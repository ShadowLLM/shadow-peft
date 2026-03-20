from __future__ import annotations

"""
Standalone experiment runner using Shadow-PEFT (no imports from `run_experiments.py`).

This script is a copy-style reimplementation of `run_experiments.py`, but uses the
`shadow_peft` library (../ShadowPEFT/src) for Shadow experiments.
"""

import argparse
import copy
import json
import os
import re
import sys
from typing import Dict, List, Optional

sys.dont_write_bytecode = True  # avoid __pycache__ permission issues in some environments

import evaluate
import numpy as np
import torch
import torch.nn.functional as F
import shadow_peft
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import EvalLoopOutput

try:
    from trl import GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The `trl` package is required for generation tasks. Install it with `pip install trl`."
    ) from exc

from data_utils import (
    ClassificationDatasetBundle,
    ExperimentDatasetBundle,
    GSM8KDatasetBundle,
    MMLUDatasetBundle,
    SquadV2DatasetBundle,
    build_classification_datasets,
    build_datasets,
    build_gsm8k_datasets,
    build_mmlu_datasets,
    build_squad_v2_datasets,
    extract_gsm8k_final_answer,
)

from shadow_peft import (
    ShadowConfig,
    ShadowForCausalLM,
    ShadowForSequenceClassification,
    get_shadow_model,
    prepare_shadow_model,
)

# MMLU subtasks from lighteval/mmlu dataset
MMLU_SUBTASKS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]

MMLU_SUITE = [
    {
        "id": "mmlu_full",
        "mmlu_subset": "all",
        "max_seq_length": 512,
    }
]

GENERATION_SUITE = MMLU_SUITE

GSM8K_SUITE = [
    {
        "id": "gsm8k_main",
        "gsm8k_subset": "main",
        "max_seq_length": 512,
    }
]

CLASSIFICATION_SUITE = [
    {
        "id": "amazon_reviews_multi_en",
        "dataset_name": "SetFit/amazon_reviews_multi_en",
        "dataset_config": None,
        "train_split": "train",
        "eval_split": "test",
        "text_column": "text",
        "label_column": "label",
        "max_seq_length": 256,
        "shadow_alpha": 4.0,
        "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    },
    {
        "id": "ag_news",
        "dataset_name": "SetFit/ag_news",
        "dataset_config": None,
        "train_split": "train",
        "eval_split": "test",
        "text_column": "text",
        "label_column": "label",
        "max_seq_length": 256,
        "shadow_alpha": 2.0,
        "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    },
]

DEFAULT_MMLU_GENERATION_TOKENS = 10
DEFAULT_GSM8K_GENERATION_TOKENS = 256
DEFAULT_SQUAD_V2_GENERATION_TOKENS = 32

REWARD_CORRECT_ANSWER = 1.0
REWARD_WRONG_ANSWER = -0.5
REWARD_INVALID_FORMAT = -1.0


def _unwrap_model(model):
    """Unwrap DDP/DataParallel wrappers."""
    return model.module if hasattr(model, "module") else model


def _save_shadow_peft_adapter(model, tokenizer, output_dir: str) -> None:
    """
    Save ONLY Shadow-PEFT adapter weights/config via ShadowPEFT's `save_pretrained`.

    This intentionally avoids saving backbone weights (e.g. `pytorch_model.bin`).
    """
    os.makedirs(output_dir, exist_ok=True)
    actual = _unwrap_model(model)
    if not hasattr(actual, "save_pretrained"):
        raise AttributeError("Shadow model does not implement save_pretrained().")
    actual.save_pretrained(output_dir)
    # Tokenizer is safe to save (no backbone weights), and helps with reload/inference.
    if tokenizer is not None:
        tokenizer.save_pretrained(output_dir)


def _shifted_ce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


def _resolve_mmlu_eval_subsets(requested_subset: Optional[str]) -> List[str]:
    if requested_subset in (None, "all"):
        return list(MMLU_SUBTASKS)
    if requested_subset == "auxiliary_train":
        raise ValueError("Evaluation subset cannot be 'auxiliary_train'.")
    return [requested_subset]


def _pick_reference_eval_dataset(eval_datasets: Dict[str, object]):
    for dataset in eval_datasets.values():
        return dataset
    raise ValueError("At least one evaluation subset is required for MMLU.")


def _evaluate_mmlu_subsets(trainer: Trainer, eval_datasets: Dict[str, object]):
    metrics: Dict[str, Dict[str, float]] = {}
    for subset_name, eval_dataset in eval_datasets.items():
        subset_metrics = trainer.evaluate(eval_dataset=eval_dataset, metric_key_prefix="eval")
        entry: Dict[str, float] = {
            "loss": subset_metrics.get("eval_loss"),
            "accuracy": subset_metrics.get("eval_accuracy"),
            "samples": subset_metrics.get("eval_samples"),
        }
        if subset_metrics.get("eval_shadow_loss") is not None:
            entry["shadow_loss"] = subset_metrics.get("eval_shadow_loss")
        if subset_metrics.get("eval_shadow_accuracy") is not None:
            entry["shadow_accuracy"] = subset_metrics.get("eval_shadow_accuracy")
        metrics[subset_name] = entry
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LoRA vs Shadow-PEFT experiments.")
    parser.add_argument(
        "--task",
        type=str,
        choices=("generation", "classification", "mmlu", "gsm8k", "squad_v2"),
        default="mmlu",
    )
    parser.add_argument(
        "--suite",
        type=str,
        choices=("generation", "classification", "mmlu", "gsm8k", "squad_v2", "all"),
        default=None,
    )
    parser.add_argument(
        "--classification_id",
        type=str,
        default=None,
        choices=[spec["id"] for spec in CLASSIFICATION_SUITE],
        help="Run only one classification suite entry when using --suite classification/all.",
    )
    parser.add_argument("--mmlu_subset", type=str, default="all")
    parser.add_argument("--gsm8k_subset", type=str, choices=("main", "socratic"), default="main")
    parser.add_argument("--gsm8k_answer_mode", type=str, choices=("thinking", "final"), default="thinking")
    parser.add_argument("--squad_answer_mode", type=str, choices=("thinking", "final"), default="final")

    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument(
        "--pretrained_shadow_model_name",
        type=str,
        default=None,
        help=(
            "Optional Hugging Face model name/path to use as the explicit Shadow backbone. "
            "If provided, Shadow-PEFT will be constructed with "
            "`get_shadow_model(base_model, shadow_config, shadow_model=shadow_model)`."
        ),
    )
    parser.add_argument(
        "--remove_shadow_embed_tokens",
        type=int,
        default=None,
        choices=[0, 1],
        help=(
            "Only relevant when --pretrained_shadow_model_name is set. If 1, call "
            "`prepare_shadow_model(shadow_model, remove_embed_tokens=True)` before constructing "
            "ShadowPEFT, so the explicit shadow backbone is driven by base `inputs_embeds`."
        ),
    )
    parser.add_argument("--dataset_name", type=str, default="tiny_shakespeare")
    parser.add_argument("--dataset_config", type=str, default=None)
    parser.add_argument("--text_column", type=str, default="text")
    parser.add_argument("--text_pair_column", type=str, default=None)
    parser.add_argument("--text_template", type=str, default=None)
    parser.add_argument("--label_column", type=str, default="label")
    parser.add_argument("--block_size", type=int, default=512)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--generation_max_length", type=int, default=None)
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="flash_attention_2",
        choices=("sdpa", "eager", "flash_attention_2"),
    )

    parser.add_argument("--per_device_train_batch_size", type=int, default=128)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=128)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--eval_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=100000)
    parser.add_argument("--save_steps", type=int, default=0)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--run_name", type=str, default="shadow_peft_experiment")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--print_shadow_output", type=int, default=1, choices=[0, 1])
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--eval_split", type=str, default="validation")
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=("lora", "dora", "shadow", "both"),
        help="Which method(s) to run. 'dora' is an alias for running LoRA with --peft_method dora.",
    )
    parser.add_argument("--trainer", type=str, default="sft", choices=("sft", "grpo", "both"))
    parser.add_argument("--fp16", type=int, default=0, choices=[0, 1])
    parser.add_argument("--bf16", type=int, default=1, choices=[0, 1])
    parser.add_argument("--classification_metric", type=str, default="accuracy")

    # LoRA
    parser.add_argument("--peft_method", type=str, default="lora", choices=("lora", "dora"))
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", type=str, nargs="+", default=["q_proj", "k_proj", "v_proj", "o_proj"])

    # Shadow-PEFT
    parser.add_argument("--shadow_layers", type=int, default=1)
    parser.add_argument("--shadow_intermediate_size", type=int, default=128)
    parser.add_argument("--shadow_num_attention_heads", type=int, default=None)
    parser.add_argument("--shadow_num_key_value_heads", type=int, default=None)
    parser.add_argument("--shadow_head_dim", type=int, default=None)
    parser.add_argument("--shadow_loss_weight", type=float, default=0.05)
    parser.add_argument("--shadow_alpha", type=float, default=1.0)
    parser.add_argument("--shadow_dropout", type=float, default=0.2)
    parser.add_argument("--injection_hidden_size", type=int, default=16)
    parser.add_argument("--gate_hidden_size", type=int, default=10)
    parser.add_argument(
        "--modules_to_save",
        type=str,
        nargs="+",
        default=None,
        help="Optional extra trainable modules to save in Shadow checkpoints (PEFT-style). "
        "Examples: seqcls: classifier_head shadow_classifier_head; causal LM: lm_head shadow_lm_head.",
    )
    parser.add_argument(
        "--shadow_lr_multiplier",
        type=float,
        default=1.0,
        help="Learning rate multiplier for shadow-only parameters (shadow_model, shadow_lm_head, shadow_adapter). "
        "E.g., 0.1 means shadow params get 10%% of base LR. Default: 1.0 (same as base).",
    )

    # MMLU fewshot
    parser.add_argument("--use_few_shot", type=int, default=0, choices=[0, 1])
    return parser.parse_args()


def _normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    """Backwards/UX compatibility normalizations."""
    # Support `--mode dora` as a shorthand for running the LoRA path with DoRA enabled.
    if getattr(args, "mode", None) == "dora":
        args.mode = "lora"
        args.peft_method = "dora"
    return args


def _make_run_name(args: argparse.Namespace, suffix: str) -> Optional[str]:
    if args.run_name is None:
        return None
    return f"{args.run_name}-{args.dataset_name}-{suffix}"


def _resolve_save_strategy(args: argparse.Namespace) -> str:
    # Avoid checkpoint saving by default (it can fail with tied/shared tensors under safetensors).
    return "no" if int(getattr(args, "save_steps", 0)) <= 0 else "steps"


def prepare_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        pad_fallback = tokenizer.eos_token or tokenizer.bos_token
        if pad_fallback is None:
            raise ValueError("Tokenizer is missing pad/eos/bos tokens.")
        tokenizer.pad_token = pad_fallback
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    return tokenizer


def _configure_model_no_thinking(model, tokenizer):
    """Best-effort: avoid generating thinking blocks for chatty models (e.g. Qwen)."""
    gen_cfg = getattr(model, "generation_config", None)
    if gen_cfg is not None and hasattr(gen_cfg, "enable_thinking"):
        gen_cfg.enable_thinking = False

    if hasattr(tokenizer, "convert_tokens_to_ids") and gen_cfg is not None:
        stop_tokens = []
        for token in ["<think>", "<|im_start|>think"]:
            token_id = tokenizer.convert_tokens_to_ids(token)
            unk_id = getattr(tokenizer, "unk_token_id", None)
            if isinstance(token_id, int) and unk_id is not None and token_id != unk_id:
                stop_tokens.append(token_id)
        if stop_tokens and hasattr(gen_cfg, "eos_token_id"):
            if isinstance(gen_cfg.eos_token_id, list):
                gen_cfg.eos_token_id.extend(stop_tokens)
            else:
                gen_cfg.eos_token_id = [gen_cfg.eos_token_id] + stop_tokens


def _set_attn_impl(model, attn_impl: str):
    if hasattr(model, "config"):
        try:
            model.config._attn_implementation = attn_impl
        except AttributeError:
            pass


def prepare_causal_model(model_name: str, attn_impl: str):
    # Support ShadowPEFT's projected shadow model wrapper (AutoModelForCausalLMWithHiddenProjection).
    # This model uses a custom `model_type` and cannot be loaded via AutoModelForCausalLM.
    try:
        from pathlib import Path
        import json as _json

        raw_cfg = None
        revision = None

        # Local folder?
        p = Path(model_name)
        cfg_path = p / "config.json"
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                raw_cfg = _json.load(f)
        else:
            # Hub repo id (optionally `repo@revision`)
            repo_spec = str(model_name)
            if "@" in repo_spec and "/" in repo_spec and not repo_spec.startswith(("/", ".", "~")):
                repo_id, revision = repo_spec.rsplit("@", 1)
            else:
                repo_id = repo_spec

            try:
                from huggingface_hub import hf_hub_download

                cfg_file = hf_hub_download(repo_id=repo_id, filename="config.json", revision=revision)
                with open(cfg_file, "r", encoding="utf-8") as f:
                    raw_cfg = _json.load(f)
            except Exception:
                raw_cfg = None

        if isinstance(raw_cfg, dict) and raw_cfg.get("model_type") in (
            "shadow_proj_causal_lm",
            "causal_lm_with_hidden_projection",
        ):
            from shadow_peft import AutoModelForCausalLMWithHiddenProjection

            model = AutoModelForCausalLMWithHiddenProjection.from_pretrained(
                model_name,
                revision=revision,
                freeze_backbone=False,
            )
            if hasattr(model.config, "use_cache"):
                model.config.use_cache = False
            return model
    except Exception as err:
        # Fall back to standard HF loading.
        print(f">>> Error loading shadow model: {err}")

    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation=attn_impl)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        _set_attn_impl(model, attn_impl)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    return model


def _sync_pad_token(model, tokenizer):
    if hasattr(model, "config") and getattr(model.config, "pad_token_id", None) is None:
        if tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer pad_token_id is undefined.")
        model.config.pad_token_id = tokenizer.pad_token_id


def _build_position_ids(attention_mask: torch.Tensor, padding_side: str) -> torch.Tensor:
    device = attention_mask.device
    seq_length = attention_mask.size(1)
    if padding_side == "left":
        position_ids = attention_mask.long().cumsum(dim=1) - 1
        position_ids.clamp_(min=0)
        position_ids = position_ids.masked_fill(attention_mask == 0, 0)
        return position_ids.to(device=device).clone()
    base = torch.arange(seq_length, device=device).unsqueeze(0)
    return base.expand_as(attention_mask).clone()


class LMDataCollatorWithPositions:
    def __init__(self, tokenizer):
        self.base = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        self.tokenizer = tokenizer

    def __call__(self, features):
        batch = self.base(features)
        attention_mask = batch.get("attention_mask")
        if attention_mask is None:
            attention_mask = (batch["input_ids"] != self.tokenizer.pad_token_id).long()
            batch["attention_mask"] = attention_mask
        attention_mask = attention_mask.to(dtype=torch.bool)
        batch["attention_mask"] = attention_mask
        batch["position_ids"] = _build_position_ids(attention_mask.long(), self.tokenizer.padding_side)
        return batch


class ClassificationCollatorWithPositions:
    def __init__(self, tokenizer):
        self.base = DataCollatorWithPadding(tokenizer=tokenizer)
        self.tokenizer = tokenizer

    def __call__(self, features):
        batch = self.base(features)
        attention_mask = batch.get("attention_mask")
        if attention_mask is None:
            attention_mask = (batch["input_ids"] != self.tokenizer.pad_token_id).long()
            batch["attention_mask"] = attention_mask
        attention_mask = attention_mask.to(dtype=torch.bool)
        batch["attention_mask"] = attention_mask
        batch["position_ids"] = _build_position_ids(attention_mask.long(), self.tokenizer.padding_side)
        return batch


class GSM8KDataCollator:
    """Data collator for GSM8K that preserves prompt and gold answer."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, features):
        gold_answers = [f.get("gold_answer", "") for f in features]

        input_ids_list = [f["input_ids"] for f in features]
        attention_list = [f.get("attention_mask", [1] * len(f["input_ids"])) for f in features]
        labels_list = [f.get("labels", f["input_ids"]) for f in features]

        max_length = max(len(ids) for ids in input_ids_list)
        batch_input_ids = []
        batch_labels = []
        batch_attention = []
        for ids, mask, labels in zip(input_ids_list, attention_list, labels_list):
            pad_len = max_length - len(ids)
            batch_input_ids.append(ids + [self.pad_token_id] * pad_len)
            batch_attention.append(mask + [0] * pad_len)
            batch_labels.append(labels + [-100] * pad_len)

        batch = {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention, dtype=torch.bool),
        }

        batch["position_ids"] = _build_position_ids(
            batch["attention_mask"].long(), self.tokenizer.padding_side
        )

        prompt_input_ids_list = [f.get("prompt_input_ids", f["input_ids"]) for f in features]
        prompt_attention_list = []
        for f, prompt_ids in zip(features, prompt_input_ids_list):
            prompt_mask = f.get("prompt_attention_mask")
            if prompt_mask is None:
                prompt_mask = [1] * len(prompt_ids)
            prompt_attention_list.append(prompt_mask)

        max_prompt_len = max(len(ids) for ids in prompt_input_ids_list)
        padded_prompt_ids = []
        padded_prompt_masks = []
        for ids, mask in zip(prompt_input_ids_list, prompt_attention_list):
            pad_len = max_prompt_len - len(ids)
            padded_prompt_ids.append(ids + [self.pad_token_id] * pad_len)
            padded_prompt_masks.append(mask + [0] * pad_len)

        batch["prompt_input_ids"] = torch.tensor(padded_prompt_ids, dtype=torch.long)
        batch["prompt_attention_mask"] = torch.tensor(padded_prompt_masks, dtype=torch.bool)
        batch["gold_answers"] = gold_answers
        return batch


class DifferentialLRMixin:
    """
    Mixin that adds differential learning rate support for shadow parameters.
    
    Usage: Mix this into any Trainer class and set self.shadow_lr_multiplier before calling super().__init__().
    """
    
    def _create_optimizer_with_differential_lr(self, shadow_lr_multiplier: float):
        """Create optimizer with different LRs for shadow vs base parameters."""
        if self.optimizer is not None:
            return self.optimizer
        
        # Identify shadow parameter patterns
        shadow_param_patterns = [
            'shadow_model.',
            'shadow_lm_head',
            'shadow_adapter',
            'shadow_proj',
            'shadow_hidden_projection',
            'shadow_classifier',
        ]
        
        # Categorize parameters into 4 groups
        base_decay = []
        base_nodecay = []
        shadow_decay = []
        shadow_nodecay = []
        
        print(">>> model architecture:")
        print(self.model)
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            # Check if this is a shadow parameter
            is_shadow = any(pattern in name for pattern in shadow_param_patterns)
            
            # Check if this parameter should have weight decay
            no_decay = len(param.shape) == 1 or name.endswith(".bias")
            
            if is_shadow:
                print(f">>> Shadow parameter: {name}")
                if no_decay:
                    shadow_nodecay.append(param)
                else:
                    shadow_decay.append(param)
            else:
                if no_decay:
                    base_nodecay.append(param)
                else:
                    base_decay.append(param)
        
        base_lr = self.args.learning_rate
        shadow_lr = base_lr * shadow_lr_multiplier
        
        # Build parameter groups
        optimizer_grouped_parameters = []
        
        if base_decay:
            optimizer_grouped_parameters.append({
                "params": base_decay,
                "weight_decay": self.args.weight_decay,
                "lr": base_lr,
            })
        
        if base_nodecay:
            optimizer_grouped_parameters.append({
                "params": base_nodecay,
                "weight_decay": 0.0,
                "lr": base_lr,
            })
        
        if shadow_decay:
            optimizer_grouped_parameters.append({
                "params": shadow_decay,
                "weight_decay": self.args.weight_decay,
                "lr": shadow_lr,
            })
        
        if shadow_nodecay:
            optimizer_grouped_parameters.append({
                "params": shadow_nodecay,
                "weight_decay": 0.0,
                "lr": shadow_lr,
            })
        
        # Get optimizer class and kwargs from training args
        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
        
        # Remove 'lr' from optimizer_kwargs if present (we set it per group)
        optimizer_kwargs.pop('lr', None)
        
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        
        # Log the setup
        if shadow_lr_multiplier != 1.0:
            print(f"\n=== Differential Learning Rate Setup ===")
            print(f"Base LR: {base_lr}")
            print(f"Shadow LR: {shadow_lr} (multiplier: {shadow_lr_multiplier})")
            print(f"Base params (decay): {len(base_decay)}")
            print(f"Base params (no decay): {len(base_nodecay)}")
            print(f"Shadow params (decay): {len(shadow_decay)}")
            print(f"Shadow params (no decay): {len(shadow_nodecay)}")
            print("=" * 40 + "\n")
        
        return self.optimizer
    
    def create_optimizer(self):
        """Override to use differential LR if shadow_lr_multiplier is set."""
        shadow_lr_multiplier = getattr(self, 'shadow_lr_multiplier', 1.0)
        if shadow_lr_multiplier != 1.0:
            return self._create_optimizer_with_differential_lr(shadow_lr_multiplier)
        return super().create_optimizer()


class SquadV2DataCollator:
    """Data collator for SQuAD v2 that preserves prompt + id/answers for metric eval."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, features):
        squad_ids = [f.get("squad_id", "") for f in features]
        squad_answers = [f.get("squad_answers", {"text": [], "answer_start": []}) for f in features]

        input_ids_list = [f["input_ids"] for f in features]
        attention_list = [f.get("attention_mask", [1] * len(f["input_ids"])) for f in features]
        labels_list = [f.get("labels", f["input_ids"]) for f in features]

        max_length = max(len(ids) for ids in input_ids_list)
        batch_input_ids = []
        batch_labels = []
        batch_attention = []
        for ids, mask, labels in zip(input_ids_list, attention_list, labels_list):
            pad_len = max_length - len(ids)
            batch_input_ids.append(ids + [self.pad_token_id] * pad_len)
            batch_attention.append(mask + [0] * pad_len)
            batch_labels.append(labels + [-100] * pad_len)

        batch = {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention, dtype=torch.bool),
        }

        batch["position_ids"] = _build_position_ids(
            batch["attention_mask"].long(), self.tokenizer.padding_side
        )

        prompt_input_ids_list = [f.get("prompt_input_ids", f["input_ids"]) for f in features]
        prompt_attention_list = []
        for f, prompt_ids in zip(features, prompt_input_ids_list):
            prompt_mask = f.get("prompt_attention_mask")
            if prompt_mask is None:
                prompt_mask = [1] * len(prompt_ids)
            prompt_attention_list.append(prompt_mask)

        max_prompt_len = max(len(ids) for ids in prompt_input_ids_list)
        padded_prompt_ids = []
        padded_prompt_masks = []
        for ids, mask in zip(prompt_input_ids_list, prompt_attention_list):
            pad_len = max_prompt_len - len(ids)
            padded_prompt_ids.append(ids + [self.pad_token_id] * pad_len)
            padded_prompt_masks.append(mask + [0] * pad_len)

        batch["prompt_input_ids"] = torch.tensor(padded_prompt_ids, dtype=torch.long)
        batch["prompt_attention_mask"] = torch.tensor(padded_prompt_masks, dtype=torch.bool)
        batch["squad_ids"] = squad_ids
        batch["squad_answers"] = squad_answers
        return batch


class MMLUDataCollator:
    """Data collator for MMLU that preserves answer information + prompt inputs."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, features):
        # Extract answer info FIRST before any processing
        answer_letters = [f.get("answer_letter", None) for f in features]
        answer_indices = [f.get("answer_idx", None) for f in features]

        # Remove answer fields from features for tokenization
        features_for_collation = []
        for f in features:
            f_clean = {k: v for k, v in f.items() if k not in ["answer_letter", "answer_idx"]}
            features_for_collation.append(f_clean)

        # Manual padding and batching
        input_ids_list = [f["input_ids"] for f in features_for_collation]
        labels_list = [f.get("labels", f["input_ids"]) for f in features_for_collation]
        max_length = max(len(ids) for ids in input_ids_list)

        batch_input_ids = []
        batch_labels = []
        for ids, labels in zip(input_ids_list, labels_list):
            pad_len = max_length - len(ids)
            batch_input_ids.append(ids + [self.pad_token_id] * pad_len)
            batch_labels.append(labels + [-100] * pad_len)

        batch = {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }

        attention_mask = (batch["input_ids"] != self.pad_token_id).bool()
        batch["attention_mask"] = attention_mask
        batch["position_ids"] = _build_position_ids(attention_mask.long(), self.tokenizer.padding_side)

        # Prepare prompt-only inputs (everything up to "Answer: ")
        prompt_input_ids_list = [f.get("prompt_input_ids", f["input_ids"]) for f in features]
        prompt_attention_list = []
        for feature, prompt_ids in zip(features, prompt_input_ids_list):
            prompt_mask = feature.get("prompt_attention_mask")
            if prompt_mask is None:
                prompt_mask = [1] * len(prompt_ids)
            elif len(prompt_mask) != len(prompt_ids):
                if len(prompt_mask) > len(prompt_ids):
                    prompt_mask = prompt_mask[: len(prompt_ids)]
                else:
                    prompt_mask = prompt_mask + [0] * (len(prompt_ids) - len(prompt_mask))
            prompt_attention_list.append(prompt_mask)

        max_prompt_len = max(len(ids) for ids in prompt_input_ids_list)
        padded_prompt_ids = []
        padded_prompt_masks = []
        for ids, mask in zip(prompt_input_ids_list, prompt_attention_list):
            pad_len = max_prompt_len - len(ids)
            padded_prompt_ids.append(ids + [self.pad_token_id] * pad_len)
            padded_prompt_masks.append(mask + [0] * pad_len)

        batch["prompt_input_ids"] = torch.tensor(padded_prompt_ids, dtype=torch.long)
        batch["prompt_attention_mask"] = torch.tensor(padded_prompt_masks, dtype=torch.bool)

        batch["answer_letters"] = answer_letters
        batch["answer_indices"] = answer_indices
        return batch


_THINK_PATTERN = re.compile(r"<think>.*?</think>", flags=re.IGNORECASE | re.DOTALL)
_IM_TAG_PATTERN = re.compile(r"<\|im_(?:start|end)\|>")
_HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
_ANSWER_PATTERN = re.compile(r"(?:answer|option)\s*[:=-]?\s*([ABCD])", flags=re.IGNORECASE)
_LETTER_PATTERN = re.compile(r"\b([ABCD])\b", flags=re.IGNORECASE)


def _clean_generated_text(text: str) -> str:
    no_think = _THINK_PATTERN.sub(" ", text)
    no_im = _IM_TAG_PATTERN.sub(" ", no_think)
    no_tags = _HTML_TAG_PATTERN.sub(" ", no_im)
    return re.sub(r"\s+", " ", no_tags).strip()


def extract_answer_from_text(text: str) -> Optional[str]:
    if not text:
        return None
    cleaned = _clean_generated_text(text)
    if not cleaned:
        return None
    match = _LETTER_PATTERN.search(cleaned)
    if match:
        return match.group(1).upper()
    match = _ANSWER_PATTERN.search(cleaned)
    if match:
        return match.group(1).upper()
    return None


def extract_gsm8k_answer_from_text(text: str) -> str:
    if not text:
        return ""
    cleaned = _clean_generated_text(text)
    return extract_gsm8k_final_answer(cleaned)


def compute_mmlu_reward(generated_texts: List[str], ground_truth_answers: List[str]) -> List[float]:
    rewards = []
    for gen_text, true_answer in zip(generated_texts, ground_truth_answers):
        pred = extract_answer_from_text(gen_text) if gen_text else None
        if pred is None or pred not in ["A", "B", "C", "D"]:
            reward = REWARD_INVALID_FORMAT
        elif pred == true_answer.upper():
            reward = REWARD_CORRECT_ANSWER
        else:
            reward = REWARD_WRONG_ANSWER
        rewards.append(reward)
    return rewards


def _shadow_config_from_args(args: argparse.Namespace) -> ShadowConfig:
    return ShadowConfig(
        num_shadow_layers=int(args.shadow_layers),
        shadow_intermediate_size=int(args.shadow_intermediate_size),
        shadow_num_attention_heads=(
            None if getattr(args, "shadow_num_attention_heads", None) is None else int(args.shadow_num_attention_heads)
        ),
        shadow_num_key_value_heads=(
            None if getattr(args, "shadow_num_key_value_heads", None) is None else int(args.shadow_num_key_value_heads)
        ),
        shadow_head_dim=None if getattr(args, "shadow_head_dim", None) is None else int(args.shadow_head_dim),
        injection_hidden_size=int(args.injection_hidden_size),
        gate_hidden_size=int(args.gate_hidden_size),
        alpha=float(args.shadow_alpha),
        dropout=float(args.shadow_dropout),
        modules_to_save=list(getattr(args, "modules_to_save", None) or []),
    )


def _build_shadow_peft_causal_lm(
    args: argparse.Namespace, base_model, *, shadow_model=None
) -> ShadowForCausalLM:
    peft = get_shadow_model(base_model, _shadow_config_from_args(args), shadow_model=shadow_model)
    return ShadowForCausalLM(
        peft,
        shadow_loss_weight=float(args.shadow_loss_weight),
        inference_mode="base_shadow",
    )


def _build_shadow_peft_seqcls(
    args: argparse.Namespace, base_model, *, shadow_model=None
) -> ShadowForSequenceClassification:
    peft = get_shadow_model(base_model, _shadow_config_from_args(args), shadow_model=shadow_model)
    return ShadowForSequenceClassification(peft, shadow_loss_weight=float(args.shadow_loss_weight), inference_mode="base_shadow")


def generate_from_shadow(model: ShadowForCausalLM, **gen_kwargs) -> torch.Tensor:
    prev = getattr(model, "inference_mode", "base_shadow")
    model.set_inference_mode("shadow_only")
    try:
        return model.generate(**gen_kwargs)
    finally:
        model.set_inference_mode(prev)


class ShadowSFTTrainer(DifferentialLRMixin, SFTTrainer):
    """SFTTrainer variant that evaluates shadow_logits during evaluation."""

    def __init__(self, *args, shadow_lr_multiplier=1.0, **kwargs):
        self.shadow_lr_multiplier = shadow_lr_multiplier
        super().__init__(*args, **kwargs)

    def evaluation_loop(self, dataloader, description, prediction_loss_only=None, ignore_keys=None, metric_key_prefix="eval"):
        model = self._wrap_model(self.model, training=False, dataloader=dataloader)
        model.eval()

        total_loss = 0.0
        total_shadow_loss = 0.0
        num_samples = 0

        for step, inputs in enumerate(dataloader):
            inputs = self._prepare_inputs(inputs)
            with torch.no_grad():
                outputs = model(**inputs)
                if outputs.loss is not None:
                    total_loss += outputs.loss.item()

                if getattr(outputs, "shadow_logits", None) is not None:
                    labels = inputs.get("labels")
                    if labels is not None:
                        total_shadow_loss += float(_shifted_ce_loss(outputs.shadow_logits, labels).item())

                batch_size = inputs["input_ids"].shape[0]
                num_samples += batch_size

        avg_loss = total_loss / (step + 1) if step >= 0 else 0.0
        metrics = {
            f"{metric_key_prefix}_loss": avg_loss,
            f"{metric_key_prefix}_samples": num_samples,
        }
        if total_shadow_loss > 0:
            metrics[f"{metric_key_prefix}_shadow_loss"] = total_shadow_loss / (step + 1)

        return EvalLoopOutput(predictions=None, label_ids=None, metrics=metrics, num_samples=num_samples)


class MMLUTrainer(DifferentialLRMixin, SFTTrainer):
    """SFTTrainer variant for MMLU with answer extraction and accuracy computation."""

    def __init__(self, *args, shadow_lr_multiplier=1.0, **kwargs):
        self.shadow_lr_multiplier = shadow_lr_multiplier
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        inputs.pop("prompt_input_ids", None)
        inputs.pop("prompt_attention_mask", None)
        inputs.pop("answer_letter", None)
        inputs.pop("answer_idx", None)
        return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

    def evaluation_loop(self, dataloader, description, prediction_loss_only=None, ignore_keys=None, metric_key_prefix="eval"):
        model = self._wrap_model(self.model, training=False, dataloader=dataloader)
        model.eval()

        all_predictions = []
        all_shadow_predictions = []
        all_labels = []
        total_loss = 0.0
        total_shadow_loss = 0.0
        num_samples = 0

        actual_model = model.module if hasattr(model, "module") else model
        do_shadow_gen = bool(getattr(self.args, "print_shadow_output", False))

        compute_dtype = torch.bfloat16 if self.args.bf16 else (torch.float16 if self.args.fp16 else torch.float32)
        max_new_tokens = getattr(self.args, "generation_max_length", None) or DEFAULT_MMLU_GENERATION_TOKENS
        pad_token_id = getattr(self.processing_class, "pad_token_id", getattr(self.model.config, "pad_token_id", None))
        eos_token_id = getattr(self.processing_class, "eos_token_id", getattr(self.model.config, "eos_token_id", None))
        if pad_token_id is None and eos_token_id is not None:
            pad_token_id = eos_token_id

        for step, inputs in enumerate(dataloader):
            inputs = self._prepare_inputs(inputs)
            prompt_input_ids = inputs.pop("prompt_input_ids", None)
            prompt_attention_mask = inputs.pop("prompt_attention_mask", None)
            answer_letters = inputs.pop("answer_letters", None)

            with torch.no_grad():
                outputs = model(**inputs)
                if outputs.loss is not None:
                    total_loss += outputs.loss.item()
                labels = inputs.get("labels")
                if labels is not None and getattr(outputs, "shadow_logits", None) is not None:
                    total_shadow_loss += float(_shifted_ce_loss(outputs.shadow_logits, labels).item())

                input_ids = inputs["input_ids"]
                attention_mask = inputs["attention_mask"]
                batch_size = input_ids.shape[0]

                for i in range(batch_size):
                    if prompt_input_ids is not None and prompt_attention_mask is not None:
                        gen_input_ids = prompt_input_ids[i : i + 1, :]
                        gen_attention_mask = prompt_attention_mask[i : i + 1, :]
                    else:
                        gen_input_ids = input_ids[i : i + 1, :]
                        gen_attention_mask = attention_mask[i : i + 1, :]

                    if gen_attention_mask is not None:
                        true_len = int(gen_attention_mask.long().sum(dim=-1).item())
                        true_len = max(true_len, 1)
                        padding_side = getattr(self.processing_class, "padding_side", "right")
                        if padding_side == "left":
                            gen_input_ids = gen_input_ids[:, -true_len:]
                            gen_attention_mask = gen_attention_mask[:, -true_len:]
                        else:
                            gen_input_ids = gen_input_ids[:, :true_len]
                            gen_attention_mask = gen_attention_mask[:, :true_len]

                    with torch.autocast(
                        device_type="cuda",
                        dtype=compute_dtype,
                        enabled=bool(self.args.bf16 or self.args.fp16),
                    ):
                        gen_kwargs = dict(
                            input_ids=gen_input_ids,
                            attention_mask=gen_attention_mask,
                            max_new_tokens=max_new_tokens,
                            do_sample=False,
                            pad_token_id=pad_token_id,
                            eos_token_id=eos_token_id,
                            use_cache=False,
                        )
                        generated = actual_model.generate(**gen_kwargs)

                    prompt_length = gen_input_ids.shape[-1]
                    new_tokens = generated[0, prompt_length:]
                    gen_text_raw = self.processing_class.decode(new_tokens, skip_special_tokens=True)
                    generated_text = _clean_generated_text(gen_text_raw)
                    predicted_answer = extract_answer_from_text(generated_text) or "INVALID"
                    all_predictions.append(predicted_answer.upper() if predicted_answer else "INVALID")

                    shadow_generated_text = None
                    shadow_predicted = None
                    shadow_info = ""

                    if do_shadow_gen and hasattr(actual_model, "shadow_lm_head"):
                        with torch.autocast(
                            device_type="cuda",
                            dtype=compute_dtype,
                            enabled=bool(self.args.bf16 or self.args.fp16),
                        ):
                            shadow_generated = generate_from_shadow(actual_model, **gen_kwargs)
                        shadow_new_tokens = shadow_generated[0, prompt_length:]
                        shadow_text_raw = self.processing_class.decode(shadow_new_tokens, skip_special_tokens=True)
                        shadow_generated_text = _clean_generated_text(shadow_text_raw)
                        shadow_predicted = extract_answer_from_text(shadow_generated_text) or "INVALID"
                        all_shadow_predictions.append(
                            shadow_predicted.upper() if shadow_predicted else "INVALID"
                        )

                    if answer_letters and i < len(answer_letters) and answer_letters[i] is not None:
                        true_answer = str(answer_letters[i]).upper()
                        all_labels.append(true_answer)
                        if do_shadow_gen and shadow_generated_text is not None:
                            shadow_info = (
                                f" | shadow_gen: '{shadow_generated_text}' | shadow_pred: {shadow_predicted}"
                            )
                        print(
                            f"[Sample {num_samples}] generated: '{generated_text}' | predicted: {predicted_answer} | true: {true_answer}{shadow_info}",
                            flush=True,
                        )
                    else:
                        print(
                            f"[Sample {num_samples}] Warning: No answer_letter found for sample {i}",
                            flush=True,
                        )
                    num_samples += 1

        correct = sum(1 for p, l in zip(all_predictions, all_labels) if p == l)
        accuracy = correct / len(all_labels) if all_labels else 0.0
        avg_loss = total_loss / (step + 1) if step >= 0 else 0.0

        metrics: Dict[str, float] = {
            f"{metric_key_prefix}_loss": avg_loss,
            f"{metric_key_prefix}_accuracy": accuracy,
            f"{metric_key_prefix}_samples": num_samples,
        }
        if total_shadow_loss > 0 and step >= 0:
            metrics[f"{metric_key_prefix}_shadow_loss"] = total_shadow_loss / (step + 1)
        if do_shadow_gen and all_shadow_predictions and all_labels:
            shadow_correct = sum(1 for p, l in zip(all_shadow_predictions, all_labels) if p == l)
            metrics[f"{metric_key_prefix}_shadow_accuracy"] = shadow_correct / len(all_labels)

        return EvalLoopOutput(predictions=all_predictions, label_ids=all_labels, metrics=metrics, num_samples=num_samples)


class GSM8KTrainer(DifferentialLRMixin, SFTTrainer):
    """SFTTrainer variant for GSM8K with final-answer extraction accuracy."""

    def __init__(self, *args, shadow_lr_multiplier=1.0, **kwargs):
        self.shadow_lr_multiplier = shadow_lr_multiplier
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        inputs.pop("prompt_input_ids", None)
        inputs.pop("prompt_attention_mask", None)
        inputs.pop("gold_answers", None)
        return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

    def evaluation_loop(
        self,
        dataloader,
        description,
        prediction_loss_only=None,
        ignore_keys=None,
        metric_key_prefix="eval",
    ):
        model = self._wrap_model(self.model, training=False, dataloader=dataloader)
        model.eval()

        total_loss = 0.0
        total_shadow_loss = 0.0
        num_batches = 0
        num_samples = 0

        correct = 0
        shadow_correct = 0
        total_answered = 0

        actual_model = model.module if hasattr(model, "module") else model
        has_shadow = hasattr(actual_model, "shadow_lm_head")
        do_shadow_gen = bool(getattr(self.args, "print_shadow_output", False))

        compute_dtype = (
            torch.bfloat16
            if self.args.bf16
            else (torch.float16 if self.args.fp16 else torch.float32)
        )
        max_new_tokens = getattr(self.args, "generation_max_length", None)
        if max_new_tokens is None:
            max_new_tokens = DEFAULT_GSM8K_GENERATION_TOKENS

        pad_token_id = getattr(
            self.processing_class,
            "pad_token_id",
            getattr(self.model.config, "pad_token_id", None),
        )
        eos_token_id = getattr(
            self.processing_class,
            "eos_token_id",
            getattr(self.model.config, "eos_token_id", None),
        )
        if pad_token_id is None and eos_token_id is not None:
            pad_token_id = eos_token_id

        use_autocast = bool(torch.cuda.is_available() and (self.args.bf16 or self.args.fp16))

        for _, inputs in enumerate(dataloader):
            inputs = self._prepare_inputs(inputs)
            prompt_input_ids = inputs.pop("prompt_input_ids", None)
            prompt_attention_mask = inputs.pop("prompt_attention_mask", None)
            gold_answers = inputs.pop("gold_answers", None)

            with torch.inference_mode():
                outputs = model(**inputs)
                if outputs.loss is not None:
                    total_loss += float(outputs.loss.item())
                num_batches += 1

                labels = inputs.get("labels")
                if labels is not None and getattr(outputs, "shadow_logits", None) is not None:
                    total_shadow_loss += float(_shifted_ce_loss(outputs.shadow_logits, labels).item())

            if prompt_input_ids is None or prompt_attention_mask is None or gold_answers is None:
                continue

            batch_size = prompt_input_ids.shape[0]
            for i in range(batch_size):
                gen_input_ids = prompt_input_ids[i : i + 1, :]
                gen_attention_mask = prompt_attention_mask[i : i + 1, :]

                # Trim per-sample padding before generation.
                if gen_attention_mask is not None:
                    true_len = int(gen_attention_mask.long().sum(dim=-1).item())
                    true_len = max(true_len, 1)
                    padding_side = getattr(self.processing_class, "padding_side", "right")
                    if padding_side == "left":
                        gen_input_ids = gen_input_ids[:, -true_len:]
                        gen_attention_mask = gen_attention_mask[:, -true_len:]
                    else:
                        gen_input_ids = gen_input_ids[:, :true_len]
                        gen_attention_mask = gen_attention_mask[:, :true_len]

                if use_autocast:
                    ctx = torch.autocast(device_type="cuda", dtype=compute_dtype, enabled=True)
                else:
                    ctx = torch.no_grad()

                with ctx:
                    generated = model.generate(
                        input_ids=gen_input_ids,
                        attention_mask=gen_attention_mask,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=pad_token_id,
                        eos_token_id=eos_token_id,
                        use_cache=False,
                    )

                prompt_length = gen_input_ids.shape[-1]
                new_tokens = generated[0, prompt_length:]
                generated_text_raw = self.processing_class.decode(new_tokens, skip_special_tokens=True)
                pred = extract_gsm8k_answer_from_text(generated_text_raw)
                gold = str(gold_answers[i]).strip()

                if pred and gold and pred == gold:
                    correct += 1
                total_answered += 1

                shadow_text_raw = None
                shadow_pred = None
                if has_shadow and do_shadow_gen:
                    with ctx:
                        shadow_generated_ids = generate_from_shadow(
                            model=actual_model,
                            input_ids=gen_input_ids,
                            attention_mask=gen_attention_mask,
                            max_new_tokens=max_new_tokens,
                            pad_token_id=pad_token_id,
                            eos_token_id=eos_token_id,
                            use_cache=False,
                        )
                    shadow_new_tokens = shadow_generated_ids[0, prompt_length:]
                    shadow_text_raw = self.processing_class.decode(
                        shadow_new_tokens, skip_special_tokens=True
                    )
                    shadow_pred = extract_gsm8k_answer_from_text(shadow_text_raw)
                    if shadow_pred and gold and shadow_pred == gold:
                        shadow_correct += 1

                running_acc = correct / total_answered if total_answered > 0 else 0.0
                running_loss = total_loss / max(num_batches, 1)
                msg = (
                    f"[Sample {num_samples}] "
                    f"acc: {running_acc:.4f} ({correct}/{total_answered}) | "
                    f"loss: {running_loss:.4f} | "
                    f"generated: '{generated_text_raw}' | predicted: {pred} | true: {gold}"
                )
                if has_shadow and do_shadow_gen and shadow_text_raw is not None:
                    running_shadow_acc = shadow_correct / total_answered if total_answered > 0 else 0.0
                    msg += (
                        f" | shadow_acc: {running_shadow_acc:.4f} ({shadow_correct}/{total_answered})"
                        f" | shadow_gen: '{shadow_text_raw}' | shadow_pred: {shadow_pred}"
                    )
                print(msg, flush=True)
                num_samples += 1

        avg_loss = total_loss / max(num_batches, 1)
        metrics = {
            f"{metric_key_prefix}_loss": avg_loss,
            f"{metric_key_prefix}_samples": num_samples,
        }
        if total_answered > 0:
            metrics[f"{metric_key_prefix}_answer_accuracy"] = correct / total_answered
            if has_shadow and do_shadow_gen:
                metrics[f"{metric_key_prefix}_shadow_answer_accuracy"] = shadow_correct / total_answered
        if total_shadow_loss > 0 and num_batches > 0:
            metrics[f"{metric_key_prefix}_shadow_loss"] = total_shadow_loss / num_batches

        return EvalLoopOutput(predictions=None, label_ids=None, metrics=metrics, num_samples=num_samples)


class SquadV2Trainer(DifferentialLRMixin, SFTTrainer):
    """SFTTrainer variant for SQuAD v2 using the official EM/F1 metric."""

    def __init__(self, *args, shadow_lr_multiplier=1.0, **kwargs):
        self.shadow_lr_multiplier = shadow_lr_multiplier
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        inputs.pop("prompt_input_ids", None)
        inputs.pop("prompt_attention_mask", None)
        inputs.pop("squad_ids", None)
        inputs.pop("squad_answers", None)
        return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

    def evaluation_loop(
        self,
        dataloader,
        description,
        prediction_loss_only=None,
        ignore_keys=None,
        metric_key_prefix="eval",
    ):
        model = self._wrap_model(self.model, training=False, dataloader=dataloader)
        model.eval()

        metric = evaluate.load("squad_v2")
        shadow_metric = None

        total_loss = 0.0
        total_shadow_loss = 0.0
        num_batches = 0
        num_samples = 0

        actual_model = model.module if hasattr(model, "module") else model
        do_shadow_gen = bool(getattr(self.args, "print_shadow_output", False))
        has_shadow = hasattr(actual_model, "shadow_lm_head")
        if has_shadow and do_shadow_gen:
            shadow_metric = evaluate.load("squad_v2")

        compute_dtype = (
            torch.bfloat16
            if self.args.bf16
            else (torch.float16 if self.args.fp16 else torch.float32)
        )
        max_new_tokens = getattr(self.args, "generation_max_length", None)
        if max_new_tokens is None:
            max_new_tokens = DEFAULT_SQUAD_V2_GENERATION_TOKENS

        pad_token_id = getattr(
            self.processing_class,
            "pad_token_id",
            getattr(self.model.config, "pad_token_id", None),
        )
        eos_token_id = getattr(
            self.processing_class,
            "eos_token_id",
            getattr(self.model.config, "eos_token_id", None),
        )
        if pad_token_id is None and eos_token_id is not None:
            pad_token_id = eos_token_id

        use_autocast = bool(torch.cuda.is_available() and (self.args.bf16 or self.args.fp16))

        for _, inputs in enumerate(dataloader):
            inputs = self._prepare_inputs(inputs)
            prompt_input_ids = inputs.pop("prompt_input_ids", None)
            prompt_attention_mask = inputs.pop("prompt_attention_mask", None)
            squad_ids = inputs.pop("squad_ids", None)
            squad_answers = inputs.pop("squad_answers", None)

            with torch.inference_mode():
                outputs = model(**inputs)
                if outputs.loss is not None:
                    total_loss += float(outputs.loss.item())
                num_batches += 1

                labels = inputs.get("labels")
                if labels is not None and getattr(outputs, "shadow_logits", None) is not None:
                    total_shadow_loss += float(_shifted_ce_loss(outputs.shadow_logits, labels).item())

            if (
                prompt_input_ids is None
                or prompt_attention_mask is None
                or squad_ids is None
                or squad_answers is None
            ):
                continue

            batch_size = prompt_input_ids.shape[0]
            for i in range(batch_size):
                gen_input_ids = prompt_input_ids[i : i + 1, :]
                gen_attention_mask = prompt_attention_mask[i : i + 1, :]

                # Trim per-sample padding before generation.
                if gen_attention_mask is not None:
                    true_len = int(gen_attention_mask.long().sum(dim=-1).item())
                    true_len = max(true_len, 1)
                    padding_side = getattr(self.processing_class, "padding_side", "right")
                    if padding_side == "left":
                        gen_input_ids = gen_input_ids[:, -true_len:]
                        gen_attention_mask = gen_attention_mask[:, -true_len:]
                    else:
                        gen_input_ids = gen_input_ids[:, :true_len]
                        gen_attention_mask = gen_attention_mask[:, :true_len]

                if use_autocast:
                    ctx = torch.autocast(device_type="cuda", dtype=compute_dtype, enabled=True)
                else:
                    ctx = torch.no_grad()

                with ctx:
                    generated = actual_model.generate(
                        input_ids=gen_input_ids,
                        attention_mask=gen_attention_mask,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=pad_token_id,
                        eos_token_id=eos_token_id,
                        use_cache=False,
                    )

                prompt_length = gen_input_ids.shape[-1]
                new_tokens = generated[0, prompt_length:]
                pred_raw = self.processing_class.decode(new_tokens, skip_special_tokens=True).strip()
                pred_clean = _clean_generated_text(pred_raw)

                # Map "unanswerable" => no-answer.
                if pred_clean.lower() in {"unanswerable", "no answer", "n/a", "none", ""}:
                    prediction_text = ""
                    no_answer_probability = 1.0
                else:
                    prediction_text = pred_clean
                    no_answer_probability = 0.0

                qid = str(squad_ids[i])
                shadow_info = ""
                if has_shadow and do_shadow_gen:
                    with ctx:
                        shadow_generated = generate_from_shadow(
                            model=actual_model,
                            input_ids=gen_input_ids,
                            attention_mask=gen_attention_mask,
                            max_new_tokens=max_new_tokens,
                            pad_token_id=pad_token_id,
                            eos_token_id=eos_token_id,
                            use_cache=False,
                        )
                    shadow_new_tokens = shadow_generated[0, prompt_length:]
                    shadow_raw = self.processing_class.decode(
                        shadow_new_tokens, skip_special_tokens=True
                    ).strip()
                    shadow_text = _clean_generated_text(shadow_raw)
                    shadow_info = f" | shadow_gen: '{shadow_text}'"
                    if shadow_metric is not None:
                        if shadow_text.lower() in {"unanswerable", "no answer", "n/a", "none", ""}:
                            shadow_prediction_text = ""
                            shadow_no_answer_probability = 1.0
                        else:
                            shadow_prediction_text = shadow_text
                            shadow_no_answer_probability = 0.0
                        shadow_info = f" | shadow_gen: '{shadow_prediction_text}'"
                        shadow_metric.add(
                            prediction={
                                "id": qid,
                                "prediction_text": shadow_prediction_text,
                                "no_answer_probability": float(shadow_no_answer_probability),
                            },
                            reference={
                                "id": qid,
                                "answers": squad_answers[i],
                            },
                        )

                print(
                    f"[Sample {num_samples}] generated: '{prediction_text}' | true: {squad_answers[i]}{shadow_info}",
                    flush=True,
                )
                metric.add(
                    prediction={
                        "id": qid,
                        "prediction_text": prediction_text,
                        "no_answer_probability": float(no_answer_probability),
                    },
                    reference={
                        "id": qid,
                        "answers": squad_answers[i],
                    },
                )
                num_samples += 1

        avg_loss = total_loss / max(num_batches, 1)
        squad_metrics = metric.compute()
        metrics = {
            f"{metric_key_prefix}_loss": avg_loss,
            f"{metric_key_prefix}_samples": num_samples,
        }
        for k, v in squad_metrics.items():
            metrics[f"{metric_key_prefix}_{k}"] = v
        if shadow_metric is not None:
            shadow_squad_metrics = shadow_metric.compute()
            for k, v in shadow_squad_metrics.items():
                metrics[f"{metric_key_prefix}_shadow_{k}"] = v
        if total_shadow_loss > 0 and num_batches > 0:
            metrics[f"{metric_key_prefix}_shadow_loss"] = total_shadow_loss / num_batches

        return EvalLoopOutput(
            predictions=None,
            label_ids=None,
            metrics=metrics,
            num_samples=num_samples,
        )


def _build_peft_config(args: argparse.Namespace, task_type):
    base_kwargs = dict(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=task_type,
    )
    if getattr(args, "peft_method", "lora") != "dora":
        return LoraConfig(**base_kwargs)
    for flag in ("use_dora", "enable_dora", "dora"):
        try:
            return LoraConfig(**base_kwargs, **{flag: True})
        except TypeError:
            continue
    cfg = LoraConfig(**base_kwargs)
    for attr in ("use_dora", "enable_dora", "dora"):
        if hasattr(cfg, attr):
            setattr(cfg, attr, True)
            return cfg
    raise RuntimeError("Your installed `peft` does not appear to support DoRA.")


def run_generation_trainer(
    model,
    tokenizer,
    training_args: SFTConfig,
    datasets: ExperimentDatasetBundle,
    trainer_cls=SFTTrainer,
    trainer_kwargs: Optional[Dict] = None,
    *,
    save_model: bool = True,
):
    data_collator = LMDataCollatorWithPositions(tokenizer)
    dataset_text_field = "text" if "text" in datasets.train_dataset.column_names else None
    init_kwargs = dict(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=datasets.train_dataset,
        eval_dataset=datasets.eval_dataset,
        data_collator=data_collator,
    )
    if dataset_text_field is not None:
        init_kwargs["dataset_text_field"] = dataset_text_field
    if trainer_kwargs:
        init_kwargs.update(trainer_kwargs)
    trainer = trainer_cls(**init_kwargs)
    trainer.train()
    metrics = trainer.evaluate()
    if save_model:
        trainer.save_model(training_args.output_dir)
    return metrics


def generation_lora(args, tokenizer, datasets):
    model = prepare_causal_model(args.model_name, args.attn_implementation)
    _sync_pad_token(model, tokenizer)
    if args.bf16:
        model = model.to(torch.bfloat16)
    elif args.fp16:
        model = model.to(torch.float16)
    lora_config = _build_peft_config(args, task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    training_args = SFTConfig(
        output_dir=os.path.join(args.output_dir, getattr(args, "peft_method", "lora")),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_accumulation_steps=args.eval_accumulation_steps,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_strategy=_resolve_save_strategy(args),
        save_steps=args.save_steps,
        save_total_limit=1,
        warmup_ratio=args.warmup_ratio,
        report_to=args.report_to,
        run_name=_make_run_name(args, f"{getattr(args, 'peft_method', 'lora')}-gen"),
        # save_safetensors=False,
        fp16=args.fp16,
        bf16=args.bf16,
    )
    setattr(training_args, "generation_max_length", getattr(args, "generation_max_length", None))
    return run_generation_trainer(model, tokenizer, training_args, datasets, save_model=True)


def generation_shadow(args, tokenizer, datasets):
    base_model = prepare_causal_model(args.model_name, args.attn_implementation)
    _sync_pad_token(base_model, tokenizer)
    shadow_model = None
    if getattr(args, "pretrained_shadow_model_name", None):
        shadow_model = prepare_causal_model(args.pretrained_shadow_model_name, args.attn_implementation)
        _sync_pad_token(shadow_model, tokenizer)
        if int(getattr(args, "remove_shadow_embed_tokens", 0) or 0) == 1:
            shadow_model = prepare_shadow_model(shadow_model, remove_embed_tokens=True)
    model = _build_shadow_peft_causal_lm(args, base_model, shadow_model=shadow_model)
    if args.bf16:
        model = model.to(torch.bfloat16)
    elif args.fp16:
        model = model.to(torch.float16)
    model.print_trainable_parameters()
    training_args = SFTConfig(
        output_dir=os.path.join(args.output_dir, "shadow_peft"),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_accumulation_steps=args.eval_accumulation_steps,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_strategy=_resolve_save_strategy(args),
        save_steps=args.save_steps,
        save_total_limit=1,
        warmup_ratio=args.warmup_ratio,
        report_to=args.report_to,
        run_name=_make_run_name(args, "shadow_peft-gen"),
        # save_safetensors=False,
        fp16=args.fp16,
        bf16=args.bf16,
    )
    setattr(training_args, "generation_max_length", getattr(args, "generation_max_length", None))
    metrics = run_generation_trainer(
        model,
        tokenizer,
        training_args,
        datasets,
        trainer_cls=ShadowSFTTrainer,
        trainer_kwargs={"shadow_lr_multiplier": float(getattr(args, "shadow_lr_multiplier", 1.0))},
        save_model=False,  # avoid saving backbone weights via Trainer
    )
    _save_shadow_peft_adapter(model, tokenizer, training_args.output_dir)
    return metrics


def run_generation_task(args: argparse.Namespace) -> Dict[str, Dict[str, float]]:
    tokenizer = prepare_tokenizer(args.model_name)
    datasets = build_datasets(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        tokenizer=tokenizer,
        block_size=args.block_size,
        train_split=args.train_split,
        eval_split=args.eval_split,
        text_column=args.text_column,
        text_template=args.text_template,
    )
    os.makedirs(args.output_dir, exist_ok=True)
    results: Dict[str, Dict[str, float]] = {}
    if args.mode in ("both", "lora"):
        results["lora"] = generation_lora(args, tokenizer, datasets)
    if args.mode in ("both", "shadow"):
        results["shadow_peft"] = generation_shadow(args, tokenizer, datasets)
    summary_path = os.path.join(args.output_dir, "metrics.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"Wrote aggregated metrics to {summary_path}")
    return results


def gsm8k_lora(args, tokenizer, datasets: GSM8KDatasetBundle):
    model = prepare_causal_model(args.model_name, args.attn_implementation)
    _sync_pad_token(model, tokenizer)
    if args.bf16:
        model = model.to(torch.bfloat16)
    elif args.fp16:
        model = model.to(torch.float16)

    lora_config = _build_peft_config(args, task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    _configure_model_no_thinking(model, tokenizer)

    training_args = SFTConfig(
        output_dir=os.path.join(args.output_dir, getattr(args, "peft_method", "lora")),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_accumulation_steps=args.eval_accumulation_steps,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_strategy=_resolve_save_strategy(args),
        save_steps=args.save_steps,
        save_total_limit=1,
        warmup_ratio=args.warmup_ratio,
        report_to=args.report_to,
        run_name=_make_run_name(args, f"{getattr(args, 'peft_method', 'lora')}-gsm8k"),
        # save_safetensors=False,
        fp16=args.fp16,
        bf16=args.bf16,
        max_length=args.max_seq_length,
        remove_unused_columns=False,
    )
    setattr(training_args, "print_shadow_output", bool(getattr(args, "print_shadow_output", False)))
    setattr(training_args, "generation_max_length", getattr(args, "generation_max_length", None))

    collator = GSM8KDataCollator(tokenizer)
    trainer = GSM8KTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets.train_dataset,
        eval_dataset=datasets.eval_dataset,
        data_collator=collator,
        processing_class=tokenizer,
    )
    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(training_args.output_dir)
    return metrics


def gsm8k_shadow(args, tokenizer, datasets: GSM8KDatasetBundle):
    base_model = prepare_causal_model(args.model_name, args.attn_implementation)
    _sync_pad_token(base_model, tokenizer)
    shadow_model = None
    if getattr(args, "pretrained_shadow_model_name", None):
        shadow_model = prepare_causal_model(args.pretrained_shadow_model_name, args.attn_implementation)
        _sync_pad_token(shadow_model, tokenizer)
        if int(getattr(args, "remove_shadow_embed_tokens", 0) or 0) == 1:
            shadow_model = prepare_shadow_model(shadow_model, remove_embed_tokens=True)
    model = _build_shadow_peft_causal_lm(args, base_model, shadow_model=shadow_model)
    if args.bf16:
        model = model.to(torch.bfloat16)
    elif args.fp16:
        model = model.to(torch.float16)
    model.print_trainable_parameters()

    _configure_model_no_thinking(model, tokenizer)

    training_args = SFTConfig(
        output_dir=os.path.join(args.output_dir, "shadow_peft"),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_accumulation_steps=args.eval_accumulation_steps,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_strategy=_resolve_save_strategy(args),
        save_steps=args.save_steps,
        save_total_limit=1,
        warmup_ratio=args.warmup_ratio,
        report_to=args.report_to,
        run_name=_make_run_name(args, "shadow_peft-gsm8k"),
        # save_safetensors=False,
        fp16=args.fp16,
        bf16=args.bf16,
        max_length=args.max_seq_length,
        remove_unused_columns=False,
    )
    setattr(training_args, "print_shadow_output", bool(getattr(args, "print_shadow_output", False)))
    setattr(training_args, "generation_max_length", getattr(args, "generation_max_length", None))

    collator = GSM8KDataCollator(tokenizer)
    trainer = GSM8KTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets.train_dataset,
        eval_dataset=datasets.eval_dataset,
        data_collator=collator,
        processing_class=tokenizer,
        shadow_lr_multiplier=float(getattr(args, "shadow_lr_multiplier", 1.0)),
    )
    trainer.train()
    metrics = trainer.evaluate()
    _save_shadow_peft_adapter(model, tokenizer, training_args.output_dir)
    return metrics


def run_gsm8k_task(args: argparse.Namespace) -> Dict[str, Dict[str, float]]:
    if args.trainer not in ("sft", "both"):
        raise NotImplementedError("GSM8K currently supports only --trainer sft in run_shadow_peft.py.")

    tokenizer = prepare_tokenizer(args.model_name)
    datasets: GSM8KDatasetBundle = build_gsm8k_datasets(
        tokenizer=tokenizer,
        subset=args.gsm8k_subset,
        max_length=args.max_seq_length,
        answer_mode=("final" if getattr(args, "gsm8k_answer_mode", "thinking") == "final" else "thinking"),
    )

    os.makedirs(args.output_dir, exist_ok=True)
    results: Dict[str, Dict[str, float]] = {}
    if args.mode in ("both", "lora"):
        results["lora_sft"] = gsm8k_lora(args, tokenizer, datasets)
    if args.mode in ("both", "shadow"):
        results["shadow_peft_sft"] = gsm8k_shadow(args, tokenizer, datasets)

    summary_path = os.path.join(args.output_dir, "metrics.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"Wrote aggregated metrics to {summary_path}")
    return results


def squad_v2_lora(args, tokenizer, datasets: SquadV2DatasetBundle):
    model = prepare_causal_model(args.model_name, args.attn_implementation)
    _sync_pad_token(model, tokenizer)
    if args.bf16:
        model = model.to(torch.bfloat16)
    elif args.fp16:
        model = model.to(torch.float16)

    lora_config = _build_peft_config(args, task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    _configure_model_no_thinking(model, tokenizer)

    training_args = SFTConfig(
        output_dir=os.path.join(args.output_dir, getattr(args, "peft_method", "lora")),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_accumulation_steps=args.eval_accumulation_steps,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_strategy=_resolve_save_strategy(args),
        save_steps=args.save_steps,
        save_total_limit=1,
        warmup_ratio=args.warmup_ratio,
        report_to=args.report_to,
        run_name=_make_run_name(args, f"{getattr(args, 'peft_method', 'lora')}-squad_v2"),
        # save_safetensors=False,
        fp16=args.fp16,
        bf16=args.bf16,
        max_length=args.max_seq_length,
        remove_unused_columns=False,
    )
    setattr(training_args, "print_shadow_output", bool(getattr(args, "print_shadow_output", False)))
    setattr(training_args, "generation_max_length", getattr(args, "generation_max_length", None))

    collator = SquadV2DataCollator(tokenizer)
    trainer = SquadV2Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets.train_dataset,
        eval_dataset=datasets.eval_dataset,
        data_collator=collator,
        processing_class=tokenizer,
    )
    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(training_args.output_dir)
    return metrics


def squad_v2_shadow(args, tokenizer, datasets: SquadV2DatasetBundle):
    base_model = prepare_causal_model(args.model_name, args.attn_implementation)
    _sync_pad_token(base_model, tokenizer)
    shadow_model = None
    if getattr(args, "pretrained_shadow_model_name", None):
        shadow_model = prepare_causal_model(args.pretrained_shadow_model_name, args.attn_implementation)
        _sync_pad_token(shadow_model, tokenizer)
        if int(getattr(args, "remove_shadow_embed_tokens", 0) or 0) == 1:
            shadow_model = prepare_shadow_model(shadow_model, remove_embed_tokens=True)
    model = _build_shadow_peft_causal_lm(args, base_model, shadow_model=shadow_model)
    if args.bf16:
        model = model.to(torch.bfloat16)
    elif args.fp16:
        model = model.to(torch.float16)
    model.print_trainable_parameters()

    _configure_model_no_thinking(model, tokenizer)

    training_args = SFTConfig(
        output_dir=os.path.join(args.output_dir, "shadow_peft"),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_accumulation_steps=args.eval_accumulation_steps,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_strategy=_resolve_save_strategy(args),
        save_steps=args.save_steps,
        save_total_limit=1,
        warmup_ratio=args.warmup_ratio,
        report_to=args.report_to,
        run_name=_make_run_name(args, "shadow_peft-squad_v2"),
        # save_safetensors=False,
        fp16=args.fp16,
        bf16=args.bf16,
        max_length=args.max_seq_length,
        remove_unused_columns=False,
    )
    setattr(training_args, "print_shadow_output", bool(getattr(args, "print_shadow_output", False)))
    setattr(training_args, "generation_max_length", getattr(args, "generation_max_length", None))

    collator = SquadV2DataCollator(tokenizer)
    trainer = SquadV2Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets.train_dataset,
        eval_dataset=datasets.eval_dataset,
        data_collator=collator,
        processing_class=tokenizer,
        shadow_lr_multiplier=float(getattr(args, "shadow_lr_multiplier", 1.0)),
    )
    trainer.train()
    metrics = trainer.evaluate()
    _save_shadow_peft_adapter(model, tokenizer, training_args.output_dir)
    return metrics


def run_squad_v2_task(args: argparse.Namespace) -> Dict[str, Dict[str, float]]:
    if args.trainer not in ("sft", "both"):
        raise NotImplementedError("SQuAD v2 currently supports only --trainer sft in run_shadow_peft.py.")

    tokenizer = prepare_tokenizer(args.model_name)
    datasets: SquadV2DatasetBundle = build_squad_v2_datasets(
        tokenizer=tokenizer,
        max_length=args.max_seq_length,
        answer_mode=("final" if getattr(args, "squad_answer_mode", "final") == "final" else "thinking"),
    )

    os.makedirs(args.output_dir, exist_ok=True)
    results: Dict[str, Dict[str, float]] = {}
    if args.mode in ("both", "lora"):
        results["lora_sft"] = squad_v2_lora(args, tokenizer, datasets)
    if args.mode in ("both", "shadow"):
        results["shadow_peft_sft"] = squad_v2_shadow(args, tokenizer, datasets)

    summary_path = os.path.join(args.output_dir, "metrics.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"Wrote aggregated metrics to {summary_path}")
    return results


def mmlu_lora(args, tokenizer, datasets: MMLUDatasetBundle):
    model = prepare_causal_model(args.model_name, args.attn_implementation)
    _sync_pad_token(model, tokenizer)
    if args.bf16:
        model = model.to(torch.bfloat16)
    elif args.fp16:
        model = model.to(torch.float16)
    lora_config = _build_peft_config(args, task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    training_args = SFTConfig(
        output_dir=os.path.join(args.output_dir, getattr(args, "peft_method", "lora")),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_accumulation_steps=args.eval_accumulation_steps,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_strategy=_resolve_save_strategy(args),
        save_steps=args.save_steps,
        save_total_limit=1,
        warmup_ratio=args.warmup_ratio,
        report_to=args.report_to,
        run_name=_make_run_name(args, f"{getattr(args, 'peft_method', 'lora')}-mmlu"),
        # save_safetensors=False,
        fp16=args.fp16,
        bf16=args.bf16,
        max_length=args.max_seq_length,
        remove_unused_columns=False,
    )
    setattr(training_args, "print_shadow_output", bool(getattr(args, "print_shadow_output", False)))
    setattr(training_args, "shadow_gen_max_samples", int(getattr(args, "shadow_gen_max_samples", 0)))
    setattr(training_args, "shadow_gen_sample_rate", float(getattr(args, "shadow_gen_sample_rate", 1.0)))
    setattr(training_args, "generation_max_length", getattr(args, "generation_max_length", None))
    data_collator = MMLUDataCollator(tokenizer)
    eval_reference_dataset = _pick_reference_eval_dataset(datasets.eval_datasets)
    trainer = MMLUTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets.train_dataset,
        eval_dataset=eval_reference_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )
    trainer.train()
    metrics = _evaluate_mmlu_subsets(trainer, datasets.eval_datasets)
    trainer.save_model(training_args.output_dir)
    return metrics


def mmlu_shadow(args, tokenizer, datasets: MMLUDatasetBundle):
    base_model = prepare_causal_model(args.model_name, args.attn_implementation)
    _sync_pad_token(base_model, tokenizer)
    shadow_model = None
    if getattr(args, "pretrained_shadow_model_name", None):
        shadow_model = prepare_causal_model(args.pretrained_shadow_model_name, args.attn_implementation)
        _sync_pad_token(shadow_model, tokenizer)
        if int(getattr(args, "remove_shadow_embed_tokens", 0) or 0) == 1:
            shadow_model = prepare_shadow_model(shadow_model, remove_embed_tokens=True)
    model = _build_shadow_peft_causal_lm(args, base_model, shadow_model=shadow_model)
    if args.bf16:
        model = model.to(torch.bfloat16)
    elif args.fp16:
        model = model.to(torch.float16)
    model.print_trainable_parameters()
    training_args = SFTConfig(
        output_dir=os.path.join(args.output_dir, "shadow_peft"),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_accumulation_steps=args.eval_accumulation_steps,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_strategy=_resolve_save_strategy(args),
        save_steps=args.save_steps,
        save_total_limit=1,
        warmup_ratio=args.warmup_ratio,
        report_to=args.report_to,
        run_name=_make_run_name(args, "shadow_peft-mmlu"),
        # save_safetensors=False,
        fp16=args.fp16,
        bf16=args.bf16,
        max_length=args.max_seq_length,
        remove_unused_columns=False,
    )
    setattr(training_args, "print_shadow_output", bool(getattr(args, "print_shadow_output", False)))
    setattr(training_args, "shadow_gen_max_samples", int(getattr(args, "shadow_gen_max_samples", 0)))
    setattr(training_args, "shadow_gen_sample_rate", float(getattr(args, "shadow_gen_sample_rate", 1.0)))
    setattr(training_args, "generation_max_length", getattr(args, "generation_max_length", None))
    data_collator = MMLUDataCollator(tokenizer)
    eval_reference_dataset = _pick_reference_eval_dataset(datasets.eval_datasets)
    trainer = MMLUTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets.train_dataset,
        eval_dataset=eval_reference_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        shadow_lr_multiplier=float(getattr(args, "shadow_lr_multiplier", 1.0)),
    )
    trainer.train()
    metrics = _evaluate_mmlu_subsets(trainer, datasets.eval_datasets)
    _save_shadow_peft_adapter(model, tokenizer, training_args.output_dir)
    return metrics


def run_mmlu_task(args: argparse.Namespace) -> Dict[str, Dict[str, Dict[str, float]]]:
    tokenizer = prepare_tokenizer(args.model_name)
    eval_subsets = _resolve_mmlu_eval_subsets(args.mmlu_subset)
    use_few_shot = getattr(args, "use_few_shot", False)
    datasets = build_mmlu_datasets(
        tokenizer=tokenizer,
        max_length=args.max_seq_length,
        train_subset="auxiliary_train",
        train_split="train",
        eval_subsets=eval_subsets,
        eval_split="test",
        use_few_shot=use_few_shot,
    )
    os.makedirs(args.output_dir, exist_ok=True)
    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    if args.trainer in ("sft", "both"):
        if args.mode in ("both", "lora"):
            results["lora_sft"] = mmlu_lora(args, tokenizer, datasets)
        if args.mode in ("both", "shadow"):
            results["shadow_peft_sft"] = mmlu_shadow(args, tokenizer, datasets)
    summary_path = os.path.join(args.output_dir, "metrics.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"Wrote aggregated metrics to {summary_path}")
    return results


def classification_lora(args, tokenizer, datasets: ClassificationDatasetBundle):
    base_model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(datasets.label2id),
        id2label=datasets.id2label,
        label2id=datasets.label2id,
    )
    _set_attn_impl(base_model, args.attn_implementation)
    _sync_pad_token(base_model, tokenizer)
    if args.bf16:
        base_model = base_model.to(torch.bfloat16)
    elif args.fp16:
        base_model = base_model.to(torch.float16)
    lora_config = _build_peft_config(args, task_type=TaskType.SEQ_CLS)
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    collator = ClassificationCollatorWithPositions(tokenizer)
    metric = evaluate.load(args.classification_metric)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        preds = np.argmax(logits, axis=-1)
        return metric.compute(predictions=preds, references=labels)

    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, getattr(args, "peft_method", "lora")),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_accumulation_steps=args.eval_accumulation_steps,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_strategy=_resolve_save_strategy(args),
        save_steps=args.save_steps,
        save_total_limit=1,
        warmup_ratio=args.warmup_ratio,
        report_to=args.report_to,
        run_name=_make_run_name(args, f"{getattr(args, 'peft_method', 'lora')}-cls"),
        # save_safetensors=False,
        fp16=bool(args.fp16),
        bf16=bool(args.bf16),
        # ShadowPEFT task wrappers use `*args/**kwargs` forwards, so Trainer can't infer
        # which columns are used; keep token columns to avoid collator getting only labels.
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets.train_dataset,
        eval_dataset=datasets.eval_dataset,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(training_args.output_dir)
    return metrics


def classification_shadow(args, tokenizer, datasets: ClassificationDatasetBundle):
    base_model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(datasets.label2id),
        id2label=datasets.id2label,
        label2id=datasets.label2id,
    )
    _set_attn_impl(base_model, args.attn_implementation)
    _sync_pad_token(base_model, tokenizer)
    shadow_model = None
    if getattr(args, "pretrained_shadow_model_name", None):
        try:
            shadow_model = AutoModelForSequenceClassification.from_pretrained(
                args.pretrained_shadow_model_name,
                num_labels=len(datasets.label2id),
                id2label=datasets.id2label,
                label2id=datasets.label2id,
            )
            _set_attn_impl(shadow_model, args.attn_implementation)
        except Exception as e:
            print(f"Error loading shadow model: {e}")
            shadow_model = prepare_causal_model(args.pretrained_shadow_model_name, args.attn_implementation)
        _sync_pad_token(shadow_model, tokenizer)
        if int(getattr(args, "remove_shadow_embed_tokens", 0) or 0) == 1:
            shadow_model = prepare_shadow_model(shadow_model, remove_embed_tokens=True)
    model = _build_shadow_peft_seqcls(args, base_model, shadow_model=shadow_model)
    if args.bf16:
        model = model.to(torch.bfloat16)
    elif args.fp16:
        model = model.to(torch.float16)
    model.peft_model.print_trainable_parameters()  # type: ignore[attr-defined]
    collator = ClassificationCollatorWithPositions(tokenizer)
    metric = evaluate.load(args.classification_metric)
    shadow_metric = evaluate.load(args.classification_metric)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        shadow_logits = None
        if isinstance(logits, (tuple, list)):
            main_logits = logits[0]
            if len(logits) > 1:
                shadow_logits = logits[1]
        else:
            main_logits = logits
        preds = np.argmax(main_logits, axis=-1)
        results = metric.compute(predictions=preds, references=labels)
        if shadow_logits is not None:
            shadow_preds = np.argmax(shadow_logits, axis=-1)
            shadow_results = shadow_metric.compute(predictions=shadow_preds, references=labels)
            results.update({f"shadow_{k}": v for k, v in shadow_results.items()})
        return results

    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, "shadow_peft"),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_accumulation_steps=args.eval_accumulation_steps,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_strategy=_resolve_save_strategy(args),
        save_steps=args.save_steps,
        save_total_limit=1,
        warmup_ratio=args.warmup_ratio,
        report_to=args.report_to,
        run_name=_make_run_name(args, "shadow_peft-cls"),
        # save_safetensors=False,
        fp16=bool(args.fp16),
        bf16=bool(args.bf16),
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets.train_dataset,
        eval_dataset=datasets.eval_dataset,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate()
    _save_shadow_peft_adapter(model, tokenizer, training_args.output_dir)
    return metrics


def run_classification_task(args: argparse.Namespace) -> Dict[str, Dict[str, float]]:
    tokenizer = prepare_tokenizer(args.model_name)
    datasets = build_classification_datasets(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        tokenizer=tokenizer,
        text_column=args.text_column,
        text_pair_column=args.text_pair_column,
        label_column=args.label_column,
        max_seq_length=args.max_seq_length,
        train_split=args.train_split,
        eval_split=args.eval_split,
    )
    os.makedirs(args.output_dir, exist_ok=True)
    results: Dict[str, Dict[str, float]] = {}
    if args.mode in ("both", "lora"):
        results["lora"] = classification_lora(args, tokenizer, datasets)
    if args.mode in ("both", "shadow"):
        results["shadow_peft"] = classification_shadow(args, tokenizer, datasets)
    summary_path = os.path.join(args.output_dir, "metrics.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"Wrote aggregated metrics to {summary_path}")
    return results


def run_suite(args: argparse.Namespace) -> Dict[str, Dict[str, Dict[str, float]]]:
    suite_results: Dict[str, Dict[str, Dict[str, float]]] = {}
    selections: List[tuple[str, List[Dict]]] = []
    if args.suite in ("mmlu", "generation", "all"):
        selections.append(("mmlu", MMLU_SUITE))
    if args.suite in ("gsm8k", "all"):
        selections.append(("gsm8k", GSM8K_SUITE))
    if args.suite in ("squad_v2", "all"):
        selections.append(("squad_v2", [{"id": "squad_v2", "model_name": args.model_name, "max_seq_length": args.max_seq_length}]))
    if args.suite in ("classification", "all"):
        selections.append(("classification", CLASSIFICATION_SUITE))

    for task, specs in selections:
        if task == "classification" and getattr(args, "classification_id", None):
            specs = [s for s in specs if s.get("id") == args.classification_id]
            if not specs:
                raise ValueError(f"Unknown classification_id={args.classification_id}")
        for spec in specs:
            run_args = copy.deepcopy(args)
            run_args.task = task
            for key, value in spec.items():
                if key == "id":
                    continue
                setattr(run_args, key, value)
            run_args.output_dir = os.path.join(args.output_dir, f"{task}_{spec['id']}")
            if args.run_name is not None:
                run_args.run_name = f"{args.run_name}_{spec['id']}"
            print(f"=== Running {task} dataset '{spec['id']}' ===")
            if task == "mmlu":
                result = run_mmlu_task(run_args)
            elif task == "generation":
                result = run_generation_task(run_args)
            elif task == "gsm8k":
                result = run_gsm8k_task(run_args)
            elif task == "squad_v2":
                result = run_squad_v2_task(run_args)
            elif task == "classification":
                result = run_classification_task(run_args)
            else:
                raise NotImplementedError(f"Task not yet implemented in run_shadow_peft.py: {task}")
            suite_results[f"{task}:{spec['id']}"] = result

    summary_path = os.path.join(args.output_dir, "suite_metrics_shadow_peft.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(suite_results, handle, indent=2)
    print(f"Wrote suite metrics to {summary_path}")
    return suite_results


def main():
    args = _normalize_args(parse_args())
    set_seed(args.seed)
    if args.suite is not None:
        run_suite(args)
        return

    if args.task == "mmlu":
        run_mmlu_task(args)
    elif args.task == "generation":
        run_generation_task(args)
    elif args.task == "gsm8k":
        run_gsm8k_task(args)
    elif args.task == "squad_v2":
        run_squad_v2_task(args)
    elif args.task == "classification":
        run_classification_task(args)
    else:
        raise NotImplementedError(
            f"Task not yet implemented in run_shadow_peft.py: {args.task}. "
            "Use run_experiments.py for now."
        )


if __name__ == "__main__":
    main()


