from __future__ import annotations

import argparse
import copy
import json
import os
import re
from typing import Dict, List, Optional

import evaluate
import numpy as np
import torch
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
    from trl import SFTConfig, SFTTrainer, GRPOConfig, GRPOTrainer
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError(
        "The `trl` package is required for generation tasks. "
        "Install it with `pip install trl`."
    ) from exc

from data_utils import (
    ClassificationDatasetBundle,
    ExperimentDatasetBundle,
    MMLUDatasetBundle,
    GSM8KDatasetBundle,
    SquadV2DatasetBundle,
    build_classification_datasets,
    build_datasets,
    build_gsm8k_datasets,
    build_mmlu_datasets,
    build_squad_v2_datasets,
    extract_gsm8k_final_answer,
)
from shadow_model import ShadowModel, ShadowForSequenceClassificationModel, ShadowForCausalLM

# MMLU subtasks from lighteval/mmlu dataset
MMLU_SUBTASKS = [
    'abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge',
    'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics',
    'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics',
    'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic',
    'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science',
    'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics',
    'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics',
    'high_school_physics', 'high_school_psychology', 'high_school_statistics',
    'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality',
    'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning',
    'management', 'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes',
    'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting',
    'professional_law', 'professional_medicine', 'professional_psychology', 'public_relations',
    'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions'
]

MMLU_SUITE = [
    {
        "id": "mmlu_full",
        "mmlu_subset": "all",  # Train on auxiliary set, evaluate on every subset
        "max_seq_length": 512,
    }
]

GENERATION_SUITE = MMLU_SUITE  # Use MMLU as the generation suite

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
        "model_name": "Qwen/Qwen3-0.6B",
        "shadow_beta": 4.0,
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
        "model_name": "Qwen/Qwen3-0.6B",
        "shadow_beta": 2.0,
        "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    },
]

DEFAULT_MMLU_GENERATION_TOKENS = 10
DEFAULT_GSM8K_GENERATION_TOKENS = 256
DEFAULT_SQUAD_V2_GENERATION_TOKENS = 32  # 16

# GRPO Reward weights
REWARD_CORRECT_ANSWER = 1.0
REWARD_WRONG_ANSWER = -0.5
REWARD_INVALID_FORMAT = -1.0


def _resolve_mmlu_eval_subsets(requested_subset: Optional[str]) -> List[str]:
    """Return the list of subsets to evaluate on, excluding the 'all' aggregate."""
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
        # If the trainer exposes shadow head metrics, keep them as well.
        # (These come from MMLUTrainer.evaluation_loop when the model has shadow_lm_head.)
        if subset_metrics.get("eval_shadow_loss") is not None:
            entry["shadow_loss"] = subset_metrics.get("eval_shadow_loss")
        if subset_metrics.get("eval_shadow_accuracy") is not None:
            entry["shadow_accuracy"] = subset_metrics.get("eval_shadow_accuracy")
        metrics[subset_name] = entry
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LoRA vs Shadow fine-tuning experiments."
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=("generation", "classification", "mmlu", "gsm8k", "squad_v2"),
        default="mmlu",
        help="Task family to run when --suite is not provided.",
    )
    parser.add_argument(
        "--suite",
        type=str,
        choices=("generation", "classification", "mmlu", "gsm8k", "squad_v2", "all"),
        default=None,
        help="Run the predefined experiment suites.",
    )
    parser.add_argument(
        "--classification_id",
        type=str,
        default=None,
        choices=[spec["id"] for spec in CLASSIFICATION_SUITE],
        help=(
            "When using `--suite classification` (or `--suite all`), run only a single "
            "classification suite entry by id (e.g. `ag_news` or `amazon_reviews_multi_en`)."
        ),
    )
    parser.add_argument(
        "--mmlu_subset",
        type=str,
        default="all",
        help="MMLU subset/subtask to run (e.g., 'abstract_algebra', 'anatomy').",
    )
    parser.add_argument(
        "--gsm8k_subset",
        type=str,
        choices=("main", "socratic"),
        default="main",
        help="GSM8K configuration/subset to use (main or socratic).",
    )
    parser.add_argument(
        "--gsm8k_answer_mode",
        type=str,
        choices=("thinking", "final"),
        default="thinking",
        help="GSM8K mode: 'thinking' trains full solution; 'final' trains direct final answer only.",
    )
    parser.add_argument(
        "--squad_answer_mode",
        type=str,
        choices=("thinking", "final"),
        default="final",
        help="SQuAD v2 mode: 'final' enforces short span/'unanswerable' answers; 'thinking' is more permissive.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Base model checkpoint to fine-tune.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="tiny_shakespeare",
        help="Dataset identifier on the Hugging Face hub.",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default=None,
        help="Optional dataset configuration name.",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default="text",
        help="Column that contains the raw text.",
    )
    parser.add_argument(
        "--text_pair_column",
        type=str,
        default=None,
        help="Optional secondary text column for classification datasets.",
    )
    parser.add_argument(
        "--text_template",
        type=str,
        default=None,
        help=(
            "Python format string to build a synthetic text column. "
            "Example: 'Question: {question}\\nAnswer: {answer}'"
        ),
    )
    parser.add_argument(
        "--label_column",
        type=str,
        default="label",
        help="Label column for classification datasets.",
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=512,
        help="Sequence length used for language modeling chunks.",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="Max sequence length for classification tasks.",
    )
    parser.add_argument(
        "--generation_max_length",
        type=int,
        default=None,
        help=(
            "Max number of new tokens to generate during evaluation. "
            "If unset, task-specific defaults are used."
        ),
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="flash_attention_2",
        choices=("sdpa", "eager", "flash_attention_2"),
        help="Attention backend forced on loaded models to avoid flash-attn issues.",
    )
    parser.add_argument(
        "--per_device_train_batch_size", type=int, default=128, help="Train batch size."
    )
    parser.add_argument(
        "--per_device_eval_batch_size", type=int, default=128, help="Eval batch size."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--eval_accumulation_steps",
        type=int,
        default=1,
        help="Number of eval batches to accumulate on GPU before moving to CPU.",
    )
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="LR.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument(
        "--num_train_epochs", type=int, default=1, help="Training epochs."
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="Override number of training steps. Use -1 to disable.",
    )
    parser.add_argument(
        "--logging_steps", type=int, default=10, help="Logging interval."
    )
    parser.add_argument("--eval_steps", type=int, default=1000, help="Eval interval.")
    parser.add_argument("--save_steps", type=int, default=0, help="Save interval.")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Root directory for experiment artifacts.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help="Integration for logging (e.g., none, tensorboard, wandb).",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="shadow_experiment",
        help="Optional experiment run name (used by trackers like wandb).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--print_shadow_output",
        type=int,
        default=1,
        choices=[0, 1],
        help="0: disable shadow generation during evaluation, 1: enable shadow generation during evaluation.",
    )
    parser.add_argument(
        "--train_split",
        type=str,
        default="train",
        help="Dataset split used for training.",
    )
    parser.add_argument(
        "--eval_split",
        type=str,
        default="validation",
        help="Dataset split used for evaluation.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=("lora", "shadow", "both"),
        help="Which experiments to run.",
    )
    parser.add_argument(
        "--trainer",
        type=str,
        default="sft",
        choices=("sft", "grpo", "both"),
        help="Which trainer to use: sft (Supervised Fine-Tuning), grpo (GRPO RL), or both.",
    )
    parser.add_argument(
        "--fp16",
        type=int,
        default=0,
        choices=[0, 1],
        help="0: disable FP16 training, 1: enable FP16 training.",
    )
    parser.add_argument(
        "--bf16",
        type=int,
        default=1,
        choices=[0, 1],
        help="0: disable BF16 training, 1: enable BF16 training.",
    )
    parser.add_argument(
        "--classification_metric",
        type=str,
        default="accuracy",
        help="Evaluate metric name supported by the `evaluate` library.",
    )

    # LoRA specific
    parser.add_argument(
        "--peft_method",
        type=str,
        default="lora",
        choices=("lora", "dora"),
        help=(
            "PEFT adapter to use when --mode includes 'lora'. "
            "'dora' enables DoRA (Weight-Decomposed LoRA) if supported by your installed `peft`."
        ),
    )
    parser.add_argument("--lora_r", type=int, default=32, help="LoRA rank.")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha.")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout.")
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj"],
        help="Target modules for LoRA adapters.",
    )

    # Shadow specific
    parser.add_argument(
        "--shadow_layers", type=int, default=1, help="Number of shadow layers."
    )
    parser.add_argument(
        "--shadow_intermediate_size",
        type=int,
        default=128,
        help="Intermediate size ratio for shadow MLP (generation).",
    )
    parser.add_argument(
        "--shadow_cls_hidden_ratio",
        type=float,
        default=0.25,
        help="Width ratio for the classification shadow head.",
    )
    parser.add_argument(
        "--num_alpha_heads",
        type=int,
        default=8,
        help=(
            "Number of alpha heads (experts) for ShadowModel routing. "
            "Defaults to 8 to preserve existing behavior."
        ),
    )
    parser.add_argument(
        "--shadow_gate_heads",
        type=int,
        default=None,
        help="Number of heads for the gate MLP in ShadowModel.",
    )
    parser.add_argument(
        "--shadow_loss_weight",
        type=float,
        default=0.05,
        help="Auxiliary shadow-head task loss weight. If <= 0, skip computing shadow task loss.",
    )
    parser.add_argument(
        "--shadow_beta",
        type=float,
        default=1.0,
        help="Global scaling factor for shadow injection/update inside ShadowModel.",
    )
    parser.add_argument(
        "--shadow_dropout",
        type=float,
        default=0.1,
        help="Dropout rate for shadow model.",
    )
    # MMLU specific
    parser.add_argument(
        "--use_few_shot",
        type=int,
        default=0,
        choices=[0, 1],
        help="Use few-shot examples in MMLU prompts to encourage single-letter answers.",
    )

    return parser.parse_args()


def _build_peft_config(args: argparse.Namespace, task_type):
    """
    Build a PEFT config for LoRA / DoRA.

    DoRA support in `peft` has changed API names across versions, so we try a few:
    - LoraConfig(..., use_dora=True)
    - LoraConfig(..., enable_dora=True)
    - LoraConfig(...); then set attribute if present
    """
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

    # Try constructor flags first (most robust)
    for flag in ("use_dora", "enable_dora", "dora"):
        try:
            return LoraConfig(**base_kwargs, **{flag: True})
        except TypeError:
            continue

    # Fallback: set attribute after construction
    cfg = LoraConfig(**base_kwargs)
    for attr in ("use_dora", "enable_dora", "dora"):
        if hasattr(cfg, attr):
            setattr(cfg, attr, True)
            return cfg

    raise RuntimeError(
        "Your installed `peft` does not appear to support DoRA. "
        "Try upgrading: pip install -U peft"
    )


def prepare_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        pad_fallback = tokenizer.eos_token or tokenizer.bos_token
        if pad_fallback is None:
            raise ValueError(
                "Tokenizer is missing pad/eos/bos tokens; please specify a pad token."
            )
        tokenizer.pad_token = pad_fallback
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    return tokenizer


def _set_attn_impl(model, attn_impl: str):
    if hasattr(model, "config"):
        try:
            model.config._attn_implementation = attn_impl
        except AttributeError:
            pass


def prepare_causal_model(model_name: str, attn_impl: str) -> AutoModelForCausalLM:
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, attn_implementation=attn_impl
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        _set_attn_impl(model, attn_impl)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    return model


def _sync_pad_token(model, tokenizer):
    if hasattr(model, "config") and getattr(model.config, "pad_token_id", None) is None:
        if tokenizer.pad_token_id is None:
            raise ValueError(
                "Tokenizer pad_token_id is undefined; unable to set model pad_token_id."
            )
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
        batch["position_ids"] = _build_position_ids(
            attention_mask.long(), self.tokenizer.padding_side
        )
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
        batch["position_ids"] = _build_position_ids(
            attention_mask.long(), self.tokenizer.padding_side
        )
        return batch


class MMLUDataCollator:
    """Data collator for MMLU that preserves answer information."""
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
        labels_list = [f.get("labels", f["input_ids"]) for f in features_for_collation]  # Use existing labels if available
        max_length = max(len(ids) for ids in input_ids_list)
        
        batch_input_ids = []
        batch_labels = []
        for ids, labels in zip(input_ids_list, labels_list):
            # Pad input_ids
            padding_length = max_length - len(ids)
            padded_ids = ids + [self.pad_token_id] * padding_length
            batch_input_ids.append(padded_ids)
            
            # Pad labels (use -100 for padding)
            padded_labels = labels + [-100] * padding_length
            batch_labels.append(padded_labels)
        
        batch = {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }
        
        # Create attention mask
        attention_mask = (batch["input_ids"] != self.pad_token_id).bool()
        batch["attention_mask"] = attention_mask
        
        # Create position_ids
        batch["position_ids"] = _build_position_ids(
            attention_mask.long(), self.tokenizer.padding_side
        )
        
        # Prepare prompt-only inputs (everything up to "Answer: ")
        prompt_input_ids_list = [
            f.get("prompt_input_ids", f["input_ids"]) for f in features
        ]
        prompt_attention_list = []
        for feature, prompt_ids in zip(features, prompt_input_ids_list):
            prompt_mask = feature.get("prompt_attention_mask")
            if prompt_mask is None:
                prompt_mask = [1] * len(prompt_ids)
            elif len(prompt_mask) != len(prompt_ids):
                # Ensure per-sample mask length matches prompt length before batch padding
                if len(prompt_mask) > len(prompt_ids):
                    prompt_mask = prompt_mask[: len(prompt_ids)]
                else:
                    prompt_mask = prompt_mask + [0] * (len(prompt_ids) - len(prompt_mask))
            prompt_attention_list.append(prompt_mask)
        max_prompt_len = max(len(ids) for ids in prompt_input_ids_list)
        padded_prompt_ids = []
        padded_prompt_masks = []
        for ids, mask in zip(prompt_input_ids_list, prompt_attention_list):
            padding_length = max_prompt_len - len(ids)
            padded_prompt_ids.append(ids + [self.pad_token_id] * padding_length)
            padded_prompt_masks.append(mask + [0] * padding_length)
        batch["prompt_input_ids"] = torch.tensor(padded_prompt_ids, dtype=torch.long)
        batch["prompt_attention_mask"] = torch.tensor(
            padded_prompt_masks, dtype=torch.bool
        )
        
        # Add back answer info for evaluation
        batch["answer_letters"] = answer_letters
        batch["answer_indices"] = answer_indices
        
        return batch


class GSM8KDataCollator:
    """Data collator for GSM8K that preserves prompt and gold answer."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, features):
        gold_answers = [f.get("gold_answer", "") for f in features]

        input_ids_list = [f["input_ids"] for f in features]
        attention_list = [
            f.get("attention_mask", [1] * len(f["input_ids"])) for f in features
        ]
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


class SquadV2DataCollator:
    """Data collator for SQuAD v2 that preserves prompt + id/answers for metric eval."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, features):
        squad_ids = [f.get("squad_id", "") for f in features]
        squad_answers = [f.get("squad_answers", {"text": [], "answer_start": []}) for f in features]

        input_ids_list = [f["input_ids"] for f in features]
        attention_list = [
            f.get("attention_mask", [1] * len(f["input_ids"])) for f in features
        ]
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
    """Extract answer letter (A/B/C/D) from generated text."""
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
    """Extract normalized GSM8K final answer from generated text."""
    if not text:
        return ""
    cleaned = _clean_generated_text(text)
    return extract_gsm8k_final_answer(cleaned)


def compute_mmlu_reward(generated_texts: List[str], ground_truth_answers: List[str]) -> List[float]:
    """
    Compute rewards for MMLU generation task.
    
    Args:
        generated_texts: List of generated answer texts
        ground_truth_answers: List of correct answer letters (A/B/C/D)
    
    Returns:
        List of reward scores
    """
    rewards = []
    for gen_text, true_answer in zip(generated_texts, ground_truth_answers):
        # Extract answer more robustly
        try:
            if "Answer: " in gen_text:
                predicted_answer = gen_text.split("Answer: ")[1].strip().upper()
            else:
                predicted_answer = extract_answer_from_text(gen_text)
        except (IndexError, AttributeError):
            predicted_answer = None
        
        if predicted_answer is None or predicted_answer not in ['A', 'B', 'C', 'D']:
            # Invalid format - heavily penalize
            reward = REWARD_INVALID_FORMAT
        elif predicted_answer == true_answer.upper():
            # Correct answer
            reward = REWARD_CORRECT_ANSWER
        else:
            # Wrong answer
            reward = REWARD_WRONG_ANSWER
        
        rewards.append(reward)
    
    return rewards


def generate_from_shadow(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int = 1,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
) -> torch.Tensor:
    """
    Generate tokens using the shadow_lm_head instead of the main lm_head.
    
    Args:
        model: The ShadowForCausalLM model (unwrapped)
        input_ids: Input token ids [batch_size, seq_len]
        attention_mask: Attention mask [batch_size, seq_len]
        max_new_tokens: Maximum number of tokens to generate
        pad_token_id: Padding token id
        eos_token_id: End of sequence token id
    
    Returns:
        Generated token ids [batch_size, seq_len + max_new_tokens]
    """
    generated_ids = input_ids.clone()
    current_attention_mask = attention_mask.clone()
    
    for _ in range(max_new_tokens):
        # Run forward pass to get shadow hidden states
        shadow_outputs = model.shadow_model(
            input_ids=generated_ids,
            attention_mask=current_attention_mask,
        )
        
        # Get shadow logits at the last position
        shadow_hidden = shadow_outputs.last_shadow_hidden_state[:, -1:, :]
        shadow_next_logits = model.shadow_lm_head(shadow_hidden)
        
        # Greedy decode: get the token with highest probability
        next_token = shadow_next_logits.argmax(dim=-1)  # [batch_size, 1]
        
        # Check if we should stop (EOS token generated)
        if eos_token_id is not None and (next_token == eos_token_id).all():
            break
        
        # Append the new token
        generated_ids = torch.cat([generated_ids, next_token], dim=-1)
        
        # Update attention mask
        new_mask = torch.ones((current_attention_mask.shape[0], 1), 
                             dtype=current_attention_mask.dtype, 
                             device=current_attention_mask.device)
        current_attention_mask = torch.cat([current_attention_mask, new_mask], dim=-1)
    
    return generated_ids


class ShadowSFTTrainer(SFTTrainer):
    """SFTTrainer variant that evaluates shadow_logits during evaluation."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Override to handle shadow model outputs."""
        return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)
    
    def evaluation_loop(self, dataloader, description, prediction_loss_only=None, ignore_keys=None, metric_key_prefix="eval"):
        """Override evaluation to also compute shadow loss."""
        model = self._wrap_model(self.model, training=False, dataloader=dataloader)
        model.eval()
        
        total_loss = 0.0
        total_shadow_loss = 0.0
        num_samples = 0
        
        # Check if model has shadow_logits
        has_shadow = hasattr(model, 'shadow_lm_head') or (hasattr(model, 'module') and hasattr(model.module, 'shadow_lm_head'))
        
        for step, inputs in enumerate(dataloader):
            inputs = self._prepare_inputs(inputs)
            
            with torch.no_grad():
                outputs = model(**inputs)
                if outputs.loss is not None:
                    total_loss += outputs.loss.item()
                
                # Compute shadow loss separately if available
                if has_shadow and hasattr(outputs, 'shadow_logits') and outputs.shadow_logits is not None:
                    shadow_logits = outputs.shadow_logits
                    labels = inputs.get("labels")
                    if labels is not None:
                        # Get the actual model (unwrap if needed)
                        actual_model = model.module if hasattr(model, 'module') else model
                        if hasattr(actual_model, 'shadow_model'):
                            # Compute shadow loss using the base model's loss function
                            shadow_loss = actual_model.shadow_model.base_model.loss_function(
                                logits=shadow_logits, 
                                labels=labels, 
                                vocab_size=actual_model.shadow_model.config.vocab_size
                            )
                            total_shadow_loss += shadow_loss.item()
                
                batch_size = inputs["input_ids"].shape[0]
                num_samples += batch_size
        
        avg_loss = total_loss / (step + 1) if step >= 0 else 0.0
        metrics = {
            f"{metric_key_prefix}_loss": avg_loss,
            f"{metric_key_prefix}_samples": num_samples,
        }
        
        # Add shadow metrics if available
        if has_shadow and total_shadow_loss > 0:
            avg_shadow_loss = total_shadow_loss / (step + 1)
            metrics[f"{metric_key_prefix}_shadow_loss"] = avg_shadow_loss
        
        return EvalLoopOutput(
            predictions=None,
            label_ids=None,
            metrics=metrics,
            num_samples=num_samples,
        )


class MMLUTrainer(SFTTrainer):
    """SFTTrainer variant for MMLU with answer extraction and accuracy computation."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Override compute_loss to remove MMLU-specific fields."""
        # Remove MMLU-specific evaluation fields that SFTTrainer doesn't need
        inputs.pop("prompt_input_ids", None)
        inputs.pop("prompt_attention_mask", None)
        inputs.pop("answer_letter", None)
        inputs.pop("answer_idx", None)
        return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)
    
    def evaluation_loop(self, dataloader, description, prediction_loss_only=None, ignore_keys=None, metric_key_prefix="eval"):
        """Override evaluation to compute accuracy on generated answers."""
        model = self._wrap_model(self.model, training=False, dataloader=dataloader)
        model.eval()
        
        all_predictions = []
        all_shadow_predictions = []
        all_labels = []
        total_loss = 0.0
        total_shadow_loss = 0.0
        num_samples = 0
        
        # Check if model has shadow_logits
        actual_model = model.module if hasattr(model, 'module') else model
        has_shadow = hasattr(actual_model, 'shadow_lm_head')
        do_shadow_gen = bool(getattr(self.args, "print_shadow_output", False))
        
        # Determine the compute dtype
        compute_dtype = torch.bfloat16 if self.args.bf16 else (torch.float16 if self.args.fp16 else torch.float32)
        
        max_new_tokens = getattr(self.args, "generation_max_length", None)
        if max_new_tokens is None:
            max_new_tokens = DEFAULT_MMLU_GENERATION_TOKENS
        pad_token_id = getattr(
            self.processing_class, "pad_token_id", getattr(self.model.config, "pad_token_id", None)
        )
        eos_token_id = getattr(
            self.processing_class, "eos_token_id", getattr(self.model.config, "eos_token_id", None)
        )
        if pad_token_id is None and eos_token_id is not None:
            pad_token_id = eos_token_id

        for step, inputs in enumerate(dataloader):
            # Move to device
            inputs = self._prepare_inputs(inputs)
            prompt_input_ids = inputs.pop("prompt_input_ids", None)
            prompt_attention_mask = inputs.pop("prompt_attention_mask", None)
            answer_letters = inputs.pop("answer_letters", None)
            answer_indices = inputs.pop("answer_indices", None)
            
            with torch.no_grad():
                # Compute loss
                outputs = model(**inputs)
                if outputs.loss is not None:
                    total_loss += outputs.loss.item()
                
                # Compute shadow loss separately if available
                if has_shadow and hasattr(outputs, 'shadow_logits') and outputs.shadow_logits is not None:
                    shadow_logits = outputs.shadow_logits
                    labels = inputs.get("labels")
                    if labels is not None:
                        if hasattr(actual_model, 'shadow_model'):
                            # Compute shadow loss using the base model's loss function
                            shadow_loss = actual_model.shadow_model.base_model.loss_function(
                                logits=shadow_logits, 
                                labels=labels, 
                                vocab_size=actual_model.shadow_model.config.vocab_size
                            )
                            total_shadow_loss += shadow_loss.item()
                
                # Generate answer (just the next token after "Answer: ")
                # Find position of "Answer:" in input
                input_ids = inputs["input_ids"]
                attention_mask = inputs["attention_mask"]
                batch_size = input_ids.shape[0]
                
                for i in range(batch_size):
                    # Prepare generation inputs with proper dtype
                    if prompt_input_ids is not None and prompt_attention_mask is not None:
                        gen_input_ids = prompt_input_ids[i:i+1, :]
                        gen_attention_mask = prompt_attention_mask[i:i+1, :]
                    else:
                        gen_input_ids = input_ids[i:i+1, :]
                        gen_attention_mask = attention_mask[i:i+1, :]

                    # IMPORTANT: trim per-sample padding before generation.
                    # Otherwise many samples end with PAD, causing repetitive next-token outputs (e.g., "COPYING").
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
                    
                    # Generate with autocast to ensure proper dtype (limit to the answer token only)
                    with torch.autocast(
                        device_type='cuda',
                        dtype=compute_dtype,
                        enabled=bool(self.args.bf16 or self.args.fp16),
                    ):
                        # Both LoRA and Shadow now support caching
                        # Build generation kwargs
                        gen_kwargs = {
                            "input_ids": gen_input_ids,
                            "attention_mask": gen_attention_mask,
                            "max_new_tokens": max_new_tokens,
                            "do_sample": False,
                            "pad_token_id": pad_token_id,
                            "eos_token_id": eos_token_id,
                            "use_cache": False,
                        }
                        
                        # Add thinking suppression if tokenizer supports it
                        if hasattr(self.processing_class, 'thinking_tokens'):
                            # Stop at thinking tokens if they exist
                            stop_strings = ["<think>", "<|im_start|>think"]
                            gen_kwargs["stop_strings"] = stop_strings
                        
                        generated = model.generate(**gen_kwargs)
                    
                    # Decode only the new tokens (everything after the prompt context)
                    prompt_length = gen_input_ids.shape[-1]
                    new_tokens = generated[0, prompt_length:]
                    generated_text_raw = self.processing_class.decode(
                        new_tokens,
                        skip_special_tokens=True
                    )
                    # Clean thinking blocks and other tags from generated text
                    generated_text = _clean_generated_text(generated_text_raw)
                    # Extract answer
                    predicted_answer = extract_answer_from_text(generated_text)
                    if predicted_answer:
                        all_predictions.append(predicted_answer.upper())
                    else:
                        all_predictions.append("INVALID")
                    
                    # For shadow models, also generate from shadow_logits to compute shadow_accuracy
                    shadow_generated_text = None
                    if has_shadow and do_shadow_gen:
                        # Generate from shadow model
                        with torch.autocast(
                            device_type='cuda',
                            dtype=compute_dtype,
                            enabled=bool(self.args.bf16 or self.args.fp16),
                        ):
                            shadow_generated_ids = generate_from_shadow(
                                model=actual_model,
                                input_ids=gen_input_ids,
                                attention_mask=gen_attention_mask,
                                max_new_tokens=max_new_tokens,
                                pad_token_id=pad_token_id,
                                eos_token_id=eos_token_id,
                            )
                            
                            # Decode only the new tokens (everything after the prompt context)
                            shadow_generated_text_raw = self.processing_class.decode(
                                shadow_generated_ids[0, prompt_length:],
                                skip_special_tokens=True
                            )
                            # Clean thinking blocks and other tags from shadow generated text
                            shadow_generated_text = _clean_generated_text(shadow_generated_text_raw)
                        
                        shadow_predicted = extract_answer_from_text(shadow_generated_text)
                        if shadow_predicted:
                            all_shadow_predictions.append(shadow_predicted.upper())
                        else:
                            all_shadow_predictions.append("INVALID")
                    
                    # Get the true answer label
                    if answer_letters and i < len(answer_letters) and answer_letters[i] is not None:
                        true_answer = answer_letters[i].upper()
                        all_labels.append(true_answer)
                        if has_shadow and do_shadow_gen and shadow_generated_text is not None:
                            shadow_pred = all_shadow_predictions[-1]
                            shadow_info = f" | shadow_gen: '{shadow_generated_text}' | shadow_pred: {shadow_pred}"
                        else:
                            shadow_info = ""
                        print(f"[Sample {num_samples}] generated: '{generated_text}' | predicted: {predicted_answer} | true: {true_answer}{shadow_info}")
                    else:
                        print(f"[Sample {num_samples}] Warning: No answer_letter found for sample {i}")
                    
                    num_samples += 1
        
        # Compute accuracy
        correct = sum(1 for p, l in zip(all_predictions, all_labels) if p == l)
        accuracy = correct / len(all_labels) if all_labels else 0.0
        avg_loss = total_loss / (step + 1) if step >= 0 else 0.0
        
        metrics = {
            f"{metric_key_prefix}_loss": avg_loss,
            f"{metric_key_prefix}_accuracy": accuracy,
            f"{metric_key_prefix}_samples": num_samples,
        }
        
        # Add shadow metrics if available
        if has_shadow:
            if total_shadow_loss > 0:
                avg_shadow_loss = total_shadow_loss / (step + 1)
                metrics[f"{metric_key_prefix}_shadow_loss"] = avg_shadow_loss
            
            if do_shadow_gen and all_shadow_predictions and all_labels:
                shadow_correct = sum(
                    1 for p, l in zip(all_shadow_predictions, all_labels) if p == l
                )
                shadow_accuracy = shadow_correct / len(all_labels)
                metrics[f"{metric_key_prefix}_shadow_accuracy"] = shadow_accuracy
        
        return EvalLoopOutput(
            predictions=all_predictions,
            label_ids=all_labels,
            metrics=metrics,
            num_samples=num_samples,
        )


class GSM8KTrainer(SFTTrainer):
    """SFTTrainer variant for GSM8K with final-answer extraction accuracy."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        inputs.pop("prompt_input_ids", None)
        inputs.pop("prompt_attention_mask", None)
        inputs.pop("gold_answers", None)
        return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

    def evaluation_loop(self, dataloader, description, prediction_loss_only=None, ignore_keys=None, metric_key_prefix="eval"):
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

        compute_dtype = torch.bfloat16 if self.args.bf16 else (torch.float16 if self.args.fp16 else torch.float32)
        max_new_tokens = getattr(self.args, "generation_max_length", None)
        if max_new_tokens is None:
            max_new_tokens = DEFAULT_GSM8K_GENERATION_TOKENS
        pad_token_id = getattr(
            self.processing_class, "pad_token_id", getattr(self.model.config, "pad_token_id", None)
        )
        eos_token_id = getattr(
            self.processing_class, "eos_token_id", getattr(self.model.config, "eos_token_id", None)
        )
        if pad_token_id is None and eos_token_id is not None:
            pad_token_id = eos_token_id

        for step, inputs in enumerate(dataloader):
            inputs = self._prepare_inputs(inputs)
            prompt_input_ids = inputs.pop("prompt_input_ids", None)
            prompt_attention_mask = inputs.pop("prompt_attention_mask", None)
            gold_answers = inputs.pop("gold_answers", None)

            with torch.inference_mode():
                outputs = model(**inputs)
                if outputs.loss is not None:
                    total_loss += outputs.loss.item()
                num_batches += 1

                if has_shadow and hasattr(outputs, "shadow_logits") and outputs.shadow_logits is not None:
                    labels = inputs.get("labels")
                    if labels is not None and hasattr(actual_model, "shadow_model"):
                        shadow_loss = actual_model.shadow_model.base_model.loss_function(
                            logits=outputs.shadow_logits,
                            labels=labels,
                            vocab_size=actual_model.shadow_model.config.vocab_size,
                        )
                        total_shadow_loss += shadow_loss.item()

                if prompt_input_ids is None or prompt_attention_mask is None or gold_answers is None:
                    continue

                batch_size = prompt_input_ids.shape[0]
                for i in range(batch_size):
                    gen_input_ids = prompt_input_ids[i:i+1, :]
                    gen_attention_mask = prompt_attention_mask[i:i+1, :]

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

                    with torch.autocast(
                        device_type="cuda",
                        dtype=compute_dtype,
                        enabled=bool(self.args.bf16 or self.args.fp16),
                    ):
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
                    running_acc = correct / total_answered if total_answered > 0 else 0.0
                    running_loss = total_loss / max(num_batches, 1)

                    shadow_text_raw = None
                    shadow_pred = None
                    running_shadow_acc = None
                    if has_shadow and do_shadow_gen:
                        with torch.autocast(
                            device_type="cuda",
                            dtype=compute_dtype,
                            enabled=bool(self.args.bf16 or self.args.fp16),
                        ):
                            shadow_generated_ids = generate_from_shadow(
                                model=actual_model,
                                input_ids=gen_input_ids,
                                attention_mask=gen_attention_mask,
                                max_new_tokens=max_new_tokens,
                                pad_token_id=pad_token_id,
                                eos_token_id=eos_token_id,
                            )
                        shadow_new_tokens = shadow_generated_ids[0, prompt_length:]
                        shadow_text_raw = self.processing_class.decode(shadow_new_tokens, skip_special_tokens=True)
                        shadow_pred = extract_gsm8k_answer_from_text(shadow_text_raw)
                        if shadow_pred and gold and shadow_pred == gold:
                            shadow_correct += 1
                        running_shadow_acc = shadow_correct / total_answered if total_answered > 0 else 0.0

                    msg = (
                        f"[Sample {num_samples}] "
                        f"acc: {running_acc:.4f} ({correct}/{total_answered}) | "
                        f"loss: {running_loss:.4f} | "
                        f"generated: '{generated_text_raw}' | predicted: {pred} | true: {gold}"
                    )
                    if has_shadow and do_shadow_gen and shadow_text_raw is not None:
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
        if has_shadow and total_shadow_loss > 0 and num_batches > 0:
            metrics[f"{metric_key_prefix}_shadow_loss"] = total_shadow_loss / num_batches

        return EvalLoopOutput(
            predictions=None,
            label_ids=None,
            metrics=metrics,
            num_samples=num_samples,
        )


class SquadV2Trainer(SFTTrainer):
    """SFTTrainer variant for SQuAD v2 using the official EM/F1 metric."""

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

        total_loss = 0.0
        num_batches = 0
        num_samples = 0

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

        for step, inputs in enumerate(dataloader):
            inputs = self._prepare_inputs(inputs)
            prompt_input_ids = inputs.pop("prompt_input_ids", None)
            prompt_attention_mask = inputs.pop("prompt_attention_mask", None)
            squad_ids = inputs.pop("squad_ids", None)
            squad_answers = inputs.pop("squad_answers", None)

            with torch.no_grad():
                outputs = model(**inputs)
                if outputs.loss is not None:
                    total_loss += outputs.loss.item()
                num_batches += 1

            if prompt_input_ids is None or prompt_attention_mask is None or squad_ids is None or squad_answers is None:
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

                with torch.autocast(
                    device_type="cuda",
                    dtype=compute_dtype,
                    enabled=bool(self.args.bf16 or self.args.fp16),
                ):
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
                pred_raw = self.processing_class.decode(new_tokens, skip_special_tokens=True).strip()
                pred_clean = _clean_generated_text(pred_raw)

                # Map "unanswerable" => no-answer.
                if pred_clean.lower() in {"unanswerable", "no answer", "n/a", "none", ""}:
                    prediction_text = ""
                    no_answer_probability = 1.0
                else:
                    prediction_text = pred_clean
                    no_answer_probability = 0.0

                print(f"[Sample {num_samples}] prediction: {prediction_text} | true: {squad_answers[i]}")
                qid = str(squad_ids[i])
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

        return EvalLoopOutput(
            predictions=None,
            label_ids=None,
            metrics=metrics,
            num_samples=num_samples,
        )


def run_generation_trainer(
    model,
    tokenizer,
    training_args: SFTConfig,
    datasets: ExperimentDatasetBundle,
    trainer_cls=SFTTrainer,
    trainer_kwargs: Optional[Dict] = None,
):
    data_collator = LMDataCollatorWithPositions(tokenizer)
    dataset_text_field = (
        "text" if "text" in datasets.train_dataset.column_names else None
    )
    init_kwargs = {
        "model": model,
        "args": training_args,
        "tokenizer": tokenizer,
        "train_dataset": datasets.train_dataset,
        "eval_dataset": datasets.eval_dataset,
        "data_collator": data_collator,
    }
    if dataset_text_field is not None:
        init_kwargs["dataset_text_field"] = dataset_text_field
    if trainer_kwargs:
        init_kwargs.update(trainer_kwargs)
    trainer = trainer_cls(**init_kwargs)
    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(training_args.output_dir)
    return metrics


def _configure_model_no_thinking(model, tokenizer):
    """Configure model and tokenizer to avoid generating thinking tokens."""
    # Set generation config to disable thinking if supported
    if hasattr(model, 'generation_config') and model.generation_config is not None:
        if hasattr(model.generation_config, 'enable_thinking'):
            model.generation_config.enable_thinking = False
        # Add stop tokens for thinking blocks if tokenizer has them
        if hasattr(tokenizer, 'convert_tokens_to_ids'):
            stop_tokens = []
            for token in ['<think>', '<|im_start|>think']:
                token_id = tokenizer.convert_tokens_to_ids(token)
                if isinstance(token_id, int) and token_id != tokenizer.unk_token_id:
                    stop_tokens.append(token_id)
            if stop_tokens and hasattr(model.generation_config, 'eos_token_id'):
                if isinstance(model.generation_config.eos_token_id, list):
                    model.generation_config.eos_token_id.extend(stop_tokens)
                else:
                    model.generation_config.eos_token_id = [model.generation_config.eos_token_id] + stop_tokens


def _make_run_name(args: argparse.Namespace, suffix: str) -> Optional[str]:
    if args.run_name is None:
        return None
    return f"{args.run_name}-{args.dataset_name}-{suffix}"


def generation_lora(args, tokenizer, datasets):
    model = prepare_causal_model(args.model_name, args.attn_implementation)
    _sync_pad_token(model, tokenizer)
    
    # Cast base model to proper dtype before applying LoRA
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
        save_steps=args.save_steps,
        save_total_limit=1,
        warmup_ratio=args.warmup_ratio,
        report_to=args.report_to,
        run_name=_make_run_name(args, f"{getattr(args, 'peft_method', 'lora')}-gen"),
        save_safetensors=False,
        fp16=args.fp16,
        bf16=args.bf16,
    )
    # Inject generation settings for custom evaluation loops (via self.args).
    setattr(training_args, "generation_max_length", getattr(args, "generation_max_length", None))
    return run_generation_trainer(model, tokenizer, training_args, datasets)


def generation_shadow(args, tokenizer, datasets):
    base_model = prepare_causal_model(args.model_name, args.attn_implementation)
    _sync_pad_token(base_model, tokenizer)
    shadow_model = ShadowModel(
        base_model=base_model,
        num_shadow_layers=args.shadow_layers,
        intermediate_size=args.shadow_intermediate_size,
        num_alpha_heads=args.num_alpha_heads,
        shadow_beta=float(getattr(args, "shadow_beta", 1.0)),
        shadow_dropout=float(getattr(args, "shadow_dropout", 0.1)),
        shadow_gate_heads=args.shadow_gate_heads,
    )
    model = ShadowForCausalLM(
        shadow_model=shadow_model,
    )
    # Configure auxiliary shadow-head task loss weight.
    if hasattr(model, "config"):
        setattr(model.config, "shadow_loss_weight", float(getattr(args, "shadow_loss_weight", 0.05)))
    model = model.to(torch.bfloat16)
    model.print_trainable_parameters()

    training_args = SFTConfig(
        output_dir=os.path.join(args.output_dir, "shadow"),
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
        save_steps=args.save_steps,
        save_total_limit=1,
        warmup_ratio=args.warmup_ratio,
        report_to=args.report_to,
        run_name=_make_run_name(args, "shadow-gen"),
        save_safetensors=False,
        fp16=args.fp16,
        bf16=args.bf16,
    )
    # Inject generation settings for custom evaluation loops (via self.args).
    setattr(training_args, "generation_max_length", getattr(args, "generation_max_length", None))
    return run_generation_trainer(
        model,
        tokenizer,
        training_args,
        datasets,
        trainer_cls=ShadowSFTTrainer,
    )


def mmlu_lora(args, tokenizer, datasets: MMLUDatasetBundle):
    """Train LoRA model on MMLU with accuracy evaluation using SFTTrainer."""
    model = prepare_causal_model(args.model_name, args.attn_implementation)
    _sync_pad_token(model, tokenizer)
    
    # Cast base model to proper dtype before applying LoRA
    if args.bf16:
        model = model.to(torch.bfloat16)
    elif args.fp16:
        model = model.to(torch.float16)
    
    lora_config = _build_peft_config(args, task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_config)
    print(">>> Model Architecture:")
    print(model)
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
        save_steps=args.save_steps,
        save_total_limit=1,
        warmup_ratio=args.warmup_ratio,
        report_to=args.report_to,
        run_name=_make_run_name(args, f"{getattr(args, 'peft_method', 'lora')}-mmlu"),
        save_safetensors=False,
        fp16=args.fp16,
        bf16=args.bf16,
        max_length=args.max_seq_length,
        remove_unused_columns=False,  # Keep custom MMLU fields
        # Dataset is pre-tokenized with labels already masked for answer-only training
    )
    # Pass CLI-only flags into the trainer args object (SFTConfig),
    # so custom evaluation_loop logic can see them via `self.args`.
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
    """Train Shadow model on MMLU with accuracy evaluation using SFTTrainer."""
    base_model = prepare_causal_model(args.model_name, args.attn_implementation)
    _sync_pad_token(base_model, tokenizer)
    shadow_model = ShadowModel(
        base_model=base_model,
        num_shadow_layers=args.shadow_layers,
        intermediate_size=args.shadow_intermediate_size,
        num_alpha_heads=args.num_alpha_heads,
        shadow_beta=float(getattr(args, "shadow_beta", 1.0)),
        shadow_dropout=float(getattr(args, "shadow_dropout", 0.1)),
        shadow_gate_heads=args.shadow_gate_heads,
    )
    model = ShadowForCausalLM(shadow_model=shadow_model)
    # Configure auxiliary shadow-head task loss weight.
    if hasattr(model, "config"):
        setattr(model.config, "shadow_loss_weight", float(getattr(args, "shadow_loss_weight", 0.05)))
    model = model.to(torch.bfloat16)
    model.print_trainable_parameters()
    
    # Configure model to avoid generating thinking tokens
    _configure_model_no_thinking(model, tokenizer)
    
    training_args = SFTConfig(
        output_dir=os.path.join(args.output_dir, "shadow"),
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
        save_steps=args.save_steps,
        save_total_limit=1,
        warmup_ratio=args.warmup_ratio,
        report_to=args.report_to,
        run_name=_make_run_name(args, "shadow-mmlu"),
        save_safetensors=False,
        fp16=args.fp16,
        bf16=args.bf16,
        max_length=args.max_seq_length,
        remove_unused_columns=False,  # Keep custom MMLU fields
        # Dataset is pre-tokenized with labels already masked for answer-only training
    )
    # Pass CLI-only flags into the trainer args object (SFTConfig),
    # so custom evaluation_loop logic can see them via `self.args`.
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


def mmlu_lora_grpo(args, tokenizer, datasets: MMLUDatasetBundle):
    """Train LoRA model on MMLU using GRPO (Reinforcement Learning)."""
    model = prepare_causal_model(args.model_name, args.attn_implementation)
    _sync_pad_token(model, tokenizer)
    
    # Cast base model to proper dtype before applying LoRA
    if args.bf16:
        model = model.to(torch.bfloat16)
    elif args.fp16:
        model = model.to(torch.float16)
    
    lora_config = _build_peft_config(args, task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Prepare dataset for GRPO - need prompts and ground truth answers
    train_dataset = datasets.train_dataset.map(
        lambda x: {
            "prompt": tokenizer.decode(x["prompt_input_ids"], skip_special_tokens=True),
            "answer_letter": x["answer_letter"],  # Keep answer for reward function
        },
        remove_columns=[col for col in datasets.train_dataset.column_names 
                       if col not in ["prompt", "answer_letter"]],
    )
    
    def reward_function(completions, answer_letter, **kwargs):
        """
        Compute rewards for MMLU completions.
        
        Args:
            completions: List of generated completion texts
            answer_letter: List of correct answer letters from dataset
            **kwargs: Additional fields from dataset (ignored)
            
        Returns:
            List of reward scores
        """
        # answer_letter comes from the dataset and will be a list
        return compute_mmlu_reward(completions, answer_letter)
    
    training_args = GRPOConfig(
        output_dir=os.path.join(args.output_dir, f"{getattr(args, 'peft_method', 'lora')}_grpo"),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=1,
        warmup_ratio=args.warmup_ratio,
        report_to=args.report_to,
        run_name=_make_run_name(args, f"{getattr(args, 'peft_method', 'lora')}-mmlu-grpo"),
        save_safetensors=False,
        bf16=args.bf16,
        fp16=args.fp16,
    )
    # Inject generation settings for any generation-time behaviors (via self.args).
    setattr(training_args, "generation_max_length", getattr(args, "generation_max_length", None))
    
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_function,  # Correct parameter name
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    
    trainer.train()
    trainer.save_model(training_args.output_dir)
    
    # Evaluate on all subsets
    eval_reference_dataset = _pick_reference_eval_dataset(datasets.eval_datasets)
    metrics = {"grpo_trained": True}  # Placeholder for GRPO metrics
    return metrics


def mmlu_shadow_grpo(args, tokenizer, datasets: MMLUDatasetBundle):
    """Train Shadow model on MMLU using GRPO (Reinforcement Learning)."""
    base_model = prepare_causal_model(args.model_name, args.attn_implementation)
    _sync_pad_token(base_model, tokenizer)
    shadow_model = ShadowModel(
        base_model=base_model,
        num_shadow_layers=args.shadow_layers,
        intermediate_size=args.shadow_intermediate_size,
        num_alpha_heads=args.num_alpha_heads,
        shadow_beta=float(getattr(args, "shadow_beta", 1.0)),
        shadow_dropout=float(getattr(args, "shadow_dropout", 0.1)),
        shadow_gate_heads=args.shadow_gate_heads,
    )
    model = ShadowForCausalLM(shadow_model=shadow_model)
    # Configure auxiliary shadow-head task loss weight.
    if hasattr(model, "config"):
        setattr(model.config, "shadow_loss_weight", float(getattr(args, "shadow_loss_weight", 0.05)))
    model = model.to(torch.bfloat16)
    model.print_trainable_parameters()
    
    # Configure model to avoid generating thinking tokens
    _configure_model_no_thinking(model, tokenizer)
    
    # Prepare dataset for GRPO - need prompts and ground truth answers
    train_dataset = datasets.train_dataset.map(
        lambda x: {
            "prompt": tokenizer.decode(x["prompt_input_ids"], skip_special_tokens=True),
            "answer_letter": x["answer_letter"],  # Keep answer for reward function
        },
        remove_columns=[col for col in datasets.train_dataset.column_names 
                       if col not in ["prompt", "answer_letter"]],
    )
    
    def reward_function(completions, answer_letter, **kwargs):
        """
        Compute rewards for MMLU completions.
        
        Args:
            completions: List of generated completion texts
            answer_letter: List of correct answer letters from dataset
            **kwargs: Additional fields from dataset (ignored)
            
        Returns:
            List of reward scores
        """
        # answer_letter comes from the dataset and will be a list
        return compute_mmlu_reward(completions, answer_letter)
    
    training_args = GRPOConfig(
        output_dir=os.path.join(args.output_dir, "shadow_grpo"),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=1,
        warmup_ratio=args.warmup_ratio,
        report_to=args.report_to,
        run_name=_make_run_name(args, "shadow-mmlu-grpo"),
        save_safetensors=False,
        bf16=args.bf16,
        fp16=args.fp16,
    )
    # Inject generation settings for any generation-time behaviors (via self.args).
    setattr(training_args, "generation_max_length", getattr(args, "generation_max_length", None))
    
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_function,  # Correct parameter name
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    
    trainer.train()
    trainer.save_model(training_args.output_dir)
    
    # Evaluate on all subsets
    metrics = {"grpo_trained": True}  # Placeholder for GRPO metrics
    return metrics


def classification_lora(args, tokenizer, datasets: ClassificationDatasetBundle):
    base_model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(datasets.label2id),
        id2label=datasets.id2label,
        label2id=datasets.label2id,
    )
    _set_attn_impl(base_model, args.attn_implementation)
    _sync_pad_token(base_model, tokenizer)
    
    # Cast base model to proper dtype before applying LoRA
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
        save_steps=args.save_steps,
        save_total_limit=1,
        warmup_ratio=args.warmup_ratio,
        report_to=args.report_to,
        run_name=_make_run_name(args, f"{getattr(args, 'peft_method', 'lora')}-cls"),
        save_safetensors=False,
        fp16=args.fp16,
        bf16=args.bf16,
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
    shadow_model = ShadowModel(
        base_model=base_model,
        num_shadow_layers=args.shadow_layers,
        intermediate_size=args.shadow_intermediate_size,
        num_alpha_heads=args.num_alpha_heads,
        shadow_beta=float(getattr(args, "shadow_beta", 1.0)),
        shadow_dropout=float(getattr(args, "shadow_dropout", 0.1)),
        shadow_gate_heads=args.shadow_gate_heads,
    )
    model = ShadowForSequenceClassificationModel(
        shadow_model=shadow_model,
    )
    # Configure auxiliary shadow-head task loss weight.
    if hasattr(model, "config"):
        setattr(model.config, "shadow_loss_weight", float(getattr(args, "shadow_loss_weight", 0.05)))
    
    # Cast model to proper dtype
    if args.bf16:
        model = model.to(torch.bfloat16)
    elif args.fp16:
        model = model.to(torch.float16)
    
    model.print_trainable_parameters()
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
        output_dir=os.path.join(args.output_dir, "shadow"),
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
        save_steps=args.save_steps,
        save_total_limit=1,
        warmup_ratio=args.warmup_ratio,
        report_to=args.report_to,
        run_name=_make_run_name(args, "shadow-cls"),
        save_safetensors=False,
        fp16=args.fp16,
        bf16=args.bf16,
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
        print("Running LoRA generation experiment...")
        results["lora"] = generation_lora(args, tokenizer, datasets)
    if args.mode in ("both", "shadow"):
        print("Running Shadow generation experiment...")
        results["shadow"] = generation_shadow(args, tokenizer, datasets)

    summary_path = os.path.join(args.output_dir, "metrics.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"Wrote aggregated metrics to {summary_path}")
    return results


def run_gsm8k_task(args: argparse.Namespace) -> Dict[str, Dict[str, float]]:
    """Train on GSM8K train split and evaluate on GSM8K test split."""
    if args.trainer not in ("sft", "both"):
        raise ValueError("GSM8K currently supports SFT training only (set --trainer sft).")

    tokenizer = prepare_tokenizer(args.model_name)
    datasets: GSM8KDatasetBundle = build_gsm8k_datasets(
        tokenizer=tokenizer,
        subset=args.gsm8k_subset,
        max_length=args.max_seq_length,
        answer_mode=("final" if getattr(args, "gsm8k_answer_mode", "thinking") == "final" else "thinking"),
    )
    os.makedirs(args.output_dir, exist_ok=True)
    results: Dict[str, Dict[str, float]] = {}

    collator = GSM8KDataCollator(tokenizer)

    if args.mode in ("both", "lora"):
        base_model = prepare_causal_model(args.model_name, args.attn_implementation)
        _sync_pad_token(base_model, tokenizer)

        if args.bf16:
            base_model = base_model.to(torch.bfloat16)
        elif args.fp16:
            base_model = base_model.to(torch.float16)

        lora_config = _build_peft_config(args, task_type="CAUSAL_LM")
        model = get_peft_model(base_model, lora_config)
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
            save_steps=args.save_steps,
            save_total_limit=1,
            warmup_ratio=args.warmup_ratio,
            report_to=args.report_to,
            run_name=_make_run_name(args, f"{getattr(args, 'peft_method', 'lora')}-gsm8k-{args.gsm8k_subset}"),
            save_safetensors=False,
            fp16=args.fp16,
            bf16=args.bf16,
            max_length=args.max_seq_length,
            remove_unused_columns=False,
        )
        # Pass CLI-only flags into the trainer args object (SFTConfig),
        # so GSM8KTrainer.evaluation_loop can see them via `self.args`.
        setattr(training_args, "print_shadow_output", bool(getattr(args, "print_shadow_output", False)))
        setattr(training_args, "shadow_gen_max_samples", int(getattr(args, "shadow_gen_max_samples", 0)))
        setattr(training_args, "shadow_gen_sample_rate", float(getattr(args, "shadow_gen_sample_rate", 1.0)))
        setattr(training_args, "generation_max_length", getattr(args, "generation_max_length", None))

        trainer = GSM8KTrainer(
            model=model,
            args=training_args,
            train_dataset=datasets.train_dataset,
            eval_dataset=datasets.eval_dataset,
            data_collator=collator,
            processing_class=tokenizer,
        )
        trainer.train()
        results["lora_sft"] = trainer.evaluate()
        trainer.save_model(training_args.output_dir)

    if args.mode in ("both", "shadow"):
        base_model = prepare_causal_model(args.model_name, args.attn_implementation)
        _sync_pad_token(base_model, tokenizer)
        shadow_model = ShadowModel(
            base_model=base_model,
            num_shadow_layers=args.shadow_layers,
            intermediate_size=args.shadow_intermediate_size,
            num_alpha_heads=args.num_alpha_heads,
            shadow_beta=float(getattr(args, "shadow_beta", 1.0)),
            shadow_dropout=float(getattr(args, "shadow_dropout", 0.1)),
            shadow_gate_heads=args.shadow_gate_heads,
        )
        model = ShadowForCausalLM(shadow_model=shadow_model)
        # Configure auxiliary shadow-head task loss weight.
        if hasattr(model, "config"):
            setattr(model.config, "shadow_loss_weight", float(getattr(args, "shadow_loss_weight", 0.05)))
        if args.bf16:
            model = model.to(torch.bfloat16)
        elif args.fp16:
            model = model.to(torch.float16)
        model.print_trainable_parameters()

        training_args = SFTConfig(
            output_dir=os.path.join(args.output_dir, "shadow"),
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
            save_steps=args.save_steps,
            save_total_limit=1,
            warmup_ratio=args.warmup_ratio,
            report_to=args.report_to,
            run_name=_make_run_name(args, f"shadow-gsm8k-{args.gsm8k_subset}"),
            save_safetensors=False,
            fp16=args.fp16,
            bf16=args.bf16,
            max_length=args.max_seq_length,
            remove_unused_columns=False,
        )
        # Pass CLI-only flags into the trainer args object (SFTConfig),
        # so GSM8KTrainer.evaluation_loop can see them via `self.args`.
        setattr(training_args, "print_shadow_output", bool(getattr(args, "print_shadow_output", False)))
        setattr(training_args, "shadow_gen_max_samples", int(getattr(args, "shadow_gen_max_samples", 0)))
        setattr(training_args, "shadow_gen_sample_rate", float(getattr(args, "shadow_gen_sample_rate", 1.0)))
        setattr(training_args, "generation_max_length", getattr(args, "generation_max_length", None))

        trainer = GSM8KTrainer(
            model=model,
            args=training_args,
            train_dataset=datasets.train_dataset,
            eval_dataset=datasets.eval_dataset,
            data_collator=collator,
            processing_class=tokenizer,
        )
        trainer.train()
        results["shadow_sft"] = trainer.evaluate()
        trainer.save_model(training_args.output_dir)

    summary_path = os.path.join(args.output_dir, "metrics.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"Wrote aggregated metrics to {summary_path}")
    return results


def run_mmlu_task(args: argparse.Namespace) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Run MMLU task with accuracy-based evaluation."""
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
    results: Dict[str, Dict[str, float]] = {}
    
    # Run SFT training if requested
    if args.trainer in ("sft", "both"):
        if args.mode in ("both", "lora"):
            print(f"Running LoRA MMLU SFT experiment on {args.mmlu_subset}...")
            results["lora_sft"] = mmlu_lora(args, tokenizer, datasets)
        
        if args.mode in ("both", "shadow"):
            print(f"Running Shadow MMLU SFT experiment on {args.mmlu_subset}...")
            results["shadow_sft"] = mmlu_shadow(args, tokenizer, datasets)
    
    # Run GRPO training if requested
    if args.trainer in ("grpo", "both"):
        if args.mode in ("both", "lora"):
            print(f"Running LoRA MMLU GRPO experiment on {args.mmlu_subset}...")
            results["lora_grpo"] = mmlu_lora_grpo(args, tokenizer, datasets)
        
        if args.mode in ("both", "shadow"):
            print(f"Running Shadow MMLU GRPO experiment on {args.mmlu_subset}...")
            results["shadow_grpo"] = mmlu_shadow_grpo(args, tokenizer, datasets)

    summary_path = os.path.join(args.output_dir, "metrics.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"Wrote aggregated metrics to {summary_path}")

    # Also write shadow_lm_head-only metrics (if present) for easier analysis.
    shadow_only: Dict[str, Dict[str, Dict[str, float]]] = {}
    for run_key, run_metrics in results.items():
        if not isinstance(run_metrics, dict):
            continue
        for subset_name, subset_metrics in run_metrics.items():
            if not isinstance(subset_metrics, dict):
                continue
            if ("shadow_loss" in subset_metrics) or ("shadow_accuracy" in subset_metrics):
                shadow_only.setdefault(run_key, {})[subset_name] = {
                    k: v
                    for k, v in subset_metrics.items()
                    if k in ("shadow_loss", "shadow_accuracy", "samples")
                }
    shadow_path = os.path.join(args.output_dir, "shadow_metrics.json")
    with open(shadow_path, "w", encoding="utf-8") as handle:
        json.dump(shadow_only, handle, indent=2)
    print(f"Wrote shadow metrics to {shadow_path}")
    return results


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
        print("Running LoRA classification experiment...")
        results["lora"] = classification_lora(args, tokenizer, datasets)
    if args.mode in ("both", "shadow"):
        print("Running Shadow classification experiment...")
        results["shadow"] = classification_shadow(args, tokenizer, datasets)
    summary_path = os.path.join(args.output_dir, "metrics.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"Wrote aggregated metrics to {summary_path}")
    return results


def run_squad_v2_task(args: argparse.Namespace) -> Dict[str, Dict[str, float]]:
    """Train on rajpurkar/squad_v2 train split and evaluate on validation split."""
    if args.trainer not in ("sft", "both"):
        raise ValueError("SQuAD v2 currently supports SFT training only (set --trainer sft).")

    tokenizer = prepare_tokenizer(args.model_name)
    datasets: SquadV2DatasetBundle = build_squad_v2_datasets(
        tokenizer=tokenizer,
        max_length=args.max_seq_length,
        answer_mode=("final" if getattr(args, "squad_answer_mode", "final") == "final" else "thinking"),
    )
    os.makedirs(args.output_dir, exist_ok=True)
    results: Dict[str, Dict[str, float]] = {}

    collator = SquadV2DataCollator(tokenizer)

    if args.mode in ("both", "lora"):
        base_model = prepare_causal_model(args.model_name, args.attn_implementation)
        _sync_pad_token(base_model, tokenizer)

        if args.bf16:
            base_model = base_model.to(torch.bfloat16)
        elif args.fp16:
            base_model = base_model.to(torch.float16)

        lora_config = _build_peft_config(args, task_type="CAUSAL_LM")
        model = get_peft_model(base_model, lora_config)
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
            save_steps=args.save_steps,
            save_total_limit=1,
            warmup_ratio=args.warmup_ratio,
            report_to=args.report_to,
            run_name=_make_run_name(args, f"{getattr(args, 'peft_method', 'lora')}-squad_v2"),
            save_safetensors=False,
            fp16=args.fp16,
            bf16=args.bf16,
            max_length=args.max_seq_length,
            remove_unused_columns=False,
        )
        setattr(training_args, "print_shadow_output", bool(getattr(args, "print_shadow_output", False)))
        setattr(training_args, "shadow_gen_max_samples", int(getattr(args, "shadow_gen_max_samples", 0)))
        setattr(training_args, "shadow_gen_sample_rate", float(getattr(args, "shadow_gen_sample_rate", 1.0)))
        setattr(training_args, "generation_max_length", getattr(args, "generation_max_length", None))

        trainer = SquadV2Trainer(
            model=model,
            args=training_args,
            train_dataset=datasets.train_dataset,
            eval_dataset=datasets.eval_dataset,
            data_collator=collator,
            processing_class=tokenizer,
        )
        trainer.train()
        results["lora_sft"] = trainer.evaluate()
        trainer.save_model(training_args.output_dir)

    if args.mode in ("both", "shadow"):
        base_model = prepare_causal_model(args.model_name, args.attn_implementation)
        _sync_pad_token(base_model, tokenizer)
        shadow_model = ShadowModel(
            base_model=base_model,
            num_shadow_layers=args.shadow_layers,
            intermediate_size=args.shadow_intermediate_size,
            num_alpha_heads=args.num_alpha_heads,
            shadow_beta=float(getattr(args, "shadow_beta", 1.0)),
            shadow_dropout=float(getattr(args, "shadow_dropout", 0.1)),
            shadow_gate_heads=args.shadow_gate_heads,
        )
        model = ShadowForCausalLM(shadow_model=shadow_model)
        if hasattr(model, "config"):
            setattr(model.config, "shadow_loss_weight", float(getattr(args, "shadow_loss_weight", 0.05)))
        if args.bf16:
            model = model.to(torch.bfloat16)
        elif args.fp16:
            model = model.to(torch.float16)
        model.print_trainable_parameters()

        training_args = SFTConfig(
            output_dir=os.path.join(args.output_dir, "shadow"),
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
            save_steps=args.save_steps,
            save_total_limit=1,
            warmup_ratio=args.warmup_ratio,
            report_to=args.report_to,
            run_name=_make_run_name(args, "shadow-squad_v2"),
            save_safetensors=False,
            fp16=args.fp16,
            bf16=args.bf16,
            max_length=args.max_seq_length,
            remove_unused_columns=False,
        )
        setattr(training_args, "print_shadow_output", bool(getattr(args, "print_shadow_output", False)))
        setattr(training_args, "shadow_gen_max_samples", int(getattr(args, "shadow_gen_max_samples", 0)))
        setattr(training_args, "shadow_gen_sample_rate", float(getattr(args, "shadow_gen_sample_rate", 1.0)))
        setattr(training_args, "generation_max_length", getattr(args, "generation_max_length", None))

        trainer = SquadV2Trainer(
            model=model,
            args=training_args,
            train_dataset=datasets.train_dataset,
            eval_dataset=datasets.eval_dataset,
            data_collator=collator,
            processing_class=tokenizer,
        )
        trainer.train()
        results["shadow_sft"] = trainer.evaluate()
        trainer.save_model(training_args.output_dir)

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
                raise ValueError(
                    f"Unknown classification_id={args.classification_id}. "
                    f"Valid: {[s['id'] for s in CLASSIFICATION_SUITE]}"
                )
        for spec in specs:
            run_args = copy.deepcopy(args)
            run_args.task = task
            for key, value in spec.items():
                if key == "id":
                    continue
                setattr(run_args, key, value)
            run_args.output_dir = os.path.join(args.output_dir, f"{task}_{spec['id']}")
            # Ensure unique run_name by incorporating the dataset id
            if args.run_name is not None:
                run_args.run_name = f"{args.run_name}_{spec['id']}"
            print(f"=== Running {task} dataset '{spec['id']}' ===")
            if task == "mmlu":
                result = run_mmlu_task(run_args)
            elif task == "gsm8k":
                result = run_gsm8k_task(run_args)
            elif task == "squad_v2":
                result = run_squad_v2_task(run_args)
            elif task == "generation":
                result = run_generation_task(run_args)
            else:
                result = run_classification_task(run_args)
            suite_results[f"{task}:{spec['id']}"] = result

    summary_path = os.path.join(args.output_dir, "suite_metrics.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(suite_results, handle, indent=2)
    print(f"Wrote suite metrics to {summary_path}")
    return suite_results


def main():
    args = parse_args()
    set_seed(args.seed)
    if args.suite is not None:
        run_suite(args)
        return

    if args.task == "mmlu":
        run_mmlu_task(args)
    elif args.task == "gsm8k":
        run_gsm8k_task(args)
    elif args.task == "squad_v2":
        run_squad_v2_task(args)
    elif args.task == "generation":
        run_generation_task(args)
    else:
        run_classification_task(args)


if __name__ == "__main__":
    main()
