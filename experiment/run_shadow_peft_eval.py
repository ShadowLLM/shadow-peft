"""
Evaluation script for ShadowPEFT models on benchmark datasets.

This script uses the shadow_peft library to load and evaluate Shadow-PEFT checkpoints
on MMLU, GSM8K, and SQuAD v2 benchmarks.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

# Add shadow_peft to path
SHADOW_PEFT_PATH = Path(__file__).parent.parent / "ShadowPEFT" / "src"
if str(SHADOW_PEFT_PATH) not in sys.path:
    sys.path.insert(0, str(SHADOW_PEFT_PATH))

from shadow_peft import (
    AutoModelForCausalLMWithHiddenProjection,
    ShadowConfig,
    ShadowForCausalLM,
    prepare_shadow_model,
)

# Import data utilities for consistent task prompts
from data_utils import (
    build_gsm8k_datasets,
    build_mmlu_datasets,
    build_squad_v2_datasets,
)

# Import trainers and collators from run_experiments (reuse existing evaluation logic)
from run_experiments import (
    DEFAULT_GSM8K_GENERATION_TOKENS,
    DEFAULT_MMLU_GENERATION_TOKENS,
    DEFAULT_SQUAD_V2_GENERATION_TOKENS,
    GSM8K_SUITE,
    MMLU_SUITE,
    GSM8KDataCollator,
    GSM8KTrainer,
    MMLUDataCollator,
    MMLUTrainer,
    SquadV2DataCollator,
    SquadV2Trainer,
    _evaluate_mmlu_subsets,
    _pick_reference_eval_dataset,
    _resolve_mmlu_eval_subsets,
    set_seed,
)


SQUAD_V2_SUITE = [
    {
        "id": "squad_v2",
        "model_name": "Qwen/Qwen3-0.6B",
        "max_seq_length": 512,
    }
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate ShadowPEFT checkpoints on benchmark datasets."
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=("mmlu", "gsm8k", "squad_v2"),
        default="mmlu",
        help="Benchmark task to evaluate.",
    )
    parser.add_argument(
        "--suite",
        type=str,
        choices=("mmlu", "gsm8k", "squad_v2", "all"),
        default=None,
        help="Evaluate predefined benchmark suites.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Base model id/path.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="ShadowPEFT checkpoint directory (contains shadow_config.json and shadow_adapter.safetensors).",
    )
    parser.add_argument(
        "--explicit_shadow_model_name_or_path",
        type=str,
        default=None,
        help=(
            "Optional explicit shadow model checkpoint. When provided, this model is passed into "
            "`ShadowForCausalLM.from_pretrained(..., shadow_model=...)` so adapter checkpoints whose shadow "
            "backbone differs from the base model (e.g. 0.6B shadow for an 8B base) can be loaded. "
            "By default, the adapter's saved `shadow_model.*` weights will still be loaded. "
            "Use `--force_use_explicit_shadow_model` to re-apply the explicit shadow weights after loading."
        ),
    )
    parser.add_argument(
        "--force_use_explicit_shadow_model",
        action="store_true",
        help=(
            "If set (and `--explicit_shadow_model_name_or_path` is provided), force the runtime shadow backbone "
            "weights (and projection when compatible) to match the explicit shadow checkpoint, overriding any "
            "`shadow_model.*` weights loaded from the ShadowPEFT adapter."
        ),
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="flash_attention_2",
        choices=("sdpa", "eager", "flash_attention_2"),
    )
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16)
    parser.add_argument("--eval_accumulation_steps", type=int, default=1)
    parser.add_argument("--bf16", type=int, default=1, choices=[0, 1])
    parser.add_argument("--fp16", type=int, default=0, choices=[0, 1])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="eval_outputs")

    # Task-specific arguments
    parser.add_argument("--mmlu_subset", type=str, default="all")
    parser.add_argument("--use_few_shot", type=int, default=0, choices=[0, 1])
    parser.add_argument(
        "--gsm8k_subset",
        type=str,
        default="main",
        choices=("main", "socratic"),
    )
    parser.add_argument(
        "--gsm8k_answer_mode",
        type=str,
        default="thinking",
        choices=("thinking", "final"),
    )
    parser.add_argument(
        "--squad_answer_mode",
        type=str,
        default="final",
        choices=("thinking", "final"),
    )

    # Generation parameters
    parser.add_argument(
        "--generation_max_length",
        type=int,
        default=None,
        help="Max new tokens to generate during evaluation (overrides task defaults).",
    )
    parser.add_argument(
        "--inference_mode",
        type=str,
        default="base_shadow",
        choices=("base_shadow", "shadow_only"),
        help="Inference mode: 'base_shadow' uses both paths, 'shadow_only' uses only shadow path.",
    )
    parser.add_argument(
        "--shadow_loss_weight",
        type=float,
        default=0.05,
        help="Weight for shadow auxiliary loss during evaluation.",
    )

    return parser.parse_args()


def prepare_tokenizer(model_name_or_path: str) -> AutoTokenizer:
    """Load and configure tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def prepare_base_model(model_name: str, attn_implementation: str) -> AutoModelForCausalLM:
    """Load base causal LM model."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
        trust_remote_code=True,
    )
    return model


def load_shadow_peft_model(
    args: argparse.Namespace,
    tokenizer: AutoTokenizer,
) -> ShadowForCausalLM:
    """
    Load a ShadowPEFT model from checkpoint.
    
    Returns:
        ShadowForCausalLM: The loaded model ready for evaluation.
    """
    # Load base model
    base_model = prepare_base_model(args.model_name, args.attn_implementation)
    
    # Sync pad token
    if tokenizer.pad_token_id != base_model.config.pad_token_id:
        base_model.config.pad_token_id = tokenizer.pad_token_id
        if hasattr(base_model, "generation_config") and base_model.generation_config is not None:
            base_model.generation_config.pad_token_id = tokenizer.pad_token_id
    
    # Optional explicit shadow model used to *construct* ShadowPeftModel so adapter loading succeeds
    # even when the adapter's shadow backbone hidden size != base hidden size.
    explicit_shadow = None
    if getattr(args, "explicit_shadow_model_name_or_path", None):
        explicit_path = str(args.explicit_shadow_model_name_or_path)
        # Best-effort dtype choice (base_model is loaded in bf16 in this script).
        explicit_dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
        explicit_shadow = AutoModelForCausalLMWithHiddenProjection.from_pretrained(
            explicit_path,
            torch_dtype=explicit_dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

    # Load ShadowPEFT checkpoint
    shadow_model = ShadowForCausalLM.from_pretrained(
        base_model,
        args.checkpoint_dir,
        is_trainable=False,
        shadow_loss_weight=args.shadow_loss_weight,
        inference_mode=args.inference_mode,
        shadow_model=explicit_shadow,
    )
    
    # Apply precision
    if args.bf16:
        shadow_model = shadow_model.to(torch.bfloat16)
    elif args.fp16:
        shadow_model = shadow_model.to(torch.float16)

    # Optional: force runtime shadow backbone to the explicit shadow checkpoint (override adapter weights).
    # This mirrors the notebook-style "shadow weight replacing" behavior.
    if explicit_shadow is not None and getattr(args, "force_use_explicit_shadow_model", False):
        explicit_backbone = prepare_shadow_model(explicit_shadow, remove_embed_tokens=False)
        try:
            shadow_model.peft_model.shadow_model.load_state_dict(
                explicit_backbone.state_dict(),
                strict=True,
            )
        except RuntimeError:
            # Fallback: remove shadow embeddings so state_dict matches adapters that share base embeddings.
            explicit_backbone_no_embed = prepare_shadow_model(explicit_shadow, remove_embed_tokens=True)
            shadow_model.peft_model.shadow_model.load_state_dict(
                explicit_backbone_no_embed.state_dict(),
                strict=True,
            )

        # Also override shadow->base projection weights when both sides expose a compatible Linear.
        try:
            exp_proj = getattr(explicit_shadow, "shadow_hidden_projection", None)
            tgt_proj = getattr(shadow_model.peft_model, "shadow_hidden_projection", None)
            if (
                isinstance(exp_proj, torch.nn.Linear)
                and isinstance(tgt_proj, torch.nn.Linear)
                and exp_proj.in_features == tgt_proj.in_features
                and exp_proj.out_features == tgt_proj.out_features
            ):
                tgt_proj.load_state_dict(exp_proj.state_dict(), strict=True)
        except Exception:
            pass

    # Avoid holding onto a large explicit model longer than needed.
    explicit_shadow = None
    
    return shadow_model


def make_eval_args(args: argparse.Namespace, run_name: str) -> TrainingArguments:
    """Create TrainingArguments for evaluation."""
    eval_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        eval_accumulation_steps=args.eval_accumulation_steps,
        report_to="none",
        run_name=run_name,
        bf16=bool(args.bf16),
        fp16=bool(args.fp16),
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        dataloader_num_workers=0,
    )
    
    # Add custom attributes for generation
    setattr(eval_args, "generation_max_length", args.generation_max_length)
    setattr(eval_args, "max_length", args.max_seq_length)
    
    return eval_args


def eval_mmlu(args: argparse.Namespace) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Evaluate on MMLU benchmark."""
    tokenizer = prepare_tokenizer(args.model_name)
    model = load_shadow_peft_model(args, tokenizer)
    
    eval_subsets = _resolve_mmlu_eval_subsets(args.mmlu_subset)
    datasets = build_mmlu_datasets(
        tokenizer=tokenizer,
        max_length=args.max_seq_length,
        train_subset="auxiliary_train",
        train_split="train",
        eval_subsets=eval_subsets,
        eval_split="test",
        use_few_shot=bool(args.use_few_shot),
    )
    
    eval_reference_dataset = _pick_reference_eval_dataset(datasets.eval_datasets)
    
    trainer = MMLUTrainer(
        model=model,
        args=make_eval_args(args, run_name=f"eval-mmlu-shadow_peft"),
        train_dataset=datasets.train_dataset,
        eval_dataset=eval_reference_dataset,
        data_collator=MMLUDataCollator(tokenizer),
        processing_class=tokenizer,
    )
    
    return _evaluate_mmlu_subsets(trainer, datasets.eval_datasets)


def eval_gsm8k(args: argparse.Namespace) -> Dict[str, float]:
    """Evaluate on GSM8K benchmark."""
    tokenizer = prepare_tokenizer(args.model_name)
    model = load_shadow_peft_model(args, tokenizer)
    
    datasets = build_gsm8k_datasets(
        tokenizer=tokenizer,
        subset=args.gsm8k_subset,
        max_length=args.max_seq_length,
        answer_mode=("final" if args.gsm8k_answer_mode == "final" else "thinking"),
    )
    
    trainer = GSM8KTrainer(
        model=model,
        args=make_eval_args(args, run_name=f"eval-gsm8k-shadow_peft"),
        train_dataset=datasets.train_dataset,
        eval_dataset=datasets.eval_dataset,
        data_collator=GSM8KDataCollator(tokenizer),
        processing_class=tokenizer,
    )
    
    return trainer.evaluate()


def eval_squad_v2(args: argparse.Namespace) -> Dict[str, float]:
    """Evaluate on SQuAD v2 benchmark."""
    tokenizer = prepare_tokenizer(args.model_name)
    model = load_shadow_peft_model(args, tokenizer)
    
    datasets = build_squad_v2_datasets(
        tokenizer=tokenizer,
        max_length=args.max_seq_length,
        answer_mode=("final" if args.squad_answer_mode == "final" else "thinking"),
    )
    
    trainer = SquadV2Trainer(
        model=model,
        args=make_eval_args(args, run_name=f"eval-squad_v2-shadow_peft"),
        train_dataset=datasets.train_dataset,
        eval_dataset=datasets.eval_dataset,
        data_collator=SquadV2DataCollator(tokenizer),
        processing_class=tokenizer,
    )
    
    return trainer.evaluate()


def run_single_task(args: argparse.Namespace) -> Dict:
    """Run evaluation on a single task."""
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.task == "mmlu":
        results = eval_mmlu(args)
    elif args.task == "gsm8k":
        results = eval_gsm8k(args)
    elif args.task == "squad_v2":
        results = eval_squad_v2(args)
    else:
        raise ValueError(f"Unknown task: {args.task}")
    
    # Save results
    out_path = os.path.join(args.output_dir, "metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[ShadowPEFT Eval] Wrote metrics to {out_path}")
    
    return results


def run_suite(args: argparse.Namespace) -> Dict[str, object]:
    """Run evaluation on a suite of tasks."""
    suite_results: Dict[str, object] = {}
    selections = []
    
    if args.suite in ("mmlu", "all"):
        selections.append(("mmlu", MMLU_SUITE))
    if args.suite in ("gsm8k", "all"):
        selections.append(("gsm8k", GSM8K_SUITE))
    if args.suite in ("squad_v2", "all"):
        selections.append(("squad_v2", SQUAD_V2_SUITE))
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    for task, specs in selections:
        for spec in specs:
            import copy
            run_args = copy.deepcopy(args)
            run_args.task = task
            
            # Apply spec overrides
            for k, v in spec.items():
                if k == "id":
                    continue
                setattr(run_args, k, v)
            
            run_args.output_dir = os.path.join(
                args.output_dir, f"{task}_{spec['id']}"
            )
            
            print(f"=== Evaluating {task} '{spec['id']}' (ShadowPEFT) ===")
            suite_results[f"{task}:{spec['id']}"] = run_single_task(run_args)
    
    # Save suite results
    out_path = os.path.join(args.output_dir, "suite_metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(suite_results, f, indent=2)
    print(f"[ShadowPEFT Eval] Wrote suite metrics to {out_path}")
    
    return suite_results


def main():
    args = parse_args()
    set_seed(args.seed)
    
    if args.suite is not None:
        run_suite(args)
    else:
        run_single_task(args)


if __name__ == "__main__":
    main()
