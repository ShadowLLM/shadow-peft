"""
Out-of-Domain (OOD) Evaluation script for ShadowPEFT, LoRA, and DoRA models.

This script evaluates models trained on one task on different tasks to assess
generalization capabilities. Supports few-shot demonstrations with consistent
sampling across different models for fair comparison.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

# Add shadow_peft to path
SHADOW_PEFT_PATH = Path(__file__).parent.parent / "ShadowPEFT" / "src"
if str(SHADOW_PEFT_PATH) not in sys.path:
    sys.path.insert(0, str(SHADOW_PEFT_PATH))

from shadow_peft import ShadowConfig, ShadowForCausalLM

# Import data utilities for consistent task prompts
from data_utils import (
    build_gsm8k_datasets,
    build_mmlu_datasets,
    build_squad_v2_datasets,
    extract_gsm8k_final_answer,
)

# Import trainers and collators from run_experiments
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
    _clean_generated_text,
)

# Regex patterns for extracting boxed answers
_BOXED_ANSWER_RE = re.compile(r"\\boxed\s*\{\s*([^}]*)\s*\}")
_NUMBER_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")


SQUAD_V2_SUITE = [
    {
        "id": "squad_v2",
        "model_name": "Qwen/Qwen3-0.6B",
        "max_seq_length": 512,
    }
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Out-of-Domain evaluation for ShadowPEFT, LoRA, and DoRA models."
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=("mmlu", "gsm8k", "squad_v2"),
        default="mmlu",
        help="Benchmark task to evaluate (OOD task).",
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
        "--mode",
        type=str,
        required=True,
        choices=("shadow", "lora", "dora"),
        help="Model type: 'shadow' (ShadowPEFT), 'lora' (LoRA), or 'dora' (DoRA).",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Checkpoint directory (ShadowPEFT adapter dir for shadow; LoRA/DoRA adapter dir for lora/dora).",
    )
    parser.add_argument(
        "--trained_on",
        type=str,
        choices=("mmlu", "gsm8k", "squad_v2"),
        default=None,
        help="Task the model was trained on (for documentation purposes).",
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
    parser.add_argument("--output_dir", type=str, default="ood_eval_outputs")

    # Few-shot demonstration
    parser.add_argument(
        "--k_shot",
        type=int,
        default=2,
        help="Number of few-shot examples to prepend from training set (default: 2).",
    )
    parser.add_argument(
        "--few_shot_seed",
        type=int,
        default=42,
        help="Seed for sampling few-shot examples (ensures consistency across models).",
    )

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
        choices=("thinking", "final", "boxed"),
        help="Answer format for GSM8K: 'thinking' (standard), 'final' (number only), 'boxed' (with \\boxed format)",
    )
    parser.add_argument(
        "--force_ood_format",
        type=int,
        default=1,
        choices=[0, 1],
        help="For OOD evaluation, force boxed format (default: 1). Set to 0 to use standard format.",
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
    
    # ShadowPEFT-specific
    parser.add_argument(
        "--inference_mode",
        type=str,
        default="base_shadow",
        choices=("base_shadow", "shadow_only"),
        help="Inference mode for ShadowPEFT: 'base_shadow' uses both paths, 'shadow_only' uses only shadow path.",
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


def load_model(
    args: argparse.Namespace,
    tokenizer: AutoTokenizer,
):
    """
    Load model based on mode (shadow, lora, or dora).
    
    Returns:
        Model ready for evaluation.
    """
    # Load base model
    base_model = prepare_base_model(args.model_name, args.attn_implementation)
    
    # Sync pad token
    if tokenizer.pad_token_id != base_model.config.pad_token_id:
        base_model.config.pad_token_id = tokenizer.pad_token_id
        if hasattr(base_model, "generation_config") and base_model.generation_config is not None:
            base_model.generation_config.pad_token_id = tokenizer.pad_token_id
    
    if args.mode == "shadow":
    # Load ShadowPEFT checkpoint
        model = ShadowForCausalLM.from_pretrained(
        base_model,
        args.checkpoint_dir,
        is_trainable=False,
        shadow_loss_weight=args.shadow_loss_weight,
        inference_mode=args.inference_mode,
    )
    elif args.mode in ("lora", "dora"):
        # Load LoRA or DoRA checkpoint
        model = PeftModel.from_pretrained(base_model, args.checkpoint_dir)
        model.eval()
    else:
        raise ValueError(f"Unknown mode: {args.mode}")
    
    # Apply precision
    if args.bf16:
        model = model.to(torch.bfloat16)
    elif args.fp16:
        model = model.to(torch.float16)
    
    return model


def sample_few_shot_examples(dataset, k: int, seed: int) -> List[int]:
    """
    Sample k indices from the dataset for few-shot demonstrations.
    
    Uses a fixed seed to ensure consistency across different model evaluations.
    """
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    sampled_indices = rng.sample(indices, min(k, len(indices)))
    return sorted(sampled_indices)


def format_few_shot_prompt(examples: List[Dict], task: str, tokenizer) -> str:
    """
    Format few-shot examples into a prompt string.
    
    Args:
        examples: List of dataset examples
        task: Task name (mmlu, gsm8k, squad_v2)
        tokenizer: Tokenizer for formatting
    
    Returns:
        Formatted few-shot prompt string
    """
    if not examples:
        return ""
    
    prompt_parts = ["Here are some examples:\n\n"]
    
    for i, example in enumerate(examples, 1):
        if task == "mmlu":
            # For MMLU, show question, choices, and answer
            # Reconstruct from tokenized data (we'll need to decode)
            prompt_parts.append(f"Example {i}:\n")
            # Note: MMLU examples in dataset are already tokenized
            # We'd need the original question/answer, which are stored in dataset
            prompt_parts.append("[MMLU example formatting would go here]\n\n")
            
        elif task == "gsm8k":
            # For GSM8K, show question and answer
            prompt_parts.append(f"Example {i}:\n")
            prompt_parts.append("[GSM8K example formatting would go here]\n\n")
            
        elif task == "squad_v2":
            # For SQuAD v2, show context, question, and answer
            prompt_parts.append(f"Example {i}:\n")
            prompt_parts.append("[SQuAD v2 example formatting would go here]\n\n")
    
    prompt_parts.append("Now, please solve the following:\n\n")
    return "".join(prompt_parts)


def extract_gsm8k_boxed_answer(text: str) -> str:
    """
    Extract answer from boxed format: $\\boxed{answer}$ or $$\\boxed{answer}$$
    
    This is used for OOD evaluation where models are prompted to output:
    - Thinking process
    - Final answer in boxed format
    
    Multiple extraction strategies for robustness:
    1. Look for boxed pattern
    2. Look for "Final Answer:" pattern
    3. Look for "#### " pattern (GSM8K style)
    4. Fallback to last number
    """
    if not text:
        return ""
    
    # Strategy 1: Look for \boxed{...} pattern (single or double $)
    match = _BOXED_ANSWER_RE.search(text)
    if match:
        boxed_content = match.group(1).strip()
        # Extract number from boxed content
        num = _NUMBER_RE.search(boxed_content)
        if num:
            return num.group(0).replace(",", "").strip()
        return boxed_content
    
    # Strategy 2: Look for "Final Answer:" pattern
    final_answer_pattern = re.compile(r"Final Answer:\s*\$?\\boxed\{([^}]+)\}\$?", re.IGNORECASE)
    match = final_answer_pattern.search(text)
    if match:
        answer_content = match.group(1).strip()
        num = _NUMBER_RE.search(answer_content)
        if num:
            return num.group(0).replace(",", "").strip()
        return answer_content
    
    # Strategy 3: Look for "#### " pattern (standard GSM8K)
    gsm8k_pattern = re.compile(r"####\s*([^\n\r]+)")
    match = gsm8k_pattern.search(text)
    if match:
        answer_content = match.group(1).strip()
        num = _NUMBER_RE.search(answer_content)
        if num:
            return num.group(0).replace(",", "").strip()
        return answer_content
    
    # Strategy 4: Fallback - take the last number in the output
    nums = _NUMBER_RE.findall(text)
    if nums:
        return nums[-1].replace(",", "").strip()
    
    return text.strip()


def _patch_gsm8k_extractor_for_boxed_eval():
    """
    Patch GSM8KTrainer to extract boxed answers for OOD evaluation.
    """
    import run_experiments as _rex
    
    original = _rex.extract_gsm8k_answer_from_text
    
    def _boxed_extractor(generated_text: str) -> str:
        # Clean the text first
        cleaned = _clean_generated_text(generated_text)
        # Extract from boxed format
        return extract_gsm8k_boxed_answer(cleaned)
    
    _rex.extract_gsm8k_answer_from_text = _boxed_extractor
    return _rex, original


def _gsm8k_make_boxed_demos(
    *,
    subset: str,
    k_shot: int,
    seed: int,
) -> tuple[str, list[int]]:
    """
    Create a fixed few-shot demonstration prefix for GSM8K with boxed answers.

    We sample k examples from the GSM8K **train** split using `seed` so the same
    demonstrations are reused across different models (fair comparison).
    """
    if k_shot <= 0:
        return "", []

    raw_train = load_dataset(
        "openai/gsm8k", subset, split="train", download_mode="reuse_dataset_if_exists"
    )
    demo_indices = sample_few_shot_examples(raw_train, k_shot, seed)

    parts: list[str] = []
    parts.append(
        "Here are examples showing the required answer format.\n"
    )
    for j, idx in enumerate(demo_indices, 1):
        ex = raw_train[int(idx)]
        q = ex["question"]
        a = ex["answer"]
        thinking_process = re.sub(r"#### .*", '', ex["answer"])
        gold_final = extract_gsm8k_final_answer(a)
        parts.append(f"[Demonstration Example {j}]:\n")
        parts.append(f"Problem:\n{q}\n\n")
        parts.append("Solution:\n")
        parts.append(f"{thinking_process}\n")
        parts.append(f"Final Answer: $\\boxed{{{gold_final}}}$\n\n")

    parts.append(
        "Important: the **final numeric answer** must appear as:\n"
        "Final Answer: $\\boxed{ANSWER}$\n\n"
    )
    return "".join(parts), demo_indices


def build_gsm8k_boxed_dataset_ood(
    tokenizer,
    subset: str,
    split: str,
    max_length: int,
    *,
    k_shot: int = 0,
    few_shot_seed: int = 42,
):
    """
    Build GSM8K dataset for OOD evaluation with boxed answer format.
    
    Prompts the model to output thinking process followed by:
    $$
    \\boxed{answer}
    $$
    """
    raw = load_dataset("openai/gsm8k", subset, split=split, download_mode="reuse_dataset_if_exists")
    demos_text, demo_indices = _gsm8k_make_boxed_demos(
        subset=subset, k_shot=int(k_shot), seed=int(few_shot_seed)
    )

    print(">>> demo text:", demos_text)

    def format_example(example):
        question = example["question"]
        answer = example["answer"]
        gold_final = extract_gsm8k_final_answer(answer)

        # Few-shot prefix + explicit required format.
        user_content = (
            f"{demos_text}"
            f"Problem:\n{question}\n\n"
            "Solution:\n"
        )
        
        user_message = {"role": "user", "content": user_content}
        prompt_text = tokenizer.apply_chat_template(
            [user_message],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        target_text = f"{answer}\n\n Final Answer: $\\boxed{{{gold_final}}}$\n\n"
        im_end_token = getattr(tokenizer, "im_end_token", None)
        if im_end_token is None:
            im_end_token = getattr(tokenizer, "eos_token", "")

        full_text = prompt_text + target_text
        if im_end_token:
            full_text += im_end_token
        if not full_text.endswith("\n"):
            full_text += "\n"

        tokenized = tokenizer(
            full_text,
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        prompt_tokenized = tokenizer(
            prompt_text,
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        
        prompt_length = len(prompt_tokenized["input_ids"])
        labels = [-100] * prompt_length + tokenized["input_ids"][prompt_length:]
        tokenized["labels"] = labels

        tokenized["prompt_input_ids"] = prompt_tokenized["input_ids"]
        tokenized["prompt_attention_mask"] = prompt_tokenized["attention_mask"]
        tokenized["gold_answer"] = str(gold_final).strip()
        return tokenized

    ds = raw.map(
        format_example,
        remove_columns=raw.column_names,
        load_from_cache_file=False,
    )
    # Store demo indices so callers can log them.
    ds = ds.add_column("__few_shot_indices__", [demo_indices] * len(ds))
    return ds


def add_few_shot_to_datasets(train_dataset, eval_dataset, k_shot: int, seed: int, task: str):
    """
    Add few-shot demonstrations to evaluation dataset.
    
    Note: This is a placeholder. The actual implementation would need to:
    1. Sample k examples from train_dataset
    2. Format them as demonstrations
    3. Prepend to each example in eval_dataset
    
    For now, we just return the original datasets and document the few-shot indices.
    """
    if k_shot == 0:
        return train_dataset, eval_dataset, []
    
    few_shot_indices = sample_few_shot_examples(train_dataset, k_shot, seed)
    print(f"[OOD Eval] Using few-shot examples at indices: {few_shot_indices}")
    
    # TODO: Implement actual few-shot prepending
    # This would require modifying the dataset examples to include demonstrations
    
    return train_dataset, eval_dataset, few_shot_indices


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
    model = load_model(args, tokenizer)
    
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
    
    # Add few-shot demonstrations
    train_ds, eval_ref, few_shot_indices = add_few_shot_to_datasets(
        datasets.train_dataset,
        _pick_reference_eval_dataset(datasets.eval_datasets),
        args.k_shot,
        args.few_shot_seed,
        "mmlu"
    )
    
    eval_reference_dataset = eval_ref
    
    trainer = MMLUTrainer(
        model=model,
        args=make_eval_args(args, run_name=f"ood-eval-mmlu-{args.mode}"),
        train_dataset=train_ds,
        eval_dataset=eval_reference_dataset,
        data_collator=MMLUDataCollator(tokenizer),
        processing_class=tokenizer,
    )
    
    results = _evaluate_mmlu_subsets(trainer, datasets.eval_datasets)
    
    # Add metadata
    results["_metadata"] = {
        "mode": args.mode,
        "trained_on": args.trained_on,
        "k_shot": args.k_shot,
        "few_shot_indices": few_shot_indices,
    }
    
    return results


def eval_gsm8k(args: argparse.Namespace) -> Dict[str, float]:
    """Evaluate on GSM8K benchmark."""
    tokenizer = prepare_tokenizer(args.model_name)
    model = load_model(args, tokenizer)
    
    # For OOD evaluation, use boxed answer format (thinking + boxed answer)
    # For in-domain, use the specified answer_mode
    # Can be overridden with --force_ood_format 0
    use_boxed_format = (
        args.trained_on is not None 
        and args.trained_on != "gsm8k" 
        and bool(args.force_ood_format)
    )
    
    if use_boxed_format:
        print(f"[OOD Eval] GSM8K using boxed answer format (trained_on={args.trained_on})")
        # Boxed OOD prompts can trigger repetitive output; if the user didn't specify a
        # generation length, use a smaller default to reduce repetition.
        if args.generation_max_length is None:
            args.generation_max_length = 160
            print("[OOD Eval] generation_max_length was not set; defaulting to 160 for GSM8K boxed OOD")
        # Build custom dataset with boxed format
        eval_dataset = build_gsm8k_boxed_dataset_ood(
            tokenizer=tokenizer,
            subset=args.gsm8k_subset,
            split="test",
            max_length=args.max_seq_length,
            k_shot=int(args.k_shot),
            few_shot_seed=int(args.few_shot_seed),
        )
        # Trainer expects train_dataset; for pure eval we can reuse eval_dataset.
        train_dataset = eval_dataset
        answer_format = "boxed"
        # Demo indices are stored in the dataset for logging/metadata.
        try:
            demo_indices = eval_dataset[0].get("__few_shot_indices__", [])
        except Exception:
            demo_indices = []
    else:
        print(f"[OOD Eval] GSM8K using standard format (trained_on={args.trained_on})")
        answer_mode = ("final" if args.gsm8k_answer_mode == "final" else "thinking")
        datasets = build_gsm8k_datasets(
            tokenizer=tokenizer,
            subset=args.gsm8k_subset,
            max_length=args.max_seq_length,
                answer_mode=answer_mode,
        )
        train_dataset = datasets.train_dataset
        eval_dataset = datasets.eval_dataset
        answer_format = answer_mode
        demo_indices = sample_few_shot_examples(train_dataset, int(args.k_shot), int(args.few_shot_seed)) if int(args.k_shot) > 0 else []
    
    # For GSM8K boxed mode, few-shot demonstrations are embedded in the prompt already.
    # For standard mode, we still don't prepend demos (placeholder), but we record indices for consistency.
    train_ds, eval_ds, few_shot_indices = train_dataset, eval_dataset, demo_indices
    
    # Patch the answer extractor if using boxed format
    if use_boxed_format:
        _rex, _orig = _patch_gsm8k_extractor_for_boxed_eval()
        try:
            trainer = GSM8KTrainer(
                model=model,
                args=make_eval_args(args, run_name=f"ood-eval-gsm8k-{args.mode}"),
                train_dataset=train_ds,
                eval_dataset=eval_ds,
                data_collator=GSM8KDataCollator(tokenizer),
                processing_class=tokenizer,
            )
            results = trainer.evaluate()
        finally:
            # Restore original extractor
            _rex.extract_gsm8k_answer_from_text = _orig
    else:
        trainer = GSM8KTrainer(
            model=model,
                args=make_eval_args(args, run_name=f"ood-eval-gsm8k-{args.mode}"),
                train_dataset=train_ds,
                eval_dataset=eval_ds,
            data_collator=GSM8KDataCollator(tokenizer),
            processing_class=tokenizer,
        )
        results = trainer.evaluate()
    
    # Add metadata
    results["_metadata"] = {
        "mode": args.mode,
        "trained_on": args.trained_on,
        "k_shot": args.k_shot,
        "few_shot_indices": few_shot_indices,
        "answer_format": answer_format,
    }
    
    return results


def eval_squad_v2(args: argparse.Namespace) -> Dict[str, float]:
    """Evaluate on SQuAD v2 benchmark."""
    tokenizer = prepare_tokenizer(args.model_name)
    model = load_model(args, tokenizer)
    
    # For OOD evaluation, always use "final" mode to get concise answers
    answer_mode = "final" if args.trained_on != "squad_v2" else ("final" if args.squad_answer_mode == "final" else "thinking")
    
    print(f"[OOD Eval] SQuAD v2 answer mode: {answer_mode} (trained_on={args.trained_on})")
    
    datasets = build_squad_v2_datasets(
        tokenizer=tokenizer,
        max_length=args.max_seq_length,
        answer_mode=answer_mode,
    )
    
    # Add few-shot demonstrations
    train_ds, eval_ds, few_shot_indices = add_few_shot_to_datasets(
        datasets.train_dataset,
        datasets.eval_dataset,
        args.k_shot,
        args.few_shot_seed,
        "squad_v2"
    )
    
    trainer = SquadV2Trainer(
        model=model,
        args=make_eval_args(args, run_name=f"ood-eval-squad_v2-{args.mode}"),
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=SquadV2DataCollator(tokenizer),
        processing_class=tokenizer,
    )
    
    results = trainer.evaluate()
    
    # Add metadata
    results["_metadata"] = {
        "mode": args.mode,
        "trained_on": args.trained_on,
        "k_shot": args.k_shot,
        "few_shot_indices": few_shot_indices,
        "answer_mode": answer_mode,
    }
    
    return results


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
    out_path = os.path.join(args.output_dir, "ood_metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[OOD Eval] Wrote metrics to {out_path}")
    
    # Also save a summary
    summary = {
        "mode": args.mode,
        "trained_on": args.trained_on,
        "evaluated_on": args.task,
        "k_shot": args.k_shot,
        "checkpoint": args.checkpoint_dir,
        "model": args.model_name,
    }
    summary_path = os.path.join(args.output_dir, "ood_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[OOD Eval] Wrote summary to {summary_path}")
    
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
            
            print(f"=== OOD Evaluating {task} '{spec['id']}' ({args.mode}) ===")
            if args.trained_on:
                print(f"    (Model trained on: {args.trained_on})")
            suite_results[f"{task}:{spec['id']}"] = run_single_task(run_args)
    
    # Save suite results
    out_path = os.path.join(args.output_dir, "ood_suite_metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(suite_results, f, indent=2)
    print(f"[OOD Eval] Wrote suite metrics to {out_path}")
    
    return suite_results


def main():
    args = parse_args()
    set_seed(args.seed)
    
    print("="*60)
    print("Out-of-Domain (OOD) Evaluation")
    print("="*60)
    print(f"Mode: {args.mode}")
    print(f"Trained on: {args.trained_on or 'Not specified'}")
    print(f"Evaluating on: {args.task if not args.suite else f'Suite: {args.suite}'}")
    print(f"K-shot: {args.k_shot}")
    print(f"Checkpoint: {args.checkpoint_dir}")
    print("="*60 + "\n")
    
    if args.suite is not None:
        run_suite(args)
    else:
        run_single_task(args)


if __name__ == "__main__":
    main()
