from __future__ import annotations

import argparse
import copy
import json
import os
import re
from typing import Dict, List, Optional

import torch
from peft import PeftModel
from datasets import load_dataset

# Reuse dataset builders + trainers + model constructors from run_experiments
from run_experiments import (  # noqa: F401
    GSM8K_SUITE,
    MMLU_SUITE,
    DEFAULT_MMLU_GENERATION_TOKENS,
    DEFAULT_GSM8K_GENERATION_TOKENS,
    DEFAULT_SQUAD_V2_GENERATION_TOKENS,
    GSM8KDataCollator,
    MMLUDataCollator,
    SquadV2DataCollator,
    GSM8KTrainer,
    MMLUTrainer,
    SquadV2Trainer,
    ShadowForCausalLM,
    ShadowModel,
    _evaluate_mmlu_subsets,
    _pick_reference_eval_dataset,
    _resolve_mmlu_eval_subsets,
    build_gsm8k_datasets,
    build_mmlu_datasets,
    build_squad_v2_datasets,
    prepare_causal_model,
    prepare_tokenizer,
    _sync_pad_token,
    SFTConfig,
    set_seed,
)

_BOXED_ANSWER_RE = re.compile(r"\\boxed\s*\{\s*([^}]*)\s*\}")
_NUMBER_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")


def extract_gsm8k_final_answer_base(text: str) -> str:
    """
    Base-model GSM8K extractor that prefers LaTeX boxed answers:
      $$ \\boxed{13} $$

    If multiple numbers exist, we take the number inside \\boxed{...}.
    """
    if not text:
        return ""
    match = _BOXED_ANSWER_RE.search(text)
    if match:
        boxed = match.group(1).strip()
        num = _NUMBER_RE.search(boxed)
        if num:
            return num.group(0).replace(",", "").strip()
        return boxed
    # Fallback: take the last number in the output (more robust than first).
    nums = _NUMBER_RE.findall(text)
    if nums:
        return nums[-1].replace(",", "").strip()
    return text.strip()


def _patch_gsm8k_extractor_for_base_eval():
    """
    GSM8KTrainer.evaluation_loop calls `run_experiments.extract_gsm8k_answer_from_text(...)`.
    For base-model eval we want to extract the boxed answer instead.
    """
    import run_experiments as _rex

    original = _rex.extract_gsm8k_answer_from_text

    def _boxed_extractor(generated_text: str) -> str:
        # Keep the same cleaning behavior as run_experiments, then extract boxed answer.
        cleaned = _rex._clean_generated_text(generated_text)
        return extract_gsm8k_final_answer_base(cleaned)

    _rex.extract_gsm8k_answer_from_text = _boxed_extractor
    return _rex, original


SQUAD_V2_SUITE = [
    {
        "id": "squad_v2",
        "model_name": "Qwen/Qwen3-0.6B",
        "max_seq_length": 512,
    }
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate pretrained checkpoints on benchmark datasets.")
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
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B", help="Base model id/path.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=("base", "lora", "shadow"),
        default="base",
        help=(
            "Model type to evaluate. "
            "'base' loads a plain causal LM; "
            "'lora' loads base + PEFT adapter from --checkpoint_dir; "
            "'shadow' builds ShadowForCausalLM and optionally loads weights from --checkpoint_dir."
        ),
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Optional checkpoint directory (adapter dir for lora; full model dir for base/shadow).",
    )
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2", choices=("sdpa", "eager", "flash_attention_2"))
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16)
    parser.add_argument("--eval_accumulation_steps", type=int, default=1)
    parser.add_argument("--bf16", type=int, default=1, choices=[0, 1])
    parser.add_argument("--fp16", type=int, default=0, choices=[0, 1])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="eval_outputs")

    # Task-specific
    parser.add_argument("--mmlu_subset", type=str, default="all")
    parser.add_argument("--use_few_shot", type=int, default=0, choices=[0, 1])
    parser.add_argument("--gsm8k_subset", type=str, default="main", choices=("main", "socratic"))
    parser.add_argument("--gsm8k_answer_mode", type=str, default="thinking", choices=("thinking", "final"))
    parser.add_argument("--squad_answer_mode", type=str, default="final", choices=("thinking", "final"))

    # Generation
    parser.add_argument(
        "--generation_max_length",
        type=int,
        default=None,
        help="Max new tokens to generate during evaluation (overrides task defaults).",
    )

    # Optional: shadow generation printing (reuses same flag shape)
    parser.add_argument("--print_shadow_output", type=int, default=0, choices=[0, 1])
    return parser.parse_args()


def _load_state_dict_from_dir(checkpoint_dir: str) -> Dict[str, torch.Tensor]:
    # run_experiments uses save_safetensors=False, so pytorch_model.bin is typical.
    bin_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
    if os.path.exists(bin_path):
        return torch.load(bin_path, map_location="cpu")
    # fallback for safetensors users
    st_path = os.path.join(checkpoint_dir, "model.safetensors")
    if os.path.exists(st_path):
        from safetensors.torch import load_file  # local import (optional dep)
        return load_file(st_path, device="cpu")
    raise FileNotFoundError(f"Could not find model weights in {checkpoint_dir} (expected pytorch_model.bin or model.safetensors).")


def load_eval_model(args, tokenizer):
    """
    Returns a model ready for evaluation.
    - base: AutoModelForCausalLM from model_name or checkpoint_dir
    - lora: base model_name + adapter from checkpoint_dir
    - shadow: ShadowForCausalLM(base model_name) + optional state dict from checkpoint_dir
    """
    if args.mode == "base":
        model_id = args.checkpoint_dir or args.model_name
        model = prepare_causal_model(model_id, args.attn_implementation)
        _sync_pad_token(model, tokenizer)
        return model

    if args.mode == "lora":
        if not args.checkpoint_dir:
            raise ValueError("--checkpoint_dir is required for --mode lora (adapter directory).")
        base = prepare_causal_model(args.model_name, args.attn_implementation)
        _sync_pad_token(base, tokenizer)
        model = PeftModel.from_pretrained(base, args.checkpoint_dir)
        return model

    if args.mode == "shadow":
        base = prepare_causal_model(args.model_name, args.attn_implementation)
        _sync_pad_token(base, tokenizer)
        shadow_model = ShadowModel(
            base_model=base,
            num_shadow_layers=1,
            intermediate_size=128,
            num_alpha_heads=8,
        )
        model = ShadowForCausalLM(shadow_model=shadow_model)
        if args.checkpoint_dir:
            state = _load_state_dict_from_dir(args.checkpoint_dir)
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing:
                print(f"[run_eval] Warning: missing keys when loading shadow checkpoint (showing up to 20): {missing[:20]}")
            if unexpected:
                print(f"[run_eval] Warning: unexpected keys when loading shadow checkpoint (showing up to 20): {unexpected[:20]}")
        return model

    raise ValueError(f"Unknown mode: {args.mode}")


def _make_eval_args(args: argparse.Namespace, run_name: str) -> SFTConfig:
    eval_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        eval_accumulation_steps=args.eval_accumulation_steps,
        report_to="none",
        run_name=run_name,
        bf16=bool(args.bf16),
        fp16=bool(args.fp16),
        remove_unused_columns=False,
        max_length=args.max_seq_length,
        save_safetensors=False,
    )
    setattr(eval_args, "generation_max_length", getattr(args, "generation_max_length", None))
    setattr(eval_args, "print_shadow_output", bool(getattr(args, "print_shadow_output", 0)))
    return eval_args


def _build_gsm8k_boxed_dataset(
    *,
    tokenizer,
    subset: str,
    split: str,
    max_length: int,
) :
    """
    GSM8K evaluation dataset that prompts the model to output:

    $$
    \\boxed{answer}
    $$
    """
    from data_utils import extract_gsm8k_final_answer  # local import to avoid cycles

    raw = load_dataset("openai/gsm8k", subset, split=split, download_mode="reuse_dataset_if_exists")

    def format_example(example):
        question = example["question"]
        answer = example["answer"]
        gold_final = extract_gsm8k_final_answer(answer)

        user_content = (
            f"Question: {question}\n"
            "Return ONLY the final answer in the exact format:\n"
            "$$\n"
            "\\boxed{answer}\n"
            "$$\n"
            "Answer:"
        )
        user_message = {"role": "user", "content": user_content}
        prompt_text = tokenizer.apply_chat_template(
            [user_message],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        # Supervise with boxed format so eval loss is consistent with the prompt.
        target_text = "$$\n\\boxed{" + str(gold_final).strip() + "}\n$$"

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

    return raw.map(
        format_example,
        remove_columns=raw.column_names,
        load_from_cache_file=False,
    )


def eval_mmlu(args: argparse.Namespace) -> Dict[str, Dict[str, Dict[str, float]]]:
    tokenizer = prepare_tokenizer(args.model_name if args.checkpoint_dir is None else (args.checkpoint_dir if args.mode == "base" else args.model_name))
    model = load_eval_model(args, tokenizer)
    if args.bf16:
        model = model.to(torch.bfloat16)
    elif args.fp16:
        model = model.to(torch.float16)

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
        args=_make_eval_args(args, run_name=f"eval-mmlu-{args.mode}"),
        train_dataset=datasets.train_dataset,
        eval_dataset=eval_reference_dataset,
        data_collator=MMLUDataCollator(tokenizer),
        processing_class=tokenizer,
    )
    return _evaluate_mmlu_subsets(trainer, datasets.eval_datasets)


def eval_gsm8k(args: argparse.Namespace) -> Dict[str, Dict[str, float]]:
    tokenizer = prepare_tokenizer(args.model_name if args.checkpoint_dir is None else (args.checkpoint_dir if args.mode == "base" else args.model_name))
    model = load_eval_model(args, tokenizer)
    if args.bf16:
        model = model.to(torch.bfloat16)
    elif args.fp16:
        model = model.to(torch.float16)

    if args.mode == "base":
        # Special evaluation prompt for base models:
        # force $$\\boxed{answer}$$ output format.
        eval_dataset = _build_gsm8k_boxed_dataset(
            tokenizer=tokenizer,
            subset=args.gsm8k_subset,
            split="test",
            max_length=args.max_seq_length,
        )
        # GSM8KTrainer expects a train_dataset argument; we reuse eval_dataset (no training happens).
        train_dataset = eval_dataset
    else:
        datasets = build_gsm8k_datasets(
            tokenizer=tokenizer,
            subset=args.gsm8k_subset,
            max_length=args.max_seq_length,
            answer_mode=("final" if args.gsm8k_answer_mode == "final" else "thinking"),
        )
        train_dataset = datasets.train_dataset
        eval_dataset = datasets.eval_dataset
    if args.mode == "base":
        _rex, _orig = _patch_gsm8k_extractor_for_base_eval()
        try:
            trainer = GSM8KTrainer(
                model=model,
                args=_make_eval_args(args, run_name=f"eval-gsm8k-{args.mode}"),
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=GSM8KDataCollator(tokenizer),
                processing_class=tokenizer,
            )
            return trainer.evaluate()
        finally:
            _rex.extract_gsm8k_answer_from_text = _orig

    trainer = GSM8KTrainer(
        model=model,
        args=_make_eval_args(args, run_name=f"eval-gsm8k-{args.mode}"),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=GSM8KDataCollator(tokenizer),
        processing_class=tokenizer,
    )
    return trainer.evaluate()


def eval_squad_v2(args: argparse.Namespace) -> Dict[str, Dict[str, float]]:
    tokenizer = prepare_tokenizer(args.model_name if args.checkpoint_dir is None else (args.checkpoint_dir if args.mode == "base" else args.model_name))
    model = load_eval_model(args, tokenizer)
    if args.bf16:
        model = model.to(torch.bfloat16)
    elif args.fp16:
        model = model.to(torch.float16)

    datasets = build_squad_v2_datasets(
        tokenizer=tokenizer,
        max_length=args.max_seq_length,
        answer_mode=("final" if args.squad_answer_mode == "final" else "thinking"),
    )
    trainer = SquadV2Trainer(
        model=model,
        args=_make_eval_args(args, run_name=f"eval-squad_v2-{args.mode}"),
        train_dataset=datasets.train_dataset,
        eval_dataset=datasets.eval_dataset,
        data_collator=SquadV2DataCollator(tokenizer),
        processing_class=tokenizer,
    )
    return trainer.evaluate()


def run_task(args: argparse.Namespace):
    os.makedirs(args.output_dir, exist_ok=True)
    if args.task == "mmlu":
        results = eval_mmlu(args)
    elif args.task == "gsm8k":
        results = eval_gsm8k(args)
    elif args.task == "squad_v2":
        results = eval_squad_v2(args)
    else:
        raise ValueError(f"Unknown task: {args.task}")

    out_path = os.path.join(args.output_dir, "metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[run_eval] Wrote metrics to {out_path}")
    return results


def run_suite(args: argparse.Namespace):
    suite_results: Dict[str, object] = {}
    selections: List[tuple[str, List[Dict]]] = []
    if args.suite in ("mmlu", "all"):
        selections.append(("mmlu", MMLU_SUITE))
    if args.suite in ("gsm8k", "all"):
        selections.append(("gsm8k", GSM8K_SUITE))
    if args.suite in ("squad_v2", "all"):
        selections.append(("squad_v2", SQUAD_V2_SUITE))

    os.makedirs(args.output_dir, exist_ok=True)
    for task, specs in selections:
        for spec in specs:
            run_args = copy.deepcopy(args)
            run_args.task = task
            for k, v in spec.items():
                if k == "id":
                    continue
                setattr(run_args, k, v)
            run_args.output_dir = os.path.join(args.output_dir, f"{task}_{spec['id']}")
            print(f"=== Evaluating {task} '{spec['id']}' ({args.mode}) ===")
            suite_results[f"{task}:{spec['id']}"] = run_task(run_args)

    out_path = os.path.join(args.output_dir, "suite_metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(suite_results, f, indent=2)
    print(f"[run_eval] Wrote suite metrics to {out_path}")
    return suite_results


def main():
    args = parse_args()
    set_seed(args.seed)
    if args.suite is not None:
        run_suite(args)
    else:
        run_task(args)


if __name__ == "__main__":
    main()