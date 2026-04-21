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
from transformers.trainer_utils import EvalLoopOutput

# Reuse dataset builders + trainers + model constructors from run_shadow_peft
from run_shadow_peft import (  # noqa: F401
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
    _clean_generated_text,
    extract_answer_from_text,
    generate_from_shadow,
    _shifted_ce_loss,
    ShadowForCausalLM,
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
    GSM8KTrainer.evaluation_loop calls `run_shadow_peft.extract_gsm8k_answer_from_text(...)`.
    For base-model eval we want to extract the boxed answer instead.
    """
    import run_shadow_peft as _rsp

    original = _rsp.extract_gsm8k_answer_from_text

    def _boxed_extractor(generated_text: str) -> str:
        # Keep the same cleaning behavior as run_shadow_peft, then extract boxed answer.
        cleaned = _rsp._clean_generated_text(generated_text)
        return extract_gsm8k_final_answer_base(cleaned)

    _rsp.extract_gsm8k_answer_from_text = _boxed_extractor
    return _rsp, original


SQUAD_V2_SUITE = [
    {
        "id": "squad_v2",
        "max_seq_length": 512,
    }
]


class RunEvalMMLUTrainer(MMLUTrainer):
    """
    Keep run_eval.py on top of run_shadow_peft, but restore the old MMLU
    generation behavior that suppresses explicit thinking blocks.
    """

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
            inputs = self._prepare_inputs(inputs)
            prompt_input_ids = inputs.pop("prompt_input_ids", None)
            prompt_attention_mask = inputs.pop("prompt_attention_mask", None)
            answer_letters = inputs.pop("answer_letters", None)
            answer_indices = inputs.pop("answer_indices", None)

            with torch.no_grad():
                outputs = model(**inputs)
                if outputs.loss is not None:
                    total_loss += outputs.loss.item()

                if getattr(outputs, "shadow_logits", None) is not None:
                    labels = inputs.get("labels")
                    if labels is not None:
                        total_shadow_loss += float(_shifted_ce_loss(outputs.shadow_logits, labels).item())

                input_ids = inputs["input_ids"]
                attention_mask = inputs["attention_mask"]
                batch_size = input_ids.shape[0]

                for i in range(batch_size):
                    if prompt_input_ids is not None and prompt_attention_mask is not None:
                        gen_input_ids = prompt_input_ids[i:i+1, :]
                        gen_attention_mask = prompt_attention_mask[i:i+1, :]
                    else:
                        gen_input_ids = input_ids[i:i+1, :]
                        gen_attention_mask = attention_mask[i:i+1, :]

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
                        gen_kwargs = {
                            "input_ids": gen_input_ids,
                            "attention_mask": gen_attention_mask,
                            "max_new_tokens": max_new_tokens,
                            "do_sample": False,
                            "pad_token_id": pad_token_id,
                            "eos_token_id": eos_token_id,
                            "use_cache": False,
                        }
                        if hasattr(self.processing_class, "thinking_tokens"):
                            gen_kwargs["stop_strings"] = ["<think>", "<|im_start|>think"]
                        generated = actual_model.generate(**gen_kwargs)

                    prompt_length = gen_input_ids.shape[-1]
                    new_tokens = generated[0, prompt_length:]
                    generated_text_raw = self.processing_class.decode(
                        new_tokens,
                        skip_special_tokens=True
                    )
                    generated_text = _clean_generated_text(generated_text_raw)
                    predicted_answer = extract_answer_from_text(generated_text)
                    if predicted_answer:
                        all_predictions.append(predicted_answer.upper())
                    else:
                        all_predictions.append("INVALID")

                    shadow_generated_text = None
                    if do_shadow_gen and hasattr(actual_model, "shadow_lm_head"):
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
                                use_cache=False,
                            )

                            shadow_generated_text_raw = self.processing_class.decode(
                                shadow_generated_ids[0, prompt_length:],
                                skip_special_tokens=True
                            )
                            shadow_generated_text = _clean_generated_text(shadow_generated_text_raw)

                        shadow_predicted = extract_answer_from_text(shadow_generated_text)
                        if shadow_predicted:
                            all_shadow_predictions.append(shadow_predicted.upper())
                        else:
                            all_shadow_predictions.append("INVALID")

                    if answer_letters and i < len(answer_letters) and answer_letters[i] is not None:
                        true_answer = answer_letters[i].upper()
                        all_labels.append(true_answer)
                        if do_shadow_gen and shadow_generated_text is not None:
                            shadow_pred = all_shadow_predictions[-1]
                            shadow_info = f" | shadow_gen: '{shadow_generated_text}' | shadow_pred: {shadow_pred}"
                        else:
                            shadow_info = ""
                        print(f"[Sample {num_samples}] generated: '{generated_text}' | predicted: {predicted_answer} | true: {true_answer}{shadow_info}")
                    else:
                        print(f"[Sample {num_samples}] Warning: No answer_letter found for sample {i}")

                    num_samples += 1

        correct = sum(1 for p, l in zip(all_predictions, all_labels) if p == l)
        accuracy = correct / len(all_labels) if all_labels else 0.0
        avg_loss = total_loss / (step + 1) if step >= 0 else 0.0

        metrics = {
            f"{metric_key_prefix}_loss": avg_loss,
            f"{metric_key_prefix}_accuracy": accuracy,
            f"{metric_key_prefix}_samples": num_samples,
        }

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
        if not args.checkpoint_dir:
            raise ValueError("--checkpoint_dir is required for --mode shadow (checkpoint directory).")
        base = prepare_causal_model(args.model_name, args.attn_implementation)
        _sync_pad_token(base, tokenizer)
        model = ShadowForCausalLM.from_pretrained(
            base,
            args.checkpoint_dir,
            is_trainable=False,
        )
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
        # save_safetensors=False,
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
    trainer = RunEvalMMLUTrainer(
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