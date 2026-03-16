from __future__ import annotations

"""
Standalone **shadow model pretraining** script.

Goal
----
Pretrain a provided Hugging Face **transformer causal LM** on domain datasets
(GSM8K / SQuAD v2 / MMLU / AG News / Amazon Reviews) using **next-token prediction**
on a unified text format.

Important behavior
------------------
- We **do not build a new shadow model architecture** in this script.
  You provide the model checkpoint/path, and we load it with `AutoModelForCausalLM`.
- We keep `embed_tokens` and `lm_head` **frozen** during pretraining so they remain
  consistent (e.g. if you exported a shadow model whose embeddings/head were copied
  from a backbone model).
"""

import argparse
import copy
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import concatenate_datasets, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

from data_utils import (
    extract_gsm8k_final_answer,
    tokenize_and_group,
)


# ---- suite specs (mirrors run_shadow_peft.py) ----
MMLU_SUITE = [{"id": "mmlu_full", "mmlu_subset": "all", "max_seq_length": 512}]
GSM8K_SUITE = [{"id": "gsm8k_main", "gsm8k_subset": "main", "max_seq_length": 512}]
SQUAD_V2_SUITE = [{"id": "squad_v2", "max_seq_length": 512}]
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
    },
]


def _prepare_tokenizer(model_name: str):
    # Some shadow-only model repos may have incomplete/invalid configs; always allow the
    # caller to point tokenization at a known-good tokenizer repo.
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.bos_token
    if tok.pad_token_id is None and tok.pad_token is not None:
        tok.pad_token_id = tok.convert_tokens_to_ids(tok.pad_token)
    return tok


def _sync_pad_token(model, tokenizer) -> None:
    if hasattr(model, "config") and getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id


def _freeze_module(module) -> None:
    if module is None:
        return
    for p in module.parameters():
        p.requires_grad = False


def _freeze_embeddings_and_lm_head(model) -> None:
    _freeze_module(model.get_input_embeddings())
    _freeze_module(model.get_output_embeddings())


def _freeze_embeddings_only(model) -> None:
    _freeze_module(model.get_input_embeddings())


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


class CausalLMDataCollatorWithPositions:
    """
    Pads `input_ids/attention_mask` and preserves existing `labels` (pads with -100).
    Also adds `position_ids` for models that need it.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict]):
        # Keep labels separate because tokenizer.pad won't pad them.
        labels_list = [f.get("labels") for f in features]
        to_pad = []
        for f in features:
            to_pad.append({k: v for k, v in f.items() if k in ("input_ids", "attention_mask")})
        batch = self.tokenizer.pad(to_pad, return_tensors="pt")

        if "attention_mask" not in batch:
            batch["attention_mask"] = (batch["input_ids"] != self.tokenizer.pad_token_id).long()

        max_len = batch["input_ids"].shape[1]
        padded_labels = []
        for lab in labels_list:
            if lab is None:
                # Fall back to LM-style labels if dataset didn't provide them.
                padded = batch["input_ids"].new_tensor(batch["input_ids"][len(padded_labels)].tolist())
            else:
                padded = torch.tensor(lab, dtype=torch.long)
            if padded.numel() < max_len:
                pad = torch.full((max_len - padded.numel(),), -100, dtype=torch.long)
                padded = torch.cat([padded, pad], dim=0)
            else:
                padded = padded[:max_len]
            padded_labels.append(padded)
        batch["labels"] = torch.stack(padded_labels, dim=0)

        attn_bool = batch["attention_mask"].to(dtype=torch.bool)
        batch["attention_mask"] = attn_bool
        batch["position_ids"] = _build_position_ids(attn_bool.long(), self.tokenizer.padding_side)
        return batch


def _load_shadow_causal_lm(*, model_name_or_path: str, tokenizer, attn_implementation: str):
    """
    Load the provided CausalLM and freeze embeddings + lm_head.
    """
    # Support ShadowPEFT's projected shadow model wrapper (AutoModelForCausalLMWithHiddenProjection).
    # This model uses a custom `model_type` and cannot be loaded via AutoModelForCausalLM.
    try:
        from pathlib import Path
        import json as _json

        raw_cfg = None
        revision = None

        # Local folder?
        p = Path(model_name_or_path)
        cfg_path = p / "config.json"
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                raw_cfg = _json.load(f)
        else:
            # Hub repo id (optionally `repo@revision`)
            repo_spec = str(model_name_or_path)
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
                model_name_or_path,
                revision=revision,
                freeze_backbone=False,
                freeze_embed_tokens=True,
                freeze_lm_head=True,
            )
            _sync_pad_token(model, tokenizer)
            if hasattr(model.config, "use_cache"):
                model.config.use_cache = False
            _freeze_embeddings_and_lm_head(model)
            return model
    except Exception as err:
        # Fall back to standard HF loading.
        print(f">>> Error loading shadow model: {err}")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, attn_implementation=attn_implementation
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    _sync_pad_token(model, tokenizer)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    _freeze_embeddings_and_lm_head(model)
    return model


def _resolve_mmlu_eval_subsets(requested_subset: Optional[str]) -> List[str]:
    if requested_subset in (None, "all"):
        # For pretraining we don't need exhaustive eval; keep default light.
        return ["abstract_algebra"]
    if requested_subset == "auxiliary_train":
        raise ValueError("Evaluation subset cannot be 'auxiliary_train'.")
    return [requested_subset]


def _format_chat_text(tokenizer, user_content: str, assistant_content: str) -> str:
    """
    Convert (user, assistant) into one unified text stream using the model chat template.
    All tasks (QA/classification) are converted to this same next-token-pretraining format.
    """
    user_message = {"role": "user", "content": user_content}
    prompt_text = tokenizer.apply_chat_template(
        [user_message],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    im_end_token = getattr(tokenizer, "im_end_token", None) or getattr(tokenizer, "eos_token", "")
    full_text = prompt_text + str(assistant_content)
    if im_end_token:
        full_text += im_end_token
    if not full_text.endswith("\n"):
        full_text += "\n"
    return full_text


def _build_lm_dataset_gsm8k(tokenizer, *, subset: str, split: str, answer_mode: str):
    ds = load_dataset("openai/gsm8k", subset, split=split, download_mode="reuse_dataset_if_exists")

    def fmt(ex):
        q = ex["question"]
        a = ex["answer"]
        if answer_mode == "thinking":
            user = f"Question: {q}\nAnswer:"
            target = a
        else:
            user = (
                f"Question: {q}\n"
                "Give only the final answer as a number.\n"
                "Answer:"
            )
            target = extract_gsm8k_final_answer(a)
        return {"text": _format_chat_text(tokenizer, user, target)}

    return ds.map(fmt, remove_columns=ds.column_names, load_from_cache_file=False)


def _build_lm_dataset_squad_v2(tokenizer, *, split: str, answer_mode: str):
    ds = load_dataset("rajpurkar/squad_v2", split=split, download_mode="reuse_dataset_if_exists")

    def fmt(ex):
        context = ex["context"]
        question = ex["question"]
        answers = ex.get("answers", {"text": [], "answer_start": []}) or {"text": [], "answer_start": []}
        texts = answers.get("text") or []
        has = bool(texts)
        gold = str(texts[0]).strip() if has else "unanswerable"
        if answer_mode == "thinking":
            user = (
                "Answer the question using the provided context.\n"
                "If the question is unanswerable from the context, respond with exactly: unanswerable\n\n"
                f"Context:\n{context}\n\n"
                f"Question:\n{question}\n\n"
                "Answer:"
            )
        else:
            user = (
                "Answer with the exact span from the context.\n"
                "If the question is unanswerable from the context, respond with exactly: unanswerable\n\n"
                f"Context:\n{context}\n\n"
                f"Question:\n{question}\n\n"
                "Answer (span or 'unanswerable' only):"
            )
        return {"text": _format_chat_text(tokenizer, user, gold)}

    return ds.map(fmt, remove_columns=ds.column_names, load_from_cache_file=False)


def _build_lm_dataset_mmlu(tokenizer, *, subset: str, split: str, use_few_shot: bool):
    ds = load_dataset("cais/mmlu", subset, split=split, download_mode="reuse_dataset_if_exists")
    # unwrap nested auxiliary split
    if len(ds.column_names) == 1 and isinstance(ds[0][ds.column_names[0]], dict):
        col = ds.column_names[0]
        ds = ds.map(lambda ex: ex[col], remove_columns=ds.column_names)

    letters = ["A", "B", "C", "D"]

    def fmt(ex):
        q = ex["question"]
        choices = ex["choices"]
        ans_idx = int(ex["answer"])
        if use_few_shot:
            prefix = (
                "Examples of correct response format:\n\n"
                "Question: What is 2+2?\nOptions:\nA: 3\nB: 4\nC: 5\nD: 6\nAnswer: B\n\n"
                "Question: What color is the sky?\nOptions:\nA: Green\nB: Blue\nC: Red\nD: Yellow\nAnswer: B\n\n"
                "Now answer the following question:\n\n"
            )
        else:
            prefix = ""
        options = "\n".join([f"{l}: {choices[i]}" for i, l in enumerate(letters)])
        user = f"{prefix}Question: {q}\nOptions:\n{options}\nAnswer:"
        target = letters[ans_idx] if 0 <= ans_idx < 4 else "A"
        return {"text": _format_chat_text(tokenizer, user, target)}

    return ds.map(fmt, remove_columns=ds.column_names, load_from_cache_file=False)


def _build_lm_dataset_classification(
    tokenizer,
    *,
    dataset_name: str,
    dataset_config: Optional[str],
    split: str,
    text_column: str,
    label_column: str,
):
    ds = load_dataset(dataset_name, dataset_config, split=split, download_mode="reuse_dataset_if_exists")
    label_feat = ds.features.get(label_column)
    label_names = getattr(label_feat, "names", None)

    def fmt(ex):
        text = str(ex[text_column])
        lab = ex[label_column]
        if label_names is not None and isinstance(lab, (int, np.integer)) and 0 <= int(lab) < len(label_names):
            lab_str = str(label_names[int(lab)])
        else:
            lab_str = str(lab)
        user = (
            "Classify the following text.\n"
            "Answer with the label only.\n\n"
            f"Text:\n{text}\n\n"
            "Label:"
        )
        return {"text": _format_chat_text(tokenizer, user, lab_str)}

    return ds.map(fmt, remove_columns=ds.column_names, load_from_cache_file=False)


def _build_tokenized_lm_bundle(*, train_text_ds, eval_text_ds, tokenizer, block_size: int):
    train_ds = tokenize_and_group(train_text_ds, tokenizer, block_size=int(block_size))
    eval_ds = tokenize_and_group(eval_text_ds, tokenizer, block_size=int(block_size))
    return train_ds, eval_ds


def _maybe_cap_examples(ds, n: Optional[int], *, seed: int):
    if n is None:
        return ds
    n = int(n)
    if n <= 0:
        raise ValueError(f"cap must be > 0, got {n}")
    if len(ds) <= n:
        return ds
    return ds.shuffle(seed=seed).select(range(n))


def _run_combined(args: argparse.Namespace, *, out_dir: str) -> None:
    """
    Train one shadow model on a combined mixture of all task datasets, unified as LM text.
    """
    model_path = args.shadow_model_name_or_path or args.model_name
    tok = _prepare_tokenizer(args.tokenizer_name_or_path or model_path)
    seed = int(getattr(args, "seed", 42))

    train_parts = []
    eval_parts = []

    # GSM8K
    gsm_answer_mode = "final" if args.gsm8k_answer_mode == "final" else "thinking"
    gsm_train = _build_lm_dataset_gsm8k(tok, subset=args.gsm8k_subset, split="train", answer_mode=gsm_answer_mode)
    gsm_eval = _build_lm_dataset_gsm8k(tok, subset=args.gsm8k_subset, split="test", answer_mode=gsm_answer_mode)
    train_parts.append(_maybe_cap_examples(gsm_train, args.combined_max_train_examples_per_source, seed=seed))
    eval_parts.append(_maybe_cap_examples(gsm_eval, args.combined_max_eval_examples_per_source, seed=seed))

    # SQuAD v2
    squad_answer_mode = "final" if args.squad_answer_mode == "final" else "thinking"
    sq_train = _build_lm_dataset_squad_v2(tok, split="train", answer_mode=squad_answer_mode)
    sq_eval = _build_lm_dataset_squad_v2(tok, split="validation", answer_mode=squad_answer_mode)
    train_parts.append(_maybe_cap_examples(sq_train, args.combined_max_train_examples_per_source, seed=seed + 1))
    eval_parts.append(_maybe_cap_examples(sq_eval, args.combined_max_eval_examples_per_source, seed=seed + 1))

    # MMLU
    eval_subsets = _resolve_mmlu_eval_subsets(args.mmlu_subset)
    mmlu_train = _build_lm_dataset_mmlu(tok, subset="auxiliary_train", split="train", use_few_shot=False)
    mmlu_eval = _build_lm_dataset_mmlu(tok, subset=eval_subsets[0], split="test", use_few_shot=False)
    train_parts.append(_maybe_cap_examples(mmlu_train, args.combined_max_train_examples_per_source, seed=seed + 2))
    eval_parts.append(_maybe_cap_examples(mmlu_eval, args.combined_max_eval_examples_per_source, seed=seed + 2))

    # Classification suite: amazon + ag_news (unless user narrowed via classification_id)
    cls_specs = CLASSIFICATION_SUITE
    if getattr(args, "classification_id", None):
        cls_specs = [s for s in cls_specs if s.get("id") == args.classification_id]
        if not cls_specs:
            raise ValueError(f"Unknown classification_id={args.classification_id}")
    for idx, spec in enumerate(cls_specs):
        tr = _build_lm_dataset_classification(
            tok,
            dataset_name=spec["dataset_name"],
            dataset_config=spec.get("dataset_config"),
            split=spec.get("train_split", "train"),
            text_column=spec.get("text_column", "text"),
            label_column=spec.get("label_column", "label"),
        )
        ev = _build_lm_dataset_classification(
            tok,
            dataset_name=spec["dataset_name"],
            dataset_config=spec.get("dataset_config"),
            split=spec.get("eval_split", "test"),
            text_column=spec.get("text_column", "text"),
            label_column=spec.get("label_column", "label"),
        )
        train_parts.append(
            _maybe_cap_examples(tr, args.combined_max_train_examples_per_source, seed=seed + 10 + idx)
        )
        eval_parts.append(
            _maybe_cap_examples(ev, args.combined_max_eval_examples_per_source, seed=seed + 10 + idx)
        )

    train_text = concatenate_datasets(train_parts).shuffle(seed=seed)
    eval_text = concatenate_datasets(eval_parts).shuffle(seed=seed)

    train_ds, eval_ds = _build_tokenized_lm_bundle(
        train_text_ds=train_text, eval_text_ds=eval_text, tokenizer=tok, block_size=args.block_size
    )

    model = _load_shadow_causal_lm(
        model_name_or_path=model_path,
        tokenizer=tok,
        attn_implementation=args.attn_implementation,
    )
    if args.bf16:
        model = model.to(torch.bfloat16)
    elif args.fp16:
        model = model.to(torch.float16)

    train_args = _trainer_args(args, out_dir)
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=CausalLMDataCollatorWithPositions(tok),
        tokenizer=tok,
    )
    trainer.train()
    trainer.save_model(out_dir)
    tok.save_pretrained(out_dir)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pretrain a standalone shadow model on domain datasets.")
    p.add_argument(
        "--task",
        type=str,
        choices=("mmlu", "gsm8k", "squad_v2", "classification", "combined"),
        default=None,
        help="Which dataset/task to pretrain on (ignored when --suite is set).",
    )
    p.add_argument(
        "--suite",
        type=str,
        choices=("all", "mmlu", "gsm8k", "squad_v2", "classification", "combined"),
        default=None,
        help="Run a suite of tasks (mirrors run_shadow_peft.py).",
    )
    p.add_argument(
        "--classification_id",
        type=str,
        default=None,
        choices=[spec["id"] for spec in CLASSIFICATION_SUITE],
        help="Run only one classification suite entry when using --suite classification/all.",
    )

    p.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Hugging Face model name/path to pretrain (the shadow model you provide).",
    )
    p.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help=(
            "Hugging Face tokenizer name/path to use for tokenization. "
            "Use this when --model_name points to a shadow-only repo with an invalid/incomplete config. "
            "Example: --tokenizer_name_or_path Qwen/Qwen3-0.6B"
        ),
    )
    p.add_argument(
        "--shadow_model_name_or_path",
        type=str,
        default=None,
        help=(
            "Deprecated. Use --model_name for the model you want to pretrain. "
            "If set, this overrides --model_name."
        ),
    )
    p.add_argument(
        "--attn_implementation",
        type=str,
        default="flash_attention_2",
        choices=("sdpa", "eager", "flash_attention_2"),
    )

    # NOTE: We intentionally do not provide "shadow sizing" knobs here; this script does not
    # build a model from scratch. You must pass a model checkpoint via --model_name.

    # Dataset knobs (mirrors run_shadow_peft.py)
    p.add_argument("--mmlu_subset", type=str, default="all")
    p.add_argument("--gsm8k_subset", type=str, choices=("main", "socratic"), default="main")
    p.add_argument("--gsm8k_answer_mode", type=str, choices=("thinking", "final"), default="thinking")
    p.add_argument("--squad_answer_mode", type=str, choices=("thinking", "final"), default="final")

    # Classification dataset fields
    p.add_argument("--dataset_name", type=str, default=None)
    p.add_argument("--dataset_config", type=str, default=None)
    p.add_argument("--text_column", type=str, default="text")
    p.add_argument("--text_pair_column", type=str, default=None)
    p.add_argument("--label_column", type=str, default="label")
    p.add_argument("--train_split", type=str, default="train")
    p.add_argument("--eval_split", type=str, default="validation")
    p.add_argument("--max_seq_length", type=int, default=512)
    p.add_argument(
        "--block_size",
        type=int,
        default=512,
        help="Chunk size for next-token pretraining (labels = input_ids).",
    )
    p.add_argument(
        "--combined_max_train_examples_per_source",
        type=int,
        default=None,
        help=(
            "Only for --task/--suite combined: cap the number of *raw text* examples taken from each "
            "source dataset before concatenation (helps balance sizes)."
        ),
    )
    p.add_argument(
        "--combined_max_eval_examples_per_source",
        type=int,
        default=None,
        help="Only for --task/--suite combined: cap eval examples per source dataset before concatenation.",
    )

    # Training
    p.add_argument("--per_device_train_batch_size", type=int, default=8)
    p.add_argument("--per_device_eval_batch_size", type=int, default=8)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--eval_accumulation_steps", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--num_train_epochs", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--eval_steps", type=int, default=1000)
    p.add_argument("--save_steps", type=int, default=1000)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--output_dir", type=str, default="outputs_shadow_pretrain")
    p.add_argument("--report_to", type=str, default="none")
    p.add_argument("--run_name", type=str, default="shadow_pretrain")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fp16", type=int, default=0, choices=[0, 1])
    p.add_argument("--bf16", type=int, default=1, choices=[0, 1])
    return p.parse_args()


def _trainer_args(args: argparse.Namespace, output_dir: str) -> TrainingArguments:
    return TrainingArguments(
        output_dir=output_dir,
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
        save_strategy="steps" if int(args.save_steps) > 0 else "no",
        save_total_limit=1,
        warmup_ratio=args.warmup_ratio,
        report_to=args.report_to,
        run_name=args.run_name,
        fp16=bool(args.fp16),
        bf16=bool(args.bf16),
        remove_unused_columns=False,  # datasets include auxiliary fields we want to keep/ignore safely
    )


def _run_mmlu(args: argparse.Namespace, *, out_dir: str) -> None:
    model_path = args.shadow_model_name_or_path or args.model_name
    tok = _prepare_tokenizer(args.tokenizer_name_or_path or model_path)
    eval_subsets = _resolve_mmlu_eval_subsets(args.mmlu_subset)
    train_text = _build_lm_dataset_mmlu(tok, subset="auxiliary_train", split="train", use_few_shot=False)
    eval_text = _build_lm_dataset_mmlu(tok, subset=eval_subsets[0], split="test", use_few_shot=False)
    train_ds, eval_ds = _build_tokenized_lm_bundle(
        train_text_ds=train_text, eval_text_ds=eval_text, tokenizer=tok, block_size=args.block_size
    )
    model = _load_shadow_causal_lm(
        model_name_or_path=model_path,
        tokenizer=tok,
        attn_implementation=args.attn_implementation,
    )
    if args.bf16:
        model = model.to(torch.bfloat16)
    elif args.fp16:
        model = model.to(torch.float16)

    train_args = _trainer_args(args, out_dir)
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=CausalLMDataCollatorWithPositions(tok),
        tokenizer=tok,
    )
    trainer.train()
    trainer.save_model(out_dir)
    tok.save_pretrained(out_dir)


def _run_gsm8k(args: argparse.Namespace, *, out_dir: str) -> None:
    model_path = args.shadow_model_name_or_path or args.model_name
    tok = _prepare_tokenizer(args.tokenizer_name_or_path or model_path)
    train_text = _build_lm_dataset_gsm8k(
        tok,
        subset=args.gsm8k_subset,
        split="train",
        answer_mode=("final" if args.gsm8k_answer_mode == "final" else "thinking"),
    )
    eval_text = _build_lm_dataset_gsm8k(
        tok,
        subset=args.gsm8k_subset,
        split="test",
        answer_mode=("final" if args.gsm8k_answer_mode == "final" else "thinking"),
    )
    train_ds, eval_ds = _build_tokenized_lm_bundle(
        train_text_ds=train_text, eval_text_ds=eval_text, tokenizer=tok, block_size=args.block_size
    )
    model = _load_shadow_causal_lm(
        model_name_or_path=model_path,
        tokenizer=tok,
        attn_implementation=args.attn_implementation,
    )
    if args.bf16:
        model = model.to(torch.bfloat16)
    elif args.fp16:
        model = model.to(torch.float16)

    train_args = _trainer_args(args, out_dir)
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=CausalLMDataCollatorWithPositions(tok),
        tokenizer=tok,
    )
    trainer.train()
    trainer.save_model(out_dir)
    tok.save_pretrained(out_dir)


def _run_squad_v2(args: argparse.Namespace, *, out_dir: str) -> None:
    model_path = args.shadow_model_name_or_path or args.model_name
    tok = _prepare_tokenizer(args.tokenizer_name_or_path or model_path)
    train_text = _build_lm_dataset_squad_v2(
        tok,
        split="train",
        answer_mode=("final" if args.squad_answer_mode == "final" else "thinking"),
    )
    eval_text = _build_lm_dataset_squad_v2(
        tok,
        split="validation",
        answer_mode=("final" if args.squad_answer_mode == "final" else "thinking"),
    )
    train_ds, eval_ds = _build_tokenized_lm_bundle(
        train_text_ds=train_text, eval_text_ds=eval_text, tokenizer=tok, block_size=args.block_size
    )
    model = _load_shadow_causal_lm(
        model_name_or_path=model_path,
        tokenizer=tok,
        attn_implementation=args.attn_implementation,
    )
    if args.bf16:
        model = model.to(torch.bfloat16)
    elif args.fp16:
        model = model.to(torch.float16)

    train_args = _trainer_args(args, out_dir)
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=CausalLMDataCollatorWithPositions(tok),
        tokenizer=tok,
    )
    trainer.train()
    trainer.save_model(out_dir)
    tok.save_pretrained(out_dir)


def _run_classification(args: argparse.Namespace, *, out_dir: str) -> None:
    if not args.dataset_name:
        raise ValueError("classification requires --dataset_name (or use --suite classification/all).")
    model_path = args.shadow_model_name_or_path or args.model_name
    tok = _prepare_tokenizer(args.tokenizer_name_or_path or model_path)
    train_text = _build_lm_dataset_classification(
        tok,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.train_split,
        text_column=args.text_column,
        label_column=args.label_column,
    )
    eval_text = _build_lm_dataset_classification(
        tok,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.eval_split,
        text_column=args.text_column,
        label_column=args.label_column,
    )
    train_ds, eval_ds = _build_tokenized_lm_bundle(
        train_text_ds=train_text, eval_text_ds=eval_text, tokenizer=tok, block_size=args.block_size
    )
    model = _load_shadow_causal_lm(
        model_name_or_path=model_path,
        tokenizer=tok,
        attn_implementation=args.attn_implementation,
    )
    if args.bf16:
        model = model.to(torch.bfloat16)
    elif args.fp16:
        model = model.to(torch.float16)

    train_args = _trainer_args(args, out_dir)
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=CausalLMDataCollatorWithPositions(tok),
        tokenizer=tok,
    )
    trainer.train()
    trainer.save_model(out_dir)
    tok.save_pretrained(out_dir)


def run_suite(args: argparse.Namespace) -> None:
    if args.suite == "combined":
        out_dir = os.path.join(args.output_dir, "combined_all")
        os.makedirs(out_dir, exist_ok=True)
        print("=== Pretraining shadow model on combined mixture (all tasks) ===", flush=True)
        _run_combined(args, out_dir=out_dir)
        return

    selections: List[Tuple[str, List[Dict]]] = []
    if args.suite in ("mmlu", "all"):
        selections.append(("mmlu", MMLU_SUITE))
    if args.suite in ("gsm8k", "all"):
        selections.append(("gsm8k", GSM8K_SUITE))
    if args.suite in ("squad_v2", "all"):
        selections.append(("squad_v2", SQUAD_V2_SUITE))
    if args.suite in ("classification", "all"):
        selections.append(("classification", CLASSIFICATION_SUITE))

    for task, specs in selections:
        if task == "classification" and getattr(args, "classification_id", None):
            specs = [s for s in specs if s.get("id") == args.classification_id]
        for spec in specs:
            run_args = copy.deepcopy(args)
            run_args.task = task
            for k, v in spec.items():
                if k == "id":
                    continue
                setattr(run_args, k, v)
            out_dir = os.path.join(args.output_dir, f"{task}_{spec['id']}")
            os.makedirs(out_dir, exist_ok=True)
            print(f"=== Pretraining shadow model on {task}:{spec['id']} ===", flush=True)
            if task == "mmlu":
                _run_mmlu(run_args, out_dir=out_dir)
            elif task == "gsm8k":
                _run_gsm8k(run_args, out_dir=out_dir)
            elif task == "squad_v2":
                _run_squad_v2(run_args, out_dir=out_dir)
            elif task == "classification":
                _run_classification(run_args, out_dir=out_dir)
            else:
                raise NotImplementedError(task)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    # Persist config for reproducibility.
    with open(os.path.join(args.output_dir, "pretrain_args.json"), "w", encoding="utf-8") as f:
        f.write(str(args) + "\n")

    if args.suite is not None:
        run_suite(args)
        return

    if args.task is None:
        raise ValueError("Either --task or --suite must be specified.")

    out_dir = os.path.join(args.output_dir, f"{args.task}_run")
    os.makedirs(out_dir, exist_ok=True)
    if args.task == "mmlu":
        _run_mmlu(args, out_dir=out_dir)
    elif args.task == "gsm8k":
        _run_gsm8k(args, out_dir=out_dir)
    elif args.task == "squad_v2":
        _run_squad_v2(args, out_dir=out_dir)
    elif args.task == "classification":
        _run_classification(args, out_dir=out_dir)
    elif args.task == "combined":
        _run_combined(args, out_dir=out_dir)
    else:
        raise NotImplementedError(args.task)


if __name__ == "__main__":
    main()

