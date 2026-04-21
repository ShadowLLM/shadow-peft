"""
Dataset helpers for causal language modeling experiments.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, List, Optional, Tuple

from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizerBase


def _build_chat_texts(
    tokenizer: PreTrainedTokenizerBase,
    user_content: str,
    answer_letter: str,
) -> Tuple[str, str, List[Dict[str, str]]]:
    """Return (full_text, prompt_text) for a user question and gold answer."""
    user_message = {"role": "user", "content": user_content}
    prompt_text = tokenizer.apply_chat_template(
        [user_message],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    
    im_end_token = getattr(tokenizer, "im_end_token", None)
    if im_end_token is None:
        im_end_token = getattr(tokenizer, "eos_token", "")
    
    full_text = prompt_text + answer_letter
    if im_end_token:
        full_text += im_end_token
    if not full_text.endswith("\n"):
        full_text += "\n"

    return full_text, prompt_text


def load_text_dataset(
    dataset_name: str,
    split: str,
    text_column: str = "text",
    dataset_config: Optional[str] = None,
    text_template: Optional[str] = None,
) -> Dataset:
    """Download and return a Hugging Face dataset with a single `text` column."""
    dataset = load_dataset(dataset_name, dataset_config, split=split, download_mode="reuse_dataset_if_exists")

    template_column = "__formatted_text__"
    if text_template is not None:
        def apply_template(example):
            try:
                text_value = text_template.format(**{k: str(v) for k, v in example.items()})
            except KeyError as exc:
                raise ValueError(
                    f"Placeholder '{exc.args[0]}' not found in dataset columns "
                    f"{dataset.column_names}"
                ) from exc
            return {template_column: text_value}

        dataset = dataset.map(apply_template)
        text_column = template_column

    if text_column not in dataset.column_names:
        raise ValueError(
            f"Column '{text_column}' not found. Available columns: {dataset.column_names}"
        )

    if text_column != "text":
        dataset = dataset.rename_column(text_column, "text")

    return dataset.remove_columns([c for c in dataset.column_names if c != "text"])


def tokenize_and_group(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    block_size: int = 512,
) -> Dataset:
    """Tokenize raw text and chunk it into fixed-length sequences."""
    if block_size > tokenizer.model_max_length:
        block_size = tokenizer.model_max_length

    def tokenize_function(examples: Dict[str, List[str]]):
        return tokenizer(examples["text"])

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
    )

    def group_texts(examples: Dict[str, List[List[int]]]):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples["input_ids"])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [
                concatenated_examples[k][i : i + block_size]
                for i in range(0, total_length, block_size)
            ]
            for k in concatenated_examples.keys()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    return tokenized_dataset.map(group_texts, batched=True)


@dataclass
class ExperimentDatasetBundle:
    train_dataset: Dataset
    eval_dataset: Dataset


@dataclass
class ClassificationDatasetBundle:
    train_dataset: Dataset
    eval_dataset: Dataset
    label2id: Dict[str, int]
    id2label: Dict[int, str]


def build_datasets(
    dataset_name: str,
    tokenizer: PreTrainedTokenizerBase,
    dataset_config: Optional[str] = None,
    block_size: int = 512,
    train_split: str = "train",
    eval_split: str = "validation",
    text_column: str = "text",
    text_template: Optional[str] = None,
) -> ExperimentDatasetBundle:
    """Prepare tokenized train/eval datasets."""
    train_raw = load_text_dataset(
        dataset_name,
        split=train_split,
        text_column=text_column,
        dataset_config=dataset_config,
        text_template=text_template,
    )
    eval_raw = load_text_dataset(
        dataset_name,
        split=eval_split,
        text_column=text_column,
        dataset_config=dataset_config,
        text_template=text_template,
    )

    train_dataset = tokenize_and_group(train_raw, tokenizer, block_size)
    eval_dataset = tokenize_and_group(eval_raw, tokenizer, block_size)

    return ExperimentDatasetBundle(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )


def build_classification_datasets(
    dataset_name: str,
    tokenizer: PreTrainedTokenizerBase,
    dataset_config: Optional[str] = None,
    text_column: str = "text",
    text_pair_column: Optional[str] = None,
    label_column: str = "label",
    max_seq_length: int = 512,
    train_split: str = "train",
    eval_split: str = "validation",
) -> ClassificationDatasetBundle:
    """Tokenize datasets for sequence classification."""
    dataset_dict = load_dataset(dataset_name, dataset_config, download_mode="reuse_dataset_if_exists")
    if train_split not in dataset_dict or eval_split not in dataset_dict:
        raise ValueError(f"Splits '{train_split}'/'{eval_split}' not found in dataset.")

    remove_columns = dataset_dict[train_split].column_names
    raw_labels = dataset_dict[train_split].unique(label_column)
    label2id = {str(label): idx for idx, label in enumerate(sorted(map(str, raw_labels)))}
    id2label = {idx: label for label, idx in label2id.items()}

    def preprocess_function(examples: Dict[str, List[str]]):
        text_pairs = examples[text_pair_column] if text_pair_column else None
        tokenized = tokenizer(
            examples[text_column],
            text_pair=text_pairs,
            truncation=True,
            max_length=max_seq_length,
        )
        labels = []
        for label in examples[label_column]:
            label_str = str(label)
            if label_str not in label2id:
                raise ValueError(f"Label '{label_str}' not found in label set {label2id.keys()}")
            labels.append(label2id[label_str])
        tokenized["labels"] = labels
        return tokenized

    train_dataset = dataset_dict[train_split].map(
        preprocess_function,
        batched=True,
        remove_columns=remove_columns,
    )
    eval_dataset = dataset_dict[eval_split].map(
        preprocess_function,
        batched=True,
        remove_columns=remove_columns,
    )

    return ClassificationDatasetBundle(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        label2id=label2id,
        id2label=id2label,
    )


def _unwrap_mmlu_split(dataset: Dataset) -> Dataset:
    """Flatten auxiliary MMLU splits that nest the real columns under a single key."""
    if len(dataset.column_names) != 1:
        return dataset
    column = dataset.column_names[0]
    sample = dataset[0][column]
    if not isinstance(sample, dict):
        return dataset

    def unwrap(example):
        return example[column]

    return dataset.map(unwrap, remove_columns=dataset.column_names)


def load_mmlu_dataset(
    subset: str,
    split: str,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 512,
    use_few_shot: bool = False,
) -> Dataset:
    """Load and format cais/mmlu subset for multiple-choice QA."""
    dataset = load_dataset("cais/mmlu", subset, split=split, download_mode="reuse_dataset_if_exists")
    dataset = _unwrap_mmlu_split(dataset)
    
    def format_mmlu_example(example):
        question = example["question"]
        choices = example["choices"]
        answer_idx = example["answer"]  # This is an integer 0-3
        
        # Format as multiple choice question with explicit single-letter response instruction
        choice_letters = ["A", "B", "C", "D"]
        
        # Add few-shot examples if requested
        if use_few_shot:
            few_shot_prefix = (
                "Examples of correct response format:\n\n"
                "Question: What is 2+2?\nOptions:\nA: 3\nB: 4\nC: 5\nD: 6\nAnswer: B\n\n"
                "Question: What color is the sky?\nOptions:\nA: Green\nB: Blue\nC: Red\nD: Yellow\nAnswer: B\n\n"
                "Now answer the following question:\n\n"
            )
        else:
            few_shot_prefix = ""
        
        formatted_text = f"{few_shot_prefix}Question: {question}\n\nOptions:\n"
        for i, choice in enumerate(choices):
            formatted_text += f"{choice_letters[i]}: {choice}\n"
        formatted_text += (
            "\nInstructions: Answer with ONLY the letter (A, B, C, or D). "
            "Do not include any explanation, reasoning, or additional text. "
            "Answer:\n\n"
        )
        
        answer_letter = choice_letters[answer_idx]

        # Pre-tokenize using chat template
        full_text, prompt_text= _build_chat_texts(
            tokenizer,
            formatted_text,
            answer_letter,
        )
        
        # Tokenize full sequence
        tokenized = tokenizer(
            full_text,
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        
        # Tokenize prompt only to determine where answer starts
        prompt_tokenized = tokenizer(
            prompt_text,
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        
        # Create labels: mask prompt tokens with -100, keep answer tokens
        prompt_length = len(prompt_tokenized["input_ids"])
        labels = [-100] * prompt_length + tokenized["input_ids"][prompt_length:]
        tokenized["labels"] = labels
        
        # Keep MMLU-specific fields for evaluation
        tokenized["answer_letter"] = answer_letter
        tokenized["answer_idx"] = answer_idx
        tokenized["prompt_input_ids"] = prompt_tokenized["input_ids"]
        tokenized["prompt_attention_mask"] = prompt_tokenized["attention_mask"]

        return tokenized
    
    processed = dataset.map(
        format_mmlu_example,
        remove_columns=dataset.column_names,
        load_from_cache_file=False,  # Force reprocessing to avoid cached version
    )
    
    return processed


@dataclass
class MMLUDatasetBundle:
    """Bundle for MMLU training dataset and per-subset eval datasets."""
    train_dataset: Dataset
    eval_datasets: Dict[str, Dataset]


def build_mmlu_datasets(
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 512,
    train_subset: str = "auxiliary_train",
    train_split: str = "train",
    eval_subsets: Optional[List[str]] = None,
    eval_split: str = "test",
    use_few_shot: bool = False,
) -> MMLUDatasetBundle:
    """Build MMLU datasets with shared auxiliary train split and per-subset eval data."""
    if eval_subsets is None:
        raise ValueError("eval_subsets must be provided for MMLU evaluation.")
    
    train_dataset = load_mmlu_dataset(train_subset, train_split, tokenizer, max_length, use_few_shot)
    eval_datasets: Dict[str, Dataset] = {}
    for subset in eval_subsets:
        if subset == train_subset:
            continue
        eval_datasets[subset] = load_mmlu_dataset(subset, eval_split, tokenizer, max_length, use_few_shot)
    
    return MMLUDatasetBundle(
        train_dataset=train_dataset,
        eval_datasets=eval_datasets,
    )


_GSM8K_ANSWER_LINE = re.compile(r"####\s*([^\n\r]+)")
_GSM8K_NUMBER = re.compile(r"[-+]?\d[\d,]*\.?\d*")


def extract_gsm8k_final_answer(text: str) -> str:
    """
    Extract and normalize the final GSM8K answer.

    GSM8K gold answers typically end with a line like: "#### 72".
    We return a normalized string version of the number when possible.
    """
    if text is None:
        return ""
    match = _GSM8K_ANSWER_LINE.search(text)
    candidate = match.group(1).strip() if match else text.strip()
    num = _GSM8K_NUMBER.search(candidate)
    if not num:
        return candidate
    return num.group(0).replace(",", "").strip()


@dataclass
class GSM8KDatasetBundle:
    train_dataset: Dataset
    eval_dataset: Dataset


@dataclass
class SquadV2DatasetBundle:
    train_dataset: Dataset
    eval_dataset: Dataset


def load_gsm8k_dataset(
    subset: str,
    split: str,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 512,
    answer_mode: str = "thinking",
) -> Dataset:
    """Load and format openai/gsm8k subset for SFT with answer extraction."""
    dataset = load_dataset("openai/gsm8k", subset, split=split, download_mode="reuse_dataset_if_exists")

    def format_gsm8k_example(example):
        question = example["question"]
        answer = example["answer"]
        gold_final = extract_gsm8k_final_answer(answer)

        if answer_mode == "thinking":
            user_content = f"Question: {question}\nAnswer:"
            target_text = answer
        elif answer_mode in ("final", "non_thinking"):
            user_content = (
                f"Question: {question}\n"
                "Give only the final answer as a number.\n"
                "Answer:"
            )
            target_text = gold_final
        else:
            raise ValueError(
                "answer_mode must be 'thinking' or 'final' (alias: non_thinking). "
                f"Got: {answer_mode}"
            )

        user_message = {"role": "user", "content": user_content}
        prompt_text = tokenizer.apply_chat_template(
            [user_message],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

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

        # Keep prompt-only inputs for generation evaluation.
        tokenized["prompt_input_ids"] = prompt_tokenized["input_ids"]
        tokenized["prompt_attention_mask"] = prompt_tokenized["attention_mask"]
        tokenized["gold_answer"] = gold_final
        return tokenized

    processed = dataset.map(
        format_gsm8k_example,
        remove_columns=dataset.column_names,
        load_from_cache_file=False,
    )
    return processed


def build_gsm8k_datasets(
    tokenizer: PreTrainedTokenizerBase,
    subset: str = "main",
    max_length: int = 512,
    answer_mode: str = "thinking",
) -> GSM8KDatasetBundle:
    train_dataset = load_gsm8k_dataset(
        subset=subset,
        split="train",
        tokenizer=tokenizer,
        max_length=max_length,
        answer_mode=answer_mode,
    )
    eval_dataset = load_gsm8k_dataset(
        subset=subset,
        split="test",
        tokenizer=tokenizer,
        max_length=max_length,
        answer_mode=answer_mode,
    )
    return GSM8KDatasetBundle(train_dataset=train_dataset, eval_dataset=eval_dataset)


def load_squad_v2_dataset(
    split: str,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 512,
    answer_mode: str = "thinking",
) -> Dataset:
    """
    Load and format rajpurkar/squad_v2 for SFT with generation-style answers.

    Common practice: train on train split, evaluate on validation split.

    - If the question is answerable, supervise with an extractive answer span text.
    - If unanswerable, supervise with the token "unanswerable".
    """
    dataset = load_dataset("rajpurkar/squad_v2", split=split, download_mode="reuse_dataset_if_exists")

    def format_example(example):
        qid = str(example["id"])
        context = example["context"]
        question = example["question"]
        answers = example.get("answers", {"text": [], "answer_start": []}) or {"text": [], "answer_start": []}
        answer_texts = answers.get("text") or []

        has_answer = bool(answer_texts)
        gold_text = str(answer_texts[0]).strip() if has_answer else "unanswerable"

        if answer_mode == "thinking":
            user_content = (
                "Answer the question using the provided context.\n"
                "If the question is unanswerable from the context, respond with exactly: unanswerable\n\n"
                f"Context:\n{context}\n\n"
                f"Question:\n{question}\n\n"
                "Answer:"
            )
            target_text = gold_text
        elif answer_mode in ("final", "non_thinking"):
            # Same supervision, but explicitly discourage extra words.
            user_content = (
                "Answer with the exact span from the context.\n"
                "If the question is unanswerable from the context, respond with exactly: unanswerable\n\n"
                f"Context:\n{context}\n\n"
                f"Question:\n{question}\n\n"
                "Answer (span or 'unanswerable' only):"
            )
            target_text = gold_text
        else:
            raise ValueError(
                "answer_mode must be 'thinking' or 'final' (alias: non_thinking). "
                f"Got: {answer_mode}"
            )

        user_message = {"role": "user", "content": user_content}
        prompt_text = tokenizer.apply_chat_template(
            [user_message],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

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

        # Keep raw fields for evaluation with the official metric.
        tokenized["squad_id"] = qid
        tokenized["squad_answers"] = answers
        tokenized["gold_text"] = gold_text
        tokenized["has_answer"] = int(has_answer)
        return tokenized

    processed = dataset.map(
        format_example,
        remove_columns=dataset.column_names,
        load_from_cache_file=False,
    )
    return processed


def build_squad_v2_datasets(
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 512,
    answer_mode: str = "thinking",
) -> SquadV2DatasetBundle:
    train_dataset = load_squad_v2_dataset(
        split="train",
        tokenizer=tokenizer,
        max_length=max_length,
        answer_mode=answer_mode,
    )
    eval_dataset = load_squad_v2_dataset(
        split="validation",
        tokenizer=tokenizer,
        max_length=max_length,
        answer_mode=answer_mode,
    )
    return SquadV2DatasetBundle(train_dataset=train_dataset, eval_dataset=eval_dataset)

