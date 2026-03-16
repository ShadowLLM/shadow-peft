#!/usr/bin/env python3
"""
LM Head Alignment Training Script

This script fine-tunes the shadow_hidden_projection of a AutoModelForCausalLMWithHiddenProjection
model using knowledge distillation from a reference (teacher) model.

The goal is to adapt a model trained with one lm_head (e.g., from a smaller model) to work
optimally with a different lm_head (e.g., from a larger model).

Usage:
    # With KL divergence loss (default)
    CUDA_VISIBLE_DEVICES=4 python run_lm_head_alignment.py \
        --student_model_path ./initial_pretrained_shadows/qwen3-0.6b-with-hidden-projection-for-qwen3-8B \
        --teacher_model_path Qwen/Qwen3-0.6B \
        --dataset_name /sharedata/lxm/fineweb-edu-dedup-5B \
        --output_dir ./aligned_model/qwen3-0.6b-with-hidden-projection-for-qwen3-8B/fineweb-edu-dedup-5B-MSE_KLD \
        --num_train_epochs 2 \
        --learning_rate 1e-4 \
        --per_device_train_batch_size 32 \
        --alpha 1.0 \
        --temperature 1.0 \
        --use_mse \
        --use_kld \
        --bf16

    CUDA_VISIBLE_DEVICES=6 python run_lm_head_alignment.py \
        --student_model_path ./aligned_model/qwen3-0.6b-with-hidden-projection-for-qwen3-8B/fineweb-edu-dedup-5B-MSE_KLD/final \
        --teacher_model_path Qwen/Qwen3-0.6B \
        --dataset_name /sharedata/lxm/wudao \
        --output_dir ./aligned_model/qwen3-0.6b-with-hidden-projection-for-qwen3-8B/wudao-5B-MSE_KLD \
        --num_train_epochs 2 \
        --learning_rate 5e-5 \
        --per_device_train_batch_size 64 \
        --alpha 1.0 \
        --temperature 1.0 \
        --use_kld \
        --bf16

"""

import argparse
import copy
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

os.environ["HF_HOME"] = "/sharedata/lxm/.cache"

# Add ShadowPEFT to path
_HERE = Path(__file__).parent
_SHADOW_PEFT_SRC = _HERE.parent / "ShadowPEFT" / "src"
if str(_SHADOW_PEFT_SRC) not in sys.path:
    sys.path.insert(0, str(_SHADOW_PEFT_SRC))

from shadow_peft import AutoModelForCausalLMWithHiddenProjection

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class AlignmentTrainer(Trainer):
    """
    Custom trainer that uses knowledge distillation to fine-tune the shadow_hidden_projection.
    
    OPTIMIZATION: Since only the projection is trainable and the student/teacher share the same
    backbone, we REUSE the student's backbone output to compute teacher logits. This eliminates
    redundant computation and cuts training time nearly in half!
    """

    def __init__(
        self,
        teacher_lm_head: nn.Module,
        alpha: float = 0.5,
        temperature: float = 2.0,
        adaptive_temperature: bool = False,
        initial_temp: float = 3.0,
        final_temp: float = 1.0,
        regularization_weight: float = 0.0,
        use_kld: bool = True,
        use_mse: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        # Store only the teacher's lm_head (not the full model!)
        self.teacher_lm_head = teacher_lm_head
        self.teacher_lm_head.eval()
        
        # Disable gradients for teacher lm_head
        for param in self.teacher_lm_head.parameters():
            param.requires_grad = False
        
        # Move teacher lm_head to device
        if self.args.n_gpu > 0:
            self.teacher_lm_head = self.teacher_lm_head.to(self.args.device)

        self.alpha = alpha
        self.temperature = temperature
        self.adaptive_temperature = adaptive_temperature
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.regularization_weight = regularization_weight
        self.use_kld = use_kld
        self.use_mse = use_mse

        # Validate loss configuration
        if not use_kld and not use_mse:
            raise ValueError("At least one of --use_kld or --use_mse must be enabled")

        # Store initial projection weights for regularization
        if regularization_weight > 0:
            self.initial_projection_weight = (
                self.model.shadow_hidden_projection.weight.data.clone().detach()
            )
        
        logger.info(f"AlignmentTrainer initialized:")
        logger.info(f"  Alpha (distillation weight): {alpha}")
        logger.info(f"  Temperature: {temperature}")
        logger.info(f"  Adaptive temperature: {adaptive_temperature}")
        if adaptive_temperature:
            logger.info(f"    Initial temp: {initial_temp}, Final temp: {final_temp}")
        logger.info(f"  Regularization weight: {regularization_weight}")
        logger.info(f"  Loss types: KLD={use_kld}, MSE={use_mse}")
        logger.info(f"  Optimization: Reusing student backbone output (2x faster!)")

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Compute combined loss:
        - (1-alpha) * standard LM loss
        - alpha * distillation loss (KL divergence and/or MSE with teacher)
        - regularization * L2 distance from initial projection
        
        OPTIMIZATION: Runs backbone ONCE, then forks to student/teacher paths.
        """
        labels = inputs.get("labels")

        # ============================================================
        # OPTIMIZATION: Run backbone ONCE, reuse for both paths
        # ============================================================
        
        # 1. Run backbone once to get hidden states
        backbone_out = model.shadow_model(
            input_ids=inputs.get("input_ids"),
            attention_mask=inputs.get("attention_mask"),
            position_ids=inputs.get("position_ids"),
            inputs_embeds=inputs.get("inputs_embeds"),
            use_cache=False,
            return_dict=True,
        )
        hidden = backbone_out.last_hidden_state  # [batch, seq, shadow_hidden]
        
        # 2. Student path: hidden → projection → student lm_head (WITH gradients)
        projected_hidden = model.shadow_hidden_projection(hidden)
        student_logits = model.lm_head(projected_hidden)
        
        # 3. Teacher path: hidden → teacher lm_head (NO gradients)
        with torch.no_grad():
            teacher_logits = self.teacher_lm_head(hidden)
        
        # 4. Manually compute LM loss (shift for causal LM)
        if labels is not None and self.alpha < 1.0:
            # Shift so that tokens < n predict n
            shift_logits = student_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = torch.nn.CrossEntropyLoss()
            lm_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        else:
            lm_loss = torch.tensor(0.0, device=student_logits.device)

        # Adaptive temperature scheduling
        if self.adaptive_temperature and self.state.max_steps > 0:
            progress = self.state.global_step / self.state.max_steps
            current_temp = self.initial_temp + (self.final_temp - self.initial_temp) * progress
        else:
            current_temp = self.temperature

        # Distillation losses
        kl_loss = torch.tensor(0.0, device=student_logits.device)
        mse_loss = torch.tensor(0.0, device=student_logits.device)
        
        if labels is not None:
            # Mask for valid tokens (not -100 padding)
            mask = (labels != -100).float()

            # KL divergence loss
            if self.use_kld:
                # Temperature-scaled softmax
                student_log_probs = F.log_softmax(student_logits / current_temp, dim=-1)
                teacher_probs = F.softmax(teacher_logits / current_temp, dim=-1)

                # KL divergence per token
                kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)

                # Apply mask and average
                kl_loss = (kl_loss * mask).sum() / mask.sum()
                kl_loss = kl_loss * (current_temp ** 2)  # Scale back
            
            # MSE loss (on logits directly)
            if self.use_mse:
                # Compute MSE per token
                mse_per_token = F.mse_loss(student_logits, teacher_logits, reduction="none").mean(dim=-1)
                
                # Apply mask and average
                mse_loss = (mse_per_token * mask).sum() / mask.sum()

        # Combine distillation losses
        if self.use_kld and self.use_mse:
            # Both enabled: average them
            distill_loss = (kl_loss + mse_loss) / 2.0
        elif self.use_kld:
            distill_loss = kl_loss
        else:
            distill_loss = mse_loss

        # Regularization: keep projection close to initialization
        reg_loss = torch.tensor(0.0, device=student_logits.device)
        if self.regularization_weight > 0:
            proj_weight = model.shadow_hidden_projection.weight
            proj_drift = (
                proj_weight - self.initial_projection_weight.to(proj_weight.device)
            ).pow(2).mean()
            reg_loss = self.regularization_weight * proj_drift

        # Combined loss
        total_loss = (1 - self.alpha) * lm_loss + self.alpha * distill_loss + reg_loss

        # Log individual components
        if self.state.global_step % self.args.logging_steps == 0:
            log_dict = {
                "train/lm_loss": lm_loss.item(),
                "train/distill_loss": distill_loss.item(),
                "train/reg_loss": reg_loss.item(),
                "train/temperature": current_temp,
                "train/total_loss": total_loss.item(),
            }
            if self.use_kld:
                log_dict["train/kl_loss"] = kl_loss.item()
            if self.use_mse:
                log_dict["train/mse_loss"] = mse_loss.item()
            self.log(log_dict)

        # Return outputs if requested (for compatibility)
        if return_outputs:
            # Create a minimal output object for compatibility
            outputs = CausalLMOutputWithPast(
                loss=total_loss,
                logits=student_logits,
                past_key_values=None,
                hidden_states=None,
                attentions=None,
            )
            return (total_loss, outputs)
        
        return total_loss


def prepare_dataset(
    dataset_name: str,
    dataset_config: Optional[str],
    num_train_samples: Optional[int] = None,
    num_eval_samples: Optional[int] = None,
):
    """
    Load raw dataset WITHOUT pre-tokenization (tokenization happens on-the-fly).
    This is much faster for large datasets!
    """
    logger.info(f"Loading dataset: {dataset_name}")
    if dataset_config:
        dataset = load_dataset(dataset_name, dataset_config)
    else:
        dataset = load_dataset(dataset_name)

    # Determine text column
    text_column = None
    for col in ["text", "content", "question", "input"]:
        if col in dataset["train"].column_names:
            text_column = col
            break

    if text_column is None:
        raise ValueError(f"Could not find text column in dataset. Available columns: {dataset['train'].column_names}")

    logger.info(f"Using text column: {text_column}")

    # Subsample if requested (BEFORE tokenization for speed)
    if num_train_samples and len(dataset["train"]) > num_train_samples:
        dataset["train"] = dataset["train"].shuffle(seed=42).select(range(num_train_samples))

    if "test" in dataset:
        eval_split = "test"
    elif "validation" in dataset:
        eval_split = "validation"
    else:
        # Split train into train/eval
        split = dataset["train"].train_test_split(test_size=0.1, seed=42)
        dataset["train"] = split["train"]
        dataset["validation"] = split["test"]
        eval_split = "validation"

    if num_eval_samples and len(dataset[eval_split]) > num_eval_samples:
        dataset[eval_split] = dataset[eval_split].shuffle(seed=42).select(range(num_eval_samples))

    logger.info(f"Train samples: {len(dataset['train'])}")
    logger.info(f"Eval samples: {len(dataset[eval_split])}")
    logger.info(f"✓ Dataset loaded (tokenization will happen on-the-fly during training)")

    return dataset["train"], dataset[eval_split], text_column


class OnTheFlyTokenizationCollator:
    """
    Data collator that tokenizes text on-the-fly during training.
    Much faster than pre-tokenizing large datasets!
    """
    def __init__(self, tokenizer, text_column: str = "text", max_length: int = 512):
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.max_length = max_length

    def __call__(self, examples):
        # Extract text from examples
        texts = [example[self.text_column] for example in examples]
        
        # Tokenize on-the-fly
        batch = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",  # Pad to max_length for efficient batching
            return_tensors="pt",
        )
        
        # Create labels (same as input_ids for causal LM)
        batch["labels"] = batch["input_ids"].clone()
        
        # Mask padding tokens in labels
        if self.tokenizer.pad_token_id is not None:
            batch["labels"][batch["labels"] == self.tokenizer.pad_token_id] = -100
        
        return batch


def main():
    parser = argparse.ArgumentParser(description="Align shadow_hidden_projection using knowledge distillation")

    # Model arguments
    parser.add_argument(
        "--student_model_path",
        type=str,
        required=True,
        help="Path to the AutoModelForCausalLMWithHiddenProjection model (student)",
    )
    parser.add_argument(
        "--teacher_model_path",
        type=str,
        required=True,
        help="Path or HF model ID for the reference teacher model",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Path to tokenizer (defaults to student_model_path)",
    )

    # Dataset arguments
    parser.add_argument("--dataset_name", type=str, required=True, help="HuggingFace dataset name")
    parser.add_argument("--dataset_config", type=str, default=None, help="Dataset configuration/subset")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--num_train_samples", type=int, default=None, help="Limit training samples")
    parser.add_argument("--num_eval_samples", type=int, default=None, help="Limit evaluation samples")

    # Training arguments
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for checkpoints")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--max_steps", type=int, default=-1, help="Max training steps (overrides epochs)")
    parser.add_argument("--per_device_train_batch_size", type=int, default=32, help="Train batch size per device")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32, help="Eval batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="LR scheduler type")

    # Distillation arguments
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.7,
        help="Distillation weight (0-1). Higher = more distillation. Recommended: 0.7",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Distillation temperature. Higher = softer distributions. Recommended: 2.0",
    )
    parser.add_argument(
        "--adaptive_temperature",
        action="store_true",
        help="Use adaptive temperature scheduling (starts high, ends low)",
    )
    parser.add_argument("--initial_temp", type=float, default=3.0, help="Initial temperature (if adaptive)")
    parser.add_argument("--final_temp", type=float, default=1.0, help="Final temperature (if adaptive)")
    parser.add_argument(
        "--regularization_weight",
        type=float,
        default=0.0,
        help="L2 regularization weight to keep projection close to initialization",
    )
    parser.add_argument(
        "--use_kld",
        action="store_true",
        default=False,
        help="Use KL divergence loss for distillation",
    )
    parser.add_argument(
        "--use_mse",
        action="store_true",
        default=False,
        help="Use MSE loss for distillation (on logits)",
    )

    # Logging arguments
    parser.add_argument("--logging_steps", type=int, default=50, help="Log every N steps")
    parser.add_argument("--save_steps", type=int, default=1000, help="Save checkpoint every N steps")
    parser.add_argument("--save_total_limit", type=int, default=1, help="Maximum number of checkpoints to keep")
    
    # Other arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 training")
    parser.add_argument("--bf16", action="store_true", help="Use BF16 training")
    parser.add_argument(
        "--dataloader_pin_memory",
        action="store_true",
        default=True,
        help="Pin memory in data loaders for faster GPU transfer",
    )

    args = parser.parse_args()

    # Default to KLD if neither is specified
    if not args.use_kld and not args.use_mse:
        logger.info("Neither --use_kld nor --use_mse specified, defaulting to --use_kld")
        args.use_kld = True

    # Set up logging
    logger.info("=" * 80)
    logger.info("LM Head Alignment Training")
    logger.info("=" * 80)

    # Load tokenizer
    tokenizer_path = args.tokenizer_path or args.teacher_model_path
    logger.info(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load student model
    logger.info(f"Loading student model from: {args.student_model_path}")
    student_model = AutoModelForCausalLMWithHiddenProjection.from_pretrained(
        args.student_model_path,
        freeze_backbone=True,  # Freeze backbone
        freeze_embed_tokens=True,  # Freeze embeddings
        freeze_lm_head=True,  # Freeze lm_head
        torch_dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32),
    )

    # Only shadow_hidden_projection should be trainable
    trainable_params = sum(p.numel() for p in student_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in student_model.parameters())
    logger.info(f"Student model loaded:")
    logger.info(f"  Trainable params: {trainable_params:,}")
    logger.info(f"  Total params: {total_params:,}")
    logger.info(f"  Trainable %: {100 * trainable_params / total_params:.4f}%")

    # Load teacher model's lm_head only (OPTIMIZATION: no need for full model)
    logger.info(f"Loading teacher lm_head from: {args.teacher_model_path}")
    logger.info("  (Loading full model temporarily to extract lm_head...)")
    teacher_model_temp = AutoModelForCausalLM.from_pretrained(
        args.teacher_model_path,
        torch_dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32),
        trust_remote_code=True,
    )
    
    # Extract and clone the lm_head
    teacher_lm_head = teacher_model_temp.get_output_embeddings()
    if teacher_lm_head is None:
        raise ValueError("Teacher model does not have an output embeddings layer (lm_head)")
    
    # Clone the lm_head as a standalone module
    teacher_lm_head = copy.deepcopy(teacher_lm_head)
    teacher_lm_head.eval()
    for param in teacher_lm_head.parameters():
        param.requires_grad = False
    
    # Delete the full teacher model to save memory
    del teacher_model_temp
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    logger.info(f"  Teacher lm_head extracted: {teacher_lm_head.weight.shape}")
    logger.info(f"  ✓ Full teacher model deleted (memory saved!)")

    # Load and prepare dataset (raw, no pre-tokenization)
    train_dataset, eval_dataset, text_column = prepare_dataset(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        num_train_samples=args.num_train_samples,
        num_eval_samples=args.num_eval_samples,
    )

    # Data collator (tokenizes on-the-fly)
    data_collator = OnTheFlyTokenizationCollator(
        tokenizer=tokenizer,
        text_column=text_column,
        max_length=args.max_length,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        eval_strategy="no",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=args.bf16,
        fp16=args.fp16,
        dataloader_num_workers=4,
        dataloader_pin_memory=args.dataloader_pin_memory,
        remove_unused_columns=False,
        report_to=["none"],
        gradient_checkpointing=False,  # Not needed for small trainable params
        seed=args.seed,
        optim="adamw_torch",
    )

    # Create trainer
    trainer = AlignmentTrainer(
        model=student_model,
        teacher_lm_head=teacher_lm_head,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        alpha=args.alpha,
        temperature=args.temperature,
        adaptive_temperature=args.adaptive_temperature,
        initial_temp=args.initial_temp,
        final_temp=args.final_temp,
        regularization_weight=args.regularization_weight,
        use_kld=args.use_kld,
        use_mse=args.use_mse,
    )

    # Log training setup
    logger.info("\n" + "=" * 80)
    logger.info("Training Configuration")
    logger.info("=" * 80)
    logger.info(f"Device: {training_args.device}")
    logger.info(f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    logger.info(f"Total optimization steps: {trainer.state.max_steps if trainer.state.max_steps > 0 else 'epoch-based'}")
    logger.info("=" * 80 + "\n")

    # Train
    logger.info("Starting training...")
    train_result = trainer.train()

    # Save final model
    logger.info(f"Saving final model to {args.output_dir}/final")
    trainer.save_model(os.path.join(args.output_dir, "final"))
    trainer.save_state()

    # Log metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    # Final evaluation
    logger.info("Running final evaluation...")
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    logger.info("=" * 80)
    logger.info("Training complete!")
    logger.info(f"Final model saved to: {args.output_dir}/final")
    logger.info(f"Final eval loss: {eval_metrics['eval_loss']:.6f}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

