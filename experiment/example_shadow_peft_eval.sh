#!/bin/bash
# Example usage of run_shadow_peft_eval.py

# Example 1: Evaluate a ShadowPEFT checkpoint on MMLU
echo "=== Example 1: MMLU Evaluation ==="
CUDA_VISIBLE_DEVICES=7 python run_shadow_peft_eval.py \
  --task mmlu \
  --model_name Qwen/Qwen3-4B \
  --checkpoint_dir erin99/Qwen3-4B-MMLU-Shadow \
  --mmlu_subset all \
  --max_seq_length 512 \
  --per_device_eval_batch_size 16 \
  --bf16 1 \
  --output_dir ./eval_outputs/mmlu \
  --seed 42

# Example 2: Evaluate on GSM8K with thinking mode
echo "=== Example 2: GSM8K Evaluation (Thinking Mode) ==="
CUDA_VISIBLE_DEVICES=7 python run_shadow_peft_eval.py \
  --task gsm8k \
  --model_name Qwen/Qwen3-4B \
  --checkpoint_dir erin99/Qwen3-4B-GSM8k-Shadow \
  --gsm8k_subset main \
  --gsm8k_answer_mode thinking \
  --max_seq_length 512 \
  --per_device_eval_batch_size 8 \
  --bf16 1 \
  --output_dir ./eval_outputs/gsm8k_thinking \
  --seed 42


# Example 4: Evaluate on SQuAD v2
echo "=== Example 4: SQuAD v2 Evaluation ==="
CUDA_VISIBLE_DEVICES=7 python run_shadow_peft_eval.py \
  --task squad_v2 \
  --model_name Qwen/Qwen3-4B \
  --checkpoint_dir erin99/Qwen3-4B-SquadV2-Shadow \
  --squad_answer_mode final \
  --max_seq_length 512 \
  --per_device_eval_batch_size 16 \
  --bf16 1 \
  --output_dir ./eval_outputs/squad_v2 \
  --seed 42

echo "=== All examples completed ==="

