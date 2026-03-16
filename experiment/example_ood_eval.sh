#!/bin/bash
# Example usage of run_shadow_peft_ood_eval.py for Out-of-Domain evaluation

# Scenario 1: Model trained on GSM8K, evaluated on MMLU (OOD)
echo "=== Example 1: GSM8K → MMLU (ShadowPEFT) ==="
CUDA_VISIBLE_DEVICES=7 python run_shadow_peft_ood_eval.py \
  --mode shadow \
  --trained_on gsm8k \
  --task mmlu \
  --model_name Qwen/Qwen3-4B \
  --checkpoint_dir erin99/Qwen3-4B-GSM8k-Shadow \
  --mmlu_subset all \
  --k_shot 2 \
  --max_seq_length 512 \
  --per_device_eval_batch_size 16 \
  --bf16 1 \
  --output_dir ./ood_eval/gsm8k_to_mmlu_shadow \
  --seed 42

# Scenario 2: GSM8k -> SquadV2
echo "=== Example 2: GSM8K → SQuAD v2 (ShadowPEFT) ==="
CUDA_VISIBLE_DEVICES=7 python run_shadow_peft_ood_eval.py \
  --mode shadow \
  --trained_on gsm8k \
  --task squad_v2 \
  --model_name Qwen/Qwen3-4B \
  --checkpoint_dir erin99/Qwen3-4B-GSM8k-Shadow \
  --squad_answer_mode final \
  --k_shot 2 \
  --max_seq_length 512 \
  --per_device_eval_batch_size 16 \
  --bf16 1 \
  --output_dir ./ood_eval/gsm8k_to_squad_shadow \
  --seed 42




# LoRA, GSM8k -> MMLU
CUDA_VISIBLE_DEVICES=7 python run_shadow_peft_ood_eval.py \
  --mode lora \
  --trained_on gsm8k \
  --task mmlu \
  --model_name Qwen/Qwen3-4B \
  --checkpoint_dir erin99/Qwen3-4B-GSM8k-LoRA \
  --mmlu_subset all \
  --k_shot 2 \
  --max_seq_length 512 \
  --per_device_eval_batch_size 16 \
  --bf16 1 \
  --output_dir ./ood_eval/gsm8k_to_mmlu_lora \
  --seed 42

# LoRA, GSM8k -> SquadV2
CUDA_VISIBLE_DEVICES=7 python run_shadow_peft_ood_eval.py \
  --mode lora \
  --trained_on gsm8k \
  --task squad_v2 \
  --model_name Qwen/Qwen3-4B \
  --checkpoint_dir erin99/Qwen3-4B-GSM8k-LoRA \
  --squad_answer_mode final \
  --k_shot 2 \
  --max_seq_length 512 \
  --per_device_eval_batch_size 16 \
  --bf16 1 \
  --output_dir ./ood_eval/gsm8k_to_squad_lora \
  --seed 42



# DoRA, GSM8k -> MMLU
CUDA_VISIBLE_DEVICES=7 python run_shadow_peft_ood_eval.py \
  --mode dora \
  --trained_on gsm8k \
  --task mmlu \
  --model_name Qwen/Qwen3-4B \
  --checkpoint_dir erin99/Qwen3-4B-GSM8k-DoRA \
  --mmlu_subset all \
  --k_shot 2 \
  --max_seq_length 512 \
  --per_device_eval_batch_size 16 \
  --bf16 1 \
  --output_dir ./ood_eval/gsm8k_to_mmlu_dora \
  --seed 42

# LoRA, GSM8k -> SquadV2
CUDA_VISIBLE_DEVICES=7 python run_shadow_peft_ood_eval.py \
  --mode dora \
  --trained_on gsm8k \
  --task squad_v2 \
  --model_name Qwen/Qwen3-4B \
  --checkpoint_dir erin99/Qwen3-4B-GSM8k-DoRA \
  --squad_answer_mode final \
  --k_shot 2 \
  --max_seq_length 512 \
  --per_device_eval_batch_size 16 \
  --bf16 1 \
  --output_dir ./ood_eval/gsm8k_to_squad_dora \
  --seed 42



# Scenario 1: Model trained on GSM8K, evaluated on MMLU (OOD)
echo "=== Example 1: SQUADv2 → GSM8k (ShadowPEFT) ==="
CUDA_VISIBLE_DEVICES=7 python run_shadow_peft_ood_eval.py \
  --mode shadow \
  --trained_on squad_v2 \
  --task gsm8k \
  --gsm8k_answer_mode boxed \
  --model_name Qwen/Qwen3-4B \
  --checkpoint_dir erin99/Qwen3-4B-SquadV2-Shadow \
  --gsm8k_subset main \
  --k_shot 2 \
  --max_seq_length 1024 \
  --generation_max_length 512 \
  --per_device_eval_batch_size 16 \
  --bf16 1 \
  --output_dir ./ood_eval/squad_v2_to_gsm8k_shadow \
  --seed 42

# Scenario 2: GSM8k -> SquadV2
echo "=== Example 2: SQuAD v2 -> MMLU ==="
CUDA_VISIBLE_DEVICES=7 python run_shadow_peft_ood_eval.py \
  --mode shadow \
  --trained_on squad_v2 \
  --task mmlu \
  --model_name Qwen/Qwen3-4B \
  --checkpoint_dir erin99/Qwen3-4B-SquadV2-Shadow \
  --mmlu_subset all \
  --k_shot 2 \
  --max_seq_length 512 \
  --per_device_eval_batch_size 16 \
  --bf16 1 \
  --output_dir ./ood_eval/squad_v2_to_mmlu_shadow \
  --seed 42




# Scenario 1: Model trained on GSM8K, evaluated on MMLU (OOD)
echo "=== Example 1: SQUADv2 → GSM8k (LoRA) ==="
CUDA_VISIBLE_DEVICES=7 python run_shadow_peft_ood_eval.py \
  --mode lora \
  --trained_on squad_v2 \
  --task gsm8k \
  --model_name Qwen/Qwen3-4B \
  --checkpoint_dir erin99/Qwen3-4B-SquadV2-LoRA \
  --gsm8k_subset main \
  --k_shot 2 \
  --max_seq_length 1024 \
  --generation_max_length 512 \
  --per_device_eval_batch_size 16 \
  --bf16 1 \
  --output_dir ./ood_eval/squad_v2_to_gsm8k_lora \
  --seed 42

# Scenario 2: GSM8k -> SquadV2
echo "=== Example 2: SQuAD v2 -> MMLU (LoRA) ==="
CUDA_VISIBLE_DEVICES=7 python run_shadow_peft_ood_eval.py \
  --mode lora \
  --trained_on squad_v2 \
  --task mmlu \
  --model_name Qwen/Qwen3-4B \
  --checkpoint_dir erin99/Qwen3-4B-SquadV2-LoRA \
  --mmlu_subset all \
  --k_shot 2 \
  --max_seq_length 512 \
  --per_device_eval_batch_size 16 \
  --bf16 1 \
  --output_dir ./ood_eval/squad_v2_to_mmlu_lora \
  --seed 42

