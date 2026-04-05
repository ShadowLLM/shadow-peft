#!/bin/bash
# Example usage of run_shadow_peft_ood_eval.py for Out-of-Domain evaluation

# Scenario 1: Model trained on GSM8K, evaluated on MMLU (OOD)
echo "=== 0.6B ==="

37.34
CUDA_VISIBLE_DEVICES=5 python run_eval.py \
  --mode base \
  --model_name Qwen/Qwen3-0.6B \
  --attn_implementation sdpa \
  --task mmlu \
  --suite mmlu \
  --mmlu_subset all \
  --use_few_shot 1 \
  --max_seq_length 512 \
  --per_device_eval_batch_size 16 \
  --bf16 1 \
  --seed 42


47.69
CUDA_VISIBLE_DEVICES=6 python run_eval.py \
  --mode base \
  --model_name Qwen/Qwen3-0.6B \
  --attn_implementation sdpa \
  --task gsm8k \
  --suite gsm8k \
  --gsm8k_subset main \
  --gsm8k_answer_mode thinking \
  --max_seq_length 512 \
  --per_device_eval_batch_size 16 \
  --bf16 1 \
  --seed 42



{
  "eval_loss": NaN,
  "eval_samples": 11873,
  "eval_exact": 49.47359555293523,
  "eval_f1": 49.87334447577307,
  "eval_total": 11873,
  "eval_HasAns_exact": 0.11808367071524967,
  "eval_HasAns_f1": 0.9187278948808825,
  "eval_HasAns_total": 5928,
  "eval_NoAns_exact": 98.68797308662742,
  "eval_NoAns_f1": 98.68797308662742,
  "eval_NoAns_total": 5945,
  "eval_best_exact": 50.09685841825992,
  "eval_best_exact_thresh": 0.0,
  "eval_best_f1": 50.22479125924713,
  "eval_best_f1_thresh": 0.0,
  "eval_runtime": 1393.3885,
  "eval_samples_per_second": 8.521,
  "eval_steps_per_second": 0.533,
  "epoch": 0
}
CUDA_VISIBLE_DEVICES=2 python run_eval.py \
  --mode base \
  --model_name Qwen/Qwen3-0.6B \
  --attn_implementation sdpa \
  --task squad_v2 \
  --suite squad_v2 \
  --squad_answer_mode final \
  --max_seq_length 512 \
  --per_device_eval_batch_size 16 \
  --bf16 1 \
  --seed 42


# 4B
67.54
CUDA_VISIBLE_DEVICES=1 python run_eval.py \
  --mode base \
  --model_name Qwen/Qwen3-4B \
  --attn_implementation sdpa \
  --task mmlu \
  --suite mmlu \
  --mmlu_subset all \
  --use_few_shot 1 \
  --max_seq_length 512 \
  --per_device_eval_batch_size 16 \
  --bf16 1 \
  --seed 42 \
  --output_dir eval_outputs/mmlu_qwen_4B_sdpa


77.18
CUDA_VISIBLE_DEVICES=5 python run_eval.py \
  --mode base \
  --model_name Qwen/Qwen3-4B \
  --attn_implementation sdpa \
  --task gsm8k \
  --suite gsm8k \
  --gsm8k_subset main \
  --gsm8k_answer_mode thinking \
  --max_seq_length 512 \
  --per_device_eval_batch_size 16 \
  --bf16 1 \
  --seed 42 \
  --output_dir eval_outputs/gsm8k_qwen_4B_sdpa


{
  "eval_loss": NaN,
  "eval_samples": 11873,
  "eval_exact": 68.48311294533816,
  "eval_f1": 75.49457423188097,
  "eval_total": 11873,
  "eval_HasAns_exact": 62.432523616734144,
  "eval_HasAns_f1": 76.47555328190339,
  "eval_HasAns_total": 5928,
  "eval_NoAns_exact": 74.51640033641716,
  "eval_NoAns_f1": 74.51640033641716,
  "eval_NoAns_total": 5945,
  "eval_best_exact": 68.48311294533816,
  "eval_best_exact_thresh": 0.0,
  "eval_best_f1": 75.49457423188092,
  "eval_best_f1_thresh": 0.0,
  "eval_runtime": 2903.6649,
  "eval_samples_per_second": 4.089,
  "eval_steps_per_second": 0.256,
  "epoch": 0
}
CUDA_VISIBLE_DEVICES=2 python run_eval.py \
  --mode base \
  --model_name Qwen/Qwen3-4B \
  --attn_implementation sdpa \
  --task squad_v2 \
  --suite squad_v2 \
  --squad_answer_mode final \
  --max_seq_length 512 \
  --per_device_eval_batch_size 16 \
  --bf16 1 \
  --seed 42 \
  --output_dir eval_outputs/squad_v2_qwen_4B_sdpa



# 8B

71.92
CUDA_VISIBLE_DEVICES=1 python run_eval.py \
  --mode base \
  --model_name Qwen/Qwen3-8B \
  --attn_implementation sdpa \
  --task mmlu \
  --suite mmlu \
  --mmlu_subset all \
  --use_few_shot 1 \
  --max_seq_length 512 \
  --per_device_eval_batch_size 16 \
  --bf16 1 \
  --seed 42 \
  --output_dir eval_outputs/mmlu_qwen_8B_sdpa


68.84
CUDA_VISIBLE_DEVICES=6 python run_eval.py \
  --mode base \
  --model_name Qwen/Qwen3-8B \
  --attn_implementation sdpa \
  --task gsm8k \
  --suite gsm8k \
  --gsm8k_subset main \
  --gsm8k_answer_mode thinking \
  --max_seq_length 512 \
  --per_device_eval_batch_size 16 \
  --bf16 1 \
  --seed 42 \
  --output_dir eval_outputs/gsm8k_qwen_8B_sdpa


{
  "eval_loss": NaN,
  "eval_samples": 11873,
  "eval_exact": 65.98163901288638,
  "eval_f1": 73.82122220918419,
  "eval_total": 11873,
  "eval_HasAns_exact": 63.29284750337382,
  "eval_HasAns_f1": 78.99449583158614,
  "eval_HasAns_total": 5928,
  "eval_NoAns_exact": 68.66274179983179,
  "eval_NoAns_f1": 68.66274179983179,
  "eval_NoAns_total": 5945,
  "eval_best_exact": 65.99006148403942,
  "eval_best_exact_thresh": 0.0,
  "eval_best_f1": 73.82964468033705,
  "eval_best_f1_thresh": 0.0,
  "eval_runtime": 3641.8064,
  "eval_samples_per_second": 3.26,
  "eval_steps_per_second": 0.204,
  "epoch": 0
}
CUDA_VISIBLE_DEVICES=2 python run_eval.py \
  --mode base \
  --model_name Qwen/Qwen3-8B \
  --attn_implementation sdpa \
  --task squad_v2 \
  --suite squad_v2 \
  --squad_answer_mode final \
  --max_seq_length 512 \
  --per_device_eval_batch_size 16 \
  --bf16 1 \
  --seed 42 \
  --output_dir eval_outputs/squad_v2_qwen_8B_sdpa