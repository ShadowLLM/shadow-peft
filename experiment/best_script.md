# 1. GSM8k

## 0.6B

**48.90**
```bash
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=7 python run_shadow_peft.py \
  --suite gsm8k \
  --gsm8k_subset main \
  --output_dir outputs/gsm8k_suite_shadow_scaled \
  --mode shadow \
  --bf16 1 \
  --learning_rate 3e-3 \
  --injection_hidden_size 16 \
  --gate_hidden_size 10 \
  --shadow_intermediate_size 256 \
  --shadow_layers 1 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --run_name gsm8k-shadow-scaled5 \
  --eval_steps 3000 \
  --num_train_epochs 4 \
  --trainer sft \
  --shadow_loss_weight 0.05 \
  --shadow_alpha 0.1 \
  --shadow_dropout 0.2 \
  --warmup_ratio 0.01 \
  --print_shadow_output 0 \
  --report_to none
```

## 4B

**79**

```bash
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=2 python run_shadow_peft.py \
  --model_name Qwen/Qwen3-4B \
  --suite gsm8k \
  --gsm8k_subset main \
  --output_dir outputs/gsm8k_suite_shadow_4B \
  --mode shadow \
  --bf16 1 \
  --learning_rate 3e-3 \
  --injection_hidden_size 16 \
  --gate_hidden_size 8 \
  --shadow_intermediate_size 256 \
  --shadow_num_attention_heads 16 \
  --shadow_layers 1 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --run_name gsm8k-shadow-4B \
  --eval_steps 3000 \
  --num_train_epochs 4 \
  --trainer sft \
  --shadow_loss_weight 0.05 \
  --shadow_alpha 0.1 \
  --shadow_dropout 0.2 \
  --warmup_ratio 0.01 \
  --print_shadow_output 0 \
  --report_to none
```

## 8B

**80.74**
```bash
CUDA_VISIBLE_DEVICES=7 python run_shadow_peft.py \
  --model_name Qwen/Qwen3-8B \
  --suite gsm8k \
  --gsm8k_subset main \
  --output_dir outputs/gsm8k_suite_shadow_8B \
  --mode shadow \
  --bf16 1 \
  --learning_rate 3e-3 \
  --injection_hidden_size 16 \
  --gate_hidden_size 8 \
  --shadow_intermediate_size 256 \
  --shadow_num_attention_heads 8 \
  --shadow_layers 1 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --run_name gsm8k-shadow-8B \
  --eval_steps 3000 \
  --num_train_epochs 4 \
  --trainer sft \
  --shadow_loss_weight 0.05 \
  --shadow_alpha 0.1 \
  --shadow_dropout 0.2 \
  --warmup_ratio 0.01 \
  --print_shadow_output 0 \
  --report_to none
```

# 2. squad v2

## 0.6B
  "shadow_peft_sft": {
    "eval_loss": NaN,
    "eval_samples": 11873,
    "eval_exact": 80.5356691653331,
    "eval_f1": 83.86790342263112,
    "eval_total": 11873,
    "eval_HasAns_exact": 76.21457489878543,
    "eval_HasAns_f1": 82.88859941580648,
    "eval_HasAns_total": 5928,
    "eval_NoAns_exact": 84.8444070647603,
    "eval_NoAns_f1": 84.8444070647603,
    "eval_NoAns_total": 5945,
    "eval_best_exact": 80.5356691653331,
    "eval_best_exact_thresh": 0.0,
    "eval_best_f1": 83.86790342263097,
    "eval_best_f1_thresh": 0.0,
    "eval_shadow_exact": 42.103933294028465,
    "eval_shadow_f1": 42.41573334211894,
    "eval_shadow_total": 11873,
    "eval_shadow_HasAns_exact": 0.43859649122807015,
    "eval_shadow_HasAns_f1": 1.0630907508398222,
    "eval_shadow_HasAns_total": 5928,
    "eval_shadow_NoAns_exact": 83.65012615643398,
    "eval_shadow_NoAns_f1": 83.65012615643398,
    "eval_shadow_NoAns_total": 5945,
    "eval_shadow_best_exact": 50.07159100480081,
    "eval_shadow_best_exact_thresh": 0.0,
    "eval_shadow_best_f1": 50.08001347595385,
    "eval_shadow_best_f1_thresh": 0.0,
    "eval_runtime": 3331.9618,
    "eval_samples_per_second": 3.563,
    "eval_steps_per_second": 0.112,
    "epoch": 2.0
  }

```bash
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=4 python run_shadow_peft.py \
  --suite squad_v2 \
  --output_dir outputs/squad_v2_suite_shadow_alpha5 \
  --mode shadow \
  --bf16 1 \
  --learning_rate 1e-3 \
  --injection_hidden_size 16 \
  --gate_hidden_size 10 \
  --shadow_intermediate_size 256 \
  --shadow_layers 1 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --run_name squad-v2-shadow-beta3 \
  --eval_steps 1000000 \
  --num_train_epochs 2 \
  --trainer sft \
  --shadow_loss_weight 0.05 \
  --shadow_alpha 0.5 \
  --shadow_dropout 0.2 \
  --warmup_ratio 0.01 \
  --print_shadow_output 1 \
  --report_to none
```

## 4B

{
  "shadow_peft_sft": {
    "eval_loss": NaN,
    "eval_samples": 11873,
    "eval_exact": 86.83567758780426,
    "eval_f1": 89.8164571998564,
    "eval_total": 11873,
    "eval_HasAns_exact": 83.50202429149797,
    "eval_HasAns_f1": 89.47213163527265,
    "eval_HasAns_total": 5928,
    "eval_NoAns_exact": 90.15979814970564,
    "eval_NoAns_f1": 90.15979814970564,
    "eval_NoAns_total": 5945,
    "eval_best_exact": 86.83567758780426,
    "eval_best_exact_thresh": 0.0,
    "eval_best_f1": 89.8164571998561,
    "eval_best_f1_thresh": 0.0,
    "eval_shadow_exact": 41.98601869788596,
    "eval_shadow_f1": 42.190897210018605,
    "eval_shadow_total": 11873,
    "eval_shadow_HasAns_exact": 0.25303643724696356,
    "eval_shadow_HasAns_f1": 0.6633810011050897,
    "eval_shadow_HasAns_total": 5928,
    "eval_shadow_NoAns_exact": 83.59966358284272,
    "eval_shadow_NoAns_f1": 83.59966358284272,
    "eval_shadow_NoAns_total": 5945,
    "eval_shadow_best_exact": 50.07159100480081,
    "eval_shadow_best_exact_thresh": 0.0,
    "eval_shadow_best_f1": 50.07159100480081,
    "eval_shadow_best_f1_thresh": 0.0,
    "eval_runtime": 4276.9529,
    "eval_samples_per_second": 2.776,
    "eval_steps_per_second": 0.174,
    "epoch": 2.0
  }

```bash
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=6 python run_shadow_peft.py \
  --model_name Qwen/Qwen3-4B \
  --suite squad_v2 \
  --output_dir outputs/squad_v2_suite_shadow_4B \
  --mode shadow \
  --bf16 1 \
  --learning_rate 1e-3 \
  --injection_hidden_size 16 \
  --gate_hidden_size 8 \
  --shadow_intermediate_size 256 \
  --shadow_num_attention_heads 16 \
  --shadow_layers 1 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --run_name squad-v2-shadow-4B \
  --eval_steps 1000000 \
  --num_train_epochs 2 \
  --trainer sft \
  --shadow_loss_weight 0.05 \
  --shadow_alpha 0.1 \
  --shadow_dropout 0.2 \
  --warmup_ratio 0.01 \
  --print_shadow_output 1 \
  --report_to none
```


## 8B

{
  "shadow_peft_sft": {
    "eval_loss": NaN,
    "eval_samples": 11873,
    "eval_exact": 87.50947528004717,
    "eval_f1": 90.46589932704052,
    "eval_total": 11873,
    "eval_HasAns_exact": 84.0587044534413,
    "eval_HasAns_f1": 89.98003082151688,
    "eval_HasAns_total": 5928,
    "eval_NoAns_exact": 90.95037846930194,
    "eval_NoAns_f1": 90.95037846930194,
    "eval_NoAns_total": 5945,
    "eval_best_exact": 87.50947528004717,
    "eval_best_exact_thresh": 0.0,
    "eval_best_f1": 90.46589932704006,
    "eval_best_f1_thresh": 0.0,
    "eval_shadow_exact": 44.352733091889164,
    "eval_shadow_f1": 44.47596247394108,
    "eval_shadow_total": 11873,
    "eval_shadow_HasAns_exact": 0.1349527665317139,
    "eval_shadow_HasAns_f1": 0.38176492123860545,
    "eval_shadow_HasAns_total": 5928,
    "eval_shadow_NoAns_exact": 88.44407064760303,
    "eval_shadow_NoAns_f1": 88.44407064760303,
    "eval_shadow_NoAns_total": 5945,
    "eval_shadow_best_exact": 50.07159100480081,
    "eval_shadow_best_exact_thresh": 0.0,
    "eval_shadow_best_f1": 50.07159100480081,
    "eval_shadow_best_f1_thresh": 0.0,
    "eval_runtime": 2809.216,
    "eval_samples_per_second": 4.226,
    "eval_steps_per_second": 0.264,
    "epoch": 2.0
  }

```bash
CUDA_VISIBLE_DEVICES=7 python run_shadow_peft.py \
  --model_name Qwen/Qwen3-8B \
  --suite squad_v2 \
  --output_dir outputs/squad_v2_suite_shadow_8B \
  --mode shadow \
  --bf16 1 \
  --learning_rate 1e-3 \
  --injection_hidden_size 16 \
  --gate_hidden_size 8 \
  --shadow_intermediate_size 256 \
  --shadow_num_attention_heads 8 \
  --shadow_layers 1 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --run_name squad-v2-shadow-8B \
  --eval_steps 1000000 \
  --num_train_epochs 2 \
  --trainer sft \
  --shadow_loss_weight 0.05 \
  --shadow_alpha 0.1 \
  --shadow_dropout 0.2 \
  --warmup_ratio 0.01 \
  --print_shadow_output 1 \
  --report_to none
```

# MMLU

## 0.6B

50.63, (shadow 24.62)
```bash
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=2 python run_shadow_peft.py \
  --suite mmlu \
  --mmlu_subset anatomy \
  --output_dir outputs/mmlu_suite_shadow_clamp4 \
  --mode shadow \
  --bf16 1 \
  --run_name mmlu-shadow-clamp \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --eval_steps 200  \
  --num_train_epochs 2 \
  --print_shadow_output 1 \
  --injection_hidden_size 16 \
  --gate_hidden_size 10 \
  --shadow_intermediate_size 256 \
  --shadow_layers 1 \
  --shadow_alpha 1.5 \
  --shadow_loss_weight 0.15 \
  --shadow_dropout 0.1 \
  --learning_rate 2e-4 \
  --warmup_ratio 0.01 \
  --report_to none
```

## 4B

72.91, 27.01

```bash
CUDA_VISIBLE_DEVICES=3 python run_shadow_peft.py \
  --model_name Qwen/Qwen3-4B \
  --suite mmlu \
  --mmlu_subset anatomy \
  --output_dir outputs/mmlu_suite_shadow_clamp_4B-2 \
  --mode shadow \
  --bf16 1 \
  --run_name mmlu-shadow-clamp-4B-2 \
  --eval_steps 200  \
  --num_train_epochs 2 \
  --injection_hidden_size 16 \
  --gate_hidden_size 8 \
  --shadow_intermediate_size 256 \
  --shadow_num_attention_heads 16 \
  --shadow_layers 1 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --shadow_alpha 0.1 \
  --shadow_loss_weight 0.1 \
  --shadow_dropout 0.1 \
  --learning_rate 2e-4 \
  --warmup_ratio 0.01 \
  --report_to none
```

## 8B

[best] **76.51**
```bash
CUDA_VISIBLE_DEVICES=7 python run_shadow_peft.py \
  --model_name Qwen/Qwen3-8B \
  --suite mmlu \
  --mmlu_subset anatomy \
  --output_dir outputs/mmlu_suite_shadow_clamp_8B-2 \
  --mode shadow \
  --bf16 1 \
  --run_name mmlu-shadow-clamp-8B \
  --eval_steps 200  \
  --num_train_epochs 1 \
  --injection_hidden_size 16 \
  --gate_hidden_size 8 \
  --shadow_intermediate_size 256 \
  --shadow_num_attention_heads 8 \
  --shadow_layers 1 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --shadow_alpha 0.1 \
  --shadow_loss_weight 0.05 \
  --shadow_dropout 0.2 \
  --learning_rate 2e-4 \
  --warmup_ratio 0.01 \
  --report_to none
```


# amazon_reviews_multi_en

## 0.6B
{
  "shadow_peft": {
    "eval_loss": 0.9495033025741577,
    "eval_accuracy": 0.6118,
    "eval_shadow_accuracy": 0.509,
    "eval_runtime": 14.2704,
    "eval_samples_per_second": 350.376,
    "eval_steps_per_second": 2.803,
    "epoch": 2.0
  }
```bash
CUDA_VISIBLE_DEVICES=4 python run_shadow_peft.py \
  --suite classification \
  --classification_id amazon_reviews_multi_en \
  --output_dir outputs/cls_suite_shadow \
  --mode shadow \
  --bf16 1 \
  --shadow_intermediate_size 256 \
  --shadow_layers 1 \
  --injection_hidden_size 16 \
  --gate_hidden_size 10 \
  --shadow_alpha 2.0 \
  --shadow_loss_weight 0.05 \
  --shadow_dropout 0.2 \
  --run_name shadow-cls-shadow \
  --eval_steps 100 \
  --per_device_train_batch_size 128 \
  --per_device_eval_batch_size 128 \
  --num_train_epochs 2 \
  --report_to none
```

## 4B

"shadow_peft": {
    "eval_loss": 0.9113936424255371,
    "eval_accuracy": 0.6266,
    "eval_shadow_accuracy": 0.5188,
    "eval_runtime": 44.7103,
    "eval_samples_per_second": 111.831,
    "eval_steps_per_second": 3.511,
    "epoch": 2.0
  }

```bash
CUDA_VISIBLE_DEVICES=1 python run_shadow_peft.py \
  --model_name Qwen/Qwen3-4B \
  --suite classification \
  --classification_id amazon_reviews_multi_en \
  --output_dir outputs/cls_suite_shadow_4B-2 \
  --mode shadow \
  --bf16 1 \
  --injection_hidden_size 16 \
  --gate_hidden_size 8 \
  --shadow_intermediate_size 256 \
  --shadow_num_attention_heads 16 \
  --shadow_layers 1 \
  --shadow_alpha 0.5 \
  --shadow_loss_weight 0.05 \
  --shadow_dropout 0.1 \
  --run_name shadow-cls-shadow-4B \
  --eval_steps 100 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --num_train_epochs 2 \
  --learning_rate 5e-5 \
  --warmup_ratio 0.01 \
  --report_to none
```


## 8B

**62.84**
```bash
CUDA_VISIBLE_DEVICES=5 python run_shadow_peft.py \
  --model_name Qwen/Qwen3-8B \
  --suite classification \
  --classification_id amazon_reviews_multi_en \
  --output_dir outputs/cls_suite_shadow_8B-2 \
  --mode shadow \
  --bf16 1 \
  --injection_hidden_size 16 \
  --gate_hidden_size 8 \
  --shadow_intermediate_size 256 \
  --shadow_num_attention_heads 8 \
  --shadow_layers 1 \
  --shadow_alpha 1.0 \
  --shadow_loss_weight 0.05 \
  --shadow_dropout 0.1 \
  --run_name shadow-cls-shadow-8B \
  --eval_steps 100 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --num_train_epochs 2 \
  --learning_rate 2e-5 \
  --warmup_ratio 0.01 \
  --report_to none
```

# agnews
{
  "shadow_peft": {
    "eval_loss": 0.17759943008422852,
    "eval_accuracy": 0.9468421052631579,
    "eval_shadow_accuracy": 0.9227631578947368,
    "eval_runtime": 14.5342,
    "eval_samples_per_second": 522.906,
    "eval_steps_per_second": 4.128,
    "epoch": 2.0
  }
  
```bash
CUDA_VISIBLE_DEVICES=2 python run_shadow_peft.py \
  --suite classification \
  --classification_id ag_news \
  --output_dir outputs/cls_suite_shadow \
  --mode shadow \
  --bf16 1 \
  --shadow_intermediate_size 256 \
  --shadow_layers 1 \
  --injection_hidden_size 16 \
  --gate_hidden_size 10 \
  --shadow_alpha 2.0 \
  --shadow_loss_weight 0.05 \
  --shadow_dropout 0.2 \
  --run_name shadow-cls-shadow \
  --eval_steps 100 \
  --per_device_train_batch_size 64 \
  --per_device_eval_batch_size 64 \
  --num_train_epochs 2 \
  --warmup_ratio 0.01 \
  --report_to none
```


## 4B

**95.45**
```bash
CUDA_VISIBLE_DEVICES=2 python run_shadow_peft.py \
  --model_name Qwen/Qwen3-4B \
  --suite classification \
  --classification_id ag_news \
  --output_dir outputs/cls_suite_shadow_4B \
  --mode shadow \
  --bf16 1 \
  --injection_hidden_size 16 \
  --gate_hidden_size 8 \
  --shadow_intermediate_size 256 \
  --shadow_num_attention_heads 16 \
  --shadow_layers 1 \
  --shadow_alpha 0.3 \
  --shadow_loss_weight 0.2 \
  --shadow_dropout 0.2 \
  --run_name shadow-cls-shadow-4B \
  --eval_steps 100 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --num_train_epochs 2 \
  --learning_rate 2e-4 \
  --warmup_ratio 0.01 \
  --report_to none
```

## 8B

**95.41**
```bash
CUDA_VISIBLE_DEVICES=6 python run_shadow_peft.py \
  --model_name Qwen/Qwen3-8B \
  --suite classification \
  --classification_id ag_news \
  --output_dir outputs/cls_suite_shadow_8B-2 \
  --mode shadow \
  --bf16 1 \
  --injection_hidden_size 16 \
  --gate_hidden_size 8 \
  --shadow_intermediate_size 256 \
  --shadow_num_attention_heads 8 \
  --shadow_layers 1 \
  --shadow_alpha 1.0 \
  --shadow_loss_weight 0.5 \
  --shadow_dropout 0.2 \
  --run_name shadow-cls-shadow-8B \
  --eval_steps 100 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --num_train_epochs 2 \
  --learning_rate 2e-4 \
  --warmup_ratio 0.01 \
  --report_to none
```