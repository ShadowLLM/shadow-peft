# GSM8k

## LoRA
### 0.1B
parameter: 122,683,392

81.35
```bash
CUDA_VISIBLE_DEVICES=0 python run_shadow_peft.py \
  --model_name Qwen/Qwen3-8B \
  --suite gsm8k \
  --gsm8k_subset main \
  --mode lora \
  --bf16 1 \
  --learning_rate 2e-4 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --lora_r 128 \
  --lora_alpha 128 \
  --output_dir outputs/gsm8k_suite_scaling_lora_0.1B \
  --run_name gsm8k-lora-8B \
  --eval_steps 300000 \
  --num_train_epochs 2 \
  --trainer sft \
  --report_to none
```


### 0.2B

parameter: 184,025,088

80.59

```bash
CUDA_VISIBLE_DEVICES=1 python run_shadow_peft.py \
  --model_name Qwen/Qwen3-8B \
  --suite gsm8k \
  --gsm8k_subset main \
  --output_dir outputs/gsm8k_suite_scaling_lora_0.2B \
  --mode lora \
  --bf16 1 \
  --learning_rate 2e-4 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --lora_r 192 \
  --lora_alpha 192 \
  --run_name gsm8k-lora-8B \
  --eval_steps 300000 \
  --num_train_epochs 2 \
  --trainer sft \
  --report_to none
```

### 0.3B
parameter: 291,373,056

79.91
```bash
CUDA_VISIBLE_DEVICES=2 python run_shadow_peft.py \
  --model_name Qwen/Qwen3-8B \
  --suite gsm8k \
  --gsm8k_subset main \
  --output_dir outputs/gsm8k_suite_scaling_lora_0.3B \
  --mode lora \
  --bf16 1 \
  --learning_rate 2e-4 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --lora_r 304 \
  --lora_alpha 304 \
  --run_name gsm8k-lora-8B \
  --eval_steps 300000 \
  --num_train_epochs 2 \
  --trainer sft \
  --report_to none
```


### 0.4B
parameters: 410,222,592

79.30

```bash
CUDA_VISIBLE_DEVICES=5 python run_shadow_peft.py \
  --model_name Qwen/Qwen3-8B \
  --suite gsm8k \
  --gsm8k_subset main \
  --output_dir outputs/gsm8k_suite_scaling_lora_0.4B \
  --mode lora \
  --bf16 1 \
  --learning_rate 2e-4 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --lora_r 428 \
  --lora_alpha 428 \
  --run_name gsm8k-lora-8B \
  --eval_steps 300000 \
  --num_train_epochs 2 \
  --trainer sft \
  --report_to none
```

### 0.5B

parameters: 502,235,136

77.10
```bash
CUDA_VISIBLE_DEVICES=6 python run_shadow_peft.py \
  --model_name Qwen/Qwen3-8B \
  --suite gsm8k \
  --gsm8k_subset main \
  --output_dir outputs/gsm8k_suite_scaling_lora_0.5B \
  --mode lora \
  --bf16 1 \
  --learning_rate 2e-4 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --lora_r 524 \
  --lora_alpha 524 \
  --run_name gsm8k-lora-8B \
  --eval_steps 300000 \
  --num_train_epochs 2 \
  --trainer sft \
  --report_to none
```

### 0.6B

parameters: 594,247,680

76.88
```bash
CUDA_VISIBLE_DEVICES=2 python run_shadow_peft.py \
  --model_name Qwen/Qwen3-8B \
  --suite gsm8k \
  --gsm8k_subset main \
  --output_dir outputs/gsm8k_suite_scaling_lora_0.6B \
  --mode lora \
  --bf16 1 \
  --learning_rate 2e-4 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --lora_r 620 \
  --lora_alpha 620 \
  --run_name gsm8k-lora-8B \
  --eval_steps 300000 \
  --num_train_epochs 2 \
  --trainer sft \
  --report_to none
```

## DoRA

### 0.1B

parameters: 123,052,032

81.12
```bash
CUDA_VISIBLE_DEVICES=5 python run_shadow_peft.py \
  --model_name Qwen/Qwen3-8B \
  --suite gsm8k \
  --gsm8k_subset main \
  --lora_r 128 \
  --lora_alpha 128 \
  --output_dir outputs/gsm8k_suite_scaling_dora_0.1B \
  --mode lora \
  --peft_method dora \
  --bf16 1 \
  --learning_rate 2e-4 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --run_name gsm8k-dora-8B \
  --eval_steps 300000 \
  --num_train_epochs 2 \
  --trainer sft \
  --report_to none
```



### 0.2B

parameters: 184,393,728
80.36
```bash
CUDA_VISIBLE_DEVICES=7 python run_shadow_peft.py \
  --model_name Qwen/Qwen3-8B \
  --suite gsm8k \
  --gsm8k_subset main \
  --lora_r 192 \
  --lora_alpha 192 \
  --output_dir outputs/gsm8k_suite_scaling_dora_0.2B \
  --mode lora \
  --peft_method dora \
  --bf16 1 \
  --learning_rate 2e-4 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --run_name gsm8k-dora-8B \
  --eval_steps 300000 \
  --num_train_epochs 2 \
  --trainer sft \
  --report_to none
```



### 0.3B

parameters: 291,741,696

79.76

```bash
CUDA_VISIBLE_DEVICES=1 python run_shadow_peft.py \
  --model_name Qwen/Qwen3-8B \
  --suite gsm8k \
  --gsm8k_subset main \
  --lora_r 304 \
  --lora_alpha 304 \
  --output_dir outputs/gsm8k_suite_scaling_dora_0.3B \
  --mode lora \
  --peft_method dora \
  --bf16 1 \
  --learning_rate 2e-4 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --run_name gsm8k-dora-8B \
  --eval_steps 300000 \
  --num_train_epochs 2 \
  --trainer sft \
  --report_to none
```


### 0.4B

parameters: 410,591,232

79.08

```bash
CUDA_VISIBLE_DEVICES=2 python run_shadow_peft.py \
  --model_name Qwen/Qwen3-8B \
  --suite gsm8k \
  --gsm8k_subset main \
  --lora_r 428 \
  --lora_alpha 428 \
  --output_dir outputs/gsm8k_suite_scaling_dora_0.4B \
  --mode lora \
  --peft_method dora \
  --bf16 1 \
  --learning_rate 2e-4 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --run_name gsm8k-dora-8B \
  --eval_steps 300000 \
  --num_train_epochs 2 \
  --trainer sft \
  --report_to none
```



### 0.5B

parameters: 502,603,776

77.79
```bash
CUDA_VISIBLE_DEVICES=5 python run_shadow_peft.py \
  --model_name Qwen/Qwen3-8B \
  --suite gsm8k \
  --gsm8k_subset main \
  --lora_r 524 \
  --lora_alpha 524 \
  --output_dir outputs/gsm8k_suite_scaling_dora_0.5B \
  --mode lora \
  --peft_method dora \
  --bf16 1 \
  --learning_rate 2e-4 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --run_name gsm8k-dora-8B \
  --eval_steps 300000 \
  --num_train_epochs 2 \
  --trainer sft \
  --report_to none
```



### 0.6B

parameters: 594,616,320

77.79 
```bash
CUDA_VISIBLE_DEVICES=7 python run_shadow_peft.py \
  --model_name Qwen/Qwen3-8B \
  --suite gsm8k \
  --gsm8k_subset main \
  --lora_r 620 \
  --lora_alpha 620 \
  --output_dir outputs/gsm8k_suite_scaling_dora_0.6B \
  --mode lora \
  --peft_method dora \
  --bf16 1 \
  --learning_rate 2e-4 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --run_name gsm8k-dora-8B \
  --eval_steps 300000 \
  --num_train_epochs 2 \
  --trainer sft \
  --report_to none
```

## Shadow

### 0.1B

parameters: 100,979,968

81.35 | 02.12
```bash
CUDA_VISIBLE_DEVICES=1 python run_shadow_peft.py \
  --model_name Qwen/Qwen3-8B \
  --suite gsm8k \
  --gsm8k_subset main \
  --output_dir outputs/gsm8k_suite_scaling_shadow_0.1B \
  --mode shadow \
  --bf16 1 \
  --injection_hidden_size 16 \
  --gate_hidden_size 8 \
  --shadow_intermediate_size 128 \
  --shadow_num_attention_heads 8 \
  --shadow_layers 5 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --run_name gsm8k-shadow-scaling \
  --eval_steps 300000 \
  --num_train_epochs 2 \
  --trainer sft \
  --shadow_loss_weight 0.05 \
  --shadow_alpha 0.1 \
  --shadow_dropout 0.2 \
  --learning_rate 3e-3 \
  --warmup_ratio 0.01 \
  --print_shadow_output 1 \
  --report_to none
```



### 0.2B

parameters: 192,772,608

81.50 | 2.20
```bash
CUDA_VISIBLE_DEVICES=2 python run_shadow_peft.py \
  --model_name Qwen/Qwen3-8B \
  --suite gsm8k \
  --gsm8k_subset main \
  --output_dir outputs/gsm8k_suite_scaling_shadow_0.2B \
  --mode shadow \
  --bf16 1 \
  --injection_hidden_size 16 \
  --gate_hidden_size 8 \
  --shadow_intermediate_size 128 \
  --shadow_num_attention_heads 8 \
  --shadow_layers 10 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --run_name gsm8k-shadow-scaling \
  --eval_steps 300000 \
  --num_train_epochs 2 \
  --trainer sft \
  --shadow_loss_weight 0.05 \
  --shadow_alpha 0.1 \
  --shadow_dropout 0.2 \
  --learning_rate 3e-3 \
  --warmup_ratio 0.01 \
  --print_shadow_output 1 \
  --report_to none
```



### 0.3B

parameters: 302,923,776

81.96 | 01.67
```bash
CUDA_VISIBLE_DEVICES=5 python run_shadow_peft.py \
  --model_name Qwen/Qwen3-8B \
  --suite gsm8k \
  --gsm8k_subset main \
  --output_dir outputs/gsm8k_suite_scaling_shadow_0.3B \
  --mode shadow \
  --bf16 1 \
  --injection_hidden_size 16 \
  --gate_hidden_size 8 \
  --shadow_intermediate_size 128 \
  --shadow_num_attention_heads 8 \
  --shadow_layers 16 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --run_name gsm8k-shadow-scaling \
  --eval_steps 300000 \
  --num_train_epochs 2 \
  --trainer sft \
  --shadow_loss_weight 0.05 \
  --shadow_alpha 0.1 \
  --shadow_dropout 0.2 \
  --learning_rate 3e-3 \
  --warmup_ratio 0.01 \
  --print_shadow_output 1 \
  --report_to none
```



### 0.4B

parameters: 413,074,944

82.18 | 1.82
```bash
CUDA_VISIBLE_DEVICES=7 python run_shadow_peft.py \
  --model_name Qwen/Qwen3-8B \
  --suite gsm8k \
  --gsm8k_subset main \
  --output_dir outputs/gsm8k_suite_scaling_shadow_0.4B \
  --mode shadow \
  --bf16 1 \
  --injection_hidden_size 16 \
  --gate_hidden_size 8 \
  --shadow_intermediate_size 128 \
  --shadow_num_attention_heads 8 \
  --shadow_layers 22 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --run_name gsm8k-shadow-scaling \
  --eval_steps 300000 \
  --num_train_epochs 2 \
  --trainer sft \
  --shadow_loss_weight 0.05 \
  --shadow_alpha 0.1 \
  --shadow_dropout 0.2 \
  --learning_rate 3e-3 \
  --warmup_ratio 0.01 \
  --print_shadow_output 1 \
  --report_to none
```


### 0.5B

parameters: 523,226,112

81.8 | 0.3
```bash
CUDA_VISIBLE_DEVICES=1 python run_shadow_peft.py \
  --model_name Qwen/Qwen3-8B \
  --suite gsm8k \
  --gsm8k_subset main \
  --output_dir outputs/gsm8k_suite_scaling_shadow_0.5B \
  --mode shadow \
  --bf16 1 \
  --injection_hidden_size 16 \
  --gate_hidden_size 8 \
  --shadow_intermediate_size 128 \
  --shadow_num_attention_heads 8 \
  --shadow_layers 28 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --run_name gsm8k-shadow-scaling \
  --eval_steps 300000 \
  --num_train_epochs 2 \
  --trainer sft \
  --shadow_loss_weight 0.05 \
  --shadow_alpha 0.1 \
  --shadow_dropout 0.2 \
  --learning_rate 3e-3 \
  --warmup_ratio 0.01 \
  --print_shadow_output 1 \
  --report_to none
```

# Amazon

## LoRA
### 0.1B
parameter: 122,683,392


```bash
CUDA_VISIBLE_DEVICES=0 python run_shadow_peft.py \
  --model_name Qwen/Qwen3-8B \
  --suite classification \
  --classification_id amazon_reviews_multi_en \
  --mode lora \
  --bf16 1 \
  --learning_rate 2e-5 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --lora_r 128 \
  --lora_alpha 128 \
  --output_dir outputs/amazone_suite_scaling_lora_0.1B \
  --eval_steps 5000 \
  --num_train_epochs 1 \
  --trainer sft \
  --report_to none
```


### 0.2B

parameter: 184,025,088


```bash
CUDA_VISIBLE_DEVICES=2 python run_shadow_peft.py \
  --model_name Qwen/Qwen3-8B \
  --suite classification \
  --classification_id amazon_reviews_multi_en \
  --output_dir outputs/amazon_suite_scaling_lora_0.2B \
  --mode lora \
  --bf16 1 \
  --learning_rate 2e-5 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --lora_r 192 \
  --lora_alpha 192 \
  --eval_steps 5000 \
  --num_train_epochs 1 \
  --trainer sft \
  --report_to none
```

### 0.3B
parameter: 291,373,056

```bash
CUDA_VISIBLE_DEVICES=5 python run_shadow_peft.py \
  --model_name Qwen/Qwen3-8B \
  --suite classification \
  --classification_id amazon_reviews_multi_en \
  --output_dir outputs/amazon_suite_scaling_lora_0.3B \
  --mode lora \
  --bf16 1 \
  --learning_rate 2e-5 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --lora_r 304 \
  --lora_alpha 304 \
  --eval_steps 5000 \
  --num_train_epochs 1 \
  --trainer sft \
  --report_to none
```


### 0.4B
parameters: 410,222,592

```bash
CUDA_VISIBLE_DEVICES=0 python run_shadow_peft.py \
  --model_name Qwen/Qwen3-8B \
  --suite classification \
  --classification_id amazon_reviews_multi_en \
  --output_dir outputs/amazon_suite_scaling_lora_0.4B \
  --mode lora \
  --bf16 1 \
  --learning_rate 2e-5 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --lora_r 428 \
  --lora_alpha 428 \
  --eval_steps 5000 \
  --num_train_epochs 1 \
  --trainer sft \
  --report_to none
```

### 0.5B 

parameters: 502,235,136

40.18
```bash
CUDA_VISIBLE_DEVICES=2 python run_shadow_peft.py \
  --model_name Qwen/Qwen3-8B \
  --suite classification \
  --classification_id amazon_reviews_multi_en \
  --output_dir outputs/amazon_suite_scaling_lora_0.5B \
  --lora_r 524 \
  --lora_alpha 524 \
  --mode lora \
  --peft_method lora \
  --bf16 1 \
  --learning_rate 2e-5 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --eval_steps 5000 \
  --num_train_epochs 1 \
  --trainer sft \
  --report_to none
```


## DoRA

### 0.1B

parameters: 123,052,032

40.34
```bash
CUDA_VISIBLE_DEVICES=5 python run_shadow_peft.py \
  --model_name Qwen/Qwen3-8B \
  --suite classification \
  --classification_id amazon_reviews_multi_en \
  --lora_r 128 \
  --lora_alpha 128 \
  --output_dir outputs/amazon_suite_scaling_dora_0.1B \
  --mode lora \
  --peft_method lora \
  --bf16 1 \
  --learning_rate 2e-5 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --eval_steps 5000 \
  --num_train_epochs 1 \
  --trainer sft \
  --report_to none
```



### 0.2B

parameters: 184,393,728

52.3
```bash
CUDA_VISIBLE_DEVICES=5 python run_shadow_peft.py \
  --model_name Qwen/Qwen3-8B \
  --suite classification \
  --classification_id amazon_reviews_multi_en \
  --lora_r 192 \
  --lora_alpha 192 \
  --output_dir outputs/amazon_suite_scaling_dora_0.2B \
  --mode lora \
  --peft_method dora \
  --bf16 1 \
  --learning_rate 2e-5 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --run_name gsm8k-dora-8B \
  --eval_steps 5000 \
  --num_train_epochs 1 \
  --trainer sft \
  --report_to none
```



### 0.3B 

parameters: 291,741,696

49.72
```bash
CUDA_VISIBLE_DEVICES=0 python run_shadow_peft.py \
  --model_name Qwen/Qwen3-8B \
  --suite classification \
  --classification_id amazon_reviews_multi_en \
  --lora_r 304 \
  --lora_alpha 304 \
  --output_dir outputs/amazon_suite_scaling_dora_0.3B \
  --mode lora \
  --peft_method dora \
  --bf16 1 \
  --learning_rate 2e-5 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --eval_steps 5000 \
  --num_train_epochs 1 \
  --trainer sft \
  --report_to none
```


### 0.4B

parameters: 410,591,232

25.44

```bash
CUDA_VISIBLE_DEVICES=0 python run_shadow_peft.py \
  --model_name Qwen/Qwen3-8B \
  --suite classification \
  --classification_id amazon_reviews_multi_en \
  --lora_r 428 \
  --lora_alpha 428 \
  --output_dir outputs/amazon_suite_scaling_dora_0.4B \
  --mode lora \
  --peft_method dora \
  --bf16 1 \
  --learning_rate 2e-5 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --eval_steps 5000 \
  --num_train_epochs 1 \
  --trainer sft \
  --report_to none
```



### 0.5B

parameters: 502,603,776

```bash
CUDA_VISIBLE_DEVICES=2 python run_shadow_peft.py \
  --model_name Qwen/Qwen3-8B \
  --suite classification \
  --classification_id amazon_reviews_multi_en \
  --lora_r 524 \
  --lora_alpha 524 \
  --output_dir outputs/amazon_suite_scaling_dora_0.5B \
  --mode lora \
  --peft_method dora \
  --bf16 1 \
  --learning_rate 2e-5 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --eval_steps 5000 \
  --num_train_epochs 1 \
  --trainer sft \
  --report_to none
```


## Shadow

### 0.1B 

```bash
CUDA_VISIBLE_DEVICES=5 python run_shadow_peft.py \
  --model_name Qwen/Qwen3-8B \
  --suite classification \
  --classification_id amazon_reviews_multi_en \
  --output_dir outputs/cls_suite_scaling_shadow_0.1B \
  --mode shadow \
  --bf16 1 \
  --injection_hidden_size 16 \
  --gate_hidden_size 8 \
  --shadow_intermediate_size 128 \
  --shadow_num_attention_heads 8 \
  --shadow_layers 5 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --run_name gsm8k-shadow-scaling \
  --eval_steps 300000 \
  --num_train_epochs 1 \
  --trainer sft \
  --shadow_loss_weight 0.05 \
  --shadow_alpha 0.1 \
  --shadow_dropout 0.2 \
  --learning_rate 2e-5 \
  --warmup_ratio 0.01 \
  --print_shadow_output 1 \
  --report_to none
```


### 0.2B
```bash
TRANSFORMERS_OFFLINE="1" CUDA_VISIBLE_DEVICES=1 python run_shadow_peft.py \
  --model_name Qwen/Qwen3-8B \
  --suite classification \
  --classification_id amazon_reviews_multi_en \
  --output_dir outputs/cls_suite_scaling_shadow_0.2B \
  --mode shadow \
  --bf16 1 \
  --injection_hidden_size 16 \
  --gate_hidden_size 8 \
  --shadow_intermediate_size 128 \
  --shadow_num_attention_heads 8 \
  --shadow_layers 10 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --run_name gsm8k-shadow-scaling \
  --eval_steps 5000 \
  --num_train_epochs 1 \
  --trainer sft \
  --shadow_loss_weight 0.1 \
  --shadow_alpha 1.0 \
  --shadow_dropout 0.2 \
  --shadow_lr_multiplier 10 \
  --learning_rate 2e-5 \
  --warmup_ratio 0.01 \
  --print_shadow_output 1 \
  --report_to none
```

### 0.3B

```bash
CUDA_VISIBLE_DEVICES=5 python run_shadow_peft.py \
  --model_name Qwen/Qwen3-8B \
  --suite classification \
  --classification_id amazon_reviews_multi_en \
  --output_dir outputs/cls_suite_scaling_shadow_0.3B \
  --mode shadow \
  --bf16 1 \
  --injection_hidden_size 16 \
  --gate_hidden_size 8 \
  --shadow_intermediate_size 128 \
  --shadow_num_attention_heads 8 \
  --shadow_layers 16 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --run_name gsm8k-shadow-scaling \
  --eval_steps 5000 \
  --num_train_epochs 1 \
  --trainer sft \
  --shadow_loss_weight 0.1 \
  --shadow_alpha 1.0 \
  --shadow_dropout 0.2 \
  --shadow_lr_multiplier 10 \
  --learning_rate 2e-5 \
  --warmup_ratio 0.01 \
  --print_shadow_output 1 \
  --report_to none
```


### 0.4B

```bash
CUDA_VISIBLE_DEVICES=0 python run_shadow_peft.py \
  --model_name Qwen/Qwen3-8B \
  --suite classification \
  --classification_id amazon_reviews_multi_en \
  --output_dir outputs/cls_suite_scaling_shadow_0.4B \
  --mode shadow \
  --bf16 1 \
  --injection_hidden_size 16 \
  --gate_hidden_size 8 \
  --shadow_intermediate_size 128 \
  --shadow_num_attention_heads 8 \
  --shadow_layers 22 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --run_name gsm8k-shadow-scaling \
  --eval_steps 5000 \
  --num_train_epochs 1 \
  --trainer sft \
  --shadow_loss_weight 0.2 \
  --shadow_alpha 1.0 \
  --shadow_dropout 0.2 \
  --shadow_lr_multiplier 10 \
  --learning_rate 2e-5 \
  --warmup_ratio 0.01 \
  --print_shadow_output 1 \
  --report_to none
```

### 0.5B

```bash
CUDA_VISIBLE_DEVICES=2 python run_shadow_peft.py \
  --model_name Qwen/Qwen3-8B \
  --suite classification \
  --classification_id amazon_reviews_multi_en \
  --output_dir outputs/cls_suite_scaling_shadow_0.5B \
  --mode shadow \
  --bf16 1 \
  --injection_hidden_size 16 \
  --gate_hidden_size 8 \
  --shadow_intermediate_size 128 \
  --shadow_num_attention_heads 8 \
  --shadow_layers 28 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --run_name gsm8k-shadow-scaling \
  --eval_steps 5000 \
  --num_train_epochs 1 \
  --trainer sft \
  --shadow_loss_weight 0.2 \
  --shadow_alpha 1.0 \
  --shadow_dropout 0.2 \
  --shadow_lr_multiplier 10 \
  --learning_rate 2e-5 \
  --warmup_ratio 0.01 \
  --print_shadow_output 1 \
  --report_to none
```