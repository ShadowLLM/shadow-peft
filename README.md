# Shadow-PEFT

Shadow Efficient Tuning for decoder-only LLMs (PEFT-style adapter).

### Supported LLMs

This implementation is **architecture-agnostic** for most Hugging Face *decoder-only* transformer models that expose a decoder layer stack under one of:
- `model.model.layers` (common: LLaMA, Mistral, Qwen, Gemma)
- `model.transformer.h` (GPT2-style)
- `model.model.decoder.layers` (some nested layouts)


### Install (uv)

```bash
git clone https://github.com/ShadowLLM/shadow-peft.git
cd shadow-peft
uv pip install -e .
uv pip install -e ".[dev]"
```

### Usage

#### 1) Implicit `shadow_model`

```python
from transformers import AutoModelForCausalLM
from shadow_peft import get_shadow_model, ShadowConfig

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")

shadow_config = ShadowConfig(
    num_shadow_layers=1,  # used for implicit shadow model init
    # Optional: force the implicit shadow model's MLP/GLU intermediate size
    # (when provided).
    shadow_intermediate_size=None,
    injection_hidden_size=16,
    gate_hidden_size=10,
    alpha=0.1,
    dropout=0.1,
)

model = get_shadow_model(model, shadow_config)  # wraps base model in-place
model.print_trainable_parameters()
```

#### 2) Explicit `shadow_model`

```python
from transformers import AutoModelForCausalLM
from shadow_peft import get_shadow_model, ShadowConfig

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B")
shadow_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")

shadow_config = ShadowConfig(
    injection_hidden_size=16,
    gate_hidden_size=10,
    alpha=0.1,
    dropout=0.1,
)

model = get_shadow_model(model, shadow_config, shadow_model=shadow_model)
model.print_trainable_parameters()
```

#### 3) `ShadowForCausalLM` (base+shadow vs shadow-only inference)

`ShadowForCausalLM` is a task wrapper that returns:
- **base+shadow**: `logits` (base path) + `shadow_logits` (shadow-only path)
- **shadow-only**: `logits == shadow_logits` (shadow-only path)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from shadow_peft import ShadowConfig, ShadowForCausalLM, get_shadow_model

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

peft = get_shadow_model(base, ShadowConfig(num_shadow_layers=1))
model = ShadowForCausalLM(peft, inference_mode="base_shadow")

inputs = tokenizer("Hello", return_tensors="pt")
out = model(**inputs)
print(out.logits.shape, out.shadow_logits.shape)

# Shadow-only inference
model.set_inference_mode("shadow_only")
out2 = model(**inputs)
print(out2.logits.shape)  # shadow-only logits
```

Generation (note: cache is disabled for Shadow):

```python
gen_ids = model.generate(**inputs, use_cache=False, max_new_tokens=16)
print(tokenizer.decode(gen_ids[0], skip_special_tokens=True))
```

#### 4) `ShadowForSequenceClassification` (base+shadow vs shadow-only inference)

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from shadow_peft import ShadowConfig, ShadowForSequenceClassification, get_shadow_model

base = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2,
)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

peft = get_shadow_model(base, ShadowConfig(num_shadow_layers=1))
model = ShadowForSequenceClassification(peft, inference_mode="base_shadow")

inputs = tokenizer("good movie", return_tensors="pt")
out = model(**inputs)
print(out.logits, out.shadow_logits)

model.set_inference_mode("shadow_only")
out2 = model(**inputs)
print(out2.logits)  # shadow-only logits
```

### Saving / Loading

#### Save

```python
model.save_pretrained("/path/to/save")
```

This saves **only**:
- `shadow_model` parameters
- `shadow_injection_model` parameters
- `shadow_update_model` parameters
- `shadow_config.json`
 - `shadow_adapter.safetensors`

#### Load

```python
from transformers import AutoModelForCausalLM
from shadow_peft import ShadowPeftModel

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
model = ShadowPeftModel.from_pretrained(base, "/path/to/save", is_trainable=False)
```

### Notes / Limitations

- **KV cache is disabled** inside the wrapper forward pass (`use_cache=False`) because Shadow updates require full-sequence processing.
- If you use `generate()`, set `use_cache=False` in generation config/kwargs to avoid input slicing behavior in some Transformers versions:

```python
outputs = model.generate(input_ids, use_cache=False)
```
