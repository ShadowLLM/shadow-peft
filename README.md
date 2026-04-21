# ShadowPEFT

**ShadowPEFT** is a parameter-efficient fine-tuning (PEFT) framework that augments a frozen large base model with a lightweight, centralized, and pretrainable *Shadow* network. The shadow network runs in parallel with the base model, injecting learned corrections into each decoder layer to enable effective adaptation with a fraction of the parameters. Once jointly trained with the base model, the shadow network can be detached and deployed independently, benefiting edge computing scenarios.

<p align="center">
  <img src="assets/ShadowPEFT-preview.png" alt="ShadowPEFT Preview" />
</p>

## How It Works

```
Input
  │
  ├──► Shadow Model (small, trainable) ──► shadow_hidden_states
  │
  └──► Base Model (frozen, large)
         │
         layer_0 ──────────────────────────────────────────────────► hidden_0
         layer_1 ◄── ShadowInjection(hidden_0, shadow[0]) ─────────► hidden_1
         layer_2 ◄── ShadowInjection(hidden_1, shadow[1]) ─────────► ...
         ...        [ShadowUpdate updates shadow state each step]
```

<p align="center">
  <img src="assets/ShadowPEFT-framework.png" alt="ShadowPEFT Framework" />
</p>

Three trainable components control the adaptation:

- **Shadow Model** — a small copy of the base architecture with fewer/smaller layers
- **ShadowInjectionModel** — projects the difference between base and shadow hidden states back into the base at each layer
- **ShadowUpdateModel** — uses a gated update to evolve the shadow hidden states as the base model processes each layer

---

## Table of Contents

- [Supported Models](#supported-models)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Examples](#examples)
- [Usage](#usage)
  - [1. Implicit Shadow Model](#1-implicit-shadow-model)
  - [2. Explicit Shadow Model (cross-architecture)](#2-explicit-shadow-model-cross-architecture)
  - [3. ShadowForCausalLM — generation & training](#3-shadowforcausallm--generation--training)
  - [4. ShadowForSequenceClassification](#4-shadowforsequenceclassification)
  - [5. AutoModelForCausalLMWithHiddenProjection](#5-automodelforcausallmwithhiddenprojection)
- [Configuration Reference](#configuration-reference)
- [Saving and Loading](#saving-and-loading)
  - [Save a checkpoint](#save-a-checkpoint)
  - [Load a checkpoint](#load-a-checkpoint)
  - [Push to the Hugging Face Hub](#push-to-the-hugging-face-hub)
  - [Load from the Hub](#load-from-the-hub)
- [Exporting the Shadow Model](#exporting-the-shadow-model)
- [Training with HF Trainer](#training-with-hf-trainer)
- [Notes and Limitations](#notes-and-limitations)

---

## Supported Models

ShadowPEFT is **architecture-agnostic** for most Hugging Face *decoder-only* transformer models whose decoder layer stack is accessible via one of:


| Attribute path               | Example architectures       |
| ---------------------------- | --------------------------- |
| `model.model.layers`         | LLaMA, Mistral, Qwen, Gemma |
| `model.transformer.h`        | GPT-2-style                 |
| `model.model.decoder.layers` | Some nested decoder layouts |


---

## Installation

```bash
uv pip install shadow-peft
```

or 

```bash
git clone https://github.com/ShadowLLM/shadow-peft.git
cd shadow-peft
uv pip install -e .
# Optional: dev/test dependencies
uv pip install -e ".[dev]"
```

**Requirements:** Python ≥ 3.10, PyTorch ≥ 2.1, Transformers > 5.0

---

## Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from shadow_peft import get_shadow_model, ShadowConfig

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

# Wrap the base model with a Shadow adapter (1-layer implicit shadow)
model = get_shadow_model(model, ShadowConfig(num_shadow_layers=1))
model.print_trainable_parameters()
# Trainable params: ~18M  ||  Total params: ~770M  ||  Trainable%: ~2.30%

# Only shadow-related parameters are trainable; base model is frozen.
```

---

## Examples

The `examples/` folder contains interactive playground notebooks for common ShadowPEFT workflows:

- [`examples/different_llm_backbones_playground.ipynb`](examples/different_llm_backbones_playground.ipynb) - explore ShadowPEFT across different LLM backbones
- [`examples/pretraining_shadow_via_pseudo_inverse.ipynb`](examples/pretraining_shadow_via_pseudo_inverse.ipynb) - initilize pretraining shadow model with the pseudo-inverse recipe
- [`examples/robot_intent_playground.ipynb`](examples/robot_intent_playground.ipynb) - robot intent generation
- [`examples/classification_playground.ipynb`](examples/classification_playground.ipynb) - experiment with sequence-classification workflows

---

## Usage

### 1. Implicit Shadow Model

The simplest way to use ShadowPEFT. A shadow model is automatically constructed from the same architecture as the base model, with fewer layers and optionally reduced MLP/attention sizes.

```python
from transformers import AutoModelForCausalLM
from shadow_peft import get_shadow_model, ShadowConfig

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")

shadow_config = ShadowConfig(
    num_shadow_layers=1,          # number of layers in the implicit shadow model
    injection_hidden_size=16,     # bottleneck dim for injection adapter
    gate_hidden_size=8,           # hidden dim for the update gate
    alpha=0.1,                    # scale factor for the injection delta
    dropout=0.1,

    # Optional: override implicit shadow model dimensions
    shadow_intermediate_size=None,       # MLP intermediate size (None = same as base)
    shadow_num_attention_heads=None,     # attention heads (None = same as base)
    shadow_num_key_value_heads=None,     # KV heads (None = same as base)
    shadow_head_dim=None,                # head dimension (None = same as base)
)

model = get_shadow_model(model, shadow_config)
model.print_trainable_parameters()
```

### 2. Explicit Shadow Model [Recommendation]

Use a separately pre-trained shadow model — for example, a smaller model that has been pre-trained to align with a larger base model's hidden space via `AutoModelForCausalLMWithHiddenProjection`.

When the shadow model's hidden size differs from the base model's hidden size, ShadowPEFT automatically inserts a `shadow_hidden_projection` linear layer to bridge the gap.

```python
from transformers import AutoModelForCausalLM
from shadow_peft import get_shadow_model, ShadowConfig, AutoModelForCausalLMWithHiddenProjection

# Large base model (frozen)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B")

# Pre-trained shadow model aligned to the 8B hidden space
shadow_model = AutoModelForCausalLMWithHiddenProjection.from_pretrained(
    "shadow-llm/Qwen3-0.6B-H8B"
)

shadow_config = ShadowConfig(
    injection_hidden_size=16,
    gate_hidden_size=8,
    alpha=0.1,
    dropout=0.1,
)

model = get_shadow_model(model, shadow_config, shadow_model=shadow_model)
model.print_trainable_parameters()
```

> **Tip:** When `shadow_model` carries a `shadow_hidden_projection` Linear layer (as produced by `AutoModelForCausalLMWithHiddenProjection`), ShadowPEFT reuses its trained weights instead of randomly initializing the projection.

### 3. `ShadowForCausalLM` — generation & training

`ShadowForCausalLM` is a task wrapper that adds a language modeling head to the Shadow setup. It supports two inference modes:


| Mode                      | `logits`           | `shadow_logits`    |
| ------------------------- | ------------------ | ------------------ |
| `"base_shadow"` (default) | Base model output  | Shadow path output |
| `"shadow_only"`           | Shadow path output | Shadow path output |


```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from shadow_peft import ShadowConfig, ShadowForCausalLM, get_shadow_model

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

# Pre-trained shadow model aligned to the 8B hidden space
shadow_model = AutoModelForCausalLMWithHiddenProjection.from_pretrained(
    "shadow-llm/Qwen3-0.6B-H8B"
)

shadow_config = ShadowConfig(
    injection_hidden_size=16,
    gate_hidden_size=8,
    alpha=0.1,
    dropout=0.1,
)

model = get_shadow_model(model, shadow_config, shadow_model=shadow_model)
model = ShadowForCausalLM(peft, inference_mode="base_shadow")

inputs = tokenizer("Hello", return_tensors="pt")

# base_shadow: returns both base logits and shadow logits
out = model(**inputs)
print(out.logits.shape)         # [1, seq_len, vocab]
print(out.shadow_logits.shape)  # [1, seq_len, vocab]

# Switch to shadow-only inference (lightweight, no base model forward pass)
model.set_inference_mode("shadow_only")
out = model(**inputs)
print(out.logits.shape)         # shadow logits only
```

**Training with labels:**

When `labels` are provided, `ShadowForCausalLM` computes a combined loss:

```
loss = base_CE_loss + shadow_loss_weight * shadow_CE_loss
```

```python
model = ShadowForCausalLM(peft, shadow_loss_weight=0.05)

inputs = tokenizer("Hello world", return_tensors="pt")
labels = inputs["input_ids"].clone()

out = model(**inputs, labels=labels)
print(out.loss)  # combined loss for backprop
```

**Text generation:**

KV cache is disabled inside Shadow; always pass `use_cache=False`:

```python
gen_ids = model.generate(**inputs, use_cache=False, max_new_tokens=32)
print(tokenizer.decode(gen_ids[0], skip_special_tokens=True))
```

**Loading from a saved checkpoint:**

```python
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B")
shadow_model = AutoModelForCausalLMWithHiddenProjection.from_pretrained(
    "shadow-llm/Qwen3-0.6B-H8B"
)
model = ShadowForCausalLM.from_pretrained(
    base,
    "/path/to/shadow_checkpoint",
    is_trainable=False,
    inference_mode="base_shadow",
    shadow_model=shadow_model,  # explicitly set shadow model
)
```

### 4. `ShadowForSequenceClassification`

Drop-in equivalent of `ShadowForCausalLM` for classification tasks.

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from shadow_peft import ShadowConfig, ShadowForSequenceClassification, get_shadow_model

base = AutoModelForSequenceClassification.from_pretrained(
    "Qwen/Qwen3-0.6B",
    num_labels=2,
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

peft = get_shadow_model(base, ShadowConfig(num_shadow_layers=1))
model = ShadowForSequenceClassification(peft, inference_mode="base_shadow")

inputs = tokenizer("This movie was great!", return_tensors="pt")

out = model(**inputs)
print(out.logits)         # base classifier logits [1, 2]
print(out.shadow_logits)  # shadow classifier logits [1, 2]

# Switch to shadow-only (no base forward pass)
model.set_inference_mode("shadow_only")
out = model(**inputs)
print(out.logits)  # shadow logits only
```

By default, **both** `classifier_head` and `shadow_classifier_head` are trainable. Use `ShadowConfig.modules_to_save` to control which heads are saved alongside the adapter:

```python
shadow_config = ShadowConfig(
    num_shadow_layers=1,
    modules_to_save=["classifier_head", "shadow_classifier_head"],
)
```

**Loading from a saved checkpoint:**

```python
base = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
model = ShadowForSequenceClassification.from_pretrained(
    base,
    "/path/to/shadow_checkpoint",
    is_trainable=False,
)
```

### 5. `AutoModelForCausalLMWithHiddenProjection`

A standalone HF-compatible model that wraps a small shadow backbone with:

- A **projection layer** mapping shadow hidden size → base hidden size
- A **frozen `lm_head`** from the larger base model

This is the canonical format for distributing pre-trained shadow models that target a larger base model's vocabulary space.

**Loading a pre-trained projected shadow model:**

```python
from shadow_peft import AutoModelForCausalLMWithHiddenProjection

# Load directly from the Hub (or a local path)
shadow_model = AutoModelForCausalLMWithHiddenProjection.from_pretrained(
    "shadow-llm/Qwen3-0.6B-H8B",
    freeze_backbone=False,      # keep backbone trainable (default)
    freeze_embed_tokens=True,   # freeze input embeddings (default)
    freeze_lm_head=True,        # freeze lm_head (default)
)
```

**Creating from scratch (wrapping existing models) via pseudo-inverse:**

```python
import torch.nn as nn
from transformers import AutoModelForCausalLM
from shadow_peft import AutoModelForCausalLMWithHiddenProjection

small = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
large = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B")

# Wrap: small backbone + projection (1024→4096) + large lm_head
wrapped = AutoModelForCausalLMWithHiddenProjection.wrap(
    shadow_model=small,
    shadow_hidden_projection=nn.Linear(1024, 4096, bias=False),
    lm_head=large.lm_head,
    # Optionally solve for the optimal initial projection via pseudoinverse:
    init_optimal_projection=True,
    reference_lm_head=small.lm_head,
)

wrapped.save_pretrained("/path/to/Qwen3-0.6B-H8B")
```

When `init_optimal_projection=True`, the projection is initialized to minimize `‖W_lm_large @ W_proj - W_lm_small‖`, providing a better starting point for fine-tuning.

---

## Configuration Reference

```python
from shadow_peft import ShadowConfig

ShadowConfig(
    # ── Shadow model architecture ──────────────────────────────────────────
    num_shadow_layers: int = 1,
    #   Number of transformer layers in the implicit shadow model.
    #   Ignored when an explicit shadow_model is provided.

    shadow_intermediate_size: int | None = None,
    #   Override the MLP intermediate size of the implicit shadow model.
    #   None = same as the base model.

    shadow_num_attention_heads: int | None = None,
    #   Override the number of attention heads. None = same as base.

    shadow_num_key_value_heads: int | None = None,
    #   Override the number of KV heads (GQA). None = same as base.

    shadow_head_dim: int | None = None,
    #   Override per-head dimension. None = same as base.

    # ── Adapter hyperparameters ────────────────────────────────────────────
    injection_hidden_size: int = 16,
    #   Bottleneck dimension of the ShadowInjectionModel.
    #   Larger = more expressive injection but more parameters.

    gate_hidden_size: int = 10,
    #   Hidden dimension of the ShadowUpdateModel gate.

    alpha: float = 0.1,
    #   Scale factor applied to the injection delta:
    #     hidden' = hidden + alpha * injection_delta

    dropout: float = 0.2,
    #   Dropout applied inside injection and update adapters.

    # ── Modules to save ────────────────────────────────────────────────────
    modules_to_save: list[str] = [],
    #   Extra modules to make trainable and persist in the checkpoint.
    #   CausalLM options:  ["lm_head", "shadow_lm_head"]
    #   SeqCls options:    ["classifier_head", "shadow_classifier_head"]
)
```

---

## Saving and Loading

### Save a checkpoint

Calling `save_pretrained` saves **only the adapter weights** (shadow model + injection/update modules), not the base model:

```python
# From ShadowPeftModel
# From ShadowForCausalLM or ShadowForSequenceClassification
# Also saves trainable task heads if modules_to_save is set.
model.save_pretrained("/path/to/checkpoint")
```

Saved files:

- `shadow_config.json` — adapter configuration
- `shadow_adapter.safetensors` — adapter weights (shadow model + injection + update)
- `shadow_modules.safetensors` — task-specific heads (if `modules_to_save` is set)

### Load a checkpoint

```python
from transformers import AutoModelForCausalLM
from shadow_peft import ShadowPeftModel

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
# implicit or explicit shadow model
shadow_model = None

# Inference (frozen)
model = ShadowPeftModel.from_pretrained(base, "/path/to/checkpoint", is_trainable=False, shadow_model=shadow_model)

# Resume training
model = ShadowPeftModel.from_pretrained(base, "/path/to/checkpoint", is_trainable=True, shadow_model=shadow_model)
```

Or use the task wrappers directly:

```python
from shadow_peft import ShadowForCausalLM

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
model = ShadowForCausalLM.from_pretrained(base, "/path/to/checkpoint", is_trainable=False, shadow_model=shadow_model)
```

### Push to the Hugging Face Hub

```python
# From ShadowPeftModel or ShadowForCausalLM / ShadowForSequenceClassification
model.push_to_hub(
    "your-org/my-shadow-adapter",
    commit_message="Add ShadowPEFT adapter for Qwen3-0.6B",
    private=True,
    token="hf_...",
)
```

### Load from the Hub

```python
from transformers import AutoModelForCausalLM
from shadow_peft import ShadowPeftModel

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
# implicit or explicit shadow model
shadow_model = None

# Supports repo_id or repo_id@revision
model = ShadowPeftModel.from_pretrained(base, "your-org/my-shadow-adapter", shadow_model=shadow_model)
```

---

## Exporting the Shadow Model

After training, you can extract the shadow backbone as a fully self-contained HF model — useful for independent evaluation or shadow-only inference:

```python
# Export a standalone HF model from the trained adapter
shadow_only = model.peft_model.export_shadow()
shadow_only.save_pretrained("/path/to/exported_shadow")

# Load and use it independently
import shadow_peft
from transformers import AutoModelForCausalLM
standalone = AutoModelForCausalLM.from_pretrained("/path/to/exported_shadow")
```

**When the shadow and base have different hidden sizes**, `export_shadow` returns an `AutoModelForCausalLMWithHiddenProjection` that bundles the backbone, the trained projection, and the base model's `lm_head` into a single loadable checkpoint.

---

## Training with HF Trainer

`ShadowForCausalLM` and `ShadowForSequenceClassification` are compatible with `transformers.Trainer`. The adapter's `state_dict` returns only the trainable adapter weights, so Trainer's safetensors checkpointing works without any patching.

```python
from transformers import Trainer, TrainingArguments
from shadow_peft import ShadowConfig, ShadowForCausalLM, get_shadow_model

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
peft = get_shadow_model(base, ShadowConfig(num_shadow_layers=1))
model = ShadowForCausalLM(peft, shadow_loss_weight=0.05)

training_args = TrainingArguments(
    output_dir="./shadow-output",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    # Gradient checkpointing is forwarded to the base model automatically:
    gradient_checkpointing=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=...,
)
trainer.train()

# Save only the adapter
model.save_pretrained("./shadow-checkpoint")
```

---

## Notes and Limitations

- **KV cache is disabled.** Shadow requires full-sequence processing to compute injections at every layer. `use_cache=False` is enforced automatically in all forward passes and generation calls.
- **Generation requires `use_cache=False`.** Some Transformers versions will still try to slice inputs when cache is active. Always pass it explicitly:
  ```python
  outputs = model.generate(input_ids, use_cache=False, max_new_tokens=64)
  ```
- **Base model is always frozen.** `ShadowPeftModel` sets `requires_grad=False` on all base model parameters during construction. If you need to fine-tune both base and shadow, manage `requires_grad` manually after wrapping.
- **Minimum 2 decoder layers required.** Shadow injection starts at layer 1, so the base model must have at least 2 decoder layers.
- **Embedding sharing.** For implicit shadow models, `embed_tokens` is removed from the shadow backbone and replaced by the base model's embeddings. This saves memory and keeps token representations consistent. Explicit shadow models keep their own embeddings by default; pass `remove_embed_tokens=True` to `prepare_shadow_model` to opt in to sharing.

## Credits

ShadowPEFT's API and code structure are heavily inspired by [PEFT](https://github.com/huggingface/peft) (Hugging Face). Concepts such as `get_shadow_model`, `ShadowPeftModel.from_pretrained / save_pretrained`, and `modules_to_save` deliberately mirror PEFT's conventions to provide a familiar experience for users already accustomed to LoRA and similar adapters.
