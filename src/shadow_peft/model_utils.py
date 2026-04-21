from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch
import torch.nn as nn
from transformers import PreTrainedModel


def _get_backbone(model: PreTrainedModel) -> nn.Module:
    """
    Return the module that contains the transformer decoder stack.

    This mirrors common HF naming conventions (LLaMA/Mistral/Qwen2/Qwen3: `.model`,
    GPT2/Falcon: `.transformer`).
    """
    for attr in ("model", "transformer", "base_model", "decoder"):
        backbone = getattr(model, attr, None)
        if backbone is not None:
            return backbone
    # Some HF classes are already the "backbone" (e.g. LlamaModel, Qwen3Model).
    if hasattr(model, "layers") and isinstance(model.layers, nn.ModuleList):
        return model
    if hasattr(model, "h") and isinstance(model.h, nn.ModuleList):
        return model
    if hasattr(model, "decoder") and hasattr(model.decoder, "layers") and isinstance(
        model.decoder.layers, nn.ModuleList
    ):
        return model.decoder
    raise AttributeError("Unable to locate transformer backbone inside the supplied model.")


def _get_decoder_layers(model: PreTrainedModel) -> tuple[nn.Module, nn.ModuleList, str]:
    """
    Return (backbone, layers, layers_attr_name).

    Supported:
    - backbone.layers (LLaMA/Mistral/Qwen)
    - backbone.h (GPT2-style)
    """
    backbone = _get_backbone(model)
    if hasattr(backbone, "layers") and isinstance(backbone.layers, nn.ModuleList):
        return backbone, backbone.layers, "layers"
    if hasattr(backbone, "h") and isinstance(backbone.h, nn.ModuleList):
        return backbone, backbone.h, "h"
    if hasattr(backbone, "decoder") and hasattr(backbone.decoder, "layers") and isinstance(
        backbone.decoder.layers, nn.ModuleList
    ):
        # Some model backbones (incl. some Gemma/encoder-decoder style internals) nest layers here.
        return backbone.decoder, backbone.decoder.layers, "layers"
    raise AttributeError(
        "Unsupported model: cannot find a ModuleList of decoder layers on the backbone "
        "(expected `.layers` or `.h`)."
    )


def _get_hidden_size(model: PreTrainedModel) -> int:
    cfg = model.config
    for attr in ("hidden_size", "n_embd", "d_model"):
        if hasattr(cfg, attr):
            return int(getattr(cfg, attr))
    raise AttributeError("Unable to infer hidden size from model.config.")


def _set_intermediate_size(cfg: Any, value: int) -> bool:
    """
    Best-effort set of the MLP/GLU intermediate size across common HF configs.

    Returns True if a known field was set.
    """
    for attr in ("intermediate_size", "ffn_dim", "n_inner"):
        if hasattr(cfg, attr):
            setattr(cfg, attr, int(value))
            return True
    return False


def _set_num_attention_heads(cfg: Any, value: int) -> bool:
    for attr in ("num_attention_heads", "n_head", "num_heads", "attention_heads"):
        if hasattr(cfg, attr):
            setattr(cfg, attr, int(value))
            return True
    return False


def _set_num_key_value_heads(cfg: Any, value: int) -> bool:
    for attr in ("num_key_value_heads", "num_kv_heads", "n_kv_head"):
        if hasattr(cfg, attr):
            setattr(cfg, attr, int(value))
            return True
    return False


def _set_head_dim(cfg: Any, value: int) -> bool:
    for attr in ("head_dim", "attention_head_dim"):
        if hasattr(cfg, attr):
            setattr(cfg, attr, int(value))
            return True
    return False


def _set_num_hidden_layers(cfg: Any, n: int) -> None:
    for attr in ("num_hidden_layers", "n_layer", "num_layers"):
        if hasattr(cfg, attr):
            setattr(cfg, attr, int(n))
            return
    raise AttributeError("Unable to set number of layers on this config.")


def _get_num_hidden_layers(cfg: Any) -> int:
    for attr in ("num_hidden_layers", "n_layer", "num_layers"):
        if hasattr(cfg, attr):
            return int(getattr(cfg, attr))
    raise AttributeError("Unable to get number of layers from this config.")


def build_implicit_shadow_model(
    base_model: PreTrainedModel,
    *,
    num_shadow_layers: int,
    shadow_intermediate_size: int | None = None,
    shadow_num_attention_heads: int | None = None,
    shadow_num_key_value_heads: int | None = None,
    shadow_head_dim: int | None = None,
) -> PreTrainedModel:
    """
    Create an implicit shadow model by instantiating the same class as `base_model`
    with a copied config but fewer layers and (optionally) smaller MLP size.

    This shadow model is randomly initialized (like the raw Qwen3 code path).
    """
    # Prefer instantiating the *backbone* class (e.g. Qwen3Model) rather than the task model
    # (e.g. Qwen3ForCausalLM) so the implicit shadow model stays lightweight.
    backbone = None
    for attr in ("model", "transformer", "base_model", "decoder"):
        cand = getattr(base_model, attr, None)
        # In HF, these are typically PreTrainedModel instances too.
        if isinstance(cand, PreTrainedModel):
            backbone = cand
            break
    if backbone is None:
        # Fall back to using the base model itself (covers cases where base_model is already the backbone).
        backbone = base_model

    cfg = deepcopy(backbone.config)
    if num_shadow_layers < 1:
        raise ValueError(f"num_shadow_layers must be >= 1, got {num_shadow_layers}")
    _set_num_hidden_layers(cfg, num_shadow_layers)

    # If explicitly provided, override the MLP/GLU intermediate size on the shadow config.
    if shadow_intermediate_size is not None:
        _set_intermediate_size(cfg, shadow_intermediate_size)

    # Optionally shrink attention projections (dominant param cost for large backbones).
    if shadow_num_attention_heads is not None and not _set_num_attention_heads(cfg, shadow_num_attention_heads):
        raise ValueError(
            "shadow_num_attention_heads was set, but this model config does not expose a "
            "recognized attention-heads field (e.g. num_attention_heads / n_head)."
        )
    if shadow_num_key_value_heads is not None and not _set_num_key_value_heads(cfg, shadow_num_key_value_heads):
        raise ValueError(
            "shadow_num_key_value_heads was set, but this model config does not expose a "
            "recognized kv-heads field (e.g. num_key_value_heads / num_kv_heads)."
        )
    if shadow_head_dim is not None and not _set_head_dim(cfg, shadow_head_dim):
        raise ValueError(
            "shadow_head_dim was set, but this model config does not expose a recognized "
            "head-dim field (e.g. head_dim)."
        )

    if hasattr(cfg, "use_cache"):
        cfg.use_cache = False

    shadow = backbone.__class__(cfg)
    return shadow


@torch.no_grad()
def mark_only_shadow_modules_trainable(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False

    for name, module in model.named_modules():
        if name.startswith("shadow_"):
            for p in module.parameters(recurse=True):
                p.requires_grad = True


def count_parameters(module: nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    total = sum(p.numel() for p in module.parameters())
    return trainable, total


