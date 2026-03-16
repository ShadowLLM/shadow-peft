from __future__ import annotations

import inspect
import tempfile
import weakref
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from safetensors.torch import load_file as safetensors_load_file
from safetensors.torch import save_file as safetensors_save_file
from torch.nn.modules.module import _IncompatibleKeys
from transformers import PreTrainedModel

from .config import ShadowConfig
from .model_utils import (
    _get_backbone,
    _get_decoder_layers,
    _get_hidden_size,
    build_implicit_shadow_model,
    count_parameters,
)
from .modules import ShadowInjectionModel, ShadowUpdateModel
from .projected_causal_lm import AutoModelForCausalLMWithHiddenProjection


def resolve_shadow_checkpoint_dir(pretrained_shadow_path: str | Path) -> Path:
    """
    Resolve a ShadowPEFT checkpoint directory.

    Accepts either:
    - a local directory path containing `shadow_config.json` / `shadow_adapter.safetensors`, or
    - a Hugging Face Hub repo id (optionally `repo_id@revision`), which will be downloaded.
    """
    p = Path(pretrained_shadow_path)
    if p.exists():
        return p

    repo_spec = str(pretrained_shadow_path)
    revision = None
    # Support `repo_id@revision` (common UX). Only attempt to parse when it looks like a hub id.
    if "@" in repo_spec and "/" in repo_spec and not repo_spec.startswith(("/", ".", "~")):
        repo_id, revision = repo_spec.rsplit("@", 1)
    else:
        repo_id = repo_spec

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Loading Shadow checkpoints from the Hugging Face Hub requires `huggingface_hub`. "
            "Install it with `pip install huggingface_hub`."
        ) from exc

    allow_patterns = [
        "shadow_config.json",
        "shadow_adapter.safetensors",
        "shadow_adapter.pt",
        "shadow_modules.safetensors",
    ]
    local_dir = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        allow_patterns=allow_patterns,
    )
    return Path(local_dir)


def _push_folder_to_hub(
    *,
    folder_path: str | Path,
    repo_id: str,
    commit_message: str = "Add ShadowPEFT adapter",
    private: bool = False,
    token: str | None = None,
    revision: str | None = None,
    create_pr: bool = False,
) -> str:
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Pushing Shadow checkpoints to the Hugging Face Hub requires `huggingface_hub`. "
            "Install it with `pip install huggingface_hub`."
        ) from exc

    create_repo(repo_id=repo_id, token=token, private=private, exist_ok=True)
    api = HfApi(token=token)
    commit_info = api.upload_folder(
        repo_id=repo_id,
        folder_path=str(folder_path),
        revision=revision,
        commit_message=commit_message,
        create_pr=create_pr,
    )
    # `commit_info` is a CommitInfo; return a stable string for callers.
    return getattr(commit_info, "commit_url", None) or getattr(commit_info, "url", None) or str(commit_info)


class _ShadowLayerWrapper(nn.Module):
    """
    Wrap a single decoder layer to apply Shadow injection/update.

    This is designed to be architecture-agnostic as long as the layer's first positional
    arg (or `hidden_states` kwarg) is the hidden state tensor, and the layer returns
    either:
      - a Tensor (hidden states)
      - a tuple whose first element is hidden states
    """

    def __init__(self, layer: nn.Module, *, layer_idx: int, adapter: ShadowPeftModel) -> None:
        super().__init__()
        self.layer = layer
        self.layer_idx = int(layer_idx)
        # IMPORTANT: don't store `adapter` (an nn.Module) as an attribute, otherwise this wrapper
        # registers it as a submodule and creates a cycle in `repr()`:
        # adapter -> base_model -> wrapped layer -> adapter
        object.__setattr__(self, "_adapter_ref", weakref.ref(adapter))

    def _get_adapter(self) -> ShadowPeftModel:
        adapter = self._adapter_ref()
        if adapter is None:
            raise RuntimeError("Shadow adapter reference is gone.")
        return adapter

    def __getattr__(self, name: str):
        """
        Delegate unknown attributes to the wrapped decoder layer.

        Some HF model implementations access per-layer attributes (e.g. Qwen3 uses
        `decoder_layer.attention_type`) while iterating over the ModuleList. Since we
        replace layers with wrappers, we must proxy those attributes.
        """
        try:
            return nn.Module.__getattr__(self, name)
        except AttributeError:
            layer = nn.Module.__getattr__(self, "layer")
            return getattr(layer, name)

    def forward(self, *args, **kwargs):
        adapter = self._get_adapter()
        # Extract hidden states.
        if "hidden_states" in kwargs:
            hidden_states = kwargs["hidden_states"]
            args = args
            use_kw = True
        elif len(args) > 0:
            hidden_states = args[0]
            use_kw = False
        else:
            raise TypeError("Decoder layer wrapper could not find hidden_states in args/kwargs.")

        # No-op for layer 0 (mirrors raw implementation: idx > 0).
        if self.layer_idx > 0:
            shadow = adapter._shadow_hidden_states
            if shadow is None:
                raise RuntimeError(
                    "Shadow state was not initialized. "
                    "Call the wrapper model's forward(), not the base model directly."
                )
            sidx = self.layer_idx - 1
            hidden_states = adapter.shadow_injection_model(hidden_states, shadow, sidx)
            if use_kw:
                kwargs["hidden_states"] = hidden_states
            else:
                args = (hidden_states,) + args[1:]

        out = self.layer(*args, **kwargs)

        # Pull hidden states back out to update shadow state.
        if isinstance(out, torch.Tensor):
            hs_out = out
            rest = None
        elif isinstance(out, tuple):
            hs_out = out[0]
            rest = out[1:]
        else:
            # Conservatively support HF ModelOutput-like objects.
            if hasattr(out, "hidden_states") and out.hidden_states is not None:
                hs_out = out.hidden_states
            elif hasattr(out, "last_hidden_state"):
                hs_out = out.last_hidden_state
            else:
                raise TypeError(
                    f"Unsupported decoder layer output type: {type(out)}. "
                    "Expected Tensor/tuple."
                )
            rest = None

        if self.layer_idx > 0:
            shadow = adapter._shadow_hidden_states
            sidx = self.layer_idx - 1
            adapter._shadow_hidden_states = adapter.shadow_update_model(hs_out, shadow, sidx)

        if rest is None:
            return out
        return (hs_out,) + rest


class ShadowPeftModel(nn.Module):
    """
    PEFT-style wrapper that augments a frozen base decoder-only model with Shadow modules.

    The base model is modified in-place by wrapping its decoder layers, but this wrapper
    owns adapter modules and provides save/load utilities.
    """

    def __init__(
        self,
        base_model: PreTrainedModel,
        shadow_config: ShadowConfig,
        *,
        shadow_model: PreTrainedModel | None = None,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.shadow_config = shadow_config

        # Freeze the base model.
        for p in self.base_model.parameters():
            p.requires_grad = False

        base_backbone, base_layers, base_layers_attr = _get_decoder_layers(self.base_model)

        num_base_layers = len(base_layers)
        if num_base_layers < 2:
            raise ValueError("Shadow requires at least 2 decoder layers to apply injection.")
        num_adapt_layers = num_base_layers - 1

        hidden_size = _get_hidden_size(self.base_model)

        # Shadow model: explicit or implicit (always keep *backbone-only* here).
        self.shadow_model: PreTrainedModel
        self._explicit_shadow_model = shadow_model is not None
        extracted_shadow_projection: nn.Linear | None = None
        if shadow_model is None:
            self.shadow_model = build_implicit_shadow_model(
                self.base_model,
                num_shadow_layers=shadow_config.num_shadow_layers,
                shadow_intermediate_size=shadow_config.shadow_intermediate_size,
                shadow_num_attention_heads=shadow_config.shadow_num_attention_heads,
                shadow_num_key_value_heads=shadow_config.shadow_num_key_value_heads,
                shadow_head_dim=shadow_config.shadow_head_dim,
            )
        else:
            if isinstance(shadow_model, AutoModelForCausalLMWithHiddenProjection):
                # If the provided explicit shadow model is a projected wrapper, extract the
                # projection weights for ShadowPeftModel.shadow_hidden_projection, but do NOT
                # keep the projection module inside `self.shadow_model` (otherwise strict loading
                # of `shadow_model.*` weights will fail).
                if hasattr(shadow_model, "shadow_hidden_projection") and isinstance(
                    shadow_model.shadow_hidden_projection, nn.Linear
                ):
                    extracted_shadow_projection = deepcopy(shadow_model.shadow_hidden_projection)
                shadow_model = prepare_shadow_model(shadow_model)
            else:
                # Generic explicit shadow model: if it already carries a projection module, extract
                # and remove it so it doesn't participate in the backbone state_dict.
                if hasattr(shadow_model, "shadow_hidden_projection") and isinstance(
                    shadow_model.shadow_hidden_projection, nn.Linear
                ):
                    extracted_shadow_projection = deepcopy(shadow_model.shadow_hidden_projection)
                    try:
                        delattr(shadow_model, "shadow_hidden_projection")
                    except Exception:
                        # Best-effort: remove from module registry if present.
                        shadow_model._modules.pop("shadow_hidden_projection", None)  # type: ignore[attr-defined]
            self.shadow_model = self._extract_backbone_model(shadow_model)

        # For explicit shadow models, embedding sharing/removal is opt-in via
        # `prepare_shadow_model(..., remove_embed_tokens=True)`.
        #
        # If the explicit shadow model has no input embeddings, we auto-enable embedding sharing
        # so the shadow backbone will be driven via base `inputs_embeds`.
        self._explicit_share_base_embeddings = False
        if self._explicit_shadow_model:
            try:
                get_inp = getattr(self.shadow_model, "get_input_embeddings", None)
                if callable(get_inp) and get_inp() is None:
                    self._explicit_share_base_embeddings = True
            except Exception:
                pass

        # If an explicit shadow model has a different hidden size than the base model,
        # project shadow hidden states into the base hidden size so injection/update
        # math (and downstream heads) remain compatible.
        shadow_hidden_size = _get_hidden_size(self.shadow_model)
        self.shadow_hidden_size = int(shadow_hidden_size)
        self.base_hidden_size = int(hidden_size)
        if int(shadow_hidden_size) != int(hidden_size):
            # If the caller provided an explicit shadow model that already includes a trained
            # shadow->base projection, reuse it instead of initializing a random one.
            # Expected shape: Linear(in=shadow_hidden_size, out=base_hidden_size, bias=False).
            cand = None
            if extracted_shadow_projection is not None:
                cand = extracted_shadow_projection
            else:
                for obj in (shadow_model, self.shadow_model):
                    if obj is None:
                        continue
                    c = getattr(obj, "shadow_hidden_projection", None)
                    if isinstance(c, nn.Linear):
                        cand = c
                        break

            if cand is not None and cand.in_features == int(shadow_hidden_size) and cand.out_features == int(hidden_size):
                # Clone weights so they are owned by this adapter (avoid double-registration).
                self.shadow_hidden_projection = deepcopy(cand)
            else:
                self.shadow_hidden_projection = nn.Linear(int(shadow_hidden_size), int(hidden_size), bias=False)
        else:
            # Keep an always-present module so state_dict/load_state_dict stay simple.
            self.shadow_hidden_projection = nn.Identity()

        # Share base input embeddings: we compute `inputs_embeds` from the base model and feed
        # them into the shadow backbone. When supported, we also remove shadow `embed_tokens`
        # so the shadow model doesn't keep a duplicate Embedding in memory/repr/state_dict.
        self._shadow_supports_inputs_embeds = self._configure_shadow_embedding_sharing()

        # Adapter modules (trainable).
        self.shadow_injection_model = ShadowInjectionModel(
            num_layers=num_adapt_layers,
            hidden_size=hidden_size,
            injection_hidden_size=shadow_config.injection_hidden_size,
            dropout=shadow_config.dropout,
            alpha=shadow_config.alpha,
        )
        self.shadow_update_model = ShadowUpdateModel(
            num_layers=num_adapt_layers,
            hidden_size=hidden_size,
            gate_hidden_size=shadow_config.gate_hidden_size,
            dropout=shadow_config.dropout,
        )

        # Internal mutable state during forward.
        self._shadow_hidden_states: torch.Tensor | None = None

        # Wrap base layers in-place.
        wrapped = nn.ModuleList([])
        for i, layer in enumerate(base_layers):
            if isinstance(layer, _ShadowLayerWrapper) and layer._adapter_ref() is self:
                wrapped.append(layer)
            else:
                wrapped.append(_ShadowLayerWrapper(layer, layer_idx=i, adapter=self))
        setattr(base_backbone, base_layers_attr, wrapped)

        # Ensure the shadow model is on the same device/dtype when `.to()` is called.
        self.shadow_model.to(next(self.base_model.parameters()).device)

    def __getattr__(self, name: str):
        """
        Delegate unknown attributes to the wrapped HF model.

        Important: we must preserve `nn.Module`'s default behavior of resolving submodules
        via `self._modules` (e.g. `self.base_model`). So we first try `nn.Module.__getattr__`,
        then fall back to `getattr(self.base_model, name)`.
        """
        try:
            return nn.Module.__getattr__(self, name)
        except AttributeError:
            base = nn.Module.__getattr__(self, "base_model")
            return getattr(base, name)

    def print_trainable_parameters(self) -> None:
        trainable, total = count_parameters(self)
        pct = (100.0 * trainable / total) if total else 0.0
        print(
            f"Trainable params: {trainable:,} || Total params: {total:,} || Trainable%: {pct:.2f}%"
        )

    def _compute_initial_shadow_hidden(
        self,
        *,
        input_ids: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None,
        **kwargs: Any,
    ) -> torch.Tensor:
        # We call the shadow *backbone* to get last_hidden_state (no logits needed).
        shadow_backbone = _get_backbone(self.shadow_model)

        # Embedding policy:
        # - implicit shadow model: prefer base embeddings (share) and remove shadow embed_tokens
        # - explicit shadow model: keep/use its own embed_tokens by default
        # - explicit shadow model (opt-in): share base embeddings + remove shadow embed_tokens
        share_base = (not self._explicit_shadow_model) or self._explicit_share_base_embeddings
        if self._shadow_supports_inputs_embeds and inputs_embeds is None and share_base:
            if input_ids is None:
                raise ValueError("Either `input_ids` or `inputs_embeds` must be provided.")
            base_embed = self.base_model.get_input_embeddings()
            if base_embed is None:
                raise AttributeError("Base model does not expose input embeddings.")
            inputs_embeds = base_embed(input_ids)

        # Make a best-effort to disable caching, since Shadow updates require full sequences.
        kwargs = dict(kwargs)
        kwargs["use_cache"] = False
        kwargs["past_key_values"] = None
        kwargs["output_hidden_states"] = False
        kwargs["return_dict"] = True

        # If we have `inputs_embeds`, use it. Otherwise, fall back to `input_ids` so an explicit
        # shadow model can use its own embedding table.
        if self._shadow_supports_inputs_embeds and inputs_embeds is not None:
            out = shadow_backbone(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **kwargs,
            )
        else:
            # Fallback for unusual models that don't support inputs_embeds.
            out = shadow_backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **kwargs,
            )

        if hasattr(out, "last_hidden_state"):
            shadow_hidden = out.last_hidden_state
            # Ensure shadow hidden state matches base hidden size.
            if self.shadow_hidden_projection is not None:
                shadow_hidden = self.shadow_hidden_projection(shadow_hidden)
            return shadow_hidden
        raise TypeError(
            "Shadow backbone did not return an object with `last_hidden_state`. "
            "This model architecture is not currently supported."
        )

    @staticmethod
    def _remove_embed_tokens(module: nn.Module) -> None:
        if hasattr(module, "embed_tokens") and isinstance(module.embed_tokens, nn.Module):
            # Setting to None removes it from `module._modules` and from repr/state_dict.
            try:
                module.embed_tokens = None
            except Exception:
                # Some models may implement embed_tokens as a read-only property.
                module._modules.pop("embed_tokens", None)

    @staticmethod
    def _extract_backbone_model(model: PreTrainedModel) -> PreTrainedModel:
        """
        Normalize an explicitly provided HF task model to its backbone-only model.

        Example:
        - Qwen3ForCausalLM -> Qwen3Model (via `.model`)
        - SequenceClassification heads are similarly stripped.
        """
        for attr in ("model", "transformer", "base_model", "decoder"):
            cand = getattr(model, attr, None)
            if isinstance(cand, PreTrainedModel):
                return cand
        return model

    def _configure_shadow_embedding_sharing(self) -> bool:
        """
        Returns True if shadow backbone supports `inputs_embeds`.

        If supported, we remove `embed_tokens` from the shadow model/backbone so the shadow
        architecture doesn't carry a duplicate embedding table.
        """
        shadow_backbone = _get_backbone(self.shadow_model)
        try:
            sig = inspect.signature(shadow_backbone.forward)
            supports_inputs_embeds = "inputs_embeds" in sig.parameters
        except (TypeError, ValueError):
            supports_inputs_embeds = False

        # Remove embed_tokens when we intend to feed `inputs_embeds` instead.
        # - implicit shadow models: always share base embeddings (memory save)
        # - explicit shadow models: only when embed_tokens were removed via `prepare_shadow_model`
        if supports_inputs_embeds and (
            not getattr(self, "_explicit_shadow_model", False) or getattr(self, "_explicit_share_base_embeddings", False)
        ):
            # Remove from both the outer container (if any) and the actual backbone module.
            self._remove_embed_tokens(self.shadow_model)
            if shadow_backbone is not self.shadow_model:
                self._remove_embed_tokens(shadow_backbone)
        return supports_inputs_embeds

    def forward(self, *args, **kwargs):
        # Initialize per-forward shadow state.
        input_ids = kwargs.get("input_ids")
        attention_mask = kwargs.get("attention_mask")
        position_ids = kwargs.get("position_ids")
        inputs_embeds = kwargs.get("inputs_embeds")

        self._shadow_hidden_states = self._compute_initial_shadow_hidden(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )

        # Force no-cache (same reason as raw implementation).
        kwargs["use_cache"] = False
        kwargs["past_key_values"] = None

        try:
            return self.base_model(*args, **kwargs)
        finally:
            # Avoid holding onto activations across calls.
            self._shadow_hidden_states = None

    def forward_with_shadow(self, *args, **kwargs) -> tuple[object, torch.Tensor]:
        """
        Forward the wrapped base model and also return the **initial** `shadow_hidden_states`
        produced by the shadow model backbone.

        This is useful for task wrappers (CausalLM / SeqCls) that want to compute
        `shadow_logits` for inference and/or auxiliary loss.
        """
        # Initialize per-forward shadow state.
        input_ids = kwargs.get("input_ids")
        attention_mask = kwargs.get("attention_mask")
        position_ids = kwargs.get("position_ids")
        inputs_embeds = kwargs.get("inputs_embeds")

        initial_shadow_hidden = self._compute_initial_shadow_hidden(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )
        self._shadow_hidden_states = initial_shadow_hidden

        # Force no-cache (same reason as raw implementation).
        kwargs["use_cache"] = False
        kwargs["past_key_values"] = None

        try:
            outputs = self.base_model(*args, **kwargs)
            return outputs, initial_shadow_hidden
        finally:
            self._shadow_hidden_states = None

    def adapter_state_dict(self) -> dict[str, torch.Tensor]:
        state: dict[str, torch.Tensor] = {}
        for prefix, module in [
            ("shadow_model", self.shadow_model),
            ("shadow_hidden_projection", self.shadow_hidden_projection),
            ("shadow_injection_model", self.shadow_injection_model),
            ("shadow_update_model", self.shadow_update_model),
        ]:
            for k, v in module.state_dict().items():
                t = v.detach().cpu()
                if t.is_sparse:
                    t = t.to_dense()
                state[f"{prefix}.{k}"] = t.contiguous()
        return state

    def state_dict(self, destination=None, prefix: str = "", keep_vars: bool = False):  # type: ignore[override]
        """
        Return **adapter-only** state for compatibility with `transformers.Trainer` safetensors saving.

        Why: most HF causal LMs tie `embed_tokens.weight` and `lm_head.weight`, which makes
        `safetensors.torch.save_file(state_dict)` raise when saving a *full* model state dict.
        Shadow-PEFT checkpoints are intended to store only:
        - shadow backbone weights
        - shadow injection/update weights
        """
        if destination is None:
            destination = {}
        if not isinstance(destination, dict):
            # Best-effort: support OrderedDict-like destinations.
            destination = dict(destination)

        if keep_vars:
            state: dict[str, torch.Tensor] = {}
            for pfx, module in [
                ("shadow_model", self.shadow_model),
                ("shadow_hidden_projection", self.shadow_hidden_projection),
                ("shadow_injection_model", self.shadow_injection_model),
                ("shadow_update_model", self.shadow_update_model),
            ]:
                for k, v in module.state_dict(prefix="", keep_vars=True).items():
                    state[f"{pfx}.{k}"] = v
        else:
            state = self.adapter_state_dict()

        if prefix:
            state = {f"{prefix}{k}": v for k, v in state.items()}

        destination.update(state)
        return destination

    def load_state_dict(self, state_dict: Mapping[str, torch.Tensor], strict: bool = True, assign: bool = False):  # type: ignore[override]
        """
        Load **adapter-only** checkpoints saved by `state_dict()` above.

        If the caller passes a full HF model state dict, we ignore everything except
        adapter keys. This keeps `Trainer` resume flows working without requiring the
        base model weights to be present in the checkpoint.
        """
        # Materialize to a plain dict and strip common wrappers.
        sd = dict(state_dict)
        if any(k.startswith("module.") for k in sd):
            sd = {k[len("module.") :]: v for k, v in sd.items()}
        if any(k.startswith("peft_model.") for k in sd):
            sd = {k[len("peft_model.") :]: v for k, v in sd.items()}

        adapter_prefixes = (
            "shadow_model.",
            "shadow_hidden_projection.",
            "shadow_injection_model.",
            "shadow_update_model.",
        )
        adapter_sd = {k: v for k, v in sd.items() if k.startswith(adapter_prefixes)}

        if not adapter_sd:
            if strict:
                raise RuntimeError(
                    "ShadowPeftModel expected an adapter-only checkpoint (keys starting with "
                    f"{adapter_prefixes}), but none were found."
                )
            return _IncompatibleKeys(missing_keys=[], unexpected_keys=list(sd.keys()))

        self.load_adapter_state_dict(adapter_sd)
        # We intentionally ignore non-adapter keys.
        return _IncompatibleKeys(missing_keys=[], unexpected_keys=[])

    def load_adapter_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        # Split by prefix.
        def sub(prefix: str) -> dict[str, torch.Tensor]:
            out: dict[str, torch.Tensor] = {}
            for k, v in state.items():
                if k.startswith(prefix + "."):
                    out[k[len(prefix) + 1 :]] = v
            return out

        self.shadow_model.load_state_dict(sub("shadow_model"), strict=True)
        # Backward compatibility: old checkpoints won't have projection weights.
        proj_sd = sub("shadow_hidden_projection")
        if proj_sd:
            print("Loading shadow hidden projection weights")
            self.shadow_hidden_projection.load_state_dict(proj_sd, strict=True)
        self.shadow_injection_model.load_state_dict(sub("shadow_injection_model"), strict=True)
        self.shadow_update_model.load_state_dict(sub("shadow_update_model"), strict=True)

    @torch.no_grad()
    def export_shadow(self) -> PreTrainedModel:
        """
        Export a standalone, HF-compatible **shadow model** suitable for pretraining/inference.

        This reconstructs a *task model* (same HF class as `base_model`) whose transformer
        backbone weights come from this adapter's `shadow_model`, but whose `embed_tokens`
        and `lm_head` come from the base model (they may have been removed for embedding
        sharing during Shadow training).

        Returns
        -------
        PreTrainedModel
            A decoupled model with independent parameters (no shared storage with `base_model`).
        """
        shadow_h = int(getattr(self, "shadow_hidden_size", _get_hidden_size(self.shadow_model)))
        base_h = int(getattr(self, "base_hidden_size", _get_hidden_size(self.base_model)))
        hidden_match = shadow_h == base_h

        base_embed = self.base_model.get_input_embeddings()
        base_head = self.base_model.get_output_embeddings()

        # Choose a stable dtype/device for export.
        # Using `next(self.base_model.parameters())` is unreliable because many models keep some
        # params in fp32 (e.g. norms) even when the main weights are bf16/fp16.
        target_device = None
        target_dtype = None
        try:
            if base_head is not None and hasattr(base_head, "weight") and base_head.weight is not None:
                target_device = base_head.weight.device
                target_dtype = base_head.weight.dtype
            elif base_embed is not None and hasattr(base_embed, "weight") and base_embed.weight is not None:
                target_device = base_embed.weight.device
                target_dtype = base_embed.weight.dtype
        except Exception:
            target_device = None
            target_dtype = None

        # Shadow backbone config (e.g. fewer layers).
        shadow_cfg = deepcopy(self.shadow_model.config)
        shadow_cfg = self._normalize_export_config(shadow_cfg)

        # Make the exported model fully self-contained.
        #
        # Case A: hidden sizes match -> copy base embeddings/head (original behavior).
        # If hidden sizes differ, we still export a standard HF model (loadable by AutoModel*).
        # Use `AutoModelForCausalLMWithHiddenProjection` if you want a single checkpoint that includes projection + head.
        if hidden_match:
            exported = self.base_model.__class__(shadow_cfg)
            # Load shadow backbone weights.
            exported_backbone = _get_backbone(exported)
            shadow_backbone = _get_backbone(self.shadow_model)
            missing, unexpected = exported_backbone.load_state_dict(shadow_backbone.state_dict(), strict=False)
            _ = (missing, unexpected)

            if base_embed is None:
                raise AttributeError("Base model does not expose input embeddings (`get_input_embeddings()`).")
            if base_head is None:
                raise AttributeError(
                    "Base model does not expose output embeddings / lm_head (`get_output_embeddings()`)."
                )
            exported.set_input_embeddings(deepcopy(base_embed))
            exported.set_output_embeddings(deepcopy(base_head))

            # Re-tie weights if the architecture expects it (common for decoder-only LMs).
            tie_fn = getattr(exported, "tie_weights", None)
            if callable(tie_fn):
                tie_fn()
        else:
            # Hidden sizes differ (e.g. 1024-dim shadow + 4096-dim base).
            # If a trained shadow_hidden_projection is present (Linear, not Identity),
            # bundle backbone + projection + base lm_head into
            # AutoModelForCausalLMWithHiddenProjection so the checkpoint is fully
            # self-contained and directly usable for shadow-only inference.
            has_proj = (
                isinstance(getattr(self, "shadow_hidden_projection", None), nn.Linear)
                and base_head is not None
            )

            if has_proj:
                from .projected_causal_lm import AutoModelForCausalLMWithHiddenProjection

                # Build a task model carrying the shadow backbone weights.
                shadow_task = self.base_model.__class__(shadow_cfg)
                exported_bb = _get_backbone(shadow_task)
                shadow_bb   = _get_backbone(self.shadow_model)
                exported_bb.load_state_dict(shadow_bb.state_dict(), strict=False)

                # Restore shadow embed_tokens (may have been removed during training).
                shadow_embed = None
                get_inp = getattr(self.shadow_model, "get_input_embeddings", None)
                if callable(get_inp):
                    try:
                        shadow_embed = get_inp()
                    except Exception:
                        shadow_embed = None
                if shadow_embed is not None:
                    shadow_task.set_input_embeddings(deepcopy(shadow_embed))

                # Wrap backbone + trained projection + frozen lm_head.
                exported = AutoModelForCausalLMWithHiddenProjection.wrap(
                    shadow_model=shadow_task,
                    shadow_hidden_projection=deepcopy(self.shadow_hidden_projection),
                    lm_head=deepcopy(base_head),
                    init_optimal_projection=False,  # keep the trained weights
                )
            else:
                # Fallback: export a plain task model (backbone weights only).
                exported = self.base_model.__class__(shadow_cfg)
                exported_backbone = _get_backbone(exported)
                shadow_backbone = _get_backbone(self.shadow_model)
                exported_backbone.load_state_dict(shadow_backbone.state_dict(), strict=False)

                shadow_embed = None
                get_inp = getattr(self.shadow_model, "get_input_embeddings", None)
                if callable(get_inp):
                    try:
                        shadow_embed = get_inp()
                    except Exception:
                        shadow_embed = None
                if shadow_embed is not None:
                    exported.set_input_embeddings(deepcopy(shadow_embed))

        # Match device/dtype to the base model's head/embeddings (preferred) or fall back.
        if target_device is None or target_dtype is None:
            base_param = next(self.base_model.parameters(), None)
            if base_param is not None:
                target_device = base_param.device
                target_dtype = base_param.dtype
        if target_device is not None and target_dtype is not None:
            exported = exported.to(device=target_device, dtype=target_dtype)

        return exported

    @staticmethod
    def _normalize_export_config(cfg):
        """
        Best-effort fixups so exported HF configs can be reloaded.

        Some model configs store per-layer lists (e.g. Qwen3 `layer_types`) that must match
        `num_hidden_layers`. When exporting a smaller shadow backbone, those lists can become
        inconsistent (e.g. num_hidden_layers=1 but layer_types has length 28), causing
        `AutoConfig.from_pretrained()` to raise.
        """
        # Ensure `layer_types` length matches `num_hidden_layers` if both exist.
        try:
            num_layers = int(cfg.num_hidden_layers)
        except Exception:
            num_layers = None

        if num_layers is not None and hasattr(cfg, "layer_types"):
            lt = getattr(cfg, "layer_types", None)
            if lt is None:
                pass
            else:
                # Normalize to a Python list.
                if isinstance(lt, tuple):
                    lt_list = list(lt)
                elif isinstance(lt, list):
                    lt_list = lt
                else:
                    # Unexpected type; try to coerce.
                    lt_list = list(lt)  # type: ignore[arg-type]

                if len(lt_list) == 0:
                    # If missing, choose a safe default.
                    lt_list = ["full_attention"] * num_layers
                elif len(lt_list) > num_layers:
                    lt_list = lt_list[:num_layers]
                elif len(lt_list) < num_layers:
                    lt_list = lt_list + [lt_list[-1]] * (num_layers - len(lt_list))
                cfg.layer_types = lt_list

        # Some configs also carry max-window metadata tied to layer count.
        if num_layers is not None and hasattr(cfg, "max_window_layers"):
            try:
                mw = cfg.max_window_layers
                if mw is not None and int(mw) > num_layers:
                    cfg.max_window_layers = int(num_layers)
            except Exception:
                pass

        return cfg

    def save_pretrained(self, save_directory: str | Path) -> None:
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.shadow_config.save_pretrained(save_dir)
        safetensors_save_file(
            self.adapter_state_dict(),
            str(save_dir / "shadow_adapter.safetensors"),
        )

    def push_to_hub(
        self,
        repo_id: str,
        *,
        commit_message: str = "Add ShadowPEFT adapter",
        private: bool = False,
        token: str | None = None,
        revision: str | None = None,
        create_pr: bool = False,
    ) -> str:
        """
        Push this ShadowPEFT adapter checkpoint to the Hugging Face Hub.

        The pushed files include only adapter artifacts (no backbone weights):
        - shadow_config.json
        - shadow_adapter.safetensors
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            self.save_pretrained(tmpdir)
            return _push_folder_to_hub(
                folder_path=tmpdir,
                repo_id=repo_id,
                commit_message=commit_message,
                private=private,
                token=token,
                revision=revision,
                create_pr=create_pr,
            )

    @classmethod
    def from_pretrained(
        cls,
        model: PreTrainedModel,
        pretrained_shadow_path: str | Path,
        *,
        is_trainable: bool = False,
        shadow_model: PreTrainedModel | None = None,
    ) -> ShadowPeftModel:
        ckpt_dir = resolve_shadow_checkpoint_dir(pretrained_shadow_path)
        cfg = ShadowConfig.from_pretrained(ckpt_dir)
        peft_model = cls(model, cfg, shadow_model=shadow_model)
        st_path = ckpt_dir / "shadow_adapter.safetensors"
        pt_path = ckpt_dir / "shadow_adapter.pt"
        if st_path.exists():
            state = safetensors_load_file(str(st_path))
        elif pt_path.exists():
            # Backward compatibility.
            state = torch.load(pt_path, map_location="cpu")
        else:
            raise FileNotFoundError(
                f"Missing adapter checkpoint. Expected {st_path.name} (preferred) or {pt_path.name} in {ckpt_dir}"
            )
        peft_model.load_adapter_state_dict(state)
        peft_model.train(is_trainable)
        for p in peft_model.parameters():
            p.requires_grad = False
        if is_trainable:
            for p in peft_model.shadow_model.parameters():
                p.requires_grad = True
            # Train projection when present (needed for hidden-size mismatch).
            for p in peft_model.shadow_hidden_projection.parameters():
                p.requires_grad = True
            for p in peft_model.shadow_injection_model.parameters():
                p.requires_grad = True
            for p in peft_model.shadow_update_model.parameters():
                p.requires_grad = True
        return peft_model


def get_shadow_model(
    model: PreTrainedModel,
    shadow_config: ShadowConfig,
    *,
    shadow_model: PreTrainedModel | None = None,
) -> ShadowPeftModel:
    return ShadowPeftModel(model, shadow_config, shadow_model=shadow_model)


def prepare_shadow_model(
    shadow_model: PreTrainedModel,
    *,
    remove_embed_tokens: bool = False,
) -> PreTrainedModel:
    """
    Prepare an **explicit** shadow model for ShadowPEFT.

    This is a small UX helper that:
    - strips task heads and keeps backbone-only (same behavior as ShadowPEFT internally)
    - optionally removes `embed_tokens` so the shadow backbone can be driven by base `inputs_embeds`

    When `remove_embed_tokens=True`, ShadowPEFT will automatically detect the missing input
    embeddings and switch to embedding-sharing mode (no need to set a long config flag).
    """
    if isinstance(shadow_model, AutoModelForCausalLMWithHiddenProjection):
        shadow_model = shadow_model.shadow_model

    backbone = ShadowPeftModel._extract_backbone_model(shadow_model)
    if remove_embed_tokens:
        # Remove from both the outer container (if any) and backbone module.
        ShadowPeftModel._remove_embed_tokens(shadow_model)
        if backbone is not shadow_model:
            ShadowPeftModel._remove_embed_tokens(backbone)
    return backbone
