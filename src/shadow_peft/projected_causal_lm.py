from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from .model_utils import _get_backbone


def _extract_backbone_model(model: PreTrainedModel) -> PreTrainedModel:
    """
    Normalize a task model to its backbone-only model.

    Example:
    - Qwen3ForCausalLM -> Qwen3Model (via `.model`)
    """
    for attr in ("model", "transformer", "base_model", "decoder"):
        cand = getattr(model, attr, None)
        if isinstance(cand, PreTrainedModel):
            return cand
    return model


def _import_from_path(path: str):
    """
    Import "some.module:ClassName" or "some.module.ClassName" and return the symbol.
    """
    if ":" in path:
        mod, name = path.split(":", 1)
    else:
        mod, name = path.rsplit(".", 1)
    m = importlib.import_module(mod)
    return getattr(m, name)


def _shifted_ce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    # Standard causal LM loss: predict token t+1 from token t.
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


class AutoModelForCausalLMWithHiddenProjectionConfig(PretrainedConfig):
    """
    Config for a projected CausalLM:

      shadow_causal_lm (hidden=shadow_h)
        -> shadow_hidden_projection (shadow_h -> base_h)
        -> lm_head (base_h -> vocab)

    This class is intentionally default-constructible (Transformers calls `config.__class__()`
    in some save/push flows).
    """

    model_type = "causal_lm_with_hidden_projection"

    def __init__(
        self,
        *,
        shadow_model_class: str = "",
        shadow_model_config_class: str = "",
        shadow_model_config: dict[str, Any] | None = None,
        base_hidden_size: int = 0,
        vocab_size: int = 0,
        **kwargs,
    ) -> None:
        # CRITICAL: Set hidden_size to base_hidden_size so Transformers utilities
        # (like weight tying, config inspection) use the correct dimension.
        # The shadow model's internal hidden size is stored separately in shadow_model_config.
        if "hidden_size" not in kwargs and base_hidden_size > 0:
            kwargs["hidden_size"] = int(base_hidden_size)
        
        super().__init__(vocab_size=int(vocab_size or kwargs.get("vocab_size", 0)), **kwargs)
        self.shadow_model_class = str(shadow_model_class)
        self.shadow_model_config_class = str(shadow_model_config_class)
        self.shadow_model_config = dict(shadow_model_config or {})
        self.base_hidden_size = int(base_hidden_size)
        
        # Force hidden_size to match base_hidden_size (in case from_dict overwrites it).
        if self.base_hidden_size > 0:
            self.hidden_size = int(self.base_hidden_size)
        
        # Expose shadow model attributes needed for generation and other HF utilities.
        # These delegate to the shadow_model_config dict.
        _shadow_cfg = self.shadow_model_config
        if _shadow_cfg:
            # Copy key attributes from shadow model config that are needed for generation.
            for attr in (
                "num_hidden_layers",
                "num_attention_heads",
                "num_key_value_heads",
                "intermediate_size",
                "max_position_embeddings",
                "rope_theta",
                "rms_norm_eps",
                "layer_types",
                "sliding_window",
                "attention_dropout",
                "head_dim",
                "use_sliding_window",
                "max_window_layers",
            ):
                if attr in _shadow_cfg and not hasattr(self, attr):
                    setattr(self, attr, _shadow_cfg[attr])


@dataclass
class AutoModelForCausalLMWithHiddenProjectionOutput(CausalLMOutputWithPast):
    """
    Same as CausalLMOutputWithPast but explicit for this model type.
    """


class AutoModelForCausalLMWithHiddenProjection(PreTrainedModel, GenerationMixin):
    """
    A CausalLM wrapper that is still saved/loaded like a normal HF model, but projects
    the shadow hidden size to a base hidden size before applying `lm_head`.
    """

    config_class = AutoModelForCausalLMWithHiddenProjectionConfig

    # Required by transformers ≥ 4.47 GenerationMixin._supports_default_dynamic_cache().
    _is_stateful: bool = False

    def __init__(self, config: AutoModelForCausalLMWithHiddenProjectionConfig) -> None:
        super().__init__(config)

        if not config.shadow_model_class:
            raise ValueError("Missing `shadow_model_class` in AutoModelForCausalLMWithHiddenProjectionConfig.")
        if not config.shadow_model_config_class:
            raise ValueError("Missing `shadow_model_config_class` in AutoModelForCausalLMWithHiddenProjectionConfig.")
        if not config.shadow_model_config:
            raise ValueError("Missing `shadow_model_config` in AutoModelForCausalLMWithHiddenProjectionConfig.")
        if int(getattr(config, "base_hidden_size", 0)) <= 0:
            raise ValueError("Missing/invalid `base_hidden_size` in AutoModelForCausalLMWithHiddenProjectionConfig.")
        if int(getattr(config, "vocab_size", 0)) <= 0:
            raise ValueError("Missing/invalid `vocab_size` in AutoModelForCausalLMWithHiddenProjectionConfig.")

        cfg_cls = _import_from_path(config.shadow_model_config_class)
        if not issubclass(cfg_cls, PretrainedConfig):
            raise TypeError(
                f"shadow_model_config_class must be a PretrainedConfig, got {cfg_cls} from {config.shadow_model_config_class}"
            )
        shadow_cfg = cfg_cls.from_dict(config.shadow_model_config)

        # Infer shadow hidden size from config FIRST (before instantiating any models).
        shadow_hidden_size = int(getattr(shadow_cfg, "hidden_size", getattr(shadow_cfg, "n_embd", 0)))
        if shadow_hidden_size <= 0:
            raise ValueError("Could not infer shadow hidden size from shadow model config.")

        # CRITICAL: Instantiate ONLY the backbone class (never a task model with lm_head).
        # If shadow_model_class points to a task model (e.g., Qwen3ForCausalLM),
        # infer and instantiate the backbone class (e.g., Qwen3Model) instead.
        model_cls = _import_from_path(config.shadow_model_class)
        model_cls_name = model_cls.__name__

        # Detect and convert task model class names to backbone class names.
        if model_cls_name.endswith("ForCausalLM"):
            backbone_cls_name = model_cls_name.replace("ForCausalLM", "Model")
            try:
                backbone_cls = getattr(__import__(model_cls.__module__, fromlist=[backbone_cls_name]), backbone_cls_name)
                self.shadow_model = backbone_cls(shadow_cfg)
            except (AttributeError, ImportError) as e:
                raise ValueError(
                    f"Could not instantiate backbone class '{backbone_cls_name}' from module '{model_cls.__module__}'. "
                    f"Please update the saved config's 'shadow_model_class' to point directly to the backbone "
                    f"(e.g., 'transformers.models.qwen3.modeling_qwen3:Qwen3Model' instead of ':Qwen3ForCausalLM'). "
                    f"Error: {e}"
                ) from e
        else:
            # Assume it's already a backbone class or will work as-is.
            self.shadow_model = model_cls(shadow_cfg)
            # If it has a .model attribute, extract it.
            if hasattr(self.shadow_model, "model") and isinstance(self.shadow_model.model, nn.Module):
                backbone = self.shadow_model.model
                if hasattr(self.shadow_model, "config") and not hasattr(backbone, "config"):
                    backbone.config = self.shadow_model.config
                self.shadow_model = backbone
        
        # Verify shadow_model has no lm_head (which would cause loading conflicts).
        if hasattr(self.shadow_model, "lm_head"):
            import warnings
            warnings.warn(
                f"shadow_model ({type(self.shadow_model).__name__}) has an 'lm_head' attribute, "
                "which may cause weight loading conflicts. Consider using the backbone class instead."
            )

        self.shadow_hidden_projection = nn.Linear(shadow_hidden_size, int(config.base_hidden_size), bias=False)
        self.lm_head = nn.Linear(int(config.base_hidden_size), int(config.vocab_size), bias=False)

        # Keep module order in repr: projection before lm_head.
        try:
            lm = self._modules.pop("lm_head", None)
            if lm is not None:
                self._modules["lm_head"] = lm
        except Exception:
            pass

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *model_args,
        freeze_backbone: bool = False,
        freeze_embed_tokens: bool = True,
        freeze_lm_head: bool = True,
        **kwargs,
    ) -> AutoModelForCausalLMWithHiddenProjection:
        """
        Load a pretrained AutoModelForCausalLMWithHiddenProjection from disk or Hub.

        Parameters
        ----------
        pretrained_model_name_or_path : str
            Path to the saved model directory or Hub repo ID.
        freeze_backbone : bool, default False
            If True, set `requires_grad=False` on the shadow model's parameters.
        freeze_embed_tokens : bool, default False
            If True, set `requires_grad=False` on the shadow model's input embeddings.
        freeze_lm_head : bool, default False
            If True, set `requires_grad=False` on the `lm_head`.
        **kwargs
            Additional arguments passed to the base `PreTrainedModel.from_pretrained`.

        Returns
        -------
        AutoModelForCausalLMWithHiddenProjection
            The loaded model with optional frozen layers.
        """
        # Remove custom kwargs before passing to super
        kwargs_for_super = {k: v for k, v in kwargs.items() if k not in ("freeze_backbone", "freeze_embed_tokens", "freeze_lm_head")}
        
        # Use the standard HF loading mechanism
        model = super().from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs_for_super
        )

        if freeze_backbone:
            for param in model.shadow_model.parameters():
                param.requires_grad = False

        # Apply freezing if requested
        if freeze_embed_tokens:
            embed = model.get_input_embeddings()
            if embed is not None:
                for param in embed.parameters():
                    param.requires_grad = False
        
        if freeze_lm_head:
            for param in model.lm_head.parameters():
                param.requires_grad = False
        
        return model

    @classmethod
    def wrap(
        cls,
        *,
        shadow_model: PreTrainedModel,
        shadow_hidden_projection: nn.Linear,
        lm_head: nn.Module,
        init_optimal_projection: bool = True,
        reference_lm_head: nn.Module | None = None,
    ) -> AutoModelForCausalLMWithHiddenProjection:
        """
        Convenience constructor to wrap an already-instantiated shadow model + projection + head.
        
        This creates a new AutoModelForCausalLMWithHiddenProjection by wrapping the provided components.
        
        Parameters
        ----------
        shadow_model : PreTrainedModel
            The shadow model (will be extracted to backbone-only).
        shadow_hidden_projection : nn.Linear
            Projection layer (shadow_hidden -> base_hidden).
        lm_head : nn.Module
            Language modeling head (base_hidden -> vocab).
        init_optimal_projection : bool, default True
            If True, initialize shadow_hidden_projection optimally using pseudoinverse to approximate
            `reference_lm_head`. This is useful when adapting a model trained with a different lm_head
            (e.g., using a larger model's lm_head with a smaller model's backbone).
        reference_lm_head : nn.Module, optional
            The original model's lm_head to approximate. Required if `init_optimal_projection=True`.
            Should have shape [vocab, shadow_hidden] where shadow_hidden is the shadow model's hidden size.
        
        Returns
        -------
        AutoModelForCausalLMWithHiddenProjection
            The wrapped model with optionally optimized projection.
        
        Example
        -------
        >>> # Adapt Qwen3-0.6B to use Qwen3-8B's lm_head
        >>> small_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
        >>> large_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B")
        >>> 
        >>> wrapped = AutoModelForCausalLMWithHiddenProjection.wrap(
        ...     shadow_model=small_model,
        ...     shadow_hidden_projection=nn.Linear(1024, 4096, bias=False),
        ...     lm_head=large_model.lm_head,
        ...     init_optimal_projection=True,
        ...     reference_lm_head=small_model.lm_head,
        ... )
        """
        shadow_backbone = _extract_backbone_model(shadow_model)
        shadow_cfg_dict = shadow_backbone.config.to_dict()
        
        cfg = AutoModelForCausalLMWithHiddenProjectionConfig(
            shadow_model_class=f"{shadow_backbone.__class__.__module__}:{shadow_backbone.__class__.__name__}",
            shadow_model_config_class=f"{shadow_backbone.config.__class__.__module__}:{shadow_backbone.config.__class__.__name__}",
            shadow_model_config=shadow_cfg_dict,
            base_hidden_size=int(getattr(lm_head, "in_features", getattr(cls, "base_hidden_size", 0))),
            vocab_size=int(getattr(shadow_backbone.config, "vocab_size", 0) or getattr(lm_head, "out_features", 0)),
        )
        out = cls(cfg)
        # Match device/dtype to the provided shadow model before loading weights (prevents fp32 params).
        ref = next(shadow_backbone.parameters(), None)
        if ref is not None:
            out = out.to(device=ref.device, dtype=ref.dtype)

        out.shadow_model.load_state_dict(shadow_backbone.state_dict(), strict=True)
        
        # Optionally initialize projection optimally using pseudoinverse
        if init_optimal_projection:
            if reference_lm_head is None:
                raise ValueError(
                    "When init_optimal_projection=True, you must provide reference_lm_head "
                    "(the original model's lm_head to approximate)."
                )

            print("Initializing shadow_hidden_projection optimally via pseudoinverse (it will take a few minutes, please wait)...")
            W_old = reference_lm_head.weight.data  # [vocab, shadow_hidden]
            W_lm_frozen = lm_head.weight.data  # [vocab, base_hidden]

            print(f"  Reference lm_head shape: {W_old.shape}")
            print(f"  Target lm_head shape: {W_lm_frozen.shape}")

            # Solve: W_lm_frozen @ W_proj.T = W_old
            # => W_proj.T = pinv(W_lm_frozen) @ W_old
            # Compute pseudoinverse (use float32 for numerical stability)
            W_lm_pinv = torch.linalg.pinv(W_lm_frozen.float())  # [base_hidden, vocab]
            W_proj_optimal_T = W_lm_pinv @ W_old.float()  # [base_hidden, shadow_hidden]

            # shadow_hidden_projection.weight expects shape [out_features, in_features]
            # = [base_hidden, shadow_hidden]
            out.shadow_hidden_projection.weight.data = W_proj_optimal_T.to(
                out.shadow_hidden_projection.weight.dtype
            )
            
            # Verify approximation quality
            reconstructed = out.lm_head.weight @ out.shadow_hidden_projection.weight
            reconstruction_error = (reconstructed - W_old.to(reconstructed.dtype)).norm() / W_old.norm()
            print(f"  Reconstruction error: {reconstruction_error.item():.6f}")
            print("  ✓ Optimal projection initialized")
        else:
            # Use provided projection weights
            out.shadow_hidden_projection.load_state_dict(shadow_hidden_projection.state_dict(), strict=True)

        out.lm_head.load_state_dict(lm_head.state_dict(), strict=True)
        return out

    def get_input_embeddings(self):
        get_inp = getattr(self.shadow_model, "get_input_embeddings", None)
        return get_inp() if callable(get_inp) else None

    def set_input_embeddings(self, value):
        set_inp = getattr(self.shadow_model, "set_input_embeddings", None)
        if callable(set_inp):
            return set_inp(value)
        raise AttributeError("Underlying shadow model does not support set_input_embeddings().")

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def tie_weights(self):
        """
        Override to prevent automatic weight tying.
        Our lm_head and embeddings are intentionally separate (different hidden sizes).
        """
        pass

    def _init_weights(self, module):
        """
        Override to prevent automatic weight initialization during loading.
        Weights will be loaded from checkpoint instead.
        """
        pass

    def prepare_inputs_for_generation(self, *args, **kwargs):
        # Delegate if possible (HF models often implement this for KV cache).
        fn = getattr(self.shadow_model, "prepare_inputs_for_generation", None)
        if callable(fn):
            return fn(*args, **kwargs)
        return super().prepare_inputs_for_generation(*args, **kwargs)

    def _reorder_cache(self, past_key_values, beam_idx):
        fn = getattr(self.shadow_model, "_reorder_cache", None)
        if callable(fn):
            return fn(past_key_values, beam_idx)
        return past_key_values

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        use_cache: bool | None = None,
        past_key_values=None,
        return_dict: bool | None = None,
        **kwargs,
    ):
        if return_dict is None:
            return_dict = True

        # Call the shadow *backbone* to get hidden states, then apply projection + base head.
        backbone = _get_backbone(self.shadow_model)
        out = backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=True,
            **kwargs,
        )

        hidden = out.last_hidden_state
        hidden_base = self.shadow_hidden_projection(hidden)
        logits = self.lm_head(hidden_base)

        loss = None
        if labels is not None:
            loss = _shifted_ce_loss(logits, labels)

        if not return_dict:
            return (loss, logits, getattr(out, "past_key_values", None), None, None) if loss is not None else (
                logits,
                getattr(out, "past_key_values", None),
                None,
                None,
            )

        return AutoModelForCausalLMWithHiddenProjectionOutput(
            loss=loss,
            logits=logits,
            past_key_values=getattr(out, "past_key_values", None),
            hidden_states=getattr(out, "hidden_states", None),
            attentions=getattr(out, "attentions", None),
        )


