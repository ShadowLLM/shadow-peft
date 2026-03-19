from __future__ import annotations

import logging
import tempfile
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file as safetensors_load_file
from safetensors.torch import save_file as safetensors_save_file
from transformers import GenerationConfig, GenerationMixin, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast, SequenceClassifierOutput

from .peft_model import ShadowPeftModel, _push_folder_to_hub, resolve_shadow_checkpoint_dir

InferenceMode = Literal["base_shadow", "shadow_only"]

_SHADOW_MODULES_FILE = "shadow_modules.safetensors"

logger = logging.getLogger(__name__)


def _to_cpu_dense(t: torch.Tensor) -> torch.Tensor:
    out = t.detach().cpu()
    if out.is_sparse:
        out = out.to_dense()
    return out.contiguous()


def _modules_to_save_state_dict(modules: dict[str, nn.Module]) -> dict[str, torch.Tensor]:
    """
    Flatten per-module state_dicts into a single dict with keys like:
      "{module_name}.{param_name}"
    """
    state: dict[str, torch.Tensor] = {}
    for module_name, module in modules.items():
        if module is None:
            continue
        for k, v in module.state_dict().items():
            state[f"{module_name}.{k}"] = _to_cpu_dense(v)
    return state


def _is_trainable_module(module: nn.Module) -> bool:
    return any(getattr(p, "requires_grad", False) for p in module.parameters())


def _save_modules_to_save(
    save_directory: str | Path,
    modules: dict[str, nn.Module],
    *,
    requested_modules: list[str] | None = None,
) -> None:
    """
    Save only modules that are trainable, optionally constrained by `requested_modules`.

    This mirrors PEFT behavior where "modules_to_save" are the extra trainable modules
    persisted alongside adapter weights.
    """
    if requested_modules:
        candidates = {k: v for k, v in modules.items() if k in set(requested_modules)}
    else:
        candidates = dict(modules)
    candidates = {k: v for k, v in candidates.items() if v is not None and _is_trainable_module(v)}
    state = _modules_to_save_state_dict(candidates)
    if not state:
        return
    save_dir = Path(save_directory)
    save_dir.mkdir(parents=True, exist_ok=True)
    safetensors_save_file(state, str(save_dir / _SHADOW_MODULES_FILE))


def _load_modules_to_save_state(pretrained_shadow_path: str | Path) -> dict[str, torch.Tensor]:
    ckpt_dir = resolve_shadow_checkpoint_dir(pretrained_shadow_path)
    st_path = ckpt_dir / _SHADOW_MODULES_FILE
    if not st_path.exists():
        return {}
    return safetensors_load_file(str(st_path))


def _load_modules_to_save_into(modules: dict[str, nn.Module], flat_state: dict[str, torch.Tensor]) -> None:
    if not flat_state:
        return
    for module_name, module in modules.items():
        prefix = f"{module_name}."
        sub = {k[len(prefix) :]: v for k, v in flat_state.items() if k.startswith(prefix)}
        if not sub:
            continue
        # Be strict when we have weights for this module; a mismatch should surface early.
        module.load_state_dict(sub, strict=True)


def _shifted_ce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    # Standard causal LM loss: predict token t+1 from token t.
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


@dataclass
class ShadowCausalLMOutputWithPast(CausalLMOutputWithPast):
    shadow_logits: torch.Tensor | None = None


@dataclass
class ShadowSequenceClassifierOutput(SequenceClassifierOutput):
    shadow_logits: torch.Tensor | None = None


class ShadowForCausalLM(nn.Module, GenerationMixin):
    """
    Task wrapper for causal LM.

    - **base_shadow**: returns both `logits` (base path) and `shadow_logits` (shadow path).
    - **shadow_only**: returns `logits` equal to shadow logits (and also in `shadow_logits`).
    """

    main_input_name = "input_ids"

    # Required by transformers ≥ 4.47 GenerationMixin._supports_default_dynamic_cache().
    # False = standard stateless model (no special cache handling needed).
    _is_stateful: bool = False

    def __init__(
        self,
        peft_model: ShadowPeftModel,
        *,
        shadow_loss_weight: float = 0.05,
        inference_mode: InferenceMode = "base_shadow",
    ) -> None:
        super().__init__()
        self.peft_model = peft_model
        self.shadow_loss_weight = float(shadow_loss_weight)
        self.inference_mode: InferenceMode = inference_mode

        # Expose config/generation_config like a HF model.
        self.config = getattr(self.peft_model.base_model, "config", None)
        if hasattr(self.config, "use_cache"):
            self.config.use_cache = False
        base_gen_cfg = getattr(self.peft_model.base_model, "generation_config", None)
        self.generation_config = (
            base_gen_cfg if base_gen_cfg is not None else GenerationConfig.from_model_config(self.config)
        )
        if hasattr(self.generation_config, "use_cache"):
            self.generation_config.use_cache = False

        # Heads: use base model embeddings/head if available.
        self.lm_head = self.peft_model.base_model.get_output_embeddings()
        if self.lm_head is None:
            raise AttributeError("Base model does not expose output embeddings/lm_head.")
        self.shadow_lm_head = deepcopy(self.lm_head)

        # Keep heads frozen to satisfy "save only Shadow/Injection/Update" requirement.
        for p in self.lm_head.parameters():
            p.requires_grad = False
        for p in self.shadow_lm_head.parameters():
            p.requires_grad = False

        # If configured, allow training (and thus saving) specific task modules.
        cfg = getattr(self.peft_model, "shadow_config", None)
        requested = list(getattr(cfg, "modules_to_save", []) or [])
        if "lm_head" in requested:
            if not _is_trainable_module(self.lm_head):
                logger.info(
                    "ShadowConfig.modules_to_save requested lm_head; enabling requires_grad=True for lm_head."
                )
            for p in self.lm_head.parameters():
                p.requires_grad = True
        if "shadow_lm_head" in requested:
            if not _is_trainable_module(self.shadow_lm_head):
                logger.info(
                    "ShadowConfig.modules_to_save requested shadow_lm_head; enabling requires_grad=True for shadow_lm_head."
                )
            for p in self.shadow_lm_head.parameters():
                p.requires_grad = True

    def save_pretrained(self, save_directory: str | Path) -> None:
        self.peft_model.save_pretrained(save_directory)
        cfg = getattr(self.peft_model, "shadow_config", None)
        requested = list(getattr(cfg, "modules_to_save", []) or [])
        # Save only trainable task modules (optionally requested).
        _save_modules_to_save(
            save_directory,
            {
                "lm_head": self.lm_head,
                "shadow_lm_head": self.shadow_lm_head,
            },
            requested_modules=(requested or None),
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
        Push this ShadowPEFT checkpoint (adapter + optional task modules) to the Hub.

        This never uploads backbone weights.
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

    def state_dict(self, destination=None, prefix: str = "", keep_vars: bool = False):  # type: ignore[override]
        # Delegate to adapter-only state dict to keep `Trainer` safetensors saves working.
        return self.peft_model.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], strict: bool = True, assign: bool = False):  # type: ignore[override]
        return self.peft_model.load_state_dict(state_dict, strict=strict, assign=assign)

    @classmethod
    def from_pretrained(
        cls,
        model: PreTrainedModel,
        pretrained_shadow_path: str | Path,
        *,
        is_trainable: bool = False,
        shadow_model: PreTrainedModel | None = None,
        shadow_loss_weight: float = 0.05,
        inference_mode: InferenceMode = "base_shadow",
    ) -> ShadowForCausalLM:
        peft = ShadowPeftModel.from_pretrained(
            model,
            pretrained_shadow_path,
            is_trainable=is_trainable,
            shadow_model=shadow_model,
        )
        wrapper = cls(
            peft,
            shadow_loss_weight=shadow_loss_weight,
            inference_mode=inference_mode,
        )
        # If a checkpoint includes task modules, load what we recognize.
        flat = _load_modules_to_save_state(pretrained_shadow_path)
        cfg = getattr(wrapper.peft_model, "shadow_config", None)
        requested = list(getattr(cfg, "modules_to_save", []) or [])
        if requested:
            candidates = {k: v for k, v in {
                "shadow_lm_head": getattr(wrapper, "shadow_lm_head", None),
                "lm_head": getattr(wrapper, "lm_head", None),
            }.items() if k in set(requested)}
        else:
            candidates = {
                "shadow_lm_head": getattr(wrapper, "shadow_lm_head", None),
                "lm_head": getattr(wrapper, "lm_head", None),
            }
        _load_modules_to_save_into(
            candidates,
            flat,
        )
        return wrapper

    def print_trainable_parameters(self) -> None:
        self.peft_model.print_trainable_parameters()

    def set_inference_mode(self, mode: InferenceMode) -> None:
        self.inference_mode = mode

    @property
    def device(self) -> torch.device:
        """
        Hugging Face generation utilities expect `.device` on the model.
        """
        base = getattr(self.peft_model, "base_model", None)
        if base is not None and hasattr(base, "device"):
            return base.device  # type: ignore[return-value]
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        base = getattr(self.peft_model, "base_model", None)
        if base is not None and hasattr(base, "dtype"):
            return base.dtype  # type: ignore[return-value]
        return next(self.parameters()).dtype

    # ---- transformers.Trainer compatibility ----
    def gradient_checkpointing_enable(
        self, gradient_checkpointing_kwargs: dict[str, Any] | None = None
    ) -> None:
        fn = getattr(self.peft_model.base_model, "gradient_checkpointing_enable", None)
        if callable(fn):
            if gradient_checkpointing_kwargs is None:
                fn()
            else:
                fn(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self) -> None:
        fn = getattr(self.peft_model.base_model, "gradient_checkpointing_disable", None)
        if callable(fn):
            fn()

    def forward(
        self,
        *args,
        labels: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> ShadowCausalLMOutputWithPast:
        # Force disable caching (Shadow requires full-seq processing).
        kwargs["use_cache"] = False
        kwargs["past_key_values"] = None

        # Normalize common HF kwargs so we never pass them twice (e.g. generation sets return_dict=True).
        if "labels" in kwargs:
            if labels is None:
                labels = kwargs.pop("labels")
            else:
                kwargs.pop("labels", None)
        if labels is not None:
            kwargs["labels"] = labels
        if "return_dict" not in kwargs:
            kwargs["return_dict"] = True

        if self.inference_mode == "shadow_only":
            # Compute shadow hidden using the shadow backbone, then score with shadow head.
            input_ids = kwargs.get("input_ids")
            attention_mask = kwargs.get("attention_mask")
            position_ids = kwargs.get("position_ids")
            inputs_embeds = kwargs.get("inputs_embeds")
            shadow_hidden = self.peft_model._compute_initial_shadow_hidden(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
            )
            shadow_logits = self.shadow_lm_head(shadow_hidden)
            loss = _shifted_ce_loss(shadow_logits, labels) if labels is not None else None
            return ShadowCausalLMOutputWithPast(
                loss=loss,
                logits=shadow_logits,
                shadow_logits=shadow_logits,
                past_key_values=None,
            )

        # base_shadow: run full model (base path) and also get final shadow hidden.
        outputs, shadow_hidden = self.peft_model.forward_with_shadow(*args, **kwargs)

        # HF outputs for CausalLM provide logits (+ optional loss).
        base_logits = getattr(outputs, "logits", None)
        if base_logits is None:
            raise TypeError("Base model output missing `logits`; expected a CausalLM model.")

        shadow_logits = self.shadow_lm_head(shadow_hidden)

        loss = getattr(outputs, "loss", None)
        if labels is not None:
            if loss is None:
                loss = _shifted_ce_loss(base_logits, labels)
            if self.shadow_loss_weight > 0:
                loss = loss + self.shadow_loss_weight * _shifted_ce_loss(shadow_logits, labels)

        return ShadowCausalLMOutputWithPast(
            loss=loss,
            logits=base_logits,
            shadow_logits=shadow_logits,
            past_key_values=None,
            hidden_states=getattr(outputs, "hidden_states", None),
            attentions=getattr(outputs, "attentions", None),
        )

    # GenerationMixin hooks (ensure full-sequence forward, no cache).
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        position_ids = kwargs.get("position_ids")
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "use_cache": False,
            "past_key_values": None,
        }

    def _update_model_kwargs_for_generation(self, outputs, model_kwargs, is_encoder_decoder=False):
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=is_encoder_decoder
        )
        model_kwargs["past_key_values"] = None
        model_kwargs["use_cache"] = False
        return model_kwargs


class ShadowForSequenceClassification(nn.Module):
    """
    Task wrapper for sequence classification.

    Expects `peft_model.base_model` to be an HF sequence classification model
    returning `SequenceClassifierOutput` with `logits`.
    """

    def __init__(
        self,
        peft_model: ShadowPeftModel,
        *,
        shadow_loss_weight: float = 0.05,
        inference_mode: InferenceMode = "base_shadow",
    ) -> None:
        super().__init__()
        self.peft_model = peft_model
        self.shadow_loss_weight = float(shadow_loss_weight)
        self.inference_mode: InferenceMode = inference_mode
        self.config = getattr(self.peft_model.base_model, "config", None)

        # Try to locate an existing classifier head on the base model.
        head = None
        for attr in ("score", "classifier"):
            cand = getattr(self.peft_model.base_model, attr, None)
            if isinstance(cand, nn.Module):
                head = cand
                break
        if head is None:
            raise AttributeError(
                "Base model does not expose a classifier head (`score` or `classifier`). "
                "Use an AutoModelForSequenceClassification-compatible model."
            )
        self.classifier_head = head
        self.shadow_classifier_head = deepcopy(head)

        # Default behavior for seqcls: heads are trainable, since classification often
        # benefits from adapting the final classifier layer.
        #
        # If `ShadowConfig.modules_to_save` is explicitly set (non-empty), we treat it as
        # an override: only listed modules are made trainable/saved.
        for p in self.classifier_head.parameters():
            p.requires_grad = False
        for p in self.shadow_classifier_head.parameters():
            p.requires_grad = False

        cfg = getattr(self.peft_model, "shadow_config", None)
        requested = list(getattr(cfg, "modules_to_save", []) or [])
        if requested:
            if "classifier_head" in requested:
                if not _is_trainable_module(self.classifier_head):
                    logger.info(
                        "ShadowConfig.modules_to_save requested classifier_head; enabling requires_grad=True for classifier_head."
                    )
                for p in self.classifier_head.parameters():
                    p.requires_grad = True
            if "shadow_classifier_head" in requested:
                if not _is_trainable_module(self.shadow_classifier_head):
                    logger.info(
                        "ShadowConfig.modules_to_save requested shadow_classifier_head; enabling requires_grad=True for shadow_classifier_head."
                    )
                for p in self.shadow_classifier_head.parameters():
                    p.requires_grad = True
        else:
            # Default: both heads trainable.
            logger.info(
                "Shadow seqcls: enabling requires_grad=True for classifier_head and shadow_classifier_head by default "
                "(set ShadowConfig.modules_to_save to override)."
            )
            for p in self.classifier_head.parameters():
                p.requires_grad = True
            for p in self.shadow_classifier_head.parameters():
                p.requires_grad = True

    def save_pretrained(self, save_directory: str | Path) -> None:
        self.peft_model.save_pretrained(save_directory)
        cfg = getattr(self.peft_model, "shadow_config", None)
        requested = list(getattr(cfg, "modules_to_save", []) or [])
        # Save only trainable task modules (optionally requested).
        _save_modules_to_save(
            save_directory,
            {
                "classifier_head": self.classifier_head,
                "shadow_classifier_head": self.shadow_classifier_head,
            },
            requested_modules=(requested or None),
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
        Push this ShadowPEFT checkpoint (adapter + optional task modules) to the Hub.

        This never uploads backbone weights.
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

    def state_dict(self, destination=None, prefix: str = "", keep_vars: bool = False):  # type: ignore[override]
        return self.peft_model.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], strict: bool = True, assign: bool = False):  # type: ignore[override]
        return self.peft_model.load_state_dict(state_dict, strict=strict, assign=assign)

    @classmethod
    def from_pretrained(
        cls,
        model: PreTrainedModel,
        pretrained_shadow_path: str | Path,
        *,
        is_trainable: bool = False,
        shadow_model: PreTrainedModel | None = None,
        shadow_loss_weight: float = 0.05,
        inference_mode: InferenceMode = "base_shadow",
    ) -> ShadowForSequenceClassification:
        peft = ShadowPeftModel.from_pretrained(
            model,
            pretrained_shadow_path,
            is_trainable=is_trainable,
            shadow_model=shadow_model,
        )
        wrapper = cls(peft, shadow_loss_weight=shadow_loss_weight, inference_mode=inference_mode)
        flat = _load_modules_to_save_state(pretrained_shadow_path)
        cfg = getattr(wrapper.peft_model, "shadow_config", None)
        requested = list(getattr(cfg, "modules_to_save", []) or [])
        if requested:
            candidates = {k: v for k, v in {
                "classifier_head": wrapper.classifier_head,
                "shadow_classifier_head": wrapper.shadow_classifier_head,
            }.items() if k in set(requested)}
        else:
            candidates = {
                "classifier_head": wrapper.classifier_head,
                "shadow_classifier_head": wrapper.shadow_classifier_head,
            }
        _load_modules_to_save_into(
            candidates,
            flat,
        )
        return wrapper

    def set_inference_mode(self, mode: InferenceMode) -> None:
        self.inference_mode = mode

    @property
    def device(self) -> torch.device:
        base = getattr(self.peft_model, "base_model", None)
        if base is not None and hasattr(base, "device"):
            return base.device  # type: ignore[return-value]
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        base = getattr(self.peft_model, "base_model", None)
        if base is not None and hasattr(base, "dtype"):
            return base.dtype  # type: ignore[return-value]
        return next(self.parameters()).dtype

    # ---- transformers.Trainer compatibility ----
    def gradient_checkpointing_enable(
        self, gradient_checkpointing_kwargs: dict[str, Any] | None = None
    ) -> None:
        fn = getattr(self.peft_model.base_model, "gradient_checkpointing_enable", None)
        if callable(fn):
            if gradient_checkpointing_kwargs is None:
                fn()
            else:
                fn(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self) -> None:
        fn = getattr(self.peft_model.base_model, "gradient_checkpointing_disable", None)
        if callable(fn):
            fn()

    def forward(self, *args, labels: torch.Tensor | None = None, **kwargs: Any) -> ShadowSequenceClassifierOutput:
        kwargs["use_cache"] = False
        kwargs["past_key_values"] = None
        kwargs["return_dict"] = True
        # Computing full hidden-states for seqcls is expensive and not needed for training/eval.
        # Only enable if the caller explicitly requested it.
        kwargs.setdefault("output_hidden_states", False)

        # Avoid passing labels twice if the caller provided it via kwargs.
        if "labels" in kwargs:
            if labels is None:
                labels = kwargs.pop("labels")
            else:
                kwargs.pop("labels", None)
        if labels is not None:
            kwargs["labels"] = labels

        if self.inference_mode == "shadow_only":
            input_ids = kwargs.get("input_ids")
            attention_mask = kwargs.get("attention_mask")
            position_ids = kwargs.get("position_ids")
            inputs_embeds = kwargs.get("inputs_embeds")
            shadow_hidden = self.peft_model._compute_initial_shadow_hidden(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
            )
            if attention_mask is None:
                pooled = shadow_hidden[:, -1, :]
            else:
                token_counts = attention_mask.long().sum(dim=1) - 1
                token_counts = token_counts.clamp(min=0)
                batch_idx = torch.arange(shadow_hidden.size(0), device=shadow_hidden.device)
                pooled = shadow_hidden[batch_idx, token_counts]
            shadow_logits = self.shadow_classifier_head(pooled)
            loss = None
            if labels is not None:
                loss = F.cross_entropy(shadow_logits, labels)
            return ShadowSequenceClassifierOutput(loss=loss, logits=shadow_logits, shadow_logits=shadow_logits)

        outputs, shadow_hidden = self.peft_model.forward_with_shadow(*args, **kwargs)
        base_logits = getattr(outputs, "logits", None)
        if base_logits is None:
            raise TypeError("Base model output missing `logits`; expected a sequence classification model.")

        attention_mask = kwargs.get("attention_mask")
        if attention_mask is None:
            pooled = shadow_hidden[:, -1, :]
        else:
            token_counts = attention_mask.long().sum(dim=1) - 1
            token_counts = token_counts.clamp(min=0)
            batch_idx = torch.arange(shadow_hidden.size(0), device=shadow_hidden.device)
            pooled = shadow_hidden[batch_idx, token_counts]
        shadow_logits = self.shadow_classifier_head(pooled)

        loss = getattr(outputs, "loss", None)
        if labels is not None:
            if loss is None:
                loss = F.cross_entropy(base_logits, labels)
            if self.shadow_loss_weight > 0:
                loss = loss + self.shadow_loss_weight * F.cross_entropy(shadow_logits, labels)

        return ShadowSequenceClassifierOutput(
            loss=loss,
            logits=base_logits,
            shadow_logits=shadow_logits,
            hidden_states=getattr(outputs, "hidden_states", None),
            attentions=getattr(outputs, "attentions", None),
        )


