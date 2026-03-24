"""
Utilities for building the lightweight Shadow adapter that augments a frozen
base causal language model with a small number of trainable decoder layers.
"""
from __future__ import annotations

from typing import Optional, Tuple
from dataclasses import dataclass
from copy import deepcopy

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from transformers import PreTrainedModel
from transformers.generation import GenerationMixin, GenerationConfig
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
    SequenceClassifierOutput,
    BaseModelOutputWithPast,
)
import torch.nn.functional as F
from transformers.models.qwen3.modeling_qwen3 import (
    create_causal_mask,
    create_sliding_window_causal_mask,
    Qwen3DecoderLayer,
)
from transformers.cache_utils import DynamicCache, Cache


def _get_transformer_backbone(base_model: PreTrainedModel) -> nn.Module:
    """Return the transformer block that hosts the decoder layers."""
    for attr in ("model", "transformer", "base_model", "deberta", "bert", "roberta"):
        backbone = getattr(base_model, attr, None)
        if backbone is not None:
            return backbone
    raise AttributeError(
        "Unable to locate transformer backbone inside the supplied base model."
    )


@dataclass
class ShadowOutput:
    last_hidden_state: torch.Tensor
    last_shadow_hidden_state: torch.Tensor
    base_outputs: BaseModelOutputWithPast


@dataclass
class ShadowCausalLMOutputWithPast(CausalLMOutputWithPast):
    shadow_logits: Optional[torch.Tensor] = None


@dataclass
class ShadowSequenceClassifierOutput(SequenceClassifierOutput):
    shadow_logits: Optional[torch.Tensor] = None


class ShadowModel(nn.Module):
    """Small stack of decoder layers cloned from the base model."""

    def __init__(
        self,
        base_model: PreTrainedModel,
        num_shadow_layers: int = 1,
        num_alpha_heads: Optional[int] = 8,
        intermediate_size: Optional[int] = None,
        intermediate_size_ratio: float = 0.1,
        shadow_beta: float = 1.0,
        shadow_dropout: float = 0.1,
        shadow_gate_heads: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.backbone = _get_transformer_backbone(base_model)

        self.base_model = base_model
        self.config = base_model.config
        self.num_shadow_layers = num_shadow_layers
        self.rotary_emb = self.backbone.rotary_emb
        self.embed_tokens = self.backbone.embed_tokens
        self.norm = self.backbone.norm
        self.layer_types = getattr(
            self.config,
            "layer_types",
            ["full_attention"] * len(self.backbone.layers),
        )
        self.has_sliding_layers = any(
            layer_type == "sliding_attention" for layer_type in self.layer_types
        )

        original_intermediate_size = self.config.intermediate_size
        if intermediate_size is None:
            shadow_intermediate_size = max(
                1, int(original_intermediate_size * intermediate_size_ratio)
            )
        else:
            shadow_intermediate_size = intermediate_size

        self.config.intermediate_size = shadow_intermediate_size
        self.shadow_layers = nn.ModuleList([Qwen3DecoderLayer(config=self.config, layer_idx=i) for i in range(num_shadow_layers)])
        # Multiple alpha "heads"/experts per layer. If not specified, default to
        # the model's attention head count when available.
        if num_alpha_heads is None:
            num_alpha_heads = int(getattr(self.config, "num_attention_heads", 1))
        if num_alpha_heads < 1:
            raise ValueError(f"num_alpha_heads must be >= 1, got {num_alpha_heads}")
        self.num_alpha_heads = num_alpha_heads

        # Shape: [num_layers, hidden_size]
        actual_num_layers = self.config.num_hidden_layers - 1
        self.alpha_downs = nn.Parameter(
            torch.randn((actual_num_layers, self.config.hidden_size, self.num_alpha_heads))
        )
        # nn.init.normal_(self.alpha_downs, mean=0.0, std=0.02)  # best
        nn.init.normal_(self.alpha_downs, mean=0.0, std=0.02)

        self.alpha_ups = nn.Parameter(
            torch.zeros((actual_num_layers, self.num_alpha_heads, self.config.hidden_size))
        )
        '''
        self.alphas = nn.Parameter(
            torch.zeros((actual_num_layers, self.num_alpha_heads, self.config.hidden_size))
        )
        self.alpha_routers = nn.ModuleList([
            nn.Linear(self.config.hidden_size, self.num_alpha_heads)
            for _ in range(actual_num_layers)
        ])
        '''

        self.hidden_norm = nn.LayerNorm(self.config.hidden_size)
        '''
        self.update_gate = nn.Sequential(
            nn.Linear(self.config.hidden_size, shadow_intermediate_size, bias=False),
            nn.SiLU(),
            nn.Linear(shadow_intermediate_size, self.config.hidden_size, bias=False),
            nn.Sigmoid(),
        )
        self.update_transform = nn.Sequential(
            nn.Linear(self.config.hidden_size, shadow_intermediate_size, bias=False),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(shadow_intermediate_size, self.config.hidden_size, bias=False),
        )
        '''
        if shadow_gate_heads is None:
            head_size = shadow_intermediate_size // actual_num_layers
        else:
            head_size = shadow_gate_heads
        assert head_size > 1, f"head_size must be > 1, got {head_size}"
        print(f">>> head_size: {head_size}")

        '''
        self.gemmas_downs = nn.Parameter(
            torch.randn((actual_num_layers, self.config.hidden_size, head_size))
        )
        nn.init.normal_(self.gemmas_downs, mean=0.0, std=0.02)

        self.gemmas_ups = nn.Parameter(
            torch.zeros((actual_num_layers, head_size, self.config.hidden_size))
        )
        '''
        self.update_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.config.hidden_size, head_size, bias=False),
                # nn.SiLU(),
                nn.Dropout(shadow_dropout),
                nn.Linear(head_size, self.config.hidden_size, bias=False),
                nn.Sigmoid(),
            )
            for _ in range(actual_num_layers)
        ])

        self.update_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.config.hidden_size, head_size, bias=False),
                nn.SiLU(),
                nn.Dropout(shadow_dropout),
                nn.Linear(head_size, self.config.hidden_size, bias=False),
            )
            for _ in range(actual_num_layers)
        ])

        self.shadow_dropout = nn.Dropout(shadow_dropout)
        self.shadow_beta = shadow_beta
        self.freeze_base_model()

    def freeze_base_model(self) -> None:
        for param in self.base_model.parameters():
            param.requires_grad = False

    def _build_position_ids(self, attention_mask: Optional[torch.Tensor], seq_len: int, device) -> torch.Tensor:
        if attention_mask is None:
            return torch.arange(seq_len, device=device).unsqueeze(0)
        position_ids = attention_mask.long().cumsum(dim=1) - 1
        position_ids.clamp_(min=0)
        return position_ids.masked_fill(attention_mask == 0, 0)

    def _build_causal_masks(
        self,
        mask_source: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: torch.Tensor,
        past_key_values: Optional[tuple] = None,
        cache_position: Optional[torch.Tensor] = None,
    ):
        if cache_position is None:
            cache_position = torch.arange(
                mask_source.shape[1], device=mask_source.device
            )

        mask_kwargs = {
            "config": self.config,
            "input_embeds": mask_source,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }
        full_attn_mask = create_causal_mask(**mask_kwargs)
        sliding_mask = (
            create_sliding_window_causal_mask(**mask_kwargs)
            if self.has_sliding_layers
            else None
        )
        return full_attn_mask, sliding_mask

    def _forward(self,
        layers: nn.ModuleList,
        shadow_hidden_states: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_values: Optional[tuple] = None,
        cache_position: Optional[torch.Tensor] = None,
        causal_mask_mapping: Optional[dict] = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.backbone.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            full_attn_mask, sliding_mask = self._build_causal_masks(
                mask_source=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
            )
            causal_mask_mapping = {
                "full_attention": full_attn_mask,
            }
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = sliding_mask

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for idx, decoder_layer in enumerate(layers):
            do_shadow = shadow_hidden_states is not None and idx > 0
            shadow_idx = idx - 1
            if do_shadow:  # best
                # w = torch.softmax(self.alpha_routers[shadow_idx](self.hidden_norm(hidden_states)), dim=-1)
                # alpha_eff = torch.einsum("btk,kd->btd", w, self.alphas[shadow_idx])
                # delta = alpha_eff * shadow_hidden_states

                # delta = hidden_states - shadow_hidden_states
                # delta_t = self.alpha_routers[shadow_idx](delta)
                # delta_t = torch.einsum("btk,kd->btd", delta_t, self.alphas[shadow_idx])

                # delta = hidden_states - shadow_hidden_states
                delta = shadow_hidden_states - hidden_states
                delta_t = torch.einsum("btd,dk->btk", delta, self.alpha_downs[shadow_idx])
                # delta_t = self.shadow_activation(delta_t)
                delta_t = self.shadow_dropout(delta_t)
                delta_t = torch.einsum("btk,kd->btd", delta_t, self.alpha_ups[shadow_idx])
                hidden_states = hidden_states + self.shadow_beta * delta_t
    
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

            if do_shadow:
                # update gate
                # g = self.update_gate(h_in)
                # ht = self.update_transform(h_in)
                h_in = self.hidden_norm(hidden_states)
                ht = self.update_transforms[shadow_idx](h_in)
                g = self.update_gates[shadow_idx](h_in)
                shadow_hidden_states = shadow_hidden_states + g * (ht - shadow_hidden_states)
                '''
                delta = hidden_states - shadow_hidden_states
                # delta_t = self.gemmas_routers[shadow_idx](delta)
                # delta_t = torch.einsum("btk,kd->btd", delta_t, self.gemmas[shadow_idx])
                delta_t = torch.einsum("btd,dk->btk", delta, self.gemmas_downs[shadow_idx])
                delta_t = torch.einsum("btk,kd->btd", delta_t, self.gemmas_ups[shadow_idx])
                shadow_hidden_states = shadow_hidden_states + self.shadow_beta * delta_t
                '''

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        shadow_kwargs = deepcopy(kwargs)
        shadow_kwargs["use_cache"] = False
        shadow_kwargs["past_key_values"] = None
        shadow_output = self._forward(
            layers=self.shadow_layers,
            shadow_hidden_states=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **shadow_kwargs,
        ).last_hidden_state

        base_outputs = self._forward(
            layers=self.backbone.layers[: self.config.num_hidden_layers],
            shadow_hidden_states=shadow_output,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs
        )

        return ShadowOutput(
            last_hidden_state=base_outputs.last_hidden_state,
            last_shadow_hidden_state=shadow_output,
            base_outputs=base_outputs,
        )

    def to(self, *args, **kwargs):
        self.base_model.to(*args, **kwargs)
        return super().to(*args, **kwargs)


class ShadowPreTrainedModel(PreTrainedModel):
    """Base class for all shadow models."""

    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_gradient_checkpointing = True

    _is_stateful = False
    _supports_cache_class = False
    _supports_static_cache = False
    _supports_quantized_cache = False

    def __init__(self, shadow_model: ShadowModel, *args, **kwargs) -> None:
        config = deepcopy(shadow_model.config)
        super().__init__(config, *args, **kwargs)
        self.shadow_model = shadow_model

    def _set_gradient_checkpointing(self, module, value=False):
        """Enable/disable gradient checkpointing for shadow layers."""
        if isinstance(module, Qwen3DecoderLayer):
            module.gradient_checkpointing = value
            if value:
                module._gradient_checkpointing_func = checkpoint
            else:
                if hasattr(module, '_gradient_checkpointing_func'):
                    delattr(module, '_gradient_checkpointing_func')

    def print_trainable_parameters(self) -> None:
        trainable_params = sum(p.numel() for p in self.shadow_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.shadow_model.parameters())
        percent = 100 * trainable_params / total_params if total_params else 0.0
        print(
            f"Trainable params: {trainable_params:,} || "
            f"Total params: {total_params:,} || Trainable%: {percent:.2f}%"
        )

    def forward(self, *args, **kwargs) -> Tuple[torch.Tensor, BaseModelOutputWithPast]:
        return self.shadow_model(*args, **kwargs)

    def to(self, *args, **kwargs):  # type: ignore[override]
        return super().to(*args, **kwargs)


class ShadowForCausalLM(ShadowPreTrainedModel, GenerationMixin):
    """Combine the frozen base model with the lightweight shadow adapter."""
    
    _supports_gradient_checkpointing = True

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config = self.shadow_model.config
        # Weight for auxiliary shadow-head task loss (0 disables it).
        self.shadow_loss_weight: float = float(getattr(self.config, "shadow_loss_weight", 0.05))
        
        # Disable use_cache in config for shadow model
        if hasattr(self.config, 'use_cache'):
            self.config.use_cache = False

        self.lm_head = getattr(self.shadow_model.base_model, "lm_head", None)
        self.shadow_lm_head = deepcopy(self.lm_head)

        self.main_input_name = "input_ids"
        
        # Initialize generation_config - required by GenerationMixin
        if hasattr(self.shadow_model.base_model, 'generation_config'):
            self.generation_config = self.shadow_model.base_model.generation_config
        else:
            self.generation_config = GenerationConfig.from_model_config(self.config)
        
        # Ensure generation also doesn't use cache
        if hasattr(self.generation_config, 'use_cache'):
            self.generation_config.use_cache = False

        self.set_trainable_lm_head(self.lm_head, False)
        self.set_trainable_lm_head(self.shadow_lm_head, True)

    def set_trainable_lm_head(self, lm_head: nn.Module, trainable: bool = False) -> None:
        for param in lm_head.parameters():
            param.requires_grad = trainable

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing for the model."""
        if not self._supports_gradient_checkpointing:
            raise ValueError(f"{self.__class__.__name__} does not support gradient checkpointing.")
        
        if gradient_checkpointing_kwargs is None:
            gradient_checkpointing_kwargs = {}
        
        # Enable gradient checkpointing on shadow model's layers
        self.shadow_model.apply(lambda module: self._set_gradient_checkpointing(module, True))

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing for the model."""
        if self._supports_gradient_checkpointing:
            self.shadow_model.apply(lambda module: self._set_gradient_checkpointing(module, False))

    def forward(
        self,
        *args,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = None,
        **kwargs,
    ) -> ShadowCausalLMOutputWithPast:
        # CRITICAL FIX: Disable KV cache entirely for shadow model
        # The shadow architecture requires processing full sequences through shadow layers,
        # which is incompatible with incremental token generation using KV cache.
        # TODO: Implement proper caching for shadow layers to enable fast generation.
        kwargs["past_key_values"] = None
        kwargs["use_cache"] = False

        shadow_output = super().forward(*args, **kwargs)
        logits = self.lm_head(shadow_output.last_hidden_state)
        shadow_logits = self.shadow_lm_head(shadow_output.last_shadow_hidden_state)

        loss = None
        if labels is not None:
            loss = self.shadow_model.base_model.loss_function(
                logits=logits, labels=labels, vocab_size=self.shadow_model.config.vocab_size, **kwargs)
            weight = float(getattr(self, "shadow_loss_weight", getattr(self.config, "shadow_loss_weight", 0.05)))
            if weight > 0:
                loss2 = self.shadow_model.base_model.loss_function(
                    logits=shadow_logits,
                    labels=labels,
                    vocab_size=self.shadow_model.config.vocab_size,
                    **kwargs,
                )
                loss = loss + weight * loss2

        return ShadowCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            shadow_logits=shadow_logits,
            past_key_values=None,
            hidden_states=shadow_output.base_outputs.hidden_states,
            attentions=shadow_output.base_outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        """
        Prepare inputs for generation, following the standard transformers pattern.
        This method is called by GenerationMixin.generate() to prepare inputs for each forward pass.
        
        CRITICAL: Shadow model doesn't use KV cache, so we must ALWAYS pass the full sequence.
        When generate() is called with use_cache=True, it will slice input_ids to only the new token,
        but we need the full context, so we ignore that slicing.
        """

        # Create position_ids from attention_mask if not provided
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)

        model_inputs = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "use_cache": False,  # Always disabled for shadow
            "past_key_values": None,  # Always None for shadow
        }
        
        return model_inputs

    def _update_model_kwargs_for_generation(self, outputs, model_kwargs, is_encoder_decoder=False):
        """
        Override to prevent input slicing for non-cached generation.
        
        Shadow model doesn't use KV cache, so we must always keep the full input_ids
        instead of slicing to just the last token.
        """
        # Call parent to update other kwargs
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder
        )
        
        # Force cache to None (in case parent tried to set it)
        model_kwargs["past_key_values"] = None
        model_kwargs["use_cache"] = False
        
        return model_kwargs


class ShadowForSequenceClassificationModel(ShadowPreTrainedModel):
    """Shadow adapter head for sequence classification tasks."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config = self.shadow_model.config
        # Weight for auxiliary shadow-head task loss (0 disables it).
        self.shadow_loss_weight: float = float(getattr(self.config, "shadow_loss_weight", 0.05))
        self.num_labels = self.config.num_labels
        self.problem_type = getattr(self.config, "problem_type", None)

        self.score = nn.Linear(
            self.config.hidden_size,
            self.num_labels if self.num_labels > 1 else 1,
        )
        self.score2 = nn.Linear(
            self.config.hidden_size,
            self.num_labels if self.num_labels > 1 else 1,
        )

    def _last_token_pool(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if attention_mask is None:
            return hidden_states[:, -1, :]
        token_counts = attention_mask.long().sum(dim=1) - 1
        token_counts = token_counts.clamp(min=0)
        batch_indices = torch.arange(
            hidden_states.size(0), device=hidden_states.device
        )
        return hidden_states[batch_indices, token_counts]

    def compute_loss(self,
        logits: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        pooled_logits = self._last_token_pool(logits, attention_mask)

        loss = None
        if labels is not None:
            loss = self.shadow_model.base_model.loss_function(
                logits=logits,
                labels=labels,
                pooled_logits=pooled_logits,
                config=self.shadow_model.config
            )

        return loss, pooled_logits

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> ShadowSequenceClassifierOutput:
        shadow_output = self.shadow_model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs,
        )

        loss = None
        if labels is not None:
            logits = self.score(shadow_output.last_hidden_state)
            loss, pooled_logits = self.compute_loss(logits, attention_mask, labels)
            shadow_logits = self.score2(shadow_output.last_shadow_hidden_state)
            weight = float(getattr(self, "shadow_loss_weight", getattr(self.config, "shadow_loss_weight", 0.05)))
            if weight > 0:
                loss2, pooled_shadow_logits = self.compute_loss(shadow_logits, attention_mask, labels)
                loss = loss + weight * loss2
            else:
                pooled_shadow_logits = self._last_token_pool(shadow_logits, attention_mask)
        else:
            logits = self.score(shadow_output.last_hidden_state)
            pooled_logits = self._last_token_pool(logits, attention_mask)
            shadow_logits = self.score2(shadow_output.last_shadow_hidden_state)
            pooled_shadow_logits = self._last_token_pool(shadow_logits, attention_mask)

        return ShadowSequenceClassifierOutput(
            loss=loss,
            logits=pooled_logits,
            shadow_logits=pooled_shadow_logits,
        )
