from .config import ShadowConfig
from .peft_model import ShadowPeftModel, get_shadow_model, prepare_shadow_model
from .projected_causal_lm import (
    AutoModelForCausalLMWithHiddenProjection,
    AutoModelForCausalLMWithHiddenProjectionConfig,
)
from .task_models import ShadowForCausalLM, ShadowForSequenceClassification

__all__ = [
    "ShadowConfig",
    "AutoModelForCausalLMWithHiddenProjection",
    "AutoModelForCausalLMWithHiddenProjectionConfig",
    "ShadowForCausalLM",
    "ShadowForSequenceClassification",
    "ShadowPeftModel",
    "get_shadow_model",
    "prepare_shadow_model",
]

# Register with transformers' Auto classes so that checkpoints with
# model_type="causal_lm_with_hidden_projection" are loaded automatically
# via AutoConfig.from_pretrained / AutoModelForCausalLM.from_pretrained.
from transformers import AutoConfig, AutoModelForCausalLM

AutoConfig.register(
    AutoModelForCausalLMWithHiddenProjectionConfig.model_type,
    AutoModelForCausalLMWithHiddenProjectionConfig,
)
AutoModelForCausalLM.register(
    AutoModelForCausalLMWithHiddenProjectionConfig,
    AutoModelForCausalLMWithHiddenProjection,
)
