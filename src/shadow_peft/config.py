from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ShadowConfig:
    """
    Configuration for the Shadow adapter.

    Notes:
    - `num_shadow_layers` is used only for implicit shadow model creation.
    - `injection_hidden_size`, `gate_hidden_size`, `alpha`, and `dropout` control the adapter math.
    """

    num_shadow_layers: int = 1
    injection_hidden_size: int = 16
    gate_hidden_size: int = 10
    alpha: float = 0.1
    dropout: float = 0.1

    # Optional knobs for implicit shadow model sizing.
    shadow_intermediate_size: int | None = None
    shadow_num_attention_heads: int | None = None
    shadow_num_key_value_heads: int | None = None
    shadow_head_dim: int | None = None

    # Optional: task-specific modules to train/save alongside the Shadow adapter.
    # Mirrors PEFT's "modules_to_save" concept. Examples:
    # - sequence classification: ["classifier_head", "shadow_classifier_head"]
    # - causal LM: ["lm_head", "shadow_lm_head"] (large; disabled by default)
    modules_to_save: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ShadowConfig:
        # Backward/forward compatibility: ignore unknown keys.
        allowed = set(cls.__dataclass_fields__.keys())
        filtered = {k: v for k, v in data.items() if k in allowed}
        return cls(**filtered)

    def save_pretrained(self, save_directory: str | Path) -> None:
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / "shadow_config.json").write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_pretrained(cls, pretrained_path: str | Path) -> ShadowConfig:
        path = Path(pretrained_path) / "shadow_config.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing shadow config at: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(data)


def _default_shadow_config_from_kwargs(
    *,
    num_shadow_layers: int | None = None,
    injection_hidden_size: int | None = None,
    gate_hidden_size: int | None = None,
    alpha: float | None = None,
    dropout: float | None = None,
    shadow_intermediate_size: int | None = None,
    shadow_num_attention_heads: int | None = None,
    shadow_num_key_value_heads: int | None = None,
    shadow_head_dim: int | None = None,
) -> ShadowConfig:
    base = ShadowConfig()
    return ShadowConfig(
        num_shadow_layers=base.num_shadow_layers if num_shadow_layers is None else num_shadow_layers,
        injection_hidden_size=(
            base.injection_hidden_size if injection_hidden_size is None else injection_hidden_size
        ),
        gate_hidden_size=base.gate_hidden_size if gate_hidden_size is None else gate_hidden_size,
        alpha=base.alpha if alpha is None else alpha,
        dropout=base.dropout if dropout is None else dropout,
        shadow_intermediate_size=(
            base.shadow_intermediate_size
            if shadow_intermediate_size is None
            else shadow_intermediate_size
        ),
        shadow_num_attention_heads=(
            base.shadow_num_attention_heads
            if shadow_num_attention_heads is None
            else shadow_num_attention_heads
        ),
        shadow_num_key_value_heads=(
            base.shadow_num_key_value_heads
            if shadow_num_key_value_heads is None
            else shadow_num_key_value_heads
        ),
        shadow_head_dim=base.shadow_head_dim if shadow_head_dim is None else shadow_head_dim,
    )
