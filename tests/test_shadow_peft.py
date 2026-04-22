from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

from shadow_peft import (  # noqa: E402
    ShadowConfig,
    ShadowForCausalLM,  # noqa: E402
    ShadowPeftModel,
    get_shadow_model,
)


def _tiny_llama(vocab_size: int = 128, num_layers: int = 4) -> LlamaForCausalLM:
    cfg = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
    )
    return LlamaForCausalLM(cfg)


def _tiny_qwen(num_layers: int = 4):
    Qwen2Config = getattr(transformers, "Qwen2Config", None)
    Qwen2ForCausalLM = getattr(transformers, "Qwen2ForCausalLM", None)
    if Qwen2Config is None or Qwen2ForCausalLM is None:
        pytest.skip("Qwen2 is not available in this transformers version")
    cfg = Qwen2Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
    )
    return Qwen2ForCausalLM(cfg)


def _tiny_gemma(num_layers: int = 4):
    GemmaConfig = getattr(transformers, "GemmaConfig", None)
    GemmaForCausalLM = getattr(transformers, "GemmaForCausalLM", None)
    if GemmaConfig is None or GemmaForCausalLM is None:
        pytest.skip("Gemma is not available in this transformers version")
    cfg = GemmaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
    )
    return GemmaForCausalLM(cfg)


def test_get_shadow_model_wraps_and_freezes_base():
    torch.manual_seed(0)
    base = _tiny_llama(num_layers=4)
    cfg = ShadowConfig(
        num_shadow_layers=1,
        injection_hidden_size=8,
        gate_hidden_size=10,
        alpha=0.1,
        dropout=0.0,
    )
    m = get_shadow_model(base, cfg)

    # Base params frozen.
    assert all(not p.requires_grad for p in m.base_model.parameters())

    # Adapter params trainable.
    assert any(p.requires_grad for p in m.shadow_model.parameters())
    assert any(p.requires_grad for p in m.shadow_injection_model.parameters())
    assert any(p.requires_grad for p in m.shadow_update_model.parameters())


def test_save_load_roundtrip_matches_outputs(tmp_path: Path):
    torch.manual_seed(0)
    base1 = _tiny_llama(num_layers=4)
    base1.eval()
    base1_state = {k: v.detach().clone() for k, v in base1.state_dict().items()}

    cfg = ShadowConfig(
        num_shadow_layers=1,
        injection_hidden_size=8,
        gate_hidden_size=10,
        alpha=0.1,
        dropout=0.0,
    )
    m1 = get_shadow_model(base1, cfg)
    m1.eval()

    input_ids = torch.randint(0, base1.config.vocab_size, (2, 8))
    with torch.no_grad():
        out1 = m1(input_ids=input_ids).logits

    save_dir = tmp_path / "shadow_adapter"
    m1.save_pretrained(save_dir)

    # Create a new base with identical weights.
    base2 = _tiny_llama(num_layers=4)
    base2.load_state_dict(base1_state)
    base2.eval()

    m2 = ShadowPeftModel.from_pretrained(base2, save_dir, is_trainable=False)
    m2.eval()

    with torch.no_grad():
        out2 = m2(input_ids=input_ids).logits

    assert torch.allclose(out1, out2, atol=0.0, rtol=0.0)

    # Saved artifact contains only adapter weights.
    from safetensors.torch import load_file as safetensors_load_file

    state = safetensors_load_file(str(save_dir / "shadow_adapter.safetensors"))
    assert all(k.startswith(("shadow_model.", "shadow_injection_model.", "shadow_update_model.")) for k in state)


def test_explicit_shadow_model_supported():
    torch.manual_seed(0)
    base = _tiny_llama(num_layers=4)
    shadow = _tiny_llama(num_layers=1)
    cfg = ShadowConfig(
        num_shadow_layers=1,
        injection_hidden_size=8,
        gate_hidden_size=10,
        alpha=0.1,
        dropout=0.0,
    )
    m = get_shadow_model(base, cfg, shadow_model=shadow)
    m.eval()
    input_ids = torch.randint(0, base.config.vocab_size, (1, 6))
    with torch.no_grad():
        _ = m(input_ids=input_ids).logits


def test_qwen_supported():
    torch.manual_seed(0)
    base = _tiny_qwen(num_layers=4)
    cfg = ShadowConfig(
        num_shadow_layers=1,
        injection_hidden_size=8,
        gate_hidden_size=10,
        alpha=0.1,
        dropout=0.0,
    )
    m = get_shadow_model(base, cfg)
    m.eval()
    input_ids = torch.randint(0, base.config.vocab_size, (1, 6))
    with torch.no_grad():
        _ = m(input_ids=input_ids).logits


def test_gemma_supported():
    torch.manual_seed(0)
    base = _tiny_gemma(num_layers=4)
    cfg = ShadowConfig(
        num_shadow_layers=1,
        injection_hidden_size=8,
        gate_hidden_size=10,
        alpha=0.1,
        dropout=0.0,
    )
    m = get_shadow_model(base, cfg)
    m.eval()
    input_ids = torch.randint(0, base.config.vocab_size, (1, 6))
    with torch.no_grad():
        _ = m(input_ids=input_ids).logits


def test_shadow_for_causallm_base_shadow_and_shadow_only():
    torch.manual_seed(0)
    base = _tiny_llama(num_layers=4)
    cfg = ShadowConfig(
        num_shadow_layers=1,
        injection_hidden_size=8,
        gate_hidden_size=10,
        alpha=0.1,
        dropout=0.0,
    )
    peft = get_shadow_model(base, cfg)
    m = ShadowForCausalLM(peft, inference_mode="base_shadow")
    m.eval()

    input_ids = torch.randint(0, base.config.vocab_size, (2, 8))
    with torch.no_grad():
        out = m(input_ids=input_ids)
    assert out.logits is not None
    assert out.shadow_logits is not None
    assert out.logits.shape == out.shadow_logits.shape

    m.set_inference_mode("shadow_only")
    with torch.no_grad():
        out2 = m(input_ids=input_ids)
    assert torch.allclose(out2.logits, out2.shadow_logits, atol=0.0, rtol=0.0)


def test_implicit_shadow_model_uses_shadow_intermediate_size():
    torch.manual_seed(0)
    base = _tiny_llama(num_layers=4)
    cfg = ShadowConfig(
        num_shadow_layers=1,
        shadow_intermediate_size=12,
        injection_hidden_size=8,
        gate_hidden_size=10,
        alpha=0.1,
        dropout=0.0,
    )
    m = get_shadow_model(base, cfg)
    assert m.shadow_model.config.intermediate_size == 12
