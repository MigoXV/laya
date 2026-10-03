"""用小型编码器验证两种精度下的动作头输入与有限输出。"""

from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("dtype_name", ["float16", "float32"])
def test_decision_head_precision(dtype_name):
    torch = pytest.importorskip("torch")
    from laya.reference import DecisionModel
    from laya.runtime import move_model

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=64)
            self.embedding = torch.nn.Embedding(16, 64)
            self.register_buffer("inv_freq", torch.tensor([0.123456789]), persistent=False)

        def forward(self, input_ids, attention_mask):
            return SimpleNamespace(last_hidden_state=self.embedding(input_ids))

    dtype = getattr(torch, dtype_name)
    model = DecisionModel(Encoder(), head_layers=0)
    original_frequency = model.encoder.inv_freq.clone()
    move_model(model, torch.device("cpu"), dtype).eval()
    assert model.encoder.inv_freq.dtype == torch.float32
    assert torch.equal(model.encoder.inv_freq, original_frequency)
    assert "encoder.inv_freq" not in model.state_dict()
    assert all(p.dtype == dtype for p in model.parameters())
    seen = []
    handle = model.act_head.register_forward_pre_hook(
        lambda module, args: seen.append(args[0].dtype)
    )
    try:
        with torch.inference_mode():
            logits, actions = model(
                torch.tensor([[1, 2, 3, 4]]), torch.ones((1, 4), dtype=torch.long),
                torch.tensor([[1, 2]]), torch.ones((1, 2), dtype=torch.bool),
                torch.tensor([0]),
            )
        assert seen == [dtype]
        assert actions.dtype == dtype
        assert torch.isfinite(logits).all() and torch.isfinite(actions).all()
    finally:
        handle.remove()


def test_cpu_fp16_requires_explicit_supported_precision(tmp_path):
    from pydantic import ValidationError
    from laya.config import Config

    for name in ("model.safetensors", "config.json", "tokenizer.json"):
        (tmp_path / name).touch()
    with pytest.raises(ValidationError, match="CPU 推理请显式指定 dtype=fp32"):
        Config(model_dir=tmp_path, device="cpu", dtype="fp16", _env_file=None)


def test_modernbert_rope_config_matches_saved_frequencies():
    torch = pytest.importorskip("torch")
    from transformers import AutoModel
    from laya.runtime import encoder_config_for_runtime

    settings = {
        "model_type": "modernbert", "hidden_size": 64, "num_attention_heads": 1,
        "num_hidden_layers": 2, "intermediate_size": 64, "vocab_size": 16,
        "pad_token_id": 0,
        "rope_parameters": {
            "full_attention": {"rope_type": "default", "rope_theta": 160000},
            "sliding_attention": {"rope_type": "default", "rope_theta": 160000},
        },
    }
    config = encoder_config_for_runtime(settings)
    config.reference_compile = False
    model = AutoModel.from_config(config, attn_implementation="sdpa")
    expected = 1 / (160000 ** (torch.arange(0, 64, 2).float() / 64))
    frequencies = [b for n, b in model.named_buffers() if "inv_freq" in n]
    assert frequencies
    for frequency in frequencies:
        torch.testing.assert_close(frequency, expected, rtol=0, atol=0)
    assert "local_rope_theta" not in settings
