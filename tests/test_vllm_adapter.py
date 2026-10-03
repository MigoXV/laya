"""完整头、序列隔离、原始 logits 传输与 backend 配置契约。"""

from types import SimpleNamespace

import pytest
import torch

from laya.config import Config


def test_native_config_preserves_rope_and_inclusive_window():
    from laya.vllm_runner import native_config

    source = {"encoder": {
        "local_attention": 128, "layer_types": ["full_attention", "sliding_attention"],
        "rope_parameters": {
            "full_attention": {"rope_type": "default", "rope_theta": 160000},
            "sliding_attention": {"rope_type": "default", "rope_theta": 160000},
        }}, "decision_head": {"layers": 2, "num_actions": 2}}
    class Tokenizer:
        mask_token_id = 4

        def __call__(self, text, **kwargs):
            return {"input_ids": [len(text)]}

    tokenizer = Tokenizer()
    translated = native_config(source, tokenizer)
    assert translated["global_rope_theta"] == translated["local_rope_theta"] == 160000
    window = translated["local_attention"] // 2
    assert 64 < window and not 65 < window  # HF 的 distance <= 64
    assert source["encoder"]["local_attention"] == 128
    assert "rope_parameters" in source["encoder"] and "layer_types" in source["encoder"]
    assert "rope_parameters" not in translated and "layer_types" not in translated


def test_backend_rejects_cpu_before_loading(tmp_path):
    for name in ("model.safetensors", "config.json", "tokenizer.json"):
        (tmp_path / name).touch()
    with pytest.raises(ValueError, match="vLLM 使用 cuda:0"):
        Config(model_dir=tmp_path, device="cpu", dtype="fp32", runner="vllm", _env_file=None)


@pytest.mark.parametrize("radius", [None, 1])
def test_sdpa_has_bidirectional_inclusive_window(radius):
    pytest.importorskip("vllm")
    from laya.vllm_model import CanonicalSDPA

    length, heads, dim = 5, 2, 8
    query = torch.zeros((length, heads * dim))
    values = torch.nn.functional.pad(torch.eye(length), (0, dim - length)).repeat(1, heads)
    actual = CanonicalSDPA(heads, dim, radius)(query, query, values)
    expected = torch.zeros_like(actual)
    for i in range(length):
        allowed = [j for j in range(length) if radius is None or abs(i - j) <= radius]
        for head in range(heads):
            for j in allowed:
                expected[i, head * dim + j] = 1 / len(allowed)
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-7)


def test_pooler_preserves_complete_heads_and_sequence_isolation():
    pytest.importorskip("vllm")
    from laya.vllm_model import DecisionPooler

    pooler = DecisionPooler(SimpleNamespace(
        hidden_size=64, laya_head_layers=1, laya_num_actions=2,
        laya_mask_id=4, laya_type_prefixes=[[11, 9], [12, 9], [13, 9]],
    )).eval()
    sequences = [torch.tensor([1, 11, 9, 4, 8, 4, 7]),
                 torch.tensor([1, 13, 9, 4, 4])]
    hidden = torch.randn(12, 64)
    metadata = SimpleNamespace(
        get_pooling_cursor=lambda: SimpleNamespace(is_partial_prefill=lambda: False),
        get_prompt_token_ids=lambda: sequences,
    )
    with torch.inference_mode():
        results = pooler(hidden, metadata)
        for i, (start, length, qtype, markers) in enumerate(
            [(0, 7, 0, [3, 5]), (7, 5, 2, [3, 4])]
        ):
            logits, acts = pooler.model.score_hidden(
                hidden[start:start + length][None], torch.ones((1, length)),
                torch.tensor([markers]), torch.ones((1, 2), dtype=torch.bool),
                torch.tensor([qtype]),
            )
            torch.testing.assert_close(results[i], torch.cat((logits[0], acts[0])), rtol=0, atol=0)
    assert pooler.get_pooling_updates("embed").requires_token_ids
    assert pooler.question_layout(torch.zeros(5, dtype=torch.long)) == (0, [0, 4])
    with pytest.raises(ValueError, match="unknown_question_type"):
        pooler.question_layout(torch.tensor([1, 99, 9, 4, 4]))
    with pytest.raises(ValueError, match="missing_option_markers"):
        pooler.question_layout(torch.tensor([1, 11, 9, 4, 8]))
