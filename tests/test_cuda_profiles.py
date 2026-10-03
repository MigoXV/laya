"""固定位置缓存保持原始频率、输出精度和前缀语义。"""

import pytest
import torch

from laya.config import Config
from laya.cuda_runner import CachedRotary


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_cached_rotary_preserves_original_prefix_and_dtype(dtype):
    from transformers.models.modernbert.modeling_modernbert import ModernBertRotaryEmbedding
    from transformers import ModernBertConfig

    original = ModernBertRotaryEmbedding(ModernBertConfig(
        hidden_size=64, num_attention_heads=1, max_position_embeddings=1024,
        global_rope_theta=160000, local_rope_theta=160000,
    ))
    positions = torch.arange(1024)[None]
    cached = CachedRotary(original, positions, dtype)
    for length in [22, 33, 569, 1024]:
        x = torch.empty((1, length, 3, 1, 64), dtype=dtype)
        expected = original(x, positions[:, :length])
        actual = cached(x, positions[:, :length])
        for a, b in zip(actual, expected):
            assert a.dtype == dtype
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert cached.original.inv_freq.dtype == torch.float32


def test_graph_backend_rejects_cpu_before_model_allocation(tmp_path):
    for name in ["config.json", "tokenizer.json", "model.safetensors"]:
        (tmp_path / name).touch()
    for runner in ["cuda-graph", "cuda-graph-compile"]:
        with pytest.raises(ValueError, match="需要 CUDA"):
            Config(model_dir=tmp_path, runner=runner, device="cpu", dtype="fp32", _env_file=None)


@pytest.mark.parametrize("changes", [
    {"runner": "eager", "graph_prewarm_profiles": [(1, 27, 2)]},
    {"graph_cache_size": 1, "graph_prewarm_profiles": [(1, 27, 2), (2, 27, 2)]},
    {"graph_prewarm_profiles": [(3, 27, 2)]},
    {"max_batch_tokens": 1024, "graph_prewarm_profiles": [(16, 512, 2)]},
])
def test_prewarm_rejects_invalid_or_unbounded_profiles(tmp_path, changes):
    for name in ["config.json", "tokenizer.json", "model.safetensors"]:
        (tmp_path / name).touch()
    options = {"model_dir": tmp_path, "runner": "cuda-graph", "max_batch_size": 16,
               "_env_file": None, **changes}
    with pytest.raises(ValueError):
        Config(**options)
