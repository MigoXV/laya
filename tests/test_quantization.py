"""权重格式严格校验；CPU 单元测试不导入 Triton，也不运行量化推理。"""

import pytest
import torch
from safetensors.torch import load_file, save_file

from laya.quantization.config import QuantizationConfig, read_quantization
from laya.quantization.transforms import eligible, install_linears, quantize_weight, validate_weights


def test_quantization_format_and_unknown_version():
    assert read_quantization({"format_version": 1}) is None
    config = QuantizationConfig(quantized_modules=["0"])
    assert read_quantization({"quantization": config.model_dump()}) == config
    for change in ({"version": 2}, {"method": "other"}, {"compute_dtype": "bf16"},
                   {"quantized_modules": ["0", "0"]}, {"quantized_modules": []}):
        with pytest.raises(ValueError):
            read_quantization({"quantization": {**config.model_dump(), **change}})


def test_model_path_must_be_explicit(tmp_path, monkeypatch):
    from laya.configs.settings import Config

    monkeypatch.delenv("LAYA_MODEL_DIR", raising=False)
    with pytest.raises(ValueError, match="model_dir"):
        Config(_env_file=None)
    for name in ("model.safetensors", "config.json", "tokenizer.json"):
        (tmp_path / name).touch()
    monkeypatch.setenv("LAYA_MODEL_DIR", str(tmp_path))
    assert Config(_env_file=None).model_dir == tmp_path


@pytest.mark.parametrize("device,dtype", [("cpu", "fp32"), ("cuda:0", "bf16"), ("cuda:0", "fp32")])
def test_w8a8_rejects_unsupported_runtime_before_loading(tmp_path, device, dtype):
    import json
    from laya.configs.settings import Config
    from laya.runtime.resources import Runtime

    for name in ("model.safetensors", "tokenizer.json"):
        (tmp_path / name).touch()
    (tmp_path / "config.json").write_text(json.dumps({"format_version": 1,
        "quantization": QuantizationConfig(quantized_modules=["0"]).model_dump()}))
    with pytest.raises(ValueError, match="w8a8_requires_cuda_and_fp16"):
        Runtime(Config(model_dir=tmp_path, device=device, dtype=dtype, _env_file=None))


def test_int8_state_roundtrip_and_scale_precision(tmp_path):
    from laya.models.loading import move_model

    # 架构构造时决策头可能仍为 FP32；文件参数为 FP16，bias 也须保持 FP16。
    model = torch.nn.Sequential(torch.nn.Linear(32, 256), torch.nn.Linear(256, 1))
    state = {name: value.half() for name, value in model.state_dict().items()}
    qweight, scales = quantize_weight(state.pop("0.weight"))
    state.update({"0.qweight": qweight, "0.weight_scales": scales})
    metadata = QuantizationConfig(quantized_modules=[name for name, _ in eligible(model)])
    install_linears(model, metadata)
    path = tmp_path / "model.safetensors"
    save_file(state, str(path))
    restored = load_file(str(path))
    validate_weights(model, restored)
    model.load_state_dict(restored, strict=True)
    move_model(model, torch.device("cpu"), torch.float16)
    assert model[0].qweight.dtype == torch.int8
    assert model[0].weight_scales.dtype == torch.float32
    assert model[0].bias.dtype == torch.float16
    assert torch.equal(model[0].qweight, qweight)
    assert torch.equal(model[0].weight_scales, scales)
    with pytest.raises(ValueError, match="module_list_mismatch"):
        install_linears(torch.nn.Sequential(torch.nn.Linear(32, 256)),
                        QuantizationConfig(quantized_modules=["absent"]))


@pytest.mark.parametrize("damage", ["missing", "shape", "dtype", "zero_scale", "nan", "range"])
def test_invalid_quantized_tensors_are_rejected(damage):
    model = torch.nn.Sequential(torch.nn.Linear(32, 256)).half()
    install_linears(model, QuantizationConfig(quantized_modules=["0"]))
    state = {"0.qweight": torch.zeros(256, 32, dtype=torch.int8),
             "0.weight_scales": torch.ones(256), "0.bias": torch.zeros(256, dtype=torch.float16)}
    if damage == "missing":
        state.pop("0.weight_scales")
    elif damage == "shape":
        state["0.qweight"] = state["0.qweight"][:1]
    elif damage == "dtype":
        state["0.qweight"] = state["0.qweight"].half()
    elif damage == "zero_scale":
        state["0.weight_scales"][0] = 0
    elif damage == "nan":
        state["0.weight_scales"][0] = float("nan")
    else:
        state["0.qweight"][0, 0] = -128
    with pytest.raises(ValueError, match="w8a8_"):
        validate_weights(model, state)
