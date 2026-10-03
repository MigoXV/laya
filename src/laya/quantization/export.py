"""W8A8 权重导出。"""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import torch
from safetensors.torch import load_file, save_file
from .config import QuantizationConfig, read_quantization
from .transforms import explicit_heads, eligible, quantize_weight, install_linears, validate_weights
from laya.runtime.fingerprints import source_fingerprint

def file_record(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


@torch.inference_mode()
def export_w8a8(source: Path, destination: Path, device: str):
    from transformers import AutoModel
    from laya.models.decision import DecisionModel
    from laya.models.loading import encoder_config_for_runtime, move_model

    source, destination = source.resolve(), destination.absolute()
    if destination.exists():
        raise ValueError(f"output_directory_exists: {destination}")
    raw = json.loads((source / "config.json").read_text())
    if raw.get("format_version") != 1 or "quantization" in raw:
        raise ValueError("export_requires_unquantized_laya_v1")
    target = torch.device(device)
    if target.type != "cuda" or target.index is None or not torch.cuda.is_available():
        raise ValueError("export_requires_explicit_cuda_device")
    torch.cuda.set_device(target)
    torch.set_num_threads(4)
    cfg = encoder_config_for_runtime(raw["encoder"])
    cfg.reference_compile = False
    encoder = AutoModel.from_config(cfg, dtype=torch.float16, attn_implementation="sdpa", trust_remote_code=False)
    head = raw["decision_head"]
    model = DecisionModel(encoder, head["layers"], head["num_actions"])
    state = load_file(str(source / "model.safetensors"))
    if any(t.dtype != (torch.float32 if name == "temperature" else torch.float16)
           for name, t in state.items()):
        raise ValueError("export_requires_fp16_source_weights")
    model.load_state_dict(state, strict=True)
    del state
    move_model(model, torch.device("cpu"), torch.float16).eval()
    explicit_heads(model)
    candidates = eligible(model)
    tensors = dict(model.state_dict())
    for name, module in candidates:
        qweight, scales = quantize_weight(module.weight.to(target))
        tensors.pop(f"{name}.weight")
        tensors[f"{name}.qweight"] = qweight.cpu()
        tensors[f"{name}.weight_scales"] = scales.cpu()
    raw["quantization"] = QuantizationConfig(quantized_modules=[name for name, _ in candidates]).model_dump()
    install_linears(model, read_quantization(raw))
    validate_weights(model, tensors)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        save_file({name: value.contiguous() for name, value in tensors.items()}, str(temporary / "model.safetensors"))
        (temporary / "config.json").write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n")
        for name in ("tokenizer.json", "LICENSE"):
            shutil.copyfile(source / name, temporary / name)
        original = json.loads((source / "manifest.json").read_text())
        manifest = {"format_version": 1, "name": destination.name, "source": original["source"],
                    "license": original["license"], "encoder": original["encoder"],
                    "parent_files": {name: file_record(source / name) for name in
                                     ("config.json", "model.safetensors", "tokenizer.json")},
                    "export_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                    "export_code_sha256": source_fingerprint([
                        "quantization/export.py", "quantization/config.py", "quantization/modules.py",
                        "quantization/transforms.py", "models/decision.py", "models/loading.py",
                        "runtime/fingerprints.py",
                    ]),
                    "quantized_modules": len(candidates),
                    "quantized_weight_count": sum(m.weight.numel() for _, m in candidates),
                    "files": {name: file_record(temporary / name) for name in
                              ("config.json", "model.safetensors", "tokenizer.json")},
                    "transformations": ["动态 W8A8；权重逐输出通道 INT8，激活逐 token 动态 INT8，浮点部分 FP16。",
                                        "决策头显式 QKV；97 个大矩阵量化，embedding 与动作头保留 FP16。"]}
        (temporary / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        (temporary / "README.md").write_text(
            f"# {destination.name}\n\n"
            "当前 Laya 多语言模型的动态 W8A8 仓库，供 Laya 项目加载；模型实现位于项目中。\n\n"
            "## 格式\n\n"
            "量化信息统一写入 config.json 的 quantization。laya_w8a8 v1 保存 INT8 qweight（out,in）"
            "与 FP32 weight_scales；其余参数为 FP16。尺度为每行 absmax.clamp_min(1e-8)/127，"
            "round ties-to-even 后限制至 [-127,127]。激活逐 token 动态量化，不需要固定激活尺度。"
            "决策头采用显式 QKV；归一化融合由项目在加载之后建立。\n\n"
            f"量化 {manifest['quantized_modules']} 个矩阵、{manifest['quantized_weight_count']:,} 个权重。"
            "embedding、SDPA、动作头及小末端层仍使用浮点计算。\n\n"
            "## 加载\n\n```bash\n"
            f"LAYA_REQUEST_TIMEOUT=300 poetry run laya serve --model-dir {destination} --runner cuda-graph-compile --max-batch-size 16\n"
            "```\n\n"
            "量化格式自动识别；仅支持 CUDA SM80+ 与 FP16 浮点计算，现有 A100 已验证。"
            "没有默认模型路径，必须显式选择模型仓库。\n\n"
            "## 精度与来源\n\n"
            "导出源为原 FP16 多语言模型，权重哈希及上游 revision 见 manifest.json。"
            "未经 QAT；冻结小测试集 659 个分类问题的实验 FP16/W8A8 准确率为 79.97%/79.82%，"
            "23 个答案发生变化。这是小测试集结果，不是业务精度保证。"
            "保存后模型的对齐结果见项目 tests/w8a8 的验证报告。\n\n"
            "许可沿用原仓库 Apache-2.0，见 LICENSE。\n")
        # 仅发布完整仓库；不覆盖任何已有目录。
        if destination.exists():
            raise ValueError(f"output_directory_exists: {destination}")
        (temporary / "model.safetensors").chmod(0o644)
        temporary.chmod(0o755)
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return manifest
