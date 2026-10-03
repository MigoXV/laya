"""权重身份与按相对路径稳定计算的源码指纹。"""
import hashlib
from pathlib import Path

def fingerprint(root: Path):
    digest = hashlib.sha256()
    for name in (
        "config.json",
        "tokenizer.json",
        "model.safetensors",
    ):
        digest.update(name.encode())
        with (root / name).open("rb") as source:
            for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def source_fingerprint(paths):
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for name in sorted(paths):
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


def runtime_source_fingerprint(runner, quantized):
    paths = ["configs/settings.py", "models/decision.py", "models/loading.py",
             "inferencers/contracts.py", "inferencers/types.py", "inferencers/decision.py",
             "inferencers/preprocessing.py", "inferencers/postprocessing.py",
             "inferencers/batching.py", "runtime/resources.py", "runtime/fingerprints.py",
             "runners/eager.py"]
    if runner.startswith("cuda-graph") or quantized:
        paths += ["runners/prepared.py"]
    if runner.startswith("cuda-graph"):
        paths += ["runners/cuda_graph.py", "runners/inputs.py"]
    if quantized:
        paths += ["quantization/config.py", "quantization/modules.py",
                  "quantization/transforms.py", "quantization/int8_kernels.py"]
    return source_fingerprint(paths)
