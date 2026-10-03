"""独立矩阵与完整模型 GPU 基准；输入、量化、输出头都在 Graph 内。"""

import gc
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from datetime import datetime, timezone

import numpy as np
import torch
import triton
import typer

from laya.configs.settings import Config
from laya.api.contracts import DecisionRequest
from laya.runners.inputs import INPUT_KEYS
from laya.inferencers.batching import collate_items
from laya.inferencers.decision import DecisionInferencer

from .kernels import EVIDENCE, TILES, TUNING, int_mm, quantize, quantize_weight, triton_gemm
from .model import calibrate, eligible, explicit_heads, replace_linears, restore_linears


ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT
app = typer.Typer()
MATRIX_MODES = ("cublas-dynamic", "cublas-static", "triton-dynamic", "triton-static-fused")
MODES = (*MATRIX_MODES, "triton-dynamic-ln", "triton-static-ln", "triton-dynamic-ln-native")


def command(args):
    return subprocess.check_output(args, text=True).strip()


def census():
    return command(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv"])


def assert_isolated(uuid):
    def owned(pid):
        # Inductor 的编译/autotune 子进程也可能持有 CUDA context。
        # 只允许本次实验自己的进程树；其他项目和服务仍必须停止。
        while pid > 1:
            if pid == os.getpid():
                return True
            try:
                status = Path(f"/proc/{pid}/status").read_text()
                pid = int(next(line.split()[1] for line in status.splitlines() if line.startswith("PPid:")))
            except (FileNotFoundError, StopIteration):
                return False
        return False

    for line in census().splitlines()[1:]:
        pid, device, _ = [value.strip() for value in line.split(",")]
        if device == uuid and not owned(int(pid)):
            raise RuntimeError(f"GPU 未独占，发现 PID {pid}；请先停止该卡上的服务。")


@torch.inference_mode()
def capture(fn, repeat=1):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(repeat):
            outputs = fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph.replay()
    torch.cuda.synchronize()
    return graph, outputs


def measure(graph, samples, repeat=1):
    for _ in range(8):
        graph.replay()
    events = []
    for _ in range(samples):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        events.append((start, end))
    torch.cuda.synchronize()
    return [start.elapsed_time(end) / repeat for start, end in events]


def summary(rounds, batch=1):
    raw = [value for row in rounds for value in row]
    return {"mean_ms": float(np.mean(raw)), "p50_ms": float(np.percentile(raw, 50)),
            "p95_ms": float(np.percentile(raw, 95)), "p99_ms": float(np.percentile(raw, 99)),
            "samples_per_second": 1000 * batch / float(np.mean(raw)),
            "round_mean_ms": [float(np.mean(row)) for row in rounds], "raw_ms": rounds}


@torch.inference_mode()
def matrix_benchmark(samples):
    results = []
    # 与当前模型吻合：QKV、attention output、GLU input/output、决策头 FFN。
    dims = [(768, 2304), (768, 768), (1152, 768), (768, 3072), (3072, 768)]
    for m in (27, 432, 512, 8192):
        for k, n in dims:
            print(f"matrix M={m} K={k} N={n}", flush=True)
            x = torch.randn(m, k, device="cuda", dtype=torch.float16)
            w = torch.randn(n, k, device="cuda", dtype=torch.float16) * 0.02
            qw, ws = quantize_weight(w)
            scale = x.float().abs().max() / 127
            q, dynamic_scales = quantize(x, scale, True)
            # 用实际矩阵 shape 为 Triton 选择 tile；每个候选先捕获再测速。
            for fused in (False, True):
                candidates = []
                a, scales = (x, scale) if fused else (q, dynamic_scales)
                for tile in TILES:
                    graph, _ = capture(lambda: triton_gemm(a, qw, scales, ws, None, fused, tile), 10)
                    times = measure(graph, 8, 10)
                    candidates.append({"tile": tile, "ms": float(np.median(times))})
                    del graph
                TUNING[(m, n, k, fused)] = {"tile": min(candidates, key=lambda r: r["ms"])["tile"],
                                           "candidates": candidates}

            def quantized(mode):
                if mode == "triton-static-fused":
                    return triton_gemm(x, qw, scale, ws, None, True)
                a, scales = quantize(x, scale, "dynamic" in mode)
                if mode.startswith("cublas"):
                    return int_mm(a, qw, scales, ws, None)
                return triton_gemm(a, qw, scales, ws, None, False)

            functions = {"fp16": lambda: torch.nn.functional.linear(x, w),
                         **{mode: lambda mode=mode: quantized(mode) for mode in MATRIX_MODES}}
            # 额外测试纯整数 GEMM，区分硬件收益和激活量化/反量化成本。
            functions["int8-gemm-only"] = lambda: torch._int_mm(q, qw.t())
            graphs = {name: capture(fn, 10)[0] for name, fn in functions.items()}
            raw = {name: [] for name in graphs}
            for round_index in range(3):
                names = list(graphs) if round_index % 2 == 0 else list(reversed(graphs))
                for name in names:
                    raw[name].append(measure(graphs[name], samples, 10))
            rows = {name: summary(values) for name, values in raw.items()}
            for name, row in rows.items():
                row["speedup_vs_fp16"] = rows["fp16"]["mean_ms"] / row["mean_ms"]
                row["effective_tops"] = 2 * m * n * k / (row["mean_ms"] * 1.0e9)
            results.append({"m": m, "k": k, "n": n, "variants": rows})
            graphs.clear()
            gc.collect()
    return results


def drift(reference, output):
    return [{"max_abs": float((a.float() - b.float()).abs().max()),
             "mean_abs": float((a.float() - b.float()).abs().mean()),
             "finite": bool(torch.isfinite(b).all()),
             "argmax_agreement": float((a.argmax(-1) == b.argmax(-1)).float().mean())}
            for a, b in zip(reference[:2], output[:2])]


def profile_graph(graph):
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as profiler:
        graph.replay()
        torch.cuda.synchronize()
    kernels = {}
    for event in profiler.events():
        if event.device_type == torch.autograd.DeviceType.CUDA:
            row = kernels.setdefault(event.name, {"name": event.name, "calls": 0, "total_ms": 0.0})
            row["calls"] += 1
            row["total_ms"] += event.device_time_total / 1000
    return sorted(kernels.values(), key=lambda row: row["total_ms"], reverse=True)


@torch.inference_mode()
def model_benchmark(model_dir, samples, compiled=False):
    inferencer = DecisionInferencer(Config(model_dir=model_dir, runner="cuda-graph", dtype="fp16",
                             max_batch_size=16, graph_streams=8, graph_prewarm_profiles=[]))
    runner = inferencer.runtime.runner
    prepared = runner.prepared
    inputs = {}
    for case in ("short", "long"):
        request = DecisionRequest.model_validate_json((PROJECT / f"scripts/inputs/{case}.json").read_text())
        item = inferencer.prepare(request)[0][2]
        for batch in (1, 16):
            collated = collate_items([[item] * batch], inferencer.runtime.tok.pad_token_id)
            inputs[(case, batch)] = [collated[key].cuda() for key in INPUT_KEYS]
    # 原生产实现先保存；显式注意力用于 FP16/W8A8 相同数学和相同张量布局。
    production = {}
    references = {}
    for key, tensors in inputs.items():
        entry = runner.capture(tensors)
        graph, _, outputs, _, _ = entry
        # 保留捕获前创建的静态输入、stream 和输出；它们不由 Graph 的私有池持有。
        production[key] = entry
        graph.replay()
        torch.cuda.synchronize()
        references[key] = tuple(value.clone() for value in outputs)
    original_heads = list(inferencer.runtime.runner.model.head.layers)
    explicit_heads(inferencer.runtime.runner.model)
    with torch.autocast("cuda", dtype=torch.float16):
        scales = calibrate(prepared, [inputs[(case, 1)] for case in ("short", "long")])
    covered = [{"name": name, "shape": list(module.weight.shape)}
               for name, module in eligible(prepared)]
    results = []
    for key, tensors in inputs.items():
        case, batch = key
        print(f"full model {case} batch={batch}", flush=True)
        production_entry = production.pop(key)
        graphs = {"fp16-production-streams8": production_entry[0]}
        outputs, cold, quantized_modules = {}, {}, {}
        use_autocast = True
        function = prepared

        def forward():
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_autocast):
                logits, acts = function(*tensors)
                return logits.float(), acts.float(), torch.softmax(acts.float(), -1)

        start = perf_counter()
        graphs["fp16-batched"], out = capture(forward)
        cold["fp16-batched"] = (perf_counter() - start) * 1000
        outputs["fp16-batched"] = tuple(value.clone() for value in out)
        use_autocast = False
        start = perf_counter()
        graphs["fp16-batched-native"], out = capture(forward)
        cold["fp16-batched-native"] = (perf_counter() - start) * 1000
        outputs["fp16-batched-native"] = tuple(value.clone() for value in out)
        for mode in MODES:
            start = perf_counter()
            use_autocast = not mode.endswith("-native")
            originals = replace_linears(prepared, scales, mode.removesuffix("-native"))
            # Graph 的外部权重 buffer 必须活到最后一次 replay，避免被 allocator 复用。
            quantized_modules[mode] = [getattr(owner, attribute) for owner, attribute, _ in originals]
            try:
                graphs[mode], out = capture(forward)
                outputs[mode] = tuple(value.clone() for value in out)
                cold[mode] = (perf_counter() - start) * 1000
            finally:
                restore_linears(originals)
        if compiled and batch == 16:
            for name in ("fp16-native-compile", "triton-dynamic-ln-native-compile"):
                use_autocast = False
                originals = []
                start = perf_counter()
                if name.startswith("triton"):
                    originals = replace_linears(prepared, scales, "triton-dynamic-ln")
                    quantized_modules[name] = [getattr(owner, attribute) for owner, attribute, _ in originals]
                try:
                    function = torch.compile(prepared, fullgraph=True, dynamic=False,
                                             options={"triton.cudagraphs": False})
                    graphs[name], out = capture(forward)
                    outputs[name] = tuple(value.clone() for value in out)
                    cold[name] = (perf_counter() - start) * 1000
                finally:
                    restore_linears(originals)
                    function = prepared
        torch.cuda.synchronize()
        raw = {name: [] for name in graphs}
        for round_index in range(3):
            names = list(graphs) if round_index % 2 == 0 else list(reversed(graphs))
            for name in names:
                raw[name].append(measure(graphs[name], samples))
        rows = {name: summary(values, batch) for name, values in raw.items()}
        for name, row in rows.items():
            row["speedup_vs_same_batch_fp16"] = rows["fp16-batched"]["mean_ms"] / row["mean_ms"]
            row["speedup_vs_native_fp16"] = rows["fp16-batched-native"]["mean_ms"] / row["mean_ms"]
            if "fp16-native-compile" in rows:
                row["speedup_vs_compiled_fp16"] = rows["fp16-native-compile"]["mean_ms"] / row["mean_ms"]
            row["speedup_vs_production_fp16"] = rows["fp16-production-streams8"]["mean_ms"] / row["mean_ms"]
            row["capture_and_quantize_ms"] = cold.get(name)
            if name in outputs:
                row["drift_vs_production_fp16"] = drift(references[key], outputs[name])
        if case == "long" and batch == 16:
            best = min((name for name in rows if name.startswith(("triton", "cublas"))),
                       key=lambda name: rows[name]["mean_ms"])
            for name in ("fp16-batched", "fp16-batched-native", best):
                rows[name]["kernel_profile"] = profile_graph(graphs[name])
        results.append({"case": case, "batch": batch, "length": tensors[0].shape[1], "variants": rows})
        for name, row in rows.items():
            print(f"  {name}: {row['mean_ms']:.3f} ms, {row['samples_per_second']:.1f} samples/s", flush=True)
        graphs.clear()
        quantized_modules.clear()
        gc.collect()
    del original_heads
    return {"cases": results, "quantized_linears": covered,
            "quantized_weight_parameters": sum(np.prod(row["shape"]) for row in covered),
            "fp16_operations": ["embedding", "layer norm", "RoPE", "SDPA QK/AV",
                                 "GELU/GLU", "scorer final 768->1", "action head 772->256->2"],
            "static_scale_source": "short/long benchmark inputs, batch=1, not an accuracy calibration corpus"}


@app.command()
def main(
    output: Path = typer.Option(Path(__file__).parent / "results.json"),
    model_dir: Path = typer.Option(..., envvar="LAYA_MODEL_DIR"),
    samples: int = typer.Option(30, min=3, max=100),
    matrices: bool = typer.Option(True),
    full_model: bool = typer.Option(True),
    tuning_file: Path | None = typer.Option(None),
    compiled: bool = typer.Option(False),
):
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "4")
    torch.manual_seed(42)
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    uuid = command(["nvidia-smi", "-i", "0", "--query-gpu=uuid", "--format=csv,noheader"])
    assert_isolated(uuid)
    result = {"created_at": datetime.now(timezone.utc).isoformat(),
              "revision": command(["git", "rev-parse", "HEAD"]),
              "workspace_status": command(["git", "status", "--short"]),
              "torch": torch.__version__, "triton": triton.__version__, "cuda": torch.version.cuda,
              "gpu": torch.cuda.get_device_name(), "capability": torch.cuda.get_device_capability(),
              "gpu_uuid": uuid, "model_dir": str(model_dir), "seed": 42,
              "command": "poetry run python -m tests.w8a8.benchmark",
              "argv": sys.argv, "compiled_batch16": compiled,
              "gpu_environment": command(["nvidia-smi", "-i", "0",
                  "--query-gpu=name,driver_version,pstate,clocks.sm,clocks.mem,power.limit", "--format=csv"]),
              "measurement": "GPU CUDA Graph, inputs resident on device; no HTTP/tokenizer/H2D/D2H; quantization included except int8-gemm-only",
              "rounds": 3, "samples_per_round": samples, "process_census_before": census()}
    if tuning_file:
        for row in json.loads(tuning_file.read_text())["tuning"]:
            TUNING[tuple(row["shape"])] = {key: value for key, value in row.items() if key != "shape"}
        result["tuning_file"] = str(tuning_file)
    try:
        if matrices:
            result["matrices"] = matrix_benchmark(samples)
        if full_model:
            result["model"] = model_benchmark(model_dir, samples, compiled)
        result["int8_evidence"] = EVIDENCE
        result["tuning"] = [{"shape": key, **value} for key, value in TUNING.items()]
        result["process_census_after"] = census()
        assert_isolated(uuid)
    finally:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=int) + "\n")
    print(f"saved {output}", flush=True)


if __name__ == "__main__":
    app()
