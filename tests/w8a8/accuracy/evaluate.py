"""原始 FP16、优化 FP16、最快 W8A8 的带标签评测；不把一致率当准确率。"""

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from time import perf_counter

import numpy as np
import torch
import typer

from laya.configs.settings import Config
from laya.api.contracts import DecisionRequest
from laya.runners.inputs import INPUT_KEYS
from laya.runners.prepared import PreparedModel
from laya.inferencers.batching import collate_items
from laya.inferencers.decision import DecisionInferencer
from ..benchmark import capture
from ..kernels import EVIDENCE, TUNING
from ..model import eligible, explicit_heads, replace_linears, restore_linears


HERE = Path(__file__).resolve().parent
app = typer.Typer()


def load_cases(data):
    cases = []
    manifest = json.loads((data / "manifest.json").read_text())
    for source in manifest["sources"]:
        path = data / source["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == source["sha256"]
        for index, line in enumerate(path.read_text().splitlines()):
            if not line or line.startswith("#"):
                continue
            case = json.loads(line)
            tags = case["tags"]
            ident = next(tag[3:] for tag in tags if tag.startswith("id:"))
            suite = f"xnli-{source['config']}" if source["dataset"] == "facebook/xnli" else "zh-decision"
            # 中英是平行翻译，bootstrap 时归为同一 semantic case。
            cluster = ident if suite.startswith("xnli") else f"zh-decision:{ident}"
            cases.append({"id": f"{suite}:{ident}", "cluster": cluster, "suite": suite,
                          "tags": tags, "language": case["language"], "request": case,
                          "source_line": index + 1})
    return cases, manifest


def request_from_case(case):
    # 公共评测格式的 expected/tags/language 只属于评测器，不送入模型或 API。
    return DecisionRequest.model_validate({"state": case["state"], "questions": case["questions"]})


def prediction(inferencer, question, item, logits, acts, expected):
    answer = inferencer.answer(question, item, np.asarray(logits), torch.tensor(acts).softmax(-1).numpy())
    probabilities = list(answer["probabilities"].values())
    if question.type == "choice":
        target = list(answer["probabilities"]).index(expected)
        label = answer["choice"]
        correct = label == expected
    elif question.type == "noul":
        target = int(expected)
        label = answer["noul"] >= 0.5
        correct = label == expected
    else:
        target = int(expected)
        label = int(np.argmax(probabilities))
        correct = None  # score 返回连续期望值，单独报告 MAE，而不混进分类准确率。
    return {"answer": answer, "probabilities": probabilities, "label": label, "correct": correct,
            "target_index": target, "logits": [float(x) for x in logits],
            "act_logits": [float(x) for x in acts],
            "score_abs_error": abs(answer["score"] - expected) if question.type == "score" else None}


def aggregate(rows, variant):
    classified = [row for row in rows if row["type"] != "score"]
    scores = [row for row in rows if row["type"] == "score"]
    good = sum(row["predictions"][variant]["correct"] for row in classified)
    flips, lost, gained, probabilities, tv, score_diffs = 0, 0, 0, [], [], []
    nll, brier, clusters = [], [], defaultdict(lambda: [0, 0])
    for row in rows:
        a, b = row["predictions"]["fp16-original"], row["predictions"][variant]
        diff = np.abs(np.asarray(a["probabilities"]) - np.asarray(b["probabilities"]))
        probabilities.extend(diff.tolist())
        tv.append(float(diff.sum() / 2))
        if row["type"] == "score":
            score_diffs.append(abs(a["answer"]["score"] - b["answer"]["score"]))
        else:
            flips += a["label"] != b["label"]
            lost += a["correct"] and not b["correct"]
            gained += not a["correct"] and b["correct"]
            p = np.asarray(b["probabilities"])
            gold = np.zeros_like(p)
            gold[b["target_index"]] = 1
            nll.append(-np.log(max(p[b["target_index"]], 1.0e-12)))
            brier.append(float(((p - gold) ** 2).sum()))
            clusters[row["cluster"]][0] += int(b["correct"]) - int(a["correct"])
            clusters[row["cluster"]][1] += 1
    baseline_good = sum(row["predictions"]["fp16-original"]["correct"] for row in classified)
    ci = None
    if clusters:
        values = np.asarray(list(clusters.values()))
        rng = np.random.default_rng(42)
        selections = rng.integers(0, len(values), (4000, len(values)))
        totals = values[selections].sum(1)
        ci = [float(x) for x in np.percentile(100 * totals[:, 0] / totals[:, 1], [2.5, 97.5])]
    return {"questions": len(rows), "classification_n": len(classified), "classification_correct": int(good),
            "accuracy": good / len(classified) if classified else None,
            "accuracy_delta_pp_vs_original": 100 * (good - baseline_good) / len(classified) if classified else None,
            "accuracy_delta_95pct_cluster_bootstrap_pp": ci,
            "label_flips": int(flips), "label_flip_rate": flips / len(classified) if classified else None,
            "original_correct_to_wrong": int(lost), "original_wrong_to_correct": int(gained),
            "probability_max_abs": float(max(probabilities)), "probability_mean_abs": float(np.mean(probabilities)),
            "mean_total_variation": float(np.mean(tv)),
            "nll": float(np.mean(nll)) if nll else None, "brier_sum": float(np.mean(brier)) if brier else None,
            "score_n": len(scores),
            "score_mae": float(np.mean([row["predictions"][variant]["score_abs_error"] for row in scores])) if scores else None,
            "score_mean_abs_drift": float(np.mean(score_diffs)) if score_diffs else None}


@app.command()
@torch.inference_mode()
def main(data: Path = typer.Option(HERE / "data"), output: Path = typer.Option(HERE / "results.json"),
         model_dir: Path = typer.Option(..., envvar="LAYA_MODEL_DIR"),
         tuning_file: Path = typer.Option(HERE.parent / "tuning.json"), batch_size: int = typer.Option(16)):
    if batch_size != 16:
        raise ValueError("本实验固定 batch=16，与性能实验的最快路径保持一致")
    torch.manual_seed(42)
    torch.set_num_threads(4)
    torch._dynamo.config.recompile_limit = 16
    for row in json.loads(tuning_file.read_text())["tuning"]:
        TUNING[tuple(row["shape"])] = {k: v for k, v in row.items() if k != "shape"}
    cases, manifest = load_cases(data)
    inferencer = DecisionInferencer(Config(model_dir=model_dir, runner="eager", dtype="fp16", max_batch_size=16))
    model = inferencer.runtime.runner.model
    work, rejected = [], []
    for case in cases:
        try:
            request = request_from_case(case["request"])
            prepared = inferencer.prepare(request)
            for qid, question, item in prepared:
                work.append({**case, "qid": qid, "question": question, "item": item,
                             "gold": case["request"]["expected"][qid], "predictions": {}})
        except ValueError as exc:
            rejected.append({"id": case["id"], "error": str(exc)})
    # 不截断、不筛掉难例；任何预算/格式错误都要求显式修复或单独说明。
    if rejected:
        raise ValueError(f"评测样本存在 {len(rejected)} 个不能完整编码的请求: {rejected[:3]}")
    result = {"created_at": datetime.now(timezone.utc).isoformat(),
              "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "runtime": inferencer.runtime.info,
              "data_manifest": manifest, "requests": len(cases), "questions": len(work), "rejected": rejected,
              "quantization": "dynamic per-token A8, per-output-channel W8; no calibration data and no QAT",
              "reference": "original project FP16 autocast, exact unpadded sequence, batch=1"}
    start = perf_counter()
    for index, row in enumerate(work):
        batch = collate_items([[row["item"]]], inferencer.runtime.tok.pad_token_id)
        tensors = [batch[key].cuda() for key in INPUT_KEYS]
        with torch.autocast("cuda", dtype=torch.float16):
            logits, acts = model(*tensors)
        row["predictions"]["fp16-original"] = prediction(
            inferencer, row["question"], row["item"], logits[0].float().cpu().tolist(),
            acts[0].float().cpu().tolist(), row["gold"])
        if (index + 1) % 100 == 0:
            print(f"original FP16 {index + 1}/{len(work)}", flush=True)
    result["original_forward_elapsed_s"] = perf_counter() - start
    prepared = PreparedModel(model, inferencer.runtime.cfg["input_limits"]["max_len"]).eval()
    explicit_heads(model)
    length = max(len(row["item"]["ids"]) for row in work)
    length = min(1024, 1 << (length - 1).bit_length())
    markers = max(len(row["item"]["markers"]) for row in work)
    result["candidate_profile"] = {"batch": batch_size, "length": length, "markers": markers,
        "padding": "attention/marker masks preserve actual lengths; final batch duplicates valid rows and discards extras",
        "compilation": "full model, native FP16 residuals, no emulate_precision_casts; CUDA Graph"}
    print(f"candidate static profile: B={batch_size}, L={length}, K={markers}", flush=True)

    def cpu_inputs(rows):
        items = [row["item"] for row in rows]
        items += [items[0]] * (batch_size - len(items))
        batch = collate_items([items], inferencer.runtime.tok.pad_token_id)
        padded = {"input_ids": torch.full((batch_size, length), inferencer.runtime.tok.pad_token_id, dtype=torch.long),
                  "attention_mask": torch.zeros(batch_size, length, dtype=torch.long),
                  "marker_pos": torch.zeros(batch_size, markers, dtype=torch.long),
                  "marker_mask": torch.zeros(batch_size, markers, dtype=torch.bool),
                  "qtype": batch["qtype"]}
        for key in INPUT_KEYS[:-1]:
            padded[key][:, :batch[key].shape[1]] = batch[key]
        return [padded[key] for key in INPUT_KEYS]

    static = [tensor.cuda() for tensor in cpu_inputs(work[:batch_size])]
    scales = {name: torch.ones((), device="cuda", dtype=torch.float32) for name, _ in eligible(prepared)}
    for variant in ("fp16-native-compile", "w8a8-native-compile"):
        originals = []
        start = perf_counter()
        if variant.startswith("w8a8"):
            originals = replace_linears(prepared, scales, "triton-dynamic-ln")
            # 实际运行一次以预热自定义 kernel、采集 PTX；编译期间不能访问 kernel.asm。
            prepared(*static)
        try:
            compiled = torch.compile(prepared, fullgraph=True, dynamic=False, options={"triton.cudagraphs": False})
            graph, outputs = capture(lambda function=compiled: function(*static))
            result[f"{variant}_capture_compile_s"] = perf_counter() - start
            for offset in range(0, len(work), batch_size):
                rows = work[offset:offset + batch_size]
                for destination, source in zip(static, cpu_inputs(rows)):
                    destination.copy_(source)
                graph.replay()
                logits, acts = [value.float().cpu().numpy() for value in outputs]
                if not np.isfinite(logits).all() or not np.isfinite(acts).all():
                    raise ValueError(f"{variant}: non-finite output at offset={offset}")
                for i, row in enumerate(rows):
                    k = len(row["item"]["markers"])
                    row["predictions"][variant] = prediction(
                        inferencer, row["question"], row["item"], logits[i, :k].tolist(), acts[i].tolist(), row["gold"])
            print(f"{variant} evaluated {len(work)} questions", flush=True)
            del graph, compiled, outputs
        finally:
            restore_linears(originals)
    raw = [{"id": row["id"], "cluster": row["cluster"], "suite": row["suite"], "tags": row["tags"],
            "qid": row["qid"], "type": row["question"].type, "gold": row["gold"],
            "tokens": len(row["item"]["ids"]), "predictions": row["predictions"]} for row in work]
    groups = {"all": raw}
    for row in raw:
        groups.setdefault(row["suite"], []).append(row)
        groups.setdefault(f"type:{row['type']}", []).append(row)
        if row["suite"] == "zh-decision":
            groups.setdefault(f"zh-decision:{row['tags'][0]}:{row['qid']}", []).append(row)
    variants = ("fp16-original", "fp16-native-compile", "w8a8-native-compile")
    result["summaries"] = {name: {variant: aggregate(rows, variant) for variant in variants}
                           for name, rows in groups.items()}
    result["int8_evidence"] = EVIDENCE
    result["raw"] = raw
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    for name in ("all", "xnli-zh", "xnli-en", "zh-decision"):
        print(name, {variant: result["summaries"][name][variant]["accuracy"] for variant in variants}, flush=True)
    print(f"saved {output}", flush=True)


if __name__ == "__main__":
    app()
