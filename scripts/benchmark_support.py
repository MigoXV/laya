"""基准与对齐检查共用的进程、GPU 盘点及输入辅助函数。"""

import json
import subprocess
import time


def command(args):
    return subprocess.check_output(args, text=True).strip()


def gpu_processes():
    return command(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv"])


def wait_for_gpu(uuid):
    # 进程退出后，驱动可能稍晚释放/更新 CUDA context。
    deadline = time.monotonic() + 10
    while uuid in gpu_processes():
        if time.monotonic() >= deadline:
            raise RuntimeError("GPU 0 存在推理进程，不能做隔离性能测试")
        time.sleep(0.25)


def stop(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def alignment_cases(model_dir=None):
    questions = [
        {"type": "choice", "instructions": "谁负责测试？", "criteria": ["小李", "小王"]},
        {"type": "score", "instructions": "测试的重要程度？", "criteria": ["低", "中", "高"]},
        {"type": "noul", "instructions": "小李负责测试吗？"},
    ]
    cases = [
        {"state": state, "questions": {"q": q}}
        for state in ("小李负责测试。", "小李负责测试，小王只负责发布。",
                      "讨论背景。" * 180 + "小李负责测试，小王只负责发布。")
        for q in questions
    ]
    cases.extend([
        {"state": {"测试": "小李", "发布": "小王"}, "questions": {
            str(i): q for i, q in enumerate(questions)}},
        {"state": "Alice owns testing; Bob handles releases.", "questions": {"q": {
            "type": "choice", "instructions": "Who owns testing?", "criteria": ["Alice", "Bob"]}}},
        {"state": "版本编号是 7。", "questions": {"q": {
            "type": "choice", "instructions": "版本编号？", "criteria": [str(i) for i in range(16)]}}},
    ])
    # 由公共 tokenizer 构造真实 1024 token 边界输入。
    from laya.configs.settings import Config
    from laya.api.contracts import DecisionRequest
    from laya.inferencers.preprocessing import checked_sequence
    from transformers import PreTrainedTokenizerFast

    root = Config(**({"model_dir": model_dir} if model_dir is not None else {})).model_dir
    config = json.loads((root / "config.json").read_text())
    tok = PreTrainedTokenizerFast(tokenizer_file=str(root / "tokenizer.json"), **config["tokenizer"])
    q = DecisionRequest.model_validate(cases[0]).questions["q"]
    head, _ = checked_sequence(tok, "", q, config["input_limits"])
    state = ("a " * (1024 - len(head))).strip()
    ids, _ = checked_sequence(tok, state, q, config["input_limits"])
    assert len(ids) == 1024
    cases.append({"state": state, "questions": {"q": questions[0]}})
    return cases


def compare(reference, actual):
    maxima = dict(probabilities=0.0, confidence=0.0, score=0.0, noul=0.0, act_probability=0.0)
    choices = 0
    for expected, result in zip(reference, actual):
        assert expected["usage"] == result["usage"]
        assert expected["answers"].keys() == result["answers"].keys()
        for key, a in expected["answers"].items():
            b = result["answers"][key]
            assert a.keys() == b.keys()
            choices += int(a.get("choice") != b.get("choice"))
            for option, value in a["probabilities"].items():
                maxima["probabilities"] = max(maxima["probabilities"], abs(value - b["probabilities"][option]))
            for field in set(maxima) - {"probabilities"}:
                if field in a:
                    maxima[field] = max(maxima[field], abs(a[field] - b[field]))
    return {"max_abs": maxima, "choice_mismatches": choices}
