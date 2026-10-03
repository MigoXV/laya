"""独立设备进程；仅通过有界 JSON 行传输领域输入输出。"""

import json
import os
import sys


def main():
    from laya.configs.settings import Config
    from laya.inferencers.contracts import InferenceRequest
    from laya.inferencers.decision import DecisionInferencer

    config = Config.model_validate_json(sys.argv[1])
    # 模型库和它的子进程可写 stdout；协议独占原 stdout 的副本。
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    inferencer = DecisionInferencer(config)
    warmup = InferenceRequest(
        state="测试服务是否就绪。",
        questions={
            "ready": {
                "type": "choice",
                "instructions": "选择主题",
                "criteria": ["测试", "财务"],
            }
        },
    )
    try:
        inferencer.infer(warmup)
        print(json.dumps({"ready": inferencer.runtime.info}), file=protocol, flush=True)
        for line in sys.stdin:
            try:
                payload = json.loads(line)
                if payload.get("op") == "infer_batch":
                    if payload.get("version") != 1 or not 1 <= len(payload["items"]) <= config.max_batch_size:
                        raise ValueError("invalid_worker_batch")
                    entries, requests, replies = payload["items"], [], []
                    if [entry["id"] for entry in entries] != list(range(len(entries))):
                        raise ValueError("invalid_worker_batch_ids")
                    for entry in entries:
                        try:
                            requests.append(InferenceRequest.model_validate(entry["payload"]))
                        except ValueError as exc:
                            requests.append(None)
                            replies.append({"id": entry["id"], "error": str(exc), "status": 422})
                    valid = [request for request in requests if request is not None]
                    results = iter(inferencer.infer_many(valid))
                    for entry, request in zip(entries, requests):
                        if request is None:
                            continue
                        result = next(results)
                        reply = {"error": str(result), "status": 422} if isinstance(result, ValueError) else {"result": result}
                        replies.append({"id": entry["id"], **reply})
                    response = {"version": 1, "replies": replies,
                                "batches": inferencer.last_batches,
                                "runner_metrics": getattr(inferencer.runtime.runner, "metrics", lambda: {})()}
                else:
                    result = inferencer.infer(InferenceRequest.model_validate(payload))
                    response = {"result": result, "batches": inferencer.last_batches,
                                "runner_metrics": getattr(inferencer.runtime.runner, "metrics", lambda: {})()}
            except ValueError as exc:
                response = {"error": str(exc), "status": 422}
            except Exception:
                import traceback

                traceback.print_exc(file=sys.stderr)
                response = {"error": "inference_failed", "status": 500}
            print(json.dumps(response, ensure_ascii=False), file=protocol, flush=True)
    finally:
        inferencer.close()
        protocol.close()


if __name__ == "__main__":
    main()
