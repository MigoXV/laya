"""独立设备进程；仅通过有界 JSON 行传输领域输入输出。"""

import json
import sys


def main():
    from .config import Config
    from .contracts import DecisionRequest
    from .runtime import Runtime

    config = Config.model_validate_json(sys.argv[1])
    runtime = Runtime(config)
    warmup = DecisionRequest(
        state="测试服务是否就绪。",
        questions={
            "ready": {
                "type": "choice",
                "instructions": "选择主题",
                "criteria": ["测试", "财务"],
            }
        },
    )
    runtime.infer(warmup)
    print(json.dumps({"ready": runtime.info}), flush=True)
    for line in sys.stdin:
        try:
            payload = json.loads(line)
            result = runtime.infer(DecisionRequest.model_validate(payload))
            response = {"result": result}
        except ValueError as exc:
            response = {"error": str(exc), "status": 422}
        except Exception:
            import traceback

            traceback.print_exc(file=sys.stderr)
            response = {"error": "inference_failed", "status": 500}
        print(json.dumps(response, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
