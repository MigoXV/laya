"""所有运行入口共享 Runtime 或 HTTP 契约。"""

import asyncio
import json
import logging
from pathlib import Path
from time import perf_counter

import typer

from laya.config import Config
from laya.contracts import DecisionRequest

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
app = typer.Typer()


@app.command()
def inspect(model_dir: Path | None = typer.Option(None, envvar="LAYA_MODEL_DIR")):
    config = Config(**({"model_dir": model_dir} if model_dir is not None else {}))
    typer.echo(
        json.dumps(
            {
                "path": str(config.model_dir),
                "config": json.loads((config.model_dir / "config.json").read_text()),
                "weights_bytes": (config.model_dir / "model.safetensors").stat().st_size,
            },
            ensure_ascii=False,
        )
    )


@app.command()
def infer(
    input_path: Path = typer.Argument(...),
    model_dir: Path | None = typer.Option(None, envvar="LAYA_MODEL_DIR"),
    device: str | None = typer.Option(None, envvar="LAYA_DEVICE"),
    dtype: str | None = typer.Option(None, envvar="LAYA_DTYPE"),
    runner: str | None = typer.Option(None, envvar="LAYA_RUNNER"),
):
    from laya.runtime import Runtime

    config = Config(
        **{key: value for key, value in (
            ("model_dir", model_dir), ("device", device), ("dtype", dtype), ("runner", runner)
        ) if value is not None}
    )
    runtime = Runtime(config)
    try:
        result = runtime.infer(DecisionRequest.model_validate_json(input_path.read_text()))
    finally:
        runtime.close()
    typer.echo(json.dumps(result, ensure_ascii=False))


@app.command()
def serve(
    model_dir: Path | None = typer.Option(None, envvar="LAYA_MODEL_DIR"),
    device: str | None = typer.Option(None, envvar="LAYA_DEVICE"),
    dtype: str | None = typer.Option(None, envvar="LAYA_DTYPE"),
    runner: str | None = typer.Option(None, envvar="LAYA_RUNNER"),
    max_batch_size: int | None = typer.Option(None, envvar="LAYA_MAX_BATCH_SIZE"),
    batch_wait_ms: float | None = typer.Option(None, envvar="LAYA_BATCH_WAIT_MS"),
    graph_streams: int | None = typer.Option(None, envvar="LAYA_GRAPH_STREAMS"),
    host: str = typer.Option("0.0.0.0", envvar="LAYA_HOST"),
    port: int = typer.Option(10002, envvar="LAYA_PORT"),
):
    import uvicorn
    from laya.api import create_app

    config = Config(
        **{key: value for key, value in (
            ("model_dir", model_dir), ("device", device), ("dtype", dtype), ("runner", runner),
            ("max_batch_size", max_batch_size), ("batch_wait_ms", batch_wait_ms),
            ("graph_streams", graph_streams)
        ) if value is not None},
        host=host,
        port=port,
    )
    uvicorn.run(
        create_app(config),
        host=config.host,
        port=config.port,
        workers=1,
        access_log=False,
    )


@app.command()
def benchmark(
    url: str = typer.Option("http://127.0.0.1:10002", envvar="LAYA_URL"),
    rounds: int = typer.Option(3, min=3),
    samples: int = typer.Option(32, min=8),
    concurrency: list[int] | None = typer.Option(None, min=1, max=256),
    warmup: int = typer.Option(16, min=3),
    input_path: Path | None = typer.Option(None, exists=True, dir_okay=False),
):
    """完整 HTTP 响应延迟；固定并发闭环负载，逐轮输出原始数据。"""
    import httpx

    async def run():
        failed = False
        levels = concurrency or [1, 2, 4, 8, 40]
        async with httpx.AsyncClient(
            base_url=url, timeout=60, trust_env=False,
            limits=httpx.Limits(
                max_connections=max(levels), max_keepalive_connections=max(levels)
            ),
        ) as client:
            info = await client.get("/v1/info")
            info.raise_for_status()
            metadata = info.json()
            payload = {
                "state": "小李负责测试，小王负责发布。",
                "questions": {
                    "owner": {
                        "type": "choice",
                        "instructions": "谁负责测试？",
                        "criteria": ["小李", "小王"],
                    }
                },
            }
            if input_path is not None:
                payload = json.loads(input_path.read_text())
            payload = DecisionRequest.model_validate(payload).model_dump()
            reference = None
            for _ in range(warmup):
                response = await client.post("/v1/decisions", json=payload)
                response.raise_for_status()
                reference = response.json()
            for level in levels:
                responses = await asyncio.gather(*(
                    client.post("/v1/decisions", json=payload) for _ in range(level)
                ))
                for response in responses:
                    response.raise_for_status()
                for round_id in range(rounds):
                    count = max(samples, level)

                    async def lane(lane_id):
                        rows = []
                        for index in range(lane_id, count, level):
                            start = perf_counter()
                            try:
                                response = await client.post(
                                    "/v1/decisions", json=payload
                                )
                                # post() 已读完全部响应正文；排队、网络均计入。
                                row = {
                                    "index": index,
                                    "ms": (perf_counter() - start) * 1000,
                                    "status": response.status_code,
                                }
                                if response.status_code == 200:
                                    result = response.json()
                                    row.update(
                                        timings=result.get("timings", {}),
                                        aligned=result["answers"] == reference["answers"],
                                    )
                            except httpx.HTTPError as exc:
                                row = {
                                    "index": index,
                                    "ms": (perf_counter() - start) * 1000,
                                    "error": type(exc).__name__,
                                }
                            rows.append(row)
                        return rows

                    start = perf_counter()
                    lanes = await asyncio.gather(*(lane(i) for i in range(level)))
                    elapsed = perf_counter() - start
                    raw = sorted((row for lane_rows in lanes for row in lane_rows),
                                 key=lambda row: row["index"])
                    succeeded = sum(row.get("status") == 200 for row in raw)
                    errors = len(raw) - succeeded
                    mismatches = sum(row.get("aligned") is False for row in raw)
                    failed |= bool(errors or mismatches)
                    times = sorted(row["ms"] for row in raw)
                    typer.echo(
                        json.dumps(
                            {
                                "model": metadata,
                                "payload": payload,
                                "reference": reference,
                                "load": "closed_loop",
                                "warmup_serial": warmup,
                                "warmup_concurrent": level,
                                "concurrency": level,
                                "round": round_id,
                                "elapsed_s": elapsed,
                                "successful": succeeded,
                                "errors": errors,
                                "answer_mismatches": mismatches,
                                "raw": raw,
                                "successful_qps": succeeded / elapsed,
                                "latency_ms": {
                                    f"p{p}": times[int((len(times) - 1) * p / 100)]
                                    for p in (50, 95, 99, 100)
                                },
                            }
                        )
                    )
        if failed:
            raise typer.Exit(1)

    asyncio.run(run())


if __name__ == "__main__":
    app()
