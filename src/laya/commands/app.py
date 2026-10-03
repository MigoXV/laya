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
def inspect(model_dir: Path = typer.Option(..., envvar="LAYA_MODEL_DIR")):
    config = Config(model_dir=model_dir)
    typer.echo(
        json.dumps(
            {
                "path": str(config.model_dir),
                "config": json.loads((model_dir / "rl_agent_config.json").read_text()),
                "weights_bytes": (model_dir / "model.safetensors").stat().st_size,
            },
            ensure_ascii=False,
        )
    )


@app.command()
def infer(
    input_path: Path = typer.Argument(...),
    model_dir: Path = typer.Option(..., envvar="LAYA_MODEL_DIR"),
    device: str = typer.Option("cpu", envvar="LAYA_DEVICE"),
):
    from laya.runtime import Runtime

    runtime = Runtime(Config(model_dir=model_dir, device=device))
    result = runtime.infer(DecisionRequest.model_validate_json(input_path.read_text()))
    typer.echo(json.dumps(result, ensure_ascii=False))


@app.command()
def serve(
    model_dir: Path = typer.Option(..., envvar="LAYA_MODEL_DIR"),
    device: str = typer.Option("cpu", envvar="LAYA_DEVICE"),
    host: str = typer.Option("0.0.0.0", envvar="LAYA_HOST"),
    port: int = typer.Option(10002, envvar="LAYA_PORT"),
):
    import uvicorn
    from laya.api import create_app

    config = Config(model_dir=model_dir, device=device, host=host, port=port)
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
):
    """只使用内置合成文本；逐轮原始延迟和汇总输出到 stdout。"""
    import httpx

    async def run():
        async with httpx.AsyncClient(base_url=url, timeout=60) as client:
            metadata = (await client.get("/v1/info")).json()
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
            for _ in range(3):
                (await client.post("/v1/decisions", json=payload)).raise_for_status()
            for concurrency in (1, 2, 4, 8, 40):
                for round_id in range(rounds):
                    sem = asyncio.Semaphore(concurrency)

                    async def once():
                        async with sem:
                            start = perf_counter()
                            try:
                                response = await client.post(
                                    "/v1/decisions", json=payload
                                )
                                return {
                                    "ms": (perf_counter() - start) * 1000,
                                    "status": response.status_code,
                                }
                            except httpx.HTTPError as exc:
                                return {
                                    "ms": (perf_counter() - start) * 1000,
                                    "error": type(exc).__name__,
                                }

                    start = perf_counter()
                    raw = await asyncio.gather(
                        *(once() for _ in range(max(samples, concurrency)))
                    )
                    elapsed = perf_counter() - start
                    times = sorted(row["ms"] for row in raw)
                    typer.echo(
                        json.dumps(
                            {
                                "model": metadata,
                                "concurrency": concurrency,
                                "round": round_id,
                                "raw": raw,
                                "successful_qps": sum(
                                    row.get("status") == 200 for row in raw
                                )
                                / elapsed,
                                "latency_ms": {
                                    f"p{p}": times[int((len(times) - 1) * p / 100)]
                                    for p in (50, 95, 99)
                                },
                            }
                        )
                    )

    asyncio.run(run())


if __name__ == "__main__":
    app()
