"""有界无状态调度、子进程监督和取消；模型永不在 API 线程执行。"""

import asyncio
from collections import Counter, deque
from dataclasses import dataclass
import json
import logging
import subprocess
import sys
from time import monotonic


class EngineError(RuntimeError):
    def __init__(self, message, status=503):
        super().__init__(message)
        self.status = status


@dataclass
class Job:
    payload: dict
    future: asyncio.Future
    enqueued: float
    deadline: float


class Engine:
    def __init__(self, config, command=None):
        self.config = config
        self.command = command or [
            sys.executable,
            "-m",
            "laya.worker",
            config.model_dump_json(),
        ]
        self.queue = asyncio.Queue(maxsize=config.queue_size)
        self.ready = False
        self.info = {}
        self.process = None
        self.dispatcher = self.monitor = None
        self.counters = Counter()
        self.latencies = deque(maxlen=4096)
        self.stage_times = {
            key: deque(maxlen=4096)
            for key in ("queue_ms", "preprocess_ms", "inference_ms")
        }
        self.active = None

    async def start(self):
        try:
            self.process = await asyncio.create_subprocess_exec(
                *self.command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                limit=1024 * 1024,
            )
            line = await asyncio.wait_for(
                self.process.stdout.readline(), self.config.startup_timeout
            )
            self.info = json.loads(line)["ready"]
            revision = subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True
            ).stdout.strip()
            self.info.update(
                service_revision=revision or "unknown", worker_pid=self.process.pid
            )
            logging.getLogger(__name__).info("Worker ready: %s", self.info)
            self.ready = True
            self.dispatcher = asyncio.create_task(self.dispatch())
            self.monitor = asyncio.create_task(self.supervise())
        except BaseException:
            await self.close()
            raise

    def fail_pending(self, reason):
        if self.active and not self.active.future.done():
            self.active.future.set_exception(EngineError(reason))
        while not self.queue.empty():
            job = self.queue.get_nowait()
            if not job.future.done():
                job.future.set_exception(EngineError(reason))
            self.queue.task_done()

    async def supervise(self):
        await self.process.wait()
        self.ready = False
        self.counters["worker_exits"] += 1
        self.fail_pending("worker_exited")
        if self.dispatcher:
            self.dispatcher.cancel()

    async def submit(self, payload):
        if not self.ready:
            raise EngineError("not_ready")
        now = monotonic()
        future = asyncio.get_running_loop().create_future()
        job = Job(payload, future, now, now + self.config.request_timeout)
        try:
            self.queue.put_nowait(job)
        except asyncio.QueueFull:
            self.counters["rejected"] += 1
            raise EngineError("queue_full", 429)
        self.counters["submitted"] += 1
        self.counters["queue_high_water"] = max(
            self.counters["queue_high_water"], self.queue.qsize()
        )
        try:
            return await asyncio.wait_for(
                asyncio.shield(future), self.config.request_timeout
            )
        except asyncio.TimeoutError:
            future.cancel()
            self.counters["timeouts"] += 1
            raise EngineError("deadline_exceeded", 504)
        except asyncio.CancelledError:
            future.cancel()
            self.counters["cancelled"] += 1
            raise
        finally:
            self.latencies.append((monotonic() - now) * 1000)

    async def dispatch(self):
        try:
            while self.ready:
                job = await self.queue.get()
                self.active = job
                try:
                    if job.future.done() or monotonic() >= job.deadline:
                        if not job.future.done():
                            job.future.set_exception(EngineError("queue_deadline", 504))
                        self.counters["skipped"] += 1
                        continue
                    queue_ms = (monotonic() - job.enqueued) * 1000
                    self.process.stdin.write(
                        (json.dumps(job.payload, ensure_ascii=False) + "\n").encode()
                    )
                    await self.process.stdin.drain()
                    # 单请求 deadline 过后仍需读走对应结果；硬挂起则失败并终止 Worker。
                    raw = await asyncio.wait_for(
                        self.process.stdout.readline(),
                        self.config.request_timeout + self.config.shutdown_timeout,
                    )
                    if not raw:
                        raise EngineError("worker_disconnected")
                    reply = json.loads(raw)
                    if job.future.done() or monotonic() >= job.deadline:
                        self.counters["discarded"] += 1
                        if not job.future.done():
                            job.future.set_exception(
                                EngineError("deadline_exceeded", 504)
                            )
                    elif "error" in reply:
                        self.counters["failed"] += 1
                        job.future.set_exception(
                            EngineError(reply["error"], reply.get("status", 500))
                        )
                    else:
                        result = reply["result"]
                        result.setdefault("timings", {})["queue_ms"] = queue_ms
                        for key, values in self.stage_times.items():
                            if key in result["timings"]:
                                values.append(result["timings"][key])
                        job.future.set_result(result)
                        self.counters["completed"] += 1
                finally:
                    self.queue.task_done()
                    # 异常路径必须保留 active，供统一失败处理完成 Future。
                    if job.future.done():
                        self.active = None
        except asyncio.CancelledError:
            raise
        except Exception:
            self.ready = False
            self.counters["engine_failures"] += 1
            self.fail_pending("worker_failed")
            if self.process and self.process.returncode is None:
                self.process.terminate()

    def metrics(self):
        values = sorted(self.latencies)
        return {
            "ready": self.ready,
            "queue_depth": self.queue.qsize(),
            "queue_capacity": self.config.queue_size,
            "active": int(self.active is not None),
            "counters": dict(self.counters),
            "latency_ms": {
                f"p{p}": values[min(len(values) - 1, int((len(values) - 1) * p / 100))]
                if values
                else None
                for p in (50, 95, 99)
            },
            "latency_window": len(values),
            "stages_mean_ms": {
                k: sum(v) / len(v) if v else None for k, v in self.stage_times.items()
            },
        }

    async def close(self):
        self.ready = False
        self.fail_pending("service_stopping")
        for task in (self.dispatcher, self.monitor):
            if task:
                task.cancel()
        await asyncio.gather(
            *(t for t in (self.dispatcher, self.monitor) if t), return_exceptions=True
        )
        p = self.process
        if p and p.returncode is None:
            if p.stdin:
                p.stdin.close()
            try:
                await asyncio.wait_for(p.wait(), self.config.shutdown_timeout)
            except asyncio.TimeoutError:
                p.terminate()
                try:
                    await asyncio.wait_for(p.wait(), 2)
                except asyncio.TimeoutError:
                    p.kill()
                    await p.wait()
        self.active = None
