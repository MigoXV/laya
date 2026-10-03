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
            "laya.engine.worker",
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
        self.request_batches = Counter()
        self.model_batches = Counter()
        self.batch_tokens = self.padded_tokens = 0
        self.runner_metrics = {}

    async def start(self):
        try:
            self.process = await asyncio.create_subprocess_exec(
                *self.command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                # choice 的获选标签会在 probabilities 和 choice 中各出现一次。
                # 批次正文与返回行都必须有容量，不能沿用单请求的 1 MiB 上限。
                limit=1024 * 1024 + 2 * self.config.max_body_bytes
                * getattr(self.config, "max_batch_size", 1),
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
        for job in self.active or []:
            if not job.future.done():
                job.future.set_exception(EngineError(reason))
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
                first = await self.queue.get()
                self.active = [first]
                try:
                    capacity = getattr(self.config, "max_batch_size", 1)
                    until = min(first.enqueued + getattr(self.config, "batch_wait_ms", 0) / 1000,
                                first.deadline)
                    while len(self.active) < capacity:
                        if not self.queue.empty():
                            self.active.append(self.queue.get_nowait())
                        elif monotonic() < until:
                            try:
                                self.active.append(await asyncio.wait_for(self.queue.get(), until - monotonic()))
                            except asyncio.TimeoutError:
                                break
                        else:
                            break
                    jobs, queue_times = [], []
                    for job in self.active:
                        if job.future.done() or monotonic() >= job.deadline:
                            if not job.future.done():
                                job.future.set_exception(EngineError("queue_deadline", 504))
                            self.counters["skipped"] += 1
                        else:
                            jobs.append(job)
                            queue_times.append((monotonic() - job.enqueued) * 1000)
                    if not jobs:
                        continue
                    payload = jobs[0].payload if capacity == 1 else {
                        "version": 1, "op": "infer_batch",
                        "items": [{"id": i, "payload": job.payload} for i, job in enumerate(jobs)],
                    }
                    self.request_batches[len(jobs)] += 1
                    self.counters["worker_calls"] += 1
                    self.process.stdin.write(
                        (json.dumps(payload, ensure_ascii=False) + "\n").encode()
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
                    for batch in reply.get("batches", []):
                        self.model_batches[batch["size"]] += 1
                        self.batch_tokens += batch["tokens"]
                        self.padded_tokens += batch.get("padded_size", batch["size"]) * batch["length"]
                    self.runner_metrics = reply.get("runner_metrics", {})
                    if "error" in reply:
                        replies = [reply] * len(jobs)
                    elif capacity == 1:
                        replies = [reply]
                    else:
                        if reply.get("version") != 1 or len(reply["replies"]) != len(jobs):
                            raise EngineError("worker_batch_protocol_mismatch")
                        by_id = {item["id"]: item for item in reply["replies"]}
                        if set(by_id) != set(range(len(jobs))):
                            raise EngineError("worker_batch_ids_mismatch")
                        replies = [by_id[i] for i in range(len(jobs))]
                    for job, response, queue_ms in zip(jobs, replies, queue_times):
                        if job.future.done() or monotonic() >= job.deadline:
                            self.counters["discarded"] += 1
                            if not job.future.done():
                                job.future.set_exception(EngineError("deadline_exceeded", 504))
                        elif "error" in response:
                            self.counters["failed"] += 1
                            job.future.set_exception(EngineError(response["error"], response.get("status", 500)))
                        else:
                            result = response["result"]
                            result.setdefault("timings", {})["queue_ms"] = queue_ms
                            for key, values in self.stage_times.items():
                                if key in result["timings"]:
                                    values.append(result["timings"][key])
                            job.future.set_result(result)
                            self.counters["completed"] += 1
                finally:
                    for job in self.active or []:
                        self.queue.task_done()
                    # 异常路径必须保留 active，供统一失败处理完成 Future。
                    if all(job.future.done() for job in self.active or []):
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
            "active": len(self.active or []),
            "counters": dict(self.counters),
            "latency_ms": {
                f"p{p}": values[min(len(values) - 1, int((len(values) - 1) * p / 100))]
                if values
                else None
                for p in (50, 95, 99)
            },
            "latency_window": len(values),
            "batching": {
                "max_batch_size": getattr(self.config, "max_batch_size", 1),
                "batch_wait_ms": getattr(self.config, "batch_wait_ms", 0),
                "request_batch_histogram": dict(self.request_batches),
                "model_batch_histogram": dict(self.model_batches),
                "token_fill_ratio": self.batch_tokens / self.padded_tokens if self.padded_tokens else None,
            },
            "runner": self.runner_metrics,
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
