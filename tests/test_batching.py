"""批调度的关联、取消、过载与 Worker 故障，使用轻量假 Worker。"""

import asyncio
import sys

import pytest

from laya.engine.core import Engine, EngineError
from tests.test_service import make_engine


WORKER = """import sys,json,time
print(json.dumps({'ready':{'fingerprint':'batch-fake'}}),flush=True)
for line in sys.stdin:
 p=json.loads(line)
 entries=p['items']
 if any(e['payload'].get('crash') for e in entries): sys.exit(7)
 time.sleep(max(e['payload'].get('delay',0) for e in entries))
 replies=[]
 for e in entries:
  value=e['payload']
  reply={'error':'bad_item','status':422} if value.get('invalid') else {'result':{'answers':value,'timings':{}}}
  replies.append({'id':e['id'],**reply})
 print(json.dumps({'version':1,'replies':list(reversed(replies)),
                   'batches':[{'size':len(entries),'length':32,'tokens':20*len(entries)}]}),flush=True)
"""


def batched_engine(**overrides):
    config = make_engine(queue_size=32, max_batch_size=4, batch_wait_ms=10,
                         request_timeout=2).config
    for key, value in overrides.items():
        setattr(config, key, value)
    return Engine(config, [sys.executable, "-u", "-c", WORKER])


async def test_batch_association_and_item_error_isolation():
    engine = batched_engine()
    await engine.start()
    try:
        tasks = [asyncio.create_task(engine.submit({"id": i, "invalid": i == 2})) for i in range(8)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        assert isinstance(results[2], EngineError) and results[2].status == 422
        assert all(result["answers"]["id"] == i for i, result in enumerate(results) if i != 2)
        assert engine.request_batches == {4: 2}
        assert engine.model_batches == {4: 2}
        assert engine.metrics()["batching"]["token_fill_ratio"] == 20 / 32
    finally:
        await engine.close()
    assert engine.queue._unfinished_tasks == 0


async def test_cancellation_in_active_batch_keeps_other_results():
    engine = batched_engine()
    await engine.start()
    try:
        tasks = [asyncio.create_task(engine.submit({"id": i, "delay": 0.08})) for i in range(4)]
        while engine.counters["worker_calls"] == 0:
            await asyncio.sleep(0)
        tasks[1].cancel()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        assert isinstance(results[1], asyncio.CancelledError)
        assert [results[i]["answers"]["id"] for i in (0, 2, 3)] == [0, 2, 3]
        assert engine.counters["discarded"] == 1
        assert (await engine.submit({"id": 9}))["answers"]["id"] == 9
    finally:
        await engine.close()


async def test_batch_worker_crash_fails_all_active_and_queued():
    engine = batched_engine()
    await engine.start()
    try:
        tasks = [asyncio.create_task(engine.submit({"id": i, "crash": i == 0})) for i in range(8)]
        results = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 2)
        assert all(isinstance(result, EngineError) for result in results)
        assert not engine.ready
    finally:
        await engine.close()


async def test_batch_deadline_and_bounded_queue():
    engine = batched_engine(queue_size=4, request_timeout=0.04)
    await engine.start()
    try:
        active = [asyncio.create_task(engine.submit({"id": i, "delay": 0.12})) for i in range(4)]
        while engine.counters["worker_calls"] == 0:
            await asyncio.sleep(0)
        queued = [asyncio.create_task(engine.submit({"id": i + 4})) for i in range(4)]
        await asyncio.sleep(0)
        with pytest.raises(EngineError, match="queue_full"):
            await engine.submit({"id": 99})
        results = await asyncio.gather(*active, *queued, return_exceptions=True)
        assert all(isinstance(result, EngineError) and result.status == 504 for result in results)
        await engine.queue.join()
        assert engine.counters["discarded"] == 4
        assert engine.counters["skipped"] == 4
    finally:
        await engine.close()


async def test_bounded_batch_reply_can_exceed_one_megabyte():
    engine = batched_engine(max_body_bytes=300000)
    await engine.start()
    try:
        payloads = [{"id": i, "data": " " * 280000} for i in range(4)]
        results = await asyncio.gather(*(engine.submit(payload) for payload in payloads))
        assert [result["answers"] for result in results] == payloads
        assert engine.counters["completed"] == 4 and engine.ready
    finally:
        await engine.close()
