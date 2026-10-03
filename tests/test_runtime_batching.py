"""领域组批保证问题顺序、Token 用量及错误归属，不需模型权重。"""

from types import SimpleNamespace

import numpy as np

from laya.api.contracts import DecisionRequest
from laya.inferencers.decision import DecisionInferencer
from laya.inferencers.preprocessing import InputTooLong


def test_runtime_mixed_questions_and_independent_validation():
    inferencer = DecisionInferencer.__new__(DecisionInferencer)
    inferencer.runtime = SimpleNamespace()
    inferencer.runtime.config = SimpleNamespace(max_batch_size=2, max_batch_tokens=1024, runner="eager")
    inferencer.runtime.tok = SimpleNamespace(pad_token_id=0)
    inferencer.runtime.cfg = {"calibration": {"temperature": [1, 1, 1], "temperature_by_options": {}}}
    inferencer.runtime.info = {"runner": "fake"}
    shapes = []

    def prepare(request):
        if request.state == "invalid":
            raise InputTooLong("state_token_budget_exceeded")
        result = []
        for qid, question in request.questions.items():
            length = int(qid)
            result.append((qid, question, {"ids": [1] * length, "markers": [1, 2],
                           "qtype": 0, "target": [0, 0], "label": -1,
                           "episode": 0, "ep_step": 0}))
        return result

    def execute(batch):
        shapes.append(tuple(batch["input_ids"].shape))
        n = len(batch["input_ids"])
        return np.tile([0.0, 1.0], (n, 1)), np.tile([0.75, 0.25], (n, 1))

    inferencer.prepare = prepare
    inferencer.runtime.runner = SimpleNamespace(execute=execute)
    question = {"type": "choice", "instructions": "x", "criteria": ["a", "b"]}
    requests = [DecisionRequest.model_validate({"state": state, "questions": questions})
                for state, questions in [("ok", {"33": question, "8": question}),
                                         ("invalid", {"8": question}),
                                         ("ok", {"600": question, "8": question, "34": question})]]
    results = inferencer.infer_many(requests)
    assert isinstance(results[1], InputTooLong)
    assert list(results[0]["answers"]) == ["33", "8"]
    assert list(results[2]["answers"]) == ["600", "8", "34"]
    assert results[0]["usage"]["input_tokens"] == 41
    assert results[2]["usage"]["input_tokens"] == 642
    assert shapes == [(2, 34), (2, 8), (1, 600)]
    assert all(shape[0] <= 2 and shape[0] * shape[1] <= 1024 for shape in shapes)
    assert results[0]["answers"]["33"]["choice"] == "b"


def test_graph_batches_never_pad_sequence_or_option_dimensions():
    inferencer = DecisionInferencer.__new__(DecisionInferencer)
    inferencer.runtime = SimpleNamespace()
    inferencer.runtime.config = SimpleNamespace(max_batch_size=16, max_batch_tokens=1024, runner="cuda-graph")
    inferencer.runtime.tok = SimpleNamespace(pad_token_id=0)
    inferencer.runtime.cfg = {"calibration": {"temperature": [1, 1, 1], "temperature_by_options": {}}}
    inferencer.runtime.info = {}
    seen = []
    q = DecisionRequest.model_validate({"state": "x", "questions": {"q": {
        "type": "choice", "instructions": "x", "criteria": ["a", "b"]}}})

    def prepare(request):
        length = len(request.state)
        return [("q", request.questions["q"], {"ids": [1] * length, "markers": [1, 2],
                 "qtype": 0, "target": [0, 0], "label": -1, "episode": 0, "ep_step": 0})]

    def execute(batch):
        assert batch["attention_mask"].all()
        seen.append(tuple(batch["input_ids"].shape))
        n = len(batch["input_ids"])
        return np.tile([0.0, 1.0], (n, 1)), np.tile([1.0, 0.0], (n, 1))

    inferencer.prepare = prepare
    inferencer.runtime.runner = SimpleNamespace(execute=execute)
    inferencer.infer_many([q.model_copy(update={"state": "a" * length}) for length in [27, 28, 27]])
    assert seen == [(2, 27), (1, 28)]
    seen.clear()
    inferencer.infer_many([q.model_copy(update={"state": "a" * 100})] * 9)
    assert seen == [(8, 100), (1, 100)]
    assert all(batch["padded_size"] * batch["length"] <= 1024 for batch in inferencer.last_batches)
