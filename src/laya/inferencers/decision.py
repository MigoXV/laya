"""领域预处理、问题组批和结果后处理。"""
from time import perf_counter
from laya.configs.settings import Config
from laya.runtime.resources import Runtime
from laya.inferencers.contracts import InferenceRequest
from .preprocessing import checked_sequence
from .postprocessing import answer
from .batching import collate_items
from .types import QTYPES


class DecisionInferencer:
    def __init__(self, config: Config):
        self.runtime = Runtime(config)

    def close(self):
        self.runtime.close()

    def prepare(self, request: InferenceRequest):
        prepared = []
        for qid, question in request.questions.items():
            ids, markers = checked_sequence(
                self.runtime.tok, request.state, question, self.runtime.cfg["input_limits"]
            )
            prepared.append(
                (
                    qid,
                    question,
                    {
                        "ids": ids,
                        "markers": markers,
                        "qtype": QTYPES[question.type],
                        "target": [0.0] * len(markers),
                        "label": -1,
                        "episode": 0,
                        "ep_step": 0,
                        "ep_len": 1,
                        "src": "api",
                    },
                )
            )
        return prepared

    def answer(self, question, item, logits, act):
        return answer(question, item, logits, act, self.runtime.cfg["calibration"])

    def infer_many(self, requests):
        """跨请求组批；预处理错误只拒绝其所属请求，结果始终按输入顺序返回。"""
        started = perf_counter()
        prepared, results = [], []
        for index, request in enumerate(requests):
            prep_start = perf_counter()
            try:
                entries = self.prepare(request)
            except ValueError as exc:
                results.append(exc)
                continue
            results.append({
                "model": self.runtime.info, "answers": {},
                "usage": {"input_tokens": sum(len(item["ids"]) for _, _, item in entries), "output_tokens": 0},
                "timings": {"preprocess_ms": (perf_counter() - prep_start) * 1000},
            })
            prepared.extend((index, qid, question, item) for qid, question, item in entries)
        # 同一长度桶内 oldest-first；所有结果通过 index/qid 恢复关联。
        groups = {}
        for entry in prepared:
            length = len(entry[3]["ids"])
            bucket = ((length, len(entry[3]["markers"])) if self.runtime.config.runner.startswith("cuda-graph")
                      else 1 << (length - 1).bit_length())
            groups.setdefault(bucket, []).append(entry)
        execution = perf_counter()
        batches = []
        for entries in groups.values():
            cursor = 0
            while cursor < len(entries):
                group = []
                max_length = 0
                while cursor < len(entries) and len(group) < self.runtime.config.max_batch_size:
                    candidate = entries[cursor]
                    length = max(max_length, len(candidate[3]["ids"]))
                    size = len(group) + 1
                    padded_size = 1 << (size - 1).bit_length() if self.runtime.config.runner.startswith("cuda-graph") else size
                    if group and length * padded_size > self.runtime.config.max_batch_tokens:
                        break
                    group.append(candidate)
                    max_length = length
                    cursor += 1
                batch = collate_items([[entry[3] for entry in group]], self.runtime.tok.pad_token_id)
                logits, acts = self.runtime.runner.execute(batch)
                padded_size = (1 << (len(group) - 1).bit_length()
                               if self.runtime.config.runner.startswith("cuda-graph") else len(group))
                batches.append({"size": len(group), "padded_size": padded_size, "length": max_length,
                                "tokens": batch["n_tokens"]})
                for row, (index, qid, question, item) in enumerate(group):
                    results[index]["answers"][qid] = self.answer(question, item, logits[row], acts[row])
        execution_ms = (perf_counter() - execution) * 1000
        for request, result in zip(requests, results):
            if isinstance(result, Exception):
                continue
            result["answers"] = {key: result["answers"][key] for key in request.questions}
            result["timings"].update(inference_ms=execution_ms, total_ms=(perf_counter() - started) * 1000)
        self.last_batches = batches
        return results

    def infer(self, request: InferenceRequest):
        result = self.infer_many([request])[0]
        if isinstance(result, Exception):
            raise result
        return result
