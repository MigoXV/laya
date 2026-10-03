"""概率与校准；源自 convaiinnovations/laya，见 THIRD_PARTY.md。"""
import math
import numpy as np
from .types import QTYPE_NAMES

def confidence_from_probs(p: np.ndarray, k: int) -> float:
    """Jev-style confidence: 1 - normalized entropy of the answer distribution."""
    if k < 2:
        return 1.0
    p = p[:k]
    ent = -(p * np.log(np.clip(p, 1e-12, 1))).sum()
    return float(1 - ent / math.log(k))


def temp_bucket(qtype: int, k: int) -> str:
    """Key for per-cardinality temperature fitting: a 2-option noul and a 20-option choice need different scaling."""
    size = "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"
    return "%s:%s" % (QTYPE_NAMES[int(qtype)], size)


def answer(question, item, logits, act, calibration):
    k, qt = len(item["markers"]), item["qtype"]
    temperature = calibration["temperature_by_options"].get(
        temp_bucket(qt, k), calibration["temperature"][qt]
    )
    z = logits[:k] / temperature
    probabilities = np.exp(z - z.max())
    probabilities /= probabilities.sum()
    keys = list(question.criteria) if question.type == "choice" else [str(i) for i in range(k)]
    answer = {
        "type": question.type,
        "probabilities": dict(zip(keys, map(float, probabilities))),
        "confidence": confidence_from_probs(probabilities, k),
        "act_probability": float(act[0]),
    }
    if question.type == "choice":
        answer["choice"] = keys[int(probabilities.argmax())]
    elif question.type == "score":
        answer["score"] = float((np.arange(k) * probabilities).sum())
    else:
        answer["noul"] = float(probabilities[1])
    return answer

