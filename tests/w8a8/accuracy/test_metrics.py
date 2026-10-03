"""确认准确率损失、答案翻转和连续评分使用不同指标。"""

import pytest


def test_net_accuracy_change_does_not_hide_label_flips():
    from .evaluate import aggregate

    def classified(index, gold, old, new):
        def prediction(probabilities):
            label = int(probabilities[1] > probabilities[0])
            return {"label": label, "correct": label == gold, "target_index": gold,
                    "probabilities": probabilities, "answer": {}}
        return {"cluster": str(index), "type": "choice", "predictions": {
            "fp16-original": prediction(old), "int8": prediction(new)}}

    rows = [classified(0, 0, [0.9, 0.1], [0.1, 0.9]),
            classified(1, 1, [0.6, 0.4], [0.4, 0.6]),
            classified(2, 0, [0.8, 0.2], [0.7, 0.3])]
    rows.append({"cluster": "score", "type": "score", "predictions": {
        "fp16-original": {"probabilities": [0.1, 0.1, 0.8], "answer": {"score": 1.7}},
        "int8": {"probabilities": [0.2, 0.2, 0.6], "answer": {"score": 1.4}, "score_abs_error": 0.6}}})
    result = aggregate(rows, "int8")
    assert result["classification_n"] == 3
    assert result["accuracy"] == pytest.approx(2 / 3)
    assert result["accuracy_delta_pp_vs_original"] == 0
    assert result["label_flip_rate"] == pytest.approx(2 / 3)
    assert result["original_correct_to_wrong"] == result["original_wrong_to_correct"] == 1
    assert result["score_n"] == 1
    assert result["score_mae"] == pytest.approx(0.6)
    assert result["score_mean_abs_drift"] == pytest.approx(0.3)
    assert result["probability_max_abs"] == pytest.approx(0.8)


def test_gold_and_metadata_are_not_sent_to_model():
    from .evaluate import request_from_case

    case = {"state": "Alice handles testing.", "questions": {"q": {
        "type": "choice", "instructions": "Who handles testing?", "criteria": ["Alice", "Bob"]}},
        "expected": {"q": "Alice"}, "language": "en", "tags": ["id:test"], "source_row_index": 5}
    request = request_from_case(case)
    assert set(request.model_dump()) == {"state", "questions"}
    assert request.state == case["state"]
    assert request.questions["q"].instructions == "Who handles testing?"
