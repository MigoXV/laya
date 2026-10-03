"""Laya HTTP 与 CLI 输入契约。"""
from pydantic import Field, model_validator
from laya.inferencers.contracts import InferenceQuestion, InferenceRequest

class Question(InferenceQuestion):
    """原有 HTTP/CLI 协议保留必填且非空的指令。"""

    instructions: str = Field(min_length=1, max_length=8192)


class DecisionRequest(InferenceRequest):
    questions: dict[str, Question] = Field(min_length=1, max_length=16)

    @model_validator(mode="after")
    def question_ids(self):
        if any(not key or len(key) > 128 for key in self.questions):
            raise ValueError("问题 ID 必须为 1–128 字符")
        return self
