"""内部推理请求；不依赖外部 SDK。"""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

class InferenceQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["choice", "score", "noul"]
    instructions: str = Field(default="", max_length=8192)
    criteria: dict[str, str | None] | list[str] | None = None

    @model_validator(mode="after")
    def check_criteria(self):
        c = self.criteria
        if self.type == "choice":
            if isinstance(c, list):
                if len(set(c)) != len(c):
                    raise ValueError("重复选项")
                c = self.criteria = dict.fromkeys(c)
            if (
                not isinstance(c, dict)
                or not 2 <= len(c) <= 16
                or any(not k for k in c)
            ):
                raise ValueError("choice 需要 2–16 个非空唯一选项")
        elif self.type == "score":
            if not isinstance(c, list) or not 2 <= len(c) <= 16:
                raise ValueError("score 需要 2–16 个有序等级说明")
        elif c is not None and (not isinstance(c, dict) or set(c) - {"false", "true"}):
            raise ValueError("noul criteria 只允许 false/true")
        return self


class InferenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    state: str | dict | list
    questions: dict[str, InferenceQuestion] = Field(min_length=1, max_length=16)
