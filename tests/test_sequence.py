"""单次分词路径必须保持参考输入，并在分词器配置变化后仍拒绝截断。"""

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from laya.api.contracts import Question
from tests.support.sequence import build_sequence
from laya.inferencers.preprocessing import checked_sequence, InputTooLong


@pytest.fixture
def tokenizer():
    tokens = ["[UNK]", "[CLS]", "[SEP]", "[MASK]", "[PAD]", "a", "b", "choice", "score", "noul"]
    backend = Tokenizer(models.WordLevel(dict(zip(tokens, range(len(tokens)))), unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]",
                                  cls_token="[CLS]", sep_token="[SEP]",
                                  mask_token="[MASK]", pad_token="[PAD]")


@pytest.mark.parametrize("state", ["小李负责测试。", {"人": "小王", "tasks": ["test", "publish"]}, ["a", "b"]])
@pytest.mark.parametrize("question", [
    {"type": "choice", "instructions": "谁负责测试？", "criteria": {"a": "测试", "b": None}},
    {"type": "score", "instructions": "How important?", "criteria": ["low", "medium", "high"]},
    {"type": "noul", "instructions": "a handles b?", "criteria": {"true": "yes", "false": "no"}},
])
def test_checked_sequence_matches_original_builder(tokenizer, state, question):
    question = Question.model_validate(question)
    limits = {"max_len": 1024, "head_max_len": 256}
    actual = checked_sequence(tokenizer, state, question, limits)
    expected = build_sequence(tokenizer, state, {"t": question.type,
                              "ins": question.instructions, "crit": question.criteria}, **limits)
    assert actual == expected


def test_prior_tokenizer_truncation_cannot_hide_overflow(tokenizer):
    tokenizer.backend_tokenizer.enable_truncation(max_length=8)
    tokenizer.backend_tokenizer.enable_padding(length=16)
    q = Question(type="choice", instructions="a", criteria=["a", "b"])
    with pytest.raises(InputTooLong, match="state_token_budget_exceeded"):
        checked_sequence(tokenizer, "a " * 100, q, {"max_len": 64, "head_max_len": 64})
    with pytest.raises(InputTooLong, match="question_head_exceeded"):
        checked_sequence(tokenizer, "a", q.model_copy(update={"instructions": "a " * 100}),
                         {"max_len": 1024, "head_max_len": 64})
    with pytest.raises(InputTooLong, match="question_head_exceeded"):
        checked_sequence(tokenizer, "a", q.model_copy(update={"criteria": {"a " * 49: None, "b": None}}),
                         {"max_len": 1024, "head_max_len": 256})
    with pytest.raises(ValueError, match="保留"):
        checked_sequence(tokenizer, {"text": "a [MASK] b"}, q, {"max_len": 1024, "head_max_len": 256})
