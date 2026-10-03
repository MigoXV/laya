"""问题渲染与无截断序列构建；源自 convaiinnovations/laya，见 THIRD_PARTY.md。"""
import json
from typing import Dict, List

def serialize_state(state) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False)


def render_options(q: Dict) -> List[str]:
    """Option texts in label-index order. Noul is always [false, true] so p[1] == noul."""
    t, crit = q["t"], q.get("crit")
    if t == "choice":
        return [k if not v else "%s: %s" % (k, v) for k, v in crit.items()]
    if t == "score":
        return ["level %d: %s" % (i, c) for i, c in enumerate(crit)]
    crit = crit or {}
    return [
        "false: " + (crit.get("false") or "no, the statement does not hold"),
        "true: " + (crit.get("true") or "yes, the statement holds"),
    ]


class InputTooLong(ValueError):
    pass


def checked_sequence(tok, state, question, config):
    """先验证参考构建器不会截断任何指令、选项或原文。"""
    q = {"t": question.type, "ins": question.instructions, "crit": question.criteria}
    mask = tok.mask_token
    state_text = serialize_state(state)
    rendered = render_options(q)
    if (
        mask in state_text
        or mask in question.instructions
        or any(mask in o for o in rendered)
    ):
        raise ValueError("输入包含模型保留的 MASK 标记")
    backend = tok.backend_tokenizer
    backend.no_truncation()
    backend.no_padding()
    texts = [f"{q['t']} question: {q['ins']}", *[" " + o for o in rendered], state_text]
    head, *options, state_ids = [encoding.ids for encoding in backend.encode_batch(
        texts, add_special_tokens=False
    )]
    option_len = sum(1 + len(o) for o in options)
    budget = config["head_max_len"] - option_len
    if any(len(o) > 48 for o in options) or budget < 16 or len(head) > max(8, budget):
        raise InputTooLong("question_head_exceeded")
    if 4 + len(head) + option_len + len(state_ids) > config["max_len"]:
        raise InputTooLong("state_token_budget_exceeded")
    # 上面的预算已证明不需要截断；复用同一批 token，避免参考构建器重复分词。
    ids = [tok.cls_token_id] + head + [tok.sep_token_id]
    markers = []
    for option in options:
        markers.append(len(ids))
        ids.extend([tok.mask_token_id] + option)
    return ids + [tok.sep_token_id] + state_ids + [tok.sep_token_id], markers
