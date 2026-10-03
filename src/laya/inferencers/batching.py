"""CPU 输入张量组装；源自 convaiinnovations/laya，见 THIRD_PARTY.md。"""
import torch

def collate_items(batch, pad_id: int):
    items = [it for group in batch for it in group]
    if not items:
        return None
    n, L = len(items), max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    target = torch.zeros((n, kmax), dtype=torch.float32)
    ep_group = torch.full((n,), -1, dtype=torch.long)
    group_of = {}
    for i, it in enumerate(items):
        ids[i, : len(it["ids"])] = torch.tensor(it["ids"])
        att[i, : len(it["ids"])] = 1
        k = len(it["markers"])
        mpos[i, :k] = torch.tensor(it["markers"])
        mmask[i, :k] = True
        target[i, :k] = torch.tensor(it["target"], dtype=torch.float32)
    # episodes: all prefixes of the same record share a group id (used for TD(lambda) targets)
    for i, it in enumerate(items):
        if it["episode"]:
            ep_group[i] = group_of.setdefault(it.get("rec_uid", -1 - i), len(group_of))
    return {
        "input_ids": ids,
        "attention_mask": att,
        "marker_pos": mpos,
        "marker_mask": mmask,
        "target": target,
        "qtype": torch.tensor([it["qtype"] for it in items]),
        "label": torch.tensor([it["label"] for it in items]),
        "episode": torch.tensor([it["episode"] for it in items], dtype=torch.bool),
        "ep_group": ep_group,
        "ep_step": torch.tensor([it["ep_step"] for it in items]),
        "meta": [
            {k: it[k] for k in it if k not in ("ids", "markers", "target")}
            for it in items
        ],
        "n_tokens": int(att.sum()),
    }
