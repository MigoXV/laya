"""固定抽样公开评测数据；不安装 datasets、不修改项目依赖。"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import time

import httpx
import typer


HERE = Path(__file__).resolve().parent
app = typer.Typer()
LABELS = ["entailment", "neutral", "contradiction"]


def get(client, url, **kwargs):
    for attempt in range(3):
        try:
            response = client.get(url, **kwargs)
            response.raise_for_status()
            return response
        except (httpx.HTTPError, ValueError):
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def request_for(row, language):
    if language == "zh":
        instructions = "仅根据前提判断假设：是可以推出、无法确定，还是与前提矛盾？"
        criteria = {"entailment": "前提能够推出假设为真", "neutral": "前提不足以确定假设真假",
                    "contradiction": "假设与前提矛盾，不能同时为真"}
    else:
        instructions = "Using only the premise, is the hypothesis entailed, undetermined, or contradicted?"
        criteria = {"entailment": "The premise implies that the hypothesis is true",
                    "neutral": "The premise does not determine whether the hypothesis is true",
                    "contradiction": "The hypothesis contradicts the premise"}
    return {"state": {"premise": row["premise"], "hypothesis": row["hypothesis"]},
            "questions": {"relation": {"type": "choice", "instructions": instructions,
                                        "criteria": criteria}}}


@app.command()
def main(output: Path = typer.Option(HERE / "data"), samples: int = typer.Option(200, min=10, max=500),
         seed: int = typer.Option(42)):
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(), "seed": seed, "sources": []}
    with httpx.Client(timeout=40, follow_redirects=True) as client:
        # shell 外网需要环境代理；只访问公开数据，不需要 token。
        commit_response = client.get("https://api.github.com/repos/NandhaKishorM/laya/commits/main")
        gh = commit_response.json()["sha"] if commit_response.status_code == 200 else "main"
        url = f"https://raw.githubusercontent.com/NandhaKishorM/laya/{gh}/research/evals/zh_decision_bench.jsonl"
        response = get(client, url)
        business_path = output / "zh_decision_bench.jsonl"
        business_path.write_bytes(response.content)
        business = [json.loads(line) for line in response.text.splitlines() if line and not line.startswith("#")]
        manifest["sources"].append({"dataset": "zh-decision-bench", "file": business_path.name,
            "url": url, "upstream": "https://github.com/CodyQin/zh-decision-bench", "revision": gh,
            "revision_note": "source pinned by SHA256; GitHub main resolved when API is available",
            "license": "CC BY 4.0", "sha256": hashlib.sha256(response.content).hexdigest(),
            "items": len(business), "questions": sum(len(row["questions"]) for row in business),
            "composition": "179 MASSIVE zh-CN dev-derived items + 40 human-adjudicated synthetic items"})
        info_url = "https://huggingface.co/api/datasets/facebook/xnli"
        revision = get(client, info_url).json()["sha"]
        rows_url = "https://datasets-server.huggingface.co/rows"
        common = {"dataset": "facebook/xnli", "split": "test"}
        first = get(client, rows_url, params={**common, "config": "en", "offset": 0, "length": 1}).json()
        total = first["num_rows_total"]
        assert next(f["type"]["names"] for f in first["features"] if f["name"] == "label") == LABELS
        # 随机选五个完整 100 行块，再从中无放回抽样；两种语言共用同一批索引。
        blocks = sorted(random.Random(seed).sample(range(total // 100), 5))
        tasks = [(language, block * 100) for language in ("zh", "en") for block in blocks]

        def fetch(task):
            language, offset = task
            data = get(client, rows_url, params={**common, "config": language,
                       "offset": offset, "length": 100}).json()
            assert data["num_rows_total"] == total
            assert not any(row["truncated_cells"] for row in data["rows"])
            return language, data["rows"]

        pools = {language: {} for language in ("zh", "en")}
        with ThreadPoolExecutor(max_workers=4) as workers:
            for language, rows in workers.map(fetch, tasks):
                pools[language].update({row["row_idx"]: row["row"] for row in rows})
        assert pools["zh"].keys() == pools["en"].keys()
        selected = sorted(random.Random(seed).sample(sorted(pools["en"]), samples))
        for idx in selected:
            assert pools["zh"][idx]["label"] == pools["en"][idx]["label"]
        assert get(client, info_url).json()["sha"] == revision
        for language in ("zh", "en"):
            path = output / f"xnli-{language}.jsonl"
            records = []
            for idx in selected:
                source = pools[language][idx]
                records.append({**request_for(source, language), "expected": {"relation": LABELS[source["label"]]},
                    "language": language, "tags": ["xnli", f"xnli-{language}", f"id:xnli_{idx}"],
                    "source_row_index": idx})
            payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records)
            path.write_text(payload)
            manifest["sources"].append({"dataset": "facebook/xnli", "config": language, "split": "test",
                "url": "https://huggingface.co/datasets/facebook/xnli", "revision": revision,
                "file": path.name, "sha256": hashlib.sha256(payload.encode()).hexdigest(),
                "items": samples, "total_test_rows": total, "candidate_blocks": blocks,
                "selected_indices": selected, "license": "See upstream XNLI dataset card and source license",
                "sampling": f"5 random full 100-row blocks, then random sample without replacement; seed={seed}",
                "translation_pairing": "Chinese/English share row indices and gold labels"})
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"prepared {len(business)} Chinese decision cases + {samples} zh/{samples} en XNLI cases in {output}")


if __name__ == "__main__":
    app()
