"""Inspect a small HF Viewer sample of Ai2's Molmo2-SynMultiImageQA.

This is an audit helper only. It does not download image binaries, transform
answers into training labels, or write records to a training dataset.

The dataset card lists ODC-BY and says use is intended for research and
education under Ai2's Responsible Use Guidelines. It also notes that images
were rendered from code generated with Claude Sonnet 4.5 and questions were
generated with GPT-5, with the corresponding provider terms applying. Check
those terms and independently audit labels/images before using any records.

Example:
    python scripts/audit_molmo2_synmultiimageqa.py --configs chart doc table
"""

from __future__ import annotations

import argparse
import json
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

DATASET = "allenai/Molmo2-SynMultiImageQA"
CONFIGS = (
    "chart",
    "chemical",
    "circuit",
    "diagram",
    "doc",
    "graphic",
    "music",
    "table",
)
VIEWER_ROWS_URL = "https://datasets-server.huggingface.co/rows"


def fetch_rows(config: str, split: str, offset: int, length: int) -> list[dict]:
    query = urlencode(
        {
            "dataset": DATASET,
            "config": config,
            "split": split,
            "offset": offset,
            "length": length,
        }
    )
    request = Request(
        f"{VIEWER_ROWS_URL}?{query}",
        headers={"User-Agent": "jev-gemma-dataset-audit/0.1"},
    )
    with urlopen(request, timeout=45) as response:
        payload = json.load(response)
    return payload.get("rows", [])


def row_summary(item: dict) -> dict:
    row = item.get("row", {})
    images = row.get("images", [])
    if isinstance(images, str):
        try:
            images = json.loads(images)
        except json.JSONDecodeError:
            images = []
    count = len(images) if isinstance(images, list) else None
    metadata = row.get("metadata", {})
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            metadata = {}
    if count is None and isinstance(metadata, dict):
        count = metadata.get("num_images")

    qa = row.get("qa_pairs", {})
    if isinstance(qa, str):
        try:
            qa = json.loads(qa)
        except json.JSONDecodeError:
            qa = {}
    questions = qa.get("question", []) if isinstance(qa, dict) else []
    answers = qa.get("answer", []) if isinstance(qa, dict) else []
    examples = []
    if isinstance(questions, list) and isinstance(answers, list):
        for question, answer in zip(questions, answers):
            examples.append({"question": question, "answer": answer})
            if len(examples) == 2:
                break

    return {
        "id": row.get("id"),
        "image_count": count,
        "eligible_for_1_to_4_image_scope": isinstance(count, int) and 1 <= count <= 4,
        "question_count": len(questions) if isinstance(questions, list) else None,
        "sample_qa": examples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--configs", nargs="+", default=["chart", "doc", "table"], choices=CONFIGS
    )
    parser.add_argument("--split", default="train", choices=("train", "validation"))
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--length", type=int, default=10, help="Rows per config; capped at 25"
    )
    args = parser.parse_args()

    if args.offset < 0 or not 1 <= args.length <= 25:
        parser.error("--offset must be nonnegative and --length must be in 1..25")

    report = {
        "dataset": DATASET,
        "split": args.split,
        "offset": args.offset,
        "requested_rows_per_config": args.length,
        "configs": {},
        "training_records_written": 0,
    }
    for config in args.configs:
        try:
            rows = fetch_rows(config, args.split, args.offset, args.length)
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            print(f"Could not fetch config={config}: {exc}", file=sys.stderr)
            return 2
        summaries = [row_summary(item) for item in rows]
        report["configs"][config] = {
            "returned_rows": len(summaries),
            "eligible_1_to_4_image_rows_in_sample": sum(
                item["eligible_for_1_to_4_image_scope"] for item in summaries
            ),
            "sample": summaries,
        }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
