"""Check that a saved adapter produces stable logits after fresh reloads."""

import argparse
import gc
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jev_core import load_model, predict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", default="runs/gate_smoke_50")
    args = parser.parse_args()
    fixture_path = ROOT / "data/generated/fixtures.jsonl"
    rows = [json.loads(line) for line in fixture_path.read_text().splitlines() if line]
    row = next(row for row in rows if len(row["images"]) == 4)
    scores = []
    for _ in range(2):
        model, processor = load_model(
            str(ROOT / "models/gemma-4-e4b-it-4bit"), str(ROOT / args.adapter)
        )
        result = predict(model, processor, row, root=fixture_path.parent)
        scores.append(result["raw_logits"])
        del model, processor
        gc.collect()
    difference = max(abs(a - b) for a, b in zip(*scores))
    if difference > 1e-4:
        raise AssertionError(f"adapter reload changed logits: {difference}")
    print(
        json.dumps(
            {"max_logit_abs_diff": difference, "tolerance": 1e-4, "image_count": 4}
        )
    )


if __name__ == "__main__":
    main()
