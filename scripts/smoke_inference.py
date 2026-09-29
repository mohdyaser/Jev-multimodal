"""Run one frozen text and four-image decision on the local checkpoint."""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jev_core import load_model, predict


def main() -> None:
    import mlx.core as mx

    start = time.perf_counter()
    model, processor = load_model(str(ROOT / "models/gemma-4-e4b-it-4bit"))
    print(json.dumps({"model_load_seconds": time.perf_counter() - start}), flush=True)
    path = ROOT / "data/generated/fixtures.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    for count in (0, 4):
        row = next(item for item in rows if len(item["images"]) == count)
        start = time.perf_counter()
        result = predict(model, processor, row, root=path.parent)
        print(
            json.dumps(
                {
                    "id": row["id"],
                    "image_count": count,
                    "target_id": row["target_id"],
                    "selected_id": result["selected_id"],
                    "probabilities": result["probabilities"],
                    "sequence_length": result["sequence_length"],
                    "image_tokens": result["image_tokens"],
                    "valid_marker_mass": result["valid_marker_mass"],
                    "seconds": time.perf_counter() - start,
                    "peak_memory_gb": mx.get_peak_memory() / 1e9,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
