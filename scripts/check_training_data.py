"""Verify completion masks and image features on all forty gate fixtures."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jev_train import (
    DecisionDataset,
    check_dataset,
    iterate_single_example_batches,
    read_records,
)


def main() -> None:
    from mlx_vlm.models.gemma4.processing_gemma4 import Gemma4Processor

    processor = Gemma4Processor.from_pretrained(ROOT / "models/gemma-4-e4b-it-4bit")
    records = read_records(ROOT / "data/generated/fixtures.jsonl")
    dataset = DecisionDataset(records, processor)
    print(json.dumps(check_dataset(dataset), indent=2))
    from mlx_vlm.trainer.sft_trainer import iterate_batches

    four_image = DecisionDataset(
        [row for row in records if len(row["images"]) == 4], processor
    )
    batch = next(iterate_batches(four_image, 1, 2048))
    print(
        json.dumps(
            {"upstream_four_image_batch_pixel_shape": list(batch["pixel_values"].shape)}
        )
    )
    corrected = next(iterate_single_example_batches(four_image, 1, 2048))
    shape = list(corrected["pixel_values"].shape)
    if shape[0] != 4 or len(shape) != 4:
        raise AssertionError(f"corrected four-image tensor has wrong shape: {shape}")
    print(json.dumps({"decision_four_image_batch_pixel_shape": shape}))


if __name__ == "__main__":
    main()
