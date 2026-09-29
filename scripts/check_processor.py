"""Check 0–4 ordered-image preprocessing against the pinned Gemma tokenizer."""

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jev_core import prepare, validate_request


def main() -> None:
    from mlx_vlm.models.gemma4.processing_gemma4 import Gemma4Processor

    model_dir = ROOT / "models/gemma-4-e4b-it-4bit"
    fixture_path = ROOT / "data/generated/fixtures.jsonl"
    processor = Gemma4Processor.from_pretrained(model_dir)
    rows = [json.loads(line) for line in fixture_path.read_text().splitlines() if line]
    results = []
    for count in range(5):
        sample = next(row for row in rows if len(row["images"]) == count)
        request = validate_request(sample, fixture_path.parent)
        prompt, prepared = prepare(processor, request)
        if prompt.count("<|image|>") != count:
            raise AssertionError("rendered prompt lost or added an image")
        for index in range(1, count + 1):
            if prompt.index(f"image_{index}:") > prompt.index(
                "<|image|>", prompt.index(f"image_{index}:")
            ):
                raise AssertionError("image label does not precede its placeholder")
        result = {
            "count": count,
            "id": sample["id"],
            "sequence_length": prepared["sequence_length"],
            "image_tokens": prepared["image_tokens"],
            "pixel_shape": list(prepared["inputs"]["pixel_values"].shape)
            if count
            else None,
        }
        results.append(result)
    import numpy as np
    from PIL import Image

    with tempfile.TemporaryDirectory() as tmp:
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
        paths = []
        for index, color in enumerate(colors, 1):
            path = Path(tmp) / f"order_{index}.png"
            Image.new("RGB", (224, 224), color).save(path)
            paths.append(str(path))
        request = validate_request(
            {
                "question": "Which image is blue?",
                "images": paths,
                "output_type": "choice",
                "options": [
                    {"id": f"image_{i}", "description": f"image_{i}"}
                    for i in range(1, 5)
                ],
            }
        )
        _, prepared = prepare(processor, request)
        pixels = np.asarray(prepared["inputs"]["pixel_values"])
        observed = [
            tuple(int(v) for v in (pixels[i].mean(axis=(1, 2)) * 255).round())
            for i in range(4)
        ]
        if observed != colors:
            raise AssertionError(f"image order changed: {observed} != {colors}")
        results.append({"ordered_image_pixel_colors": observed})
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
