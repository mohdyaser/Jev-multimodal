"""Verify causal answer logits agree with inference for all four-image fixtures."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jev_core import marker_ids, prepare, raw_logits
from jev_train import DecisionDataset, read_records


def main() -> None:
    import mlx.core as mx
    from mlx_vlm import load
    from mlx_vlm.trainer.utils import apply_lora_layers

    path = ROOT / "data/generated/fixtures.jsonl"
    records = [r for r in read_records(path) if len(r["images"]) == 4]
    model, processor = load(str(ROOT / "models/gemma-4-e4b-it-4bit"))
    model = apply_lora_layers(model, str(ROOT / "runs/gate_causal_one_step"))
    model.eval()
    dataset = DecisionDataset(records, processor)
    maximum_causal_difference = 0.0
    stock_mask_mismatches = 0
    for index, row in enumerate(records):
        item = dataset[index]
        prefix_len = int(mx.sum(item["completion_mask"] == 0).item())
        prompt, prep = prepare(processor, row)
        ids = marker_ids(processor, prompt, len(row["options"]))
        infer_scores, _ = raw_logits(model, prep["inputs"], ids)
        input_ids = item["input_ids"][:, :-1]
        extras = {
            key: value
            for key, value in item.items()
            if key
            not in {"input_ids", "attention_mask", "completion_mask", "pixel_values"}
        }
        for key in ("mm_token_type_ids", "token_type_ids"):
            if key in extras and getattr(extras[key], "ndim", 0) == 2:
                extras[key] = extras[key][:, :-1]
        causal_output = model(input_ids, item["pixel_values"], None, **extras)
        stock_output = model(
            input_ids, item["pixel_values"], item["attention_mask"][:, :-1], **extras
        )
        causal_scores = [
            float(causal_output.logits[0, prefix_len - 1, token].item())
            for token in ids
        ]
        stock_scores = [
            float(stock_output.logits[0, prefix_len - 1, token].item()) for token in ids
        ]
        causal_difference = max(abs(a - b) for a, b in zip(infer_scores, causal_scores))
        stock_difference = max(abs(a - b) for a, b in zip(infer_scores, stock_scores))
        maximum_causal_difference = max(maximum_causal_difference, causal_difference)
        stock_mask_mismatches += stock_difference > 1e-3
    if maximum_causal_difference > 1e-4:
        raise AssertionError(
            f"causal answer logits changed with future tokens: {maximum_causal_difference}"
        )
    print(
        json.dumps(
            {
                "four_image_examples": len(records),
                "max_causal_logit_abs_diff": maximum_causal_difference,
                "stock_mask_mismatch_examples": stock_mask_mismatches,
            }
        )
    )


if __name__ == "__main__":
    main()
