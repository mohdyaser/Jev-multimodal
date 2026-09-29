"""Exercise the public request and typed-result contract without loading a model."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jev_core import result_from_logits, validate_request


def main() -> None:
    source = ROOT / "data/generated/fixtures.jsonl"
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    example = next(row for row in rows if len(row["images"]) == 4)
    root = source.parent
    validate_request(example, root)

    cases = {}
    fifth = copy.deepcopy(example)
    fifth["images"].append(fifth["images"][0])
    cases["fifth_image"] = fifth
    missing = copy.deepcopy(example)
    missing["question"] = "What is visible in image_5?"
    cases["missing_reference"] = missing
    absent = copy.deepcopy(example)
    absent["images"][0] = "images/not_present.png"
    cases["missing_file"] = absent
    many = copy.deepcopy(example)
    many["options"] = [{"id": str(i), "description": f"Choice {i}"} for i in range(17)]
    cases["seventeen_options"] = many
    duplicate = copy.deepcopy(example)
    duplicate["options"][1]["id"] = duplicate["options"][0]["id"]
    cases["duplicate_option"] = duplicate

    for label, request in cases.items():
        try:
            validate_request(request, root)
        except (TypeError, ValueError):
            continue
        raise AssertionError(f"invalid input was accepted: {label}")

    validated = validate_request(example, root)
    logits = [float(i) for i in range(len(validated["options"]))]
    result = result_from_logits(validated, logits)
    probabilities = result["probabilities"]
    if (
        set(probabilities) != {option["id"] for option in validated["options"]}
        or abs(sum(probabilities.values()) - 1.0) > 1e-12
        or result["selected_id"] not in probabilities
    ):
        raise AssertionError("typed output failed schema or normalization check")
    print(
        json.dumps(
            {
                "accepted_valid_four_image_request": True,
                "rejected_cases": sorted(cases),
                "probability_sum": sum(probabilities.values()),
            }
        )
    )


if __name__ == "__main__":
    main()
