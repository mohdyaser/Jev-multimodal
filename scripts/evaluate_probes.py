"""Score paired later-image, reorder, and irrelevant-change predictions."""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairs", type=Path, default=Path("data/generated/probe_manifest.jsonl")
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    pairs = read_jsonl(args.pairs)
    predictions = {row["id"]: row for row in read_jsonl(args.predictions)}
    if len(predictions) != len(read_jsonl(args.predictions)):
        raise ValueError("prediction IDs must be unique")
    results = []
    groups = defaultdict(lambda: {"passed": 0, "total": 0})
    for pair in pairs:
        before = predictions[pair["before_id"]]
        after = predictions[pair["after_id"]]
        expected_before, expected_after = pair["expected_targets"]
        if (
            before["target_id"] != expected_before
            or after["target_id"] != expected_after
        ):
            raise ValueError(f"probe target mismatch for {pair['pair_id']}")
        passed = (
            before["selected_id"] == expected_before
            and after["selected_id"] == expected_after
        )
        if pair["relation"] in {
            "later_image_changes_count",
            "order_invariant_winner_remapped",
        }:
            passed = passed and before["selected_id"] != after["selected_id"]
        elif pair["relation"] == "nonwinner_change_keeps_answer":
            passed = passed and before["selected_id"] == after["selected_id"]
        else:
            raise ValueError(f"unknown relation: {pair['relation']}")
        group = groups[(pair["relation"], pair["count"])]
        group["total"] += 1
        group["passed"] += int(passed)
        results.append(
            {
                "pair_id": pair["pair_id"],
                "relation": pair["relation"],
                "image_count": pair["count"],
                "expected_targets": pair["expected_targets"],
                "predicted_targets": [before["selected_id"], after["selected_id"]],
                "passed": passed,
            }
        )
    report = {
        "total_pairs": len(results),
        "passed_pairs": sum(row["passed"] for row in results),
        "groups": [
            {"relation": relation, "image_count": count, **totals}
            for (relation, count), totals in sorted(groups.items())
        ],
        "pairs": results,
    }
    output = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
