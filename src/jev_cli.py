"""Local prediction and manifest scoring for the Gemma decision prototype."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jev_core import load_model, predict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/gemma-4-e4b-it-4bit")
    parser.add_argument(
        "--adapter",
        help="directory containing adapter_config.json and adapters.safetensors",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--temperature-report",
        help="evaluation report JSON whose fitted temperature is applied",
    )
    parser.add_argument("--input", required=True, help="JSON request or JSONL manifest")
    parser.add_argument("--output", help="write JSON or JSONL instead of stdout")
    parser.add_argument(
        "--jsonl", action="store_true", help="input is a manifest of decisions"
    )
    args = parser.parse_args()

    if args.temperature_report:
        if args.temperature != 1.0:
            parser.error("use either --temperature or --temperature-report")
        report = json.loads(Path(args.temperature_report).read_text(encoding="utf-8"))
        expected_model = "adapter" if args.adapter else "frozen"
        if report.get("model_names") != [expected_model]:
            parser.error(f"temperature report must describe the {expected_model} model")
        args.temperature = float(report["temperature"]["value"])

    input_path = Path(args.input)
    model, processor = load_model(args.model, args.adapter)
    model_name = "adapter" if args.adapter else "frozen"
    if args.jsonl:
        rows = []
        with input_path.open(encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                request = json.loads(line)
                result = predict(
                    model,
                    processor,
                    request,
                    root=input_path.parent,
                    temperature=args.temperature,
                )
                rows.append(
                    {
                        "id": request.get("id"),
                        "split": request.get("split"),
                        "image_count": len(request.get("images", [])),
                        "output_type": request["output_type"],
                        "source_kind": request.get("source_kind", "unknown"),
                        "target_id": request.get("target_id"),
                        "option_ids": result["option_ids"],
                        "raw_logits": result["raw_logits"],
                        "level_values": request.get("level_values"),
                        "model_name": model_name,
                        "selected_id": result["selected_id"],
                        "valid_marker_mass": result["valid_marker_mass"],
                        "sequence_length": result["sequence_length"],
                        "image_tokens": result["image_tokens"],
                        "prompt_sha256": result["prompt_sha256"],
                    }
                )
        serialized = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    else:
        request = json.loads(input_path.read_text(encoding="utf-8"))
        result = predict(
            model,
            processor,
            request,
            root=input_path.parent,
            temperature=args.temperature,
        )
        serialized = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(serialized, encoding="utf-8")
    else:
        print(serialized, end="")


if __name__ == "__main__":
    main()
