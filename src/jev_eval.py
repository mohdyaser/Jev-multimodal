"""Evaluate option-logit decision records and fit one calibration temperature.

Input is JSONL, one record per example::

    {"id":"ex-1", "split":"calibration", "image_count":0,
     "output_type":"choice", "source_kind":"synthetic",
     "target_id":"B", "option_ids":["A","B"], "raw_logits":[0.2,1.3],
     "level_values":{"low":0,"high":1}}

``level_values`` is optional and is used for score MAE. It may be a mapping
from option ID to a numeric value, or a list aligned with ``option_ids``.
Temperature is fit using only rows whose split is ``calibration``. By default,
metrics are reported on ``test`` only after the fit is complete.

This module intentionally uses only the Python standard library.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

DEFAULT_TEMPERATURE_MIN = 0.05
DEFAULT_TEMPERATURE_MAX = 20.0


class EvaluationError(ValueError):
    """Raised when records do not satisfy the evaluation contract."""


def _is_finite_number(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _logsumexp(values: Sequence[float]) -> float:
    peak = max(values)
    return peak + math.log(sum(math.exp(value - peak) for value in values))


def _softmax(logits: Sequence[float], temperature: float) -> list[float]:
    if not _is_finite_number(temperature) or temperature <= 0:
        raise EvaluationError("temperature must be a finite positive number")
    scaled = [float(value) / temperature for value in logits]
    normalizer = _logsumexp(scaled)
    return [math.exp(value - normalizer) for value in scaled]


def _row_nll(row: Mapping[str, Any], temperature: float) -> float:
    logits = row["raw_logits"]
    target_index = row["option_ids"].index(row["target_id"])
    scaled = [float(value) / temperature for value in logits]
    return _logsumexp(scaled) - scaled[target_index]


def validate_record(record: Any, line_number: int | None = None) -> dict[str, Any]:
    """Validate and normalize one input row, with a useful location in errors."""
    where = f"line {line_number}: " if line_number is not None else ""
    if not isinstance(record, dict):
        raise EvaluationError(f"{where}record must be a JSON object")
    required = (
        "id",
        "split",
        "image_count",
        "output_type",
        "source_kind",
        "target_id",
        "option_ids",
        "raw_logits",
    )
    missing = [field for field in required if field not in record]
    if missing:
        raise EvaluationError(f"{where}missing required fields: {', '.join(missing)}")
    for field in ("id", "split", "output_type", "source_kind", "target_id"):
        if not isinstance(record[field], str) or not record[field]:
            raise EvaluationError(f"{where}{field} must be a non-empty string")
    image_count = record["image_count"]
    if (
        isinstance(image_count, bool)
        or not isinstance(image_count, int)
        or not 0 <= image_count <= 4
    ):
        raise EvaluationError(f"{where}image_count must be an integer from 0 through 4")
    if record["output_type"] not in {"noul", "choice", "score"}:
        raise EvaluationError(f"{where}output_type must be noul, choice, or score")

    option_ids = record["option_ids"]
    logits = record["raw_logits"]
    if not isinstance(option_ids, list) or not 2 <= len(option_ids) <= 16:
        raise EvaluationError(f"{where}option_ids must be a list of 2 through 16 IDs")
    if any(not isinstance(option_id, str) or not option_id for option_id in option_ids):
        raise EvaluationError(f"{where}option_ids must contain non-empty strings")
    if len(set(option_ids)) != len(option_ids):
        raise EvaluationError(f"{where}option_ids must be unique")
    if record["output_type"] == "noul" and len(option_ids) != 2:
        raise EvaluationError(f"{where}noul records must have exactly two options")
    if record["target_id"] not in option_ids:
        raise EvaluationError(f"{where}target_id is not present in option_ids")
    if not isinstance(logits, list) or len(logits) != len(option_ids):
        raise EvaluationError(f"{where}raw_logits must match the length of option_ids")
    for index, logit in enumerate(logits):
        if not _is_finite_number(logit):
            raise EvaluationError(f"{where}raw_logits[{index}] must be a finite number")

    normalized = dict(record)
    normalized["image_count"] = image_count
    normalized["option_ids"] = list(option_ids)
    normalized["raw_logits"] = [float(value) for value in logits]
    if "level_values" in record and record["level_values"] is not None:
        values = record["level_values"]
        if isinstance(values, list):
            if len(values) != len(option_ids):
                raise EvaluationError(
                    f"{where}level_values list must match option_ids length"
                )
            mapped = dict(zip(option_ids, values))
        elif isinstance(values, dict):
            mapped = values
        else:
            raise EvaluationError(
                f"{where}level_values must be an object or aligned list"
            )
        if set(mapped) != set(option_ids):
            raise EvaluationError(
                f"{where}level_values must provide exactly one value per option"
            )
        if any(not _is_finite_number(mapped[option_id]) for option_id in option_ids):
            raise EvaluationError(
                f"{where}level_values must contain finite numeric values"
            )
        normalized["level_values"] = {
            option_id: float(mapped[option_id]) for option_id in option_ids
        }
    return normalized


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load and validate JSONL records; reject duplicate IDs."""
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    try:
        with Path(path).open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as error:
                    raise EvaluationError(
                        f"line {line_number}: invalid JSON: {error.msg}"
                    ) from error
                row = validate_record(raw, line_number)
                if row["id"] in seen:
                    raise EvaluationError(
                        f"line {line_number}: duplicate id {row['id']!r}"
                    )
                seen.add(row["id"])
                records.append(row)
    except OSError as error:
        raise EvaluationError(f"cannot read {path}: {error}") from error
    if not records:
        raise EvaluationError("input contains no records")
    return records


def fit_temperature(
    calibration_records: Sequence[Mapping[str, Any]],
    temperature_min: float = DEFAULT_TEMPERATURE_MIN,
    temperature_max: float = DEFAULT_TEMPERATURE_MAX,
) -> float:
    """Fit a single temperature by minimizing mean calibration NLL.

    Searches uniformly in log-temperature, then refines around the best point.
    The bounded search is deterministic and needs no numerical dependencies.
    """
    if not calibration_records:
        raise EvaluationError("no calibration records found; cannot fit temperature")
    if not (_is_finite_number(temperature_min) and _is_finite_number(temperature_max)):
        raise EvaluationError("temperature bounds must be finite")
    if temperature_min <= 0 or temperature_max <= temperature_min:
        raise EvaluationError("temperature bounds must satisfy 0 < min < max")

    lower, upper = math.log(float(temperature_min)), math.log(float(temperature_max))

    def objective(log_temperature: float) -> float:
        temperature = math.exp(log_temperature)
        return sum(_row_nll(row, temperature) for row in calibration_records) / len(
            calibration_records
        )

    # Coarse global sweep avoids relying on a unimodal local initialization.
    count = 401
    step = (upper - lower) / (count - 1)
    values = [
        (objective(lower + index * step), lower + index * step)
        for index in range(count)
    ]
    _, best = min(values)
    left, right = max(lower, best - step), min(upper, best + step)
    # Golden-section refinement within the neighboring grid interval.
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    x1, x2 = right - ratio * (right - left), left + ratio * (right - left)
    f1, f2 = objective(x1), objective(x2)
    for _ in range(100):
        if right - left < 1e-12:
            break
        if f1 <= f2:
            right, x2, f2 = x2, x1, f1
            x1 = right - ratio * (right - left)
            f1 = objective(x1)
        else:
            left, x1, f1 = x1, x2, f2
            x2 = left + ratio * (right - left)
            f2 = objective(x2)
    candidates = [
        (objective(lower), lower),
        (objective(upper), upper),
        (objective(best), best),
        (objective((left + right) / 2), (left + right) / 2),
    ]
    return math.exp(min(candidates)[1])


def _metrics(
    records: Sequence[Mapping[str, Any]], temperature: float
) -> dict[str, Any]:
    if not records:
        return {
            "n": 0,
            "accuracy": None,
            "multiclass_nll": None,
            "brier": None,
            "score_mae": None,
            "score_mae_n": 0,
        }
    correct = 0
    nll_total = 0.0
    brier_total = 0.0
    score_error_total = 0.0
    score_count = 0
    for row in records:
        probabilities = _softmax(row["raw_logits"], temperature)
        target_index = row["option_ids"].index(row["target_id"])
        predicted_index = max(range(len(probabilities)), key=probabilities.__getitem__)
        correct += predicted_index == target_index
        nll_total += -math.log(max(probabilities[target_index], 1e-300))
        brier_total += sum(
            (probability - (1.0 if index == target_index else 0.0)) ** 2
            for index, probability in enumerate(probabilities)
        )
        if row.get("level_values") is not None:
            level_values = row["level_values"]
            expected_value = sum(
                probability * level_values[option_id]
                for probability, option_id in zip(probabilities, row["option_ids"])
            )
            target_value = level_values[row["target_id"]]
            score_error_total += abs(expected_value - target_value)
            score_count += 1
    return {
        "n": len(records),
        "accuracy": correct / len(records),
        "multiclass_nll": nll_total / len(records),
        # Multiclass Brier uses the sum of squared class errors per example.
        "brier": brier_total / len(records),
        "score_mae": score_error_total / score_count if score_count else None,
        "score_mae_n": score_count,
    }


def _group_metrics(
    records: Sequence[Mapping[str, Any]], temperature: float, key: str
) -> list[dict[str, Any]]:
    groups: dict[Any, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        groups[row[key]].append(row)
    result = []
    for value in sorted(groups, key=lambda item: (str(type(item)), item)):
        rows = groups[value]
        result.append(
            {
                "value": value,
                "raw": _metrics(rows, 1.0),
                "calibrated": _metrics(rows, temperature),
            }
        )
    return result


def evaluate(
    records: Sequence[Mapping[str, Any]],
    eval_split: str = "test",
    temperature_min: float = DEFAULT_TEMPERATURE_MIN,
    temperature_max: float = DEFAULT_TEMPERATURE_MAX,
) -> dict[str, Any]:
    """Fit on calibration rows and report raw/calibrated metrics for eval_split."""
    if not eval_split:
        raise EvaluationError("eval_split must be non-empty")
    records = [validate_record(dict(row)) for row in records]
    names = [row["id"] for row in records]
    if len(names) != len(set(names)):
        raise EvaluationError("record IDs must be unique")
    model_names_present = {
        row["model_name"] for row in records if isinstance(row.get("model_name"), str)
    }
    if len(model_names_present) > 1:
        raise EvaluationError("input must contain predictions from only one model")
    calibration = [row for row in records if row["split"] == "calibration"]
    evaluation = [row for row in records if row["split"] == eval_split]
    if not calibration:
        raise EvaluationError(
            "no calibration records found; calibration split is required"
        )
    if not evaluation:
        raise EvaluationError(f"no records found for evaluation split {eval_split!r}")
    temperature = fit_temperature(calibration, temperature_min, temperature_max)
    calibration_nll_before = sum(_row_nll(row, 1.0) for row in calibration) / len(
        calibration
    )
    calibration_nll_after = sum(
        _row_nll(row, temperature) for row in calibration
    ) / len(calibration)
    model_names = sorted(
        {row["model_name"] for row in records if isinstance(row.get("model_name"), str)}
    )
    output: dict[str, Any] = {
        "model_names": model_names,
        "temperature": {
            "value": temperature,
            "fit_split": "calibration",
            "fit_n": len(calibration),
            "fit_nll_before": calibration_nll_before,
            "fit_nll_after": calibration_nll_after,
            "bounds": [temperature_min, temperature_max],
        },
        "evaluation": {
            "split": eval_split,
            "n": len(evaluation),
            "raw": _metrics(evaluation, 1.0),
            "calibrated": _metrics(evaluation, temperature),
            "groups": {
                "image_count": _group_metrics(evaluation, temperature, "image_count"),
                "output_type": _group_metrics(evaluation, temperature, "output_type"),
                "source_kind": _group_metrics(evaluation, temperature, "source_kind"),
            },
        },
    }
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_jsonl", help="records JSONL containing calibration and evaluation rows"
    )
    parser.add_argument(
        "--eval-split", default="test", help="split to report (default: test)"
    )
    parser.add_argument(
        "--temperature-min", type=float, default=DEFAULT_TEMPERATURE_MIN
    )
    parser.add_argument(
        "--temperature-max", type=float, default=DEFAULT_TEMPERATURE_MAX
    )
    parser.add_argument(
        "--output", "-o", help="write report JSON to this path (default: stdout)"
    )
    args = parser.parse_args(argv)
    try:
        report = evaluate(
            load_jsonl(args.input_jsonl),
            args.eval_split,
            args.temperature_min,
            args.temperature_max,
        )
    except EvaluationError as error:
        print(f"jev_eval: {error}", file=sys.stderr)
        return 2
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        try:
            Path(args.output).write_text(rendered, encoding="utf-8")
        except OSError as error:
            print(f"jev_eval: cannot write {args.output}: {error}", file=sys.stderr)
            return 2
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
