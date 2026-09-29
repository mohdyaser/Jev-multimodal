"""Select an adapter checkpoint by completion-only dev cross-entropy.

Only records with ``split == "dev"`` are accepted. The script scores each
``*_adapters.safetensors`` file directly under ``--run-dir`` using the same
``DecisionDataset`` and per-example iterator as training, selects the lowest
mean full-vocabulary completion loss, and writes a loadable adapter directory.

Example::

    .venv/bin/python scripts/select_checkpoint.py \
      --dev data/generated/dev.jsonl --run-dir runs/train
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class SelectionError(ValueError):
    """Raised for invalid dev data, checkpoints, or adapter state."""


def _verify_run_provenance(
    run_dir: Path, base_identity: dict[str, Any]
) -> dict[str, Any]:
    from jev_train import ITERATOR_PROVENANCE, LOSS_PROVENANCE

    invalid_marker = run_dir / "INVALID.txt"
    if invalid_marker.exists():
        details = invalid_marker.read_text(encoding="utf-8").strip()
        raise SelectionError(
            f"refusing run marked INVALID at {invalid_marker}: {details or '(no reason recorded)'}"
        )
    config_path = run_dir / "training_config.json"
    if not config_path.is_file():
        raise SelectionError(f"missing training configuration: {config_path}")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SelectionError(
            f"invalid training configuration {config_path}: {error}"
        ) from error
    if config.get("loss_provenance") != LOSS_PROVENANCE:
        raise SelectionError(
            f"run loss provenance is missing or unsupported; expected {LOSS_PROVENANCE!r}"
        )
    if config.get("iterator_provenance") != ITERATOR_PROVENANCE:
        raise SelectionError(
            f"run iterator provenance is missing or unsupported; expected {ITERATOR_PROVENANCE!r}"
        )
    for key in ("base_model_id", "base_revision", "base_manifest_sha256"):
        if config.get(key) != base_identity.get(key):
            raise SelectionError(
                f"run {key} is missing or does not match the verified local base model"
            )
    return config


def _load_checkpoint_config(run_dir: Path) -> tuple[dict[str, Any], int, int]:
    config_path = run_dir / "adapter_config.json"
    if not config_path.is_file():
        raise SelectionError(f"missing adapter configuration: {config_path}")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        parameters = config["lora_parameters"]
        rank = int(parameters["rank"])
        scale = float(parameters["scale"])
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise SelectionError(
            f"invalid adapter configuration {config_path}: {error}"
        ) from error
    if rank <= 0 or not math.isfinite(scale) or scale <= 0:
        raise SelectionError("adapter rank and scale must be positive")
    alpha = round(rank * scale)
    if not math.isclose(alpha / rank, scale, rel_tol=1e-7, abs_tol=1e-9):
        raise SelectionError("adapter scale cannot be represented by the trainer alpha")
    return config, rank, alpha


def _dev_records(path: Path) -> list[dict[str, Any]]:
    from jev_train import read_records

    records = read_records(path)
    non_dev = [row.get("split") for row in records if row.get("split") != "dev"]
    if non_dev:
        raise SelectionError(
            "dev input must contain only records marked split='dev'; "
            f"found splits including {sorted(set(map(str, non_dev)))}"
        )
    ids = [row.get("id") for row in records]
    if len(ids) != len(set(ids)):
        raise SelectionError("dev record IDs must be unique")
    return records


def _candidates(run_dir: Path) -> list[Path]:
    candidates = sorted(
        path for path in run_dir.glob("*_adapters.safetensors") if path.is_file()
    )
    if not candidates:
        raise SelectionError(
            f"no *_adapters.safetensors checkpoints found in {run_dir}"
        )
    return candidates


def _step_from_name(path: Path) -> int | None:
    match = re.search(r"(?:^|/)(\d+)_adapters\.safetensors$", str(path))
    return int(match.group(1)) if match else None


def _validate_checkpoint_keys(model: Any, checkpoint: Path) -> None:
    """Ensure each file has exactly the expected adapter tensors and shapes."""
    from mlx.utils import tree_flatten
    from safetensors import safe_open

    expected = dict(tree_flatten(model.trainable_parameters()))
    if not expected:
        raise SelectionError("model has no trainable adapter parameters")
    try:
        with safe_open(str(checkpoint), framework="np") as weights:
            actual_keys = set(weights.keys())
            if actual_keys != set(expected):
                missing = sorted(set(expected) - actual_keys)
                extra = sorted(actual_keys - set(expected))
                raise SelectionError(
                    f"{checkpoint.name} adapter tensor keys differ; "
                    f"missing={missing[:8]}, extra={extra[:8]}"
                )
            for key, parameter in expected.items():
                shape = tuple(weights.get_slice(key).get_shape())
                if shape != tuple(parameter.shape):
                    raise SelectionError(
                        f"{checkpoint.name} shape mismatch for {key}: "
                        f"checkpoint={shape}, expected={tuple(parameter.shape)}"
                    )
    except SelectionError:
        raise
    except Exception as error:
        raise SelectionError(
            f"cannot inspect checkpoint {checkpoint}: {error}"
        ) from error


def _score_checkpoint(
    model: Any,
    dataset: Any,
    records: list[dict[str, Any]],
    checkpoint: Path,
    max_seq_length: int,
) -> dict[str, Any]:
    import mlx.core as mx

    from jev_train import decision_loss, iterate_single_example_batches

    _validate_checkpoint_keys(model, checkpoint)
    model.load_weights(str(checkpoint), strict=False)
    model.eval()

    by_image_count: dict[str, list[float]] = defaultdict(list)
    per_example: list[float] = []
    iterator = iterate_single_example_batches(
        dataset, batch_size=1, max_seq_length=max_seq_length, train=False
    )
    for record, batch in zip(records, iterator):
        loss = decision_loss(model, batch)
        mx.eval(loss)
        value = float(loss.item())
        if not math.isfinite(value):
            raise SelectionError(
                f"non-finite dev loss for {record.get('id')} at {checkpoint.name}: {value}"
            )
        per_example.append(value)
        by_image_count[str(len(record["images"]))].append(value)
    if len(per_example) != len(records):
        raise SelectionError(
            f"iterator returned {len(per_example)} examples for {len(records)} dev rows"
        )
    return {
        "checkpoint": checkpoint.name,
        "microstep": _step_from_name(checkpoint),
        "dev_count": len(per_example),
        "mean_completion_ce": sum(per_example) / len(per_example),
        "by_image_count": {
            count: {
                "n": len(values),
                "mean_completion_ce": sum(values) / len(values),
            }
            for count, values in sorted(
                by_image_count.items(), key=lambda item: int(item[0])
            )
        },
        # Retained internally for the reload check; removed from the report.
        "_first_example_loss": per_example[0],
    }


def select_checkpoint(
    *,
    model_path: str,
    dev_path: Path,
    run_dir: Path,
    selected_dir: Path | None = None,
    report_path: Path | None = None,
    max_seq_length: int = 2048,
) -> dict[str, Any]:
    import mlx.core as mx
    from mlx_vlm import load

    from jev_core import load_model, verify_base_model
    from jev_train import DecisionDataset, attach_upper_qo_adapters

    if max_seq_length <= 0:
        raise SelectionError("max_seq_length must be positive")
    run_dir = run_dir.resolve()
    dev_path = dev_path.resolve()
    selected_dir = (selected_dir or run_dir / "selected").resolve()
    report_path = (report_path or run_dir / "selection.json").resolve()
    try:
        base_identity = verify_base_model(model_path)
    except (OSError, RuntimeError) as error:
        raise SelectionError(
            f"local pinned base model verification failed: {error}"
        ) from error
    training_config = _verify_run_provenance(run_dir, base_identity)
    candidates = _candidates(run_dir)
    adapter_config, rank, alpha = _load_checkpoint_config(run_dir)
    records = _dev_records(dev_path)

    # Fail early if a similarly named but unsupported LoRA layout was trained.
    if adapter_config.get("fine_tune_type", "lora") != "lora":
        raise SelectionError("only LoRA adapter checkpoints are supported")

    model, processor = load(model_path)
    keys = attach_upper_qo_adapters(model, rank=rank, alpha=alpha)
    expected_adapter_config = model.config.lora
    architecture_config = {
        key: value
        for key, value in adapter_config.items()
        if key not in {"base_model_id", "base_revision", "base_manifest_sha256"}
    }
    if architecture_config != expected_adapter_config:
        raise SelectionError(
            "run adapter_config.json does not match the selected upper-eight Q/O LoRA architecture"
        )
    dataset = DecisionDataset(records, processor)

    scored: list[dict[str, Any]] = []
    for checkpoint in candidates:
        print(f"Scoring {checkpoint.name} on {len(records)} dev records", flush=True)
        scored.append(
            _score_checkpoint(model, dataset, records, checkpoint, max_seq_length)
        )
        mx.clear_cache()

    best = min(scored, key=lambda row: row["mean_completion_ce"])
    best_source = run_dir / best["checkpoint"]
    selected_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_source, selected_dir / "adapters.safetensors")
    selected_adapter_config = {
        **architecture_config,
        "base_model_id": base_identity["base_model_id"],
        "base_revision": base_identity["base_revision"],
        "base_manifest_sha256": base_identity["base_manifest_sha256"],
    }
    (selected_dir / "adapter_config.json").write_text(
        json.dumps(selected_adapter_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    first_dev = records[0]
    expected_reload_loss = float(best["_first_example_loss"])
    del dataset, processor, model
    gc.collect()
    mx.clear_cache()

    # Reload from the same directory that jev_cli uses, then reproduce the
    # selected checkpoint's first dev example loss as a serialization check.
    reloaded_model, reloaded_processor = load_model(model_path, str(selected_dir))
    reload_dataset = DecisionDataset([first_dev], reloaded_processor)
    reload_batch = reload_dataset[0]
    from jev_train import decision_loss

    reload_loss = decision_loss(reloaded_model, reload_batch)
    mx.eval(reload_loss)
    actual_reload_loss = float(reload_loss.item())
    if not math.isfinite(actual_reload_loss) or not math.isclose(
        expected_reload_loss, actual_reload_loss, rel_tol=1e-5, abs_tol=1e-5
    ):
        raise SelectionError(
            "selected adapter reload check did not reproduce the first dev loss: "
            f"scored={expected_reload_loss}, reloaded={actual_reload_loss}"
        )

    for row in scored:
        row.pop("_first_example_loss", None)
    report = {
        "selection_metric": "mean per-example completion-only full-vocabulary cross-entropy",
        "selection_split": "dev",
        "dev_path": str(dev_path),
        "dev_count": len(records),
        "max_seq_length": max_seq_length,
        "model_path": model_path,
        "training_provenance": {
            "loss": training_config["loss_provenance"],
            "iterator": training_config["iterator_provenance"],
        },
        "base_model": base_identity,
        "adapter_modules": keys,
        "selected_checkpoint": best["checkpoint"],
        "selected_directory": str(selected_dir),
        "selected_mean_completion_ce": best["mean_completion_ce"],
        "reload_check": {
            "passed": True,
            "example_id": first_dev.get("id"),
            "scored_loss": expected_reload_loss,
            "reloaded_loss": actual_reload_loss,
        },
        "checkpoints": scored,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/gemma-4-e4b-it-4bit")
    parser.add_argument("--dev", type=Path, default=Path("data/generated/dev.jsonl"))
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="training directory containing adapter_config.json and periodic checkpoints",
    )
    parser.add_argument(
        "--selected-dir",
        type=Path,
        help="destination adapter directory (default: RUN_DIR/selected)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="selection JSON path (default: RUN_DIR/selection.json)",
    )
    parser.add_argument("--max-seq-length", type=int, default=2048)
    args = parser.parse_args()
    try:
        report = select_checkpoint(
            model_path=args.model,
            dev_path=args.dev,
            run_dir=args.run_dir,
            selected_dir=args.selected_dir,
            report_path=args.report,
            max_seq_length=args.max_seq_length,
        )
    except (SelectionError, ValueError, OSError) as error:
        print(f"select_checkpoint: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "selected_checkpoint": report["selected_checkpoint"],
                "selected_mean_completion_ce": report["selected_mean_completion_ce"],
                "dev_count": report["dev_count"],
                "selected_directory": report["selected_directory"],
                "report": str(
                    (args.report or args.run_dir / "selection.json").resolve()
                ),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
