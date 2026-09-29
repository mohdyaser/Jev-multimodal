"""Completion-only QLoRA for Gemma 4 decisions on Apple Silicon."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from jev_core import (
    MARKERS,
    MAX_SEQUENCE,
    build_messages,
    format_prompt,
    marker_ids,
    process_multimodal,
    validate_request,
    verify_base_model,
)

LOSS_PROVENANCE = "jev_train.decision_loss:causal-completion-v1"
ITERATOR_PROVENANCE = "jev_train.iterate_single_example_batches:ordered-image-v1"


def read_records(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(validate_request(json.loads(line), path.parent))
    if not rows:
        raise ValueError(f"no records in {path}")
    return rows


def read_training_records(path: Path) -> list[dict[str, Any]]:
    """Allow only training data or explicit wiring fixtures as optimizer input."""
    rows = read_records(path)
    invalid = sorted(
        {
            str(row.get("split"))
            for row in rows
            if row.get("split") not in {"train", "fixture"}
        }
    )
    if invalid:
        raise ValueError(
            "training dataset may contain only split='train' or split='fixture'; "
            f"found {invalid} in {path}"
        )
    return rows


def attach_upper_qo_adapters(model: Any, rank: int = 8, alpha: int = 16) -> list[str]:
    """Attach LoRA only to q/o projections in the final eight text layers."""
    from mlx import nn
    from mlx_vlm.trainer.lora_layers import LoRALinear
    from mlx_vlm.trainer.utils import freeze_model, set_module_by_name

    freeze_model(model)
    # Gemma's audio tower has direct array parameters that the upstream
    # recursive freeze fallback leaves active. Freeze every module's own leaves.
    for _, module in model.named_modules():
        if not hasattr(module, "_no_grad"):
            module._no_grad = set()
        module.freeze(recurse=False)
    layers = model.language_model.model.layers
    start = len(layers) - 8
    if start < 0:
        raise ValueError("Gemma text stack has fewer than eight layers")
    keys = []
    for layer_idx in range(start, len(layers)):
        for projection in ("q_proj", "o_proj"):
            local_name = f"model.layers.{layer_idx}.self_attn.{projection}"
            submodule = model.language_model.model.layers[layer_idx].self_attn
            original = getattr(submodule, projection)
            if not isinstance(original, (nn.Linear, nn.QuantizedLinear)):
                raise TypeError(
                    f"unsupported projection: {local_name}: {type(original)}"
                )
            adapter = LoRALinear.from_base(
                original, r=rank, scale=alpha / rank, dropout=0.0
            )
            set_module_by_name(model.language_model, local_name, adapter)
            keys.append(f"language_model.{local_name}")
    model.config.lora = {
        "fine_tune_type": "lora",
        "num_layers": -1,
        "lora_parameters": {
            "rank": rank,
            "scale": alpha / rank,
            "dropout": 0.0,
            "keys": keys,
        },
    }
    from mlx.utils import tree_flatten

    trainable = [name for name, _ in tree_flatten(model.trainable_parameters())]
    expected = {f"{key}.{side}" for key in keys for side in ("lora_a", "lora_b")}
    if set(trainable) != expected or len(keys) != 16:
        raise AssertionError(
            f"upper-eight q/o adapter selection failed: keys={len(keys)}, "
            f"missing={sorted(expected - set(trainable))}, "
            f"unexpected={sorted(set(trainable) - expected)[:30]}"
        )
    return keys


class DecisionDataset:
    """Same exact answer boundary for training and next-token inference."""

    def __init__(self, records: list[dict[str, Any]], processor: Any):
        self.records = records
        self.processor = processor
        self.config = {"image_token_id": processor.image_token_id}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import mlx.core as mx

        row = self.records[index]
        prompt = format_prompt(self.processor, row)
        ids = marker_ids(self.processor, prompt, len(row["options"]))
        target_idx = next(
            i
            for i, option in enumerate(row["options"])
            if option["id"] == row["target_id"]
        )
        marker = MARKERS[target_idx]
        messages = build_messages(row) + [{"role": "assistant", "content": marker}]
        full_prompt = self.processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        prefix = process_multimodal(self.processor, row["images"], prompt)
        full = process_multimodal(self.processor, row["images"], full_prompt)
        prefix_ids = prefix["input_ids"][0]
        full_ids = full["input_ids"][0]
        prefix_len = int(prefix_ids.shape[0])
        full_len = int(full_ids.shape[0])
        if full_len > MAX_SEQUENCE:
            raise ValueError(f"{row.get('id')} has {full_len} tokens, above the limit")
        if full_len <= prefix_len or not bool(
            mx.all(full_ids[:prefix_len] == prefix_ids).item()
        ):
            raise ValueError(f"{row.get('id')} completion changed the input prefix")
        if int(full_ids[prefix_len].item()) != ids[target_idx]:
            raise ValueError(f"{row.get('id')} answer marker moved or split")
        completion_mask = mx.zeros_like(full["input_ids"])
        completion_mask[:, prefix_len:] = 1
        return {
            "input_ids": full["input_ids"],
            "attention_mask": full.get(
                "attention_mask", mx.ones_like(full["input_ids"])
            ),
            "pixel_values": full.get("pixel_values"),
            "completion_mask": completion_mask,
            **{
                k: v
                for k, v in full.items()
                if k
                not in {
                    "input_ids",
                    "attention_mask",
                    "pixel_values",
                    "completion_mask",
                }
            },
        }


def check_dataset(
    dataset: DecisionDataset, max_rows: int | None = None
) -> dict[str, Any]:
    """Fail on missing media, completion leaks, or sequence overflow."""
    from collections import Counter

    lengths = Counter()
    max_length = 0
    for index in range(min(len(dataset), max_rows or len(dataset))):
        item = dataset[index]
        count = len(dataset.records[index]["images"])
        lengths[count] += 1
        max_length = max(max_length, int(item["input_ids"].shape[-1]))
    return {
        "checked": sum(lengths.values()),
        "by_image_count": dict(lengths),
        "max_sequence": max_length,
    }


def iterate_single_example_batches(
    dataset: DecisionDataset, batch_size: int, max_seq_length: int, train: bool = False
):
    """Keep Gemma's per-example image tensor [N, C, H, W] intact.

    MLX-VLM 0.7.3 stacks this tensor into [1, N, C, H, W] for microbatch 1,
    which the Gemma vision tower does not accept for N > 1.
    """
    import numpy as np

    if batch_size != 1:
        raise ValueError("this multi-image iterator requires microbatch size 1")
    indices = np.arange(len(dataset))
    while True:
        if train:
            np.random.shuffle(indices)
        for index in indices:
            item = dataset[int(index)]
            if item["input_ids"].shape[-1] > max_seq_length:
                raise ValueError("processed sequence exceeds max_seq_length")
            yield item
        if not train:
            break


def decision_loss(
    model: Any,
    batch: dict[str, Any],
    train_on_completions: bool = True,
    assistant_id: int | None = None,
):
    """Causal Gemma loss on answer tokens only.

    MLX-VLM 0.7.3's generic VLM loss passes its 2D padding mask as Gemma's
    attention mask. That bypasses Gemma's causal mask construction and lets a
    prompt position see future answer tokens. Keep `mask=None` and pass the
    processor's multimodal token types so Gemma builds its own causal mask.
    """
    import mlx.core as mx
    from mlx import nn

    del assistant_id
    if not train_on_completions or "completion_mask" not in batch:
        raise ValueError("decision_loss requires an explicit completion mask")
    full_ids = batch["input_ids"]
    inputs = full_ids[:, :-1]
    labels = full_ids[:, 1:]
    extras = {
        key: value
        for key, value in batch.items()
        if key not in {"input_ids", "attention_mask", "completion_mask", "pixel_values"}
    }
    for key in ("mm_token_type_ids", "token_type_ids"):
        if key in extras and getattr(extras[key], "ndim", 0) == 2:
            extras[key] = extras[key][:, :-1]
    output = model(inputs, batch["pixel_values"], None, **extras)
    logits = output.logits.astype(mx.float32)
    if logits.shape[:2] != labels.shape:
        raise ValueError("model logits and next-token labels are misaligned")
    loss_mask = (
        batch["completion_mask"][:, 1:] * batch["attention_mask"][:, 1:]
    ).astype(mx.float32)
    ce = nn.losses.cross_entropy(logits, labels)
    return (ce * loss_mask).sum() / mx.maximum(loss_mask.sum(), 1)


def run_training(args: argparse.Namespace) -> None:
    import mlx.core as mx
    import mlx.optimizers as optim
    import numpy as np
    from mlx_vlm import load
    from mlx_vlm.trainer import sft_trainer

    random.seed(args.seed)
    np.random.seed(args.seed)
    mx.random.seed(args.seed)
    records = read_training_records(Path(args.dataset))
    selected_count = 4 if args.four_image_only else args.image_count
    if selected_count is not None:
        records = [row for row in records if len(row["images"]) == selected_count]
        if not records:
            raise ValueError(f"no {selected_count}-image rows")
    base_identity = verify_base_model(args.model)
    model, processor = load(args.model)
    dataset = DecisionDataset(records, processor)
    preflight = check_dataset(dataset, args.preflight_rows)
    print(json.dumps({"preflight": preflight}, sort_keys=True), flush=True)
    keys = attach_upper_qo_adapters(model, args.rank, args.alpha)
    active = [name for name, _ in model.named_modules() if name in keys]
    print(
        json.dumps({"adapter_modules": active, "total_modules": len(keys)}), flush=True
    )
    if len(active) != 16:
        raise AssertionError("not all selected adapter modules exist")
    microsteps = args.iters or len(dataset)
    if microsteps % args.gradient_accumulation:
        raise ValueError(
            "microsteps must be divisible by gradient accumulation so no final gradients are discarded"
        )
    optimizer_steps = microsteps // args.gradient_accumulation
    warmup = max(1, int(0.05 * optimizer_steps))
    schedule = optim.join_schedules(
        [
            optim.linear_schedule(
                args.learning_rate / warmup, args.learning_rate, warmup
            ),
            optim.cosine_decay(
                args.learning_rate,
                max(1, optimizer_steps - warmup),
                end=args.learning_rate * 0.1,
            ),
        ],
        [warmup],
    )
    optimizer = optim.AdamW(learning_rate=schedule)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config = {
        "model_path": args.model,
        **base_identity,
        "dataset_path": str(Path(args.dataset).resolve()),
        "dataset_sha256": hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest(),
        "converted_revision": "475b9088d29754a3379866cf5aeb6b41acd313c2",
        "prompt_version": "decision-v1",
        "rank": args.rank,
        "alpha": args.alpha,
        "learning_rate": args.learning_rate,
        "microbatch_size": 1,
        "gradient_accumulation": args.gradient_accumulation,
        "microsteps": microsteps,
        "optimizer_steps": optimizer_steps,
        "seed": args.seed,
        "max_sequence": MAX_SEQUENCE,
        "image_soft_tokens": 140,
        "loss_provenance": LOSS_PROVENANCE,
        "iterator_provenance": ITERATOR_PROVENANCE,
        "adapter_keys": keys,
        "mlx_vlm_version": __import__(
            "importlib.metadata", fromlist=["version"]
        ).version("mlx-vlm"),
    }
    (output / "training_config.json").write_text(json.dumps(config, indent=2))
    training_args = sft_trainer.TrainingArgs(
        batch_size=1,
        iters=microsteps,
        steps_per_report=max(1, args.report_every),
        steps_per_eval=max(1, args.eval_every),
        steps_per_save=max(1, args.save_every),
        max_seq_length=MAX_SEQUENCE,
        adapter_file=str(output / "adapters.safetensors"),
        grad_checkpoint=args.grad_checkpoint,
        learning_rate=args.learning_rate,
        grad_clip=1.0,
        gradient_accumulation_steps=args.gradient_accumulation,
    )
    sft_trainer.iterate_batches = iterate_single_example_batches
    sft_trainer.train(
        model,
        optimizer,
        dataset,
        val_dataset=None,
        args=training_args,
        loss_fn=decision_loss,
        train_on_completions=True,
    )
    print(json.dumps({"peak_memory_gb": mx.get_peak_memory() / 1e9}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/gemma-4-e4b-it-4bit")
    parser.add_argument("--dataset", default="data/generated/fixtures.jsonl")
    parser.add_argument("--output", default="runs/gate")
    parser.add_argument("--iters", type=int)
    parser.add_argument("--four-image-only", action="store_true")
    parser.add_argument("--image-count", type=int, choices=range(5))
    parser.add_argument("--preflight-rows", type=int)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--report-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=400)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--grad-checkpoint", action="store_true")
    args = parser.parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
