"""Validate generated Jev decision data, manifests, images, and split isolation."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

EXPECTED = {
    "train": {0: 500, 1: 500, 2: 600, 3: 200, 4: 200},
    "dev": {0: 20, 1: 20, 2: 20, 3: 20, 4: 20},
    "calibration": {0: 20, 1: 20, 2: 20, 3: 20, 4: 20},
    "test": {0: 40, 1: 40, 2: 40, 3: 40, 4: 40},
}


def read(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def validate(root: Path):
    problems = []
    allrows = {}
    groups = {}
    assets = {}
    ids = set()
    for split, expected in EXPECTED.items():
        p = root / f"{split}.jsonl"
        if not p.exists():
            problems.append(f"missing {p}")
            continue
        rows = read(p)
        allrows[split] = rows
        got = Counter(r.get("num_images") for r in rows)
        for n, want in expected.items():
            if got[n] != want:
                problems.append(
                    f"{split}: expected {want} rows with {n} images, found {got[n]}"
                )
        for r in rows:
            rid = r.get("id")
            if rid in ids:
                problems.append(f"duplicate example id {rid}")
            ids.add(rid)
            if r.get("split") != split:
                problems.append(f"record {rid} stored in wrong split")
            group = r.get("split_group_id")
            if group in groups and groups[group] != split:
                problems.append(f"group crosses split boundary: {group}")
            groups[group] = split
            imgs = r.get("images", [])
            metas = r.get("image_assets", [])
            if len(imgs) != r.get("num_images") or len(metas) != len(imgs):
                problems.append(f"{rid}: image count mismatch")
            if imgs != [i.get("path") for i in metas]:
                problems.append(f"{rid}: ordered paths differ from asset metadata")
            if [i.get("image_id") for i in metas] != r.get("image_order"):
                problems.append(f"{rid}: order list differs from assets")
            if len(imgs) > 4:
                problems.append(f"{rid}: exceeds 4 images")
            for im in metas:
                iid = im.get("image_id")
                if iid in assets and assets[iid] != split:
                    problems.append(f"asset crosses split boundary: {iid}")
                assets[iid] = split
                f = root / im["path"]
                if not f.is_file():
                    problems.append(f"{rid}: missing asset {f}")
                    continue
                if hashlib.sha256(f.read_bytes()).hexdigest() != im.get("sha256"):
                    problems.append(f"{rid}: hash mismatch for {f}")
            opts = r.get("options", [])
            if not 2 <= len(opts) <= 16:
                problems.append(f"{rid}: option count outside 2..16 ({len(opts)})")
            markers = [o.get("marker") for o in opts]
            if len(markers) != len(set(markers)):
                problems.append(f"{rid}: duplicate markers")
            hit = [o for o in opts if o.get("id") == r.get("answer_id")]
            if len(hit) != 1 or hit[0].get("marker") != r.get("answer_marker"):
                problems.append(f"{rid}: answer mapping invalid")
            if r.get("completion") != r.get("answer_marker"):
                problems.append(f"{rid}: completion is not just its marker")
            if r.get("target_id") != r.get("answer_id"):
                problems.append(f"{rid}: target_id mismatch")
            if r.get("output_type") == "noul" and {o.get("id") for o in opts} != {
                "false",
                "true",
            }:
                problems.append(f"{rid}: noul options must have false/true IDs")
            ann = r.get("raw_annotation", {})
            task = ann.get("task")
            vals = ann.get("values", [])
            signed = ann.get("signed", [])
            statuses = ann.get("statuses", [])
            expected_target = None
            if task in ("text_status", "image_status", "image_fields") and statuses:
                expected_target = "true" if statuses[0] == "PASS" else "false"
            elif task in ("text_count", "text_score", "image_score", "joint_count"):
                expected_target = str(sum(bool(x) for x in signed))
            elif task == "joint_match":
                expected_target = "true" if len(set(vals)) <= 1 else "false"
            elif task in ("text_category", "text_category_12", "image_category"):
                expected_target = ann.get("class_id")
            elif task == "joint_rank" and vals:
                expected_target = (
                    f"image_{max(range(len(vals)), key=lambda i: (vals[i], -i)) + 1}"
                )
            elif task == "joint_change" and len(vals) >= 2:
                inc = all(vals[i + 1] > vals[i] for i in range(len(vals) - 1))
                dec = all(vals[i + 1] < vals[i] for i in range(len(vals) - 1))
                if len(vals) == 2:
                    expected_target = (
                        "increased" if inc else "decreased" if dec else "same"
                    )
                else:
                    expected_target = (
                        "increased" if inc else "decreased" if dec else "changed"
                    )
            if expected_target is not None and expected_target != r.get("target_id"):
                problems.append(f"{rid}: deterministic label does not match annotation")
            if r.get("output_type") == "score":
                vals = [o.get("id") for o in opts]
                if vals != [str(i) for i in range(len(vals))]:
                    problems.append(f"{rid}: score levels not in ordinal order")
            if len(opts) >= 8 and r.get("task_family") != "text_category_12":
                problems.append(
                    f"{rid}: high-option slice is not the audited category task"
                )
            if r.get("task_family") == "text_category_12" and len(opts) != 12:
                problems.append(
                    f"{rid}: 12-way support classification must have 12 options"
                )
    train_questions = {r.get("question") for r in allrows.get("train", [])}
    for split, question_expected, visual_expected in (
        ("dev", 50, 40),
        ("calibration", 50, 40),
        ("test", 100, 80),
    ):
        rows = allrows.get(split, [])
        qrows = [r for r in rows if r.get("question_family", "").startswith("heldout_")]
        vrows = [
            r
            for r in rows
            if any(im.get("layout") == "ledger_v1" for im in r.get("image_assets", []))
        ]
        if len(qrows) != question_expected:
            problems.append(
                f"{split}: expected {question_expected} heldout question rows, found {len(qrows)}"
            )
        if len(vrows) != visual_expected:
            problems.append(
                f"{split}: expected {visual_expected} heldout visual rows, found {len(vrows)}"
            )
        for r in qrows:
            if r.get("question") in train_questions:
                problems.append(f"{r.get('id')}: heldout wording appears in train")
            wanted = f"heldout_{split}_{'ledger' if r.get('num_images', 0) > 0 else 'text'}_v2"
            if r.get("template_family") != wanted:
                problems.append(
                    f"{r.get('id')}: template family does not match heldout layout"
                )
            if r.get("num_images", 0) > 0 and any(
                im.get("layout") != "ledger_v1" for im in r.get("image_assets", [])
            ):
                problems.append(
                    f"{r.get('id')}: heldout visual row contains default-layout asset"
                )
    for split, rows in allrows.items():
        high = [r for r in rows if len(r.get("options", [])) >= 8]
        wanted = {"train": 100, "dev": 2, "calibration": 2, "test": 2}.get(split, 0)
        if len(high) != wanted:
            problems.append(
                f"{split}: expected {wanted} high-option records, found {len(high)}"
            )
    for task in ("text_count", "image_fields", "image_score"):
        count = sum(
            r.get("task_family") == task
            for split in ("train", "dev", "calibration")
            for r in allrows.get(split, [])
        )
        if count < 20:
            problems.append(
                f"label audit needs at least 20 non-test {task} rows, found {count}"
            )
    category_targets = Counter(
        r.get("target_id")
        for r in allrows.get("train", [])
        if r.get("task_family") == "text_category_12"
    )
    if len(category_targets) != 12 or (
        category_targets
        and max(category_targets.values()) - min(category_targets.values()) > 1
    ):
        problems.append(
            f"12-way train target balance invalid: {dict(category_targets)}"
        )
    try:
        manifest = json.loads((root / "manifest.json").read_text())
        for split, rows in allrows.items():
            expected = manifest["split_counts"][split]
            actual = {str(n): sum(r["num_images"] == n for r in rows) for n in range(5)}
            if expected != actual:
                problems.append(f"manifest image coverage differs for {split}")
        if manifest.get("external_datasets") != []:
            problems.append("unexpected external dataset listed")
    except Exception as e:  # noqa: BLE001 - collect malformed manifest details
        problems.append(f"manifest invalid: {e}")
    fixture_path = root / "fixtures.jsonl"
    if fixture_path.exists():
        fixtures = read(fixture_path)
        fc = Counter(r.get("num_images") for r in fixtures)
        if len(fixtures) != 40 or any(fc[n] != 8 for n in range(5)):
            problems.append(f"fixture coverage invalid: {dict(fc)}")
        for n, task in (
            (1, "image_category"),
            (2, "joint_rank"),
            (3, "joint_rank"),
            (4, "joint_rank"),
        ):
            targets = {
                r["target_id"]
                for r in fixtures
                if r.get("num_images") == n and r.get("task_family") == task
            }
            if len(targets) < 2:
                problems.append(
                    f"fixtures lack target diversity for {task} with {n} images"
                )
        for n in (2, 3, 4):
            counted = [
                r
                for r in fixtures
                if r.get("num_images") == n and r.get("task_family") == "joint_count"
            ]
            changes = [
                r
                for r in fixtures
                if r.get("num_images") == n and r.get("task_family") == "joint_change"
            ]
            if len({r["target_id"] for r in counted}) < 2:
                problems.append(f"fixtures lack count-label diversity at {n} images")
            if len({tuple(r["raw_annotation"].get("values", [])) for r in changes}) < 2:
                problems.append(f"fixtures repeat change trajectories at {n} images")
        for n in (2, 4):
            matches = {
                r["target_id"]
                for r in fixtures
                if r.get("num_images") == n and r.get("task_family") == "joint_match"
            }
            if len(matches) < 2:
                problems.append(
                    f"fixtures lack joint-match positives/negatives at {n} images"
                )
        score_targets = {
            r["target_id"] for r in fixtures if r.get("task_family") == "image_score"
        }
        if score_targets != {"0", "1"}:
            problems.append("image_score fixtures need both 0 and 1 labels")
        for r in fixtures:
            if len(r.get("images", [])) != r.get("num_images"):
                problems.append(f"fixture {r.get('id')}: bad image count")
            for im in r.get("image_assets", []):
                f = root / im["path"]
                if not f.exists() or hashlib.sha256(
                    f.read_bytes()
                ).hexdigest() != im.get("sha256"):
                    problems.append(
                        f"fixture asset missing or changed: {im.get('path')}"
                    )
    else:
        problems.append("missing fixtures.jsonl")
    try:
        probe_path = root / "probes.jsonl"
        probe_manifest = json.loads((root / "probe_manifest.json").read_text())
        probe_rows = read(probe_path)
        by_id = {r["id"]: r for r in probe_rows}
        pair_manifest_path = root / "probe_manifest.jsonl"
        pair_rows = read(pair_manifest_path)
        if len(probe_rows) != 18 or probe_manifest.get("records") != 18:
            problems.append(f"expected 18 probe rows, found {len(probe_rows)}")
        if hashlib.sha256(probe_path.read_bytes()).hexdigest() != probe_manifest.get(
            "file_sha256"
        ):
            problems.append("probe JSONL hash mismatch")
        if hashlib.sha256(
            pair_manifest_path.read_bytes()
        ).hexdigest() != probe_manifest.get("pair_manifest_sha256"):
            problems.append("probe pair-manifest hash mismatch")
        if pair_rows != probe_manifest.get("pairs"):
            problems.append("pair JSONL manifest differs from summary manifest")
        if probe_manifest.get("split") != "probe":
            problems.append("probe manifest has wrong split")
        for r in probe_rows:
            if (
                r.get("split") != "probe"
                or r.get("source_kind") != "owned_synthetic_probe"
            ):
                problems.append(
                    f"{r.get('id')}: probe row leaks into a production split"
                )
            if r.get("source_group_id") in groups:
                problems.append(
                    f"{r.get('id')}: probe group overlaps production groups"
                )
            imgs = r.get("images", [])
            metas = r.get("image_assets", [])
            ann = r.get("raw_annotation", {})
            if (
                len(imgs) != r.get("num_images")
                or len(metas) != len(imgs)
                or imgs != [m.get("path") for m in metas]
            ):
                problems.append(f"{r.get('id')}: probe image path/count mismatch")
            if not 2 <= len(imgs) <= 4:
                problems.append(f"{r.get('id')}: unsupported probe image count")
            for im in metas:
                f = root / im["path"]
                if not f.is_file() or hashlib.sha256(
                    f.read_bytes()
                ).hexdigest() != im.get("sha256"):
                    problems.append(
                        f"{r.get('id')}: probe asset missing or hash mismatch"
                    )
            opts = r.get("options", [])
            answer = [o for o in opts if o.get("id") == r.get("target_id")]
            if len(answer) != 1 or answer[0].get("marker") != r.get("answer_marker"):
                problems.append(f"{r.get('id')}: probe answer mapping invalid")
            if ann.get("task") == "joint_count":
                if str(sum(bool(x) for x in ann.get("signed", []))) != r.get(
                    "target_id"
                ):
                    problems.append(f"{r.get('id')}: probe count label invalid")
            elif ann.get("task") == "joint_rank":
                vals = ann.get("values", [])
                expected = (
                    f"image_{max(range(len(vals)), key=lambda i: (vals[i], -i)) + 1}"
                    if vals
                    else None
                )
                if expected != r.get("target_id"):
                    problems.append(f"{r.get('id')}: probe rank label invalid")
                pos = (
                    int(r.get("target_id", "").split("_")[-1]) - 1
                    if r.get("target_id")
                    else -1
                )
                if (
                    pos < 0
                    or pos >= len(metas)
                    or ann.get("target_asset_id") != metas[pos].get("image_id")
                ):
                    problems.append(
                        f"{r.get('id')}: probe rank target does not map to winning image"
                    )
        for pair in probe_manifest.get("pairs", []):
            before = by_id.get(pair.get("before_id"))
            after = by_id.get(pair.get("after_id"))
            if before is None or after is None:
                problems.append(f"missing probe pair row: {pair.get('pair_id')}")
                continue
            n = pair.get("count")
            relation = pair.get("relation")
            if before.get("probe_pair_id") != pair.get("pair_id") or after.get(
                "probe_pair_id"
            ) != pair.get("pair_id"):
                problems.append(f"bad pair IDs: {pair.get('pair_id')}")
            if before.get("num_images") != n or after.get("num_images") != n:
                problems.append(f"bad pair image count: {pair.get('pair_id')}")
            bm, am = before["image_assets"], after["image_assets"]
            if relation == "later_image_changes_count":
                if (
                    before["images"][:-1] != after["images"][:-1]
                    or bm[-1]["sha256"] == am[-1]["sha256"]
                ):
                    problems.append(
                        f"later-image pair did not change only the final visual asset: {pair['pair_id']}"
                    )
                if (
                    before["target_id"] != "0"
                    or after["target_id"] != "1"
                    or before["raw_annotation"]["signed"][:-1]
                    != after["raw_annotation"]["signed"][:-1]
                ):
                    problems.append(
                        f"later-image pair label relation invalid: {pair['pair_id']}"
                    )
                if (
                    before["raw_annotation"]["signed"][-1]
                    or not after["raw_annotation"]["signed"][-1]
                ):
                    problems.append(
                        f"later-image signature did not flip: {pair['pair_id']}"
                    )
            elif relation == "order_invariant_winner_remapped":
                if after["images"] != list(reversed(before["images"])):
                    problems.append(
                        f"order pair is not a full reversal: {pair['pair_id']}"
                    )
                if before["raw_annotation"].get("target_asset_id") != after[
                    "raw_annotation"
                ].get("target_asset_id"):
                    problems.append(
                        f"order pair changed winning asset: {pair['pair_id']}"
                    )
                if before["target_id"] == after["target_id"]:
                    problems.append(
                        f"order pair failed to remap target position: {pair['pair_id']}"
                    )
            elif relation == "nonwinner_change_keeps_answer":
                if (
                    before["images"][:-1] != after["images"][:-1]
                    or bm[-1]["sha256"] == am[-1]["sha256"]
                ):
                    problems.append(
                        f"irrelevant-change pair did not isolate the final asset: {pair['pair_id']}"
                    )
                if before["target_id"] != "image_1" or after["target_id"] != "image_1":
                    problems.append(
                        f"irrelevant-change pair target changed: {pair['pair_id']}"
                    )
                if (
                    before["raw_annotation"]["values"][:-1]
                    != after["raw_annotation"]["values"][:-1]
                    or before["raw_annotation"]["values"][-1]
                    == after["raw_annotation"]["values"][-1]
                ):
                    problems.append(
                        f"irrelevant-change values invalid: {pair['pair_id']}"
                    )
            else:
                problems.append(f"unknown probe relation: {relation}")
        if len(probe_manifest.get("pairs", [])) != 9:
            problems.append("probe manifest must contain nine deterministic pairs")
    except Exception as e:  # noqa: BLE001 - collect malformed probe details
        problems.append(f"probe manifest invalid: {e}")
    if problems:
        print("DATA VALIDATION FAILED")
        for p in problems[:100]:
            print("-", p)
        return 1
    print(
        f"Validated {sum(map(len, allrows.values()))} records, {len(assets)} distinct production image assets, and 40 fixtures."
    )
    print(
        "Image counts and grouped split boundaries are valid; assets match their recorded SHA-256 hashes."
    )
    return 0


if __name__ == "__main__":
    sys.exit(
        validate(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/generated"))
    )
