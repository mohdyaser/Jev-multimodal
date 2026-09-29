"""Deterministic, owned synthetic data for the Gemma E4B decision prototype.

No network access or third-party packages are needed. Images are small PNGs
rendered from primitive shapes and a bundled 5x7 bitmap font, so every label is
inspectable and reproducible. This is a data authoring utility, not a claim
that the generated image tasks represent real-world visual performance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import struct
import zlib
from collections.abc import Iterable
from pathlib import Path

SEED = 271828
RIGHTS = {
    "source": "locally_generated",
    "license": "owned_synthetic",
    "attribution": None,
    "notes": "Rendered deterministically by src/jev_data.py; no external assets or datasets.",
}
SPLIT_COUNTS = {
    "train": {0: 500, 1: 500, 2: 600, 3: 200, 4: 200},
    "dev": {0: 20, 1: 20, 2: 20, 3: 20, 4: 20},
    "calibration": {0: 20, 1: 20, 2: 20, 3: 20, 4: 20},
    "test": {0: 40, 1: 40, 2: 40, 3: 40, 4: 40},
}
TASKS = {
    0: ["text_status", "text_count", "text_score", "text_category"],
    1: ["image_fields", "image_status", "image_score", "image_category"],
    2: ["joint_match", "joint_count", "joint_rank", "joint_change"],
    3: ["joint_count", "joint_rank", "joint_change"],
    4: ["joint_count", "joint_rank", "joint_match", "joint_change"],
}
COLOR_NAMES = ["blue", "orange", "green", "purple", "red"]
SUPPORT_TOPICS = [
    ("payment_issue", "Payment problem", "The card was charged twice for this order."),
    (
        "refund_status",
        "Refund status",
        "The returned item arrived back, but the money has not been sent back.",
    ),
    (
        "delivery_delay",
        "Delivery delay",
        "The expected delivery date passed and the parcel still has not arrived.",
    ),
    (
        "tracking_issue",
        "Tracking issue",
        "Tracking has not changed since the parcel left the warehouse.",
    ),
    (
        "account_access",
        "Account access",
        "I cannot sign in even though the account details are correct.",
    ),
    ("password_reset", "Password reset", "The password reset email never arrives."),
    (
        "subscription_billing",
        "Subscription billing",
        "My monthly plan was billed at the wrong rate.",
    ),
    (
        "cancel_subscription",
        "Cancel subscription",
        "Please stop the recurring plan before the next renewal.",
    ),
    ("damaged_item", "Damaged item", "The item arrived cracked and cannot be used."),
    (
        "return_label",
        "Return label",
        "I need a shipping label to send this order back.",
    ),
    (
        "warranty_claim",
        "Warranty claim",
        "The device stopped working during the covered period.",
    ),
    (
        "address_change",
        "Address change",
        "Please update the delivery address before dispatch.",
    ),
]
SPLIT_WORDING = {
    "train": [
        "Review the record(s) below.",
        "Inspect the supplied evidence.",
        "Use the visible information.",
    ],
    "dev": ["Evaluate the presented evidence.", "Review the shown records."],
    "calibration": [
        "Assess the available evidence.",
        "Consider the displayed record(s).",
    ],
    "test": [
        "Determine the answer from the evidence.",
        "Inspect these record(s) and decide.",
    ],
    "fixture": [
        "Fixture check: inspect the evidence.",
        "Gate example: use the visible information.",
    ],
    "probe": [
        "Probe case: decide from the ordered images.",
        "Use the supplied evidence for this probe.",
    ],
}

# Tiny uppercase/digit font. Unsupported punctuation is drawn as a blank.
FONT = {
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "B": ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
    "C": ("01111", "10000", "10000", "10000", "10000", "10000", "01111"),
    "D": ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "F": ("11111", "10000", "10000", "11110", "10000", "10000", "10000"),
    "G": ("01111", "10000", "10000", "10111", "10001", "10001", "01111"),
    "H": ("10001", "10001", "10001", "11111", "10001", "10001", "10001"),
    "I": ("11111", "00100", "00100", "00100", "00100", "00100", "11111"),
    "J": ("00111", "00010", "00010", "00010", "10010", "10010", "01100"),
    "K": ("10001", "10010", "10100", "11000", "10100", "10010", "10001"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    "N": ("10001", "11001", "10101", "10011", "10001", "10001", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "P": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
    "Q": ("01110", "10001", "10001", "10001", "10101", "10010", "01101"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    "U": ("10001", "10001", "10001", "10001", "10001", "10001", "01110"),
    "V": ("10001", "10001", "10001", "10001", "10001", "01010", "00100"),
    "W": ("10001", "10001", "10001", "10101", "10101", "10101", "01010"),
    "X": ("10001", "10001", "01010", "00100", "01010", "10001", "10001"),
    "Y": ("10001", "10001", "01010", "00100", "00100", "00100", "00100"),
    "Z": ("11111", "00001", "00010", "00100", "01000", "10000", "11111"),
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11110", "00001", "00001", "01110", "00001", "00001", "11110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "10000", "11110", "00001", "00001", "11110"),
    "6": ("01110", "10000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00001", "01110"),
}


class Canvas:
    def __init__(self, w: int = 480, h: int = 320, bg=(250, 250, 247)):
        self.w, self.h = w, h
        self.p = bytearray(bytes(bg) * (w * h))

    def rect(self, x, y, w, h, color, outline=None, thick=2):
        x0, x1 = max(0, x), min(self.w, x + w)
        y0, y1 = max(0, y), min(self.h, y + h)
        if x1 <= x0 or y1 <= y0:
            return
        fill = bytes(color) * (x1 - x0)
        for yy in range(y0, y1):
            row_start = (yy * self.w + x0) * 3
            horizontal = outline is not None and (yy < y + thick or yy >= y + h - thick)
            if outline is None or horizontal:
                line = bytes(outline or color) * (x1 - x0)
                self.p[row_start : row_start + len(line)] = line
            else:
                self.p[row_start : row_start + len(fill)] = fill
                border = bytes(outline) * min(thick, x1 - x0)
                self.p[row_start : row_start + len(border)] = border
                self.p[row_start + len(fill) - len(border) : row_start + len(fill)] = (
                    border
                )

    def pixel(self, x, y, c):
        if 0 <= x < self.w and 0 <= y < self.h:
            i = (y * self.w + x) * 3
            self.p[i : i + 3] = bytes(c)

    def text(self, x, y, s, scale=3, color=(28, 43, 55)):
        cursor = x
        for ch in str(s).upper():
            glyph = FONT.get(ch)
            if glyph:
                for gy, row in enumerate(glyph):
                    for gx, on in enumerate(row):
                        if on == "1":
                            self.rect(
                                cursor + gx * scale, y + gy * scale, scale, scale, color
                            )
            cursor += 6 * scale if ch in FONT else 4 * scale

    def png(self):
        raw = b"".join(
            b"\0" + self.p[y * self.w * 3 : (y + 1) * self.w * 3] for y in range(self.h)
        )

        def chunk(tag, body):
            return (
                struct.pack(">I", len(body))
                + tag
                + body
                + struct.pack(">I", zlib.crc32(tag + body) & 0xFFFFFFFF)
            )

        return (
            b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">2I5B", self.w, self.h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9))
            + chunk(b"IEND", b"")
        )


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def opaque_id(prefix: str, key: str) -> str:
    return prefix + "_" + hashlib.sha256(key.encode()).hexdigest()[:12]


def render_asset(
    out: Path,
    record_key: str,
    slot: int,
    value: int,
    status: str,
    signed: bool,
    color_idx: int,
    layout="card_v1",
) -> dict:
    """Render a readable fictional record card; content is generated from labels."""
    c = Canvas()
    colors = [
        (54, 126, 183),
        (210, 117, 51),
        (61, 148, 104),
        (147, 91, 166),
        (196, 78, 89),
    ]
    x, y = 26, 24
    if layout == "ledger_v1":
        # Held-out landscape ledger: value and signature occupy separate panels,
        # with status in a full-width footer rather than the card's lower row.
        c.rect(x, y, 428, 272, (255, 255, 255), outline=(48, 72, 83), thick=3)
        c.rect(x + 18, y + 16, 392, 48, colors[color_idx], outline=None)
        c.text(x + 32, y + 27, "AUDIT RECORD", 3, (255, 255, 255))
        c.rect(
            x + 18, y + 80, 178, 112, (239, 245, 246), outline=(87, 112, 120), thick=2
        )
        c.text(x + 32, y + 92, "VALUE", 3, (54, 73, 81))
        c.text(x + 37, y + 128, f"{value:02d}", 6, colors[color_idx])
        c.rect(
            x + 212, y + 80, 198, 112, (247, 243, 235), outline=(129, 112, 77), thick=2
        )
        c.text(x + 229, y + 92, "SIGNATURE", 2, (82, 71, 54))
        c.rect(
            x + 265,
            y + 128,
            96,
            46,
            (225, 241, 230) if signed else (247, 230, 224),
            outline=(65, 116, 80) if signed else (165, 81, 66),
        )
        c.text(
            x + 285,
            y + 142,
            "YES" if signed else "NO",
            3,
            (40, 96, 61) if signed else (150, 65, 51),
        )
        c.rect(
            x + 18,
            y + 207,
            392,
            44,
            (225, 241, 230) if status == "PASS" else (247, 230, 224),
            outline=(65, 116, 80) if status == "PASS" else (165, 81, 66),
        )
        c.text(
            x + 34,
            y + 219,
            f"STATUS {status}",
            3,
            (40, 96, 61) if status == "PASS" else (150, 65, 51),
        )
    else:
        c.rect(x, y, 428, 272, (255, 255, 255), outline=(31, 53, 68), thick=3)
        c.rect(x + 16, y + 16, 396, 42, colors[color_idx], outline=None)
        c.text(x + 28, y + 24, f"CARD {slot + 1}", 3, (255, 255, 255))
        c.text(x + 30, y + 82, "VALUE", 3, (70, 84, 91))
        c.text(x + 30, y + 112, f"{value:02d}", 7, colors[color_idx])
        c.text(x + 30, y + 188, "STATUS", 3, (70, 84, 91))
        c.text(
            x + 170,
            y + 184,
            status,
            4,
            (35, 99, 69) if status == "PASS" else (168, 57, 60),
        )
        c.rect(
            x + 332,
            y + 185,
            48,
            48,
            (235, 245, 239) if signed else (248, 236, 231),
            outline=(65, 116, 80) if signed else (165, 81, 66),
        )
        c.text(
            x + 339,
            y + 201,
            "YES" if signed else "NO",
            2,
            (40, 96, 61) if signed else (150, 65, 51),
        )
    payload = c.png()
    asset_id = opaque_id("img", f"{record_key}:{slot}")
    rel = Path("images") / (asset_id + ".png")
    dest = out / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(payload)
    return {
        "image_id": asset_id,
        "path": rel.as_posix(),
        "sha256": digest(payload),
        "role": f"image_{slot + 1}",
        "width": 480,
        "height": 320,
        "layout": layout,
    }


def shuffled_options(
    rng: random.Random, labels: list[tuple[str, str]], keep_order=False
) -> tuple[list[dict], str]:
    pairs = list(labels)
    if not keep_order:
        rng.shuffle(pairs)
    markers = "ABCDEFGHIJKLMNOP"
    options = [
        {"id": oid, "description": label, "marker": markers[i]}
        for i, (oid, label) in enumerate(pairs)
    ]
    answer = next(o["marker"] for o in options if o["id"] == _CURRENT_ANSWER[0])
    return options, answer


# Set only within a single synchronous record build, then immediately consume.
_CURRENT_ANSWER = [""]


def alternate_question(task: str, count: int) -> str:
    variants = {
        "text_status": "Is the report's recorded outcome PASS?",
        "image_status": "Does image_1 show a PASS status?",
        "image_fields": "Does image_1 show a PASS status?",
        "text_category": "Which customer-service group owns this case?",
        "text_category_12": "Which service topic best matches this customer message?",
        "image_category": "What hue is shown in the header band of image_1?",
        "text_count": "What number of listed entries carry a signed mark?",
        "joint_count": "How many of the displayed records carry a signature mark?",
        "text_score": "How many checks are marked complete?",
        "image_score": "How many of the 1 records carry a visible signature mark?",
        "joint_match": "Are the VALUE readings identical in every image?",
        "joint_rank": "Which image has the highest VALUE?",
    }
    if task == "joint_change":
        return (
            "Between image_1 and image_2, is the value higher, lower, or unchanged?"
            if count == 2
            else "Across the ordered sequence, does the value rise at every step, fall at every step, or reverse direction?"
        )
    return variants[task]


def make_record(
    out: Path,
    split: str,
    count: int,
    index: int,
    task: str,
    family: str,
    seed: int,
    fixture_note: str | None = None,
) -> dict:
    key = f"{split}:{family}:{index:05d}:{task}"
    rng = random.Random(seed)
    fixture = split == "fixture"
    fixture_occurrence = index // len(TASKS[count])
    eval_split = split in ("dev", "calibration", "test")
    heldout_question = eval_split and index % 2 == 0
    heldout_visual = heldout_question and count > 0
    layout = "ledger_v1" if heldout_visual else "card_v1"
    # Text-only records still contain a small explicitly written set of checks.
    n = rng.randrange(2, 5) if count == 0 else count
    values = [rng.randrange(10, 100) for _ in range(n)]
    statuses = ["PASS" if rng.random() < 0.55 else "HOLD" for _ in range(n)]
    signed = [rng.random() < 0.5 for _ in range(n)]
    colors = [rng.randrange(5) for _ in range(n)]
    class_id = None
    # Permute order-invariant sets before assigning references or targets.
    # Position-based tasks are recomputed below from the resulting order;
    # trajectory tasks keep their meaningful chronological sequence.
    permutation = list(range(n))
    if task in ("joint_count", "joint_rank", "joint_match") and n > 1:
        rng.shuffle(permutation)
        values = [values[i] for i in permutation]
        statuses = [statuses[i] for i in permutation]
        signed = [signed[i] for i in permutation]
        colors = [colors[i] for i in permutation]
    # Every multi-image target consumes all members, and changes to any member
    # can change the target (count, ordered delta, all-match, or ranking).
    if task in ("text_status", "image_status"):
        truth = bool(fixture_occurrence % 2) if fixture else bool(rng.randrange(2))
        labels = [("false", "No"), ("true", "Yes")]
        answer_id = "true" if truth else "false"
        statuses[0] = "PASS" if truth else "HOLD"
        question = (
            "Is the record marked PASS?" if count else "Does the report state PASS?"
        )
        context = f"Report status: {'PASS' if truth else 'HOLD'}."
    elif task == "text_category":
        labels = [
            ("billing", "Billing"),
            ("delivery", "Delivery"),
            ("account", "Account access"),
            ("returns", "Returns"),
        ]
        class_id, description = rng.choice(labels)
        answer_id = class_id
        context = f"Case routing record: assigned department = {description}."
        question = "Which department is assigned to this case?"
    elif task == "text_category_12":
        topic_id, _topic_label, message = SUPPORT_TOPICS[index % len(SUPPORT_TOPICS)]
        class_id = topic_id
        answer_id = topic_id
        labels = [(tid, label) for tid, label, _ in SUPPORT_TOPICS]
        context = f"Customer message: {message}"
        question = "Which service topic best fits the customer's message?"
    elif task == "image_category":
        if fixture:
            colors[0] = index % len(COLOR_NAMES)
        labels = [(name, name.title()) for name in COLOR_NAMES]
        class_id = COLOR_NAMES[colors[0]]
        answer_id = class_id
        context = "Identify the color of the header bar in image_1."
        question = "Which color is the header bar?"
    elif task in ("text_count", "joint_count"):
        if fixture and task == "joint_count":
            signed = [i < min(fixture_occurrence, n) for i in range(n)]
        k = sum(signed)
        labels = [(str(i), str(i)) for i in range(n + 1)]
        answer_id = str(k)
        question = (
            f"How many of the {n} records have a signed mark?"
            if count
            else "How many of the listed entries are marked complete?"
        )
        context = (
            "Entries: "
            + ", ".join(
                f"entry {i + 1} {'complete' if v else 'incomplete'}"
                for i, v in enumerate(signed)
            )
            if not count
            else ""
        )
    elif task in ("text_score", "image_score"):
        k = sum(signed)
        labels = [(str(i), str(i)) for i in range(5)]
        answer_id = str(k)
        if task == "image_score":
            question = "How many of the 1 records show a signed mark?"
            context = "Inspect the visible signature mark in image_1."
        else:
            question = "How many required checks are visibly complete?"
            context = "Required checks: " + ", ".join(
                f"check {i + 1} {'complete' if v else 'incomplete'}"
                for i, v in enumerate(signed)
            )
    elif task == "image_fields":
        truth = bool(fixture_occurrence % 2) if fixture else bool(rng.randrange(2))
        labels = [("false", "No"), ("true", "Yes")]
        answer_id = "true" if truth else "false"
        statuses[0] = "PASS" if truth else "HOLD"
        question = "Is this record marked PASS?"
        context = "Inspect image_1."
    elif task == "joint_match":
        same = (fixture_occurrence % 2 == 0) if fixture else bool(rng.randrange(2))
        values = (
            [values[0]] * n
            if same
            else [10 + ((values[0] + i * 17) % 89) for i in range(n)]
        )
        if len(set(values)) == 1 and not same:
            values[-1] = 99 if values[-1] != 99 else 98
        labels = [("false", "No"), ("true", "Yes")]
        answer_id = "true" if same else "false"
        question = "Do all shown records have the same value?"
        context = "Compare the VALUE field in every image."
    elif task == "joint_rank":
        if fixture:
            top_index = fixture_occurrence % n
            values = [20 + 10 * i for i in range(n)]
            values[top_index] = 99
        top = max(range(n), key=lambda i: (values[i], -i))
        labels = [(f"image_{i + 1}", f"image_{i + 1}") for i in range(n)]
        answer_id = f"image_{top + 1}"
        question = "Which image shows the largest value?"
        context = "Compare the VALUE field across all images."
    elif task == "joint_change":
        # Ordered trajectory: each adjacent pair matters for a monotone target.
        mode = fixture_occurrence % 3 if fixture else rng.randrange(3)
        if n == 2:
            if mode == 0:
                values = [30, values[1] + 40 if values[1] < 59 else 99]
            elif mode == 1:
                values = [80, 20]
            else:
                values = [values[0], values[0]]
            labels = [
                ("decreased", "Decreased"),
                ("same", "Stayed the same"),
                ("increased", "Increased"),
            ]
            answer_id = (
                "increased"
                if values[1] > values[0]
                else "decreased"
                if values[1] < values[0]
                else "same"
            )
            question = "From image_1 to image_2, did the value increase, decrease, or stay the same?"
        else:
            if mode == 0:
                values = [20 + i * 8 for i in range(n)]
                answer_id = "increased"
            elif mode == 1:
                values = [90 - i * 8 for i in range(n)]
                answer_id = "decreased"
            else:
                values = [20 + i * 8 for i in range(n)]
                values[n // 2] -= 20
                answer_id = "changed"
            labels = [
                ("decreased", "Decreased at every step"),
                ("increased", "Increased at every step"),
                ("changed", "Changed direction"),
            ]
            question = f"Across image_1 through image_{n} in order, did values increase at every step, decrease at every step, or change direction?"
        context = (
            "Images are ordered from earlier to later. Compare each adjacent pair."
        )
    else:
        raise ValueError(task)

    if heldout_question:
        question = alternate_question(task, count)
    images = []
    for i in range(count):
        images.append(
            render_asset(
                out, key, i, values[i], statuses[i], signed[i], colors[i], layout=layout
            )
        )
    _CURRENT_ANSWER[0] = answer_id
    options, answer_marker = shuffled_options(
        rng,
        labels,
        keep_order=task in ("text_score", "text_count", "image_score", "joint_count"),
    )
    wording = rng.choice(SPLIT_WORDING[split])
    image_refs = " ".join(f"[{im['role']}]" for im in images)
    prompt = (
        f"{wording}\n{context}\n{image_refs}\nQuestion: {question}\nOptions:\n"
        + "\n".join(f"{o['marker']}. {o['description']}" for o in options)
        + "\nReply with one option marker only."
    )
    record = {
        "id": opaque_id("ex", key),
        "split": split,
        "num_images": count,
        "image_order": [im["image_id"] for im in images],
        "images": [im["path"] for im in images],
        "image_assets": images,
        "context": context,
        "question": question,
        "output_type": "score"
        if task in ("text_score", "text_count", "image_score", "joint_count")
        else (
            "choice"
            if task
            in (
                "joint_rank",
                "joint_change",
                "text_category",
                "text_category_12",
                "image_category",
            )
            else "noul"
        ),
        "options": options,
        "answer_id": answer_id,
        "target_id": answer_id,
        "answer_marker": answer_marker,
        "completion": answer_marker,
        "prompt": prompt,
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer_marker},
        ],
        "target": {
            "id": answer_id,
            "marker": answer_marker,
            "score_value": int(answer_id)
            if task in ("text_score", "text_count", "image_score", "joint_count")
            else None,
        },
        "task_family": task,
        "source_group_id": family,
        "template_family": (
            f"heldout_{split}_{'ledger' if heldout_visual else 'text'}_v2"
            if heldout_question
            else ("text_v1" if count == 0 else "card_v1")
        ),
        "question_family": f"heldout_{split}_questions_v2"
        if heldout_question
        else "default_questions_v1",
        "split_group_id": family,
        "source_kind": "owned_synthetic",
        "level_values": list(range(len(options)))
        if task in ("text_score", "text_count", "image_score", "joint_count")
        else None,
        "label_method": "deterministic_from_render_spec",
        "raw_annotation": {
            "values": values,
            "statuses": statuses,
            "signed": signed,
            "task": task,
            "class_id": class_id,
            "order_permutation": permutation,
        },
        "rights": RIGHTS.copy(),
        "seed": seed,
    }
    if fixture_note:
        record["fixture_note"] = fixture_note
    return record


def _probe_options(labels, answer_id, seed):
    rng = random.Random(seed)
    pairs = list(labels)
    rng.shuffle(pairs)
    options = [
        {"id": oid, "description": label, "marker": "ABCDEFGHIJKLMNOP"[i]}
        for i, (oid, label) in enumerate(pairs)
    ]
    marker = next(o["marker"] for o in options if o["id"] == answer_id)
    return options, marker


def _probe_row(
    out,
    pair_id,
    role,
    relation,
    count,
    task,
    images,
    values,
    signed,
    target_id,
    labels,
    question,
    target_asset_id=None,
    seed=SEED,
):
    options, marker = _probe_options(labels, target_id, seed)
    context = "Images are supplied in the listed order. Compare the visible VALUE and signature fields as the question requires."
    prompt = (
        f"Probe case: inspect all supplied images in order.\n{context}\n"
        + " ".join(f"[image_{i + 1}]" for i in range(count))
        + f"\nQuestion: {question}\nOptions:\n"
        + "\n".join(f"{o['marker']}. {o['description']}" for o in options)
        + "\nReply with one option marker only."
    )
    iid = opaque_id("probe", f"{pair_id}:{role}")
    return {
        "id": iid,
        "split": "probe",
        "num_images": count,
        "images": [im["path"] for im in images],
        "image_assets": [
            {**im, "role": f"image_{i + 1}"} for i, im in enumerate(images)
        ],
        "image_order": [im["image_id"] for im in images],
        "context": context,
        "question": question,
        "output_type": "score" if task == "joint_count" else "choice",
        "options": options,
        "answer_id": target_id,
        "target_id": target_id,
        "answer_marker": marker,
        "completion": marker,
        "prompt": prompt,
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": marker},
        ],
        "target": {
            "id": target_id,
            "marker": marker,
            "score_value": int(target_id) if task == "joint_count" else None,
        },
        "level_values": list(range(len(options))) if task == "joint_count" else None,
        "source_kind": "owned_synthetic_probe",
        "source_group_id": f"probe_fixture_{pair_id}",
        "rights": RIGHTS.copy(),
        "label_method": "deterministic_probe_spec",
        "raw_annotation": {
            "task": task,
            "values": values,
            "signed": signed,
            "target_asset_id": target_asset_id,
            "image_ids": [im["image_id"] for im in images],
        },
        "probe_pair_id": pair_id,
        "probe_role": role,
        "probe_relation": relation,
    }


def generate_probes(output: Path):
    """Generate independent held-out pairs; never derived from the test split."""
    records = []
    pairs = []
    for n in (2, 3, 4):
        # Later-image dependence: toggle only the signature on the final image.
        pair_id = f"later_signature_n{n}"
        base = [
            render_asset(
                output, f"probe:{pair_id}:common", i, 20 + 7 * i, "PASS", False, i % 5
            )
            for i in range(n - 1)
        ]
        before_last = render_asset(
            output,
            f"probe:{pair_id}:last_before",
            n - 1,
            20 + 7 * (n - 1),
            "PASS",
            False,
            (n - 1) % 5,
        )
        after_last = render_asset(
            output,
            f"probe:{pair_id}:last_after",
            n - 1,
            20 + 7 * (n - 1),
            "PASS",
            True,
            (n - 1) % 5,
        )
        labels = [(str(i), str(i)) for i in range(n + 1)]
        q = f"How many of the {n} records show a signed mark?"
        before = _probe_row(
            output,
            pair_id,
            "before",
            "later_image_changes_count",
            n,
            "joint_count",
            base + [before_last],
            [20 + 7 * i for i in range(n)],
            [False] * n,
            "0",
            labels,
            q,
            seed=SEED + n,
        )
        after = _probe_row(
            output,
            pair_id,
            "after",
            "later_image_changes_count",
            n,
            "joint_count",
            base + [after_last],
            [20 + 7 * i for i in range(n)],
            [False] * (n - 1) + [True],
            "1",
            labels,
            q,
            seed=SEED + n,
        )
        records.extend([before, after])
        pairs.append(
            {
                "pair_id": pair_id,
                "relation": "later_image_changes_count",
                "count": n,
                "before_id": before["id"],
                "after_id": after["id"],
                "decisive_position": n,
                "invariant_positions": list(range(1, n)),
                "expected_targets": ["0", "1"],
                "changed_field": "signature mark on final image",
            }
        )

        # Reordering preserves the winning asset but changes its image_N ID.
        pair_id = f"reorder_rank_n{n}"
        ordered = [
            render_asset(
                output,
                f"probe:{pair_id}:asset",
                i,
                15 + 20 * i,
                "PASS",
                bool(i % 2),
                i % 5,
            )
            for i in range(n)
        ]
        reversed_assets = list(reversed(ordered))
        values = [15 + 20 * i for i in range(n)]
        winner_asset = ordered[-1]["image_id"]
        q = "Which image shows the largest value?"
        labels = [(f"image_{i + 1}", f"image_{i + 1}") for i in range(n)]
        before_target = f"image_{ordered.index(next(im for im in ordered if im['image_id'] == winner_asset)) + 1}"
        after_target = f"image_{reversed_assets.index(next(im for im in reversed_assets if im['image_id'] == winner_asset)) + 1}"
        before = _probe_row(
            output,
            pair_id,
            "before",
            "order_invariant_winner_remapped",
            n,
            "joint_rank",
            ordered,
            values,
            [bool(i % 2) for i in range(n)],
            before_target,
            labels,
            q,
            target_asset_id=winner_asset,
            seed=SEED + 20 + n,
        )
        after_values = [
            values[
                ordered.index(
                    next(im for im in ordered if im["image_id"] == im2["image_id"])
                )
            ]
            for im2 in reversed_assets
        ]
        after_signed = [bool((n - 1 - i) % 2) for i in range(n)]
        after = _probe_row(
            output,
            pair_id,
            "after",
            "order_invariant_winner_remapped",
            n,
            "joint_rank",
            reversed_assets,
            after_values,
            after_signed,
            after_target,
            labels,
            q,
            target_asset_id=winner_asset,
            seed=SEED + 20 + n,
        )
        records.extend([before, after])
        pairs.append(
            {
                "pair_id": pair_id,
                "relation": "order_invariant_winner_remapped",
                "count": n,
                "before_id": before["id"],
                "after_id": after["id"],
                "expected_targets": [before_target, after_target],
                "same_winning_asset_id": winner_asset,
                "order_operation": "reverse all images; recompute image_N target",
            }
        )

        # Change the final lower-ranked distractor while the winner remains image_1.
        pair_id = f"irrelevant_rank_n{n}"
        vals = [90] + [25 + 10 * i for i in range(n - 1)]
        base_assets = [
            render_asset(
                output,
                f"probe:{pair_id}:common",
                i,
                vals[i],
                "PASS",
                bool(i % 2),
                i % 5,
            )
            for i in range(n - 1)
        ]
        last_before = render_asset(
            output,
            f"probe:{pair_id}:distractor",
            n - 1,
            vals[-1],
            "PASS",
            bool((n - 1) % 2),
            (n - 1) % 5,
        )
        changed_value = min(75, vals[-1] + 9)
        last_after = render_asset(
            output,
            f"probe:{pair_id}:distractor_changed",
            n - 1,
            changed_value,
            "PASS",
            bool((n - 1) % 2),
            (n - 1) % 5,
        )
        labels = [(f"image_{i + 1}", f"image_{i + 1}") for i in range(n)]
        q = "Which image shows the largest value?"
        before = _probe_row(
            output,
            pair_id,
            "before",
            "nonwinner_change_keeps_answer",
            n,
            "joint_rank",
            base_assets + [last_before],
            vals,
            [bool(i % 2) for i in range(n)],
            "image_1",
            labels,
            q,
            target_asset_id=base_assets[0]["image_id"],
            seed=SEED + 30 + n,
        )
        after_vals = vals[:-1] + [changed_value]
        after = _probe_row(
            output,
            pair_id,
            "after",
            "nonwinner_change_keeps_answer",
            n,
            "joint_rank",
            base_assets + [last_after],
            after_vals,
            [bool(i % 2) for i in range(n)],
            "image_1",
            labels,
            q,
            target_asset_id=base_assets[0]["image_id"],
            seed=SEED + 30 + n,
        )
        records.extend([before, after])
        pairs.append(
            {
                "pair_id": pair_id,
                "relation": "nonwinner_change_keeps_answer",
                "count": n,
                "before_id": before["id"],
                "after_id": after["id"],
                "expected_targets": ["image_1", "image_1"],
                "winner_asset_id": base_assets[0]["image_id"],
                "changed_position": n,
                "changed_field": "VALUE on nonwinning image",
            }
        )

    write_jsonl(output / "probes.jsonl", records)
    write_jsonl(output / "probe_manifest.jsonl", pairs)
    manifest = {
        "version": "jev-probes-v1",
        "split": "probe",
        "source_kind": "owned_synthetic_probe",
        "rights": RIGHTS,
        "records": len(records),
        "pairs": pairs,
        "pair_count": len(pairs),
        "file_sha256": digest((output / "probes.jsonl").read_bytes()),
        "pair_manifest_sha256": digest((output / "probe_manifest.jsonl").read_bytes()),
        "interpretation": "Expected relations are exact deterministic label/asset checks. Passing this manifest validates probe construction, not model behavior.",
    }
    (output / "probe_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def build_split_plan() -> list[dict]:
    """Allocate template/source groups to splits before any image set is made."""
    plan = []
    for split, by_count in SPLIT_COUNTS.items():
        for count, total in by_count.items():
            # Distinct per-split source families; records inherit a single group.
            group_total = max(8, min(total, 24))
            for i in range(total):
                group = f"{split}_c{count}_family_{(i * 7 + count) % group_total:02d}"
                if split == "train" and count == 0:
                    task = (
                        "text_category_12"
                        if i < 100
                        else "text_category"
                        if i < 250
                        else "text_status"
                        if i < 400
                        else "text_count"
                        if i < 450
                        else "text_score"
                    )
                elif split == "train" and count == 1:
                    task = (
                        "image_category"
                        if i < 250
                        else "image_status"
                        if i < 350
                        else "image_fields"
                        if i < 450
                        else "image_score"
                    )
                elif split in ("dev", "calibration", "test") and count == 0 and i < 2:
                    task = "text_category_12"
                else:
                    tasks = TASKS[count]
                    task = tasks[(i + count) % len(tasks)]
                plan.append(
                    {
                        "split": split,
                        "count": count,
                        "index": i,
                        "group": group,
                        "task": task,
                        "seed": SEED
                        + int(
                            hashlib.sha256(f"{split}:{count}:{i}".encode()).hexdigest()[
                                :8
                            ],
                            16,
                        ),
                    }
                )
    return plan


def write_jsonl(path: Path, records: Iterable[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")


def generate(output: Path, fixtures_only=False):
    output.mkdir(parents=True, exist_ok=True)
    fixtures = []
    # Eight hand-curated task patterns for each image-count gate, with stable
    # seed and notes explaining the dependency each fixture is meant to expose.
    for count in range(5):
        tasks = TASKS[count]
        for i in range(8):
            task = tasks[i % len(tasks)]
            fixture_seed = SEED + int(
                hashlib.sha256(f"fixture:{count}:{i}".encode()).hexdigest()[:8], 16
            )
            fixtures.append(
                make_record(
                    output,
                    "fixture",
                    count,
                    i,
                    task,
                    f"fixture_c{count}_{i:02d}",
                    fixture_seed,
                    fixture_note=(
                        "Inspect all ordered images; this target is derived from the displayed values/marks."
                        if count > 1
                        else "Inspect the text or image evidence and compare it with the target."
                    ),
                )
            )
    write_jsonl(output / "fixtures.jsonl", fixtures)
    if fixtures_only:
        return

    # Allocate the entire split/group plan first. Only then render assets and
    # construct multi-image records, enforcing one-split ownership per group.
    plan = build_split_plan()
    records_by_split = {s: [] for s in SPLIT_COUNTS}
    for spec in plan:
        r = make_record(
            output,
            spec["split"],
            spec["count"],
            spec["index"],
            spec["task"],
            spec["group"],
            spec["seed"],
        )
        records_by_split[spec["split"]].append(r)
    for split, records in records_by_split.items():
        write_jsonl(output / f"{split}.jsonl", records)
    all_assets = {
        im["image_id"]: im
        for rs in records_by_split.values()
        for r in rs
        for im in r["image_assets"]
    }
    write_jsonl(output / "assets.jsonl", (all_assets[k] for k in sorted(all_assets)))
    generate_probes(output)
    data_hashes = {
        f"{s}.jsonl": digest((output / f"{s}.jsonl").read_bytes())
        for s in records_by_split
    }
    data_hashes["fixtures.jsonl"] = digest((output / "fixtures.jsonl").read_bytes())
    data_hashes["assets.jsonl"] = digest((output / "assets.jsonl").read_bytes())
    fixture_counts = {
        str(n): sum(r["num_images"] == n for r in fixtures) for n in range(5)
    }
    summary = {
        "seed": SEED,
        "rights": RIGHTS,
        "split_counts": {
            s: {str(n): sum(r["num_images"] == n for r in rs) for n in range(5)}
            for s, rs in records_by_split.items()
        },
        "records": sum(map(len, records_by_split.values())),
        "fixture_records": len(fixtures),
        "fixture_counts": fixture_counts,
        "image_count": sum(
            len(r["images"]) for rs in records_by_split.values() for r in rs
        ),
        "split_group_counts": {
            s: len({r["split_group_id"] for r in rs})
            for s, rs in records_by_split.items()
        },
        "file_sha256": data_hashes,
        "generator_sha256": digest(Path(__file__).read_bytes()),
        "asset_manifest": "assets.jsonl",
        "external_datasets": [],
        "prompt_version": "jev-decision-v1",
        "answer_markers": "A-P placeholders; tokenizer gate required before training",
    }
    (output / "manifest.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (output / "coverage.json").write_text(
        json.dumps(coverage(records_by_split), indent=2) + "\n", encoding="utf-8"
    )


def coverage(splits):
    return {
        split: {
            "records": len(rs),
            "by_image_count": {
                str(n): sum(r["num_images"] == n for r in rs) for n in range(5)
            },
            "by_type": {
                t: sum(r["output_type"] == t for r in rs)
                for t in ("noul", "choice", "score")
            },
            "by_task": {
                t: sum(r["task_family"] == t for r in rs)
                for t in sorted({r["task_family"] for r in rs})
            },
            "by_template_family": {
                t: sum(r["template_family"] == t for r in rs)
                for t in sorted({r["template_family"] for r in rs})
            },
            "heldout_question_rows": sum(
                r.get("question_family", "").startswith("heldout_") for r in rs
            ),
            "heldout_visual_rows": sum(
                any(im.get("layout") == "ledger_v1" for im in r.get("image_assets", []))
                for r in rs
            ),
            "by_option_count_band": {
                "2-5": sum(2 <= len(r["options"]) <= 5 for r in rs),
                "6-7": sum(6 <= len(r["options"]) <= 7 for r in rs),
                "8-16": sum(8 <= len(r["options"]) <= 16 for r in rs),
            },
        }
        for split, rs in splits.items()
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=Path("data/generated"))
    ap.add_argument("--fixtures-only", action="store_true")
    args = ap.parse_args()
    generate(args.output, args.fixtures_only)


if __name__ == "__main__":
    main()
