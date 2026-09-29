# Production label and legibility audit

## Scope and method

Audited final regenerated production examples from `data/generated/train.jsonl`, `dev.jsonl`, and `calibration.jsonl`. `test.jsonl` was excluded and not opened. The audit checks whether each target agrees with the visible source record(s) and whether the rendered wording and values are legible. It does not assess model performance or real-world generalization.

Sampling was deterministic (seed `20260929`). Within each of the 13 task families, selected at least 20 rows by round-robin across split, image count, and target class. Also included 20 rows from each heldout ledger template (`heldout_dev_ledger_v2`, `heldout_calibration_ledger_v2`) and all 10 rows from each heldout text template (`heldout_dev_text_v2`, `heldout_calibration_text_v2`); these supplemental selections can overlap the task-family selections. The merged sample contains **271 unique records**: 132 train, 73 dev, and 66 calibration. It includes **169 image-bearing records and 349 image references** (80 one-image, 29 two-image, 29 three-image, and 31 four-image records), plus 102 text-only records.

Every image-bearing selected row was reviewed in contact sheets grouped by split/template/task/image count. The referenced generated PNGs are 480×320. Text-only rows were checked against their context, question, and option set. For every one of the 349 selected image references, the path exists and its SHA-256 matches the row's recorded hash. All 271 rows have matching `target_id` and `answer_id`.

## Reviewed rows and target coverage

| Task family | Rows | Target counts in audited sample |
|---|---:|---|
| `image_category` | 20 | blue 3, green 6, orange 4, purple 2, red 5 |
| `image_fields` | 20 | false 11, true 9 |
| `image_score` | 20 | 0: 13, 1: 7 |
| `image_status` | 20 | false 6, true 14 |
| `joint_change` | 23 | changed 6, decreased 8, increased 8, same 1 |
| `joint_count` | 23 | 0: 3, 1: 7, 2: 8, 3: 4, 4: 1 |
| `joint_match` | 20 | false 11, true 9 |
| `joint_rank` | 23 | image_1 7, image_2 10, image_3 4, image_4 2 |
| `text_category` | 20 | account 4, billing 6, delivery 6, returns 4 |
| `text_category_12` | 20 | account_access 1, address_change 1, cancel_subscription 1, damaged_item 1, delivery_delay 1, password_reset 2, payment_issue 3, refund_status 4, return_label 1, subscription_billing 1, tracking_issue 2, warranty_claim 2 |
| `text_count` | 20 | 0: 5, 1: 2, 2: 7, 3: 3, 4: 3 |
| `text_score` | 22 | 0: 3, 1: 5, 2: 9, 3: 2, 4: 3 |
| `text_status` | 20 | false 9, true 11 |

The reviewed template counts are: `card_v1` 114, `text_v1` 82, heldout dev ledger 28, heldout calibration ledger 27, heldout dev text 10, and heldout calibration text 10. The ledger template totals include the extra heldout rows and overlap with the task-family sample.

## Findings and re-audit

- **Label mismatches:** 0 found in 271 reviewed rows. Negatives were included across yes/no, status, equality, signed-mark count, and change tasks; both outcomes were checked where available. The 2–4 image rows were checked against all ordered images, including the winning image index and equal/non-equal value cases.
- **Unreadable or misplaced image references:** 0. The cards and ledger records were legible at their rendered 480×320 size, and all referenced PNGs resolved with matching hashes.
- **Text/option mapping errors:** 0 found. Audited answer IDs mapped to the stated answer in the context; reordered option lists and heldout wording variants retained the correct answer mapping.
- **Re-audit after final regeneration:** repeated checks of `image_score`, joint change/count/match/rank, and both heldout ledger layouts after the generator fixes. In particular, score labels matched the visible signature mark, three-image change labels agreed with their displayed ordered values, and joint targets agreed with every image in the set. No remaining production-sample issue was found.

The synthetic examples deliberately reuse a small visual vocabulary and generated card layouts, so this audit establishes internal label/render consistency only. It is not evidence that the examples cover natural images or that a fine-tuned model will transfer beyond these patterns. The deterministic sample is broad across task families, splits, targets, and image counts, but it is still a sample rather than an exhaustive manual review of all production rows.
