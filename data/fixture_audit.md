# Generated fixture visual audit

Audit date: 2026-09-29

## Scope and method

Manually reviewed all 40 records in `data/generated/fixtures.jsonl` against their referenced PNGs. Read every text-only context directly and visually inspected all 80 image references in contact sheets grouped by image count. Checked that each image path exists, its SHA-256 matches the record manifest, image order matches the listed `image_assets`, and the target marker matches the completion.

## Counts and findings

| Ordered image count | Records | Images reviewed | Result |
|---:|---:|---:|---|
| 0 | 8 | 0 | All 8 targets agree with the text context. |
| 1 | 8 | 8 | All 8 targets agree with the visible record fields and signed-mark badges. |
| 2 | 8 | 16 | All 8 targets agree with the displayed values and marks. |
| 3 | 8 | 24 | All 8 targets agree with the displayed values and marks. |
| 4 | 8 | 32 | All 8 targets agree with the displayed values and marks. |
| **Total** | **40** | **80** | **No target mismatches found in the reviewed fixtures.** |

All 80 referenced images exist, match their declared hashes, and are unique within this fixture set. Every record is marked `fixture`; all records have unique IDs, source-group IDs, and template-family IDs. No IDs, image assets, split groups, or source groups overlap with the current train, dev, calibration, or test files. The ordered `image_assets` paths match the record image lists; multi-image cards visibly appear in `CARD 1` through `CARD n` order.

The PNGs are 480×320. Their large pixel-style values and PASS/HOLD and YES/NO labels are readable at native size. This visual review does not establish that all text remains legible after Gemma's 140-token-per-image processing profile.

## Regenerated score wording

The earlier version asked two one-image `image_score` records “How many required checks are visibly complete?” even though each image showed one signed-mark badge and no checklist. The generator was updated and the fixtures regenerated. Both now ask “How many of the 1 records show a signed mark?”; I re-inspected the current images and verified the score target against the YES/NO badge:

- `ex_ab31ea918630`: badge NO, target 0.
- `ex_483ac50272c0`: badge YES, target 1.

The wording now names the quantity represented in the image, and the regenerated fixtures include both score levels.

## Repetition and leakage limits

This is a wiring/coverage fixture set, not a generalization or calibration sample. Repeated question templates remain by design, but regenerated 2–4 image targets now vary: the two-image rank, count, and change tasks cover both outcomes; three-image counts cover 0, 1, and 2, ranks select each image position, and change tasks include increase and decrease; four-image tasks include both match outcomes, two count levels, two rank positions, and increase/decrease. The repeated value sequences noted in the first audit have been removed. All records remain in the fixture split and do not leak into current train, dev, calibration, or test assets/groups.

## Limitations

The review was one manual visual pass, including renewed inspection of the regenerated one-image and 2–4 image rows. It verifies visible labels and straightforward comparisons at native resolution; it does not measure model-readable resolution, test the processor's image ordering at runtime, establish broad task diversity, or independently validate the generator implementation. The fixtures support wiring checks, not claims about generalization or behavior quality.
