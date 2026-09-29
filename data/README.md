# Owned synthetic decision data

`python src/jev_data.py --output data/generated` deterministically creates the
40 image-count gate fixtures and the full initial corpus. It uses only Python
standard-library code and makes no network requests. Every raster image is a
locally rendered fictional card; provenance is recorded as
`owned_synthetic` / `locally_generated` in each record and in the manifest.

The generated row schema exposes `id`, `split`, `context`, ordered `images`
(relative PNG paths), `question`, `output_type`, `options` (`id`, `description`,
and a provisional `marker`), `target_id`, `level_values`, `source_kind`,
`source_group_id`, `rights`, and `label_method`. `image_assets` gives dimensions
and SHA-256 hashes for every path. `completion` is the selected marker for a
completion-only trainer. Markers A–P are placeholders; verify their one-token
property against the actual Gemma tokenizer and answer prefix before training.
Binary `noul` options use `false` and `true` IDs with `No` and `Yes`
descriptions. The train type mix is about 52% choice, 28% binary, and 21% score.
The 12-way support-topic task contributes 100 of 2,000 training rows (5%),
with balanced labels and twelve explicit options. Small development,
calibration, and test slices contain two such rows each.

The split plan allocates source/template groups first, then forms examples and
ordered image sets. Each image is rendered per example, so no asset is reused
across records or splits. Split-specific wording variants and family IDs are
recorded for auditing. Order-invariant comparison sets are deterministically
permuted with annotations and position-based answers assigned after ordering;
chronological trajectory tasks preserve their meaningful order. Test is held
out by split; do not use it to debug.

Half of each development, calibration, and test split uses a held-out question
family; for rows with images that slice also uses a separate ledger layout with
the value, signature, and status in distinct panels. The standard card and
ledger layout render the same target fields. Per-split family counts and heldout
question/visual row counts are recorded in `coverage.json`.

To generate only the gate fixtures:

```sh
python src/jev_data.py --output data/generated --fixtures-only
```

To validate the complete output:

```sh
python data/validate_data.py data/generated
```

The 40 fixtures are 8 records at each image count from 0 through 4. They are
intended for direct inspection via `fixtures.jsonl` and the referenced PNGs.
The data generator records deterministic label source fields, but it does not
assert visual verification or real-world model performance. Synthetic examples
are not a substitute for the rights-checked real-image slice required by the
plan; `manifest.json` explicitly records that no external datasets were used.

The separate `probes.jsonl` contains 9 held-out pairs (18 rows) across 2, 3,
and 4 images. Its `split` is `probe`; none of these rows comes from `test.jsonl`
or is intended for training. `probe_manifest.jsonl` records each pair and its
expected relation: flipping the final signature changes the count; reversing
the same images preserves the winning asset while remapping its `image_N`
answer; changing a nonwinning value keeps the top-ranked image unchanged.
`probe_manifest.json` includes checksums. `python data/validate_data.py
data/generated` verifies probe asset hashes, answer labels, pair isolation,
order remapping, and the manifest checksum.
