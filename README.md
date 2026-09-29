# Jev inspired Gemma 4 E4B decision prototype

This prototype adapts Gemma 4 E4B for typed decisions from a question and zero
to four ordered images. It scores one-token option markers and normalizes over
the supplied options. Jev inspired describes the interaction pattern; this
does not reproduce Jev's private architecture or establish Jev-equivalent
behavior.

## Setup and pinned model

The experiment used Python 3.11, MLX-VLM 0.7.3, and MLX 0.32.2 on an Apple
M5 Pro MacBook Pro with 64 GB unified memory. Training requires Apple Silicon
and macOS.

```sh
uv venv --python 3.11 .venv
UV_CACHE_DIR=.uv-cache uv pip install --python .venv/bin/python -r requirements.txt
HF_HOME=.hf-cache HF_HUB_DISABLE_IMPLICIT_TOKEN=1 hf download \
  mlx-community/gemma-4-e4b-it-4bit \
  --revision 475b9088d29754a3379866cf5aeb6b41acd313c2 \
  --local-dir models/gemma-4-e4b-it-4bit
```

Accept Google's Gemma terms and authenticate with Hugging Face if required.
`MODEL_MANIFEST.json` pins the converted checkpoint's revision, file sizes,
and SHA-256 hashes. Training, checkpoint selection, and inference verify the
local files against this manifest; unchanged files use a local verification
cache. `config/experiment.json` records the default system, selected adapter
hash, split hashes, prompt/processor settings, and both calibration scalars.

## Try the local UI

With the environment and model files from setup in place, start the browser
interface:

```sh
.venv/bin/python src/jev_ui.py --open
```

Open `http://127.0.0.1:7860/` if a browser does not open automatically. Choose
the **noul** (yes/no), **choice**, or **score** template, edit its question and
options, and optionally add up to four ordered PNG, JPEG, or WebP images. The
result shows the selected option and a normalized probability for every
option; score also shows the expected numeric score. Use the model selector
to compare the frozen base with the experimental adapter. The frozen base is
the default. The server listens only on this machine, loads the model on the
first prediction, and applies the calibration temperature recorded in
`config/experiment.json`. Stop it with Ctrl+C. These probabilities compare
the supplied options; they are not a guarantee that an answer is correct.

## Data and verification

The deterministic local generator creates 2,000 training decisions, 100
development, 100 calibration, and 200 test decisions, plus 40 separate wiring
fixtures and paired image probes. All production and evaluated images are
synthetic. The [label audit](data/label_audit.md) reports a manual sample of
271 non-test decisions and 349 ordered image references; it found no sampled
label/render mismatch. The [online source audit](data/source_audit.md) records
why no third-party data entered this run. There is no rights-checked real-image
slice.

```sh
.venv/bin/python src/jev_data.py --output data/generated
.venv/bin/python data/validate_data.py data/generated
.venv/bin/python scripts/check_processor.py
.venv/bin/python scripts/check_training_data.py
```

The processor check covers image counts and ordering; the data check visits
all fixtures for completion boundaries, image tensors, and sequence limits.
All 0–4 image counts passed. A four-image fixture used 504 visual tokens and
601 processed tokens; the longest fixture sequence was 629 tokens, below the
2,048 limit. Mixed landscape/portrait input was checked in both training and
inference and normalized to a stacked `[2, 3, 528, 528]` tensor.

MLX-VLM 0.7.3's generic VLM loss passes a 2D padding mask that bypasses Gemma
4's causal mask and leaks future answer tokens. The earlier runs using that
loss are marked `INVALID.txt`. The corrected training and selector use
`jev_train.decision_loss` with `mask=None`, sliced multimodal token types, and
an explicit completion mask, plus `iterate_single_example_batches` to preserve
each example's ordered `[N, C, H, W]` image tensor. A causal four-image gate
completed with finite loss, 8/8 accuracy on its tiny training subset, and
10.080 GB peak MLX memory. Reloaded adapter logits and causal training versus
inference logits matched on the checked fixtures. These are wiring checks.

## Training and checkpoint selection

For a small wiring run, 400 microsteps with accumulation 8 make 50 optimizer
updates:

```sh
.venv/bin/python src/jev_train.py \
  --dataset data/generated/fixtures.jsonl \
  --output runs/gate_causal_smoke \
  --iters 400 --gradient-accumulation 8 --save-every 50
```

The completed one-epoch experiment used the following command:

```sh
.venv/bin/python src/jev_train.py \
  --dataset data/generated/train.jsonl \
  --output runs/main_1epoch \
  --save-every 400
```

It trained all 2,000 rows for one pass: 2,000 microsteps, 250 optimizer
updates, microbatch 1, accumulation 8, rank 8/alpha 16 Q/O adapters on the
upper eight language layers, and a peak learning rate of `2e-5`. It took about
16 minutes, peaked at 10.080 GB MLX memory, and used no swap. See
`runs/main_1epoch/training_config.json` for configuration and dataset hash.
The causal-loss, iterator, and base-identity provenance fields were recorded
after the run completed based on the observed launch configuration; the file
documents this retrofit.

The selector considers development records only, scores each periodic adapter
with mean completion-only full-vocabulary cross-entropy, records losses and
counts, copies the lowest-loss checkpoint to `runs/main_1epoch/selected/`, and
verifies a fresh reload:

```sh
.venv/bin/python scripts/select_checkpoint.py \
  --dev data/generated/dev.jsonl --run-dir runs/main_1epoch
```

The final 2,000-microstep adapter had the lowest dev loss (0.1823). The test
split did not select the checkpoint or tune the prompt.

## Predict and evaluate

Requests contain a question, optional context, ordered image paths, an
`output_type` (`noul`, `choice`, or `score`), and 2–16 options with IDs and
descriptions. Binary `noul` options use IDs `false` and `true`. Score requests
can include increasing numeric `level_values`. Frozen inference is the
default; select the adapter explicitly for experiments:

```sh
.venv/bin/python src/jev_cli.py --input data/request.json --output runs/response.json
.venv/bin/python src/jev_cli.py --input data/request.json \
  --adapter runs/main_1epoch/selected --output runs/adapter_response.json
```

To reproduce the saved comparison, score calibration plus test examples, then
let `jev_eval.py` fit temperature on calibration and report metrics on test:

```sh
cat data/generated/calibration.jsonl data/generated/test.jsonl > data/generated/calibration_test.jsonl
.venv/bin/python src/jev_cli.py --input data/generated/calibration_test.jsonl --jsonl \
  --output runs/baseline_predictions.jsonl
.venv/bin/python src/jev_eval.py runs/baseline_predictions.jsonl \
  --output runs/baseline_report.json
.venv/bin/python src/jev_cli.py --input data/generated/calibration_test.jsonl --jsonl \
  --adapter runs/main_1epoch/selected --output runs/tuned_predictions.jsonl
.venv/bin/python src/jev_eval.py runs/tuned_predictions.jsonl \
  --output runs/tuned_report.json
```

Evaluation collected raw logits on calibration and test records, fit one
temperature on calibration, and reported test metrics. The frozen model's
temperature was 4.048; the selected adapter's was 1.491. Test accuracy was
79.0% (158/200) for both. Calibrated NLL was 0.551 frozen and 0.494 adapter;
calibrated Brier was 0.289 and 0.272; score MAE was 0.549 and 0.487 over 63
score examples. The adapter corrected 11 frozen errors and introduced 11 new
ones. Paired image probes passed 6/9 for each model; both missed some later
image changes and one three-image reorder.

The CLI therefore keeps the frozen base as its default. Reports and raw
predictions are in `runs/baseline_report.json`, `runs/tuned_report.json`, and
the corresponding `*_predictions.jsonl` files. The full results and
per-image-count breakdown are in [RESULTS.md](RESULTS.md). Calibration
temperatures can be applied with `--temperature-report runs/baseline_report.json`
or `--temperature-report runs/tuned_report.json`.

These results cover a small synthetic corpus with limited visual variety.
They do not establish natural-image, dense OCR, unfamiliar-content, production
confidence, or general Jev-like performance. The next data step is a
rights-checked and visually audited real-image multi-image slice followed by
a new held-out evaluation.
