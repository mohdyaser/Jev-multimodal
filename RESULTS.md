# Gemma 4 E4B decision prototype: local experiment

Run date: 29 September 2026. Machine: Apple M5 Pro MacBook Pro, 64 GB unified memory. The base is the pinned `mlx-community/gemma-4-e4b-it-4bit` conversion at revision `475b9088d29754a3379866cf5aeb6b41acd313c2`; [MODEL_MANIFEST.json](MODEL_MANIFEST.json) verifies the local files. MLX-VLM is 0.7.3 and MLX is 0.32.2.

## Data and gate

The deterministic local generator produced 2,000 training decisions (500/500/600/200/200 for 0/1/2/3/4 images), 100 development, 100 calibration, and 200 test decisions, plus 40 separate wiring fixtures. All 3,900 production images and every evaluated example are synthetic. The validator found no image, hash, or split overlap issue. A manual audit found no label/render mismatch among 271 sampled non-test production rows, including 349 ordered image references; [data/label_audit.md](data/label_audit.md) records the sample and its limits. No rights-checked real-image slice was included.

All five image counts reach the processor in order. With 140 requested visual tokens per image, the checked four-image fixture used 504 actual image tokens and 601 processed tokens; the longest fixture training sequence was 629, below the 2,048 cap. Markers A–P are each one token at the answer boundary. Training uses the same prompt boundary as inference and a causal, completion-only loss. The stock MLX-VLM 0.7.3 generic loss leaked future answer tokens on all eight tested four-image fixtures, so its earlier smoke artifacts are marked `INVALID.txt`.

The corrected 400-microstep four-image gate reached 8/8 accuracy on its tiny training subset, with finite loss and 10.080 GB peak MLX memory. Adapter reload reproduced the checked answer logits exactly; causal full-sequence training logits also matched inference logits exactly across all eight four-image fixtures. This is a wiring check, not a generalization result. Mixed-aspect inference was also checked with a landscape plus portrait pair: inference and training both processed a stacked `[2, 3, 528, 528]` image tensor, and the model returned normalized probabilities.

Eight-microstep single-count checks, each from fresh adapters, provided these approximate rates and peak MLX allocations. The short windows include warm-up and are not end-to-end latency guarantees.

| Images | Microsteps/s | Peak MLX GB |
|---:|---:|---:|
| 0 | 6.142 | 5.982 |
| 1 | 3.046 | 6.744 |
| 2 | 1.801 | 7.731 |
| 3 | 1.268 | 8.772 |
| 4 | 1.025 | 9.746 |

## One-epoch adapter experiment

The full run used all 2,000 training rows once, with microbatch 1, gradient accumulation 8, 250 optimizer updates, AdamW, 2e-5 peak learning rate, rank 8/alpha 16 LoRA in the query/output projections of the upper eight language layers, and frozen base weights. It took about 16 minutes including preflight, peaked at 10.080 GB MLX memory, and used no swap. Its configuration and dataset hash are in `runs/main_1epoch/training_config.json`. Causal-loss and base-identity fields were added to that file after the already running process ended; the file explicitly records that retrofit. New runs write them at launch.

Five periodic adapters were scored on the 100-row development split using full-vocabulary completion cross-entropy. The final 2,000-microstep adapter had the lowest mean loss (0.1823) and was copied to `runs/main_1epoch/selected/`. A fresh reload reproduced the selected example's dev loss exactly. The test split did not select the checkpoint or change the prompt.

## Held-out synthetic test results

Each system fit one temperature on the 100-row calibration split. Frozen temperature: 4.048. Selected adapter temperature: 1.491. Temperature scaling changes probabilities, not the selected option. All values below use the same untouched 200-row synthetic test manifest; lower NLL, Brier, and score MAE are better.

| System | Accuracy | Raw NLL | Calibrated NLL | Raw Brier | Calibrated Brier | Calibrated score MAE (n=63) |
|---|---:|---:|---:|---:|---:|---:|
| Frozen | 158/200 (79.0%) | 1.066 | 0.551 | 0.366 | 0.289 | 0.549 |
| Adapter | 158/200 (79.0%) | 0.521 | 0.494 | 0.284 | 0.272 | 0.487 |

| Image count | Frozen correct | Adapter correct |
|---:|---:|---:|
| 0 | 38/40 | 39/40 |
| 1 | 35/40 | 35/40 |
| 2 | 36/40 | 37/40 |
| 3 | 26/40 | 25/40 |
| 4 | 23/40 | 22/40 |

By output type, frozen/adapter accuracy was `choice` 70/88 vs 72/88, `noul` 49/49 vs 49/49, and `score` 39/63 vs 37/63. The adapter corrected 11 frozen errors and introduced 11 new errors. The three- and four-image regressions are one example each, but those slices are weak in both systems. The 9 paired image probes passed 6/9 for each system. Both missed some changes to later images and one three-image reorder, so the training did not establish reliable multi-image dependence.

The adapter improves probability quality on this synthetic test but did not improve answer accuracy or paired-probe success. The CLI therefore keeps the frozen model as its default; the selected adapter remains available with `--adapter runs/main_1epoch/selected` for experiments. Use `--temperature-report runs/baseline_report.json` or `runs/tuned_report.json` with the corresponding system. The two complete reports and raw-logit predictions are under `runs/`.

The sample is small and synthetic with a limited visual vocabulary. These counts do not establish performance on natural photos, real documents, dense OCR, unfamiliar wording, production confidence, or Jev-equivalent behavior. The practical next data step is an independently rights-checked and visually audited real-image multi-image slice, followed by a new held-out evaluation.

The [online source audit](data/source_audit.md) records the candidate dataset check and why no external rows were used in this run.
