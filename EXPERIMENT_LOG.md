# TAO Dehazing Retrospective Experiment Ledger

> **Provenance notice.** The initial runs did not save complete raw terminal stdout logs. This ledger is reconstructed only from verified commands, artifacts, metrics, hashes, source inspection, and Git history. It is a structured retrospective experiment record, not a reconstructed console transcript. No fabricated timestamps, losses, iteration logs, or console output are included.

## Common Environment

- Operating system: Windows
- Conda environment: `tao`
- Python: 3.10.20
- PyTorch: 2.7.1+cu128
- torchvision: 0.22.1+cu128
- PyTorch CUDA runtime: 12.8
- cuDNN: 9.7.1
- GPU: NVIDIA GeForce RTX 5060 Laptop GPU

## EXP-001 — Single-image smoke test

- **Experiment ID:** EXP-001
- **Objective:** Verify that the TAO dehazing pipeline can run end to end and save a restored image.
- **Input:** `0586.jpg`
- **Input source:** `test_samples/HSTS_256x256/synthetic/0586.jpg`
- **Local temporary input:** `test_samples/HSTS_single_input/`
- **Command:** The exact shell command was not retained. The verified input/output paths are recorded here; no command transcript is fabricated.
- **Environment:** Common environment above.
- **Important configuration:** No complete run-specific configuration record beyond the unchanged `sample_dehazing.py` invocation context was retained.
- **Output artifacts:** `results_single/0586.jpg`
- **Quantitative result:** No formal metric record was retained for this smoke test.
- **Observation:** The expected output image exists, demonstrating that the pipeline completed successfully for the selected input.
- **Conclusion:** **PASSED** — the TAO dehazing pipeline executed end to end.
- **Public / local-only artifact status:** `results_single/` and `test_samples/HSTS_single_input/` are local-only smoke-test artifacts and are excluded from public Git tracking.

## EXP-002 — HSTS Batch10 reproduction

- **Experiment ID:** EXP-002
- **Objective:** Reproduce TAO dehazing on all 10 processed HSTS hazy inputs in one Python process and compare the outputs under a unified evaluator.
- **Input:** 10 HSTS synthetic hazy images from `test_samples/HSTS_256x256/synthetic/`.
- **Command:**

  ```bash
  python sample_dehazing.py \
    --sample_dir "test_samples/HSTS_256x256/synthetic" \
    --result_dir "results_batch_ours"
  ```

- **Environment:** Common environment above.
- **Important configuration:** `batch_size=1`, `inference_num=2`, `guidance_scale=25000`, `diffusion_steps=1000`, `use_fp16=True`, and per-image `seed=10`.
- **Output artifacts:** `results_batch_ours/` (10 JPEG images), `reproduction_metrics.csv`.
- **Quantitative result:**

  | Evaluated images | PSNR (dB) | SSIM |
  |---|---:|---:|
  | Hazy Input | 14.7446 | 0.753592 |
  | Author-provided TAO | 22.3276 | 0.856213 |
  | Our Reproduction | 18.7119 | 0.797267 |

- **Observation:** Our reproduction improves over the hazy input under the unified RGB evaluator, but its aggregate result remains below the author-provided images.
- **Conclusion:** The public pipeline ran successfully, but this environment and execution did not exactly reproduce the author-provided outputs.
- **Public / local-only artifact status:** Outputs, evaluator, CSV, documentation, and qualitative comparison are public reproduction artifacts. The pretrained checkpoint remains local-only.

## EXP-003 — Fresh sub_B controlled experiment

- **Experiment ID:** EXP-003
- **Objective:** Test whether restoring the author's 5-process input topology for `sub_syn_B`, and thereby changing cross-image process state, explains the main reproduction gap.
- **Input:** `test_samples/HSTS_256x256/sub_syn_B/1381.jpg` and `3146.jpg`.
- **Command:**

  ```bash
  python sample_dehazing.py \
    --sample_dir "test_samples/HSTS_256x256/sub_syn_B" \
    --result_dir "results_subB_fresh"
  ```

- **Environment:** Common environment above; one fresh Python process for this two-image split.
- **Important configuration:** The same source, checkpoint, seed, FP16 setting, and default dehazing parameters as EXP-002.
- **Output artifacts:** `results_subB_fresh/1381.jpg`, `results_subB_fresh/3146.jpg`.
- **Quantitative result:**

  | Image | Author PSNR | Batch10 PSNR | Fresh B PSNR |
  |---|---:|---:|---:|
  | 1381 | 19.7618 | 12.3171 | 11.6284 |
  | 3146 | 18.3737 | 13.2259 | 12.7284 |

- **Observation:** Changing the process boundary changed the outputs, but the Fresh B scores did not approach the author-provided results and were slightly lower than Batch10.
- **Conclusion:** Process history affects the observed result, but restoring this author-style process boundary does not explain the main 5–8 dB Author-vs-Ours gap. The experiment did not isolate optimizer reset as a single causal variable.
- **Public / local-only artifact status:** The two Fresh B outputs are tracked public audit artifacts.

## EXP-004 — Fresh sub_B repeatability

- **Experiment ID:** EXP-004
- **Objective:** Test run-to-run reproducibility in the current environment using the same inputs, source, parameters, seed, checkpoint, and a second fresh Python process.
- **Input:** `test_samples/HSTS_256x256/sub_syn_B/1381.jpg` and `3146.jpg`.
- **Command:**

  ```bash
  python sample_dehazing.py \
    --sample_dir "test_samples/HSTS_256x256/sub_syn_B" \
    --result_dir "results_subB_fresh_repeat"
  ```

- **Environment:** Common environment above; a separate fresh Python process from EXP-003.
- **Important configuration:** Identical to EXP-003.
- **Output artifacts:** `results_subB_fresh_repeat/1381.jpg`, `results_subB_fresh_repeat/3146.jpg`.
- **Quantitative result:**

  | Image | SHA-256 shared by Fresh and Repeat | Cross-run PSNR | Cross-run SSIM |
  |---|---|---:|---:|
  | 1381.jpg | `A34B073707F59B9C60F33CF00D431475EA8F49B2E0DA99DAB5B7F50073C18C17` | ∞ | 1.0 |
  | 3146.jpg | `177D6A7E3D68CAC4B38FE2C81A932BA3857A4F662D9D5F384DAA02B1EEA3E13B` | ∞ | 1.0 |

- **Observation:** Both output pairs are byte-identical, and their metrics relative to GT are also identical.
- **Conclusion:** The tested Fresh B runs are byte-level reproducible in the current environment. The Author-vs-Ours gap is not explained by ordinary run-to-run randomness under these conditions.
- **Public / local-only artifact status:** Both repeat outputs and their checksums are tracked public audit artifacts.

## Logging Policy for Future Research

Future image restoration and TTA experiments should preserve the following as a minimum experiment record:

- Experiment ID
- Git commit
- Exact command
- Complete configuration and resolved defaults
- Seed and RNG policy
- Operating system, Python, framework, CUDA, cuDNN, driver, and GPU
- Checkpoint filename, source, size, and cryptographic hash
- Dataset name, version, preprocessing, and tested file manifest
- Raw stdout/stderr log
- Metrics CSV and exact evaluation protocol
- WandB/TensorBoard run identifier and export, if used
- Result images and artifact checksums
- Observation, conclusion, failure state, and known limitations
