# TAO Dehazing Reproducibility Manifest

This manifest records only information verified from the current environment, local artifacts, source tree, Git objects, and remote repository state. It does not claim parity with runtime details that the original authors did not publish.

## Repository Provenance

- Publication repository: [`wanyi7534-wq/2024-ICML-TAO`](https://github.com/wanyi7534-wq/2024-ICML-TAO)
- Repository lineage/base audited during reproduction: [`XLearning-SCU/2024-ICML-TAO`](https://github.com/XLearning-SCU/2024-ICML-TAO)
- Original TAO repository attributed in the current README and configured as `upstream`: [`YBGou/2024-ICML-TAO`](https://github.com/YBGou/2024-ICML-TAO)
- Original TAO authors: Yuanbiao Gou, Haiyu Zhao, Boyun Li, Xinyan Xiao, and Xi Peng

The TAO method and upstream source are the original authors' work. The additions in this fork are a reproduction and empirical audit.

## Relevant Git Commits

| Commit | Verified role |
|---|---|
| `3ef198ac85624853ffa7f7f61b447344ade374e9` | Upstream base used before the reproduction commits |
| `681f70948aaba0f35ead43d1439becfc5cf16fdd` | Restore upstream `gen_dif_pri` ordinary source tree |
| `7a4e76ec06fa61d9bce781d51f380fbabadbfd72` | TAO dehazing reproduction study |
| `9de6173479f1d76f81f003c70c6dbe6f7786ac8b` | Official historical source used to restore `gen_dif_pri/` |

At the time of the completeness audit, local `main` and `origin/main` both resolved to `7a4e76ec06fa61d9bce781d51f380fbabadbfd72` before this manifest commit was created.

## Reproduction Environment

| Component | Verified value |
|---|---|
| Operating system | Windows |
| Conda environment | `tao` (`D:\miniconda3\envs\tao`) |
| Python | 3.10.20 |
| PyTorch | 2.7.1+cu128 |
| torchvision | 0.22.1+cu128 |
| PyTorch CUDA runtime | 12.8 |
| cuDNN | 9.7.1 (`torch.backends.cudnn.version() == 90701`) |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU |
| NVIDIA driver | 582.05 |
| `nvidia-smi` driver capability | CUDA 13.0 |
| NumPy | 2.2.6 |
| SciPy | 1.15.3 |
| scikit-image | 0.25.2 |

PyTorch's CUDA runtime version and the maximum CUDA version reported by `nvidia-smi` describe different layers and are therefore both recorded.

## Diffusion Checkpoint

- Path: `test_models/256x256_diffusion_uncond.pt`
- Status: local-only; ignored by Git and not committed to GitHub
- Size: `2,211,383,297` bytes
- SHA-256: `A37C32FFFD316CD494CF3F35B339936DEBDC1576DAD13FE57C42399A5DBC78B1`
- Source described by upstream README: pretrained unconditional ImageNet-256 DDPM from OpenAI guided-diffusion

The repository does not provide the checksum of the exact checkpoint used for the authors' committed results. Therefore, this manifest cannot claim that the local checkpoint is byte-identical to the authors' runtime checkpoint.

## VGG Perceptual Backbone

Source inspection of `sample_dehazing.py` verifies:

- API: `torchvision.models.vgg16`
- Weights enum: `VGG16_Weights.IMAGENET1K_V1`
- Feature slice: `features[:9]`

A local torchvision cache file existed and was inspected without downloading:

- Cache file: `%USERPROFILE%/.cache/torch/hub/checkpoints/vgg16-397923af.pth`
- Size: `553,433,881` bytes
- SHA-256: `397923AF8E79CDBB6A7127F12361ACD7A2F83E06B05044DDF496E83DE57A5BF0`
- Status: local-only; not tracked by this repository

## Data and Execution Scope

- Dataset: processed HSTS at 256×256
- Tested hazy inputs: 10 JPEG images under `test_samples/HSTS_256x256/synthetic/`
- Clean references: corresponding images under `test_samples/HSTS_256x256/original/`
- Author-provided images: `test_samples/HSTS_256x256/results/`
- Our Batch10 outputs: `results_batch_ours/`
- Controlled Fresh B outputs: `results_subB_fresh/`
- Repeatability outputs: `results_subB_fresh_repeat/`
- Pretrained diffusion prior was not retrained.
- Clean GT was not used by the restoration pipeline; it was used only for post-hoc evaluation.

Key resolved defaults for the Batch10 run were `batch_size=1`, `inference_num=2`, `guidance_scale=25000`, `diffusion_steps=1000`, `use_fp16=True`, and per-image `seed=10`.

## Evaluation Protocol

- Evaluator: `evaluate_reproduction.py`
- Image representation: PIL-converted RGB uint8 arrays
- PSNR: `skimage.metrics.peak_signal_noise_ratio`, `data_range=255`
- SSIM: `skimage.metrics.structural_similarity`, `data_range=255`, `channel_axis=2`
- Detailed values: `reproduction_metrics.csv`

This custom unified RGB evaluator provides an internally consistent comparison among hazy inputs, author-provided images, and our outputs. It is **not identical to a verified official paper SSIM protocol** because the official IQA implementation is absent from the audited working tree.

## Artifact Integrity

- Tracked artifact checksums: `reproduction_checksums.sha256`
- Detailed experiment record: `REPRODUCTION.md`
- Retrospective ledger: `EXPERIMENT_LOG.md`

The checkpoint hash is documented separately above and is intentionally excluded from the tracked artifact checksum file.

## Known Provenance Limits

- The exact environment, checkpoint checksum, GPU, and complete runtime logs used to generate the author-provided images are unavailable.
- Initial reproduction runs did not preserve complete raw terminal stdout.
- Cross-environment numerical effects remain hypotheses rather than isolated causes of the reproduction gap.
- Fresh B process-boundary experiments did not isolate optimizer reset as a single independent variable.
- Byte-level repeatability was verified for the two Fresh B images in the current environment, not universally across all platforms or inputs.
