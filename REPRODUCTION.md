# TAO Dehazing Reproduction

本文档记录对论文 *Test-Time Degradation Adaptation for Open-Set Image Restoration*（TAO，ICML 2024）中 image dehazing 实验的一次实际复现。除特别标注为 hypothesis 或 limitation 的内容外，本文仅陈述当前仓库源码、已有产物和已完成实验能够支持的事实。

## 1. Objective

本次实验的目标是：

- 理解并复现 TAO 在 image dehazing 上的 test-time adaptation pipeline；
- 验证公开代码能否在当前 Windows 环境运行；
- 使用统一评价程序，定量比较 hazy input、author-provided TAO results 和 our reproduction；
- 调查 author-provided results 与复现输出之间的 reproduction gap；
- 为后续 image restoration + test-time adaptation（TTA）研究提炼可复用的工程与实验设计经验。

## 2. Environment

### 2.1 Reproduction environment

以下版本从本次实际使用的 `tao` conda environment 中读取：

| Component | Version |
|---|---|
| Operating system | Windows |
| Conda environment | `tao` |
| Python | 3.10.20 |
| PyTorch | 2.7.1+cu128 |
| torchvision | 0.22.1+cu128 |
| CUDA runtime reported by PyTorch | 12.8 |
| cuDNN reported by PyTorch | 9.7.1 |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU |
| NumPy | 2.2.6 |
| SciPy | 1.15.3 |
| scikit-image | 0.25.2 |

### 2.2 Author-reported environment

The repository [README](README.md) states that the code was tested with:

| Component | Version |
|---|---|
| Operating system | Ubuntu 20.04 |
| PyTorch | 2.0.1 |
| torchvision | 0.15.2 (`requirements.txt`) |
| CUDA | 11.7 |
| cuDNN package | 8.5.0.96 (`requirements.txt`) |
| SciPy | 1.11.2 (`requirements.txt`) |
| scikit-image | 0.22.0 (`requirements.txt`) |

两套环境并不完全一致。仓库没有可靠指定作者所用 Python 小版本或具体 GPU，因此本文不对这些信息作推测。

## 3. Dataset and File Roles

HSTS 256×256 数据与实验产物的角色如下：

| Relative path | Role | Used during restoration | Used during evaluation |
|---|---|---:|---:|
| `test_samples/HSTS_256x256/synthetic/` | Hazy input | Yes | Yes |
| `test_samples/HSTS_256x256/original/` | Clean ground truth | No | Yes, as reference |
| `test_samples/HSTS_256x256/results/` | Author-provided TAO restoration | No | Yes |
| `results_batch_ours/` | Our 10-image reproduction | Output | Yes |

`sub_syn_A`–`sub_syn_E` 是 HSTS hazy inputs 的五个输入分片。每组包含两张图片；本次逐文件 SHA-256 核验确认，分片文件与 `synthetic/` 中同名文件逐字节相同：

| Split | Images |
|---|---|
| `sub_syn_A` | `0586.jpg`, `1352.jpg` |
| `sub_syn_B` | `1381.jpg`, `3146.jpg` |
| `sub_syn_C` | `4184.jpg`, `4561.jpg` |
| `sub_syn_D` | `5576.jpg`, `5920.jpg` |
| `sub_syn_E` | `7471.jpg`, `8180.jpg` |

[`tta_scripts.sh`](tta_scripts.sh) 为 A–E 分片分别启动后台 Python process，并在最后执行 `wait`；因此作者脚本表达的是 **5 independent Python processes × 2 images**。Clean GT 不传入 [`sample_dehazing.py`](sample_dehazing.py) 的 restoration pipeline，只在事后 PSNR/SSIM evaluation 中使用。

## 4. Reproduction Pipeline

当前 dehazing 数据流可概括为：

```text
Hazy input
→ ImageFolderDataset
→ pretrained unconditional diffusion prior
→ current clean estimate
→ Test-time Degradation Adapter (GenerativeDegradation)
→ predicted degraded counterpart
→ AIR loss guidance
→ guided reverse diffusion
→ restored image
```

具体而言：

1. [`ImageFolderDataset`](gen_dif_pri/scripts/imagenet_dataloader/imagenet_dataset.py) 从 `--sample_dir` 读取 hazy image。
2. [`sample_dehazing.py`](sample_dehazing.py) 加载 `test_models/256x256_diffusion_uncond.pt`，即 unconditional ImageNet-256 diffusion prior。此次复现使用预训练权重，没有重新训练 PDM。
3. Reverse diffusion 给出当前 clean estimate；`GenerativeDegradation` 将其映射为 predicted degraded counterpart。
4. TDA 在 test time 在线更新 degradation generator，并通过其内部 discriminator 计算 adversarial component。
5. Predicted degraded counterpart 与观测 hazy input 之间的 AIR losses 产生梯度 guidance，作用于 guided reverse diffusion。
6. 最终样本被映射到 uint8 RGB，并保存至 `--result_dir`。Clean GT 在以上流程中不可见。

本次 10-image reproduction 使用的命令为：

```powershell
python sample_dehazing.py --sample_dir "test_samples/HSTS_256x256/synthetic" --result_dir "results_batch_ours"
```

未在命令行覆盖的关键代码默认值包括：`batch_size=1`、`inference_num=2`、`guidance_scale=25000`、`diffusion_steps=1000` 和 `use_fp16=True`。每个 dataloader iteration 都调用 `init_seed(seed=10)`。代码生成两个 candidates，但只保存前 `batch_size` 个，即当前配置下只保存第一个 candidate。

## 5. Quantitative Results

评价程序为 [`evaluate_reproduction.py`](evaluate_reproduction.py)，逐图结果保存在 [`reproduction_metrics.csv`](reproduction_metrics.csv)。三组图像均以 clean GT 为 reference，并经过同一流程：PIL 转 RGB uint8，随后调用 `skimage.metrics.peak_signal_noise_ratio` 与 `structural_similarity(channel_axis=2, data_range=255)`。

### 5.1 Aggregate results

| Method | PSNR (dB) | SSIM |
|---|---:|---:|
| Hazy Input | 14.7446 | 0.753592 |
| Author TAO | 22.3276 | 0.856213 |
| Our Reproduction | 18.7119 | 0.797267 |

- Our Reproduction vs Hazy Input: **+3.9673 dB PSNR**, **+0.043675 SSIM**.
- Author TAO vs Our Reproduction: approximately **+3.6157 dB PSNR**, **+0.058946 SSIM**.

> **Metric protocol note:** 这些指标适合三组结果之间的内部公平比较，但不能直接声称等于论文官方 SSIM protocol。当前仓库的 `img_qua_ass/` 为空，无法在本地复核 README 所述官方 IQA implementation。另需注意，CSV 将逐图指标截取为六位小数；直接对这些已舍入行重新求均值时，Author SSIM 为 `0.8562135`，显示到六位小数可能成为 `0.856214`。上表保留实验运行时记录的 aggregate value `0.856213`，该 `1e-6` 级舍入差异不影响结论。

### 5.2 Per-image results

| Image | Hazy PSNR | Hazy SSIM | Author PSNR | Author SSIM | Ours PSNR | Ours SSIM | Author − Ours PSNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0586 | 17.2157 | 0.847935 | 28.1876 | 0.931393 | 24.7379 | 0.899478 | +3.4497 |
| 1352 | 18.0409 | 0.868745 | 28.4338 | 0.935294 | 21.4073 | 0.870731 | +7.0265 |
| 1381 | 10.4681 | 0.574031 | 19.7618 | 0.773161 | 12.3171 | 0.607088 | +7.4447 |
| 3146 | 11.9834 | 0.656183 | 18.3737 | 0.779909 | 13.2259 | 0.672351 | +5.1478 |
| 4184 | 11.6026 | 0.657234 | 19.0823 | 0.824037 | 16.0838 | 0.774011 | +2.9986 |
| 4561 | 14.2284 | 0.758610 | 16.5636 | 0.811783 | 17.1468 | 0.801929 | −0.5832 |
| 5576 | 12.5440 | 0.634467 | 20.4426 | 0.764551 | 15.7262 | 0.697040 | +4.7164 |
| 5920 | 16.2267 | 0.825860 | 26.4240 | 0.920414 | 26.6925 | 0.916933 | −0.2685 |
| 7471 | 20.7878 | 0.912758 | 25.3248 | 0.941011 | 21.5444 | 0.903205 | +3.7804 |
| 8180 | 14.3489 | 0.800098 | 20.6819 | 0.880582 | 18.2375 | 0.829904 | +2.4444 |

负的 `Author − Ours PSNR` 表示 our reproduction 的 PSNR 更高；它不表示所有视觉属性或其他 IQA 指标都更优。

## 6. Qualitative Comparison

下图四列依次为 **Hazy Input | Our TAO | Author TAO | Ground Truth**：

![TAO dehazing qualitative comparison](reproduction_assets/tao_dehazing_comparison_4x4.png)

图中选择 `1381`、`3146`、`4561` 和 `5920`：前两张在统一 evaluator 下具有明显 PSNR reproduction gap；后两张的 our PSNR 分别比 author-provided result 高约 0.58 dB 和 0.27 dB。该图用于展示逐图行为并不一致，不据此扩展为超出指标支持范围的整体视觉质量结论。生成脚本见 [`reproduction_assets/make_dehazing_comparison.py`](reproduction_assets/make_dehazing_comparison.py)。

## 7. Reproduction Gap Investigation

### 7.1 Initial hypothesis: cross-image optimizer state

源码审计得到以下状态生命周期：

- `GANLoss.__init__` 和 `GenerativeDegradation.__init__` 在进入 dataloader loop 前分别创建 Adam optimizer。
- 每张图片开始时，`init_gene_net()` 重新初始化 degradation generator weights，并同时重新初始化 `GenerativeDegradation` 内部 discriminator weights。
- 上述两个 Adam optimizer 没有在每张图片开始时重新创建或清空，因此其 moments 可沿同一 Python process 跨图片保留。
- 脚本还创建了一个独立的 AIR `adversarial_loss` 并逐图重置其 discriminator weights；但当前 guidance 代码令 `adv = 0`，对应的 forward/update 被注释。因此，不能把这个当前未启用的 standalone guidance discriminator 与 TDA 内部、实际参与更新的 discriminator 混为一谈。

作者脚本表达的执行方式为 **5 independent Python processes × 2 images**；我们的初次批量复现为 **1 Python process × 10 images**。据此提出 cross-image optimizer state / process-history dependency hypothesis。这里使用“state dependency”，不将其直接称作 optimizer bug。

### 7.2 Controlled Fresh sub_B Experiment

随后用 fresh Python process 单独运行 `sub_syn_B`（`1381.jpg`、`3146.jpg`），使 process boundary 与作者分片方式一致：

| Image | Author PSNR | Batch10 PSNR | Fresh B PSNR | Fresh B − Batch10 |
|---|---:|---:|---:|---:|
| 1381 | 19.7618 | 12.3171 | 11.6284 | −0.6886 |
| 3146 | 18.3737 | 13.2259 | 12.7284 | −0.4975 |

恢复 process boundary 后，Fresh B 没有接近 author-provided results，反而比 Batch10 略低。这表明 process history 会影响当前实现的输出，并与源码中跨图保留 Adam state 的机制一致；但本实验没有执行“只重置 optimizer、其他状态完全不变”的单因素消融，因此不能把观察到的全部变化严格归因于 Adam moments。

无论采用更强或更弱的因果措辞，该实验都足以排除“仅把 10 张拆成作者的 2-image process 就能消除 5–8 dB 逐图 gap”这一解释。Cross-image state 可能贡献部分差异，但不是已观察到的大幅 Author vs Ours gap 的充分解释。

## 8. Reproducibility Check

Fresh `sub_syn_B` 在第二个全新 Python process 中以相同代码、参数和 seed 再运行一次，输出到独立目录。结果如下：

| Image | Fresh B SHA-256 | Repeat SHA-256 | Byte-identical | Cross-run PSNR | Cross-run SSIM | Repeat vs GT PSNR | Repeat vs GT SSIM |
|---|---|---|---:|---:|---:|---:|---:|
| 1381.jpg | `A34B0737…73C18C17` | `A34B0737…73C18C17` | Yes | ∞ | 1.0 | 11.628415 | 0.573495 |
| 3146.jpg | `177D6A7E…EEA3E13B` | `177D6A7E…EEA3E13B` | Yes | ∞ | 1.0 | 12.728378 | 0.660774 |

完整 SHA-256：

- `1381.jpg`: `A34B073707F59B9C60F33CF00D431475EA8F49B2E0DA99DAB5B7F50073C18C17`
- `3146.jpg`: `177D6A7E3D68CAC4B38FE2C81A932BA3857A4F662D9D5F384DAA02B1EEA3E13B`

两次输出相对于 GT 的指标完全一致。由此可知，在当前 Windows + `tao` 环境中，相同代码、参数、seed 和 fresh Python process 条件下，这两张图具有逐字节可重复性。因此 Author vs Ours gap 不能解释为当前环境中的普通 run-to-run randomness。

## 9. Remaining Possible Causes

以下均为 **remaining hypotheses，而非已经证明的原因**：

1. **Cross-environment stochastic diffusion / FP16 numerical trajectory differences.** 当前代码的 reverse diffusion、online optimization 与 guidance 构成闭环；不同初始或中间数值轨迹可能被后续迭代放大。其实际贡献尚未通过 matched-environment experiment 测量。
2. **PyTorch/CUDA/cuDNN/GPU implementation differences.** 两套软件栈和 GPU 平台确实不同，但“环境不同”本身不能证明它造成了当前 3–7 dB 的逐图差距。
3. **Author result generation provenance is incomplete.** README 说明已提供结果并称论文结果通过 `tta_scripts.sh` 获得，但仓库没有保存逐图生成日志、完整 runtime metadata 或输出 checksum，无法端到端证明 `results/` 的精确生成过程。
4. **Exact checkpoint/runtime asset identity cannot be fully verified.** 本地 checkpoint 可以识别和校验，但缺少作者运行时 checkpoint checksum 与所有 runtime assets 的完整记录，无法证明二者逐字节一致。

这些限制不构成“作者结果有问题”或“作者使用隐藏参数”的证据。README 明确说明针对 degradation type 用 representative images 调整 loss weights 和 guidance scale，并称仓库提供了论文所用 degradations 的参数；当前调查无法进一步验证 committed outputs 的生成 provenance。

## 10. Key Technical Findings

- TAO performs actual online optimization at test time；它不是固定网络的一次普通 forward inference。
- TDA learns degradation behavior online，并将当前 clean estimate 映射回观测 degradation domain。
- AIR 在不使用 clean GT 的条件下，以 adapter-predicted degradation consistency 引导 reverse diffusion。
- 当 Python process 被多个样本复用时，test-time optimizer state 可能依赖先前样本。
- Diffusion sampling 与 online adaptation 形成数值敏感的 closed-loop system。
- 严谨复现需要记录 process boundaries、RNG/seed policy、environment、checkpoint identity、candidate policy 和 evaluation protocol。

## 11. Relevance to Future Restoration + TTA Research

以下是由本次复现提炼出的 future research directions / lessons，不代表已经实现的方法：

- 明确定义 episodic TTA 与 continual TTA，并使 state lifecycle 与问题设定一致；
- 将 optimizer reset policy 作为显式实验变量，而不只重置 model weights；
- 探索 confidence-aware adaptation，减少不可靠 pseudo supervision 的影响；
- 研究如何检测和阻止 harmful test-time updates；
- 评估 teacher/student 或 temporal stabilization 是否能降低在线优化震荡；
- 同时报告 restoration quality 与 downstream-task performance，避免只优化单一像素指标；
- 保存 seeds、process topology、软件栈、checkpoint hash、逐样本日志和输出 hash，以提高 reproducibility。

## 12. Limitations

- 当前 Windows/PyTorch 2.7.1/CUDA 12.8 环境与作者报告的 Ubuntu/PyTorch 2.0.1/CUDA 11.7 环境不同。
- Our reproduction 改善了 hazy baseline，但没有完全复现 author-provided committed outputs。
- Author results 的 exact provenance、逐图生成日志和 runtime state 不可完全验证。
- 当前统一 RGB SSIM evaluator 与 README 引用的官方 IQA implementation 不完全一致；当前 `img_qua_ass/` 为空，无法进行 protocol parity check。
- 没有为了追逐 committed metrics 而大规模 sweep hyperparameters。
- 没有重新训练 pretrained diffusion prior。
- Fresh B experiment 改变了整个 process boundary；未执行只重置 optimizer state 的单因素 ablation。
- 可重复性验证覆盖 `sub_syn_B` 的两张图，不能自动推广为所有硬件、软件栈或数据的普遍确定性保证。

## 13. Artifacts

以下产物均已在当前工作区中确认存在：

| Artifact | Purpose |
|---|---|
| [`evaluate_reproduction.py`](evaluate_reproduction.py) | Unified RGB PSNR/SSIM evaluator |
| [`reproduction_metrics.csv`](reproduction_metrics.csv) | 10-image per-method quantitative results |
| [`reproduction_assets/tao_dehazing_comparison_4x4.png`](reproduction_assets/tao_dehazing_comparison_4x4.png) | Four-by-four qualitative comparison figure |
| [`reproduction_assets/make_dehazing_comparison.py`](reproduction_assets/make_dehazing_comparison.py) | Reproducible figure-generation script |
| [`results_batch_ours/`](results_batch_ours/) | Our 10-image reproduction outputs |
| [`results_subB_fresh/`](results_subB_fresh/) | First controlled Fresh B outputs |
| [`results_subB_fresh_repeat/`](results_subB_fresh_repeat/) | Independent repeatability-check outputs |

Runtime dependencies that exist locally but are not themselves experimental results include `test_models/256x256_diffusion_uncond.pt` and the restored `gen_dif_pri/` source tree. The checkpoint is approximately 2.21 GB and should not be committed to ordinary Git history without an explicit artifact-management decision.
