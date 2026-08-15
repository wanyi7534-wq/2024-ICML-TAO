import os
import csv
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity


# ============================================================
# TAO Dehazing Reproduction Evaluation
# HSTS 256x256
#
# Compare:
#   1. Hazy input
#   2. Author-provided TAO results
#   3. Our reproduced TAO results
#
# Reference:
#   Clean ground-truth images in original/
# ============================================================


GT_DIR = r"test_samples\HSTS_256x256\original"
HAZY_DIR = r"test_samples\HSTS_256x256\synthetic"
AUTHOR_DIR = r"test_samples\HSTS_256x256\results"
OURS_DIR = r"results_batch_ours"

OUTPUT_CSV = "reproduction_metrics.csv"


def load_rgb(path):
    """Load image as RGB uint8 numpy array."""
    image = Image.open(path).convert("RGB")
    return np.array(image)


def calculate_metrics(reference, prediction):
    """Calculate RGB PSNR and SSIM."""
    if reference.shape != prediction.shape:
        raise ValueError(
            f"Image shape mismatch: GT={reference.shape}, prediction={prediction.shape}"
        )

    psnr = peak_signal_noise_ratio(
        reference,
        prediction,
        data_range=255
    )

    ssim = structural_similarity(
        reference,
        prediction,
        channel_axis=2,
        data_range=255
    )

    return psnr, ssim


def evaluate_folder(name, folder, filenames):
    print("\n" + "=" * 65)
    print(f"Evaluating: {name}")
    print("=" * 65)

    rows = []
    psnr_values = []
    ssim_values = []

    for filename in filenames:

        gt_path = os.path.join(GT_DIR, filename)
        pred_path = os.path.join(folder, filename)

        if not os.path.exists(pred_path):
            print(f"[WARNING] Missing: {pred_path}")
            continue

        gt = load_rgb(gt_path)
        pred = load_rgb(pred_path)

        psnr, ssim = calculate_metrics(gt, pred)

        psnr_values.append(psnr)
        ssim_values.append(ssim)

        rows.append({
            "method": name,
            "image": filename,
            "psnr": psnr,
            "ssim": ssim
        })

        print(
            f"{filename:12s} | "
            f"PSNR = {psnr:8.4f} dB | "
            f"SSIM = {ssim:.6f}"
        )

    mean_psnr = float(np.mean(psnr_values))
    mean_ssim = float(np.mean(ssim_values))

    print("-" * 65)
    print(
        f"AVERAGE      | "
        f"PSNR = {mean_psnr:8.4f} dB | "
        f"SSIM = {mean_ssim:.6f}"
    )

    return rows, mean_psnr, mean_ssim


def main():

    filenames = sorted([
        f for f in os.listdir(GT_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    print("\nTAO Dehazing Reproduction Evaluation")
    print("====================================")
    print(f"Number of GT images: {len(filenames)}")

    all_rows = []
    summary = []

    experiments = [
        ("Hazy Input", HAZY_DIR),
        ("Author TAO", AUTHOR_DIR),
        ("Our Reproduction", OURS_DIR),
    ]

    for name, folder in experiments:

        rows, mean_psnr, mean_ssim = evaluate_folder(
            name,
            folder,
            filenames
        )

        all_rows.extend(rows)

        summary.append({
            "method": name,
            "psnr": mean_psnr,
            "ssim": mean_ssim
        })

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "Method",
            "Image",
            "PSNR_dB",
            "SSIM"
        ])

        for row in all_rows:
            writer.writerow([
                row["method"],
                row["image"],
                f"{row['psnr']:.6f}",
                f"{row['ssim']:.6f}"
            ])

    print("\n")
    print("=" * 65)
    print("FINAL SUMMARY")
    print("=" * 65)

    print(
        f"{'Method':20s} | "
        f"{'PSNR (dB)':>12s} | "
        f"{'SSIM':>10s}"
    )

    print("-" * 65)

    for item in summary:

        print(
            f"{item['method']:20s} | "
            f"{item['psnr']:12.4f} | "
            f"{item['ssim']:10.6f}"
        )

    print("=" * 65)

    print(f"\nDetailed results saved to: {OUTPUT_CSV}")
    print("\nEvaluation finished successfully.")


if __name__ == "__main__":
    main()