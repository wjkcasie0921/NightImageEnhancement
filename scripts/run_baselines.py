import argparse
import csv
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
RESIZE_HW = (320, 320)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run traditional low-light baselines (HE/MSRCR/Gamma) and export PSNR/SSIM CSV."
    )
    parser.add_argument("--low_dir", type=str, required=True, help="Low-light image directory.")
    parser.add_argument("--normal_dir", type=str, required=True, help="Normal-light (GT) image directory.")
    parser.add_argument("--output_csv", type=str, required=True, help="Output CSV path.")
    return parser.parse_args()


def collect_images(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def build_stem_lookup(paths: List[Path]) -> Dict[str, Path]:
    lookup: Dict[str, Path] = {}
    for p in paths:
        if p.stem not in lookup:
            lookup[p.stem] = p
    return lookup


def pair_images(low_dir: Path, normal_dir: Path) -> List[Tuple[Path, Path]]:
    low_paths = collect_images(low_dir)
    normal_paths = collect_images(normal_dir)
    if len(low_paths) == 0:
        raise RuntimeError(f"No low-light images found in: {low_dir}")
    if len(normal_paths) == 0:
        raise RuntimeError(f"No normal-light images found in: {normal_dir}")

    normal_lookup = build_stem_lookup(normal_paths)
    pairs: List[Tuple[Path, Path]] = []
    missing = 0
    for low_path in low_paths:
        gt_path = normal_lookup.get(low_path.stem)
        if gt_path is None:
            missing += 1
            continue
        pairs.append((low_path, gt_path))

    if len(pairs) == 0:
        raise RuntimeError("No filename-stem pairs found. Please check naming consistency.")

    if missing > 0:
        warnings.warn(f"{missing} low-light images have no matching normal image and were skipped.")
    return pairs


def load_rgb_uint8(path: Path, resize_hw: Tuple[int, int] = RESIZE_HW) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    img = img.resize((resize_hw[1], resize_hw[0]), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def to_float01(img_uint8: np.ndarray) -> np.ndarray:
    return img_uint8.astype(np.float32) / 255.0


def compute_metrics(pred_uint8: np.ndarray, gt_uint8: np.ndarray) -> Tuple[float, float]:
    pred = to_float01(pred_uint8)
    gt = to_float01(gt_uint8)
    psnr = peak_signal_noise_ratio(gt, pred, data_range=1.0)
    ssim = structural_similarity(gt, pred, data_range=1.0, channel_axis=2)
    return float(psnr), float(ssim)


def he_yuv_rgb(low_rgb: np.ndarray) -> np.ndarray:
    yuv = cv2.cvtColor(low_rgb, cv2.COLOR_RGB2YUV)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    out_rgb = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
    return out_rgb


def gamma_correction(low_rgb: np.ndarray, gamma: float = 0.7) -> np.ndarray:
    x = low_rgb.astype(np.float32) / 255.0
    y = np.power(np.clip(x, 0.0, 1.0), gamma)
    return np.clip(y * 255.0, 0.0, 255.0).astype(np.uint8)


def init_msrcr_processor() -> Tuple[str, Optional[object]]:
    # Priority 1: cv2.createMSRCR
    if hasattr(cv2, "createMSRCR"):
        try:
            return "opencv_factory", cv2.createMSRCR()
        except Exception as e:
            warnings.warn(f"cv2.createMSRCR exists but initialization failed: {e}")

    # Priority 2: cv2.xphoto.createMsrCR
    xphoto = getattr(cv2, "xphoto", None)
    if xphoto is not None and hasattr(xphoto, "createMsrCR"):
        try:
            return "opencv_xphoto", xphoto.createMsrCR()
        except Exception as e:
            warnings.warn(f"cv2.xphoto.createMsrCR exists but initialization failed: {e}")

    warnings.warn(
        "MSRCR factory is not exposed in current OpenCV build. "
        "Falling back to numpy/OpenCV implementation."
    )
    return "fallback_numpy", None


def msrcr_fallback(
    low_rgb: np.ndarray,
    sigmas: Tuple[float, float, float] = (15.0, 80.0, 250.0),
    gain: float = 5.0,
    bias: float = 25.0,
    alpha: float = 125.0,
    beta: float = 46.0,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Fallback MSRCR approximation using Multi-Scale Retinex + color restoration.
    """
    img = low_rgb.astype(np.float32) + 1.0
    img = np.maximum(img, eps)

    # Multi-scale Retinex in log domain.
    retinex = np.zeros_like(img, dtype=np.float32)
    for sigma in sigmas:
        blur = cv2.GaussianBlur(img, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma)
        blur = np.maximum(blur, eps)
        retinex += np.log(img) - np.log(blur)
    retinex /= float(len(sigmas))

    # Color restoration term.
    intensity = np.sum(img, axis=2, keepdims=True)
    intensity = np.maximum(intensity, eps)
    color_restore = beta * (np.log(alpha * img) - np.log(intensity))

    msrcr = gain * (retinex * color_restore + bias)

    # Robust per-channel normalization to uint8.
    out = np.zeros_like(msrcr, dtype=np.uint8)
    for c in range(3):
        ch = msrcr[:, :, c]
        lo = np.percentile(ch, 1.0)
        hi = np.percentile(ch, 99.0)
        if hi <= lo:
            ch_n = np.clip(ch, 0.0, 255.0)
        else:
            ch_n = (ch - lo) * 255.0 / (hi - lo)
        out[:, :, c] = np.clip(ch_n, 0.0, 255.0).astype(np.uint8)
    return out


def run_msrcr(low_rgb: np.ndarray, backend: str, processor: Optional[object]) -> np.ndarray:
    if backend == "fallback_numpy":
        return msrcr_fallback(low_rgb)
    if processor is None:
        raise RuntimeError("MSRCR processor is not initialized.")

    # OpenCV algorithms usually expect BGR input.
    low_bgr = cv2.cvtColor(low_rgb, cv2.COLOR_RGB2BGR)
    out = processor.process(low_bgr)
    if out is None:
        raise RuntimeError("MSRCR processor returned None.")

    out = np.asarray(out)
    if out.dtype != np.uint8:
        out = np.clip(out, 0.0, 255.0).astype(np.uint8)
    if out.ndim != 3 or out.shape[2] != 3:
        raise RuntimeError(f"Unexpected MSRCR output shape: {out.shape}")

    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def nan_mean_std(values: List[float]) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan"), float("nan")
    return float(np.nanmean(arr)), float(np.nanstd(arr))


def fmt(v: float) -> str:
    return "nan" if np.isnan(v) else f"{v:.6f}"


def main() -> None:
    args = parse_args()
    low_dir = Path(args.low_dir)
    normal_dir = Path(args.normal_dir)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    pairs = pair_images(low_dir, normal_dir)
    msrcr_backend, msrcr_processor = init_msrcr_processor()

    rows = []
    he_psnr_vals: List[float] = []
    he_ssim_vals: List[float] = []
    msrcr_psnr_vals: List[float] = []
    msrcr_ssim_vals: List[float] = []
    gamma_psnr_vals: List[float] = []
    gamma_ssim_vals: List[float] = []

    for low_path, gt_path in pairs:
        low_rgb = load_rgb_uint8(low_path, RESIZE_HW)
        gt_rgb = load_rgb_uint8(gt_path, RESIZE_HW)

        # HE baseline
        he_rgb = he_yuv_rgb(low_rgb)
        he_psnr, he_ssim = compute_metrics(he_rgb, gt_rgb)
        he_psnr_vals.append(he_psnr)
        he_ssim_vals.append(he_ssim)

        # Gamma baseline
        gamma_rgb = gamma_correction(low_rgb, gamma=0.7)
        gamma_psnr, gamma_ssim = compute_metrics(gamma_rgb, gt_rgb)
        gamma_psnr_vals.append(gamma_psnr)
        gamma_ssim_vals.append(gamma_ssim)

        # MSRCR baseline (optional)
        msrcr_psnr = float("nan")
        msrcr_ssim = float("nan")
        try:
            msrcr_rgb = run_msrcr(low_rgb, msrcr_backend, msrcr_processor)
            msrcr_psnr, msrcr_ssim = compute_metrics(msrcr_rgb, gt_rgb)
        except Exception as e:
            warnings.warn(f"MSRCR failed on {low_path.name}, set to NaN: {e}")
        msrcr_psnr_vals.append(msrcr_psnr)
        msrcr_ssim_vals.append(msrcr_ssim)

        rows.append(
            {
                "image_name": low_path.name,
                "HE_PSNR": he_psnr,
                "HE_SSIM": he_ssim,
                "MSRCR_PSNR": msrcr_psnr,
                "MSRCR_SSIM": msrcr_ssim,
                "Gamma_PSNR": gamma_psnr,
                "Gamma_SSIM": gamma_ssim,
            }
        )

    he_psnr_mean, he_psnr_std = nan_mean_std(he_psnr_vals)
    he_ssim_mean, he_ssim_std = nan_mean_std(he_ssim_vals)
    msrcr_psnr_mean, msrcr_psnr_std = nan_mean_std(msrcr_psnr_vals)
    msrcr_ssim_mean, msrcr_ssim_std = nan_mean_std(msrcr_ssim_vals)
    gamma_psnr_mean, gamma_psnr_std = nan_mean_std(gamma_psnr_vals)
    gamma_ssim_mean, gamma_ssim_std = nan_mean_std(gamma_ssim_vals)

    fieldnames = [
        "image_name",
        "HE_PSNR",
        "HE_SSIM",
        "MSRCR_PSNR",
        "MSRCR_SSIM",
        "Gamma_PSNR",
        "Gamma_SSIM",
    ]

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "image_name": r["image_name"],
                    "HE_PSNR": fmt(r["HE_PSNR"]),
                    "HE_SSIM": fmt(r["HE_SSIM"]),
                    "MSRCR_PSNR": fmt(r["MSRCR_PSNR"]),
                    "MSRCR_SSIM": fmt(r["MSRCR_SSIM"]),
                    "Gamma_PSNR": fmt(r["Gamma_PSNR"]),
                    "Gamma_SSIM": fmt(r["Gamma_SSIM"]),
                }
            )

        writer.writerow(
            {
                "image_name": "AVG",
                "HE_PSNR": fmt(he_psnr_mean),
                "HE_SSIM": fmt(he_ssim_mean),
                "MSRCR_PSNR": fmt(msrcr_psnr_mean),
                "MSRCR_SSIM": fmt(msrcr_ssim_mean),
                "Gamma_PSNR": fmt(gamma_psnr_mean),
                "Gamma_SSIM": fmt(gamma_ssim_mean),
            }
        )
        writer.writerow(
            {
                "image_name": "STD",
                "HE_PSNR": fmt(he_psnr_std),
                "HE_SSIM": fmt(he_ssim_std),
                "MSRCR_PSNR": fmt(msrcr_psnr_std),
                "MSRCR_SSIM": fmt(msrcr_ssim_std),
                "Gamma_PSNR": fmt(gamma_psnr_std),
                "Gamma_SSIM": fmt(gamma_ssim_std),
            }
        )

    print("Baseline evaluation finished.")
    print(f"Paired images: {len(rows)}")
    print(f"CSV saved to: {output_csv}")
    print(f"MSRCR backend: {msrcr_backend}")


if __name__ == "__main__":
    main()
