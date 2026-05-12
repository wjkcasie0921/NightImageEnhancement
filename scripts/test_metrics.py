import argparse
import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import sys

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.retinex_net import RetinexNetLite

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("PyYAML is required. Please install pyyaml.") from exc


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _load_yaml(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping: {cfg_path}")
    return data


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Retinex model on paired test set. "
            "Compute per-image PSNR/SSIM, then export CSV with summary."
        )
    )
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file.")
    parser.add_argument("--low_dir", type=str, default=None, help="Low-light images directory.")
    parser.add_argument("--normal_dir", type=str, default=None, help="Normal-light images directory.")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to checkpoint (retinex_lite_best.pth).")
    parser.add_argument("--output_csv", type=str, default=None, help="CSV output path.")
    parser.add_argument("--image_size", type=int, default=None, help="Inference resize resolution.")
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Backbone width override. If omitted, script tries to read width from checkpoint.",
    )
    parser.add_argument("--device", type=str, default=None, help="cuda / cpu (default: auto).")
    return parser


def _apply_config_defaults(parser: argparse.ArgumentParser, cfg: Dict[str, Any]) -> None:
    parser.set_defaults(
        low_dir=cfg.get("low_dir", "./datasets/LOL/test/low"),
        normal_dir=cfg.get("normal_dir", "./datasets/LOL/test/normal"),
        ckpt=cfg.get("ckpt", "./checkpoints/retinex_lite_best.pth"),
        output_csv=cfg.get("output_csv", "./metrics_results.csv"),
        image_size=cfg.get("image_size", 320),
        width=cfg.get("width", 24),
        device=cfg.get("device", None),
    )


def collect_images(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def build_normal_lookup(normal_paths: List[Path]) -> Dict[str, Path]:
    lookup: Dict[str, Path] = {}
    for p in normal_paths:
        if p.stem not in lookup:
            lookup[p.stem] = p
    return lookup


def make_gaussian_window(window_size: int = 11, sigma: float = 1.5, channels: int = 3) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    g = g / g.sum()
    kernel_2d = torch.outer(g, g)
    kernel_2d = kernel_2d / kernel_2d.sum()
    window = kernel_2d.view(1, 1, window_size, window_size).repeat(channels, 1, 1, 1)
    return window


def compute_psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> float:
    mse = F.mse_loss(pred, target).item()
    if mse < eps:
        return 100.0
    return 10.0 * math.log10(1.0 / mse)


def compute_ssim(pred: torch.Tensor, target: torch.Tensor, window: torch.Tensor) -> float:
    c1 = (0.01 ** 2)
    c2 = (0.03 ** 2)
    padding = window.shape[-1] // 2

    mu_x = F.conv2d(pred, window, padding=padding, groups=pred.shape[1])
    mu_y = F.conv2d(target, window, padding=padding, groups=target.shape[1])

    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.conv2d(pred * pred, window, padding=padding, groups=pred.shape[1]) - mu_x_sq
    sigma_y_sq = F.conv2d(target * target, window, padding=padding, groups=target.shape[1]) - mu_y_sq
    sigma_xy = F.conv2d(pred * target, window, padding=padding, groups=pred.shape[1]) - mu_xy

    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
    )
    return ssim_map.mean().item()


def load_model(ckpt_path: str, device: str, width_override: Optional[int] = None) -> RetinexNetLite:
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
        width = width_override if width_override is not None else ckpt.get("width", 24)
    else:
        state_dict = ckpt
        width = width_override if width_override is not None else 24

    model = RetinexNetLite(width=width).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def compute_mean_std(values: List[float]) -> Tuple[float, float]:
    if len(values) == 0:
        return 0.0, 0.0
    t = torch.tensor(values, dtype=torch.float32)
    mean = t.mean().item()
    std = t.std(unbiased=False).item()
    return mean, std


def main():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default="./configs/eval.yaml")
    pre_args, remaining = pre_parser.parse_known_args()

    cfg = _load_yaml(pre_args.config)
    parser = _build_arg_parser()
    _apply_config_defaults(parser, cfg)
    args = parser.parse_args(remaining)
    args.config = pre_args.config

    device = args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")

    low_dir = Path(args.low_dir)
    normal_dir = Path(args.normal_dir)
    if not low_dir.exists():
        raise FileNotFoundError(f"low_dir not found: {low_dir}")
    if not normal_dir.exists():
        raise FileNotFoundError(f"normal_dir not found: {normal_dir}")

    low_paths = collect_images(low_dir)
    normal_paths = collect_images(normal_dir)
    if len(low_paths) == 0:
        raise RuntimeError(f"No images found in low_dir: {low_dir}")
    if len(normal_paths) == 0:
        raise RuntimeError(f"No images found in normal_dir: {normal_dir}")

    normal_lookup = build_normal_lookup(normal_paths)
    pairs: List[Tuple[Path, Path]] = []
    missing = 0
    for low_path in low_paths:
        match = normal_lookup.get(low_path.stem, None)
        if match is None:
            missing += 1
            continue
        pairs.append((low_path, match))

    if len(pairs) == 0:
        raise RuntimeError("No paired images found by filename stem. Please check naming consistency.")
    if missing > 0:
        print(f"Warning: {missing} low-light images have no matching normal image and were skipped.")

    model = load_model(args.ckpt, device, args.width)

    transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
        ]
    )

    ssim_window = make_gaussian_window(window_size=11, sigma=1.5, channels=3).to(device)

    rows = []
    psnr_list: List[float] = []
    ssim_list: List[float] = []

    with torch.no_grad():
        for low_path, normal_path in tqdm(pairs, desc="Evaluating", ncols=100):
            low_img = Image.open(low_path).convert("RGB")
            normal_img = Image.open(normal_path).convert("RGB")

            low = transform(low_img).unsqueeze(0).to(device)
            normal = transform(normal_img).unsqueeze(0).to(device)

            illum, refl = model(low)
            enhanced = model.compose(illum, refl).clamp(0.0, 1.0)

            psnr_val = compute_psnr(enhanced, normal)
            ssim_val = compute_ssim(enhanced, normal, ssim_window)

            psnr_list.append(psnr_val)
            ssim_list.append(ssim_val)

            rows.append(
                {
                    "image_name": low_path.name,
                    "low_path": str(low_path),
                    "normal_path": str(normal_path),
                    "psnr": f"{psnr_val:.6f}",
                    "ssim": f"{ssim_val:.6f}",
                }
            )

    psnr_mean, psnr_std = compute_mean_std(psnr_list)
    ssim_mean, ssim_std = compute_mean_std(ssim_list)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_name", "low_path", "normal_path", "psnr", "ssim"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

        writer.writerow({"image_name": "AVG", "psnr": f"{psnr_mean:.6f}", "ssim": f"{ssim_mean:.6f}"})
        writer.writerow({"image_name": "STD", "psnr": f"{psnr_std:.6f}", "ssim": f"{ssim_std:.6f}"})

    print("Evaluation finished.")
    print(f"Paired images: {len(rows)}")
    print(f"PSNR mean/std: {psnr_mean:.6f} / {psnr_std:.6f}")
    print(f"SSIM mean/std: {ssim_mean:.6f} / {ssim_std:.6f}")
    print(f"CSV saved to: {output_csv}")


if __name__ == "__main__":
    main()
