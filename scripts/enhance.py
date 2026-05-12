import argparse
from pathlib import Path

import sys

from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.retinex_net import RetinexNetLite


def parse_args():
    parser = argparse.ArgumentParser(description="Enhance one low-light image using trained Retinex model.")
    parser.add_argument("--input", type=str, required=True, help="Path to input low-light image.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to trained checkpoint (.pth).")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Directory to save output.")
    parser.add_argument("--image_size", type=int, default=320, help="Inference image size.")
    parser.add_argument(
        "--brightness_gain",
        type=float,
        default=1.08,
        help="Final brightness gain (>1.0 makes output brighter).",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.95,
        help="Gamma correction (<1.0 brightens shadows).",
    )
    parser.add_argument(
        "--denoise_strength",
        type=float,
        default=0.15,
        help="Blend weight of light smoothing in flat areas (0~1).",
    )
    parser.add_argument(
        "--wb_strength",
        type=float,
        default=0.2,
        help="Gray-world white balance strength (0~1), helps reduce color cast.",
    )
    return parser.parse_args()


def load_model(ckpt_path: str, device: str) -> RetinexNetLite:
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        width = ckpt.get("width", 24)
        state_dict = ckpt["model"]
    else:
        width = 24
        state_dict = ckpt
    model = RetinexNetLite(width=width).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def gray_world_white_balance(x: torch.Tensor, strength: float = 0.2, eps: float = 1e-6) -> torch.Tensor:
    """
    Apply gray-world white balance with controllable strength.
    x shape: [N,3,H,W], range [0,1].
    """
    strength = max(0.0, min(1.0, strength))
    if strength <= 0.0:
        return x

    ch_mean = x.mean(dim=(2, 3), keepdim=True)
    gray_mean = ch_mean.mean(dim=1, keepdim=True)
    gains = gray_mean / (ch_mean + eps)
    balanced = (x * gains).clamp(0.0, 1.0)
    return torch.lerp(x, balanced, strength)


def edge_aware_denoise(x: torch.Tensor, strength: float = 0.15) -> torch.Tensor:
    """
    Lightweight edge-aware smoothing:
    stronger in flat regions, weaker on strong edges.
    """
    strength = max(0.0, min(1.0, strength))
    if strength <= 0.0:
        return x

    smooth = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
    gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
    grad_x = torch.abs(gray[:, :, :, 1:] - gray[:, :, :, :-1])
    grad_y = torch.abs(gray[:, :, 1:, :] - gray[:, :, :-1, :])
    grad_x = F.pad(grad_x, (0, 1, 0, 0), mode="replicate")
    grad_y = F.pad(grad_y, (0, 0, 0, 1), mode="replicate")
    edge = (grad_x + grad_y).clamp(0.0, 1.0)
    flat_weight = torch.exp(-10.0 * edge)
    alpha = strength * flat_weight
    return torch.lerp(x, smooth, alpha).clamp(0.0, 1.0)


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(args.input).convert("RGB")
    transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
        ]
    )
    inv_transform = transforms.ToPILImage()

    x = transform(image).unsqueeze(0).to(device)
    model = load_model(args.ckpt, device)

    with torch.no_grad():
        illum, refl = model(x)
        enhanced = model.compose(illum, refl).clamp(0.0, 1.0)
        # Mild post enhancement to avoid globally dark output.
        if abs(args.gamma - 1.0) > 1e-6:
            enhanced = torch.pow(enhanced.clamp(1e-6, 1.0), args.gamma)
        if abs(args.brightness_gain - 1.0) > 1e-6:
            enhanced = (enhanced * args.brightness_gain).clamp(0.0, 1.0)

        enhanced = gray_world_white_balance(enhanced, args.wb_strength)
        enhanced = edge_aware_denoise(enhanced, args.denoise_strength)

        # Slight contrast recovery after gamma/gain.
        enhanced = torch.clamp((enhanced - 0.5) * 1.03 + 0.5, 0.0, 1.0)

    stem = Path(args.input).stem
    enhanced_path = output_dir / f"{stem}_enhanced.png"
    compare_path = output_dir / f"{stem}_compare.png"

    save_image(enhanced[0], enhanced_path.as_posix())

    # Build side-by-side comparison image (left: original, right: enhanced).
    original_pil = inv_transform(x[0].cpu())
    enhanced_pil = inv_transform(enhanced[0].cpu())
    compare = Image.new("RGB", (original_pil.width * 2, original_pil.height))
    compare.paste(original_pil, (0, 0))
    compare.paste(enhanced_pil, (original_pil.width, 0))
    compare.save(compare_path)

    print(f"Enhanced image saved to: {enhanced_path}")
    print(f"Comparison image saved to: {compare_path}")


if __name__ == "__main__":
    main()

