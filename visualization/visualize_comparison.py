import argparse
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import sys

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.retinex_net import RetinexNetLite


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
METHOD_NAMES = ["Original", "HE", "MSRCR", "Gamma", "Our Model"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize enhancement comparisons across multiple methods.")
    parser.add_argument("--low_dir", type=str, required=True, help="Low-light image directory.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to trained model checkpoint.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save comparison images.")
    parser.add_argument("--image_size", type=int, default=320, help="Resize H/W for all methods.")
    parser.add_argument("--num_images", type=int, default=5, help="Number of images to process; <=0 means all.")
    parser.add_argument(
        "--save_individual",
        action="store_true",
        help="If set, also save individual outputs into method subfolders.",
    )
    return parser.parse_args()


def collect_images(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def load_model(ckpt_path: Path, device: str) -> RetinexNetLite:
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model = RetinexNetLite(width=24).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def load_and_resize_rgb(path: Path, image_size: int) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    img = img.resize((image_size, image_size), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def he_yuv_rgb(img_rgb: np.ndarray) -> np.ndarray:
    yuv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YUV)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)


def gamma_correction(img_rgb: np.ndarray, gamma: float = 0.7) -> np.ndarray:
    x = img_rgb.astype(np.float32) / 255.0
    y = np.power(np.clip(x, 0.0, 1.0), gamma)
    return np.clip(y * 255.0, 0.0, 255.0).astype(np.uint8)


def init_msrcr_processor() -> Tuple[str, Optional[object]]:
    xphoto = getattr(cv2, "xphoto", None)
    if xphoto is not None and hasattr(xphoto, "createMsrCR"):
        try:
            return "opencv_xphoto", xphoto.createMsrCR()
        except Exception as e:
            warnings.warn(f"cv2.xphoto.createMsrCR init failed: {e}")
    warnings.warn("MSRCR OpenCV xphoto API unavailable, using fallback MSRCR implementation.")
    return "fallback", None


def msrcr_fallback(
    img_rgb: np.ndarray,
    sigmas: Tuple[float, float, float] = (15.0, 80.0, 250.0),
    gain: float = 5.0,
    bias: float = 25.0,
    alpha: float = 125.0,
    beta: float = 46.0,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Multi-scale Retinex with Color Restoration fallback implementation.
    """
    img = img_rgb.astype(np.float32) + 1.0
    img = np.maximum(img, eps)

    retinex = np.zeros_like(img, dtype=np.float32)
    for sigma in sigmas:
        blur = cv2.GaussianBlur(img, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma)
        blur = np.maximum(blur, eps)
        retinex += np.log(img) - np.log(blur)
    retinex /= float(len(sigmas))

    intensity = np.sum(img, axis=2, keepdims=True)
    intensity = np.maximum(intensity, eps)
    color_restore = beta * (np.log(alpha * img) - np.log(intensity))
    msrcr = gain * (retinex * color_restore + bias)

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


def run_msrcr(img_rgb: np.ndarray, backend: str, processor: Optional[object]) -> np.ndarray:
    if backend == "opencv_xphoto":
        if processor is None:
            raise RuntimeError("MSRCR processor is None.")
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        out = processor.process(img_bgr)
        if out is None:
            raise RuntimeError("MSRCR processor returned None.")
        out = np.asarray(out)
        if out.dtype != np.uint8:
            out = np.clip(out, 0.0, 255.0).astype(np.uint8)
        return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    return msrcr_fallback(img_rgb)


def run_our_model(img_rgb: np.ndarray, model: RetinexNetLite, device: str) -> np.ndarray:
    tf = transforms.ToTensor()
    x = tf(Image.fromarray(img_rgb)).unsqueeze(0).to(device)
    with torch.no_grad():
        illum, refl = model(x)
        out = model.compose(illum, refl).clamp(0.0, 1.0)[0].cpu().numpy()
    out = np.transpose(out, (1, 2, 0))
    return np.clip(out * 255.0, 0.0, 255.0).astype(np.uint8)


def make_labeled_strip(images: List[np.ndarray], labels: List[str], pad_top: int = 30) -> Image.Image:
    if len(images) != len(labels):
        raise ValueError("images and labels must have same length.")
    h, w = images[0].shape[:2]
    canvas = Image.new("RGB", (w * len(images), h + pad_top), color=(20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for i, (img, label) in enumerate(zip(images, labels)):
        tile = Image.fromarray(img)
        x = i * w
        canvas.paste(tile, (x, pad_top))
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        tx = x + max(0, (w - text_w) // 2)
        ty = max(0, (pad_top - text_h) // 2)
        draw.text((tx, ty), label, fill=(255, 255, 255), font=font)

    return canvas


def sanitize_stem(name: str) -> str:
    # Avoid duplicate ".png" like "xxx.png_comparison.png"
    return Path(name).stem


def ensure_method_dirs(base_dir: Path) -> dict:
    method_to_dir = {}
    for name in METHOD_NAMES:
        d = base_dir / name.replace(" ", "_")
        d.mkdir(parents=True, exist_ok=True)
        method_to_dir[name] = d
    return method_to_dir


def main() -> None:
    args = parse_args()
    low_dir = Path(args.low_dir)
    ckpt = Path(args.ckpt)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not low_dir.exists():
        raise FileNotFoundError(f"low_dir not found: {low_dir}")
    if not ckpt.exists():
        raise FileNotFoundError(f"ckpt not found: {ckpt}")

    low_paths = collect_images(low_dir)
    if len(low_paths) == 0:
        raise RuntimeError(f"No images found in low_dir: {low_dir}")

    if args.num_images > 0:
        low_paths = low_paths[: args.num_images]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(ckpt, device)
    msrcr_backend, msrcr_processor = init_msrcr_processor()

    method_dirs = ensure_method_dirs(output_dir / "individual") if args.save_individual else None

    for low_path in low_paths:
        img_rgb = load_and_resize_rgb(low_path, args.image_size)

        original = img_rgb
        he = he_yuv_rgb(img_rgb)
        gamma = gamma_correction(img_rgb, gamma=0.7)

        try:
            msrcr = run_msrcr(img_rgb, msrcr_backend, msrcr_processor)
        except Exception as e:
            warnings.warn(f"MSRCR failed on {low_path.name}, fallback to original for panel: {e}")
            msrcr = original.copy()

        try:
            ours = run_our_model(img_rgb, model, device)
        except Exception as e:
            warnings.warn(f"Our model inference failed on {low_path.name}, fallback to original: {e}")
            ours = original.copy()

        outputs = [original, he, msrcr, gamma, ours]
        strip = make_labeled_strip(outputs, METHOD_NAMES, pad_top=30)

        out_name = f"{sanitize_stem(low_path.name)}_comparison.png"
        strip.save(output_dir / out_name)

        if method_dirs is not None:
            Image.fromarray(original).save(method_dirs["Original"] / f"{sanitize_stem(low_path.name)}.png")
            Image.fromarray(he).save(method_dirs["HE"] / f"{sanitize_stem(low_path.name)}.png")
            Image.fromarray(msrcr).save(method_dirs["MSRCR"] / f"{sanitize_stem(low_path.name)}.png")
            Image.fromarray(gamma).save(method_dirs["Gamma"] / f"{sanitize_stem(low_path.name)}.png")
            Image.fromarray(ours).save(method_dirs["Our Model"] / f"{sanitize_stem(low_path.name)}.png")

    print("Visualization finished.")
    print(f"Processed images: {len(low_paths)}")
    print(f"MSRCR backend: {msrcr_backend}")
    print(f"Output dir: {output_dir}")
    if args.save_individual:
        print(f"Individual outputs dir: {output_dir / 'individual'}")


if __name__ == "__main__":
    main()

