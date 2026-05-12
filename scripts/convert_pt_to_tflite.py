import argparse
import sys
from pathlib import Path
from typing import Tuple

import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.retinex_net import RetinexNetLite


def parse_input_shape(text: str) -> Tuple[int, int, int, int]:
    parts = [int(x.strip()) for x in text.split(",")]
    if len(parts) != 4:
        raise ValueError("--input-shape must be like 1,3,320,320")
    if parts[1] != 3:
        raise ValueError("This script expects PyTorch NCHW input shape like 1,3,320,320")
    return tuple(parts)  # type: ignore[return-value]


def load_model(ckpt_path: Path, width: int) -> RetinexNetLite:
    model = RetinexNetLite(width=width)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model") or checkpoint.get("state_dict") or checkpoint
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported checkpoint format: {type(state_dict)!r}")

    cleaned = {}
    for key, value in state_dict.items():
        new_key = key.replace("module.", "", 1) if key.startswith("module.") else key
        cleaned[new_key] = value

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"[WARN] Missing keys: {missing}")
    if unexpected:
        print(f"[WARN] Unexpected keys: {unexpected}")

    model.eval()
    return model


def inspect_torch_output(model: torch.nn.Module, sample_input: torch.Tensor) -> None:
    with torch.no_grad():
        outputs = model(sample_input)

    if isinstance(outputs, tuple) and len(outputs) == 2:
        illum, refl = outputs
        print("Torch output shapes:")
        print(f"  illum: {tuple(illum.shape)}")
        print(f"  refl : {tuple(refl.shape)}")
    else:
        print("Torch output shape:", tuple(outputs.shape))


def export_with_ai_edge_torch(model: torch.nn.Module, sample_input: torch.Tensor, output_path: Path) -> None:
    try:
        import ai_edge_torch
    except ImportError as exc:
        raise ImportError(
            "ai_edge_torch is not installed. Run: pip install ai-edge-torch"
        ) from exc

    # 官方推荐：用样例输入触发 tracing / lowering。
    edge_model = ai_edge_torch.convert(model, (sample_input,))
    edge_model.export(str(output_path))
    print(f"TFLite saved to: {output_path}")

    # 兼容不同版本的 ai_edge_torch API：尽量打印可用信息
    try:
        if hasattr(edge_model, "signature"):
            print("Edge model signature:", edge_model.signature)
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert PyTorch RetinexNetLite checkpoint to TFLite")
    parser.add_argument("--ckpt", type=Path, default=Path("checkpoints/retinex_lite_best.pth"), help="Path to PyTorch checkpoint")
    parser.add_argument("--out-dir", type=Path, default=Path("mobile/tflite_out"), help="Output directory")
    parser.add_argument("--input-shape", type=str, default="1,3,320,320", help="PyTorch NCHW shape, e.g. 1,3,320,320")
    parser.add_argument("--width", type=int, default=24, help="RetinexNetLite width")
    parser.add_argument("--output-name", type=str, default="retinex_lite.tflite", help="TFLite filename")
    args = parser.parse_args()

    input_shape = parse_input_shape(args.input_shape)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.out_dir / args.output_name

    model = load_model(args.ckpt, args.width)
    sample_input = torch.randn(*input_shape)

    print(f"Loaded checkpoint: {args.ckpt}")
    print(f"Sample input shape: {tuple(sample_input.shape)}")
    inspect_torch_output(model, sample_input)

    export_with_ai_edge_torch(model, sample_input, output_path)


if __name__ == "__main__":
    main()
