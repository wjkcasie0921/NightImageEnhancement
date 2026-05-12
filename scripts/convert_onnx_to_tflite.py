import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, Tuple

import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.retinex_net import RetinexNetLite


PNNX_EXE = r"E:\anaconda3\envs\zhou_3_8\Scripts\pnnx.exe"


def parse_input_shape(text: str) -> Tuple[int, int, int, int]:
    parts = [int(x.strip()) for x in text.split(",")]
    if len(parts) != 4:
        raise ValueError("--input-shape must be like 1,3,320,320")
    if parts[0] != 1 or parts[1] != 3:
        raise ValueError("Expected PyTorch NCHW input shape like 1,3,320,320")
    return tuple(parts)  # type: ignore[return-value]


def load_checkpoint(ckpt_path: Path, width: int) -> RetinexNetLite:
    model = RetinexNetLite(width=width)
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model") or checkpoint.get("state_dict") or checkpoint
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported checkpoint format: {type(state_dict)!r}")

    cleaned_state_dict: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key.replace("module.", "", 1) if key.startswith("module.") else key
        cleaned_state_dict[new_key] = value

    missing, unexpected = model.load_state_dict(cleaned_state_dict, strict=False)
    if missing:
        print(f"[WARN] Missing keys: {missing}")
    if unexpected:
        print(f"[WARN] Unexpected keys: {unexpected}")

    model.eval()
    return model


def export_onnx(model: torch.nn.Module, sample_input: torch.Tensor, onnx_path: Path) -> Path:
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        sample_input,
        str(onnx_path),
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["illum", "refl"],
        dynamic_axes={
            "input": {0: "batch", 2: "height", 3: "width"},
            "illum": {0: "batch", 2: "height", 3: "width"},
            "refl": {0: "batch", 2: "height", 3: "width"},
        },
    )
    print(f"ONNX model saved to: {onnx_path}")
    return onnx_path


def run_pnnx_tools(onnx_path: Path, out_dir: Path, model_name: str) -> Tuple[Path, Path]:
    if not Path(PNNX_EXE).exists():
        raise FileNotFoundError(f"PNNX executable not found: {PNNX_EXE}")
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    param_path = out_dir / f"{model_name}.pnnx.param"
    bin_path = out_dir / f"{model_name}.pnnx.bin"
    ncnn_param_path = out_dir / f"{model_name}.ncnn.param"
    ncnn_bin_path = out_dir / f"{model_name}.ncnn.bin"

    # PNNX 命令通常只需要输入模型路径，输出文件名通过参数或默认规则生成。
    # 这里显式使用输入模型绝对路径，并在输出目录中执行，避免相对路径解析错误。
    cmd = [
        PNNX_EXE,
        str(onnx_path.resolve()),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(out_dir.resolve()))

    # PNNX 通常会在输出目录生成 .pnnx.* 和 .ncnn.* 文件，这里统一收集到固定路径。
    generated_param = out_dir / f"{model_name}.pnnx.param"
    generated_bin = out_dir / f"{model_name}.pnnx.bin"
    generated_ncnn_param = out_dir / f"{model_name}.ncnn.param"
    generated_ncnn_bin = out_dir / f"{model_name}.ncnn.bin"

    if generated_param.exists():
        param_path.write_bytes(generated_param.read_bytes())
    elif generated_ncnn_param.exists():
        param_path.write_bytes(generated_ncnn_param.read_bytes())
    else:
        raise FileNotFoundError("PNNX did not generate param file")

    if generated_bin.exists():
        bin_path.write_bytes(generated_bin.read_bytes())
    elif generated_ncnn_bin.exists():
        bin_path.write_bytes(generated_ncnn_bin.read_bytes())
    else:
        raise FileNotFoundError("PNNX did not generate bin file")

    return param_path, bin_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert RetinexNetLite checkpoint to NCNN via PNNX")
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=Path("checkpoints/retinex_lite_best.pth"),
        help="Path to PyTorch checkpoint",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("mobile/tflite_out"),
        help="Output directory",
    )
    parser.add_argument(
        "--input-shape",
        type=str,
        default="1,3,320,320",
        help="PyTorch NCHW input shape, e.g. 1,3,320,320",
    )
    parser.add_argument("--width", type=int, default=24, help="RetinexNetLite width")
    args = parser.parse_args()

    input_shape = parse_input_shape(args.input_shape)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model = load_checkpoint(args.ckpt, args.width)
    sample_input = torch.randn(*input_shape)

    print(f"Loaded checkpoint: {args.ckpt}")
    print(f"Sample input shape: {tuple(sample_input.shape)}")

    onnx_path = args.out_dir / "retinex_lite.onnx"
    export_onnx(model, sample_input, onnx_path)

    param_path, bin_path = run_pnnx_tools(onnx_path, args.out_dir, "retinex_lite")

    print("PNNX conversion completed successfully.")
    print(f"ONNX:  {onnx_path}")
    print(f"PARAM: {param_path}")
    print(f"BIN:   {bin_path}")


if __name__ == "__main__":
    main()
