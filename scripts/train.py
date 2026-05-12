import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import sys

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import LowLightPairDataset
from models.retinex_net import RetinexNetLite, count_parameters

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("PyYAML is required. Please install pyyaml.") from exc

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover
    SummaryWriter = None


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


def _to_gray(x: torch.Tensor) -> torch.Tensor:
    return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]


def illumination_smoothness_loss(illum: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    gray = _to_gray(image)

    grad_i_x = torch.abs(illum[:, :, :, 1:] - illum[:, :, :, :-1])
    grad_i_y = torch.abs(illum[:, :, 1:, :] - illum[:, :, :-1, :])
    grad_g_x = torch.abs(gray[:, :, :, 1:] - gray[:, :, :, :-1])
    grad_g_y = torch.abs(gray[:, :, 1:, :] - gray[:, :, :-1, :])

    weight_x = torch.exp(-10.0 * grad_g_x)
    weight_y = torch.exp(-10.0 * grad_g_y)
    return (grad_i_x * weight_x).mean() + (grad_i_y * weight_y).mean()


def gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    tgt_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    tgt_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    return F.l1_loss(pred_dx, tgt_dx) + F.l1_loss(pred_dy, tgt_dy)


def color_consistency_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_mean = pred.mean(dim=(2, 3))
    tgt_mean = target.mean(dim=(2, 3))
    return F.l1_loss(pred_mean, tgt_mean)


def exposure_control_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_luma = _to_gray(pred).mean(dim=(2, 3))
    tgt_luma = _to_gray(target).mean(dim=(2, 3))
    return F.l1_loss(pred_luma, tgt_luma)


def color_ratio_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred_sum = pred.sum(dim=1, keepdim=True) + eps
    tgt_sum = target.sum(dim=1, keepdim=True) + eps
    pred_ratio = pred / pred_sum
    tgt_ratio = target / tgt_sum
    return F.l1_loss(pred_ratio, tgt_ratio)


def edge_aware_denoise_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    gray_tgt = _to_gray(target)
    grad_x = torch.abs(gray_tgt[:, :, :, 1:] - gray_tgt[:, :, :, :-1])
    grad_y = torch.abs(gray_tgt[:, :, 1:, :] - gray_tgt[:, :, :-1, :])
    weight_x = torch.exp(-12.0 * grad_x)
    weight_y = torch.exp(-12.0 * grad_y)

    pred_dx = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
    pred_dy = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
    return (pred_dx * weight_x).mean() + (pred_dy * weight_y).mean()


def compute_batch_psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> float:
    mse = F.mse_loss(pred, target, reduction="none").mean(dim=(1, 2, 3))
    psnr = 10.0 * torch.log10(1.0 / (mse + eps))
    return psnr.mean().item()


@torch.no_grad()
def evaluate_psnr(model: RetinexNetLite, loader: DataLoader, device: str) -> float:
    model.eval()
    total_psnr = 0.0
    total_count = 0
    for low, normal in loader:
        low = low.to(device, non_blocking=True)
        normal = normal.to(device, non_blocking=True)
        illum, refl = model(low)
        recon = model.compose(illum, refl).clamp(0.0, 1.0)
        batch_size = low.shape[0]
        total_psnr += compute_batch_psnr(recon, normal) * batch_size
        total_count += batch_size
    return total_psnr / max(1, total_count)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train lightweight Retinex low-light enhancement model.")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file.")
    parser.add_argument("--data_root", type=str, default=None, help="Dataset root path.")
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs.")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Initial learning rate.")
    parser.add_argument("--min_lr", type=float, default=None, help="Cosine scheduler minimum lr.")
    parser.add_argument("--image_size", type=int, default=None, help="Input image size.")
    parser.add_argument("--num_workers", type=int, default=None, help="DataLoader workers.")
    parser.add_argument("--lambda_smooth", type=float, default=None, help="Illumination smoothness weight.")
    parser.add_argument("--lambda_grad", type=float, default=None, help="Gradient-structure loss weight.")
    parser.add_argument("--lambda_color", type=float, default=None, help="Color consistency loss weight.")
    parser.add_argument("--lambda_exposure", type=float, default=None, help="Exposure control loss weight.")
    parser.add_argument("--lambda_ratio", type=float, default=None, help="Color ratio loss weight.")
    parser.add_argument("--lambda_denoise", type=float, default=None, help="Edge-aware denoise loss weight.")
    parser.add_argument("--save_dir", type=str, default=None, help="Checkpoint output directory.")
    parser.add_argument("--max_param_mb", type=float, default=None, help="Maximum model size in MB.")
    parser.add_argument("--width", type=int, default=None, help="Backbone width.")
    parser.add_argument("--grad_clip", type=float, default=None, help="Gradient clipping value.")
    parser.add_argument("--eval_every", type=int, default=None, help="Run validation every N epochs.")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint if available.")
    parser.add_argument("--no_resume", action="store_true", help="Disable auto-resume from last checkpoint.")
    parser.add_argument("--log_dir", type=str, default=None, help="TensorBoard log directory.")
    parser.add_argument("--disable_tb", action="store_true", help="Disable TensorBoard logging.")
    return parser


def _apply_config_defaults(parser: argparse.ArgumentParser, cfg: Dict[str, Any]) -> None:
    loss_cfg = cfg.get("loss_weights", {}) if isinstance(cfg.get("loss_weights", {}), dict) else {}
    defaults = {
        "data_root": cfg.get("data_root", "./datasets/LOL"),
        "epochs": cfg.get("epochs", 120),
        "batch_size": cfg.get("batch_size", 8),
        "lr": cfg.get("lr", 3e-4),
        "min_lr": cfg.get("min_lr", 1e-6),
        "image_size": cfg.get("image_size", 320),
        "num_workers": cfg.get("num_workers", 0),
        "lambda_smooth": loss_cfg.get("lambda_smooth", cfg.get("lambda_smooth", 0.08)),
        "lambda_grad": loss_cfg.get("lambda_grad", cfg.get("lambda_grad", 0.2)),
        "lambda_color": loss_cfg.get("lambda_color", cfg.get("lambda_color", 0.05)),
        "lambda_exposure": loss_cfg.get("lambda_exposure", cfg.get("lambda_exposure", 0.15)),
        "lambda_ratio": loss_cfg.get("lambda_ratio", cfg.get("lambda_ratio", 0.08)),
        "lambda_denoise": loss_cfg.get("lambda_denoise", cfg.get("lambda_denoise", 0.04)),
        "save_dir": cfg.get("save_dir", "./checkpoints"),
        "max_param_mb": cfg.get("max_param_mb", 2.0),
        "width": cfg.get("width", 24),
        "grad_clip": cfg.get("grad_clip", 1.0),
        "eval_every": cfg.get("eval_every", 1),
        "log_dir": cfg.get("log_dir", None),
    }
    parser.set_defaults(**defaults)


def _lazy_import_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    return plt


def _save_loss_plot(save_dir: Path, history: Dict[str, list]) -> None:
    plt = _lazy_import_matplotlib()
    if plt is None:
        return
    if len(history.get("train_loss", [])) == 0:
        return

    fig, ax1 = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(history["train_loss"]) + 1)
    ax1.plot(epochs, history["train_loss"], label="train_loss", color="#1f77b4")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)

    if len(history.get("val_psnr", [])) > 0:
        ax2 = ax1.twinx()
        ax2.plot(epochs[: len(history["val_psnr"])], history["val_psnr"], label="val_psnr", color="#ff7f0e")
        ax2.set_ylabel("PSNR")
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")
    else:
        ax1.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(save_dir / "training_curve.png", dpi=150)
    plt.close(fig)


def _load_checkpoint_if_available(
    save_dir: Path,
    model: RetinexNetLite,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingLR,
    device: str,
) -> Dict[str, Any]:
    last_ckpt = save_dir / "retinex_lite_last.pth"
    if not last_ckpt.exists():
        return {
            "start_epoch": 1,
            "history": {"train_loss": [], "val_psnr": [], "lr": []},
            "best_loss": float("inf"),
            "best_val_psnr": float("-inf"),
            "has_ckpt": False,
        }

    ckpt = torch.load(last_ckpt, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"], strict=True)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        history = ckpt.get("history", {"train_loss": [], "val_psnr": [], "lr": []})
        return {
            "start_epoch": int(ckpt.get("epoch", 0)) + 1,
            "history": history,
            "best_loss": float(ckpt.get("best_loss", float("inf"))),
            "best_val_psnr": float(ckpt.get("best_val_psnr", float("-inf"))),
            "has_ckpt": True,
        }

    return {
        "start_epoch": 1,
        "history": {"train_loss": [], "val_psnr": [], "lr": []},
        "best_loss": float("inf"),
        "best_val_psnr": float("-inf"),
        "has_ckpt": True,
    }


def main():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default="./configs/train.yaml")
    pre_args, remaining = pre_parser.parse_known_args()

    cfg = _load_yaml(pre_args.config)
    parser = _build_arg_parser()
    _apply_config_defaults(parser, cfg)
    args = parser.parse_args(remaining)
    args.config = pre_args.config

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_low_dir = Path(args.data_root) / "train" / "low"
    train_normal_dir = Path(args.data_root) / "train" / "normal"
    val_low_dir = Path(args.data_root) / "eval15" / "low"
    val_normal_dir = Path(args.data_root) / "eval15" / "normal"

    if not train_low_dir.exists():
        raise FileNotFoundError(
            f"Training low-light folder not found: {train_low_dir}. "
            "Please check that datasets/LOL/train/low exists."
        )
    if not train_normal_dir.exists():
        raise FileNotFoundError(
            f"Training normal-light folder not found: {train_normal_dir}. "
            "Please check that datasets/LOL/train/normal exists."
        )

    train_set = LowLightPairDataset(
        root=args.data_root,
        split="train",
        image_size=args.image_size,
        augment=True,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = None
    if val_low_dir.exists() and val_normal_dir.exists():
        val_set = LowLightPairDataset(
            root=args.data_root,
            split="eval15",
            image_size=args.image_size,
            augment=False,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        print(f"Validation split detected: {len(val_set)} pairs")
    else:
        print("Validation split not found at datasets/LOL/eval15. Best checkpoint will be selected by training loss.")

    model = RetinexNetLite(width=args.width).to(device)

    param_count = count_parameters(model)
    model_size_mb = param_count * 4 / (1024 ** 2)
    print(f"Trainable parameters: {param_count:,}")
    print(f"Estimated model size (FP32): {model_size_mb:.4f} MB")
    if model_size_mb > args.max_param_mb:
        raise RuntimeError(
            f"Model size {model_size_mb:.4f} MB exceeds limit {args.max_param_mb:.2f} MB. Please reduce backbone width."
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir) if args.log_dir else (save_dir / "tb_logs")
    writer = None
    if not args.disable_tb and SummaryWriter is not None:
        writer = SummaryWriter(log_dir=str(log_dir))
        print(f"TensorBoard logs: {log_dir}")
    elif SummaryWriter is None:
        print("TensorBoard is unavailable; logging disabled.")

    should_resume = bool(args.resume) and not bool(args.no_resume)
    state = {
        "start_epoch": 1,
        "history": {"train_loss": [], "val_psnr": [], "lr": []},
        "best_loss": float("inf"),
        "best_val_psnr": float("-inf"),
        "has_ckpt": False,
    }

    if should_resume:
        state = _load_checkpoint_if_available(save_dir, model, optimizer, scheduler, device)
        if state["has_ckpt"]:
            resumed_epoch = state["start_epoch"] - 1
            if resumed_epoch >= args.epochs:
                print(
                    f"Checkpoint epoch {resumed_epoch} is already >= target epochs {args.epochs}. "
                    "Use --no_resume to start a new run, or increase --epochs to continue training."
                )
                if writer is not None:
                    writer.close()
                return
            print(f"Resumed from epoch {resumed_epoch} using {save_dir / 'retinex_lite_last.pth'}")
        else:
            print("No checkpoint found; starting from epoch 1.")
    else:
        state = {
            "start_epoch": 1,
            "history": {"train_loss": [], "val_psnr": [], "lr": []},
            "best_loss": float("inf"),
            "best_val_psnr": float("-inf"),
            "has_ckpt": False,
        }

    history = state["history"]
    best_loss = state["best_loss"]
    best_val_psnr = state["best_val_psnr"]

    epoch = 0
    epoch_loss = 0.0
    val_psnr = None

    try:
        for epoch in range(state["start_epoch"], args.epochs + 1):
            model.train()
            running_loss = 0.0
            val_psnr = None
            epoch_loss = 0.0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", ncols=100)
            for low, normal in pbar:
                low = low.to(device, non_blocking=True)
                normal = normal.to(device, non_blocking=True)

                illum, refl = model(low)
                recon = model.compose(illum, refl)

                loss_l1 = F.l1_loss(recon, normal)
                loss_smooth = illumination_smoothness_loss(illum, low)
                loss_grad = gradient_loss(recon, normal)
                loss_color = color_consistency_loss(recon, normal)
                loss_exposure = exposure_control_loss(recon, normal)
                loss_ratio = color_ratio_loss(recon, normal)
                loss_denoise = edge_aware_denoise_loss(recon, normal)

                loss = (
                    loss_l1
                    + args.lambda_smooth * loss_smooth
                    + args.lambda_grad * loss_grad
                    + args.lambda_color * loss_color
                    + args.lambda_exposure * loss_exposure
                    + args.lambda_ratio * loss_ratio
                    + args.lambda_denoise * loss_denoise
                )

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                optimizer.step()

                running_loss += loss.item()
                pbar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "l1": f"{loss_l1.item():.4f}",
                        "smooth": f"{loss_smooth.item():.4f}",
                        "grad": f"{loss_grad.item():.4f}",
                        "color": f"{loss_color.item():.4f}",
                        "exp": f"{loss_exposure.item():.4f}",
                        "ratio": f"{loss_ratio.item():.4f}",
                        "denoise": f"{loss_denoise.item():.4f}",
                    }
                )

            epoch_loss = running_loss / max(1, len(train_loader))
            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"[Epoch {epoch}] avg_loss={epoch_loss:.6f}, lr={current_lr:.8f}")

            if val_loader is not None and epoch % max(1, args.eval_every) == 0:
                val_psnr = evaluate_psnr(model, val_loader, device)
                print(f"[Epoch {epoch}] val_psnr={val_psnr:.4f}")

            history.setdefault("train_loss", []).append(epoch_loss)
            history.setdefault("val_psnr", []).append(val_psnr if val_psnr is not None else float("nan"))
            history.setdefault("lr", []).append(current_lr)

            if writer is not None:
                writer.add_scalar("train/loss", epoch_loss, epoch)
                writer.add_scalar("train/lr", current_lr, epoch)
                if val_psnr is not None:
                    writer.add_scalar("val/psnr", val_psnr, epoch)

            _save_loss_plot(save_dir, history)

            last_ckpt = save_dir / "retinex_lite_last.pth"
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "loss": epoch_loss,
                    "val_psnr": val_psnr,
                    "width": args.width,
                    "param_count": param_count,
                    "best_loss": best_loss,
                    "best_val_psnr": best_val_psnr,
                    "history": history,
                    "config": vars(args),
                },
                last_ckpt,
            )

            should_update_best = val_psnr > best_val_psnr if val_psnr is not None else epoch_loss < best_loss
            if should_update_best:
                best_loss = epoch_loss
                if val_psnr is not None:
                    best_val_psnr = val_psnr
                best_ckpt = save_dir / "retinex_lite_best.pth"
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "epoch": epoch,
                        "loss": best_loss,
                        "val_psnr": val_psnr,
                        "width": args.width,
                        "param_count": param_count,
                        "best_loss": best_loss,
                        "best_val_psnr": best_val_psnr,
                        "history": history,
                        "config": vars(args),
                    },
                    best_ckpt,
                )
                if val_psnr is not None:
                    print(f"Best model updated by val_psnr={val_psnr:.4f}: {best_ckpt}")
                else:
                    print(f"Best model updated by train loss: {best_ckpt}")

            if writer is not None:
                writer.flush()

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving last checkpoint before exit...")
        last_ckpt = save_dir / "retinex_lite_last.pth"
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch if "epoch" in locals() else 0,
                "loss": epoch_loss if "epoch_loss" in locals() else None,
                "val_psnr": val_psnr if "val_psnr" in locals() else None,
                "width": args.width,
                "param_count": param_count,
                "best_loss": best_loss,
                "best_val_psnr": best_val_psnr,
                "history": history,
                "config": vars(args),
            },
            last_ckpt,
        )
        _save_loss_plot(save_dir, history)
        print(f"Checkpoint saved to: {last_ckpt}")
        raise
    finally:
        if writer is not None:
            writer.close()

    print("Training finished.")


if __name__ == "__main__":
    main()
