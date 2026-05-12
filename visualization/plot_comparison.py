import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


METHOD_ORDER = ["HE", "MSRCR", "Gamma", "Ours"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot baseline vs ours comparison (PSNR/SSIM) from summary CSV rows."
    )
    parser.add_argument("--baseline_csv", type=str, required=True, help="Path to baseline_metrics.csv")
    parser.add_argument("--ours_csv", type=str, required=True, help="Path to metrics_after_color_denoise.csv")
    parser.add_argument(
        "--output_png",
        type=str,
        default="comparison_chart.png",
        help="Output chart path (default: comparison_chart.png)",
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def find_summary_rows(rows: List[Dict[str, str]]) -> Tuple[Dict[str, str], Dict[str, str]]:
    avg_row = None
    std_row = None
    for r in rows:
        name = str(r.get("image_name", "")).strip().upper()
        if name == "AVG":
            avg_row = r
        elif name == "STD":
            std_row = r
    if avg_row is None or std_row is None:
        raise ValueError("AVG/STD rows not found in CSV.")
    return avg_row, std_row


def as_float(v: str) -> float:
    return float(str(v).strip())


def parse_baseline_stats(path: Path) -> Dict[str, Dict[str, float]]:
    rows = read_csv_rows(path)
    avg, std = find_summary_rows(rows)

    # Normalize potential naming variations by searching candidate headers.
    header_map = {k.lower(): k for k in avg.keys()}

    def get_col(*candidates: str) -> str:
        for c in candidates:
            k = c.lower()
            if k in header_map:
                return header_map[k]
        raise KeyError(f"None of columns found: {candidates}")

    he_psnr_col = get_col("HE_PSNR", "he_psnr", "psnr_he")
    he_ssim_col = get_col("HE_SSIM", "he_ssim", "ssim_he")
    msrcr_psnr_col = get_col("MSRCR_PSNR", "msrcr_psnr", "psnr_msrcr")
    msrcr_ssim_col = get_col("MSRCR_SSIM", "msrcr_ssim", "ssim_msrcr")
    gamma_psnr_col = get_col("Gamma_PSNR", "gamma_psnr", "psnr_gamma")
    gamma_ssim_col = get_col("Gamma_SSIM", "gamma_ssim", "ssim_gamma")

    return {
        "HE": {
            "psnr_mean": as_float(avg[he_psnr_col]),
            "psnr_std": as_float(std[he_psnr_col]),
            "ssim_mean": as_float(avg[he_ssim_col]),
            "ssim_std": as_float(std[he_ssim_col]),
        },
        "MSRCR": {
            "psnr_mean": as_float(avg[msrcr_psnr_col]),
            "psnr_std": as_float(std[msrcr_psnr_col]),
            "ssim_mean": as_float(avg[msrcr_ssim_col]),
            "ssim_std": as_float(std[msrcr_ssim_col]),
        },
        "Gamma": {
            "psnr_mean": as_float(avg[gamma_psnr_col]),
            "psnr_std": as_float(std[gamma_psnr_col]),
            "ssim_mean": as_float(avg[gamma_ssim_col]),
            "ssim_std": as_float(std[gamma_ssim_col]),
        },
    }


def parse_ours_stats(path: Path) -> Dict[str, Dict[str, float]]:
    rows = read_csv_rows(path)
    avg, std = find_summary_rows(rows)
    header_map = {k.lower(): k for k in avg.keys()}

    psnr_col = header_map.get("psnr")
    ssim_col = header_map.get("ssim")
    if psnr_col is None or ssim_col is None:
        raise KeyError("Could not find 'psnr'/'ssim' columns in ours CSV.")

    return {
        "Ours": {
            "psnr_mean": as_float(avg[psnr_col]),
            "psnr_std": as_float(std[psnr_col]),
            "ssim_mean": as_float(avg[ssim_col]),
            "ssim_std": as_float(std[ssim_col]),
        }
    }


def merge_stats(
    baseline_stats: Dict[str, Dict[str, float]],
    ours_stats: Dict[str, Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    out = {}
    out.update(baseline_stats)
    out.update(ours_stats)
    for m in METHOD_ORDER:
        if m not in out:
            raise KeyError(f"Method '{m}' missing in merged stats.")
    return out


def plot_bars(stats: Dict[str, Dict[str, float]], output_png: Path) -> None:
    methods = METHOD_ORDER
    psnr_mean = [stats[m]["psnr_mean"] for m in methods]
    psnr_std = [stats[m]["psnr_std"] for m in methods]
    ssim_mean = [stats[m]["ssim_mean"] for m in methods]
    ssim_std = [stats[m]["ssim_std"] for m in methods]

    x = np.arange(len(methods))
    colors = ["#5DA5DA", "#60BD68", "#F17CB0", "#F5A623"]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    axes[0].bar(x, psnr_mean, yerr=psnr_std, color=colors, capsize=4, alpha=0.9, edgecolor="black")
    axes[0].set_xticks(x, methods)
    axes[0].set_ylabel("PSNR (dB)")
    axes[0].set_title("PSNR Comparison")
    axes[0].grid(axis="y", linestyle="--", alpha=0.35)

    axes[1].bar(x, ssim_mean, yerr=ssim_std, color=colors, capsize=4, alpha=0.9, edgecolor="black")
    axes[1].set_xticks(x, methods)
    axes[1].set_ylabel("SSIM")
    axes[1].set_title("SSIM Comparison")
    axes[1].grid(axis="y", linestyle="--", alpha=0.35)

    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def latex_table(stats: Dict[str, Dict[str, float]]) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Quantitative comparison on LOL-eval15 (mean $\pm$ std).}",
        r"\label{tab:lol_eval15_comparison}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Method & PSNR (dB) & SSIM \\",
        r"\midrule",
    ]
    for m in METHOD_ORDER:
        s = stats[m]
        lines.append(
            f"{m} & {s['psnr_mean']:.2f} $\\pm$ {s['psnr_std']:.2f} & "
            f"{s['ssim_mean']:.3f} $\\pm$ {s['ssim_std']:.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    baseline_stats = parse_baseline_stats(Path(args.baseline_csv))
    ours_stats = parse_ours_stats(Path(args.ours_csv))
    stats = merge_stats(baseline_stats, ours_stats)

    output_png = Path(args.output_png)
    plot_bars(stats, output_png)

    print("Chart saved to:", output_png)
    print("\nLaTeX table:\n")
    print(latex_table(stats))


if __name__ == "__main__":
    main()

