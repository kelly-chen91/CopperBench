#!/usr/bin/env python3
"""Plot the Stage 1 baseline vs Stage 3 prompts comparison as separate figures.

Reads `replicate_runs/baseline_stage3_mae_accuracy_comparison.csv` and produces
three independent PNGs so MAE (lower-is-better) and accuracy (higher-is-better)
are never displayed in the same figure.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

MPLCONFIGDIR = Path("/private/tmp/copperbench_mplconfig")
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib.pyplot as plt


STAGE3_DIR = Path(__file__).resolve().parent
RUNS_DIR = STAGE3_DIR / "replicate_runs"
INPUT_CSV = RUNS_DIR / "baseline_stage3_mae_accuracy_comparison.csv"

PROMPT_ORDER = ["Baseline", "Combined persona + few-shot", "Persona"]
COLORS = {
    "Baseline": "#4C78A8",
    "Combined persona + few-shot": "#F58518",
    "Persona": "#54A24B",
}


def read_rows() -> list[dict[str, str]]:
    with INPUT_CSV.open(newline="") as f:
        return list(csv.DictReader(f))


def annotate_bars(ax, bars, values, fmt: str) -> None:
    ymax = max(values) if values else 0.0
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + ymax * 0.02,
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )


def plot_single(
    rows: list[dict[str, str]],
    value_key: str,
    ylabel: str,
    title: str,
    fmt: str,
    output_path: Path,
    is_percent: bool,
) -> None:
    ordered = sorted(rows, key=lambda row: PROMPT_ORDER.index(row["prompt"]))
    labels = [row["prompt"].replace(" + ", " +\n") for row in ordered]
    values = [float(row[value_key]) for row in ordered]
    colors = [COLORS[row["prompt"]] for row in ordered]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    bars = ax.bar(labels, values, color=colors, width=0.6)
    ax.set_title(title, fontsize=14, pad=12)
    ax.set_ylabel(ylabel, fontsize=11)
    if is_percent:
        ax.set_ylim(0, 100)
    else:
        ax.set_ylim(0, max(values) * 1.25)
    ax.grid(axis="y", linestyle="--", alpha=0.32)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    annotate_bars(ax, bars, values, fmt)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rows = read_rows()
    plot_single(
        rows,
        "MAE",
        "Mean absolute error (mg copper per serving)",
        "Stage 1 Baseline vs Stage 3 Prompts: MAE (lower is better)",
        "{:.4f}",
        RUNS_DIR / "baseline_stage3_mae.png",
        is_percent=False,
    )
    plot_single(
        rows,
        "accuracy_10_percent",
        "Per-recipe accuracy within 10% (%)",
        "Stage 1 Baseline vs Stage 3 Prompts: Accuracy within 10% (higher is better)",
        "{:.1f}%",
        RUNS_DIR / "baseline_stage3_accuracy_10_percent.png",
        is_percent=True,
    )
    plot_single(
        rows,
        "accuracy_20_percent",
        "Per-recipe accuracy within 20% (%)",
        "Stage 1 Baseline vs Stage 3 Prompts: Accuracy within 20% (higher is better)",
        "{:.1f}%",
        RUNS_DIR / "baseline_stage3_accuracy_20_percent.png",
        is_percent=True,
    )
    print("Wrote:")
    for name in (
        "baseline_stage3_mae.png",
        "baseline_stage3_accuracy_10_percent.png",
        "baseline_stage3_accuracy_20_percent.png",
    ):
        print(f"  {RUNS_DIR / name}")


if __name__ == "__main__":
    main()
