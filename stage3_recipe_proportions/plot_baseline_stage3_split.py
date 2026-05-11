#!/usr/bin/env python3
"""Split MAE and Per-Recipe Accuracy within 20% into two separate bar charts.

Reads `replicate_runs/baseline_stage3_mae_accuracy_comparison.csv` and writes:
  - replicate_runs/baseline_stage3_mae_bar.png
  - replicate_runs/baseline_stage3_accuracy_20_percent_bar.png
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt


RUNS_DIR = Path(__file__).resolve().parent / "replicate_runs"
CSV_PATH = RUNS_DIR / "baseline_stage3_mae_accuracy_comparison.csv"

PROMPT_ORDER = ["Baseline", "Combined persona + few-shot", "Persona"]
COLORS = ["#4C78A8", "#F58518", "#54A24B"]


def read_rows() -> list[dict[str, str]]:
    with CSV_PATH.open(newline="") as f:
        return list(csv.DictReader(f))


def plot_bar(labels, values, title, ylabel, fmt, output, is_percent):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    bars = ax.bar(labels, values, color=COLORS, width=0.6)
    ax.set_title(title, fontsize=14, pad=12)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_ylim(0, 100 if is_percent else max(values) * 1.25)
    ax.grid(axis="y", linestyle="--", alpha=0.32)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.02,
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rows = sorted(read_rows(), key=lambda r: PROMPT_ORDER.index(r["prompt"]))
    labels = [r["prompt"].replace(" + ", " +\n") for r in rows]

    plot_bar(
        labels,
        [float(r["MAE"]) for r in rows],
        "MAE",
        "Mean absolute error (mg copper per serving)",
        "{:.4f}",
        RUNS_DIR / "baseline_stage3_mae_bar.png",
        is_percent=False,
    )
    plot_bar(
        labels,
        [float(r["accuracy_20_percent"]) for r in rows],
        "Per-Recipe Accuracy within 20%",
        "Accuracy within 20% (%)",
        "{:.1f}%",
        RUNS_DIR / "baseline_stage3_accuracy_20_percent_bar.png",
        is_percent=True,
    )
    print(f"Wrote {RUNS_DIR / 'baseline_stage3_mae_bar.png'}")
    print(f"Wrote {RUNS_DIR / 'baseline_stage3_accuracy_20_percent_bar.png'}")


if __name__ == "__main__":
    main()
