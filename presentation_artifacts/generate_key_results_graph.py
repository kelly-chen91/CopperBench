#!/usr/bin/env python3
"""Generate per-stage key-results figures for the presentation.

Each stage gets its own PNG so MAE (lower is better) and accuracy
(higher is better) are never shown in the same figure.
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


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "presentation_artifacts"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


SHORT_NAME_OVERRIDES = {
    "gpt_4_1": "4.1",
    "gpt_4_1_mini": "4.1-mini",
    "gpt_4_1_nano": "4.1-nano",
    "gpt_4o": "4o",
    "gpt_4o_mini": "4o-mini",
    "gpt_5_4__reasoning_low": "5.4\nlow",
    "gpt_5_4__reasoning_medium": "5.4\nmedium",
    "gpt_5_4__reasoning_high": "5.4\nhigh",
    "gpt_5_4_mini__reasoning_low": "5.4-mini\nlow",
    "gpt_5_4_mini__reasoning_medium": "5.4-mini\nmedium",
    "gpt_5_4_mini__reasoning_high": "5.4-mini\nhigh",
    "gpt_5_4_nano__reasoning_low": "5.4-nano\nlow",
    "gpt_5_4_nano__reasoning_medium": "5.4-nano\nmedium",
    "gpt_5_4_nano__reasoning_high": "5.4-nano\nhigh",
    "gpt_5_4_pro__reasoning_high": "5.4-pro\nhigh",
    "gpt_5_5__reasoning_low": "5.5\nlow",
    "gpt_5_5__reasoning_medium": "5.5\nmedium",
    "gpt_5_5__reasoning_high": "5.5\nhigh",
}


def short_model_name(name: str) -> str:
    if name in SHORT_NAME_OVERRIDES:
        return SHORT_NAME_OVERRIDES[name]
    cleaned = name.removeprefix("gpt_").replace("__reasoning_", "\n")
    return cleaned.replace("_", "-")


def annotate_bars(ax, bars, values, suffix="", decimals=2, offset=0.02):
    ymax = max(values) if values else 0.0
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + ymax * offset,
            f"{value:.{decimals}f}{suffix}",
            ha="center",
            va="bottom",
            fontsize=10,
        )


def style(ax) -> None:
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_stage1_accuracy() -> Path:
    stage1 = read_csv(ROOT / "stage1_recipe_proportions" / "evaluation_results" / "model_ranking.csv")
    stage1_top = sorted(stage1, key=lambda row: float(row["accuracy_percent"]), reverse=True)[:6]
    labels = [short_model_name(row["model_name"]) for row in stage1_top]
    values = [float(row["accuracy_percent"]) for row in stage1_top]
    colors = [
        "#2A9D8F" if row["model_name"] == "gpt_5_4_mini__reasoning_medium" else "#7A8CA3"
        for row in stage1_top
    ]

    fig, ax = plt.subplots(figsize=(9, 5.6))
    bars = ax.bar(labels, values, color=colors, width=0.68)
    ax.set_title("Stage 1: best per-recipe accuracy within ±10%", fontsize=14, pad=12)
    ax.set_ylabel("Accuracy within ±10% (%)  — higher is better", fontsize=11)
    ax.set_ylim(0, max(values) * 1.32)
    ax.tick_params(axis="x", labelsize=9)
    annotate_bars(ax, bars, values, suffix="%", decimals=1)
    style(ax)
    out = OUT_DIR / "key_results_stage1_accuracy.png"
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_stage1_mae() -> Path:
    stage1 = read_csv(ROOT / "stage1_recipe_proportions" / "evaluation_results" / "model_ranking.csv")
    stage1_top = sorted(stage1, key=lambda row: float(row["MAE"]))[:6]
    labels = [short_model_name(row["model_name"]) for row in stage1_top]
    values = [float(row["MAE"]) for row in stage1_top]
    colors = [
        "#2A9D8F" if row["model_name"] == "gpt_5_4_mini__reasoning_medium" else "#7A8CA3"
        for row in stage1_top
    ]

    fig, ax = plt.subplots(figsize=(9, 5.6))
    bars = ax.bar(labels, values, color=colors, width=0.68)
    ax.set_title("Stage 1: best MAE", fontsize=14, pad=12)
    ax.set_ylabel("MAE (mg copper per serving)  — lower is better", fontsize=11)
    ax.set_ylim(0, max(values) * 1.32)
    ax.tick_params(axis="x", labelsize=9)
    annotate_bars(ax, bars, values, decimals=4)
    style(ax)
    out = OUT_DIR / "key_results_stage1_mae.png"
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_stage2_mae() -> Path:
    stage2 = read_csv(
        ROOT
        / "stage2.5_combined_prompts"
        / "replicate_runs"
        / "averaged_evaluation_results"
        / "averaged_model_ranking.csv"
    )
    order = ["Combined persona + few-shot", "Persona", "Baseline", "Few-shot", "CoT"]
    by_prompt = {row["prompt_variant"]: row for row in stage2}
    labels = ["Combined\npersona +\nfew-shot", "Persona", "Baseline", "Few-shot", "CoT"]
    values = [float(by_prompt[name]["avg_MAE"]) for name in order]
    colors = ["#2A9D8F", "#2A9D8F", "#7A8CA3", "#C06C84", "#C06C84"]

    fig, ax = plt.subplots(figsize=(9, 5.6))
    bars = ax.bar(labels, values, color=colors, width=0.68)
    ax.set_title("Stage 2: no-proportions prompt comparison", fontsize=14, pad=12)
    ax.set_ylabel("MAE (mg copper per serving)  — lower is better", fontsize=11)
    ax.set_ylim(0, max(values) * 1.28)
    ax.tick_params(axis="x", labelsize=9)
    annotate_bars(ax, bars, values, decimals=3)
    style(ax)
    out = OUT_DIR / "key_results_stage2_mae.png"
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_stage3_mae() -> Path:
    stage3 = read_csv(ROOT / "stage3_recipe_proportions" / "replicate_runs" / "replicate_metric_averages.csv")
    baseline = read_csv(
        ROOT / "stage3_recipe_proportions" / "replicate_runs" / "baseline_stage3_mae_accuracy_comparison.csv"
    )
    baseline_row = next(row for row in baseline if row["prompt"] == "Baseline")
    persona_row = next(row for row in stage3 if row["prompt_variant"] == "Persona")
    combined_row = next(row for row in stage3 if row["prompt_variant"] == "Combined persona + few-shot")
    labels = ["Baseline", "Persona", "Combined\npersona +\nfew-shot"]
    values = [
        float(baseline_row["MAE"]),
        float(persona_row["avg_MAE"]),
        float(combined_row["avg_MAE"]),
    ]
    colors = ["#7A8CA3", "#2A9D8F", "#2A9D8F"]

    fig, ax = plt.subplots(figsize=(8, 5.6))
    bars = ax.bar(labels, values, color=colors, width=0.6)
    ax.set_title("Stage 3: with-proportions mitigation", fontsize=14, pad=12)
    ax.set_ylabel("MAE (mg copper per serving)  — lower is better", fontsize=11)
    ax.set_ylim(0, max(values) * 1.32)
    ax.tick_params(axis="x", labelsize=9)
    annotate_bars(ax, bars, values, decimals=3)
    style(ax)
    out = OUT_DIR / "key_results_stage3_mae.png"
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    outputs = [
        plot_stage1_accuracy(),
        plot_stage1_mae(),
        plot_stage2_mae(),
        plot_stage3_mae(),
    ]
    print("Wrote:")
    for path in outputs:
        print(f"  {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
