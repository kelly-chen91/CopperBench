#!/usr/bin/env python3
"""Plot averaged Stage 2/2.5 replicate evaluation metrics."""

from __future__ import annotations

import csv
import os
from collections import defaultdict
from pathlib import Path
from typing import Any


STAGE25_DIR = Path(__file__).resolve().parent
RUNS_DIR = STAGE25_DIR / "replicate_runs"
AVERAGES_CSV = RUNS_DIR / "replicate_metric_averages.csv"
METRICS_CSV = RUNS_DIR / "replicate_metrics.csv"
OUTPUT_DIR = RUNS_DIR / "averaged_evaluation_results"

VARIANT_ORDER = [
    "Baseline",
    "Persona",
    "Few-shot",
    "CoT",
    "Combined persona + few-shot",
]


def require_matplotlib():
    cache_dir = STAGE25_DIR / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir / "xdg"))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def variant_sort_key(row: dict[str, Any]) -> tuple[int, str]:
    prompt_variant = str(row["prompt_variant"])
    try:
        return (VARIANT_ORDER.index(prompt_variant), prompt_variant)
    except ValueError:
        return (len(VARIANT_ORDER), prompt_variant)


def load_average_rows() -> list[dict[str, Any]]:
    rows = []
    for row in read_csv(AVERAGES_CSV):
        rows.append(
            {
                "stage": row["stage"],
                "prompt_variant": row["prompt_variant"],
                "model_name": row["model_name"],
                "run_count": int(row["run_count"]),
                "avg_MAE": float(row["avg_MAE"]),
                "stddev_MAE": float(row["stddev_MAE"]),
                "avg_mean_signed_error_mg_per_serving": float(
                    row["avg_mean_signed_error_mg_per_serving"]
                ),
                "stddev_mean_signed_error_mg_per_serving": float(
                    row["stddev_mean_signed_error_mg_per_serving"]
                ),
                "avg_accuracy_20_percent": float(row["avg_accuracy_20_percent"]),
                "stddev_accuracy_20_percent": float(row["stddev_accuracy_20_percent"]),
            }
        )
    return rows


def averaged_model_ranking_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(rows, key=lambda row: (row["avg_MAE"], -row["avg_accuracy_20_percent"]))
    return [
        {
            "rank": index,
            "stage": row["stage"],
            "prompt_variant": row["prompt_variant"],
            "model_name": row["model_name"],
            "run_count": row["run_count"],
            "avg_MAE": row["avg_MAE"],
            "stddev_MAE": row["stddev_MAE"],
            "avg_mean_signed_error_mg_per_serving": row[
                "avg_mean_signed_error_mg_per_serving"
            ],
            "stddev_mean_signed_error_mg_per_serving": row[
                "stddev_mean_signed_error_mg_per_serving"
            ],
            "avg_accuracy_20_percent": row["avg_accuracy_20_percent"],
            "stddev_accuracy_20_percent": row["stddev_accuracy_20_percent"],
        }
        for index, row in enumerate(ranked, start=1)
    ]


def plot_bar_with_error(
    rows: list[dict[str, Any]],
    value_key: str,
    error_key: str,
    title: str,
    ylabel: str,
    output_path: Path,
    *,
    zero_line: bool = False,
) -> None:
    plt = require_matplotlib()
    ordered = sorted(rows, key=variant_sort_key)
    labels = [row["prompt_variant"] for row in ordered]
    values = [row[value_key] for row in ordered]
    errors = [row[error_key] for row in ordered]
    colors = ["#4C78A8" if value >= 0 else "#F58518" for value in values]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(labels, values, yerr=errors, capsize=5, color=colors)
    if zero_line:
        ax.axhline(0, color="#222222", linewidth=1)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", labelrotation=25, labelsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.45)

    for index, value in enumerate(values):
        vertical_offset = 4 if value >= 0 else -14
        va = "bottom" if value >= 0 else "top"
        ax.annotate(
            f"{value:.3f}" if "accuracy" not in value_key else f"{value:.1f}%",
            xy=(index, value),
            xytext=(0, vertical_offset),
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_run_trends(metric_rows: list[dict[str, str]], output_dir: Path) -> list[Path]:
    plt = require_matplotlib()
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in metric_rows:
        grouped[row["prompt_variant"]].append(row)

    metric_specs = [
        ("MAE", "MAE per Run (lower is better)", "MAE (mg copper per serving)", "per_run_mae_trend.png"),
        (
            "mean_signed_error_mg_per_serving",
            "Mean Signed Error per Run (closer to zero is better)",
            "Mean signed error (mg copper per serving)",
            "per_run_mean_signed_error_trend.png",
        ),
        (
            "accuracy_20_percent",
            "Accuracy Within 20% per Run (higher is better)",
            "Accuracy within 20% (%)",
            "per_run_accuracy_20_percent_trend.png",
        ),
    ]

    written: list[Path] = []
    for metric_key, title, ylabel, filename in metric_specs:
        fig, ax = plt.subplots(figsize=(8, 5))
        for prompt_variant in VARIANT_ORDER:
            rows = sorted(grouped.get(prompt_variant, []), key=lambda row: int(row["run"]))
            if not rows:
                continue
            ax.plot(
                [int(row["run"]) for row in rows],
                [float(row[metric_key]) for row in rows],
                marker="o",
                label=prompt_variant,
            )
        if metric_key == "mean_signed_error_mg_per_serving":
            ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
        ax.set_title(title)
        ax.set_xlabel("Run")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", linestyle=":", alpha=0.45)
        ax.legend(fontsize=8, loc="best")
        fig.tight_layout()
        out = output_dir / filename
        fig.savefig(out, dpi=180)
        plt.close(fig)
        written.append(out)
    return written


def aggregate_recipe_type_rows() -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    metadata_by_model: dict[str, dict[str, str]] = {}

    for run_dir in sorted(RUNS_DIR.glob("run_*")):
        path = run_dir / "evaluation_results_20_percent" / "recipe_type_metrics.csv"
        for row in read_csv(path):
            key = (row["model_name"], row["recipe_type"])
            grouped[key].append(row)
            metadata_by_model[row["model_name"]] = row

    rows: list[dict[str, Any]] = []
    for (model_name, recipe_type), group_rows in sorted(grouped.items()):
        metadata = metadata_by_model[model_name]
        rows.append(
            {
                "model_name": model_name,
                "prompt_variant": prompt_variant_for_model(model_name),
                "recipe_type": recipe_type,
                "run_count": len(group_rows),
                "avg_MAE": sum(float(row["MAE"]) for row in group_rows) / len(group_rows),
                "avg_accuracy_20_percent": sum(
                    float(row["accuracy_percent"]) for row in group_rows
                )
                / len(group_rows),
                "valid_predictions_count": metadata["valid_predictions_count"],
                "total_recipes": metadata["total_recipes"],
            }
        )
    return rows


def prompt_variant_for_model(model_name: str) -> str:
    if model_name == "gpt_5_4_mini__reasoning_medium":
        return "Baseline"
    if model_name.startswith("persona__"):
        return "Persona"
    if model_name.startswith("few_shot__"):
        return "Few-shot"
    if model_name.startswith("cot__"):
        return "CoT"
    if model_name.startswith("combined_persona_few_shot__"):
        return "Combined persona + few-shot"
    return model_name


def plot_recipe_type_heatmap(rows: list[dict[str, Any]], output_path: Path) -> None:
    plt = require_matplotlib()
    recipe_types = sorted({row["recipe_type"] for row in rows})
    variants = [variant for variant in VARIANT_ORDER if any(row["prompt_variant"] == variant for row in rows)]
    value_by_key = {
        (row["prompt_variant"], row["recipe_type"]): float(row["avg_MAE"])
        for row in rows
    }
    matrix = [
        [value_by_key[(variant, recipe_type)] for variant in variants]
        for recipe_type in recipe_types
    ]

    fig, ax = plt.subplots(figsize=(11, 5.8))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_title("Average MAE by Recipe Type Across Replicate Runs")
    ax.set_xticks(range(len(variants)), variants)
    ax.set_yticks(range(len(recipe_types)), recipe_types)
    ax.tick_params(axis="x", labelrotation=25, labelsize=8)

    for y, recipe_type in enumerate(recipe_types):
        for x, variant in enumerate(variants):
            value = value_by_key[(variant, recipe_type)]
            ax.text(x, y, f"{value:.3f}", ha="center", va="center", fontsize=8, color="white")

    fig.colorbar(image, ax=ax, label="Average MAE")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    average_rows = load_average_rows()
    ranking_rows = averaged_model_ranking_rows(average_rows)
    write_csv(OUTPUT_DIR / "averaged_model_ranking.csv", ranking_rows)

    recipe_type_rows = aggregate_recipe_type_rows()
    write_csv(OUTPUT_DIR / "averaged_recipe_type_metrics.csv", recipe_type_rows)

    metric_rows = read_csv(METRICS_CSV)
    plot_bar_with_error(
        average_rows,
        "avg_MAE",
        "stddev_MAE",
        "Average MAE Across 3 Stage 2/2.5 Runs",
        "Average MAE (mg copper per serving)",
        OUTPUT_DIR / "average_mae_comparison.png",
    )
    plot_bar_with_error(
        average_rows,
        "avg_mean_signed_error_mg_per_serving",
        "stddev_mean_signed_error_mg_per_serving",
        "Average Mean Signed Error Across 3 Stage 2/2.5 Runs",
        "Average mean signed error (mg copper per serving)",
        OUTPUT_DIR / "average_mean_signed_error_comparison.png",
        zero_line=True,
    )
    plot_bar_with_error(
        average_rows,
        "avg_accuracy_20_percent",
        "stddev_accuracy_20_percent",
        "Average Accuracy Within 20% Across 3 Stage 2/2.5 Runs",
        "Average accuracy within 20% (%)",
        OUTPUT_DIR / "average_accuracy_20_percent_comparison.png",
    )
    plot_run_trends(metric_rows, OUTPUT_DIR)
    plot_recipe_type_heatmap(
        recipe_type_rows,
        OUTPUT_DIR / "average_mae_by_recipe_type_heatmap.png",
    )

    print(f"Wrote averaged evaluation outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
