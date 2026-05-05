#!/usr/bin/env python3
"""Plot Stage 3 mitigation MAE over Stage 1.5 failure-mode clusters."""

from __future__ import annotations

import csv
import os
import statistics
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
STAGE15_DIR = ROOT_DIR / "stage1.5_clustering_analysis"
STAGE3_DIR = ROOT_DIR / "stage3_recipe_proportions"
RUNS_DIR = STAGE3_DIR / "replicate_runs"
BASELINE_MODEL = "gpt_5_4_mini__reasoning_medium"

os.environ.setdefault("MPLCONFIGDIR", str(STAGE3_DIR / ".matplotlib_cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(STAGE3_DIR / ".cache"))


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


def load_cluster_recipes() -> tuple[dict[int, set[int]], dict[int, str]]:
    cluster_recipes: dict[int, set[int]] = defaultdict(set)
    for row in read_csv(STAGE15_DIR / "clusters_with_labels.csv"):
        cluster_recipes[int(row["cluster_id"])].add(int(row["recipe_index"]))

    cluster_labels = {
        int(row["cluster_id"]): str(row["failure_mode_label"])
        for row in read_csv(STAGE15_DIR / "error_cluster_analysis.csv")
    }
    return dict(cluster_recipes), cluster_labels


def load_baseline_recipe_mae() -> dict[int, float]:
    values: dict[int, float] = {}
    for row in read_csv(
        ROOT_DIR / "stage1_recipe_proportions" / "evaluation_results" / "per_recipe_predictions.csv"
    ):
        if row["model_name"] != BASELINE_MODEL or row["valid"] != "True":
            continue
        values[int(row["recipe_index"])] = float(row["absolute_error"])
    return values


def load_stage3_recipe_mae() -> dict[str, dict[int, float]]:
    errors_by_prompt_recipe: dict[str, dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in read_csv(RUNS_DIR / "per_recipe_replicate_metrics.csv"):
        errors_by_prompt_recipe[row["prompt_variant"]][int(row["recipe_index"])].append(
            float(row["MAE"])
        )

    return {
        prompt: {
            recipe_index: statistics.fmean(errors)
            for recipe_index, errors in recipe_errors.items()
        }
        for prompt, recipe_errors in errors_by_prompt_recipe.items()
    }


def mean_for_recipes(recipe_mae: dict[int, float], recipe_ids: set[int]) -> float:
    values = [recipe_mae[recipe_id] for recipe_id in sorted(recipe_ids) if recipe_id in recipe_mae]
    if not values:
        raise ValueError("No matching recipe MAE values found for cluster.")
    return statistics.fmean(values)


def build_rows() -> list[dict[str, Any]]:
    cluster_recipes, cluster_labels = load_cluster_recipes()
    baseline_mae = load_baseline_recipe_mae()
    stage3_mae = load_stage3_recipe_mae()

    rows: list[dict[str, Any]] = []
    prompt_order = [
        ("Baseline", baseline_mae),
        ("Combined persona + few-shot", stage3_mae["Combined persona + few-shot"]),
        ("Persona", stage3_mae["Persona"]),
    ]
    for cluster_id in sorted(cluster_recipes):
        recipe_ids = cluster_recipes[cluster_id]
        for prompt, recipe_mae in prompt_order:
            rows.append(
                {
                    "cluster_id": cluster_id,
                    "failure_mode_label": cluster_labels[cluster_id],
                    "unique_recipe_count": len(recipe_ids),
                    "prompt": prompt,
                    "mean_absolute_error": mean_for_recipes(recipe_mae, recipe_ids),
                }
            )
    return rows


def cluster_axis_label(cluster_id: int, failure_mode_label: str) -> str:
    wrapped = "\n".join(textwrap.wrap(failure_mode_label, width=24))
    return f"Cluster {cluster_id}\n{wrapped}"


def plot(rows: list[dict[str, Any]]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    prompt_order = ["Baseline", "Combined persona + few-shot", "Persona"]
    colors = {
        "Baseline": "#4c78a8",
        "Combined persona + few-shot": "#f58518",
        "Persona": "#54a24b",
    }
    prompt_labels = {
        "Baseline": "Baseline",
        "Combined persona + few-shot": "Combined\npersona +\nfew-shot",
        "Persona": "Persona",
    }

    clusters = sorted({int(row["cluster_id"]) for row in rows})
    cluster_labels = {
        int(row["cluster_id"]): str(row["failure_mode_label"])
        for row in rows
    }
    values = {
        (int(row["cluster_id"]), row["prompt"]): float(row["mean_absolute_error"])
        for row in rows
    }

    x = np.arange(len(clusters))
    width = 0.24
    offsets = {
        "Baseline": -width,
        "Combined persona + few-shot": 0,
        "Persona": width,
    }
    y_max = max(values.values()) * 1.16

    plt.rcParams.update(
        {
            "axes.titlesize": 15,
            "axes.labelsize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
        }
    )

    fig, ax = plt.subplots(figsize=(14, 7.5))
    for prompt in prompt_order:
        bar_values = [values[(cluster_id, prompt)] for cluster_id in clusters]
        bars = ax.bar(
            x + offsets[prompt],
            bar_values,
            width,
            label=prompt_labels[prompt],
            color=colors[prompt],
            edgecolor="none",
        )
        for bar, value in zip(bars, bar_values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                max(value - y_max * 0.035, value * 0.5),
                f"{value:.3f}",
                ha="center",
                va="center",
                color="white",
                fontsize=9,
                fontweight="bold",
            )

    ax.set_title("Mean Absolute Error by Cluster: Baseline vs Stage 3 Mitigations", pad=14)
    ax.set_ylabel("Mean Absolute Error (mg)")
    ax.set_xlabel("Failure mode cluster")
    ax.set_ylim(0, y_max)
    ax.set_xticks(
        x,
        [cluster_axis_label(cluster_id, cluster_labels[cluster_id]) for cluster_id in clusters],
    )
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    output_path = RUNS_DIR / "stage3_cluster_mitigation_mae_by_cluster.png"
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def main() -> None:
    rows = build_rows()
    output_csv = RUNS_DIR / "stage3_cluster_mitigation_mae_by_cluster.csv"
    write_csv(output_csv, rows)
    output_png = plot(rows)
    print(f"Wrote {output_csv}")
    print(f"Wrote {output_png}")


if __name__ == "__main__":
    main()
