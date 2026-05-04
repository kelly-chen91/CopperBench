#!/usr/bin/env python3
"""Decompose Stage 2 signed error by Stage 1.5 failure-mode cluster."""

from __future__ import annotations

import csv
import json
import os
import statistics
from collections import Counter, defaultdict
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
STAGE2_DIR = ROOT_DIR / "stage2_recipe_no_proportions"
STAGE15_DIR = ROOT_DIR / "stage1.5_clustering_analysis"
RESULTS_DIR = STAGE2_DIR / "evaluation_results"

PREDICTIONS_CSV = RESULTS_DIR / "per_recipe_predictions.csv"
CLUSTERS_CSV = STAGE15_DIR / "clusters_with_labels.csv"
LABELS_JSON = STAGE15_DIR / "failure_mode_labels.json"
DECOMPOSITION_CSV = RESULTS_DIR / "signed_error_by_failure_mode.csv"
ASSIGNMENTS_CSV = RESULTS_DIR / "recipe_failure_mode_assignments.csv"
HEATMAP_PNG = RESULTS_DIR / "signed_error_by_failure_mode_heatmap.png"

MODEL_LABELS = {
    "gpt_5_4_mini__reasoning_medium": "Baseline",
    "few_shot__gpt_5_4_mini__reasoning_medium": "Few-shot",
    "cot__gpt_5_4_mini__reasoning_medium": "CoT",
    "persona__gpt_5_4_mini__reasoning_medium": "Persona",
}
MODEL_ORDER = [
    "gpt_5_4_mini__reasoning_medium",
    "few_shot__gpt_5_4_mini__reasoning_medium",
    "cot__gpt_5_4_mini__reasoning_medium",
    "persona__gpt_5_4_mini__reasoning_medium",
]
UNCLUSTERED_ID = "unclustered"
UNCLUSTERED_LABEL = "No high-error Stage 1.5 cluster"
MIXED_ID = "mixed"
MIXED_LABEL = "Mixed Stage 1.5 failure modes"


def require_matplotlib():
    cache_dir = STAGE2_DIR / "stage1_recipe_proportions" / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir / "xdg"))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    return plt, TwoSlopeNorm


def load_failure_mode_labels() -> dict[str, str]:
    labels = json.loads(LABELS_JSON.read_text())
    return {
        cluster_id: str(payload.get("label", f"Cluster {cluster_id}"))
        for cluster_id, payload in labels.items()
    }


def load_recipe_assignments() -> dict[int, dict[str, str]]:
    labels = load_failure_mode_labels()
    cluster_counts_by_recipe: dict[int, Counter[str]] = defaultdict(Counter)
    recipe_names: dict[int, str] = {}

    with CLUSTERS_CSV.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            recipe_index = int(row["recipe_index"])
            cluster_counts_by_recipe[recipe_index][row["cluster_id"]] += 1
            recipe_names[recipe_index] = row["recipe_name"]

    assignments: dict[int, dict[str, str]] = {}
    for recipe_index, cluster_counts in cluster_counts_by_recipe.items():
        max_count = max(cluster_counts.values())
        top_cluster_ids = sorted(
            cluster_id
            for cluster_id, count in cluster_counts.items()
            if count == max_count
        )
        if len(top_cluster_ids) > 1:
            cluster_id = MIXED_ID
            failure_mode_label = MIXED_LABEL
        else:
            cluster_id = top_cluster_ids[0]
            failure_mode_label = labels.get(cluster_id, f"Cluster {cluster_id}")

        assignments[recipe_index] = {
            "recipe_index": str(recipe_index),
            "recipe_name": recipe_names[recipe_index],
            "assigned_cluster_id": cluster_id,
            "failure_mode_label": failure_mode_label,
            "stage15_cluster_counts": json.dumps(
                dict(sorted(cluster_counts.items())),
                sort_keys=True,
            ),
        }
    return assignments


def write_assignments(prediction_rows: list[dict[str, str]]) -> dict[int, dict[str, str]]:
    assignments = load_recipe_assignments()
    recipe_names = {
        int(row["recipe_index"]): row["recipe_name"]
        for row in prediction_rows
        if row["model_name"] == MODEL_ORDER[0]
    }
    rows: list[dict[str, str]] = []
    for recipe_index, recipe_name in sorted(recipe_names.items()):
        assignment = assignments.get(recipe_index)
        if assignment is None:
            assignment = {
                "recipe_index": str(recipe_index),
                "recipe_name": recipe_name,
                "assigned_cluster_id": UNCLUSTERED_ID,
                "failure_mode_label": UNCLUSTERED_LABEL,
                "stage15_cluster_counts": "{}",
            }
            assignments[recipe_index] = assignment
        rows.append(assignment)

    with ASSIGNMENTS_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return assignments


def load_prediction_rows() -> list[dict[str, str]]:
    with PREDICTIONS_CSV.open(newline="") as handle:
        return list(csv.DictReader(handle))


def summarize_signed_errors(
    prediction_rows: list[dict[str, str]],
    assignments: dict[int, dict[str, str]],
) -> list[dict[str, object]]:
    errors: dict[tuple[str, str], list[float]] = defaultdict(list)
    labels_by_cluster_id: dict[str, str] = {}

    for row in prediction_rows:
        model_name = row["model_name"]
        if model_name not in MODEL_LABELS or row["valid"] != "True":
            continue
        recipe_index = int(row["recipe_index"])
        assignment = assignments[recipe_index]
        cluster_id = assignment["assigned_cluster_id"]
        labels_by_cluster_id[cluster_id] = assignment["failure_mode_label"]
        prediction = float(row["predicted_copper_mg_per_serving"])
        truth = float(row["ground_truth_copper_mg_per_serving"])
        errors[(model_name, cluster_id)].append(prediction - truth)

    cluster_order = sorted(
        labels_by_cluster_id,
        key=lambda cluster_id: (
            cluster_id == UNCLUSTERED_ID,
            cluster_id == MIXED_ID,
            int(cluster_id) if cluster_id.lstrip("-").isdigit() else 999,
        ),
    )
    rows: list[dict[str, object]] = []
    for model_name in MODEL_ORDER:
        for cluster_id in cluster_order:
            values = errors.get((model_name, cluster_id), [])
            rows.append(
                {
                    "model_name": model_name,
                    "prompt_variant": MODEL_LABELS[model_name],
                    "assigned_cluster_id": cluster_id,
                    "failure_mode_label": labels_by_cluster_id[cluster_id],
                    "recipe_count": len(values),
                    "mean_signed_error_mg_per_serving": statistics.fmean(values)
                    if values
                    else "",
                    "median_signed_error_mg_per_serving": statistics.median(values)
                    if values
                    else "",
                    "overestimate_count": sum(1 for value in values if value > 0),
                    "underestimate_count": sum(1 for value in values if value < 0),
                    "zero_error_count": sum(1 for value in values if value == 0),
                }
            )
    return rows


def write_decomposition(rows: list[dict[str, object]]) -> None:
    with DECOMPOSITION_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_heatmap(rows: list[dict[str, object]]) -> None:
    plt, TwoSlopeNorm = require_matplotlib()
    cluster_ids = []
    for row in rows:
        cluster_id = str(row["assigned_cluster_id"])
        if cluster_id not in cluster_ids:
            cluster_ids.append(cluster_id)

    values_by_key = {
        (str(row["model_name"]), str(row["assigned_cluster_id"])): row
        for row in rows
        if row["mean_signed_error_mg_per_serving"] != ""
    }
    matrix = [
        [
            float(values_by_key[(model_name, cluster_id)]["mean_signed_error_mg_per_serving"])
            for cluster_id in cluster_ids
        ]
        for model_name in MODEL_ORDER
    ]

    max_abs = max(abs(value) for model_values in matrix for value in model_values)
    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)
    labels = [
        "\n".join(str(rows_by_cluster[0]["failure_mode_label"]).split()[:4])
        for rows_by_cluster in (
            [row for row in rows if row["assigned_cluster_id"] == cluster_id]
            for cluster_id in cluster_ids
        )
    ]

    fig_width = max(11, len(cluster_ids) * 2.2)
    fig, ax = plt.subplots(figsize=(fig_width, 5.8))
    image = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", norm=norm)
    ax.set_title("Stage 2 Mean Signed Error by Stage 1.5 Failure Mode")
    ax.set_xticks(range(len(cluster_ids)), labels)
    ax.set_yticks(range(len(MODEL_ORDER)), [MODEL_LABELS[name] for name in MODEL_ORDER])
    ax.tick_params(axis="x", labelrotation=25, labelsize=8)

    for y, model_name in enumerate(MODEL_ORDER):
        for x, cluster_id in enumerate(cluster_ids):
            row = values_by_key[(model_name, cluster_id)]
            value = float(row["mean_signed_error_mg_per_serving"])
            count = int(row["recipe_count"])
            ax.text(
                x,
                y,
                f"{value:+.3f}\nn={count}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if abs(value) > max_abs * 0.55 else "black",
            )

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Mean signed error (mg copper per serving)")
    fig.tight_layout()
    fig.savefig(HEATMAP_PNG, dpi=180)
    plt.close(fig)


def main() -> None:
    prediction_rows = load_prediction_rows()
    assignments = write_assignments(prediction_rows)
    decomposition_rows = summarize_signed_errors(prediction_rows, assignments)
    write_decomposition(decomposition_rows)
    plot_heatmap(decomposition_rows)
    print(f"Wrote {DECOMPOSITION_CSV}")
    print(f"Wrote {ASSIGNMENTS_CSV}")
    print(f"Wrote {HEATMAP_PNG}")


if __name__ == "__main__":
    main()
