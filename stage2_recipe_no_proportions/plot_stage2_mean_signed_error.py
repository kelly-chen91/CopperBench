#!/usr/bin/env python3
"""Plot Stage 2 mean signed copper prediction error by prompt variant."""

from __future__ import annotations

import csv
import os
import sys
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


STAGE2_DIR = Path(__file__).resolve().parent
RESULTS_DIR = STAGE2_DIR / "evaluation_results"
ROOT_DIR = STAGE2_DIR.parent
RECIPES_JSON = STAGE2_DIR / "recipes_no_proportions.json"
MODEL_OUTPUTS_DIR = STAGE2_DIR / "model_outputs"
OUTPUT_PNG = RESULTS_DIR / "mean_signed_error_comparison.png"
OUTPUT_CSV = RESULTS_DIR / "mean_signed_error_summary.csv"

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


sys.path.insert(0, str(ROOT_DIR / "stage1_recipe_proportions"))
import evaluate_model_outputs  # noqa: E402


def require_matplotlib():
    cache_dir = STAGE2_DIR / "stage1_recipe_proportions" / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir / "xdg"))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def load_signed_errors() -> dict[str, list[float]]:
    truths = evaluate_model_outputs.load_truth(RECIPES_JSON)
    truth_by_index = {truth.recipe_index: truth for truth in truths}
    errors_by_model: dict[str, list[float]] = defaultdict(list)
    for model_name in MODEL_ORDER:
        model_file = MODEL_OUTPUTS_DIR / f"{model_name}.jsonl"
        predictions, _metadata = evaluate_model_outputs.load_model_predictions(model_file)
        for prediction in predictions:
            if (
                not prediction.valid
                or prediction.recipe_index is None
                or prediction.predicted_copper_mg_per_serving is None
            ):
                continue
            truth = truth_by_index.get(prediction.recipe_index)
            if truth is None:
                continue
            errors_by_model[model_name].append(
                prediction.predicted_copper_mg_per_serving
                - truth.copper_per_serving_mg
            )
    return errors_by_model


def write_summary(errors_by_model: dict[str, list[float]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model_name in MODEL_ORDER:
        errors = errors_by_model.get(model_name, [])
        if not errors:
            rows.append(
                {
                    "model_name": model_name,
                    "prompt_variant": MODEL_LABELS[model_name],
                    "valid_predictions_count": 0,
                    "mean_signed_error_mg_per_serving": "",
                }
            )
            continue
        rows.append(
            {
                "model_name": model_name,
                "prompt_variant": MODEL_LABELS[model_name],
                "valid_predictions_count": len(errors),
                "mean_signed_error_mg_per_serving": statistics.fmean(errors),
            }
        )

    with OUTPUT_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def plot_summary(rows: list[dict[str, object]]) -> None:
    plt = require_matplotlib()
    labels = [
        f"{row['prompt_variant']}\n(n={row['valid_predictions_count']})"
        for row in rows
    ]
    values = [float(row["mean_signed_error_mg_per_serving"]) for row in rows]
    colors = ["#4C78A8" if value >= 0 else "#F58518" for value in values]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, values, color=colors)
    ax.axhline(0, color="#222222", linewidth=1)
    ax.set_title("Stage 2 Mean Signed Error by Prompt Variant")
    ax.set_ylabel("Mean signed error (mg copper per serving)")
    ax.grid(axis="y", linestyle=":", alpha=0.45)

    for index, value in enumerate(values):
        vertical_offset = 3 if value >= 0 else -14
        va = "bottom" if value >= 0 else "top"
        ax.annotate(
            f"{value:+.3f}",
            xy=(index, value),
            xytext=(0, vertical_offset),
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=180)
    plt.close(fig)


def main() -> None:
    errors_by_model = load_signed_errors()
    rows = write_summary(errors_by_model)
    plot_summary(rows)
    print(f"Wrote {OUTPUT_PNG}")
    print(f"Wrote {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
