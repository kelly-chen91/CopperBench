#!/usr/bin/env python3
"""Evaluate Stage 2 outputs with a 20% relative accuracy threshold."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


STAGE2_DIR = Path(__file__).resolve().parent
ROOT_DIR = STAGE2_DIR.parent
RESULTS_DIR = STAGE2_DIR / "evaluation_results_20_percent"
ACCURACY_RELATIVE_TOLERANCE = 0.20

sys.path.insert(0, str(ROOT_DIR / "stage1_recipe_proportions"))
import evaluate_model_outputs  # noqa: E402


def relabel_detail_rows(detail_rows: list[dict[str, Any]]) -> None:
    for row in detail_rows:
        row["within_20_percent"] = row.pop("within_10_percent")


def write_20_percent_plots(
    overall_rows: list[dict[str, Any]],
    recipe_type_rows: list[dict[str, Any]],
    results_dir: Path,
) -> None:
    evaluate_model_outputs.plot_overall_bar(
        overall_rows,
        "MAE",
        "MAE Comparison Across Models",
        "Mean absolute error (mg copper per serving)",
        results_dir / "mae_comparison.png",
    )
    evaluate_model_outputs.plot_overall_bar(
        overall_rows,
        "accuracy_percent",
        "Accuracy Within +/-20% Across Models",
        "Accuracy (%)",
        results_dir / "accuracy_comparison.png",
    )
    evaluate_model_outputs.plot_recipe_type_heatmap(
        recipe_type_rows,
        results_dir / "mae_by_recipe_type_heatmap.png",
    )
    evaluate_model_outputs.plot_reasoning_effect(
        overall_rows,
        results_dir / "reasoning_effort_effect.png",
    )


def main() -> None:
    evaluate_model_outputs.ACCURACY_RELATIVE_TOLERANCE = ACCURACY_RELATIVE_TOLERANCE
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_truths = evaluate_model_outputs.load_truth(
        STAGE2_DIR / "recipes_no_proportions.json"
    )
    model_files = sorted((STAGE2_DIR / "model_outputs").glob("*.jsonl"))
    if not model_files:
        raise SystemExit(f"No JSONL files found in {STAGE2_DIR / 'model_outputs'}")

    overall_rows: list[dict[str, Any]] = []
    recipe_type_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []

    for model_file in model_files:
        model_name = evaluate_model_outputs.model_name_from_path(model_file)
        predictions, file_metadata = evaluate_model_outputs.load_model_predictions(model_file)
        model_overall, model_recipe_types, model_details = (
            evaluate_model_outputs.summarize_predictions(
                {model_name: predictions},
                all_truths,
            )
        )
        relabel_detail_rows(model_details)
        overall_rows.extend(model_overall)
        recipe_type_rows.extend(model_recipe_types)
        detail_rows.extend(model_details)
        file_metadata["scored_recipe_count"] = len(all_truths)
        file_metadata["excluded_recipe_count"] = 0
        metadata.append(file_metadata)

    overall_rows.sort(
        key=lambda row: (
            float("inf") if row["MAE"] == "" else row["MAE"],
            float("-inf") if row["accuracy_percent"] == "" else -row["accuracy_percent"],
            row["model_name"],
        )
    )
    evaluate_model_outputs.write_csv(RESULTS_DIR / "model_ranking.csv", overall_rows)
    evaluate_model_outputs.write_csv(RESULTS_DIR / "recipe_type_metrics.csv", recipe_type_rows)
    evaluate_model_outputs.write_csv(RESULTS_DIR / "per_recipe_predictions.csv", detail_rows)
    evaluate_model_outputs.write_json(
        RESULTS_DIR / "evaluation_metadata.json",
        {
            "accuracy_relative_tolerance": ACCURACY_RELATIVE_TOLERANCE,
            "source_recipe_count": len(all_truths),
            "recipe_count": len(all_truths),
            "excluded_recipe_count": 0,
            "model_file_count": len(model_files),
            "model_files": [path.name for path in model_files],
            "model_output_metadata": metadata,
            "exclusion_policy": "All prompt variants are scored on all Stage 2 recipes.",
        },
    )
    write_20_percent_plots(overall_rows, recipe_type_rows, RESULTS_DIR)

    best = overall_rows[0]
    print(
        "Best model: "
        f"{best['model_name']} "
        f"(MAE={best['MAE']:.6f}, accuracy={best['accuracy_percent']:.2f}%, "
        f"valid={best['valid_predictions_count']}/{best['total_recipes']})"
    )
    print(f"Wrote 20% evaluation outputs to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
