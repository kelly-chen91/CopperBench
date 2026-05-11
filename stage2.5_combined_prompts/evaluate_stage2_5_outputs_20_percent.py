#!/usr/bin/env python3
"""Evaluate Stage 2.5 combined prompt outputs with a 20% relative threshold."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


STAGE25_DIR = Path(__file__).resolve().parent
ROOT_DIR = STAGE25_DIR.parent
STAGE2_DIR = ROOT_DIR / "stage2_recipe_no_proportions"
DEFAULT_ARGS = [
    "--recipes",
    str(STAGE2_DIR / "recipes_no_proportions.json"),
    "--results-dir",
    str(STAGE25_DIR / "evaluation_results_20_percent"),
]
MODEL_OUTPUT_DIRS = [
    STAGE2_DIR / "model_outputs",
    STAGE25_DIR / "model_outputs",
]


sys.path.insert(0, str(ROOT_DIR / "stage1_recipe_proportions"))
import evaluate_model_outputs  # noqa: E402


def relabel_detail_rows(detail_rows: list[dict[str, object]]) -> None:
    for row in detail_rows:
        row["within_20_percent"] = row.pop("within_10_percent")


def write_20_percent_plots(
    overall_rows: list[dict[str, object]],
    recipe_type_rows: list[dict[str, object]],
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


def load_predictions_from_dirs() -> tuple[dict[str, list[Any]], list[dict[str, Any]], list[str]]:
    predictions_by_model: dict[str, list[Any]] = {}
    metadata: list[dict[str, Any]] = []
    model_file_names: list[str] = []

    for output_dir in MODEL_OUTPUT_DIRS:
        model_files = sorted(output_dir.glob("*.jsonl"))
        if not model_files:
            raise SystemExit(f"No JSONL files found in {output_dir}")

        for model_file in model_files:
            model_name = evaluate_model_outputs.model_name_from_path(model_file)
            predictions, file_metadata = evaluate_model_outputs.load_model_predictions(model_file)
            predictions_by_model[model_name] = predictions
            file_metadata["source_output_dir"] = str(output_dir)
            metadata.append(file_metadata)
            model_file_names.append(str(model_file.relative_to(ROOT_DIR)))

    return predictions_by_model, metadata, model_file_names


def main() -> None:
    evaluate_model_outputs.ACCURACY_RELATIVE_TOLERANCE = 0.20
    sys.argv[1:1] = DEFAULT_ARGS
    args = evaluate_model_outputs.parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    truths = evaluate_model_outputs.load_truth(args.recipes)
    predictions_by_model, metadata, model_file_names = load_predictions_from_dirs()

    overall_rows, recipe_type_rows, detail_rows = evaluate_model_outputs.summarize_predictions(
        predictions_by_model,
        truths,
    )
    relabel_detail_rows(detail_rows)
    evaluate_model_outputs.write_csv(args.results_dir / "model_ranking.csv", overall_rows)
    evaluate_model_outputs.write_csv(args.results_dir / "recipe_type_metrics.csv", recipe_type_rows)
    evaluate_model_outputs.write_csv(args.results_dir / "per_recipe_predictions.csv", detail_rows)
    evaluate_model_outputs.write_json(
        args.results_dir / "evaluation_metadata.json",
        {
            "accuracy_relative_tolerance": evaluate_model_outputs.ACCURACY_RELATIVE_TOLERANCE,
            "source_recipe_count": len(truths),
            "recipe_count": len(truths),
            "excluded_recipe_count": 0,
            "excluded_recipes": [],
            "model_file_count": len(model_file_names),
            "model_files": model_file_names,
            "model_output_dirs": [str(path) for path in MODEL_OUTPUT_DIRS],
            "model_output_metadata": metadata,
        },
    )

    if not args.skip_plots:
        write_20_percent_plots(overall_rows, recipe_type_rows, args.results_dir)

    best = overall_rows[0]
    print(
        "Best model: "
        f"{best['model_name']} "
        f"(MAE={best['MAE']:.6f}, accuracy={best['accuracy_percent']:.2f}%, "
        f"valid={best['valid_predictions_count']}/{best['total_recipes']})"
    )
    print(f"Wrote 20% evaluation outputs to {args.results_dir}")


if __name__ == "__main__":
    main()
