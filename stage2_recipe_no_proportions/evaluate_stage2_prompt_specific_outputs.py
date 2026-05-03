#!/usr/bin/env python3
"""Evaluate Stage 2 outputs with few-shot example recipes excluded only for few-shot runs."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


STAGE2_DIR = Path(__file__).resolve().parent
ROOT_DIR = STAGE2_DIR.parent
DEFAULT_FEW_SHOT_EXCLUSIONS = {
    "Trail Mix Snack",
    "Chicken Sausage Garden Pasta",
    "Fish Tacos with Crunchy Slaw",
}
DEFAULT_ARGS = [
    "--recipes",
    str(STAGE2_DIR / "recipes_no_proportions.json"),
    "--model-outputs-dir",
    str(STAGE2_DIR / "model_outputs"),
    "--results-dir",
    str(STAGE2_DIR / "evaluation_results"),
]


sys.path.insert(0, str(ROOT_DIR / "stage1_recipe_proportions"))
import evaluate_model_outputs  # noqa: E402


def truths_for_model(model_name: str, all_truths: list[Any]) -> tuple[list[Any], list[Any]]:
    if not model_name.startswith("few_shot__"):
        return all_truths, []

    included = [
        truth
        for truth in all_truths
        if truth.recipe_name not in DEFAULT_FEW_SHOT_EXCLUSIONS
    ]
    excluded = [
        truth
        for truth in all_truths
        if truth.recipe_name in DEFAULT_FEW_SHOT_EXCLUSIONS
    ]
    return included, excluded


def main() -> None:
    sys.argv[1:1] = DEFAULT_ARGS
    args = evaluate_model_outputs.parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    all_truths = evaluate_model_outputs.load_truth(args.recipes)
    model_files = sorted(args.model_outputs_dir.glob("*.jsonl"))
    if not model_files:
        raise SystemExit(f"No JSONL files found in {args.model_outputs_dir}")

    overall_rows: list[dict[str, Any]] = []
    recipe_type_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    exclusions_by_model: dict[str, list[dict[str, Any]]] = {}

    for model_file in model_files:
        model_name = evaluate_model_outputs.model_name_from_path(model_file)
        predictions, file_metadata = evaluate_model_outputs.load_model_predictions(model_file)
        truths, excluded_truths = truths_for_model(model_name, all_truths)
        model_overall, model_recipe_types, model_details = (
            evaluate_model_outputs.summarize_predictions(
                {model_name: predictions},
                truths,
            )
        )
        overall_rows.extend(model_overall)
        recipe_type_rows.extend(model_recipe_types)
        detail_rows.extend(model_details)
        file_metadata["scored_recipe_count"] = len(truths)
        file_metadata["excluded_recipe_count"] = len(excluded_truths)
        metadata.append(file_metadata)
        exclusions_by_model[model_name] = [
            {
                "recipe_index": truth.recipe_index,
                "recipe_type": truth.recipe_type,
                "recipe_name": truth.recipe_name,
                "ground_truth_copper_mg_per_serving": truth.copper_per_serving_mg,
            }
            for truth in excluded_truths
        ]

    overall_rows.sort(
        key=lambda row: (
            float("inf") if row["MAE"] == "" else row["MAE"],
            float("-inf") if row["accuracy_percent"] == "" else -row["accuracy_percent"],
            row["model_name"],
        )
    )
    evaluate_model_outputs.write_csv(args.results_dir / "model_ranking.csv", overall_rows)
    evaluate_model_outputs.write_csv(args.results_dir / "recipe_type_metrics.csv", recipe_type_rows)
    evaluate_model_outputs.write_csv(args.results_dir / "per_recipe_predictions.csv", detail_rows)
    evaluate_model_outputs.write_json(
        args.results_dir / "evaluation_metadata.json",
        {
            "accuracy_relative_tolerance": evaluate_model_outputs.ACCURACY_RELATIVE_TOLERANCE,
            "source_recipe_count": len(all_truths),
            "model_file_count": len(model_files),
            "model_files": [path.name for path in model_files],
            "model_output_metadata": metadata,
            "exclusion_policy": (
                "Few-shot prompt example recipes are excluded only from few-shot "
                "model scoring. Baseline and persona outputs are scored on all recipes."
            ),
            "few_shot_excluded_recipe_names": sorted(DEFAULT_FEW_SHOT_EXCLUSIONS),
            "excluded_recipes_by_model": exclusions_by_model,
        },
    )

    if not args.skip_plots:
        evaluate_model_outputs.write_plots(overall_rows, recipe_type_rows, args.results_dir)

    best = overall_rows[0]
    print(
        "Best model: "
        f"{best['model_name']} "
        f"(MAE={best['MAE']:.6f}, accuracy={best['accuracy_percent']:.2f}%, "
        f"valid={best['valid_predictions_count']}/{best['total_recipes']})"
    )
    print(f"Wrote evaluation outputs to {args.results_dir}")


if __name__ == "__main__":
    main()
