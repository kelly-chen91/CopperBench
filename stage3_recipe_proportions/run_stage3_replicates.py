#!/usr/bin/env python3
"""Run Stage 3 prompt variants repeatedly and aggregate 20% metrics."""

from __future__ import annotations

import csv
import json
import os
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STAGE3_DIR = Path(__file__).resolve().parent
ROOT_DIR = STAGE3_DIR.parent
RUNS_DIR = STAGE3_DIR / "replicate_runs"
RECIPE_PATH = ROOT_DIR / "recipes.json"
RUN_COUNT = 3

RUNNERS = [
    {
        "stage": "stage3",
        "prompt_variant": "Persona",
        "script": STAGE3_DIR / "run_stage3_persona.py",
        "model_name": "persona__gpt_5_4_mini__reasoning_medium",
    },
    {
        "stage": "stage3",
        "prompt_variant": "Combined persona + few-shot",
        "script": STAGE3_DIR / "run_stage3_combined_persona_few_shot.py",
        "model_name": "combined_persona_few_shot__gpt_5_4_mini__reasoning_medium",
    },
]
RUNNER_BY_MODEL = {runner["model_name"]: runner for runner in RUNNERS}


sys.path.insert(0, str(ROOT_DIR / "stage1_recipe_proportions"))
import evaluate_model_outputs  # noqa: E402


def load_dotenv_if_needed() -> None:
    if os.environ.get("OPENAI_API_KEY"):
        return

    env_path = ROOT_DIR / ".env"
    if not env_path.exists():
        return

    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def relabel_and_add_error_rows(detail_rows: list[dict[str, Any]]) -> None:
    for row in detail_rows:
        row["within_20_percent"] = row.pop("within_10_percent")
        if row["valid"]:
            prediction = float(row["predicted_copper_mg_per_serving"])
            truth = float(row["ground_truth_copper_mg_per_serving"])
            signed_error = prediction - truth
            row["signed_error"] = signed_error
            row["squared_error"] = signed_error**2
        else:
            row["signed_error"] = ""
            row["squared_error"] = ""


def write_20_percent_plots(
    overall_rows: list[dict[str, Any]],
    recipe_type_rows: list[dict[str, Any]],
    results_dir: Path,
) -> None:
    evaluate_model_outputs.plot_overall_bar(
        overall_rows,
        "MAE",
        "MAE Comparison Across Stage 3 Prompts",
        "Mean absolute error (mg copper per serving)",
        results_dir / "mae_comparison.png",
    )
    evaluate_model_outputs.plot_overall_bar(
        overall_rows,
        "accuracy_percent",
        "Accuracy Within +/-20% Across Stage 3 Prompts",
        "Accuracy (%)",
        results_dir / "accuracy_20_percent_comparison.png",
    )
    evaluate_model_outputs.plot_recipe_type_heatmap(
        recipe_type_rows,
        results_dir / "mae_by_recipe_type_heatmap.png",
    )


def run_prompt_variants(run_index: int, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for runner in RUNNERS:
        print(
            f"[run {run_index}/{RUN_COUNT}] running {runner['prompt_variant']} -> {output_dir}",
            flush=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(runner["script"]),
                "--output-dir",
                str(output_dir),
                "--overwrite",
            ],
            cwd=ROOT_DIR,
            check=True,
            env=os.environ.copy(),
        )


def load_predictions(output_dir: Path) -> tuple[dict[str, list[Any]], list[dict[str, Any]], list[str]]:
    predictions_by_model: dict[str, list[Any]] = {}
    metadata: list[dict[str, Any]] = []
    model_file_names: list[str] = []

    for model_file in sorted(output_dir.glob("*.jsonl")):
        model_name = evaluate_model_outputs.model_name_from_path(model_file)
        predictions, file_metadata = evaluate_model_outputs.load_model_predictions(model_file)
        predictions_by_model[model_name] = predictions
        file_metadata["source_output_dir"] = str(output_dir)
        metadata.append(file_metadata)
        model_file_names.append(str(model_file.relative_to(ROOT_DIR)))

    missing_models = sorted(set(RUNNER_BY_MODEL) - set(predictions_by_model))
    if missing_models:
        raise RuntimeError(f"Missing expected model outputs: {', '.join(missing_models)}")

    return predictions_by_model, metadata, model_file_names


def rows_for_model(detail_rows: list[dict[str, Any]], model_name: str) -> list[dict[str, Any]]:
    return [
        row
        for row in detail_rows
        if row["model_name"] == model_name and row["valid"]
    ]


def evaluate_run(
    run_index: int,
    output_dir: Path,
    results_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    results_dir.mkdir(parents=True, exist_ok=True)
    evaluate_model_outputs.ACCURACY_RELATIVE_TOLERANCE = 0.20

    truths = evaluate_model_outputs.load_truth(RECIPE_PATH)
    predictions_by_model, metadata, model_file_names = load_predictions(output_dir)
    overall_rows, recipe_type_rows, detail_rows = evaluate_model_outputs.summarize_predictions(
        predictions_by_model,
        truths,
    )
    relabel_and_add_error_rows(detail_rows)

    evaluate_model_outputs.write_csv(results_dir / "model_ranking.csv", overall_rows)
    evaluate_model_outputs.write_csv(results_dir / "recipe_type_metrics.csv", recipe_type_rows)
    evaluate_model_outputs.write_csv(results_dir / "per_recipe_predictions.csv", detail_rows)
    evaluate_model_outputs.write_json(
        results_dir / "evaluation_metadata.json",
        {
            "accuracy_relative_tolerance": 0.20,
            "source_recipe_count": len(truths),
            "recipe_count": len(truths),
            "excluded_recipe_count": 0,
            "excluded_recipes": [],
            "model_file_count": len(model_file_names),
            "model_files": model_file_names,
            "model_output_metadata": metadata,
        },
    )
    write_20_percent_plots(overall_rows, recipe_type_rows, results_dir)

    metric_rows: list[dict[str, Any]] = []
    per_recipe_metric_rows: list[dict[str, Any]] = []
    for row in overall_rows:
        model_name = str(row["model_name"])
        runner = RUNNER_BY_MODEL[model_name]
        valid_detail_rows = rows_for_model(detail_rows, model_name)
        squared_errors = [float(detail["squared_error"]) for detail in valid_detail_rows]
        absolute_errors = [float(detail["absolute_error"]) for detail in valid_detail_rows]
        metric_rows.append(
            {
                "run": run_index,
                "stage": runner["stage"],
                "prompt_variant": runner["prompt_variant"],
                "model_name": model_name,
                "MAE": row["MAE"],
                "MSE": statistics.fmean(squared_errors) if squared_errors else "",
                "SAE": sum(absolute_errors) if absolute_errors else "",
                "accuracy_20_percent": row["accuracy_percent"],
                "valid_predictions_count": row["valid_predictions_count"],
                "invalid_predictions_count": row["invalid_predictions_count"],
                "total_recipes": row["total_recipes"],
            }
        )

        for detail in valid_detail_rows:
            per_recipe_metric_rows.append(
                {
                    "run": run_index,
                    "stage": runner["stage"],
                    "prompt_variant": runner["prompt_variant"],
                    "model_name": model_name,
                    "recipe_index": detail["recipe_index"],
                    "recipe_type": detail["recipe_type"],
                    "recipe_name": detail["recipe_name"],
                    "ground_truth_copper_mg_per_serving": detail[
                        "ground_truth_copper_mg_per_serving"
                    ],
                    "predicted_copper_mg_per_serving": detail[
                        "predicted_copper_mg_per_serving"
                    ],
                    "MAE": detail["absolute_error"],
                    "SAE": detail["signed_error"],
                    "squared_error": detail["squared_error"],
                    "within_20_percent": detail["within_20_percent"],
                }
            )

    write_csv(results_dir / "run_metrics.csv", metric_rows)
    write_csv(results_dir / "per_recipe_run_metrics.csv", per_recipe_metric_rows)
    return metric_rows, per_recipe_metric_rows


def aggregate_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["model_name"]), []).append(row)

    aggregate_rows: list[dict[str, Any]] = []
    for model_name in sorted(grouped):
        model_rows = sorted(grouped[model_name], key=lambda row: int(row["run"]))
        first = model_rows[0]
        aggregate_rows.append(
            {
                "stage": first["stage"],
                "prompt_variant": first["prompt_variant"],
                "model_name": model_name,
                "run_count": len(model_rows),
                "avg_MAE": statistics.fmean(float(row["MAE"]) for row in model_rows),
                "stddev_MAE": statistics.stdev(float(row["MAE"]) for row in model_rows)
                if len(model_rows) > 1
                else 0.0,
                "avg_MSE": statistics.fmean(float(row["MSE"]) for row in model_rows),
                "stddev_MSE": statistics.stdev(float(row["MSE"]) for row in model_rows)
                if len(model_rows) > 1
                else 0.0,
                "avg_SAE": statistics.fmean(float(row["SAE"]) for row in model_rows),
                "stddev_SAE": statistics.stdev(float(row["SAE"]) for row in model_rows)
                if len(model_rows) > 1
                else 0.0,
                "avg_accuracy_20_percent": statistics.fmean(
                    float(row["accuracy_20_percent"]) for row in model_rows
                ),
                "stddev_accuracy_20_percent": statistics.stdev(
                    float(row["accuracy_20_percent"]) for row in model_rows
                )
                if len(model_rows) > 1
                else 0.0,
            }
        )

    aggregate_rows.sort(key=lambda row: (float(row["avg_MAE"]), str(row["model_name"])))
    return aggregate_rows


def main() -> None:
    load_dotenv_if_needed()
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Missing OPENAI_API_KEY in environment or .env.")

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()
    all_metric_rows: list[dict[str, Any]] = []
    all_per_recipe_metric_rows: list[dict[str, Any]] = []

    for run_index in range(1, RUN_COUNT + 1):
        run_dir = RUNS_DIR / f"run_{run_index:02d}"
        output_dir = run_dir / "model_outputs"
        results_dir = run_dir / "evaluation_results_20_percent"
        run_prompt_variants(run_index, output_dir)
        metric_rows, per_recipe_metric_rows = evaluate_run(run_index, output_dir, results_dir)
        all_metric_rows.extend(metric_rows)
        all_per_recipe_metric_rows.extend(per_recipe_metric_rows)

    write_csv(RUNS_DIR / "replicate_metrics.csv", all_metric_rows)
    write_csv(RUNS_DIR / "per_recipe_replicate_metrics.csv", all_per_recipe_metric_rows)
    aggregate_rows = aggregate_metrics(all_metric_rows)
    write_csv(RUNS_DIR / "replicate_metric_averages.csv", aggregate_rows)
    (RUNS_DIR / "replicate_metadata.json").write_text(
        json.dumps(
            {
                "started_at_utc": started_at,
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "run_count": RUN_COUNT,
                "accuracy_relative_tolerance": 0.20,
                "recipe_path": str(RECIPE_PATH.relative_to(ROOT_DIR)),
                "runs_dir": str(RUNS_DIR.relative_to(ROOT_DIR)),
                "prompt_variants": RUNNERS,
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n"
    )

    print(f"Wrote per-run metrics to {RUNS_DIR / 'replicate_metrics.csv'}")
    print(f"Wrote per-recipe metrics to {RUNS_DIR / 'per_recipe_replicate_metrics.csv'}")
    print(f"Wrote aggregate metrics to {RUNS_DIR / 'replicate_metric_averages.csv'}")


if __name__ == "__main__":
    main()
