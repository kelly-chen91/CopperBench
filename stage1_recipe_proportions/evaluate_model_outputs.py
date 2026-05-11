#!/usr/bin/env python3
"""Evaluate copper-per-serving predictions against recipes.json ground truth."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_RECIPES_PATH = Path("recipes.json")
DEFAULT_MODEL_OUTPUTS_DIR = Path("model_outputs")
DEFAULT_RESULTS_DIR = Path("stage1_recipe_proportions/evaluation_results")
ACCURACY_RELATIVE_TOLERANCE = 0.10


@dataclass(frozen=True)
class RecipeTruth:
    recipe_index: int
    recipe_type: str
    recipe_name: str
    copper_per_serving_mg: float


@dataclass(frozen=True)
class Prediction:
    model_name: str
    model_file: str
    reasoning_level: str
    recipe_index: int | None
    recipe_type: str | None
    recipe_name: str | None
    predicted_copper_mg_per_serving: float | None
    valid: bool
    failure_reason: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank model copper predictions against recipes.json ground truth."
    )
    parser.add_argument(
        "--recipes",
        type=Path,
        default=DEFAULT_RECIPES_PATH,
        help=f"Ground-truth recipes JSON. Default: {DEFAULT_RECIPES_PATH}",
    )
    parser.add_argument(
        "--model-outputs-dir",
        type=Path,
        default=DEFAULT_MODEL_OUTPUTS_DIR,
        help=f"Directory containing model JSONL outputs. Default: {DEFAULT_MODEL_OUTPUTS_DIR}",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=f"Directory for CSV and visualization outputs. Default: {DEFAULT_RESULTS_DIR}",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Write CSV/JSON outputs but skip Matplotlib visualizations.",
    )
    parser.add_argument(
        "--exclude-recipe-name",
        action="append",
        default=[],
        help=(
            "Recipe name to exclude from scoring. May be passed multiple times. "
            "Names must match recipes.json exactly."
        ),
    )
    return parser.parse_args()


def iter_recipe_dicts(node: Any):
    if isinstance(node, dict):
        yield node
        return

    if isinstance(node, list):
        for item in node:
            yield from iter_recipe_dicts(item)


def load_truth(path: Path) -> list[RecipeTruth]:
    data = json.loads(path.read_text())
    truths: list[RecipeTruth] = []
    recipe_index = 0

    for group in data:
        recipe_type = group.get("recipe_type")
        for recipe in iter_recipe_dicts(group.get("recipes", [])):
            if not isinstance(recipe, dict):
                continue
            missing = [
                key
                for key in ("name", "copper_per_serving_mg")
                if key not in recipe
            ]
            if missing:
                raise ValueError(
                    f"Recipe at index {recipe_index} is missing required keys: {missing}"
                )
            truths.append(
                RecipeTruth(
                    recipe_index=recipe_index,
                    recipe_type=str(recipe_type),
                    recipe_name=str(recipe["name"]),
                    copper_per_serving_mg=float(recipe["copper_per_serving_mg"]),
                )
            )
            recipe_index += 1

    if not truths:
        raise ValueError(f"No recipes found in {path}")
    return truths


def filter_truths_by_name(
    truths: list[RecipeTruth],
    excluded_recipe_names: list[str],
) -> tuple[list[RecipeTruth], list[RecipeTruth]]:
    excluded_names = set(excluded_recipe_names)
    if not excluded_names:
        return truths, []

    matching_exclusions = [
        truth for truth in truths if truth.recipe_name in excluded_names
    ]
    found_names = {truth.recipe_name for truth in matching_exclusions}
    missing_names = sorted(excluded_names - found_names)
    if missing_names:
        raise ValueError(
            "Excluded recipe names were not found in recipes.json: "
            + ", ".join(missing_names)
        )

    filtered_truths = [
        truth for truth in truths if truth.recipe_name not in excluded_names
    ]
    return filtered_truths, matching_exclusions


def model_name_from_path(path: Path) -> str:
    return path.stem


def reasoning_level_from_config(config: dict[str, Any] | None, model_file: Path) -> str:
    if config and config.get("reasoning_effort"):
        return str(config["reasoning_effort"])
    match = re.search(r"__reasoning_([a-z]+)$", model_file.stem)
    if match:
        return match.group(1)
    return "none"


def extract_prediction_value(record: dict[str, Any]) -> tuple[float | None, str | None]:
    parsed_output = record.get("parsed_output")
    if not record.get("success", False):
        return None, "record_success_false"
    if not isinstance(parsed_output, dict):
        return None, "missing_parsed_output"

    value = parsed_output.get("copper_mg_per_serving")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None, "missing_numeric_copper_mg_per_serving"
    value = float(value)
    if not math.isfinite(value):
        return None, "non_finite_copper_mg_per_serving"
    return value, None


def load_model_predictions(path: Path) -> tuple[list[Prediction], dict[str, Any]]:
    model_name = model_name_from_path(path)
    config: dict[str, Any] | None = None
    latest_by_recipe: dict[int, Prediction] = {}
    unindexed_predictions: list[Prediction] = []
    json_errors: list[dict[str, Any]] = []
    trial_record_count = 0

    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            json_errors.append({"line": line_number, "error": str(exc)})
            continue

        if record.get("record_type") == "experiment_config":
            config = record.get("config") if isinstance(record.get("config"), dict) else config
            continue
        if record.get("record_type") != "trial_result":
            continue

        trial_record_count += 1
        reasoning_level = reasoning_level_from_config(config, path)
        predicted_value, failure_reason = extract_prediction_value(record)
        recipe_index = record.get("recipe_index")
        if isinstance(recipe_index, bool) or not isinstance(recipe_index, int):
            recipe_index = None

        prediction = Prediction(
            model_name=model_name,
            model_file=path.name,
            reasoning_level=reasoning_level,
            recipe_index=recipe_index,
            recipe_type=record.get("recipe_type"),
            recipe_name=record.get("recipe_name"),
            predicted_copper_mg_per_serving=predicted_value,
            valid=failure_reason is None,
            failure_reason=failure_reason,
        )

        if recipe_index is None:
            unindexed_predictions.append(prediction)
        else:
            latest_by_recipe[recipe_index] = prediction

    metadata = {
        "model_file": path.name,
        "model_name": model_name,
        "reasoning_level": reasoning_level_from_config(config, path),
        "trial_record_count": trial_record_count,
        "latest_prediction_count": len(latest_by_recipe),
        "unindexed_prediction_count": len(unindexed_predictions),
        "json_error_count": len(json_errors),
        "json_errors": json_errors,
    }
    return [latest_by_recipe[index] for index in sorted(latest_by_recipe)], metadata


def absolute_error(prediction: float, truth: float) -> float:
    return abs(prediction - truth)


def is_within_tolerance(prediction: float, truth: float) -> bool:
    if truth == 0:
        return prediction == 0
    return absolute_error(prediction, truth) <= abs(truth) * ACCURACY_RELATIVE_TOLERANCE


def summarize_predictions(
    predictions_by_model: dict[str, list[Prediction]],
    truths: list[RecipeTruth],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    truth_by_index = {truth.recipe_index: truth for truth in truths}
    overall_rows: list[dict[str, Any]] = []
    recipe_type_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []

    for model_name, predictions in sorted(predictions_by_model.items()):
        details_for_model: list[dict[str, Any]] = []
        predictions_by_index = {
            prediction.recipe_index: prediction
            for prediction in predictions
            if prediction.recipe_index is not None
        }
        reasoning_level = predictions[0].reasoning_level if predictions else "unknown"
        model_file = predictions[0].model_file if predictions else f"{model_name}.jsonl"

        for truth in truths:
            prediction = predictions_by_index.get(truth.recipe_index)
            if prediction is None:
                row = {
                    "model_name": model_name,
                    "model_file": model_file,
                    "reasoning_level": reasoning_level,
                    "recipe_index": truth.recipe_index,
                    "recipe_type": truth.recipe_type,
                    "recipe_name": truth.recipe_name,
                    "ground_truth_copper_mg_per_serving": truth.copper_per_serving_mg,
                    "predicted_copper_mg_per_serving": "",
                    "absolute_error": "",
                    "relative_error_percent": "",
                    "within_10_percent": False,
                    "valid": False,
                    "failure_reason": "missing_prediction",
                }
            elif not prediction.valid or prediction.predicted_copper_mg_per_serving is None:
                row = {
                    "model_name": model_name,
                    "model_file": prediction.model_file,
                    "reasoning_level": prediction.reasoning_level,
                    "recipe_index": truth.recipe_index,
                    "recipe_type": truth.recipe_type,
                    "recipe_name": truth.recipe_name,
                    "ground_truth_copper_mg_per_serving": truth.copper_per_serving_mg,
                    "predicted_copper_mg_per_serving": "",
                    "absolute_error": "",
                    "relative_error_percent": "",
                    "within_10_percent": False,
                    "valid": False,
                    "failure_reason": prediction.failure_reason,
                }
            else:
                error = absolute_error(
                    prediction.predicted_copper_mg_per_serving,
                    truth.copper_per_serving_mg,
                )
                row = {
                    "model_name": model_name,
                    "model_file": prediction.model_file,
                    "reasoning_level": prediction.reasoning_level,
                    "recipe_index": truth.recipe_index,
                    "recipe_type": truth.recipe_type,
                    "recipe_name": truth.recipe_name,
                    "ground_truth_copper_mg_per_serving": truth.copper_per_serving_mg,
                    "predicted_copper_mg_per_serving": prediction.predicted_copper_mg_per_serving,
                    "absolute_error": error,
                    "relative_error_percent": (error / truth.copper_per_serving_mg * 100)
                    if truth.copper_per_serving_mg
                    else "",
                    "within_10_percent": is_within_tolerance(
                        prediction.predicted_copper_mg_per_serving,
                        truth.copper_per_serving_mg,
                    ),
                    "valid": True,
                    "failure_reason": "",
                }
            details_for_model.append(row)
            detail_rows.append(row)

        valid_rows = [row for row in details_for_model if row["valid"]]
        invalid_rows = [row for row in details_for_model if not row["valid"]]
        mae = statistics.fmean(row["absolute_error"] for row in valid_rows) if valid_rows else ""
        accuracy = (
            sum(1 for row in valid_rows if row["within_10_percent"]) / len(valid_rows) * 100
            if valid_rows
            else ""
        )
        overall_rows.append(
            {
                "model_name": model_name,
                "model_file": model_file,
                "valid_predictions_count": len(valid_rows),
                "invalid_predictions_count": len(invalid_rows),
                "total_recipes": len(truths),
                "MAE": mae,
                "accuracy_percent": accuracy,
                "recipe_type": "all",
                "reasoning_level": reasoning_level,
                "failed_recipes": "; ".join(row["recipe_name"] for row in invalid_rows),
            }
        )

        for recipe_type in sorted({truth.recipe_type for truth in truths}):
            rows_for_type = [
                row
                for row in details_for_model
                if row["recipe_type"] == recipe_type and row["valid"]
            ]
            invalid_for_type = [
                row
                for row in details_for_model
                if row["recipe_type"] == recipe_type and not row["valid"]
            ]
            recipe_type_rows.append(
                {
                    "model_name": model_name,
                    "model_file": model_file,
                    "valid_predictions_count": len(rows_for_type),
                    "invalid_predictions_count": len(invalid_for_type),
                    "total_recipes": len(rows_for_type) + len(invalid_for_type),
                    "MAE": statistics.fmean(row["absolute_error"] for row in rows_for_type)
                    if rows_for_type
                    else "",
                    "accuracy_percent": (
                        sum(1 for row in rows_for_type if row["within_10_percent"])
                        / len(rows_for_type)
                        * 100
                    )
                    if rows_for_type
                    else "",
                    "recipe_type": recipe_type,
                    "reasoning_level": reasoning_level,
                    "failed_recipes": "; ".join(row["recipe_name"] for row in invalid_for_type),
                }
            )

    overall_rows.sort(
        key=lambda row: (
            float("inf") if row["MAE"] == "" else row["MAE"],
            float("-inf") if row["accuracy_percent"] == "" else -row["accuracy_percent"],
            row["model_name"],
        )
    )
    return overall_rows, recipe_type_rows, detail_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def short_model_label(model_name: str) -> str:
    return (
        model_name.replace("gpt_", "gpt-")
        .replace("_", "-")
        .replace("--reasoning-", "\n")
        .replace("-reasoning-", "\n")
    )


def require_matplotlib():
    cache_dir = Path(__file__).resolve().parent / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir / "xdg"))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "Matplotlib is required for visualizations. Install dependencies with "
            "`python3 -m pip install -r stage1_recipe_proportions/requirements.txt`, "
            "or rerun with --skip-plots."
        ) from exc
    return plt


def plot_overall_bar(
    rows: list[dict[str, Any]],
    value_key: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    plt = require_matplotlib()
    valid_rows = [row for row in rows if row[value_key] != ""]
    labels = [short_model_label(row["model_name"]) for row in valid_rows]
    values = [row[value_key] for row in valid_rows]

    fig_width = max(12, len(valid_rows) * 0.55)
    fig, ax = plt.subplots(figsize=(fig_width, 7))
    bars = ax.bar(labels, values, color="#4C78A8", edgecolor="none")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", labelrotation=65, labelsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.45)
    y_max = max(values) * 1.12 if values else 1.0
    ax.set_ylim(0, y_max)
    value_format = "{:.1f}%" if value_key == "accuracy_percent" else "{:.3f}"
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            max(value - y_max * 0.04, value * 0.5),
            value_format.format(value),
            ha="center",
            va="center",
            color="white",
            fontsize=8,
            fontweight="bold",
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_recipe_type_heatmap(rows: list[dict[str, Any]], output_path: Path) -> None:
    plt = require_matplotlib()
    model_names = sorted({row["model_name"] for row in rows})
    recipe_types = sorted({row["recipe_type"] for row in rows})
    value_by_key = {
        (row["model_name"], row["recipe_type"]): row["MAE"]
        for row in rows
        if row["MAE"] != ""
    }
    matrix = [
        [value_by_key.get((model_name, recipe_type), float("nan")) for model_name in model_names]
        for recipe_type in recipe_types
    ]

    fig_width = max(12, len(model_names) * 0.55)
    fig, ax = plt.subplots(figsize=(fig_width, 5))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_title("MAE by Recipe Type and Model")
    ax.set_xticks(range(len(model_names)), [short_model_label(name) for name in model_names])
    ax.set_yticks(range(len(recipe_types)), recipe_types)
    ax.tick_params(axis="x", labelrotation=65, labelsize=8)
    fig.colorbar(image, ax=ax, label="MAE")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_reasoning_effect(rows: list[dict[str, Any]], output_path: Path) -> None:
    plt = require_matplotlib()
    effort_order = {"low": 0, "medium": 1, "high": 2}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["reasoning_level"] in effort_order and row["MAE"] != "":
            base_model = re.sub(r"__reasoning_[a-z]+$", "", row["model_name"])
            if "gpt_5" in base_model:
                grouped[base_model].append(row)

    efforts = ["low", "medium", "high"]
    model_names = sorted(grouped)
    x_positions = list(range(len(efforts)))
    bar_width = min(0.16, 0.82 / max(len(model_names), 1))
    center_offset = (len(model_names) - 1) * bar_width / 2
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    values_by_key = {
        (base_model, row["reasoning_level"]): float(row["MAE"])
        for base_model, model_rows in grouped.items()
        for row in model_rows
    }
    y_max = max(values_by_key.values()) * 1.12 if values_by_key else 1.0

    fig, ax = plt.subplots(figsize=(10.5, 6))
    for model_index, base_model in enumerate(model_names):
        bar_x = [
            position - center_offset + model_index * bar_width
            for position in x_positions
        ]
        bar_values = [
            values_by_key.get((base_model, effort), 0.0)
            for effort in efforts
        ]
        bars = ax.bar(
            bar_x,
            bar_values,
            width=bar_width,
            color=colors[model_index % len(colors)],
            edgecolor="none",
            label=short_model_label(base_model),
        )
        for bar, value in zip(bars, bar_values):
            if value <= 0:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                max(value - y_max * 0.035, value * 0.5),
                f"{value:.3f}",
                ha="center",
                va="center",
                color="white",
                fontsize=8,
                fontweight="bold",
            )
    ax.set_title("Stage 1 GPT-5 Reasoning Effort Effect")
    ax.set_xlabel("Reasoning effort")
    ax.set_ylabel("MAE")
    ax.set_xticks(x_positions, efforts)
    ax.set_ylim(0, y_max)
    ax.grid(axis="y", linestyle=":", alpha=0.45)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_plots(
    overall_rows: list[dict[str, Any]],
    recipe_type_rows: list[dict[str, Any]],
    results_dir: Path,
) -> None:
    plot_overall_bar(
        overall_rows,
        "MAE",
        "MAE Comparison Across Models",
        "Mean absolute error (mg copper per serving)",
        results_dir / "mae_comparison.png",
    )
    plot_overall_bar(
        overall_rows,
        "accuracy_percent",
        "Stage 1 Accuracy Within +/-10% Across Models",
        "Accuracy (%)",
        results_dir / "accuracy_comparison.png",
    )
    plot_recipe_type_heatmap(recipe_type_rows, results_dir / "mae_by_recipe_type_heatmap.png")
    plot_reasoning_effect(overall_rows, results_dir / "reasoning_effort_effect.png")


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    all_truths = load_truth(args.recipes)
    truths, excluded_truths = filter_truths_by_name(
        all_truths,
        args.exclude_recipe_name,
    )
    model_files = sorted(args.model_outputs_dir.glob("*.jsonl"))
    if not model_files:
        raise SystemExit(f"No JSONL files found in {args.model_outputs_dir}")

    predictions_by_model: dict[str, list[Prediction]] = {}
    metadata: list[dict[str, Any]] = []
    for model_file in model_files:
        predictions, file_metadata = load_model_predictions(model_file)
        predictions_by_model[model_name_from_path(model_file)] = predictions
        metadata.append(file_metadata)

    overall_rows, recipe_type_rows, detail_rows = summarize_predictions(
        predictions_by_model,
        truths,
    )
    write_csv(args.results_dir / "model_ranking.csv", overall_rows)
    write_csv(args.results_dir / "recipe_type_metrics.csv", recipe_type_rows)
    write_csv(args.results_dir / "per_recipe_predictions.csv", detail_rows)
    write_json(
        args.results_dir / "evaluation_metadata.json",
        {
            "accuracy_relative_tolerance": ACCURACY_RELATIVE_TOLERANCE,
            "source_recipe_count": len(all_truths),
            "recipe_count": len(truths),
            "excluded_recipe_count": len(excluded_truths),
            "excluded_recipes": [
                {
                    "recipe_index": truth.recipe_index,
                    "recipe_type": truth.recipe_type,
                    "recipe_name": truth.recipe_name,
                    "ground_truth_copper_mg_per_serving": truth.copper_per_serving_mg,
                }
                for truth in excluded_truths
            ],
            "model_file_count": len(model_files),
            "model_files": [path.name for path in model_files],
            "model_output_metadata": metadata,
        },
    )

    if not args.skip_plots:
        write_plots(overall_rows, recipe_type_rows, args.results_dir)

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
