#!/usr/bin/env python3
import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_RECIPES_PATH = "recipes.json"
DEFAULT_OUTPUT_DIR = "model_outputs"
DEFAULT_SUMMARY_PATH = "copper_recipe_type_rankings.csv"
DEFAULT_DETAILS_PATH = "copper_recipe_type_details.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank model outputs by error within each recipe type."
    )
    parser.add_argument(
        "--recipes",
        default=DEFAULT_RECIPES_PATH,
        help=f"Path to recipes.json. Default: {DEFAULT_RECIPES_PATH}",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory containing model JSONL outputs. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--summary-csv",
        default=DEFAULT_SUMMARY_PATH,
        help=f"Where to write recipe-type ranking summary CSV. Default: {DEFAULT_SUMMARY_PATH}",
    )
    parser.add_argument(
        "--details-csv",
        default=DEFAULT_DETAILS_PATH,
        help=f"Where to write recipe-type detail CSV. Default: {DEFAULT_DETAILS_PATH}",
    )
    parser.add_argument(
        "--include-pro",
        action="store_true",
        help="Include *pro* model files in the analysis.",
    )
    return parser.parse_args()


def iter_recipe_dicts(node: Any):
    if isinstance(node, dict):
        yield node
        return
    if isinstance(node, list):
        for item in node:
            yield from iter_recipe_dicts(item)


def load_reference_recipes(path: Path) -> dict[int, dict[str, Any]]:
    data = json.loads(path.read_text())
    recipes: dict[int, dict[str, Any]] = {}
    recipe_index = 0
    for group in data:
        recipe_type = group.get("recipe_type")
        for recipe in iter_recipe_dicts(group.get("recipes", [])):
            if not isinstance(recipe, dict):
                continue
            recipes[recipe_index] = {
                "recipe_index": recipe_index,
                "recipe_type": recipe_type,
                "recipe_name": recipe.get("name"),
                "actual_copper_mg_per_serving": recipe.get("copper_per_serving_mg"),
            }
            recipe_index += 1
    return recipes


def maybe_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def is_trial_result(record: dict[str, Any]) -> bool:
    return record.get("record_type") == "trial_result"


def is_pro_file(path: Path) -> bool:
    return "_pro" in path.stem


def choose_latest_records(path: Path) -> dict[int, dict[str, Any]]:
    latest: dict[int, dict[str, Any]] = {}
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not is_trial_result(record):
                continue
            recipe_index = record.get("recipe_index")
            if not isinstance(recipe_index, int):
                continue
            latest[recipe_index] = record
    return latest


def extract_json_object_from_text(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if not stripped:
        raise ValueError("Empty text.")

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL)
    if fenced:
        stripped = fenced.group(1)

    decoder = json.JSONDecoder()
    try:
        obj, _ = decoder.raw_decode(stripped)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    for match in re.finditer(r"\{", stripped):
        try:
            obj, _ = decoder.raw_decode(stripped[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj

    raise ValueError("Could not find JSON object.")


def extract_prediction(record: dict[str, Any]) -> tuple[float | None, str | None]:
    parsed = record.get("parsed_output")
    if isinstance(parsed, dict):
        value = maybe_float(parsed.get("copper_mg_per_serving"))
        if value is not None:
            return value, "parsed_output.copper_mg_per_serving"

    raw_output = record.get("raw_output")
    if not isinstance(raw_output, str) or not raw_output.strip():
        return None, None

    try:
        parsed = extract_json_object_from_text(raw_output)
    except ValueError:
        parsed = None
    if isinstance(parsed, dict):
        value = maybe_float(parsed.get("copper_mg_per_serving"))
        if value is not None:
            return value, "raw_output.json.copper_mg_per_serving"

    patterns = [
        r"copper_mg_per_serving\"?\s*[:=]\s*([0-9]*\.?[0-9]+)",
        r"per serving[^0-9]{0,40}([0-9]*\.?[0-9]+)\s*mg",
        r"([0-9]*\.?[0-9]+)\s*mg\s*(?:copper\s*)?per serving",
    ]
    lower_text = raw_output.lower()
    for pattern in patterns:
        match = re.search(pattern, lower_text, re.IGNORECASE)
        if not match:
            continue
        try:
            return float(match.group(1)), f"raw_output.regex:{pattern}"
        except ValueError:
            continue

    return None, None


def round_or_none(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(value, digits)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    recipes = load_reference_recipes(Path(args.recipes))
    output_dir = Path(args.output_dir)
    jsonl_files = sorted(output_dir.glob("*.jsonl"))
    if not args.include_pro:
        jsonl_files = [path for path in jsonl_files if not is_pro_file(path)]

    detail_rows: list[dict[str, Any]] = []
    grouped_errors: dict[tuple[str, str], list[float]] = defaultdict(list)
    grouped_signed_errors: dict[tuple[str, str], list[float]] = defaultdict(list)
    grouped_coverage: dict[tuple[str, str], int] = defaultdict(int)
    grouped_recipe_totals: dict[str, int] = defaultdict(int)
    grouped_model_meta: dict[tuple[str, str], dict[str, Any]] = {}

    for recipe in recipes.values():
        grouped_recipe_totals[recipe["recipe_type"]] += 1

    for path in jsonl_files:
        latest_records = choose_latest_records(path)
        model_name = next(
            (record.get("model") for record in latest_records.values() if isinstance(record, dict)),
            None,
        )
        reasoning_effort = next(
            (
                record.get("reasoning_effort")
                for record in latest_records.values()
                if isinstance(record, dict)
            ),
            None,
        )
        model_variant_label = path.stem

        for recipe_index, recipe in recipes.items():
            record = latest_records.get(recipe_index)
            actual = maybe_float(recipe.get("actual_copper_mg_per_serving"))
            predicted = None
            source = None
            if record is not None:
                predicted, source = extract_prediction(record)

            error = None
            abs_error = None
            if actual is not None and predicted is not None:
                error = predicted - actual
                abs_error = abs(error)
                key = (recipe["recipe_type"], model_variant_label)
                grouped_errors[key].append(abs_error)
                grouped_signed_errors[key].append(error)
                grouped_coverage[key] += 1
                grouped_model_meta[key] = {
                    "file_name": path.name,
                    "model": model_name,
                    "reasoning_effort": reasoning_effort,
                    "recipe_type": recipe["recipe_type"],
                    "model_variant_label": model_variant_label,
                }

            detail_rows.append(
                {
                    "recipe_type": recipe["recipe_type"],
                    "file_name": path.name,
                    "model": model_name,
                    "reasoning_effort": reasoning_effort,
                    "model_variant_label": model_variant_label,
                    "recipe_index": recipe_index,
                    "recipe_name": recipe["recipe_name"],
                    "actual_copper_mg_per_serving": actual,
                    "predicted_copper_mg_per_serving": predicted,
                    "error_mg": round_or_none(error),
                    "absolute_error_mg": round_or_none(abs_error),
                    "prediction_source": source,
                }
            )

    summary_rows: list[dict[str, Any]] = []
    for (recipe_type, model_variant_label), abs_errors in grouped_errors.items():
        signed_errors = grouped_signed_errors[(recipe_type, model_variant_label)]
        meta = grouped_model_meta[(recipe_type, model_variant_label)]
        total_in_type = grouped_recipe_totals[recipe_type]
        coverage = grouped_coverage[(recipe_type, model_variant_label)]
        rmse = math.sqrt(statistics.fmean([x * x for x in signed_errors])) if signed_errors else None

        summary_rows.append(
            {
                "recipe_type": recipe_type,
                "file_name": meta["file_name"],
                "model": meta["model"],
                "reasoning_effort": meta["reasoning_effort"],
                "model_variant_label": model_variant_label,
                "recipes_in_type": total_in_type,
                "matched_predictions": coverage,
                "coverage_rate": round_or_none(coverage / total_in_type if total_in_type else None),
                "mean_signed_error_mg": round_or_none(statistics.fmean(signed_errors) if signed_errors else None),
                "mae_mg": round_or_none(statistics.fmean(abs_errors) if abs_errors else None),
                "rmse_mg": round_or_none(rmse),
            }
        )

    rankings_by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        rankings_by_type[row["recipe_type"]].append(row)

    ranked_rows: list[dict[str, Any]] = []
    for recipe_type, rows in rankings_by_type.items():
        rows.sort(
            key=lambda row: (
                row["mae_mg"] is None,
                -(row["coverage_rate"] or 0),
                row["mae_mg"] if row["mae_mg"] is not None else float("inf"),
            )
        )
        for rank, row in enumerate(rows, start=1):
            ranked = dict(row)
            ranked["rank_within_recipe_type"] = rank
            ranked_rows.append(ranked)

    write_csv(
        Path(args.summary_csv),
        ranked_rows,
        [
            "recipe_type",
            "rank_within_recipe_type",
            "file_name",
            "model",
            "reasoning_effort",
            "model_variant_label",
            "recipes_in_type",
            "matched_predictions",
            "coverage_rate",
            "mean_signed_error_mg",
            "mae_mg",
            "rmse_mg",
        ],
    )
    write_csv(
        Path(args.details_csv),
        detail_rows,
        [
            "recipe_type",
            "file_name",
            "model",
            "reasoning_effort",
            "model_variant_label",
            "recipe_index",
            "recipe_name",
            "actual_copper_mg_per_serving",
            "predicted_copper_mg_per_serving",
            "error_mg",
            "absolute_error_mg",
            "prediction_source",
        ],
    )

    print(f"Wrote summary: {args.summary_csv}")
    print(f"Wrote details: {args.details_csv}")
    print()
    for recipe_type, rows in sorted(rankings_by_type.items()):
        best = min(
            rows,
            key=lambda row: (
                row["mae_mg"] is None,
                row["mae_mg"] if row["mae_mg"] is not None else float("inf"),
            ),
        )
        print(
            f"{recipe_type}: best={best['file_name']} "
            f"(coverage={best['matched_predictions']}/{best['recipes_in_type']}, "
            f"MAE={best['mae_mg']}, RMSE={best['rmse_mg']})"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
