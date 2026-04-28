#!/usr/bin/env python3
import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any


DEFAULT_RECIPES_PATH = "recipes.json"
DEFAULT_OUTPUT_DIR = "model_outputs"
DEFAULT_REPORT_PATH = "copper_analysis_summary.csv"
DEFAULT_DETAILS_PATH = "copper_analysis_details.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare model copper predictions against reference copper values in recipes.json."
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
        default=DEFAULT_REPORT_PATH,
        help=f"Where to write model-level summary CSV. Default: {DEFAULT_REPORT_PATH}",
    )
    parser.add_argument(
        "--details-csv",
        default=DEFAULT_DETAILS_PATH,
        help=f"Where to write row-level details CSV. Default: {DEFAULT_DETAILS_PATH}",
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
                "servings": recipe.get("servings"),
                "actual_copper_mg_per_serving": recipe.get("copper_per_serving_mg"),
            }
            recipe_index += 1

    return recipes


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


def maybe_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def extract_prediction_from_parsed_output(record: dict[str, Any]) -> tuple[float | None, str | None]:
    parsed = record.get("parsed_output")
    if not isinstance(parsed, dict):
        return None, None

    for key in ("copper_mg_per_serving", "total_copper_mg"):
        value = maybe_float(parsed.get(key))
        if value is not None and key == "copper_mg_per_serving":
            return value, "parsed_output.copper_mg_per_serving"

    return None, None


def extract_prediction_from_raw_output(record: dict[str, Any]) -> tuple[float | None, str | None]:
    raw_output = record.get("raw_output")
    if not isinstance(raw_output, str) or not raw_output.strip():
        return None, None

    parsed_from_text = None
    try:
        parsed_from_text = extract_json_object_from_text(raw_output)
    except ValueError:
        parsed_from_text = None

    if isinstance(parsed_from_text, dict):
        value = maybe_float(parsed_from_text.get("copper_mg_per_serving"))
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
    value, source = extract_prediction_from_parsed_output(record)
    if value is not None:
        return value, source

    value, source = extract_prediction_from_raw_output(record)
    if value is not None:
        return value, source

    return None, None


def safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return statistics.fmean(values)


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
    summary_rows: list[dict[str, Any]] = []

    for path in jsonl_files:
        latest_records = choose_latest_records(path)
        abs_errors: list[float] = []
        sq_errors: list[float] = []
        signed_errors: list[float] = []
        matched_count = 0
        api_success_count = 0
        usable_output_count = 0
        extracted_count = 0

        for recipe_index, recipe in recipes.items():
            record = latest_records.get(recipe_index)
            actual = maybe_float(recipe.get("actual_copper_mg_per_serving"))
            predicted = None
            extraction_source = None
            has_raw_output = False
            api_success = False

            if record is not None:
                api_success = bool(record.get("api_success"))
                raw_output = record.get("raw_output")
                has_raw_output = isinstance(raw_output, str) and bool(raw_output.strip())
                if api_success:
                    api_success_count += 1
                if has_raw_output:
                    usable_output_count += 1
                predicted, extraction_source = extract_prediction(record)
                if predicted is not None:
                    extracted_count += 1

            error = None
            abs_error = None
            sq_error = None
            if actual is not None and predicted is not None:
                error = predicted - actual
                abs_error = abs(error)
                sq_error = error * error
                signed_errors.append(error)
                abs_errors.append(abs_error)
                sq_errors.append(sq_error)
                matched_count += 1

            detail_rows.append(
                {
                    "file_name": path.name,
                    "model": record.get("model") if record else None,
                    "reasoning_effort": record.get("reasoning_effort") if record else None,
                    "recipe_index": recipe_index,
                    "recipe_type": recipe.get("recipe_type"),
                    "recipe_name": recipe.get("recipe_name"),
                    "actual_copper_mg_per_serving": actual,
                    "predicted_copper_mg_per_serving": predicted,
                    "error_mg": round_or_none(error),
                    "absolute_error_mg": round_or_none(abs_error),
                    "prediction_source": extraction_source,
                    "api_success": api_success,
                    "has_raw_output": has_raw_output,
                    "extracted_json_present": record.get("extracted_json_present") if record else None,
                    "response_status": record.get("response_status") if record else None,
                    "response_incomplete_details": (
                        json.dumps(record.get("response_incomplete_details"))
                        if record and record.get("response_incomplete_details") is not None
                        else None
                    ),
                    "raw_output_preview": (
                        record.get("raw_output", "")[:180].replace("\n", " ")
                        if record and isinstance(record.get("raw_output"), str)
                        else None
                    ),
                }
            )

        rmse = math.sqrt(statistics.fmean(sq_errors)) if sq_errors else None
        summary_rows.append(
            {
                "file_name": path.name,
                "model": next(
                    (record.get("model") for record in latest_records.values() if isinstance(record, dict)),
                    None,
                ),
                "reasoning_effort": next(
                    (
                        record.get("reasoning_effort")
                        for record in latest_records.values()
                        if isinstance(record, dict)
                    ),
                    None,
                ),
                "recipes_in_reference": len(recipes),
                "trial_results_found": len(latest_records),
                "api_success_count": api_success_count,
                "nonempty_raw_output_count": usable_output_count,
                "predictions_extracted_count": extracted_count,
                "matched_with_reference_count": matched_count,
                "coverage_rate": round_or_none(matched_count / len(recipes) if recipes else None),
                "mean_signed_error_mg": round_or_none(safe_mean(signed_errors)),
                "mae_mg": round_or_none(safe_mean(abs_errors)),
                "rmse_mg": round_or_none(rmse),
            }
        )

    write_csv(
        Path(args.summary_csv),
        summary_rows,
        [
            "file_name",
            "model",
            "reasoning_effort",
            "recipes_in_reference",
            "trial_results_found",
            "api_success_count",
            "nonempty_raw_output_count",
            "predictions_extracted_count",
            "matched_with_reference_count",
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
            "file_name",
            "model",
            "reasoning_effort",
            "recipe_index",
            "recipe_type",
            "recipe_name",
            "actual_copper_mg_per_serving",
            "predicted_copper_mg_per_serving",
            "error_mg",
            "absolute_error_mg",
            "prediction_source",
            "api_success",
            "has_raw_output",
            "extracted_json_present",
            "response_status",
            "response_incomplete_details",
            "raw_output_preview",
        ],
    )

    summary_rows_sorted = sorted(
        summary_rows,
        key=lambda row: (
            row["mae_mg"] is None,
            row["mae_mg"] if row["mae_mg"] is not None else float("inf"),
        ),
    )
    print(f"Wrote summary: {args.summary_csv}")
    print(f"Wrote details: {args.details_csv}")
    print()
    print("Top models by MAE:")
    for row in summary_rows_sorted[:10]:
        print(
            f"{row['file_name']}: "
            f"coverage={row['matched_with_reference_count']}/{row['recipes_in_reference']}, "
            f"MAE={row['mae_mg']}, RMSE={row['rmse_mg']}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
