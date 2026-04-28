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
DEFAULT_SUMMARY_PATH = "copper_cost_adjusted_rankings.csv"
DEFAULT_DETAILS_PATH = "copper_cost_adjusted_details.csv"

# USD per 1M tokens. These defaults are based on current OpenAI pricing pages
# and can be overridden with --pricing-json.
DEFAULT_PRICING = {
    "gpt-4.1": {"input_per_m": 2.00, "output_per_m": 8.00},
    "gpt-4.1-mini": {"input_per_m": 0.40, "output_per_m": 1.60},
    "gpt-4.1-nano": {"input_per_m": 0.10, "output_per_m": 0.40},
    "gpt-4o": {"input_per_m": 2.50, "output_per_m": 10.00},
    "gpt-4o-mini": {"input_per_m": 0.15, "output_per_m": 0.60},
    "gpt-5.4": {"input_per_m": 2.50, "output_per_m": 15.00},
    "gpt-5.4-mini": {"input_per_m": 0.75, "output_per_m": 4.50},
    "gpt-5.4-nano": {"input_per_m": 0.20, "output_per_m": 1.25},
    "gpt-5.5": {"input_per_m": 5.00, "output_per_m": 30.00},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank models by cost-adjusted accuracy using recipes.json and model output logs."
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
        help=f"Where to write cost-adjusted ranking summary CSV. Default: {DEFAULT_SUMMARY_PATH}",
    )
    parser.add_argument(
        "--details-csv",
        default=DEFAULT_DETAILS_PATH,
        help=f"Where to write per-model cost details CSV. Default: {DEFAULT_DETAILS_PATH}",
    )
    parser.add_argument(
        "--pricing-json",
        help="Optional JSON file mapping model names to {input_per_m, output_per_m}.",
    )
    parser.add_argument(
        "--include-pro",
        action="store_true",
        help="Include *pro* model files in the analysis.",
    )
    parser.add_argument(
        "--accuracy-weight",
        type=float,
        default=0.7,
        help="Weight on normalized MAE in the composite score. Default: 0.7",
    )
    parser.add_argument(
        "--cost-weight",
        type=float,
        default=0.3,
        help="Weight on normalized cost-per-usable-output in the composite score. Default: 0.3",
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


def load_pricing(path: str | None) -> dict[str, dict[str, float]]:
    pricing = dict(DEFAULT_PRICING)
    if not path:
        return pricing
    override = json.loads(Path(path).read_text())
    pricing.update(override)
    return pricing


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


def estimate_visible_tokens(text: str | None) -> int:
    if not isinstance(text, str) or not text:
        return 0
    # Rough token estimate for plain text when exact usage was not logged.
    return max(1, math.ceil(len(text) / 4))


def record_has_usable_output(record: dict[str, Any]) -> bool:
    if not record.get("api_success"):
        return False
    raw_output = record.get("raw_output")
    if not isinstance(raw_output, str) or not raw_output.strip():
        return False
    response_status = record.get("response_status")
    if response_status not in {None, "completed"}:
        return False
    if record.get("response_incomplete_details"):
        return False
    return True


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
    if args.accuracy_weight < 0 or args.cost_weight < 0:
        raise SystemExit("Weights must be non-negative.")
    if args.accuracy_weight == 0 and args.cost_weight == 0:
        raise SystemExit("At least one of --accuracy-weight or --cost-weight must be positive.")

    recipes = load_reference_recipes(Path(args.recipes))
    pricing = load_pricing(args.pricing_json)
    output_dir = Path(args.output_dir)
    jsonl_files = sorted(output_dir.glob("*.jsonl"))
    if not args.include_pro:
        jsonl_files = [path for path in jsonl_files if not is_pro_file(path)]

    summary_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []

    for path in jsonl_files:
        latest_records = choose_latest_records(path)
        if not latest_records:
            continue

        model = next(
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

        model_pricing = pricing.get(model or "")
        total_input_tokens_est = 0
        total_output_tokens_est = 0
        abs_errors: list[float] = []
        signed_errors: list[float] = []
        matched_count = 0
        usable_output_count = 0

        for recipe_index, recipe in recipes.items():
            record = latest_records.get(recipe_index)
            if not record:
                continue

            prompt = record.get("prompt")
            raw_output = record.get("raw_output")
            input_tokens_est = estimate_visible_tokens(prompt)
            output_tokens_est = estimate_visible_tokens(raw_output)
            usable_output = record_has_usable_output(record)
            if usable_output:
                usable_output_count += 1

            predicted, prediction_source = extract_prediction(record)
            actual = maybe_float(recipe.get("actual_copper_mg_per_serving"))
            error = None
            abs_error = None
            if actual is not None and predicted is not None:
                error = predicted - actual
                abs_error = abs(error)
                signed_errors.append(error)
                abs_errors.append(abs_error)
                matched_count += 1

            total_input_tokens_est += input_tokens_est
            total_output_tokens_est += output_tokens_est

            detail_rows.append(
                {
                    "file_name": path.name,
                    "model": model,
                    "reasoning_effort": reasoning_effort,
                    "recipe_index": recipe_index,
                    "recipe_type": recipe.get("recipe_type"),
                    "recipe_name": recipe.get("recipe_name"),
                    "input_tokens_est": input_tokens_est,
                    "output_tokens_est": output_tokens_est,
                    "usable_output": usable_output,
                    "actual_copper_mg_per_serving": actual,
                    "predicted_copper_mg_per_serving": predicted,
                    "absolute_error_mg": round_or_none(abs_error),
                    "prediction_source": prediction_source,
                }
            )

        estimated_cost_usd = None
        estimated_cost_per_usable_output_usd = None
        if model_pricing is not None:
            estimated_cost_usd = (
                (total_input_tokens_est / 1_000_000) * model_pricing["input_per_m"]
                + (total_output_tokens_est / 1_000_000) * model_pricing["output_per_m"]
            )
            if usable_output_count > 0:
                estimated_cost_per_usable_output_usd = estimated_cost_usd / usable_output_count

        rmse = math.sqrt(statistics.fmean([x * x for x in signed_errors])) if signed_errors else None
        mae = statistics.fmean(abs_errors) if abs_errors else None
        summary_rows.append(
            {
                "file_name": path.name,
                "model": model,
                "reasoning_effort": reasoning_effort,
                "recipes_in_reference": len(recipes),
                "matched_predictions": matched_count,
                "usable_outputs": usable_output_count,
                "coverage_rate": round_or_none(matched_count / len(recipes) if recipes else None),
                "mae_mg": round_or_none(mae),
                "rmse_mg": round_or_none(rmse),
                "mean_signed_error_mg": round_or_none(statistics.fmean(signed_errors) if signed_errors else None),
                "estimated_input_tokens_visible": total_input_tokens_est,
                "estimated_output_tokens_visible": total_output_tokens_est,
                "estimated_total_cost_usd_visible": round_or_none(estimated_cost_usd, 8),
                "estimated_cost_per_usable_output_usd_visible": round_or_none(
                    estimated_cost_per_usable_output_usd, 8
                ),
                "pricing_available": model_pricing is not None,
            }
        )

    comparable_rows = [
        row
        for row in summary_rows
        if row["mae_mg"] is not None and row["estimated_cost_per_usable_output_usd_visible"] is not None
    ]
    min_mae = min((row["mae_mg"] for row in comparable_rows), default=None)
    min_cost = min(
        (row["estimated_cost_per_usable_output_usd_visible"] for row in comparable_rows),
        default=None,
    )

    for row in summary_rows:
        normalized_mae = None
        normalized_cost = None
        composite_score = None
        if min_mae and row["mae_mg"] is not None:
            normalized_mae = row["mae_mg"] / min_mae
        if min_cost and row["estimated_cost_per_usable_output_usd_visible"] is not None:
            normalized_cost = row["estimated_cost_per_usable_output_usd_visible"] / min_cost
        if normalized_mae is not None and normalized_cost is not None:
            composite_score = (
                args.accuracy_weight * normalized_mae
                + args.cost_weight * normalized_cost
            )

        row["normalized_mae"] = round_or_none(normalized_mae)
        row["normalized_cost"] = round_or_none(normalized_cost)
        row["cost_adjusted_score"] = round_or_none(composite_score)

    summary_rows.sort(
        key=lambda row: (
            row["cost_adjusted_score"] is None,
            row["cost_adjusted_score"] if row["cost_adjusted_score"] is not None else float("inf"),
        )
    )
    for rank, row in enumerate(summary_rows, start=1):
        row["overall_rank"] = rank

    write_csv(
        Path(args.summary_csv),
        summary_rows,
        [
            "overall_rank",
            "file_name",
            "model",
            "reasoning_effort",
            "recipes_in_reference",
            "matched_predictions",
            "usable_outputs",
            "coverage_rate",
            "mae_mg",
            "rmse_mg",
            "mean_signed_error_mg",
            "estimated_input_tokens_visible",
            "estimated_output_tokens_visible",
            "estimated_total_cost_usd_visible",
            "estimated_cost_per_usable_output_usd_visible",
            "normalized_mae",
            "normalized_cost",
            "cost_adjusted_score",
            "pricing_available",
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
            "input_tokens_est",
            "output_tokens_est",
            "usable_output",
            "actual_copper_mg_per_serving",
            "predicted_copper_mg_per_serving",
            "absolute_error_mg",
            "prediction_source",
        ],
    )

    print(f"Wrote summary: {args.summary_csv}")
    print(f"Wrote details: {args.details_csv}")
    print()
    print(
        f"Composite score = {args.accuracy_weight} * normalized_MAE + "
        f"{args.cost_weight} * normalized_cost_per_usable_output"
    )
    print("Lower is better.")
    print()
    for row in summary_rows[:10]:
        print(
            f"#{row['overall_rank']} {row['file_name']}: "
            f"MAE={row['mae_mg']}, "
            f"cost/output=${row['estimated_cost_per_usable_output_usd_visible']}, "
            f"score={row['cost_adjusted_score']}"
        )
    print()
    print(
        "Note: cost estimates are based on visible prompt/output text only. "
        "They do not include hidden reasoning tokens, so GPT-5-family runs may be undercounted."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
