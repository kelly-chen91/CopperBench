#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_INPUT = "recipes.json"
DEFAULT_OUTPUT_DIR = "model_outputs"
DEFAULT_MODELS = [
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.4-nano",
    "gpt-5.4-pro",
    "gpt-5.5",
    "gpt-5.5-pro",
]
DEFAULT_REASONING_EFFORTS = ["low", "medium", "high"]
PROMPT_TEMPLATE = """Estimate the copper content (mg) per serving for the following recipe. Return JSON with: dish_name, servings, total_copper_mg, copper_mg_per_serving.

Recipe Name: {recipe_name}
Ingredients: {ingredients}
Servings: {servings}
"""
COPPER_ESTIMATE_SCHEMA = {
    "type": "object",
    "properties": {
        "dish_name": {"type": "string"},
        "servings": {"type": ["string", "number"]},
        "total_copper_mg": {"type": "number"},
        "copper_mg_per_serving": {"type": "number"},
    },
    "required": [
        "dish_name",
        "servings",
        "total_copper_mg",
        "copper_mg_per_serving",
    ],
    "additionalProperties": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run copper-estimate prompts for each recipe across multiple OpenAI models."
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help=f"Path to recipe JSON file. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for per-model JSONL output files. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=DEFAULT_MODELS,
        help="Override the model list.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=None,
        help="Optional max output tokens per API call. Default: unset",
    )
    parser.add_argument(
        "--response-format",
        choices=["json_schema", "json_object", "text"],
        default="text",
        help="Response format to request from the API. Default: text",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--reasoning-efforts",
        nargs="*",
        default=DEFAULT_REASONING_EFFORTS,
        help="Reasoning efforts to use for GPT-5-family models.",
    )
    parser.add_argument(
        "--reasoning-summary",
        default="auto",
        help="Reasoning summary setting for reasoning-capable models. Default: auto",
    )
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=0.0,
        help="Delay between requests.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files instead of appending/skipping completed recipes.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately on the first request failure.",
    )
    return parser.parse_args()


def require_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Missing OPENAI_API_KEY in environment.")
    return api_key


def require_openai_sdk():
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit(
            "Missing Python package 'openai'. Install it with: pip install openai"
        ) from exc
    return OpenAI


def iter_recipe_dicts(recipes_node: Any):
    if isinstance(recipes_node, dict):
        yield recipes_node
        return

    if isinstance(recipes_node, list):
        for item in recipes_node:
            yield from iter_recipe_dicts(item)


def load_recipes(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    flat_recipes: list[dict[str, Any]] = []
    recipe_index = 0

    for group in data:
        recipe_type = group.get("recipe_type")
        for recipe in iter_recipe_dicts(group.get("recipes", [])):
            if not isinstance(recipe, dict):
                continue
            flat_recipes.append(
                {
                    "recipe_index": recipe_index,
                    "recipe_type": recipe_type,
                    "recipe_name": recipe["name"],
                    "servings": recipe["servings"],
                    "ingredients": recipe["ingredients"],
                }
            )
            recipe_index += 1

    return flat_recipes


def render_ingredients(ingredients: list[dict[str, str]]) -> str:
    return "; ".join(
        f"{item.get('proportion', '').strip()} {item.get('name', '').strip()}".strip()
        for item in ingredients
    )


def build_prompt(recipe: dict[str, Any]) -> str:
    return PROMPT_TEMPLATE.format(
        recipe_name=recipe["recipe_name"],
        ingredients=render_ingredients(recipe["ingredients"]),
        servings=recipe["servings"],
    )


def is_reasoning_model(model: str) -> bool:
    return model.startswith("gpt-5")


def supported_reasoning_efforts(model: str, requested_efforts: list[str]) -> list[str]:
    if not is_reasoning_model(model):
        return [None]

    if model.endswith("-pro"):
        return ["high"]

    return requested_efforts


def expand_model_variants(models: list[str], reasoning_efforts: list[str]) -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = []
    for model in models:
        for effort in supported_reasoning_efforts(model, reasoning_efforts):
            variants.append({"model": model, "reasoning_effort": effort})
    return variants


def call_responses_api(
    client: Any,
    model: str,
    reasoning_effort: str | None,
    reasoning_summary: str | None,
    response_format: str,
    prompt: str,
    max_output_tokens: int,
    temperature: float,
    retries: int = 3,
) -> Any:
    for attempt in range(1, retries + 1):
        try:
            request_kwargs: dict[str, Any] = {
                "model": model,
                "input": prompt,
            }
            if max_output_tokens is not None:
                request_kwargs["max_output_tokens"] = max_output_tokens

            if response_format == "json_schema":
                request_kwargs["text"] = {
                    "format": {
                        "type": "json_schema",
                        "name": "copper_estimate",
                        "schema": COPPER_ESTIMATE_SCHEMA,
                        "strict": True,
                    }
                }
            elif response_format == "json_object":
                request_kwargs["text"] = {"format": {"type": "json_object"}}

            if reasoning_effort is not None:
                reasoning_config: dict[str, Any] = {"effort": reasoning_effort}
                if reasoning_summary:
                    reasoning_config["summary"] = reasoning_summary
                request_kwargs["reasoning"] = reasoning_config
            else:
                request_kwargs["temperature"] = temperature

            return client.responses.create(**request_kwargs)
        except Exception as exc:  # noqa: BLE001
            status = getattr(exc, "status_code", None)
            if status in {408, 409, 429, 500, 502, 503, 504} and attempt < retries:
                time.sleep(2 ** attempt)
                continue
            raise

    raise RuntimeError("Request failed after retries.")


def extract_output_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    texts: list[str] = []

    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", None) != "message":
            continue
        for content in getattr(item, "content", []) or []:
            content_type = getattr(content, "type", None)
            text = getattr(content, "text", None)
            if content_type in {"output_text", "text"} and text:
                texts.append(text)

    if texts:
        return "\n".join(texts).strip()

    return ""


def extract_reasoning_summary(response: Any) -> tuple[bool, str | None]:
    reasoning_texts: list[str] = []
    reasoning_item_present = False

    for output_item in getattr(response, "output", []) or []:
        if getattr(output_item, "type", None) != "reasoning":
            continue
        reasoning_item_present = True
        for summary_item in getattr(output_item, "summary", []) or []:
            text = getattr(summary_item, "text", None)
            if text:
                reasoning_texts.append(text)

    return reasoning_item_present, "\n".join(reasoning_texts) if reasoning_texts else None


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if not stripped:
        raise ValueError("Empty model output.")

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL)
    if fenced:
        stripped = fenced.group(1)

    decoder = json.JSONDecoder()
    try:
        obj, _ = decoder.raw_decode(stripped)
        if not isinstance(obj, dict):
            raise ValueError("Model output was valid JSON but not a JSON object.")
        return obj
    except json.JSONDecodeError:
        pass

    # Some model replies include prose before the JSON block.
    for match in re.finditer(r"\{", stripped):
        try:
            obj, _ = decoder.raw_decode(stripped[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj

    raise ValueError("Could not find a JSON object in model output.")


def model_output_path(output_dir: Path, model: str) -> Path:
    safe_name = model.replace(".", "_").replace("-", "_")
    return output_dir / f"{safe_name}.jsonl"


def model_variant_output_path(output_dir: Path, model: str, reasoning_effort: str | None) -> Path:
    if reasoning_effort is None:
        return model_output_path(output_dir, model)
    safe_name = model.replace(".", "_").replace("-", "_")
    safe_effort = reasoning_effort.replace("-", "_")
    return output_dir / f"{safe_name}__reasoning_{safe_effort}.jsonl"


def record_has_usable_output(entry: dict[str, Any]) -> bool:
    if not entry.get("api_success"):
        return False
    if not isinstance(entry.get("recipe_index"), int):
        return False
    raw_output = entry.get("raw_output")
    if not isinstance(raw_output, str) or not raw_output.strip():
        return False
    response_status = entry.get("response_status")
    if response_status not in {None, "completed"}:
        return False
    if entry.get("response_incomplete_details"):
        return False
    return True


def load_completed_recipe_ids(path: Path) -> set[int]:
    completed: set[int] = set()
    if not path.exists():
        return completed

    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record_has_usable_output(entry):
                completed.add(entry["recipe_index"])
    return completed


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def main() -> int:
    args = parse_args()
    require_api_key()
    OpenAI = require_openai_sdk()
    client = OpenAI()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    recipes = load_recipes(input_path)
    model_variants = expand_model_variants(args.models, args.reasoning_efforts)
    total_requests = len(recipes) * len(model_variants)
    print(
        f"Loaded {len(recipes)} recipes. Preparing {total_requests} requests across {len(model_variants)} model variants.",
        file=sys.stderr,
    )

    for variant in model_variants:
        model = variant["model"]
        reasoning_effort = variant["reasoning_effort"]
        output_path = model_variant_output_path(output_dir, model, reasoning_effort)

        if args.overwrite and output_path.exists():
            output_path.unlink()

        completed_ids = load_completed_recipe_ids(output_path)
        remaining = [r for r in recipes if r["recipe_index"] not in completed_ids]

        print(
            f"[{model} | reasoning={reasoning_effort or 'none'}] {len(completed_ids)} completed, {len(remaining)} remaining. Writing to {output_path}.",
            file=sys.stderr,
        )

        experiment_config = {
            "model": model,
            "reasoning_effort": reasoning_effort,
            "reasoning_summary": args.reasoning_summary if reasoning_effort is not None else None,
            "input_json_path": str(input_path),
            "num_recipes": len(recipes),
            "response_format": args.response_format,
            "max_output_tokens": args.max_output_tokens,
            "temperature": None if reasoning_effort is not None else args.temperature,
        }

        if output_path.stat().st_size == 0 if output_path.exists() else True:
            append_jsonl(
                output_path,
                {
                    "record_type": "experiment_config",
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "config": experiment_config,
                },
            )

        for recipe in remaining:
            prompt = build_prompt(recipe)
            timestamp = datetime.now(timezone.utc).isoformat()
            raw_output = None

            try:
                response_json = call_responses_api(
                    client=client,
                    model=model,
                    reasoning_effort=reasoning_effort,
                    reasoning_summary=args.reasoning_summary,
                    response_format=args.response_format,
                    prompt=prompt,
                    max_output_tokens=args.max_output_tokens,
                    temperature=args.temperature,
                )
                raw_output = extract_output_text(response_json)
                reasoning_item_present, reasoning_summary_text = extract_reasoning_summary(response_json)
                response_status = getattr(response_json, "status", None)
                incomplete_details = getattr(response_json, "incomplete_details", None)
                if incomplete_details is not None:
                    incomplete_details = (
                        incomplete_details.to_dict()
                        if hasattr(incomplete_details, "to_dict")
                        else str(incomplete_details)
                    )

                parsed_output = None
                extracted_json_present = False
                extracted_json_error = None
                json_extraction_skipped_reason = None
                if not isinstance(raw_output, str) or not raw_output.strip():
                    json_extraction_skipped_reason = "empty_raw_output"
                elif response_status not in {None, "completed"}:
                    json_extraction_skipped_reason = f"response_status_{response_status}"
                elif incomplete_details:
                    json_extraction_skipped_reason = "response_incomplete"
                else:
                    try:
                        parsed_output = extract_json_object(raw_output)
                        extracted_json_present = True
                    except Exception as exc:  # noqa: BLE001
                        extracted_json_error = str(exc)

                record = {
                    "record_type": "trial_result",
                    "timestamp_utc": timestamp,
                    "success": True,
                    "api_success": True,
                    "model": model,
                    "reasoning_effort": reasoning_effort,
                    "reasoning_summary_requested": args.reasoning_summary if reasoning_effort is not None else None,
                    "reasoning_item_present": reasoning_item_present,
                    "reasoning_summary_text": reasoning_summary_text,
                    "cot_requested": reasoning_effort is not None,
                    "cot_exposed_by_api": False,
                    "recipe_index": recipe["recipe_index"],
                    "recipe_type": recipe["recipe_type"],
                    "recipe_name": recipe["recipe_name"],
                    "servings_input": recipe["servings"],
                    "prompt": prompt,
                    "parsed_output": parsed_output,
                    "extracted_json_present": extracted_json_present,
                    "extracted_json_error": extracted_json_error,
                    "json_extraction_skipped_reason": json_extraction_skipped_reason,
                    "raw_output": raw_output,
                    "response_id": getattr(response_json, "id", None),
                    "response_status": response_status,
                    "response_incomplete_details": incomplete_details,
                }
                append_jsonl(output_path, record)
                print(
                    f"[{model} | reasoning={reasoning_effort or 'none'}] recipe {recipe['recipe_index'] + 1}/{len(recipes)} ok: {recipe['recipe_name']}",
                    file=sys.stderr,
                )
            except Exception as exc:  # noqa: BLE001
                record = {
                    "record_type": "trial_result",
                    "timestamp_utc": timestamp,
                    "success": False,
                    "api_success": False,
                    "model": model,
                    "reasoning_effort": reasoning_effort,
                    "reasoning_summary_requested": args.reasoning_summary if reasoning_effort is not None else None,
                    "cot_requested": reasoning_effort is not None,
                    "cot_exposed_by_api": False,
                    "recipe_index": recipe["recipe_index"],
                    "recipe_type": recipe["recipe_type"],
                    "recipe_name": recipe["recipe_name"],
                    "servings_input": recipe["servings"],
                    "prompt": prompt,
                    "parsed_output": None,
                    "extracted_json_present": False,
                    "extracted_json_error": None,
                    "json_extraction_skipped_reason": None,
                    "raw_output": raw_output,
                    "error": str(exc),
                }
                append_jsonl(output_path, record)
                print(
                    f"[{model} | reasoning={reasoning_effort or 'none'}] recipe {recipe['recipe_index'] + 1}/{len(recipes)} failed: {recipe['recipe_name']} :: {exc}",
                    file=sys.stderr,
                )
                if args.fail_fast:
                    return 1

            if args.delay_seconds > 0:
                time.sleep(args.delay_seconds)

        append_jsonl(
            output_path,
            {
                "record_type": "experiment_end",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "config": experiment_config,
            },
        )

    print("Finished.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
