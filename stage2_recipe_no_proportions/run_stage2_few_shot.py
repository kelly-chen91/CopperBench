#!/usr/bin/env python3
"""Run the Stage 2 few-shot prompt experiment."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


STAGE2_DIR = Path(__file__).resolve().parent
ROOT_DIR = STAGE2_DIR.parent
DEFAULT_ARGS = [
    "--input",
    str(STAGE2_DIR / "recipes_no_proportions.json"),
    "--output-dir",
    str(STAGE2_DIR / "model_outputs"),
    "--models",
    "gpt-5.4-mini",
    "--reasoning-efforts",
    "medium",
    "--response-format",
    "json_object",
    "--temperature",
    "0.0",
]


sys.path.insert(0, str(ROOT_DIR))
import run_copper_estimates  # noqa: E402


ORIGINAL_MODEL_VARIANT_OUTPUT_PATH = run_copper_estimates.model_variant_output_path


def render_ingredients(ingredients: list[dict[str, str]]) -> str:
    return "; ".join(item.get("name", "").strip() for item in ingredients)


def load_stage2_few_shot_template() -> str:
    return (STAGE2_DIR / "prompt_templates" / "few_shot_prompt.txt").read_text()


def build_stage2_few_shot_prompt(recipe: dict[str, Any]) -> str:
    return (
        load_stage2_few_shot_template()
        .replace("{recipe_name}", str(recipe["recipe_name"]))
        .replace("{ingredients}", render_ingredients(recipe["ingredients"]))
        .replace("{servings}", str(recipe["servings"]))
    )


def few_shot_output_path(
    output_dir: Path,
    model: str,
    reasoning_effort: str | None,
) -> Path:
    path = ORIGINAL_MODEL_VARIANT_OUTPUT_PATH(
        output_dir,
        model,
        reasoning_effort,
    )
    return path.with_name(f"few_shot__{path.name}")


def main() -> int:
    run_copper_estimates.build_prompt = build_stage2_few_shot_prompt
    run_copper_estimates.model_variant_output_path = few_shot_output_path
    sys.argv[1:1] = DEFAULT_ARGS
    return run_copper_estimates.main()


if __name__ == "__main__":
    raise SystemExit(main())
