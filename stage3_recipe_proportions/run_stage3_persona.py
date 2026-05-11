#!/usr/bin/env python3
"""Run the Stage 3 persona-prompt experiment on recipes with proportions."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


STAGE3_DIR = Path(__file__).resolve().parent
ROOT_DIR = STAGE3_DIR.parent
DEFAULT_ARGS = [
    "--input",
    str(ROOT_DIR / "recipes.json"),
    "--output-dir",
    str(STAGE3_DIR / "model_outputs"),
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


def load_persona_template() -> str:
    return (STAGE3_DIR / "prompt_templates" / "persona_prompt.txt").read_text()


def build_persona_prompt(recipe: dict[str, Any]) -> str:
    return (
        load_persona_template()
        .replace("{recipe_name}", str(recipe["recipe_name"]))
        .replace("{ingredients}", run_copper_estimates.render_ingredients(recipe["ingredients"]))
        .replace("{servings}", str(recipe["servings"]))
    )


def persona_output_path(
    output_dir: Path,
    model: str,
    reasoning_effort: str | None,
) -> Path:
    path = ORIGINAL_MODEL_VARIANT_OUTPUT_PATH(output_dir, model, reasoning_effort)
    return path.with_name(f"persona__{path.name}")


def main() -> int:
    run_copper_estimates.build_prompt = build_persona_prompt
    run_copper_estimates.model_variant_output_path = persona_output_path
    sys.argv[1:1] = DEFAULT_ARGS
    return run_copper_estimates.main()


if __name__ == "__main__":
    raise SystemExit(main())
