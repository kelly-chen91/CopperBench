#!/usr/bin/env python3
"""Evaluate Stage 2 model outputs with the Stage 1 evaluator defaults overridden."""

from __future__ import annotations

import sys
from pathlib import Path


STAGE2_DIR = Path(__file__).resolve().parent
ROOT_DIR = STAGE2_DIR.parent
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


def main() -> None:
    sys.argv[1:1] = DEFAULT_ARGS
    evaluate_model_outputs.main()


if __name__ == "__main__":
    main()
