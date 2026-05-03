#!/usr/bin/env python3
"""Build the Stage 2 no-proportions recipe dataset from the root recipes file."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT_RECIPES_PATH = Path(__file__).resolve().parents[1] / "recipes.json"
OUTPUT_PATH = Path(__file__).resolve().parent / "recipes_no_proportions.json"


def iter_recipe_dicts(node: Any):
    if isinstance(node, dict):
        yield node
        return

    if isinstance(node, list):
        for item in node:
            yield from iter_recipe_dicts(item)


def strip_proportions() -> list[dict[str, Any]]:
    source_data = json.loads(ROOT_RECIPES_PATH.read_text())
    stage2_groups: list[dict[str, Any]] = []

    for group in source_data:
        stage2_recipes = []
        for recipe in iter_recipe_dicts(group.get("recipes", [])):
            ingredient_names = [
                {"name": str(ingredient["name"])}
                for ingredient in recipe.get("ingredients", [])
                if isinstance(ingredient, dict) and ingredient.get("name")
            ]
            recipe_name = str(recipe["name"])
            stage2_recipes.append(
                {
                    "recipe_name": recipe_name,
                    "name": recipe_name,
                    "servings": recipe["servings"],
                    "ingredients": ingredient_names,
                    "copper_per_serving_mg": recipe["copper_per_serving_mg"],
                }
            )

        stage2_groups.append(
            {
                "recipe_type": group.get("recipe_type"),
                "recipes": stage2_recipes,
            }
        )

    return stage2_groups


def main() -> None:
    OUTPUT_PATH.write_text(json.dumps(strip_proportions(), indent=2) + "\n")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
