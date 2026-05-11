# Stage 1: Recipe Copper Evaluation Without Optional Ingredients

This analysis repeats the `stage1_recipe_proportions` evaluation while excluding
recipes whose ingredient list strictly says `optional`.

## Excluded Recipes

The following recipes are excluded from scoring:

| Recipe Index | Recipe Type | Recipe | Ground Truth Copper Per Serving |
| ---: | --- | --- | ---: |
| 4 | `breakfast` | `Glorious Green Smoothie Bowl` | 0.17 mg |
| 10 | `anytime_meals` | `Fish Tacos with Crunchy Slaw` | 0.14 mg |
| 12 | `anytime_meals` | `Roasted Cauliflower Power Salad` | 0.12 mg |
| 19 | `desserts` | `Pumpkin Cinnamon Bars` | 0.06 mg |

## Outputs

Run results are written to
`stage1_recipe_proportions_no_optional/evaluation_results/`:

- `model_ranking.csv`: overall model ranking by MAE.
- `recipe_type_metrics.csv`: MAE and accuracy grouped by recipe type.
- `per_recipe_predictions.csv`: per-recipe prediction and error details.
- `evaluation_metadata.json`: source recipe count, excluded recipes, scored
  recipe count, model file count, and parse metadata.
- `mae_comparison.png`: MAE bar chart for all models.
- `accuracy_comparison.png`: +/-10% accuracy bar chart for all models.
- `mae_by_recipe_type_heatmap.png`: recipe-type MAE heatmap.
- `reasoning_effort_effect.png`: GPT-5-family reasoning-effort line plot.

## Current Results

This filtered dataset scores 22 recipes after removing the 4 optional-ingredient
recipes. The current run discovered 18 model output files.

Top three models by MAE:

| Rank | Model | MAE | Accuracy Within +/-10% | Valid Predictions |
| --- | --- | ---: | ---: | ---: |
| 1 | `gpt_5_4_mini__reasoning_low` | 0.048182 | 13.64% | 22/22 |
| 2 | `gpt_5_4_mini__reasoning_medium` | 0.051318 | 27.27% | 22/22 |
| 3 | `gpt_5_4_pro__reasoning_high` | 0.058455 | 18.18% | 22/22 |

The best MAE model remains `gpt_5_4_mini__reasoning_low`. As in the original
stage 1 analysis, `gpt_5_4_mini__reasoning_medium` has meaningfully higher
within-10% accuracy despite slightly worse MAE.

## Reproducibility

Run the filtered evaluation:

```bash
python3 stage1_recipe_proportions/evaluate_model_outputs.py \
  --results-dir stage1_recipe_proportions_no_optional/evaluation_results \
  --exclude-recipe-name "Glorious Green Smoothie Bowl" \
  --exclude-recipe-name "Fish Tacos with Crunchy Slaw" \
  --exclude-recipe-name "Roasted Cauliflower Power Salad" \
  --exclude-recipe-name "Pumpkin Cinnamon Bars"
```

Run CSV/JSON generation without visualizations:

```bash
python3 stage1_recipe_proportions/evaluate_model_outputs.py \
  --skip-plots \
  --results-dir stage1_recipe_proportions_no_optional/evaluation_results \
  --exclude-recipe-name "Glorious Green Smoothie Bowl" \
  --exclude-recipe-name "Fish Tacos with Crunchy Slaw" \
  --exclude-recipe-name "Roasted Cauliflower Power Salad" \
  --exclude-recipe-name "Pumpkin Cinnamon Bars"
```

## Verification

Completed checks:

- Confirmed metadata reports 26 source recipes, 4 excluded recipes, and 22
  scored recipes.
- Confirmed the four excluded recipe names do not appear in
  `per_recipe_predictions.csv`.
- Generated all CSV, JSON, and visualization outputs in the separate results
  directory.
