# Copper Results Summary

## Main Takeaway

If cost is ignored, the strongest overall performer in the current experiment was `gpt-5.4-mini` with `low` reasoning.

## Overall Best Model

| Metric | Best Result |
|---|---|
| Model | `gpt-5.4-mini` |
| Reasoning | `low` |
| File | `gpt_5_4_mini__reasoning_low.jsonl` |
| MAE | `0.049615` |
| RMSE | `0.064599` |
| Source | [copper_analysis_summary.csv](/Users/mehadichowdhury/Documents/CopperBench/copper_analysis_summary.csv) |

## Best By Recipe Type

| Recipe Type | Best Model | Reasoning | Source File |
|---|---|---|---|
| `breakfast` | `gpt-5.4-mini` | `medium` | `gpt_5_4_mini__reasoning_medium.jsonl` |
| `anytime_meals` | `gpt-5.4-mini` | `low` | `gpt_5_4_mini__reasoning_low.jsonl` |
| `desserts` | `gpt-5.4` | `medium` | `gpt_5_4__reasoning_medium.jsonl` |
| `snacks` | `gpt-5.4-mini` | `medium` | `gpt_5_4_mini__reasoning_medium.jsonl` |

Source: [copper_recipe_type_rankings.csv](/Users/mehadichowdhury/Documents/CopperBench/copper_recipe_type_rankings.csv)

## CSV Guide

| File | Purpose | Best Use |
|---|---|---|
| [copper_analysis_summary.csv](/Users/mehadichowdhury/Documents/CopperBench/copper_analysis_summary.csv) | Overall model leaderboard across all recipes | Find the best model overall |
| [copper_analysis_details.csv](/Users/mehadichowdhury/Documents/CopperBench/copper_analysis_details.csv) | Per-recipe error breakdown for each model | Inspect specific wins, misses, and extracted predictions |
| [copper_recipe_type_rankings.csv](/Users/mehadichowdhury/Documents/CopperBench/copper_recipe_type_rankings.csv) | Ranking within each recipe category | Find the best model for breakfasts, desserts, snacks, or anytime meals |
| [copper_recipe_type_details.csv](/Users/mehadichowdhury/Documents/CopperBench/copper_recipe_type_details.csv) | Per-recipe support table grouped by recipe type | Diagnose category-specific behavior |
| [copper_cost_adjusted_rankings.csv](/Users/mehadichowdhury/Documents/CopperBench/copper_cost_adjusted_rankings.csv) | Accuracy and estimated cost combined into one score | Pick the best accuracy-to-cost tradeoff |
| [copper_cost_adjusted_details.csv](/Users/mehadichowdhury/Documents/CopperBench/copper_cost_adjusted_details.csv) | Per-recipe token and cost estimation support | Audit the cost-adjusted calculation |

## Important Columns

| File | Column | Meaning |
|---|---|---|
| `copper_analysis_summary.csv` | `mae_mg` | Average absolute error in mg, lower is better |
| `copper_analysis_summary.csv` | `rmse_mg` | Root mean squared error, lower is better |
| `copper_analysis_summary.csv` | `matched_with_reference_count` | Number of recipes with usable predictions |
| `copper_analysis_summary.csv` | `coverage_rate` | Fraction of recipes with usable predictions |
| `copper_analysis_details.csv` | `actual_copper_mg_per_serving` | Reference copper value from `recipes.json` |
| `copper_analysis_details.csv` | `predicted_copper_mg_per_serving` | Model-predicted copper value |
| `copper_analysis_details.csv` | `absolute_error_mg` | Absolute difference between actual and predicted |
| `copper_recipe_type_rankings.csv` | `rank_within_recipe_type` | Model rank inside a recipe category |
| `copper_cost_adjusted_rankings.csv` | `cost_adjusted_score` | Weighted score combining accuracy and estimated cost, lower is better |

## Cost Caveat

| Issue | Meaning |
|---|---|
| Visible-token pricing only | Cost estimates are based on logged prompt and output text |
| Hidden reasoning not included | GPT-5-family reasoning runs may be underpriced in the estimate |
| Cost-adjusted ranking is approximate | Good for relative comparison, not exact billing |

## Bottom Line

| Question | Answer |
|---|---|
| Best overall ignoring cost | `gpt-5.4-mini` with `low` reasoning |
| Best model family overall | `gpt-5.4-mini` |
| Same winner for every recipe type? | No |
| Strongest default choice from this experiment | `gpt-5.4-mini` |
