# Stage 1: Recipe Copper Proportion Evaluation

This stage evaluates model predictions for `copper_mg_per_serving` against the
ground-truth `copper_per_serving_mg` values in `recipes.json`.

## Methodology

The evaluator in `evaluate_model_outputs.py`:

1. Loads and flattens `recipes.json`, preserving `recipe_index`, `recipe_type`,
   recipe name, and ground-truth copper per serving.
2. Dynamically discovers every `*.jsonl` file in `model_outputs/`.
3. Parses `experiment_config` metadata and `trial_result` records.
4. Keeps the latest trial result per `recipe_index` for each model file. This
   handles append-style reruns without double-counting duplicate recipe trials.
5. Extracts `parsed_output.copper_mg_per_serving`.
6. Marks failed or malformed predictions invalid and excludes them from MAE and
   accuracy calculations.

The primary ranking metric is mean absolute error (MAE), lower is better.
The secondary metric is the percentage of valid predictions within +/-10% of
the ground-truth value.

## Outputs

Run results are written to `stage1_recipe_proportions/evaluation_results/`:

- `model_ranking.csv`: overall model ranking by MAE, with valid/invalid counts.
- `recipe_type_metrics.csv`: MAE and accuracy grouped by recipe type.
- `per_recipe_predictions.csv`: per-recipe prediction, error, validity, and
  failure details.
- `evaluation_metadata.json`: recipe count, model file count, discovered files,
  and parse metadata.
- `mae_comparison.png`: MAE bar chart for all models.
- `accuracy_comparison.png`: +/-10% accuracy bar chart for all models.
- `mae_by_recipe_type_heatmap.png`: recipe-type MAE heatmap.
- `reasoning_effort_effect.png`: GPT-5-family reasoning-effort line plot.

## Current Results

The current dataset contains 26 recipes and 18 discovered model output files.

Top three models by MAE:

| Rank | Model | MAE | Accuracy Within +/-10% | Valid Predictions |
| --- | --- | ---: | ---: | ---: |
| 1 | `gpt_5_4_mini__reasoning_low` | 0.049615 | 15.38% | 26/26 |
| 2 | `gpt_5_4_mini__reasoning_medium` | 0.054962 | 26.92% | 26/26 |
| 3 | `gpt_5_4_pro__reasoning_high` | 0.059269 | 19.23% | 26/26 |

Best-performing model: `gpt_5_4_mini__reasoning_medium`.

Rationale: While it does not have the lowest overall MAE while producing valid predictions for
all 26 recipes, `gpt_5_4_mini__reasoning_medium` has better +/-10% accuracy. There are not much difference between MAE scores for `gpt_5_4_mini__reasoning_low` and `gpt_5_4_mini__reasoning_medium` and there is a big substantial increase in per-recipe accuracy (~11.5 percentage points increase).

## Reasoning Effort Notes

Reasoning effort did not monotonically improve MAE in the current outputs:

- `gpt_5_4`: low 0.065385, medium 0.064346, high 0.067038.
- `gpt_5_4_mini`: low 0.049615, medium 0.054962, high 0.060308.
- `gpt_5_4_nano`: low 0.392962, medium 0.126714, high 0.158615.
- `gpt_5_5`: low 0.063192, medium 0.071923, high 0.065462.

The lowest-reasoning `gpt_5_4_mini` run is currently the best MAE result.

## Analysis
Even with `gpt_5_4_mini__reasoning_medium` being the best performing model, the per-recipe accuracy is still quite low - since we do have the ability to examine the CoT process, our next step would be that we should do an analysis regarding the CoT for each prompt and find out where did reasoning fail. 

## Reproducibility

Create the uv environment and install plotting dependencies:

```bash
uv venv stage1_recipe_proportions/.venv
uv pip install --python stage1_recipe_proportions/.venv/bin/python -r stage1_recipe_proportions/requirements.txt
```

Run the full evaluation:

```bash
stage1_recipe_proportions/.venv/bin/python stage1_recipe_proportions/evaluate_model_outputs.py
```

Run CSV/JSON generation without visualizations:

```bash
python3 stage1_recipe_proportions/evaluate_model_outputs.py --skip-plots
```

## Verification

Completed checks:

- Loaded 26 recipes with `copper_per_serving_mg` values.
- Discovered and parsed all 19 model JSONL files.
- Parsed `gpt_4o.jsonl` trial results successfully.
- Generated overall CSV ranking for all 19 model files.
- Generated all four visualization files with Matplotlib.
- Spot check: the first `gpt_4o` recipe has prediction 0.2925 mg, ground truth
  0.15 mg, and absolute error `abs(0.2925 - 0.15) = 0.1425`, matching
  `per_recipe_predictions.csv`.

## Open Questions

- The prompts request per-serving values and all valid model outputs are scored
  from `copper_mg_per_serving`; total copper values are not used for ranking.
- `recipes.json` is treated as the source of truth for WDA cookbook copper
  values.
- Outliers are not removed. They remain visible in `per_recipe_predictions.csv`
  and recipe-type summaries for follow-up analysis.
