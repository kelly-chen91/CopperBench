| Stage | Input condition | System | Runs | MAE | Mean signed error | +/-20% accuracy | Takeaway |
|---|---|---|---:|---:|---:|---:|---|
| Stage 1 | Full recipes with ingredient proportions | `gpt_5_4_mini__reasoning_medium` | 1 | 0.0550 | +0.0315 | 42.31% | Strong baseline when quantities are available. |
| Stage 2 | Ingredient names only; no proportions | `gpt_5_4_mini__reasoning_medium` | 3 | 0.1023 | +0.0933 | 21.79% | Removing quantities roughly doubles MAE and increases overestimation. |
| Stage 3 | Full recipes with targeted prompt mitigation | `combined_persona_few_shot__gpt_5_4_mini__reasoning_medium` | 3 | 0.0452 | +0.0102 | 47.44% | Failure-mode-informed prompting gives the best MAE. |
