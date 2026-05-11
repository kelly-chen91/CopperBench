# Stage 2: Recipe Copper Evaluation Without Proportions

This stage evaluates copper-per-serving estimates for recipes where ingredient
proportions are omitted. The baseline uses the best-performing Stage 1 model,
`gpt_5_4_mini__reasoning_medium`.

## Goals

1. Create a baseline evaluation for recipes that include only:
   - recipe name
   - ingredient names
   - serving size
2. Use the Stage 1 best model: `gpt_5_4_mini__reasoning_medium`.
3. Leave the few-shot and persona prompt templates blank for later tuning.
4. Compare baseline results against later few-shot and persona variants.

## Prompt templates

Baseline prompt template:
- `stage2_recipe_no_proportions/prompt_templates/baseline_prompt.txt`

Few-shot prompt template:
- `stage2_recipe_no_proportions/prompt_templates/few_shot_prompt.txt`

Persona prompt template:
- `stage2_recipe_no_proportions/prompt_templates/persona_prompt.txt`

## Implementation plan

1. Prepare the Stage 2 dataset.
   - Add a JSON file like `stage2_recipe_no_proportions/recipes_no_proportions.json`.
   - Each recipe should include `recipe_name`, `servings`, `ingredients`, and
     `copper_per_serving_mg` ground-truth values.
   - Ingredient entries should omit numeric proportions (e.g., `"ingredients":
     [{"name": "eggs"}, {"name": "broccoli"}, ...]`).

2. Use the baseline prompt.
   - Start with `prompt_templates/baseline_prompt.txt`.
   - Fill `{recipe_name}`, `{ingredients}`, and `{servings}` for each recipe.
   - Keep the prompt focused on estimating copper per serving without amounts.

3. Run the baseline evaluation.
   - Use the existing `run_copper_estimates.py` driver.
   - Target model: `gpt-5.4-mini` with `reasoning_effort=medium`.
   - Example command:
     ```bash
     python3 run_copper_estimates.py \
       --input stage2_recipe_no_proportions/recipes_no_proportions.json \
       --output-dir stage2_recipe_no_proportions/model_outputs \
       --models gpt-5.4-mini \
       --reasoning-efforts medium \
       --response-format json_object \
       --temperature 0.0
     ```

4. Evaluate the results.
   - Reuse the Stage 1 evaluator from
     `stage1_recipe_proportions/evaluate_model_outputs.py`.
   - Point it at the Stage 2 recipe file and Stage 2 model outputs.
   - Example command:
     ```bash
     python3 stage1_recipe_proportions/evaluate_model_outputs.py \
       --recipes stage2_recipe_no_proportions/recipes_no_proportions.json \
       --model-outputs-dir stage2_recipe_no_proportions/model_outputs \
       --results-dir stage2_recipe_no_proportions/evaluation_results
     ```

5. Add few-shot and persona experiments.
   - Keep the baseline prompt as the control.
   - Later populate `few_shot_prompt.txt` and `persona_prompt.txt`.
   - Compare against baseline using the same model and evaluation metrics.

## Evaluation metrics

- Mean absolute error (MAE) for `copper_mg_per_serving`
- Valid prediction rate
- Percentage of predictions within +/-20% of ground truth
- Per-recipe error breakdown

## Notes

- The baseline should measure how well the model performs with minimal
  ingredient information.
- Few-shot and persona prompting are separate experiments; leave their templates
  blank until the next development step.
- The Stage 1 evaluator already handles model JSONL outputs and computes MAE,
  so reuse it for Stage 2 where possible.
