# Stage 2 Implementation Summary

Implemented the Stage 2 no-proportions setup without modifying `README.md`.

## Added

- `recipes_no_proportions.json`: generated Stage 2 dataset with ingredient names only, serving sizes, recipe names, and copper-per-serving ground truth.
- `generate_recipes_no_proportions.py`: reproducible generator that derives the Stage 2 dataset from the root `recipes.json`.
- `run_stage2_baseline.py`: local baseline runner that uses `prompt_templates/baseline_prompt.txt` with `gpt-5.4-mini` and medium reasoning by default.
- `run_stage2_persona.py`: local persona-prompt runner that uses `prompt_templates/persona_prompt.txt` and writes `persona__*.jsonl` outputs without overwriting the baseline.
- `run_stage2_few_shot.py`: local few-shot prompt runner that uses `prompt_templates/few_shot_prompt.txt` and writes `few_shot__*.jsonl` outputs without overwriting other experiments.
- `run_stage2_cot.py`: local chain-of-thought prompt runner that uses `prompt_templates/cot_prompt.txt`, requests text output, and writes `cot__*.jsonl` outputs so explicit reasoning plus final JSON can be captured.
- `evaluate_stage2_outputs.py`: local evaluator wrapper that points the Stage 1 evaluator at Stage 2 inputs and output directories.
- `evaluate_stage2_prompt_specific_outputs.py`: local evaluator that scores baseline/persona on all 26 recipes while excluding few-shot prompt examples only from few-shot model scoring.

## Notes

- `few_shot_prompt.txt` and `persona_prompt.txt` remain blank for later tuning.
- The dataset includes both `recipe_name` and `name`: `recipe_name` follows the Stage 2 plan, while `name` preserves compatibility with the existing Stage 1 evaluator.
- Baseline outputs are expected under `model_outputs/`.
- Evaluation outputs are expected under `evaluation_results/`.


## Results
Baseline metrics:

  - MAE: 0.069231
  - Accuracy within +/-10%: 19.23%
  - Valid predictions: 26/26
  
Mean signed error (prediction - ground truth):

  - Baseline: +0.100846 mg per serving, 26 recipes
  - Persona: +0.103385 mg per serving, 26 recipes
  - Few-shot: +0.081304 mg per serving, 23 recipes
