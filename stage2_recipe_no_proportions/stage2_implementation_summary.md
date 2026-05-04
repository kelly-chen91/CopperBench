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

Mean signed error (prediction - ground truth):

  Baseline: +0.100846  n=26
  Few-shot: +0.097378  n=26
  CoT:      +0.122215  n=26
  Persona:  +0.103385  n=26

  All variants are overestimating on average.

Overall with 10% accuracy threshold:
  - Few-shot: MAE 0.111521, accuracy 7.69%, valid 26/26
  - Baseline: MAE 0.109923, accuracy 11.54%, valid 26/26
  - Persona: MAE 0.124692, accuracy 3.85%, valid 26/26
  - CoT: MAE 0.132081, accuracy 11.54%, valid 26/26

Overall with 20% accuracy threshold: 
  - Few-shot: MAE 0.111521, accuracy 26.92%, valid 26/26
  - Baseline: MAE 0.109923, accuracy 15.38%, valid 26/26
  - Persona: MAE 0.124692, accuracy 26.92%, valid 26/26
  - CoT: MAE 0.132081, accuracy 26.92%, valid 26/26


## Design Decisions
*Few Shot Prompts* Due to the absence of external ground truth recipes, we had to get manually generate few shot examples of failure modes from the recipe dataset.


Low:    true value < 0.2 mg/serving
Medium: 0.2 – 0.8 mg/serving
High:   > 0.8 mg/serving