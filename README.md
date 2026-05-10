# CopperBench

**Evaluating LLM Copper-Per-Serving Estimation for Recipes**

All code for CopperBench is publicly available on GitHub at https://github.com/kelly-chen91/CopperBench. This project did not build on top of any existing code base, so every file was created by the authors.

CopperBench measures how well large language models can estimate the copper micronutrient content (mg per serving) of cooking recipes, and tests whether failure-mode analysis combined with targeted prompt engineering can reduce systematic estimation errors. 

---

## 1. Original Source

This is original work built from scratch for this project. There is no upstream codebase that was forked or modified.

- **Repository:** https://github.com/kelly-chen91/CopperBench
- **External services used:**
  - **OpenAI Chat Completions / Responses API** — https://platform.openai.com/ — used for every model inference call.
- **Dataset construction:** The 26-recipe seed set in `recipes.json` was assembled by hand by the authors using the “The Copper Conscious Cookbook: A guidebook with recipes for eating well with Wilson disease” which can be found at this [link](!https://www.amazon.com/Copper-Conscious-Cookbook-guidebook-recipes/dp/B0G528MJ2W). 

---

## 2. Files Authored for This Project

Because the project is original, the tables below list the files **written for** the project rather than files modified from an upstream source. Files are grouped by stage, with the principal functions that implement each file's role.

### 2.1 Project root

| File | Role | Key functions / contents |
|---|---|---|
| `recipes.json` | Dataset (26 labeled recipes) | Ground-truth `copper_per_serving_mg`, ingredients, servings |
| `prompt_template.txt` | Documentation copy of the Stage 1 inference prompt (the same string is hardcoded as `PROMPT_TEMPLATE` in `run_copper_estimates.py`) | — |
| `run_copper_estimates.py` | Stage 1 inference driver — runs every (model, reasoning-effort) pair against `recipes.json` | `parse_args`, `load_recipes`, `build_prompt`, `call_model`, `write_jsonl`, `main` |
| `analyze_copper_predictions.py` | Lightweight metrics summarizer over `model_outputs/*.jsonl` | `parse_jsonl`, `compute_metrics`, `main` |
| `rank_models_by_recipe_type.py` | Per-recipe-type ranking | `aggregate_by_type`, `rank_models`, `main` |
| `rank_models_cost_adjusted.py` | Cost-adjusted ranking | `compute_cost`, `rank`, `main` |

### 2.2 Stage 1 — Model benchmark with proportions

`stage1_recipe_proportions/`

| File | Role | Key functions |
|---|---|---|
| `evaluate_model_outputs.py` | Stage 1 evaluator — produces `model_ranking.csv`, MAE bar chart, accuracy chart, MAE-by-recipe-type heatmap, reasoning-effort plot | `flatten_recipes`, `discover_outputs`, `extract_prediction`, `compute_mae`, `compute_relative_accuracy`, `plot_mae_comparison`, `plot_accuracy_comparison`, `plot_recipe_type_heatmap`, `plot_reasoning_effort_effect`, `main` |
| `prompt_templates/baseline.txt` | Stage 1 inference prompt (mirrors the `PROMPT_TEMPLATE` constant in `run_copper_estimates.py`) | — |
| `requirements.txt` | Stage 1 Python dependencies | — |

### 2.3 Stage 1.5 — Failure-mode clustering

`stage1.5_clustering_analysis/`

| File | Role | Key functions |
|---|---|---|
| `analyze_clusters.py` | Extracts lexical / calculation / confidence / coverage / method features from `reasoning_summary_text`, runs HDBSCAN on standardized features, optionally LLM-labels clusters, writes per-cluster artifacts | `load_predictions`, `extract_features`, `cluster_with_hdbscan`, `merge_clusters`, `label_clusters_with_llm`, `summarize_clusters`, `write_artifacts`, `main` |
| `requirements.txt` | Stage 1.5 dependencies (`hdbscan`, `scikit-learn`, …) | — |

### 2.4 Stage 2 — Prompt Mitigation Evaluation Without Proportions

Stage 2 screens five candidate prompts (baseline, persona, few-shot, chain-of-thought, and combined persona + few-shot) against the no-proportions dataset and runs the full set across three replicates for a stable ranking.

`stage2_recipe_no_proportions/`

| File | Role | Key functions |
|---|---|---|
| `generate_recipes_no_proportions.py` | Builds `recipes_no_proportions.json` by stripping quantities from `recipes.json` | `strip_proportions`, `main` |
| `run_stage2_baseline.py` | Runs baseline prompt against no-proportions recipes | `run_for_recipe`, `main` |
| `run_stage2_persona.py` | Runs persona prompt | `run_for_recipe`, `main` |
| `run_stage2_few_shot.py` | Runs few-shot prompt | `run_for_recipe`, `main` |
| `run_stage2_cot.py` | Runs chain-of-thought prompt | `run_for_recipe`, `main` |
| `evaluate_stage2_outputs.py` | Stage 2 evaluator | `evaluate_predictions`, `main` |
| `evaluate_stage2_prompt_specific_outputs_20_percent.py` | ±20% accuracy evaluation per prompt variant | `evaluate_predictions_20`, `main` |
| `plot_stage2_mean_signed_error.py` | Plots mean signed error | `plot`, `main` |
| `plot_stage2_signed_error_by_failure_mode.py` | Plots signed error stratified by Stage 1.5 cluster | `plot_by_cluster`, `main` |

`stage2.5_combined_prompts/`

| File | Role | Key functions |
|---|---|---|
| `run_stage2_5_combined_persona_few_shot.py` | Runs the combined persona + few-shot prompt | `run_for_recipe`, `main` |
| `run_stage2_replicates.py` | Drives 3 replicate runs across all five Stage 2 prompts | `run_replicate`, `main` |
| `evaluate_stage2_5_outputs_20_percent.py` | Per-replicate ±20% evaluation | `evaluate_predictions_20`, `main` |
| `plot_replicate_average_evaluation.py` | Aggregates and plots replicate-averaged results — emits one PNG per metric (MAE, mean signed error, ±20% accuracy) | `load_average_rows`, `averaged_model_ranking_rows`, `plot_bar_with_error`, `plot_run_trends`, `aggregate_recipe_type_rows`, `plot_recipe_type_heatmap`, `main` |

### 2.5 Stage 3 — Mitigation with proportions and replicates

`stage3_recipe_proportions/`

| File | Role | Key functions |
|---|---|---|
| `run_stage3_persona.py` | Runs Stage 3 persona prompt with proportions | `run_for_recipe`, `main` |
| `run_stage3_combined_persona_few_shot.py` | Runs Stage 3 combined prompt with proportions | `run_for_recipe`, `main` |
| `run_stage3_replicates.py` | Drives the 3 replicate runs and aggregates 10% / 20% metrics | `run_replicate`, `aggregate_metrics`, `main` |
| `plot_stage3_cluster_mitigation_mae.py` | Plots MAE per Stage 1.5 failure cluster, Baseline vs Persona vs Combined | `plot_cluster_mae`, `main` |
| `plot_baseline_stage3_comparison.py` | Emits separate single-metric PNGs (MAE, ±10% accuracy, ±20% accuracy) for the Stage 1 baseline vs Stage 3 prompts | `plot_single`, `main` |

### 2.6 Presentation artifacts (optional)

These scripts only produce figures and summary tables for the slide deck — they are not required to reproduce the experimental results.

`presentation_artifacts/`

| File | Role |
|---|---|
| `generate_presentation_artifacts.py` | Builds `stage_progression_mae.png`, `prediction_vs_ground_truth_by_stage.png`, and the summary markdown table |
| `generate_key_results_graph.py` | Builds the per-stage key-results PNGs (`key_results_stage1_accuracy.png`, `key_results_stage1_mae.png`, `key_results_stage2_mae.png`, `key_results_stage3_mae.png`) — each on its own figure so MAE and accuracy are never juxtaposed |

---

## 3. Commands to Train and Test the Baseline and the Built Systems

No model fine-tuning is performed — every model in this project is a frozen OpenAI API model. "Training the baseline" is therefore replaced by **running the prompt against the API**, and "testing" is **running the evaluator**.

All commands assume the repository root is the working directory and `OPENAI_API_KEY` is exported.

### 3.1 Stage 1 — Model benchmark with proportions

```bash
# Run all 11 models × 3 reasoning levels against the labeled recipes
python3 run_copper_estimates.py \
  --input recipes.json \
  --output-dir model_outputs

# Evaluate (produces model_ranking.csv, MAE chart, accuracy chart, heatmap)
python3 stage1_recipe_proportions/evaluate_model_outputs.py
```

### 3.2 Stage 1.5 — Failure-mode clustering

```bash
# Cluster the top-25% absolute-error predictions by reasoning-summary features
python3 stage1.5_clustering_analysis/analyze_clusters.py
```

Outputs: `clusters_with_labels.csv`, `failure_mode_labels.json`, `error_cluster_analysis.csv`, `visualizations/*.png`.

### 3.3 Stage 2 — Prompt Mitigation Evaluation Without Proportions

```bash
# 1. Build the no-proportions dataset from recipes.json
python3 stage2_recipe_no_proportions/generate_recipes_no_proportions.py

# 2. Single-shot runs of the five candidate prompts (against gpt-5.4-mini medium reasoning)
python3 stage2_recipe_no_proportions/run_stage2_baseline.py
python3 stage2_recipe_no_proportions/run_stage2_persona.py
python3 stage2_recipe_no_proportions/run_stage2_few_shot.py
python3 stage2_recipe_no_proportions/run_stage2_cot.py
python3 stage2.5_combined_prompts/run_stage2_5_combined_persona_few_shot.py

# 3. Single-shot evaluation + plots
python3 stage2_recipe_no_proportions/evaluate_stage2_outputs.py
python3 stage2_recipe_no_proportions/evaluate_stage2_prompt_specific_outputs_20_percent.py
python3 stage2_recipe_no_proportions/plot_stage2_mean_signed_error.py
python3 stage2_recipe_no_proportions/plot_stage2_signed_error_by_failure_mode.py

# 4. Full 3-replicate sweep across all five prompts, with per-replicate evaluation and averaged plots
python3 stage2.5_combined_prompts/run_stage2_replicates.py
python3 stage2.5_combined_prompts/evaluate_stage2_5_outputs_20_percent.py
python3 stage2.5_combined_prompts/plot_replicate_average_evaluation.py
```

### 3.4 Stage 3 — Mitigation with proportions over 3 replicates

```bash
# Single-shot runs of the two surviving prompts
python3 stage3_recipe_proportions/run_stage3_persona.py
python3 stage3_recipe_proportions/run_stage3_combined_persona_few_shot.py

# Full 3-replicate sweep with 10% and 20% accuracy metrics
python3 stage3_recipe_proportions/run_stage3_replicates.py

# Plots (one PNG per metric)
python3 stage3_recipe_proportions/plot_stage3_cluster_mitigation_mae.py
python3 stage3_recipe_proportions/plot_baseline_stage3_comparison.py
```

### 3.5 Presentation artifacts (optional)

```bash
python3 presentation_artifacts/generate_presentation_artifacts.py
python3 presentation_artifacts/generate_key_results_graph.py
```

---

## 4. Trained Models and Training Data

**No models were trained.** Every model used in this project is a frozen, hosted OpenAI model accessed by API. There are therefore no trained-model checkpoints to release.

**Models evaluated** (Stage 1):

- `gpt-4.1`, `gpt-4.1-mini`, `gpt-4.1-nano`
- `gpt-4o`, `gpt-4o-mini`
- `gpt-5.4`, `gpt-5.4-mini`, `gpt-5.4-nano`, `gpt-5.4-pro` (each at low / medium / high reasoning effort where supported)
- `gpt-5.5`

**Best-performing model carried forward to Stages 2 and 3:** `gpt-5.4-mini` at medium reasoning effort.

**Data**

| Artifact | Path | Description |
|---|---|---|
| Labeled dataset | `recipes.json` | 26 recipes with hand-curated `copper_per_serving_mg` ground truth |
| No-proportions dataset | `stage2_recipe_no_proportions/recipes_no_proportions.json` | Same 26 recipes with quantities removed |
| Stage 1 raw outputs | `model_outputs/*.jsonl` | One JSONL file per (model, reasoning-effort) pair |
| Stage 1 metrics | `stage1_recipe_proportions/evaluation_results/` | `model_ranking.csv`, plots, per-recipe predictions |
| Stage 1.5 clustering output | `stage1.5_clustering_analysis/{clusters_with_labels.csv, error_cluster_analysis.csv, failure_mode_labels.json}` | Failure-mode clusters and per-cluster summaries |
| Stage 2 replicate outputs | `stage2.5_combined_prompts/replicate_runs/` | 3 runs × all prompt variants |
| Stage 3 replicate outputs | `stage3_recipe_proportions/replicate_runs/` | 3 runs × persona and combined prompts, with 10% and 20% metric breakdowns |

---

## 5. Prompts

All prompts are stored in plain text files in the repository, organized by stage:

| Stage | Prompt | File |
|---|---|---|
| Stage 1 | Baseline copper-estimation prompt | `stage1_recipe_proportions/prompt_templates/baseline.txt` (also mirrored at the repo root as `prompt_template.txt` for convenience, and embedded as the `PROMPT_TEMPLATE` constant in `run_copper_estimates.py`) |
| Stage 2 | Baseline (no proportions) | `stage2_recipe_no_proportions/prompt_templates/baseline_prompt.txt` |
| Stage 2 | Persona | `stage2_recipe_no_proportions/prompt_templates/persona_prompt.txt` |
| Stage 2 | Few-shot | `stage2_recipe_no_proportions/prompt_templates/few_shot_prompt.txt` |
| Stage 2 | Chain-of-thought | `stage2_recipe_no_proportions/prompt_templates/cot_prompt.txt` |
| Stage 2 | Combined persona + few-shot | `stage2.5_combined_prompts/prompt_templates/combined_persona_few_shot.txt` |
| Stage 3 | Persona (with-proportions variant) | `stage3_recipe_proportions/prompt_templates/persona_prompt.txt` |
| Stage 3 | Combined persona + few-shot (with-proportions variant) | `stage3_recipe_proportions/prompt_templates/combined_persona_few_shot_prompt.txt` |

All prompt templates use Python `str.format` placeholders: `{recipe_name}`, `{ingredients}`, `{servings}`.

---

## 6. Software Requirements

**Language and runtime**

- Python 3.10 or newer
- macOS, Linux, or WSL

**Core Python packages**

| Package | Minimum version | Used by |
|---|---|---|
| `openai` | 1.0 | All scripts that generate responses |
| `numpy` | 1.26 | All evaluators |
| `pandas` | 2.0 | Evaluators, ranking scripts |
| `matplotlib` | 3.8 | All plotting scripts |
| `scikit-learn` | 1.4 | Stage 1.5 feature scaling and silhouette |
| `hdbscan` | 0.8.33 | Stage 1.5 clustering |

**Environment variable**

- `OPENAI_API_KEY` must be set before replication of this project, which is the API key generated by the ChatGPT API dashboard.

**Install**

Each stage that needs extra packages ships a `requirements.txt`. To install everything in one shot:

```bash
pip install "openai>=1.0" "numpy>=1.26" "pandas>=2.0" \
            "matplotlib>=3.8" "scikit-learn>=1.4" "hdbscan>=0.8.33"
```

Or per stage:

```bash
pip install -r stage1_recipe_proportions/requirements.txt
pip install -r stage1.5_clustering_analysis/requirements.txt
```

**No GPU is required.** All inference is done via the OpenAI API; clustering and evaluation are CPU-only.

---

## Repository Layout

```
CopperBench/
├── recipes.json                        # Labeled dataset (26 recipes)
├── prompt_template.txt                 # Stage 1 inference prompt (documentation copy)
├── run_copper_estimates.py             # Stage 1 inference driver
├── analyze_copper_predictions.py       # Quick metrics
├── rank_models_*.py                    # Model ranking scripts
├── model_outputs/                      # Stage 1 raw outputs (per-model JSONL)
├── stage1_recipe_proportions/          # Stage 1 evaluator + results
├── stage1.5_clustering_analysis/       # Failure-mode clustering
├── stage2_recipe_no_proportions/       # Stage 2 — no-proportions prompt screening
├── stage2.5_combined_prompts/          # Stage 2 — combined prompt and 3-run replicates
├── stage3_recipe_proportions/          # Mitigation replicates
└── presentation_artifacts/             # Cross-stage plots and summaries
```

---

## Results

In Stage 1, we are able to find that GPT 5.4 mini medium reasoning had the best accuracy and the second lowest MAE. Compared to the other model that had the lowest MAE, 5.4 mini low reasoning, its accuracy was 15.4% compared to the 26.9% of the medium reasoning version. This gap was significant enough for us to move onto Stage 2 using this model. After analysing the first round for failure modes, the main failure modes pointed towards overestimation, as each cluster refers to a type of overestimation (overestimating trace copper amounts, overestimating because of density, and regular ingredient overestimation). Next in Stage 2, we determined the top two mitigation strategies across our evaluation metrics of accuracy, mean absolute error, and mean signed error, which were persona prompting and persona + few-shot prompting. We then ran Stage 3 with these prompts, resulting in MAE being the lowest with persona + few-shot. In accuracy with a 10% threshold, baseline did the best; however, when we switch our threshold over to 20%, persona + few-shot does better by a significant margin.

**Stage 3 head-to-head (with proportions, persona + combined are 3-run averages):**

| Prompt | MAE (mg/serving) — lower is better | ±10% accuracy — higher is better | ±20% accuracy — higher is better |
|---|---:|---:|---:|
| Baseline (Stage 1 reference) | 0.0550 | **26.9%** | 42.3% |
| Persona | 0.0461 | 24.4% | 44.9% |
| Combined persona + few-shot | **0.0452** | 20.5% | **47.4%** |

Taken together, these results demonstrate that failure-mode-informed prompting produces a measurable improvement in LLM copper-per-serving estimation. The combined persona + few-shot prompt reduces MAE by roughly 18% relative to the Stage 1 medium-reasoning baseline and lifts ±20% accuracy by 5.1 percentage points, while Persona alone is essentially tied with Combined and still well ahead of Baseline. However more work needs to be done! 