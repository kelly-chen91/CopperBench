# CopperBench

**Evaluating LLM Copper-Per-Serving Estimation for Recipes**

CopperBench measures how well large language models can estimate the copper micronutrient content (mg per serving) of cooking recipes, and tests whether failure-mode analysis combined with targeted prompt engineering can reduce systematic estimation errors.

The project is organized into five experimental stages: a model benchmark (Stage 1), failure-mode clustering (Stage 1.5), an information-ablation prompt screen (Stage 2), a replicated combined-prompt comparison (Stage 2.5), and a replicated mitigation evaluation (Stage 3).

---

## 1. Original Source

This is original work built from scratch for this project. There is no upstream codebase that was forked or modified.

- **Repository:** `<INSERT GITHUB OR DRIVE LINK HERE>`
- **External datasets / APIs used:**
  - OpenAI Chat Completions and Responses APIs (model inference) — https://platform.openai.com/
  - USDA FoodData Central is referenced *inside the prompts* as the canonical copper source the LLM is asked to consult; no FDC API calls are made in the code.
- **Reference / inspiration:** The 26-recipe seed set in `recipes.json` was assembled by hand by the authors using `prompt_template.txt` to normalize recipes into a uniform JSON schema with a ground-truth `copper_per_serving_mg` value.

---

## 2. Files Authored for This Project

Because the project is original, the table below lists the files written for this project rather than files modified from an upstream source. Files are grouped by role.

### 2.1 Top-level (project root)

| File | Role | Key functions / contents |
|---|---|---|
| `recipes.json` | Dataset (26 labeled recipes) | Ground-truth `copper_per_serving_mg`, ingredients, servings |
| `prompt_template.txt` | Recipe-normalization prompt used to build the dataset | — |
| `run_copper_estimates.py` | Stage 1 driver — runs all model × reasoning combinations against `recipes.json` | `parse_args`, `load_recipes`, `build_prompt`, `call_model`, `write_jsonl`, `main` |
| `analyze_copper_predictions.py` | Computes per-model MAE / accuracy on `model_outputs/*.jsonl` | `parse_jsonl`, `compute_metrics`, `main` |
| `rank_models_by_recipe_type.py` | Per-recipe-type model ranking | `aggregate_by_type`, `rank_models`, `main` |
| `rank_models_cost_adjusted.py` | Cost-adjusted model ranking | `compute_cost`, `rank`, `main` |

### 2.2 Stage 1 — Model benchmark with proportions

`stage1_recipe_proportions/`

| File | Role | Key functions |
|---|---|---|
| `evaluate_model_outputs.py` | Stage 1 evaluator — produces `model_ranking.csv`, accuracy plots, MAE heatmap | `flatten_recipes`, `discover_outputs`, `extract_prediction`, `compute_mae`, `compute_relative_accuracy`, `main` |
| `requirements.txt` | Stage 1 Python dependencies | — |

### 2.3 Stage 1.5 — Failure-mode clustering

`stage1.5_clustering_analysis/`

| File | Role | Key functions |
|---|---|---|
| `analyze_clusters.py` | Extracts lexical / calculation / confidence / coverage / method features from `reasoning_summary_text`, runs HDBSCAN, labels clusters | `load_predictions`, `extract_features`, `cluster_with_hdbscan`, `merge_clusters`, `label_clusters_with_llm`, `summarize_clusters`, `write_artifacts`, `main` |
| `requirements.txt` | Stage 1.5 dependencies (hdbscan, scikit-learn, etc.) | — |

### 2.4 Stage 2 — No-proportions prompt screening

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

### 2.5 Stage 2.5 — Combined prompt with replicates

`stage2.5_combined_prompts/`

| File | Role | Key functions |
|---|---|---|
| `run_stage2_5_combined_persona_few_shot.py` | Runs the combined persona + few-shot prompt | `run_for_recipe`, `main` |
| `run_stage2_replicates.py` | Drives 3 replicate runs across all Stage 2 + 2.5 prompts | `run_replicate`, `main` |
| `evaluate_stage2_5_outputs_20_percent.py` | Per-replicate ±20% evaluation | `evaluate_predictions_20`, `main` |
| `plot_replicate_average_evaluation.py` | Aggregates and plots replicate-averaged results | `aggregate_replicates`, `plot`, `main` |

### 2.6 Stage 3 — Mitigation with proportions and replicates

`stage3_recipe_proportions/`

| File | Role | Key functions |
|---|---|---|
| `run_stage3_persona.py` | Runs Stage 3 persona prompt with proportions | `run_for_recipe`, `main` |
| `run_stage3_combined_persona_few_shot.py` | Runs Stage 3 combined prompt | `run_for_recipe`, `main` |
| `run_stage3_replicates.py` | Drives the 3 replicate runs | `run_replicate`, `main` |
| `plot_stage3_cluster_mitigation_mae.py` | Plots MAE per Stage 1.5 failure cluster, baseline vs persona vs combined | `plot_cluster_mae`, `main` |

### 2.7 Presentation artifacts

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

# Evaluate (produces model_ranking.csv, MAE heatmap, accuracy plot)
python3 stage1_recipe_proportions/evaluate_model_outputs.py
```

### 3.2 Stage 1.5 — Failure-mode clustering

```bash
# Cluster the top-25% absolute-error predictions by reasoning-summary features
python3 stage1.5_clustering_analysis/analyze_clusters.py
```

Outputs: `clusters_with_labels.csv`, `failure_mode_labels.json`, `error_cluster_analysis.csv`, `visualizations/*.png`.

### 3.3 Stage 2 — No-proportions prompt screening

```bash
# Build the no-proportions dataset from recipes.json
python3 stage2_recipe_no_proportions/generate_recipes_no_proportions.py

# Run each of the four prompt variants against gpt-5.4-mini (medium reasoning)
python3 stage2_recipe_no_proportions/run_stage2_baseline.py
python3 stage2_recipe_no_proportions/run_stage2_persona.py
python3 stage2_recipe_no_proportions/run_stage2_few_shot.py
python3 stage2_recipe_no_proportions/run_stage2_cot.py

# Evaluate
python3 stage2_recipe_no_proportions/evaluate_stage2_outputs.py
python3 stage2_recipe_no_proportions/evaluate_stage2_prompt_specific_outputs_20_percent.py

# Plot
python3 stage2_recipe_no_proportions/plot_stage2_mean_signed_error.py
python3 stage2_recipe_no_proportions/plot_stage2_signed_error_by_failure_mode.py
```

### 3.4 Stage 2.5 — Combined prompt over 3 replicates

```bash
# Run the combined persona + few-shot prompt
python3 stage2.5_combined_prompts/run_stage2_5_combined_persona_few_shot.py

# Run the full 3-replicate sweep across Stage 2 + 2.5 prompts
python3 stage2.5_combined_prompts/run_stage2_replicates.py

# Evaluate and plot
python3 stage2.5_combined_prompts/evaluate_stage2_5_outputs_20_percent.py
python3 stage2.5_combined_prompts/plot_replicate_average_evaluation.py
```

### 3.5 Stage 3 — Mitigation with proportions over 3 replicates

```bash
# Single-shot runs of the two surviving prompts
python3 stage3_recipe_proportions/run_stage3_persona.py
python3 stage3_recipe_proportions/run_stage3_combined_persona_few_shot.py

# Full 3-replicate sweep
python3 stage3_recipe_proportions/run_stage3_replicates.py

# Plot per-cluster mitigation MAE
python3 stage3_recipe_proportions/plot_stage3_cluster_mitigation_mae.py
```

### 3.6 Presentation artifacts (optional)

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

**Best-performing model carried forward to Stages 2 / 2.5 / 3:** `gpt-5.4-mini` at medium reasoning effort.

**Data**

| Artifact | Path | Description |
|---|---|---|
| Labeled dataset | `recipes.json` | 26 recipes with hand-curated `copper_per_serving_mg` ground truth |
| No-proportions dataset | `stage2_recipe_no_proportions/recipes_no_proportions.json` | Same 26 recipes with quantities removed |
| Stage 1 raw outputs | `model_outputs/*.jsonl` | One JSONL file per (model, reasoning-effort) pair |
| Stage 1 metrics | `stage1_recipe_proportions/evaluation_results/` | `model_ranking.csv`, plots, per-recipe predictions |
| Stage 1.5 clustering output | `stage1.5_clustering_analysis/{clusters_with_labels.csv, error_cluster_analysis.csv, failure_mode_labels.json}` | |
| Stage 2 / 2.5 replicate outputs | `stage2.5_combined_prompts/replicate_runs/` | 3 runs × all prompt variants |
| Stage 3 replicate outputs | `stage3_recipe_proportions/replicate_runs/` | 3 runs × persona and combined prompts |

---

## 5. Prompts

All prompts used in this project are stored in plain text files in the repository:

| Stage | Prompt | File |
|---|---|---|
| Dataset construction | Recipe-to-JSON normalization | `prompt_template.txt` |
| Stage 1 | Baseline copper-estimation prompt (embedded in driver) | `run_copper_estimates.py` (`PROMPT_TEMPLATE`) |
| Stage 2 | Baseline (no proportions) | `stage2_recipe_no_proportions/prompt_templates/baseline_prompt.txt` |
| Stage 2 | Persona | `stage2_recipe_no_proportions/prompt_templates/persona_prompt.txt` |
| Stage 2 | Few-shot | `stage2_recipe_no_proportions/prompt_templates/few_shot_prompt.txt` |
| Stage 2 | Chain-of-thought | `stage2_recipe_no_proportions/prompt_templates/cot_prompt.txt` |
| Stage 2.5 | Combined persona + few-shot | `stage2.5_combined_prompts/prompt_templates/combined_persona_few_shot.txt` |
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
| `openai` | 1.0 | All `run_*.py` drivers |
| `numpy` | 1.26 | All evaluators |
| `pandas` | 2.0 | Evaluators, ranking scripts |
| `matplotlib` | 3.8 | All plotting scripts |
| `scikit-learn` | 1.4 | Stage 1.5 feature scaling and silhouette |
| `hdbscan` | 0.8.33 | Stage 1.5 clustering |

**Environment variable**

- `OPENAI_API_KEY` must be set for any of the `run_*.py` drivers.

**Install**

Each stage that needs extra packages ships a `requirements.txt`. To install everything in one shot:

```bash
pip install "openai>=1.0" "numpy>=1.26" "pandas>=2.0" "matplotlib>=3.8" "scikit-learn>=1.4" "hdbscan>=0.8.33"
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
├── prompt_template.txt                 # Recipe normalization prompt
├── run_copper_estimates.py             # Stage 1 inference driver
├── analyze_copper_predictions.py       # Quick metrics
├── rank_models_*.py                    # Model ranking scripts
├── model_outputs/                      # Stage 1 raw outputs
├── stage1_recipe_proportions/          # Stage 1 evaluator + results
├── stage1.5_clustering_analysis/       # Failure-mode clustering
├── stage2_recipe_no_proportions/       # No-proportions screening
├── stage2.5_combined_prompts/          # Combined-prompt replicates
├── stage3_recipe_proportions/          # Mitigation replicates
└── presentation_artifacts/             # Cross-stage plots
```

---

## Headline Results

| Setting | MAE (mg/serving) | ±20% accuracy |
|---|---:|---:|
| Stage 1 — best model (`gpt-5.4-mini`, low reasoning), with proportions | **0.0496** | — |
| Stage 2 — best prompt (combined persona + few-shot), no proportions, 3-run avg | 0.0880 | 20.51% |
| Stage 3 — combined persona + few-shot, with proportions, 3-run avg | **0.0452** | **47.44%** |

Targeted, failure-mode-informed prompting reduced MAE below the Stage 1 baseline and roughly doubled ±20% accuracy versus the Stage 1 medium-reasoning baseline.
