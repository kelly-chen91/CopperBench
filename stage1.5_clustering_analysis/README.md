# CoT Analysis & Failure Mode Clustering

## Implementation Status
This directory now contains an executable clustering pipeline in `analyze_clusters.py` plus generated analysis artifacts.

## Methodology
- Parsed 335 valid trial results from `model_outputs/*.jsonl` with non-empty `reasoning_summary_text`.
- Clustered the top 25% by absolute error: 84 predictions at or above 0.100 mg.
- Extracted lexical, calculation, confidence, ingredient-coverage, and method-indicator features from `reasoning_summary_text` plus `raw_output`.
- Clustered standardized feature vectors with `hdbscan`.
- Applied manual cluster merges after HDBSCAN: [{"source_cluster_id": 1, "target_cluster_id": 0}].
- Computed error magnitude and signed error post hoc; signed error is not used as a clustering feature.
- Labeled clusters with OpenAI when `OPENAI_API_KEY` is available, otherwise with deterministic heuristic labels.

## Cluster Quality
- Non-noise clusters: 2
- Noise points: 50
- Silhouette score: 0.49504607817576474

## Most Frequent Failure/Reasoning Modes
- Cluster -1 (50 predictions): Systematic ingredient copper overestimation | avg MAE 0.200 mg | direction over
- Cluster 0 (22 predictions): Incorrect copper values for snack ingredients | avg MAE 0.145 mg | direction mixed
- Cluster 2 (12 predictions): Systematic copper overestimation via weight-based method | avg MAE 0.942 mg | direction over

## Highest Error Clusters
- Cluster 2: Systematic copper overestimation via weight-based method | avg MAE 0.942 mg | avg relative error 2026.7% | worst model gpt_5_4_nano__reasoning_low
- Cluster -1: Systematic ingredient copper overestimation | avg MAE 0.200 mg | avg relative error 268.3% | worst model gpt_5_5__reasoning_high
- Cluster 0: Incorrect copper values for snack ingredients | avg MAE 0.145 mg | avg relative error 95.6% | worst model gpt_5_4_nano__reasoning_medium

## Output Artifacts
- `features_extracted.csv` contains one row per top-absolute-error reasoning-summary prediction with extracted features and error metrics.
- `clusters_with_labels.csv` adds `cluster_id` to each prediction row.
- `failure_mode_labels.json` maps each cluster to a label, description, label source, and representative samples.
- `error_cluster_analysis.csv` summarizes error, recipe type, model diversity, and over/under-estimation direction by cluster.
- `visualizations/` contains cluster/error plots. PNG files are written when matplotlib is installed; SVG fallbacks are written otherwise.

## Actionable Recommendations
- Inspect high-MAE clusters first; they identify reasoning styles most associated with large copper-estimation misses.
- Compare `ingredient_coverage_ratio` and `grouping_marker_count` against MAE to find recipes where grouped negligible assumptions hide meaningful copper sources.
- Use `error_direction` to separate reasoning-pattern fixes from calibration fixes; directional clusters suggest systematic numeric bias after the reasoning mode is identified.

## Reproduce
```bash
uv run python analyze_clusters.py
```
