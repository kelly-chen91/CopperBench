# Stage 1.5 Clustering Analysis Results

## Run Summary

Command run from repository root:

```bash
stage1_recipe_proportions/.venv/bin/python3 stage1.5_clustering_analysis/analyze_cot_clusters.py
```

- Parsed trial records: 476
- Feature rows generated: 476
- Model output files: 18
- Recipes covered: 26
- Clustering method: HDBSCAN
- HDBSCAN parameters: `min_cluster_size=8`, `min_samples=4`
- Non-noise clusters: 26
- Noise points: 137
- Silhouette score: 0.7682
- Cluster label source: OpenAI `gpt-4.1-mini`

The run used the intended HDBSCAN path and OpenAI-based cluster labeling path. The local label parser was updated to handle JSON returned inside Markdown code fences, which is how the OpenAI API responded even when strict JSON was requested.

## Overall Error Pattern

Of the 467 rows with numeric signed error:

- Overestimates: 337
- Underestimates: 113
- Exact matches: 17
- Overall mean absolute error: 0.1074 mg copper per serving
- Overall mean relative error: 146.5%

The main cross-cluster pattern is systematic overestimation. OpenAI labels identified 18 of 27 cluster groups, including noise, as some form of copper overestimation or ingredient copper inflation. Six clusters were labeled as systematic underestimation.

## Highest-Severity Clusters

| Cluster | Label | Size | Avg Abs Error | Avg Rel Error | Direction | Recipe Distribution |
|---:|---|---:|---:|---:|---|---|
| 14 | systematic overestimation bias | 11 | 0.8964 mg | 2240.9% | 100% over | anytime_meals: 11 |
| 19 | systematic overestimation bias | 10 | 0.2720 mg | 302.2% | 100% over | anytime_meals: 10 |
| 25 | systematic overestimation bias | 12 | 0.1734 mg | 97.1% | 100% over | anytime_meals: 7; breakfast: 5 |
| 22 | systematic overestimation bias | 10 | 0.1700 mg | 141.7% | 100% over | anytime_meals: 10 |
| 1 | systematic overestimation due to ingredient copper content inflation | 32 | 0.1397 mg | 104.5% | 84.4% over | snacks: 27; breakfast: 3; desserts: 2 |

Cluster 14 is the most severe by a wide margin. It is entirely anytime meals and entirely overestimates, with one extreme model/recipe point driving very high relative error. Cluster 1 is less severe per row but much larger and concentrated in snacks, so it is a better target for broad prompt or evaluation improvements.

## Largest Clusters

| Cluster | Label | Size | Avg Abs Error | Avg Rel Error | Notes |
|---:|---|---:|---:|---:|---|
| -1 | systematic overestimation bias in ingredient copper values | 137 | 0.1054 mg | 134.6% | HDBSCAN noise/outliers; mostly anytime meals |
| 1 | systematic overestimation due to ingredient copper content inflation | 32 | 0.1397 mg | 104.5% | Mostly snacks |
| 21 | systematic overestimation bias | 19 | 0.0577 mg | 36.5% | 100% overestimates |
| 11 | systematic overestimation bias | 18 | 0.0783 mg | 56.0% | 100% overestimates |
| 2 | systematic miscalculation of ingredient copper content | 17 | 0.0200 mg | 15.2% | Low-error cluster despite the label |

Noise points are retained as cluster `-1` for anomaly inspection. They are numerous and still show a strong overestimation tendency: 78.8% overestimates, 18.2% underestimates.

## Lower-Error Clusters

The best low-error clusters are mostly underestimation or near-exact groups:

| Cluster | Label | Size | Avg Abs Error | Avg Rel Error | Direction |
|---:|---|---:|---:|---:|---|
| 5 | systematic underestimation bias | 16 | 0.0144 mg | 13.8% | 100% under |
| 6 | systematic underestimation bias | 16 | 0.0144 mg | 14.7% | 100% under |
| 2 | systematic miscalculation of ingredient copper content | 17 | 0.0200 mg | 15.2% | signed average 0 |

These clusters suggest that mild underestimation is less damaging than the high-copper overestimation pattern in this dataset.

## Findings

1. The dominant failure mode is overestimating copper values, especially in anytime meals and snacks.
2. HDBSCAN separated the feature space cleanly by the engineered CoT features, with a high silhouette score of 0.7682.
3. The largest non-noise error cluster is snack-heavy and appears to inflate ingredient copper content.
4. Anytime meals dominate the most severe overestimation clusters.
5. Low-error clusters often underpredict slightly or balance over/under errors, producing much lower MAE than the overestimation clusters.

## Recommendations

1. Prioritize fixing overestimation behavior before underestimation behavior.
2. Add prompt checks that discourage inflated copper values for low-copper ingredients and require sanity checks against serving-level totals.
3. Add targeted evaluations for anytime meals and snacks, since they dominate the severe clusters.
4. Inspect cluster `14` manually for the extreme anytime-meal outlier before using it to drive broad conclusions.
5. Keep cluster `-1` as an anomaly bucket rather than forcing reassignment; it contains many high-error outlier reasoning traces.

## Output Artifacts

- `features_extracted.csv`: extracted CoT features and error metrics for all predictions
- `clusters_with_labels.csv`: feature rows with HDBSCAN cluster IDs and PCA coordinates
- `failure_mode_labels.json`: OpenAI-generated label, description, confidence, and representative samples per cluster
- `error_cluster_analysis.csv`: cluster-level error statistics, model distribution, and recipe-type distribution
- `clustering_metadata.json`: run metadata, feature columns, HDBSCAN parameters, and standardization stats
- `visualizations/feature_space_clusters.png`
- `visualizations/cluster_sizes.png`
- `visualizations/average_error_by_cluster.png`
- `visualizations/error_distribution_by_cluster.png`
- `visualizations/failure_mode_recipe_type_heatmap.png`
