## Plan: CoT Analysis & Failure Mode Clustering

### Overview
Analyze Chain-of-Thought (CoT) reasoning across all 17 models to identify failure modes using HDBSCAN clustering. Extract custom features from raw_output, cluster globally, label with GPT, then cross-check clusters against error magnitudes (absolute + relative).

### Design Decisions
- **Clustering Scope**: Global clustering across all models + recipes (identify universal failure patterns)
- **Feature Extraction**: Custom NLP features extracted from CoT text (reasoning depth, confidence signals, ingredient coverage, calculation method, systematic over/underestimators)
- **Clustering Algorithm**: HDBSCAN with auto-detection (data-driven cluster count, no manual tuning)
- **Error Metrics**: Both absolute error (MAE in mg) and relative error (%) for comprehensive cross-validation
- **LLM Labeling**: OpenAI GPT API (consistent with run_copper_estimates.py)

### Data Sources
- `model_outputs/*.jsonl` — raw_output (CoT text), parsed_output (predictions)
- `stage1_recipe_proportions/evaluation_results/` — per_recipe_predictions.csv (error magnitudes)
- `recipes.json` — ground truth copper values + recipe metadata

---

## Implementation Phases

### Phase 1: Feature Extraction from raw_output
1. Load all model_outputs/*.jsonl files and extract trial_result records (~450+ total)
2. For each CoT response (raw_output), extract custom features:
   - **Reasoning Depth**: word count, paragraph count, number of refinement steps/corrections
   - **Confidence Signals**: count USDA references, count uncertainty markers ("I estimate", "roughly", "approximately"), count correction phrases
   - **Ingredient Coverage**: % of recipe ingredients with explicit individual estimates vs. grouped/skipped
   - **Calculation Method**: text patterns (weight-based "per 100g" vs. portion-based, USDA lookup vs. approximation)
   - **Recipe Complexity Response**: how model handles multi-ingredient recipes
   - **Systematic overestimators**: models that consistently include optional ingredients or use high copper reference values
   - **Systematic underestimators**: models that miss ingredients or use low reference values
3. Create `features_extracted.csv` with: recipe_name, model_name, predicted_copper_mg, ground_truth_copper_mg, absolute_error, relative_error_percent, and all extracted features

### Phase 2: HDBSCAN Clustering
4. Normalize feature vectors using StandardScaler
5. Run HDBSCAN with:
   - `min_cluster_size` tuned based on data (~5-10)
   - `metric='euclidean'` or `'manhattan'`
   - Let algorithm auto-detect cluster count
6. Assign cluster labels; flag noise points (label = -1)
7. Create `clusters_with_labels.csv` with cluster_id column added
8. Calculate cluster statistics:
   - Cluster size (# predictions)
   - Average error magnitude per cluster (absolute + relative %)
   - Model diversity per cluster
   - Recipe type distribution

### Phase 3: LLM-based Cluster Labeling
9. For each cluster:
   - Sample 3-5 representative CoT responses (median error + extremes)
   - Create prompt for GPT: "What failure mode or reasoning pattern do these CoT samples share?"
   - Call OpenAI GPT to generate failure mode label + description
10. Create `failure_mode_labels.json` mapping cluster_id → {label, description, representative_samples}

### Phase 4: Error-Failure Mode Cross-Check Analysis
11. Calculate correlation between failure modes and error magnitudes:
    - Which clusters have highest/lowest MAE?
    - Which clusters have highest/lowest relative error %?
    - Do certain failure modes produce systematic under/over-estimation?
12. Analyze error patterns by recipe type within each cluster
13. Create `error_cluster_analysis.csv` with: cluster_id, failure_mode_label, avg_absolute_error, avg_relative_error_percent, best/worst_performing_model, recipe_type_distribution

### Phase 5: Visualization & Insights
14. Generate matplotlib visualizations:
    - Scatter plot: Feature space colored by cluster (PCA/t-SNE dimensionality reduction)
    - Bar chart: Average error (absolute + relative %) per failure mode
    - Bar chart: Cluster sizes + noise point count
    - Heatmap: Failure mode vs. recipe type (error intensity)
    - Box plot: Error distribution per cluster
15. Save visualizations to `stage1.5_clustering_analysis/visualizations/`

### Phase 6: Documentation & Insights Report
16. Write insights section:
    - Top 3-5 failure modes (by frequency + error severity)
    - Which models appear most/least in each failure mode cluster
    - Actionable recommendations for model improvement
17. Update README.md with methodology, findings, and recommendations

---

## Verification Checklist
- [ ] All JSONL files parsed; ~450+ trial_result records extracted
- [ ] Feature extraction: verify word count, marker counts, coverage % are reasonable (spot-check 5 samples)
- [ ] HDBSCAN produces 3-10 clusters (reasonable separation, not all noise)
- [ ] Silhouette score computed; confirm acceptable cluster quality
- [ ] 3-5 sample CoT responses per cluster labeled via GPT API
- [ ] error_cluster_analysis.csv shows meaningful correlation between failure modes and error magnitudes
- [ ] Visualizations render correctly with all clusters visible

---

## Key Questions & Edge Cases
1. **Noise points**: How to handle predictions HDBSCAN marks as noise (-1 cluster)? Options: (a) treat as anomalies, (b) assign to nearest cluster, (c) exclude from analysis
2. **Feature normalization**: Does StandardScaler preserve important differences? Test with/without normalization
3. **Semantic coherence**: Verify that clusters actually share reasoning patterns (spot-check), not just mathematical proximity
4. **LLM label consistency**: Ensure consistent labeling across clusters via fixed GPT model + temperature + confidence ratings
5. **Per-model analysis**: Should we also analyze which failure modes are model-specific vs. universal across all models?

---

## Output Artifacts
- `features_extracted.csv` — all predictions with extracted features
- `clusters_with_labels.csv` — cluster assignments
- `failure_mode_labels.json` — failure mode definitions
- `error_cluster_analysis.csv` — error correlation analysis
- `visualizations/` — plots and heatmaps
- `README.md` — final findings and recommendations
