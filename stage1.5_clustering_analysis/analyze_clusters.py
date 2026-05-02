#!/usr/bin/env python3
"""Cluster copper-estimation reasoning patterns and cross-check error modes.

The preferred path uses HDBSCAN when installed. The script also includes small
deterministic fallbacks so the core artifacts can still be produced in a bare
Python environment.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
import textwrap
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE_DIR = ROOT / "stage1.5_clustering_analysis"
DEFAULT_RECIPES = ROOT / "recipes.json"
DEFAULT_OUTPUTS = ROOT / "model_outputs"
DEFAULT_EVAL = ROOT / "stage1_recipe_proportions" / "evaluation_results" / "per_recipe_predictions.csv"
DEFAULT_VIS_DIR = DEFAULT_STAGE_DIR / "visualizations"
DEFAULT_TOP_ABSOLUTE_ERROR_FRACTION = 0.25
DEFAULT_CLUSTER_MERGES = ((0, 1),)
os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_STAGE_DIR / ".matplotlib_cache"))

UNCERTAINTY_MARKERS = (
    "about",
    "approximately",
    "estimate",
    "estimated",
    "roughly",
    "around",
    "typical",
    "varies",
    "may vary",
    "assume",
    "likely",
)
CORRECTION_MARKERS = (
    "adjust",
    "correction",
    "instead",
    "recalculate",
    "refine",
    "revised",
    "however",
    "but",
)
USDA_MARKERS = ("usda", "nutrition database", "fooddata", "standard nutrition")
GROUPING_MARKERS = ("negligible", "trace", "to taste", "remaining ingredients", "other ingredients")
WEIGHT_MARKERS = ("per 100g", "100 g", "grams", "gram", " g)")
PORTION_MARKERS = ("cup", "tablespoon", "teaspoon", "tbsp", "tsp", "serving", "medium", "large")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipes", type=Path, default=DEFAULT_RECIPES)
    parser.add_argument("--model-outputs-dir", type=Path, default=DEFAULT_OUTPUTS)
    parser.add_argument("--evaluation-csv", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--stage-dir", type=Path, default=DEFAULT_STAGE_DIR)
    parser.add_argument("--min-cluster-size", type=int, default=7)
    parser.add_argument(
        "--top-absolute-error-fraction",
        type=float,
        default=DEFAULT_TOP_ABSOLUTE_ERROR_FRACTION,
        help="Only cluster the highest-error fraction by absolute error after requiring reasoning summaries.",
    )
    parser.add_argument("--skip-llm-labels", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument(
        "--llm-model",
        default=os.environ.get("CLUSTER_LABEL_MODEL", "gpt-4.1-mini"),
        help="OpenAI model for cluster labels when OPENAI_API_KEY is set.",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def iter_recipe_dicts(node: Any):
    if isinstance(node, dict):
        yield node
    elif isinstance(node, list):
        for item in node:
            yield from iter_recipe_dicts(item)


def load_recipes(path: Path) -> dict[int, dict[str, Any]]:
    recipes: dict[int, dict[str, Any]] = {}
    recipe_index = 0
    for group in read_json(path):
        recipe_type = str(group.get("recipe_type", "unknown"))
        for recipe in iter_recipe_dicts(group.get("recipes", [])):
            ingredients = recipe.get("ingredients", [])
            recipes[recipe_index] = {
                "recipe_index": recipe_index,
                "recipe_type": recipe_type,
                "recipe_name": str(recipe.get("name", "")),
                "servings": recipe.get("servings", ""),
                "ingredients": ingredients if isinstance(ingredients, list) else [],
                "ground_truth_copper_mg_per_serving": float(recipe["copper_per_serving_mg"]),
            }
            recipe_index += 1
    return recipes


def model_name_from_path(path: Path) -> str:
    return path.stem


def reasoning_level(config: dict[str, Any] | None, path: Path) -> str:
    if config and config.get("reasoning_effort"):
        return str(config["reasoning_effort"])
    match = re.search(r"__reasoning_([a-z]+)$", path.stem)
    return match.group(1) if match else "none"


def load_eval_rows(path: Path) -> dict[tuple[str, int], dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="") as handle:
        return {
            (row["model_name"], int(row["recipe_index"])): row
            for row in csv.DictReader(handle)
            if row.get("model_name") and row.get("recipe_index")
        }


def extract_prediction(record: dict[str, Any]) -> float | None:
    parsed = record.get("parsed_output")
    if not isinstance(parsed, dict):
        return None
    value = parsed.get("copper_mg_per_serving")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def reasoning_text(record: dict[str, Any]) -> str:
    parts = []
    summary = record.get("reasoning_summary_text")
    raw = record.get("raw_output")
    if isinstance(summary, str) and summary.strip():
        parts.append(summary.strip())
    if isinstance(raw, str) and raw.strip():
        parts.append(raw.strip())
    return "\n\n".join(parts)


def load_trial_rows(
    model_outputs_dir: Path,
    recipes: dict[int, dict[str, Any]],
    eval_rows: dict[tuple[str, int], dict[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(model_outputs_dir.glob("*.jsonl")):
        config: dict[str, Any] | None = None
        latest_by_recipe: dict[int, dict[str, Any]] = {}
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("record_type") == "experiment_config":
                if isinstance(record.get("config"), dict):
                    config = record["config"]
                continue
            if record.get("record_type") != "trial_result":
                continue
            recipe_index = record.get("recipe_index")
            if isinstance(recipe_index, bool) or not isinstance(recipe_index, int):
                continue
            latest_by_recipe[recipe_index] = {
                "record": record,
                "line_number": line_number,
                "reasoning_level": reasoning_level(config, path),
            }

        model_name = model_name_from_path(path)
        for recipe_index, payload in sorted(latest_by_recipe.items()):
            if recipe_index not in recipes:
                continue
            record = payload["record"]
            predicted = extract_prediction(record)
            if predicted is None:
                continue
            recipe = recipes[recipe_index]
            truth = recipe["ground_truth_copper_mg_per_serving"]
            eval_row = eval_rows.get((model_name, recipe_index), {})
            absolute_error = abs(predicted - truth)
            relative_error = (absolute_error / truth * 100) if truth else 0.0
            absolute_error = float(eval_row.get("absolute_error") or absolute_error)
            relative_error = float(eval_row.get("relative_error_percent") or relative_error)
            reasoning_summary = record.get("reasoning_summary_text") or ""
            if not isinstance(reasoning_summary, str) or not reasoning_summary.strip():
                continue
            rows.append(
                {
                    "model_name": model_name,
                    "model_file": path.name,
                    "reasoning_level": payload["reasoning_level"],
                    "recipe_index": recipe_index,
                    "recipe_type": recipe["recipe_type"],
                    "recipe_name": recipe["recipe_name"],
                    "predicted_copper_mg": predicted,
                    "ground_truth_copper_mg": truth,
                    "absolute_error": absolute_error,
                    "relative_error_percent": relative_error,
                    "signed_error": predicted - truth,
                    "raw_output": record.get("raw_output") or "",
                    "reasoning_summary_text": reasoning_summary,
                    "reasoning_text": reasoning_text(record),
                    "ingredients": recipe["ingredients"],
                }
            )
    return rows


def filter_top_absolute_error_rows(
    rows: list[dict[str, Any]],
    top_fraction: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not 0 < top_fraction <= 1:
        raise ValueError("--top-absolute-error-fraction must be > 0 and <= 1")
    if not rows:
        return [], {
            "top_absolute_error_fraction": top_fraction,
            "top_absolute_error_count": 0,
            "absolute_error_cutoff_mg": None,
            "reasoning_summary_candidate_rows": 0,
        }
    target_count = max(1, math.ceil(len(rows) * top_fraction))
    ordered = sorted(rows, key=lambda row: float(row["absolute_error"]), reverse=True)
    cutoff = float(ordered[target_count - 1]["absolute_error"])
    filtered = ordered[:target_count]
    filtered.sort(key=lambda row: (str(row["model_name"]), int(row["recipe_index"])))
    return filtered, {
        "top_absolute_error_fraction": top_fraction,
        "top_absolute_error_count": len(filtered),
        "absolute_error_cutoff_mg": cutoff,
        "reasoning_summary_candidate_rows": len(rows),
    }


def normalize_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def ingredient_terms(ingredient: dict[str, Any]) -> set[str]:
    name = str(ingredient.get("name", "")).lower()
    terms = {
        word
        for word in normalize_words(name)
        if len(word) >= 4
        and word
        not in {
            "with",
            "fresh",
            "chopped",
            "divided",
            "large",
            "small",
            "medium",
            "reduced",
            "optional",
            "taste",
            "thinly",
            "sliced",
            "roughly",
        }
    }
    return terms


def count_markers(text: str, markers: tuple[str, ...]) -> int:
    lowered = text.lower()
    return sum(lowered.count(marker) for marker in markers)


def bool_marker(text: str, markers: tuple[str, ...]) -> int:
    lowered = text.lower()
    return int(any(marker in lowered for marker in markers))


def estimate_individual_ingredient_coverage(text: str, ingredients: list[dict[str, Any]]) -> tuple[int, float]:
    lowered = text.lower()
    covered = 0
    for ingredient in ingredients:
        terms = ingredient_terms(ingredient)
        if not terms:
            continue
        if any(term in lowered for term in terms):
            covered += 1
    total = len(ingredients)
    return covered, (covered / total if total else 0.0)


def extract_features(row: dict[str, Any]) -> dict[str, Any]:
    text = str(row["reasoning_text"])
    words = normalize_words(text)
    paragraphs = [part for part in re.split(r"\n\s*\n+", text.strip()) if part.strip()]
    bullet_lines = [
        line
        for line in text.splitlines()
        if re.match(r"\s*(?:[-*]|\d+[.)]|#{1,6}\s)", line)
    ]
    numeric_mg_mentions = len(re.findall(r"\b\d+(?:\.\d+)?\s*mg\b", text, flags=re.I))
    covered, coverage_ratio = estimate_individual_ingredient_coverage(text, row["ingredients"])
    ingredient_count = len(row["ingredients"])
    coverage_gap = max(ingredient_count - covered, 0)
    features = {
        "word_count": len(words),
        "paragraph_count": len(paragraphs),
        "bullet_line_count": len(bullet_lines),
        "numeric_mg_mentions": numeric_mg_mentions,
        "calculation_operator_count": len(re.findall(r"[+=/×x-]", text)),
        "usda_reference_count": count_markers(text, USDA_MARKERS),
        "uncertainty_marker_count": count_markers(text, UNCERTAINTY_MARKERS),
        "correction_marker_count": count_markers(text, CORRECTION_MARKERS),
        "grouping_marker_count": count_markers(text, GROUPING_MARKERS),
        "ingredient_count": ingredient_count,
        "ingredient_mention_count": covered,
        "ingredient_coverage_ratio": coverage_ratio,
        "ingredient_coverage_gap": coverage_gap,
        "uses_weight_based_method": bool_marker(text, WEIGHT_MARKERS),
        "uses_portion_based_method": bool_marker(text, PORTION_MARKERS),
        "uses_usda_or_database_method": bool_marker(text, USDA_MARKERS),
        "mentions_serving_division": int(bool(re.search(r"\bper serving\b|\bservings?\b|/\s*\d+", text, flags=re.I))),
        "recipe_complexity_response": coverage_ratio * math.log1p(ingredient_count),
        "raw_json_only": int(len(words) < 60 and "{" in text and "}" in text),
    }
    return features


def build_feature_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    numeric_columns = [
        "word_count",
        "paragraph_count",
        "bullet_line_count",
        "numeric_mg_mentions",
        "calculation_operator_count",
        "usda_reference_count",
        "uncertainty_marker_count",
        "correction_marker_count",
        "grouping_marker_count",
        "ingredient_count",
        "ingredient_mention_count",
        "ingredient_coverage_ratio",
        "ingredient_coverage_gap",
        "uses_weight_based_method",
        "uses_portion_based_method",
        "uses_usda_or_database_method",
        "mentions_serving_division",
        "recipe_complexity_response",
        "raw_json_only",
    ]
    feature_rows = []
    for row in rows:
        features = extract_features(row)
        output = {
            key: value
            for key, value in row.items()
            if key not in {"ingredients", "raw_output", "reasoning_summary_text"}
        }
        output.update(features)
        output["raw_output_excerpt"] = compact_text(row["reasoning_text"], 500)
        feature_rows.append(output)
    return feature_rows, numeric_columns


def matrix_from_rows(rows: list[dict[str, Any]], columns: list[str]) -> list[list[float]]:
    return [[float(row[column]) for column in columns] for row in rows]


def standardize(matrix: list[list[float]]) -> list[list[float]]:
    if not matrix:
        return []
    width = len(matrix[0])
    means = [statistics.fmean(row[i] for row in matrix) for i in range(width)]
    stdevs = []
    for i in range(width):
        values = [row[i] for row in matrix]
        stdev = statistics.pstdev(values)
        stdevs.append(stdev if stdev > 0 else 1.0)
    return [[(row[i] - means[i]) / stdevs[i] for i in range(width)] for row in matrix]


def euclidean(left: list[float], right: list[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def fallback_density_cluster(matrix: list[list[float]], min_cluster_size: int) -> tuple[list[int], dict[str, Any]]:
    """A tiny DBSCAN-like fallback used only when hdbscan is unavailable."""
    if not matrix:
        return [], {"algorithm": "fallback_density", "reason": "empty matrix"}
    distances = []
    k = max(2, min(min_cluster_size, len(matrix) - 1))
    for i, point in enumerate(matrix):
        point_distances = sorted(euclidean(point, other) for j, other in enumerate(matrix) if i != j)
        if point_distances:
            distances.append(point_distances[min(k - 1, len(point_distances) - 1)])
    eps = percentile(distances, 55) if distances else 0.0
    eps = eps or 1.0
    labels = [-99] * len(matrix)
    cluster_id = 0
    for i in range(len(matrix)):
        if labels[i] != -99:
            continue
        neighbors = [j for j, point in enumerate(matrix) if euclidean(matrix[i], point) <= eps]
        if len(neighbors) < min_cluster_size:
            labels[i] = -1
            continue
        labels[i] = cluster_id
        seeds = [n for n in neighbors if n != i]
        while seeds:
            current = seeds.pop()
            if labels[current] == -1:
                labels[current] = cluster_id
            if labels[current] != -99:
                continue
            labels[current] = cluster_id
            current_neighbors = [
                j for j, point in enumerate(matrix) if euclidean(matrix[current], point) <= eps
            ]
            if len(current_neighbors) >= min_cluster_size:
                seeds.extend(n for n in current_neighbors if labels[n] in {-99, -1})
        cluster_id += 1
    labels = [-1 if label == -99 else label for label in labels]
    return labels, {"algorithm": "fallback_density", "eps": eps, "min_cluster_size": min_cluster_size}


def cluster_rows(
    feature_rows: list[dict[str, Any]],
    numeric_columns: list[str],
    min_cluster_size: int,
) -> tuple[list[int], dict[str, Any]]:
    scaled = standardize(matrix_from_rows(feature_rows, numeric_columns))
    try:
        import hdbscan  # type: ignore

        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean")
        labels = [int(label) for label in clusterer.fit_predict(scaled)]
        return labels, {
            "algorithm": "hdbscan",
            "min_cluster_size": min_cluster_size,
            "metric": "euclidean",
        }
    except Exception as exc:
        labels, metadata = fallback_density_cluster(scaled, min_cluster_size)
        metadata["hdbscan_unavailable_reason"] = str(exc)
        return labels, metadata


def apply_cluster_merges(
    rows: list[dict[str, Any]],
    labels: list[int],
    merges: tuple[tuple[int, int], ...],
) -> tuple[list[int], list[dict[str, int]]]:
    merge_map = {source: target for target, source in merges}
    if not merge_map:
        return labels, []
    merged_labels = [merge_map.get(label, label) for label in labels]
    for row, cluster_id in zip(rows, merged_labels):
        row["cluster_id"] = cluster_id
    return merged_labels, [
        {"source_cluster_id": source, "target_cluster_id": target}
        for target, source in merges
    ]


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * pct / 100
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[int(index)]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def silhouette_score(matrix: list[list[float]], labels: list[int]) -> float | None:
    cluster_ids = sorted(label for label in set(labels) if label != -1)
    if len(cluster_ids) < 2:
        return None
    scores = []
    for i, point in enumerate(matrix):
        label = labels[i]
        if label == -1:
            continue
        same = [j for j, other_label in enumerate(labels) if other_label == label and j != i]
        if not same:
            continue
        a = statistics.fmean(euclidean(point, matrix[j]) for j in same)
        b_candidates = []
        for other_label in cluster_ids:
            if other_label == label:
                continue
            others = [j for j, candidate_label in enumerate(labels) if candidate_label == other_label]
            if others:
                b_candidates.append(statistics.fmean(euclidean(point, matrix[j]) for j in others))
        if not b_candidates:
            continue
        b = min(b_candidates)
        scores.append((b - a) / max(a, b) if max(a, b) else 0.0)
    return statistics.fmean(scores) if scores else None


def direction_label(rows: list[dict[str, Any]]) -> tuple[str, float, float]:
    over = sum(1 for row in rows if float(row["signed_error"]) > 0)
    under = sum(1 for row in rows if float(row["signed_error"]) < 0)
    total = len(rows) or 1
    over_ratio = over / total
    under_ratio = under / total
    if over_ratio >= 0.85:
        return "over", over_ratio, under_ratio
    if under_ratio >= 0.85:
        return "under", over_ratio, under_ratio
    return "mixed", over_ratio, under_ratio


def summarize_clusters(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["cluster_id"])].append(row)
    summaries = []
    for cluster_id, cluster_rows in sorted(grouped.items()):
        direction, over_ratio, under_ratio = direction_label(cluster_rows)
        recipe_types = Counter(str(row["recipe_type"]) for row in cluster_rows)
        models = Counter(str(row["model_name"]) for row in cluster_rows)
        signed_errors = [float(row["signed_error"]) for row in cluster_rows]
        summaries.append(
            {
                "cluster_id": cluster_id,
                "cluster_size": len(cluster_rows),
                "avg_absolute_error": statistics.fmean(float(row["absolute_error"]) for row in cluster_rows),
                "median_absolute_error": statistics.median(float(row["absolute_error"]) for row in cluster_rows),
                "avg_relative_error_percent": statistics.fmean(
                    float(row["relative_error_percent"]) for row in cluster_rows
                ),
                "median_relative_error_percent": statistics.median(
                    float(row["relative_error_percent"]) for row in cluster_rows
                ),
                "avg_signed_error": statistics.fmean(signed_errors),
                "error_direction": direction,
                "overestimate_ratio": over_ratio,
                "underestimate_ratio": under_ratio,
                "model_diversity": len(models),
                "recipe_type_distribution": json.dumps(dict(sorted(recipe_types.items())), sort_keys=True),
                "top_models": "; ".join(f"{name}:{count}" for name, count in models.most_common(5)),
                "top_recipe_types": "; ".join(f"{name}:{count}" for name, count in recipe_types.most_common()),
            }
        )
    return summaries


def compact_text(text: str, limit: int = 700) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def representative_samples(rows: list[dict[str, Any]], cluster_id: int, limit: int = 5) -> list[dict[str, Any]]:
    cluster_rows = [row for row in rows if int(row["cluster_id"]) == cluster_id]
    if not cluster_rows:
        return []
    errors = [float(row["absolute_error"]) for row in cluster_rows]
    mean_error = statistics.fmean(errors)
    stdev_error = statistics.pstdev(errors) if len(errors) > 1 else 0.0
    if stdev_error > 0:
        lower_bound = mean_error - 2 * stdev_error
        upper_bound = mean_error + 2 * stdev_error
        candidates = [
            row
            for row in cluster_rows
            if lower_bound <= float(row["absolute_error"]) <= upper_bound
        ]
    else:
        candidates = list(cluster_rows)
    if not candidates:
        candidates = list(cluster_rows)

    by_distance_to_mean = sorted(
        candidates,
        key=lambda row: abs(float(row["absolute_error"]) - mean_error),
    )
    picks = []
    seen_recipes = set()
    for row in by_distance_to_mean:
        recipe_name = str(row["recipe_name"])
        if recipe_name in seen_recipes:
            continue
        picks.append(row)
        seen_recipes.add(recipe_name)
        if len(picks) >= limit:
            break
    if len(picks) < limit:
        picks.extend(row for row in by_distance_to_mean if row not in picks)
    unique = []
    seen = set()
    for row in picks:
        key = (row["model_name"], row["recipe_index"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(
            {
                "model_name": row["model_name"],
                "recipe_name": row["recipe_name"],
                "predicted_copper_mg": row["predicted_copper_mg"],
                "ground_truth_copper_mg": row["ground_truth_copper_mg"],
                "absolute_error": row["absolute_error"],
                "relative_error_percent": row["relative_error_percent"],
                "signed_error": row["signed_error"],
                "raw_reasoning_text": row.get("reasoning_text", row["raw_output_excerpt"]),
            }
        )
        if len(unique) >= limit:
            break
    return unique


def heuristic_cluster_label(cluster_rows: list[dict[str, Any]]) -> tuple[str, str]:
    avg = lambda column: statistics.fmean(float(row[column]) for row in cluster_rows)
    raw_json = avg("raw_json_only")
    coverage = avg("ingredient_coverage_ratio")
    uncertainty = avg("uncertainty_marker_count")
    mg_mentions = avg("numeric_mg_mentions")
    grouping = avg("grouping_marker_count")
    usda = avg("usda_reference_count")
    weight = avg("uses_weight_based_method")
    if raw_json >= 0.6:
        return (
            "Final-answer-only estimates",
            "Responses expose little reasoning text, so clustering is driven by sparse JSON/final-answer structure.",
        )
    if coverage >= 0.75 and mg_mentions >= 8:
        return (
            "Detailed ingredient-by-ingredient arithmetic",
            "Responses enumerate most ingredients with numeric copper estimates and explicit summation.",
        )
    if grouping >= 2 or coverage < 0.35:
        return (
            "Grouped or skipped ingredient estimates",
            "Responses collapse several ingredients as negligible or omit many ingredient-specific estimates.",
        )
    if uncertainty >= 5:
        return (
            "High-uncertainty approximation",
            "Responses lean on approximate language and broad assumptions rather than precise lookup-style accounting.",
        )
    if usda > 0 or weight >= 0.5:
        return (
            "Database or weight-normalized lookup reasoning",
            "Responses cite USDA/database-style sources or weight-normalized copper values.",
        )
    return (
        "Portion-based estimation",
        "Responses use common household portions and rough ingredient contributions without a strong database signal.",
    )


def heuristic_label_payload(cluster_rows: list[dict[str, Any]]) -> dict[str, str]:
    label, description = heuristic_cluster_label(cluster_rows)
    direction, _, _ = direction_label(cluster_rows)
    return {
        "label": label,
        "description": description,
        "error_direction": direction,
        "distinguishing_feature": "Deterministic feature summary; rerun with OPENAI_API_KEY for ingredient-specific failure labeling.",
    }


LABEL_FEATURE_COLUMNS = (
    "word_count",
    "paragraph_count",
    "bullet_line_count",
    "numeric_mg_mentions",
    "calculation_operator_count",
    "usda_reference_count",
    "uncertainty_marker_count",
    "correction_marker_count",
    "grouping_marker_count",
    "ingredient_count",
    "ingredient_mention_count",
    "ingredient_coverage_ratio",
    "ingredient_coverage_gap",
    "uses_weight_based_method",
    "uses_portion_based_method",
    "uses_usda_or_database_method",
    "mentions_serving_division",
    "recipe_complexity_response",
    "raw_json_only",
)


def feature_contrast_summary(
    cluster_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    limit: int = 8,
) -> list[dict[str, Any]]:
    contrasts = []
    if not cluster_rows or not comparison_rows:
        return contrasts
    for column in LABEL_FEATURE_COLUMNS:
        cluster_mean = statistics.fmean(float(row[column]) for row in cluster_rows)
        comparison_mean = statistics.fmean(float(row[column]) for row in comparison_rows)
        comparison_values = [float(row[column]) for row in comparison_rows]
        comparison_stdev = statistics.pstdev(comparison_values) or 1.0
        contrasts.append(
            {
                "feature": column,
                "cluster_mean": round(cluster_mean, 3),
                "other_clusters_mean": round(comparison_mean, 3),
                "difference": round(cluster_mean - comparison_mean, 3),
                "z_vs_other_clusters": round((cluster_mean - comparison_mean) / comparison_stdev, 3),
            }
        )
    return sorted(contrasts, key=lambda item: abs(float(item["z_vs_other_clusters"])), reverse=True)[:limit]


def llm_label_cluster(
    cluster_id: int,
    cluster_rows: list[dict[str, Any]],
    all_rows: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    model: str,
) -> tuple[dict[str, str], str]:
    if not os.environ.get("OPENAI_API_KEY"):
        return heuristic_label_payload(cluster_rows), "heuristic_no_openai_api_key"
    try:
        from openai import OpenAI  # type: ignore

        cluster_mae = statistics.fmean(float(row["absolute_error"]) for row in cluster_rows)
        cluster_mape = statistics.fmean(float(row["relative_error_percent"]) for row in cluster_rows)
        error_direction, over_ratio, under_ratio = direction_label(cluster_rows)
        gt_values = [float(row["ground_truth_copper_mg"]) for row in cluster_rows]
        gt_min = min(gt_values)
        gt_max = max(gt_values)
        recipe_types = Counter(str(row["recipe_type"]) for row in cluster_rows)
        recipes = Counter(str(row["recipe_name"]) for row in cluster_rows)
        comparison_rows = [row for row in all_rows if int(row["cluster_id"]) != cluster_id]
        feature_contrasts = feature_contrast_summary(cluster_rows, comparison_rows)
        feature_summary = {
            "mean_word_count": round(statistics.fmean(float(row["word_count"]) for row in cluster_rows), 3),
            "mean_numeric_mg_mentions": round(
                statistics.fmean(float(row["numeric_mg_mentions"]) for row in cluster_rows), 3
            ),
            "mean_uncertainty_marker_count": round(
                statistics.fmean(float(row["uncertainty_marker_count"]) for row in cluster_rows), 3
            ),
            "mean_ingredient_coverage_ratio": round(
                statistics.fmean(float(row["ingredient_coverage_ratio"]) for row in cluster_rows), 3
            ),
            "mean_grouping_marker_count": round(
                statistics.fmean(float(row["grouping_marker_count"]) for row in cluster_rows), 3
            ),
            "uses_weight_based_method_ratio": round(
                statistics.fmean(float(row["uses_weight_based_method"]) for row in cluster_rows), 3
            ),
            "uses_portion_based_method_ratio": round(
                statistics.fmean(float(row["uses_portion_based_method"]) for row in cluster_rows), 3
            ),
            "uses_usda_or_database_method_ratio": round(
                statistics.fmean(float(row["uses_usda_or_database_method"]) for row in cluster_rows), 3
            ),
        }
        representative_samples = json.dumps(samples, indent=2)
        prompt = f"""
These reasoning traces all performed ingredient-by-ingredient copper estimation
but produced large errors. They have already been clustered by feature similarity.

Cluster error summary:
- Mean absolute error: {cluster_mae:.3f} mg
- Mean relative error: {cluster_mape:.1f}%
- Error direction: {error_direction}
- Overestimate ratio: {over_ratio:.2f}
- Underestimate ratio: {under_ratio:.2f}
- Ground truth range: {gt_min:.3f} - {gt_max:.3f} mg/serving
- Recipe type distribution: {json.dumps(dict(recipe_types.most_common()), sort_keys=True)}
- Top recipes: {json.dumps(dict(recipes.most_common(8)), sort_keys=True)}
- Method feature summary: {json.dumps(feature_summary, sort_keys=True)}
- Feature contrasts vs all other clusters: {json.dumps(feature_contrasts, indent=2)}

Your job is to identify what specifically went wrong in THIS cluster - not what
the model did generally, but the specific failure mechanism that caused the error.
These traces were clustered using engineered feature columns from
features_extracted.csv. Use BOTH the feature-column contrasts and the raw
reasoning text when labeling. Do not infer the cluster label from reasoning text
alone. If the raw traces suggest one ingredient type but the feature columns show
the stronger distinction is method-based, recipe-complexity-based, coverage-based,
or database-use-based, prefer the feature-column distinction.

Look for:
- Which specific ingredients had wrong copper reference values
- Whether unit conversions were applied incorrectly (e.g., volume to weight)
- Whether any ingredients were skipped or grouped incorrectly
- Whether the per-serving division step was wrong
- Whether optional ingredients were incorrectly included or excluded
- Whether the model anchored to a round number or dish-level estimate

Do NOT label this cluster "detailed ingredient-by-ingredient arithmetic" -
that describes all clusters. Find what makes THIS cluster's errors distinctive.
Do NOT use generic labels like "incorrect copper values for ingredients" unless
you also name the ingredient class or estimation style that makes the cluster
distinctive. If two clusters both use wrong ingredient reference values, separate
them by the specific ingredient type, recipe family, or method style shown in the
summary and traces, such as grain/flour values, snack toppings, spice/minor
ingredients, portion-based guesses, or weight-based per-100g lookups.
Make the label and description agree with the cluster error direction. If the
error direction is mixed, do not label it as pure overestimation or pure
underestimation unless the description explicitly explains both directions.

Reasoning traces:
{representative_samples}

Return strict JSON:
{{
  "label": "specific failure mechanism in 5 words or fewer",
  "description": "one sentence: what specifically went wrong and why",
  "error_direction": "over / under / mixed",
  "distinguishing_feature": "what makes this cluster different from generic ingredient arithmetic"
}}
""".strip()
        client = OpenAI()
        response = client.responses.create(
            model=model,
            temperature=0,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "cluster_label",
                    "schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "label": {"type": "string"},
                            "description": {"type": "string"},
                            "error_direction": {
                                "type": "string",
                                "enum": ["over", "under", "mixed"],
                            },
                            "distinguishing_feature": {"type": "string"},
                        },
                        "required": [
                            "label",
                            "description",
                            "error_direction",
                            "distinguishing_feature",
                        ],
                    },
                    "strict": True,
                }
            },
            input=[
                {
                    "role": "system",
                    "content": "Identify specific failure mechanisms in copper-estimation reasoning clusters.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        content = response.output_text
        data = parse_json_object(content)
        return {
            "label": str(data["label"]),
            "description": str(data["description"]),
            "error_direction": str(data["error_direction"]),
            "distinguishing_feature": str(data["distinguishing_feature"]),
        }, "openai"
    except Exception as exc:
        payload = heuristic_label_payload(cluster_rows)
        payload["description"] = f"{payload['description']} LLM labeling failed: {exc}"
        return payload, "heuristic_llm_failed"


def parse_json_object(content: str) -> dict[str, Any]:
    content = content.strip()
    if not content:
        raise ValueError("empty model response")
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, flags=re.S)
        if fenced:
            data = json.loads(fenced.group(1))
        else:
            start = content.find("{")
            end = content.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise
            data = json.loads(content[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("model response JSON was not an object")
    return data


def label_clusters(
    rows: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    model: str,
    skip_llm: bool,
) -> dict[str, Any]:
    labels: dict[str, Any] = {}
    for summary in summaries:
        cluster_id = int(summary["cluster_id"])
        cluster_rows = [row for row in rows if int(row["cluster_id"]) == cluster_id]
        samples = representative_samples(rows, cluster_id)
        if skip_llm:
            payload = heuristic_label_payload(cluster_rows)
            source = "heuristic_skip_llm"
        else:
            payload, source = llm_label_cluster(cluster_id, cluster_rows, rows, samples, model)
        labels[str(cluster_id)] = {
            "label": payload["label"],
            "description": payload["description"],
            "error_direction": payload["error_direction"],
            "distinguishing_feature": payload["distinguishing_feature"],
            "label_source": source,
            "representative_samples": samples,
        }
    return labels


def attach_labels_to_analysis(
    summaries: list[dict[str, Any]],
    labels: dict[str, Any],
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for summary in summaries:
        cluster_id = int(summary["cluster_id"])
        label = labels[str(cluster_id)]
        cluster_rows = [row for row in rows if int(row["cluster_id"]) == cluster_id]
        model_errors: dict[str, list[float]] = defaultdict(list)
        for row in cluster_rows:
            model_errors[str(row["model_name"])].append(float(row["absolute_error"]))
        model_mae = {
            model: statistics.fmean(errors)
            for model, errors in model_errors.items()
            if errors
        }
        best_model = min(model_mae, key=model_mae.get) if model_mae else ""
        worst_model = max(model_mae, key=model_mae.get) if model_mae else ""
        row = dict(summary)
        row.update(
            {
                "failure_mode_label": label["label"],
                "failure_mode_description": label["description"],
                "failure_mode_distinguishing_feature": label.get("distinguishing_feature", ""),
                "label_error_direction": label.get("error_direction", ""),
                "best_performing_model": best_model,
                "worst_performing_model": worst_model,
            }
        )
        output.append(row)
    return output


def safe_slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "cluster"


def write_svg_bar_chart(path: Path, title: str, items: list[tuple[str, float]], y_label: str) -> None:
    width, height = 1100, 520
    margin_left, margin_bottom, margin_top, margin_right = 90, 150, 55, 30
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    max_value = max([value for _, value in items] or [1.0]) or 1.0
    bar_width = plot_width / max(len(items), 1)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2}" y="30" text-anchor="middle" font-family="Arial" font-size="20">{escape_xml(title)}</text>',
        f'<text x="20" y="{margin_top + plot_height / 2}" transform="rotate(-90 20 {margin_top + plot_height / 2})" text-anchor="middle" font-family="Arial" font-size="13">{escape_xml(y_label)}</text>',
        f'<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{width - margin_right}" y2="{margin_top + plot_height}" stroke="#333"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#333"/>',
    ]
    for index, (label, value) in enumerate(items):
        x = margin_left + index * bar_width + bar_width * 0.15
        bar_h = (value / max_value) * plot_height
        y = margin_top + plot_height - bar_h
        color = "#4f7cac" if not label.startswith("-1") else "#9a9a9a"
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width * 0.7:.1f}" height="{bar_h:.1f}" fill="{color}"/>')
        parts.append(f'<text x="{x + bar_width * 0.35:.1f}" y="{y - 5:.1f}" text-anchor="middle" font-family="Arial" font-size="11">{value:.2f}</text>')
        parts.append(
            f'<text x="{x + bar_width * 0.35:.1f}" y="{margin_top + plot_height + 18}" '
            f'text-anchor="end" transform="rotate(-45 {x + bar_width * 0.35:.1f} {margin_top + plot_height + 18})" '
            f'font-family="Arial" font-size="11">{escape_xml(label)}</text>'
        )
    parts.append("</svg>\n")
    path.write_text("\n".join(parts))


def write_svg_box_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    items = []
    for cluster_id in sorted({int(row["cluster_id"]) for row in rows}):
        values = sorted(float(row["absolute_error"]) for row in rows if int(row["cluster_id"]) == cluster_id)
        if not values:
            continue
        items.append(
            (
                str(cluster_id),
                percentile(values, 0),
                percentile(values, 25),
                percentile(values, 50),
                percentile(values, 75),
                percentile(values, 100),
            )
        )
    width, height = 900, 520
    margin_left, margin_bottom, margin_top, margin_right = 75, 70, 55, 30
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    max_value = max([item[5] for item in items] or [1.0]) or 1.0
    slot = plot_width / max(len(items), 1)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="450" y="30" text-anchor="middle" font-family="Arial" font-size="20">Absolute Error Distribution by Cluster</text>',
        f'<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="#333"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#333"/>',
        f'<text x="22" y="{height / 2}" transform="rotate(-90 22 {height / 2})" text-anchor="middle" font-family="Arial" font-size="13">Absolute error (mg)</text>',
    ]
    def y_for(value: float) -> float:
        return height - margin_bottom - value / max_value * plot_height

    for index, (label, low, q1, median, q3, high) in enumerate(items):
        x = margin_left + index * slot + slot / 2
        box_width = slot * 0.42
        parts.append(f'<line x1="{x:.1f}" y1="{y_for(low):.1f}" x2="{x:.1f}" y2="{y_for(high):.1f}" stroke="#333"/>')
        parts.append(
            f'<rect x="{x - box_width / 2:.1f}" y="{y_for(q3):.1f}" width="{box_width:.1f}" '
            f'height="{max(y_for(q1) - y_for(q3), 1):.1f}" fill="#b7d0e6" stroke="#333"/>'
        )
        parts.append(f'<line x1="{x - box_width / 2:.1f}" y1="{y_for(median):.1f}" x2="{x + box_width / 2:.1f}" y2="{y_for(median):.1f}" stroke="#c7522a" stroke-width="2"/>')
        parts.append(f'<text x="{x:.1f}" y="{height - margin_bottom + 22}" text-anchor="middle" font-family="Arial" font-size="12">{escape_xml(label)}</text>')
    parts.append("</svg>\n")
    path.write_text("\n".join(parts))


def write_svg_heatmap(path: Path, rows: list[dict[str, Any]]) -> None:
    clusters = sorted({int(row["cluster_id"]) for row in rows})
    recipe_types = sorted({str(row["recipe_type"]) for row in rows})
    values: dict[tuple[int, str], float] = {}
    for cluster_id in clusters:
        for recipe_type in recipe_types:
            subset = [
                float(row["absolute_error"])
                for row in rows
                if int(row["cluster_id"]) == cluster_id and str(row["recipe_type"]) == recipe_type
            ]
            values[(cluster_id, recipe_type)] = statistics.fmean(subset) if subset else 0.0
    max_value = max(values.values() or [1.0]) or 1.0
    cell_w, cell_h = 155, 58
    left, top = 135, 75
    width = left + cell_w * len(recipe_types) + 40
    height = top + cell_h * len(clusters) + 70
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2}" y="30" text-anchor="middle" font-family="Arial" font-size="20">Average Absolute Error: Cluster vs Recipe Type</text>',
    ]
    for col, recipe_type in enumerate(recipe_types):
        x = left + col * cell_w + cell_w / 2
        parts.append(f'<text x="{x:.1f}" y="58" text-anchor="middle" font-family="Arial" font-size="12">{escape_xml(recipe_type)}</text>')
    for row_index, cluster_id in enumerate(clusters):
        y = top + row_index * cell_h
        parts.append(f'<text x="{left - 18}" y="{y + cell_h / 2 + 5:.1f}" text-anchor="end" font-family="Arial" font-size="13">Cluster {cluster_id}</text>')
        for col, recipe_type in enumerate(recipe_types):
            value = values[(cluster_id, recipe_type)]
            intensity = value / max_value
            red = int(245 - 60 * (1 - intensity))
            green = int(245 - 135 * intensity)
            blue = int(245 - 165 * intensity)
            x = left + col * cell_w
            parts.append(f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" fill="rgb({red},{green},{blue})" stroke="#ffffff"/>')
            parts.append(f'<text x="{x + cell_w / 2:.1f}" y="{y + cell_h / 2 + 5:.1f}" text-anchor="middle" font-family="Arial" font-size="12">{value:.3f}</text>')
    parts.append("</svg>\n")
    path.write_text("\n".join(parts))


def write_svg_scatter(path: Path, rows: list[dict[str, Any]]) -> None:
    width, height = 900, 650
    margin = 70
    x_values = [float(row["word_count"]) for row in rows]
    y_values = [float(row["ingredient_coverage_ratio"]) for row in rows]
    x_min, x_max = min(x_values or [0]), max(x_values or [1])
    y_min, y_max = 0.0, 1.0
    if x_min == x_max:
        x_max = x_min + 1
    colors = ["#4f7cac", "#c7522a", "#2f9c67", "#8e5ea2", "#d99a2b", "#5f5f5f", "#2a9db0", "#b34d7a"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="450" y="30" text-anchor="middle" font-family="Arial" font-size="20">Reasoning Feature Scatter by Cluster</text>',
        f'<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#333"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="#333"/>',
        f'<text x="{width / 2}" y="{height - 20}" text-anchor="middle" font-family="Arial" font-size="13">word_count</text>',
        f'<text x="20" y="{height / 2}" transform="rotate(-90 20 {height / 2})" text-anchor="middle" font-family="Arial" font-size="13">ingredient_coverage_ratio</text>',
    ]
    for row in rows:
        x = margin + (float(row["word_count"]) - x_min) / (x_max - x_min) * (width - 2 * margin)
        y = height - margin - (float(row["ingredient_coverage_ratio"]) - y_min) / (y_max - y_min) * (height - 2 * margin)
        cluster = int(row["cluster_id"])
        color = "#999999" if cluster == -1 else colors[cluster % len(colors)]
        parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}" opacity="0.75">'
            f'<title>{escape_xml(str(row["model_name"]))} | {escape_xml(str(row["recipe_name"]))} | cluster {cluster}</title></circle>'
        )
    parts.append("</svg>\n")
    path.write_text("\n".join(parts))


def escape_xml(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def generate_visualizations(rows: list[dict[str, Any]], analysis: list[dict[str, Any]], vis_dir: Path) -> None:
    vis_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt  # type: ignore
        import numpy as np  # type: ignore

        plt.rcParams.update(
            {
                "axes.titlesize": 15,
                "axes.labelsize": 13,
                "xtick.labelsize": 11,
                "ytick.labelsize": 11,
                "legend.fontsize": 11,
                "legend.title_fontsize": 12,
            }
        )

        clusters = sorted({int(row["cluster_id"]) for row in rows})
        palette = {
            -1: "#6b7280",
            0: "#2563eb",
            2: "#dc2626",
        }
        fallback_colors = ["#059669", "#7c3aed", "#d97706", "#0891b2"]
        cluster_colors = {
            cluster_id: palette.get(cluster_id, fallback_colors[index % len(fallback_colors)])
            for index, cluster_id in enumerate(clusters)
        }

        def cluster_label(row: dict[str, Any]) -> str:
            wrapped = "\n".join(textwrap.wrap(str(row["failure_mode_label"]), width=24))
            return f'Cluster {row["cluster_id"]}\n{wrapped}'

        labels = [cluster_label(row) for row in analysis]
        avg_abs = [float(row["avg_absolute_error"]) for row in analysis]
        sizes = [int(row["cluster_size"]) for row in analysis]
        rel = [float(row["avg_relative_error_percent"]) for row in analysis]
        bar_colors = [cluster_colors[int(row["cluster_id"])] for row in analysis]

        fig, ax = plt.subplots(figsize=(13, 7))
        bars = ax.bar(labels, avg_abs, color=bar_colors)
        ax.set_title("Average Absolute Error by Cluster", pad=14)
        plt.ylabel("Average absolute error (mg)")
        ax.set_xlabel("Failure mode cluster")
        ax.bar_label(bars, labels=[f"{value:.3f}" for value in avg_abs], padding=3, fontsize=11)
        plt.xticks(rotation=0, ha="center")
        fig.tight_layout()
        plt.savefig(vis_dir / "average_absolute_error_by_cluster.png", dpi=180)
        plt.close()

        fig, ax = plt.subplots(figsize=(13, 7))
        bars = ax.bar(labels, rel, color=bar_colors)
        ax.set_title("Average Relative Error by Cluster", pad=14)
        ax.set_ylabel("Average relative error (%)")
        ax.set_xlabel("Failure mode cluster")
        ax.bar_label(bars, labels=[f"{value:.1f}%" for value in rel], padding=3, fontsize=11)
        ax.text(
            0.99,
            0.94,
            "Note: Cluster 2 includes a 6.96 mg error on a 0.04 mg true value.",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=11,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "#fff7ed", "edgecolor": "#fdba74"},
        )
        plt.xticks(rotation=0, ha="center")
        fig.tight_layout()
        plt.savefig(vis_dir / "average_relative_error_by_cluster.png", dpi=180)
        plt.close()

        fig, ax = plt.subplots(figsize=(9, 5.5))
        size_labels = [f'Cluster {row["cluster_id"]}' for row in analysis]
        bars = ax.bar(size_labels, sizes, color=bar_colors)
        total = sum(sizes) or 1
        ax.set_title("Cluster Sizes", pad=14)
        ax.set_ylabel("Predictions")
        ax.set_xlabel("Cluster")
        ax.bar_label(
            bars,
            labels=[f"{size} ({size / total:.0%})" for size in sizes],
            padding=3,
            fontsize=11,
        )
        fig.tight_layout()
        plt.savefig(vis_dir / "cluster_sizes.png", dpi=180)
        plt.close()

        fig, ax = plt.subplots(figsize=(10, 6.5))
        for cluster_id in clusters:
            cluster_rows = [row for row in rows if int(row["cluster_id"]) == cluster_id]
            x_values = [float(row["word_count"]) for row in cluster_rows]
            y_values = []
            for index, row in enumerate(cluster_rows):
                coverage = float(row["ingredient_coverage_ratio"])
                jitter = ((index % 9) - 4) * 0.006 if abs(coverage - 1.0) < 1e-9 else 0.0
                y_values.append(min(1.025, max(-0.025, coverage + jitter)))
            ax.scatter(
                x_values,
                y_values,
                s=32,
                alpha=0.78,
                label=f"Cluster {cluster_id}",
                color=cluster_colors[cluster_id],
                edgecolors="white",
                linewidths=0.4,
            )
        ax.set_title("Reasoning Length vs Ingredient Coverage", pad=14)
        ax.set_xlabel("Reasoning length (words)")
        ax.set_ylabel("Ingredient coverage ratio")
        ax.set_ylim(-0.03, 1.04)
        ax.legend(title="Cluster", ncol=2)
        fig.tight_layout()
        plt.savefig(vis_dir / "feature_scatter_word_count_coverage.png", dpi=180)
        plt.close()

        box_data = [
            [float(row["absolute_error"]) for row in rows if int(row["cluster_id"]) == cluster_id]
            for cluster_id in clusters
        ]
        fig, ax = plt.subplots(figsize=(10, 6.5))
        box = ax.boxplot(
            box_data,
            tick_labels=[f"Cluster {cluster_id}" for cluster_id in clusters],
            patch_artist=True,
            showfliers=False,
        )
        for patch, cluster_id in zip(box["boxes"], clusters):
            patch.set_facecolor(cluster_colors[cluster_id])
            patch.set_alpha(0.45)
        for position, cluster_id in enumerate(clusters, start=1):
            values = [float(row["absolute_error"]) for row in rows if int(row["cluster_id"]) == cluster_id]
            jitter = [((index % 9) - 4) * 0.035 for index, _ in enumerate(values)]
            ax.scatter(
                [position + offset for offset in jitter],
                values,
                color=cluster_colors[cluster_id],
                alpha=0.72,
                s=30,
                edgecolors="white",
                linewidths=0.4,
            )
        ax.set_yscale("log")
        ax.set_title("Absolute Error Distribution by Cluster", pad=14)
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Absolute error (mg, log scale)")
        ax.grid(axis="y", which="both", alpha=0.25)
        fig.tight_layout()
        plt.savefig(vis_dir / "error_distribution_by_cluster.png", dpi=180)
        plt.close()

        recipe_types = sorted({str(row["recipe_type"]) for row in rows})
        heatmap = []
        annotations = []
        for cluster_id in clusters:
            heatmap_row = []
            annotation_row = []
            for recipe_type in recipe_types:
                subset = [
                    float(row["absolute_error"])
                    for row in rows
                    if int(row["cluster_id"]) == cluster_id and str(row["recipe_type"]) == recipe_type
                ]
                if subset:
                    value = statistics.fmean(subset)
                    heatmap_row.append(value)
                    annotation_row.append(f"{value:.3f}")
                else:
                    heatmap_row.append(float("nan"))
                    annotation_row.append("No data")
            heatmap.append(heatmap_row)
            annotations.append(annotation_row)
        heatmap_array = np.array(heatmap, dtype=float)
        cmap = plt.cm.YlOrRd.copy()
        cmap.set_bad(color="#e5e7eb")
        fig, ax = plt.subplots(figsize=(9.8, 5.6))
        image = ax.imshow(np.ma.masked_invalid(heatmap_array), cmap=cmap, aspect="auto")
        plt.colorbar(image, ax=ax, label="Average absolute error (mg)")
        ax.set_title("Average Absolute Error by Cluster and Recipe Type", pad=14)
        ax.set_xticks(range(len(recipe_types)), recipe_types, rotation=30, ha="right")
        ax.set_yticks(range(len(clusters)), [f"Cluster {cluster_id}" for cluster_id in clusters])
        ax.set_xlabel("Recipe type")
        ax.set_ylabel("Cluster")
        for row_index, annotation_row in enumerate(annotations):
            for col_index, text in enumerate(annotation_row):
                value = heatmap_array[row_index, col_index]
                color = "white" if not np.isnan(value) and value > np.nanmax(heatmap_array) * 0.55 else "#111827"
                ax.text(col_index, row_index, text, ha="center", va="center", color=color, fontsize=10)
        fig.tight_layout()
        plt.savefig(vis_dir / "failure_mode_recipe_type_heatmap.png", dpi=180)
        plt.close()
        return
    except Exception:
        pass

    write_svg_scatter(vis_dir / "feature_scatter_word_count_coverage.svg", rows)
    write_svg_bar_chart(
        vis_dir / "average_absolute_error_by_cluster.svg",
        "Average Absolute Error by Cluster",
        [(f'{row["cluster_id"]}: {row["failure_mode_label"][:28]}', float(row["avg_absolute_error"])) for row in analysis],
        "Average absolute error (mg)",
    )
    write_svg_bar_chart(
        vis_dir / "average_relative_error_by_cluster.svg",
        "Average Relative Error by Cluster",
        [(f'{row["cluster_id"]}: {row["failure_mode_label"][:28]}', float(row["avg_relative_error_percent"])) for row in analysis],
        "Average relative error (%)",
    )
    write_svg_bar_chart(
        vis_dir / "cluster_sizes.svg",
        "Cluster Sizes",
        [(str(row["cluster_id"]), float(row["cluster_size"])) for row in analysis],
        "Predictions",
    )
    write_svg_box_plot(vis_dir / "error_distribution_by_cluster.svg", rows)
    write_svg_heatmap(vis_dir / "failure_mode_recipe_type_heatmap.svg", rows)


def write_report(
    readme_path: Path,
    rows: list[dict[str, Any]],
    analysis: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    top_frequency = sorted(analysis, key=lambda row: int(row["cluster_size"]), reverse=True)[:5]
    top_error = sorted(analysis, key=lambda row: float(row["avg_absolute_error"]), reverse=True)[:5]
    noise_count = sum(1 for row in rows if int(row["cluster_id"]) == -1)
    cluster_count = len([row for row in analysis if int(row["cluster_id"]) != -1])

    lines = [
        "# CoT Analysis & Failure Mode Clustering",
        "",
        "## Implementation Status",
        "This directory now contains an executable clustering pipeline in `analyze_clusters.py` plus generated analysis artifacts.",
        "",
        "## Methodology",
        f"- Parsed {metadata['input_filter']['reasoning_summary_candidate_rows']} valid trial results from `model_outputs/*.jsonl` with non-empty `reasoning_summary_text`.",
        f"- Clustered the top {float(metadata['top_absolute_error_fraction']) * 100:g}% by absolute error: {len(rows)} predictions at or above {float(metadata['absolute_error_cutoff_mg']):.3f} mg.",
        "- Extracted lexical, calculation, confidence, ingredient-coverage, and method-indicator features from `reasoning_summary_text` plus `raw_output`.",
        f"- Clustered standardized feature vectors with `{metadata['cluster_metadata']['algorithm']}`.",
        f"- Applied manual cluster merges after HDBSCAN: {json.dumps(metadata.get('cluster_merges', []), sort_keys=True)}.",
        "- Computed error magnitude and signed error post hoc; signed error is not used as a clustering feature.",
        "- Labeled clusters with OpenAI when `OPENAI_API_KEY` is available, otherwise with deterministic heuristic labels.",
        "",
        "## Cluster Quality",
        f"- Non-noise clusters: {cluster_count}",
        f"- Noise points: {noise_count}",
        f"- Silhouette score: {metadata.get('silhouette_score') if metadata.get('silhouette_score') is not None else 'not available'}",
        "",
        "## Most Frequent Failure/Reasoning Modes",
    ]
    for row in top_frequency:
        lines.append(
            f"- Cluster {row['cluster_id']} ({row['cluster_size']} predictions): "
            f"{row['failure_mode_label']} | avg MAE {float(row['avg_absolute_error']):.3f} mg | "
            f"direction {row['error_direction']}"
        )
    lines.extend(["", "## Highest Error Clusters"])
    for row in top_error:
        lines.append(
            f"- Cluster {row['cluster_id']}: {row['failure_mode_label']} | "
            f"avg MAE {float(row['avg_absolute_error']):.3f} mg | "
            f"avg relative error {float(row['avg_relative_error_percent']):.1f}% | "
            f"worst model {row['worst_performing_model']}"
        )
    lines.extend(
        [
            "",
            "## Output Artifacts",
            "- `features_extracted.csv` contains one row per top-absolute-error reasoning-summary prediction with extracted features and error metrics.",
            "- `clusters_with_labels.csv` adds `cluster_id` to each prediction row.",
            "- `failure_mode_labels.json` maps each cluster to a label, description, label source, and representative samples.",
            "- `error_cluster_analysis.csv` summarizes error, recipe type, model diversity, and over/under-estimation direction by cluster.",
            "- `visualizations/` contains cluster/error plots. PNG files are written when matplotlib is installed; SVG fallbacks are written otherwise.",
            "",
            "## Actionable Recommendations",
            "- Inspect high-MAE clusters first; they identify reasoning styles most associated with large copper-estimation misses.",
            "- Compare `ingredient_coverage_ratio` and `grouping_marker_count` against MAE to find recipes where grouped negligible assumptions hide meaningful copper sources.",
            "- Use `error_direction` to separate reasoning-pattern fixes from calibration fixes; directional clusters suggest systematic numeric bias after the reasoning mode is identified.",
            "",
            "## Reproduce",
            "```bash",
            "uv run python analyze_clusters.py",
            "```",
            "",
        ]
    )
    readme_path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    args.stage_dir.mkdir(parents=True, exist_ok=True)
    recipes = load_recipes(args.recipes)
    eval_rows = load_eval_rows(args.evaluation_csv)
    trial_rows = load_trial_rows(
        args.model_outputs_dir,
        recipes,
        eval_rows,
    )
    if not trial_rows:
        raise SystemExit("No valid trial_result rows with reasoning_summary_text found.")
    trial_rows, filter_metadata = filter_top_absolute_error_rows(
        trial_rows,
        args.top_absolute_error_fraction,
    )

    feature_rows, numeric_columns = build_feature_rows(trial_rows)
    labels, cluster_metadata = cluster_rows(feature_rows, numeric_columns, args.min_cluster_size)
    for row, cluster_id in zip(feature_rows, labels):
        row["cluster_id"] = cluster_id
    labels, cluster_merges = apply_cluster_merges(feature_rows, labels, DEFAULT_CLUSTER_MERGES)

    scaled = standardize(matrix_from_rows(feature_rows, numeric_columns))
    silhouette = silhouette_score(scaled, labels)
    summaries = summarize_clusters(feature_rows)
    cluster_labels = label_clusters(feature_rows, summaries, args.llm_model, args.skip_llm_labels)
    analysis = attach_labels_to_analysis(summaries, cluster_labels, feature_rows)

    features_only_rows = [
        {key: value for key, value in row.items() if key != "cluster_id"}
        for row in feature_rows
    ]
    write_csv(args.stage_dir / "features_extracted.csv", features_only_rows)
    write_csv(args.stage_dir / "clusters_with_labels.csv", feature_rows)
    write_json(args.stage_dir / "failure_mode_labels.json", cluster_labels)
    write_csv(args.stage_dir / "error_cluster_analysis.csv", analysis)

    metadata = {
        "trial_rows": len(trial_rows),
        "feature_columns": numeric_columns,
        "cluster_metadata": cluster_metadata,
        "silhouette_score": silhouette,
        "cluster_count_excluding_noise": len({label for label in labels if label != -1}),
        "noise_count": sum(1 for label in labels if label == -1),
        "cluster_merges": cluster_merges,
        **filter_metadata,
        "input_filter": {
            "requires_reasoning_summary_text": True,
            "requires_top_absolute_error_fraction": True,
            "reasoning_summary_candidate_rows": filter_metadata["reasoning_summary_candidate_rows"],
        },
    }
    write_json(args.stage_dir / "analysis_metadata.json", metadata)
    if not args.skip_plots:
        generate_visualizations(feature_rows, analysis, args.stage_dir / "visualizations")
    write_report(args.stage_dir / "README.md", feature_rows, analysis, metadata)


if __name__ == "__main__":
    main()
