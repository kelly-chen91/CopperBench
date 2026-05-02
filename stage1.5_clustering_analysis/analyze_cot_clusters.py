#!/usr/bin/env python3
"""Cluster copper-estimation reasoning traces and analyze failure modes."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


STAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = STAGE_DIR.parent
DEFAULT_RECIPES_PATH = REPO_ROOT / "recipes.json"
DEFAULT_MODEL_OUTPUTS_DIR = REPO_ROOT / "model_outputs"
DEFAULT_STAGE1_PREDICTIONS = (
    REPO_ROOT / "stage1_recipe_proportions" / "evaluation_results" / "per_recipe_predictions.csv"
)
DEFAULT_OUTPUT_DIR = STAGE_DIR
DEFAULT_LABEL_MODEL = "gpt-4.1-mini"

UNCERTAINTY_MARKERS = (
    "estimate",
    "estimated",
    "roughly",
    "approximately",
    "approx",
    "about",
    "around",
    "typical",
    "average",
    "assume",
    "likely",
)
CORRECTION_MARKERS = (
    "however",
    "instead",
    "recalculate",
    "recomputed",
    "correction",
    "correcting",
    "revise",
    "revised",
    "adjust",
    "adjusted",
)
HIGH_REFERENCE_MARKERS = (
    "high in copper",
    "rich in copper",
    "significant copper",
    "substantial copper",
)
LOW_REFERENCE_MARKERS = (
    "negligible",
    "trace",
    "little copper",
    "very low",
    "no copper",
)
GROUPING_MARKERS = (
    "remaining ingredients",
    "other ingredients",
    "minor ingredients",
    "negligible ingredients",
    "combined",
    "together",
)


@dataclass(frozen=True)
class RecipeTruth:
    recipe_index: int
    recipe_type: str
    recipe_name: str
    servings: str
    ingredients: list[str]
    copper_per_serving_mg: float


@dataclass(frozen=True)
class TrialRecord:
    model_name: str
    model_file: str
    reasoning_level: str
    recipe_index: int
    recipe_type: str
    recipe_name: str
    predicted_copper_mg: float | None
    raw_output: str
    success: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract CoT features, cluster reasoning traces, and analyze failure modes."
    )
    parser.add_argument("--recipes", type=Path, default=DEFAULT_RECIPES_PATH)
    parser.add_argument("--model-outputs-dir", type=Path, default=DEFAULT_MODEL_OUTPUTS_DIR)
    parser.add_argument("--stage1-predictions", type=Path, default=DEFAULT_STAGE1_PREDICTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-cluster-size", type=int, default=8)
    parser.add_argument("--label-model", default=DEFAULT_LABEL_MODEL)
    parser.add_argument("--skip-llm-labels", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def iter_recipe_dicts(node: Any):
    if isinstance(node, dict):
        yield node
        return
    if isinstance(node, list):
        for item in node:
            yield from iter_recipe_dicts(item)


def load_truths(path: Path) -> dict[int, RecipeTruth]:
    data = json.loads(path.read_text())
    truths: dict[int, RecipeTruth] = {}
    recipe_index = 0
    for group in data:
        recipe_type = str(group.get("recipe_type", "unknown"))
        for recipe in iter_recipe_dicts(group.get("recipes", [])):
            ingredients = [
                str(item.get("name", ""))
                for item in recipe.get("ingredients", [])
                if isinstance(item, dict) and item.get("name")
            ]
            truths[recipe_index] = RecipeTruth(
                recipe_index=recipe_index,
                recipe_type=recipe_type,
                recipe_name=str(recipe["name"]),
                servings=str(recipe.get("servings", "")),
                ingredients=ingredients,
                copper_per_serving_mg=float(recipe["copper_per_serving_mg"]),
            )
            recipe_index += 1
    if not truths:
        raise ValueError(f"No recipes found in {path}")
    return truths


def model_name_from_path(path: Path) -> str:
    return path.stem


def reasoning_level(config: dict[str, Any] | None, path: Path) -> str:
    if config and config.get("reasoning_effort"):
        return str(config["reasoning_effort"])
    match = re.search(r"__reasoning_([a-z]+)$", path.stem)
    return match.group(1) if match else "none"


def maybe_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    if isinstance(value, str):
        try:
            parsed = float(value.strip())
        except ValueError:
            return None
        return parsed if math.isfinite(parsed) else None
    return None


def extract_prediction(record: dict[str, Any]) -> float | None:
    parsed = record.get("parsed_output")
    if isinstance(parsed, dict):
        value = maybe_float(parsed.get("copper_mg_per_serving"))
        if value is not None:
            return value
    return None


def load_stage1_errors(path: Path) -> dict[tuple[str, int], dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            (row["model_name"], int(row["recipe_index"])): row
            for row in csv.DictReader(handle)
            if row.get("model_name") and row.get("recipe_index")
        }


def load_trial_records(paths: list[Path]) -> list[TrialRecord]:
    records: list[TrialRecord] = []
    for path in paths:
        config: dict[str, Any] | None = None
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"Skipping invalid JSON in {path}:{line_number}: {exc}")
                continue
            if record.get("record_type") == "experiment_config":
                config = record.get("config") if isinstance(record.get("config"), dict) else config
                continue
            if record.get("record_type") != "trial_result":
                continue
            recipe_index = record.get("recipe_index")
            if isinstance(recipe_index, bool) or not isinstance(recipe_index, int):
                continue
            records.append(
                TrialRecord(
                    model_name=model_name_from_path(path),
                    model_file=path.name,
                    reasoning_level=reasoning_level(config, path),
                    recipe_index=recipe_index,
                    recipe_type=str(record.get("recipe_type", "")),
                    recipe_name=str(record.get("recipe_name", "")),
                    predicted_copper_mg=extract_prediction(record),
                    raw_output=str(record.get("raw_output") or ""),
                    success=bool(record.get("success", False)),
                )
            )
    return records


def count_regex(pattern: str, text: str) -> int:
    return len(re.findall(pattern, text, flags=re.IGNORECASE | re.MULTILINE))


def keyword_count(text: str, keywords: tuple[str, ...]) -> int:
    lower = text.lower()
    return sum(lower.count(keyword) for keyword in keywords)


def ingredient_tokens(name: str) -> set[str]:
    stopwords = {
        "and",
        "or",
        "the",
        "fresh",
        "dried",
        "large",
        "medium",
        "small",
        "reduced",
        "fat",
        "low",
        "plain",
        "chopped",
        "shredded",
        "divided",
        "warmed",
        "ground",
        "canned",
        "drained",
        "rinsed",
        "optional",
        "about",
        "any",
        "color",
    }
    tokens = set(re.findall(r"[a-z][a-z-]{2,}", name.lower()))
    return {token.strip("-") for token in tokens if token not in stopwords}


def ingredient_coverage(raw_output: str, ingredients: list[str]) -> tuple[int, int, float]:
    lower = raw_output.lower()
    lines = [line.lower() for line in raw_output.splitlines()]
    mentioned = 0
    explicit_estimates = 0
    for ingredient in ingredients:
        tokens = ingredient_tokens(ingredient)
        if not tokens:
            continue
        is_mentioned = any(token in lower for token in tokens)
        if is_mentioned:
            mentioned += 1
        if any("mg" in line and any(token in line for token in tokens) for line in lines):
            explicit_estimates += 1
    total = len(ingredients)
    return mentioned, explicit_estimates, (explicit_estimates / total if total else 0.0)


def arithmetic_expression_count(text: str) -> int:
    return count_regex(r"\d+(?:\.\d+)?\s*[+*/-]\s*\d+", text)


def extract_features(record: TrialRecord, truth: RecipeTruth) -> dict[str, Any]:
    text = record.raw_output
    lower = text.lower()
    words = re.findall(r"[A-Za-z0-9_']+", text)
    paragraphs = [part for part in re.split(r"\n\s*\n", text.strip()) if part.strip()]
    mentioned, explicit, coverage = ingredient_coverage(text, truth.ingredients)
    predicted = record.predicted_copper_mg
    ground_truth = truth.copper_per_serving_mg
    signed_error = predicted - ground_truth if predicted is not None else ""
    absolute_error = abs(signed_error) if signed_error != "" else ""
    relative_error = absolute_error / ground_truth * 100 if absolute_error != "" and ground_truth else ""
    total_ingredients = len(truth.ingredients)

    features = {
        "word_count": len(words),
        "paragraph_count": len(paragraphs),
        "sentence_count": count_regex(r"[.!?](?:\s|$)", text),
        "bullet_count": count_regex(r"^\s*(?:[-*]|\d+[.)])\s+", text),
        "numbered_step_count": count_regex(r"^\s*\d+[.)]\s+", text),
        "json_block_present": int("```json" in lower or '"copper_mg_per_serving"' in lower),
        "usda_reference_count": keyword_count(lower, ("usda", "fdc", "fooddata", "database")),
        "uncertainty_marker_count": keyword_count(lower, UNCERTAINTY_MARKERS),
        "correction_phrase_count": keyword_count(lower, CORRECTION_MARKERS),
        "refinement_step_count": keyword_count(lower, CORRECTION_MARKERS),
        "ingredient_count": total_ingredients,
        "ingredient_mentioned_count": mentioned,
        "explicit_ingredient_estimate_count": explicit,
        "ingredient_coverage_percent": coverage * 100,
        "skipped_or_negligible_count": keyword_count(lower, LOW_REFERENCE_MARKERS),
        "grouping_marker_count": keyword_count(lower, GROUPING_MARKERS),
        "per_100g_count": count_regex(r"per\s+100\s*g|/\s*100\s*g|100g", lower),
        "gram_conversion_count": count_regex(r"(?:~|≈|about|approx\.?)?\s*\d+(?:\.\d+)?\s*g\b", lower),
        "portion_based_count": keyword_count(lower, ("cup", "tablespoon", "tbsp", "teaspoon", "tsp", "serving")),
        "division_per_serving_count": count_regex(r"(?:divide|dividing|/)\s*(?:by\s*)?\d+|per serving", lower),
        "sum_statement_count": keyword_count(lower, ("sum", "summing", "total", "subtotal")),
        "arithmetic_expression_count": arithmetic_expression_count(text),
        "high_copper_reference_count": keyword_count(lower, HIGH_REFERENCE_MARKERS),
        "low_copper_reference_count": keyword_count(lower, LOW_REFERENCE_MARKERS),
        "estimated_numeric_mg_count": count_regex(r"\d+(?:\.\d+)?\s*mg", lower),
        "calculation_density": arithmetic_expression_count(text) / max(1, len(words)),
        "coverage_gap_count": max(0, total_ingredients - explicit),
        "complexity_response_ratio": explicit / max(1, total_ingredients),
        "overestimate_indicator": int(signed_error != "" and signed_error > 0),
        "underestimate_indicator": int(signed_error != "" and signed_error < 0),
    }
    row = {
        "model_name": record.model_name,
        "model_file": record.model_file,
        "reasoning_level": record.reasoning_level,
        "recipe_index": record.recipe_index,
        "recipe_type": truth.recipe_type,
        "recipe_name": truth.recipe_name,
        "predicted_copper_mg": predicted if predicted is not None else "",
        "ground_truth_copper_mg": ground_truth,
        "signed_error": signed_error,
        "absolute_error": absolute_error,
        "relative_error_percent": relative_error,
        "success": record.success,
        "raw_output": text,
    }
    row.update(features)
    return row


FEATURE_COLUMNS = [
    "word_count",
    "paragraph_count",
    "sentence_count",
    "bullet_count",
    "numbered_step_count",
    "json_block_present",
    "usda_reference_count",
    "uncertainty_marker_count",
    "correction_phrase_count",
    "refinement_step_count",
    "ingredient_count",
    "ingredient_mentioned_count",
    "explicit_ingredient_estimate_count",
    "ingredient_coverage_percent",
    "skipped_or_negligible_count",
    "grouping_marker_count",
    "per_100g_count",
    "gram_conversion_count",
    "portion_based_count",
    "division_per_serving_count",
    "sum_statement_count",
    "arithmetic_expression_count",
    "high_copper_reference_count",
    "low_copper_reference_count",
    "estimated_numeric_mg_count",
    "calculation_density",
    "coverage_gap_count",
    "complexity_response_ratio",
    "overestimate_indicator",
    "underestimate_indicator",
]


def numeric_matrix(rows: list[dict[str, Any]]) -> np.ndarray:
    matrix = []
    for row in rows:
        matrix.append([float(row[column] or 0.0) for column in FEATURE_COLUMNS])
    return np.asarray(matrix, dtype=float)


def standardize(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0)
    stds[stds == 0] = 1.0
    return (matrix - means) / stds, means, stds


def run_hdbscan_or_fallback(
    scaled: np.ndarray,
    min_cluster_size: int,
) -> tuple[np.ndarray, str, dict[str, Any]]:
    try:
        import hdbscan  # type: ignore

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=max(2, min_cluster_size // 2),
            metric="euclidean",
        )
        labels = clusterer.fit_predict(scaled)
        return labels.astype(int), "hdbscan", {
            "min_cluster_size": min_cluster_size,
            "min_samples": max(2, min_cluster_size // 2),
        }
    except Exception as exc:
        labels = fallback_kmeans_with_noise(scaled, target_clusters=6)
        return labels, "fallback_kmeans_distance_noise", {
            "reason": f"{type(exc).__name__}: {exc}",
            "target_clusters": 6,
            "note": "Install hdbscan to run the README-specified clustering algorithm.",
        }


def fallback_kmeans_with_noise(
    scaled: np.ndarray,
    target_clusters: int,
    iterations: int = 80,
) -> np.ndarray:
    if len(scaled) == 0:
        return np.asarray([], dtype=int)
    k = min(target_clusters, max(1, len(scaled)))
    rng = np.random.default_rng(42)
    centroids = scaled[rng.choice(len(scaled), size=k, replace=False)]
    labels = np.zeros(len(scaled), dtype=int)
    for _ in range(iterations):
        distances = np.linalg.norm(scaled[:, None, :] - centroids[None, :, :], axis=2)
        new_labels = distances.argmin(axis=1)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        for cluster_id in range(k):
            members = scaled[labels == cluster_id]
            if len(members):
                centroids[cluster_id] = members.mean(axis=0)
    nearest = np.linalg.norm(scaled - centroids[labels], axis=1)
    threshold = float(np.quantile(nearest, 0.98))
    labels = labels.astype(int)
    labels[nearest > threshold] = -1
    return labels


def silhouette_score(matrix: np.ndarray, labels: np.ndarray) -> float | None:
    cluster_labels = sorted(label for label in set(labels.tolist()) if label != -1)
    if len(cluster_labels) < 2:
        return None
    sample_scores = []
    for index, point in enumerate(matrix):
        label = labels[index]
        if label == -1:
            continue
        own = matrix[labels == label]
        if len(own) <= 1:
            continue
        a = float(np.linalg.norm(own - point, axis=1).sum() / (len(own) - 1))
        b = min(
            float(np.linalg.norm(matrix[labels == other] - point, axis=1).mean())
            for other in cluster_labels
            if other != label and np.any(labels == other)
        )
        sample_scores.append((b - a) / max(a, b) if max(a, b) else 0.0)
    return float(statistics.fmean(sample_scores)) if sample_scores else None


def pca_2d(scaled: np.ndarray) -> np.ndarray:
    centered = scaled - scaled.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:2].T


def coerce_float(value: Any) -> float | None:
    if value == "" or value is None:
        return None
    return float(value)


def fmean_or_none(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def cluster_statistics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_cluster: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_cluster[int(row["cluster_id"])].append(row)

    stats = []
    for cluster_id, cluster_rows in sorted(by_cluster.items()):
        abs_errors = [coerce_float(row["absolute_error"]) for row in cluster_rows]
        rel_errors = [coerce_float(row["relative_error_percent"]) for row in cluster_rows]
        signed_errors = [coerce_float(row["signed_error"]) for row in cluster_rows]
        abs_values = [value for value in abs_errors if value is not None]
        rel_values = [value for value in rel_errors if value is not None]
        signed_values = [value for value in signed_errors if value is not None]
        model_counts = Counter(row["model_name"] for row in cluster_rows)
        recipe_type_counts = Counter(row["recipe_type"] for row in cluster_rows)
        stats.append(
            {
                "cluster_id": cluster_id,
                "cluster_size": len(cluster_rows),
                "avg_absolute_error": fmean_or_none(abs_values),
                "median_absolute_error": statistics.median(abs_values) if abs_values else None,
                "avg_relative_error_percent": fmean_or_none(rel_values),
                "median_relative_error_percent": statistics.median(rel_values) if rel_values else None,
                "avg_signed_error": fmean_or_none(signed_values),
                "overestimate_rate_percent": (
                    sum(1 for value in signed_values if value > 0) / len(signed_values) * 100
                    if signed_values
                    else None
                ),
                "underestimate_rate_percent": (
                    sum(1 for value in signed_values if value < 0) / len(signed_values) * 100
                    if signed_values
                    else None
                ),
                "model_diversity": len(model_counts),
                "top_models": "; ".join(f"{name}:{count}" for name, count in model_counts.most_common(5)),
                "recipe_type_distribution": "; ".join(
                    f"{name}:{count}" for name, count in recipe_type_counts.most_common()
                ),
                "best_performing_model": best_or_worst_model(cluster_rows, best=True),
                "worst_performing_model": best_or_worst_model(cluster_rows, best=False),
            }
        )
    return stats


def best_or_worst_model(rows: list[dict[str, Any]], best: bool) -> str:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = coerce_float(row["absolute_error"])
        if value is not None:
            grouped[row["model_name"]].append(value)
    if not grouped:
        return ""
    scored = [(name, statistics.fmean(values), len(values)) for name, values in grouped.items()]
    scored.sort(key=lambda item: (item[1], -item[2], item[0]), reverse=not best)
    name, avg_error, count = scored[0]
    return f"{name} (MAE={avg_error:.4f}, n={count})"


def representative_samples(rows: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    valid_rows = [row for row in rows if coerce_float(row["absolute_error"]) is not None]
    if not valid_rows:
        valid_rows = rows
    sorted_rows = sorted(valid_rows, key=lambda row: coerce_float(row["absolute_error"]) or 0.0)
    picks: list[dict[str, Any]] = []
    if sorted_rows:
        picks.extend([sorted_rows[0], sorted_rows[len(sorted_rows) // 2], sorted_rows[-1]])
    for row in sorted_rows:
        if row not in picks:
            picks.append(row)
        if len(picks) >= limit:
            break
    result = []
    for row in picks[:limit]:
        result.append(
            {
                "model_name": row["model_name"],
                "recipe_name": row["recipe_name"],
                "recipe_type": row["recipe_type"],
                "absolute_error": row["absolute_error"],
                "relative_error_percent": row["relative_error_percent"],
                "raw_output_excerpt": str(row["raw_output"]).strip()[:1200],
            }
        )
    return result


def heuristic_cluster_label(cluster_id: int, rows: list[dict[str, Any]]) -> dict[str, Any]:
    feature_means = {
        feature: statistics.fmean(float(row[feature] or 0.0) for row in rows)
        for feature in FEATURE_COLUMNS
    }
    avg_signed = fmean_or_none(
        [coerce_float(row["signed_error"]) for row in rows if coerce_float(row["signed_error"]) is not None]
    )
    avg_abs = fmean_or_none(
        [
            coerce_float(row["absolute_error"])
            for row in rows
            if coerce_float(row["absolute_error"]) is not None
        ]
    )
    over_rate = feature_means["overestimate_indicator"]
    under_rate = feature_means["underestimate_indicator"]
    if cluster_id == -1:
        label = "Anomalous reasoning traces"
        description = "Noise or outlier traces that do not closely match the dominant feature patterns."
    elif (
        feature_means["ingredient_coverage_percent"] < 55
        and avg_signed is not None
        and avg_signed > 0.05
        and over_rate >= 0.6
    ):
        label = "Sparse overestimation"
        description = "Reasoning gives limited ingredient-level support while skewing high against ground truth."
    elif (
        feature_means["ingredient_coverage_percent"] < 55
        and avg_signed is not None
        and avg_signed < -0.02
        and under_rate >= 0.6
    ):
        label = "Sparse underestimation"
        description = "Reasoning gives limited ingredient-level support while skewing low against ground truth."
    elif feature_means["ingredient_coverage_percent"] < 55 and avg_abs is not None and avg_abs < 0.04:
        label = "Sparse low-error estimates"
        description = "Reasoning is brief and low coverage, but absolute error stays comparatively small."
    elif feature_means["per_100g_count"] >= 4 and feature_means["gram_conversion_count"] >= 8:
        label = "Dense weight-based calculation"
        description = "Reasoning relies heavily on gram conversions and per-100g copper references."
    elif feature_means["skipped_or_negligible_count"] >= 5:
        label = "Negligible-ingredient pruning"
        description = "Reasoning repeatedly marks ingredients as negligible or trace copper contributors."
    elif feature_means["ingredient_coverage_percent"] < 55:
        label = "Low ingredient coverage"
        description = "Reasoning tends to skip explicit per-ingredient estimates or group ingredients broadly."
    elif avg_signed is not None and avg_signed > 0.08:
        label = "Systematic overestimation"
        description = "Predictions in this cluster skew high relative to the recipe ground truth."
    elif avg_signed is not None and avg_signed < -0.08:
        label = "Systematic underestimation"
        description = "Predictions in this cluster skew low relative to the recipe ground truth."
    elif feature_means["uncertainty_marker_count"] >= 12:
        label = "Approximation-heavy reasoning"
        description = "Reasoning uses many uncertainty and approximation markers."
    else:
        label = "Structured ingredient summation"
        description = "Reasoning follows a conventional ingredient-by-ingredient sum and serving division."
    return {
        "label": label,
        "description": description,
        "confidence": "medium",
        "label_source": "heuristic",
    }


def label_clusters(
    rows: list[dict[str, Any]],
    label_model: str,
    skip_llm: bool,
) -> dict[str, Any]:
    by_cluster: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_cluster[int(row["cluster_id"])].append(row)

    labels: dict[str, Any] = {}
    for cluster_id, cluster_rows in sorted(by_cluster.items()):
        base_label = heuristic_cluster_label(cluster_id, cluster_rows)
        samples = representative_samples(cluster_rows)
        if not skip_llm:
            llm_label = try_llm_label(cluster_id, cluster_rows, samples, label_model)
            if llm_label:
                base_label = llm_label
        base_label["representative_samples"] = samples
        labels[str(cluster_id)] = base_label
    return labels


def try_llm_label(
    cluster_id: int,
    rows: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    label_model: str,
) -> dict[str, Any] | None:
    if not os.environ.get("OPENAI_API_KEY"):
        return None
    try:
        from openai import OpenAI  # type: ignore
    except Exception:
        return None

    stats = cluster_statistics([{**row, "cluster_id": cluster_id} for row in rows])[0]
    prompt = {
        "task": "Name the shared failure mode or reasoning pattern in these copper-estimation CoT samples.",
        "cluster_statistics": stats,
        "samples": samples,
        "response_schema": {
            "label": "short noun phrase",
            "description": "one to three sentences",
            "confidence": "low|medium|high",
        },
    }
    try:
        client = OpenAI()
        response = client.responses.create(
            model=label_model,
            input=(
                "Return strict JSON only. Identify the shared reasoning pattern, "
                "especially any failure mode tied to copper-estimation error.\n"
                + json.dumps(prompt, indent=2)
            ),
            temperature=0,
        )
        text = response.output_text.strip()
        parsed = parse_llm_json(text)
        return {
            "label": str(parsed.get("label", "Unlabeled cluster")),
            "description": str(parsed.get("description", "")),
            "confidence": str(parsed.get("confidence", "medium")),
            "label_source": f"openai:{label_model}",
        }
    except Exception as exc:
        return {
            **heuristic_cluster_label(cluster_id, rows),
            "llm_label_error": f"{type(exc).__name__}: {exc}",
        }


def parse_llm_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1)
    elif not cleaned.startswith("{"):
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start : end + 1]
    parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise ValueError("LLM label response must be a JSON object")
    return parsed


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = fieldnames or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def require_matplotlib():
    cache_dir = STAGE_DIR / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir / "xdg"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def short_cluster_label(cluster_id: int, labels: dict[str, Any], max_length: int = 34) -> str:
    label = str(labels.get(str(cluster_id), {}).get("label") or "")
    if not label:
        return str(cluster_id)
    text = f"{cluster_id}: {label}"
    return text if len(text) <= max_length else text[: max_length - 1].rstrip() + "..."


def plot_cluster_label_key(labels: dict[str, Any], output_dir: Path) -> None:
    if not labels:
        return
    visual_dir = output_dir / "visualizations"
    plt = require_matplotlib()
    ordered = sorted(labels.items(), key=lambda item: int(item[0]))
    rows = [[cluster_id, str(info.get("label", ""))] for cluster_id, info in ordered]
    fig_height = max(6, len(rows) * 0.32)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["Cluster ID", "Failure Mode Label"],
        cellLoc="left",
        colLoc="left",
        loc="center",
        colWidths=[0.14, 0.86],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.25)
    ax.set_title("Cluster ID to Failure Mode Label", pad=16)
    fig.tight_layout()
    fig.savefig(visual_dir / "cluster_label_key.png", dpi=180)
    plt.close(fig)


def plot_outputs(
    rows: list[dict[str, Any]],
    stats: list[dict[str, Any]],
    labels_json: dict[str, Any],
    output_dir: Path,
) -> None:
    visual_dir = output_dir / "visualizations"
    visual_dir.mkdir(parents=True, exist_ok=True)
    plt = require_matplotlib()

    x = [float(row["pca_x"]) for row in rows]
    y = [float(row["pca_y"]) for row in rows]
    labels = [int(row["cluster_id"]) for row in rows]
    fig, ax = plt.subplots(figsize=(9, 7))
    scatter = ax.scatter(x, y, c=labels, cmap="tab10", s=28, alpha=0.8)
    ax.set_title("CoT Feature Space by Cluster")
    ax.set_xlabel("PCA component 1")
    ax.set_ylabel("PCA component 2")
    fig.colorbar(scatter, ax=ax, label="Cluster ID; see cluster_label_key.png for labels")
    fig.tight_layout()
    fig.savefig(visual_dir / "feature_space_clusters.png", dpi=180)
    plt.close(fig)

    cluster_labels = [short_cluster_label(int(row["cluster_id"]), labels_json) for row in stats]
    sizes = [row["cluster_size"] for row in stats]
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(cluster_labels, sizes, color="#4C78A8")
    ax.set_title("Cluster Sizes by Failure Mode")
    ax.set_xlabel("Cluster ID: failure mode label")
    ax.set_ylabel("Prediction count")
    ax.tick_params(axis="x", labelrotation=75, labelsize=8)
    fig.tight_layout()
    fig.savefig(visual_dir / "cluster_sizes.png", dpi=180)
    plt.close(fig)

    avg_abs = [row["avg_absolute_error"] or 0 for row in stats]
    avg_rel = [row["avg_relative_error_percent"] or 0 for row in stats]
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    axes[0].bar(cluster_labels, avg_abs, color="#F58518")
    axes[0].set_title("Average Absolute Error by Failure Mode")
    axes[0].set_xlabel("Cluster ID: failure mode label")
    axes[0].set_ylabel("mg copper per serving")
    axes[1].bar(cluster_labels, avg_rel, color="#54A24B")
    axes[1].set_title("Average Relative Error by Failure Mode")
    axes[1].set_xlabel("Cluster ID: failure mode label")
    axes[1].set_ylabel("Percent")
    for axis in axes:
        axis.tick_params(axis="x", labelrotation=75, labelsize=8)
    fig.tight_layout()
    fig.savefig(visual_dir / "average_error_by_cluster.png", dpi=180)
    plt.close(fig)

    data = []
    box_labels = []
    for cluster_id in sorted(set(labels)):
        values = [
            coerce_float(row["absolute_error"])
            for row in rows
            if int(row["cluster_id"]) == cluster_id and coerce_float(row["absolute_error"]) is not None
        ]
        if values:
            data.append(values)
            box_labels.append(short_cluster_label(cluster_id, labels_json))
    fig, ax = plt.subplots(figsize=(15, 6))
    try:
        ax.boxplot(data, tick_labels=box_labels, showfliers=False)
    except TypeError:
        ax.boxplot(data, labels=box_labels, showfliers=False)
    ax.set_title("Absolute Error Distribution by Failure Mode")
    ax.set_xlabel("Cluster ID: failure mode label")
    ax.set_ylabel("mg copper per serving")
    ax.tick_params(axis="x", labelrotation=75, labelsize=8)
    fig.tight_layout()
    fig.savefig(visual_dir / "error_distribution_by_cluster.png", dpi=180)
    plt.close(fig)

    recipe_types = sorted(set(row["recipe_type"] for row in rows))
    cluster_ids = sorted(set(labels))
    matrix = []
    for recipe_type in recipe_types:
        row_values = []
        for cluster_id in cluster_ids:
            values = [
                coerce_float(row["absolute_error"])
                for row in rows
                if row["recipe_type"] == recipe_type
                and int(row["cluster_id"]) == cluster_id
                and coerce_float(row["absolute_error"]) is not None
            ]
            row_values.append(statistics.fmean(values) if values else float("nan"))
        matrix.append(row_values)
    fig, ax = plt.subplots(figsize=(16, max(5, len(recipe_types) * 0.55)))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_title("Failure Mode vs Recipe Type Error Intensity")
    ax.set_xticks(
        range(len(cluster_ids)),
        [short_cluster_label(cluster_id, labels_json, max_length=26) for cluster_id in cluster_ids],
    )
    ax.set_yticks(range(len(recipe_types)), recipe_types)
    ax.set_xlabel("Cluster ID: failure mode label")
    ax.tick_params(axis="x", labelrotation=75, labelsize=8)
    fig.colorbar(image, ax=ax, label="MAE")
    fig.tight_layout()
    fig.savefig(visual_dir / "failure_mode_recipe_type_heatmap.png", dpi=180)
    plt.close(fig)

    plot_cluster_label_key(labels_json, output_dir)


def enrich_error_cluster_analysis(
    stats: list[dict[str, Any]],
    labels: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for row in stats:
        label_info = labels.get(str(row["cluster_id"]), {})
        rows.append(
            {
                "cluster_id": row["cluster_id"],
                "failure_mode_label": label_info.get("label", ""),
                "failure_mode_description": label_info.get("description", ""),
                "cluster_size": row["cluster_size"],
                "avg_absolute_error": row["avg_absolute_error"],
                "median_absolute_error": row["median_absolute_error"],
                "avg_relative_error_percent": row["avg_relative_error_percent"],
                "median_relative_error_percent": row["median_relative_error_percent"],
                "avg_signed_error": row["avg_signed_error"],
                "overestimate_rate_percent": row["overestimate_rate_percent"],
                "underestimate_rate_percent": row["underestimate_rate_percent"],
                "best_performing_model": row["best_performing_model"],
                "worst_performing_model": row["worst_performing_model"],
                "model_diversity": row["model_diversity"],
                "top_models": row["top_models"],
                "recipe_type_distribution": row["recipe_type_distribution"],
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    truths = load_truths(args.recipes)
    stage1_errors = load_stage1_errors(args.stage1_predictions)
    model_files = sorted(args.model_outputs_dir.glob("*.jsonl"))
    if not model_files:
        raise SystemExit(f"No JSONL files found in {args.model_outputs_dir}")

    records = load_trial_records(model_files)
    rows = []
    for record in records:
        truth = truths.get(record.recipe_index)
        if truth is None:
            continue
        row = extract_features(record, truth)
        stage1_row = stage1_errors.get((record.model_name, record.recipe_index))
        if stage1_row and row["absolute_error"] == "":
            row["absolute_error"] = stage1_row.get("absolute_error", "")
            row["relative_error_percent"] = stage1_row.get("relative_error_percent", "")
        rows.append(row)

    matrix = numeric_matrix(rows)
    scaled, means, stds = standardize(matrix)
    labels, clustering_method, clustering_params = run_hdbscan_or_fallback(
        scaled,
        min_cluster_size=args.min_cluster_size,
    )
    coords = pca_2d(scaled)
    for index, row in enumerate(rows):
        row["cluster_id"] = int(labels[index])
        row["pca_x"] = float(coords[index, 0])
        row["pca_y"] = float(coords[index, 1])

    stats = cluster_statistics(rows)
    labels_json = label_clusters(rows, args.label_model, args.skip_llm_labels)
    error_analysis = enrich_error_cluster_analysis(stats, labels_json)
    score = silhouette_score(scaled, labels)

    base_fields = [
        "model_name",
        "model_file",
        "reasoning_level",
        "recipe_index",
        "recipe_type",
        "recipe_name",
        "predicted_copper_mg",
        "ground_truth_copper_mg",
        "signed_error",
        "absolute_error",
        "relative_error_percent",
        "success",
    ]
    write_csv(args.output_dir / "features_extracted.csv", rows, base_fields + FEATURE_COLUMNS)
    write_csv(
        args.output_dir / "clusters_with_labels.csv",
        rows,
        base_fields + ["cluster_id", "pca_x", "pca_y"] + FEATURE_COLUMNS,
    )
    write_csv(args.output_dir / "error_cluster_analysis.csv", error_analysis)
    write_json(args.output_dir / "failure_mode_labels.json", labels_json)
    write_json(
        args.output_dir / "clustering_metadata.json",
        {
            "trial_record_count": len(records),
            "feature_row_count": len(rows),
            "model_file_count": len(model_files),
            "model_files": [path.name for path in model_files],
            "recipe_count": len(truths),
            "feature_columns": FEATURE_COLUMNS,
            "clustering_method": clustering_method,
            "clustering_params": clustering_params,
            "cluster_count_excluding_noise": len({label for label in labels.tolist() if label != -1}),
            "noise_point_count": int(np.sum(labels == -1)),
            "silhouette_score": score,
            "standardization": {
                "means": dict(zip(FEATURE_COLUMNS, means.tolist(), strict=True)),
                "stds": dict(zip(FEATURE_COLUMNS, stds.tolist(), strict=True)),
            },
        },
    )

    if not args.skip_plots:
        plot_outputs(rows, stats, labels_json, args.output_dir)

    print(f"Extracted {len(rows)} feature rows from {len(model_files)} model files.")
    print(
        f"Clustering method: {clustering_method}; "
        f"clusters={len({label for label in labels.tolist() if label != -1})}, "
        f"noise={int(np.sum(labels == -1))}, "
        f"silhouette={score if score is not None else 'n/a'}"
    )
    print(f"Wrote clustering outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
