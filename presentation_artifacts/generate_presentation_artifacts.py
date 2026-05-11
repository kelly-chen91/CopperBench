#!/usr/bin/env python3
"""Generate presentation-level figures and summary tables for CopperBench."""

from __future__ import annotations

import csv
import os
from collections import defaultdict
from pathlib import Path

MPLCONFIGDIR = Path("/private/tmp/copperbench_mplconfig")
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "presentation_artifacts"


STAGE1_MODEL = "gpt_5_4_mini__reasoning_medium"
STAGE2_MODEL = "gpt_5_4_mini__reasoning_medium"
STAGE3_MODEL = "combined_persona_few_shot__gpt_5_4_mini__reasoning_medium"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def to_float(value: str) -> float:
    return float(value.strip())


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def stage1_predictions() -> list[dict[str, object]]:
    rows = read_csv(
        ROOT / "stage1_recipe_proportions" / "evaluation_results" / "per_recipe_predictions.csv"
    )
    predictions = []
    for row in rows:
        if row["model_name"] != STAGE1_MODEL or row["valid"] != "True":
            continue
        truth = to_float(row["ground_truth_copper_mg_per_serving"])
        pred = to_float(row["predicted_copper_mg_per_serving"])
        predictions.append(
            {
                "recipe_index": int(row["recipe_index"]),
                "recipe_name": row["recipe_name"],
                "recipe_type": row["recipe_type"],
                "truth": truth,
                "prediction": pred,
                "absolute_error": abs(pred - truth),
                "signed_error": pred - truth,
            }
        )
    return predictions


def averaged_replicate_predictions(
    runs_dir: Path,
    model_name: str,
    evaluation_subdir: str = "evaluation_results_20_percent",
) -> list[dict[str, object]]:
    grouped: dict[int, dict[str, object]] = {}
    predictions_by_recipe: dict[int, list[float]] = defaultdict(list)

    for run_dir in sorted(runs_dir.glob("run_*")):
        per_recipe_path = run_dir / evaluation_subdir / "per_recipe_predictions.csv"
        if not per_recipe_path.exists():
            continue
        for row in read_csv(per_recipe_path):
            if row["model_name"] != model_name or row["valid"] != "True":
                continue
            recipe_index = int(row["recipe_index"])
            truth = to_float(row["ground_truth_copper_mg_per_serving"])
            pred = to_float(row["predicted_copper_mg_per_serving"])
            grouped[recipe_index] = {
                "recipe_index": recipe_index,
                "recipe_name": row["recipe_name"],
                "recipe_type": row["recipe_type"],
                "truth": truth,
            }
            predictions_by_recipe[recipe_index].append(pred)

    averaged = []
    for recipe_index, preds in sorted(predictions_by_recipe.items()):
        base = grouped[recipe_index]
        pred = mean(preds)
        truth = float(base["truth"])
        averaged.append(
            {
                **base,
                "prediction": pred,
                "absolute_error": abs(pred - truth),
                "signed_error": pred - truth,
            }
        )
    return averaged


def metrics_from_predictions(predictions: list[dict[str, object]]) -> dict[str, float]:
    abs_errors = [float(row["absolute_error"]) for row in predictions]
    signed_errors = [float(row["signed_error"]) for row in predictions]
    within_20 = [
        abs(float(row["prediction"]) - float(row["truth"])) <= 0.2 * float(row["truth"])
        for row in predictions
    ]
    return {
        "mae": mean(abs_errors),
        "mean_signed_error": mean(signed_errors),
        "accuracy_20": 100 * sum(within_20) / len(within_20),
        "valid_predictions": len(predictions),
    }


def stage2_official_metrics() -> dict[str, float]:
    rows = read_csv(ROOT / "stage2.5_combined_prompts" / "replicate_runs" / "replicate_metrics.csv")
    rows = [row for row in rows if row["model_name"] == STAGE2_MODEL and row["prompt_variant"] == "Baseline"]
    return {
        "mae": mean([to_float(row["MAE"]) for row in rows]),
        "mean_signed_error": mean([to_float(row["mean_signed_error_mg_per_serving"]) for row in rows]),
        "accuracy_20": mean([to_float(row["accuracy_20_percent"]) for row in rows]),
        "valid_predictions": int(rows[0]["valid_predictions_count"]),
    }


def stage3_official_metrics() -> dict[str, float]:
    rows = read_csv(ROOT / "stage3_recipe_proportions" / "replicate_runs" / "replicate_metrics.csv")
    rows = [row for row in rows if row["model_name"] == STAGE3_MODEL]

    per_recipe_rows = read_csv(
        ROOT / "stage3_recipe_proportions" / "replicate_runs" / "per_recipe_replicate_metrics.csv"
    )
    signed_by_run: dict[int, list[float]] = defaultdict(list)
    for row in per_recipe_rows:
        if row["model_name"] != STAGE3_MODEL:
            continue
        signed_by_run[int(row["run"])].append(to_float(row["SAE"]))

    return {
        "mae": mean([to_float(row["MAE"]) for row in rows]),
        "mean_signed_error": mean([mean(values) for values in signed_by_run.values()]),
        "accuracy_20": mean([to_float(row["accuracy_20_percent"]) for row in rows]),
        "valid_predictions": int(rows[0]["valid_predictions_count"]),
    }


def write_summary_table(stage_rows: list[dict[str, object]]) -> None:
    csv_path = OUT_DIR / "stage_progression_summary.csv"
    md_path = OUT_DIR / "stage_progression_summary.md"

    fieldnames = [
        "stage",
        "input_condition",
        "system",
        "run_count",
        "valid_predictions",
        "mae_mg_per_serving",
        "mean_signed_error_mg_per_serving",
        "accuracy_within_20_percent",
        "presentation_takeaway",
    ]

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in stage_rows:
            writer.writerow({key: row[key] for key in fieldnames})

    headers = [
        "Stage",
        "Input condition",
        "System",
        "Runs",
        "MAE",
        "Mean signed error",
        "+/-20% accuracy",
        "Takeaway",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for row in stage_rows:
        lines.append(
            "| {stage} | {input_condition} | `{system}` | {run_count} | "
            "{mae_mg_per_serving:.4f} | {mean_signed_error_mg_per_serving:+.4f} | "
            "{accuracy_within_20_percent:.2f}% | {presentation_takeaway} |".format(**row)
        )
    md_path.write_text("\n".join(lines) + "\n")


def plot_stage_progression(stage_rows: list[dict[str, object]]) -> None:
    labels = [row["stage"] for row in stage_rows]
    maes = [float(row["mae_mg_per_serving"]) for row in stage_rows]
    acc20 = [float(row["accuracy_within_20_percent"]) for row in stage_rows]
    colors = ["#355C7D", "#C06C84", "#2A9D8F"]

    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    bars = ax.bar(labels, maes, color=colors, width=0.58)
    ax.set_title("CopperBench Stage Progression", fontsize=16, pad=14)
    ax.set_ylabel("Mean absolute error (mg copper per serving)")
    ax.set_ylim(0, max(maes) * 1.28)
    ax.grid(axis="y", linestyle="--", alpha=0.32)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, mae, acc in zip(bars, maes, acc20):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(maes) * 0.03,
            f"MAE {mae:.4f}\n{acc:.1f}% within 20%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.text(
        0.5,
        -0.18,
        "Stage 1 and Stage 2 use the same model baseline; Stage 3 uses the best targeted prompt.",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        color="#444444",
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "stage_progression_mae.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_prediction_scatter(stage_series: list[tuple[str, list[dict[str, object]], str]]) -> None:
    all_values = []
    for _, rows, _ in stage_series:
        for row in rows:
            all_values.append(float(row["truth"]))
            all_values.append(float(row["prediction"]))
    axis_max = max(all_values) * 1.08

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharex=True, sharey=True)
    for ax, (label, rows, color) in zip(axes, stage_series):
        truths = [float(row["truth"]) for row in rows]
        preds = [float(row["prediction"]) for row in rows]
        ax.scatter(truths, preds, s=48, color=color, alpha=0.78, edgecolor="white", linewidth=0.6)
        ax.plot([0, axis_max], [0, axis_max], color="#333333", linestyle="--", linewidth=1)
        ax.set_title(label, fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        worst = max(rows, key=lambda row: float(row["absolute_error"]))
        ax.annotate(
            str(worst["recipe_name"]).split(" with ")[0],
            xy=(float(worst["truth"]), float(worst["prediction"])),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
            color="#333333",
        )

    axes[0].set_ylabel("Predicted copper (mg/serving)")
    for ax in axes:
        ax.set_xlabel("Ground truth copper (mg/serving)")
        ax.set_xlim(0, axis_max)
        ax.set_ylim(0, axis_max)

    fig.suptitle("Prediction vs. Ground Truth by Stage", fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "prediction_vs_ground_truth_by_stage.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    stage1 = stage1_predictions()
    stage2 = averaged_replicate_predictions(
        ROOT / "stage2.5_combined_prompts" / "replicate_runs",
        STAGE2_MODEL,
    )
    stage3 = averaged_replicate_predictions(
        ROOT / "stage3_recipe_proportions" / "replicate_runs",
        STAGE3_MODEL,
    )

    stage1_metrics = metrics_from_predictions(stage1)
    stage2_metrics = stage2_official_metrics()
    stage3_metrics = stage3_official_metrics()

    stage_rows = [
        {
            "stage": "Stage 1",
            "input_condition": "Full recipes with ingredient proportions",
            "system": STAGE1_MODEL,
            "run_count": 1,
            "valid_predictions": stage1_metrics["valid_predictions"],
            "mae_mg_per_serving": stage1_metrics["mae"],
            "mean_signed_error_mg_per_serving": stage1_metrics["mean_signed_error"],
            "accuracy_within_20_percent": stage1_metrics["accuracy_20"],
            "presentation_takeaway": "Strong baseline when quantities are available.",
        },
        {
            "stage": "Stage 2",
            "input_condition": "Ingredient names only; no proportions",
            "system": STAGE2_MODEL,
            "run_count": 3,
            "valid_predictions": stage2_metrics["valid_predictions"],
            "mae_mg_per_serving": stage2_metrics["mae"],
            "mean_signed_error_mg_per_serving": stage2_metrics["mean_signed_error"],
            "accuracy_within_20_percent": stage2_metrics["accuracy_20"],
            "presentation_takeaway": "Removing quantities roughly doubles MAE and increases overestimation.",
        },
        {
            "stage": "Stage 3",
            "input_condition": "Full recipes with targeted prompt mitigation",
            "system": STAGE3_MODEL,
            "run_count": 3,
            "valid_predictions": stage3_metrics["valid_predictions"],
            "mae_mg_per_serving": stage3_metrics["mae"],
            "mean_signed_error_mg_per_serving": stage3_metrics["mean_signed_error"],
            "accuracy_within_20_percent": stage3_metrics["accuracy_20"],
            "presentation_takeaway": "Failure-mode-informed prompting gives the best MAE.",
        },
    ]

    write_summary_table(stage_rows)
    plot_stage_progression(stage_rows)
    plot_prediction_scatter(
        [
            ("Stage 1: proportions baseline", stage1, "#355C7D"),
            ("Stage 2: no proportions baseline", stage2, "#C06C84"),
            ("Stage 3: targeted prompt", stage3, "#2A9D8F"),
        ]
    )

    print("Generated:")  # noqa: T201
    for path in [
        OUT_DIR / "stage_progression_mae.png",
        OUT_DIR / "prediction_vs_ground_truth_by_stage.png",
        OUT_DIR / "stage_progression_summary.csv",
        OUT_DIR / "stage_progression_summary.md",
    ]:
        print(f"- {path.relative_to(ROOT)}")  # noqa: T201


if __name__ == "__main__":
    main()
