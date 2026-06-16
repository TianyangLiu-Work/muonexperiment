from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.reporting import fmt, markdown_table, write_markdown
from e11_condition_geometry.statistics import ci95, corr_ci95
from scripts.e11_write_cifar100_resnet_condition_score_next import FEATURE_COLUMNS
from scripts.e11_write_cifar100_resnet_condition_score_next import add_protocol_features
from scripts.e11_write_cifar100_resnet_condition_score_next import predict_residual
from scripts.e11_write_cifar100_resnet_condition_score_next import standardized_theory_score
from scripts.e11_write_cifar100_resnet_condition_score_next import target_residual_from_source_depth


CONDITION_SCORE_DIR = Path("results/e11_cifar100_resnet_condition_score_next")
SOURCE_LAYER_SUMMARY = Path("results/e11_cifar100_resnet_layer_jvp_checkpoint_prediction/layer_summary.csv")
COEFFICIENTS_PATH = CONDITION_SCORE_DIR / "calibration_coefficients.csv"
OUTPUT_DIR = CONDITION_SCORE_DIR / "heldout_score_evaluation"
FIGURE_DIR = Path("figures/e11_cifar100_resnet_condition_score_next/heldout_score_evaluation")
DISCUSSION_PATH = Path("discussion/e11_cifar100_resnet_condition_score_next_heldout_evaluation.md")


@dataclass(frozen=True)
class HeldoutSplit:
    split_id: str
    role: str
    display_name: str
    layer_summary_path: Path
    config_path: Path


HELDOUT_SPLITS = (
    HeldoutSplit(
        split_id="primary_heldout_architecture_resnet34",
        role="primary_heldout_architecture",
        display_name="ResNet34 CIFAR-100-LT held-out architecture",
        layer_summary_path=CONDITION_SCORE_DIR / "heldout_architecture" / "layer_summary.csv",
        config_path=CONDITION_SCORE_DIR / "heldout_architecture" / "config.json",
    ),
    HeldoutSplit(
        split_id="primary_heldout_data_cifar10lt_resnet18",
        role="primary_heldout_data",
        display_name="ResNet18 CIFAR-10-LT held-out data",
        layer_summary_path=CONDITION_SCORE_DIR / "heldout_data" / "layer_summary.csv",
        config_path=CONDITION_SCORE_DIR / "heldout_data" / "config.json",
    ),
)


def load_frozen_fits(coefficients: pd.DataFrame) -> dict[int, dict[str, object]]:
    fits: dict[int, dict[str, object]] = {}
    feature_names = list(FEATURE_COLUMNS)
    for source_step, group in coefficients.groupby("source_warmup_steps", observed=True, sort=True):
        by_term = group.set_index("term")
        missing_terms = [
            term
            for term in ("depth_intercept", "depth_slope_log_early_layer_prior", "residual_intercept", *feature_names)
            if term not in by_term.index
        ]
        if missing_terms:
            raise ValueError(f"coefficient table for source step {source_step} missing terms: {missing_terms}")
        fits[int(source_step)] = {
            "depth_intercept": float(by_term.loc["depth_intercept", "coefficient"]),
            "depth_slope": float(by_term.loc["depth_slope_log_early_layer_prior", "coefficient"]),
            "residual_intercept": float(by_term.loc["residual_intercept", "coefficient"]),
            "feature_mean": np.array([float(by_term.loc[name, "feature_mean"]) for name in feature_names]),
            "feature_std": np.array([float(by_term.loc[name, "feature_std"]) for name in feature_names]),
            "beta": np.array([float(by_term.loc[name, "coefficient"]) for name in feature_names]),
        }
    if not fits:
        raise ValueError("coefficient table did not contain any frozen source fits")
    return fits


def _target_top_k(target: pd.DataFrame, target_residual: pd.Series) -> set[str]:
    return set(
        pd.DataFrame({"parameter": target["parameter"], "target_residual": target_residual})
        .nlargest(min(5, len(target)), "target_residual")["parameter"]
        .astype(str)
    )


def _score_rows_for_frame(
    *,
    split: HeldoutSplit,
    source_step: int,
    target_step: int,
    target: pd.DataFrame,
    target_residual: pd.Series,
    fit: dict[str, object],
    score_values: dict[str, np.ndarray],
    predicted_log_values: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    target_below_one = target["geomean_observed_tail_drift_sq_ratio_spectral_over_fro"].astype(float) < 1.0
    target_top5 = _target_top_k(target, target_residual)
    target_series = pd.Series(target_residual.to_numpy(dtype=float))
    for score_name, values in score_values.items():
        score_series = pd.Series(np.asarray(values, dtype=float))
        spearman, spearman_low, spearman_high, points = corr_ci95(
            score_series,
            target_series,
            method="spearman",
        )
        pearson, pearson_low, pearson_high, _ = corr_ci95(
            score_series,
            target_series,
            method="pearson",
        )
        predicted_top5 = set(
            pd.DataFrame({"parameter": target["parameter"], "score": values})
            .nlargest(min(5, len(target)), "score")["parameter"]
            .astype(str)
        )
        if score_name in predicted_log_values:
            predicted_below_one = np.asarray(predicted_log_values[score_name], dtype=float) < 0.0
            threshold_accuracy = float((predicted_below_one == target_below_one.to_numpy()).mean())
            predicted_below_one_fraction = float(predicted_below_one.mean())
        else:
            threshold_accuracy = math.nan
            predicted_below_one_fraction = math.nan
        rows.append(
            {
                "split_id": split.split_id,
                "split_role": split.role,
                "split_display_name": split.display_name,
                "source_warmup_steps": int(source_step),
                "target_warmup_steps": int(target_step),
                "score": score_name,
                "score_family": (
                    "primary_candidate"
                    if score_name == "condition_score_v2_calibrated_residual"
                    else "positive_control"
                    if score_name == "source_observed_drift_positive_control"
                    else "baseline"
                ),
                "points": int(points),
                "spearman_score_vs_target_residual": spearman,
                "spearman_ci95_low": spearman_low,
                "spearman_ci95_high": spearman_high,
                "pearson_score_vs_target_residual": pearson,
                "pearson_ci95_low": pearson_low,
                "pearson_ci95_high": pearson_high,
                "top5_residual_risk_overlap_fraction": len(predicted_top5 & target_top5)
                / max(len(target_top5), 1),
                "threshold_below_one_accuracy": threshold_accuracy,
                "target_observed_below_one_fraction": float(target_below_one.mean()),
                "predicted_below_one_fraction": predicted_below_one_fraction,
                "frozen_depth_intercept": float(fit["depth_intercept"]),
                "frozen_depth_slope": float(fit["depth_slope"]),
            }
        )
    return rows


def score_heldout_split(
    split: HeldoutSplit,
    heldout_layer_summary: pd.DataFrame,
    source_layer_summary: pd.DataFrame,
    frozen_fits: dict[int, dict[str, object]],
) -> pd.DataFrame:
    target_frame = add_protocol_features(heldout_layer_summary)
    source_frame = add_protocol_features(source_layer_summary)
    rows: list[dict[str, object]] = []

    for source_step, fit in frozen_fits.items():
        source = source_frame[source_frame["warmup_steps"].eq(source_step)].copy()
        source_control = pd.DataFrame()
        if not source.empty:
            source_residual = source["log_observed"].astype(float) - (
                float(fit["depth_intercept"])
                + float(fit["depth_slope"]) * source["log_early_layer_prior"].astype(float)
            )
            source_control = pd.DataFrame(
                {
                    "parameter": source["parameter"].astype(str).to_numpy(),
                    "source_observed_residual_positive_control": source_residual.to_numpy(dtype=float),
                    "source_log_observed_positive_control": source["log_observed"].to_numpy(dtype=float),
                }
            )
        for target_step in sorted(int(value) for value in target_frame["warmup_steps"].unique()):
            target = target_frame[target_frame["warmup_steps"].eq(target_step)].copy()
            target_residual = target_residual_from_source_depth(
                target,
                float(fit["depth_intercept"]),
                float(fit["depth_slope"]),
            )
            primary_residual = predict_residual(target, fit)
            predicted_log_observed = (
                float(fit["depth_intercept"])
                + float(fit["depth_slope"]) * target["log_early_layer_prior"].to_numpy(dtype=float)
                + primary_residual
            )
            score_values = {
                "condition_score_v2_calibrated_residual": primary_residual,
                "early_layer_prior": target["log_early_layer_prior"].to_numpy(dtype=float),
                "legacy_scaled_jvp_ratio": target["log_scaled_jvp_ratio"].to_numpy(dtype=float),
                "theory_sign_composite": standardized_theory_score(target, fit),
            }
            predicted_log_values = {
                "condition_score_v2_calibrated_residual": predicted_log_observed,
                "legacy_scaled_jvp_ratio": target["log_scaled_jvp_ratio"].to_numpy(dtype=float),
            }
            rows.extend(
                _score_rows_for_frame(
                    split=split,
                    source_step=source_step,
                    target_step=target_step,
                    target=target,
                    target_residual=target_residual,
                    fit=fit,
                    score_values=score_values,
                    predicted_log_values=predicted_log_values,
                )
            )

            if not source_control.empty:
                matched = target[["parameter"]].astype({"parameter": str}).merge(
                    source_control,
                    on="parameter",
                    how="inner",
                )
                if not matched.empty:
                    target_matched = target[target["parameter"].astype(str).isin(set(matched["parameter"]))].copy()
                    target_matched = matched[["parameter"]].merge(target_matched, on="parameter", how="inner")
                    target_residual_matched = target_residual_from_source_depth(
                        target_matched,
                        float(fit["depth_intercept"]),
                        float(fit["depth_slope"]),
                    )
                    rows.extend(
                        _score_rows_for_frame(
                            split=split,
                            source_step=source_step,
                            target_step=target_step,
                            target=target_matched,
                            target_residual=target_residual_matched,
                            fit=fit,
                            score_values={
                                "source_observed_drift_positive_control": matched[
                                    "source_observed_residual_positive_control"
                                ].to_numpy(dtype=float)
                            },
                            predicted_log_values={
                                "source_observed_drift_positive_control": matched[
                                    "source_log_observed_positive_control"
                                ].to_numpy(dtype=float)
                            },
                        )
                    )
    return pd.DataFrame(rows)


def summarize_heldout_scores(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (split_id, score), group in pairs.groupby(["split_id", "score"], observed=True, sort=False):
        spearman, spearman_low, spearman_high = ci95(group["spearman_score_vs_target_residual"])
        pearson, pearson_low, pearson_high = ci95(group["pearson_score_vs_target_residual"])
        top5, top5_low, top5_high = ci95(group["top5_residual_risk_overlap_fraction"])
        threshold, threshold_low, threshold_high = ci95(group["threshold_below_one_accuracy"])
        if math.isfinite(top5_low):
            top5_low = max(0.0, top5_low)
        if math.isfinite(top5_high):
            top5_high = min(1.0, top5_high)
        if math.isfinite(threshold_low):
            threshold_low = max(0.0, threshold_low)
        if math.isfinite(threshold_high):
            threshold_high = min(1.0, threshold_high)
        rows.append(
            {
                "split_id": str(split_id),
                "split_role": str(group["split_role"].iloc[0]),
                "split_display_name": str(group["split_display_name"].iloc[0]),
                "score": str(score),
                "score_family": str(group["score_family"].iloc[0]),
                "split_transfer_pairs": int(len(group)),
                "mean_points": float(group["points"].mean()),
                "mean_spearman_score_vs_target_residual": spearman,
                "spearman_ci95_low": spearman_low,
                "spearman_ci95_high": spearman_high,
                "mean_pearson_score_vs_target_residual": pearson,
                "pearson_ci95_low": pearson_low,
                "pearson_ci95_high": pearson_high,
                "mean_top5_residual_risk_overlap_fraction": top5,
                "top5_residual_risk_overlap_ci95_low": top5_low,
                "top5_residual_risk_overlap_ci95_high": top5_high,
                "mean_threshold_below_one_accuracy": threshold,
                "threshold_below_one_accuracy_ci95_low": threshold_low,
                "threshold_below_one_accuracy_ci95_high": threshold_high,
            }
        )
    return pd.DataFrame(rows)


def heldout_gate_report(summary: pd.DataFrame, splits: tuple[HeldoutSplit, ...]) -> pd.DataFrame:
    rows = []
    all_primary_pass = True
    for split in splits:
        split_summary = (
            summary[summary["split_id"].eq(split.split_id)]
            if "split_id" in summary.columns
            else pd.DataFrame(columns=summary.columns)
        )
        primary = split_summary[split_summary["score"].eq("condition_score_v2_calibrated_residual")]
        early = split_summary[split_summary["score"].eq("early_layer_prior")]
        source_control = split_summary[split_summary["score"].eq("source_observed_drift_positive_control")]
        if primary.empty:
            all_primary_pass = False
            rows.append(
                {
                    "gate_id": f"{split.role}_generated",
                    "scope": split.display_name,
                    "status": "not_run",
                    "evidence": f"missing held-out layer summary at {split.layer_summary_path}",
                }
            )
            continue
        primary_row = primary.iloc[0]
        residual_pass = float(primary_row["spearman_ci95_low"]) > 0.0
        threshold_pass = float(primary_row["mean_threshold_below_one_accuracy"]) >= 0.8
        baseline_pass = not early.empty and float(
            primary_row["mean_spearman_score_vs_target_residual"]
        ) > float(early.iloc[0]["mean_spearman_score_vs_target_residual"])
        all_primary_pass = all_primary_pass and residual_pass and threshold_pass and baseline_pass
        rows.extend(
            [
                {
                    "gate_id": f"{split.role}_residual_spearman",
                    "scope": split.display_name,
                    "status": "pass" if residual_pass else "fail",
                    "evidence": (
                        f"primary residual Spearman={fmt(primary_row['mean_spearman_score_vs_target_residual'])} "
                        f"CI=[{fmt(primary_row['spearman_ci95_low'])}, {fmt(primary_row['spearman_ci95_high'])}]"
                    ),
                },
                {
                    "gate_id": f"{split.role}_threshold_accuracy",
                    "scope": split.display_name,
                    "status": "pass" if threshold_pass else "fail",
                    "evidence": (
                        "primary below-one threshold accuracy="
                        f"{fmt(primary_row['mean_threshold_below_one_accuracy'])}"
                    ),
                },
                {
                    "gate_id": f"{split.role}_early_prior_comparison",
                    "scope": split.display_name,
                    "status": "pass" if baseline_pass else "fail",
                    "evidence": (
                        f"primary Spearman={fmt(primary_row['mean_spearman_score_vs_target_residual'])}; "
                        f"early-layer prior Spearman="
                        f"{fmt(early.iloc[0]['mean_spearman_score_vs_target_residual']) if not early.empty else 'missing'}"
                    ),
                },
                {
                    "gate_id": f"{split.role}_source_observed_control_reported",
                    "scope": split.display_name,
                    "status": "reported" if not source_control.empty else "not_applicable",
                    "evidence": (
                        f"matched-parameter source-observed control points={fmt(source_control.iloc[0]['mean_points'])}"
                        if not source_control.empty
                        else "no matched source parameters available for this split"
                    ),
                },
            ]
        )
    rows.append(
        {
            "gate_id": "p0_predictive_condition_heldout_claim",
            "scope": "registered P0 condition-score held-out splits",
            "status": "pass" if all_primary_pass and len(rows) > 1 else "not_ready",
            "evidence": "All generated primary held-out splits must pass residual Spearman and threshold gates.",
        }
    )
    return pd.DataFrame(rows)


def write_figure(summary: pd.DataFrame, figure_dir: Path) -> Path:
    figure_dir.mkdir(parents=True, exist_ok=True)
    if "score" in summary.columns:
        primary = summary[summary["score"].eq("condition_score_v2_calibrated_residual")].copy()
        baselines = summary[summary["score"].isin(["early_layer_prior", "legacy_scaled_jvp_ratio"])].copy()
    else:
        primary = pd.DataFrame()
        baselines = pd.DataFrame()
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8))
    if not primary.empty:
        y_positions = list(range(len(primary)))
        estimates = primary["mean_spearman_score_vs_target_residual"].to_numpy(dtype=float)
        low = primary["spearman_ci95_low"].to_numpy(dtype=float)
        high = primary["spearman_ci95_high"].to_numpy(dtype=float)
        axes[0].barh(y_positions, estimates, color="#009E73", alpha=0.88)
        axes[0].errorbar(
            estimates,
            y_positions,
            xerr=[estimates - low, high - estimates],
            fmt="none",
            color="black",
            capsize=3,
            linewidth=1,
        )
        axes[0].set_yticks(y_positions)
        axes[0].set_yticklabels(primary["split_role"].tolist(), fontsize=8)
    axes[0].axvline(0.0, color="black", linestyle="--", linewidth=1)
    axes[0].set_xlabel("held-out residual Spearman")
    axes[0].set_title("Primary condition-score gate")

    if not baselines.empty:
        labels = [f"{row.split_role}: {row.score}" for row in baselines.itertuples()]
        y_positions = list(range(len(baselines)))
        axes[1].barh(
            y_positions,
            baselines["mean_spearman_score_vs_target_residual"].to_numpy(dtype=float),
            color="#0072B2",
            alpha=0.82,
        )
        axes[1].set_yticks(y_positions)
        axes[1].set_yticklabels(labels, fontsize=7)
    axes[1].axvline(0.0, color="black", linestyle="--", linewidth=1)
    axes[1].set_xlabel("held-out residual Spearman")
    axes[1].set_title("Baseline comparisons")
    fig.suptitle("Registered condition-score v2 held-out evaluation")
    fig.tight_layout()
    path = figure_dir / "cifar100_resnet_condition_score_heldout_evaluation.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def write_discussion(
    summary: pd.DataFrame,
    gates: pd.DataFrame,
    figure_path: Path,
    discussion_path: Path,
) -> None:
    text = f"""# E11 CIFAR ResNet Condition-Score v2 Held-Out Evaluation

This generated report applies the frozen `condition_score_v2_calibrated_residual`
coefficients from the registered ResNet18 checkpoint analysis to the primary
held-out architecture and held-out data splits. It does not refit coefficients
on the held-out target results.

![Held-out condition-score evaluation](../{figure_path.as_posix()})

## Held-Out Summary

{markdown_table(summary, ["split_role", "score", "score_family", "split_transfer_pairs", "mean_points", "mean_spearman_score_vs_target_residual", "spearman_ci95_low", "spearman_ci95_high", "mean_top5_residual_risk_overlap_fraction", "mean_threshold_below_one_accuracy"])}

## Gate Report

{markdown_table(gates, ["gate_id", "scope", "status", "evidence"])}

## Boundary

This report can support the P0 predictive-condition claim only if both primary
held-out splits pass the residual-Spearman and threshold gates. A failed or
missing split keeps the paper at the current mechanism-only claim boundary.

Artifacts:
- [heldout_score_pairs.csv](../{(OUTPUT_DIR / 'heldout_score_pairs.csv').as_posix()})
- [heldout_score_summary.csv](../{(OUTPUT_DIR / 'heldout_score_summary.csv').as_posix()})
- [heldout_gate_report.csv](../{(OUTPUT_DIR / 'heldout_gate_report.csv').as_posix()})
- [config.json](../{(OUTPUT_DIR / 'config.json').as_posix()})
"""
    write_markdown(discussion_path, text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-layer-summary", type=Path, default=SOURCE_LAYER_SUMMARY)
    parser.add_argument("--coefficients", type=Path, default=COEFFICIENTS_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--figure-dir", type=Path, default=FIGURE_DIR)
    parser.add_argument("--discussion-path", type=Path, default=DISCUSSION_PATH)
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Write a not-run gate report instead of failing when held-out layer summaries are missing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_layer_summary = pd.read_csv(args.source_layer_summary)
    coefficients = pd.read_csv(args.coefficients)
    frozen_fits = load_frozen_fits(coefficients)

    missing = [split for split in HELDOUT_SPLITS if not split.layer_summary_path.exists()]
    if missing and not args.allow_missing:
        raise FileNotFoundError(
            "missing held-out layer summaries: "
            + ", ".join(split.layer_summary_path.as_posix() for split in missing)
        )

    pair_frames = []
    split_configs = {}
    for split in HELDOUT_SPLITS:
        if not split.layer_summary_path.exists():
            continue
        heldout_layer_summary = pd.read_csv(split.layer_summary_path)
        pair_frames.append(score_heldout_split(split, heldout_layer_summary, source_layer_summary, frozen_fits))
        if split.config_path.exists():
            split_configs[split.split_id] = json.loads(split.config_path.read_text(encoding="utf-8"))

    if pair_frames:
        pairs = pd.concat(pair_frames, ignore_index=True)
        summary = summarize_heldout_scores(pairs)
    else:
        pairs = pd.DataFrame(columns=["split_id", "split_role", "source_warmup_steps", "target_warmup_steps", "score"])
        summary = pd.DataFrame(
            columns=[
                "split_id",
                "split_role",
                "split_display_name",
                "score",
                "score_family",
                "split_transfer_pairs",
                "mean_points",
                "mean_spearman_score_vs_target_residual",
                "spearman_ci95_low",
                "spearman_ci95_high",
                "mean_top5_residual_risk_overlap_fraction",
                "mean_threshold_below_one_accuracy",
            ]
        )
    gates = heldout_gate_report(summary, HELDOUT_SPLITS)

    pairs.to_csv(args.output_dir / "heldout_score_pairs.csv", index=False)
    summary.to_csv(args.output_dir / "heldout_score_summary.csv", index=False)
    gates.to_csv(args.output_dir / "heldout_gate_report.csv", index=False)
    (args.output_dir / "config.json").write_text(
        json.dumps(
            {
                "source_layer_summary": args.source_layer_summary.as_posix(),
                "coefficients": args.coefficients.as_posix(),
                "heldout_splits": [
                    {
                        "split_id": split.split_id,
                        "role": split.role,
                        "display_name": split.display_name,
                        "layer_summary_path": split.layer_summary_path.as_posix(),
                        "generated": split.layer_summary_path.exists(),
                    }
                    for split in HELDOUT_SPLITS
                ],
                "split_configs": split_configs,
                "feature_columns": FEATURE_COLUMNS,
                "frozen_source_fits": sorted(frozen_fits),
                "allow_missing": bool(args.allow_missing),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    figure_path = write_figure(summary, args.figure_dir)
    write_discussion(summary, gates, figure_path, args.discussion_path)
    print(f"saved held-out condition-score evaluation to {args.output_dir}")
    print(f"figure: {figure_path}")
    print(f"discussion: {args.discussion_path}")
    if not summary.empty:
        print(summary.to_string(index=False))
    print(gates.to_string(index=False))


if __name__ == "__main__":
    main()
