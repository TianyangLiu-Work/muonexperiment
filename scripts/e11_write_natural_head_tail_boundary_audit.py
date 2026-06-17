from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.reporting import markdown_table, write_markdown


RESULT_DIR = Path("results/e11_natural_head_tail_boundary")
DISCUSSION_PATH = Path("discussion/e11_natural_head_tail_boundary.md")


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    path: Path
    family: str
    natural_task: str
    scope: str


SOURCES: tuple[SourceSpec, ...] = (
    SourceSpec(
        "digits_one_step",
        Path("results/e11_long_tail_one_step/pair_summary.csv"),
        "sklearn_digits_mlp",
        "digits long-tail head/tail partition",
        "baseline natural matched-head-gain diagnostic",
    ),
    SourceSpec(
        "digits_imbalance_ablation",
        Path("results/e11_long_tail_imbalance_ablation/summary.csv"),
        "sklearn_digits_mlp",
        "digits long-tail imbalance sweep",
        "tail-frequency robustness",
    ),
    SourceSpec(
        "digits_checkpoint_sweep",
        Path("results/e11_long_tail_checkpoint_sweep/summary.csv"),
        "sklearn_digits_mlp",
        "digits long-tail checkpoint sweep",
        "checkpoint robustness",
    ),
    SourceSpec(
        "digits_class_partition_sweep",
        Path("results/e11_long_tail_class_partition_sweep/summary.csv"),
        "sklearn_digits_mlp",
        "digits long-tail class partitions",
        "class-partition robustness",
    ),
    SourceSpec(
        "digits_rho_sweep",
        Path("results/e11_long_tail_rho_sweep/summary.csv"),
        "sklearn_digits_mlp",
        "digits long-tail head-gain rho sweep",
        "target-head-gain robustness",
    ),
    SourceSpec(
        "cifar100_resnet_one_step_rho005",
        Path("results/e11_cifar100_resnet_one_step/pair_summary.csv"),
        "cifar100_lt_resnet18",
        "CIFAR-100-LT ResNet18 one-step rho=0.005",
        "baseline ResNet natural matched-head-gain diagnostic",
    ),
    SourceSpec(
        "cifar100_resnet_one_step_rho002",
        Path("results/e11_cifar100_resnet_one_step_rho002/pair_summary.csv"),
        "cifar100_lt_resnet18",
        "CIFAR-100-LT ResNet18 one-step rho=0.002",
        "smaller head-gain robustness",
    ),
    SourceSpec(
        "cifar100_resnet_checkpoint_sweep",
        Path("results/e11_cifar100_resnet_checkpoint_sweep/pair_summary.csv"),
        "cifar100_lt_resnet18",
        "CIFAR-100-LT ResNet18 checkpoint sweep",
        "checkpoint robustness",
    ),
    SourceSpec(
        "cifar100_resnet_tail_quality_control",
        Path("results/e11_cifar100_resnet_tail_quality_control/pair_summary.csv"),
        "cifar100_resnet18_tail_rich",
        "CIFAR-100 ResNet18 tail-quality control",
        "tail-quality control",
    ),
    SourceSpec(
        "cifar100_resnet_imbalance_sweep",
        Path("results/e11_cifar100_resnet_imbalance_sweep/pair_summary.csv"),
        "cifar100_lt_resnet18",
        "CIFAR-100-LT ResNet18 imbalance sweep",
        "tail-frequency robustness",
    ),
    SourceSpec(
        "cifar100_resnet_fc_condition_scatter",
        Path("results/e11_cifar100_resnet_fc_condition_scatter/summary.csv"),
        "cifar100_lt_resnet18_fc_only",
        "CIFAR-100-LT ResNet18 final-layer condition scatter",
        "final-layer downstream-aware condition diagnostic",
    ),
)


SETTING_FIELDS: tuple[str, ...] = (
    "dataset",
    "model",
    "warmup_steps",
    "head_train_per_class",
    "tail_train_per_class",
    "imbalance_ratio",
    "partition_name",
    "head_classes",
    "tail_classes",
    "target_head_gain_fraction",
)


RATIO_METRICS: tuple[dict[str, str], ...] = (
    {
        "metric_id": "tail_output_drift",
        "metric_role": "primary_full_tail_logit_vector_drift",
        "estimate": "geomean_tail_output_drift_sq_ratio_spectral_over_fro",
        "ci_low": "tail_output_drift_sq_ratio_ci95_low",
        "ci_high": "tail_output_drift_sq_ratio_ci95_high",
    },
    {
        "metric_id": "centered_tail_output_drift",
        "metric_role": "primary_variant_centered_tail_logit_vector_drift",
        "estimate": "geomean_centered_tail_output_drift_sq_ratio_spectral_over_fro",
        "ci_low": "centered_tail_output_drift_sq_ratio_ci95_low",
        "ci_high": "centered_tail_output_drift_sq_ratio_ci95_high",
    },
    {
        "metric_id": "true_logit_delta",
        "metric_role": "component_true_class_logit_drift",
        "estimate": "geomean_true_logit_delta_sq_ratio_spectral_over_fro",
        "ci_low": "true_logit_delta_sq_ratio_ci95_low",
        "ci_high": "true_logit_delta_sq_ratio_ci95_high",
    },
    {
        "metric_id": "competitor_logit_delta",
        "metric_role": "component_competitor_logit_drift",
        "estimate": "geomean_competitor_logit_delta_sq_ratio_spectral_over_fro",
        "ci_low": "competitor_logit_delta_sq_ratio_ci95_low",
        "ci_high": "competitor_logit_delta_sq_ratio_ci95_high",
    },
    {
        "metric_id": "margin_delta",
        "metric_role": "component_margin_delta_drift",
        "estimate": "geomean_margin_delta_sq_ratio_spectral_over_fro",
        "ci_low": "margin_delta_sq_ratio_ci95_low",
        "ci_high": "margin_delta_sq_ratio_ci95_high",
    },
)


DIFF_METRICS: tuple[dict[str, str], ...] = (
    {
        "metric_id": "tail_loss_increase",
        "metric_role": "secondary_tail_loss_tradeoff",
        "estimate": "mean_tail_loss_increase_diff_spectral_minus_fro",
        "ci_low": "tail_loss_increase_diff_ci95_low",
        "ci_high": "tail_loss_increase_diff_ci95_high",
        "positive_means": "spectral_higher_loss_increase",
    },
    {
        "metric_id": "tail_margin_drop",
        "metric_role": "secondary_tail_margin_tradeoff",
        "estimate": "mean_tail_margin_drop_diff_spectral_minus_fro",
        "ci_low": "tail_margin_drop_diff_ci95_low",
        "ci_high": "tail_margin_drop_diff_ci95_high",
        "positive_means": "spectral_higher_margin_drop",
    },
    {
        "metric_id": "tail_accuracy_drop",
        "metric_role": "secondary_tail_accuracy_tradeoff",
        "estimate": "mean_tail_accuracy_drop_diff_spectral_minus_fro",
        "ci_low": "tail_accuracy_drop_diff_ci95_low",
        "ci_high": "tail_accuracy_drop_diff_ci95_high",
        "positive_means": "spectral_higher_accuracy_drop",
    },
)


def _clean(value: object) -> str:
    if pd.isna(value):
        return "nan"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _setting_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in SETTING_FIELDS if column in frame.columns]


def _setting_values(row: pd.Series, columns: list[str]) -> str:
    if not columns:
        return "pooled"
    return ";".join(f"{column}={_clean(row[column])}" for column in columns)


def _setting_id(source_id: str, row: pd.Series, columns: list[str]) -> str:
    suffix = _setting_values(row, columns)
    return f"{source_id}::{suffix}"


def _ratio_status(ci_low: float, ci_high: float) -> tuple[str, str]:
    if pd.isna(ci_low) or pd.isna(ci_high):
        return "missing_ci", "ratio_ci_missing"
    if ci_high < 1.0:
        return "spectral_better", "ratio_ci_strictly_below_one"
    if ci_low > 1.0:
        return "spectral_worse", "ratio_ci_strictly_above_one"
    return "mixed_or_uncertain", "ratio_ci_crosses_or_touches_one"


def _diff_status(ci_low: float, ci_high: float) -> tuple[str, str]:
    if pd.isna(ci_low) or pd.isna(ci_high):
        return "missing_ci", "diff_ci_missing"
    if ci_high < 0.0:
        return "spectral_better", "diff_ci_strictly_below_zero"
    if ci_low > 0.0:
        return "spectral_worse", "diff_ci_strictly_above_zero"
    return "mixed_or_uncertain", "diff_ci_crosses_or_touches_zero"


def _present_metrics(frame: pd.DataFrame, metric_specs: tuple[dict[str, str], ...]) -> list[dict[str, str]]:
    columns = set(frame.columns)
    return [
        spec
        for spec in metric_specs
        if {spec["estimate"], spec["ci_low"], spec["ci_high"]}.issubset(columns)
    ]


def scan_sources() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    registry_rows: list[dict[str, object]] = []
    primary_rows: list[dict[str, object]] = []
    secondary_rows: list[dict[str, object]] = []

    for source in SOURCES:
        if not source.path.exists():
            registry_rows.append(
                {
                    "source_id": source.source_id,
                    "path": str(source.path),
                    "family": source.family,
                    "natural_task": source.natural_task,
                    "scope": source.scope,
                    "rows": 0,
                    "setting_axes": "missing",
                    "ratio_metric_count": 0,
                    "diff_metric_count": 0,
                    "primary_tail_drift_present": "no",
                    "status": "missing_source",
                }
            )
            continue

        frame = pd.read_csv(source.path)
        setting_columns = _setting_columns(frame)
        ratio_metrics = _present_metrics(frame, RATIO_METRICS)
        diff_metrics = _present_metrics(frame, DIFF_METRICS)
        registry_rows.append(
            {
                "source_id": source.source_id,
                "path": str(source.path),
                "family": source.family,
                "natural_task": source.natural_task,
                "scope": source.scope,
                "rows": len(frame),
                "setting_axes": ",".join(setting_columns) if setting_columns else "pooled",
                "ratio_metric_count": len(ratio_metrics),
                "diff_metric_count": len(diff_metrics),
                "primary_tail_drift_present": (
                    "yes"
                    if any(spec["metric_id"] == "tail_output_drift" for spec in ratio_metrics)
                    else "no"
                ),
                "status": "included",
            }
        )

        for row_index, row in frame.iterrows():
            setting_values = _setting_values(row, setting_columns)
            setting_id = _setting_id(source.source_id, row, setting_columns)
            base = {
                "source_id": source.source_id,
                "family": source.family,
                "natural_task": source.natural_task,
                "source_path": str(source.path),
                "row_index": int(row_index),
                "setting_id": setting_id,
                "setting_axes": ",".join(setting_columns) if setting_columns else "pooled",
                "setting_values": setting_values,
            }
            for spec in ratio_metrics:
                estimate = float(row[spec["estimate"]])
                ci_low = float(row[spec["ci_low"]])
                ci_high = float(row[spec["ci_high"]])
                status, interpretation = _ratio_status(ci_low, ci_high)
                claim_boundary = (
                    "strict_primary_tail_drift"
                    if spec["metric_id"] == "tail_output_drift"
                    else "component_metric_boundary"
                )
                primary_rows.append(
                    {
                        **base,
                        "metric_id": spec["metric_id"],
                        "metric_role": spec["metric_role"],
                        "estimate_column": spec["estimate"],
                        "ratio_spectral_over_fro": estimate,
                        "ci95_low": ci_low,
                        "ci95_high": ci_high,
                        "boundary_status": status,
                        "interpretation": interpretation,
                        "claim_boundary": claim_boundary,
                    }
                )
            for spec in diff_metrics:
                estimate = float(row[spec["estimate"]])
                ci_low = float(row[spec["ci_low"]])
                ci_high = float(row[spec["ci_high"]])
                status, interpretation = _diff_status(ci_low, ci_high)
                secondary_rows.append(
                    {
                        **base,
                        "metric_id": spec["metric_id"],
                        "metric_role": spec["metric_role"],
                        "estimate_column": spec["estimate"],
                        "mean_diff_spectral_minus_fro": estimate,
                        "ci95_low": ci_low,
                        "ci95_high": ci_high,
                        "boundary_status": status,
                        "interpretation": interpretation,
                        "positive_means": spec["positive_means"],
                        "claim_boundary": "secondary_outcome_tradeoff",
                    }
                )

    return (
        pd.DataFrame(registry_rows),
        pd.DataFrame(primary_rows),
        pd.DataFrame(secondary_rows),
    )


def _status_counts(frame: pd.DataFrame) -> dict[str, int]:
    counts = frame["boundary_status"].value_counts().to_dict() if not frame.empty else {}
    return {
        "spectral_better_count": int(counts.get("spectral_better", 0)),
        "spectral_worse_count": int(counts.get("spectral_worse", 0)),
        "mixed_or_uncertain_count": int(counts.get("mixed_or_uncertain", 0)),
        "missing_ci_count": int(counts.get("missing_ci", 0)),
    }


def _summary_row(summary_id: str, frame: pd.DataFrame, claim_status: str) -> dict[str, object]:
    if frame.empty:
        return {
            "summary_id": summary_id,
            "row_count": 0,
            "source_count": 0,
            "metric_count": 0,
            "min_estimate": float("nan"),
            "max_estimate": float("nan"),
            "max_ci95_high": float("nan"),
            "min_ci95_low": float("nan"),
            **_status_counts(frame),
            "claim_status": claim_status,
        }
    estimate_column = (
        "ratio_spectral_over_fro"
        if "ratio_spectral_over_fro" in frame.columns
        else "mean_diff_spectral_minus_fro"
    )
    return {
        "summary_id": summary_id,
        "row_count": int(len(frame)),
        "source_count": int(frame["source_id"].nunique()),
        "metric_count": int(frame["metric_id"].nunique()),
        "min_estimate": float(frame[estimate_column].min()),
        "max_estimate": float(frame[estimate_column].max()),
        "max_ci95_high": float(frame["ci95_high"].max()),
        "min_ci95_low": float(frame["ci95_low"].min()),
        **_status_counts(frame),
        "claim_status": claim_status,
    }


def make_summary(primary: pd.DataFrame, secondary: pd.DataFrame) -> pd.DataFrame:
    primary_tail = primary[primary["metric_id"].eq("tail_output_drift")]
    component = primary[~primary["metric_id"].eq("tail_output_drift")]
    secondary_worse = secondary[secondary["boundary_status"].eq("spectral_worse")]
    primary_tail_worse = primary_tail[primary_tail["boundary_status"].eq("spectral_worse")]
    component_worse = component[component["boundary_status"].eq("spectral_worse")]

    rows = [
        _summary_row(
            "primary_tail_output_drift",
            primary_tail,
            (
                "strict_natural_primary_counterexample_found"
                if not primary_tail_worse.empty
                else "no_strict_natural_primary_counterexample_in_committed_scan"
            ),
        ),
        _summary_row(
            "all_ratio_metrics",
            primary,
            (
                "ratio_metric_boundary_candidates_found"
                if not primary[primary["boundary_status"].eq("spectral_worse")].empty
                else "no_ratio_metric_boundary_candidates_in_committed_scan"
            ),
        ),
        _summary_row(
            "component_ratio_metrics",
            component,
            (
                "component_boundary_candidates_found"
                if not component_worse.empty
                else "no_component_boundary_candidates_in_committed_scan"
            ),
        ),
        _summary_row(
            "secondary_tail_outcomes",
            secondary,
            (
                "secondary_outcome_tradeoffs_found"
                if not secondary_worse.empty
                else "no_strict_secondary_outcome_tradeoffs_in_committed_scan"
            ),
        ),
    ]
    return pd.DataFrame(rows)


def make_candidates(primary: pd.DataFrame, secondary: pd.DataFrame) -> pd.DataFrame:
    primary_candidates = primary[primary["boundary_status"].eq("spectral_worse")].copy()
    primary_candidates["scan"] = "ratio"
    primary_candidates["estimate"] = primary_candidates["ratio_spectral_over_fro"]
    secondary_candidates = secondary[secondary["boundary_status"].eq("spectral_worse")].copy()
    secondary_candidates["scan"] = "secondary_outcome"
    secondary_candidates["estimate"] = secondary_candidates["mean_diff_spectral_minus_fro"]
    columns = [
        "scan",
        "claim_boundary",
        "source_id",
        "family",
        "setting_id",
        "setting_values",
        "metric_id",
        "metric_role",
        "estimate",
        "ci95_low",
        "ci95_high",
        "boundary_status",
        "interpretation",
    ]
    candidates = pd.concat([primary_candidates, secondary_candidates], ignore_index=True)
    if candidates.empty:
        return pd.DataFrame(columns=columns)
    return candidates[columns].sort_values(
        ["claim_boundary", "scan", "source_id", "metric_id", "estimate"],
        ascending=[True, True, True, True, False],
    )


def write_outputs(
    registry: pd.DataFrame,
    primary: pd.DataFrame,
    secondary: pd.DataFrame,
    summary: pd.DataFrame,
    candidates: pd.DataFrame,
) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    registry.to_csv(RESULT_DIR / "search_registry.csv", index=False)
    primary.to_csv(RESULT_DIR / "primary_drift_scan.csv", index=False)
    secondary.to_csv(RESULT_DIR / "secondary_outcome_scan.csv", index=False)
    summary.to_csv(RESULT_DIR / "boundary_summary.csv", index=False)
    candidates.to_csv(RESULT_DIR / "candidate_negative_cases.csv", index=False)

    primary_tail = primary[primary["metric_id"].eq("tail_output_drift")]
    worst_primary = primary_tail.sort_values(["ci95_high", "ratio_spectral_over_fro"], ascending=False).head(8)
    ratio_candidates = candidates[candidates["scan"].eq("ratio")]
    secondary_candidates = candidates[candidates["scan"].eq("secondary_outcome")]
    primary_status = summary.set_index("summary_id").loc["primary_tail_output_drift", "claim_status"]

    if candidates.empty:
        candidate_text = "No strict negative candidate was found in the committed natural-task scan."
    else:
        candidate_text = markdown_table(
            candidates.head(16),
            [
                "scan",
                "claim_boundary",
                "source_id",
                "setting_values",
                "metric_id",
                "estimate",
                "ci95_low",
                "ci95_high",
            ],
        )

    ratio_candidate_text = (
        "No ratio-metric negative candidate was found."
        if ratio_candidates.empty
        else markdown_table(
            ratio_candidates,
            [
                "claim_boundary",
                "source_id",
                "setting_values",
                "metric_id",
                "estimate",
                "ci95_low",
                "ci95_high",
            ],
        )
    )
    secondary_candidate_text = (
        "No strict secondary outcome tradeoff was found."
        if secondary_candidates.empty
        else markdown_table(
            secondary_candidates.head(12),
            [
                "source_id",
                "setting_values",
                "metric_id",
                "estimate",
                "ci95_low",
                "ci95_high",
            ],
        )
    )

    text = f"""# E11 Natural Head-to-Tail Boundary Audit

This generated audit scans the already committed natural matched-head-gain
diagnostics for settings where the spectral/polar direction is worse than the
Frobenius direction. It is a fixed-rule scan over existing result tables, not a
replacement for a future pre-registered natural negative-search experiment.

## Boundary Rule

- Primary natural counterexample: the full tail-output logit-vector squared
  drift ratio has CI lower endpoint above 1.
- Component boundary candidate: a true-logit, competitor-logit, centered-drift,
  or margin-delta ratio has CI lower endpoint above 1.
- Secondary outcome tradeoff: a spectral-minus-Frobenius tail loss increase,
  margin drop, or accuracy drop has CI lower endpoint above 0.

The current narrow paper claim is about local full tail-output drift at matched
head gain. Component and outcome tradeoffs constrain stronger loss, margin,
accuracy, or predictive-score claims, but they do not by themselves falsify the
primary full-drift statement.

## Search Registry

{markdown_table(registry, ["source_id", "family", "rows", "setting_axes", "ratio_metric_count", "diff_metric_count", "status"])}

## Summary

{markdown_table(summary, ["summary_id", "row_count", "source_count", "metric_count", "spectral_better_count", "spectral_worse_count", "mixed_or_uncertain_count", "claim_status"])}

Primary status: `{primary_status}`.

## Worst Primary Full-Drift Settings

{markdown_table(worst_primary, ["source_id", "setting_values", "ratio_spectral_over_fro", "ci95_low", "ci95_high", "boundary_status"])}

## Strict Negative Candidates

{candidate_text}

## Ratio-Metric Boundary Candidates

{ratio_candidate_text}

## Secondary Outcome Tradeoffs

{secondary_candidate_text}

## Claim Boundary

This audit supports only the following scoped statement: among the committed
natural matched-head-gain diagnostics scanned here, the primary full tail-output
drift metric has the status shown above. If the paper needs natural negative
examples, the next step is a pre-registered search over additional natural
architectures, data partitions, and checkpoints using the same CI rules and
reporting table.
"""
    write_markdown(DISCUSSION_PATH, text)


def main() -> None:
    registry, primary, secondary = scan_sources()
    summary = make_summary(primary, secondary)
    candidates = make_candidates(primary, secondary)
    write_outputs(registry, primary, secondary, summary, candidates)
    print(f"saved natural head-to-tail boundary audit to {RESULT_DIR} and {DISCUSSION_PATH}")


if __name__ == "__main__":
    main()
