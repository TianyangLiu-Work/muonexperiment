from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from statistics import NormalDist

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.reporting import fmt, markdown_table, write_markdown


RESULT_ROOT = Path("results/e11_cifar100_resnet_lt_tuned_benchmark")
SELECTION_DIR = RESULT_ROOT / "validation_selection"
POWER_DIR = RESULT_ROOT / "final_power_audit"
OUTPUT_DIR = RESULT_ROOT / "variance_prior_audit"
DISCUSSION_PATH = Path("discussion/e11_cifar100_resnet_lt_tuned_benchmark_variance_prior_audit.md")
ASSUMED_PAIRED_DIFF_SD = 0.03


def t_critical(df: int, two_sided_alpha: float) -> float:
    try:
        from scipy.stats import t

        return float(t.ppf(1.0 - two_sided_alpha / 2.0, df))
    except Exception:
        return NormalDist().inv_cdf(1.0 - two_sided_alpha / 2.0)


def infer_sd_from_ci(ci_low: float, ci_high: float, n: int) -> float:
    half_width = (float(ci_high) - float(ci_low)) / 2.0
    crit = t_critical(n - 1, 0.05)
    return half_width * math.sqrt(n) / crit


def mde_for_sd(sd: float, final_seed_count: int, family_size: int, familywise: bool) -> float:
    alpha = 0.05 / family_size if familywise else 0.05
    crit = t_critical(final_seed_count - 1, alpha)
    return crit * sd / math.sqrt(final_seed_count)


def final_outputs_present() -> bool:
    final_root = RESULT_ROOT / "final_claim"
    if not final_root.exists():
        return False
    return any(final_root.rglob("*"))


def load_run_registry() -> pd.DataFrame:
    registry_path = SELECTION_DIR / "run_registry.csv"
    if registry_path.exists():
        return pd.read_csv(registry_path)
    return pd.DataFrame()


def build_validation_setting_variance(run_registry: pd.DataFrame) -> pd.DataFrame:
    registry_lookup: dict[str, dict[str, object]] = {}
    if not run_registry.empty and "setting_id" in run_registry:
        registry_lookup = run_registry.set_index("setting_id").to_dict("index")

    rows: list[dict[str, object]] = []
    for group_path in sorted((RESULT_ROOT / "validation_tuning").glob("*/group_metrics.csv")):
        setting_id = group_path.parent.name
        summary_path = group_path.with_name("summary.csv")
        if not summary_path.exists():
            continue
        group_metrics = pd.read_csv(group_path)
        summary = pd.read_csv(summary_path)
        registry_row = registry_lookup.get(setting_id, {})
        for frequency_group, frame in group_metrics.groupby("frequency_group", sort=True):
            n = int(frame["seed"].nunique())
            summary_row = summary[summary["frequency_group"].eq(frequency_group)].iloc[0]
            direct_sd = float(frame["balanced_accuracy"].std(ddof=1)) if n > 1 else 0.0
            ci_sd = infer_sd_from_ci(
                float(summary_row["balanced_accuracy_ci95_low"]),
                float(summary_row["balanced_accuracy_ci95_high"]),
                n,
            )
            rows.append(
                {
                    "setting_id": setting_id,
                    "array_index": registry_row.get("array_index", ""),
                    "recipe_family": registry_row.get("recipe_family", ""),
                    "recipe_name": registry_row.get("recipe_name", str(summary_row["recipe"])),
                    "frequency_group": frequency_group,
                    "validation_seed_count": n,
                    "mean_balanced_accuracy": float(summary_row["mean_balanced_accuracy"]),
                    "seed_sd_balanced_accuracy": direct_sd,
                    "ci_implied_sd_balanced_accuracy": ci_sd,
                    "variance_role": "validation-only within-setting seed variability; not a final paired-diff estimate",
                }
            )
    return pd.DataFrame(rows)


def build_spent_pilot_paired_variance() -> pd.DataFrame:
    sources = [
        (
            "augmented_recipe_pilot",
            Path("results/e11_cifar100_resnet_lt_recipe_benchmark/pair_summary.csv"),
        ),
        (
            "negative_ns_muon_pilot",
            Path("results/e11_cifar100_resnet_lt_muon_final_benchmark/pair_summary.csv"),
        ),
    ]
    rows: list[dict[str, object]] = []
    for source_id, path in sources:
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        for row in frame.to_dict("records"):
            n = int(row["seeds"])
            inferred_sd = infer_sd_from_ci(
                float(row["balanced_accuracy_diff_ci95_low"]),
                float(row["balanced_accuracy_diff_ci95_high"]),
                n,
            )
            rows.append(
                {
                    "source_id": source_id,
                    "recipe": row["recipe"],
                    "baseline_recipe": row["baseline_recipe"],
                    "frequency_group": row["frequency_group"],
                    "seed_count": n,
                    "mean_balanced_accuracy_diff": float(row["mean_balanced_accuracy_diff"]),
                    "balanced_accuracy_diff_ci95_low": float(row["balanced_accuracy_diff_ci95_low"]),
                    "balanced_accuracy_diff_ci95_high": float(row["balanced_accuracy_diff_ci95_high"]),
                    "ci_implied_paired_diff_sd": inferred_sd,
                    "variance_role": "spent pilot paired-diff variability; context only, cannot select final recipes",
                }
            )
    return pd.DataFrame(rows)


def quantile_rows(source_id: str, frame: pd.DataFrame, value_col: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for frequency_group, group in frame.groupby("frequency_group", sort=True):
        values = group[value_col].dropna().astype(float)
        if values.empty:
            continue
        rows.append(
            {
                "source_id": source_id,
                "frequency_group": frequency_group,
                "row_count": int(values.shape[0]),
                "sd_p50": float(values.quantile(0.50)),
                "sd_p80": float(values.quantile(0.80)),
                "sd_p95": float(values.quantile(0.95)),
                "sd_max": float(values.max()),
                "assumed_paired_diff_sd": ASSUMED_PAIRED_DIFF_SD,
                "p80_within_assumed_0p03": "yes"
                if float(values.quantile(0.80)) <= ASSUMED_PAIRED_DIFF_SD
                else "no",
                "interpretation": (
                    "within-setting seed SD is a loose variance prior"
                    if source_id == "validation_within_setting"
                    else "spent pilot paired-diff SD is the closest available pre-final paired-diff prior"
                ),
            }
        )
    return rows


def build_variance_prior_summary(
    validation_variance: pd.DataFrame,
    pilot_variance: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rows.extend(
        quantile_rows(
            "validation_within_setting",
            validation_variance,
            "seed_sd_balanced_accuracy",
        )
    )
    rows.extend(
        quantile_rows(
            "spent_pilot_paired_diff",
            pilot_variance,
            "ci_implied_paired_diff_sd",
        )
    )
    return pd.DataFrame(rows)


def build_mde_sensitivity(summary: pd.DataFrame) -> pd.DataFrame:
    final_design = pd.read_csv(POWER_DIR / "final_family_design.csv")
    comparison_plan = pd.read_csv(POWER_DIR / "primary_comparison_plan.csv")
    final_seed_count = int(final_design["final_seed_count"].iloc[0])
    family_size = int(comparison_plan["family_size"].iloc[0])
    rows: list[dict[str, object]] = []
    for row in summary.to_dict("records"):
        for label, value in (
            ("p50", row["sd_p50"]),
            ("p80", row["sd_p80"]),
            ("p95", row["sd_p95"]),
            ("max", row["sd_max"]),
            ("assumed_0p03", ASSUMED_PAIRED_DIFF_SD),
        ):
            sd = float(value)
            holm_mde = mde_for_sd(sd, final_seed_count, family_size, familywise=True)
            raw_mde = mde_for_sd(sd, final_seed_count, family_size, familywise=False)
            rows.append(
                {
                    "source_id": row["source_id"],
                    "frequency_group": row["frequency_group"],
                    "sd_reference": label,
                    "paired_diff_sd": sd,
                    "final_seed_count": final_seed_count,
                    "primary_comparison_family_size": family_size,
                    "raw_single_comparison_mde": raw_mde,
                    "holm_worst_case_mde": holm_mde,
                    "all_class_noninferiority_margin": -0.01,
                    "minimum_mean_all_diff_to_pass_guardrail": -0.01 + holm_mde,
                    "claim_effect": "MDE sensitivity only; final benchmark claim remains blocked until validation selection and final seeds complete",
                }
            )
    return pd.DataFrame(rows)


def build_gate_matrix(
    validation_variance: pd.DataFrame,
    pilot_variance: pd.DataFrame,
    sensitivity: pd.DataFrame,
) -> pd.DataFrame:
    final_present = final_outputs_present()
    few_pilot = pilot_variance[pilot_variance["frequency_group"].eq("few")]
    few_p80 = float(few_pilot["ci_implied_paired_diff_sd"].quantile(0.80)) if not few_pilot.empty else float("nan")
    return pd.DataFrame(
        [
            {
                "gate_id": "EVPA-1-final-seed-quarantine",
                "status": "fail" if final_present else "pass",
                "evidence": "no files under results/e11_cifar100_resnet_lt_tuned_benchmark/final_claim"
                if not final_present
                else "final_claim files are present",
                "claim_effect": "variance prior remains pre-final and cannot inspect final seeds",
            },
            {
                "gate_id": "EVPA-2-validation-sd-surface",
                "status": "pass" if len(validation_variance) >= 4 else "fail",
                "evidence": f"{len(validation_variance)} validation frequency-group variance rows",
                "claim_effect": "completed validation settings provide within-setting seed variability",
            },
            {
                "gate_id": "EVPA-3-spent-pilot-paired-sd-surface",
                "status": "pass" if len(pilot_variance) >= 4 else "fail",
                "evidence": f"{len(pilot_variance)} spent-pilot paired-diff variance rows",
                "claim_effect": "paired-diff SD prior is anchored to pre-final spent pilot context",
            },
            {
                "gate_id": "EVPA-4-primary-few-sd-anchor",
                "status": "pass" if not few_pilot.empty else "fail",
                "evidence": "few-class spent-pilot paired-diff SD p80="
                + (fmt(few_p80) if not math.isnan(few_p80) else "missing"),
                "claim_effect": "primary few-class MDE can be compared with empirical pre-final variability",
            },
            {
                "gate_id": "EVPA-5-mde-sensitivity-grid",
                "status": "pass" if {"p50", "p80", "p95", "max", "assumed_0p03"}.issubset(set(sensitivity["sd_reference"])) else "fail",
                "evidence": "MDE grid includes empirical p50/p80/p95/max and assumed_0p03 references",
                "claim_effect": "reviewers can see how final detectability changes under empirical variance priors",
            },
            {
                "gate_id": "EVPA-6-claim-boundary",
                "status": "pass",
                "evidence": "audit reads validation-only and spent-pilot summaries, not final benchmark outputs",
                "claim_effect": "no benchmark-performance claim is authorized by this audit",
            },
        ]
    )


def write_discussion(
    validation_variance: pd.DataFrame,
    pilot_variance: pd.DataFrame,
    summary: pd.DataFrame,
    sensitivity: pd.DataFrame,
    gates: pd.DataFrame,
) -> None:
    few_pilot_summary = summary[
        summary["source_id"].eq("spent_pilot_paired_diff") & summary["frequency_group"].eq("few")
    ]
    if few_pilot_summary.empty:
        few_line = "few-class spent-pilot paired-diff SD is unavailable"
    else:
        row = few_pilot_summary.iloc[0]
        few_line = (
            f"few-class spent-pilot paired-diff SD p50/p80/p95 = "
            f"{fmt(row['sd_p50'])}/{fmt(row['sd_p80'])}/{fmt(row['sd_p95'])}"
        )
    sensitivity_view = sensitivity[
        sensitivity["frequency_group"].isin(["few", "all"])
        & sensitivity["sd_reference"].isin(["p80", "p95", "assumed_0p03"])
    ].copy()
    text = f"""# E11 CIFAR-100-LT Tuned Benchmark Variance Prior Audit

This generated audit calibrates the tuned benchmark power assumptions against pre-final evidence. It reads completed validation summaries and spent pilot paired comparisons only; it does not inspect or authorize final seed outputs.

Current anchor: {few_line}. The existing power audit uses paired-diff SD `{ASSUMED_PAIRED_DIFF_SD}` as a reference. This audit keeps that value as a sensitivity point and adds empirical p50/p80/p95/max references from validation-only within-setting seed variability and spent-pilot paired-diff variability.

## Gate Matrix

{markdown_table(gates, ["gate_id", "status", "evidence", "claim_effect"])}

## Variance Prior Summary

{markdown_table(summary, ["source_id", "frequency_group", "row_count", "sd_p50", "sd_p80", "sd_p95", "sd_max", "assumed_paired_diff_sd", "p80_within_assumed_0p03", "interpretation"])}

## MDE Sensitivity Snapshot

{markdown_table(sensitivity_view, ["source_id", "frequency_group", "sd_reference", "paired_diff_sd", "holm_worst_case_mde", "minimum_mean_all_diff_to_pass_guardrail", "claim_effect"])}

## Boundary

Allowed now: use this as a variance-prior and detectable-effect sensitivity audit for the registered tuned benchmark protocol.

Blocked now: using validation variability, spent pilot variability, or the assumed SD grid to choose recipes, inspect final seeds, or claim tuned optimizer performance.

Artifacts:
- [validation_setting_variance.csv](../{(OUTPUT_DIR / 'validation_setting_variance.csv').as_posix()})
- [spent_pilot_paired_variance.csv](../{(OUTPUT_DIR / 'spent_pilot_paired_variance.csv').as_posix()})
- [variance_prior_summary.csv](../{(OUTPUT_DIR / 'variance_prior_summary.csv').as_posix()})
- [mde_sensitivity_from_empirical_sd.csv](../{(OUTPUT_DIR / 'mde_sensitivity_from_empirical_sd.csv').as_posix()})
- [gate_matrix.csv](../{(OUTPUT_DIR / 'gate_matrix.csv').as_posix()})
- [config.json](../{(OUTPUT_DIR / 'config.json').as_posix()})
"""
    write_markdown(DISCUSSION_PATH, text)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_registry = load_run_registry()
    validation_variance = build_validation_setting_variance(run_registry)
    pilot_variance = build_spent_pilot_paired_variance()
    summary = build_variance_prior_summary(validation_variance, pilot_variance)
    sensitivity = build_mde_sensitivity(summary)
    gates = build_gate_matrix(validation_variance, pilot_variance, sensitivity)

    validation_variance.to_csv(OUTPUT_DIR / "validation_setting_variance.csv", index=False)
    pilot_variance.to_csv(OUTPUT_DIR / "spent_pilot_paired_variance.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "variance_prior_summary.csv", index=False)
    sensitivity.to_csv(OUTPUT_DIR / "mde_sensitivity_from_empirical_sd.csv", index=False)
    gates.to_csv(OUTPUT_DIR / "gate_matrix.csv", index=False)
    (OUTPUT_DIR / "config.json").write_text(
        json.dumps(
            {
                "purpose": "pre-final empirical variance-prior audit for tuned benchmark MDE assumptions",
                "result_root": RESULT_ROOT.as_posix(),
                "final_power_audit": POWER_DIR.as_posix(),
                "assumed_paired_diff_sd": ASSUMED_PAIRED_DIFF_SD,
                "final_seed_outputs_inspected": False,
                "final_outputs_present": final_outputs_present(),
                "claim_boundary": "variance-prior sensitivity only; no recipe selection or final-performance claim",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    write_discussion(validation_variance, pilot_variance, summary, sensitivity, gates)
    if not gates["status"].astype(str).eq("pass").all():
        raise AssertionError(f"variance-prior gates must pass before commit: {gates.to_dict('records')}")
    print(f"saved tuned benchmark variance-prior audit to {OUTPUT_DIR} and {DISCUSSION_PATH}")


if __name__ == "__main__":
    main()
