from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.reporting import fmt, markdown_table, write_markdown
from scripts.e11_run_natural_negative_search_phase1 import (
    DEFAULT_SEEDS,
    MULTIPLICITY_FAMILY,
    PHASE1_SEARCH_IDS,
    SEARCH_OUTPUT_PREFIXES,
    all_settings,
    settings_registry,
)


RESULT_DIR = Path("results/e11_natural_negative_search_protocol")
OUTPUT_DIR = RESULT_DIR / "phase1_multiplicity_evaluation"
DISCUSSION_PATH = Path("discussion/e11_natural_negative_search_phase1_evaluation.md")
ALPHA = 0.05
HEAD_GAIN_RELATIVE_ERROR_MAX = 0.05
TAIL_ACCURACY_FLOOR_BY_DATASET = {
    "CIFAR100": 0.02,
    "CIFAR10": 0.11,
}


DECISION_COLUMNS = [
    "search_id",
    "setting_id",
    "output_status",
    "phase",
    "multiplicity_family",
    "planned_family_size",
    "dataset_name",
    "partition_id",
    "warmup_steps",
    "target_head_gain_fraction",
    "seed_count",
    "observed_seeds",
    "metric_id",
    "inference_source",
    "estimate",
    "raw_ci95_low",
    "raw_ci95_high",
    "log_ratio_mean",
    "log_ratio_se",
    "raw_one_sided_p",
    "holm_adjusted_one_sided_p",
    "bonferroni_simultaneous_ci95_low",
    "raw_ci_worse",
    "head_gain_gate",
    "tail_quality_gate",
    "quality_gate",
    "adjustment_scope_status",
    "adjusted_primary_decision",
    "claim_status",
]

SEED_RATIO_COLUMNS = [
    "search_id",
    "setting_id",
    "phase",
    "multiplicity_family",
    "planned_family_size",
    "dataset",
    "model",
    "partition_id",
    "warmup_steps",
    "target_head_gain_fraction_registered",
    "seed",
    "frobenius_tail_output_drift_fro",
    "spectral_tail_output_drift_fro",
    "tail_output_drift_sq_ratio_spectral_over_fro",
    "log_tail_output_drift_sq_ratio_spectral_over_fro",
]


def t_ppf(probability: float, df: int) -> float:
    try:
        from scipy.stats import t

        return float(t.ppf(probability, df))
    except Exception:
        return 1.644854 if probability <= 0.95 else 1.96


def t_sf(value: float, df: int) -> float:
    try:
        from scipy.stats import t

        return float(t.sf(value, df))
    except Exception:
        return 0.5 * math.erfc(value / math.sqrt(2.0))


def infer_log_ratio_test(row: pd.Series, *, family_size: int) -> tuple[float, float, float]:
    estimate = float(row["geomean_tail_output_drift_sq_ratio_spectral_over_fro"])
    low = float(row["tail_output_drift_sq_ratio_ci95_low"])
    high = float(row["tail_output_drift_sq_ratio_ci95_high"])
    seeds = int(row["seeds"])
    if not (np.isfinite(estimate) and np.isfinite(low) and np.isfinite(high) and estimate > 0 and low > 0 and high > 0):
        return math.nan, math.nan, math.nan
    if seeds <= 1 or abs(math.log(high) - math.log(low)) <= 1e-15:
        raw_p = 0.0 if estimate > 1.0 and low > 1.0 else 1.0
        return raw_p, low, math.nan
    df = max(seeds - 1, 1)
    tcrit_raw = t_ppf(0.975, df)
    se = (math.log(high) - math.log(low)) / max(2.0 * tcrit_raw, 1e-300)
    if se <= 0 or not np.isfinite(se):
        return math.nan, math.nan, math.nan
    tstat = math.log(estimate) / se
    raw_p = t_sf(tstat, df)
    tcrit_bonferroni = t_ppf(1.0 - ALPHA / max(family_size, 1), df)
    simultaneous_low = math.exp(math.log(estimate) - tcrit_bonferroni * se)
    return raw_p, simultaneous_low, se


def log_ratio_test_from_seed_rows(seed_rows: pd.DataFrame, *, family_size: int) -> dict[str, float]:
    logs = (
        seed_rows["log_tail_output_drift_sq_ratio_spectral_over_fro"]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .to_numpy(dtype=float)
    )
    if logs.size == 0:
        return {
            "estimate": math.nan,
            "raw_ci95_low": math.nan,
            "raw_ci95_high": math.nan,
            "raw_one_sided_p": math.nan,
            "bonferroni_simultaneous_ci95_low": math.nan,
            "log_ratio_mean": math.nan,
            "log_ratio_se": math.nan,
        }
    mean = float(logs.mean())
    estimate = math.exp(mean)
    if logs.size == 1:
        raw_p = 0.0 if mean > 0.0 else 1.0
        return {
            "estimate": estimate,
            "raw_ci95_low": estimate,
            "raw_ci95_high": estimate,
            "raw_one_sided_p": raw_p,
            "bonferroni_simultaneous_ci95_low": estimate,
            "log_ratio_mean": mean,
            "log_ratio_se": 0.0,
        }
    df = int(logs.size - 1)
    se = float(logs.std(ddof=1)) / math.sqrt(float(logs.size))
    if se <= 0.0 or not np.isfinite(se):
        raw_p = 0.0 if mean > 0.0 else 1.0
        return {
            "estimate": estimate,
            "raw_ci95_low": estimate,
            "raw_ci95_high": estimate,
            "raw_one_sided_p": raw_p,
            "bonferroni_simultaneous_ci95_low": estimate,
            "log_ratio_mean": mean,
            "log_ratio_se": 0.0,
        }
    tcrit_raw = t_ppf(0.975, df)
    raw_ci_low = math.exp(mean - tcrit_raw * se)
    raw_ci_high = math.exp(mean + tcrit_raw * se)
    raw_p = t_sf(mean / se, df)
    tcrit_bonferroni = t_ppf(1.0 - ALPHA / max(family_size, 1), df)
    simultaneous_low = math.exp(mean - tcrit_bonferroni * se)
    return {
        "estimate": estimate,
        "raw_ci95_low": raw_ci_low,
        "raw_ci95_high": raw_ci_high,
        "raw_one_sided_p": raw_p,
        "bonferroni_simultaneous_ci95_low": simultaneous_low,
        "log_ratio_mean": mean,
        "log_ratio_se": se,
    }


def holm_adjust(raw_p_values: pd.Series, *, family_size: int) -> pd.Series:
    adjusted = pd.Series(np.nan, index=raw_p_values.index, dtype=float)
    finite = raw_p_values.replace([np.inf, -np.inf], np.nan).dropna().sort_values()
    running = 0.0
    for rank, (idx, p_value) in enumerate(finite.items(), start=1):
        multiplier = max(family_size - rank + 1, 1)
        running = max(running, min(float(p_value) * multiplier, 1.0))
        adjusted.loc[idx] = min(running, 1.0)
    return adjusted


def expected_registry() -> pd.DataFrame:
    return settings_registry(all_settings(), seeds=DEFAULT_SEEDS)


def expected_output_files(prefix: Path) -> dict[str, Path]:
    return {
        "settings_registry": prefix / "settings_registry.csv",
        "step_metrics": prefix / "step_metrics.csv",
        "pair_summary": prefix / "pair_summary.csv",
        "layer_metrics": prefix / "layer_metrics.csv",
        "decision_template": prefix / "decision_template.csv",
        "config": prefix / "config.json",
    }


def build_run_registry(expected: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for search_id in PHASE1_SEARCH_IDS:
        prefix = SEARCH_OUTPUT_PREFIXES[search_id]
        files = expected_output_files(prefix)
        expected_rows = int(expected["search_id"].eq(search_id).sum())
        settings_rows = len(pd.read_csv(files["settings_registry"])) if files["settings_registry"].exists() else 0
        pair_rows = len(pd.read_csv(files["pair_summary"])) if files["pair_summary"].exists() else 0
        required_metric_files = ["step_metrics", "pair_summary", "layer_metrics", "decision_template", "config"]
        metric_files_present = all(files[name].exists() for name in required_metric_files)
        if metric_files_present and pair_rows == expected_rows:
            status = "complete"
        elif files["settings_registry"].exists() and not any(files[name].exists() for name in required_metric_files):
            status = "settings_only_no_metrics"
        elif any(path.exists() for path in files.values()):
            status = "incomplete_metrics"
        else:
            status = "not_run"
        rows.append(
            {
                "search_id": search_id,
                "artifact_prefix": prefix.as_posix(),
                "expected_settings": expected_rows,
                "settings_registry_rows": settings_rows,
                "pair_summary_rows": pair_rows,
                "output_status": status,
                "settings_registry_path": files["settings_registry"].as_posix(),
                "pair_summary_path": files["pair_summary"].as_posix(),
            }
        )
    return pd.DataFrame(rows)


def load_observed_pair_summary() -> pd.DataFrame:
    frames = []
    for search_id, prefix in SEARCH_OUTPUT_PREFIXES.items():
        path = prefix / "pair_summary.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if "search_id" not in frame.columns:
            frame["search_id"] = search_id
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_observed_step_metrics() -> pd.DataFrame:
    frames = []
    for search_id, prefix in SEARCH_OUTPUT_PREFIXES.items():
        path = prefix / "step_metrics.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if "search_id" not in frame.columns:
            frame["search_id"] = search_id
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=SEED_RATIO_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def seed_level_primary_ratios(step_metrics: pd.DataFrame) -> pd.DataFrame:
    if step_metrics.empty:
        return pd.DataFrame(columns=SEED_RATIO_COLUMNS)
    required = {"search_id", "setting_id", "seed", "geometry", "tail_output_drift_fro"}
    missing = required.difference(step_metrics.columns)
    if missing:
        raise ValueError(f"step_metrics missing required columns for seed-level natural evaluator: {sorted(missing)}")
    rows = []
    metadata_columns = [
        "phase",
        "multiplicity_family",
        "planned_family_size",
        "dataset",
        "model",
        "partition_id",
        "warmup_steps",
        "target_head_gain_fraction_registered",
    ]
    for (search_id, setting_id, seed), group in step_metrics.groupby(
        ["search_id", "setting_id", "seed"],
        observed=True,
        sort=False,
    ):
        by_geometry = group.set_index("geometry")
        if not {"frobenius", "spectral"}.issubset(by_geometry.index):
            continue
        fro = float(by_geometry.loc["frobenius", "tail_output_drift_fro"])
        spectral = float(by_geometry.loc["spectral", "tail_output_drift_fro"])
        ratio = spectral**2 / max(fro**2, 1e-300)
        row = {
            "search_id": search_id,
            "setting_id": setting_id,
            "seed": int(seed),
            "frobenius_tail_output_drift_fro": fro,
            "spectral_tail_output_drift_fro": spectral,
            "tail_output_drift_sq_ratio_spectral_over_fro": ratio,
            "log_tail_output_drift_sq_ratio_spectral_over_fro": math.log(ratio) if ratio > 0 else math.nan,
        }
        first = group.iloc[0]
        for column in metadata_columns:
            row[column] = first[column] if column in first.index else math.nan
        rows.append(row)
    return pd.DataFrame(rows, columns=SEED_RATIO_COLUMNS)


def quality_gates(row: pd.Series) -> tuple[bool, bool, bool]:
    head_gain_ok = (
        float(row.get("mean_actual_head_gain_relative_error_frobenius", math.inf))
        <= HEAD_GAIN_RELATIVE_ERROR_MAX
        and float(row.get("mean_actual_head_gain_relative_error_spectral", math.inf))
        <= HEAD_GAIN_RELATIVE_ERROR_MAX
    )
    dataset_name = str(row.get("dataset_name", ""))
    floor = TAIL_ACCURACY_FLOOR_BY_DATASET.get(dataset_name, 0.0)
    tail_accuracy = float(row.get("mean_tail_accuracy_before", math.nan))
    tail_quality_ok = bool(np.isfinite(tail_accuracy) and tail_accuracy >= floor)
    return head_gain_ok, tail_quality_ok, head_gain_ok and tail_quality_ok


def build_primary_decisions(
    expected: pd.DataFrame,
    observed: pd.DataFrame,
    seed_ratios: pd.DataFrame,
    run_registry: pd.DataFrame,
) -> pd.DataFrame:
    family_size = int(expected["planned_family_size"].max())
    observed_columns = [
        "search_id",
        "setting_id",
        "seeds",
        "geomean_tail_output_drift_sq_ratio_spectral_over_fro",
        "tail_output_drift_sq_ratio_ci95_low",
        "tail_output_drift_sq_ratio_ci95_high",
        "mean_tail_accuracy_before",
        "mean_actual_head_gain_relative_error_frobenius",
        "mean_actual_head_gain_relative_error_spectral",
    ]
    if observed.empty:
        merged = expected.copy()
    else:
        available = [column for column in observed_columns if column in observed.columns]
        merged = expected.merge(
            observed[available],
            on=["search_id", "setting_id"],
            how="left",
            suffixes=("", "_observed"),
        )
    complete = set(run_registry["output_status"]) == {"complete"}
    decisions = []
    for _, row in merged.iterrows():
        has_output = pd.notna(row.get("geomean_tail_output_drift_sq_ratio_spectral_over_fro", math.nan))
        record = {column: row.get(column, math.nan) for column in expected.columns}
        record["output_status"] = "observed" if has_output else "not_run"
        record["metric_id"] = "primary_tail_output_drift_ratio"
        record["observed_seeds"] = int(row["seeds_observed"] if "seeds_observed" in row and pd.notna(row["seeds_observed"]) else row.get("seeds", 0)) if has_output else 0
        if has_output:
            setting_seed_ratios = seed_ratios[seed_ratios["setting_id"].eq(row["setting_id"])]
            if len(setting_seed_ratios) >= 1:
                test = log_ratio_test_from_seed_rows(setting_seed_ratios, family_size=family_size)
                inference_source = "paired_seed_log_ratio_t_test"
                raw_p = test["raw_one_sided_p"]
                simultaneous_low = test["bonferroni_simultaneous_ci95_low"]
                raw_low = test["raw_ci95_low"]
                raw_high = test["raw_ci95_high"]
                estimate = test["estimate"]
                log_ratio_mean = test["log_ratio_mean"]
                log_ratio_se = test["log_ratio_se"]
                observed_seeds = int(setting_seed_ratios["seed"].nunique())
            else:
                raw_p, simultaneous_low, log_ratio_se = infer_log_ratio_test(row, family_size=family_size)
                inference_source = "aggregate_ci_fallback"
                raw_low = float(row["tail_output_drift_sq_ratio_ci95_low"])
                raw_high = float(row["tail_output_drift_sq_ratio_ci95_high"])
                estimate = float(row["geomean_tail_output_drift_sq_ratio_spectral_over_fro"])
                log_ratio_mean = math.log(estimate) if estimate > 0 else math.nan
                observed_seeds = int(
                    row["seeds_observed"]
                    if "seeds_observed" in row and pd.notna(row["seeds_observed"])
                    else row.get("seeds", 0)
                )
            head_gain_gate, tail_quality_gate, quality_gate = quality_gates(row)
            raw_ci_worse = bool(raw_low > 1.0)
            scope_status = "complete_family" if complete else "incomplete_provisional"
            decision_status = "pending_phase_completion" if not complete else "not_primary_worse_adjusted"
            claim_status = "not_ready" if not complete else "finite_null_or_no_primary_candidate"
        else:
            inference_source = "missing_output"
            raw_p = math.nan
            simultaneous_low = math.nan
            log_ratio_mean = math.nan
            log_ratio_se = math.nan
            observed_seeds = 0
            head_gain_gate = False
            tail_quality_gate = False
            quality_gate = False
            raw_low = math.nan
            raw_high = math.nan
            estimate = math.nan
            raw_ci_worse = False
            scope_status = "missing_output"
            decision_status = "pending_output"
            claim_status = "not_ready"
        record.update(
            {
                "seed_count": row.get("seed_count", len(DEFAULT_SEEDS)),
                "observed_seeds": observed_seeds,
                "inference_source": inference_source,
                "estimate": estimate,
                "raw_ci95_low": raw_low,
                "raw_ci95_high": raw_high,
                "log_ratio_mean": log_ratio_mean,
                "log_ratio_se": log_ratio_se,
                "raw_one_sided_p": raw_p,
                "bonferroni_simultaneous_ci95_low": simultaneous_low,
                "raw_ci_worse": raw_ci_worse,
                "head_gain_gate": head_gain_gate,
                "tail_quality_gate": tail_quality_gate,
                "quality_gate": quality_gate,
                "adjustment_scope_status": scope_status,
                "adjusted_primary_decision": decision_status,
                "claim_status": claim_status,
            }
        )
        decisions.append(record)
    frame = pd.DataFrame(decisions)
    frame["holm_adjusted_one_sided_p"] = holm_adjust(
        frame["raw_one_sided_p"],
        family_size=family_size,
    )
    observed_mask = frame["output_status"].eq("observed")
    complete_family = set(run_registry["output_status"]) == {"complete"}
    adjusted_worse = (
        observed_mask
        & complete_family
        & frame["quality_gate"].astype(bool)
        & frame["holm_adjusted_one_sided_p"].le(ALPHA)
        & frame["estimate"].gt(1.0)
    )
    frame.loc[adjusted_worse, "adjusted_primary_decision"] = "primary_worse_adjusted"
    frame.loc[adjusted_worse, "claim_status"] = "fresh_natural_primary_boundary_candidate"
    complete_observed_not_worse = observed_mask & complete_family & ~adjusted_worse
    frame.loc[complete_observed_not_worse, "adjusted_primary_decision"] = "not_primary_worse_adjusted"
    frame.loc[complete_observed_not_worse, "claim_status"] = "no_primary_counterexample_for_setting"
    return frame[DECISION_COLUMNS]


def build_gate_report(run_registry: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
    complete = set(run_registry["output_status"]) == {"complete"}
    observed_count = int(decisions["output_status"].eq("observed").sum())
    expected_count = len(decisions)
    adjusted_worse_count = int(decisions["adjusted_primary_decision"].eq("primary_worse_adjusted").sum())
    if complete and adjusted_worse_count > 0:
        primary_status = "candidate"
        primary_evidence = f"{adjusted_worse_count} adjusted primary rows pass Holm and quality gates"
    elif complete:
        primary_status = "finite_null_candidate"
        primary_evidence = "all phase1 settings evaluated with no adjusted primary worse row"
    else:
        primary_status = "not_ready"
        primary_evidence = f"{observed_count}/{expected_count} phase1 settings have primary metric rows"
    return pd.DataFrame(
        [
            {
                "gate_id": "NNS-E1-evaluator-implemented",
                "status": "pass",
                "evidence": "scripts/e11_evaluate_natural_negative_search_phase1.py wrote evaluator artifacts",
            },
            {
                "gate_id": "NNS-E2-phase1-output-completeness",
                "status": "pass" if complete else "not_ready",
                "evidence": "; ".join(
                    f"{row.search_id}={row.output_status} ({int(row.pair_summary_rows)}/{int(row.expected_settings)} rows)"
                    for row in run_registry.itertuples()
                ),
            },
            {
                "gate_id": "NNS-E3-primary-multiplicity",
                "status": "pass" if complete else "not_ready",
                "evidence": "Holm one-sided p-values use paired per-seed log-ratio tests over the registered 26-setting phase1 family",
            },
            {
                "gate_id": "NNS-E4-natural-primary-claim",
                "status": primary_status,
                "evidence": primary_evidence,
            },
            {
                "gate_id": "NNS-E5-full-reporting-boundary",
                "status": "pass",
                "evidence": "primary_decisions.csv keeps every registered setting, including not-run rows",
            },
        ]
    )


def write_discussion(
    run_registry: pd.DataFrame,
    decisions: pd.DataFrame,
    gate_report: pd.DataFrame,
    seed_ratios: pd.DataFrame,
) -> None:
    observed = int(decisions["output_status"].eq("observed").sum())
    total = len(decisions)
    lines = [
        "# E11 Natural Negative Search Phase1 Evaluation",
        "",
        "This generated evaluator is the multiplicity boundary for the registered",
        "natural negative-search phase1 family. It does not claim a natural primary",
        "counterexample until all 26 declared settings have metric rows and the Holm",
        "adjusted primary decision from paired per-seed log-ratio tests passes the",
        "quality gates.",
        "",
        f"Current primary metric coverage: {observed}/{total} settings.",
        f"Current seed-level primary rows: {len(seed_ratios)}.",
        "",
        "## Run Registry",
        "",
        markdown_table(
            run_registry,
            [
                "search_id",
                "output_status",
                "expected_settings",
                "settings_registry_rows",
                "pair_summary_rows",
                "artifact_prefix",
            ],
        ),
        "",
        "## Gate Report",
        "",
        markdown_table(gate_report, ["gate_id", "status", "evidence"]),
        "",
        "## Primary Decision Boundary",
        "",
        markdown_table(
            decisions.head(12),
            [
                "search_id",
                "setting_id",
                "output_status",
                "inference_source",
                "raw_one_sided_p",
                "holm_adjusted_one_sided_p",
                "adjusted_primary_decision",
                "claim_status",
            ],
        ),
        "",
        "Artifacts:",
        f"- [run_registry.csv](../{(OUTPUT_DIR / 'run_registry.csv').as_posix()})",
        f"- [seed_level_primary_ratios.csv](../{(OUTPUT_DIR / 'seed_level_primary_ratios.csv').as_posix()})",
        f"- [primary_decisions.csv](../{(OUTPUT_DIR / 'primary_decisions.csv').as_posix()})",
        f"- [gate_report.csv](../{(OUTPUT_DIR / 'gate_report.csv').as_posix()})",
        f"- [config.json](../{(OUTPUT_DIR / 'config.json').as_posix()})",
    ]
    write_markdown(DISCUSSION_PATH, "\n".join(lines) + "\n")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    expected = expected_registry()
    run_registry = build_run_registry(expected)
    observed = load_observed_pair_summary()
    step_metrics = load_observed_step_metrics()
    seed_ratios = seed_level_primary_ratios(step_metrics)
    decisions = build_primary_decisions(expected, observed, seed_ratios, run_registry)
    gate_report = build_gate_report(run_registry, decisions)
    run_registry.to_csv(OUTPUT_DIR / "run_registry.csv", index=False)
    seed_ratios.to_csv(OUTPUT_DIR / "seed_level_primary_ratios.csv", index=False)
    decisions.to_csv(OUTPUT_DIR / "primary_decisions.csv", index=False)
    gate_report.to_csv(OUTPUT_DIR / "gate_report.csv", index=False)
    (OUTPUT_DIR / "config.json").write_text(
        json.dumps(
            {
                "multiplicity_family": MULTIPLICITY_FAMILY,
                "planned_family_size": int(expected["planned_family_size"].max()),
                "alpha": ALPHA,
                "head_gain_relative_error_max": HEAD_GAIN_RELATIVE_ERROR_MAX,
                "tail_accuracy_floor_by_dataset": TAIL_ACCURACY_FLOOR_BY_DATASET,
                "primary_test": "paired per-seed log tail-output-drift ratio t-test with Holm one-sided phase-family adjustment",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_discussion(run_registry, decisions, gate_report, seed_ratios)
    print(f"saved natural negative-search phase1 evaluation to {OUTPUT_DIR}")
    print(f"observed primary rows={int(decisions['output_status'].eq('observed').sum())}/{len(decisions)}")
    print(gate_report.to_string(index=False))


if __name__ == "__main__":
    main()
