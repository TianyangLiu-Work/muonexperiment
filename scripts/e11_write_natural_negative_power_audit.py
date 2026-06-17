from __future__ import annotations

import math
import sys
from pathlib import Path
from statistics import NormalDist

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.reporting import fmt, markdown_table, write_markdown


OUTPUT_DIR = Path("results/e11_natural_negative_search_protocol/phase1_power_audit")
DISCUSSION_PATH = Path("discussion/e11_natural_negative_search_phase1_power_audit.md")
SEED_COUNT = 5
FAMILY_SIZE = 26
ALPHA = 0.05
LOG_RATIO_SDS = (0.05, 0.10, 0.20, 0.35, 0.50, 0.75)
TRUE_RATIOS = (1.05, 1.10, 1.25, 1.50, 2.00, 3.00)
TARGET_POWERS = (0.50, 0.80, 0.90)


def t_ppf(probability: float, df: int) -> float:
    try:
        from scipy.stats import t

        return float(t.ppf(probability, df))
    except Exception:
        return NormalDist().inv_cdf(probability)


def nct_sf(value: float, df: int, nc: float) -> float:
    try:
        from scipy.stats import nct

        return float(nct.sf(value, df, nc))
    except Exception:
        return 1.0 - NormalDist(mean=nc, sigma=1.0).cdf(value)


def power_one_sided(*, true_log_ratio: float, log_ratio_sd: float, seed_count: int, alpha: float) -> float:
    if true_log_ratio <= 0.0:
        return 0.0
    if log_ratio_sd <= 0.0:
        return 1.0
    df = seed_count - 1
    critical = t_ppf(1.0 - alpha, df)
    noncentrality = true_log_ratio / (log_ratio_sd / math.sqrt(seed_count))
    return nct_sf(critical, df, noncentrality)


def minimum_detectable_ratio(*, log_ratio_sd: float, seed_count: int, alpha: float, target_power: float) -> float:
    low = 1.0
    high = 2.0
    while power_one_sided(
        true_log_ratio=math.log(high),
        log_ratio_sd=log_ratio_sd,
        seed_count=seed_count,
        alpha=alpha,
    ) < target_power:
        high *= 2.0
        if high > 1e6:
            return math.inf
    for _ in range(80):
        mid = (low + high) / 2.0
        power = power_one_sided(
            true_log_ratio=math.log(mid),
            log_ratio_sd=log_ratio_sd,
            seed_count=seed_count,
            alpha=alpha,
        )
        if power >= target_power:
            high = mid
        else:
            low = mid
    return high


def alpha_rows() -> list[dict[str, object]]:
    return [
        {
            "alpha_scope": "raw_single_setting",
            "one_sided_alpha": ALPHA,
            "family_size": 1,
            "role": "diagnostic raw p-value only; cannot unlock the registered primary claim",
        },
        {
            "alpha_scope": "holm_bonferroni_worst_case",
            "one_sided_alpha": ALPHA / FAMILY_SIZE,
            "family_size": FAMILY_SIZE,
            "role": "conservative first-rejection threshold for the 26-setting phase1 family",
        },
    ]


def build_power_grid() -> pd.DataFrame:
    rows = []
    for alpha_spec in alpha_rows():
        alpha = float(alpha_spec["one_sided_alpha"])
        for log_sd in LOG_RATIO_SDS:
            for true_ratio in TRUE_RATIOS:
                rows.append(
                    {
                        "alpha_scope": alpha_spec["alpha_scope"],
                        "seed_count": SEED_COUNT,
                        "family_size": int(alpha_spec["family_size"]),
                        "one_sided_alpha": alpha,
                        "log_ratio_sd": log_sd,
                        "true_tail_drift_ratio": true_ratio,
                        "true_log_ratio": math.log(true_ratio),
                        "power": power_one_sided(
                            true_log_ratio=math.log(true_ratio),
                            log_ratio_sd=log_sd,
                            seed_count=SEED_COUNT,
                            alpha=alpha,
                        ),
                    }
                )
    return pd.DataFrame(rows)


def build_mde_table() -> pd.DataFrame:
    rows = []
    for alpha_spec in alpha_rows():
        alpha = float(alpha_spec["one_sided_alpha"])
        for log_sd in LOG_RATIO_SDS:
            for target_power in TARGET_POWERS:
                mde_ratio = minimum_detectable_ratio(
                    log_ratio_sd=log_sd,
                    seed_count=SEED_COUNT,
                    alpha=alpha,
                    target_power=target_power,
                )
                rows.append(
                    {
                        "alpha_scope": alpha_spec["alpha_scope"],
                        "seed_count": SEED_COUNT,
                        "family_size": int(alpha_spec["family_size"]),
                        "one_sided_alpha": alpha,
                        "log_ratio_sd": log_sd,
                        "target_power": target_power,
                        "minimum_detectable_ratio": mde_ratio,
                        "minimum_detectable_log_ratio": math.log(mde_ratio)
                        if math.isfinite(mde_ratio)
                        else math.inf,
                        "claim_interpretation": (
                            "phase1 null is informative only for effects at or above this scale "
                            "under the assumed across-seed log-ratio SD"
                        ),
                    }
                )
    return pd.DataFrame(rows)


def build_interpretation_ladder() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "case_id": "PWR-1-adjusted-positive",
                "evidence_condition": "complete 26-setting family; Holm-adjusted primary p <= 0.05; quality gates pass",
                "allowed_wording": "fresh natural primary boundary candidate",
                "blocked_wording": "broad optimizer-performance claim or universal natural counterexample claim",
            },
            {
                "case_id": "PWR-2-complete-null-above-mde",
                "evidence_condition": "complete 26-setting family; no adjusted primary row; observed SD makes target effect above the MDE",
                "allowed_wording": "finite registered null for effects at or above the audited detectable scale",
                "blocked_wording": "absence of smaller natural counterexamples",
            },
            {
                "case_id": "PWR-3-complete-null-below-mde",
                "evidence_condition": "complete 26-setting family; no adjusted primary row; target effect below MDE",
                "allowed_wording": "underpowered for small natural negative effects",
                "blocked_wording": "convincing null or mechanism generality claim",
            },
            {
                "case_id": "PWR-4-incomplete-family",
                "evidence_condition": "any declared phase1 metric output missing",
                "allowed_wording": "running registered search; no phase1 discovery decision yet",
                "blocked_wording": "primary counterexample, finite null, or selected-subset claim",
            },
        ]
    )


def write_discussion(power_grid: pd.DataFrame, mde_table: pd.DataFrame, ladder: pd.DataFrame) -> None:
    adjusted_mde = mde_table[
        (mde_table["alpha_scope"] == "holm_bonferroni_worst_case")
        & (mde_table["target_power"].isin([0.80, 0.90]))
    ].copy()
    adjusted_mde = adjusted_mde[adjusted_mde["log_ratio_sd"].isin([0.10, 0.20, 0.35, 0.50])]
    adjusted_power = power_grid[
        (power_grid["alpha_scope"] == "holm_bonferroni_worst_case")
        & (power_grid["log_ratio_sd"].isin([0.10, 0.20, 0.35]))
        & (power_grid["true_tail_drift_ratio"].isin([1.25, 1.50, 2.00, 3.00]))
    ].copy()
    text = f"""# E11 Natural Negative Search Phase1 Power Audit

This generated audit records the detectable-effect boundary for the registered
26-setting natural negative-search phase1 family. It is a design audit, not an
outcome analysis: it should be interpreted before inspecting fresh metric rows.

The primary test is the paired across-seed log ratio for
`tail_output_drift_sq_ratio_spectral_over_fro`, with a one-sided worse-than-one
alternative and Holm-adjusted phase-family decision. The conservative
first-rejection threshold is approximated as Bonferroni `0.05 / 26`.

## Adjusted Minimum Detectable Ratio

{markdown_table(adjusted_mde, ["log_ratio_sd", "target_power", "minimum_detectable_ratio", "minimum_detectable_log_ratio", "alpha_scope"])}

## Adjusted Power Grid

{markdown_table(adjusted_power, ["log_ratio_sd", "true_tail_drift_ratio", "power", "alpha_scope"])}

## Interpretation Ladder

{markdown_table(ladder, ["case_id", "evidence_condition", "allowed_wording", "blocked_wording"])}

Generated tables:

- [power_grid.csv](../results/e11_natural_negative_search_protocol/phase1_power_audit/power_grid.csv)
- [minimum_detectable_effect.csv](../results/e11_natural_negative_search_protocol/phase1_power_audit/minimum_detectable_effect.csv)
- [interpretation_ladder.csv](../results/e11_natural_negative_search_protocol/phase1_power_audit/interpretation_ladder.csv)
"""
    write_markdown(DISCUSSION_PATH, text)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    power_grid = build_power_grid()
    mde_table = build_mde_table()
    ladder = build_interpretation_ladder()
    power_grid.to_csv(OUTPUT_DIR / "power_grid.csv", index=False)
    mde_table.to_csv(OUTPUT_DIR / "minimum_detectable_effect.csv", index=False)
    ladder.to_csv(OUTPUT_DIR / "interpretation_ladder.csv", index=False)
    write_discussion(power_grid, mde_table, ladder)
    print(f"saved natural negative-search phase1 power audit to {OUTPUT_DIR} and {DISCUSSION_PATH}")


if __name__ == "__main__":
    main()
