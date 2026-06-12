from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.reporting import (
    fmt,
    markdown_table as table,
    require_one as row,
    write_markdown,
)

OUTPUT_PATH = Path("discussion/e11_optimizer_invariance_audit.md")


def volatility_rows(frame: pd.DataFrame, *, mode: str) -> list[dict[str, object]]:
    metrics = [
        "norm_rank_plane_mean_speed",
        "norm_condition_mean_speed",
        "condition_score_std_speed",
        "loss_mean_rel_speed",
        "per_update_rank_plane_mean_speed",
        "per_update_condition_mean_speed",
    ]
    families = ["All", "MatrixFactorizationInput", "MatrixSensing", "SmallMLPDigits"]
    records: list[dict[str, object]] = []
    for metric in metrics:
        for family in families:
            item = row(frame, metric=metric, problem_family=family)
            records.append(
                {
                    "mode": mode,
                    "metric": metric,
                    "problem_family": family,
                    "n_pairs": int(item["n_pairs"]),
                    "muon_lower_pairs": int(item["muon_lower_pairs"]),
                    "ratio_muon_over_adam": item["geomean_ratio_muon_over_adam"],
                    "ratio_ci95": f"[{fmt(item['ratio_ci95_low'])}, {fmt(item['ratio_ci95_high'])}]",
                    "ci_below_one": bool(item["ratio_ci95_below_one"]),
                }
            )
    return records


def main() -> None:
    raw_volatility = pd.read_csv("results/e11/volatility_summary.csv")
    equal_volatility = pd.read_csv("results/e11_equal_update/volatility_summary.csv")
    raw_update = pd.read_csv("results/e11/update_spectrum_summary.csv")
    equal_update = pd.read_csv("results/e11_equal_update/update_spectrum_summary.csv")

    raw_rank_all = row(raw_volatility, metric="norm_rank_plane_mean_speed", problem_family="All")
    equal_rank_all = row(equal_volatility, metric="norm_rank_plane_mean_speed", problem_family="All")
    raw_condition_all = row(raw_volatility, metric="norm_condition_mean_speed", problem_family="All")
    equal_condition_all = row(equal_volatility, metric="norm_condition_mean_speed", problem_family="All")
    equal_rank_ms = row(equal_volatility, metric="norm_rank_plane_mean_speed", problem_family="MatrixSensing")
    equal_condition_ms = row(equal_volatility, metric="norm_condition_mean_speed", problem_family="MatrixSensing")

    update_rows = []
    for mode, frame in [("raw", raw_update), ("equal_update", equal_update)]:
        for metric in ["nrUpdate", "stUpdate", "nrUpdateFrac", "stUpdateFrac", "update_flatness"]:
            item = row(frame, metric=metric, problem_family="All")
            update_rows.append(
                {
                    "mode": mode,
                    "metric": metric,
                    "ratio_muon_over_adam": item["geomean_ratio_muon_over_adam"],
                    "ratio_ci95": f"[{fmt(item['ratio_ci95_low'])}, {fmt(item['ratio_ci95_high'])}]",
                    "ci_above_one": bool(item["ratio_ci95_above_one"]),
                }
            )

    volatility = pd.DataFrame(volatility_rows(raw_volatility, mode="raw") + volatility_rows(equal_volatility, mode="equal_update"))
    update_spectrum = pd.DataFrame(update_rows)

    text = f"""# E11 Optimizer-Invariance Audit

This generated audit separates optimizer-intrinsic evidence from effects that depend on update scale, task family, or trajectory state. It is meant to answer whether the current results justify a statement like "Muon is more stable" across tasks.

## Short Answer

The current evidence supports a robust optimizer-intrinsic claim for **update-spectrum shaping**, but it does **not** support the broad claim that Muon is generally more stable in the state geometry.

Raw trajectories make Muon look smoother: all-task normalized rank-plane speed has Muon/Adam ratio {fmt(raw_rank_all['geomean_ratio_muon_over_adam'])} with CI [{fmt(raw_rank_all['ratio_ci95_low'])}, {fmt(raw_rank_all['ratio_ci95_high'])}], and normalized condition speed has ratio {fmt(raw_condition_all['geomean_ratio_muon_over_adam'])} with CI [{fmt(raw_condition_all['ratio_ci95_low'])}, {fmt(raw_condition_all['ratio_ci95_high'])}]. However, after matching global relative update size, those all-task ratios become {fmt(equal_rank_all['geomean_ratio_muon_over_adam'])} for rank-plane speed and {fmt(equal_condition_all['geomean_ratio_muon_over_adam'])} for condition speed. In Matrix Sensing under equal-update control, the same ratios are {fmt(equal_rank_ms['geomean_ratio_muon_over_adam'])} and {fmt(equal_condition_ms['geomean_ratio_muon_over_adam'])}, both strongly above 1.

Therefore, "Muon is smoother/more stable" is not a reliable optimizer-level conclusion. A safer statement is: **Muon reliably flattens the update spectrum; the induced state-trajectory smoothness is task- and scale-dependent.**

## Update-Spectrum Invariant

{table(update_spectrum, ["mode", "metric", "ratio_muon_over_adam", "ratio_ci95", "ci_above_one"])}

These diagnostics are robust to the equal-update control because rescaling an update does not change normalized singular-value geometry. This is the strongest current cross-task optimizer signature.

## Volatility / Stability Check

{table(volatility, ["mode", "metric", "problem_family", "n_pairs", "muon_lower_pairs", "ratio_muon_over_adam", "ratio_ci95", "ci_below_one"])}

## Interpretation

1. Raw lower volatility is partly confounded by Muon's smaller effective update scale in several settings.
2. Equal-update control removes that explanation and shows that state-geometry speed can flip direction by task family.
3. The update-spectrum difference is optimizer-intrinsic; the state-trajectory difference is an interaction between optimizer, task, and scale.
4. This audit argues against using "more stable" as the main claim unless stability is defined narrowly as a specific measured quantity in a specific controlled setting.

## Recommended Wording

Use:

> Muon has a robust update-spectrum signature: at matched update size, its update matrices have larger effective/stable rank and flatter spectra than Adam. The downstream trajectory smoothness is not invariant across task families.

Avoid:

> Muon is generally more stable than Adam.

## Sources

- [raw volatility summary](../results/e11/volatility_summary.csv)
- [equal-update volatility summary](../results/e11_equal_update/volatility_summary.csv)
- [raw update-spectrum summary](../results/e11/update_spectrum_summary.csv)
- [equal-update update-spectrum summary](../results/e11_equal_update/update_spectrum_summary.csv)
- [raw volatility figure](../figures/e11/volatility_robustness.png)
- [equal-update volatility figure](../figures/e11_equal_update/volatility_robustness.png)
- [equal-update update-spectrum figure](../figures/e11_equal_update/update_spectrum_robustness.png)
"""
    write_markdown(OUTPUT_PATH, text)
    print(f"saved optimizer-invariance audit to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
