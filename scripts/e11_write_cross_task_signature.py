from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.reporting import fmt, markdown_table, write_markdown


OUTPUT_DIR = Path("results/e11_cross_task_signature")
DISCUSSION_PATH = Path("discussion/e11_cross_task_signature.md")

FAMILIES = ["MatrixFactorizationInput", "MatrixSensing", "SmallMLPDigits"]


def read_ratio_rows(path: str, *, source: str, metrics: list[str], direction: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    rows = frame[frame["metric"].isin(metrics) & frame["problem_family"].isin(FAMILIES)].copy()
    rows["source"] = source
    rows["expected_direction"] = direction
    if direction == "above_one":
        rows["ci_supports_expected"] = rows["ratio_ci95_low"] > 1.0
        rows["opposes_expected"] = rows["ratio_ci95_high"] < 1.0
    elif direction == "below_one":
        rows["ci_supports_expected"] = rows["ratio_ci95_high"] < 1.0
        rows["opposes_expected"] = rows["ratio_ci95_low"] > 1.0
    else:
        raise ValueError(f"unknown direction: {direction}")
    return rows


def summarize_candidate(rows: pd.DataFrame) -> dict[str, object]:
    ratios = []
    ci_text = []
    signs = []
    support = []
    oppose = []
    for family in FAMILIES:
        item = rows[rows["problem_family"].eq(family)]
        if len(item) != 1:
            raise ValueError(f"expected one row for {rows['source'].iloc[0]} {rows['metric'].iloc[0]} {family}, got {len(item)}")
        rec = item.iloc[0]
        ratios.append(f"{family}={fmt(rec['geomean_ratio_muon_over_adam'])}")
        ci_text.append(f"{family}=[{fmt(rec['ratio_ci95_low'])}, {fmt(rec['ratio_ci95_high'])}]")
        signs.append("above" if rec["geomean_ratio_muon_over_adam"] > 1.0 else "below")
        support.append(bool(rec["ci_supports_expected"]))
        oppose.append(bool(rec["opposes_expected"]))
    all_same_sign = len(set(signs)) == 1
    return {
        "source": rows["source"].iloc[0],
        "metric": rows["metric"].iloc[0],
        "expected_direction": rows["expected_direction"].iloc[0],
        "family_ratios": "; ".join(ratios),
        "family_ci95": "; ".join(ci_text),
        "all_families_same_sign": all_same_sign,
        "families_support_expected": int(sum(support)),
        "families_oppose_expected": int(sum(oppose)),
        "passes_cross_task_screen": bool(all_same_sign and all(support)),
    }


def main() -> None:
    candidates = [
        read_ratio_rows(
            "results/e11_equal_update/update_spectrum_summary.csv",
            source="equal_update:update_spectrum",
            metrics=["nrUpdate", "stUpdate", "nrUpdateFrac", "stUpdateFrac", "update_flatness"],
            direction="above_one",
        ),
        read_ratio_rows(
            "results/e11_equal_update/volatility_summary.csv",
            source="equal_update:state_volatility",
            metrics=[
                "norm_rank_plane_mean_speed",
                "norm_condition_mean_speed",
                "condition_score_std_speed",
                "loss_mean_rel_speed",
                "per_update_rank_plane_mean_speed",
                "per_update_condition_mean_speed",
            ],
            direction="below_one",
        ),
        read_ratio_rows(
            "results/e11_equal_update/first_order_pair_summary.csv",
            source="equal_update:one_step_progress",
            metrics=["update_grad_inner", "delta_loss", "update_grad_cosine"],
            direction="above_one",
        ),
    ]
    long_rows = pd.concat(candidates, ignore_index=True)
    summary = pd.DataFrame(
        [
            summarize_candidate(group)
            for _, group in long_rows.groupby(["source", "metric"], observed=True, sort=False)
        ]
    ).sort_values(["passes_cross_task_screen", "source", "metric"], ascending=[False, True, True])

    passed = summary[summary["passes_cross_task_screen"]].copy()
    failed = summary[~summary["passes_cross_task_screen"]].copy()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    long_rows.to_csv(OUTPUT_DIR / "cross_task_signature_rows.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "cross_task_signature_summary.csv", index=False)

    text = f"""# E11 Cross-Task Optimizer Signature

This generated note asks which candidate optimizer-level statements survive a simple cross-task screen across Matrix Factorization with input, Matrix Sensing, and Small MLP digits.

Screening rule: a candidate passes only if all three task families have the same Muon/Adam ratio direction and all three family-level 95% CIs support the expected direction. This is deliberately conservative.

## Passing Candidates

{markdown_table(passed, ["source", "metric", "expected_direction", "family_ratios", "family_ci95", "families_support_expected", "passes_cross_task_screen"])}

## Failed Or Mixed Candidates

{markdown_table(failed, ["source", "metric", "expected_direction", "family_ratios", "family_ci95", "families_support_expected", "families_oppose_expected", "all_families_same_sign"])}

## Interpretation

The only candidates that cleanly pass this cross-task screen are update-spectrum quantities: `nrUpdate`, `stUpdate`, normalized rank fractions, and update flatness. This supports the narrow optimizer-intrinsic claim that Muon changes the singular-value geometry of update matrices.

State-trajectory volatility and one-step progress do not pass the same screen. Their signs or CI support change by task family. This means they should be presented as downstream interactions between optimizer and problem geometry, not as optimizer-intrinsic invariants.

## Implication For The Research Question

The cross-task invariant we can currently defend is:

> Muon consistently produces flatter, higher-rank update spectra than Adam at matched update size.

The mechanism question should then be:

> Under what task/layer/norm geometry does this update-spectrum shaping improve the positive descent proxy `<G,D>` and observed loss decrease?

## Sources

- [cross-task signature summary](../results/e11_cross_task_signature/cross_task_signature_summary.csv)
- [cross-task signature rows](../results/e11_cross_task_signature/cross_task_signature_rows.csv)
- [equal-update update-spectrum summary](../results/e11_equal_update/update_spectrum_summary.csv)
- [equal-update volatility summary](../results/e11_equal_update/volatility_summary.csv)
- [equal-update first-order pair summary](../results/e11_equal_update/first_order_pair_summary.csv)
"""
    write_markdown(DISCUSSION_PATH, text)
    print(f"saved cross-task signature summary to {OUTPUT_DIR}")
    print(f"saved cross-task signature discussion to {DISCUSSION_PATH}")


if __name__ == "__main__":
    main()
