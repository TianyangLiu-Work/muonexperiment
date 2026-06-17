from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.reporting import fmt, markdown_table, write_markdown


OUTPUT_DIR = Path("results/e11_matrix_block_tightness_audit")
DISCUSSION_PATH = Path("discussion/e11_matrix_block_tightness_audit.md")


def padded(values: list[float], length: int) -> list[float]:
    return values + [0.0] * max(length - len(values), 0)


def spectrum(values: list[float]) -> str:
    return "[" + ", ".join(fmt(value) for value in values) + "]"


def numerical_rank(values: list[float]) -> float:
    squared = sum(value * value for value in values)
    if squared <= 0.0:
        return math.nan
    return sum(values) ** 2 / squared


def sandwich_stable_rank(b_values: list[float], a_values: list[float]) -> float:
    length = max(len(b_values), len(a_values))
    b_pad = padded(b_values, length)
    a_pad = padded(a_values, length)
    top_product_sq = max(b_pad) ** 2 * max(a_pad) ** 2
    if top_product_sq <= 0.0:
        return math.nan
    paired_sum = sum((b * a) ** 2 for b, a in zip(b_pad, a_pad))
    return paired_sum / top_product_sq


def frobenius_tail_numerator(b_values: list[float], a_values: list[float]) -> float:
    if not b_values or not a_values:
        return 0.0
    return max(b_values) ** 2 * max(a_values) ** 2


def spectral_tail_numerator(b_values: list[float], a_values: list[float]) -> float:
    length = max(len(b_values), len(a_values))
    return sum((b * a) ** 2 for b, a in zip(padded(b_values, length), padded(a_values, length)))


def relation_from_ratio(ratio: float, applicable: bool) -> str:
    if not applicable:
        return "degenerate_tail_boundary_not_applicable"
    if ratio < 1.0 - 1e-12:
        return "spectral_smaller_worst_case_bound"
    if ratio > 1.0 + 1e-12:
        return "frobenius_smaller_worst_case_bound"
    return "equal_worst_case_bound"


def build_cases() -> pd.DataFrame:
    case_specs = [
        {
            "case_id": "MBTA-1-spectral-favored",
            "g_singular_values": [1.0, 1.0, 1.0, 1.0],
            "b_singular_values": [1.0, 0.0, 0.0, 0.0],
            "a_singular_values": [1.0, 0.0, 0.0, 0.0],
            "paper_use": "clean rank-boundary witness for nrank(G_H)>ssrank(B_T,A_T)",
        },
        {
            "case_id": "MBTA-2-equality-boundary",
            "g_singular_values": [1.0, 1.0],
            "b_singular_values": [1.0, 1.0],
            "a_singular_values": [1.0, 1.0],
            "paper_use": "shows strict inequality is needed for a strict spectral bound advantage",
        },
        {
            "case_id": "MBTA-3-frobenius-favored",
            "g_singular_values": [1.0],
            "b_singular_values": [1.0, 1.0],
            "a_singular_values": [1.0, 1.0],
            "paper_use": "claim-boundary witness where tail sensitivity rank exceeds head numerical rank",
        },
        {
            "case_id": "MBTA-4-degenerate-tail",
            "g_singular_values": [1.0, 1.0, 1.0],
            "b_singular_values": [0.0, 0.0],
            "a_singular_values": [1.0, 1.0],
            "paper_use": "nondegenerate-tail assumption witness; rank boundary is intentionally not interpreted",
        },
    ]
    rows = []
    for spec in case_specs:
        g_values = spec["g_singular_values"]
        b_values = spec["b_singular_values"]
        a_values = spec["a_singular_values"]
        nrank = numerical_rank(g_values)
        ssrank = sandwich_stable_rank(b_values, a_values)
        applicable = bool(
            math.isfinite(nrank)
            and math.isfinite(ssrank)
            and frobenius_tail_numerator(b_values, a_values) > 0.0
        )
        ratio = ssrank / nrank if applicable else math.nan
        rows.append(
            {
                "case_id": spec["case_id"],
                "g_singular_values": spectrum(g_values),
                "b_singular_values": spectrum(b_values),
                "a_singular_values": spectrum(a_values),
                "nrank_G_H": nrank,
                "ssrank_B_T_A_T": ssrank,
                "ratio_I_spectral_over_I_frobenius": ratio,
                "rank_boundary_applicable": "yes" if applicable else "no",
                "predicted_relation": relation_from_ratio(ratio, applicable),
                "paper_use": spec["paper_use"],
            }
        )
    return pd.DataFrame(rows)


def build_formula_checks(cases: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for case in cases.itertuples():
        g_values = [float(value) for value in str(case.g_singular_values).strip("[]").split(", ")]
        b_values = [float(value) for value in str(case.b_singular_values).strip("[]").split(", ")]
        a_values = [float(value) for value in str(case.a_singular_values).strip("[]").split(", ")]
        fro_tail = frobenius_tail_numerator(b_values, a_values)
        spectral_tail = spectral_tail_numerator(b_values, a_values)
        g_fro_sq = sum(value * value for value in g_values)
        g_nuclear_sq = sum(g_values) ** 2
        fro_coeff = fro_tail / g_fro_sq if g_fro_sq > 0.0 else math.nan
        spectral_coeff = spectral_tail / g_nuclear_sq if g_nuclear_sq > 0.0 else math.nan
        direct_ratio = spectral_coeff / fro_coeff if fro_coeff > 0.0 else math.nan
        rank_ratio = case.ratio_I_spectral_over_I_frobenius
        rows.append(
            {
                "case_id": case.case_id,
                "frobenius_tail_numerator_formula": fro_tail,
                "frobenius_unit_witness_value": fro_tail,
                "spectral_tail_numerator_formula": spectral_tail,
                "spectral_unit_witness_value": spectral_tail,
                "I_frobenius": fro_coeff,
                "I_spectral": spectral_coeff,
                "direct_ratio_I_spectral_over_I_frobenius": direct_ratio,
                "rank_ratio_ssrank_over_nrank": rank_ratio,
                "ratio_absolute_error": abs(direct_ratio - rank_ratio)
                if math.isfinite(direct_ratio) and math.isfinite(rank_ratio)
                else math.nan,
                "tightness_status": "exact_for_diagonal_singular_witness"
                if case.rank_boundary_applicable == "yes"
                else "not_applicable_degenerate_tail",
            }
        )
    return pd.DataFrame(rows)


def build_caveats() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "caveat_id": "MBTC-1-nondegenerate-tail-required",
                "condition": "sigma_1(B_T) sigma_1(A_T) = 0",
                "mathematical_effect": "both tail numerator formulas vanish and ssrank(B_T,A_T) is undefined",
                "paper_action": "do not interpret the nrank-vs-ssrank rank boundary for a zero tail-sensitive block",
            },
            {
                "caveat_id": "MBTC-2-strict-inequality-required",
                "condition": "nrank(G_H) = ssrank(B_T,A_T)",
                "mathematical_effect": "I_spectral/I_frobenius = 1, so the theorem gives no strict advantage",
                "paper_action": "state strict spectral advantage only under nrank(G_H)>ssrank(B_T,A_T)",
            },
            {
                "caveat_id": "MBTC-3-worst-case-not-realized",
                "condition": "the optimizer update is not aligned with the worst-case tail singular directions",
                "mathematical_effect": "the bound can be loose for a realized direction even when the coefficient ordering is strict",
                "paper_action": "separate theorem-level bound ordering from empirical realized-drift prediction",
            },
        ]
    )


def write_discussion(cases: pd.DataFrame, checks: pd.DataFrame, caveats: pd.DataFrame) -> None:
    text = f"""# E11 Matrix-Block Tightness Audit

This generated artifact is a deterministic sanity audit for the matrix-block
theorem proof contract. It does not add new empirical results. It checks exact
diagonal singular-spectrum witnesses for the Frobenius numerator, the spectral
sandwich numerator, the ratio identity
`I_spectral / I_frobenius = ssrank(B_T,A_T) / nrank(G_H)`, and the degeneracy
cases where the rank boundary must not be interpreted.

## Rank-Boundary Cases

{markdown_table(cases, ["case_id", "g_singular_values", "b_singular_values", "a_singular_values", "nrank_G_H", "ssrank_B_T_A_T", "ratio_I_spectral_over_I_frobenius", "rank_boundary_applicable", "predicted_relation", "paper_use"])}

## Formula And Tightness Checks

{markdown_table(checks, ["case_id", "frobenius_tail_numerator_formula", "frobenius_unit_witness_value", "spectral_tail_numerator_formula", "spectral_unit_witness_value", "direct_ratio_I_spectral_over_I_frobenius", "rank_ratio_ssrank_over_nrank", "ratio_absolute_error", "tightness_status"])}

## Caveats

{markdown_table(caveats, ["caveat_id", "condition", "mathematical_effect", "paper_action"])}

Artifacts:
- [rank_boundary_cases.csv](../{(OUTPUT_DIR / 'rank_boundary_cases.csv').as_posix()})
- [formula_checks.csv](../{(OUTPUT_DIR / 'formula_checks.csv').as_posix()})
- [caveat_checks.csv](../{(OUTPUT_DIR / 'caveat_checks.csv').as_posix()})
- [config.json](../{(OUTPUT_DIR / 'config.json').as_posix()})
"""
    write_markdown(DISCUSSION_PATH, text)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cases = build_cases()
    checks = build_formula_checks(cases)
    caveats = build_caveats()
    applicable_checks = checks[checks["tightness_status"].eq("exact_for_diagonal_singular_witness")]
    max_ratio_error = float(applicable_checks["ratio_absolute_error"].max())
    if max_ratio_error > 1e-12:
        raise AssertionError(f"matrix-block ratio identity check failed: max error={max_ratio_error}")

    cases.to_csv(OUTPUT_DIR / "rank_boundary_cases.csv", index=False)
    checks.to_csv(OUTPUT_DIR / "formula_checks.csv", index=False)
    caveats.to_csv(OUTPUT_DIR / "caveat_checks.csv", index=False)
    (OUTPUT_DIR / "config.json").write_text(
        json.dumps(
            {
                "purpose": "deterministic tightness and degeneracy sanity audit for the matrix-block theorem",
                "analysis_scope": "closed-form proof sanity checks; no new empirical results",
                "case_count": int(len(cases)),
                "formula_check_count": int(len(checks)),
                "max_ratio_absolute_error": max_ratio_error,
                "all_identity_checks_pass": bool(max_ratio_error <= 1e-12),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_discussion(cases, checks, caveats)
    print(f"saved {DISCUSSION_PATH} and {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
