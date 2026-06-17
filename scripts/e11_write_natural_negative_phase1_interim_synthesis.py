from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.reporting import fmt, markdown_table, write_markdown


EVAL_DIR = Path("results/e11_natural_negative_search_protocol/phase1_multiplicity_evaluation")
OUTPUT_DIR = Path("results/e11_natural_negative_search_protocol/phase1_interim_synthesis")
DISCUSSION_PATH = Path("discussion/e11_natural_negative_search_phase1_interim_synthesis.md")
TOTAL_FAMILY_SIZE = 26


def bool_sum(frame: pd.DataFrame, column: str) -> int:
    if column not in frame.columns or frame.empty:
        return 0
    return int(frame[column].astype(str).str.lower().isin({"true", "1", "yes"}).sum())


def build_family_coverage(run_registry: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in run_registry.itertuples():
        expected = int(row.expected_settings)
        observed = int(row.pair_summary_rows)
        rows.append(
            {
                "search_id": row.search_id,
                "expected_settings": expected,
                "observed_primary_rows": observed,
                "missing_primary_rows": expected - observed,
                "coverage_fraction": observed / expected if expected else 0.0,
                "output_status": row.output_status,
                "claim_use": "interim_evidence_only" if observed else "missing_required_family_rows",
            }
        )
    return pd.DataFrame(rows)


def build_observed_summary(decisions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    scopes = [("all_observed", decisions)]
    for search_id, group in decisions.groupby("search_id", sort=False):
        scopes.append((str(search_id), group))

    for scope, group in scopes:
        observed = group[group["output_status"].eq("observed")].copy()
        missing = group[group["output_status"].ne("observed")]
        estimates = observed["estimate"].dropna().astype(float) if "estimate" in observed else pd.Series(dtype=float)
        raw_high = observed["raw_ci95_high"].dropna().astype(float) if "raw_ci95_high" in observed else pd.Series(dtype=float)
        rows.append(
            {
                "scope": scope,
                "observed_primary_rows": int(len(observed)),
                "missing_primary_rows": int(len(missing)),
                "raw_worse_rows": bool_sum(observed, "raw_ci_worse"),
                "quality_gate_pass_rows": bool_sum(observed, "quality_gate"),
                "quality_gate_fail_rows": int(len(observed)) - bool_sum(observed, "quality_gate"),
                "estimate_min": float(estimates.min()) if not estimates.empty else float("nan"),
                "estimate_max": float(estimates.max()) if not estimates.empty else float("nan"),
                "raw_ci95_high_max": float(raw_high.max()) if not raw_high.empty else float("nan"),
                "claim_statuses": ",".join(sorted(set(group["claim_status"].astype(str)))),
                "interpretation": "no_claim_until_full_family_complete"
                if len(missing) > 0
                else "complete_family_ready_for_registered_decision",
            }
        )
    return pd.DataFrame(rows)


def build_claim_boundary(decisions: pd.DataFrame, gate_report: pd.DataFrame) -> pd.DataFrame:
    observed_count = int(decisions["output_status"].eq("observed").sum())
    missing_count = TOTAL_FAMILY_SIZE - observed_count
    gate_status = gate_report.set_index("gate_id")["status"].to_dict()
    raw_worse_rows = bool_sum(decisions[decisions["output_status"].eq("observed")], "raw_ci_worse")
    adjusted_worse_rows = int(decisions["adjusted_primary_decision"].eq("primary_worse_adjusted").sum())
    return pd.DataFrame(
        [
            {
                "claim_id": "NNI-1-primary-natural-counterexample",
                "current_status": "blocked_partial_family",
                "evidence": f"{observed_count}/{TOTAL_FAMILY_SIZE} observed; raw_worse_rows={raw_worse_rows}; adjusted_worse_rows={adjusted_worse_rows}; NNS-E2={gate_status.get('NNS-E2-phase1-output-completeness')}",
                "allowed_wording": "partial phase1 readout only; no natural primary counterexample",
                "blocked_wording": "fresh natural primary counterexample",
            },
            {
                "claim_id": "NNI-2-finite-null-search",
                "current_status": "blocked_partial_family",
                "evidence": f"{missing_count} declared settings are still missing, including tail-quality controls",
                "allowed_wording": "20/26 observed rows are reported under the frozen family",
                "blocked_wording": "finite null over the 26-setting phase1 family",
            },
            {
                "claim_id": "NNI-3-quality-gated-interpretation",
                "current_status": "blocked_until_quality_and_completeness_pass",
                "evidence": f"NNS-E3={gate_status.get('NNS-E3-primary-multiplicity')}; NNS-E4={gate_status.get('NNS-E4-natural-primary-claim')}",
                "allowed_wording": "quality and multiplicity gates remain the claim boundary",
                "blocked_wording": "using low-tail-quality or incomplete rows as mechanism falsification",
            },
        ]
    )


def build_remaining_work(run_registry: pd.DataFrame) -> pd.DataFrame:
    missing = run_registry[run_registry["output_status"].ne("complete")].copy()
    rows = []
    for row in missing.itertuples():
        rows.append(
            {
                "search_id": row.search_id,
                "missing_primary_rows": int(row.expected_settings) - int(row.pair_summary_rows),
                "current_status": row.output_status,
                "required_action": "wait for Slurm output, then rerun make e11-natural-negative-search-phase1-eval",
                "claim_unblocked_if_done": "fresh primary candidate or finite-null boundary, depending on adjusted decisions",
            }
        )
    return pd.DataFrame(rows)


def write_discussion(
    coverage: pd.DataFrame,
    observed_summary: pd.DataFrame,
    claim_boundary: pd.DataFrame,
    remaining_work: pd.DataFrame,
) -> None:
    observed_total = int(coverage["observed_primary_rows"].sum())
    missing_total = int(coverage["missing_primary_rows"].sum())
    all_observed = observed_summary[observed_summary["scope"].eq("all_observed")].iloc[0]
    text = f"""# E11 Natural Negative Search Phase1 Interim Synthesis

This generated artifact summarizes the partial Holm-family state for the
pre-registered natural negative-search phase1 family. It is not a discovery
claim. The current evaluator has {observed_total}/{TOTAL_FAMILY_SIZE} observed
primary rows and {missing_total} missing declared rows. Natural claims remain
blocked because the tail-quality controls are not complete and the evaluator
gates remain `not_ready`.

Observed rows have raw_worse_rows={int(all_observed["raw_worse_rows"])} and
max raw CI high {fmt(all_observed["raw_ci95_high_max"])}. This is useful
interim evidence, but no adjusted primary decision is claimable until the full
26-setting family is complete.

## Family Coverage

{markdown_table(coverage, ["search_id", "expected_settings", "observed_primary_rows", "missing_primary_rows", "coverage_fraction", "output_status", "claim_use"])}

## Observed Primary Summary

{markdown_table(observed_summary, ["scope", "observed_primary_rows", "missing_primary_rows", "raw_worse_rows", "quality_gate_pass_rows", "quality_gate_fail_rows", "estimate_min", "estimate_max", "raw_ci95_high_max", "claim_statuses", "interpretation"])}

## Claim Boundary

{markdown_table(claim_boundary, ["claim_id", "current_status", "evidence", "allowed_wording", "blocked_wording"])}

## Remaining Work

{markdown_table(remaining_work, ["search_id", "missing_primary_rows", "current_status", "required_action", "claim_unblocked_if_done"])}

Artifacts:
- [family_coverage.csv](../{(OUTPUT_DIR / 'family_coverage.csv').as_posix()})
- [observed_primary_summary.csv](../{(OUTPUT_DIR / 'observed_primary_summary.csv').as_posix()})
- [claim_boundary.csv](../{(OUTPUT_DIR / 'claim_boundary.csv').as_posix()})
- [remaining_work.csv](../{(OUTPUT_DIR / 'remaining_work.csv').as_posix()})
- [config.json](../{(OUTPUT_DIR / 'config.json').as_posix()})
"""
    write_markdown(DISCUSSION_PATH, text)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_registry = pd.read_csv(EVAL_DIR / "run_registry.csv")
    decisions = pd.read_csv(EVAL_DIR / "primary_decisions.csv")
    gate_report = pd.read_csv(EVAL_DIR / "gate_report.csv")
    coverage = build_family_coverage(run_registry)
    observed_summary = build_observed_summary(decisions)
    claim_boundary = build_claim_boundary(decisions, gate_report)
    remaining_work = build_remaining_work(run_registry)
    observed_total = int(coverage["observed_primary_rows"].sum())
    missing_total = int(coverage["missing_primary_rows"].sum())
    raw_worse_total = int(
        observed_summary[observed_summary["scope"].eq("all_observed")]["raw_worse_rows"].iloc[0]
    )

    coverage.to_csv(OUTPUT_DIR / "family_coverage.csv", index=False)
    observed_summary.to_csv(OUTPUT_DIR / "observed_primary_summary.csv", index=False)
    claim_boundary.to_csv(OUTPUT_DIR / "claim_boundary.csv", index=False)
    remaining_work.to_csv(OUTPUT_DIR / "remaining_work.csv", index=False)
    (OUTPUT_DIR / "config.json").write_text(
        json.dumps(
            {
                "purpose": "interim synthesis for partial natural negative-search phase1 outputs",
                "analysis_scope": "claim-boundary synthesis; no new empirical computation beyond registered evaluator tables",
                "observed_primary_rows": observed_total,
                "missing_primary_rows": missing_total,
                "raw_worse_rows": raw_worse_total,
                "natural_claim_allowed": False,
                "full_family_required_rows": TOTAL_FAMILY_SIZE,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_discussion(coverage, observed_summary, claim_boundary, remaining_work)
    print(f"saved {DISCUSSION_PATH} and {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
