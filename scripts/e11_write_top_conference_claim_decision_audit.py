from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.reporting import markdown_table, write_markdown


OUTPUT_DIR = Path("results/e11_top_conference_claim_decision_audit")
DISCUSSION_PATH = Path("discussion/e11_top_conference_claim_decision_audit.md")

SOURCE_FILES = {
    "proof_obligations": Path("results/e11_theory_proof_obligation_register/proof_obligations.csv"),
    "claim_scope_boundaries": Path("results/e11_theory_proof_obligation_register/claim_scope_boundaries.csv"),
    "v5_final_gates": Path("results/e11_condition_score_v5_protocol/final_score_evaluation/final_gate_report.csv"),
    "natural_interim_claims": Path(
        "results/e11_natural_negative_search_protocol/phase1_interim_synthesis/claim_boundary.csv"
    ),
    "natural_interim_summary": Path(
        "results/e11_natural_negative_search_protocol/phase1_interim_synthesis/observed_primary_summary.csv"
    ),
    "tuned_benchmark_gates": Path(
        "results/e11_cifar100_resnet_lt_tuned_benchmark/validation_selection/gate_report.csv"
    ),
    "submission_build_gates": Path("results/e11_submission_repro_audit/build_gate_summary.csv"),
    "gap_register": Path("results/e11_top_conference_gap_register/gap_register.csv"),
}


def load_sources() -> dict[str, pd.DataFrame]:
    return {name: pd.read_csv(path) for name, path in SOURCE_FILES.items()}


def one(frame: pd.DataFrame, column: str, value: str) -> pd.Series:
    matches = frame[frame[column].astype(str).eq(value)]
    if len(matches) != 1:
        raise ValueError(f"expected one {column}={value}, found {len(matches)}")
    return matches.iloc[0]


def status_line(frame: pd.DataFrame, id_column: str, status_column: str = "status") -> str:
    return "; ".join(f"{row[id_column]}={row[status_column]}" for row in frame.to_dict("records"))


def build_claim_decision_matrix(sources: dict[str, pd.DataFrame]) -> pd.DataFrame:
    proof = sources["proof_obligations"]
    scopes = sources["claim_scope_boundaries"]
    v5_gates = sources["v5_final_gates"]
    natural_claims = sources["natural_interim_claims"]
    natural_summary = sources["natural_interim_summary"]
    tuned_gates = sources["tuned_benchmark_gates"]
    submission_gates = sources["submission_build_gates"]

    pto1 = one(proof, "obligation_id", "PTO-1-local-linearization")
    pto2 = one(proof, "obligation_id", "PTO-2-matrix-block-boundary")
    pto4 = one(proof, "obligation_id", "PTO-4-v5-transport-score")
    pto5 = one(proof, "obligation_id", "PTO-5-natural-falsification")
    pto6 = one(proof, "obligation_id", "PTO-6-final-performance-separation")
    main_scope = one(scopes, "claim_scope", "main_theorem")
    diagnostic_scope = one(scopes, "claim_scope", "natural_drift_diagnostic")
    predictive_scope = one(scopes, "claim_scope", "predictive_condition")
    counterexample_scope = one(scopes, "claim_scope", "natural_counterexample")
    benchmark_scope = one(scopes, "claim_scope", "optimizer_benchmark")
    natural_all = one(natural_summary, "scope", "all_observed")
    v5_p0 = one(v5_gates, "gate_id", "v5_p0_predictive_condition_claim")
    natural_complete = int(natural_all["missing_primary_rows"]) == 0
    natural_finite_null = "finite_null_candidate" in set(natural_claims["current_status"].astype(str))

    tuned_blockers = tuned_gates[tuned_gates["blocks_final_claim"].astype(str).str.lower().eq("yes")]
    submission_not_ready = submission_gates[submission_gates["status"].astype(str).eq("not_ready")]

    return pd.DataFrame(
        [
            {
                "claim_id": "TCD-1-main-mechanism-theorem",
                "paper_section": "main theorem and mechanism",
                "current_decision": "supportable_main_with_assumptions",
                "evidence_status": f"{pto1['current_status']}; {pto2['current_status']}",
                "author_allowed_wording": main_scope["allowed_claim"],
                "author_blocked_wording": main_scope["blocked_claim"],
                "decisive_gate": main_scope["decisive_gate"],
                "required_next_action": pto2["required_upgrade"],
                "source_artifacts": "discussion/e11_matrix_block_theorem_proof.md; discussion/e11_matrix_block_tightness_audit.md; results/e11_theory_proof_obligation_register/proof_obligations.csv",
            },
            {
                "claim_id": "TCD-2-natural-drift-diagnostic",
                "paper_section": "natural and ResNet diagnostics",
                "current_decision": "supportable_diagnostic_only",
                "evidence_status": diagnostic_scope["current_status"],
                "author_allowed_wording": diagnostic_scope["allowed_claim"],
                "author_blocked_wording": diagnostic_scope["blocked_claim"],
                "decisive_gate": diagnostic_scope["decisive_gate"],
                "required_next_action": "keep every drift claim paired with the quantitative ledger caveat separating logit drift from tail accuracy",
                "source_artifacts": "discussion/e11_quantitative_claim_ledger.md; discussion/e11_natural_head_tail_boundary.md",
            },
            {
                "claim_id": "TCD-3-predictive-condition-generalization",
                "paper_section": "predictive condition",
                "current_decision": "registered_not_ready_wait_for_v5_finals",
                "evidence_status": f"{pto4['current_status']}; {v5_p0['gate_id']}={v5_p0['status']}",
                "author_allowed_wording": predictive_scope["allowed_claim"],
                "author_blocked_wording": predictive_scope["blocked_claim"],
                "decisive_gate": predictive_scope["decisive_gate"],
                "required_next_action": pto4["required_upgrade"],
                "source_artifacts": "discussion/e11_condition_score_v5_validation_freeze.md; discussion/e11_condition_score_v5_final_evaluation.md; discussion/e11_condition_score_v5_reviewer_failure_response.md",
            },
            {
                "claim_id": "TCD-4-natural-counterexample-or-finite-null",
                "paper_section": "natural falsification",
                "current_decision": (
                    "finite_null_candidate_with_caveats"
                    if natural_complete and natural_finite_null
                    else "blocked_partial_family"
                ),
                "evidence_status": (
                    f"{pto5['current_status']}; observed={int(natural_all['observed_primary_rows'])}/"
                    f"{int(natural_all['observed_primary_rows']) + int(natural_all['missing_primary_rows'])}; "
                    f"raw_worse_rows={int(natural_all['raw_worse_rows'])}; "
                    f"quality_gate_pass_rows={int(natural_all['quality_gate_pass_rows'])}; "
                    f"quality_gate_fail_rows={int(natural_all['quality_gate_fail_rows'])}; "
                    f"{status_line(natural_claims, 'claim_id', 'current_status')}"
                ),
                "author_allowed_wording": counterexample_scope["allowed_claim"],
                "author_blocked_wording": "fresh natural primary counterexample; unqualified absence of natural counterexamples outside the registered phase1 space; quality-failed rows validate the mechanism",
                "decisive_gate": counterexample_scope["decisive_gate"],
                "required_next_action": pto5["required_upgrade"],
                "source_artifacts": "discussion/e11_natural_negative_search_phase1_evaluation.md; discussion/e11_natural_negative_search_phase1_interim_synthesis.md",
            },
            {
                "claim_id": "TCD-5-optimizer-performance-benchmark",
                "paper_section": "practical optimizer scope",
                "current_decision": "blocked_protocol_pending",
                "evidence_status": f"{pto6['current_status']}; {status_line(tuned_gates, 'gate_id')}",
                "author_allowed_wording": benchmark_scope["allowed_claim"],
                "author_blocked_wording": benchmark_scope["blocked_claim"],
                "decisive_gate": benchmark_scope["decisive_gate"],
                "required_next_action": pto6["required_upgrade"],
                "source_artifacts": "discussion/e11_cifar100_resnet_lt_tuned_benchmark_protocol.md; discussion/e11_cifar100_resnet_lt_tuned_benchmark_selection.md",
            },
            {
                "claim_id": "TCD-6-artifact-reproducibility",
                "paper_section": "artifact and submission package",
                "current_decision": "supportable_with_toolchain_caveat",
                "evidence_status": (
                    f"{status_line(submission_gates, 'gate_id')}; "
                    f"not_ready_count={len(submission_not_ready)}"
                ),
                "author_allowed_wording": "the current server evidence bundle validates and rendered PDFs exist under the documented fallback toolchain",
                "author_blocked_wording": "preferred pdflatex/bibtex/xelatex clean-checkout reproducibility is complete on this server",
                "decisive_gate": "R3-preferred-latex-toolchain plus clean-checkout make e11-full in a documented venue-style environment",
                "required_next_action": "run a clean checkout with pdflatex/bibtex/xelatex before claiming full venue-toolchain reproducibility",
                "source_artifacts": "discussion/e11_submission_repro_audit.md; results/e11_submission_repro_audit/build_gate_summary.csv",
            },
        ]
    )


def build_reviewer_objection_matrix(claims: pd.DataFrame) -> pd.DataFrame:
    lookup = claims.set_index("claim_id")
    return pd.DataFrame(
        [
            {
                "objection_id": "RO-1-toy-theorem",
                "likely_objection": "The theorem is a constructed block model and may not explain real networks.",
                "current_response": "Use TCD-1 as the theorem contract and TCD-2 as diagnostic-only real-model support.",
                "response_status": "answerable_with_scope",
                "missing_gate": lookup.loc["TCD-3-predictive-condition-generalization", "decisive_gate"],
                "forbidden_shortcut": "calling the synthetic sign boundary an out-of-sample natural predictor",
            },
            {
                "objection_id": "RO-2-score-cherry-picking",
                "likely_objection": "The condition score was selected after seeing failures.",
                "current_response": "v5 freezes a transport-normalized score before final outputs and quarantines spent v2/v3/v4 final rows.",
                "response_status": lookup.loc[
                    "TCD-3-predictive-condition-generalization", "current_decision"
                ],
                "missing_gate": "unspent ResNeXt50-32x4d and CIFAR-10 cross-partition final outputs",
                "forbidden_shortcut": "using any final row to refit or reselect the score",
            },
            {
                "objection_id": "RO-3-no-natural-negative",
                "likely_objection": "Only positive natural examples are available.",
                "current_response": "The natural negative-search family is registered, powered, complete for phase1, and reports 26/26 observed rows without a raw or adjusted primary worse row.",
                "response_status": lookup.loc["TCD-4-natural-counterexample-or-finite-null", "current_decision"],
                "missing_gate": "larger-family or held-out natural replication before broad natural-null wording",
                "forbidden_shortcut": "claiming a universal finite null or natural counterexample without adjusted primary evidence",
            },
            {
                "objection_id": "RO-4-muon-overclaim",
                "likely_objection": "The manuscript implies practical Muon benchmark superiority.",
                "current_response": "TCD-5 blocks benchmark wording and keeps Muon as motivation/local compatibility until tuned validation and final seeds finish.",
                "response_status": lookup.loc["TCD-5-optimizer-performance-benchmark", "current_decision"],
                "missing_gate": lookup.loc["TCD-5-optimizer-performance-benchmark", "decisive_gate"],
                "forbidden_shortcut": "turning lower local drift into a final tail-accuracy claim",
            },
            {
                "objection_id": "RO-5-artifact-reproducibility",
                "likely_objection": "The evidence package may not reproduce in a clean venue environment.",
                "current_response": "TCD-6 records the server validation, PDF fallback, and the remaining preferred-LaTeX clean-checkout gap.",
                "response_status": lookup.loc["TCD-6-artifact-reproducibility", "current_decision"],
                "missing_gate": lookup.loc["TCD-6-artifact-reproducibility", "decisive_gate"],
                "forbidden_shortcut": "calling the preferred LaTeX toolchain complete on a server where it is absent",
            },
        ]
    )


def build_paper_sequence(claims: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sequence_step": 1,
                "claim_id": "TCD-1-main-mechanism-theorem",
                "paper_move": "State the local matched-head-gain matrix-block theorem and its assumptions before broad empirical claims.",
                "writing_rule": "Every theorem paragraph must preserve the local-step and block-model scope.",
            },
            {
                "sequence_step": 2,
                "claim_id": "TCD-2-natural-drift-diagnostic",
                "paper_move": "Present digits/CIFAR/ResNet diagnostics as matched-head-gain tail-logit drift evidence.",
                "writing_rule": "Attach the tail-accuracy caveat from the quantitative claim ledger.",
            },
            {
                "sequence_step": 3,
                "claim_id": "TCD-3-predictive-condition-generalization",
                "paper_move": "Describe v5 as a preregistered pending held-out test rather than a successful predictor.",
                "writing_rule": "Update only after both unspent final splits pass the frozen evaluator.",
            },
            {
                "sequence_step": 4,
                "claim_id": "TCD-4-natural-counterexample-or-finite-null",
                "paper_move": "Report the natural negative-search family as a complete registered phase1 finite-null candidate with caveats.",
                "writing_rule": "Do not generalize beyond the 26-setting phase1 family or convert the null candidate into a natural counterexample claim.",
            },
            {
                "sequence_step": 5,
                "claim_id": "TCD-5-optimizer-performance-benchmark",
                "paper_move": "Keep practical Muon training results in scope-control unless the tuned benchmark protocol completes.",
                "writing_rule": "Do not let local drift diagnostics imply benchmark superiority.",
            },
            {
                "sequence_step": 6,
                "claim_id": "TCD-6-artifact-reproducibility",
                "paper_move": "State validated server artifacts and the remaining preferred-LaTeX reproducibility gap.",
                "writing_rule": "Keep artifact-review wording tied to the exact build gates.",
            },
        ]
    ).merge(claims[["claim_id", "current_decision"]], on="claim_id", how="left")


def build_rebuttal_response_pack(claims: pd.DataFrame, objections: pd.DataFrame) -> pd.DataFrame:
    claim_lookup = claims.set_index("claim_id")
    objection_lookup = objections.set_index("objection_id")
    return pd.DataFrame(
        [
            {
                "rebuttal_id": "RRP-1-toy-theorem-scope",
                "objection_id": "RO-1-toy-theorem",
                "claim_id": "TCD-1-main-mechanism-theorem",
                "response_posture": "answer_now_with_scope_and_real-diagnostic_bridge",
                "evidence_to_cite": "discussion/e11_matrix_block_theorem_proof.md; discussion/e11_matrix_block_tightness_audit.md; discussion/e11_quantitative_claim_ledger.md; discussion/e11_natural_head_tail_boundary.md",
                "manuscript_edit": "State the theorem as a local matched-head-gain mechanism before experiments and cite real-model diagnostics only as diagnostic support.",
                "missing_gate": objection_lookup.loc["RO-1-toy-theorem", "missing_gate"],
                "forbidden_rebuttal": objection_lookup.loc["RO-1-toy-theorem", "forbidden_shortcut"],
            },
            {
                "rebuttal_id": "RRP-2-score-selection-leakage",
                "objection_id": "RO-2-score-cherry-picking",
                "claim_id": "TCD-3-predictive-condition-generalization",
                "response_posture": claim_lookup.loc[
                    "TCD-3-predictive-condition-generalization", "current_decision"
                ],
                "evidence_to_cite": "discussion/e11_condition_score_v5_validation_freeze.md; discussion/e11_condition_score_v5_final_evaluation.md; discussion/e11_condition_score_v5_reviewer_failure_response.md",
                "manuscript_edit": "Describe v5 as a frozen pending held-out test; keep v2/v3/v4 as spent failures and do not use final rows for refit or score selection.",
                "missing_gate": objection_lookup.loc["RO-2-score-cherry-picking", "missing_gate"],
                "forbidden_rebuttal": objection_lookup.loc["RO-2-score-cherry-picking", "forbidden_shortcut"],
            },
            {
                "rebuttal_id": "RRP-3-natural-negative-boundary",
                "objection_id": "RO-3-no-natural-negative",
                "claim_id": "TCD-4-natural-counterexample-or-finite-null",
                "response_posture": claim_lookup.loc[
                    "TCD-4-natural-counterexample-or-finite-null", "current_decision"
                ],
                "evidence_to_cite": "discussion/e11_natural_negative_search_protocol.md; discussion/e11_natural_negative_search_phase1_power_audit.md; discussion/e11_natural_negative_search_phase1_interim_synthesis.md",
                "manuscript_edit": "Report the natural negative search as a complete registered phase1 family: 26/26 observed, raw_worse_rows=0, adjusted primary worse rows=0, and finite-null-candidate wording only with detectable-effect and tail-quality caveats.",
                "missing_gate": objection_lookup.loc["RO-3-no-natural-negative", "missing_gate"],
                "forbidden_rebuttal": objection_lookup.loc["RO-3-no-natural-negative", "forbidden_shortcut"],
            },
            {
                "rebuttal_id": "RRP-4-muon-performance-overclaim",
                "objection_id": "RO-4-muon-overclaim",
                "claim_id": "TCD-5-optimizer-performance-benchmark",
                "response_posture": claim_lookup.loc[
                    "TCD-5-optimizer-performance-benchmark", "current_decision"
                ],
                "evidence_to_cite": "discussion/e11_cifar100_resnet_lt_tuned_benchmark_protocol.md; discussion/e11_cifar100_resnet_lt_tuned_benchmark_selection.md; discussion/e11_quantitative_claim_ledger.md",
                "manuscript_edit": "Keep Muon as motivation and selected-state/local compatibility; quarantine benchmark claims until validation selection and untouched final seeds finish.",
                "missing_gate": objection_lookup.loc["RO-4-muon-overclaim", "missing_gate"],
                "forbidden_rebuttal": objection_lookup.loc["RO-4-muon-overclaim", "forbidden_shortcut"],
            },
            {
                "rebuttal_id": "RRP-5-artifact-clean-checkout",
                "objection_id": "RO-5-artifact-reproducibility",
                "claim_id": "TCD-6-artifact-reproducibility",
                "response_posture": claim_lookup.loc[
                    "TCD-6-artifact-reproducibility", "current_decision"
                ],
                "evidence_to_cite": "discussion/e11_submission_repro_audit.md; results/e11_submission_repro_audit/build_gate_summary.csv; discussion/e11_artifact_manifest.md",
                "manuscript_edit": "State server validation and rendered PDFs, but keep preferred-LaTeX clean-checkout completion as a remaining artifact-review gate.",
                "missing_gate": objection_lookup.loc["RO-5-artifact-reproducibility", "missing_gate"],
                "forbidden_rebuttal": objection_lookup.loc[
                    "RO-5-artifact-reproducibility", "forbidden_shortcut"
                ],
            },
        ]
    )


def build_manuscript_edit_queue(claims: pd.DataFrame, gaps: pd.DataFrame) -> pd.DataFrame:
    claim_lookup = claims.set_index("claim_id")
    gap_lookup = gaps.set_index("gap_id")
    return pd.DataFrame(
        [
            {
                "edit_id": "MEQ-1-theory-frontload-scope",
                "target_section": "Theory / introduction",
                "claim_id": "TCD-1-main-mechanism-theorem",
                "edit_action": "Front-load local matched-head-gain assumptions, worst-case-vs-realized distinction, and theorem caveats before empirical interpretation.",
                "acceptance_check": "The section must contain local-step, matched-head-gain, sandwich-block, and not-global-optimizer wording.",
                "current_decision": claim_lookup.loc[
                    "TCD-1-main-mechanism-theorem", "current_decision"
                ],
            },
            {
                "edit_id": "MEQ-2-diagnostic-caveat-next-to-results",
                "target_section": "Experiments",
                "claim_id": "TCD-2-natural-drift-diagnostic",
                "edit_action": "Place the tail-loss/margin/accuracy caveat immediately next to every logit-drift table or figure.",
                "acceptance_check": "No result paragraph may convert matched-head-gain logit drift into tail-accuracy or benchmark wording.",
                "current_decision": claim_lookup.loc[
                    "TCD-2-natural-drift-diagnostic", "current_decision"
                ],
            },
            {
                "edit_id": "MEQ-3-v5-pending-test-language",
                "target_section": "Predictive condition / limitations",
                "claim_id": "TCD-3-predictive-condition-generalization",
                "edit_action": "Describe the v5 score as a frozen pending test and cite the registered final evaluator plus reviewer failure response.",
                "acceptance_check": gap_lookup.loc["P0-PredictiveCondition", "acceptance_gate"],
                "current_decision": claim_lookup.loc[
                    "TCD-3-predictive-condition-generalization", "current_decision"
                ],
            },
            {
                "edit_id": "MEQ-4-natural-negative-incomplete-family",
                "target_section": "Natural boundary cases",
                "claim_id": "TCD-4-natural-counterexample-or-finite-null",
                "edit_action": "Report 26/26 observed, raw_worse_rows=0, and no adjusted primary worse row as a finite registered phase1 null candidate with quality and detectable-effect caveats.",
                "acceptance_check": gap_lookup.loc["P2-NaturalBoundaryCases", "acceptance_gate"],
                "current_decision": claim_lookup.loc[
                    "TCD-4-natural-counterexample-or-finite-null", "current_decision"
                ],
            },
            {
                "edit_id": "MEQ-5-performance-benchmark-quarantine",
                "target_section": "Practical Muon / limitations",
                "claim_id": "TCD-5-optimizer-performance-benchmark",
                "edit_action": "Keep standard/recipe/negative NS-Muon pilots as benchmark context and quarantine all competitive optimizer wording.",
                "acceptance_check": gap_lookup.loc["P0-StandardBenchmark", "acceptance_gate"],
                "current_decision": claim_lookup.loc[
                    "TCD-5-optimizer-performance-benchmark", "current_decision"
                ],
            },
            {
                "edit_id": "MEQ-6-artifact-review-caveat",
                "target_section": "Reproducibility",
                "claim_id": "TCD-6-artifact-reproducibility",
                "edit_action": "State current server validation and PDF fallback while explicitly naming the preferred-LaTeX clean-checkout gap.",
                "acceptance_check": gap_lookup.loc["P2-PackagingRepro", "acceptance_gate"],
                "current_decision": claim_lookup.loc[
                    "TCD-6-artifact-reproducibility", "current_decision"
                ],
            },
        ]
    )


def write_discussion(
    claims: pd.DataFrame,
    objections: pd.DataFrame,
    sequence: pd.DataFrame,
    readiness_summary: pd.DataFrame,
    rebuttal_pack: pd.DataFrame,
    manuscript_queue: pd.DataFrame,
) -> None:
    text = f"""# E11 Top-Conference Claim Decision Audit

This generated audit is the paper-level claim contract. It is stricter than the
quantitative claim ledger: each row says whether a top-conference manuscript can
write a claim now, must present it as a registered pending test, or must block
the wording until a named gate completes. It is generated from existing
proof-obligation, v5 final-evaluator, natural-negative, tuned-benchmark, and
submission-reproducibility tables; it does not add new empirical results.

## Readiness Summary

{markdown_table(readiness_summary, ["current_decision", "claim_count"])}

## Claim Decision Matrix

{markdown_table(claims, ["claim_id", "paper_section", "current_decision", "evidence_status", "author_allowed_wording", "author_blocked_wording", "decisive_gate", "required_next_action", "source_artifacts"])}

## Reviewer Objection Matrix

{markdown_table(objections, ["objection_id", "likely_objection", "current_response", "response_status", "missing_gate", "forbidden_shortcut"])}

## Rebuttal Response Pack

{markdown_table(rebuttal_pack, ["rebuttal_id", "objection_id", "claim_id", "response_posture", "evidence_to_cite", "manuscript_edit", "missing_gate", "forbidden_rebuttal"])}

## Manuscript Edit Queue

{markdown_table(manuscript_queue, ["edit_id", "target_section", "claim_id", "edit_action", "acceptance_check", "current_decision"])}

## Paper Sequence

{markdown_table(sequence, ["sequence_step", "claim_id", "current_decision", "paper_move", "writing_rule"])}

## Operating Rule

Allowed manuscript wording is limited to rows whose `current_decision` starts
with `supportable`. Rows marked `registered_not_ready` or `blocked` can be
reported as protocols, pending tests, or claim boundaries only. Rows marked
`finite_null_candidate` can be reported only with their stated caveats. No row
can be used as a stronger positive claim unless its decisive gate is rerun and
this audit is regenerated.
"""
    write_markdown(DISCUSSION_PATH, text)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sources = load_sources()
    claims = build_claim_decision_matrix(sources)
    objections = build_reviewer_objection_matrix(claims)
    sequence = build_paper_sequence(claims)
    rebuttal_pack = build_rebuttal_response_pack(claims, objections)
    manuscript_queue = build_manuscript_edit_queue(claims, sources["gap_register"])
    readiness_summary = (
        claims.groupby("current_decision", sort=True)
        .size()
        .reset_index(name="claim_count")
        .sort_values(["current_decision"])
    )

    claims.to_csv(OUTPUT_DIR / "claim_decision_matrix.csv", index=False)
    objections.to_csv(OUTPUT_DIR / "reviewer_objection_matrix.csv", index=False)
    rebuttal_pack.to_csv(OUTPUT_DIR / "rebuttal_response_pack.csv", index=False)
    manuscript_queue.to_csv(OUTPUT_DIR / "manuscript_edit_queue.csv", index=False)
    sequence.to_csv(OUTPUT_DIR / "paper_sequence.csv", index=False)
    readiness_summary.to_csv(OUTPUT_DIR / "readiness_summary.csv", index=False)
    (OUTPUT_DIR / "config.json").write_text(
        json.dumps(
            {
                "purpose": "top-conference manuscript claim decision contract",
                "source_files": {name: path.as_posix() for name, path in SOURCE_FILES.items()},
                "new_empirical_results": False,
                "supportable_prefix": "supportable",
                "blocked_decisions": ["registered_not_ready_wait_for_v5_finals", "blocked_partial_family", "blocked_protocol_pending"],
                "caveated_decisions": ["finite_null_candidate_with_caveats"],
                "rebuttal_ready": True,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_discussion(claims, objections, sequence, readiness_summary, rebuttal_pack, manuscript_queue)
    print(f"saved {DISCUSSION_PATH} and {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
