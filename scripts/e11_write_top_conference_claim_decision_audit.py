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
    "natural_phase2_gates": Path(
        "results/e11_natural_negative_search_protocol/phase2_multiplicity_evaluation/gate_report.csv"
    ),
    "natural_phase2_decisions": Path(
        "results/e11_natural_negative_search_protocol/phase2_multiplicity_evaluation/primary_decisions.csv"
    ),
    "tuned_benchmark_gates": Path(
        "results/e11_cifar100_resnet_lt_tuned_benchmark/validation_selection/gate_report.csv"
    ),
    "tuned_leakage_guards": Path(
        "results/e11_cifar100_resnet_lt_tuned_benchmark/validation_leakage_audit/leakage_guard_matrix.csv"
    ),
    "tuned_refresh_state": Path(
        "results/e11_cifar100_resnet_lt_tuned_benchmark/validation_refresh_firewall/refresh_state.csv"
    ),
    "tuned_refresh_allowed": Path(
        "results/e11_cifar100_resnet_lt_tuned_benchmark/validation_refresh_firewall/allowed_transition_matrix.csv"
    ),
    "tuned_refresh_forbidden": Path(
        "results/e11_cifar100_resnet_lt_tuned_benchmark/validation_refresh_firewall/forbidden_action_matrix.csv"
    ),
    "tuned_final_execution_gates": Path(
        "results/e11_cifar100_resnet_lt_tuned_benchmark/final_execution_plan/gate_matrix.csv"
    ),
    "tuned_final_eval_gates": Path(
        "results/e11_cifar100_resnet_lt_tuned_benchmark/final_evaluation/claim_gate_report.csv"
    ),
    "tuned_final_launch_gates": Path(
        "results/e11_cifar100_resnet_lt_tuned_benchmark/final_launch_audit/latest_gate_snapshot.csv"
    ),
    "tuned_final_launch_decision": Path(
        "results/e11_cifar100_resnet_lt_tuned_benchmark/final_launch_audit/latest_final_launch_decision.csv"
    ),
    "muon_state_terms": Path("results/e11_muon_state_distribution_contract/state_distribution_terms.csv"),
    "muon_claim_gates": Path("results/e11_muon_state_distribution_contract/claim_gate_ladder.csv"),
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
    phase2_gates = sources["natural_phase2_gates"]
    phase2_decisions = sources["natural_phase2_decisions"]
    tuned_gates = sources["tuned_benchmark_gates"]
    tuned_leakage = sources["tuned_leakage_guards"]
    tuned_refresh_state = sources["tuned_refresh_state"]
    tuned_refresh_allowed = sources["tuned_refresh_allowed"]
    tuned_refresh_forbidden = sources["tuned_refresh_forbidden"]
    tuned_final_execution_gates = sources["tuned_final_execution_gates"]
    tuned_final_eval_gates = sources["tuned_final_eval_gates"]
    tuned_final_launch_gates = sources["tuned_final_launch_gates"]
    tuned_final_launch_decision = sources["tuned_final_launch_decision"]
    muon_terms = sources["muon_state_terms"]
    muon_gates = sources["muon_claim_gates"]
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
    msd_t1 = one(muon_terms, "term_id", "MSD-T1-local-response-integrand")
    msd_t2 = one(muon_terms, "term_id", "MSD-T2-state-occupancy-measure")
    msd_t4 = one(muon_terms, "term_id", "MSD-T4-terminal-risk-functional")
    msg_4 = one(muon_gates, "gate_id", "MSG-4-top-tier-practical-claim")
    natural_complete = int(natural_all["missing_primary_rows"]) == 0
    natural_finite_null = "finite_null_candidate" in set(natural_claims["current_status"].astype(str))
    phase2_gate_lookup = phase2_gates.set_index("gate_id")["status"].astype(str).to_dict()
    phase2_observed = int(phase2_decisions["output_status"].eq("observed").sum())
    phase2_total = int(len(phase2_decisions))
    phase2_adjusted_worse = int(phase2_decisions["adjusted_primary_decision"].eq("primary_worse_adjusted").sum())
    phase2_head_gain = phase2_decisions["head_gain_gate"].astype(str).str.lower().eq("true")
    phase2_head_gain_fail = int((~phase2_head_gain).sum())
    phase2_complete_finite_null = (
        phase2_observed == phase2_total
        and phase2_gate_lookup.get("NNS-P2-E4-heldout-architecture-claim") == "finite_null_candidate"
    )

    tuned_blockers = tuned_gates[tuned_gates["blocks_final_claim"].astype(str).str.lower().eq("yes")]
    submission_not_ready = submission_gates[submission_gates["status"].astype(str).eq("not_ready")]
    launch_row = tuned_final_launch_decision.iloc[0]
    launch_status = str(launch_row["submission_status"])
    submit_command = str(pd.Series([launch_row.get("submit_command", "")]).fillna("").iloc[0])

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
                "current_decision": "blocked_completed_final_failed_boundary",
                "evidence_status": f"{pto4['current_status']}; {v5_p0['gate_id']}={v5_p0['status']}",
                "author_allowed_wording": predictive_scope["allowed_claim"],
                "author_blocked_wording": predictive_scope["blocked_claim"],
                "decisive_gate": predictive_scope["decisive_gate"],
                "required_next_action": pto4["required_upgrade"],
                "source_artifacts": "discussion/e11_condition_score_v5_validation_freeze.md; discussion/e11_condition_score_v5_final_evaluation.md; discussion/e11_condition_score_v5_reviewer_failure_response.md; discussion/e11_condition_score_v5_direction_guardrail_failure_audit.md",
            },
            {
                "claim_id": "TCD-4-natural-counterexample-or-finite-null",
                "paper_section": "natural falsification",
                "current_decision": (
                    "finite_null_candidate_with_caveats"
                    if natural_complete and natural_finite_null and phase2_complete_finite_null
                    else "blocked_partial_family"
                ),
                "evidence_status": (
                    f"{pto5['current_status']}; observed={int(natural_all['observed_primary_rows'])}/"
                    f"{int(natural_all['observed_primary_rows']) + int(natural_all['missing_primary_rows'])}; "
                    f"raw_worse_rows={int(natural_all['raw_worse_rows'])}; "
                    f"quality_gate_pass_rows={int(natural_all['quality_gate_pass_rows'])}; "
                    f"quality_gate_fail_rows={int(natural_all['quality_gate_fail_rows'])}; "
                    f"{status_line(natural_claims, 'claim_id', 'current_status')}; "
                    f"phase2_observed={phase2_observed}/{phase2_total}; "
                    f"NNS-P2-E4={phase2_gate_lookup.get('NNS-P2-E4-heldout-architecture-claim')}; "
                    f"phase2_adjusted_worse_rows={phase2_adjusted_worse}; "
                    f"phase2_head_gain_gate_fail_rows={phase2_head_gain_fail}"
                ),
                "author_allowed_wording": "registered phase1 and phase2 searches found no adjusted primary full-drift counterexample in the declared finite families, with detectable-effect, head-gain, and quality caveats",
                "author_blocked_wording": "fresh natural primary counterexample; unqualified absence of natural counterexamples outside the registered phase1/phase2 spaces; quality-failed or head-gain-failed rows validate the mechanism",
                "decisive_gate": counterexample_scope["decisive_gate"],
                "required_next_action": pto5["required_upgrade"],
                "source_artifacts": "discussion/e11_natural_negative_search_phase1_evaluation.md; discussion/e11_natural_negative_search_phase1_interim_synthesis.md; discussion/e11_natural_negative_search_phase2_evaluation.md",
            },
            {
                "claim_id": "TCD-5-optimizer-performance-benchmark",
                "paper_section": "practical optimizer scope",
                "current_decision": "blocked_protocol_pending",
                "evidence_status": (
                    f"{pto6['current_status']}; {status_line(tuned_gates, 'gate_id')}; "
                    f"{status_line(tuned_leakage, 'guard_id')}; "
                    f"{status_line(tuned_refresh_state, 'state_id')}; "
                    f"{status_line(tuned_refresh_allowed, 'transition_id')}; "
                    f"{status_line(tuned_refresh_forbidden, 'forbidden_id')}; "
                    f"{status_line(tuned_final_execution_gates, 'gate_id')}; "
                    f"{status_line(tuned_final_eval_gates, 'gate_id')}; "
                    f"{status_line(tuned_final_launch_gates, 'gate_id')}; "
                    f"final_launch_status={launch_status}; final_submit_command={submit_command or '<empty>'}; "
                    f"{msd_t1['term_id']}={msd_t1['careful_status']}; "
                    f"{msd_t2['term_id']}={msd_t2['careful_status']}; "
                    f"{msd_t4['term_id']}={msd_t4['careful_status']}; "
                    f"{msg_4['gate_id']}={msg_4['current_status']}"
                ),
                "author_allowed_wording": (
                    f"{benchmark_scope['allowed_claim']}; state-distribution transport contract "
                    "allows sampled-state local compatibility and negative final pilot boundary only; "
                    "validation leakage audit permits progress accounting only while selection rule, "
                    "array order, and final seed quarantine remain frozen; the validation refresh firewall "
                    "permits only sequential progress accounting from the contiguous prefix; final execution, evaluator, "
                    "and launch audits are dry-run/not_ready boundaries until final rows exist"
                ),
                "author_blocked_wording": (
                    f"{benchmark_scope['blocked_claim']}; local Muon drift compatibility implies "
                    "benchmark superiority; sampled bridge states represent the full training trajectory distribution; "
                    "using partial validation leaderboard to change selection, launch order, or final seed plan; "
                    "violating the validation refresh firewall forbidden-action matrix; "
                    "running final-safe-submit before FEP/TFE/FLA gates pass; claiming final benchmark "
                    "performance from not_ready final evaluator gates"
                ),
                "decisive_gate": (
                    f"{benchmark_scope['decisive_gate']} plus final execution/evaluation/launch gates, "
                    "state-distribution occupancy logging, and MSG-4 top-tier practical gate, with TLA "
                    "leakage guards and VRF refresh-firewall transitions still passing"
                ),
                "required_next_action": (
                    f"{pto6['required_upgrade']} plus record trajectory occupancy summaries during "
                    "tuned validation and final seeds; rerun the leakage/optional-stopping audit, "
                    "validation refresh firewall, and "
                    "final execution/evaluation/launch audits after each validation refresh; submit final "
                    "jobs only after FEP/TFE/FLA gates pass"
                ),
                "source_artifacts": (
                    "discussion/e11_cifar100_resnet_lt_tuned_benchmark_protocol.md; "
                    "discussion/e11_cifar100_resnet_lt_tuned_benchmark_selection.md; "
                    "discussion/e11_cifar100_resnet_lt_tuned_benchmark_leakage_audit.md; "
                    "discussion/e11_cifar100_resnet_lt_tuned_benchmark_refresh_firewall.md; "
                    "discussion/e11_cifar100_resnet_lt_tuned_benchmark_final_execution_plan.md; "
                    "discussion/e11_cifar100_resnet_lt_tuned_benchmark_final_evaluation.md; "
                    "discussion/e11_cifar100_resnet_lt_tuned_benchmark_final_launch_audit.md; "
                    "results/e11_cifar100_resnet_lt_tuned_benchmark/validation_leakage_audit/leakage_guard_matrix.csv; "
                    "results/e11_cifar100_resnet_lt_tuned_benchmark/validation_refresh_firewall/refresh_state.csv; "
                    "results/e11_cifar100_resnet_lt_tuned_benchmark/validation_refresh_firewall/allowed_transition_matrix.csv; "
                    "results/e11_cifar100_resnet_lt_tuned_benchmark/validation_refresh_firewall/forbidden_action_matrix.csv; "
                    "results/e11_cifar100_resnet_lt_tuned_benchmark/final_execution_plan/gate_matrix.csv; "
                    "results/e11_cifar100_resnet_lt_tuned_benchmark/final_evaluation/claim_gate_report.csv; "
                    "results/e11_cifar100_resnet_lt_tuned_benchmark/final_launch_audit/latest_final_launch_decision.csv; "
                    "discussion/e11_muon_state_distribution_contract.md; "
                    "results/e11_muon_state_distribution_contract/claim_gate_ladder.csv"
                ),
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
                "current_response": "v5 froze a transport-normalized score before final outputs, consumed both final splits, preserved the failures, and quarantines spent v2/v3/v4/v5 final rows.",
                "response_status": lookup.loc[
                    "TCD-3-predictive-condition-generalization", "current_decision"
                ],
                "missing_gate": "new unspent validation/final split family before any repaired score can make a renewed P0 attempt",
                "forbidden_shortcut": "using any final row to refit or reselect the score",
            },
            {
                "objection_id": "RO-3-no-natural-negative",
                "likely_objection": "Only positive natural examples are available.",
                "current_response": "The natural negative-search family is registered, powered, complete for phase1 and phase2, and reports 26/26 phase1 rows plus 8/8 phase2 rows without an adjusted primary worse row.",
                "response_status": lookup.loc["TCD-4-natural-counterexample-or-finite-null", "current_decision"],
                "missing_gate": "larger-family, larger-dataset, or additional held-out natural replication before broad natural-null wording",
                "forbidden_shortcut": "claiming a universal finite null or natural counterexample without adjusted primary evidence",
            },
            {
                "objection_id": "RO-4-muon-overclaim",
                "likely_objection": "The manuscript implies practical Muon benchmark superiority.",
                "current_response": (
                    "TCD-5 blocks benchmark wording; the Muon state-distribution contract separates "
                    "sampled-state local compatibility from trajectory occupancy and notes local "
                    "compatibility can coexist with poor final performance until tuned validation, "
                    "occupancy logging, and final seeds finish. The tuned leakage audit and final "
                    "validation refresh firewall plus final execution/evaluator/launch audits separately block optional-stopping moves, "
                    "premature final-safe-submit, and final-performance claims from partial validation."
                ),
                "response_status": lookup.loc["TCD-5-optimizer-performance-benchmark", "current_decision"],
                "missing_gate": lookup.loc["TCD-5-optimizer-performance-benchmark", "decisive_gate"],
                "forbidden_shortcut": (
                    "turning lower local drift into a final tail-accuracy claim or treating sampled "
                    "bridge states as the full training trajectory distribution or using partial validation "
                    "leaderboard to change selection, launch order, or final seed plan or violating the "
                    "validation refresh firewall forbidden-action matrix or running "
                    "final-safe-submit before FEP/TFE/FLA gates pass"
                ),
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
                "paper_move": "Describe v5 as a preregistered completed negative held-out boundary rather than a successful predictor.",
                "writing_rule": "Preserve both failed final gates and open a new protocol before any repaired predictor claim.",
            },
            {
                "sequence_step": 4,
                "claim_id": "TCD-4-natural-counterexample-or-finite-null",
                "paper_move": "Report the natural negative-search family as complete registered phase1 and phase2 finite-null candidates with caveats.",
                "writing_rule": "Do not generalize beyond the 26-setting phase1 and 8-setting phase2 families or convert the null candidate into a natural counterexample claim.",
            },
            {
                "sequence_step": 5,
                "claim_id": "TCD-5-optimizer-performance-benchmark",
                "paper_move": (
                    "Keep practical Muon training results in scope-control and route them through "
                    "the state-distribution transport contract unless tuned benchmark protocol and "
                    "occupancy logging complete, and cite the leakage plus final-gate audits while "
                    "validation/final execution remains partial."
                ),
                "writing_rule": "Do not let sampled local drift diagnostics, partial validation leaderboards, or dry-run final launch audits imply benchmark superiority, state occupancy, selection authority, or final-submit readiness.",
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
                "evidence_to_cite": "discussion/e11_condition_score_v5_validation_freeze.md; discussion/e11_condition_score_v5_final_evaluation.md; discussion/e11_condition_score_v5_reviewer_failure_response.md; discussion/e11_condition_score_v5_direction_guardrail_failure_audit.md",
                "manuscript_edit": "Describe v5 as a frozen completed held-out failure boundary; keep v2/v3/v4/v5 finals as spent failures and do not use final rows for refit or score selection.",
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
                "evidence_to_cite": "discussion/e11_natural_negative_search_protocol.md; discussion/e11_natural_negative_search_phase1_power_audit.md; discussion/e11_natural_negative_search_phase1_interim_synthesis.md; discussion/e11_natural_negative_search_phase2_evaluation.md; discussion/e11_natural_negative_search_phase2_power_audit.md",
                "manuscript_edit": "Report the natural negative search as complete registered phase1 and phase2 families: 26/26 phase1 observed, 8/8 phase2 observed, adjusted primary worse rows=0, and finite-null-candidate wording only with detectable-effect, head-gain, and quality caveats.",
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
                "evidence_to_cite": (
                    "discussion/e11_cifar100_resnet_lt_tuned_benchmark_protocol.md; "
                    "discussion/e11_cifar100_resnet_lt_tuned_benchmark_selection.md; "
                    "discussion/e11_cifar100_resnet_lt_tuned_benchmark_leakage_audit.md; "
                    "discussion/e11_cifar100_resnet_lt_tuned_benchmark_refresh_firewall.md; "
                    "discussion/e11_cifar100_resnet_lt_tuned_benchmark_final_execution_plan.md; "
                    "discussion/e11_cifar100_resnet_lt_tuned_benchmark_final_evaluation.md; "
                    "discussion/e11_cifar100_resnet_lt_tuned_benchmark_final_launch_audit.md; "
                    "results/e11_cifar100_resnet_lt_tuned_benchmark/validation_leakage_audit/leakage_guard_matrix.csv; "
                    "results/e11_cifar100_resnet_lt_tuned_benchmark/validation_refresh_firewall/refresh_state.csv; "
                    "results/e11_cifar100_resnet_lt_tuned_benchmark/validation_refresh_firewall/allowed_transition_matrix.csv; "
                    "results/e11_cifar100_resnet_lt_tuned_benchmark/validation_refresh_firewall/forbidden_action_matrix.csv; "
                    "results/e11_cifar100_resnet_lt_tuned_benchmark/final_execution_plan/gate_matrix.csv; "
                    "results/e11_cifar100_resnet_lt_tuned_benchmark/final_evaluation/claim_gate_report.csv; "
                    "results/e11_cifar100_resnet_lt_tuned_benchmark/final_launch_audit/latest_final_launch_decision.csv; "
                    "discussion/e11_muon_state_distribution_contract.md; "
                    "results/e11_muon_state_distribution_contract/claim_gate_ladder.csv; "
                    "discussion/e11_quantitative_claim_ledger.md"
                ),
                "manuscript_edit": (
                    "Keep Muon as motivation, selected-state/local compatibility, and a "
                    "state-distribution transport contract: local compatibility can coexist with poor "
                    "final performance; quarantine benchmark claims until validation selection, "
                    "occupancy logging, leakage guards, validation refresh firewall transitions, final execution/evaluation/launch gates, and "
                    "untouched final seeds finish."
                ),
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
                "edit_id": "MEQ-3-v5-completed-boundary-language",
                "target_section": "Predictive condition / limitations",
                "claim_id": "TCD-3-predictive-condition-generalization",
                "edit_action": "Describe the v5 score as a frozen completed negative boundary and cite the registered final evaluator, reviewer failure response, and direction-guardrail failure audit.",
                "acceptance_check": gap_lookup.loc["P0-PredictiveCondition", "acceptance_gate"],
                "current_decision": claim_lookup.loc[
                    "TCD-3-predictive-condition-generalization", "current_decision"
                ],
            },
            {
                "edit_id": "MEQ-4-natural-negative-complete-finite-family",
                "target_section": "Natural boundary cases",
                "claim_id": "TCD-4-natural-counterexample-or-finite-null",
                "edit_action": "Report 26/26 phase1 observed and 8/8 phase2 observed with no adjusted primary worse row as finite registered null candidates with detectable-effect, head-gain, and quality caveats.",
                "acceptance_check": gap_lookup.loc["P2-NaturalBoundaryCases", "acceptance_gate"],
                "current_decision": claim_lookup.loc[
                    "TCD-4-natural-counterexample-or-finite-null", "current_decision"
                ],
            },
            {
                "edit_id": "MEQ-5-performance-benchmark-quarantine",
                "target_section": "Practical Muon / limitations",
                "claim_id": "TCD-5-optimizer-performance-benchmark",
                "edit_action": (
                    "Keep standard/recipe/negative NS-Muon pilots as benchmark context, cite the "
                    "state-distribution contract, tuned leakage audit, and validation refresh firewall, and quarantine all competitive "
                    "optimizer wording until final execution/evaluator/launch gates pass."
                ),
                "acceptance_check": (
                    f"{gap_lookup.loc['P0-StandardBenchmark', 'acceptance_gate']} Also require "
                    "state-distribution occupancy summaries, passing leakage guards, and passing refresh-firewall transitions for selected recipes "
                    "plus passing final execution/evaluator/launch gates before practical-performance wording."
                ),
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
write a claim now, must present it as a registered pending or completed negative
boundary, or must block the wording until a named gate completes. It is generated from existing
proof-obligation, v5 final-evaluator, natural-negative, tuned-benchmark,
Muon state-distribution, and submission-reproducibility tables; it does not add new empirical results.

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
with `supportable`. Rows marked `blocked_completed_final_failed_boundary` or
`blocked` can be reported as protocols, completed negative boundaries, or claim boundaries only. Rows marked
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
                "blocked_decisions": [
                    "blocked_completed_final_failed_boundary",
                    "blocked_partial_family",
                    "blocked_protocol_pending",
                ],
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
