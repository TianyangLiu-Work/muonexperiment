from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.reporting import fmt, markdown_table, write_markdown


OUTPUT_DIR = Path("results/e11_theory_proof_obligation_register")
DISCUSSION_PATH = Path("discussion/e11_theory_proof_obligation_register.md")


def ci(row: pd.Series, low: str, high: str) -> str:
    return f"[{fmt(row[low])}, {fmt(row[high])}]"


def status_lookup(frame: pd.DataFrame, key_col: str, value_col: str = "status") -> dict[str, str]:
    return {str(row[key_col]): str(row[value_col]) for _, row in frame.iterrows()}


def load_evidence() -> dict[str, object]:
    head_tail = pd.read_csv("results/e11_head_tail_interference/pair_summary.csv").set_index("setting")
    local_linearization = pd.read_csv("results/e11_local_linearization/summary.csv").set_index("direction")
    v5_freeze = pd.read_csv("results/e11_condition_score_v5_protocol/validation_score_freeze/freeze_status.csv")
    v5_final_gates = pd.read_csv("results/e11_condition_score_v5_protocol/final_score_evaluation/final_gate_report.csv")
    natural_gates = pd.read_csv("results/e11_natural_negative_search_protocol/phase1_multiplicity_evaluation/gate_report.csv")
    natural_decisions = pd.read_csv("results/e11_natural_negative_search_protocol/phase1_multiplicity_evaluation/primary_decisions.csv")
    natural_phase2_gates = pd.read_csv(
        "results/e11_natural_negative_search_protocol/phase2_multiplicity_evaluation/gate_report.csv"
    )
    natural_phase2_decisions = pd.read_csv(
        "results/e11_natural_negative_search_protocol/phase2_multiplicity_evaluation/primary_decisions.csv"
    )
    tuned_gates = pd.read_csv("results/e11_cifar100_resnet_lt_tuned_benchmark/validation_selection/gate_report.csv")
    v5_response_config = json.loads(
        Path("results/e11_condition_score_v5_protocol/reviewer_failure_response/config.json").read_text(
            encoding="utf-8"
        )
    )
    return {
        "head_tail": head_tail,
        "local_linearization": local_linearization,
        "v5_freeze": v5_freeze,
        "v5_final_gates": v5_final_gates,
        "natural_gates": natural_gates,
        "natural_decisions": natural_decisions,
        "natural_phase2_gates": natural_phase2_gates,
        "natural_phase2_decisions": natural_phase2_decisions,
        "tuned_gates": tuned_gates,
        "v5_response_config": v5_response_config,
    }


def build_proof_obligations(evidence: dict[str, object]) -> pd.DataFrame:
    head_tail = evidence["head_tail"]
    local_linearization = evidence["local_linearization"]
    v5_freeze_status = status_lookup(evidence["v5_freeze"], "item")
    v5_final_status = status_lookup(evidence["v5_final_gates"], "gate_id")
    natural_status = status_lookup(evidence["natural_gates"], "gate_id")
    natural_observed_count = int(evidence["natural_decisions"]["output_status"].eq("observed").sum())
    natural_phase2_status = status_lookup(evidence["natural_phase2_gates"], "gate_id")
    natural_phase2_observed_count = int(evidence["natural_phase2_decisions"]["output_status"].eq("observed").sum())
    natural_phase2_head_gain_fail_count = int(
        evidence["natural_phase2_decisions"]["head_gain_gate"].astype(str).str.lower().eq("false").sum()
    )
    tuned_status = status_lookup(evidence["tuned_gates"], "gate_id")
    selected_score = str(evidence["v5_response_config"]["primary_score"])
    natural_complete = (
        natural_status.get("NNS-E2-phase1-output-completeness") == "pass"
        and natural_status.get("NNS-E3-primary-multiplicity") == "pass"
    )
    natural_finite_null = natural_status.get("NNS-E4-natural-primary-claim") == "finite_null_candidate"
    natural_phase2_finite_null = (
        natural_phase2_status.get("NNS-P2-E2-phase2-output-completeness") == "pass"
        and natural_phase2_status.get("NNS-P2-E4-heldout-architecture-claim") == "finite_null_candidate"
    )

    positive = head_tail.loc["high_head_rank_low_tail_srank"]
    negative = head_tail.loc["low_head_rank_high_tail_srank"]
    max_linearization_error = float(local_linearization["relative_error_ci95_high"].max())

    rows = [
        {
            "obligation_id": "PTO-1-local-linearization",
            "paper_claim": "Tail-logit drift can be analyzed by a first-order layerwise response at matched head gain.",
            "formal_object": "Taylor/JVP expansion of held-out tail logits under a head-gain-normalized update direction.",
            "assumptions_to_state": "small update scale; fixed checkpoint; same batch, seed, and head-gain target across compared directions",
            "current_evidence": f"largest local-linearization relative-error CI high={fmt(max_linearization_error)} across Fro, polar, momentum-polar, and NS directions",
            "current_status": "empirical_assumption_check_passes",
            "blocks_main_theory_claim": "yes_if_removed",
            "required_upgrade": "state smoothness/local-step assumptions in the theorem and keep finite-difference residuals in the appendix",
            "forbidden_wording": "do not claim a global training-dynamics theorem from one-step linearization evidence",
        },
        {
            "obligation_id": "PTO-2-matrix-block-boundary",
            "paper_claim": "The rank/sensitivity condition gives the sign boundary for spectral-vs-Frobenius tail drift in the constructed mechanism.",
            "formal_object": "sandwiched matrix-block tail risk ||B_T D A_T||_F^2 after matching head gain",
            "assumptions_to_state": "block model; exact head-gain matching; tail operators fixed for the comparison; squared-drift metric",
            "current_evidence": (
                f"positive boundary ratio={fmt(positive['geomean_tail_output_drift_sq_ratio_spectral_over_fro'])} "
                f"{ci(positive, 'tail_output_drift_sq_ratio_ci95_low', 'tail_output_drift_sq_ratio_ci95_high')}; "
                f"negative boundary ratio={fmt(negative['geomean_tail_output_drift_sq_ratio_spectral_over_fro'])} "
                f"{ci(negative, 'tail_output_drift_sq_ratio_ci95_low', 'tail_output_drift_sq_ratio_ci95_high')}; "
                "discussion/e11_matrix_block_tightness_audit.md checks exact diagonal witnesses, ratio identities, equality boundary, Frobenius-favored boundary, and degenerate-tail caveats"
            ),
            "current_status": "main_theorem_contract_and_tightness_audit_generated",
            "blocks_main_theory_claim": "yes",
            "required_upgrade": "keep discussion/e11_matrix_block_theorem_proof.md and discussion/e11_matrix_block_tightness_audit.md synchronized with the paper theorem and appendix proof",
            "forbidden_wording": "do not present the synthetic sign boundary as an out-of-sample natural predictor",
        },
        {
            "obligation_id": "PTO-3-muon-approximation-scope",
            "paper_claim": "Muon-style momentum/NS directions are local compatibility checks, not a full optimizer theory.",
            "formal_object": "finite-iteration Newton-Schulz approximation to polar momentum states sampled along short trajectories",
            "assumptions_to_state": "selected states; fixed checkpoint or short trajectory; no convergence claim; no tuned final-performance claim",
            "current_evidence": "reviewer-risk and quantitative ledgers keep Muon as motivation plus selected-state compatibility",
            "current_status": "scoped_support_only",
            "blocks_main_theory_claim": "no",
            "required_upgrade": "derive a separate finite-NS state-distribution theorem before making optimizer-level claims",
            "forbidden_wording": "do not claim complete Muon training behavior or benchmark superiority",
        },
        {
            "obligation_id": "PTO-4-v5-transport-score",
            "paper_claim": "A measurable transport-normalized condition score predicts residual layer risk on unseen real-task splits.",
            "formal_object": "source-standardized sandwiched-tail residual score with partition/architecture transport terms",
            "assumptions_to_state": "score frozen before final rows; spent final quarantine; residual, direction, baseline-dominance, and control-reporting gates",
            "current_evidence": (
                f"validation score={selected_score}; freeze status="
                f"{v5_freeze_status['v5 transport-normalized residual score']}; "
                f"architecture residual={v5_final_status['v5_final_heldout_architecture_residual_spearman']}; "
                f"architecture direction={v5_final_status['v5_final_heldout_architecture_direction_threshold_accuracy']}; "
                f"data residual={v5_final_status['v5_final_heldout_data_partition_residual_spearman']}; "
                f"data direction={v5_final_status['v5_final_heldout_data_partition_direction_threshold_accuracy']}; "
                f"final P0 gate={v5_final_status['v5_p0_predictive_condition_claim']}"
            ),
            "current_status": "completed_final_failed_boundary",
            "blocks_main_theory_claim": "yes_for_predictive_condition_claim",
            "required_upgrade": "open a new preregistered protocol with new unspent splits before any score repair or renewed predictive-condition attempt",
            "forbidden_wording": "do not claim held-out predictive-condition generality after the frozen final gates failed",
        },
        {
            "obligation_id": "PTO-5-natural-falsification",
            "paper_claim": "The empirical story includes natural positive and negative boundary cases, not only constructed examples.",
            "formal_object": "familywise Holm-adjusted natural negative search over 26 registered phase1 settings and 8 registered phase2 ResNet34 settings",
            "assumptions_to_state": "complete metric rows for every declared setting; paired per-seed log-ratio tests; quality, head-gain, and power caveats applied before claims",
            "current_evidence": (
                f"phase1 completeness={natural_status['NNS-E2-phase1-output-completeness']}; "
                f"primary metric rows={natural_observed_count}/26; "
                f"natural primary claim={natural_status['NNS-E4-natural-primary-claim']}; "
                f"phase2 observed rows={natural_phase2_observed_count}/8; "
                f"phase2 claim={natural_phase2_status['NNS-P2-E4-heldout-architecture-claim']}; "
                f"phase2 head_gain_gate_fail_rows={natural_phase2_head_gain_fail_count}"
            ),
            "current_status": (
                "finite_registered_phase1_phase2_null_candidate_with_caveats"
                if natural_complete and natural_finite_null and natural_phase2_finite_null
                else "partial_metric_outputs"
            ),
            "blocks_main_theory_claim": "no_but_blocks_falsification_upgrade",
            "required_upgrade": (
                "replicate or extend the registered natural search on larger datasets or additional held-out architectures before making broader natural-null claims"
                if natural_complete and natural_finite_null and natural_phase2_finite_null
                else "finish the remaining phase1 metric rows and rerun the multiplicity evaluator"
            ),
            "forbidden_wording": "do not call any setting a natural counterexample without an adjusted primary worse decision, and do not generalize the finite phase1/phase2 null outside its registered spaces",
        },
        {
            "obligation_id": "PTO-6-final-performance-separation",
            "paper_claim": "The mechanism is distinct from a competitive long-tail optimizer-performance claim.",
            "formal_object": "validation/final split protocol for tuned baselines, finite-NS-Muon candidates, and familywise final comparisons",
            "assumptions_to_state": "recipe selection on validation seeds only; final seeds quarantined; many/medium/few metrics reported even if negative",
            "current_evidence": (
                f"tuned validation grid={tuned_status['TVS-1-validation-grid-complete']}; "
                f"family selection={tuned_status['TVS-2-family-selection']}; "
                f"final seed quarantine={tuned_status['TVS-3-final-seed-quarantine']}"
            ),
            "current_status": "performance_claim_blocked",
            "blocks_main_theory_claim": "no_but_blocks_optimizer_benchmark_claim",
            "required_upgrade": "complete tuned validation selection and untouched final paired seeds before performance wording",
            "forbidden_wording": "do not imply that lower local drift already proves better final tail accuracy",
        },
    ]
    return pd.DataFrame(rows)


def build_assumption_stress_tests() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "assumption_id": "AST-1-local-step",
                "assumption": "first-order local response is accurate at the chosen head-gain scale",
                "stress_test": "local_linearization_errors.tex and results/e11_local_linearization/summary.csv",
                "failure_mode": "large residual terms would make the JVP theorem irrelevant to measured logits",
                "paper_action": "keep update-scale caveats and report residual errors",
                "status": "checked",
            },
            {
                "assumption_id": "AST-2-head-gain-matching",
                "assumption": "directions are compared at matched head objective gain",
                "stress_test": "validator enforces tiny relative update/head-gain gaps across committed probes",
                "failure_mode": "spectral could win by taking a smaller useful head step rather than by geometry",
                "paper_action": "state matched-head-gain protocol before every drift ratio",
                "status": "checked",
            },
            {
                "assumption_id": "AST-3-tail-quality",
                "assumption": "tail examples are meaningful enough that drift is interpretable",
                "stress_test": "tail-quality control, imbalance sweep, and standard many/medium/few reporting baseline",
                "failure_mode": "low pre-update tail accuracy would make drift-only evidence look artificial",
                "paper_action": "present tail-quality controls as context, not as accuracy improvement",
                "status": "checked_but_limited",
            },
            {
                "assumption_id": "AST-4-transport-stability",
                "assumption": "partition and architecture transport terms preserve residual layer-risk ordering",
                "stress_test": "v5 frozen final ResNeXt50-32x4d and CIFAR-10 cross-partition gates",
                "failure_mode": "score becomes a fixed-split diagnostic rather than a predictive condition",
                "paper_action": "report the completed v5 negative boundary and require a new protocol for any repair",
                "status": "completed_final_failed_boundary",
            },
            {
                "assumption_id": "AST-5-multiplicity-integrity",
                "assumption": "natural negative examples survive familywise adjustment",
                "stress_test": "phase1 multiplicity evaluator with 26 declared settings plus phase2 ResNet34 evaluator with 8 declared settings",
                "failure_mode": "selected anecdotal counterexamples would be statistically weak",
                "paper_action": "report the finite registered phase1/phase2 results with detectable-effect, head-gain, and quality-gate caveats",
                "status": "finite_null_candidate_with_caveats",
            },
        ]
    )


def build_claim_scope_boundaries() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_scope": "main_theorem",
                "allowed_claim": "a local matched-head-gain matrix-block mechanism can favor spectral/polar directions for tail drift",
                "blocked_claim": "global convergence or optimizer superiority",
                "decisive_gate": "PTO-1 and PTO-2",
                "current_status": "supportable_if_written_with_assumptions",
            },
            {
                "claim_scope": "natural_drift_diagnostic",
                "allowed_claim": "committed natural diagnostics support lower matched-head-gain tail-logit drift in tested settings",
                "blocked_claim": "tail accuracy necessarily improves",
                "decisive_gate": "quantitative claim ledger caveats",
                "current_status": "supportable_as_diagnostic",
            },
            {
                "claim_scope": "predictive_condition",
                "allowed_claim": "v5 has a frozen candidate whose completed final gates failed under the registered protocol",
                "blocked_claim": "the v5 score predicts unseen real-task residual risk",
                "decisive_gate": "both v5 final splits pass residual, direction, baseline, and reporting gates",
                "current_status": "completed_final_failed_boundary",
            },
            {
                "claim_scope": "natural_counterexample",
                "allowed_claim": "the registered 26-setting phase1 and 8-setting phase2 searches found no adjusted primary full-drift counterexample and are finite-null candidates with caveats",
                "blocked_claim": "a natural primary counterexample exists, or no natural counterexample exists outside the registered phase1/phase2 spaces",
                "decisive_gate": "Holm-adjusted positive decision for a counterexample, or complete family plus finite-null caveats for null wording",
                "current_status": "finite_null_candidate_with_caveats",
            },
            {
                "claim_scope": "optimizer_benchmark",
                "allowed_claim": "current final-training pilots are benchmark context and include negative Muon boundaries",
                "blocked_claim": "Muon or spectral training is competitive on long-tail benchmarks",
                "decisive_gate": "tuned validation selection and untouched final paired seeds",
                "current_status": "not_ready",
            },
        ]
    )


def build_theorem_to_experiment_queue() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "priority": "P0",
                "task": "keep the exact local matrix-block theorem/proof contract synchronized with the paper appendix",
                "artifact_or_command": "make e11-matrix-block-theorem-proof && make e11-matrix-block-tightness-audit",
                "unblocks": "main_theorem wording",
                "dependency": "none",
            },
            {
                "priority": "P0",
                "task": "preserve the completed frozen v5 final failures without score edits",
                "artifact_or_command": "make e11-cifar-resnet-condition-score-v5-final-eval",
                "unblocks": "completed negative predictive-condition boundary wording",
                "dependency": "completed v5 final outputs",
            },
            {
                "priority": "P0",
                "task": "preserve the completed v5 final power audit as negative-transport boundary evidence",
                "artifact_or_command": "make e11-cifar-resnet-condition-score-v5-final-power-audit",
                "unblocks": "completed-final negative-versus-underpowered wording for failed residual-Spearman gates",
                "dependency": "completed v5 final evaluator and final power audit",
            },
            {
                "priority": "P0",
                "task": "preserve the completed-final v5 reviewer failure response and claim downgrades",
                "artifact_or_command": "make e11-cifar-resnet-condition-score-v5-reviewer-failure-response",
                "unblocks": "no-repair negative boundary wording and new unspent-protocol requirement",
                "dependency": "completed v5 final evaluator and reviewer failure response",
            },
            {
                "priority": "P1",
                "task": "replicate or extend the finite phase1/phase2 natural negative-search result before broader natural-null wording",
                "artifact_or_command": "make e11-natural-negative-search-phase1-eval && make e11-natural-negative-search-phase2-eval",
                "unblocks": "broader natural-null or held-out natural boundary wording",
                "dependency": "complete phase1/phase2 finite-null candidates with caveats",
            },
            {
                "priority": "P1",
                "task": "complete tuned benchmark validation selection and final seeds",
                "artifact_or_command": "make e11-cifar-resnet-lt-tuned-benchmark-selection plus final benchmark jobs",
                "unblocks": "optimizer_benchmark wording",
                "dependency": "validation grid outputs",
            },
            {
                "priority": "P2",
                "task": "formalize post-final transport obligations under explicit endpoint-specific invariance assumptions",
                "artifact_or_command": "make e11-cifar-resnet-condition-score-v5-theory-to-score-map",
                "unblocks": "stronger theorem-to-score alignment",
                "dependency": "completed v5 final pass/fail localization",
            },
        ]
    )


def write_discussion(
    proof_obligations: pd.DataFrame,
    stress_tests: pd.DataFrame,
    scope_boundaries: pd.DataFrame,
    queue: pd.DataFrame,
) -> None:
    text = f"""# E11 Theory Proof-Obligation Register

This generated register is the theory-facing top-conference checklist for the
current E11 paper. It separates what the paper can state as a theorem, what is
only an empirical diagnostic, what is pending GPU evidence, and what remains a
future optimizer-performance or predictive-condition claim. It is deliberately
stricter than the result ledger: every broad claim must point to a formal object,
stated assumptions, decisive evidence, and forbidden wording.

## Proof Obligations

{markdown_table(proof_obligations, ["obligation_id", "paper_claim", "formal_object", "assumptions_to_state", "current_status", "required_upgrade", "forbidden_wording"])}

## Assumption Stress Tests

{markdown_table(stress_tests, ["assumption_id", "assumption", "stress_test", "failure_mode", "paper_action", "status"])}

## Claim Scope Boundaries

{markdown_table(scope_boundaries, ["claim_scope", "allowed_claim", "blocked_claim", "decisive_gate", "current_status"])}

## Theorem-To-Experiment Queue

{markdown_table(queue, ["priority", "task", "artifact_or_command", "unblocks", "dependency"])}

Artifacts:
- [proof_obligations.csv](../{(OUTPUT_DIR / 'proof_obligations.csv').as_posix()})
- [assumption_stress_tests.csv](../{(OUTPUT_DIR / 'assumption_stress_tests.csv').as_posix()})
- [claim_scope_boundaries.csv](../{(OUTPUT_DIR / 'claim_scope_boundaries.csv').as_posix()})
- [theorem_to_experiment_queue.csv](../{(OUTPUT_DIR / 'theorem_to_experiment_queue.csv').as_posix()})
- [config.json](../{(OUTPUT_DIR / 'config.json').as_posix()})
"""
    write_markdown(DISCUSSION_PATH, text)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    evidence = load_evidence()
    proof_obligations = build_proof_obligations(evidence)
    stress_tests = build_assumption_stress_tests()
    scope_boundaries = build_claim_scope_boundaries()
    queue = build_theorem_to_experiment_queue()

    proof_obligations.to_csv(OUTPUT_DIR / "proof_obligations.csv", index=False)
    stress_tests.to_csv(OUTPUT_DIR / "assumption_stress_tests.csv", index=False)
    scope_boundaries.to_csv(OUTPUT_DIR / "claim_scope_boundaries.csv", index=False)
    queue.to_csv(OUTPUT_DIR / "theorem_to_experiment_queue.csv", index=False)
    (OUTPUT_DIR / "config.json").write_text(
        json.dumps(
            {
                "purpose": "theory-facing proof-obligation and claim-scope register",
                "analysis_scope": "top-conference theorem checklist; no new empirical results",
                "primary_score": evidence["v5_response_config"]["primary_score"],
                "proof_obligation_count": int(len(proof_obligations)),
                "claim_scope_count": int(len(scope_boundaries)),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_discussion(proof_obligations, stress_tests, scope_boundaries, queue)
    print(f"saved {DISCUSSION_PATH} and {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
