from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.reporting import markdown_table, write_markdown


OUTPUT_DIR = Path("results/e11_bold_conjecture_register")
DISCUSSION_PATH = Path("discussion/e11_bold_conjecture_register.md")


def lookup(frame: pd.DataFrame, key: str, value: str = "status") -> dict[str, str]:
    return {str(row[key]): str(row[value]) for _, row in frame.iterrows()}


def load_inputs() -> dict[str, object]:
    return {
        "proof": pd.read_csv("results/e11_theory_proof_obligation_register/proof_obligations.csv"),
        "heldout": pd.read_csv("results/e11_heldout_generality_audit/generality_claim_gate.csv"),
        "pfo": pd.read_csv("results/e11_condition_score_v5_theory_to_score_map/post_final_transport_obligations.csv"),
        "v5_final": pd.read_csv("results/e11_condition_score_v5_protocol/final_score_evaluation/final_gate_report.csv"),
        "tuned": pd.read_csv("results/e11_cifar100_resnet_lt_tuned_benchmark/validation_selection/gate_report.csv"),
        "falsifiers": pd.read_csv("results/e11_mechanism_referee_audit/falsification_trigger_matrix.csv"),
    }


def build_conjectures(inputs: dict[str, object]) -> pd.DataFrame:
    proof = inputs["proof"]
    heldout_status = lookup(inputs["heldout"], "gate_id")
    v5_status = lookup(inputs["v5_final"], "gate_id")
    tuned_status = lookup(inputs["tuned"], "gate_id")
    proof_status = lookup(proof, "obligation_id", "current_status")

    return pd.DataFrame(
        [
            {
                "conjecture_id": "BC-1-local-sandwich-drift",
                "bold_conjecture": "Matched-head-gain spectral/polar geometry lowers tail-logit drift when head-gradient nuclear rank exceeds downstream-aware tail sensitivity rank.",
                "current_evidence_state": proof_status["PTO-2-matrix-block-boundary"],
                "supporting_artifacts": "discussion/e11_matrix_block_theorem_proof.md; discussion/e11_matrix_block_tightness_audit.md; discussion/e11_heldout_generality_audit.md",
                "careful_status": "supportable_as_local_mechanism",
                "falsifier": "a pre-registered matched-head-gain family with valid linearization and head-gain gates shows adjusted primary spectral-worse drift under the favorable rank/sensitivity regime",
                "next_unspent_test": "new held-out architecture/data matched-head-gain diagnostic with fixed drift CI, tail-quality, and head-gain gates",
                "forbidden_shortcut": "treating the synthetic sign theorem as a global optimizer or final-accuracy theorem",
            },
            {
                "conjecture_id": "BC-2-endpoint-factorized-transport",
                "bold_conjecture": "Residual-risk ranking and below-one direction classification require separate architecture/data transport conditions; a single scalar score is under-specified.",
                "current_evidence_state": "post_final_supported_failure_model",
                "supporting_artifacts": "discussion/e11_condition_score_v5_theory_to_score_map.md; results/e11_condition_score_v5_theory_to_score_map/post_final_transport_obligations.csv",
                "careful_status": "conjecture_for_new_protocol_not_current_positive_claim",
                "falsifier": "a new unspent protocol shows one frozen scalar passes residual ranking, direction threshold, baseline dominance, and controls on both architecture and data final splits",
                "next_unspent_test": "register endpoint-specific architecture-direction and data-residual transport terms before any new final rows exist",
                "forbidden_shortcut": "repairing, thresholding, or reweighting the score on spent v2/v3/v4/v5 final rows",
            },
            {
                "conjecture_id": "BC-3-natural-counterexamples-are-structured",
                "bold_conjecture": "Natural primary full-drift counterexamples, if they exist, are structured and detectable by familywise registered searches rather than anecdotal cherry-picking.",
                "current_evidence_state": proof_status["PTO-5-natural-falsification"],
                "supporting_artifacts": "discussion/e11_natural_negative_search_phase1_evaluation.md; discussion/e11_natural_negative_search_phase2_evaluation.md; discussion/e11_heldout_generality_audit.md",
                "careful_status": heldout_status["HGG-5-paper-generality-claim"],
                "falsifier": "a registered family produces a Holm-adjusted primary worse row with quality, head-gain, and detectable-effect gates satisfied",
                "next_unspent_test": "larger dataset or additional architecture family with the same primary drift, tail-loss, margin, rank, JVP, and quality reporting contract",
                "forbidden_shortcut": "claiming universal absence of natural counterexamples outside the registered phase1/phase2 families",
            },
            {
                "conjecture_id": "BC-4-muon-performance-needs-state-distribution-theory",
                "bold_conjecture": "Local Muon-style drift compatibility can coexist with poor final performance; optimizer gains require a state-distribution and schedule theory, not just polar geometry.",
                "current_evidence_state": proof_status["PTO-6-final-performance-separation"],
                "supporting_artifacts": "discussion/e11_cifar100_resnet_practical_muon_bridge.md; discussion/e11_cifar100_resnet_lt_muon_final_benchmark.md; discussion/e11_cifar100_resnet_lt_tuned_benchmark_protocol.md",
                "careful_status": "benchmark_claim_blocked_until_tuned_validation_and_final_seeds",
                "falsifier": "tuned validation selection plus untouched final seeds show no trajectory-local drift advantage and no performance boundary separating Muon states from AdamW/SGD states",
                "next_unspent_test": "complete tuned validation selection, then final paired many/medium/few metrics with trajectory-local drift probes",
                "forbidden_shortcut": "turning lower local drift into a competitive optimizer-performance claim",
            },
            {
                "conjecture_id": "BC-5-layer-risk-is-predictable-but-the-current-proxy-is-incomplete",
                "bold_conjecture": "Held-out layer-risk ordering is predictable in real networks, but the current theorem proxy misses transport and endpoint factors needed for unseen splits.",
                "current_evidence_state": (
                    f"v5_p0={v5_status['v5_p0_predictive_condition_claim']}; "
                    f"architecture_direction={v5_status['v5_final_heldout_architecture_direction_threshold_accuracy']}; "
                    f"data_residual={v5_status['v5_final_heldout_data_partition_residual_spearman']}"
                ),
                "supporting_artifacts": "discussion/e11_cifar100_resnet_layer_jvp_checkpoint_prediction.md; discussion/e11_condition_score_v5_final_evaluation.md; discussion/e11_condition_score_v5_direction_guardrail_failure_audit.md",
                "careful_status": "predictive_condition_failed_but_boundary_is_informative",
                "falsifier": "positive controls stop transferring under the same checkpoint-transfer benchmark, or a transport-complete score fails both endpoint-specific validation gates before final evaluation",
                "next_unspent_test": "new transport-complete score with separate residual and direction endpoints plus fresh validation/final split quarantine",
                "forbidden_shortcut": "claiming the failed v5 score predicts unseen real-task residual risk",
            },
        ]
    )


def build_stress_tests(inputs: dict[str, object]) -> pd.DataFrame:
    falsifier = inputs["falsifiers"].set_index("trigger_id")
    pfo_ids = set(inputs["pfo"]["obligation_id"])
    required_pfos = {
        "PFO-1-endpoint-factorization",
        "PFO-2-architecture-direction-transport",
        "PFO-3-data-partition-residual-transport",
        "PFO-4-post-final-quarantine",
        "PFO-5-negative-boundary-ledger",
    }
    if pfo_ids != required_pfos:
        raise AssertionError(f"unexpected post-final obligation ids: {sorted(pfo_ids)}")
    return pd.DataFrame(
        [
            {
                "stress_id": "BST-1-local-theorem-scope",
                "conjecture_id": "BC-1-local-sandwich-drift",
                "stress_test": "local linearization, matched-head-gain validator checks, synthetic positive/negative sign boundary, and ResNet18 held-out diagnostics",
                "active_falsification_trigger": "FT-3-unit-jvp-misread",
                "trigger_status": str(falsifier.loc["FT-3-unit-jvp-misread", "current_status"]),
                "claim_response": "keep the claim local and matched-head-gain; report unit-JVP caveat",
            },
            {
                "stress_id": "BST-2-score-transport-scope",
                "conjecture_id": "BC-2-endpoint-factorized-transport",
                "stress_test": "post-final transport obligations PFO-1 through PFO-5 and v5 final gate family",
                "active_falsification_trigger": "FT-1-v5-final-fails",
                "trigger_status": str(falsifier.loc["FT-1-v5-final-fails", "current_status"]),
                "claim_response": "downgrade v5 to a completed negative boundary and require new unspent splits for repair",
            },
            {
                "stress_id": "BST-3-natural-family-scope",
                "conjecture_id": "BC-3-natural-counterexamples-are-structured",
                "stress_test": "phase1 26-setting and phase2 8-setting Holm families plus detectable-effect and head-gain caveats",
                "active_falsification_trigger": "FT-2-natural-family-incomplete",
                "trigger_status": str(falsifier.loc["FT-2-natural-family-incomplete", "current_status"]),
                "claim_response": "finite phase1/phase2 null candidates only; no universal natural-null wording",
            },
            {
                "stress_id": "BST-4-optimizer-performance-scope",
                "conjecture_id": "BC-4-muon-performance-needs-state-distribution-theory",
                "stress_test": "negative finite-NS-Muon pilot plus tuned benchmark validation/final quarantine",
                "active_falsification_trigger": "FT-4-performance-overread",
                "trigger_status": str(falsifier.loc["FT-4-performance-overread", "current_status"]),
                "claim_response": "keep Muon as local trajectory compatibility until tuned final seeds finish",
            },
            {
                "stress_id": "BST-5-heldout-predictor-scope",
                "conjecture_id": "BC-5-layer-risk-is-predictable-but-the-current-proxy-is-incomplete",
                "stress_test": "checkpoint-transfer positive controls, residualized risk controls, and failed v5 final score gates",
                "active_falsification_trigger": "FT-1-v5-final-fails",
                "trigger_status": str(falsifier.loc["FT-1-v5-final-fails", "current_status"]),
                "claim_response": "use the failure as theory-to-score guidance, not as a predictor claim",
            },
        ]
    )


def build_upgrade_ladder() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ladder_id": "BCL-1-current-submission",
                "allowed_claim": "local matched-head-gain mechanism with ResNet18-supported diagnostics and explicit held-out boundaries",
                "required_gate_for_upgrade": "none for the scoped mechanism paper",
                "blocked_upgrade": "broad architecture/data generality or optimizer-performance wording",
            },
            {
                "ladder_id": "BCL-2-predictive-condition-upgrade",
                "allowed_claim": "completed v5 negative boundary and endpoint-factorized transport conjecture",
                "required_gate_for_upgrade": "new preregistered validation/final splits where residual and direction endpoints both pass",
                "blocked_upgrade": "reuse of spent v2/v3/v4/v5 final rows for score repair",
            },
            {
                "ladder_id": "BCL-3-natural-boundary-upgrade",
                "allowed_claim": "finite registered phase1/phase2 null candidates with detectable-effect, head-gain, and quality caveats",
                "required_gate_for_upgrade": "larger or additional registered natural family with complete quality/head-gain gates",
                "blocked_upgrade": "universal natural null or anecdotal counterexample wording",
            },
            {
                "ladder_id": "BCL-4-optimizer-performance-upgrade",
                "allowed_claim": "negative pilot plus local Muon-style trajectory compatibility",
                "required_gate_for_upgrade": "tuned validation selection and untouched final paired seeds with many/medium/few metrics",
                "blocked_upgrade": "competitive Muon benchmark claim before final seed evidence",
            },
        ]
    )


def write_discussion(
    conjectures: pd.DataFrame,
    stress_tests: pd.DataFrame,
    ladder: pd.DataFrame,
) -> None:
    text = f"""# E11 Bold Conjecture Register

This generated register states the paper's aggressive theory bets while binding
each one to current evidence, falsifiers, and forbidden shortcuts. It is the
project's "bold conjecture, careful verification" ledger: conjectures can guide
the manuscript, but only rows whose gates pass can become positive claims.

## Bold Conjectures

{markdown_table(conjectures, ["conjecture_id", "bold_conjecture", "current_evidence_state", "careful_status", "falsifier", "next_unspent_test", "forbidden_shortcut"])}

## Stress Tests

{markdown_table(stress_tests, ["stress_id", "conjecture_id", "stress_test", "active_falsification_trigger", "trigger_status", "claim_response"])}

## Claim Upgrade Ladder

{markdown_table(ladder, ["ladder_id", "allowed_claim", "required_gate_for_upgrade", "blocked_upgrade"])}

## Operating Rule

Allowed now: use these conjectures to frame the theory agenda and reviewer
response. Current positive wording remains limited to the local matched-head-gain mechanism and diagnostic ResNet18 support. Score upgrades are blocked by this boundary: new unspent validation/final protocol required.

Blocked now: presenting endpoint-factorized transport, natural finite-null
families, or Muon-style local compatibility as stronger positive claims before
their named unspent gates pass.

Artifacts:
- [conjecture_register.csv](../results/e11_bold_conjecture_register/conjecture_register.csv)
- [stress_test_matrix.csv](../results/e11_bold_conjecture_register/stress_test_matrix.csv)
- [claim_upgrade_ladder.csv](../results/e11_bold_conjecture_register/claim_upgrade_ladder.csv)
- [config.json](../results/e11_bold_conjecture_register/config.json)
"""
    write_markdown(DISCUSSION_PATH, text)


def main() -> None:
    inputs = load_inputs()
    conjectures = build_conjectures(inputs)
    stress_tests = build_stress_tests(inputs)
    ladder = build_upgrade_ladder()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    conjectures.to_csv(OUTPUT_DIR / "conjecture_register.csv", index=False)
    stress_tests.to_csv(OUTPUT_DIR / "stress_test_matrix.csv", index=False)
    ladder.to_csv(OUTPUT_DIR / "claim_upgrade_ladder.csv", index=False)
    config = {
        "conjecture_rows": int(len(conjectures)),
        "stress_test_rows": int(len(stress_tests)),
        "upgrade_ladder_rows": int(len(ladder)),
        "positive_claim_boundary": "local matched-head-gain mechanism only",
        "score_upgrade_boundary": "new unspent validation/final protocol required",
    }
    (OUTPUT_DIR / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    write_discussion(conjectures, stress_tests, ladder)
    print(f"saved bold conjecture register to {DISCUSSION_PATH} and {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
