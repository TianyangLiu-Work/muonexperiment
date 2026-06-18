from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.reporting import fmt, markdown_table, require_one, write_markdown


OUTPUT_DIR = Path("results/e11_muon_state_distribution_contract")
DISCUSSION_PATH = Path("discussion/e11_muon_state_distribution_contract.md")

BRIDGE_DIR = Path("results/e11_cifar100_resnet_practical_muon_bridge")
MUON_FINAL_DIR = Path("results/e11_cifar100_resnet_lt_muon_final_benchmark")
TUNED_DIR = Path("results/e11_cifar100_resnet_lt_tuned_benchmark")
BOLD_CONJECTURE_PATH = Path("results/e11_bold_conjecture_register/conjecture_register.csv")


def _bridge_readout(frame: pd.DataFrame, state_source: str) -> pd.Series:
    return require_one(frame, state_source=state_source, direction="ns_momentum")


def _best_muon_pair(frame: pd.DataFrame, frequency_group: str) -> pd.Series:
    muon = frame[
        frame["recipe"].astype(str).str.startswith("ns_muon")
        & frame["frequency_group"].eq(frequency_group)
    ].copy()
    if muon.empty:
        raise AssertionError(f"missing NS-Muon paired rows for {frequency_group}")
    return muon.sort_values("mean_balanced_accuracy_diff", ascending=False).iloc[0]


def _bridge_drift_text(row: pd.Series) -> str:
    return (
        f"{fmt(row['geomean_tail_output_drift_sq_ratio_vs_fro'])} "
        f"[{fmt(row['tail_output_drift_sq_ratio_vs_fro_ci95_low'])}, "
        f"{fmt(row['tail_output_drift_sq_ratio_vs_fro_ci95_high'])}]"
    )


def _paired_accuracy_diff_text(row: pd.Series) -> str:
    return (
        f"{fmt(row['mean_balanced_accuracy_diff'])} "
        f"[{fmt(row['balanced_accuracy_diff_ci95_low'])}, "
        f"{fmt(row['balanced_accuracy_diff_ci95_high'])}]"
    )


def load_inputs() -> dict[str, object]:
    return {
        "bridge_summary": pd.read_csv(BRIDGE_DIR / "summary.csv"),
        "bridge_step_summary": pd.read_csv(BRIDGE_DIR / "step_summary.csv"),
        "muon_pair_summary": pd.read_csv(MUON_FINAL_DIR / "pair_summary.csv"),
        "tuned_gate_report": pd.read_csv(TUNED_DIR / "validation_selection/gate_report.csv"),
        "tuned_execution_status": pd.read_csv(TUNED_DIR / "execution_status.csv"),
        "bold_conjectures": pd.read_csv(BOLD_CONJECTURE_PATH),
    }


def build_state_distribution_terms(inputs: dict[str, object]) -> pd.DataFrame:
    bridge = inputs["bridge_summary"]
    pairs = inputs["muon_pair_summary"]
    gates = inputs["tuned_gate_report"].set_index("gate_id")["status"].to_dict()
    execution = require_one(
        inputs["tuned_execution_status"],
        registry_id="cifar100lt_resnet18_tuned_validation_grid",
    )
    adamw_state = _bridge_readout(bridge, "adamw_matrix_trajectory")
    ns_state = _bridge_readout(bridge, "ns_muon_matrix_trajectory")
    best_few = _best_muon_pair(pairs, "few")
    best_all = _best_muon_pair(pairs, "all")
    settings = int(execution["settings"])

    return pd.DataFrame(
        [
            {
                "term_id": "MSD-T1-local-response-integrand",
                "theory_object": "ell(s,d): matched-head-gain tail-logit drift response at sampled optimizer state s and direction d",
                "measurable_proxy": "NS(M_t)-versus-Fro/GD squared tail-output drift ratio in the practical Muon trajectory bridge",
                "current_evidence": (
                    "adamw_state="
                    f"{_bridge_drift_text(adamw_state)}; "
                    "ns_muon_state="
                    f"{_bridge_drift_text(ns_state)}"
                ),
                "careful_status": "local_integrand_supported_on_sampled_states",
                "missing_for_upgrade": "state occupancy weights and terminal-risk coupling are not proved by local ratios",
            },
            {
                "term_id": "MSD-T2-state-occupancy-measure",
                "theory_object": "nu_R(s): recipe-dependent distribution over checkpoints, tail quality, gradients, momentum alignment, and class exposure",
                "measurable_proxy": "per-recipe trajectory-state occupancy table with tail quality, head gain, gradient-momentum cosine, and local drift probes",
                "current_evidence": f"sampled warmup/tail-rich bridge states exist; tuned runner registers occupancy_trace.csv paths; TVS-5={gates['TVS-5-occupancy-logging-complete']}; final-training occupancy for selected recipes is absent",
                "careful_status": "occupancy_measure_missing_for_final_training",
                "missing_for_upgrade": "complete occupancy_trace.csv summaries during tuned validation and final seeds",
            },
            {
                "term_id": "MSD-T3-transition-and-schedule-operator",
                "theory_object": "T_R(s_{t+1}|s_t): schedule-, step-size-, weight-decay-, and Newton-Schulz-depth-induced state transition",
                "measurable_proxy": "registered tuned validation grid over learning rate, warmup, weight decay, class weighting, sampler, and Newton-Schulz depth",
                "current_evidence": f"{settings} validation settings registered; TVS-1={gates['TVS-1-validation-grid-complete']}",
                "careful_status": "schedule_transport_registered_not_evaluated",
                "missing_for_upgrade": "complete validation summaries before selecting final recipes",
            },
            {
                "term_id": "MSD-T4-terminal-risk-functional",
                "theory_object": "M_R: final many/medium/few balanced accuracy, loss, and margin under recipe R",
                "measurable_proxy": "paired final-training pilot differences versus AdamW augmentation",
                "current_evidence": (
                    f"best_tested_few_diff={_paired_accuracy_diff_text(best_few)}; "
                    f"best_tested_all_diff={_paired_accuracy_diff_text(best_all)}"
                ),
                "careful_status": "final_performance_negative_boundary",
                "missing_for_upgrade": "tuned validation selection and untouched final paired seeds",
            },
            {
                "term_id": "MSD-T5-claim-composition-rule",
                "theory_object": "A practical optimizer claim requires local response, state occupancy, transition stability, and terminal-risk evidence to pass together",
                "measurable_proxy": "joint gate over bridge ratios, occupancy logging, tuned selection, final paired metrics, and no-overclaim wording",
                "current_evidence": "local response positive, final pilot negative, tuned validation/final protocol incomplete",
                "careful_status": "mechanism_only_until_all_components_pass",
                "missing_for_upgrade": "all state-distribution and final-performance gates must pass under a new unspent protocol",
            },
        ]
    )


def build_evidence_links(inputs: dict[str, object]) -> pd.DataFrame:
    bridge = inputs["bridge_summary"]
    pairs = inputs["muon_pair_summary"]
    gates = inputs["tuned_gate_report"].set_index("gate_id")["evidence"].to_dict()
    conjectures = inputs["bold_conjectures"].set_index("conjecture_id")
    adamw_state = _bridge_readout(bridge, "adamw_matrix_trajectory")
    ns_state = _bridge_readout(bridge, "ns_muon_matrix_trajectory")
    best_few = _best_muon_pair(pairs, "few")
    bc4_status = str(conjectures.loc["BC-4-muon-performance-needs-state-distribution-theory", "careful_status"])

    return pd.DataFrame(
        [
            {
                "evidence_id": "MSE-1-local-adamw-state",
                "source_artifact": "results/e11_cifar100_resnet_practical_muon_bridge/summary.csv",
                "observed_readout": (
                    "AdamW-sampled NS(M_t) drift ratio "
                    f"{_bridge_drift_text(adamw_state)}"
                ),
                "interpretation": "local tail-drift integrand is favorable on AdamW-sampled states",
                "claim_boundary": "not a final optimizer-performance result",
            },
            {
                "evidence_id": "MSE-2-local-ns-muon-state",
                "source_artifact": "results/e11_cifar100_resnet_practical_muon_bridge/summary.csv",
                "observed_readout": (
                    "NS-Muon-sampled NS(M_t) drift ratio "
                    f"{_bridge_drift_text(ns_state)}"
                ),
                "interpretation": "local tail-drift integrand is favorable on NS-Muon-sampled states",
                "claim_boundary": "sampled state support only, not a trajectory occupancy theorem",
            },
            {
                "evidence_id": "MSE-3-final-pilot-negative",
                "source_artifact": "results/e11_cifar100_resnet_lt_muon_final_benchmark/pair_summary.csv",
                "observed_readout": (
                    f"best tested NS-Muon few balanced-accuracy diff={_paired_accuracy_diff_text(best_few)}"
                ),
                "interpretation": "local compatibility can coexist with poor terminal few-class accuracy",
                "claim_boundary": "blocks practical optimizer-performance wording",
            },
            {
                "evidence_id": "MSE-4-tuned-grid-registered",
                "source_artifact": "results/e11_cifar100_resnet_lt_tuned_benchmark/validation_selection/gate_report.csv",
                "observed_readout": (
                    f"TVS-1 evidence: {gates['TVS-1-validation-grid-complete']}; "
                    f"TVS-5 evidence: {gates['TVS-5-occupancy-logging-complete']}"
                ),
                "interpretation": "unspent tuned validation/final protocol and occupancy logging paths exist but have not produced selection evidence",
                "claim_boundary": "do not select from spent pilot rows",
            },
            {
                "evidence_id": "MSE-5-bold-conjecture-boundary",
                "source_artifact": "results/e11_bold_conjecture_register/conjecture_register.csv",
                "observed_readout": f"BC-4 careful_status={bc4_status}",
                "interpretation": "performance gains need state-distribution and schedule theory",
                "claim_boundary": "use as conjecture and protocol driver, not as a positive claim",
            },
        ]
    )


def build_falsification_tests() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "test_id": "MSF-1-occupancy-logging",
                "hypothesis": "Recipes with favorable terminal few-class accuracy occupy states where local NS(M_t) drift ratios stay below Fro/GD under matched head gain.",
                "required_protocol": "log occupancy_trace.csv rows with trajectory-state tail quality, head gain, gradient-momentum cosine, and local drift probes during tuned validation and final seeds",
                "pass_to_upgrade": "occupancy-weighted local drift predicts selected final recipe ordering without using final labels for tuning",
                "fail_response": "local mechanism remains true only at sampled states; no practical Muon claim",
                "compute_mode": "GPU via Slurm plus CPU aggregation",
            },
            {
                "test_id": "MSF-2-schedule-transport",
                "hypothesis": "The practical failure is caused by schedule/state transport, not by disappearance of the local drift integrand.",
                "required_protocol": "compare learning-rate, warmup, weight-decay, and Newton-Schulz-depth strata before final unblinding",
                "pass_to_upgrade": "a validation-selected stratum preserves local drift and avoids terminal collapse on final seeds",
                "fail_response": "state-transition theory is incomplete; keep performance wording blocked",
                "compute_mode": "GPU via Slurm",
            },
            {
                "test_id": "MSF-3-terminal-risk-separation",
                "hypothesis": "Tail-logit drift reduction is a necessary local safety term but not sufficient for final few-class balanced accuracy.",
                "required_protocol": "jointly report drift probes, many/medium/few accuracy, loss, margin, and paired differences for every selected final recipe",
                "pass_to_upgrade": "local drift improvement and terminal few-class improvement co-occur under the same selected recipe",
                "fail_response": "publish the negative separation as a boundary result",
                "compute_mode": "GPU via Slurm",
            },
            {
                "test_id": "MSF-4-baseline-dominance",
                "hypothesis": "Any practical Muon claim survives tuned AdamW, SGD, class-balanced loss, and sampler baselines.",
                "required_protocol": "complete validation-only selection and 10 paired final seeds for selected baseline and Muon families",
                "pass_to_upgrade": "Holm-adjusted few-class gain over tuned baselines without all-class collapse",
                "fail_response": "mechanism-only paper path remains",
                "compute_mode": "GPU via Slurm",
            },
            {
                "test_id": "MSF-5-state-distribution-counterexample",
                "hypothesis": "A recipe can have favorable sampled local drift yet poor final performance if it visits low-tail-quality or unstable transition states.",
                "required_protocol": "search for recipes with local drift ratio below one but negative terminal paired differences and inspect occupancy covariates",
                "pass_to_upgrade": "counterexample explains why local theorem does not imply optimizer superiority",
                "fail_response": "register a sharper theorem tying local response to terminal metric before broadening claims",
                "compute_mode": "CPU aggregation after GPU outputs",
            },
        ]
    )


def build_claim_gates(inputs: dict[str, object]) -> pd.DataFrame:
    gates = inputs["tuned_gate_report"].set_index("gate_id")["status"].to_dict()
    return pd.DataFrame(
        [
            {
                "gate_id": "MSG-1-local-compatibility",
                "current_status": "pass",
                "allowed_claim": "Muon-style finite Newton-Schulz momentum can reduce local matched-head-gain tail-logit drift at sampled ResNet trajectory states.",
                "blocked_claim": "This local compatibility proves final optimizer superiority.",
                "evidence_required": "existing practical bridge ratios with CI upper endpoints below one for sampled states",
            },
            {
                "gate_id": "MSG-2-state-distribution-transport",
                "current_status": "not_ready",
                "allowed_claim": "state-distribution transport is the registered missing term between local geometry and final performance",
                "blocked_claim": "the sampled bridge states represent the full training trajectory distribution",
                "evidence_required": "trajectory occupancy_trace.csv logging across validation-selected and final recipes",
            },
            {
                "gate_id": "MSG-3-tuned-final-performance",
                "current_status": gates["TVS-1-validation-grid-complete"],
                "allowed_claim": "negative pilot boundary and unspent tuned benchmark protocol",
                "blocked_claim": "competitive CIFAR-100-LT ResNet18 optimizer-performance claim",
                "evidence_required": "complete validation grid, selected recipes, and untouched 10-seed final comparisons",
            },
            {
                "gate_id": "MSG-4-top-tier-practical-claim",
                "current_status": "blocked_until_state_distribution_and_final_gates_pass",
                "allowed_claim": "mechanism paper with explicit practical-performance boundary",
                "blocked_claim": "broad long-tail Muon optimizer claim",
                "evidence_required": "MSG-1, MSG-2, MSG-3 plus no-overclaim manuscript audit all pass",
            },
        ]
    )


def write_outputs(
    terms: pd.DataFrame,
    evidence: pd.DataFrame,
    falsifiers: pd.DataFrame,
    claim_gates: pd.DataFrame,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    terms.to_csv(OUTPUT_DIR / "state_distribution_terms.csv", index=False)
    evidence.to_csv(OUTPUT_DIR / "evidence_link_matrix.csv", index=False)
    falsifiers.to_csv(OUTPUT_DIR / "falsification_tests.csv", index=False)
    claim_gates.to_csv(OUTPUT_DIR / "claim_gate_ladder.csv", index=False)
    config = {
        "term_rows": int(len(terms)),
        "evidence_rows": int(len(evidence)),
        "falsification_rows": int(len(falsifiers)),
        "claim_gate_rows": int(len(claim_gates)),
        "current_claim_boundary": "local Muon-style drift compatibility only",
        "performance_upgrade_boundary": "state-distribution plus tuned validation/final evidence required",
    }
    (OUTPUT_DIR / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    text = f"""# E11 Muon State-Distribution Contract

This generated contract formalizes the practical Muon gap: local matched-head-gain drift compatibility is an integrand, not a final optimizer theorem. A top-tier practical claim needs the state-distribution transport contract below, because the current evidence contains both positive sampled-state local drift and a negative final-training pilot.

## State-Distribution Terms

{markdown_table(terms, ["term_id", "theory_object", "measurable_proxy", "current_evidence", "careful_status", "missing_for_upgrade"])}

## Evidence Links

{markdown_table(evidence, ["evidence_id", "source_artifact", "observed_readout", "interpretation", "claim_boundary"])}

## Falsification Tests

{markdown_table(falsifiers, ["test_id", "hypothesis", "required_protocol", "pass_to_upgrade", "fail_response", "compute_mode"])}

## Claim Gate Ladder

{markdown_table(claim_gates, ["gate_id", "current_status", "allowed_claim", "blocked_claim", "evidence_required"])}

## Operating Rule

Allowed now: cite local Muon-style drift compatibility at sampled ResNet trajectory states and cite the finite-NS-Muon final-training pilot as a negative boundary.

Blocked now: any practical optimizer-performance claim until state-distribution occupancy logging, tuned validation selection, and untouched final paired seeds all pass. Local compatibility can coexist with poor final performance.

Artifacts:
- [state_distribution_terms.csv](../results/e11_muon_state_distribution_contract/state_distribution_terms.csv)
- [evidence_link_matrix.csv](../results/e11_muon_state_distribution_contract/evidence_link_matrix.csv)
- [falsification_tests.csv](../results/e11_muon_state_distribution_contract/falsification_tests.csv)
- [claim_gate_ladder.csv](../results/e11_muon_state_distribution_contract/claim_gate_ladder.csv)
- [config.json](../results/e11_muon_state_distribution_contract/config.json)
"""
    write_markdown(DISCUSSION_PATH, text)


def main() -> None:
    inputs = load_inputs()
    terms = build_state_distribution_terms(inputs)
    evidence = build_evidence_links(inputs)
    falsifiers = build_falsification_tests()
    claim_gates = build_claim_gates(inputs)
    write_outputs(terms, evidence, falsifiers, claim_gates)
    print(f"saved Muon state-distribution contract to {DISCUSSION_PATH} and {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
