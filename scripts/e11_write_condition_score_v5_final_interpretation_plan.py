from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.reporting import markdown_table, write_markdown


OUTPUT_DIR = Path("results/e11_condition_score_v5_protocol/final_interpretation_plan")
DISCUSSION_PATH = Path("discussion/e11_condition_score_v5_final_interpretation_plan.md")
FREEZE_DIR = Path("results/e11_condition_score_v5_protocol/validation_score_freeze")
FINAL_EVAL_DIR = Path("results/e11_condition_score_v5_protocol/final_score_evaluation")


def load_final_splits() -> list[dict[str, object]]:
    config = json.loads((FINAL_EVAL_DIR / "config.json").read_text(encoding="utf-8"))
    return list(config["final_splits"])


def load_final_gates() -> pd.DataFrame:
    path = FINAL_EVAL_DIR / "final_gate_report.csv"
    if not path.exists():
        return pd.DataFrame(columns=["gate_id", "scope", "status", "evidence"])
    return pd.read_csv(path)


def selected_score() -> str:
    freeze = pd.read_csv(FREEZE_DIR / "freeze_status.csv").set_index("item")
    status = str(freeze.loc["v5 transport-normalized residual score", "status"])
    score = str(freeze.loc["v5 transport-normalized residual score", "evidence"])
    if status != "frozen":
        raise ValueError(f"v5 final interpretation plan requires a frozen validation score, got {status}")
    return score


def build_final_split_status(splits: list[dict[str, object]]) -> pd.DataFrame:
    rows = []
    for split in splits:
        layer_path = Path(str(split["layer_summary_path"]))
        metrics_path = Path(str(split["metrics_path"]))
        generated = layer_path.exists() and metrics_path.exists()
        rows.append(
            {
                "split_id": split["split_id"],
                "split_role": split["role"],
                "display_name": split["display_name"],
                "layer_summary_path": layer_path.as_posix(),
                "metrics_path": metrics_path.as_posix(),
                "current_output_status": "generated" if generated else "not_run",
                "claim_gate_group": "required_for_p0",
                "can_be_replaced": "no",
                "can_be_dropped_after_result": "no",
            }
        )
    return pd.DataFrame(rows)


def build_gate_contract(score: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "gate_id": "V5-FINAL-G1-frozen-primary-score",
                "required_for": "all final interpretations",
                "pass_rule": f"primary score remains {score}",
                "failure_effect": "all final interpretations are invalid until the committed evaluator is restored",
            },
            {
                "gate_id": "V5-FINAL-G2-output-completeness",
                "required_for": "P0 predictive-condition claim",
                "pass_rule": "both registered final splits have layer_summary.csv and metrics.csv",
                "failure_effect": "P0 remains not_ready; no split may be substituted",
            },
            {
                "gate_id": "V5-FINAL-G3-residual-ranking",
                "required_for": "P0 predictive-condition claim",
                "pass_rule": "primary residual Spearman CI lower endpoint is above zero on each final split",
                "failure_effect": "the failed split becomes a transport-boundary result, not a tuning target",
            },
            {
                "gate_id": "V5-FINAL-G4-direction-guardrail",
                "required_for": "P0 predictive-condition claim",
                "pass_rule": "direction-axis below-one threshold accuracy and CI lower endpoint are both at least 0.8",
                "failure_effect": "the local spectral/Frobenius direction guardrail failed to transport",
            },
            {
                "gate_id": "V5-FINAL-G5-baseline-dominance",
                "required_for": "P0 predictive-condition claim",
                "pass_rule": "primary score beats early_layer_prior on residual Spearman and top-k overlap",
                "failure_effect": "the score is treated as a depth/stage nuisance proxy",
            },
            {
                "gate_id": "V5-FINAL-G6-control-reporting",
                "required_for": "paper reporting",
                "pass_rule": "direction, early-layer, and source-observed controls are reported for every generated final split",
                "failure_effect": "the final evaluator output is incomplete for review",
            },
        ]
    )


def build_outcome_ladder() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "outcome_pattern": "both final splits missing or incomplete (pre-output branch)",
                "claim_state": "not_ready",
                "allowed_interpretation": "registered final evaluation is pending only while outputs are missing",
                "forbidden_interpretation": "any predictive-condition or generality claim",
                "required_paper_action": "before outputs exist, report pending Slurm/output status; after outputs exist, use current_interpretation_summary.csv",
            },
            {
                "outcome_pattern": "both final splits pass all P0 gates",
                "claim_state": "p0_claim_eligible",
                "allowed_interpretation": "the frozen transport-normalized score predicts residual layer risk on the registered architecture and data finals",
                "forbidden_interpretation": "claiming final optimizer performance or broader dataset/architecture coverage",
                "required_paper_action": "report both split summaries, controls, confidence intervals, and the no-retuning boundary",
            },
            {
                "outcome_pattern": "architecture split passes, data split fails residual ranking",
                "claim_state": "data_transport_boundary",
                "allowed_interpretation": "architecture transfer survived, but data-partition transport remains unresolved",
                "forbidden_interpretation": "broad data-family predictive-condition claim",
                "required_paper_action": "preserve the negative CIFAR-10 cross-partition result and analyze partition transport terms",
            },
            {
                "outcome_pattern": "data split passes, architecture split fails residual ranking",
                "claim_state": "architecture_transport_boundary",
                "allowed_interpretation": "data-family transfer survived, but ResNeXt architecture transport remains unresolved",
                "forbidden_interpretation": "broad architecture-family predictive-condition claim",
                "required_paper_action": "preserve the negative ResNeXt50-32x4d result and analyze parameterization transport terms",
            },
            {
                "outcome_pattern": "either final split fails direction guardrail",
                "claim_state": "direction_guardrail_failure",
                "allowed_interpretation": "the local spectral/Frobenius direction comparison failed to transport to the final split",
                "forbidden_interpretation": "residual-risk predictor claim, even if residual ranking is positive elsewhere",
                "required_paper_action": "separate direction failure from scalar residual-score failure",
            },
            {
                "outcome_pattern": "either final split fails baseline dominance",
                "claim_state": "nuisance_proxy_boundary",
                "allowed_interpretation": "the frozen score did not add enough information beyond the early-layer nuisance baseline",
                "forbidden_interpretation": "theory-derived measurable score claim",
                "required_paper_action": "report early_layer_prior comparison as a failed ablation gate",
            },
            {
                "outcome_pattern": "both final splits fail any P0 gate",
                "claim_state": "local_mechanism_only",
                "allowed_interpretation": "the paper retains local drift-mechanism and falsification evidence only",
                "forbidden_interpretation": "predictive-condition claim on unseen real tasks",
                "required_paper_action": "keep failed finals in the main evidence ledger and do not retune on them",
            },
        ]
    )


def build_current_interpretation(status: pd.DataFrame, final_gates: pd.DataFrame) -> pd.DataFrame:
    outputs_generated = bool(status["current_output_status"].eq("generated").all())
    generated_count = int(status["current_output_status"].eq("generated").sum())
    total_count = int(len(status))
    if not outputs_generated:
        return pd.DataFrame(
            [
                {
                    "output_state": "pending_outputs",
                    "current_claim_state": "not_ready",
                    "active_ladder_states": "not_ready",
                    "blocking_gate_ids": "V5-FINAL-G2-output-completeness",
                    "evidence": f"{generated_count}/{total_count} registered final splits generated",
                    "allowed_current_interpretation": "registered final evaluation is pending",
                    "forbidden_current_interpretation": "any predictive-condition or generality claim",
                    "required_paper_action": "report output status and rerun the frozen evaluator after outputs exist",
                }
            ]
        )
    if final_gates.empty:
        return pd.DataFrame(
            [
                {
                    "output_state": "generated_without_gate_report",
                    "current_claim_state": "not_ready",
                    "active_ladder_states": "control_reporting_failure",
                    "blocking_gate_ids": "final_gate_report.csv",
                    "evidence": f"{generated_count}/{total_count} registered final splits generated but final_gate_report.csv is missing",
                    "allowed_current_interpretation": "generated final tables require the frozen gate report before interpretation",
                    "forbidden_current_interpretation": "reading split results without the committed final gate report",
                    "required_paper_action": "rerun the same frozen final evaluator without changing score, split, threshold, or baseline rules",
                }
            ]
        )
    gate_lookup = final_gates.set_index("gate_id")["status"].astype(str).to_dict()
    evidence_lookup = final_gates.set_index("gate_id")["evidence"].astype(str).to_dict()
    p0_status = gate_lookup.get("v5_p0_predictive_condition_claim", "missing")
    non_p0 = final_gates[~final_gates["gate_id"].eq("v5_p0_predictive_condition_claim")].copy()
    blocking = non_p0[~non_p0["status"].astype(str).eq("pass")]
    blocking_ids = blocking["gate_id"].astype(str).tolist()
    active_states: list[str] = []
    if gate_lookup.get("v5_final_heldout_architecture_residual_spearman") == "pass" and (
        gate_lookup.get("v5_final_heldout_data_partition_residual_spearman") == "fail"
    ):
        active_states.append("data_transport_boundary")
    if gate_lookup.get("v5_final_heldout_data_partition_residual_spearman") == "pass" and (
        gate_lookup.get("v5_final_heldout_architecture_residual_spearman") == "fail"
    ):
        active_states.append("architecture_transport_boundary")
    if any("direction_threshold_accuracy" in gate_id for gate_id in blocking_ids):
        active_states.append("direction_guardrail_failure")
    if any("baseline_dominance" in gate_id for gate_id in blocking_ids):
        active_states.append("nuisance_proxy_boundary")
    arch_failed = any(
        gate_id.startswith("v5_final_heldout_architecture_") and status_value != "pass"
        for gate_id, status_value in gate_lookup.items()
    )
    data_failed = any(
        gate_id.startswith("v5_final_heldout_data_partition_") and status_value != "pass"
        for gate_id, status_value in gate_lookup.items()
    )
    if arch_failed and data_failed:
        active_states.append("local_mechanism_only")
    active_states = list(dict.fromkeys(active_states))
    if p0_status == "pass":
        current_claim_state = "p0_claim_eligible"
        active_ladder_states = "p0_claim_eligible"
        allowed = "the frozen transport-normalized score predicts residual layer risk on the two registered final splits"
        forbidden = "optimizer-performance or broad dataset/architecture claims"
        required = "report both final split summaries, controls, confidence intervals, and the no-retuning boundary"
        evidence = "v5_p0_predictive_condition_claim=pass"
    else:
        current_claim_state = "completed_final_failed_boundary"
        active_ladder_states = "; ".join(active_states) if active_states else "not_ready"
        allowed = "completed final negative boundary; local mechanism only under the v5 protocol"
        forbidden = "the v5 frozen score is an unseen-task predictive condition"
        required = "preserve failed gates, do not repair on final rows, and open a new unspent protocol for any score revision"
        evidence_parts = [
            f"{gate_id}={gate_lookup[gate_id]} ({evidence_lookup.get(gate_id, 'no evidence')})"
            for gate_id in blocking_ids
        ]
        evidence_parts.append(f"v5_p0_predictive_condition_claim={p0_status}")
        evidence = "; ".join(evidence_parts)
        blocking_ids.append("v5_p0_predictive_condition_claim")
    return pd.DataFrame(
        [
            {
                "output_state": "generated",
                "current_claim_state": current_claim_state,
                "active_ladder_states": active_ladder_states,
                "blocking_gate_ids": "; ".join(blocking_ids) if blocking_ids else "none",
                "evidence": evidence,
                "allowed_current_interpretation": allowed,
                "forbidden_current_interpretation": forbidden,
                "required_paper_action": required,
            }
        ]
    )


def build_leakage_lock(score: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "locked_item": "primary_score",
                "locked_value": score,
                "forbidden_after_final_outputs": "changing score weights, features, signs, or aliases",
                "allowed_after_final_outputs": "rerun the same evaluator and report generated rows",
            },
            {
                "locked_item": "final_splits",
                "locked_value": "ResNeXt50-32x4d CIFAR-100-LT; CIFAR-10 cross partition",
                "forbidden_after_final_outputs": "dropping, replacing, or adding splits to rescue the claim",
                "allowed_after_final_outputs": "add future splits only under a new preregistered protocol",
            },
            {
                "locked_item": "residual_gate",
                "locked_value": "Spearman CI lower endpoint above zero on each final split",
                "forbidden_after_final_outputs": "using mean-only positivity or a one-split pass as P0 evidence",
                "allowed_after_final_outputs": "report partial positives as boundary evidence",
            },
            {
                "locked_item": "direction_gate",
                "locked_value": "threshold accuracy and CI lower endpoint at least 0.8",
                "forbidden_after_final_outputs": "lowering the threshold or merging it into residual ranking",
                "allowed_after_final_outputs": "report a direction failure as a separate obstruction",
            },
            {
                "locked_item": "baseline_gate",
                "locked_value": "primary beats early_layer_prior on residual ranking and top-k overlap",
                "forbidden_after_final_outputs": "omitting the early-layer nuisance comparison",
                "allowed_after_final_outputs": "downgrade to nuisance-proxy boundary if baseline wins",
            },
        ]
    )


def write_discussion(
    status: pd.DataFrame,
    current_interpretation: pd.DataFrame,
    gate_contract: pd.DataFrame,
    outcome_ladder: pd.DataFrame,
    leakage_lock: pd.DataFrame,
    score: str,
) -> None:
    text = f"""# E11 Condition-Score V5 Final Interpretation Plan

This generated artifact is the post-output interpretation lock for the v5 final
splits. The outcome-to-claim state machine was fixed before final rows were used;
now that both registered final split tables are generated, this artifact records
the current completed negative boundary while it still forbids changing the
validation-frozen score `{score}`, split set, thresholds, or baseline
comparisons. The locked ladder explicitly separates positive P0 eligibility from
data/architecture transport boundaries, direction-guardrail failures,
baseline-dominance failures, and local-mechanism-only outcomes.

## Current Interpretation Summary

{markdown_table(current_interpretation, ["output_state", "current_claim_state", "active_ladder_states", "blocking_gate_ids", "allowed_current_interpretation", "forbidden_current_interpretation", "required_paper_action"])}

## Final Split Status

{markdown_table(status, ["split_id", "split_role", "current_output_status", "claim_gate_group", "can_be_replaced", "can_be_dropped_after_result"])}

## Gate Contract

{markdown_table(gate_contract, ["gate_id", "required_for", "pass_rule", "failure_effect"])}

## Outcome Ladder

{markdown_table(outcome_ladder, ["outcome_pattern", "claim_state", "allowed_interpretation", "forbidden_interpretation", "required_paper_action"])}

## Leakage Lock

{markdown_table(leakage_lock, ["locked_item", "locked_value", "forbidden_after_final_outputs", "allowed_after_final_outputs"])}

Artifacts:
- [current_interpretation_summary.csv](../{(OUTPUT_DIR / 'current_interpretation_summary.csv').as_posix()})
- [final_split_status.csv](../{(OUTPUT_DIR / 'final_split_status.csv').as_posix()})
- [final_gate_contract.csv](../{(OUTPUT_DIR / 'final_gate_contract.csv').as_posix()})
- [outcome_interpretation_ladder.csv](../{(OUTPUT_DIR / 'outcome_interpretation_ladder.csv').as_posix()})
- [leakage_lock.csv](../{(OUTPUT_DIR / 'leakage_lock.csv').as_posix()})
- [config.json](../{(OUTPUT_DIR / 'config.json').as_posix()})
"""
    write_markdown(DISCUSSION_PATH, text)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    splits = load_final_splits()
    score = selected_score()
    status = build_final_split_status(splits)
    final_gates = load_final_gates()
    current_interpretation = build_current_interpretation(status, final_gates)
    gate_contract = build_gate_contract(score)
    outcome_ladder = build_outcome_ladder()
    leakage_lock = build_leakage_lock(score)

    current_interpretation.to_csv(OUTPUT_DIR / "current_interpretation_summary.csv", index=False)
    status.to_csv(OUTPUT_DIR / "final_split_status.csv", index=False)
    gate_contract.to_csv(OUTPUT_DIR / "final_gate_contract.csv", index=False)
    outcome_ladder.to_csv(OUTPUT_DIR / "outcome_interpretation_ladder.csv", index=False)
    leakage_lock.to_csv(OUTPUT_DIR / "leakage_lock.csv", index=False)
    (OUTPUT_DIR / "config.json").write_text(
        json.dumps(
            {
                "primary_score": score,
                "freeze_status_path": (FREEZE_DIR / "freeze_status.csv").as_posix(),
                "final_evaluator_config": (FINAL_EVAL_DIR / "config.json").as_posix(),
                "final_outputs_generated": bool(status["current_output_status"].eq("generated").all()),
                "current_claim_state": str(current_interpretation["current_claim_state"].iloc[0]),
                "analysis_scope": "post-output v5 final interpretation lock; no final-row tuning",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    write_discussion(status, current_interpretation, gate_contract, outcome_ladder, leakage_lock, score)
    print(f"saved v5 final interpretation plan to {OUTPUT_DIR} and {DISCUSSION_PATH}")
    print(status.to_string(index=False))


if __name__ == "__main__":
    main()
