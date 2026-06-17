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
                "outcome_pattern": "both final splits missing or incomplete",
                "claim_state": "not_ready",
                "allowed_interpretation": "registered final evaluation is pending",
                "forbidden_interpretation": "any predictive-condition or generality claim",
                "required_paper_action": "report pending Slurm/output status and rerun the frozen evaluator after outputs exist",
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
    gate_contract: pd.DataFrame,
    outcome_ladder: pd.DataFrame,
    leakage_lock: pd.DataFrame,
    score: str,
) -> None:
    text = f"""# E11 Condition-Score V5 Final Interpretation Plan

This generated artifact is a pre-output interpretation lock for the v5 final
splits. It fixes the outcome-to-claim state machine for the submitted
ResNeXt50-32x4d architecture final and CIFAR-10 cross-partition final before
their layer tables are available. It uses the validation-frozen score `{score}`
and forbids changing the score, split set, thresholds, or baseline comparisons
after final outputs exist. The locked ladder explicitly separates positive P0
eligibility from data/architecture transport boundaries, direction-guardrail failures,
baseline-dominance failures, and local-mechanism-only outcomes.

## Final Split Status

{markdown_table(status, ["split_id", "split_role", "current_output_status", "claim_gate_group", "can_be_replaced", "can_be_dropped_after_result"])}

## Gate Contract

{markdown_table(gate_contract, ["gate_id", "required_for", "pass_rule", "failure_effect"])}

## Outcome Ladder

{markdown_table(outcome_ladder, ["outcome_pattern", "claim_state", "allowed_interpretation", "forbidden_interpretation", "required_paper_action"])}

## Leakage Lock

{markdown_table(leakage_lock, ["locked_item", "locked_value", "forbidden_after_final_outputs", "allowed_after_final_outputs"])}

Artifacts:
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
    gate_contract = build_gate_contract(score)
    outcome_ladder = build_outcome_ladder()
    leakage_lock = build_leakage_lock(score)

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
                "analysis_scope": "pre-output v5 final interpretation lock; no final-row tuning",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    write_discussion(status, gate_contract, outcome_ladder, leakage_lock, score)
    print(f"saved v5 final interpretation plan to {OUTPUT_DIR} and {DISCUSSION_PATH}")
    print(status.to_string(index=False))


if __name__ == "__main__":
    main()
