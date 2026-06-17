from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.reporting import markdown_table, write_markdown


OUTPUT_DIR = Path("results/e11_condition_score_v5_protocol/reviewer_failure_response")
DISCUSSION_PATH = Path("discussion/e11_condition_score_v5_reviewer_failure_response.md")
FREEZE_DIR = Path("results/e11_condition_score_v5_protocol/validation_score_freeze")
FINAL_EVAL_DIR = Path("results/e11_condition_score_v5_protocol/final_score_evaluation")
INTERPRET_DIR = Path("results/e11_condition_score_v5_protocol/final_interpretation_plan")


def selected_score() -> str:
    freeze = pd.read_csv(FREEZE_DIR / "freeze_status.csv").set_index("item")
    status = str(freeze.loc["v5 transport-normalized residual score", "status"])
    score = str(freeze.loc["v5 transport-normalized residual score", "evidence"])
    if status != "frozen":
        raise ValueError(f"reviewer failure response requires frozen v5 score, got {status}")
    return score


def load_final_splits() -> list[dict[str, object]]:
    config = json.loads((FINAL_EVAL_DIR / "config.json").read_text(encoding="utf-8"))
    return list(config["final_splits"])


def build_split_status(splits: list[dict[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for split in splits:
        layer_path = Path(str(split["layer_summary_path"]))
        metrics_path = Path(str(split["metrics_path"]))
        rows.append(
            {
                "split_id": split["split_id"],
                "split_role": split["role"],
                "display_name": split["display_name"],
                "layer_summary_path": layer_path.as_posix(),
                "metrics_path": metrics_path.as_posix(),
                "current_output_status": "generated" if layer_path.exists() and metrics_path.exists() else "not_run",
                "pre_output_policy": "do_not_change_score_or_split",
            }
        )
    return pd.DataFrame(rows)


def build_failure_mode_register(score: str) -> pd.DataFrame:
    rows = [
        {
            "failure_mode_id": "V5-RFR-0-pending-outputs",
            "trigger_pattern": "one or both registered final splits are missing layer/metric tables",
            "reviewer_objection": "The paper is still promising a final predictive-condition claim without final evidence.",
            "diagnostic_separation": "This is an output-completeness state, not a score failure or theory failure.",
            "allowed_claim": "v5 has a validation-frozen score and pending unspent final tests",
            "forbidden_claim": "any final predictive-condition, architecture-transfer, or data-transfer conclusion",
            "blocks_p0": "yes",
            "paper_action": "report not_ready status, Slurm/output paths, and the frozen evaluator command",
        },
        {
            "failure_mode_id": "V5-RFR-1-both-final-splits-pass",
            "trigger_pattern": "architecture and data final splits pass residual ranking, direction guardrail, baseline dominance, and controls",
            "reviewer_objection": "A positive result may be overclaimed as optimizer performance or universal transfer.",
            "diagnostic_separation": "Positive P0 evidence is restricted to frozen-score residual layer-risk prediction.",
            "allowed_claim": f"{score} predicts residual layer risk on the two registered final splits",
            "forbidden_claim": "competitive long-tail optimizer performance or broad coverage beyond the registered split family",
            "blocks_p0": "no",
            "paper_action": "report both final tables, confidence intervals, controls, and the no-retuning boundary",
        },
        {
            "failure_mode_id": "V5-RFR-2-data-transport-boundary",
            "trigger_pattern": "architecture final passes but CIFAR-10 cross-partition final fails residual ranking",
            "reviewer_objection": "The condition score does not transport across data families.",
            "diagnostic_separation": "Architecture transfer and data-partition transport are separated before final rows exist.",
            "allowed_claim": "architecture transfer survived, while data-family transport is a negative boundary",
            "forbidden_claim": "general data-family predictive-condition claim",
            "blocks_p0": "yes",
            "paper_action": "keep the CIFAR-10 failure in the main ledger and analyze partition transport terms",
        },
        {
            "failure_mode_id": "V5-RFR-3-architecture-transport-boundary",
            "trigger_pattern": "CIFAR-10 cross-partition final passes but ResNeXt50-32x4d architecture final fails residual ranking",
            "reviewer_objection": "The condition score is tied to the ResNet18/validation architecture family.",
            "diagnostic_separation": "Data-family transfer and architecture parameterization transfer are separated.",
            "allowed_claim": "data-family transfer survived, while architecture transport is a negative boundary",
            "forbidden_claim": "general architecture-family predictive-condition claim",
            "blocks_p0": "yes",
            "paper_action": "preserve the ResNeXt failure and analyze parameterization/normalization transport terms",
        },
        {
            "failure_mode_id": "V5-RFR-4-direction-guardrail-failure",
            "trigger_pattern": "either final split fails the below-one direction threshold gate",
            "reviewer_objection": "The basic spectral-vs-Frobenius direction comparison did not transport.",
            "diagnostic_separation": "Direction failure is distinct from scalar residual ranking failure.",
            "allowed_claim": "local drift mechanism remains supported only where direction guardrails hold",
            "forbidden_claim": "residual-risk predictor claim on any split with failed direction guardrail",
            "blocks_p0": "yes",
            "paper_action": "separate direction-threshold failure from score-ranking failure in the results table",
        },
        {
            "failure_mode_id": "V5-RFR-5-baseline-dominance-failure",
            "trigger_pattern": "primary frozen score does not beat early_layer_prior on residual Spearman and top-k overlap",
            "reviewer_objection": "The learned-looking score may only be a depth/stage nuisance proxy.",
            "diagnostic_separation": "Baseline dominance is evaluated independently of residual-ranking positivity.",
            "allowed_claim": "the final split exposes a nuisance-proxy boundary",
            "forbidden_claim": "theory-derived measurable-score improvement over simple stage priors",
            "blocks_p0": "yes",
            "paper_action": "report the early-layer comparison as a failed ablation gate",
        },
        {
            "failure_mode_id": "V5-RFR-6-control-reporting-failure",
            "trigger_pattern": "direction, early-layer, or source-observed controls are missing from a generated final split",
            "reviewer_objection": "The final table hides the controls needed to interpret the score.",
            "diagnostic_separation": "Incomplete reporting blocks review even when a primary statistic exists.",
            "allowed_claim": "none until the frozen evaluator emits the full control table",
            "forbidden_claim": "any final-score conclusion based on a partial table",
            "blocks_p0": "yes",
            "paper_action": "rerun the same evaluator without changing score, split, threshold, or baseline rules",
        },
        {
            "failure_mode_id": "V5-RFR-7-both-final-splits-fail",
            "trigger_pattern": "both registered final splits fail at least one required P0 gate",
            "reviewer_objection": "The predictive-condition program failed on unseen real-task finals.",
            "diagnostic_separation": "A double failure downgrades the paper to local mechanism plus falsification evidence.",
            "allowed_claim": "local matched-head-gain drift mechanism and negative predictive-condition boundary",
            "forbidden_claim": "unseen-task predictive-condition claim",
            "blocks_p0": "yes",
            "paper_action": "make the failures main-text evidence and stop the score-development line under this protocol",
        },
        {
            "failure_mode_id": "V5-RFR-8-post-final-leakage-pressure",
            "trigger_pattern": "any attempt to edit weights, features, signs, thresholds, or split membership after final outputs",
            "reviewer_objection": "The final result may be post-hoc tuned on the test set.",
            "diagnostic_separation": "Leakage is a protocol violation, not a recoverable empirical boundary.",
            "allowed_claim": "only pre-leakage frozen results remain citable",
            "forbidden_claim": "rescued P0 claim under the same v5 protocol",
            "blocks_p0": "yes",
            "paper_action": "open a new protocol with new unspent validation/final splits if score development continues",
        },
    ]
    return pd.DataFrame(rows)


def build_reviewer_objection_map() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "reviewer_objection_id": "RFR-O1-final-evidence",
                "objection": "The final predictive-condition claim is not yet evidenced.",
                "response_table": "failure_mode_register.csv",
                "decisive_artifact": "final_score_evaluation/final_gate_report.csv",
                "remaining_evidence": "Wait for registered ResNeXt50-32x4d and CIFAR-10 cross-partition outputs.",
            },
            {
                "reviewer_objection_id": "RFR-O2-generality",
                "objection": "Positive finals may still be too narrow for broad optimizer or dataset claims.",
                "response_table": "claim_downgrade_actions.csv",
                "decisive_artifact": "outcome_interpretation_ladder.csv",
                "remaining_evidence": "Add a separately registered larger-dataset or architecture-family protocol.",
            },
            {
                "reviewer_objection_id": "RFR-O3-baselines",
                "objection": "The frozen score may not beat a simple early-layer nuisance prior.",
                "response_table": "failure_mode_register.csv",
                "decisive_artifact": "final_score_summary.csv",
                "remaining_evidence": "Keep baseline-dominance as a required final gate.",
            },
            {
                "reviewer_objection_id": "RFR-O4-leakage",
                "objection": "The score may have been repaired after seeing final rows.",
                "response_table": "claim_downgrade_actions.csv",
                "decisive_artifact": "validation_score_freeze/freeze_status.csv",
                "remaining_evidence": "Preserve frozen score identity, split registry, and no-retuning language.",
            },
            {
                "reviewer_objection_id": "RFR-O5-mechanism-vs-performance",
                "objection": "The condition score should not be confused with final optimizer performance.",
                "response_table": "next_evidence_queue.csv",
                "decisive_artifact": "tuned benchmark protocol and future final benchmark outputs",
                "remaining_evidence": "Run tuned long-horizon optimizer benchmarks under validation/final separation.",
            },
        ]
    )


def build_claim_downgrade_actions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_state": "not_ready",
                "paper_location": "main claim ledger and limitations",
                "allowed_wording": "The frozen v5 final test is pending registered GPU outputs.",
                "forbidden_wording": "The v5 score generalizes to held-out finals.",
                "action_if_observed": "Report pending status and do not include final positive wording.",
                "top_conference_delta": "Blocks the P0 predictive-condition claim but preserves protocol credibility.",
            },
            {
                "claim_state": "p0_claim_eligible",
                "paper_location": "main results, not optimizer benchmark section",
                "allowed_wording": "The frozen score predicts residual layer risk on both registered final splits.",
                "forbidden_wording": "Muon or spectral training is a stronger long-tail optimizer.",
                "action_if_observed": "Add final split CIs, controls, and the no-retuning boundary.",
                "top_conference_delta": "Upgrades the mechanism paper with a narrow unseen-task predictor result.",
            },
            {
                "claim_state": "single-axis-transport-boundary",
                "paper_location": "main limitations and failure analysis",
                "allowed_wording": "One transfer axis failed under the frozen protocol.",
                "forbidden_wording": "The passing split proves general transfer.",
                "action_if_observed": "Downgrade to architecture-only or data-only boundary evidence.",
                "top_conference_delta": "Keeps the paper falsifiable but leaves P0 unresolved.",
            },
            {
                "claim_state": "direction_guardrail_failure",
                "paper_location": "mechanism diagnostics",
                "allowed_wording": "The local direction guardrail did not transport on the failed split.",
                "forbidden_wording": "Residual ranking is sufficient despite direction failure.",
                "action_if_observed": "Separate direction and scalar-score mechanisms in the table.",
                "top_conference_delta": "Forces a narrower local-geometry claim.",
            },
            {
                "claim_state": "nuisance_proxy_boundary",
                "paper_location": "ablation table and reviewer-risk response",
                "allowed_wording": "The score did not dominate the early-layer nuisance prior.",
                "forbidden_wording": "The theory-derived score adds predictive information.",
                "action_if_observed": "Report baseline win and move score development to future protocol.",
                "top_conference_delta": "Prevents overclaiming and identifies the next theory gap.",
            },
            {
                "claim_state": "local_mechanism_only",
                "paper_location": "abstract, conclusion, and limitations",
                "allowed_wording": "The paper establishes a local drift mechanism and records failed final predictors.",
                "forbidden_wording": "The paper has an unseen-task predictive condition.",
                "action_if_observed": "Make negative finals visible and remove predictive-condition headline language.",
                "top_conference_delta": "Shifts the contribution to theory plus high-integrity falsification.",
            },
        ]
    )


def build_next_evidence_queue() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "priority": "P0",
                "evidence_item": "consume the two registered v5 final outputs",
                "resolves_failure_modes": "V5-RFR-0-pending-outputs",
                "command_or_protocol": "make e11-cifar-resnet-condition-score-v5-final-eval",
                "claim_unlocked": "narrow frozen-score P0 eligibility if both finals pass",
                "depends_on_final_outputs": "yes",
            },
            {
                "priority": "P0",
                "evidence_item": "rerun the reviewer response asset after final evaluator output changes",
                "resolves_failure_modes": "V5-RFR-1 through V5-RFR-7",
                "command_or_protocol": "make e11-cifar-resnet-condition-score-v5-reviewer-failure-response",
                "claim_unlocked": "updated claim downgrade table without score retuning",
                "depends_on_final_outputs": "yes",
            },
            {
                "priority": "P1",
                "evidence_item": "registered larger-dataset matched-head-gain diagnostic",
                "resolves_failure_modes": "data and architecture transport-boundary concerns",
                "command_or_protocol": "new preregistered protocol with unspent validation/final splits",
                "claim_unlocked": "broader mechanism generality",
                "depends_on_final_outputs": "no",
            },
            {
                "priority": "P1",
                "evidence_item": "tuned long-horizon optimizer benchmark",
                "resolves_failure_modes": "mechanism-vs-performance objection",
                "command_or_protocol": "run tuned benchmark validation grid, selection, then final paired seeds",
                "claim_unlocked": "optimizer-performance claim if positive",
                "depends_on_final_outputs": "no",
            },
            {
                "priority": "P2",
                "evidence_item": "theory refinement for transport-normalized sandwich residual terms",
                "resolves_failure_modes": "baseline dominance and transport reversal failures",
                "command_or_protocol": "derive split-invariant normalization or prove narrower fixed-partition theorem",
                "claim_unlocked": "cleaner theorem-to-score alignment",
                "depends_on_final_outputs": "no",
            },
        ]
    )


def write_discussion(
    split_status: pd.DataFrame,
    failure_modes: pd.DataFrame,
    objections: pd.DataFrame,
    downgrades: pd.DataFrame,
    next_queue: pd.DataFrame,
    score: str,
) -> None:
    generated_count = int(split_status["current_output_status"].eq("generated").sum())
    text = f"""# E11 Condition-Score V5 Reviewer Failure Response

This generated top-conference reviewer failure response is a pre-output
claim-downgrade plan for the v5 final condition-score test. It reads the
validation-frozen score `{score}` and the registered final split paths, but it
does not inspect, refit, reselect, or retune on final rows. Its purpose is to
make every plausible final outcome reviewable before the pending ResNeXt50-32x4d
and CIFAR-10 cross-partition tables exist.

Current final split outputs generated: {generated_count}/{len(split_status)}.

## Final Split Output Status

{markdown_table(split_status, ["split_id", "split_role", "current_output_status", "pre_output_policy"])}

## Failure Mode Register

{markdown_table(failure_modes, ["failure_mode_id", "trigger_pattern", "allowed_claim", "forbidden_claim", "blocks_p0", "paper_action"])}

## Reviewer Objection Map

{markdown_table(objections, ["reviewer_objection_id", "objection", "response_table", "decisive_artifact", "remaining_evidence"])}

## Claim Downgrade Actions

{markdown_table(downgrades, ["claim_state", "allowed_wording", "forbidden_wording", "action_if_observed", "top_conference_delta"])}

## Next Evidence Queue

{markdown_table(next_queue, ["priority", "evidence_item", "resolves_failure_modes", "command_or_protocol", "claim_unlocked"])}

Artifacts:
- [final_split_output_status.csv](../{(OUTPUT_DIR / 'final_split_output_status.csv').as_posix()})
- [failure_mode_register.csv](../{(OUTPUT_DIR / 'failure_mode_register.csv').as_posix()})
- [reviewer_objection_map.csv](../{(OUTPUT_DIR / 'reviewer_objection_map.csv').as_posix()})
- [claim_downgrade_actions.csv](../{(OUTPUT_DIR / 'claim_downgrade_actions.csv').as_posix()})
- [next_evidence_queue.csv](../{(OUTPUT_DIR / 'next_evidence_queue.csv').as_posix()})
- [config.json](../{(OUTPUT_DIR / 'config.json').as_posix()})
"""
    write_markdown(DISCUSSION_PATH, text)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    score = selected_score()
    split_status = build_split_status(load_final_splits())
    failure_modes = build_failure_mode_register(score)
    objections = build_reviewer_objection_map()
    downgrades = build_claim_downgrade_actions()
    next_queue = build_next_evidence_queue()

    split_status.to_csv(OUTPUT_DIR / "final_split_output_status.csv", index=False)
    failure_modes.to_csv(OUTPUT_DIR / "failure_mode_register.csv", index=False)
    objections.to_csv(OUTPUT_DIR / "reviewer_objection_map.csv", index=False)
    downgrades.to_csv(OUTPUT_DIR / "claim_downgrade_actions.csv", index=False)
    next_queue.to_csv(OUTPUT_DIR / "next_evidence_queue.csv", index=False)
    (OUTPUT_DIR / "config.json").write_text(
        json.dumps(
            {
                "primary_score": score,
                "freeze_status_path": (FREEZE_DIR / "freeze_status.csv").as_posix(),
                "final_evaluator_config": (FINAL_EVAL_DIR / "config.json").as_posix(),
                "interpretation_plan_dir": INTERPRET_DIR.as_posix(),
                "final_outputs_generated": int(split_status["current_output_status"].eq("generated").sum()),
                "analysis_scope": "pre-output reviewer failure response; no final-row tuning",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_discussion(split_status, failure_modes, objections, downgrades, next_queue, score)
    print(f"saved {DISCUSSION_PATH} and {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
