from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from statistics import NormalDist

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.reporting import fmt, markdown_table, write_markdown


OUTPUT_DIR = Path("results/e11_condition_score_v5_protocol/final_power_audit")
DISCUSSION_PATH = Path("discussion/e11_condition_score_v5_final_power_audit.md")
FREEZE_DIR = Path("results/e11_condition_score_v5_protocol/validation_score_freeze")
FINAL_EVAL_DIR = Path("results/e11_condition_score_v5_protocol/final_score_evaluation")
VALIDATION_LAYER_SUMMARY = Path(
    "results/e11_condition_score_v5_protocol/validation_cifar100_mod4_partition/layer_summary.csv"
)


def normal_critical(alpha: float) -> float:
    return NormalDist().inv_cdf(1.0 - alpha / 2.0)


def selected_score() -> str:
    freeze = pd.read_csv(FREEZE_DIR / "freeze_status.csv").set_index("item")
    status = str(freeze.loc["v5 transport-normalized residual score", "status"])
    score = str(freeze.loc["v5 transport-normalized residual score", "evidence"])
    if status != "frozen":
        raise ValueError(f"v5 final power audit requires a frozen score, got {status}")
    return score


def load_final_config() -> dict[str, object]:
    return json.loads((FINAL_EVAL_DIR / "config.json").read_text(encoding="utf-8"))


def reference_design() -> dict[str, int]:
    validation = pd.read_csv(VALIDATION_LAYER_SUMMARY)
    points_per_warmup = (
        validation.groupby("warmup_steps", observed=True)["parameter"].nunique().astype(int).tolist()
    )
    if not points_per_warmup:
        raise ValueError("validation layer summary has no warmup-level point counts")
    source_layer_summary = Path(str(load_final_config()["source_layer_summary"]))
    source = pd.read_csv(source_layer_summary)
    source_warmups = int(source["warmup_steps"].nunique())
    target_warmups = int(validation["warmup_steps"].nunique())
    return {
        "reference_points_per_transfer": int(min(points_per_warmup)),
        "reference_source_warmups": source_warmups,
        "reference_target_warmups": target_warmups,
        "reference_transfer_pairs": source_warmups * target_warmups,
    }


def build_split_power_design(config: dict[str, object], design: dict[str, int]) -> pd.DataFrame:
    rows = []
    for split in config["final_splits"]:
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
                "reference_points_per_transfer": design["reference_points_per_transfer"],
                "reference_transfer_pairs": design["reference_transfer_pairs"],
                "primary_residual_gate": "mean Spearman CI lower endpoint above zero",
                "power_boundary": "pre-output reference only; actual final evaluator reports observed mean_points",
            }
        )
    return pd.DataFrame(rows)


def fisher_resolution(points_values: list[int], validation_spearman: float = 0.645, alpha: float = 0.05) -> pd.DataFrame:
    zcrit = normal_critical(alpha)
    rows = []
    for points in sorted(set(points_values)):
        if points <= 3:
            continue
        half_width = zcrit / math.sqrt(points - 3)
        min_r = math.tanh(half_width)
        validation_z = math.atanh(max(min(validation_spearman, 0.999999), -0.999999))
        rows.append(
            {
                "points": int(points),
                "two_sided_alpha": alpha,
                "fisher_z_half_width": half_width,
                "minimum_observed_spearman_for_ci_low_above_zero": min_r,
                "validation_reference_spearman": validation_spearman,
                "validation_reference_ci_low_at_points": math.tanh(validation_z - half_width),
                "claim_interpretation": (
                    "a generated final split needs an observed residual Spearman at least this large "
                    "before a Fisher-z pairwise CI can exclude zero"
                ),
            }
        )
    return pd.DataFrame(rows)


def mean_spearman_mde(transfer_pairs: int, alpha: float = 0.05) -> pd.DataFrame:
    multiplier = 2.776 if transfer_pairs <= 5 else 1.96
    rows = []
    for across_pair_sd in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
        min_mean = multiplier * across_pair_sd / math.sqrt(transfer_pairs)
        rows.append(
            {
                "transfer_pairs": int(transfer_pairs),
                "two_sided_alpha": alpha,
                "ci_multiplier_used_by_evaluator": multiplier,
                "assumed_across_pair_spearman_sd": across_pair_sd,
                "minimum_mean_spearman_for_ci_low_above_zero": min_mean,
                "claim_interpretation": (
                    "if the final mean Spearman is below this row, a non-positive summary CI "
                    "is not a strong negative result under the assumed across-pair variability"
                ),
            }
        )
    return pd.DataFrame(rows)


def build_interpretation_ladder() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "case_id": "V5-PWR-1-pending-outputs",
                "evidence_condition": "one or both registered final split layer tables are absent",
                "allowed_wording": "registered final power audit only; P0 remains not_ready",
                "blocked_wording": "positive or negative final predictive-condition result",
            },
            {
                "case_id": "V5-PWR-2-positive-above-zero-ci",
                "evidence_condition": "both splits generated; primary mean Spearman CI lower endpoint > 0; other final gates pass",
                "allowed_wording": "narrow frozen-score residual-risk predictor on the registered final splits",
                "blocked_wording": "optimizer-performance claim or broader architecture/data-family generality",
            },
            {
                "case_id": "V5-PWR-3-null-below-detectable-scale",
                "evidence_condition": "generated split has non-positive CI but mean effect is below audited MDE for observed variability",
                "allowed_wording": "underpowered or small-effect final boundary",
                "blocked_wording": "the frozen score is disproven for all natural tasks",
            },
            {
                "case_id": "V5-PWR-4-negative-above-detectable-scale",
                "evidence_condition": "generated split fails residual ranking with effect scale above audited detectable threshold",
                "allowed_wording": "negative architecture/data transport boundary under the frozen protocol",
                "blocked_wording": "post-hoc score repair under the same v5 final protocol",
            },
            {
                "case_id": "V5-PWR-5-high-heterogeneity",
                "evidence_condition": "across-pair Spearman SD is high enough that summary CI is dominated by transfer-pair heterogeneity",
                "allowed_wording": "transport heterogeneity diagnostic requiring split-specific mechanism analysis",
                "blocked_wording": "single scalar score success/failure without heterogeneity reporting",
            },
        ]
    )


def build_outcome_state_machine() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "state_id": "V5-PWR-S1-not-run",
                "trigger": "0/2 final split summaries generated",
                "claim_state": "not_ready",
                "required_action": "wait for Slurm outputs; do not interpret missing rows as evidence",
            },
            {
                "state_id": "V5-PWR-S2-partial",
                "trigger": "1/2 final split summaries generated",
                "claim_state": "not_ready_partial_family",
                "required_action": "report the generated split as progress only; keep P0 closed",
            },
            {
                "state_id": "V5-PWR-S3-complete-positive",
                "trigger": "2/2 final split summaries generated and both pass residual, direction, baseline, and reporting gates",
                "claim_state": "p0_claim_eligible_with_power_context",
                "required_action": "report observed points, transfer-pair variability, controls, and no-retuning boundary",
            },
            {
                "state_id": "V5-PWR-S4-complete-small-null",
                "trigger": "2/2 generated; residual gate fails but observed scale is below audited MDE",
                "claim_state": "underpowered_final_boundary",
                "required_action": "downgrade to small-effect/underpowered boundary and add a new preregistered larger final family",
            },
            {
                "state_id": "V5-PWR-S5-complete-negative",
                "trigger": "2/2 generated; residual gate fails at or above audited detectable scale",
                "claim_state": "negative_transport_boundary",
                "required_action": "preserve failed final as main evidence and do not retune on final rows",
            },
        ]
    )


def write_discussion(
    split_design: pd.DataFrame,
    fisher: pd.DataFrame,
    mde: pd.DataFrame,
    ladder: pd.DataFrame,
    states: pd.DataFrame,
    score: str,
) -> None:
    generated = int(split_design["current_output_status"].eq("generated").sum())
    design_points = int(split_design["reference_points_per_transfer"].iloc[0])
    design_pairs = int(split_design["reference_transfer_pairs"].iloc[0])
    fisher_row = fisher[fisher["points"].eq(design_points)].iloc[0]
    mde_row = mde[
        mde["assumed_across_pair_spearman_sd"].eq(0.2)
    ].iloc[0]
    text = f"""# E11 Condition-Score V5 Final Power Audit

This generated audit is a pre-output detectable-effect contract for the v5
final condition-score test. It reads the validation-frozen score `{score}` and
the registered final split paths, but it does not inspect, refit, reselect, or
retune on final rows. Its purpose is to define how much residual-Spearman signal
the submitted ResNeXt50-32x4d and CIFAR-10 cross-partition final splits can
resolve before their outputs exist. The audit records Fisher-z resolution,
mean-Spearman MDE, and negative transport boundary wording before either final
split is interpreted.

Current final split outputs generated: {generated}/{len(split_design)}.

Under the reference design, each transfer correlation has about {design_points}
layer points and the final summary averages {design_pairs} source/target
transfer-pair correlations. A single transfer correlation needs observed
Spearman at least {fmt(fisher_row['minimum_observed_spearman_for_ci_low_above_zero'])}
for its Fisher-z CI lower endpoint to exceed zero. With across-pair Spearman SD
0.2, the final mean Spearman needs to be at least
{fmt(mde_row['minimum_mean_spearman_for_ci_low_above_zero'])} before the
evaluator's summary CI lower endpoint can exclude zero.

## Split Power Design

{markdown_table(split_design, ["split_id", "split_role", "current_output_status", "reference_points_per_transfer", "reference_transfer_pairs", "primary_residual_gate"])}

## Pairwise Fisher-Z Resolution

{markdown_table(fisher, ["points", "minimum_observed_spearman_for_ci_low_above_zero", "validation_reference_spearman", "validation_reference_ci_low_at_points", "claim_interpretation"])}

## Final Mean Spearman MDE

{markdown_table(mde, ["transfer_pairs", "assumed_across_pair_spearman_sd", "minimum_mean_spearman_for_ci_low_above_zero", "claim_interpretation"])}

## Interpretation Ladder

{markdown_table(ladder, ["case_id", "evidence_condition", "allowed_wording", "blocked_wording"])}

## Outcome State Machine

{markdown_table(states, ["state_id", "trigger", "claim_state", "required_action"])}

Artifacts:
- [split_power_design.csv](../{(OUTPUT_DIR / 'split_power_design.csv').as_posix()})
- [fisher_z_resolution.csv](../{(OUTPUT_DIR / 'fisher_z_resolution.csv').as_posix()})
- [mean_spearman_mde.csv](../{(OUTPUT_DIR / 'mean_spearman_mde.csv').as_posix()})
- [interpretation_ladder.csv](../{(OUTPUT_DIR / 'interpretation_ladder.csv').as_posix()})
- [outcome_state_machine.csv](../{(OUTPUT_DIR / 'outcome_state_machine.csv').as_posix()})
- [config.json](../{(OUTPUT_DIR / 'config.json').as_posix()})
"""
    write_markdown(DISCUSSION_PATH, text)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    score = selected_score()
    config = load_final_config()
    design = reference_design()
    split_design = build_split_power_design(config, design)
    points_values = [
        max(4, design["reference_points_per_transfer"] - 6),
        design["reference_points_per_transfer"],
        design["reference_points_per_transfer"] + 9,
        design["reference_points_per_transfer"] + 19,
    ]
    fisher = fisher_resolution(points_values)
    mde = mean_spearman_mde(design["reference_transfer_pairs"])
    ladder = build_interpretation_ladder()
    states = build_outcome_state_machine()

    split_design.to_csv(OUTPUT_DIR / "split_power_design.csv", index=False)
    fisher.to_csv(OUTPUT_DIR / "fisher_z_resolution.csv", index=False)
    mde.to_csv(OUTPUT_DIR / "mean_spearman_mde.csv", index=False)
    ladder.to_csv(OUTPUT_DIR / "interpretation_ladder.csv", index=False)
    states.to_csv(OUTPUT_DIR / "outcome_state_machine.csv", index=False)
    (OUTPUT_DIR / "config.json").write_text(
        json.dumps(
            {
                "primary_score": score,
                "freeze_status_path": (FREEZE_DIR / "freeze_status.csv").as_posix(),
                "final_evaluator_config": (FINAL_EVAL_DIR / "config.json").as_posix(),
                "validation_reference_layer_summary": VALIDATION_LAYER_SUMMARY.as_posix(),
                "reference_points_per_transfer": design["reference_points_per_transfer"],
                "reference_transfer_pairs": design["reference_transfer_pairs"],
                "final_outputs_generated": int(split_design["current_output_status"].eq("generated").sum()),
                "analysis_scope": "pre-output v5 final detectable-effect audit; no final-row tuning",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_discussion(split_design, fisher, mde, ladder, states, score)
    print(f"saved {DISCUSSION_PATH} and {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
