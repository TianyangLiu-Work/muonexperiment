from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.reporting import markdown_table, write_markdown


RESULT_PATH = Path("results/e11_top_conference_gap_register/gap_register.csv")
OUTPUT_PATH = Path("discussion/e11_top_conference_gap_register.md")


def gap_rows() -> list[dict[str, str]]:
    return [
        {
            "gap_id": "P0-PredictiveCondition",
            "priority": "P0",
            "claim_unblocked": "A measured downstream-aware condition predicts when the matched-head-gain spectral/polar direction will lower tail-example drift beyond constructed settings.",
            "current_state": "The checkpoint-transfer benchmark is a boundary result, the initial condition-score protocol is recorded in discussion/e11_cifar100_resnet_condition_score_protocol.md, the locked ResNet18 retrospective v2 split passes, the registered v2 held-out ResNet34/CIFAR-10-LT evaluation fails, and the frozen fresh v3 zero-fit scaled-JVP revision is evaluated in discussion/e11_condition_score_fresh_evaluation.md: CIFAR-10 alternate partition passes residual ranking, but ResNet50 architecture fails residual ranking and early-layer-prior comparison. The obstruction is summarized in discussion/e11_condition_score_failure_mechanism_audit.md. V4 freezes condition_score_v4_two_axis_amplitude_minus_direction in discussion/e11_condition_score_v4_validation_freeze.md and evaluates both unspent final splits in discussion/e11_condition_score_v4_final_evaluation.md: WideResNet50-2 passes residual ranking, but the CIFAR-10 mixed final data split fails with a negative residual Spearman CI. The v4 failure audit in discussion/e11_condition_score_v4_failure_mechanism_audit.md localizes this data-partition reversal mechanism to amplitude/depth scalar aggregation rather than the direction threshold. The v5 theory protocol in discussion/e11_condition_score_v5_theory_protocol.md registers a transport-normalized score contract, spent-final quarantine, and new unspent validation/final split requirements. The v5 theory-to-score map in discussion/e11_condition_score_v5_theory_to_score_map.md now states the transport-stable sandwich residual proposition, maps theorem terms to measurable score features, and pre-registers ablation/falsification gates. The v5 validation-freeze boundary in discussion/e11_condition_score_v5_validation_freeze.md now freezes condition_score_v5_transport_normalized_amplitude_minus_direction after the validation-only mod-4 CIFAR-100-LT split, with validation residual Spearman 0.645 [0.5898, 0.7002], a passing direction guardrail, and no final outputs before freeze. The pre-registered final evaluator in discussion/e11_condition_score_v5_final_evaluation.md currently reports not_run until the submitted ResNeXt50-32x4d and CIFAR-10 cross-partition final Slurm jobs write their layer tables. The final interpretation lock in discussion/e11_condition_score_v5_final_interpretation_plan.md fixes the outcome-to-claim state machine before final outputs exist. The reviewer failure response in discussion/e11_condition_score_v5_reviewer_failure_response.md adds a claim-downgrade plan for top-conference pass/fail objections before final rows exist.",
            "required_next_evidence": "Wait for the submitted registered v5 ResNeXt50-32x4d architecture final and CIFAR-10 cross-partition final GPU jobs to finish, then rerun make e11-cifar-resnet-condition-score-v5-final-eval using the frozen validation-selected score and interpret the result under make e11-cifar-resnet-condition-score-v5-final-interpretation-plan plus make e11-cifar-resnet-condition-score-v5-reviewer-failure-response, without using v2/v3/v4 final rows or v5 final rows for score fitting or selection.",
            "acceptance_gate": "For any v5 condition-score claim, report positive held-out residual Spearman with a CI lower endpoint above zero on the new unspent architecture and data splits, threshold accuracy above 0.8, explicit early-layer/v2/v3/v4/source-observed controls, and a statement that all failed v2/v3/v4 final held-outs were quarantined from score fitting and selection.",
            "compute_mode": "GPU via Slurm",
            "planned_artifacts": "results/e11_cifar100_resnet_condition_score_protocol/*; results/e11_cifar100_resnet_condition_score_next/*; results/e11_condition_score_fresh_protocol/*; results/e11_condition_score_failure_mechanism_audit/*; results/e11_condition_score_v4_protocol/*; results/e11_condition_score_v4_failure_mechanism_audit/*; results/e11_condition_score_v5_theory_protocol/*; results/e11_condition_score_v5_theory_to_score_map/*; results/e11_condition_score_v5_protocol/*; figures/e11_condition_score_fresh_protocol/*; figures/e11_condition_score_failure_mechanism_audit/*; figures/e11_condition_score_v4_protocol/*; figures/e11_condition_score_v4_failure_mechanism_audit/*; figures/e11_condition_score_v5_protocol/*; discussion/e11_cifar100_resnet_condition_score_next.md; discussion/e11_cifar100_resnet_condition_score_next_heldout_evaluation.md; discussion/e11_condition_score_fresh_protocol.md; discussion/e11_condition_score_fresh_evaluation.md; discussion/e11_condition_score_failure_mechanism_audit.md; discussion/e11_condition_score_v4_protocol.md; discussion/e11_condition_score_v4_validation_freeze.md; discussion/e11_condition_score_v4_final_evaluation.md; discussion/e11_condition_score_v4_failure_mechanism_audit.md; discussion/e11_condition_score_v5_theory_protocol.md; discussion/e11_condition_score_v5_theory_to_score_map.md; discussion/e11_condition_score_v5_validation_freeze.md; discussion/e11_condition_score_v5_final_evaluation.md; discussion/e11_condition_score_v5_final_interpretation_plan.md; discussion/e11_condition_score_v5_reviewer_failure_response.md",
            "risk_if_missing": "The paper can still claim a local drift mechanism, but it cannot claim the current condition score is predictive on unseen real tasks.",
        },
        {
            "gap_id": "P0-StandardBenchmark",
            "priority": "P0",
            "claim_unblocked": "Any benchmark-level statement about final long-tail performance or practical optimizer quality.",
            "current_state": "The repo has a CIFAR-100-LT IF=100 AdamW reporting baseline, a 5-seed augmented recipe pilot, and a negative 3-seed finite-NS-Muon final-training pilot. The tuned benchmark protocol in discussion/e11_cifar100_resnet_lt_tuned_benchmark_protocol.md now quarantines those pilots as context and registers validation/final seed splits, tuned AdamW/SGD/class-balanced baselines, finite-NS-Muon candidate grids, familywise final comparisons, and scope-control gates before any tuned final-performance result exists. The executable validation-grid runner scripts/e11_run_cifar100_resnet_lt_tuned_benchmark.py and Slurm wrapper scripts/slurm/e11_cifar100_resnet_lt_tuned_benchmark_validation.sbatch materialize the 164-setting validation registry under results/e11_cifar100_resnet_lt_tuned_benchmark/settings_registry.csv. The selection audit in discussion/e11_cifar100_resnet_lt_tuned_benchmark_selection.md remains not_ready until validation summaries are complete and keeps final seeds 20..29 quarantined.",
            "required_next_evidence": "Run the registered tuned multi-optimizer validation grid, rerun make e11-cifar-resnet-lt-tuned-benchmark-selection to select recipes on validation seeds only, then run untouched final claim seeds with augmentation, schedules, weight decay grids, class-balanced losses or samplers, finite-NS-Muon grids, and final many/medium/few metrics; extend beyond CIFAR-100-LT under a separate preregistered protocol if the paper wants broad benchmark scope.",
            "acceptance_gate": "Separate local-drift and final-performance claims; for any performance claim, report paired seeds, tuned baselines, all/few/many metrics, and confidence intervals with no post-hoc recipe selection.",
            "compute_mode": "GPU via Slurm",
            "planned_artifacts": "results/e11_cifar100_resnet_lt_tuned_benchmark_protocol/*; discussion/e11_cifar100_resnet_lt_tuned_benchmark_protocol.md; scripts/e11_write_cifar100_resnet_lt_tuned_benchmark_protocol.py; scripts/e11_run_cifar100_resnet_lt_tuned_benchmark.py; scripts/e11_write_cifar100_resnet_lt_tuned_benchmark_selection.py; scripts/slurm/e11_cifar100_resnet_lt_tuned_benchmark_validation.sbatch; results/e11_cifar100_resnet_lt_tuned_benchmark/*; figures/e11_cifar100_resnet_lt_tuned_benchmark/*; discussion/e11_cifar100_resnet_lt_tuned_benchmark_selection.md; discussion/e11_cifar100_resnet_lt_tuned_benchmark.md",
            "risk_if_missing": "Reviewers can reject any implied practical optimizer advantage; the safe story remains mechanism-only.",
        },
        {
            "gap_id": "P1-HeldOutGenerality",
            "priority": "P1",
            "claim_unblocked": "The matched-head-gain drift mechanism is not an artifact of one ResNet18 checkpoint family.",
            "current_state": "The current strongest visual evidence is CIFAR-100-LT ResNet18 with checkpoint, tail-quality, imbalance, all-layer JVP, and trajectory-state bridge diagnostics.",
            "required_next_evidence": "Repeat the matched-head-gain diagnostic on at least one held-out architecture family and one held-out data family, keeping the same acceptance criteria fixed before looking at results.",
            "acceptance_gate": "Every reported setting must include seed count, checkpoint quality, tail accuracy before update, drift ratio CI, tail-loss sign, and an explicit unsupported-final-accuracy caveat.",
            "compute_mode": "GPU via Slurm",
            "planned_artifacts": "results/e11_heldout_arch_data_drift/*; figures/e11_heldout_arch_data_drift/*; discussion/e11_heldout_arch_data_drift.md",
            "risk_if_missing": "The paper remains credible as a focused CIFAR-100-LT ResNet mechanism study, but generality will be a predictable reviewer concern.",
        },
        {
            "gap_id": "P1-TheoryToScore",
            "priority": "P1",
            "claim_unblocked": "The theorem and measurable real-model score are visibly connected rather than adjacent.",
            "current_state": "The synthetic sandwich-block condition gives the right sign boundary, discussion/e11_matrix_block_theorem_proof.md now records the matched-gain theorem statement, sandwich sensitivity lemma, proof steps, assumptions, and claim implications, discussion/e11_matrix_block_tightness_audit.md checks exact diagonal witnesses, the coefficient-ratio identity, equality/Frobenius-favored boundaries, and degenerate-tail caveats, discussion/e11_condition_score_theory_bridge.md separates the supported threshold-direction guardrail from failed residual-ranking targets, discussion/e11_condition_score_failure_mechanism_audit.md shows the current measurable score reverses on held-out ResNet50 bottleneck/downsample parameterization, discussion/e11_condition_score_v4_protocol.md registers a two-axis transport protocol before unspent final splits, discussion/e11_condition_score_v4_validation_freeze.md freezes the amplitude-minus-direction residual score after the validation-only split, discussion/e11_condition_score_v4_final_evaluation.md shows that this score still reverses on the CIFAR-10 mixed final data split, discussion/e11_condition_score_v4_failure_mechanism_audit.md shows the v4 reversal is not a direction-threshold failure but an amplitude/depth aggregation failure, discussion/e11_theory_proof_obligation_register.md now maps theorem claims, assumptions, blocked wording, and pending empirical gates into a proof-obligation register, discussion/e11_condition_score_v5_theory_protocol.md states the sandwiched tail-drift contract plus a required partition/architecture transport term, discussion/e11_condition_score_v5_theory_to_score_map.md gives a transport-stable sandwich residual proposition plus theorem-to-feature lineage, discussion/e11_condition_score_v5_validation_freeze.md freezes the transport-normalized residual score before final outputs exist, discussion/e11_condition_score_v5_final_evaluation.md pre-registers the final residual, direction, baseline-dominance, and control-reporting gates, and discussion/e11_condition_score_v5_final_interpretation_plan.md locks the allowed claim states for all pass/fail patterns before final outputs exist.",
            "required_next_evidence": "Keep the matrix-block theorem proof and tightness audit synchronized with the paper appendix, wait for the registered v5 unspent final split jobs to finish, rerun the frozen residual-score evaluator, and preserve negative outcomes if the transport term fails on either final split.",
            "acceptance_gate": "Add a theorem, proposition, or falsifiable conjecture plus an ablation showing which transport term makes the score fail or succeed on the checkpoint-transfer benchmark; future held-out splits must not reuse failed v2, v3, or v4 final results for tuning.",
            "compute_mode": "CPU plus optional GPU via Slurm",
            "planned_artifacts": "discussion/e11_matrix_block_theorem_proof.md; discussion/e11_matrix_block_tightness_audit.md; discussion/e11_condition_score_theory_bridge.md; discussion/e11_condition_score_failure_mechanism_audit.md; discussion/e11_condition_score_v4_protocol.md; discussion/e11_condition_score_v4_validation_freeze.md; discussion/e11_condition_score_v4_final_evaluation.md; discussion/e11_condition_score_v4_failure_mechanism_audit.md; discussion/e11_theory_proof_obligation_register.md; discussion/e11_condition_score_v5_theory_protocol.md; discussion/e11_condition_score_v5_theory_to_score_map.md; discussion/e11_condition_score_v5_validation_freeze.md; discussion/e11_condition_score_v5_final_evaluation.md; discussion/e11_condition_score_v5_final_interpretation_plan.md; results/e11_matrix_block_theorem_proof/*; results/e11_matrix_block_tightness_audit/*; results/e11_condition_score_failure_mechanism_audit/*; results/e11_condition_score_v4_protocol/*; results/e11_condition_score_v4_failure_mechanism_audit/*; results/e11_theory_proof_obligation_register/*; results/e11_condition_score_v5_theory_protocol/*; results/e11_condition_score_v5_theory_to_score_map/*; results/e11_condition_score_v5_protocol/*; results/e11_condition_score_ablation/*; figures/e11_condition_score_ablation/*",
            "risk_if_missing": "The work reads as a collection of diagnostics rather than a tight mechanism paper with a predictive mathematical object.",
        },
        {
            "gap_id": "P1-PracticalMuonBridge",
            "priority": "P1",
            "claim_unblocked": "A practical finite-NS-Muon or Muon-style recipe can be discussed without contradicting the negative final-training pilot.",
            "current_state": "Local Muon-style trajectory-state diagnostics are positive, but the current CIFAR-100-LT final-training finite-NS-Muon pilot is negative.",
            "required_next_evidence": "Tune schedules, matrix-weight treatment, Newton-Schulz depth, learning rate, weight decay, and warmup while sampling matched-head-gain drift probes along the trajectory.",
            "acceptance_gate": "Report tuned final metrics even if negative, plus trajectory-local drift readouts; keep a no-performance-claim path if tuned Muon recipes remain below AdamW or SGD baselines.",
            "compute_mode": "GPU via Slurm",
            "planned_artifacts": "results/e11_cifar100_resnet_practical_muon_tuned/*; figures/e11_cifar100_resnet_practical_muon_tuned/*; discussion/e11_cifar100_resnet_practical_muon_tuned.md",
            "risk_if_missing": "The paper must keep Muon as motivation and local compatibility evidence, not as a demonstrated final optimizer improvement.",
        },
        {
            "gap_id": "P2-NaturalBoundaryCases",
            "priority": "P2",
            "claim_unblocked": "The mechanism has natural positive and negative examples, not only a synthetic sign flip.",
            "current_state": "The natural head-to-tail boundary audit in discussion/e11_natural_head_tail_boundary.md scans committed digits and CIFAR-100-LT ResNet matched-head-gain summaries with a fixed CI rule. It reports no strict natural primary full tail-output drift counterexample in 37 primary rows, while recording component true-logit and secondary loss/margin/accuracy tradeoff candidates as claim-boundary evidence rather than as primary drift failures. The pre-registered fresh natural negative-search protocol in discussion/e11_natural_negative_search_protocol.md freezes the search-space registry, metric contract, multiplicity rule, stopping rule, acceptance gates, and claim ladder. The phase1 runner scripts/e11_run_natural_negative_search_phase1.py, Slurm wrapper scripts/slurm/e11_natural_negative_search_phase1.sbatch, settings registries, multiplicity evaluator scripts/e11_evaluate_natural_negative_search_phase1.py, and detectable-effect audit scripts/e11_write_natural_negative_power_audit.py are implemented. The first fresh phase1 readout in discussion/e11_natural_negative_search_phase1_NNS-P1-cifar100lt-resnet18-new-partitions.md completes the 12 CIFAR-100-LT ResNet18 new-partition settings, the second readout in discussion/e11_natural_negative_search_phase1_NNS-P1-cifar10lt-resnet18-cross-partitions.md completes the 8 CIFAR-10-LT cross-partition settings, and discussion/e11_natural_negative_search_phase1_evaluation.md now reports 20/26 primary metric rows with the completeness and multiplicity gates still not_ready. The interim synthesis in discussion/e11_natural_negative_search_phase1_interim_synthesis.md records raw_worse_rows=0 but blocks both natural-counterexample and finite-null wording because the tail-quality controls are still missing.",
            "required_next_evidence": "Use make e11-natural-negative-search-phase1-power-audit to keep finite-null interpretation tied to a detectable-effect boundary, wait for make e11-natural-negative-search-phase1-results to finish the remaining tail-quality-control phase1 metric files, then rerun make e11-natural-negative-search-phase1-eval so Holm-adjusted decisions preserve every declared setting, null, component-only reversal, and quality failure.",
            "acceptance_gate": "For any found boundary case, include the same drift, tail-loss, margin, rank, JVP, checkpoint-quality, and pre-update tail-quality table used for positive cases, plus adjusted primary-family decision columns and an explicit label for primary full-drift failures versus component or outcome tradeoffs.",
            "compute_mode": "CPU plus optional GPU via Slurm",
            "planned_artifacts": "results/e11_natural_head_tail_boundary/*; results/e11_natural_negative_search_protocol/*; discussion/e11_natural_head_tail_boundary.md; discussion/e11_natural_negative_search_protocol.md; discussion/e11_natural_negative_search_phase1_NNS-P1-cifar100lt-resnet18-new-partitions.md; discussion/e11_natural_negative_search_phase1_NNS-P1-cifar10lt-resnet18-cross-partitions.md; discussion/e11_natural_negative_search_phase1_evaluation.md; discussion/e11_natural_negative_search_phase1_interim_synthesis.md; discussion/e11_natural_negative_search_phase1_power_audit.md; scripts/e11_run_natural_negative_search_phase1.py; scripts/e11_evaluate_natural_negative_search_phase1.py; scripts/e11_write_natural_negative_power_audit.py; scripts/e11_write_natural_negative_phase1_interim_synthesis.py; scripts/slurm/e11_natural_negative_search_phase1.sbatch",
            "risk_if_missing": "The synthetic boundary will look sufficient for theory sanity but thin for empirical falsifiability.",
        },
        {
            "gap_id": "P2-PackagingRepro",
            "priority": "P2",
            "claim_unblocked": "Submission packaging and artifact review can reproduce the exact evidence bundle.",
            "current_state": "The server build works through Tectonic and the root Makefile defaults to the torch-enabled conda Python. The submission reproducibility audit in discussion/e11_submission_repro_audit.md records toolchain availability, rendered PDF hashes, source bundle hashes, and build gates. The preferred pdflatex/bibtex/xelatex clean-checkout gate remains not_ready because those tools are absent on the server.",
            "required_next_evidence": "Run make e11-full after regenerating the submission reproducibility audit, then run a clean-checkout artifact build in a documented environment with pdflatex/bibtex/xelatex and archive exact commands and package versions.",
            "acceptance_gate": "A fresh checkout completes make e11-full; main and two-page PDFs have no missing figures, undefined references, or serious visible layout warnings.",
            "compute_mode": "CPU",
            "planned_artifacts": "discussion/e11_submission_repro_audit.md; results/e11_submission_repro_audit/*",
            "risk_if_missing": "Scientific claims remain unchanged, but artifact-review and camera-ready packaging risk stays higher than necessary.",
        },
    ]


def main() -> None:
    frame = pd.DataFrame(gap_rows())
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(RESULT_PATH, index=False)

    priority_counts = (
        frame.groupby("priority", sort=True)
        .size()
        .reset_index(name="gap_count")
        .sort_values(["priority"])
    )
    text = f"""# E11 Top-Conference Gap Register

This generated register turns the current top-conference plan into concrete, checkable next evidence. It is intentionally stricter than the present paper: the current safe claim is local matched-head-gain tail-example logit drift, not final long-tail accuracy or general Muon optimizer performance.

## Minimum Viable Top-Tier Mechanism Paper

The shortest credible route is to keep the paper as a mechanism paper. `P0-PredictiveCondition` is now a staged held-out claim rather than an open score search: v2/v3/v4 final attempts are spent or failed, while v5 has frozen a transport-normalized validation score and now needs the two unspent final splits. `P0-StandardBenchmark` is only required if the paper wants a benchmark-level performance claim.

## Gap Counts

{markdown_table(priority_counts, ["priority", "gap_count"])}

## Gap Register

{markdown_table(frame, ["gap_id", "priority", "claim_unblocked", "current_state", "required_next_evidence", "acceptance_gate", "compute_mode", "planned_artifacts", "risk_if_missing"])}

## Operating Rule

No row in this register authorizes a stronger paper claim by itself. A row only moves from gap to evidence after the listed artifacts exist, the acceptance gate passes, and `make e11-check` validates the refreshed evidence bundle.
"""
    write_markdown(OUTPUT_PATH, text)
    print(f"saved top-conference gap register to {OUTPUT_PATH} and {RESULT_PATH}")


if __name__ == "__main__":
    main()
