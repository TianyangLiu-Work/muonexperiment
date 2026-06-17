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
            "current_state": "The checkpoint-transfer benchmark is a boundary result, the condition-score protocol is pre-registered in discussion/e11_cifar100_resnet_condition_score_protocol.md, the locked ResNet18 retrospective v2 split passes in discussion/e11_cifar100_resnet_condition_score_next.md, and the registered held-out ResNet34/CIFAR-10-LT evaluation now fails the residual-ranking gates in discussion/e11_cifar100_resnet_condition_score_next_heldout_evaluation.md.",
            "required_next_evidence": "Treat the frozen v2 held-out failure as an obstruction, not as a tuning target: derive or pre-register a new theory-linked score revision, then evaluate it on fresh held-out splits that were not used to diagnose this failure.",
            "acceptance_gate": "For a future protocol revision, report positive held-out residual Spearman with a CI lower endpoint above zero on fresh architecture and data splits, threshold accuracy above 0.8, explicit early-layer and source-observed controls, and a statement that the failed v2 held-outs were not used for score fitting.",
            "compute_mode": "GPU via Slurm",
            "planned_artifacts": "results/e11_cifar100_resnet_condition_score_protocol/*; results/e11_cifar100_resnet_condition_score_next/*; figures/e11_cifar100_resnet_condition_score_next/*; discussion/e11_cifar100_resnet_condition_score_next.md; discussion/e11_cifar100_resnet_condition_score_next_heldout_evaluation.md",
            "risk_if_missing": "The paper can still claim a local drift mechanism, but it cannot claim the current condition score is predictive on unseen real tasks.",
        },
        {
            "gap_id": "P0-StandardBenchmark",
            "priority": "P0",
            "claim_unblocked": "Any benchmark-level statement about final long-tail performance or practical optimizer quality.",
            "current_state": "The repo has a CIFAR-100-LT IF=100 AdamW reporting baseline, a 5-seed augmented recipe pilot, and a negative 3-seed finite-NS-Muon final-training pilot.",
            "required_next_evidence": "Run a tuned multi-optimizer benchmark with augmentation, schedules, weight decay grids, class-balanced losses or samplers, and final many/medium/few metrics; extend beyond CIFAR-100-LT if the paper wants broad benchmark scope.",
            "acceptance_gate": "Separate local-drift and final-performance claims; for any performance claim, report paired seeds, tuned baselines, all/few/many metrics, and confidence intervals with no post-hoc recipe selection.",
            "compute_mode": "GPU via Slurm",
            "planned_artifacts": "results/e11_cifar100_resnet_lt_tuned_benchmark/*; figures/e11_cifar100_resnet_lt_tuned_benchmark/*; discussion/e11_cifar100_resnet_lt_tuned_benchmark.md",
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
            "current_state": "The synthetic sandwich-block condition gives the right sign boundary, and discussion/e11_condition_score_theory_bridge.md now separates the supported held-out threshold-direction guardrail from the failed residual-ranking target.",
            "required_next_evidence": "Derive or state a computable downstream-aware score that reduces to the theorem quantities in the linear case, is estimable with finite-difference or autodiff JVPs in networks, and is frozen before fresh held-out architecture/data splits.",
            "acceptance_gate": "Add a theorem, proposition, or falsifiable conjecture plus an ablation showing which term makes the score fail or succeed on the checkpoint-transfer benchmark; fresh held-out splits must not reuse the failed v2 splits for tuning.",
            "compute_mode": "CPU plus optional GPU via Slurm",
            "planned_artifacts": "discussion/e11_condition_score_theory_bridge.md; results/e11_condition_score_ablation/*; figures/e11_condition_score_ablation/*",
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
            "current_state": "The synthetic negative case is clean, but the natural-task evidence mostly documents where spectral/polar lowers local drift and where score candidates fail to rank risk.",
            "required_next_evidence": "Search pre-registered natural settings where the matched-head-gain spectral/polar direction is worse, or report a null search with the same metrics and stopping rule.",
            "acceptance_gate": "For any found boundary case, include the same drift, tail-loss, margin, rank, JVP, and checkpoint-quality table used for positive cases.",
            "compute_mode": "CPU plus optional GPU via Slurm",
            "planned_artifacts": "results/e11_natural_head_tail_boundary/*; figures/e11_natural_head_tail_boundary/*; discussion/e11_natural_head_tail_boundary.md",
            "risk_if_missing": "The synthetic boundary will look sufficient for theory sanity but thin for empirical falsifiability.",
        },
        {
            "gap_id": "P2-PackagingRepro",
            "priority": "P2",
            "claim_unblocked": "Submission packaging and artifact review can reproduce the exact evidence bundle.",
            "current_state": "The server build works through Tectonic and the root Makefile defaults to the torch-enabled conda Python; pdflatex/xelatex remains unverified because those tools are absent on the server.",
            "required_next_evidence": "Run a clean-checkout artifact build in a documented environment with pdflatex or xelatex, then archive exact commands and package versions.",
            "acceptance_gate": "A fresh checkout completes make e11-full; main and two-page PDFs have no missing figures, undefined references, or serious visible layout warnings.",
            "compute_mode": "CPU",
            "planned_artifacts": "discussion/e11_submission_repro_audit.md; outputs/e11_submission_repro_audit/*",
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

The shortest credible route is to keep the paper as a mechanism paper. `P0-PredictiveCondition` is now a negative boundary rather than an open run: the frozen v2 score passed the retrospective checkpoint split but failed the registered held-out residual-ranking gates. A top-tier predictive-condition claim needs a new theory-linked score revision plus fresh held-out splits. `P0-StandardBenchmark` is only required if the paper wants a benchmark-level performance claim.

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
