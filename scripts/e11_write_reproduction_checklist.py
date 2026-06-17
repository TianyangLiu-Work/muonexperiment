from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.artifacts import APPENDIX_RUNNER_DESCRIPTIONS, APPENDIX_RUNNER_SCRIPTS, MAIN_EVIDENCE_STAGES
from e11_condition_geometry.reporting import markdown_table, write_markdown


OUTPUT_PATH = Path("discussion/e11_reproduction_checklist.md")


def main() -> None:
    main_sequence = pd.DataFrame([dict(item) for item in MAIN_EVIDENCE_STAGES])
    current_stages = {
        "Head-to-tail interference probe",
        "Long-tailed one-step diagnostic",
        "CIFAR-100-LT MLP one-step diagnostic",
        "CIFAR-100-LT ResNet18 one-step diagnostic",
        "CIFAR-100-LT ResNet18 smaller-head-gain check",
        "CIFAR-100-LT ResNet18 checkpoint-quality sweep",
        "CIFAR-100-LT ResNet18 condition-proxy scatter",
        "CIFAR-100-LT ResNet18 final-layer condition scatter",
        "CIFAR-100 ResNet18 tail-quality control",
        "CIFAR-100-LT ResNet18 imbalance sweep",
        "CIFAR-100-LT ResNet18 all-layer JVP tail-quality diagnostic",
        "CIFAR-100-LT ResNet18 NS-Muon final-training pilot",
        "CIFAR-100-LT ResNet18 practical Muon trajectory bridge",
        "Long-tailed Muon-style compatibility diagnostic",
        "Long-tailed practical-Muon trajectory compatibility",
        "Long-tailed Muon state-source control",
        "Long-tailed practical training diagnostic",
        "Long-tailed practical training LR sensitivity",
        "Head-only forgetting probe",
        "Long-tailed layerwise diagnostic",
    }
    current_sequence = main_sequence[main_sequence["stage"].isin(current_stages)].reset_index(drop=True)
    background_sequence = main_sequence[~main_sequence["stage"].isin(current_stages)].reset_index(drop=True)

    appendix_sequence = pd.DataFrame(
        [
            {
                "command": f"python3 {script}",
                "why": APPENDIX_RUNNER_DESCRIPTIONS[script],
            }
            for script in APPENDIX_RUNNER_SCRIPTS
        ]
    )

    paper_artifacts = pd.DataFrame(
        [
            {
                "artifact": "discussion/e11_main_paper_package.md",
                "role": "Smallest main figure/table package.",
            },
            {
                "artifact": "discussion/e11_quantitative_claim_ledger.md",
                "role": "Allowed wording, forbidden wording, quantitative anchors, and evidence links.",
            },
            {
                "artifact": "paper/specgrad_activation_paper/tables/e11_paper_numbers.tex",
                "role": "LaTeX macros generated from the current result CSVs and included by the paper draft.",
            },
            {
                "artifact": "discussion/e11_paper_skeleton.md",
                "role": "Current section-level paper skeleton and figure/table plan.",
            },
            {
                "artifact": "discussion/e11_reviewer_risk_audit.md",
                "role": "Known reviewer risks and safe claim decisions.",
            },
            {
                "artifact": "discussion/e11_matrix_block_theorem_proof.md",
                "role": "Machine-checkable theorem statements, assumptions, proof steps, and claim implications.",
            },
            {
                "artifact": "discussion/e11_matrix_block_tightness_audit.md",
                "role": "Deterministic exact-witness and ratio-identity audit for the matrix-block theorem.",
            },
            {
                "artifact": "discussion/e11_theory_proof_obligation_register.md",
                "role": "Theory-facing proof obligations, assumptions, claim boundaries, and pending gates.",
            },
            {
                "artifact": "discussion/e11_condition_score_v5_reviewer_failure_response.md",
                "role": "Pre-output v5 final failure-mode matrix for reviewer-safe claim downgrades.",
            },
            {
                "artifact": "discussion/e11_natural_negative_search_phase1_NNS-P1-cifar100lt-resnet18-new-partitions.md",
                "role": "Completed 12-setting CIFAR-100-LT phase1 readout for the registered natural-negative search.",
            },
            {
                "artifact": "discussion/e11_natural_negative_search_phase1_NNS-P1-cifar10lt-resnet18-cross-partitions.md",
                "role": "Completed 8-setting CIFAR-10-LT cross-partition phase1 readout for the registered natural-negative search.",
            },
            {
                "artifact": "discussion/e11_natural_negative_search_phase1_evaluation.md",
                "role": "Holm-family evaluator for the complete 26/26 observed primary rows, with finite-null-candidate wording constrained by detectable-effect and quality gates.",
            },
            {
                "artifact": "discussion/e11_natural_negative_search_phase1_interim_synthesis.md",
                "role": "Complete-family claim-boundary synthesis for raw-worse counts, quality gates, and finite registered phase1 null-candidate caveats.",
            },
            {
                "artifact": "discussion/e11_natural_negative_search_phase2_NNS-P2-heldout-architecture-boundary.md",
                "role": "Settings-only ResNet34 held-out architecture registry for phase2, with no phase2 metric rows used before Slurm completion.",
            },
            {
                "artifact": "discussion/e11_natural_negative_search_phase2_evaluation.md",
                "role": "Pre-output Holm evaluator for the 8-setting phase2 held-out architecture family, currently not_ready until metric rows exist.",
            },
            {
                "artifact": "discussion/e11_natural_negative_search_phase2_power_audit.md",
                "role": "Pre-output detectable-effect and outcome-state audit for the 3-seed, 8-setting ResNet34 phase2 family.",
            },
            {
                "artifact": "discussion/e11_top_conference_claim_decision_audit.md",
                "role": "Paper-level supportable/registered-not-ready/blocked claim and rebuttal-readiness contract for top-conference wording.",
            },
            {
                "artifact": "discussion/e11_manuscript_claim_trace.md",
                "role": "Main-tex claim trace that maps every top-conference claim decision to anchors and checks blocked positive wording is absent.",
            },
            {
                "artifact": "discussion/e11_artifact_review_packet.md",
                "role": "Artifact-review command, gate, local-state, and reviewer-response packet for reproducing the current bundle without expanding claims.",
            },
            {
                "artifact": "discussion/e11_mechanism_referee_audit.md",
                "role": "Adversarial alternative-explanation, theory-measurement, and falsification-trigger audit for top-conference reviewer scrutiny.",
            },
        ]
    )

    text = f"""# E11 Reproduction Checklist

This generated checklist separates the minimal main-paper evidence from appendix controls and generated writing assets. It is not a new experiment; it is the current reproducibility map for the E11 paper direction.

## Make Targets

```bash
make e11-main-results      # core trajectories, equal-update, head-to-tail probes, and spectral-allocation
make e11-cifar-results     # CIFAR-100-LT MLP local matched-head-gain diagnostic
make e11-cifar-resnet-results # submit the CIFAR-100-LT ResNet18 GPU diagnostic via Slurm
make e11-cifar-resnet-rho002-results # submit the CIFAR-100-LT ResNet18 rho=0.002 robustness check via Slurm
make e11-cifar-resnet-checkpoint-sweep-results # submit the top-conference ResNet checkpoint-quality sweep via Slurm
make e11-cifar-resnet-condition-proxy-results # regenerate the ResNet rank-proxy scatter from checkpoint-sweep CSVs
make e11-cifar-resnet-fc-condition-results # submit the ResNet final-layer downstream-aware condition diagnostic via Slurm
make e11-cifar-resnet-tail-quality-results # submit the tail-rich ResNet checkpoint-quality control via Slurm
make e11-cifar-resnet-imbalance-sweep-results # submit the CIFAR-100-LT ResNet18 tail-count imbalance sweep via Slurm
make e11-cifar-resnet-layer-jvp-tail-quality-results # submit the all-layer ResNet finite-difference JVP tail-quality diagnostic via Slurm
make e11-cifar-resnet-condition-score-v4-validation-freeze # freeze or block the v4 scalar aggregation after the validation split
make e11-cifar-resnet-condition-score-v4-final-eval # evaluate frozen v4 final gates after both unspent final Slurm jobs finish
make e11-matrix-block-theorem-proof # write the matched-gain theorem/proof contract and sandwich rank derivation
make e11-matrix-block-tightness-audit # verify theorem tightness, ratio identity, equality boundary, and degeneracy caveats
make e11-theory-proof-obligation-register # map theorem assumptions, proof obligations, claim scope, and blocked wording
make e11-cifar-resnet-condition-score-v5-final-eval # evaluate frozen v5 final gates after both unspent final Slurm jobs finish
make e11-cifar-resnet-condition-score-v5-final-interpretation-plan # lock the v5 final outcome-to-claim state machine before outputs exist
make e11-cifar-resnet-condition-score-v5-reviewer-failure-response # map v5 final pass/fail modes to reviewer-safe claim downgrades
make e11-cifar-resnet-lt-muon-final-benchmark-results # submit the CIFAR-100-LT ResNet18 NS-Muon final-training benchmark pilot via Slurm
make e11-cifar-resnet-lt-tuned-benchmark-protocol # register validation/final splits and benchmark claim gates
make e11-cifar-resnet-lt-tuned-benchmark-settings # write the executable tuned validation grid registry
make e11-cifar-resnet-lt-tuned-benchmark-validation-results # submit the tuned validation grid via Slurm array
make e11-cifar-resnet-lt-tuned-benchmark-selection # select final recipes from completed validation summaries
make e11-cifar-resnet-practical-muon-bridge-results # submit the ResNet practical Muon/AdamW trajectory bridge via Slurm
make e11-natural-negative-search-phase1-power-audit # keep the 26-setting natural-negative detectable-effect boundary current
make e11-natural-negative-search-phase1-results # submit the registered phase1 natural-negative GPU jobs via Slurm
make e11-natural-negative-search-phase1-eval # refresh the 26-setting Holm-adjusted primary decision table
make e11-natural-negative-search-phase1-interim-synthesis # summarize complete phase1 coverage and caveated finite-null boundaries
make e11-natural-negative-search-phase2-settings # freeze the ResNet34 held-out architecture phase2 registry
make e11-natural-negative-search-phase2-results # submit the registered ResNet34 phase2 jobs via Slurm
make e11-natural-negative-search-phase2-eval # refresh the 8-setting phase2 Holm-adjusted primary decision table
make e11-natural-negative-search-phase2-power-audit # keep the phase2 detectable-effect and outcome-state boundary current
make e11-appendix-results  # current appendix/guardrail probes
make e11-all-results       # main plus appendix/guardrail result generation
make e11-paper-assets      # current head-to-tail Markdown/TeX paper assets
make e11-top-conference-claim-decision-audit # regenerate the supportable/blocked paper claim and rebuttal-readiness contract
make e11-mechanism-referee-audit # regenerate the adversarial mechanism/referee alternative-explanation audit
make e11-guardrail-assets  # legacy condition-geometry guardrail notes
make e11-all-assets        # current paper assets plus legacy guardrail notes
make e11-paper-pdf         # rebuild paper/specgrad_activation_paper/main.pdf and two_page.pdf
make e11-check             # validation, tests, and whitespace check
make e11-full              # paper assets, PDF build, and e11-check
make e11-artifact-review-packet # regenerate the artifact-review command, gate, local-state, and reviewer-response packet
```

## Current Head-to-Tail Paper Evidence

{markdown_table(current_sequence, ["stage", "command", "produces", "paper_role"])}

## Background / Legacy E11 Evidence

These artifacts remain reproducible because they document the route to the
current head-to-tail framing and provide guardrails against broader optimizer
claims. They are not the main evidence table for the current paper draft.

{markdown_table(background_sequence, ["stage", "command", "produces", "paper_role"])}

## Appendix / Guardrail Evidence

{markdown_table(appendix_sequence, ["command", "why"])}

## Generated Paper-Facing Assets

After the result CSVs and figures exist, regenerate the current head-to-tail paper assets with:

```bash
make e11-paper-assets
```

Regenerate legacy condition-geometry guardrail notes separately with:

```bash
make e11-guardrail-assets
```

{markdown_table(paper_artifacts, ["artifact", "role"])}

## Batch / Activation Contract

The validation gate enforces the current training-versus-diagnostics contract:

- Default core E11 optimization uses noisy mini-batches, so `train_batch_size < num_samples` and `noise_std > 0`.
- Activation diagnostics use the full sampled problem instance. MLP diagnostics must record `diagnostic_A_definition == full_layer_input_activation`.
- `delta_loss` is the same-batch pre/post-update decrease, evaluated on the training batch used for the gradient/update.

## Validation Gate

Before using the numbers in a draft, run:

```bash
make e11-check
```

The stronger local gate is:

```bash
make e11-full
```

`make e11-full` regenerates current paper-facing artifacts, rebuilds the paper PDF, and then runs validation, tests, and whitespace checks. It assumes the longer experiment result CSVs already exist unless their writer script reruns the relevant probe. Legacy guardrail notes are intentionally not part of `e11-full`; use `make e11-guardrail-assets` when those background notes need refreshing.
"""
    write_markdown(OUTPUT_PATH, text)
    print(f"saved reproduction checklist to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
