# E11 End-of-Draft Self-Review

This self-review follows the research-paper-writing checklist for the current
head-to-tail interference manuscript. It is intentionally conservative: a row
passes only when the current paper state has direct evidence.

## Five-Dimension Review

| dimension | question | current answer | status | required discipline |
|---|---|---|---|---|
| Contribution | What new knowledge does the paper give? | It formulates a matched-head-gain head-to-tail function-drift diagnostic and derives the sandwich-block condition `nrank(G_H) > ssrank(B_T,A_T)`. | pass for mechanism paper | Do not present it as a new long-tailed training method. |
| Contribution | Is the insight non-obvious beyond known spectral-gradient theory? | The paper transfers same-batch spectral-gradient geometry into a cross-distribution head-to-tail drift question and adds downstream-aware `ssrank(B_T,A_T)`. | pass with scope | Keep Related Work clear that this is a different local question, not a replacement for existing long-tail methods. |
| Writing clarity | Can a reader reproduce the diagnostics? | Appendix I specifies splits, seeds, MLP architecture, batch/noise settings, matched-gain scaling, Muon-style NS details, and CI computation. | pass | Keep generated artifact and reproduction checklist in sync with the paper. |
| Writing clarity | Are terms stable? | `G_H`, `A_T`, `B_T`, `J_T`, `nrank`, `srank`, `ssrank`, squared drift ratio, and RMS drift ratio are separately defined. | pass | Do not shorten squared drift ratio to generic drift ratio in paper-facing files. |
| Experimental strength | Are the empirical effects meaningful? | One-step squared drift ratio is `0.5501 [0.5101, 0.5931]`; 8-step final squared drift ratio is `0.6167 [0.5744, 0.6622]`; trajectory NS ratio is `0.8019 [0.7583, 0.848]`. | pass for diagnostics | State these as local drift effects, not benchmark performance. |
| Experimental strength | Are final-performance claims supported? | No. One-step tail CE worsens slightly for spectral/polar, tail accuracy does not improve in the practical diagnostic, and long-tail benchmarks are absent. | not supported | The main paper must keep performance caveats adjacent to drift claims. |
| Evaluation completeness | Are important ablations present? | The paper includes boundary reversal, imbalance ablation, local linearization error, seed-clustered trajectory CI, layerwise downstream-aware diagnostics, margin certificate, and LR sensitivity. | pass for current scope | A stronger empirical paper still needs CIFAR-100-LT/ImageNet-LT-style benchmarks. |
| Evaluation completeness | Are strong baselines included? | For local matched-gain diagnostics the main comparison is Fro/GD vs spectral/polar/Muon-style directions; for practical training the comparison is Adam vs NS-Muon-style. | partial | Do not claim superiority over the full optimizer landscape. |
| Method design soundness | Is the matched-gain protocol technically sound? | The paper defines both steepest-direction and arbitrary-direction matched-gain coefficients and uses actual head alignment for `polar(M_t)` and `NS(M_t)`. The Fro/GD state-source control checks that the short-trajectory Muon-style readout is not unique to NS-Muon-generated states. | pass | Keep Muon-style results labeled as selected-state compatibility checks. |
| Method design soundness | Does the method depend on unrealistic assumptions? | The theory uses a local first-order approximation and a sandwich matrix block; experiments report local linearization error and explain first-layer ReLU gating through a frozen-gate local operator. | pass with limitation | Larger nonlinear architectures require additional layerwise operator diagnostics. |

## Claim-Evidence Map

| major claim | evidence | status |
|---|---|---|
| Spectral/polar geometry can reduce matched-head-gain squared tail-example logit drift in the tested settings. | One-step digits, 8-step forgetting, synthetic boundary, and imbalance ablation summaries. | supported |
| The matrix-block mechanism boundary is `nrank(G_H) > ssrank(B_T,A_T)`. | Theorem derivation plus positive/negative synthetic boundary reversal. | supported for the controlled sandwich block |
| Muon-style directions are locally compatible with the polar mechanism in selected states. | Fixed-checkpoint `polar(M_t)` squared drift ratio, short-trajectory NS-Muon-style diagnostics, and Fro/GD state-source control. | supported as compatibility only |
| Spectral/polar directions are intrinsically less tail-sensitive layerwise. | Layerwise unit-JVP squared drift ratios are above one in both layers. | contradicted |
| Lower tail-example logit drift implies lower tail CE, better margin, or higher accuracy. | One-step tail CE caveat and practical training tail-accuracy difference. | not supported |
| The paper establishes a general Muon long-tail optimizer advantage. | No standard long-tailed benchmark or tuned practical optimizer comparison. | not supported |

## Required Manuscript Discipline

1. The main paper should lead with the local function-drift mechanism.
2. Every empirical drift claim should specify that it is matched-head-gain squared tail-example logit drift.
3. Muon should remain motivation plus selected-state compatibility evidence unless larger practical benchmarks are added; the Fro/GD state-source control reduces but does not remove this boundary.
4. Tail loss, margin, and accuracy should be reported as separate outcomes, not consequences of the drift theorem.
5. Older condition-geometry artifacts should remain guardrails or appendix/background evidence, not part of the current paper's core claim.

## Current Residual Risks

| risk | why it remains | how to resolve |
|---|---|---|
| Dataset scale | The core real-data evidence is controlled scikit-learn digits. | Add CIFAR-100-LT, ImageNet-LT, or iNaturalist-style matched-gain diagnostics. |
| Optimizer-level Muon claim | Current Muon evidence is selected-state and small practical training only. | Run tuned practical Muon vs tuned baselines on standard long-tail tasks and sample matched-gain diagnostics along the trajectory. |
| Architecture generality | Layerwise diagnostics are for a two-layer MLP. | Add deeper ConvNet/Transformer-style layerwise `nrank`, downstream-aware rank, JVP, scaled JVP, and observed drift analysis. |
| Boundary predictiveness | The clean boundary result is constructed. | Test a pre-specified boundary predictor on held-out task/architecture families. |
