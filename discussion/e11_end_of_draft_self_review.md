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
| Experimental strength | Are final-performance claims supported? | No. One-step tail CE worsens slightly for spectral/polar, tail accuracy does not improve in the practical diagnostic, and the CIFAR-100-LT NS-Muon final-training pilot is negative versus augmented AdamW. | not supported | The main paper must keep performance caveats adjacent to drift claims and use CIFAR-100-LT benchmark pilots only as context. |
| Evaluation completeness | Are important ablations present? | The paper includes boundary reversal, imbalance ablation, local linearization error, seed-clustered trajectory CI, class-partition and rho sweeps, tail-rich ResNet controls, all-layer ResNet JVP diagnostics, condition-score failure audits, margin readouts, and LR sensitivity. | pass for current scope | A stronger empirical paper still needs tuned validation/final benchmark grids and larger long-tail datasets. |
| Evaluation completeness | Are strong baselines included? | Local matched-gain diagnostics compare Fro/GD against spectral/polar/Muon-style directions; benchmark context includes AdamW IF=100 reporting, augmented AdamW, class-balanced AdamW, SGD-momentum, and a negative finite-NS-Muon pilot. | partial for optimizer claims | Do not claim superiority over the full optimizer landscape until the registered tuned validation and untouched final-seed protocol completes. |
| Method design soundness | Is the matched-gain protocol technically sound? | The paper defines both steepest-direction and arbitrary-direction matched-gain coefficients and uses actual head alignment for `polar(M_t)` and `NS(M_t)`. The Fro/GD state-source control checks that the short-trajectory Muon-style readout is not unique to NS-Muon-generated states. | pass | Keep Muon-style results labeled as selected-state compatibility checks. |
| Method design soundness | Does the method depend on unrealistic assumptions? | The theory uses a local first-order approximation and a sandwich matrix block; experiments report local linearization error and explain first-layer ReLU gating through a frozen-gate local operator. | pass with limitation | Larger nonlinear architectures require additional layerwise operator diagnostics. |

## Claim-Evidence Map

| major claim | evidence | status |
|---|---|---|
| Spectral/polar geometry can reduce matched-head-gain squared tail-example logit drift in the tested settings. | One-step digits, 8-step forgetting, synthetic boundary, CIFAR-100-LT ResNet18 one-step/checkpoint/tail-quality/imbalance controls, and all-layer ResNet JVP summaries. | supported |
| The matrix-block mechanism boundary is `nrank(G_H) > ssrank(B_T,A_T)`. | Theorem derivation plus positive/negative synthetic boundary reversal. | supported for the controlled sandwich block |
| Muon-style directions are locally compatible with the polar mechanism in selected states. | Fixed-checkpoint `polar(M_t)` squared drift ratio, short-trajectory NS-Muon-style diagnostics, and Fro/GD state-source control. | supported as compatibility only |
| Spectral/polar directions are intrinsically less tail-sensitive layerwise. | Layerwise unit-JVP squared drift ratios are above one in both layers. | contradicted |
| Lower tail-example logit drift implies lower tail CE, better margin, or higher accuracy. | One-step tail CE caveat and practical training tail-accuracy difference. | not supported |
| The paper establishes a general Muon long-tail optimizer advantage. | The current standard/recipe/NS-Muon CIFAR-100-LT pilots are benchmark context, and the direct NS-Muon final-training pilot is negative; the registered tuned benchmark protocol is still not_ready. | not supported |

## Required Manuscript Discipline

1. The main paper should lead with the local function-drift mechanism.
2. Every empirical drift claim should specify that it is matched-head-gain squared tail-example logit drift.
3. Muon should remain motivation plus selected-state compatibility evidence unless the registered tuned benchmark protocol completes; the Fro/GD state-source control and ResNet practical bridge reduce but do not remove this boundary.
4. Tail loss, margin, and accuracy should be reported as separate outcomes, not consequences of the drift theorem.
5. Older condition-geometry artifacts should remain guardrails or appendix/background evidence, not part of the current paper's core claim.

## Current Residual Risks

| risk | why it remains | how to resolve |
|---|---|---|
| Dataset scale | CIFAR-100-LT ResNet18 diagnostics, tail-rich controls, and benchmark pilots are now present, but ImageNet-LT/iNaturalist-style evidence is still absent. | Add larger long-tail datasets under a pre-registered matched-gain and benchmark protocol. |
| Optimizer-level Muon claim | Current Muon evidence is selected-state/local plus a negative CIFAR-100-LT NS-Muon final-training pilot. | Complete the registered tuned benchmark validation grid, select recipes on validation seeds only, then run untouched final paired seeds. |
| Architecture generality | All-layer ResNet18 JVP diagnostics are present, and v5 ResNeXt50-32x4d and CIFAR-10 cross-partition final condition-score outputs are complete and failed the P0 gate family. | Preserve the completed v5 negative boundary and open a new protocol before any repaired architecture/data transport claim. |
| Boundary predictiveness | The synthetic boundary is controlled and the current condition-score program remains not_ready because completed v5 final gates failed. | Treat the frozen v5 score as a negative boundary, not a predictor; any renewed claim needs new unspent validation/final splits. |
