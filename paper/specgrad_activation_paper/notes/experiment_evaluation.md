# Experiment Evaluation for the SpecGrad Activation Paper

This note evaluates what experiments are needed before the current paper draft
can be treated as a project paper rather than a theory-oriented proposal.

## Current Status

The paper draft has a coherent framing:

> SpecGrad / Muon-like updates reshape matrix-update spectra, but whether this
> helps optimization depends on norm geometry, activation geometry, and
> downstream sensitivity.

The existing E11 evidence supports the update-spectrum and local first-order
parts of this framing:

- Muon-like updates have larger update nuclear-rank and stable-rank metrics than Adam.
- One-step loss decrease is strongly correlated with the first-order proxy
  `<G, D>`.
- Flat/polar spectral allocation wins under an operator-norm budget and loses
  under a Frobenius budget.
- Neural negative controls show that higher update rank does not automatically
  imply better one-step progress.

The direct activation-perturbation diagnostic has now been added for the
activation-defined E11 tasks. The remaining missing part is downstream-aware
evidence: the current draft should not claim that SpecGrad improves downstream
stability or long-tail behavior until the JVP/downstream and long-tail
experiments below are run.

## Required Experiments

### 1. Direct Activation Perturbation Diagnostics

**Claim tested.** Polar / SpecGrad updates naturally control per-sample or
operator-norm activation perturbation, but not necessarily batch-Frobenius
activation movement.

**Status.** Initial version complete for MF-with-input and Small MLP. Matrix
Sensing is excluded because its diagnostic `A` is a measurement-operator proxy,
not a layer activation matrix.

**Current result.** Under equal-update control, Muon/Adam ratios are below one
for relative operator perturbation and max per-sample perturbation, but not for
batch-Frobenius perturbation. See:

- `discussion/e11_activation_perturbation.md`
- `results/e11_equal_update/activation_perturbation_summary.csv`
- `paper/specgrad_activation_paper/tables/e11_activation_perturbation.tex`

**Tasks.**

- MF-with-input.
- Small MLP / MNIST MLP.
- MNIST patch or ConvNet only after the MLP version works.

**Protocol.**

At each optimizer step and each matrix layer, record the actual update
`D_i = W_i^t - W_i^{t+1}` and the full-diagnostic activation matrix `A_i`.
For neural tasks, training should use small mini-batches, but activation
diagnostics should be computed on the full sampled dataset for that problem
instance.

Record:

- `||D_i A_i||_F / ||W_i A_i||_F`
- `||D_i A_i||_op / ||W_i A_i||_op`
- `max_b ||D_i a_b||_2 / ||a_b||_2`
- `mean_b ||D_i a_b||_2 / ||a_b||_2`
- update metrics `||D_i||_F`, `||D_i||_op`, `nr(D_i)`, `sr(D_i)`
- observed `delta_loss` on the training batch and evaluation loss on the full
  diagnostic set when available

**Comparison.**

Use paired Adam vs Muon runs from identical initialization and data order.
Also evaluate local direction probes at the same state:

- GD-shaped direction under matched Frobenius budget.
- Polar direction under matched Frobenius budget.
- GD-shaped direction under matched operator budget.
- Polar direction under matched operator budget.

**Required statistics.**

- Paired mean/median differences with bootstrap confidence intervals.
- Per-task and per-layer summaries, not only pooled summaries.
- Correlation between activation perturbation metrics and observed `delta_loss`.

**Decision rule.**

This experiment supports the activation framing only if the operator/per-sample
perturbation metrics separate from batch-Frobenius metrics in the predicted
direction. If Muon has higher update rank but no consistent operator/per-sample
perturbation pattern, then the activation framing should be presented as a
theoretical interpretation rather than an empirical finding.

The current result passes this initial decision rule for operator and worst-case
per-sample perturbation, with the important caveat that batch-Frobenius and mean
per-sample perturbation remain task dependent.

### 2. Layerwise Condition Predictor: `nr(G_i)` vs `sr(A_i)`

**Claim tested.** The Davis-Drusvyatskiy style condition
`nr(G_i) > sr(A_i)` predicts when a polar / SpecGrad direction has larger
local one-step progress than a GD-shaped direction.

**Tasks.**

- MF-with-input across kappa values.
- Matrix Sensing.
- Small MLP with at least two widths.
- MNIST MLP as a neural sanity check.

**Protocol.**

At the same parameter state, compute for each layer:

- `G_i`, the parameter gradient.
- `A_i`, the full-diagnostic activation matrix.
- `nr(G_i) = ||G_i||_*^2 / ||G_i||_F^2`.
- `sr(A_i) = ||A_i||_F^2 / ||A_i||_op^2`.
- `condition_margin_i = log(nr(G_i)) - log(sr(A_i))`.
- GD-shaped and polar one-step predicted decrease under the paper's matched
  assumptions.
- Observed one-step `delta_loss` for small finite steps.

**Required statistics.**

- Accuracy/AUC of `condition_margin_i > 0` for predicting polar advantage.
- Spearman correlation between `condition_margin_i` and polar/GD advantage.
- Leave-setting-out validation: hold out a task family or kappa value.

**Decision rule.**

This can become a paper result only if the condition has predictive value out of
sample. If it works only in pooled in-sample plots, it should be described as a
diagnostic rather than a predictive law.

### 3. Downstream-Aware Perturbation / Sandwiched Stable Rank

**Claim tested.** `sr(A_i)` is only the `B = I` special case; the more relevant
quantity is downstream sensitivity through `B_i D_i A_i` or `J_i D_i`.

**Tasks.**

- Start with small MLP, because downstream Jacobian estimation is manageable.
- Then repeat on MNIST MLP or patch classifier if the result is interpretable.

**Protocol.**

For each layer and state, compare GD-shaped and polar directions using:

```text
score(D_i) = <G_i, D_i>^2 / ||J_i D_i||_F^2
```

where `J_i D_i` is estimated by JVP/VJP over the full diagnostic set.

Also compute an empirical downstream sensitivity ratio:

```text
downstream_ratio_i = ||J_i D_polar||_F / ||J_i D_gd||_F
```

If feasible, estimate the spectra of the activation matrix and downstream
Jacobian surrogate to approximate the sandwiched stable rank.

**Required statistics.**

- Paired polar/GD score ratio with confidence intervals.
- Whether downstream-aware score predicts observed one-step loss decrease
  better than `nr(G_i) > sr(A_i)`.
- Separate layerwise results; avoid only reporting whole-network averages.

**Decision rule.**

This is the most important experiment for making the current paper novel. If
downstream-aware metrics improve prediction over plain `sr(A_i)`, the paper can
claim a genuine extension of the existing layerwise condition. If not, the
sandwiched stable-rank section should be framed as a theoretical refinement
whose empirical value is not established yet.

### 4. Mini-Batch Training vs Full-Dataset Diagnostics

**Claim tested.** The optimizer may train on small batches while the geometry
diagnostics describe the full sampled problem instance.

**Tasks.**

- Small MLP / MNIST MLP.
- Optional: long-tail synthetic classification after the basic version works.

**Protocol.**

Train with small batch sizes. At each recorded step:

- compute gradients and updates from the mini-batch;
- compute `A_i`, `D_i A_i`, and downstream diagnostics on the full sampled
  dataset;
- record whether mini-batch geometry is aligned with full-dataset geometry.

**Required statistics.**

- Batch-size sweep, for example 16, 32, 64, 128.
- Correlation between mini-batch condition scores and full-dataset condition
  scores.
- Variance / instability of `nr(G_i)`, `sr(A_i)`, and activation perturbation
  metrics across batches.

**Decision rule.**

This experiment is needed if the paper discusses practical neural training.
Without it, the paper should limit neural claims to controlled diagnostics.

## Recommended But Not Required Yet

### 5. Long-Tail / Rare-Direction Probe

Run this only if the paper wants to claim anything about rare directions or
long-tail behavior. The draft currently mentions this, but E11 does not yet
support it.

Record tail-class loss, tail activation drift, tail activation coverage, and
whether Muon's high-rank update spectrum improves rare-direction movement.

This is high-risk because it can easily become a separate project.

### 6. Modern Architecture Sanity Check

A small Transformer or modern CNN probe is useful only after the activation and
downstream metrics are already validated on simpler MLP tasks. Otherwise it will
add complexity without clarifying the mechanism.

Use it as an external sanity check, not as the main evidence.

## Experiments To Avoid For Now

- Broad optimizer leaderboards.
- More rotating 3D videos or GIFs.
- Larger hyperparameter searches unless tied to a specific confound.
- Adding more problem families before the activation/downstream metrics are
  validated on the current tasks.

These would increase the artifact surface without addressing the current paper
gap.

## Suggested Immediate Work Order

1. Add local GD-vs-polar direction probes at matched Frobenius and operator
   budgets.
2. Generate tables for condition predictor
   accuracy.
3. Add downstream JVP diagnostics for the small MLP.
4. Reassess whether the paper's title should emphasize activation geometry or
   stay with the safer update-spectrum geometry framing.

## Current Paper Claim After These Experiments

The paper can become a strong project paper if it can quantitatively support:

> Muon-like updates consistently reshape update spectra; this reshaping changes
> activation perturbation geometry, and its local optimization value is
> predictable only after conditioning on activation and downstream sensitivity.

If the activation/downstream experiments are weak, the safer paper is:

> Muon-like updates are update-spectrum shaping methods whose local benefit is
> governed by norm-budget geometry; higher update rank alone is not a reliable
> explanation for optimization progress.
