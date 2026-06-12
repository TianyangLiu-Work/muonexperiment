# E11 Theory Note

This generated note records the narrow mathematical statement currently supported by the E11 evidence. It should be treated as the theory bridge for the paper, not as a completed convergence theory.

## Purpose

The empirical story says Muon produces flatter, higher-rank update spectra than Adam. The theory question is when such spectral spreading should improve the one-step decrease. The clean local answer is: **the answer depends on the norm constraint used to compare updates**.

## Local Setup

At a training state, for one matrix parameter block, write the next parameter as
\(W^+=W-D\), where \(D\) is the positive descent update. Then

\[
L(W - D) - L(W)
= -\langle G, D \rangle + O(\|D\|^2),
\]

where \(G = \nabla_W L(W)\). The recorded code metric `update_grad_inner` is \(\langle G,D\rangle\), a positive first-order decrease proxy. The E11 diagnostics are justified because the observed one-step loss decrease is well calibrated by this first-order term:

\[
\rho_\mathrm{Spearman}(\Delta L, \langle G, D\rangle)
= 0.9803,
\quad
95\%\ \mathrm{CI} =
[0.9787, 0.9817].
\]

## Notation

| symbol                         | meaning                                                                                                      |
|:-------------------------------|:-------------------------------------------------------------------------------------------------------------|
| G_i                            | Gradient matrix of layer i at the current step.                                                              |
| \Delta W_i                     | Signed parameter change W_i^+ - W_i after one optimizer step.                                                |
| D_i=-\Delta W_i                | Positive descent update W_i-W_i^+ used by `update_grad_inner`.                                               |
| \langle G_i, D_i\rangle        | Positive first-order loss decrease proxy for layer i.                                                        |
| nr(\Delta W_i), sr(\Delta W_i) | Nuclear-rank and stable-rank diagnostics of the update spectrum.                                             |
| A_i                            | Downstream activation product in MF-with-input only; not needed for the generic update-spectrum proposition. |

## Proposition: Spectral Allocation Boundary

Let \(G = U \operatorname{diag}(\sigma) V^\top\) have rank \(q\) and singular values \(\sigma_1,\ldots,\sigma_q > 0\). For the positive-decrease linearized objective \(\max_D \langle G, D \rangle\):

1. Under the Frobenius ball \(\|D\|_F \le r\), the unique optimizer for \(G \ne 0\) is
   \[
   D_F^\star = rG/\|G\|_F,
   \]
   and the optimal first-order decrease is \(r\|G\|_F\). Its update singular values are proportional to \(\sigma_j\).
2. Under the operator-norm ball \(\|D\|_{op} \le \eta\), an optimizer is
   \[
   D_{op}^\star = \eta U_q V_q^\top,
   \]
   where \(U_q,V_q\) are the singular vectors associated with the nonzero singular values of \(G\). The optimal first-order decrease is \(\eta\|G\|_*\). Its nonzero update singular values are all equal to \(\eta\), which is the flat/polar spectrum.

## Proof Sketch

For the Frobenius ball, Cauchy-Schwarz gives

\[
\langle G, D\rangle
\le \|G\|_F\|D\|_F
\le r\|G\|_F,
\]

with equality at \(D=rG/\|G\|_F\).

For the operator-norm ball, von Neumann's trace inequality gives

\[
\langle G, D\rangle
\le \sum_j \sigma_j(G)\sigma_j(D)
\le \eta\sum_j \sigma_j(G)
= \eta\|G\|_*.
\]

Equality is achieved by aligning the singular vectors and setting the update singular value to \(\eta\) on every nonzero gradient singular direction, giving \(D=\eta U_qV_q^\top\). Thus the two constraints prefer different singular-value allocations: gradient-shaped under Frobenius norm, flat/polar under operator norm. The signed parameter change is \(\Delta W=-D\).

## Consequence For Muon

Muon's polar-like update should not be described as universally better. It is better described as an optimizer that moves toward the operator-norm-constrained spectral allocation. Therefore:

- it can lose against gradient-spectrum updates when the fair comparison is Frobenius-budgeted;
- it can win when the fair comparison is operator-norm-budgeted;
- whether this local advantage converts to loss or accuracy improvement depends on the task/layer geometry and higher-order terms.

## Empirical Check

| constraint                        | linear objective maximizer   | spectral implication                                                   | Muon interpretation                                                     | probe evidence                                     |
|:----------------------------------|:-----------------------------|:-----------------------------------------------------------------------|:------------------------------------------------------------------------|:---------------------------------------------------|
| Fixed Frobenius update budget     | D^\star = rG/\|G\|_F         | Allocate update energy in proportion to the singular values of G.      | A flat/polar update can underuse dominant gradient singular directions. | flat/GD first-order ratio 0.6071 [0.5846, 0.6304]. |
| Fixed operator-norm update budget | D = \eta U_G V_G^\top        | Set every active singular direction to the same update singular value. | A polar-like update is aligned with this constrained optimum.           | flat/GD first-order ratio 1.689 [1.614, 1.768].    |

## Caveats

| point                                               | meaning                                                                                                                         |
|:----------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------|
| Local, not global                                   | The proposition concerns the first-order Taylor term at one state, not final loss or convergence.                               |
| Constraint matters                                  | Muon-like flat spectra are not inherently better; they are better only under operator-norm-like update constraints.             |
| Singular vectors are separated from singular values | The spectral-allocation probe fixes the gradient singular vectors to isolate spectrum allocation.                               |
| Optimizer implementation is richer                  | Real Adam and Muon also differ through state, momentum, scaling, and layer interactions.                                        |
| Operator-norm budget is an idealized comparison     | The proposition explains a local constrained direction; it does not claim Muon explicitly solves this problem in full training. |

## Paper Use

The paper can safely use this note to justify the following precise wording:

> Muon's flat/polar update spectrum matches the update that is locally optimal under an operator-norm update budget, but not under a Frobenius update budget.

The paper should not yet claim:

> Muon's higher update rank generally implies better optimization.

That stronger claim is contradicted by the current boundary experiments.
