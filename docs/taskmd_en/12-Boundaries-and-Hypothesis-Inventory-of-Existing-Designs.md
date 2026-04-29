<!--
Original Document: Mathematical Foundations and Statistical Formalization of Muon ($\mu$) Optimization Algorithm Experimental Design
This File: 10. Boundaries and Hypothesis Inventory of Existing Experimental Designs
Split Index: 12
-->

[TOC]

---

## 10. Boundaries and Hypothesis Inventory of Existing Designs

This section systematically enumerates all **explicit or implicit assumptions, constraints, and boundary conditions** in this experimental protocol. These boundaries determine the external validity and generalizability of the experimental conclusions.

---

### 6.1 Problem Space Boundaries

| ID | Boundary Condition | Mathematical Description | Implication |
|:---:|:---|:---|:---|
| B1 | Limited dimension range | $d \in \{50, 100, 200, 500\}$, not covering $d < 50$ or $d > 500$ | Conclusions cannot be extrapolated to extremely small or large dimensions |
| B2 | Square matrix assumption | All matrices are $d \times d$ square matrices | Conclusions for rectangular matrices ($m \times n$) require additional verification |
| B3 | Full-rank or low-rank | $r \in \{5, d/2, d\}$, not covering intermediate ranks | Behavior at intermediate ranks $r \in (5, d/2)$ is unknown |
| B4 | Fixed number of measurements | MS problems fix $m = 3d^2$, no oversampling/undersampling sweep | Sample complexity effects cannot be inferred |
| B5 | Limited decomposition depth | $L \in \{2, 3, 4\}$ | Behavior of deeper networks ($L \geq 5$) is unknown |
| B6 | Unconstrained optimization | No equality/inequality constraints | Conclusions cannot be directly generalized to constrained problems |

---

### 6.2 Data Distribution Assumptions

| ID | Assumption | Mathematical Description | Violation Consequence |
|:---:|:---|:---|:---|
| A1 | Gaussian measurement matrix | $A_{ij} \sim \mathcal{N}(0,1)$ iid | Non-Gaussian measurements (e.g., Rademacher) may alter condition number distribution |
| A2 | Gaussian ground-truth matrix | $X^\star_{ij} \sim \mathcal{N}(0,1)$ iid | Structured matrices (e.g., sparse, low-coherence) exhibit different behavior |
| A3 | Additive Gaussian noise | $\epsilon_i \sim \mathcal{N}(0, \sigma^2)$ | Non-Gaussian noise (e.g., heavy-tailed distributions) affects optimal values and convergence |
| A4 | Isotropic initialization | $W_{ij}^{(0)} \sim \mathcal{N}(0, 1/d)$ iid | Non-isotropic or structurally specific initialization may alter convergence |
| A5 | Identically distributed samples | All random variables are identically distributed | Distribution shift leads to extrapolation failure |

---

### 6.3 Algorithm Assumptions

| ID | Assumption | Mathematical Description | Violation Consequence |
|:---:|:---|:---|:---|
| A6 | Full-batch gradient | $G^{(k)} = \nabla f(\theta^{(k)})$, no mini-batch sampling | Stochastic gradient variance in mini-batch SGD may alter relative advantages |
| A7 | Fixed learning rate | $\eta^{(k)} = \eta$ constant | Learning rate schedules (e.g., cosine decay) may alter convergence dynamics |
| A8 | No momentum | Neither Muon nor SGD uses first-order momentum | SGD with momentum or Adam may differ significantly |
| A9 | No gradient clipping | Raw gradients used directly | Gradient explosion scenarios may require clipping |
| A10 | Zero weight decay | $\lambda = 0$ | Non-zero weight decay alters the objective function (adds regularization term) |
| A11 | Exact SVD computation | Muon's SVD is computed exactly | Approximate SVD may introduce numerical errors and additional FLOPs |
| A12 | Single-precision floating point | Computations in float32 precision | Numerical behavior differs in float64 or float16 |

---

### 6.4 Evaluation Metric Assumptions

| ID | Assumption | Mathematical Description | Violation Consequence |
|:---:|:---|:---|:---|
| A13 | Target accuracy attainable | $\exists \, K \leq K_{\max}: f(\theta^{(K)}) - f^\star \leq \epsilon$ | If the algorithm does not converge, the truncation value $K_\epsilon$ artificially introduces bias |
| A14 | Early stopping effective | Early stopping window $w = 100$ can detect true stagnation | A window too small may stop prematurely, too large wastes time |
| A15 | Logarithmic transformation valid | $\log(f - f^\star)$ is well-defined ($f > f^\star$) | If $f$ falls below $f^\star$ (numerical error), the logarithm is undefined |
| A16 | FLOPs model accurate | $C_{\text{Muon}} = C_{\text{SGD}} + C_{\text{SVD}}$ is exact | Actual implementations may have caching, parallelism, and other optimizations that alter true FLOPs |
| A17 | Convergence is monotonic | $f(\theta^{(k+1)}) \leq f(\theta^{(k)})$ | Non-monotonic convergence (e.g., oscillation) complicates early stopping and accuracy detection |

---

### 6.5 Statistical Assumptions

| ID | Assumption | Mathematical Description | Violation Consequence |
|:---:|:---|:---|:---|
| A18 | Paired differences normal | $\Delta_K(s) \overset{iid}{\sim} \mathcal{N}(\mu, \sigma^2)$ approximately holds | Under severe non-normality or small samples, the t-test is unreliable; can be supplemented with non-parametric tests (Wilcoxon) |
| A19 | Independent seeds | Different seeds $s_1 \neq s_2$ produce independent samples | If RNG period is insufficient or correlations exist, standard errors are underestimated |
| A20 | Inter-problem independence | Different problem instances are independent | Problem-shared dimension parameters induce structural correlations |
| A21 | Homogeneous variance | $\mathrm{Var}(\Delta_K)$ is similar across different problems | Under heterogeneous variance, the t-test is biased; partially mitigated by Welch's correction |
| A22 | Significance level appropriate | $\alpha = 0.05$ balances Type I and Type II errors | Stricter $\alpha$ reduces power, more lenient $\alpha$ increases false positives |

---

### 6.6 Generalizability Boundaries of Conclusions

Based on the above boundaries and assumptions, the valid generalization scope of this experimental conclusion is:

$$
\mathcal{G} = \left\{ \text{Problem} \; \middle| \; \begin{array}{l} \text{Square, unconstrained, unregularized,} \\ \text{full-batch gradient, fixed learning rate,} \\ \text{Gaussian measurement/initialization,} \\ \text{dimension } d \in [50, 500] \end{array} \right\}
$$

Any problem beyond $\mathcal{G}$ (e.g., mini-batch training, learning rate scheduling, non-Gaussian data, constrained optimization) requires additional experimental validation and cannot be directly extrapolated from the conclusions of this experiment.

---

## Appendix: Notation Quick Reference

| Symbol | Meaning | Domain |
|:---|:---|:---|
| $\mathcal{E}$ | Experimental quintuple | $(\mathcal{P}, \mathcal{A}, \mathcal{D}, \mathcal{M}, \mathcal{R})$ |
| $\mathcal{P}$ | Problem instance space | $\mathcal{P}_{MS} \cup \mathcal{P}_{MF}$ |
| $\mathcal{A}$ | Algorithm space | $\{\text{Muon}, \text{SGD}\}$ |
| $\mathcal{D}$ | Data/hyperparameter space | $\mathcal{D}_{\text{data}} \times \mathcal{D}_{\text{init}} \times \mathcal{D}_{\text{hyper}}$ |
| $\mathcal{M}$ | Metric space | $\{K_\epsilon, F_\epsilon, \bar{\sigma}_{\log}, \rho_K, \rho_F\}$ |
| $\mathcal{R}$ | Randomness space | Seed $\times$ Problem $\times$ Noise $\times$ Initialization |
| $d$ | Matrix dimension | $\{50, 100, 200, 500\}$ |
| $r$ | Matrix rank | $\{5, d/2, d\}$ |
| $L$ | Decomposition depth | $\{2, 3, 4\}$ |
| $m$ | Number of measurements | $3d^2$ |
| $\eta$ | Learning rate | $\{10^{-3}, 10^{-2}, 10^{-1}\}$ |
| $\lambda$ | Weight decay | $0$ |
| $\epsilon$ | Target accuracy | $10^{-6}$ |
| $K_{\max}$ | Maximum iterations | $10^5$ |
| $R$ | Number of repetitions | $10$ |
| $s$ | Random seed | $\{1, \ldots, 10\}$ |
| $\rho_K$ | Iteration efficiency ratio | $\mathbb{R}^+$ |
| $\rho_F$ | FLOPs efficiency ratio | $\mathbb{R}^+$ |
| $\kappa_{sp}$ | Spectral concentration | $[1, \sqrt{d}]$ |
| $\mathcal{S}(G)$ | SVD normalization operator | $G = U\Sigma V^T \mapsto UV^T$ |
| $\mathcal{T}_{\eta}^{\mathcal{A}}$ | Algorithm iteration operator | $\Theta \to \Theta$ |
| $\Phi$ | Data generation mapping | $\mathcal{R}_{\text{seed}} \times \mathcal{P} \to \mathcal{D}$ |

---

> **This chapter is complete.** The above formal definitions constitute the complete mathematical foundation for the Muon vs. SGD comparative experiment; all subsequent implementation, execution, and inference shall strictly adhere to the notation and definitions in this chapter.

---
