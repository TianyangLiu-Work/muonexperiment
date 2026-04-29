<!--
Original Document: Mathematical Foundations and Statistical Formalization of Experimental Design for the Muon (μ) Optimization Algorithm
This Document: 18. Systematic Review of Missing Dimensions
Split Number: 20
-->

[TOC]

---

# Part IV: Identification and Supplementation of Missing Experiments

> This section presents a systematic blind-spot review of existing Muon vs. SGD experimental designs, identifying over 40 gaps across 8 dimensions, and designs 15 immediately executable supplementary experiments (numbered E6–E20). Each supplementary experiment is equipped with complete mathematical formalism, statistical hypotheses, test statistics, and expected interpretations. This section builds upon the formalization framework in Part II and the statistical variable system in Part III, extending the experimental coverage from basic configurations to the full algorithm–problem–environment parameter space.

> **Document Positioning**: Stage 5 — a systematic blind-spot review of existing Muon vs. SGD experimental designs, identifying over 40 gaps across 8 dimensions, and designing 15 immediately executable supplementary experiments (numbered E6–E20). Each supplementary experiment is equipped with complete mathematical formalism, statistical hypotheses, test statistics, and expected interpretations.

---

## 18. Systematic Review of Missing Dimensions

The existing experimental framework covers two core problem classes: matrix sensing (MS) and matrix factorization (MF). At the algorithmic level, it establishes a comparative baseline between Muon (spectral normalization direction) and SGD (raw gradient direction), and through dimension scanning ($d=50,100,200,500$), rank structure (low-rank vs. full-rank), learning rate grid search, and 10 random repetitions, forms a preliminary foundation for statistical inference. However, from a rigorous Design of Experiments (DOE) perspective, the existing framework exhibits significant gaps in the following 8 dimensions.

---

### Dimension A: Algorithm Variants and Ablation Studies

| Sub-dimension | Existing Coverage | Missing Content | Severity |
|--------|----------|----------|------------|
| A1: Muon Internal Variants | Only original SVD spectral normalization $D^{(k)} = UV^\top$ | Truncated SVD ($r_{\text{trunc}} < \text{rank}(G)$), randomized SVD (Halko algorithm), partial spectral normalization (scaling only the first $k$ singular values), not strictly $UV^\top$ but $U\Sigma^{-\alpha}V^\top$ ($\alpha \in [0,1]$) | High |
| A2: Momentum Mechanisms | None | Polyak Heavy-Ball, Nesterov acceleration embedded in Muon; momentum coefficients $\beta \in \{0.0, 0.5, 0.9, 0.99\}$ | High |
| A3: Weight Decay | $\lambda = 0$ | Differential impact of $\lambda \in \{10^{-5}, 10^{-4}, 10^{-3}, 10^{-2}\}$ on Muon vs. SGD | Medium |
| A4: Learning Rate Schedules | Fixed $\eta$ | Dynamic effects of cosine annealing, step decay, and warmup on spectral normalization directions | Medium |
| A5: Adaptive Muon | None | Combining adaptive gains from RMSprop/Adam with spectral directions (e.g., $D^{(k)}_{\text{Adam-Muon}} = \frac{UV^\top}{\sqrt{v^{(k)}} + \epsilon}$) | Medium |
| A6: Baseline Algorithms | Only SGD | Adam, AdaGrad, RMSprop, Momentum SGD, L-BFGS, Natural Gradient Descent (NGD) approximations | High |

**Key Scientific Question**: Does Muon's core advantage solely arise from the directional normalization of $UV^\top$? When exact SVD is removed, momentum is introduced, or weight decay is applied, does the advantage persist?

---

### Dimension B: Problem Variants and Problem Space Expansion

| Sub-dimension | Existing Coverage | Missing Content | Severity |
|--------|----------|----------|------------|
| B1: Rank Ratio | $r = d/10$ and full-rank | Continuous scanning of $r/d \in \{0.01, 0.05, 0.1, 0.2, 0.5, 1.0\}$ | High |
| B2: Matrix Shape | Only square matrices ($d \times d$) | Rectangular matrices $m \times n$ (e.g., $m=2n$, $m=n/2$), sensing operators $A_i \in \mathbb{R}^{m \times n}$ | High |
| B3: Noise Level | $\sigma_\epsilon = 0$ | Impact of $\sigma_\epsilon \in \{0, 10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}\}$ on convergence rate and stability | High |
| B4: Sampling Rate | $m = 3d^2$ (oversampled) | $m/d^2 \in \{0.5, 1.0, 1.5, 2.0, 3.0, 5.0\}$, covering undersampling ($m < d^2$) and oversampling | High |
| B5: Ill-conditioning Control | Natural condition number of random matrices | Explicit construction of $\kappa_{\text{target}} \in \{10, 10^2, 10^3, 10^4, 10^6\}$ (via SVD water-filling or parameterized matrix families) | High |
| B6: Non-square Factorization | $W_i$ are all square in $W_L \cdots W_1$ | Rectangular factor decomposition: $W_1 \in \mathbb{R}^{d_1 \times d_0}, W_2 \in \mathbb{R}^{d_2 \times d_1}, \dots$, dimension-mismatched deep networks | Medium |
| B7: Matrix Distribution | $A_{ij} \sim \mathcal{N}(0,1)$ | Rademacher $\pm 1$, sub-Gaussian, spherical Gaussian, fast JL transforms (Hadamard + subsampling) | Medium |

**Key Scientific Question**: Does Muon's spectral structural advantage depend on the problem being well-conditioned and Gaussian random? When the problem becomes ill-conditioned, undersampled, noisy, or non-square, does Muon still outperform SGD?

---

### Dimension C: Initialization and Sensitivity Analysis

| Sub-dimension | Existing Coverage | Missing Content | Severity |
|--------|----------|----------|------------|
| C1: Initialization Scale | $\mathcal{N}(0, 1/d)$ | Systematic scan of $\sigma_{\text{init}} \in \{10^{-3}, 10^{-2}, 10^{-1}, 1, 10\}/d$ | High |
| C2: Initialization Distribution | Gaussian | Orthogonal initialization ($X^{(0)} = QR$), spectral initialization ($X^{(0)} = U\text{diag}(\lambda) V^\top$), zero initialization, uniform distribution | Medium |
| C3: Muon Specificity | None | Behavior of Muon's $D^{(k)}$ under zero or extreme-scale initialization (SVD degeneracy when $G^{(k)}=0$) | High |
| C4: Deep MF Initialization | L=2 has three special initializations | Initialization sensitivity for L=3,4 (does a "balanced initialization" effect exist?) | Medium |

**Key Scientific Question**: Is Muon more sensitive to initialization scale (because SVD is sensitive to gradient magnitude) or less sensitive (because spectral normalization removes scale)?

---

### Dimension D: Hyperparameter Space and Calibration

| Sub-dimension | Existing Coverage | Missing Content | Severity |
|--------|----------|----------|------------|
| D1: Learning Rate Granularity | $\eta \in \{10^{-3}, 10^{-2}, 10^{-1}\}$ | Log-uniform grid of $\eta \in \{10^{-4}, 3\times 10^{-4}, 10^{-3}, 3\times 10^{-3}, \dots, 10^{-1}\}$ | Medium |
| D2: Weight Decay | $\lambda = 0$ | Interaction effects of $\lambda \times \eta$ (two-dimensional grid) | Medium |
| D3: Schedulers | None | Coupling between warmup length, cosine period, and Muon spectral direction | Low |
| D4: Early Stopping and Tolerance | Fixed $\epsilon$ | Efficiency comparison under different precision requirements $\epsilon \in \{10^{-6}, 10^{-8}, 10^{-10}\}$ | Medium |

---

### Dimension E: Computational Efficiency and Scalability

| Sub-dimension | Existing Coverage | Missing Content | Severity |
|--------|----------|----------|------------|
| E1: Wall-clock Time | Theoretical FLOPs | Actual CPU/GPU timing (including actual SVD overhead, cache effects, BLAS implementation differences) | High |
| E2: Memory Footprint | None | Peak memory analysis of SGD $O(d^2)$ vs. Muon SVD overhead $O(d^3)$ | Medium |
| E3: SVD Implementation Variants | Exact SVD (Golub-Reinsch) | Randomized SVD ($O(d^2 r_{\text{approx}})$), incremental SVD, truncated SVD accuracy–speed tradeoffs | High |
| E4: Scale Extrapolation | $d \leq 500$ | Feasibility at $d=1000, 2000, 5000$ (the hard wall of Muon SVD $O(d^3)$) | High |
| E5: Parallelization | None | Parallelization potential of randomized SVD vs. trivial parallelization of SGD | Medium |

**Key Scientific Question**: Is Muon's theoretical FLOPs advantage (lower per-iteration cost) offset by the actual constant factor of SVD? At what scale $d^*$ does Muon's actual time efficiency fall behind that of SGD?

---

### Dimension F: Statistical Reliability and Rigor of Inference

| Sub-dimension | Existing Coverage | Missing Content | Severity |
|--------|----------|----------|------------|
| F1: Power Analysis | None | Given effect size $\delta$, variance $\sigma^2$, $\alpha=0.05$, are 10 repetitions sufficient? Compute required $n_{\min}$ | High |
| F2: Confidence Intervals | Point estimates only | 95% confidence intervals for $K_\epsilon$, $F_\epsilon$ (t-distribution or bootstrap) | High |
| F3: Multiple Testing Correction | None | Family-wise error rate (FWER) control under 20+ hypothesis tests (Bonferroni, Holm, FDR) | High |
| F4: Effect Size Reporting | None | Distribution of Cohen's $d$, log-ratio $\log(K_{\text{Muon}}/K_{\text{SGD}})$ | Medium |
| F5: Analysis of Variance | None | Multi-factor ANOVA decomposition of variance sources in $K_\epsilon$ (algorithm $\times$ problem $\times$ dimension $\times$ rank) | Medium |
| F6: Non-parametric Tests | Normality assumed | If $K_\epsilon$ distribution is skewed, use Mann-Whitney U, Kruskal-Wallis, permutation tests | Medium |

**Key Scientific Question**: Do the existing 10 repetitions provide sufficient statistical power to detect true convergence rate differences between Muon and SGD? If a difference exists but is not detected (Type II error), are the existing conclusions overly conservative?

---

### Dimension G: Dynamic Behavior and Landscape Analysis

| Sub-dimension | Existing Coverage | Missing Content | Severity |
|--------|----------|----------|------------|
| G1: Hessian Spectral Dynamics | None | Every $T$ steps, compute eigenvalue distribution of Hessian $\nabla^2 f(X^{(k)})$, observing evolution of the max/min eigenvalue ratio across iterations | High |
| G2: Gradient–Update Direction Angle | None | Temporal evolution of $\cos\theta_k = \frac{\langle G^{(k)}, D^{(k)}\rangle}{\|G^{(k)}\|_F \|D^{(k)}\|_F}$ (Muon: $D^{(k)}=UV^\top$, SGD: $D^{(k)}=G^{(k)}$) | High |
| G3: Parameter Trajectory PCA | None | Flatten $\{X^{(k)}\}_{k=0}^K$ into vectors, perform PCA, analyze low-dimensional structure of convergence trajectories | Medium |
| G4: Spectral Gap Dynamics | Static $\bar{\sigma}_{\log}$ | Evolution of the top $r$ singular values of $\sigma_i(G^{(k)})$, whether spectral gap $\sigma_r - \sigma_{r+1}$ affects Muon direction quality | High |
| G5: Critical-point Neighborhood Dynamics | None | Local convergence rate measurement (linear vs. sublinear) near convergence ($\|\nabla f\| < 10^{-3}$) | Medium |

**Key Scientific Question**: Does Muon's spectrally normalized direction always maintain high correlation with the gradient direction? As the Hessian spectral structure evolves across iterations, does Muon's direction quality degrade?

---

### Dimension H: Generalization and Distribution Shift

| Sub-dimension | Existing Coverage | Missing Content | Severity |
|--------|----------|----------|------------|
| H1: Measurement Matrix Distribution | $A_{ij} \sim \mathcal{N}(0,1)$ | Rademacher, Bernoulli, spherical Gaussian ($\text{Unif}(\mathbb{S}^{d^2-1})$), structured matrices (DFT, Hadamard) | Medium |
| H2: Noise Distribution | Gaussian noise | Laplace noise (heavy-tailed), uniform noise, multiplicative noise | Low |
| H3: Target Matrix Structure | Random low-rank | Matrices with specific eigenvalue decay (polynomial decay $\lambda_i = i^{-\alpha}$, exponential decay) | Medium |
| H4: Problem Mixing | Independent experiments | Cross-problem transfer: are hyperparameters optimal on MS also optimal on MF? | Low |

---

### Summary of Missing Dimensions

The above 8 dimensions are prioritized by the product of **scientific value $\times$ implementation cost** (higher product = higher priority for supplementation):

| Rank | Dimension | Key Gaps | Risk Description |
|------|------|----------|----------|
| 1 | B (Problem Variants) | Undersampling, noise, ill-conditioning | Existing conclusions may only hold on "easy" problems |
| 2 | A (Algorithm Baselines) | Adam, Momentum SGD, L-BFGS | Unable to claim Muon outperforms "standard" optimizers |
| 3 | E (Computational Efficiency) | Wall-clock time, $d=1000$+ | Theoretical advantages may be offset by actual overhead |
| 4 | F (Statistical Reliability) | Power analysis, confidence intervals, multiple testing | 10 repetitions may be insufficient; statistical credibility of conclusions unknown |
| 5 | G (Dynamic Behavior) | Hessian dynamics, gradient–direction angle | Lack of mechanistic understanding of "why Muon is better" |
| 6 | C (Initialization) | Extreme scales, zero initialization | Large initialization differences in practical use |
| 7 | D (Hyperparameters) | $\lambda$, schedulers | Hyperparameter interactions may change conclusions |
| 8 | H (Generalization) | Non-Gaussian matrices | Robustness boundaries unknown |

---

