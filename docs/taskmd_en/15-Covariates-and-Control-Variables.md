<!--
Original Document: Mathematical Foundations and Statistical Formalization of Muon (μ) Optimization Algorithm Experimental Design
This File: 13. Covariates & Control Variables
Split Index: 15
-->

[TOC]

---

## 13. Covariates & Control Variables (Covariates & Controls)

### 3.1 Covariate Classification

| Level | Description | Purpose |
|:---:|:---|:---|
| **Problem-Inherent Covariates** | Determined by the problem instance, cannot be altered by the algorithm | Stratified analysis, regression adjustment |
| **Initialization Covariates** | Determined by the initialization, recorded for adjustment | Variance reduction, causal inference |
| **Environmental Covariates** | Information about the experimental running environment | Heterogeneity control, meta-analysis |

### 3.2 Detailed Covariate Table

| Symbol | Name | Mathematical Definition | Domain | Type | Distribution Assumption | Experimental Purpose |
|:---:|:---|:---|:---:|:---:|:---:|:---|
| $\kappa_{\text{sp}}^{(0)}$ | **Initial Spectral Ratio** | $\kappa_{\text{sp}}^{(0)} = \|X^{(0)}\|_2 / \|X^{(0)}\|_F$ | $[1/\sqrt{d}, 1]$ | Continuous | Depends on initialization | Predicts convergence rate; Muon is sensitive to spectral ratio |
| $\kappa_{\text{cond}}^{(0)}$ | **Initial Condition Number** | $\kappa_{\text{cond}}^{(0)} = \sigma_1(X^{(0)}) / \sigma_d(X^{(0)})$ | $[1, \infty)$ | Continuous | Depends on initialization | Indicator of problem ill-conditioning; affects convergence stability |
| $\kappa_{\text{cond}}^{\star}$ | **True Condition Number** | $\kappa_{\text{cond}}^{\star} = \sigma_1(X^\star) / \sigma_r(X^\star)$ (low-rank) | $[1, \infty)$ | Continuous | Depends on construction of $X^\star$ | Intrinsic difficulty of the problem |
| $r_\epsilon$ | **Numerical $\epsilon$-Rank** | $r_\epsilon = |\{i : \sigma_i(X^\star) > \epsilon\}|$ | $\{1, \ldots, d\}$ | Discrete | Depends on spectral decay of $X^\star$ | Compared with target rank $r$ to assess over-parameterization |
| $\mu_{\text{SR}}$ | **Strong Convexity Parameter** (MS) | $\mu_{\text{SR}} = \lambda_{\min}(\frac{1}{m}\sum_i \text{vec}(A_i)\text{vec}(A_i)^\top)$ | $\mathbb{R}_{>0}$ | Continuous | Depends on measurement matrices | Condition number of MS problem: $\kappa_{\text{MS}} = M_{\text{SR}} / \mu_{\text{SR}}$ |
| $s$ | **Random Seed** | $s \in \{0, 1, \ldots, 9\}$ | $\{0, \ldots, 9\}$ | Discrete | Uniform distribution | Enables independent replications; blocking factor in fixed-effects models |
| $\text{HW}$ | **Hardware Configuration** | CPU/GPU model, memory, PyTorch version | Label vector | Categorical | Constant within experiment | Moderating variable in meta-regression; cross-experiment comparability |
| $\tau_{\text{float}}$ | **Floating-Point Precision** | $\{\text{float32}, \text{float64}\}$ | Binary | Categorical | Constant within experiment | Numerical stability control |
| $\delta_{\text{ortho}}$ | **Orthogonality Deviation** (MF, L≥3) | $\delta_{\text{ortho}} = \sum_{\ell=1}^{L-1} \|W_{\ell+1}^\top W_{\ell+1} - W_\ell W_\ell^\top\|_F$ | $\mathbb{R}_{\geq 0}$ | Continuous | Grows with $L$ | Balance measure for deep MF; affects convergence dynamics |

### 3.3 Fixed Strategy for Control Variables

| Variable | Fixed Value | Rationale | Future Extension |
|:---:|:---|:---|:---|
| $\lambda$ (weight decay) | $0$ | Isolate the effect of the algorithm itself | See §1.2 for supplementing non-zero levels |
| $\sigma_\epsilon$ (noise) | $0$ | Study deterministic convergence first | See §1.2 for supplementing noise robustness experiments |
| $\beta$ (batch size) | Full-batch | Eliminate stochastic gradient variance | Supplement mini-batch experiments |
| $\tau_{\text{float}}$ | float64 | Guarantee numerical precision | Supplement float32 experiments to assess numerical stability |
| Maximum iterations $T_{\max}$ | $10^5$ | Computational resource constraints | Adaptive adjustment according to $d$ |

### 3.4 Regression Adjustment Formula for Covariates

For the primary response variable $K_\epsilon$, a **covariate-adjusted linear model** is adopted:

$$
\log K_\epsilon = \mu + \tau_\alpha + \tau_\pi + \tau_{\alpha \times \pi} + \gamma_1 \log \kappa_{\text{cond}}^{(0)} + \gamma_2 \log \kappa_{\text{sp}}^{(0)} + \gamma_3 \delta_{\text{ortho}} + \varepsilon
$$

where $\gamma_1, \gamma_2, \gamma_3$ are covariate coefficients estimated by least squares. This adjustment can **reduce residual variance by 30--50%**, thereby improving statistical power.

---

