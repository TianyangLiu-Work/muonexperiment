<!--
Original Document: Mathematical Foundations and Statistical Formalization of Muon ($\mu$) Optimization Algorithm Experimental Design
This File: 14. Random Variables and Distribution Assumptions
Split Index: 16
-->

[TOC]

---

## 14. Random Variables and Distribution Assumptions

### 4.1 Overview of Random Variables

This experiment incorporates four categories of random sources, each with explicit probability distribution assumptions:

1. **Problem generation randomness**: $X^\star$, $\{A_i\}$, $\{\epsilon_i\}$
2. **Algorithm initialization randomness**: $W_\ell^{(0)}$
3. **Algorithm intrinsic randomness**: Random SVD, random sampling
4. **Experimental repetition randomness**: Random seed $s$

### 4.2 Detailed Table of Random Variables

| Symbol | Name | Distribution Assumption | Support | Parameters | Independence/Dependence |
|:---:|:---|:---|:---:|:---:|:---|
| $X^\star$ | Ground truth matrix | Constructive distribution (see below) | $\mathbb{R}^{d_1 \times d_2}$ | $r, \kappa_{\text{cond}}^{\star}$ | Determined by $s$; independent across seeds |
| $A_i$ | $i$-th measurement matrix | $A_{i,jk} \stackrel{\text{i.i.d.}}{\sim} \mathcal{N}(0, 1)$ | $\mathbb{R}^{d \times d}$ | None | Fixed given $s$; independent across $i$ |
| $\epsilon_i$ | Measurement noise | $\epsilon_i \stackrel{\text{i.i.d.}}{\sim} \mathcal{N}(0, \sigma_\epsilon^2)$ | $\mathbb{R}$ | $\sigma_\epsilon$ | Independent of $A_i$ |
| $W_\ell^{(0)}$ | Initialization of layer $\ell$ | $W_{\ell,ij}^{(0)} \stackrel{\text{i.i.d.}}{\sim} \mathcal{N}(0, 1/d_\ell)$ | $\mathbb{R}^{d_{\ell+1} \times d_\ell}$ | $d_\ell$ | Fixed given $s$; independent across $\ell$ |
| $s$ | Random seed | $s \sim \text{Uniform}\{0, \ldots, 9\}$ | $\{0, \ldots, 9\}$ | None | Experimental design variable |

### 4.3 Constructive Distribution of the Ground Truth Matrix $X^\star$

**Low-rank construction ($r < d$)**:

$$
X^\star = U \cdot \text{diag}(\sigma_1, \ldots, \sigma_r) \cdot V^\top
$$

where:
- $U \in \mathbb{R}^{d \times r}$: Random orthogonal matrix (sampled from Haar measure)
- $V \in \mathbb{R}^{d \times r}$: Random orthogonal matrix, independent of $U$
- Singular value construction (two modes):
  - **Uniform spectrum**: $\sigma_i = 1$ ($i = 1, \ldots, r$)
  - **Exponentially decaying spectrum**: $\sigma_i = \kappa^{-(i-1)/(r-1)}$ ($\kappa$ is the condition number parameter)

**Full-rank construction ($r = d$)**:

$$
X^\star = \frac{1}{\sqrt{d}} G, \quad G_{ij} \stackrel{\text{i.i.d.}}{\sim} \mathcal{N}(0, 1)
$$

### 4.4 Joint Distribution Factorization

The joint distribution of all random variables can be factorized according to the experimental hierarchy as:

$$
\begin{aligned}
& p(X^\star, \{A_i\}_{i=1}^m, \{\epsilon_i\}_{i=1}^m, \{W_\ell^{(0)}\}_{\ell=1}^L, s \mid \pi, d, r, L, \iota, \sigma_\epsilon) \\
&= \underbrace{p(s)}_{\text{seed}} \times \underbrace{p(X^\star \mid s, d, r, \pi)}_{\text{ground truth}} \times \underbrace{p(\{A_i\} \mid s, d, m)}_{\text{measurement matrices}} \times \underbrace{p(\{\epsilon_i\} \mid s, m, \sigma_\epsilon)}_{\text{noise}} \times \underbrace{p(\{W_\ell^{(0)}\} \mid s, d, L, \iota)}_{\text{initialization}} \\
&= \frac{1}{10} \times \delta_{X^\star(s)} \times \prod_{i=1}^m \mathcal{N}(\text{vec}(A_i) \mid 0, I_{d^2}) \times \prod_{i=1}^m \mathcal{N}(\epsilon_i \mid 0, \sigma_\epsilon^2) \times \prod_{\ell=1}^L \mathcal{N}(\text{vec}(W_\ell^{(0)}) \mid 0, \frac{1}{d_\ell} I_{d_\ell d_{\ell+1}})
\end{aligned}
$$

**Conditional independence structure**:

$$
\begin{aligned}
\{A_i\} &\perp\!\!\perp \{\epsilon_i\} \perp\!\!\perp \{W_\ell^{(0)}\} \mid s, d, m, L, \sigma_\epsilon, \iota \\
X^\star &\perp\!\!\perp \{A_i\} \mid s \quad \text{(under deterministic construction)}
\end{aligned}
$$

### 4.5 Algorithmic Randomness

| Algorithm | Random Operation | Distribution | Remarks |
|:---:|:---|:---|:---|
| Muon | Random SVD (optional) | rSVD approximation | Deterministic SVD used by default; switchable for large-scale cases |
| Muon | Random orthogonalization | Sign selection in QR decomposition | Negligible effect on convergence results |
| SGD | Mini-batch sampling (if enabled) | Uniform sampling without replacement | Current experiments use full-batch; no such randomness |

---

