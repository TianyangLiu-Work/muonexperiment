<!--
Original Document: Mathematical Foundations and Statistical Formalization of Muon (μ) Optimization Algorithm Experimental Design
This File: 5. Notation Summary and Unifying Conventions
Split Number: 06
-->

[TOC]

---

## 5. Notation Summary and Unifying Conventions

To ensure consistent notation throughout the document, a summary is provided as follows:

| Symbol | Meaning |
|--------|---------|
| $d$ | Matrix dimension |
| $r$ | Matrix rank (or target rank) |
| $m$ | Number of measurement samples |
| $L$ | Matrix factorization depth |
| $R$ | Number of experimental repetitions |
| $k$ | Iteration index |
| $K_\epsilon$ | Iteration complexity to reach accuracy $\epsilon$ |
| $F_\epsilon$ | Total FLOPs to reach accuracy $\epsilon$ |
| $X^\star$ | True / target matrix |
| $X^{(k)}$ | Parameter matrix at the $k$-th iteration |
| $G^{(k)}$ | Gradient matrix at step $k$ |
| $D^{(k)}$ | Search direction at step $k$ |
| $f(X)$ | Objective function |
| $f^\star$ | Optimal function value |
| $\delta_k$ | Function value gap $f(X^{(k)}) - f^\star$ |
| $\eta$ | Learning rate / step size |
| $\lambda$ | Weight decay coefficient |
| $\sigma_i(X)$ | The $i$-th singular value of $X$ |
| $\|X\|_2$ | Spectral norm (largest singular value) |
| $\|X\|_F$ | Frobenius norm |
| $\|X\|_*$ | Nuclear norm (sum of singular values) |
| $\kappa_{\text{cond}}(X)$ | Matrix spectral condition number $\sigma_1/\sigma_r$ |
| $\kappa_{\text{sp}}(X)$ | Spectral ratio $\|X\|_2/\|X\|_F$ |
| $\kappa(f)$ | Function condition number $L/\mu$ |
| $\mathcal{A}$ | Linear measurement operator |
| $\mathcal{A}^*$ | Adjoint operator |
| $\delta_r(\mathcal{A})$ | RIP constant |
| $\sigma_n^2$ | Noise variance |
| $\alpha$ | Significance level |
| $p$ | p-value |
| $\beta$ | Type II error probability |
| $\text{Power}$ | Statistical power $1-\beta$ |
| $\text{FLOPs}$ | Floating-point operations |
| $O(\cdot), \Theta(\cdot), \Omega(\cdot)$ | Asymptotic notation |

---

---
