<!--
Original Document: Mathematical Foundations and Statistical Formalization of Experimental Design for the Muon (μ) Optimization Algorithm
This File: 11. Factors and Treatment Variables
Split ID: 13
-->

[TOC]

---

# Part III: Statistical Variable System

> This section constructs a complete statistical variable system covering the full chain of experimental design, execution, and analysis. Building upon the framework of experimental formalization established in Part II, it further defines all factors, response variables, covariates, random variables, and derived statistics with precision, and establishes causal structures and functional dependency relationships among variables. This section can be directly referenced as the statistical specification for experimental code.

> Version: v1.0  
> Applicable Scope: Comparative algorithm experiments on Matrix Sensing (MS) and Matrix Factorization (MF) optimization problems  
> Design Principles: Reproducible, Quantifiable, Inferable, Extensible

---

## 11. Factors and Treatment Variables (Factors & Treatments)

### 1.1 Overview of Factor Classification

This experiment adopts a **Mixed Factorial Design**, where factors are classified into three categories:
- **Algorithm Factors**: Discrete variables determining optimizer behavior
- **Problem Factors**: Structural variables defining optimization problem instances
- **Environmental Factors**: Variables affecting the computation process without altering the problem essence

### 1.2 Detailed Factor Definition Table

| Symbol | Name | Type | Domain / Level Set | Role in Experiment | Supplementary Suggestions |
|:---:|:---|:---:|:---|:---|:---|
| $\alpha$ | **Algorithm Factor** | Categorical | $\{\text{Muon}, \text{SGD}\}$ | Main effect: Testing systematic differences between the two optimizers | Extensible to $\{\text{Muon}, \text{SGD}, \text{Adam}, \text{AdamW}\}$ |
| $\pi$ | **Problem Type Factor** | Categorical | $\{\text{MS}, \text{MF}\}$ | Main effect + Interaction with $\alpha$ | Can add Matrix Completion (MC), Robust PCA |
| $d$ | **Matrix Dimension** | Ordinal Integer | $\{50, 100, 200, 500\}$ | Main effect + Interaction with $\alpha, \pi$ | **Supplement** $d \in \{1000, 2000\}$ for large-scale extrapolation; supplement rectangular $d_1 \times d_2$ |
| $r$ | **Target Rank** | Discrete Integer | $\{d/10, d\}$ (low-rank vs. full-rank) | Main effect + Interaction with $\alpha$ (Muon is sensitive to low-rank structure) | **Supplement** $r \in \{d/20, d/5, d/2\}$ for rank sensitivity analysis |
| $L$ | **Factorization Depth** | Ordinal Integer | $\{2, 3, 4\}$ (MF only) | Main effect (in MF sub-experiments) + Interaction with $\alpha$ | **Supplement** $L=5, 6$ for depth extension; $L=1$ degenerate baseline |
| $\iota$ | **Initialization Pattern** | Categorical | $\{\text{Spectral}, \text{Gaussian}, \text{Orthogonal}\}$ (MF, $L=2$ only) | Main effect + Interaction with $\alpha$ | **Supplement** Xavier, He, small-norm initialization; balanced initialization for MF depth $L \geq 3$ |
| $\eta$ | **Learning Rate** | Continuous | $\{10^{-3}, 10^{-2}, 10^{-1}\}$ (grid search) | **Tuning Variable / Nuisance Variable**: Select optimal $\eta$ under each $(\alpha, \pi, d)$ configuration | Supplement log-uniform search $\eta \sim \text{LogUniform}(10^{-4}, 10^{0})$; or Bayesian optimization |
| $\lambda$ | **Weight Decay** | Continuous | $\{0\}$ (current setting) | Control variable (currently fixed) | **Supplement** $\lambda \in \{10^{-4}, 10^{-3}, 10^{-2}\}$ to test regularization effects |
| $\sigma_\epsilon$ | **Noise Level** | Continuous | $\{0\}$ (current setting) | Control variable (currently fixed) | **Supplement** $\sigma_\epsilon \in \{10^{-4}, 10^{-3}, 10^{-2}\} \cdot \|y\|$ for noise robustness analysis |
| $\rho_{os}$ | **Oversampling Factor** | Continuous | $m / d^2$ (MS only) | Covariate: Affects problem identifiability and convergence rate | Fix at $m = 5rd$ (low-rank) or $m = 2d^2$ (full-rank), or use as a scanning factor |
| $\kappa$ | **Matrix Shape** | Categorical | $\{\text{Square}, \text{Rectangular}\}$ | Extension factor | **New**: Rectangular matrices $d_1 \times d_2$ (e.g., $100 \times 50$) to test algorithm adaptability to non-square matrices |
| $\beta$ | **Batch Size** | Ordinal Integer | $\{1, \text{full-batch}\}$ | Control variable (default full-batch) | Supplement mini-batch $B \in \{32, 128, 512\}$ for stochastic gradient scenarios |
| $T_{\max}$ | **Maximum Iterations** | Discrete Integer | $10^5$ | Experiment truncation parameter | Can be adjusted proportionally to $d$: $T_{\max} = 10^4 \cdot (d/50)$ |

### 1.3 Structural Constraints Among Factors

Certain factor combinations exhibit **Structural Zeros** or **Nesting** relationships:

```
π = MS  →  L undefined (no factorization depth)
π = MF  →  ρ_os undefined (no oversampling)
r = d   →  low-rank-specific parameters invalid
L = 2   →  ι activated (three initialization patterns)
L ≥ 3   →  ι degenerates to single balanced initialization
```

**Nested Factor Table**:

| Outer Factor | Inner Factor | Nesting Relationship Description |
|:---|:---|:---|
| $\pi = \text{MS}$ | $L$ | $L$ does not exist under MS (denoted as $L = \varnothing$) |
| $\pi = \text{MF}$ | $\rho_{os}$ | Oversampling factor does not exist under MF |
| $r < d$ | $\rho_{os}$ | Oversampling factor is meaningful only in low-rank MS |
| $L = 2$ | $\iota$ | Three initialization patterns are compared only when $L=2$ |

### 1.4 Experimental Configuration Space Size

**Base Configuration Space** (considering only active factor combinations):

$$
\begin{aligned}
|\Omega_{\text{base}}| &= |\alpha| \times \Big( |\text{MS}| + |\text{MF}| \Big) \\
&= 2 \times \Big[ |d| \times |r| \times |\rho_{os}| + |d| \times |L_{\text{active}}| \times |\iota_{\text{active}}| \Big] \\
&= 2 \times \Big[ 4 \times 2 \times 1 + 4 \times 3 \times 1 \Big] \\
&= 2 \times (8 + 12) = 40 \text{ (base factor combinations)}
\end{aligned}
$$

Considering the grid search for $\eta$ ($|\eta| = 3$) and random seed replications ($n_{\text{seed}} = 10$), the **total number of experimental runs**:

$$
N_{\text{runs}} = 40 \times 3 \times 10 = 1200 \text{ independent runs}
$$

---

