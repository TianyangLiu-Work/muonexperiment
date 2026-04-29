<!--
Original Document: Mathematical Foundations and Statistical Formalization of Experimental Design for the Muon (μ) Optimization Algorithm
This Document: 17. Proposed Statistical Framework for Experimental Design
Split Number: 19
-->

[TOC]

---

## 17. Proposed Statistical Framework for Experimental Design

### 7.1 Sample Size and Power Analysis

**Current Design**: Each configuration has $n_s = 10$ replicates.

**Power Analysis** (for the main effect $\alpha$ on the response $\log K_\epsilon$):

Assuming a medium effect size $d_{\text{Cohen}} = 0.8$ (log-scale difference $d_{\log} \approx 0.5$) and significance level $\alpha = 0.05$:

$$
\beta = 1 - \Phi\left(z_{\alpha/2} + \frac{|d_{\text{Cohen}}| \sqrt{n_s/2}}{\sqrt{1 + \rho_{\text{intra}}(n_s - 1)}}\right)
$$

where $\rho_{\text{intra}}$ is the intraclass correlation coefficient (correlation among different random seeds for the same configuration, typically $< 0.1$).

**Power Calculation Results**:

| Effect Size $d_{\text{Cohen}}$ | Power at $n_s = 10$ | Power at $n_s = 20$ | Power at $n_s = 30$ |
|:---:|:---:|:---:|:---:|
| 0.5 (Medium) | 0.72 | 0.92 | 0.98 |
| 0.8 (Large) | 0.95 | >0.99 | >0.99 |
| 1.2 (Very Large) | >0.99 | >0.99 | >0.99 |

**Recommendations**:
- Core configurations ($d \leq 200$): Retain $n_s = 10$
- Large-scale configurations ($d = 500$): Reduce to $n_s = 5$ to control computational cost
- Critical interaction tests: Increase to $n_s = 20$ or adopt adaptive sample sizes

### 7.2 Factorial Design Framework

**Full Factorial Design** (structural zeros not considered):

$$
2 \times 2 \times 4 \times 2 \times 3 \times 3 \times 3 = 864 \text{ combinations (factor levels only)}
$$

**Sparse Factorial Design Recommendation** (Fractional Factorial):

Adopt $2^{6-1}$ or $2^{7-2}$ designs to screen main effects and two-way interactions, reducing the number of runs from 64/128 to 32/64.

**Recommended Hierarchical Factorial Strategy**:

```
Level 1 (Screening Experiment):
  - Factors: α × π × d × r × η
  - Levels: 2 × 2 × 3 × 2 × 3 = 72 combinations
  - n_s = 5 per combination
  - Objective: Identify significant main effects and two-way interactions

Level 2 (Refinement Experiment):
  - Based on Level 1 results, focus on significant factor combinations
  - Supplement with scans of L, ι, λ, σ_ϵ
  - n_s = 10 (core configurations) or n_s = 20 (boundary configurations)
```

### 7.3 Latin Square and Block Designs

**Latin Square Design** (for controlling two nuisance factors):

If it is necessary to simultaneously control for nuisance factors from **matrix size $d$** and **problem type $\pi$**, a Latin square can be constructed:

| | $\pi = \text{MS}$ | $\pi = \text{MF}, L=2$ | $\pi = \text{MF}, L=3$ | $\pi = \text{MF}, L=4$ |
|:---:|:---:|:---:|:---:|:---:|
| $d = 50$ | Muon | SGD | Muon | SGD |
| $d = 100$ | SGD | Muon | SGD | Muon |
| $d = 200$ | Muon | SGD | Muon | SGD |
| $d = 500$ | SGD | Muon | SGD | Muon |

This design ensures that each algorithm appears exactly once in each size-problem combination, balancing block effects.

**Block Design Recommendations**:
- **Blocking factor**: Random seed $s$ (10 blocks)
- **Treatment factors**: $(\alpha, \eta)$ combinations
- **Within-block design**: Randomize algorithm-learning rate order within each seed to eliminate time trends

### 7.4 Stratified Sampling Strategy

**Adaptive Stratified Sampling**:

Since the distribution of the response variable is typically right-skewed ($K_\epsilon$ is very large for difficult configurations), the following is recommended:

1. **Stratification by Problem Difficulty**:
   - Stratum 1 (Easy): $\kappa_{\text{cond}}^\star < 10$, $d \leq 100$
   - Stratum 2 (Medium): $\kappa_{\text{cond}}^\star \in [10, 100]$, $d \in [100, 200]$
   - Stratum 3 (Hard): $\kappa_{\text{cond}}^\star > 100$ or $d \geq 500$

2. **Neyman Optimal Allocation**:
   $$n_h \propto N_h \cdot S_h$$
   where $N_h$ is the number of configurations in stratum $h$, and $S_h$ is the within-stratum standard deviation of the response.

3. **Oversampling Strategy**:
   Apply 2× oversampling to boundary configurations (maximum size, maximum depth, most difficult initialization) to improve the reliability of extrapolation.

### 7.5 Multiple Comparison Correction

The experiment involves a large number of hypothesis tests (main effects, interaction effects, multiple response variables), and it is necessary to control the family-wise error rate (FWER) or the false discovery rate (FDR):

| Correction Method | Applicable Scenario | Formula/Description |
|:---|:---|:---|
| **Bonferroni** | Few tests ($m < 20$) | $\alpha^* = \alpha / m$; conservative |
| **Holm-Bonferroni** | Ordered p-value sequences | Stepwise correction; more powerful than Bonferroni |
| **Benjamini-Hochberg** | Large number of exploratory tests | Controls FDR $\leq \alpha$; recommended for exploratory analysis |
| **Tukey HSD** | All pairwise comparisons | Based on the studentized range distribution; suitable for post-hoc comparisons |
| **Scheffé** | All contrasts | Conservative but flexible; suitable for complex contrasts |

**Recommended Strategy**:
- **Confirmatory analysis** (main effects $\alpha$): No correction, $\alpha = 0.05$
- **Secondary hypotheses** (interaction effects): Holm-Bonferroni, family size $m = 6$ (6 interaction terms)
- **Exploratory analysis** (all pairwise configuration comparisons): Benjamini-Hochberg, FDR = 0.10

### 7.6 Missing Data Handling Plan

| Missing Type | Possible Cause | Handling Strategy |
|:---|:---|:---|
| **Non-convergence** | $K_\epsilon = T_{\max}$ (truncated) | Right-censoring; apply Cox proportional hazards model or Tobit regression |
| **Numerical overflow** | Gradient explosion, NaN | Mark as failed run; if failure rate > 20%, report as a reliability issue |
| **Timeout** | Wall-clock exceeds budget | Combine with non-convergence handling; or treat as competing risk |
| **Out of memory** | $M_{\text{peak}}$ exceeds limit | Mark as resource failure; exclude from resource-constrained scenario analysis |

**Statistical Treatment of Censored Data**:

For non-converged runs (right-censored $K_\epsilon$), define the censoring indicator $\delta = \mathbb{I}_{\text{conv}}$ and adopt a **survival analysis framework**:

$$
h(k) = \lim_{\Delta \to 0} \frac{P(k \leq K_\epsilon < k + \Delta \mid K_\epsilon \geq k)}{\Delta}
$$

where $h(k)$ is the **convergence hazard function**. Algorithm comparison can be transformed into testing $h_{\text{Muon}}(k) \stackrel{?}{=} h_{\text{SGD}}(k)$ (log-rank test).

### 7.7 Recommended Experimental Execution Pipeline

```
Phase 0: Pilot Study
  ├─ Small-scale configurations (d=50, n_s=3)
  ├─ Coarse learning rate scan (η ∈ {1e-4, 1e-3, 1e-2, 1e-1, 1})
  ├─ Estimate effect sizes, variance components, censoring proportions
  └─ Output: sample sizes and learning rate candidate sets for the formal experiment

Phase 1: Core Experiment
  ├─ Full factorial combinations (considering structural constraints)
  ├─ Optimal learning rate selection (based on Phase 0)
  ├─ n_s = 10 replicates
  ├─ Record all raw responses and covariates
  └─ Output: complete dataset

Phase 2: Boundary Extension
  ├─ Supplement d=1000, 2000 (if resources permit)
  ├─ Supplement noise experiments (σ_ϵ > 0)
  ├─ Supplement weight decay experiments (λ > 0)
  ├─ Supplement rectangular matrix experiments
  └─ Output: extrapolation validation data

Phase 3: Robustness Analysis
  ├─ Worst-case initialization scan
  ├─ Suboptimal learning rate sensitivity
  ├─ Different random matrix distributions (e.g., Rademacher instead of Gaussian)
  └─ Output: algorithm reliability boundaries

Phase 4: Statistical Inference and Reporting
  ├─ Multiple comparison correction
  ├─ Effect size calculation and confidence intervals
  ├─ Interaction effect visualization
  ├─ Causal inference sensitivity analysis
  └─ Output: final statistical report
```

---

---
