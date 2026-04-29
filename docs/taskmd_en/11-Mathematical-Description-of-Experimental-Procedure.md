<!--
Original Document: Mathematical Foundations and Statistical Formalization of Muon ($\mu$) Optimization Algorithm Experimental Design
This File: 9. Mathematical Description of Experimental Procedure
Split Number: 11
-->

[TOC]

---

## 9. Mathematical Description of Experimental Procedure

This chapter formalizes the experimental execution procedure into four phases (Phase 1--4), each with clearly defined mathematical inputs, outputs, and transformation rules.

---

### 5.1 Phase 1: Learning Rate Calibration

**Objective**: Find the optimal learning rate $\eta^\star$ for each (problem, algorithm) pair.

**Input**:
- Problem instance $p \in \mathcal{P}$
- Algorithm $\mathcal{A} \in \{\text{Muon}, \text{SGD}\}$
- Candidate learning rate set $\mathcal{H} = \{10^{-3}, 10^{-2}, 10^{-1}\}$
- Fixed random seed $s_0 = 42$ (calibration seed)

**Procedure**:

For each $\eta \in \mathcal{H}$, execute $K_{\max}$ iterations:

$$
\theta^{(k+1)} = \mathcal{T}_{\eta}^{\mathcal{A}}(\theta^{(k)}, \nabla f(\theta^{(k)})), \quad k = 0, \ldots, K_{\max}-1
$$

Record the convergence trajectory:

$$\mathcal{T}(\eta) = \{ (k, f(\theta^{(k)})) \}_{k=0}^{K_{\max}}
$$

**Output**:

$$
\eta^\star(p, \mathcal{A}) = \arg\min_{\eta \in \mathcal{H}} K_\epsilon^{(\mathcal{A})}(p, s_0; \eta)
$$

where $K_\epsilon$ denotes the number of iterations required to reach precision $\epsilon$. If multiple values of $\eta$ achieve the same minimum number of iterations, the smallest $\eta$ is selected (more conservative).

**Calibration Result Record**:

$$
\mathcal{C} = \left\{ (p, \mathcal{A}, \eta^\star(p, \mathcal{A})) \; \middle| \; p \in \mathcal{P}_{\text{cal}}, \mathcal{A} \in \mathcal{A} \right\}
$$

where $\mathcal{P}_{\text{cal}} \subseteq \mathcal{P}$ is the calibration subset (typically selected to be representative in dimension, e.g., $d = 100$).

**Note**: This experiment performs a grid search over learning rates, selecting the best $\eta$ for each configuration.

---

### 5.2 Phase 2: Formal Comparative Experiment

**Objective**: Systematically compare the performance of the two algorithms under controlled randomness.

**Input**:
- Complete problem instance space $\mathcal{P}$
- Algorithm space $\mathcal{A}$
- Calibrated learning rates $\eta^\star(p, \mathcal{A})$
- Random seed set $\mathcal{R}_{\text{seed}} = \{1, \ldots, 10\}$
- Evaluation metric space $\mathcal{M}$

**Randomization Design**: A **paired randomized design** is adopted, meaning that for the same problem instance and the same random seed, both algorithms use identical data and initialization:

$$
(\theta^{(0)}, \{A_i\}, y) = \Phi_{MS}(s) \quad \text{or} \quad (\{W_i^{(0)}\}, X^\star) = \Phi_{MF}(s)
$$

Then Muon and SGD are run separately:

$$
\begin{aligned}
\theta_{\text{Muon}}^{(k+1)} &= \mathcal{T}_{\eta^\star}^{\text{Muon}}(\theta_{\text{Muon}}^{(k)}, G^{(k)}) \\
\theta_{\text{SGD}}^{(k+1)} &= \mathcal{T}_{\eta^\star}^{\text{SGD}}(\theta_{\text{SGD}}^{(k)}, G^{(k)})
\end{aligned}
$$

Note: For MF problems, the same initialization can be used since the parameter dimensions are identical; for MS problems, the initialization is the same.

**Output**: For each $(p, \mathcal{A}, s)$ triplet, record:

$$
\mathcal{D}_{\text{raw}} = \left\{ (p, \mathcal{A}, s, \{f(\theta^{(k)}\}_{k=0}^{K_\epsilon}, K_\epsilon, F_\epsilon, \bar{\sigma}_{\log}) \right\}
$$

---

### 5.3 Phase 3: Data Aggregation and Statistical Computation

**Objective**: Compute test statistics from raw data.

**Procedure**:

**Step 1: Aggregate by problem.**

For each problem $p$ and algorithm $\mathcal{A}$, compute:

$$
\bar{K}_\epsilon^{(\mathcal{A})}(p) = \frac{1}{R} \sum_{s=1}^R K_\epsilon^{(\mathcal{A})}(p, s), \quad \hat{\sigma}_{K}^{(\mathcal{A})}(p) = \sqrt{\frac{1}{R-1} \sum_{s=1}^R \left(K_\epsilon^{(\mathcal{A})}(p, s) - \bar{K}_\epsilon^{(\mathcal{A})}(p)\right)^2}
$$

$$
\bar{F}_\epsilon^{(\mathcal{A})}(p) = \frac{1}{R} \sum_{s=1}^R F_\epsilon^{(\mathcal{A})}(p, s)
$$

**Step 2: Compute efficiency ratios.**

For each problem $p$ and seed $s$:

$$
\rho_K(p, s) = \frac{K_\epsilon^{\text{Muon}}(p, s)}{K_\epsilon^{\text{SGD}}(p, s)}, \quad \rho_F(p, s) = \frac{F_\epsilon^{\text{Muon}}(p, s)}{F_\epsilon^{\text{SGD}}(p, s)}
$$

Then compute:

$$
\bar{\rho}_K(p) = \frac{1}{R} \sum_{s=1}^R \rho_K(p, s), \quad \bar{\rho}_F(p) = \frac{1}{R} \sum_{s=1}^R \rho_F(p, s)
$$

**Step 3: Compute stability measures.**

For each algorithm, each problem, and each step $k$:

$$
\sigma_{\log}^{(\mathcal{A})}(p, k) = \sqrt{\frac{1}{R-1} \sum_{s=1}^R \left(\ell_s^{(k)} - \bar{\ell}^{(k)}\right)^2}
$$

Then:

$$
\bar{\sigma}_{\log}^{(\mathcal{A})}(p) = \frac{1}{\bar{K}_\epsilon^{(\mathcal{A})}(p)} \sum_{k=1}^{\bar{K}_\epsilon^{(\mathcal{A})}(p)} \sigma_{\log}^{(\mathcal{A})}(p, k)
$$

**Output**: Aggregated dataset

$$
\mathcal{D}_{\text{agg}} = \left\{ (p, \bar{K}_\epsilon^{\text{Muon}}, \bar{K}_\epsilon^{\text{SGD}}, \bar{\rho}_K, \bar{\rho}_F, \bar{\sigma}_{\log}^{\text{Muon}}, \bar{\sigma}_{\log}^{\text{SGD}}) \right\}_{p \in \mathcal{P}}
$$

---

### 5.4 Phase 4: Hypothesis Testing and Inference

**Objective**: Execute statistical tests for H1--H5 based on aggregated data.

**Procedure**:

For each hypothesis $H_i$, $i = 1, \ldots, 5$:

**Step 1: Construct test statistic.**

Following the definitions in Section 4, construct $T^{(i)}$ from $\mathcal{D}_{\text{agg}}$ and $\mathcal{D}_{\text{raw}}$.

**Step 2: Determine null distribution.**

Under $H_0^{(i)}$, $T^{(i)}$ follows (asymptotically or exactly):

- H1, H4, H5: t-distribution, $df = R - 1 = 9$
- H2: Welch's t-distribution, approximate degrees of freedom
- H3: Paired t-distribution, $df = n_{\text{pair}} - 1 = 39$

**Step 3: Compute p-value.**

$$
p^{(i)} = \mathbb{P}(T \geq T_{\text{obs}}^{(i)} \mid H_0^{(i)}) \quad \text{or} \quad \mathbb{P}(T \leq T_{\text{obs}}^{(i)} \mid H_0^{(i)})
$$

depending on the direction of the alternative hypothesis.

**Step 4: Multiple testing correction.**

For the 5 hypotheses, apply the Holm--Bonferroni procedure:

1. Sort p-values: $p_{(1)} \leq p_{(2)} \leq \cdots \leq p_{(5)}$
2. Find the smallest $j$ such that $p_{(j)} > \alpha / (5 - j + 1)$
3. Reject all $p_{(i)}$ satisfying $i < j$

**Step 5: Effect size calculation.**

For each rejected hypothesis, report Cohen's d effect size:

$$
d^{(i)} = \frac{|\bar{\Delta}^{(i)}|}{\hat{\sigma}^{(i)}}
$$

where $\bar{\Delta}^{(i)}$ is the sample mean of the corresponding difference, and $\hat{\sigma}^{(i)}$ is the pooled standard deviation.

**Output**: Hypothesis testing report

$$
\mathcal{R}_{\text{report}} = \left\{ (H_i, T_{\text{obs}}^{(i)}, p^{(i)}, \text{decision}, d^{(i)}) \right\}_{i=1}^5
$$

where $\text{decision} \in \{\text{Reject } H_0, \text{Fail to reject } H_0\}$.

---

