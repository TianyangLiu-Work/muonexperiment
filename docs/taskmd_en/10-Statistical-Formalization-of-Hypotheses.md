<!--
Original Document: Mathematical Foundations and Statistical Formalization of Experimental Design for the Muon (μ) Optimization Algorithm
This File: 8. Statistical Formalization of Hypotheses
Split Number: 10
-->

[TOC]

---

## 8. Statistical Formalization of Hypotheses

This chapter translates the original scientific hypotheses H1–H5 into rigorous statistical hypothesis pairs $(H_0, H_1)$. Each hypothesis pair includes: the null hypothesis ($H_0$), the alternative hypothesis ($H_1$), the test statistic, the rejection region, and statistical power.

All hypothesis tests are conducted at the significance level $\alpha = 0.05$, unless otherwise specified. The number of experimental repetitions is $R = 10$.

---

### 4.1 H1: Unconditional Convergence Advantage

**Original Statement:** On matrix sensing and matrix factorization problems, Muon's iterative convergence speed is consistently superior to that of SGD.

**Formalization:**

Let $\mathcal{P}_{test} \subseteq \mathcal{P}$ be the subset of test problems (covering all dimension, rank, and depth configurations). For each problem $p \in \mathcal{P}_{test}$ and random seed $s$, define the difference in the number of iterations:

$$
\Delta_K(p, s) = K_\epsilon^{\text{Muon}}(p, s) - K_\epsilon^{\text{SGD}}(p, s)
$$

Averaging over $R$ repetitions:

$$
\bar{\Delta}_K(p) = \frac{1}{R} \sum_{s=1}^R \Delta_K(p, s)
$$

**Statistical Hypothesis Pair H1:**

$$
\begin{aligned}
H_0^{(1)}: & \quad \mathbb{E}[\Delta_K(p)] \geq 0 \quad \text{for at least one } p \in \mathcal{P}_{test} \\
H_1^{(1)}: & \quad \mathbb{E}[\Delta_K(p)] < 0 \quad \text{for all } p \in \mathcal{P}_{test}
\end{aligned}
$$

Equivalent formulation (in terms of the efficiency ratio $\rho_K$):

$$
\begin{aligned}
H_0^{(1)}: & \quad \exists \, p \in \mathcal{P}_{test}: \; \mathbb{E}[\rho_K(p)] \geq 1 \\
H_1^{(1)}: & \quad \forall \, p \in \mathcal{P}_{test}: \; \mathbb{E}[\rho_K(p)] < 1
\end{aligned}
$$

**Test Statistic:**

For each problem $p$, a paired one-sided t-test is employed:

$$
T^{(1)}(p) = \frac{\bar{\Delta}_K(p)}{\hat{\sigma}_{\Delta}(p) / \sqrt{R}}
$$

where $\hat{\sigma}_{\Delta}(p)$ is the sample standard deviation of $\Delta_K(p, s)$.

**Rejection Region:**

With degrees of freedom $df = R - 1 = 9$, reject $H_0^{(1)}$ when:

$$
T^{(1)}(p) < -t_{0.05, 9} = -1.833
$$

**Multiple Testing Correction:** As multiple problem instances are tested, the Bonferroni correction or the Holm-Bonferroni method is used to control the Family-Wise Error Rate (FWER) at $\alpha = 0.05$.

---

### 4.2 H2: Spectral Structure Conditional Advantage

**Original Statement:** When the ratio of the spectral norm to the Frobenius norm of a matrix, $\kappa_{sp}(X)$, is large, Muon's advantage is more pronounced.

**Formalization:**

**Definition 4.1 (Spectral Concentration).** For any matrix $X \in \mathbb{R}^{d \times d}$, define the spectral concentration index:

$$
\kappa_{sp}(X) = \frac{\|X\|_2 \cdot \sqrt{d}}{\|X\|_F} = \frac{\sigma_{\max}(X) \cdot \sqrt{d}}{\sqrt{\sum_{i=1}^d \sigma_i^2(X)}}
$$

where $\|X\|_2 = \sigma_{\max}(X)$ is the spectral norm (largest singular value). From the norm inequality $\|X\|_F \leq \sqrt{d} \|X\|_2$, it follows that $\kappa_{sp}(X) \in [1, \sqrt{d}]$. When $\kappa_{sp}(X) \to 1$, energy is uniformly distributed across all singular value directions; when $\kappa_{sp}(X) \to \sqrt{d}$, energy is concentrated in a single singular value direction.

For the target matrix $X^\star$, compute its spectral concentration $\kappa_{sp}(X^\star)$ and partition all test problems into high and low groups based on this value:

$$
\mathcal{P}_{\text{low}} = \left\{ p \; \middle| \; \kappa_{sp}(X^\star) \leq \kappa_{\text{med}} \right\}, \quad \mathcal{P}_{\text{high}} = \left\{ p \; \middle| \; \kappa_{sp}(X^\star) > \kappa_{\text{med}} \right\}
$$

where $\kappa_{\text{med}}$ is the median spectral concentration across all test problems.

Define the average efficiency gain of Muon relative to SGD in each group:

$$
\bar{\rho}_K^{\text{low}} = \frac{1}{|\mathcal{P}_{\text{low}}|} \sum_{p \in \mathcal{P}_{\text{low}}} \mathbb{E}[\rho_K(p)], \quad \bar{\rho}_K^{\text{high}} = \frac{1}{|\mathcal{P}_{\text{high}}|} \sum_{p \in \mathcal{P}_{\text{high}}} \mathbb{E}[\rho_K(p)]
$$

**Statistical Hypothesis Pair H2:**

$$
\begin{aligned}
H_0^{(2)}: & \quad \bar{\rho}_K^{\text{high}} \geq \bar{\rho}_K^{\text{low}} \\
H_1^{(2)}: & \quad \bar{\rho}_K^{\text{high}} < \bar{\rho}_K^{\text{low}}
\end{aligned}
$$

Equivalent formulation (in terms of advantage gain): Define $\gamma = \bar{\rho}_K^{\text{low}} - \bar{\rho}_K^{\text{high}}$, then:

$$
\begin{aligned}
H_0^{(2)}: & \quad \gamma \leq 0 \\
H_1^{(2)}: & \quad \gamma > 0
\end{aligned}
$$

**Test Statistic:**

An independent samples t-test (Welch's t-test, as the two groups may have unequal variances) is used. Let $\{\rho_K(p) : p \in \mathcal{P}_{\text{low}}\}$ and $\{\rho_K(p) : p \in \mathcal{P}_{\text{high}}\}$ be the efficiency ratio samples for the two groups:

$$
T^{(2)} = \frac{\hat{\gamma}}{\sqrt{\frac{s_{\text{low}}^2}{n_{\text{low}}} + \frac{s_{\text{high}}^2}{n_{\text{high}}}}}
$$

where $\hat{\gamma} = \bar{\rho}_K^{\text{low}} - \bar{\rho}_K^{\text{high}}$, $s^2$ is the within-group sample variance, and $n = |\mathcal{P}|$.

**Rejection Region:**

$$
T^{(2)} > t_{\alpha/2, df^\star}
$$

where $df^\star$ is approximated by the Welch-Satterthwaite equation.

---

### 4.3 H3: Depth Effect

**Original Statement:** In matrix factorization problems, as the number of factorization layers $L$ increases, Muon's advantage over SGD grows.

**Formalization:**

For MF problems, define the depth as $L \in \{2, 3, 4\}$. For each depth $L$, compute the average efficiency ratio across all dimensions $d$ and all random seeds:

$$
\bar{\rho}_K(L) = \frac{1}{|\mathcal{D}_L| \cdot R} \sum_{d \in \{50,100,200,500\}} \sum_{s=1}^R \rho_K(d, L, s)
$$

**Statistical Hypothesis Pair H3:**

Testing whether the advantage is monotonically non-decreasing with depth:

$$
\begin{aligned}
H_0^{(3)}: & \quad \bar{\rho}_K(4) \geq \bar{\rho}_K(3) \geq \bar{\rho}_K(2) \; \text{does not hold} \\
H_1^{(3)}: & \quad \bar{\rho}_K(4) \leq \bar{\rho}_K(3) \leq \bar{\rho}_K(2) \quad \text{(Muon advantage is monotonically non-decreasing with depth)}
\end{aligned}
$$

A more precise test can be decomposed into two one-sided tests:

- **H3a ($L=2 \to 3$):**

  $$
  \begin{aligned}
  H_0^{(3a)}: & \quad \bar{\rho}_K(3) \geq \bar{\rho}_K(2) \\
  H_1^{(3a)}: & \quad \bar{\rho}_K(3) < \bar{\rho}_K(2)
  \end{aligned}
  $$

- **H3b ($L=3 \to 4$):**

  $$
  \begin{aligned}
  H_0^{(3b)}: & \quad \bar{\rho}_K(4) \geq \bar{\rho}_K(3) \\
  H_1^{(3b)}: & \quad \bar{\rho}_K(4) < \bar{\rho}_K(3)
  \end{aligned}
  $$

**Test Statistic:** A paired t-test is used for each depth pair (paired by the same $d$ and $s$):

$$
T^{(3a)} = \frac{\bar{\rho}_K(3) - \bar{\rho}_K(2)}{\hat{\sigma}_{2,3} / \sqrt{n_{pair}}}
$$

where $n_{pair} = 4 \times 10 = 40$ (4 dimensions $\times$ 10 seeds).

---

### 4.4 H4: Stability Advantage

**Original Statement:** Muon is less sensitive to random initialization than SGD (its convergence trajectory is more stable).

**Formalization:**

For a given problem instance $p$ and both algorithms, compute the standard deviation of the logarithmic error over $R$ random initializations (Definition 3.13).

Define the stability difference:

$$
\Delta_\sigma(p) = \bar{\sigma}_{\log}^{\text{Muon}}(p) - \bar{\sigma}_{\log}^{\text{SGD}}(p)
$$

**Statistical Hypothesis Pair H4:**

$$
\begin{aligned}
H_0^{(4)}: & \quad \mathbb{E}[\Delta_\sigma(p)] \leq 0 \quad \text{(Muon is not more stable)} \\
H_1^{(4)}: & \quad \mathbb{E}[\Delta_\sigma(p)] > 0 \quad \text{(Muon is more stable; note the sign: a smaller } \sigma_{\log} \text{ implies a larger } \Delta_\sigma \text{)}
\end{aligned}
$$

**Revised Formalization (clearer definition):**

Let the stability measure be $S(\mathcal{A}) = 1 / \bar{\sigma}_{\log}^{(\mathcal{A})}$ (larger values indicate greater stability), then:

$$
\begin{aligned}
H_0^{(4)}: & \quad \mathbb{E}[S(\text{Muon})] \leq \mathbb{E}[S(\text{SGD})] \\
H_1^{(4)}: & \quad \mathbb{E}[S(\text{Muon})] > \mathbb{E}[S(\text{SGD})]
\end{aligned}
$$

Equivalently, defined in terms of the ratio of standard deviations (in the form of a coefficient of variation):

$$
\begin{aligned}
H_0^{(4)}: & \quad \mathbb{E}\left[\frac{\bar{\sigma}_{\log}^{\text{Muon}}}{\bar{\sigma}_{\log}^{\text{SGD}}}\right] \geq 1 \\
H_1^{(4)}: & \quad \mathbb{E}\left[\frac{\bar{\sigma}_{\log}^{\text{Muon}}}{\bar{\sigma}_{\log}^{\text{SGD}}}\right] < 1
\end{aligned}
$$

**Test Statistic:** Paired t-test (same problem instance, same random seed, different algorithms):

$$
T^{(4)} = \frac{\bar{r}_\sigma - 1}{\hat{\sigma}_{r_\sigma} / \sqrt{R}}
$$

where $r_\sigma(s) = \bar{\sigma}_{\log}^{\text{Muon}}(s) / \bar{\sigma}_{\log}^{\text{SGD}}(s)$.

---

### 4.5 H5: FLOPs Efficiency Advantage

**Original Statement:** Even though Muon has a larger per-iteration computational cost, its efficiency in terms of total FLOPs is still superior to that of SGD.

**Formalization:**

For each run $s$, compute the FLOPs efficiency ratio $\rho_F(s)$ (Definition 3.14).

**Statistical Hypothesis Pair H5:**

$$
\begin{aligned}
H_0^{(5)}: & \quad \mathbb{E}[\rho_F] \geq 1 \quad \text{(Muon is not superior to SGD in terms of total FLOPs)} \\
H_1^{(5)}: & \quad \mathbb{E}[\rho_F] < 1 \quad \text{(Muon is superior to SGD in terms of total FLOPs)}
\end{aligned}
$$

**Test Statistic:** A one-sided t-test on the paired samples $\{\rho_F(s)\}_{s=1}^R$:

$$
T^{(5)} = \frac{\bar{\rho}_F - 1}{\hat{\sigma}_{\rho_F} / \sqrt{R}}
$$

**Rejection Region:**

$$
T^{(5)} < -t_{0.05, R-1} = -1.833
$$

---
