<!--
Original document: Mathematical Foundations and Statistical Formalization of Muon ($\mu$) Optimization Algorithm Experimental Design
This file: 3. Foundations of Statistical Inference
Split index: 04
-->

[TOC]

---

## 3. Foundations of Statistical Inference

### 3.1 Framework of Randomized Experiments

#### 3.1.1 Data Generating Process (DGP)

The statistical validity of the experiment relies on a clearly defined Data Generating Process (DGP). This experiment involves three sources of randomness: measurement matrices, observation noise, and parameter initialization.

**Definition 3.1** (Data Generating Process). The DGP for the matrix sensing experiment is defined as a quintuple
$$
\mathcal{P}_{MS} := (d, r, m, X^\star, \{A_i\}, \epsilon),
$$
where:
- $d$: matrix dimension
- $r$: rank of the target matrix ($r \ll d$)
- $m$: number of measurement samples
- $X^\star \in \mathbb{R}^{d \times d}$: true low-rank matrix with $\text{rank}(X^\star) = r$
- $A_i \overset{i.i.d.}{\sim} \mathcal{D}_A$: random measurement matrices
- $\epsilon_i \overset{i.i.d.}{\sim} \mathcal{D}_\epsilon$: observation noise

The observations are $y_i = \langle A_i, X^\star \rangle + \epsilon_i$, $i = 1, \ldots, m$.

The DGP for the matrix factorization experiment is
$$
\mathcal{P}_{MF} := (d, L, X^\star, \{W_\ell^{(0)}\}),
$$
where $L$ is the factorization depth and $\{W_\ell^{(0)}\}$ are the random initializations.

**Definition 3.2** (Random Seed Control). The experiment controls randomness through the seed $s \in \mathbb{Z}$ of a pseudorandom number generator. Denote by $\xi(s)$ the joint realization of all random variables determined by seed $s$. An experiment with a fixed seed is deterministic; varying the seed produces replications of a randomized experiment.

#### 3.1.2 Random Measurement Matrices

**Definition 3.3** (Gaussian Measurement Matrix). Each element of a standard Gaussian measurement matrix is independently distributed as a standard normal:
$$
[A_i]_{jk} \overset{i.i.d.}{\sim} \mathcal{N}(0, 1), \quad j, k = 1, \ldots, d.
$$
Equivalently, $\text{vec}(A_i) \sim \mathcal{N}(0, I_{d^2})$.

**Property**: For any matrix $X$, $\langle A_i, X \rangle \sim \mathcal{N}(0, \|X\|_F^2)$. Therefore
$$
\mathbb{E}\left[\frac{1}{m}\|\mathcal{A}(X)\|_2^2\right] = \|X\|_F^2,
$$
i.e., $\mathcal{A}$ is isometric in expectation.

**Definition 3.4** (Sub-Gaussian Measurement Matrix). A measurement matrix is called **sub-Gaussian** if its elements are independent, zero-mean sub-Gaussian random variables with parameter $K > 0$. That is, for any $t > 0$:
$$
\mathbb{P}(|[A_i]_{jk}| > t) \leq 2 \exp(-t^2/K^2).
$$

#### 3.1.3 Noise Models

**Definition 3.5** (Additive Gaussian Noise). The elements of the observation noise $\epsilon \in \mathbb{R}^m$ are independently and identically distributed:
$$\epsilon_i \overset{i.i.d.}{\sim} \mathcal{N}(0, \sigma_n^2),$$
where $\sigma_n^2$ is the noise variance. The signal-to-noise ratio is defined as
$$
\text{SNR} := \frac{\|\mathcal{A}(X^\star)\|_2^2}{\mathbb{E}\|\epsilon\|_2^2} = \frac{\sum_{i=1}^m \langle A_i, X^\star \rangle^2}{m \sigma_n^2}.$$

**Definition 3.6** (Normalized Noise Level). More commonly used in experiments is the **relative noise level**:
$$\bar{\sigma}_n := \frac{\sigma_n}{\|X^\star\|_F},$$
which eliminates the influence of the target matrix scale and facilitates comparison across different problems.

### 3.2 Hypothesis Testing Formulation

#### 3.2.1 Basic Hypothesis Testing Framework

The core statistical question of this experiment is to compare the convergence performance of Muon and SGD. Algorithm $A$ (Muon) is compared against the baseline algorithm $B$ (SGD) under various experimental conditions.

**Definition 3.7** (Algorithm Performance Random Variable). For a given problem instance $P$ and random seed $s$, define the number of iterations for algorithm $A$ to reach accuracy $\epsilon$ as $K_\epsilon^{(A)}(P, s)$. Over $R$ independent replications, we obtain samples
$$\{K_{\epsilon,r}^{(A)}\}_{r=1}^R, \quad \{K_{\epsilon,r}^{(B)}\}_{r=1}^R.$$

**Definition 3.8** (Performance Difference). Define the **performance difference** of algorithm $A$ relative to algorithm $B$ as
$$\Delta_\epsilon^{(r)} := K_{\epsilon,r}^{(A)} - K_{\epsilon,r}^{(B)}.$$
A negative value indicates that algorithm $A$ is faster.

#### 3.2.2 Paired Comparison Test

**Definition 3.9** (Null and Alternative Hypotheses). For paired differences $\{\Delta_\epsilon^{(r)}\}_{r=1}^R$, a two-sided hypothesis test is formulated:
- **Null hypothesis** $H_0$: there is no systematic difference between the two algorithms, i.e., $\mathbb{E}[\Delta_\epsilon] = 0$
- **Alternative hypothesis** $H_1$: there is a significant difference between the two algorithms, i.e., $\mathbb{E}[\Delta_\epsilon] \neq 0$

If the focus is on whether Muon is strictly better than SGD, a one-sided test is used:
- $H_0^{\text{one-sided}}$: $\mathbb{E}[\Delta_\epsilon] \geq 0$ (Muon is not faster)
- $H_1^{\text{one-sided}}$: $\mathbb{E}[\Delta_\epsilon] < 0$ (Muon is faster)

#### 3.2.3 Test Statistic and p-value

**Definition 3.10** (Paired t-Test Statistic). Let
$$\bar{\Delta} := \frac{1}{R}\sum_{r=1}^R \Delta_\epsilon^{(r)}, \quad S_\Delta^2 := \frac{1}{R-1}\sum_{r=1}^R (\Delta_\epsilon^{(r)} - \bar{\Delta})^2.$$
Under $H_0$ and assuming $\Delta_\epsilon^{(r)} \overset{i.i.d.}{\sim} \mathcal{N}(\mu_\Delta, \sigma_\Delta^2)$, the test statistic is
$$T := \frac{\bar{\Delta} - \mu_0}{S_\Delta / \sqrt{R}} \sim t_{R-1},$$
where $\mu_0 = 0$ is the expected value under $H_0$, and $t_{R-1}$ denotes the t-distribution with $R-1$ degrees of freedom.

**Definition 3.11** (p-value). The p-value for the two-sided test is
$$p := 2 \cdot \min\left\{\mathbb{P}(t_{R-1} \leq T), \mathbb{P}(t_{R-1} \geq T)\right\} = 2 \cdot \left(1 - F_{t_{R-1}}(|T|)\right),$$
where $F_{t_{R-1}}$ is the cumulative distribution function of the t-distribution.

**Decision Rule**: Given significance level $\alpha \in (0, 1)$ (typically $\alpha = 0.05$ or $\alpha = 0.01$),
- If $p \leq \alpha$, reject $H_0$ and conclude that the difference is statistically significant
- If $p > \alpha$, fail to reject $H_0$; there is insufficient evidence to conclude a significant difference

**Definition 3.12** (Significance Level). The **Type I Error** probability is
$$\alpha := \mathbb{P}(\text{reject } H_0 \mid H_0 \text{ is true}).$$
That is, the probability of erroneously claiming that a difference exists.

**Physical Interpretation**: The p-value quantifies the probability of observing the current or a more extreme result under the null hypothesis. A small p-value implies that the observed performance difference is unlikely to be due to random fluctuation. In algorithm comparison, a low p-value provides statistical evidence for algorithmic differences.

#### 3.2.4 Nonparametric Test (Wilcoxon Signed-Rank Test)

When the distribution of differences is non-normal or contains outliers, a nonparametric test is employed.

**Definition 3.13** (Wilcoxon Signed-Rank Statistic). For paired differences $\{\Delta^{(r)}\}$:
1. Compute absolute values $|\Delta^{(r)}|$ and discard zero differences
2. Assign ranks $\text{rank}_r$ to $|\Delta^{(r)}|$ from smallest to largest
3. Compute the sum of ranks for positive differences $W^+ = \sum_{\Delta^{(r)} > 0} \text{rank}_r$ and for negative differences $W^- = \sum_{\Delta^{(r)} < 0} \text{rank}_r$
4. The test statistic is $W := \min(W^+, W^-)$

**Property**: The Wilcoxon test does not rely on normality assumptions and is more robust to outliers, making it suitable for potentially heavy-tailed distributions that may arise in algorithm comparisons.

### 3.3 Construction of Confidence Intervals

#### 3.3.1 Confidence Intervals for the Mean

**Definition 3.14** (Confidence Interval for Iteration Complexity). The sample mean of $K_\epsilon$ for algorithm $A$ is $\bar{K}^{(A)} = \frac{1}{R}\sum_{r=1}^R K_{\epsilon,r}^{(A)}$. Its **confidence interval** at confidence level $(1-\alpha)$ is
$$\text{CI}_{1-\alpha}\left(\mathbb{E}[K_\epsilon^{(A)}]\right) := \left[\bar{K}^{(A)} - t_{R-1, \alpha/2} \cdot \frac{S_K^{(A)}}{\sqrt{R}}, \; \bar{K}^{(A)} + t_{R-1, \alpha/2} \cdot \frac{S_K^{(A)}}{\sqrt{R}}\right],$$
where $t_{R-1, \alpha/2}$ is the upper $\alpha/2$ quantile of the $t_{R-1}$ distribution, and $S_K^{(A)}$ is the sample standard deviation.

**Definition 3.15** (Confidence Interval for Log-Gap). For the sequence of optimization gaps $\{\delta_k^{(r)}\}_{r=1}^R$, define the sample mean of log-gaps and its confidence interval:
$$\overline{\log \delta_k} := \frac{1}{R}\sum_{r=1}^R \log \delta_k^{(r)},$$
$$\text{CI}_{1-\alpha}(\log \delta_k) := \left[\overline{\log \delta_k} - z_{\alpha/2} \cdot \frac{s_{\log, k}}{\sqrt{R}}, \; \overline{\log \delta_k} + z_{\alpha/2} \cdot \frac{s_{\log, k}}{\sqrt{R}}\right],$$
where $s_{\log, k}^2 = \frac{1}{R-1}\sum_{r=1}^R (\log \delta_k^{(r)} - \overline{\log \delta_k})^2$, and $z_{\alpha/2}$ is the standard normal quantile.

#### 3.3.2 Convergence Curve Band for Function Value Gap

**Definition 3.16** (Pointwise Confidence Band). For each step $k$, define the pointwise $(1-\alpha)$ confidence band for the function value gap as
$$\mathcal{C}_k := \left[\bar{\delta}_k - z_{\alpha/2} \cdot \frac{s_{\delta,k}}{\sqrt{R}}, \; \bar{\delta}_k + z_{\alpha/2} \cdot \frac{s_{\delta,k}}{\sqrt{R}}\right].$$
Note: $\{\mathcal{C}_k\}_{k=1}^K$ are **pointwise** confidence intervals, not **simultaneous** confidence bands. If simultaneous coverage of the entire convergence curve is required, Bonferroni correction or construction of simultaneous confidence bands should be used.

### 3.4 Sample Size and Statistical Power

#### 3.4.1 Power Analysis

**Definition 3.17** (Statistical Power). The **Type II Error** probability is
$$\beta := \mathbb{P}(\text{fail to reject } H_0 \mid H_1 \text{ is true}),$$
i.e., the probability of failing to detect a true difference. The **Statistical Power** is
$$\text{Power} := 1 - \beta = \mathbb{P}(\text{reject } H_0 \mid H_1 \text{ is true}).$$

**Definition 3.18** (Effect Size). The **standardized effect size** for hypothesis testing is defined as
$$\text{ES} := \frac{|\mathbb{E}[\Delta_\epsilon]|}{\sigma_\Delta} = \frac{|\mu_A - \mu_B|}{\sigma_{\text{pooled}}}.$$
Cohen's $d$ is a commonly used measure of effect size:
$$d := \frac{\bar{\Delta}}{S_\Delta}.$$
- $d \approx 0.2$: small effect
- $d \approx 0.5$: medium effect
- $d \approx 0.8$: large effect

**Theorem 3.1** (Relationship Between Power and Sample Size). For a paired t-test, given significance level $\alpha$, effect size $\text{ES}$, and desired power $1-\beta$, the required number of replications satisfies
$$R \geq \left(\frac{z_{\alpha/2} + z_{\beta}}{\text{ES}}\right)^2 + \frac{z_{\alpha/2}^2}{2}.$$

**Implications for Experiment Design**: When designing the experiment, $R$ must be sufficiently large to detect the effect size of interest. If the actual difference between algorithms corresponds to a large effect size ($d \geq 0.8$), then $R = 20 \sim 50$ is usually sufficient; if the effect size is medium, more replications are needed. This experiment uses $R = 50$ to ensure adequate power for detecting medium effect sizes.

### 3.5 Multiple Testing Problem and Correction Methods

#### 3.5.1 The Multiple Testing Problem

When comparing under multiple conditions (different dimensions $d$, different ranks $r$, different noise levels), multiple hypothesis tests are performed. Multiple testing increases the probability of committing a Type I error.

**Definition 3.19** (Family-Wise Error Rate). When performing $M$ independent tests, if the level of each test is $\alpha$, the probability of committing at least one Type I error is
$$\text{FWER} = \mathbb{P}(\text{at least one erroneous rejection}) = 1 - (1-\alpha)^M \approx M\alpha \quad (\text{for small } \alpha).$$

#### 3.5.2 Bonferroni Correction

**Definition 3.20** (Bonferroni Correction). For $M$ tests, the significance level of each test is adjusted to
$$\alpha^* = \frac{\alpha}{M}.$$
This guarantees that the family-wise error rate does not exceed $\alpha$.

**Property**: The Bonferroni correction is conservative, especially when the number of tests $M$ is large or when the tests are positively correlated.

#### 3.5.3 False Discovery Rate (FDR) Control

**Definition 3.21** (FDR). Suppose $M$ tests are performed, of which $M_0$ null hypotheses are true. The **False Discovery Rate** is defined as
$$\text{FDR} := \mathbb{E}\left[\frac{V}{\max(R, 1)}\right],$$
where $V$ is the number of erroneous rejections and $R$ is the total number of rejections.

**Definition 3.22** (Benjamini-Hochberg Procedure). For the ordered p-values $\{p_{(1)} \leq p_{(2)} \leq \cdots \leq p_{(M)}\}$ from $M$ tests, find
$$k^* = \max\left\{k : p_{(k)} \leq \frac{k \cdot \alpha}{M}\right\},$$
and reject the first $k^*$ hypotheses. This procedure guarantees $\text{FDR} \leq \alpha$.

**Implications for Experiment Design**: When the experiment involves multi-dimensional parameter sweeps (e.g., combinations of different $d$, $r$, $L$), multiple testing correction is required. Bonferroni is suitable when the number of tests is small; BH-FDR is suitable for large-scale sweeps, balancing explorability with statistical rigor.

### 3.6 Definitions of Random Variables and Distributional Assumptions

#### 3.6.1 Distribution of Random Initialization

**Definition 3.23** (Standard Random Initialization). In the matrix factorization experiment, the parameters of each layer are initialized as
$$[W_\ell^{(0)}]_{ij} \overset{i.i.d.}{\sim} \mathcal{N}\left(0, \sigma_w^2\right), \quad \ell = 1, \ldots, L.$$
A common choice is $\sigma_w^2 = 1/d$, which keeps the expected spectral norm of the product matrix bounded.

**Definition 3.24** (Xavier / Kaiming Initialization). For deep-network-style initialization:
- Xavier initialization: $\sigma_w^2 = 1/d$
- Kaiming initialization: $\sigma_w^2 = 2/d$

**Property**: Properly scaled random initialization ensures that the expected Frobenius norm of the product matrix $W_L^{(0)} \cdots W_1^{(0)}$ is bounded, avoiding gradient explosion or vanishing.

#### 3.6.2 Construction of the Target Matrix

**Definition 3.25** (Random Low-Rank Matrix). The target matrix constructed in the experiment is
$$X^\star := \sum_{i=1}^r \sigma_i^\star \cdot u_i^\star (v_i^\star)^\top,$$
where:
- $u_i^\star, v_i^\star \overset{i.i.d.}{\sim} \mathcal{N}(0, I_d)$ are orthonormalized (standard orthonormal columns are obtained via QR decomposition)
- $\sigma_i^\star = \sigma_1^\star \cdot \rho^{i-1}$, with $\rho \in (0, 1]$ the spectral decay rate

When $\rho = 1$, all singular values are equal (flat spectrum); when $\rho \ll 1$, the spectrum decays rapidly (strong low-rank structure).

#### 3.6.3 Formal Definitions of Gaussian Measurement Matrices and Noise

**Definition 3.26** (Formal Definition of Gaussian Measurement Operator). The complete probability space construction for the Gaussian measurement operator $\mathcal{A}: \mathbb{R}^{d \times d} \to \mathbb{R}^m$ is as follows:
Let $(\Omega, \mathcal{F}, \mathbb{P})$ be a probability space. The Gaussian measurement matrices are random variables
$$A_i: \Omega \to \mathbb{R}^{d \times d}, \quad i = 1, \ldots, m,$$
satisfying $[A_i(\omega)]_{jk} \overset{i.i.d.}{\sim} \mathcal{N}(0, 1)$. The noise random variables are
$$\epsilon_i: \Omega \to \mathbb{R}, \quad \epsilon_i \overset{i.i.d.}{\sim} \mathcal{N}(0, \sigma_n^2),$$
and $\{A_i\}$ and $\{\epsilon_i\}$ are mutually independent.

The observation random variables are
$$y_i(\omega) := \langle A_i(\omega), X^\star \rangle + \epsilon_i(\omega).$$

**Definition 3.27** (Joint Probability Space for Randomized Experiments). For $R$ independent replications, the joint probability space is the product space $(\Omega^R, \mathcal{F}^{\otimes R}, \mathbb{P}^{\otimes R})$. The random seed $s_r$ used in the $r$-th replication corresponds to a realization $\omega_r \in \Omega$ in the probability space.

**Definition 3.28** (Deterministic Equivalence and Expected Trajectory). For the random iterative sequence $\{X^{(k)}(\omega)\}$, define the **expected trajectory** as
$$\bar{X}^{(k)} := \mathbb{E}[X^{(k)}] = \int_\Omega X^{(k)}(\omega) \, d\mathbb{P}(\omega),$$
and the **expected function value gap** as
$$\bar{\delta}_k := \mathbb{E}[\delta_k] = \mathbb{E}[f(X^{(k)}) - f^\star].$$
In the experiment, $\bar{\delta}_k$ is approximated by the Monte Carlo average $\frac{1}{R}\sum_{r=1}^R \delta_k^{(r)}$.

---
