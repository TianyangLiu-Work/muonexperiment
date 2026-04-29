<!--
Original document: Mathematical Foundations and Statistical Formalization of Muon (μ) Optimization Algorithm Experimental Design
This file: Appendix
Split number: 24
-->

[TOC]

---

# Appendix

> The following appendices compile key symbols, experimental encoding rules, and the hypothesis testing framework from the full text, serving as a quick reference tool.


## A. Symbol Quick Reference

### A.1 Basic Symbols

| Symbol | Meaning | First Appearance |
|:---:|:---|:---:|
| $d$ | Matrix dimension | §1.1 |
| $r$ | Matrix rank (or target rank) | §1.1 |
| $m$ | Number of measurement samples | §1.1 |
| $L$ | Matrix factorization depth | §2.6.1 |
| $R$ | Number of experimental repetitions | §3.1.1 |
| $k$ | Iteration index | §1.1 |
| $K_\epsilon$ | Iteration complexity to reach precision $\epsilon$ | §1.2.1 |
| $F_\epsilon$ | Total FLOPs to reach precision $\epsilon$ | §4.1.4 |
| $X^\star$ | True / target matrix | §1.1 |
| $X^{(k)}$ | Parameter matrix at the $k$-th iteration | §1.1 |
| $G^{(k)}$ | Gradient matrix at step $k$ | §1.3.1 |
| $D^{(k)}$ | Search direction at step $k$ | §1.3.1 |
| $f(X)$ | Objective function | §1.1 |
| $f^\star$ | Optimal function value | §1.1 |
| $\delta_k$ | Function value gap $f(X^{(k)}) - f^\star$ | §1.1 |
| $\eta$ | Learning rate / step size | §1.3.1 |
| $\lambda$ | Weight decay coefficient | §2.2.3 |

### A.2 Norms and Spectral Analysis

| Symbol | Meaning | First Appearance |
|:---:|:---|:---:|
| $\|X\|_2$ | Spectral norm (largest singular value $\sigma_1$) | §2.1.1 |
| $\|X\|_F$ | Frobenius norm | §2.1.2 |
| $\|X\|_*$ | Nuclear norm (sum of singular values) | §2.1.3 |
| $\|X\|_{S_p}$ | Schatten $p$-norm | §2.1.4 |
| $\sigma_i(X)$ | The $i$-th singular value of $X$ | §2.1.1 |
| $\kappa_{\text{cond}}(X)$ | Matrix spectral condition number $\sigma_1/\sigma_r$ | §2.3.1 |
| $\kappa_{\text{sp}}(X)$ | Spectral ratio $\|X\|_2/\|X\|_F$ | §2.3.1 |
| $\kappa(f)$ | Function condition number $L/\mu$ | §1.3.2 |
| $r_{\text{eff}}(X)$ | Effective rank $\|X\|_F^2/\|X\|_2^2$ | §2.4.1 |
| $\text{rank}_\epsilon(X)$ | Numerical $\epsilon$-rank | §2.4.1 |

### A.3 Measurement and Operators

| Symbol | Meaning | First Appearance |
|:---:|:---|:---:|
| $\mathcal{A}$ | Linear measurement operator | §2.5.1 |
| $\mathcal{A}^*$ | Adjoint operator | §2.5.1 |
| $\delta_r(\mathcal{A})$ | RIP constant | §2.5.2 |
| $\mathcal{N}_{\text{spec}}(G)$ | Spectral normalization operator $UV^\top$ | §2.2.2 |
| $\mathcal{S}(G)$ | SVD normalization operator (same as $\mathcal{N}_{\text{spec}}$) | §7.3.1 |
| $\Pi(W_1, \ldots, W_L)$ | Product mapping $W_L \cdots W_1$ | §2.6.1 |

### A.4 Statistics and Tests

| Symbol | Meaning | First Appearance |
|:---:|:---|:---:|
| $\alpha$ | Significance level | §3.2.3 |
| $p$ | p-value | §3.2.3 |
| $\beta$ | Type II error probability | §3.4.1 |
| $\text{Power}$ | Statistical power $1-\beta$ | §3.4.1 |
| $\Delta_\epsilon^{(r)}$ | Performance difference $K_{\epsilon,r}^{(A)} - K_{\epsilon,r}^{(B)}$ | §3.2.1 |
| $\text{ES}$ | Standardized effect size | §3.4.1 |
| $d_{\text{Cohen}}$ | Cohen's d | §5.3 |
| $\text{CI}_{1-\alpha}$ | $(1-\alpha)$ confidence interval | §3.3.1 |
| $\text{FWER}$ | Family-wise error rate | §3.5.1 |
| $\text{FDR}$ | False discovery rate | §3.5.3 |

### A.5 Experimental Configuration Symbols

| Symbol | Meaning | First Appearance |
|:---:|:---|:---:|
| $\mathcal{E}$ | Experimental quintuple $(\mathcal{P}, \mathcal{A}, \mathcal{D}, \mathcal{M}, \mathcal{R})$ | §5.1 |
| $\mathcal{P}$ | Problem instance space | §5.1 |
| $\mathcal{P}_{MS}$ | Matrix sensing problem space | §5.1 |
| $\mathcal{P}_{MF}$ | Matrix factorization problem space | §5.1 |
| $\mathcal{A}$ | Algorithm space | §5.1 |
| $\mathcal{D}$ | Data / hyperparameter space | §5.1 |
| $\mathcal{M}$ | Metric space | §5.1 |
| $\mathcal{R}$ | Randomness space | §5.1 |
| $\rho_K$ | Iteration efficiency ratio $K_\epsilon^{\text{Muon}}/K_\epsilon^{\text{SGD}}$ | §7.3.4 |
| $\rho_F$ | FLOPs efficiency ratio $F_\epsilon^{\text{Muon}}/F_\epsilon^{\text{SGD}}$ | §7.3.4 |
| $\bar{\sigma}_{\log}$ | Mean log standard deviation (stability measure) | §7.3.4 |
| $\mathbb{I}_{\text{conv}}$ | Convergence flag | §12.2 |

### A.6 Supplementary Experiment Symbols

| Symbol | Meaning | First Appearance |
|:---:|:---|:---:|
| $\sigma_\epsilon$ | Noise standard deviation | §18. B3 |
| $\gamma = m/d^2$ | Sampling rate | §18. B4 |
| $\kappa_{\text{target}}$ | Target condition number | §18. B5 |
| $\sigma_{\text{init}}$ | Initialization scale | §18. C1 |
| $\delta_{\text{ortho}}$ | Orthogonality deviation | §13.3 |
| $\theta^{(k)}$ | Gradient-direction angle | §19. E12 |
| $\mathcal{A}^{(k)}$ | Hessian alignment measure | §19. E12 |
| $\tau_{\text{iter}}$ | Per-iteration wall-clock time | §19. E13 |
| $r_{\text{approx}}$ | Random SVD approximation rank | §19. E14 |
| $S_L$ | Scalability ratio | §15.1 |


## B. Experimental Configuration Encoding Scheme

### B.1 Experiment ID Encoding Rules

All experimental runs use a unified identifier encoding for convenient database storage, retrieval, and analysis.

**Encoding Format**:

```
{Problem}_{Shape}_d{dim}_r{rank}_L{depth}_{init}_s{seed}_a{algo}_lr{eta}_wd{lambda}_n{noise}
```

**Field Descriptions**:

| Field | Example Values | Description |
|:-----|:---------|:-----|
| `Problem` | `MS`, `MF` | Problem type: Matrix Sensing / Matrix Factorization |
| `Shape` | `sq`, `rect` | Shape: Square / Rectangular |
| `dim` | `100`, `200` | Matrix dimension ($d$ for square, $m \times n$ for rectangular) |
| `rank` | `10`, `full` | Target rank (numeric value or `full`) |
| `depth` | `2`, `3`, `4` | Factorization depth (valid for MF, omitted for MS) |
| `init` | `gauss`, `orth`, `spect`, `zero`, `bal` | Initialization strategy |
| `seed` | `0`–`49` | Random seed index |
| `algo` | `muon`, `sgd`, `adam`, `rmsprop`, `lbfgs` | Algorithm identifier |
| `eta` | `1e-3`, `1e-2`, `1e-1` | Learning rate |
| `lambda` | `0`, `1e-5`–`1e-2` | Weight decay coefficient |
| `noise` | `0`, `1e-4`–`1e-1` | Noise level |

**Encoding Examples**:

| Experiment Description | Experiment ID |
|:---------|:--------|
| Matrix sensing, square, $d=100$, low rank $r=10$, Gaussian initialization, seed 7, Muon, $\eta=0.01$ | `MS_sq_d100_r10_gauss_s7_a_muon_lr1e-2_wd0_n0` |
| Matrix factorization, $L=3$, $d=200$, full rank, balanced initialization, seed 12, SGD, $\eta=0.1$ | `MF_sq_d200_full_L3_bal_s12_a_sgd_lr1e-1_wd0_n0` |
| Supplementary experiment E6: noise sensitivity, $\sigma_\epsilon=10^{-2}$ | `MS_sq_d100_r10_gauss_s0_a_muon_lr1e-2_wd0_n1e-2` |
| Supplementary experiment E10: rectangular matrix, $100 \times 200$ | `MS_rect_d100x200_r10_gauss_s0_a_muon_lr1e-2_wd0_n0` |

### B.2 Configuration Dictionary and Response Dictionary Templates

**Configuration Dictionary (Config)**:

```python
config = {
    # Problem configuration
    "problem": "MS",              # MS or MF
    "shape": "square",          # square or rectangular
    "d": 100,                   # Matrix dimension
    "m": 300,                   # Number of measurements (for MS)
    "r": 10,                    # Target rank
    "L": None,                  # Factorization depth (for MF)

    # Initialization configuration
    "init_strategy": "gaussian", # gaussian / orthogonal / spectral / zero / balanced
    "init_scale": 1.0,          # Initialization scale factor

    # Algorithm configuration
    "algorithm": "Muon",        # Muon / SGD / Adam / RMSprop / L-BFGS
    "eta": 1e-2,                # Learning rate
    "lambda_wd": 0.0,           # Weight decay
    "momentum": 0.0,            # Momentum coefficient (if applicable)
    "svd_variant": "exact",     # exact / random / truncated (for Muon)

    # Experiment control
    "seed": 42,                 # Random seed
    "T_max": 100000,            # Maximum number of iterations
    "epsilon": 1e-6,            # Convergence tolerance
    "batch_size": "full",       # full or integer

    # Noise and measurements
    "sigma_eps": 0.0,           # Noise standard deviation
    "meas_dist": "gaussian",    # gaussian / rademacher / sparse / spherical / fjl
}
```

**Response Dictionary (Response)**:

```python
response = {
    # Core responses
    "K_eps": 15234,             # Convergence iteration count (T_max if not converged)
    "F_eps": 3.2e9,             # Total FLOPs to convergence
    "I_conv": 1,                # Convergence flag (0/1)
    "f_final": 1e-7,            # Final objective function value

    # Resource metrics
    "T_wall": 45.2,             # Wall-clock time (seconds)
    "M_peak": 512.0,            # Peak memory (MB)
    "tau_iter": 0.003,          # Average time per iteration

    # Accuracy metrics
    "grad_norm_final": 1e-5,    # Final gradient norm
    "f_dist_to_opt": 1e-7,      # Distance to optimal value

    # Covariates
    "kappa_sp_0": 0.32,         # Initial spectral ratio
    "kappa_cond_0": 15.7,       # Initial condition number
    "kappa_cond_star": 10.0,     # True condition number
    "r_eps": 18,                # Numerical epsilon-rank
    "mu_SR": 0.85,              # Strong convexity parameter (MS)

    # Dynamic tracking (optional, E12)
    "theta_k": [...],           # Gradient-direction angle sequence
    "A_k": [...],               # Hessian alignment sequence
    "kappa_2_k": [...],         # Hessian condition number dynamics
    "sigma_i_k": [...],         # Singular value trajectories
}
```

### B.3 Directory Organization Specification

```
experiment_data/
├── configs/                    # All configuration files (JSON)
│   ├── MS_sq_d100_r10_...
│   └── MF_sq_d200_full_...
├── raw/                        # Raw iteration trajectories
│   ├── MS_sq_d100_r10_.../loss_curve.npy
│   └── MS_sq_d100_r10_.../param_trajectory.npy
├── aggregated/                 # Aggregated statistical results
│   ├── summary_by_config.csv
│   └── summary_by_problem.csv
├── statistical_tests/          # Hypothesis testing results
│   ├── H1_results.json
│   └── H5_results.json
└── figures/                    # Auto-generated figures
    ├── convergence_curves/
    ├── efficiency_heatmaps/
    └── hypothesis_summary/
```


## C. Hypothesis Testing Summary Table

### C.1 Core Hypotheses H1–H5 (Existing Experiments)

| Hypothesis | Scientific Question | Null Hypothesis $H_0$ | Alternative Hypothesis $H_1$ | Test Statistic | Rejection Region | Distribution | Effect Size |
|:---|:---|:---|:---|:---|:---|:---|:---|
| **H1** | Muon converges unconditionally faster | $\mathbb{E}[\Delta_K(p)] \geq 0$ for at least one $p$ | $\mathbb{E}[\Delta_K(p)] < 0$ for all $p$ | $T^{(1)}(p) = \bar{\Delta}_K / (\hat{\sigma}_\Delta / \sqrt{R})$ | $T^{(1)} < -t_{0.05, 9} = -1.833$ | Paired t ($df=9$) | Cohen's $d$ |
| **H2** | Muon advantage is more pronounced at high spectral ratios | $\bar{\rho}_K^{\text{high}} \geq \bar{\rho}_K^{\text{low}}$ | $\bar{\rho}_K^{\text{high}} < \bar{\rho}_K^{\text{low}}$ | $T^{(2)} = \hat{\gamma} / \sqrt{s_{\text{low}}^2/n_{\text{low}} + s_{\text{high}}^2/n_{\text{high}}}$ | $T^{(2)} > t_{\alpha/2, df^\star}$ | Welch's t | $\hat{\gamma}$ |
| **H3a** | Advantage increases with depth $L=2 \to 3$ | $\bar{\rho}_K(3) \geq \bar{\rho}_K(2)$ | $\bar{\rho}_K(3) < \bar{\rho}_K(2)$ | $T^{(3a)} = (\bar{\rho}_K(3) - \bar{\rho}_K(2)) / (\hat{\sigma}_{2,3} / \sqrt{n_{\text{pair}}})$ | $T^{(3a)} < -t_{0.05, 39}$ | Paired t ($df=39$) | $d_{\text{Cohen}}$ |
| **H3b** | Advantage increases with depth $L=3 \to 4$ | $\bar{\rho}_K(4) \geq \bar{\rho}_K(3)$ | $\bar{\rho}_K(4) < \bar{\rho}_K(3)$ | $T^{(3b)} = (\bar{\rho}_K(4) - \bar{\rho}_K(3)) / (\hat{\sigma}_{3,4} / \sqrt{n_{\text{pair}}})$ | $T^{(3b)} < -t_{0.05, 39}$ | Paired t ($df=39$) | $d_{\text{Cohen}}$ |
| **H4** | Muon is more stable | $\mathbb{E}[S(\text{Muon})] \leq \mathbb{E}[S(\text{SGD})]$ | $\mathbb{E}[S(\text{Muon})] > \mathbb{E}[S(\text{SGD})]$ | $T^{(4)} = (\bar{r}_\sigma - 1) / (\hat{\sigma}_{r_\sigma} / \sqrt{R})$ | $T^{(4)} > t_{0.05, 9}$ | Paired t ($df=9$) | $d_{\text{Cohen}}$ |
| **H5** | Muon total FLOPs efficiency is higher | $\mathbb{E}[\rho_F] \geq 1$ | $\mathbb{E}[\rho_F] < 1$ | $T^{(5)} = (\bar{\rho}_F - 1) / (\hat{\sigma}_{\rho_F} / \sqrt{R})$ | $T^{(5)} < -t_{0.05, 9} = -1.833$ | Paired t ($df=9$) | $d_{\text{Cohen}}$ |

### C.2 Supplementary Experiments E6–E20

| ID | Experiment Name | Core Hypothesis | Null Hypothesis $H_0$ | Alternative Hypothesis $H_1$ | Test Statistic | Rejection Region | Test Type |
|:---|:---|:---|:---|:---|:---|:---|:---|
| **E6** | Noise sensitivity | Muon remains faster under noise | $\mathbb{E}[\log K_\epsilon^{\text{Muon}}] = \mathbb{E}[\log K_\epsilon^{\text{SGD}}]$ | $\mathbb{E}[\log K_\epsilon^{\text{Muon}}] < \mathbb{E}[\log K_\epsilon^{\text{SGD}}]$ | $T^{(d,\sigma)} = \bar{D} / (S_D / \sqrt{n})$ | $T < -t_{0.95, n-1}$ | Paired log t |
| **E7** | Rank ratio sweep | A sweet spot exists for $r/d$ | Algorithm$\times(r/d)$ interaction is not significant | There exists $(r/d)^*$ maximizing the difference | $F_{\text{int}} = \text{MS}_{\text{Alg} \times r/d} / \text{MS}_{\text{Error}}$ | $F > F_{\alpha, df_1, df_2}$ | Two-way ANOVA |
| **E8** | Over-/under-sampling | Sampling rate modulates advantage | $P(\delta_{\text{conv}}^{\text{Muon}}=1) = P(\delta_{\text{conv}}^{\text{SGD}}=1)$ | $P(\delta_{\text{conv}}^{\text{Muon}}=1) > P(\delta_{\text{conv}}^{\text{SGD}}=1)$ | $Z = (\hat{p}_M - \hat{p}_S) / \sqrt{\hat{p}(1-\hat{p})(2/n)}$ | $Z > z_{0.95}$ | Proportion z-test |
| **E9** | Weight decay ablation | $\lambda$ changes relative algorithm dynamics | $\mathbb{E}[\log K_\epsilon^{\text{Muon}}] = \mathbb{E}[\log K_\epsilon^{\text{SGD}}]$ for all $\lambda$ | There exists $\lambda^*$ maximizing the difference | $F_{\text{Alg} \times \lambda} = \text{MS}_{\text{Alg} \times \lambda} / \text{MS}_{\text{Error}}$ | $F > F_{\alpha, df_1, df_2}$ | Repeated measures ANOVA |
| **E10** | Rectangular matrices | Rectangularity does not eliminate advantage | $\mathbb{E}[K_\epsilon^{\text{Muon}} - K_\epsilon^{\text{SGD}}] = 0$ | Rectangularity does not eliminate (or enhances) advantage | $T_\alpha = \bar{D}_\alpha / (S_{D_\alpha} / \sqrt{n})$ | $\|T\| > t_{0.975, n-1}$ | Paired t |
| **E11** | Multi-baseline comparison | Muon outperforms all baselines | $\mathbb{E}[\log K_\epsilon^{\text{Muon}}] = \mathbb{E}[\log K_\epsilon^{\text{Algo}_i}]$ | $\mathbb{E}[\log K_\epsilon^{\text{Muon}}] < \mathbb{E}[\log K_\epsilon^{\text{Algo}_i}]$ | $T_i = \bar{D}_i / (S_{D_i} / \sqrt{n})$ | $T_i < -t_{0.05/M, n-1}$ | Holm-adjusted multiple t |
| **E12** | Hessian dynamics | Muon direction aligns better with Hessian | $\mathbb{E}[\theta_{\text{Muon}}^{(k)}] = \mathbb{E}[\theta_{\text{SGD}}^{(k)}]$ | There exists $k^*$ where Muon aligns better | Functional ANOVA F-test | $F > F_{\alpha, df}$ | Functional ANOVA |
| **E13** | Wall-clock time | SVD overhead offsets theoretical advantage | $T_\epsilon^{\text{Muon-Exact}} = T_\epsilon^{\text{SGD}}$ | $T_\epsilon^{\text{Muon-Exact}} > T_\epsilon^{\text{SGD}}$ | Wilcoxon signed-rank $W$ | $W < W_{\alpha, n}$ | Wilcoxon |
| **E14** | Random SVD trade-off | Approximate SVD preserves accuracy | $K_\epsilon^{\text{RandomSVD}} = K_\epsilon^{\text{Exact}}$ | There exists optimal $(r^*, q^*)$ for higher efficiency | Paired t (per parameter combination) | $T < -t_{0.95, n-1}$ | Response surface + t |
| **E15** | Large-scale scalability | A scale crossover point $d_{\text{cross}}$ exists | $T_\epsilon^{\text{Muon}}(d) \leq T_\epsilon^{\text{SGD}}(d)$ for all $d$ | There exists $d_{\text{cross}}$ where Muon degrades | Log-ratio t-test | $\log R_j > t_{0.95, n-1} \cdot \text{SE}$ | One-sample t |
| **E16** | Initialization scale | Muon has scale invariance | Muon $K_\epsilon$ does not depend on $\sigma_{\text{init}}$ | Muon $K_\epsilon$ depends on $\sigma_{\text{init}}$ | One-way ANOVA $F$ | $F > F_{\alpha, df}$ | ANOVA |
| **E17** | Orthogonal/spectral initialization | Muon is less sensitive to initialization | Algorithm$\times$Initialization has no interaction | Muon sensitivity is lower than SGD | $F_{\text{int}} = \text{MS}_{\text{Alg} \times \text{Init}} / \text{MS}_{\text{Error}}$ | $F > F_{\alpha, df}$ | Two-way ANOVA |
| **E18** | Condition number control | Ill-conditioning amplifies Muon advantage | $\Delta(\kappa) = \text{const}$ for all $\kappa$ | $\Delta(\kappa)$ increases monotonically with $\kappa$ | Spearman $\rho_S = \text{corr}_{\text{rank}}(\Delta(\kappa), \log \kappa)$ | $\rho_S > \rho_{\text{crit}}$ | Spearman |
| **E19** | Matrix distribution generalization | Conclusions do not depend on Gaussian assumption | Measurement distribution does not affect $\Delta K$ | There exists a distribution making $\Delta K$ significantly different | Kruskal-Wallis $H$ | $H > \chi^2_{0.95, 4}$ | Kruskal-Wallis |
| **E20** | Power and sample size | $n=10$ is insufficient | $n=10$ is sufficient to detect $\delta_{\min}=0.5$ | $n_{\min} > 10$ | Direct computation of $\hat{\delta}$ and $\hat{n}_{\min}$ | $\hat{n}_{\min} > 10$ | Bootstrap + power |

### C.3 Multiple Testing Correction Strategies

| Hypothesis Group | Number of Tests $M$ | Recommended Correction | Adjusted $\alpha^*$ | Description |
|:---|:---:|:---|:---|:---|
| H1–H5 (core hypotheses) | 5 | Holm-Bonferroni | $\alpha / (5 - j + 1)$ | Controls FWER, more powerful than Bonferroni |
| H3a + H3b (depth sub-tests) | 2 | Bonferroni | $\alpha / 2 = 0.025$ | Few sub-tests, conservative correction |
| E6–E20 (exploratory supplementary) | 15 | Benjamini-Hochberg | $p_{(i)} \leq \frac{i}{M} \cdot 0.10$ | Controls FDR = 0.10, suitable for exploratory analysis |
| E11 (multi-baseline) | 5 (algorithm$\times$problem) | Holm stepwise | $\alpha / (5 - j + 1)$ | Multiple comparison correction |
| All-configuration pairwise comparison | $\gg 40$ | Benjamini-Hochberg | FDR = 0.10 | Large-scale exploratory scanning |

### C.4 Test Distribution Quick Reference

| Test Type | Statistic | Null Distribution | Degrees of Freedom | Applicability |
|:---|:---|:---|:---|:---|
| Paired t-test | $T = \bar{D} / (S_D / \sqrt{n})$ | $t_{n-1}$ | $df = n - 1$ | Paired differences approximately normal |
| Welch's t | $T = (\bar{X}_1 - \bar{X}_2) / \sqrt{s_1^2/n_1 + s_2^2/n_2}$ | Approximate t | Welch-Satterthwaite | Two groups with unequal variances |
| One-way ANOVA | $F = \text{MS}_{\text{between}} / \text{MS}_{\text{within}}$ | $F_{k-1, N-k}$ | Between $k-1$, within $N-k$ | Multi-group mean comparison |
| Two-way ANOVA | $F_{\text{int}} = \text{MS}_{AB} / \text{MS}_E$ | $F_{(a-1)(b-1), ab(n-1)}$ | See interaction term | Interaction effect test |
| Proportion z-test | $Z = (\hat{p}_1 - \hat{p}_2) / \sqrt{\hat{p}(1-\hat{p})(1/n_1 + 1/n_2)}$ | $\mathcal{N}(0,1)$ | $\infty$ | Large-sample binomial proportions |
| Wilcoxon signed-rank | $W = \sum \text{rank}(\|D_i\|) \cdot \mathbb{I}(D_i > 0)$ | Table lookup | $n$ | Non-normal paired differences |
| Kruskal-Wallis | $H = \frac{12}{N(N+1)}\sum \frac{R_i^2}{n_i} - 3(N+1)$ | $\chi^2_{k-1}$ | $df = k - 1$ | Multi-group nonparametric |
| Spearman correlation | $\rho_S = 1 - \frac{6\sum d_i^2}{n(n^2-1)}$ | Table lookup or approximate t | $df = n - 2$ | Monotonic relationship test |
| Functional ANOVA | $F = \text{MS}_{\text{Alg} \times \text{Time}} / \text{MS}_E$ | $F$ | See design | Time series curve comparison |

> **End of Appendix**. The symbol quick reference, configuration encoding scheme, and hypothesis testing summary table in this document should be used in conjunction with the main text to ensure consistency and reproducibility of experimental design, execution, and reporting.


