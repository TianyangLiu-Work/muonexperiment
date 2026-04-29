<!--
Original Document: Mathematical Foundations and Statistical Formalization of Experimental Design for the Muon (μ) Optimization Algorithm
This File: 15. Derived Statistics
Split Index: 17
-->

[TOC]

---

## 15. Derived Statistics

### 5.1 Efficiency and Speedup Ratios

| Symbol | Name | Mathematical Definition | Interpretation |
|:---:|:---|:---|:---|
| $\rho_K$ | **Iteration Efficiency Ratio** | $\rho_K = \frac{K_\epsilon^{\text{SGD}}}{K_\epsilon^{\text{Muon}}}$ | $\rho_K > 1$: Muon converges faster per iteration; $\rho_K = 1$: equivalent |
| $\rho_F$ | **FLOPs Efficiency Ratio** | $\rho_F = \frac{F_\epsilon^{\text{SGD}}}{F_\epsilon^{\text{Muon}}}$ | Overall efficiency accounting for per-step computational cost |
| $\rho_T$ | **Time Efficiency Ratio** | $\rho_T = \frac{T_{\text{wall}}^{\text{SGD}}}{T_{\text{wall}}^{\text{Muon}}}$ | Practical wall-clock speedup |
| $S_L$ | **Scalability Ratio** | $S_L = \frac{\rho_F(d_2)}{\rho_F(d_1)}$ ($d_2 > d_1$) | Trend of efficiency change with increasing size |

### 5.2 Convergence and Stability Metrics

| Symbol | Name | Mathematical Definition | Interpretation |
|:---:|:---|:---|:---|
| $\rho_{\text{conv}}(k)$ | **Pointwise Convergence Ratio** | $\rho_{\text{conv}}(k) = \frac{f_{k+1} - f^\star}{f_k - f^\star}$ | Local convergence rate; $\rho_{\text{conv}} < 1$ indicates convergence |
| $\bar{\rho}_{\text{conv}}$ | **Average Convergence Ratio** | $\bar{\rho}_{\text{conv}} = \exp\left(\frac{1}{K_\epsilon}\sum_{k=0}^{K_\epsilon-1} \log \rho_{\text{conv}}(k)\right)$ | Geometric mean convergence rate |
| $\sigma_{\log}(k)$ | **Logarithmic Standard Deviation** | $\sigma_{\log}(k) = \sqrt{\frac{1}{n_s - 1}\sum_{s=0}^{n_s-1} (\log f_k^{(s)} - \overline{\log f_k})^2}$ | Stability across random seeds at iteration $k$ |
| $\bar{\sigma}_{\log}$ | **Average Logarithmic Standard Deviation** | $\bar{\sigma}_{\log} = \frac{1}{K_\epsilon}\sum_{k=1}^{K_\epsilon} \sigma_{\log}(k)$ | Overall convergence stability |
| $\text{CV}_K$ | **Coefficient of Variation for Iterations** | $\text{CV}_K = \frac{\sigma[K_\epsilon]}{\mathbb{E}[K_\epsilon]}$ | Relative variability of convergence time |

### 5.3 Effect Size Metrics

| Symbol | Name | Mathematical Definition | Purpose |
|:---:|:---|:---|:---|
| $d_{\text{Cohen}}$ | **Cohen's d** | $d_{\text{Cohen}} = \frac{\bar{K}_\epsilon^{\text{Muon}} - \bar{K}_\epsilon^{\text{SGD}}}{s_{\text{pooled}}}$ | Standardized mean difference; $|d| > 0.8$ indicates a large effect |
| $d_{\text{log}}$ | **Logarithmic Effect Size** | $d_{\text{log}} = \log_{10} \rho_K = \log_{10} K_\epsilon^{\text{SGD}} - \log_{10} K_\epsilon^{\text{Muon}}$ | Difference on a logarithmic scale; more suitable for right-skewed distributions |
| $A_{12}$ | **Vargha-Delaney A** | $A_{\text{VD}} = P(K_\epsilon^{\text{Muon}} < K_\epsilon^{\text{SGD}})$ | Non-parametric effect size; intuitive interpretation |
| $\Delta AUC$ | **Convergence Curve AUC Difference** | $\Delta AUC = \int_0^{T_{\max}} (f_k^{\text{SGD}} - f_k^{\text{Muon}}) \, dk$ | Overall convergence trajectory difference |

### 5.4 Confidence Intervals and Uncertainty Quantification

**Per-Configuration Confidence Intervals** (based on $n_s = 10$ repetitions):

| Statistic | Point Estimate | CI Method | Formula |
|:---:|:---|:---:|:---|
| $\mathbb{E}[K_\epsilon]$ | $\bar{K}_\epsilon = \frac{1}{n_s}\sum_{s=0}^{n_s-1} K_\epsilon^{(s)}$ | Bootstrap BCa | $[\hat{K}_{\alpha/2}^*, \hat{K}_{1-\alpha/2}^*]$ |
| $\mathbb{E}[\log K_\epsilon]$ | $\overline{\log K}_\epsilon$ | t-interval (small sample) | $\overline{\log K}_\epsilon \pm t_{n_s-1, \alpha/2} \cdot \frac{s_{\log K}}{\sqrt{n_s}}$ |
| $\rho_K$ | $\hat{\rho}_K$ | Bootstrap ratio interval | $\exp\left(\log \hat{\rho}_K \pm z_{\alpha/2} \cdot \text{SE}[\log \hat{\rho}_K]\right)$ |
| $p_{\text{conv}}$ | $\hat{p}_{\text{conv}} = \frac{1}{n_s}\sum_s \mathbb{I}_{\text{conv}}^{(s)}$ | Wilson score interval | See below |

**Wilson Score Interval** (for binary convergence indicators):

$$
\text{CI}_{p_{\text{conv}}} = \frac{\hat{p} + \frac{z^2}{2n_s} \pm z \sqrt{\frac{\hat{p}(1-\hat{p})}{n_s} + \frac{z^2}{4n_s^2}}}{1 + \frac{z^2}{n_s}}
$$

where $z = z_{1-\alpha/2}$, $\alpha = 0.05$.

**Cross-Configuration Aggregated Statistics**:

| Symbol | Name | Definition | Purpose |
|:---:|:---|:---|:---|
| $\bar{K}_\epsilon(\alpha, \pi, d)$ | Cell Mean | Mean over $n_s$ runs under the same configuration | Main effect estimation |
| $\text{Med}[K_\epsilon]$ | Cell Median | Median over $n_s$ runs under the same configuration | Robust location estimate against outliers |
| $\text{IQR}[K_\epsilon]$ | Interquartile Range | $Q_3 - Q_1$ | Robust dispersion estimate |
| $\hat{q}_{0.95}$ | 95th Quantile | Empirical 95th percentile | Worst-case assessment |

---

