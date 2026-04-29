<!--
Original Document: Mathematical Foundations and Statistical Formalization of Muon (μ) Optimization Algorithm Experimental Design
This File: 12. Response Variables
Split Index: 14
-->

[TOC]

---

## 12. Response Variables

### 2.1 Response Variable Hierarchy

| Level | Category | Number of Variables | Purpose |
|:---:|:---|:---:|:---|
| **Primary** | Primary Endpoints | 4 | Core scientific conclusion support |
| **Secondary** | Secondary Endpoints | 5 | Auxiliary explanation and mechanism analysis |
| **Tertiary** | Exploratory Endpoints | 4 | Hypothesis generation and in-depth analysis |

### 2.2 Primary Response Variables

| Symbol | Name | Mathematical Definition | Domain | Type | Distribution Assumption | Experimental Purpose |
|:---:|:---|:---|:---:|:---:|:---:|:---|
| $K_\epsilon$ | **Convergence Iteration Count** | $K_\epsilon = \min\{ k \geq 0 : f_k \leq \epsilon \}$ | $\{0, 1, \ldots, T_{\max}\}$ | Discrete | Right-skewed, approximately log-normal | **Primary response**: Core metric of algorithmic efficiency |
| $F_\epsilon$ | **Convergence FLOPs** | $F_\epsilon = \sum_{k=0}^{K_\epsilon-1} \mathcal{C}_k$ | $\mathbb{R}_{\geq 0}$ | Continuous | Right-skewed, approximately log-normal | **Primary response**: Physical metric of computational cost |
| $\mathbb{I}_{\text{conv}}$ | **Convergence Indicator** | $\mathbb{I}_{\text{conv}} = \mathbf{1}_{\{K_\epsilon < T_{\max}\}}$ | $\{0, 1\}$ | Binary | Bernoulli($p_{\text{conv}}$) | **Primary response**: Binary indicator of algorithm reliability |
| $\bar{\sigma}_{\log}$ | **Log-Convergence Stability** | $\bar{\sigma}_{\log} = \frac{1}{K_\epsilon} \sum_{k=1}^{K_\epsilon} \sigma_{\log}(k)$ | $\mathbb{R}_{\geq 0}$ | Continuous | Unknown, nonparametric methods required | **Primary response**: Stability metric of the convergence path |

where:
- $f_k = f(X_k)$ is the objective function value at step $k$
- $\epsilon = 10^{-6}$ is the convergence threshold
- $\mathcal{C}_k$ is the computational cost (FLOPs) at step $k$
- $\sigma_{\log}(k)$ is the sample standard deviation of log-objective values across seeds at step $k$

### 2.3 Secondary Response Variables

| Symbol | Name | Mathematical Definition | Domain | Type | Distribution Assumption | Experimental Purpose |
|:---:|:---|:---|:---:|:---:|:---:|:---|
| $T_{\text{wall}}$ | **Wall-Clock Time** | $T_{\text{wall}} = t_{\text{end}} - t_{\text{start}}$ | $\mathbb{R}_{\geq 0}$ | Continuous | Right-skewed, hardware-dependent | Actual engineering deployment cost |
| $M_{\text{peak}}$ | **Peak Memory Usage** | $M_{\text{peak}} = \max_k \text{mem}(X_k)$ | $\mathbb{R}_{\geq 0}$ (MB/GB) | Continuous | Approximately constant + noise | Assessment for resource-constrained scenarios |
| $f_{\text{final}}$ | **Final Accuracy** | $f_{\text{final}} = f_{K_\epsilon}$ (if converged) or $f_{T_{\max}}$ (if truncated) | $\mathbb{R}_{\geq 0}$ | Continuous | Right-skewed | Alternative metric when convergence is not achieved |
| $\|\nabla f\|_{\text{final}}$ | **Final Gradient Norm** | $\|\nabla f\|_{\text{final}} = \|\nabla f(X_{K_\epsilon})\|_F$ | $\mathbb{R}_{\geq 0}$ | Continuous | Right-skewed | First-order optimality condition verification |
| $\Delta_{\text{sp}}$ | **Spectral Ratio Change** | $\Delta_{\text{sp}} = \kappa_{\text{sp}}(X^{(0)}) - \kappa_{\text{sp}}(X^{(K)})$ | $\mathbb{R}$ | Continuous | Unknown | Implicit regularization effect of the algorithm on spectral structure |

### 2.4 Exploratory Response Variables

| Symbol | Name | Mathematical Definition | Domain | Type | Experimental Purpose |
|:---:|:---|:---|:---:|:---:|:---|
| $\{X^{(k)}\}_{k=0}^{T}$ | **Parameter Trajectory** | Full iteration sequence | $\mathbb{R}^{d \times d \times (T+1)}$ | Sequence | Convergence path visualization, dynamics analysis |
| $\{\lambda_i(H_k)\}_{i=1}^{d^2}$ | **Hessian Spectrum** | Hessian matrix eigenvalues | $\mathbb{R}^{d^2}$ | Vector | Local geometry analysis, condition number evolution |
| $\{\sigma_i(X_k)\}_{i=1}^{d}$ | **Singular Value Trajectory** | SVD singular value sequence | $\mathbb{R}_{\geq 0}^{d \times (T+1)}$ | Matrix | Spectral dynamics, implicit low-rank regularization |
| $\mathcal{R}(k)$ | **Effective Rank** | $\mathcal{R}(k) = \|X_k\|_F^2 / \|X_k\|_2^2$ | $[1, d]$ | Continuous | Numerical rank evolution, implicit rank compression |

### 2.5 Mathematical Relationships Among Response Variables

**Functional Dependence of FLOPs on Iteration Count**:

$$
F_\epsilon(\alpha, \pi, d) = \mathcal{C}_{\text{per-iter}}(\alpha, \pi, d) \times K_\epsilon(\alpha, \pi, d)
$$

where the per-iteration computational cost is:

- **Muon** (MS): $\mathcal{C}_{\text{Muon}}^{\text{MS}} = \mathcal{O}(m \cdot d^2) + \mathcal{O}(d^3)$ (including spectral normalization SVD)
- **Muon** (MF): $\mathcal{C}_{\text{Muon}}^{\text{MF}} = \mathcal{O}(L \cdot d^3)$ (chained SVD for deep factorization)
- **SGD** (MS/MF): $\mathcal{C}_{\text{SGD}} = \mathcal{O}(m \cdot d^2)$ or $\mathcal{O}(L \cdot d^3)$ (no additional SVD overhead)

**Relationship Between Gradient Norm and Convergence Iteration Count** (in expectation):

$$
\mathbb{E}[K_\epsilon] \propto \frac{\log(f_0 / \epsilon)}{\log(1 / \rho_{\text{conv}})}
$$

where $\rho_{\text{conv}}$ is the per-step convergence rate (see \S5 Derived Statistics).

---
