<!--
Original Document: Mathematical Foundations and Statistical Formalization of Muon (μ) Optimization Algorithm Experimental Design
This File: 16. Functional Dependencies and Causal Structure Among Variables
Split Index: 18
-->

[TOC]

---

## 16. Functional Dependencies and Causal Structure Among Variables

### 6.1 Causal DAG (Directed Acyclic Graph)

The following is a textual description of the causal structure of experimental variables:

```
Experimental Design Layer ──► Problem Generation Layer ──► Algorithm Execution Layer ──► Observation Response Layer

Experimental Design Layer:
  (α, π, d, r, L, ι, η, λ, σ_ϵ, s) ──► Problem Generation Layer

Problem Generation Layer:
  s ──► X* ──► κ_cond^* ──► Problem Hardness
  s ──► {A_i} ──► μ_SR, M_SR ──► Problem Geometry
  X*, {A_i} ──► y ──► Measurement Vector
  
  Problem Hardness, Problem Geometry ──► Theoretical Convergence Rate ──► Algorithm Execution Layer

Algorithm Execution Layer:
  (α, η, λ) ──► Update Rule
  s ──► W^(0) ──► κ_sp^(0), κ_cond^(0) ──► Initial Conditions
  Update Rule + Initial Conditions + Problem Instance ──► {X^(k)}_(k=0)^T ──► Convergence Process

Observation Response Layer:
  {X^(k)} ──► K_ϵ, F_ϵ, I_conv, f_final
  {X^(k)} ──► {σ_i(X_k)} ──► Δ_sp, R(k)
  Convergence Process ──► T_wall, M_peak
  Cross-Seed Aggregation ──► σ_log(k), σ̄_log
```

### 6.2 Main Effects Model

For the primary response variable $Y \in \{\log K_\epsilon, \log F_\epsilon, \mathbb{I}_{\text{conv}}\}$, we establish a **linear mixed-effects model**:

$$
\begin{aligned}
Y_{i,j,k,\ell} &= \mu + \underbrace{\tau_{\alpha_i}}_{\text{algorithm main effect}} + \underbrace{\tau_{\pi_j}}_{\text{problem main effect}} + \underbrace{\tau_{d_k}}_{\text{dimension main effect}} + \underbrace{\tau_{r_\ell}}_{\text{rank main effect}} \\
&\quad + \underbrace{\tau_{\alpha_i \times \pi_j}}_{\text{algorithm-problem interaction}} + \underbrace{\tau_{\alpha_i \times d_k}}_{\text{algorithm-dimension interaction}} + \underbrace{\tau_{\pi_j \times d_k}}_{\text{problem-dimension interaction}} \\
&\quad + \underbrace{\tau_{\alpha_i \times \pi_j \times d_k}}_{\text{three-way interaction}} + \underbrace{\gamma^\top \mathbf{c}}_{\text{covariate adjustment}} + \underbrace{\nu_{s}}_{\text{seed random effect}} + \underbrace{\varepsilon_{i,j,k,\ell,s}}_{\text{residual}}
\end{aligned}
$$

where:
- $\mu$: global intercept
- $\tau$ terms: fixed effects (main effects and interaction effects)
- $\mathbf{c}$: covariate vector ($\kappa_{\text{sp}}^{(0)}, \kappa_{\text{cond}}^{(0)}, \delta_{\text{ortho}}$)
- $\nu_s \sim \mathcal{N}(0, \sigma_s^2)$: random seed effect
- $\varepsilon \sim \mathcal{N}(0, \sigma_\varepsilon^2)$: independent residual

### 6.3 Interaction Effects to be Tested

| Interaction Term | Test Priority | Scientific Hypothesis | Test Method |
|:---|:---:|:---|:---|
| $\alpha \times \pi$ | **High** | Muon's advantage is more pronounced on MF than on MS | Two-way ANOVA; report stratified results if significant |
| $\alpha \times d$ | **High** | Muon's spectral normalization advantage amplifies at larger dimensions | Stratified comparison by $d$; test slope differences |
| $\alpha \times r$ | **High** | Muon exploits low-rank structure more effectively | Test on low-rank vs. full-rank subsamples separately |
| $\alpha \times L$ | Medium (MF sub-experiment) | Muon's depth scalability is superior to SGD | Stratified ANOVA in MF sub-experiment |
| $\alpha \times \iota$ | Medium ($L=2$ sub-experiment) | Different sensitivity to initialization for the two algorithms | Three-factor analysis in $L=2$ sub-experiment |
| $\alpha \times \eta$ | Medium | Different sensitivity to learning rate for the two algorithms | Learning rate as a moderator variable analysis |
| $\pi \times d$ | Medium | Different dimension scalability across problem types | Two-factor analysis |
| $\alpha \times \pi \times d$ | **Exploratory** | Algorithm-problem-dimension three-way interaction | Test if second-order effects are significant; requires increased sample size |

### 6.4 Conditional Independence Statements

The following independence relationships hold conditional on the parent nodes:

1. **Given the problem instance, the algorithm response is independent of the seed**:
   $$Y \perp\!\!\perp s \mid X^\star, \{A_i\}, \alpha, \eta, \lambda$$

2. **Given the initial conditions, convergence behavior is independent of initialization details**:
   $$K_\epsilon \perp\!\!\perp \iota \mid \kappa_{\text{sp}}^{(0)}, \kappa_{\text{cond}}^{(0)}, \alpha, \eta$$

3. **Responses are independent across different problem instances**:
   $$Y^{(s_1)} \perp\!\!\perp Y^{(s_2)} \mid \text{all factors}, \quad s_1 \neq s_2$$

### 6.5 Summary of Functional Dependency Relationships

| Response Variable | Direct Parents | Functional Form |
|:---|:---|:---|
| $K_\epsilon$ | $\alpha, \pi, d, r, L, \eta, \kappa_{\text{cond}}^\star, \kappa_{\text{sp}}^{(0)}, \mu_{\text{SR}}$ | $K_\epsilon = g(\alpha, \pi, d, r, L) \cdot h(\eta) \cdot q(\text{problem hardness})$ |
| $F_\epsilon$ | $K_\epsilon, \alpha, \pi, d, L$ | $F_\epsilon = K_\epsilon \times \mathcal{C}_{\text{per-iter}}(\alpha, \pi, d, L)$ |
| $\mathbb{I}_{\text{conv}}$ | $\alpha, \pi, d, r, L, \eta, T_{\max}$ | $\mathbb{I}_{\text{conv}} = \mathbf{1}_{\{K_\epsilon \leq T_{\max}\}}$ |
| $\bar{\sigma}_{\log}$ | $\{f_k^{(s)}\}_{k,s}$ | $\bar{\sigma}_{\log} = \frac{1}{K}\sum_k \text{SD}_s[\log f_k^{(s)}]$ |
| $T_{\text{wall}}$ | $F_\epsilon, \text{HW}$ | $T_{\text{wall}} = F_\epsilon / \text{FLOP-rate}(\text{HW})$ |

---
