<!--
Original Document: Mathematical Foundations and Statistical Formalization of Experimental Design for the Muon (μ) Optimization Algorithm
This File: 20. Complete Experimental Matrix after Supplementation
Split Number: 22
-->

[TOC]

---

## 20. Complete Experimental Matrix after Supplementation

Integrate existing experiments (E1–E5) with supplementary experiments (E6–E20) into a unified factorial design matrix. Matrix columns represent experimental dimensions; rows represent experiment definitions.

### 3.1 Review of Existing Experiments (E1–E5)

| ID | Experiment Name | Problem Type | Algorithm | Dimension Parameters | Key Factors | Replicates | Core Hypotheses |
|------|----------|----------|------|----------|----------|--------|----------|
| E1 | Matrix Sensing Benchmark | MS | Muon, SGD | $d=50,100,200,500$; $r=d/10$ or full-rank | Dimension × Rank Structure | 10 | H1, H2 |
| E2 | Matrix Factorization Benchmark | MF-L=2,3,4 | Muon, SGD | $d=50,100,200$; L=2,3,4 | Depth × Dimension | 10 | H3 |
| E3 | Learning Rate Calibration | MS, MF | Muon, SGD | $d=100,200$ | $\eta$ Grid Search | 10 | H4 |
| E4 | Stability Analysis | MS, MF | Muon, SGD | $d=100$ | Initialization × Noise | 10 | H4, H5 |
| E5 | FLOPs Efficiency | MS, MF | Muon, SGD | $d=50,100,200,500$ | Theoretical Computation | — | H5 |

### 3.2 Supplementary Experiment Matrix (E6–E20)

| ID | Experiment Name | Coverage Dimensions | Problem Type | Algorithm Variants | New Factors | Replicates | Core Hypotheses | Relation to Existing |
|------|----------|----------|----------|----------|----------|--------|----------|------------|
| E6 | Noise Sensitivity | B3 | MS, MF | Muon, SGD | $\sigma_\epsilon \in \{0, 10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}\}$ | 20 | Noise Advantage Boundary | Extends E1/E2 |
| E7 | Rank-Ratio Sweep | B1 | MS | Muon, SGD | $r/d \in \{0.01, 0.05, 0.1, 0.2, 0.5, 1.0\}$ | 15 | Rank–Advantage Interaction | Extends E1 |
| E8 | Over/Undersampling | B4 | MS | Muon, SGD | $\gamma = m/d^2 \in \{0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0\}$ | 15 | Sampling-Rate–Advantage Curve | Extends E1 |
| E9 | Weight Decay Ablation | A3, D2 | MS, MF | Muon, SGD | $\lambda \in \{0, 10^{-5}, 10^{-4}, 10^{-3}, 10^{-2}\}$ | 15 | $\lambda$ Interaction Effect | New Dimension |
| E10 | Rectangular Matrices | B2 | MS, MF-rect | Muon, SGD | $(m,n)$ Six Shape Configurations | 15 | Shape Generalizability | Extends E1/E2 |
| E11 | Multi-Baseline Comparison | A6 | MS, MF | Muon, SGD, Adam, RMSprop, Momentum, L-BFGS | 6 Algorithms | 15 | Muon's Position in the Spectrum | Extends Baseline |
| E12 | Hessian Spectrum Dynamics | G1, G4 | MS, MF-L2 | Muon, SGD | Time-Series Tracking (every 50 steps) | 10 | Direction–Hessian Alignment | New Dimension |
| E13 | Wall-Clock Time | E1, E2 | MS | Muon-Exact, Muon-RandSVD, Muon-Trunc, SGD | Implementation Variant × Hardware | 10 | Theory–Practice Gap | Validates E5 |
| E14 | Randomized SVD Tradeoff | E3 | MS | Muon-Exact, Muon-RandomSVD | $(r_{\text{approx}}, q, p)$ Parameters | 10 | Accuracy–Speed Pareto Frontier | Extends E13 |
| E15 | Large-Scale Scalability | E4 | MS | Muon-Exact, Muon-RandSVD, SGD | $d \in \{500, 1000, 2000\}$ | 5 | Scalability Hard Wall | Extends E1 |
| E16 | Initialization Scale Sensitivity | C1, C3 | MS, MF | Muon, SGD | $\sigma_{\text{init}}$ Five Levels | 15 | Scale Invariance Validation | Extends E2 |
| E17 | Orthogonal/Spectral Initialization | C2 | MS, MF | Muon, SGD | Four Initialization Strategies | 15 | Initialization–Algorithm Interaction | Extends E2 |
| E18 | Condition Number Control | B5 | MS | Muon, SGD | $\kappa_{\text{target}} \in \{10, 10^2, \dots, 10^6\}$ | 15 | Ill-Conditioned Advantage Amplification | New Dimension |
| E19 | Matrix Distribution Generalization | H1, H3 | MS | Muon, SGD | Five Measurement Distributions × Three Target Spectral Patterns | 15 | Distributional Robustness | Extends E1 |
| E20 | Power and Sample Size | F1, F2, F3 | MS, MF | Muon, SGD | Sample Size Sweep + Bootstrap | — | Statistical Rigor | Meta-Analysis |

### 3.3 Integrated Super-Factorial Design

Represent all experiments uniformly as a high-dimensional factor space $\mathcal{E} = (\mathcal{P}, \mathcal{A}, \mathcal{D}, \mathcal{M}, \mathcal{R})$:

**Problem Space $\mathcal{P}$**:
$$
\mathcal{P} = \left\{ \begin{array}{l}
\text{Problem-Type} \in \{\text{MS}, \text{MF-L2}, \text{MF-L3}, \text{MF-L4}, \text{MF-rect}\} \\
\text{Shape} \in \{(d,d), (m,n)\} \\
\text{Dimension } d \in \{50, 100, 200, 500, 1000, 2000\} \\
\text{Rank-Ratio } r/d \in \{0.01, 0.05, 0.1, 0.2, 0.5, 1.0\} \\
\text{Sampling-Rate } \gamma \in \{0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0\} \\
\text{Noise-Level } \sigma_\epsilon \in \{0, 10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}\} \\
\text{Condition-Number } \kappa \in \{10, 10^2, 10^3, 10^4, 10^5, 10^6\} \\
\text{Measurement-Dist} \in \{\mathcal{N}, \text{Rad}, \text{Sparse}, \text{Sphere}, \text{FJL}\} \\
\text{Target-Spec} \in \{\text{hard-cutoff}, \text{poly-}\alpha, \text{exp-}\beta\}
\end{array} \right\}
$$

**Algorithm Space $\mathcal{A}$**:
$$
\mathcal{A} = \left\{ \begin{array}{l}
\text{Algorithm} \in \{\text{Muon-Exact}, \text{Muon-RandSVD}, \text{Muon-Trunc}, \\
\quad\quad\quad\quad\quad\quad \text{SGD}, \text{Momentum-SGD}, \text{Adam}, \text{RMSprop}, \text{L-BFGS}\} \\
\text{Learning-Rate } \eta \in [10^{-4}, 10^{-1}] \text{ (continuous or logarithmic grid)} \\
\text{Weight-Decay } \lambda \in \{0, 10^{-5}, 10^{-4}, 10^{-3}, 10^{-2}\} \\
\text{Momentum } \beta \in \{0.0, 0.5, 0.9, 0.99\} \text{ (if applicable)}
\end{array} \right\}
$$

**Data and Initialization Space $\mathcal{D}$**:
$$
\mathcal{D} = \left\{ \begin{array}{l}
\text{Seed } s \in \{1, 2, \dots, n_{\max}\} \\
\text{Init-Strategy} \in \{\text{Gaussian}, \text{Orthogonal}, \text{Spectral}, \text{Zero}\} \\
\text{Init-Scale } \sigma_{\text{init}} \in \{10^{-3}, 10^{-2}, 10^{-1}, 1, 10\}/\sqrt{d}
\end{array} \right\}
$$

**Metric Space $\mathcal{M}$**:
$$
\mathcal{M} = \left\{ \begin{array}{l}
K_\epsilon \text{ (convergence iterations)}, \quad F_\epsilon \text{ (total FLOPs)}, \quad T_\epsilon \text{ (wall-clock time)} \\
f_{\text{final}} \text{ (final loss)}, \quad \delta_{\text{conv}} \in \{0, 1\} \\
\bar{\sigma}_{\log} \text{ (mean log spectral ratio)}, \quad \kappa_{\text{sp}} \text{ (spectral condition number)} \\
\text{rank}_\epsilon \text{ (numerical rank)}, \quad \kappa_{\text{cond}} \text{ (condition number)} \\
\tau_{\text{iter}} \text{ (per-iteration time)}, \quad M_{\text{peak}} \text{ (peak memory)} \\
\theta^{(k)} \text{ (gradient–direction angle)}, \quad \mathcal{A}^{(k)} \text{ (Hessian alignment)} \\
\kappa_2^{(k)}(H) \text{ (Hessian condition number dynamics)}
\end{array} \right\}
$$

**Response and Inference Space $\mathcal{R}$**:
$$
\mathcal{R} = \left\{ \begin{array}{l}
\text{Hypothesis test results (} p \text{-values, effect sizes, confidence intervals)} \\
\text{ANOVA decomposition (main effects + interaction effects)} \\
\text{Power analysis (} n_{\min} \text{, actual power)} \\
\text{Response surface model parameters } \{\beta_j\} \\
\text{Ranking and recommendation (optimal algorithm configuration)}
\end{array} \right\}
$$

### 3.4 Total Experiment Configuration Estimate

| Experiment Group | Number of Configurations (Problem × Algorithm × Parameters × Replicates) | Estimated Runtime |
|--------|-------------------------------------|-------------|
| E1–E5 Existing | $\approx 2 \times 2 \times 10 \times 10 = 400$ | Completed |
| E6 Noise | $2 \times 2 \times 3 \times 5 \times 20 = 1{,}200$ | 12 h |
| E7 Rank Sweep | $1 \times 2 \times 3 \times 6 \times 15 = 540$ | 6 h |
| E8 Sampling Rate | $1 \times 2 \times 3 \times 7 \times 15 = 630$ | 8 h |
| E9 Weight Decay | $2 \times 2 \times 2 \times 5 \times 15 = 600$ | 10 h |
| E10 Rectangular | $2 \times 2 \times 6 \times 15 = 360$ | 5 h |
| E11 Multi-Baseline | $2 \times 6 \times 2 \times 15 = 360$ | 8 h |
| E12 Hessian Dynamics | $2 \times 2 \times 2 \times 10 = 80$ (time series) | 6 h |
| E13 Wall-Clock | $1 \times 4 \times 4 \times 10 = 160$ | 4 h |
| E14 Randomized SVD | $1 \times 2 \times 12 \times 10 = 240$ | 4 h |
| E15 Large-Scale | $1 \times 3 \times 3 \times 5 = 45$ | 12 h |
| E16 Init Scale | $2 \times 2 \times 2 \times 5 \times 15 = 600$ | 8 h |
| E17 Orthogonal Init | $2 \times 2 \times 2 \times 4 \times 15 = 480$ | 6 h |
| E18 Condition Number | $1 \times 2 \times 1 \times 6 \times 2 \times 15 = 360$ | 8 h |
| E19 Distribution Generalization | $1 \times 2 \times 2 \times 5 \times 3 \times 15 = 900$ | 10 h |
| E20 Power Analysis | Meta-analysis (using existing data) | 2 h |
| **Total** | **~6,055 runs** | **~109 h (4.5 days)** |

Note: Large-scale experiments (E15) and dynamic tracking (E12) account for the majority of the runtime; the remaining experiments can be parallelized.

---
