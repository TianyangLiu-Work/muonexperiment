<!--
Original Document: Mathematical Foundations and Statistical Formalization of Muon (μ) Optimization Algorithm Experimental Design
This File: 21. Prioritization and Resource Allocation Recommendations
Split Index: 23
-->

[TOC]

---

## 21. Prioritization and Resource Allocation Recommendations

### 4.1 Priority Ranking (Scientific Value vs. Resource Cost)

Use a three-tier classification of **Must-Have**, **Should-Have**, and **Explore**:

| Priority | Experiment | Scientific Value | Resource Cost | Rationale for Classification |
|----------|------------|------------------|---------------|------------------------------|
| **P0 (Must-Have)** | E11 Multi-baseline Comparison | Extremely High | Medium | Without Adam and other baselines, one cannot claim Muon outperforms "standard" methods |
| **P0** | E13 Wall-clock Time | Extremely High | Low | Theoretical FLOPs advantages must be validated with actual timing |
| **P0** | E6 Noise Sensitivity | High | Medium | Real-world problems almost always have noise; noise-free conclusions may be severely over-optimistic |
| **P0** | E18 Condition Number Control | High | Medium | Validates Muon's core value proposition — the utilization of spectral structure |
| **P1 (Should-Have)** | E8 Over-/Undersampling | High | Medium | Determines whether conclusions apply to sampling-limited real-world scenarios |
| **P1** | E7 Rank Ratio Sweep | High | Medium | Determines the boundary of the range of advantages |
| **P1** | E16 Initialization Scale | Medium | Low | Low cost, high practical value for tuning guidance |
| **P1** | E20 Power Analysis | High | Low | Meta-analysis, low cost, significantly enhances statistical credibility |
| **P1** | E15 Large-scale Scalability | High | High | Determines feasibility of industrial deployment |
| **P2 (Exploratory)** | E9 Weight Decay | Medium | Medium | Industrially important, but belongs to the hyperparameter tuning level |
| **P2** | E10 Rectangular Matrices | Medium | Medium | Expands the shape space, but the core mechanism may remain unchanged |
| **P2** | E12 Hessian Dynamics | High | High | Deep mechanistic understanding, but costly and complex to analyze |
| **P2** | E14 Random SVD Trade-off | Medium | Medium | Escalate to P1 if E15 shows exact SVD is infeasible |
| **P2** | E17 Orthogonal Initialization | Low | Low | Low cost, but expected differences may be small |
| **P2** | E19 Distribution Generalization | Medium | High | If E18 has already validated the centrality of spectral structure, distribution may be secondary |

### 4.2 Recommended Execution Order

**Phase 1 (Execute Immediately, ~2 days)**:
1. E20 Power Analysis (using existing data, immediately determine whether additional replications are needed)
2. E11 Multi-baseline Comparison (establish the optimizer taxonomy benchmark)
3. E13 Wall-clock Time (validate theoretical efficiency)
4. E16 Initialization Scale (low cost, high practical value)

**Phase 2 (~2 days)**:
5. E6 Noise Sensitivity (core robustness)
6. E18 Condition Number Control (core mechanism validation)
7. E7 Rank Ratio Sweep (boundary of the range)
8. E8 Over-/Undersampling (sampling robustness)

**Phase 3 (~2 days)**:
9. E15 Large-scale Scalability (industrial feasibility)
10. E12 Hessian Dynamics (mechanism deepening)
11. E14 Random SVD Trade-off (if large-scale experiments require it)

**Phase 4 (Optional, ~1 day)**:
12. E9 Weight Decay
13. E10 Rectangular Matrices
14. E17 Orthogonal Initialization
15. E19 Distribution Generalization

### 4.3 Resource Budget Estimation

| Resource Type | Phase 1 | Phase 2 | Phase 3 | Phase 4 | Total |
|---------------|---------|---------|---------|---------|-------|
| CPU Core-Hours | 24 h | 40 h | 30 h | 15 h | 109 h |
| Peak Memory | 8 GB | 16 GB | 32 GB | 16 GB | — |
| Storage (Raw Data) | 2 GB | 4 GB | 3 GB | 2 GB | 11 GB |
| Human Analysis Time | 8 h | 12 h | 16 h | 8 h | 44 h |

**Acceleration Strategies**:
- All small- and medium-scale experiments ($d \leq 200$) can be fully parallelized (approximately 200 configurations can run simultaneously on 20 cores).
- E15 (large-scale) can be limited to $n=5$ replications, using the random SVD variant as the default Muon implementation.
- E12 (Hessian dynamics) can be downsampled to recording every 100 steps instead of every 50 steps.
- If E20 finds that the existing $n=10$ is insufficient, prioritize increasing the number of replications for E1–E5 to $n=20$, rather than adding new experiments.

### 4.4 Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation Strategy |
|------|------------|--------|---------------------|
| E11 shows Adam outperforms Muon | Medium | Extremely High | In the paper, explicitly position Muon's applicable scenarios (low-rank, ill-conditioned, high-precision) |
| E13 shows Muon wall-clock slower than SGD | High | Extremely High | Emphasize the random SVD variant (E14), rebrand as "theoretically motivated + approximate implementation" |
| E6 shows noise eliminates Muon's advantage | Medium | High | Establish SNR threshold theory, qualify the conditions of applicability |
| E15 shows $d=1000$ is infeasible | Low | Medium | Accept $d \leq 500$ as the valid range, industrial deployment requires approximate implementations |
| E20 shows $n=10$ has extremely low power | Medium | Medium | Increase the number of replications for existing experiments, rerun statistical tests |

---

---

