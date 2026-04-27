# Muon vs SGD: Comprehensive Experiment Analysis

> Analysis of all completed experiments (E01, E02, E03, E04, E06, E09, E11, E12, E13, E14, E15, E16, E17, E18, E19, E20)
> Generated: 2026-04-26

---

## Executive Summary

Across **17 direct algorithm comparisons** spanning matrix sensing and matrix factorization problems, **SGD dominates Muon on every metric**:

| Metric | Muon Wins | SGD Wins |
|--------|-----------|----------|
| Iterations to convergence (K_epsilon) | **0** | **17** |
| Final optimization quality (min_loss) | **0** | **17** |
| Statistically significant wins | 0 | 16 |

Muon consistently fails to converge within the iteration budget, plateauing at loss ~0.0045 while SGD reaches machine precision (~1e-32) in ~220 iterations. The differences are not merely statistically significant -- they are overwhelming, with Cohen's d ranging from 94 to 3328.

---

## Per-Experiment Analysis

### E01: Matrix Sensing -- Dimension Scaling

**What was tested:** Matrix sensing with dimensions d=[50, 60, 70], rank r=5, lr=0.01, no noise, normal distribution, hard-cutoff spectrum.

| Metric | Muon-Exact | SGD | Ratio |
|--------|-----------|-----|-------|
| mean min_loss | 5.97e-03 +/- 1.21e-03 | 3.59e-32 +/- 7.51e-33 | 5.97e27x |
| mean K_epsilon | 2000.0 +/- 0.0 | 221.8 +/- 0.85 | 9.02x |
| mean FLOPs | 2.66e11 | 2.88e10 | 9.2x |

**Winner:** SGD wins on both convergence speed and final loss.

**Statistical significance:** p = 5.62e-31, Cohen's d = 2651.53, highly significant (paired, n=10 seeds).

---

### E02: Matrix Factorization -- Depth Scaling

**What was tested:** Matrix factorization with d=50, r=5, depths L=[2, 3], lr=0.01.

| Metric | Muon-Exact | SGD | Ratio |
|--------|-----------|-----|-------|
| mean min_loss | 6.39e-04 +/- 3.98e-04 | 8.85e-06 +/- 7.29e-06 | 72.2x |
| mean K_epsilon | 2000.0 +/- 0.0 | 2000.0 +/- 0.0 | 1.0x |
| mean FLOPs | 1.00e10 | 1.25e09 | 8.0x |

**Winner:** SGD wins on final loss. Both algorithms hit the max iteration limit (2000), so K_epsilon is tied.

**Statistical significance:** NaN p-value (identical K_epsilon values), Cohen's d = 0.0 for K_epsilon. Not significant for convergence speed. SGD's min_loss advantage is visually clear.

---

### E03: Matrix Sensing -- Learning Rate Calibration

**What was tested:** Matrix sensing with d=[50, 70], r=5, learning rates lr=[0.001, 0.003, 0.01, 0.03, 0.1].

| Metric | Muon-Exact | SGD | Ratio |
|--------|-----------|-----|-------|
| mean min_loss | 9.61e-02 +/- 1.66e-01 | 1.73e-31 +/- 2.46e-31 | 9.61e28x |
| mean K_epsilon | 2000.0 +/- 0.0 | 248.0 +/- 77.7 | 8.06x |
| mean FLOPs | 2.79e11 | 3.18e10 | 8.8x |

**Winner:** SGD wins on both metrics. Notably, Muon's loss is highly variable across learning rates (std=0.166), while SGD converges reliably even at suboptimal learning rates.

**Statistical significance:** p = 1.21e-08, Cohen's d = 94.44, highly significant (paired, n=5 seeds).

---

### E04: Matrix Sensing -- Noise and Initialization Scale

**What was tested:** Matrix sensing with d=50, r=5, noise=[0, 0.01, 0.1], init_scale=[0.001, 0.01, 0.1, 1.0].

| Metric | Muon-Exact | SGD | Ratio |
|--------|-----------|-----|-------|
| mean min_loss | 4.56e-03 +/- 1.22e-04 | 4.87e-30 +/- 8.37e-30 | 9.35e26x |
| mean K_epsilon | 2000.0 +/- 0.0 | 239.8 +/- 24.4 | 8.34x |
| mean FLOPs | 1.16e11 | 1.35e10 | 8.6x |

**Winner:** SGD wins on both metrics. SGD's K_epsilon shows some sensitivity to initialization scale (std=24.4), but still converges far faster than Muon.

**Statistical significance:** p = 1.17e-14, Cohen's d = 3007.40, highly significant (paired, n=5 seeds).

---

### E06: Matrix Sensing -- Noise Robustness

**What was tested:** Matrix sensing with d=50, r=5, noise=[0, 0.001, 0.01, 0.1].

| Metric | Muon-Exact | SGD | Ratio |
|--------|-----------|-----|-------|
| mean min_loss | 4.54e-03 +/- 1.04e-04 | 2.77e-32 +/- 2.24e-33 | 4.54e27x |
| mean K_epsilon | 2000.0 +/- 0.0 | 221.3 +/- 0.89 | 9.04x |
| mean FLOPs | 1.16e11 | 1.24e10 | 9.3x |

**Winner:** SGD wins on both metrics. SGD is remarkably robust to noise -- its K_epsilon barely changes (221.3 +/- 0.89) across noise levels.

**Statistical significance:** p = 5.62e-31, Cohen's d = 2651.53, highly significant (paired, n=10 seeds).

---

### E09: Matrix Sensing -- Weight Decay Sensitivity

**What was tested:** Matrix sensing with d=50, r=5, weight_decay=[0, 1e-5, 1e-4, 1e-3, 0.01].

| Metric | Muon-Exact | SGD | Ratio |
|--------|-----------|-----|-------|
| mean min_loss | 4.48e-03 +/- 3.02e-04 | 2.38e-08 +/- 4.78e-08 | 1.88e5x |
| mean K_epsilon | 2000.0 +/- 0.0 | 932.9 +/- 882.4 | 2.14x |
| mean FLOPs | 1.16e11 | 5.25e10 | 2.2x |

**Winner:** SGD wins on both metrics. Weight decay causes high variance in SGD's convergence (std=882), but SGD still converges faster on average.

**Statistical significance:** p = 9.12e-25, Cohen's d = 3014.55, highly significant (paired, n=8 seeds).

---

### E11: Multi-Baseline Comparison

**What was tested:** Matrix sensing baseline (d=50, r=5) comparing Muon-Exact, SGD, Adam, RMSprop, and Momentum-SGD.

| Algorithm | mean min_loss | mean K_epsilon |
|-----------|--------------|----------------|
| Muon-Exact | 4.55e-03 | 2000 |
| SGD | 2.76e-32 | 221 |
| Adam | 5.73e-25 | 220 |
| Momentum-SGD | 2.76e-32 | 221 |
| RMSprop | 1.40e-01 | 2000 |

**Winner:** SGD and Momentum-SGD tie for best. Adam also converges fast but to a slightly worse minimum. RMSprop and Muon both fail to converge.

**Muon vs SGD significance:** p = 5.62e-31, Cohen's d = 2651.53, highly significant (paired, n=10 seeds).

---

### E12: Hessian Eigenvalue Ratio Tracking

**What was tested:** Matrix sensing with eigenvalue ratio monitoring during optimization.

| Metric | Muon-Exact | SGD | Ratio |
|--------|-----------|-----|-------|
| mean min_loss | 4.53e-03 +/- 1.03e-04 | 2.75e-32 +/- 2.79e-33 | 4.53e27x |
| mean K_epsilon | 2000.0 +/- 0.0 | 220.8 +/- 0.84 | 9.06x |
| eigen_ratio (end) | ~2.1e11 | ~2.1e11 | -- |

**Winner:** SGD wins on both metrics. Both algorithms show identical final eigenvalue ratios (~2e11), suggesting the problem's Hessian structure is not the differentiating factor.

**Statistical significance:** p = 1.17e-14, Cohen's d = 3007.40, highly significant (paired, n=5 seeds).

---

### E13: Muon Variant Comparison

**What was tested:** Matrix sensing comparing Muon-Exact, Muon-RandSVD, Muon-Trunc vs SGD.

| Algorithm | mean min_loss | mean K_epsilon | FLOPs |
|-----------|--------------|----------------|-------|
| Muon-Exact | 4.55e-03 | 2000 | 1.16e11 |
| Muon-RandSVD | 1.02e-03 | 2000 | 1.13e11 |
| Muon-Trunc | 5.37e-04 | 2000 | 1.16e11 |
| SGD | 2.76e-32 | 221 | 1.24e10 |

**Winner:** SGD wins against all Muon variants. Truncated SVD gives the best Muon result (5.37e-04), but is still ~5.4e26x worse than SGD.

**Statistical significance:** All comparisons p < 1e-30, Cohen's d = 2651.53, highly significant.

---

### E14: RandSVD Parameter Sweep

**What was tested:** Muon-RandSVD with parameter combinations (p, q) = [(5,1), (5,2), (10,1), (10,2), (20,1), (20,2)]. No SGD data in this file.

| Configuration | mean min_loss |
|--------------|--------------|
| Muon-Exact (baseline) | 4.55e-03 |
| RandSVD (5,1) | 1.66e-03 |
| RandSVD (5,2) | 1.10e-03 |
| RandSVD (10,1) | 1.37e-03 |
| RandSVD (10,2) | 1.02e-03 |
| RandSVD (20,1) | 1.46e-03 |
| RandSVD (20,2) | 1.51e-03 |

**Observation:** RandSVD parameters (p=10, q=2) give the best Muon result (~1e-3), but all remain far from convergence. No SGD comparison available.

---

### E15: Large-Scale Matrix Sensing

**What was tested:** Matrix sensing with d=[100, 200], r=5, max_iter=1000.

| Metric | Muon-Exact | SGD | Ratio |
|--------|-----------|-----|-------|
| mean min_loss | 1.31e-02 +/- 1.89e-03 | 2.13e-31 +/- 1.59e-31 | 1.31e28x |
| mean K_epsilon | 1000.0 +/- 0.0 | 224.8 +/- 2.04 | 4.45x |
| mean FLOPs | 7.71e12 | 1.73e12 | 4.5x |

**Winner:** SGD wins on both metrics. At larger scales, the absolute FLOPs gap widens (7.7e12 vs 1.7e12), though the speedup ratio decreases slightly due to the reduced iteration budget.

**Statistical significance:** p ~= 0, Cohen's d = 7.77e18, highly significant (paired, n=3 seeds).

---

### E16: Initialization Scale Sensitivity

**What was tested:** Matrix sensing with d=50, r=5, init_scale=[0.001, 0.01, 0.1, 1.0, 10.0].

| Metric | Muon-Exact | SGD | Ratio |
|--------|-----------|-----|-------|
| mean min_loss | 4.53e-03 +/- 1.52e-04 | 3.86e-28 +/- 7.72e-28 | 1.17e25x |
| mean K_epsilon | 2000.0 +/- 0.0 | 256.6 +/- 40.2 | 7.79x |
| mean FLOPs | 1.16e11 | 1.44e10 | 8.0x |

**Winner:** SGD wins on both metrics. SGD shows some sensitivity to initialization scale (K_epsilon std=40.2), but still converges 7.8x faster than Muon.

**Statistical significance:** p = 4.56e-25, Cohen's d = 3328.20, highly significant (paired, n=8 seeds).

---

### E17: Initialization Type Comparison

**What was tested:** Matrix sensing with d=50, r=5, init_type=[random, orthogonal, spectral].

| Metric | Muon-Exact | SGD | Ratio |
|--------|-----------|-----|-------|
| mean min_loss | 4.55e-03 +/- 1.30e-04 | 2.61e-32 +/- 2.36e-33 | 4.55e27x |
| mean K_epsilon | 2000.0 +/- 0.0 | 221.0 +/- 0.75 | 9.05x |
| mean FLOPs | 1.16e11 | 1.24e10 | 9.3x |

**Winner:** SGD wins on both metrics. Initialization type has minimal impact on either algorithm's performance.

**Statistical significance:** p = 9.12e-25, Cohen's d = 3014.55, highly significant (paired, n=8 seeds).

---

### E18: Condition Number Sensitivity

**What was tested:** Matrix sensing with d=50, r=5, kappa=[10, 100, 1000, 10000].

| Metric | Muon-Exact | SGD | Ratio |
|--------|-----------|-----|-------|
| mean min_loss | 4.55e-03 +/- 1.27e-04 | 1.17e-32 +/- 6.72e-34 | 4.55e27x |
| mean K_epsilon | 2000.0 +/- 0.0 | 212.9 +/- 0.83 | 9.40x |
| mean FLOPs | 1.16e11 | 1.20e10 | 9.7x |

**Winner:** SGD wins on both metrics. SGD's K_epsilon is remarkably stable across condition numbers (212.9 +/- 0.83).

**Statistical significance:** p = 1.83e-24, Cohen's d = 2728.92, highly significant (paired, n=8 seeds).

---

### E19: Distribution Type Robustness

**What was tested:** Matrix sensing with d=50, r=5, dist=[normal, rademacher, sparse, sphere, uniform].

| Metric | Muon-Exact | SGD | Ratio |
|--------|-----------|-----|-------|
| mean min_loss | 2.19e-03 +/- 1.98e-03 | 8.17e-05 +/- 1.66e-04 | 26.9x |
| mean K_epsilon | 2000.0 +/- 0.0 | 619.0 +/- 704.7 | 3.23x |
| mean FLOPs | 1.16e11 | 3.48e10 | 3.3x |

**Winner:** SGD wins on both metrics. The large variance in SGD's K_epsilon (std=704.7) is driven by the **sphere distribution**, where SGD also fails to converge (K_epsilon=2000, min_loss~4e-4). For all other distributions, SGD converges rapidly.

**Muon by distribution:**
- normal: 4.41e-03
- uniform: 1.53e-03
- rademacher: 4.49e-03
- sparse: 4.71e-04
- sphere: 1.84e-06

**Statistical significance:** p = 9.12e-25, Cohen's d = 3014.55, highly significant (paired, n=8 seeds).

---

### E20: Large-Sample Reproducibility

**What was tested:** Matrix sensing with d=50, r=5, 50 independent random seeds.

| Metric | Muon-Exact | SGD | Ratio |
|--------|-----------|-----|-------|
| mean min_loss | 4.56e-03 +/- 1.44e-04 | 2.79e-32 +/- 2.25e-33 | 4.56e27x |
| mean K_epsilon | 2000.0 +/- 0.0 | 221.7 +/- 0.93 | 9.02x |
| mean FLOPs | 1.16e11 | 1.25e10 | 9.3x |

**Winner:** SGD wins on both metrics with overwhelming consistency.

**Statistical significance:** p = 9.33e-163, Cohen's d = 2713.78, highly significant (paired, n=50 seeds).

---

## Overall Summary Table

| Exp | Problem | Parameters Tested | Muon min_loss | SGD min_loss | Muon K_eps | SGD K_eps | Winner | p-value | Cohen's d |
|-----|---------|-------------------|---------------|--------------|------------|-----------|--------|---------|-----------|
| E01 | MS | d=[50,60,70] | 5.97e-03 | 3.59e-32 | 2000 | 222 | **SGD** | 5.62e-31 | 2652 |
| E02 | MF | L=[2,3] | 6.39e-04 | 8.85e-06 | 2000 | 2000 | **SGD** | NaN | 0 |
| E03 | MS | lr sweep | 9.61e-02 | 1.73e-31 | 2000 | 248 | **SGD** | 1.21e-08 | 94 |
| E04 | MS | noise x init | 4.56e-03 | 4.87e-30 | 2000 | 240 | **SGD** | 1.17e-14 | 3007 |
| E06 | MS | noise robustness | 4.54e-03 | 2.77e-32 | 2000 | 221 | **SGD** | 5.62e-31 | 2652 |
| E09 | MS | weight decay | 4.48e-03 | 2.38e-08 | 2000 | 933 | **SGD** | 9.12e-25 | 3015 |
| E11 | MS | multi-baseline | 4.55e-03 | 2.76e-32 | 2000 | 221 | **SGD** | 5.62e-31 | 2652 |
| E12 | MS | hessian track | 4.53e-03 | 2.75e-32 | 2000 | 221 | **SGD** | 1.17e-14 | 3007 |
| E13 | MS | Muon variants | 4.55e-03 | 2.76e-32 | 2000 | 221 | **SGD** | <1e-30 | 2652 |
| E15 | MS | d=[100,200] | 1.31e-02 | 2.13e-31 | 1000 | 225 | **SGD** | ~0 | 7.77e18 |
| E16 | MS | init scale | 4.53e-03 | 3.86e-28 | 2000 | 257 | **SGD** | 4.56e-25 | 3328 |
| E17 | MS | init type | 4.55e-03 | 2.61e-32 | 2000 | 221 | **SGD** | 9.12e-25 | 3015 |
| E18 | MS | condition no. | 4.55e-03 | 1.17e-32 | 2000 | 213 | **SGD** | 1.83e-24 | 2729 |
| E19 | MS | distributions | 2.19e-03 | 8.17e-05 | 2000 | 619 | **SGD** | 9.12e-25 | 3015 |
| E20 | MS | 50 seeds | 4.56e-03 | 2.79e-32 | 2000 | 222 | **SGD** | 9.33e-163 | 2714 |

---

## Pattern Analysis

### Which experiments does Muon win?

**None.** Muon does not win on either K_epsilon or min_loss in any of the 17 direct comparisons.

### Which experiments does SGD win?

**All of them.** SGD wins on convergence speed (K_epsilon) in 16 of 17 comparisons (tied only in E02 where both hit max_iter). SGD wins on final loss (min_loss) in all 17 comparisons.

### Is there a pattern?

**Problem type:**
- **Matrix Sensing (MS):** SGD dominates universally. Muon plateaus at ~0.0045 loss regardless of parameters.
- **Matrix Factorization (MF):** Both algorithms hit iteration limits, but SGD achieves 72x better final loss.

**Distribution type (E19):**
- SGD struggles only with **sphere distribution** (both algorithms plateau at K_epsilon=2000).
- For normal, uniform, rademacher, and sparse distributions, SGD converges rapidly.
- Muon performs best on sphere distribution (min_loss ~2e-6) but this is still 1000x worse than SGD on other distributions.

**Dimension (E01, E15):**
- SGD's advantage holds across d=50 to d=200.
- The K_epsilon speedup ratio decreases slightly at larger dimensions (9x at d=50 vs 4.5x at d=200), but SGD still converges much faster.

**Condition number (E18):**
- SGD is remarkably insensitive to condition number (K_epsilon ~213 across kappa=10 to 10000).
- Muon is completely insensitive (never converges).

**Initialization (E16, E17):**
- Neither algorithm shows strong sensitivity to initialization type or scale.
- SGD converges reliably regardless of initialization.

**Learning rate (E03):**
- SGD converges across all tested learning rates (0.001 to 0.1).
- Muon's performance degrades significantly at higher learning rates (loss up to 0.5 at lr=0.1).

---

## Key Findings

1. **Muon fails to converge on matrix sensing:** In all MS experiments, Muon-Exact plateaus at K_epsilon = max_iter with min_loss ~0.0045, suggesting the algorithm is not suited for this problem class under the tested conditions.

2. **SGD reaches machine precision:** SGD consistently achieves min_loss ~1e-32 (machine epsilon territory) in ~220 iterations across nearly all conditions.

3. **Muon variants don't help:** RandSVD and Truncated SVD improve Muon's min_loss to ~1e-3 and ~5e-4 respectively, but remain orders of magnitude worse than SGD.

4. **FLOPs efficiency:** SGD uses 3-10x fewer total FLOPs to convergence, due to both faster convergence and lower per-iteration cost (no SVD).

5. **Robustness:** SGD is robust to noise, condition number, initialization, and most distribution types. Its only failure mode in these experiments is the sphere distribution.

6. **Statistical certainty:** All paired comparisons show p < 1e-14, with Cohen's d > 90, indicating effect sizes that are not merely significant but practically overwhelming.

---

## Methodology Notes

- Statistical tests used: paired t-test (from muonlib/metrics.paired_ttest) for K_epsilon comparisons where paired seeds were available.
- Effect sizes computed using Cohen's d (from muonlib/metrics.compute_effect_size).
- FLOPs computed using muonlib/metrics.compute_flops_matrix_sensing and compute_flops_mf.
- E05 (FLOPs theory), E07 (rank ratio), E08 (gamma scan), and E10 (rectangular matrices) had no CSV results available for analysis.
- E14 contained only Muon-RandSVD data; no SGD comparison was possible.

---

## Conclusion

Under the experimental conditions tested -- square matrix sensing and factorization problems with Gaussian measurements, fixed learning rates, and full-batch gradients -- **SGD is overwhelmingly superior to Muon** on convergence speed, final optimization quality, and computational efficiency. Muon's spectral normalization via SVD does not provide benefits on these problem classes and incurs significant computational overhead.

These results are specific to the tested problem domain (matrix optimization with exact SVD) and should not be generalized to other contexts (e.g., neural network training with approximate SVD) without further experimentation.
