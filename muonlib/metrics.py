"""Convergence metrics, FLOPs counting, and statistical analysis."""
import numpy as np
from scipy import stats

def compute_flops_matrix_sensing(d, m_meas, K, algorithm='SGD'):
    """Compute theoretical FLOPs for matrix sensing experiment.
    
    Per-task.md definitions 4.7-4.8:
    - SGD per-iteration: 2 * m * d^2 + d^2
      (linear measurements plus one dense gradient accumulation/update)
    - Muon per-iteration: 2 * m * d^2 + d^2 + SVD cost
      (same gradient + full SVD ~12d^3 + U@V^T ~2d^3)
    - FLOPs ratio: gamma = 1 + 14d^3/(2md^2 + d^2)
    """
    grad_flops = 2 * m_meas * d * d + d * d
    
    if 'Muon' in algorithm:
        if 'RandSVD' in algorithm:
            # Randomized SVD: O((p+q) * d^2) instead of O(d^3)
            svd_flops = 20 * d * d  # p=10, q=2, ~20*d^2
        elif 'Trunc' in algorithm:
            svd_flops = 20 * d * d  # sparse truncated SVD, same default scale as RandSVD
        else:
            svd_flops = 14 * d**3  # full exact SVD
    else:
        svd_flops = 0
    
    return (grad_flops + svd_flops) * K

def compute_flops_matrix_sensing_rect(m, n, m_meas, K, algorithm='SGD'):
    """Compute FLOPs for rectangular matrix sensing rows."""
    grad_flops = 2 * m_meas * m * n + m * n
    if 'Muon' not in algorithm:
        svd_flops = 0
    elif 'RandSVD' in algorithm or 'Trunc' in algorithm:
        svd_flops = 20 * m * n
    else:
        k = min(m, n)
        svd_flops = 12 * m * n * k + 2 * m * n * k
    return (grad_flops + svd_flops) * K

def compute_flops_mf(d, L, K, algorithm='SGD'):
    """Compute theoretical FLOPs for matrix factorization."""
    # Per-iteration: forward pass + backward pass through L layers
    flops_per_iter = L * 2 * d**3  # matrix multiplications
    
    if 'Muon' in algorithm:
        flops_per_iter += L * 14 * d**3  # SVD per weight matrix per layer
    
    return flops_per_iter * K

def convergence_rate(losses, epsilon=None):
    """Estimate convergence rate from loss trajectory.
    
    Returns: convergence type ('linear', 'sublinear', 'superlinear'), estimated rate
    """
    if len(losses) < 10:
        return 'insufficient_data', 0.0
    
    # Use last 50% of trajectory
    mid = len(losses) // 2
    late = losses[mid:]
    
    if len(late) < 3:
        return 'insufficient_data', 0.0
    
    # Check linear convergence: log-loss should be linear
    log_late = np.log(np.maximum(late, 1e-16))
    x = np.arange(len(log_late))
    slope, _, _, _, _ = stats.linregress(x, log_late)
    
    if slope < -0.01:
        rate = np.exp(slope)
        return 'linear', rate
    elif slope < 0:
        return 'sublinear', slope
    else:
        return 'stalled', 0.0

def compute_statistics(results_dict, alpha=0.05):
    """Compute statistical summaries for experiment results.
    
    results_dict: {config_key: [values across repetitions]}
    
    Returns: DataFrame-like dict with mean, std, ci_lower, ci_upper, n
    """
    summary = {}
    for key, values in results_dict.items():
        vals = np.array(values)
        n = len(vals)
        mean = np.mean(vals)
        std = np.std(vals, ddof=1) if n > 1 else 0.0
        
        # 95% confidence interval
        if n > 1:
            se = std / np.sqrt(n)
            ci = stats.t.ppf(1 - alpha/2, n - 1) * se
        else:
            ci = 0.0
        
        summary[key] = {
            'mean': mean, 'std': std,
            'ci_lower': mean - ci, 'ci_upper': mean + ci,
            'n': n
        }
    return summary

def welch_ttest(group1, group2):
    """Welch's t-test for comparing two groups."""
    t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)
    return t_stat, p_value

def paired_ttest(group1, group2):
    """Paired t-test."""
    t_stat, p_value = stats.ttest_rel(group1, group2)
    return t_stat, p_value

def wilcoxon_test(group1, group2):
    """Wilcoxon signed-rank test (non-parametric)."""
    stat, p_value = stats.wilcoxon(group1, group2)
    return stat, p_value

def bonferroni_correction(p_values, alpha=0.05):
    """Bonferroni correction for multiple testing."""
    m = len(p_values)
    corrected_alpha = alpha / m
    significant = [p < corrected_alpha for p in p_values]
    return corrected_alpha, significant

def benjamini_hochberg(p_values, fdr=0.10):
    """Benjamini-Hochberg FDR control."""
    m = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]
    
    thresholds = np.arange(1, m + 1) / m * fdr
    significant = np.zeros(m, dtype=bool)
    
    # Find largest k where p_(k) <= k/m * fdr
    for i in range(m - 1, -1, -1):
        if sorted_p[i] <= thresholds[i]:
            significant[sorted_indices[:i+1]] = True
            break
    
    return significant

def compute_effect_size(group1, group2, paired=False):
    """Cohen's d effect size.

    For paired samples, use the paired-standardized mean difference:
    mean(group1 - group2) / sd(group1 - group2).
    """
    x = np.asarray(group1, dtype=float)
    y = np.asarray(group2, dtype=float)
    if paired:
        if len(x) != len(y):
            raise ValueError("paired effect size requires equal-length samples")
        diff = x - y
        return np.mean(diff) / max(np.std(diff, ddof=1), 1e-16)

    n1, n2 = len(x), len(y)
    s1, s2 = np.std(x, ddof=1), np.std(y, ddof=1)
    pooled_std = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1 + n2 - 2))
    return (np.mean(x) - np.mean(y)) / max(pooled_std, 1e-16)

def enrich_result_row(row):
    """Add convergence flag and total FLOPs to a CSV result row."""
    row = dict(row)
    k_eps = int(row['K_epsilon'])
    iters = int(row.get('iters', k_eps))
    row.setdefault('I_conv', int(k_eps <= iters))

    algo = row.get('algo', 'SGD')
    if 'L' in row:
        row.setdefault('F_eps', compute_flops_mf(int(row['d']), int(row['L']), k_eps, algo))
    elif 'm' in row and 'n' in row:
        m_dim = int(row['m'])
        n_dim = int(row['n'])
        r = int(row.get('r', max(1, min(m_dim, n_dim) // 10)))
        m_meas = int(row.get('m_meas', 2 * min(m_dim, n_dim) * r))
        row.setdefault('F_eps', compute_flops_matrix_sensing_rect(m_dim, n_dim, m_meas, k_eps, algo))
    elif 'd' in row:
        d = int(row['d'])
        r = int(row.get('r', max(1, d // 10)))
        m_meas = int(row.get('m_meas', 2 * d * r))
        row.setdefault('F_eps', compute_flops_matrix_sensing(d, m_meas, k_eps, algo))
    return row

def compute_power(n, d, alpha=0.05):
    """Compute statistical power for given sample size and effect size."""
    df = 2 * n - 2
    ncp = d * np.sqrt(n / 2)
    critical = stats.t.ppf(1 - alpha/2, df)
    power = 1 - stats.nct.cdf(critical, df, ncp) + stats.nct.cdf(-critical, df, ncp)
    return power
