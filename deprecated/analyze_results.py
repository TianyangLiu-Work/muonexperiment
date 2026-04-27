#!/usr/bin/env python3
"""Analyze all experiment results comparing Muon vs SGD."""

import pandas as pd
import numpy as np
import json
import sys
sys.path.insert(0, '/data/home/tyliu/muonexperiment')

from muonlib.metrics import (
    welch_ttest, paired_ttest, wilcoxon_test, compute_effect_size,
    compute_flops_matrix_sensing, compute_flops_mf
)

RESULTS_DIR = '/data/home/tyliu/muonexperiment/results'

EXPERIMENTS = [
    'E01', 'E02', 'E03', 'E04', 'E06', 'E09', 'E11', 'E12',
    'E13', 'E14', 'E15', 'E16', 'E17', 'E18', 'E19', 'E20'
]

def load_csv(exp):
    return pd.read_csv(f'{RESULTS_DIR}/{exp}_results.csv')

def compute_mean_std(series):
    return float(series.mean()), float(series.std(ddof=1)) if len(series) > 1 else 0.0

OUTCOME_COLUMNS = {
    'algo', 'final_loss', 'min_loss', 'K_epsilon', 'F_eps', 'I_conv',
    'time_s', 'iters'
}

def paired_k_epsilon_by_full_key(muon_df, sgd_df):
    """Return K_epsilon pairs matched on the full shared experiment key."""
    key_cols = sorted((set(muon_df.columns) & set(sgd_df.columns)) - OUTCOME_COLUMNS)
    if not key_cols:
        return [], [], key_cols

    paired = muon_df[key_cols + ['K_epsilon']].merge(
        sgd_df[key_cols + ['K_epsilon']],
        on=key_cols,
        suffixes=('_muon', '_sgd')
    )
    if paired.empty:
        return [], [], key_cols
    paired = paired.sort_values(key_cols)
    return (
        paired['K_epsilon_muon'].to_numpy(),
        paired['K_epsilon_sgd'].to_numpy(),
        key_cols
    )

def analyze_experiment(exp):
    df = load_csv(exp)
    
    # Determine experiment type from columns and data
    cols = set(df.columns)
    
    # Identify problem type
    if 'L' in cols:
        problem_type = 'matrix_factorization'
    else:
        problem_type = 'matrix_sensing'
    
    # Identify parameters tested
    params = {}
    if 'd' in cols:
        params['d'] = sorted(df['d'].unique().tolist())
    if 'r' in cols:
        params['r'] = sorted(df['r'].unique().tolist())
    if 'lr' in cols:
        params['lr'] = sorted(df['lr'].unique().tolist())
    if 'noise' in cols:
        params['noise'] = sorted(df['noise'].unique().tolist())
    if 'init_scale' in cols:
        params['init_scale'] = sorted(df['init_scale'].unique().tolist())
    if 'L' in cols:
        params['L'] = sorted(df['L'].unique().tolist())
    if 'dist' in cols:
        params['dist'] = sorted(df['dist'].unique().tolist())
    if 'spectrum' in cols:
        params['spectrum'] = sorted(df['spectrum'].unique().tolist())
    if 'kappa' in cols:
        params['kappa'] = sorted(df['kappa'].unique().tolist())
    if 'wd' in cols:
        params['wd'] = sorted(df['wd'].unique().tolist())
    if 'init_type' in cols:
        params['init_type'] = sorted(df['init_type'].unique().tolist())
    
    # Get unique algorithm names
    algos = df['algo'].unique().tolist()
    
    # For comparison, we focus on Muon-Exact vs SGD when both present
    # Some experiments have multiple Muon variants
    muon_variants = [a for a in algos if 'Muon' in a]
    sgd_present = 'SGD' in algos
    
    results = {
        'experiment': exp,
        'problem_type': problem_type,
        'parameters': params,
        'algorithms_tested': algos,
        'comparisons': []
    }
    
    # For each Muon variant, compare with SGD if available
    for muon_algo in muon_variants:
        muon_df = df[df['algo'] == muon_algo]
        
        comp = {
            'muon_variant': muon_algo,
            'muon_n_runs': len(muon_df),
            'sgd_n_runs': None,
            'muon_mean_min_loss': None,
            'muon_std_min_loss': None,
            'sgd_mean_min_loss': None,
            'sgd_std_min_loss': None,
            'muon_mean_K_epsilon': None,
            'muon_std_K_epsilon': None,
            'sgd_mean_K_epsilon': None,
            'sgd_std_K_epsilon': None,
            'muon_mean_flops': None,
            'sgd_mean_flops': None,
            'winner_K_epsilon': None,
            'winner_min_loss': None,
            'K_epsilon_speedup': None,
            'min_loss_ratio': None,
            'paired_ttest_pvalue': None,
            'cohens_d': None,
            'statistical_significance': None,
            'paired_key_columns': None,
            'common_pairs': 0
        }
        
        # Muon stats
        comp['muon_mean_min_loss'], comp['muon_std_min_loss'] = compute_mean_std(muon_df['min_loss'])
        comp['muon_mean_K_epsilon'], comp['muon_std_K_epsilon'] = compute_mean_std(muon_df['K_epsilon'])
        
        # Compute FLOPs for Muon
        if problem_type == 'matrix_sensing':
            d_vals = muon_df['d'].values
            K_vals = muon_df['K_epsilon'].values
            flops = [compute_flops_matrix_sensing(int(d), m_meas=int(3*d*d), K=int(K), algorithm=muon_algo) 
                     for d, K in zip(d_vals, K_vals)]
        else:
            d_vals = muon_df['d'].values
            L_vals = muon_df['L'].values
            K_vals = muon_df['K_epsilon'].values
            flops = [compute_flops_mf(int(d), int(L), int(K), algorithm=muon_algo) 
                     for d, L, K in zip(d_vals, L_vals, K_vals)]
        comp['muon_mean_flops'] = float(np.mean(flops))
        
        if sgd_present:
            sgd_df = df[df['algo'] == 'SGD']
            comp['sgd_n_runs'] = len(sgd_df)
            comp['sgd_mean_min_loss'], comp['sgd_std_min_loss'] = compute_mean_std(sgd_df['min_loss'])
            comp['sgd_mean_K_epsilon'], comp['sgd_std_K_epsilon'] = compute_mean_std(sgd_df['K_epsilon'])
            
            # Compute FLOPs for SGD
            if problem_type == 'matrix_sensing':
                d_vals = sgd_df['d'].values
                K_vals = sgd_df['K_epsilon'].values
                flops = [compute_flops_matrix_sensing(int(d), m_meas=int(3*d*d), K=int(K), algorithm='SGD') 
                         for d, K in zip(d_vals, K_vals)]
            else:
                d_vals = sgd_df['d'].values
                L_vals = sgd_df['L'].values
                K_vals = sgd_df['K_epsilon'].values
                flops = [compute_flops_mf(int(d), int(L), int(K), algorithm='SGD') 
                         for d, L, K in zip(d_vals, L_vals, K_vals)]
            comp['sgd_mean_flops'] = float(np.mean(flops))
            
            # Determine winners
            if comp['muon_mean_K_epsilon'] < comp['sgd_mean_K_epsilon']:
                comp['winner_K_epsilon'] = 'Muon'
                comp['K_epsilon_speedup'] = float(comp['sgd_mean_K_epsilon'] / max(comp['muon_mean_K_epsilon'], 1))
            else:
                comp['winner_K_epsilon'] = 'SGD'
                comp['K_epsilon_speedup'] = float(comp['muon_mean_K_epsilon'] / max(comp['sgd_mean_K_epsilon'], 1))
            
            if comp['muon_mean_min_loss'] < comp['sgd_mean_min_loss']:
                comp['winner_min_loss'] = 'Muon'
                comp['min_loss_ratio'] = float(comp['sgd_mean_min_loss'] / max(comp['muon_mean_min_loss'], 1e-30))
            else:
                comp['winner_min_loss'] = 'SGD'
                comp['min_loss_ratio'] = float(comp['muon_mean_min_loss'] / max(comp['sgd_mean_min_loss'], 1e-30))
            
            # Statistical tests - only if full-key paired data are available.
            paired_muon, paired_sgd, key_cols = paired_k_epsilon_by_full_key(muon_df, sgd_df)
            comp['paired_key_columns'] = key_cols
            comp['common_pairs'] = len(paired_muon)
            
            if len(paired_muon) >= 3:
                try:
                    t_stat, p_val = paired_ttest(paired_muon, paired_sgd)
                    comp['paired_ttest_pvalue'] = float(p_val)
                    comp['cohens_d'] = float(compute_effect_size(paired_muon, paired_sgd, paired=True))
                    
                    if p_val < 0.001:
                        comp['statistical_significance'] = 'highly_significant (p<0.001)'
                    elif p_val < 0.01:
                        comp['statistical_significance'] = 'significant (p<0.01)'
                    elif p_val < 0.05:
                        comp['statistical_significance'] = 'marginally_significant (p<0.05)'
                    else:
                        comp['statistical_significance'] = 'not_significant'
                except Exception as e:
                    comp['statistical_significance'] = f'error: {str(e)}'
            else:
                comp['statistical_significance'] = 'unpaired_data'
                # Compute Welch's t-test instead
                try:
                    t_stat, p_val = welch_ttest(muon_df['K_epsilon'].values, sgd_df['K_epsilon'].values)
                    comp['paired_ttest_pvalue'] = float(p_val)
                    comp['cohens_d'] = float(compute_effect_size(muon_df['K_epsilon'].values, sgd_df['K_epsilon'].values))
                except Exception:
                    pass
        
        results['comparisons'].append(comp)
    
    return results

def main():
    all_results = {}
    
    for exp in EXPERIMENTS:
        print(f"Analyzing {exp}...", file=sys.stderr)
        all_results[exp] = analyze_experiment(exp)
    
    # Print structured output
    print(json.dumps(all_results, indent=2, default=str))

if __name__ == '__main__':
    main()
