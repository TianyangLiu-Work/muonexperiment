#!/usr/bin/env python3
"""Generate structured analysis report for Muon vs SGD experiments."""

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

def describe_experiment(exp):
    df = load_csv(exp)
    cols = set(df.columns)
    
    if 'L' in cols:
        problem_type = 'matrix_factorization'
    else:
        problem_type = 'matrix_sensing'
    
    # Build description
    params = []
    if 'd' in cols:
        ds = sorted(df['d'].unique().tolist())
        params.append(f"d={ds}")
    if 'r' in cols:
        params.append(f"r={sorted(df['r'].unique().tolist())}")
    if 'L' in cols:
        params.append(f"L={sorted(df['L'].unique().tolist())}")
    if 'lr' in cols:
        params.append(f"lr={sorted(df['lr'].unique().tolist())}")
    if 'noise' in cols:
        params.append(f"noise={sorted(df['noise'].unique().tolist())}")
    if 'init_scale' in cols:
        params.append(f"init_scale={sorted(df['init_scale'].unique().tolist())}")
    if 'dist' in cols:
        params.append(f"dist={sorted(df['dist'].unique().tolist())}")
    if 'spectrum' in cols:
        params.append(f"spectrum={sorted(df['spectrum'].unique().tolist())}")
    if 'kappa' in cols:
        params.append(f"kappa={sorted(df['kappa'].unique().tolist())}")
    if 'wd' in cols:
        params.append(f"wd={sorted(df['wd'].unique().tolist())}")
    if 'init_type' in cols:
        params.append(f"init_type={sorted(df['init_type'].unique().tolist())}")
    
    return problem_type, params

def analyze_comparison(exp, muon_algo='Muon-Exact'):
    df = load_csv(exp)
    cols = set(df.columns)
    
    if 'L' in cols:
        problem_type = 'matrix_factorization'
    else:
        problem_type = 'matrix_sensing'
    
    muon_df = df[df['algo'] == muon_algo]
    sgd_df = df[df['algo'] == 'SGD']
    
    if len(muon_df) == 0 or len(sgd_df) == 0:
        return None
    
    # Basic stats
    muon_min_loss_mean, muon_min_loss_std = compute_mean_std(muon_df['min_loss'])
    sgd_min_loss_mean, sgd_min_loss_std = compute_mean_std(sgd_df['min_loss'])
    muon_K_mean, muon_K_std = compute_mean_std(muon_df['K_epsilon'])
    sgd_K_mean, sgd_K_std = compute_mean_std(sgd_df['K_epsilon'])
    
    # FLOPs
    if problem_type == 'matrix_sensing':
        d_vals = muon_df['d'].values
        K_vals = muon_df['K_epsilon'].values
        muon_flops = [compute_flops_matrix_sensing(int(d), m_meas=int(3*d*d), K=int(K), algorithm=muon_algo) 
                 for d, K in zip(d_vals, K_vals)]
        d_vals = sgd_df['d'].values
        K_vals = sgd_df['K_epsilon'].values
        sgd_flops = [compute_flops_matrix_sensing(int(d), m_meas=int(3*d*d), K=int(K), algorithm='SGD') 
                 for d, K in zip(d_vals, K_vals)]
    else:
        d_vals = muon_df['d'].values
        L_vals = muon_df['L'].values
        K_vals = muon_df['K_epsilon'].values
        muon_flops = [compute_flops_mf(int(d), int(L), int(K), algorithm=muon_algo) 
                 for d, L, K in zip(d_vals, L_vals, K_vals)]
        d_vals = sgd_df['d'].values
        L_vals = sgd_df['L'].values
        K_vals = sgd_df['K_epsilon'].values
        sgd_flops = [compute_flops_mf(int(d), int(L), int(K), algorithm='SGD') 
                 for d, L, K in zip(d_vals, L_vals, K_vals)]
    
    muon_flops_mean = float(np.mean(muon_flops))
    sgd_flops_mean = float(np.mean(sgd_flops))
    
    # Winners
    winner_K = 'Muon' if muon_K_mean < sgd_K_mean else 'SGD'
    winner_loss = 'Muon' if muon_min_loss_mean < sgd_min_loss_mean else 'SGD'
    
    K_speedup = max(sgd_K_mean, 1) / max(muon_K_mean, 1) if muon_K_mean < sgd_K_mean else max(muon_K_mean, 1) / max(sgd_K_mean, 1)
    
    # Stats
    p_value = None
    cohens_d = None
    sig = None
    paired_muon, paired_sgd, paired_key_columns = paired_k_epsilon_by_full_key(muon_df, sgd_df)
    
    if len(paired_muon) >= 3:
        try:
            _, p_value = paired_ttest(paired_muon, paired_sgd)
            cohens_d = float(compute_effect_size(paired_muon, paired_sgd, paired=True))
        except:
            pass
    else:
        try:
            _, p_value = welch_ttest(muon_df['K_epsilon'].values, sgd_df['K_epsilon'].values)
            cohens_d = float(compute_effect_size(muon_df['K_epsilon'].values, sgd_df['K_epsilon'].values))
        except:
            pass
    
    if p_value is not None:
        if p_value < 0.001:
            sig = 'highly_significant'
        elif p_value < 0.01:
            sig = 'significant'
        elif p_value < 0.05:
            sig = 'marginally_significant'
        else:
            sig = 'not_significant'
    
    return {
        'experiment': exp,
        'problem_type': problem_type,
        'muon_variant': muon_algo,
        'muon_n': len(muon_df),
        'sgd_n': len(sgd_df),
        'muon_mean_min_loss': muon_min_loss_mean,
        'muon_std_min_loss': muon_min_loss_std,
        'sgd_mean_min_loss': sgd_min_loss_mean,
        'sgd_std_min_loss': sgd_min_loss_std,
        'muon_mean_K_epsilon': muon_K_mean,
        'muon_std_K_epsilon': muon_K_std,
        'sgd_mean_K_epsilon': sgd_K_mean,
        'sgd_std_K_epsilon': sgd_K_std,
        'muon_mean_flops': muon_flops_mean,
        'sgd_mean_flops': sgd_flops_mean,
        'winner_K_epsilon': winner_K,
        'winner_min_loss': winner_loss,
        'K_epsilon_speedup': K_speedup,
        'min_loss_ratio': max(muon_min_loss_mean, 1e-30) / max(sgd_min_loss_mean, 1e-30) if winner_loss == 'SGD' else max(sgd_min_loss_mean, 1e-30) / max(muon_min_loss_mean, 1e-30),
        'p_value': p_value,
        'cohens_d': cohens_d,
        'significance': sig,
        'paired_data': len(paired_muon) >= 3,
        'common_pairs': len(paired_muon),
        'paired_key_columns': paired_key_columns
    }

def main():
    report = {
        'experiments': {},
        'overall_summary': {}
    }
    
    for exp in EXPERIMENTS:
        problem_type, params = describe_experiment(exp)
        
        # Get all Muon variants present
        df = load_csv(exp)
        muon_variants = [a for a in df['algo'].unique() if 'Muon' in a]
        
        exp_data = {
            'problem_type': problem_type,
            'parameters': params,
            'comparisons': []
        }
        
        for variant in muon_variants:
            comp = analyze_comparison(exp, variant)
            if comp:
                exp_data['comparisons'].append(comp)
        
        report['experiments'][exp] = exp_data
    
    # Overall summary
    muon_wins_K = 0
    sgd_wins_K = 0
    muon_wins_loss = 0
    sgd_wins_loss = 0
    
    significant_sgd_wins = 0
    
    for exp, data in report['experiments'].items():
        for comp in data['comparisons']:
            if comp['winner_K_epsilon'] == 'Muon':
                muon_wins_K += 1
            else:
                sgd_wins_K += 1
            
            if comp['winner_min_loss'] == 'Muon':
                muon_wins_loss += 1
            else:
                sgd_wins_loss += 1
            
            if comp['winner_K_epsilon'] == 'SGD' and comp['significance'] in ['highly_significant', 'significant']:
                significant_sgd_wins += 1
    
    report['overall_summary'] = {
        'total_comparisons': muon_wins_K + sgd_wins_K,
        'muon_wins_K_epsilon': muon_wins_K,
        'sgd_wins_K_epsilon': sgd_wins_K,
        'muon_wins_min_loss': muon_wins_loss,
        'sgd_wins_min_loss': sgd_wins_loss,
        'significant_sgd_wins': significant_sgd_wins
    }
    
    print(json.dumps(report, indent=2, default=str))

if __name__ == '__main__':
    main()
