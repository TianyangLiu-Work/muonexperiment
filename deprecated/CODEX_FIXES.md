# CODEX Fixes

## Review Fixes

1. **Muon-Trunc uses truncated sparse SVD**
   - `muonlib/algorithms.py` now calls `scipy.sparse.linalg.svds` for `variant='truncated'` and sorts the returned singular triplets before forming `U @ Vt`.
   - The truncated branch no longer runs a dense full SVD except for the separate `exact` variant.

2. **RandSVD determinism**
   - `MuonOptimizer` now owns a `np.random.RandomState` seeded by `random_state` (default `0`).
   - `randomized_svd()` accepts an RNG and no longer draws from global `np.random`.

3. **Matrix-sensing FLOPs**
   - `compute_flops_matrix_sensing()` now uses the reviewed gradient/update cost:
     `2 * m * d^2 + d^2`.
   - Truncated and randomized Muon variants use quadratic SVD estimates rather than the full dense SVD cost.

4. **Cohen's d for paired comparisons**
   - `compute_effect_size(..., paired=True)` now uses:
     `mean(x - y) / std(x - y, ddof=1)`.
   - Paired analysis/report/notebook comparisons pass `paired=True`; unpaired fallbacks keep the pooled-standard-deviation formula.

5. **Matrix factorization optimizers**
   - `run_experiments.py` now creates one optimizer instance per matrix-factorization layer, so optimizer state is not shared across layers.

6. **Full-key paired matching**
   - `analyze_results.py` and `generate_report.py` now inner-join Muon and SGD rows on the full shared experiment key, excluding only algorithm/outcome columns.
   - Pairing now includes fields such as `d`, `r`, `L`, `lr`, `noise`, `seed`, etc., instead of matching on `seed` alone.

7. **Non-convergence sentinel**
   - All notebook runner scripts now set `K_epsilon = iters + 1` for non-converged runs.
   - Existing `results/*_results.csv` files were backfilled with the same convention.

8. **CSV `F_eps` and `I_conv`**
   - Added `enrich_result_row()` in `muonlib.metrics`.
   - All notebook CSV writers enrich rows with:
     - `F_eps`: total FLOPs through `K_epsilon`
     - `I_conv`: `1` if converged, `0` if censored/non-converged
   - Existing result CSVs were backfilled with both columns.

## Verification

- `python3 -m py_compile muonlib/algorithms.py muonlib/metrics.py muonlib/__init__.py run_experiments.py analyze_results.py generate_report.py notebooks/*.py`
- Targeted checks passed for:
  - Muon-Trunc calling `svds`
  - deterministic RandSVD with identical optimizer seeds
  - matrix-sensing FLOPs formula
  - paired Cohen's d formula
  - matrix-factorization one-optimizer-per-layer behavior
  - full-key paired matching
  - CSV `F_eps`/`I_conv` enrichment and non-convergence sentinel
- `python3 analyze_results.py` completed successfully for 16 experiments.
- `python3 generate_report.py` completed successfully and `analysis_report.json` was regenerated.
