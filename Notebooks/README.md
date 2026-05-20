# PyTorch Notebooks

This directory contains executed notebook experiments. Open them as reports:
the tables and figures are already saved. Inspect code cells only when changing
the grid, optimizers, stopping rule, or diagnostics.

## Reading Order

The two core metrics are training loss and relative recovery error:

$$e(\widehat X)=\frac{\lVert \widehat X-X^\star\rVert_F}{\lVert X^\star\rVert_F}.$$

Gap heatmaps use

$$\Delta_{\mathrm{Muon},b}=\log_{10} e_{\mathrm{Muon}}-\log_{10} e_b,$$

so negative values favor Muon.

| Notebook | Mathematical Question | First Quantities |
|---|---|---|
| `E01_ms_benchmark_torch.ipynb` | Matrix Sensing baseline as \(d\) varies. | \(f(X_t)\), time, steps |
| `E02_matrix_factorization_torch.ipynb` | Factorized baseline with \(X=LR^\top\). | \(g(L_t,R_t)\), time, steps |
| `E03_ms_ablations_torch.ipynb` | Sensitivity to sensing law, spectrum, \(\kappa\), and noise. | scenario-wise loss |
| `E04_mf_initialization_ablations_torch.ipynb` | Sensitivity to \((L_0,R_0)\). | scenario-wise loss and stop reason |
| `E05_ms_sample_complexity_phase_diagram_torch.ipynb` | Recovery as \(m=\alpha dr\) varies. | \(e\), gap heatmaps, success probability |
| `E06_ms_noise_robustness_torch.ipynb` | Relation between noisy loss and true recovery. | \(e\), noisy loss, clean test loss |
| `E07_mf_rank_init_phase_diagram_torch.ipynb` | Joint effect of factor rank \(q\) and scale \(s\). | \(e\), divergence, effective rank |
| `E08_mf_scale_imbalance_torch.ipynb` | Effect of \(LR^\top=(cL)(R/c)^\top\). | \(e\), balancedness, factor norms |
| `E09_muon_geometry_diagnostics_torch.ipynb` | Spectra and alignment of gradients and updates. | effective rank, cosine, step size |
| `E10_muon_variant_ablation_torch.ipynb` | Polar geometry versus update normalization. | variant error, cost, update rank |
| `E11_paper_condition_diagnostics_torch.ipynb` | Spectral-update theory condition diagnostics. | condition score, Muon advantage, update rank |
| `E12_mf_scale_imbalance_preconditioning_torch.ipynb` | Factor scale imbalance and preconditioning. | \(e\), balancedness, factor norms |
| `E13_polar_vs_normalization_ablation_torch.ipynb` | Polar orthogonalization versus normalization baselines. | \(e\), update rank, cosine, cost |
| `E14_isotropic_curvature_spectrum_control_torch.ipynb` | Controlled one-step spectrum and curvature model. | predicted local improvement |
| `E15_ms_negative_control_torch.ipynb` | Direct Matrix Sensing as a negative control. | \(e\), loss, clean loss, update rank |

Default benchmark methods are `Muon`, `Muon-Exact`, `Shampoo`, `Adam`, and
`SGD`. E10 also includes `Muon-NS-1`, `Muon-NS-10`, `Muon-Truncated`,
`Muon-RandSVD`, `NormalizedSGD`, `SpectralNormSGD`, and
`LayerwiseNormalizedSGD`. E11-E15 reuse these Muon-family and normalization
baselines for the theory-verification supplement.

## Code Cells

Skip imports, parameter grids, and worker definitions unless changing:

- the optimizer list
- the dimension, rank, seed, noise, spectrum, or initialization grid
- the early stopping rule
- the diagnostics collected per step
- the plotting functions

Execution convention:

- `runs` is first the experimental grid.
- `single_run(run)` maps one grid row to a per-step DataFrame.
- `util.run_experiments` applies `joblib.Parallel` and returns the long `runs`
  table.
- `run_summary` is derived from the final row of each run.
- `SMOKE_TEST=True` keeps the grid fixed and sets each run to at most
  `SMOKE_TEST_MAX_STEPS=10`.

Results stay in notebook memory. `runs` becomes a long per-step table with one
row per executed optimizer step, while `run_summary` and `trajectories` are
derived from it for tables and plots. The notebook does not write CSV, PNG, or
report files by default.
