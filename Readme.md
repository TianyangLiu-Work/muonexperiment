# Muon Matrix Experiments

This repo is a notebook-first experimental report on matrix optimization with
Muon and related baselines.

The notebooks are already executed. A reader should inspect the saved tables and
figures first. Code cells are needed only to modify the experimental grid,
optimizer set, stopping rule, or diagnostics.

## Reader Guide

Two objectives are studied.

Matrix Sensing observes

$$y_i=\langle A_i,X^\star\rangle+\xi_i,\qquad X^\star\in\mathbb{R}^{d\times d},\quad \operatorname{rank}(X^\star)=r,$$

and minimizes

$$f(X)=\frac{1}{2m}\sum_{i=1}^{m}(\langle A_i,X\rangle-y_i)^2.$$

Matrix Factorization parameterizes \(X=LR^\top\) and minimizes

$$g(L,R)=\frac{1}{2d^2}\lVert LR^\top-X^\star\rVert_F^2.$$

The primary recovery metric is

$$e(\widehat X)=\frac{\lVert \widehat X-X^\star\rVert_F}{\lVert X^\star\rVert_F}.$$

For two methods \(a,b\), heatmap gaps use

$$\Delta_{a,b}=\log_{10} e_a-\log_{10} e_b.$$

Thus \(\Delta_{\mathrm{Muon},b}<0\) means Muon has lower median recovery error
than method \(b\).

| Notebook | Mathematical Question | First Quantities To Inspect |
|---|---|---|
| `E01_ms_benchmark_torch.ipynb` | For Matrix Sensing, how do trajectories depend on \(d\)? | \(f(X_t)\), runtime, actual steps |
| `E02_matrix_factorization_torch.ipynb` | Under the reparameterization \(X=LR^\top\), does optimizer behavior change? | \(g(L_t,R_t)\), runtime, actual steps |
| `E03_ms_ablations_torch.ipynb` | How do measurement law, spectrum, condition number, and noise affect \(f\)? | Scenario-wise loss and log-loss |
| `E04_mf_initialization_ablations_torch.ipynb` | How does \((L_0,R_0)\) affect convergence of \(g\)? | Scenario-wise loss, stop reason |
| `E05_ms_sample_complexity_phase_diagram_torch.ipynb` | How does recovery depend on \(m=\alpha dr\)? | \(e(\widehat X)\), \(\Delta_{\mathrm{Muon},b}\), success probability |
| `E06_ms_noise_robustness_torch.ipynb` | As \(\sigma\) increases, does low training loss imply low recovery error? | \(e(\widehat X)\), \(f_\sigma(\widehat X)\), clean test loss |
| `E07_mf_rank_init_phase_diagram_torch.ipynb` | How do factor rank \(q\) and scale \(s\) affect \(g\)? | \(e(LR^\top)\), divergence rate, effective rank |
| `E08_mf_scale_imbalance_torch.ipynb` | How does scale symmetry \(LR^\top=(cL)(R/c)^\top\) affect optimization? | \(e(LR^\top)\), balancedness, factor norms |
| `E09_muon_geometry_diagnostics_torch.ipynb` | What are the spectra and alignments of gradients and updates? | effective rank, cosine, relative step size, singular-value error |
| `E10_muon_variant_ablation_torch.ipynb` | Is the effect polar geometry or normalization? | variant error, time per step, update rank, cosine |

## Implementation Model

The old script-oriented code has been removed from this branch. The repository
keeps reusable optimizer/problem code small, while experiment setup, metrics,
plotting, and conclusions live in the notebooks.

## Layout

```text
optimizers/
  MuonExact.py              # exact SVD / approximate Muon variants
  NormalizedSGD.py          # Frobenius/spectral normalized SGD baselines
  Shampoo.py                # matrix-only Shampoo

problems/
  MatrixConstruction.py     # tensor construction for matrix problems
  MatrixFactorization.py    # autograd MatrixFactorization definition
  MatrixSensing.py          # autograd MatrixSensing definition

plotting/
  ablations.py              # scenario/ablation metric plots
  colors.py                 # shared color dictionaries and color helpers
  data.py                   # summary and trajectory dataframe transforms
  metrics.py                # metric plots
  phase.py                  # phase-diagram heatmaps, gaps, and line plots
  trajectories.py           # trajectory plots
  PlottingUsage.ipynb       # usage examples for plotting functions

util/
  diagnostics.py            # recovery, rank, spectrum, and update diagnostics
  experiment.py             # joblib/tqdm execution helper

Notebooks/
  E01_ms_benchmark_torch.ipynb
  E02_matrix_factorization_torch.ipynb
  E03_ms_ablations_torch.ipynb
  E04_mf_initialization_ablations_torch.ipynb
  E05_ms_sample_complexity_phase_diagram_torch.ipynb
  E06_ms_noise_robustness_torch.ipynb
  E07_mf_rank_init_phase_diagram_torch.ipynb
  E08_mf_scale_imbalance_torch.ipynb
  E09_muon_geometry_diagnostics_torch.ipynb
  E10_muon_variant_ablation_torch.ipynb

Readme.md
requirement.yml
```

`problems/` only defines differentiable PyTorch objectives and the tensors they
need. It does not choose optimizers, run iterations, time experiments, or manage
worker serialization.

Each experiment notebook defines `single_run(run)` inline. Each joblib worker
receives one row of the `runs` table, initializes the problem, runs
optimization, records per-step data, and returns that run's DataFrame.

```python
util.run_experiments(runs, single_run, ...)
```

That keeps the experiment logic visible in the notebook while `problems/`
remains importable and joblib-serializable.

## Experiment Axes

Each notebook estimates a map from controlled variables to optimization and
recovery statistics.

| Notebook | Controlled Variables | Reported Statistics |
|---|---|---|
| E01 | \(d\), method, seed | \(f(X_t)\), final/min loss, runtime |
| E02 | \(d\), method, seed | \(g(L_t,R_t)\), final/min loss, runtime |
| E03 | measurement law, spectrum, \(\kappa\), noise | scenario-wise loss and log-loss |
| E04 | initialization law for \((L_0,R_0)\) | loss, actual steps, stop reason |
| E05 | \(\alpha\) in \(m=\alpha dr\), spectrum, method | \(e(\widehat X)\), \(\Delta_{\mathrm{Muon},b}\), success rate |
| E06 | noise scale \(\sigma\), spectrum, \(\alpha\) | \(f_\sigma(\widehat X)\), \(e(\widehat X)\), clean test loss |
| E07 | factor rank \(q\), init scale \(s\) | \(e(LR^\top)\), divergence, effective rank |
| E08 | left/right scales \(a,b\) | \(e(LR^\top)\), balancedness, factor norms |
| E09 | representative regimes and method | spectra of \(G_t,U_t\), cosine, step size |
| E10 | Muon variants and normalized baselines | \(e\), time per step, update rank, cosine |

## Optimizers

`Muon` uses official `torch.optim.Muon` when available. If the installed
PyTorch does not provide it, `Muon` falls back to the local Newton-Schulz Muon
variant in `optimizers/MuonExact.py`.

`Muon-Exact` uses exact SVD. For a matrix gradient

$$
G = U\Sigma V^\top
$$

the update direction is the polar factor:

$$
D = UV^\top
$$

`Shampoo` keeps row and column second-moment preconditioners:

$$
L_{k+1} = \beta L_k + (1-\beta)GG^\top
$$

$$
R_{k+1} = \beta R_k + (1-\beta)G^\top G
$$

and applies:

$$
\Delta = L_{k+1}^{-1/4} G R_{k+1}^{-1/4}
$$

`NormalizedSGD` and `SpectralNormSGD` are mechanism baselines for E09/E10.
They update with

$$
\Delta=-\eta\frac{G}{\lVert G\rVert_F}
$$

or

$$
\Delta=-\eta\frac{G}{\lVert G\rVert_2}
$$

respectively.

## Run

Create and activate the dedicated conda environment:

```bash
conda env create -f requirement.yml
conda activate muonexperiment-torch
```

Register the environment as a Jupyter kernel:

```bash
python -m ipykernel install --user --name muonexperiment-torch --display-name "Python (muonexperiment-torch)"
```

Open a notebook:

```bash
jupyter lab Notebooks/E01_ms_benchmark_torch.ipynb
```

To smoke-test the complete notebook code path, set `SMOKE_TEST = True` in the
parameter cell and run the notebook. Smoke mode keeps the same run grid but
sets `iters` to `SMOKE_TEST_MAX_STEPS`, which defaults to `10`.

The notebook keeps results in memory. `runs` becomes a long per-step table with
one row per executed optimizer step, while `run_summary` and `trajectories` are
derived from it for tables and plots. It does not write result tables or figures
to separate files.
