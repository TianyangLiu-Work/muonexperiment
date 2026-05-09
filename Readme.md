# Muon Matrix Experiments

This branch is a PyTorch notebook-first rewrite of the matrix experiments. The
old script-oriented code has been removed from this branch; the repository now
keeps reusable optimizer/problem code small, and keeps experiment setup,
metrics, plotting, and conclusions in the notebook.

## Layout

```text
optimizers/
  MuonExact.py              # exact SVD / approximate Muon variants
  Shampoo.py                # matrix-only Shampoo

problems/
  MatrixConstruction.py     # tensor construction for matrix problems
  MatrixFactorization.py    # autograd MatrixFactorization definition
  MatrixSensing.py          # autograd MatrixSensing definition

plotting/
  colors.py                 # shared color dictionaries and color helpers
  data.py                   # summary and trajectory dataframe transforms
  metrics.py                # metric plots
  trajectories.py           # trajectory plots
  PlottingUsage.ipynb       # usage examples for plotting functions

Notebooks/
  E01_ms_benchmark_torch.ipynb

Smoketest/
  test_problem_workers.py

Readme.md
requirement.yml
```

`problems/` only defines differentiable PyTorch objectives and the tensors they
need. It does not choose optimizers, run iterations, time experiments, or manage
worker serialization.

The main experiment notebook defines `single_run(run)` inline. Each joblib
worker receives one row of the `runs` table, initializes the MatrixSensing
problem, runs optimization, records per-step data, and returns that run's
DataFrame.

```python
delayed(single_run)(run)
```

That keeps the experiment logic visible in the notebook while `problems/`
remains importable and joblib-serializable.

## Current Experiment

`Notebooks/E01_ms_benchmark_torch.ipynb` runs the Matrix Sensing benchmark:

$$
X^\star = U \operatorname{diag}(s)V^\top
$$

$$
y_i = \langle A_i, X^\star\rangle + \varepsilon_i
$$

$$
f(X) = \frac{1}{2m}\sum_{i=1}^{m}(\langle A_i, X\rangle-y_i)^2
$$

Default full grid:

- methods: `Muon`, `Muon-Exact`, `Shampoo`, `Adam`, `SGD`
- dimensions: `50`, `60`, `70`
- seeds: `0` through `9`
- maximum iterations per run: `2000`
- early stopping: standard patience-based early stopping on relative loss
  improvement, with `min_steps=200`, `patience=200`, and `min_delta=1e-4`
- total runs: `150`
- maximum optimizer-step budget: `300000`

Runs are independent across `(method, d, seed)` and are dispatched with
`joblib.Parallel` plus a `tqdm` progress bar. Each worker uses one torch thread
to avoid CPU oversubscription. The long `runs` table stores the executed steps;
`run_summary` reports `actual_steps`, `stopped_early`, and `stop_reason`.

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

Open the notebook:

```bash
jupyter lab Notebooks/E01_ms_benchmark_torch.ipynb
```

Run the quick worker checks:

```bash
python Smoketest/test_problem_workers.py
```

The notebook keeps results in memory. `runs` becomes a long per-step table with
one row per executed optimizer step, while `run_summary` and `trajectories` are
derived from it for tables and plots. It does not write result tables or figures
to separate files.
