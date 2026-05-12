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
  ablations.py              # scenario/ablation metric plots
  colors.py                 # shared color dictionaries and color helpers
  data.py                   # summary and trajectory dataframe transforms
  metrics.py                # metric plots
  trajectories.py           # trajectory plots
  PlottingUsage.ipynb       # usage examples for plotting functions

util/
  experiment.py             # joblib/tqdm execution helper

Notebooks/
  E01_ms_benchmark_torch.ipynb
  E02_matrix_factorization_torch.ipynb
  E03_ms_ablations_torch.ipynb
  E04_mf_initialization_ablations_torch.ipynb

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

## Experiments

### E01 Matrix Sensing Dimension Benchmark

`Notebooks/E01_ms_benchmark_torch.ipynb` runs the baseline Matrix Sensing
dimension benchmark:

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
- early stopping: standard patience-based early stopping on absolute loss
  improvement, with `min_steps=200`, `patience=200`, and `min_delta=1e-4`
- total runs: `150`
- maximum optimizer-step budget: `300000`

Runs are independent across `(method, d, seed)` and are dispatched with
`joblib.Parallel` plus a `tqdm` progress bar. Each worker uses one torch thread
to avoid CPU oversubscription. The long `runs` table stores the executed steps;
`run_summary` reports `actual_steps`, `stopped_early`, and `stop_reason`.

### E02 Matrix Factorization Benchmark

`Notebooks/E02_matrix_factorization_torch.ipynb` runs the same optimizer grid on
the nonconvex low-rank factorization objective. The target is still

$$
X^\star = U \operatorname{diag}(s)V^\top
$$

but the optimized variables are factors:

$$
f(L,R) = \frac{1}{2d^2}\lVert LR^\top-X^\star\rVert_F^2.
$$

This experiment checks whether conclusions from E01 survive the switch from a
full matrix variable `X` to two matrix factors `L` and `R`.

### E03 Matrix Sensing Ablations

`Notebooks/E03_ms_ablations_torch.ipynb` keeps the Matrix Sensing objective but
changes problem assumptions one at a time:

- baseline normal measurements
- Rademacher measurements
- sphere-normalized measurements
- polynomial-decay spectrum
- exponential-decay spectrum
- ill-conditioned target with `kappa=100`
- noisy observations with `noise=0.01`

This experiment is intentionally longer than E01 because it multiplies the
optimizer/seed grid by several problem scenarios. It is the notebook to inspect
when explaining which modeling choices make the experiment take longer.

### E04 Matrix Factorization Initialization Ablations

`Notebooks/E04_mf_initialization_ablations_torch.ipynb` keeps the
MatrixFactorization objective from E02 but changes only the initialization of
the factors `L` and `R`.

It includes harder and more ill-conditioned starts:

- tiny balanced factors
- oversized factors
- left tiny / right large factors
- left large / right tiny factors
- column-wise ill-conditioned factors with condition number `100`
- column-wise ill-conditioned factors with condition number `10000`
- opposite left/right column conditioning with condition number `10000`

This experiment is the one to inspect when comparing optimizers under flat
small-initialization regions or badly scaled factor coordinates.

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
