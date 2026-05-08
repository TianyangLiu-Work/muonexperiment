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
  MatrixConstruction.py     # shared target generation, RNG, torch setup
  MatrixFactorization.py    # MatrixFactorization worker
  MatrixSensing.py          # MatrixSensing worker

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

`problems.MatrixSensing.step` and `problems.MatrixFactorization.step` are the
single optimizer-step functions. The notebook imports `step`, binds it as
`STEP_FN`, and passes it to the worker through:

```python
RUN_ONE = partial(run_matrix_sensing_spec, step_fn=STEP_FN)
```

That keeps the step visible in the notebook while still using an importable,
multiprocessing-safe function object.

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
- iterations per run: `2000`
- total runs: `150`
- total optimizer steps: `300000`

Runs are independent across `(method, d, seed)` and are dispatched with
`ProcessPoolExecutor` plus a `tqdm` progress bar. Each worker uses one torch
thread to avoid CPU oversubscription.

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

The notebook keeps results in memory as `df` and `trajectories`; it does not
write result tables or figures to separate files.
