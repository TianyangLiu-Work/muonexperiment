# Muon Matrix Experiments — PyTorch Notebook Rewrite

This branch is a clean PyTorch rewrite of the Muon matrix experiments. The old
NumPy/SciPy experiment scripts, generated result tables, logs, reports, and long
design documents have been removed from this branch.

## Design Principle

The code is notebook-first:

- Experiment setup stays in the notebook.
- The run grid stays in the notebook.
- Metrics, tables, plots, and conclusions stay in the notebook.
- Stateful optimizers and the importable single-run worker live in
  `muonlib_torch/` so multiprocessing with `spawn` works reliably.

This keeps each experiment readable without jumping through framework code.

## Current Layout

```text
muonlib_torch/
  __init__.py
  e01_matrix_sensing.py     # importable per-run worker for multiprocessing
  optimizers.py              # local Muon exact/fallback + Shampoo

notebooks_torch/
  README.md
  E01_ms_benchmark_torch.ipynb

smoketests/
  README.md
  test_e01_smoke.py

environment.yml
requirements.txt
README.md
```

## Implemented Prototype

`notebooks_torch/E01_ms_benchmark_torch.ipynb` implements a PyTorch Matrix
Sensing benchmark:

$$
X^\star = U \operatorname{diag}(s)V^\top
$$

$$
y_i = \langle A_i, X^\star\rangle + \varepsilon_i
$$

$$
f(X) = \frac{1}{2m}\sum_{i=1}^{m}(\langle A_i, X\rangle-y_i)^2
$$

The notebook compares multiple optimizers by default:

- `Muon`: official `torch.optim.Muon` when available
- `Muon-Exact`: exact SVD Muon implemented in `MuonTorch`
- `Shampoo`: matrix Shampoo implemented in `ShampooTorch`
- `Adam`: PyTorch built-in
- `SGD`: PyTorch built-in with momentum

It includes:

- explicit parameter grid
- importable torch single-run worker
- per-run multiprocessing
- `tqdm` progress over completed runs
- broad result plots: same-dimension algorithm comparisons, same-algorithm
  dimension comparisons, all algorithm-dimension grids, metric bars, and seed
  trajectory checks
- in-notebook result table, plots, and conclusion

Default mode is the full E01 grid: 5 optimizers, 3 dimensions, 10 seeds, and
2000 iterations per run, for 150 runs in total. Quick validation lives in
`smoketests/`, not in the experiment notebook.

`NUM_WORKERS` controls process-level parallelism across independent runs. Each
worker uses one torch thread to avoid CPU oversubscription.

## Optimizers

This branch targets the `torch==2.11.0` package. PyTorch 2.11 ships official
`torch.optim.Muon`, and the notebook uses it for the `Muon` row. The local
`MuonTorch` implementation remains useful for exact-SVD Muon and for older
environments where `torch.optim.Muon` is unavailable. PyTorch does not provide
a built-in `torch.optim.Shampoo`, so this branch keeps a small local
matrix-only `ShampooTorch`.

### Muon

`MuonTorch` is implemented in `muonlib_torch/optimizers.py`.

Given a matrix gradient:

$$
G = U\Sigma V^\top
$$

Muon uses the spectral-normalized direction:

$$
D = UV^\top
$$

and applies:

$$
M_{k+1} = \mu M_k + D_k
$$

$$
X_{k+1} = X_k - \eta M_{k+1} - \eta\lambda X_k
$$

Supported variants:

- `newton_schulz`
- `exact`
- `randsvd`
- `truncated`

### Shampoo

`ShampooTorch` keeps row and column second-moment preconditioners for each
matrix parameter:

$$
L_{k+1} = \beta L_k + (1-\beta)GG^\top
$$

$$
R_{k+1} = \beta R_k + (1-\beta)G^\top G
$$

and updates with:

$$
\Delta = L_{k+1}^{-1/4} G R_{k+1}^{-1/4}
$$

## Run

Create and activate the dedicated conda environment:

```bash
conda env create -f environment.yml
conda activate muonexperiment-torch
```

Register the environment as a Jupyter kernel:

```bash
python -m ipykernel install --user --name muonexperiment-torch --display-name "Python (muonexperiment-torch)"
```

Open the notebook from this environment:

```bash
jupyter lab notebooks_torch/E01_ms_benchmark_torch.ipynb
```

Or run the notebook cells manually in any Jupyter-compatible environment.

The notebook keeps results in memory as `df` and `trajectories`. It does not
write CSV, PNG, or report files by default.

`requirements.txt` is kept only as a pip-only fallback. Prefer
`environment.yml` for this branch.

Run the quick optimizer smoke test:

```bash
python smoketests/test_e01_smoke.py
```

## Notes

- This branch intentionally does not preserve old experiment outputs.
- PyTorch wall-clock results are not directly comparable with the old
  NumPy/SciPy branch, especially when CUDA is available.
