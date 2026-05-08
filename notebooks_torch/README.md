# PyTorch Notebook Rewrite

This directory contains a notebook-first PyTorch rewrite path. The existing
NumPy/SciPy experiments remain untouched in `notebooks_v2/`.

Current scope:

- `E01_ms_benchmark_torch.ipynb` is a runnable Matrix Sensing prototype.
- Default methods are `Muon`, `Muon-Exact`, `Shampoo`, `Adam`, and `SGD`.
- Experiment setup, run grid, metrics, plotting, and conclusion stay inside the
  notebook for readability.
- Runs are dispatched across worker processes with a `tqdm` progress bar.
- Plotting includes same-dimension algorithm comparisons, same-algorithm
  dimension comparisons, all-combination grids, metric bars, and seed traces.
- Plot colors are semantic: related algorithms use related hues, and dimensions
  use ordered shade/line-style variants.
- Official PyTorch `torch.optim.Muon` is used when available; exact-SVD Muon
  and Shampoo live in `muonlib_torch/optimizers.py`.
- The importable single-run worker lives in `muonlib_torch/e01_matrix_sensing.py`
  so multiprocessing can use the `spawn` start method.
- Results stay in notebook memory as `df` and `trajectories`; the notebook does
  not write CSV, PNG, or report files by default.

Default notebook mode is the full E01 grid: 5 optimizers, 3 dimensions, 10
seeds, and 2000 iterations per run. Quick validation lives in `smoketests/`,
not in this experiment notebook.
