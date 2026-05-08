# PyTorch Notebook Rewrite

This directory contains a notebook-first PyTorch rewrite path. The existing
NumPy/SciPy experiments remain untouched in `notebooks_v2/`.

Current scope:

- `E01_ms_benchmark_torch.ipynb` is a runnable Matrix Sensing prototype.
- Default methods are `Muon`, `Muon-Exact`, `Shampoo`, `Adam`, and `SGD`.
- Experiment setup, data generation, loss, run loop, metrics, plotting, and
  conclusion stay inside the notebook for readability.
- Official PyTorch `torch.optim.Muon` is used when available; exact-SVD Muon
  and Shampoo live in `muonlib_torch/optimizers.py`.
- Results stay in notebook memory as `df` and `trajectories`; the notebook does
  not write CSV, PNG, or report files by default.

Default notebook mode is the full E01 grid: 5 optimizers, 3 dimensions, 10
seeds, and 2000 iterations per run. Quick validation lives in `smoketests/`,
not in this experiment notebook.
