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

Default notebook mode is a small smoke run. Set `SMOKE_MODE = False` in the
parameter cell to run the E01-sized grid.
