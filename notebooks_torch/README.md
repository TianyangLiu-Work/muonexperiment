# PyTorch Notebook Rewrite

This directory contains a notebook-first PyTorch rewrite path. The existing
NumPy/SciPy experiments remain untouched in `notebooks_v2/`.

Current scope:

- `E01_ms_benchmark_torch.ipynb` is a runnable Matrix Sensing prototype.
- Experiment setup, data generation, loss, run loop, metrics, plotting, and CSV
  conclusion stay inside the notebook for readability.
- Only the stateful Muon optimizer is extracted to `muonlib_torch/optimizers.py`.
- Results stay in notebook memory as `df` and `trajectories`; the notebook does
  not write CSV, PNG, or report files by default.

Default notebook mode is a small smoke run. Set `SMOKE_MODE = False` in the
parameter cell to run the E01-sized grid.
