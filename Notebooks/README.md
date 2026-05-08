# PyTorch Notebooks

This directory contains notebook-first experiments.

Current scope:

- `E01_ms_benchmark_torch.ipynb` runs the full Matrix Sensing benchmark.
- Default methods are `Muon`, `Muon-Exact`, `Shampoo`, `Adam`, and `SGD`.
- Experiment setup, run grid, metrics, plotting, and conclusion stay inside the
  notebook for readability.
- Plotting functions are imported from `plotting/`; the main experiment
  notebook does not define plotting internals inline.
- `problems/` only defines autograd problems; `runners/` owns the run loop,
  optimizer construction, timing, and worker serialization.
- The notebook defines `step`, binds it as `STEP_FN`, and
  passes it into the multiprocessing runner through `functools.partial`.
- Runs are dispatched across worker processes with a `tqdm` progress bar.
- Plotting is split into short cells, with each output cell producing one
  figure or one coherent figure group.
- Plotting includes same-dimension algorithm comparisons, same-algorithm
  dimension comparisons, all-combination grids, metric bars, and seed traces.
- Official PyTorch `torch.optim.Muon` is used when available; exact-SVD Muon
  and Shampoo live in `optimizers/`.

Results stay in notebook memory as `df` and `trajectories`; the notebook does
not write CSV, PNG, or report files by default.
