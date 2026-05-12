# PyTorch Notebooks

This directory contains notebook-first experiments.

Current scope:

- `E01_ms_benchmark_torch.ipynb` runs the full Matrix Sensing benchmark.
- `E02_matrix_factorization_torch.ipynb` runs the same optimizer comparison on
  nonconvex low-rank Matrix Factorization.
- `E03_ms_ablations_torch.ipynb` runs Matrix Sensing ablations over measurement
  distribution, spectrum shape, condition number, and noise.
- `E04_mf_initialization_ablations_torch.ipynb` runs Matrix Factorization
  ablations over tiny, unbalanced, oversized, and ill-conditioned factor
  initialization.
- Default methods are `Muon`, `Muon-Exact`, `Shampoo`, `Adam`, and `SGD`.
- Experiment setup, run grid, metrics, plotting, and conclusion stay inside the
  notebook for readability.
- Each notebook has `SMOKE_TEST = False` by default. Set it to `True` to keep
  the same grid but cap each run at `SMOKE_TEST_MAX_STEPS = 10`.
- Plotting functions are imported from `plotting/`; the main experiment
  notebook does not define plotting internals inline.
- `problems/` only defines autograd problems; this notebook owns the run loop,
  optimizer construction, timing, and worker serialization.
- The notebook defines `single_run(run)` inline; each joblib worker receives one
  row from `runs` and returns that run's per-step DataFrame.
- Runs use standard patience-based early stopping on absolute loss improvement;
  the summary table reports `actual_steps`, `stopped_early`, and `stop_reason`.
- Runs are dispatched through `util.run_experiments`, which wraps
  `joblib.Parallel` and a `tqdm` progress bar.
- Plotting is split into short cells, with each output cell producing one
  figure or one coherent figure group.
- Plotting includes same-dimension algorithm comparisons, same-algorithm
  dimension comparisons, all-combination grids, metric bars, and seed traces.
- Official PyTorch `torch.optim.Muon` is used when available; exact-SVD Muon
  and Shampoo live in `optimizers/`.

Results stay in notebook memory. `runs` becomes a long per-step table with one
row per executed optimizer step, while `run_summary` and `trajectories` are
derived from it for tables and plots. The notebook does not write CSV, PNG, or
report files by default.
