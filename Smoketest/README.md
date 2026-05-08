# Smoke Tests

These scripts are quick execution checks for the PyTorch problem definitions and
optimizers. They are not the experiment itself; the full MatrixSensing
experiment lives in `Notebooks/E01_ms_benchmark_torch.ipynb`.

Run from the repository root:

```bash
conda activate muonexperiment-torch
python Smoketest/test_problem_workers.py
```
