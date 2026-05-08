# Smoke Tests

These scripts are quick execution checks for the PyTorch problem workers. They
are not the experiment itself; the full MatrixSensing experiment lives in
`notebooks_torch/E01_ms_benchmark_torch.ipynb`.

Run from the repository root:

```bash
conda activate muonexperiment-torch
python smoketests/test_problem_workers.py
```
