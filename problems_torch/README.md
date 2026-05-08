# Problem Workers

This package stores importable problem definitions for multiprocessing. The
notebooks still own the experiment grid, metrics, plots, and conclusions.

## Modules

- `common.py`: reusable torch setup, seeded random tensors, target matrix
  generation, optimizer construction, and device synchronization.
- `matrix_sensing.py`: MatrixSensing measurements, loss, and single-run worker.
- `matrix_factorization.py`: MatrixFactorization loss and single-run worker.

Each problem module exposes:

```python
run_spec(spec) -> (key, row, trajectory)
```

where `spec` is a plain dictionary that can be serialized by
`ProcessPoolExecutor` with the `spawn` start method.

