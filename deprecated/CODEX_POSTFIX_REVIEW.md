# Codex Postfix Review

Reviewed:

- `muonlib/algorithms.py`
- all 19 files in `notebooks_v2/`: `E01`, `E02`, `E03`, `E04`, `E06`, `E07`, `E08`, `E09`, `E10`, `E11`, `E12`, `E13`, `E14`, `E15`, `E16`, `E17`, `E18`, `E19`, `E20`

Verification performed:

- `python3 -m py_compile muonlib/algorithms.py notebooks_v2/*.py`
- searched target files for `svd_interval`, direct notebook `svd(...)` calls, `_last_singular_values`, `opt.step`, loss ordering, and `try/finally` logger cleanup
- exercised `MuonOptimizer(variant="truncated")` on representative small shapes

## Findings

### MAJOR - `K_epsilon` and `final_loss` semantics are now inconsistent with pre-update loss logging

The notebooks correctly moved `loss = ...` before `opt.step(...)`, so each logged loss now describes the same pre-update state as the gradient. However, the result fields still use the old post-update iteration convention:

- `K_epsilon` is still set to `step + 1`.
- `final_loss` is still `losses[-1]`.

With pre-update measurement, at `step == 0` no optimizer update has happened yet. If the initial state already satisfies `loss <= EPSILON`, the current code reports `K_epsilon = 1`, not `0`. More generally, a threshold hit observed at the start of loop step `k` means `k` updates have completed, but the code reports `k + 1`.

Likewise, after the loop completes `iters` calls to `opt.step`, `losses[-1]` is the loss before the last update, not the loss of the final returned/trained state. That makes `final_loss` stale by one update whenever `iters > 0`.

Examples:

- `notebooks_v2/E01_ms_benchmark_detailed.py:111-114`, `:129-145`
- `notebooks_v2/E02_mf_benchmark_detailed.py:94-101`, `:115-130`

The same pattern is present in all 19 notebooks:

- `E01`: `K_epsilon` at line 129, `final_loss` at line 144
- `E02`: `K_epsilon` at line 115, `final_loss` at line 129
- `E03`: `K_epsilon` at line 106, `final_loss` at line 121
- `E04`: `K_epsilon` at line 105, `final_loss` at line 120
- `E06`: `K_epsilon` at line 103, `final_loss` at line 118
- `E07`: `K_epsilon` at line 103, `final_loss` at line 118
- `E08`: `K_epsilon` at line 103, `final_loss` at line 119
- `E09`: `K_epsilon` at line 103, `final_loss` at line 119
- `E10`: `K_epsilon` at line 101, `final_loss` at line 116
- `E11`: `K_epsilon` at line 101, `final_loss` at line 116
- `E12`: `K_epsilon` at line 121, `final_loss` at line 141
- `E13`: `K_epsilon` at line 101, `final_loss` at line 116
- `E14`: `K_epsilon` at line 101, `final_loss` at line 117
- `E15`: `K_epsilon` at line 98, `final_loss` at line 111
- `E16`: `K_epsilon` at line 84, `final_loss` at line 93
- `E17`: `K_epsilon` at line 87, `final_loss` at line 94
- `E18`: `K_epsilon` at line 75, `final_loss` at line 81
- `E19`: `K_epsilon` at line 75, `final_loss` at line 81
- `E20`: `K_epsilon` at line 79, `final_loss` at line 85

Recommendation: either record an explicit post-loop loss for `final_loss`, or reduce the loop to pure pre-update logging semantics and define `final_loss` as the final logged pre-update value. For convergence counts, use `k_epsilon = step` for pre-update threshold checks if the metric means "number of completed updates to reach epsilon".

### MAJOR - `MuonOptimizer` truncated variant crashes on 1-row or 1-column matrices

`muonlib/algorithms.py:65-80` handles `min_dim <= 1` by computing `G_eff`, but it never assigns `s`. The new tracking line then reads `s.copy()`, causing `UnboundLocalError`.

Reproducer:

```python
import numpy as np
from muonlib.algorithms import MuonOptimizer

opt = MuonOptimizer(variant="truncated", r=5)
opt.step(np.zeros((1, 5)), np.ones((1, 5)), 0.1)
```

Observed:

```text
UnboundLocalError: local variable 's' referenced before assignment
```

This is not hit by the current 19 notebooks because their matrices have minimum dimension above 1, but it is a regression in the public optimizer path introduced by storing `_last_singular_values`.

Recommendation: assign `s` in the `min_dim <= 1` branch, for example `s = np.linalg.svd(G, compute_uv=False)` or `s = np.array([np.linalg.norm(G)])` depending on the intended logging meaning for this fallback.

### MINOR - Crash protection starts after logger creation and setup work

All 19 notebooks now close the logger in a `finally` block for exceptions raised inside the training loop. That covers the main crash-protection bug.

However, each notebook constructs `DetailedLogger(...)` before data generation, initialization, and optimizer creation, while the `try/finally` starts later. If setup fails after the logger is created but before the loop begins, `logger.close()` is not called and the run directory may not get its incomplete summary marker.

Examples:

- `notebooks_v2/E01_ms_benchmark_detailed.py:88` creates the logger; `:109` starts the protected region.
- `notebooks_v2/E02_mf_benchmark_detailed.py:72` creates the logger; `:92` starts the protected region.

Recommendation: if the goal is full per-run crash accounting, start the `try/finally` immediately after logger construction and include setup plus the training loop.

### MINOR - Removed direct notebook SVD calls, but stale `svd` imports remain

The double-SVD fix is functionally correct in the notebooks: direct notebook calls to `svd(...)` were removed and Muon singular values are read from optimizer state after `opt.step(...)`.

The imports remain unused in all 19 notebooks:

```python
from numpy.linalg import svd
```

This is not a behavioral regression, but it is stale code and can confuse future readers into thinking the notebooks still compute a logging SVD.

Recommendation: remove the unused `svd` imports from the 19 notebook exports.

### OK - `svd_interval=10` was removed from the reviewed target files

No `svd_interval` references remain in `notebooks_v2/` or `muonlib/algorithms.py`. The `DetailedLogger` constructor also no longer takes that argument, so the notebook call sites now match the logger API.

### OK - Singular-value logging now uses optimizer internal state

Muon notebooks now read `opt._last_singular_values` or, in matrix factorization, `opts[0]._last_singular_values` after the optimizer step. This avoids the previous extra notebook-side SVD and logs the singular values computed by the optimizer update.

For `Muon-Exact`, these are the full gradient singular values. For `Muon-RandSVD` and `Muon-Trunc`, these are the singular values from the approximate/truncated update path, which is consistent with the stated fix of reading optimizer internal state.

### OK - Loss is measured before `opt.step(...)`

All 19 notebooks now compute and append loss before computing the gradient/update. This fixes the state-ordering issue for per-step logs: `loss`, `grad_norm`, and logged singular values all correspond to the pre-update iterate and gradient used for that step.

The remaining issue is the result-summary convention covered in the first finding.

### OK - Training-loop `try/finally` cleanup is present in all 19 notebooks

Each reviewed run function wraps the training loop with `try/finally` and calls `logger.close()` in the `finally`. This means buffered trajectory data is flushed and an incomplete marker can be written when the loop itself crashes.

## Summary

No critical findings.

The fixes mostly address the four reported bugs, but two functional issues remain:

1. Every notebook still reports convergence/final-loss summaries using the old post-update convention after moving loss measurement to pre-update.
2. `MuonOptimizer(variant="truncated")` now crashes on 1D matrix shapes because `s` is undefined in the fallback branch.

