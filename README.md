# Muon Matrix Experiments — PyTorch Notebook Rewrite

This branch is a clean PyTorch rewrite of the Muon matrix experiments. The old
NumPy/SciPy experiment scripts, generated result tables, logs, reports, and long
design documents have been removed from this branch.

## Design Principle

The code is notebook-first:

- Experiment setup stays in the notebook.
- Data generation stays in the notebook.
- Loss functions stay in the notebook.
- Run loops stay in the notebook.
- Tables and plots stay in the notebook.
- Only local fallback optimizers are extracted into a tiny Python module.

This keeps each experiment readable without jumping through framework code.

## Current Layout

```text
muonlib_torch/
  __init__.py
  optimizers.py              # local Muon exact/fallback + Shampoo

notebooks_torch/
  README.md
  E01_ms_benchmark_torch.ipynb

requirements.txt
README.md
```

## Implemented Prototype

`notebooks_torch/E01_ms_benchmark_torch.ipynb` implements a PyTorch Matrix
Sensing benchmark:

$$
X^\star = U \operatorname{diag}(s)V^\top
$$

$$
y_i = \langle A_i, X^\star\rangle + \varepsilon_i
$$

$$
f(X) = \frac{1}{2m}\sum_{i=1}^{m}(\langle A_i, X\rangle-y_i)^2
$$

The notebook compares multiple optimizers by default:

- `Muon`: official `torch.optim.Muon` when available
- `Muon-Exact`: exact SVD Muon implemented in `MuonTorch`
- `Shampoo`: matrix Shampoo implemented in `ShampooTorch`
- `Adam`: PyTorch built-in
- `SGD`: PyTorch built-in with momentum

It includes:

- explicit parameter grid
- torch target matrix generation
- torch measurement tensor generation
- matrix sensing loss via `torch.einsum`
- full run loop
- live table and plot helpers
- in-notebook result table, plots, and conclusion

Default mode is a fast smoke run:

```python
SMOKE_MODE = True
```

Set it to `False` in the notebook to run the E01-sized grid.

## Optimizers

This branch targets PyTorch `2.11.0`. PyTorch 2.11 ships official
`torch.optim.Muon`, and the notebook uses it for the `Muon` row. The local
`MuonTorch` implementation remains useful for exact-SVD Muon and for older
environments where `torch.optim.Muon` is unavailable. PyTorch does not provide
a built-in `torch.optim.Shampoo`, so this branch keeps a small local
matrix-only `ShampooTorch`.

### Muon

`MuonTorch` is implemented in `muonlib_torch/optimizers.py`.

Given a matrix gradient:

$$
G = U\Sigma V^\top
$$

Muon uses the spectral-normalized direction:

$$
D = UV^\top
$$

and applies:

$$
M_{k+1} = \mu M_k + D_k
$$

$$
X_{k+1} = X_k - \eta M_{k+1} - \eta\lambda X_k
$$

Supported variants:

- `newton_schulz`
- `exact`
- `randsvd`
- `truncated`

### Shampoo

`ShampooTorch` keeps row and column second-moment preconditioners for each
matrix parameter:

$$
L_{k+1} = \beta L_k + (1-\beta)GG^\top
$$

$$
R_{k+1} = \beta R_k + (1-\beta)G^\top G
$$

and updates with:

$$
\Delta = L_{k+1}^{-1/4} G R_{k+1}^{-1/4}
$$

## Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Open the notebook:

```bash
jupyter lab notebooks_torch/E01_ms_benchmark_torch.ipynb
```

Or run the notebook cells manually in any Jupyter-compatible environment.

The notebook keeps results in memory as `df` and `trajectories`. It does not
write CSV, PNG, or report files by default.

## Notes

- This branch intentionally does not preserve old experiment outputs.
- PyTorch wall-clock results are not directly comparable with the old
  NumPy/SciPy branch, especially when CUDA is available.
