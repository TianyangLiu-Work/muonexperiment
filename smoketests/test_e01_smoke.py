"""Quick execution check for the PyTorch E01 optimizer set.

This smoke test intentionally uses a tiny matrix sensing instance. Its purpose
is to catch broken imports, optimizer API changes, and non-finite losses before
starting the full notebook experiment.
"""

from __future__ import annotations

import sys
from pathlib import Path
from math import isfinite

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from muonlib_torch import MuonTorch, ShampooTorch


DEVICE = torch.device("cpu")
DTYPE = torch.float64

ALGOS = ["Muon", "Muon-Exact", "Shampoo", "Adam", "SGD"]
D = 20
RANK = 3
LR = 0.01
ITERS = 20
M_MEAS = 2 * D * RANK


def make_generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(int(seed))


def randn(shape: tuple[int, ...], seed: int) -> torch.Tensor:
    return torch.randn(shape, generator=make_generator(seed), device=DEVICE, dtype=DTYPE)


def generate_target_matrix(d: int, rank: int, seed: int) -> torch.Tensor:
    u, _ = torch.linalg.qr(randn((d, d), seed))
    v, _ = torch.linalg.qr(randn((d, d), seed + 17))
    s = torch.zeros(d, device=DEVICE, dtype=DTYPE)
    s[:rank] = 1.0
    return u @ torch.diag(s) @ v.T


def matrix_sensing_loss(a: torch.Tensor, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    pred = torch.einsum("mij,ij->m", a, x)
    return 0.5 * torch.mean((pred - y) ** 2)


def make_optimizer(algo: str, params, lr: float) -> torch.optim.Optimizer:
    if algo == "Muon":
        if hasattr(torch.optim, "Muon"):
            return torch.optim.Muon(
                params,
                lr=lr,
                weight_decay=0.0,
                momentum=0.9,
                nesterov=False,
                ns_steps=5,
            )
        return MuonTorch(params, lr=lr, momentum=0.9, variant="newton_schulz", ns_steps=5)
    if algo == "Muon-Exact":
        return MuonTorch(params, lr=lr, momentum=0.9, variant="exact")
    if algo == "Shampoo":
        return ShampooTorch(params, lr=lr, beta2=0.9, epsilon=1e-8)
    if algo == "Adam":
        return torch.optim.Adam(params, lr=lr)
    if algo == "SGD":
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    raise ValueError(f"unknown algo: {algo}")


def run_algo(algo: str) -> float:
    x_star = generate_target_matrix(D, RANK, seed=100)
    a = randn((M_MEAS, D, D), seed=200)
    y = torch.einsum("mij,ij->m", a, x_star)

    x = torch.nn.Parameter(0.01 * randn((D, D), seed=300))
    opt = make_optimizer(algo, [x], LR)

    loss = matrix_sensing_loss(a, y, x)
    for _ in range(ITERS):
        opt.zero_grad(set_to_none=True)
        loss = matrix_sensing_loss(a, y, x)
        loss.backward()
        opt.step()

    final_loss = float(loss.detach().cpu())
    if not isfinite(final_loss):
        raise AssertionError(f"{algo} produced non-finite loss: {final_loss}")
    return final_loss


def main() -> None:
    print(f"torch={torch.__version__}, official_muon={hasattr(torch.optim, 'Muon')}")
    for algo in ALGOS:
        final_loss = run_algo(algo)
        print(f"{algo:10s} final_loss={final_loss:.6e}")


if __name__ == "__main__":
    main()
