from __future__ import annotations

from typing import Any
import math

import torch
import torch.nn.functional as F
from sklearn.datasets import load_digits

from .diagnostics import derive_step_metrics, matrix_effective_rank, matrix_view, singular_values, stable_rank


def dtype_from_name(name: str) -> torch.dtype:
    dtype = getattr(torch, name)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"unknown torch dtype: {name}")
    return dtype


def make_generator(seed: int, device: torch.device) -> torch.Generator:
    generator = torch.Generator(device=device) if device.type == "cuda" else torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def load_digits_tensors(device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    data = load_digits()
    x = torch.tensor(data.data, device=device, dtype=dtype) / 16.0
    y = torch.tensor(data.target, device=device, dtype=torch.long)
    return x, y


def sample_indices(
    y: torch.Tensor,
    *,
    classes: tuple[int, ...],
    count_per_class: int,
    generator: torch.Generator,
) -> torch.Tensor:
    selected = []
    for label in classes:
        candidates = torch.nonzero(y == int(label), as_tuple=False).flatten()
        if candidates.numel() < count_per_class:
            raise ValueError(f"class {label} has {candidates.numel()} examples, need {count_per_class}")
        perm = torch.randperm(candidates.numel(), generator=generator, device=y.device)[:count_per_class]
        selected.append(candidates[perm])
    return torch.cat(selected)


def split_long_tail_digits(
    config: Any,
    *,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x, y = load_digits_tensors(device, dtype)
    generator = make_generator(seed + 11000, device)
    head_train = sample_indices(y, classes=config.head_classes, count_per_class=config.head_train_per_class, generator=generator)
    tail_pool = sample_indices(
        y,
        classes=config.tail_classes,
        count_per_class=config.tail_train_per_class + config.tail_eval_per_class,
        generator=generator,
    )
    train_tail_count = len(config.tail_classes) * config.tail_train_per_class
    tail_train = tail_pool[:train_tail_count]
    tail_eval = tail_pool[train_tail_count:]
    train_indices = torch.cat([head_train, tail_train])
    return x, y, train_indices, head_train, tail_train, tail_eval


class TinyMLP:
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        generator: torch.Generator,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.w1 = torch.nn.Parameter(1e-2 * torch.randn((hidden_dim, input_dim), generator=generator, device=device, dtype=dtype))
        self.w2 = torch.nn.Parameter(1e-2 * torch.randn((output_dim, hidden_dim), generator=generator, device=device, dtype=dtype))

    def parameters(self) -> list[torch.nn.Parameter]:
        return [self.w1, self.w2]

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(x @ self.w1.T)
        return hidden @ self.w2.T

    def activation_matrices(self, x: torch.Tensor) -> list[torch.Tensor]:
        with torch.no_grad():
            hidden = F.relu(x @ self.w1.detach().T)
        return [x.detach(), hidden.detach()]


def batch(indices: torch.Tensor, batch_size: int, *, generator: torch.Generator) -> torch.Tensor:
    if batch_size >= indices.numel():
        return indices
    perm = torch.randperm(indices.numel(), generator=generator, device=indices.device)[:batch_size]
    return indices[perm]


def add_noise(x: torch.Tensor, noise_std: float, *, generator: torch.Generator) -> torch.Tensor:
    if noise_std <= 0.0:
        return x
    scale = x.square().mean().sqrt().clamp_min(torch.finfo(x.dtype).eps)
    noise = float(noise_std) * scale * torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)
    return x + noise


def train_digits_checkpoint(
    model: TinyMLP,
    x: torch.Tensor,
    y: torch.Tensor,
    train_indices: torch.Tensor,
    config: Any,
    *,
    seed: int,
    seed_offset: int,
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    for step in range(config.warmup_steps):
        generator = make_generator(seed + seed_offset + step, x.device)
        batch_indices = batch(train_indices, config.warmup_batch_size, generator=generator)
        batch_x = add_noise(x[batch_indices], config.train_noise_std, generator=generator)
        loss = F.cross_entropy(model.logits(batch_x), y[batch_indices])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


def margins(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    true_logits = logits[torch.arange(logits.shape[0], device=logits.device), labels]
    masked = logits.clone()
    masked[torch.arange(logits.shape[0], device=logits.device), labels] = -torch.inf
    return true_logits - torch.max(masked, dim=1).values


def snapshot(params: list[torch.nn.Parameter]) -> list[torch.Tensor]:
    return [param.detach().clone() for param in params]


def restore(params: list[torch.nn.Parameter], before: list[torch.Tensor]) -> None:
    with torch.no_grad():
        for param, value in zip(params, before):
            param.copy_(value)


def global_grad_fro(params: list[torch.nn.Parameter]) -> float:
    total = 0.0
    for param in params:
        if param.grad is not None:
            total += float(torch.linalg.norm(param.grad.detach()).cpu()) ** 2
    return total**0.5


def direction_list(params: list[torch.nn.Parameter], geometry: str) -> list[torch.Tensor]:
    if geometry == "frobenius":
        grad_fro = global_grad_fro(params)
        return [param.grad.detach() / max(grad_fro, 1e-300) for param in params]
    if geometry == "spectral":
        directions = []
        for param in params:
            grad_matrix = matrix_view(param.grad.detach())
            u, _, vh = torch.linalg.svd(grad_matrix, full_matrices=False)
            directions.append((u @ vh).reshape_as(param))
        return directions
    raise ValueError(f"unknown geometry: {geometry}")


def alignment(params: list[torch.nn.Parameter], directions: list[torch.Tensor]) -> float:
    total = 0.0
    for param, direction in zip(params, directions):
        total += float(torch.sum(param.grad.detach() * direction).cpu())
    return total


def apply_direction(
    params: list[torch.nn.Parameter],
    before: list[torch.Tensor],
    directions: list[torch.Tensor],
    step_size: float,
) -> None:
    with torch.no_grad():
        for param, value, direction in zip(params, before, directions):
            param.copy_(value - float(step_size) * direction)


def tail_output_jvp(
    tail_x: torch.Tensor,
    before: list[torch.Tensor],
    directions: list[torch.Tensor],
    step_size: float,
) -> torch.Tensor:
    if len(before) != 2:
        raise ValueError("TinyMLP tail-output JVP expects exactly two matrix parameters")

    def logits_from_weights(w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(tail_x @ w1.T)
        return hidden @ w2.T

    primals = tuple(weight.detach().clone().requires_grad_(True) for weight in before)
    tangents = tuple(-float(step_size) * direction.detach() for direction in directions)
    _base, jvp = torch.autograd.functional.jvp(logits_from_weights, primals, tangents, create_graph=False, strict=False)
    return jvp.detach()


def local_linearization_metrics(
    tail_x: torch.Tensor,
    base_tail_logits: torch.Tensor,
    actual_tail_logits: torch.Tensor,
    before: list[torch.Tensor],
    directions: list[torch.Tensor],
    step_size: float,
) -> dict[str, float]:
    actual_delta = actual_tail_logits.detach() - base_tail_logits.detach()
    jvp_delta = tail_output_jvp(tail_x, before, directions, step_size)
    residual = actual_delta - jvp_delta
    jvp_norm = torch.linalg.norm(jvp_delta).clamp_min(torch.finfo(jvp_delta.dtype).eps)
    return {
        "tail_output_jvp_fro": float(torch.linalg.norm(jvp_delta).cpu()),
        "tail_output_linearization_residual_fro": float(torch.linalg.norm(residual).cpu()),
        "tail_output_linearization_relative_error": float((torch.linalg.norm(residual) / jvp_norm).cpu()),
    }


def update_norms(directions: list[torch.Tensor], step_size: float) -> tuple[float, float]:
    fro_sq = 0.0
    op_values = []
    for direction in directions:
        update = float(step_size) * direction
        fro_sq += float(torch.linalg.norm(update).cpu()) ** 2
        op_values.append(float(torch.linalg.matrix_norm(matrix_view(update), ord=2).cpu()))
    return fro_sq**0.5, max(op_values) if op_values else math.nan


def full_metrics(model: TinyMLP, x: torch.Tensor, y: torch.Tensor, indices: torch.Tensor) -> dict[str, float]:
    logits = model.logits(x[indices])
    loss = F.cross_entropy(logits, y[indices])
    pred = logits.argmax(dim=1)
    return {
        "loss": float(loss.detach().cpu()),
        "accuracy": float((pred == y[indices]).to(torch.float64).mean().cpu()),
        "mean_margin": float(margins(logits.detach(), y[indices]).mean().cpu()),
    }


def layer_rank_rows(params: list[torch.nn.Parameter], activations: list[torch.Tensor]) -> tuple[list[dict], dict]:
    sigma_g = [singular_values(param.grad.detach()) for param in params]
    sigma_a = [singular_values(activation.detach()) for activation in activations]
    derived = derive_step_metrics(sigma_g, sigma_a)
    rows = []
    for layer, (grad_sigmas, activation_sigmas, layer_metrics) in enumerate(
        zip(sigma_g, sigma_a, derived["layer_metrics"]),
        start=1,
    ):
        rows.append(
            {
                "layer": layer,
                "head_gradient_nuclear_rank": matrix_effective_rank(grad_sigmas),
                "tail_activation_stable_rank": stable_rank(activation_sigmas),
                "condition_score": layer_metrics["condition_score"],
            }
        )
    return rows, derived


def layer_direction(param: torch.nn.Parameter, geometry: str) -> torch.Tensor:
    grad = param.grad.detach()
    if geometry == "frobenius":
        return grad / max(float(torch.linalg.norm(grad).cpu()), 1e-300)
    if geometry == "spectral":
        grad_matrix = matrix_view(grad)
        u, _, vh = torch.linalg.svd(grad_matrix, full_matrices=False)
        return (u @ vh).reshape_as(param)
    raise ValueError(f"unknown geometry: {geometry}")


def activation_for_layer(model: TinyMLP, x: torch.Tensor, layer: int) -> torch.Tensor:
    with torch.no_grad():
        if layer == 1:
            return x.detach()
        if layer == 2:
            return F.relu(x @ model.w1.detach().T).detach()
    raise ValueError(f"unknown layer: {layer}")


def _sandwiched_stable_rank(b_sigmas: torch.Tensor, a_sigmas: torch.Tensor) -> float:
    count = min(int(b_sigmas.numel()), int(a_sigmas.numel()))
    if count == 0:
        return math.nan
    b_values = b_sigmas[:count]
    a_values = a_sigmas[:count]
    denom = b_values[0].square() * a_values[0].square()
    if float(denom.detach().cpu()) <= 0.0:
        return math.nan
    value = torch.sum(b_values.square() * a_values.square()) / denom
    return float(value.detach().cpu())


def tail_downstream_rank_metrics(model: TinyMLP, tail_x: torch.Tensor, layer: int) -> dict[str, float]:
    """Return downstream-aware tail-rank quantities at a fixed TinyMLP checkpoint.

    Layer 2 is an exact sandwich block, with B=I and A equal to the hidden tail
    activation matrix. Layer 1 has sample-dependent ReLU gates, so it is not a
    single B D A sandwich; for that layer we report the exact local linear
    operator stable rank under the frozen ReLU mask.
    """

    with torch.no_grad():
        if layer == 1:
            pre_activation = tail_x @ model.w1.detach().T
            mask = (pre_activation > 0).to(tail_x.dtype)
            operator = torch.einsum("ch,nh,ni->nchi", model.w2.detach(), mask, tail_x).reshape(
                tail_x.shape[0] * model.w2.shape[0],
                model.w1.numel(),
            )
            op_norm = torch.linalg.matrix_norm(operator, ord=2)
            fro_sq = torch.sum(operator.square())
            local_operator_stable_rank = fro_sq / op_norm.square().clamp_min(torch.finfo(tail_x.dtype).eps)
            return {
                "tail_sandwiched_stable_rank": math.nan,
                "tail_local_operator_stable_rank": float(local_operator_stable_rank.cpu()),
            }
        if layer == 2:
            hidden = F.relu(tail_x @ model.w1.detach().T)
            a_sigmas = torch.linalg.svdvals(hidden.T)
            b_sigmas = torch.ones(model.w2.shape[0], device=tail_x.device, dtype=tail_x.dtype)
            ssrank = _sandwiched_stable_rank(b_sigmas, a_sigmas)
            operator = (
                torch.eye(model.w2.shape[0], device=tail_x.device, dtype=tail_x.dtype)[None, :, :, None]
                * hidden[:, None, None, :]
            ).reshape(tail_x.shape[0] * model.w2.shape[0], model.w2.numel())
            op_norm = torch.linalg.matrix_norm(operator, ord=2)
            fro_sq = torch.sum(operator.square())
            local_operator_stable_rank = fro_sq / op_norm.square().clamp_min(torch.finfo(tail_x.dtype).eps)
            return {
                "tail_sandwiched_stable_rank": ssrank,
                "tail_local_operator_stable_rank": float(local_operator_stable_rank.cpu()),
            }
    raise ValueError(f"unknown layer: {layer}")


def jvp_tail_drift_sq(
    model: TinyMLP,
    param: torch.nn.Parameter,
    direction: torch.Tensor,
    tail_x: torch.Tensor,
    base_tail_logits: torch.Tensor,
    epsilon: float,
) -> float:
    old = param.detach().clone()
    with torch.no_grad():
        param.copy_(old + float(epsilon) * direction)
        perturbed = model.logits(tail_x).detach()
        param.copy_(old)
    jvp = (perturbed - base_tail_logits) / float(epsilon)
    return float(torch.sum(jvp.square()).cpu())


def apply_layer_update(param: torch.nn.Parameter, direction: torch.Tensor, step_size: float) -> torch.Tensor:
    old = param.detach().clone()
    with torch.no_grad():
        param.copy_(old - float(step_size) * direction)
    return old


def restore_layer(param: torch.nn.Parameter, old: torch.Tensor) -> None:
    with torch.no_grad():
        param.copy_(old)
