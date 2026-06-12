from __future__ import annotations

import time
from collections.abc import Iterable

import pandas as pd
import torch

from .config import ExperimentConfig, ProblemSpec
from .diagnostics import derive_step_metrics, derive_update_metrics, json_dumps_nested, matrix_view, singular_values
from .optimizers import ExactMuon
from .problems import (
    DeepMNISTMLPProblem,
    MatrixFactorizationInputProblem,
    MatrixSensingProblem,
    MNISTConvNetProblem,
    MNISTPatchClassifierProblem,
    MNISTMLPProblem,
    SmallMLPDigitsProblem,
    TrainProblem,
)


def dtype_from_name(name: str) -> torch.dtype:
    dtype = getattr(torch, name)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"unknown torch dtype: {name}")
    return dtype


def build_problem(spec: ProblemSpec, seed: int, *, device: torch.device, dtype: torch.dtype) -> TrainProblem:
    if spec.family == "MatrixFactorizationInput":
        return MatrixFactorizationInputProblem(spec, seed, device=device, dtype=dtype)
    if spec.family == "MatrixSensing":
        return MatrixSensingProblem(spec, seed, device=device, dtype=dtype)
    if spec.family == "SmallMLPDigits":
        return SmallMLPDigitsProblem(spec, seed, device=device, dtype=dtype)
    if spec.family == "MNISTMLP":
        return MNISTMLPProblem(spec, seed, device=device, dtype=dtype)
    if spec.family == "DeepMNISTMLP":
        return DeepMNISTMLPProblem(spec, seed, device=device, dtype=dtype)
    if spec.family == "MNISTPatchClassifier":
        return MNISTPatchClassifierProblem(spec, seed, device=device, dtype=dtype)
    if spec.family == "MNISTConvNet":
        return MNISTConvNetProblem(spec, seed, device=device, dtype=dtype)
    raise ValueError(f"unknown problem family: {spec.family}")


def build_optimizer(algo: str, params: Iterable[torch.nn.Parameter], lr: float) -> torch.optim.Optimizer:
    if algo == "Adam":
        return torch.optim.Adam(list(params), lr=lr)
    if algo == "Muon":
        return ExactMuon(list(params), lr=lr)
    raise ValueError(f"unknown optimizer: {algo}")


def effective_train_batch_size(spec: ProblemSpec) -> int:
    mini_batch_families = {"SmallMLPDigits", "MNISTMLP", "DeepMNISTMLP", "MNISTPatchClassifier", "MNISTConvNet"}
    if spec.family not in mini_batch_families or spec.batch_size is None:
        return int(spec.num_samples)
    return int(max(1, min(spec.batch_size, spec.num_samples)))


def collect_diagnostics(problem: TrainProblem) -> tuple[list[list[float]], list[list[float]]]:
    params = problem.parameters()
    sigma_g = [singular_values(param.grad) for param in params]
    sigma_a = [singular_values(matrix) for matrix in problem.activation_matrices()]
    if len(sigma_g) != len(sigma_a):
        raise ValueError(f"G/A layer count mismatch: {len(sigma_g)} vs {len(sigma_a)}")
    return sigma_g, sigma_a


def snapshot_parameters(params: list[torch.nn.Parameter]) -> list[torch.Tensor]:
    return [param.detach().clone() for param in params]


def update_norms(before: list[torch.Tensor], after: list[torch.nn.Parameter]) -> dict[str, float]:
    fro_sq = 0.0
    param_fro_sq = 0.0
    op_norms = []
    relative_layer_norms = []
    for old, param in zip(before, after):
        new = param.detach()
        diff = new - old
        diff_fro = float(torch.linalg.norm(diff).cpu())
        param_fro = float(torch.linalg.norm(old).cpu())
        fro_sq += diff_fro**2
        param_fro_sq += param_fro**2
        op_norms.append(float(torch.linalg.matrix_norm(matrix_view(diff), ord=2).cpu()))
        relative_layer_norms.append(diff_fro / max(param_fro, 1e-300))
    update_fro = float(fro_sq**0.5)
    param_fro = float(param_fro_sq**0.5)
    return {
        "update_fro_norm": update_fro,
        "update_op_norm": float(max(op_norms)) if op_norms else float("nan"),
        "relative_update_fro_norm": update_fro / max(param_fro, 1e-300),
        "mean_relative_layer_update_norm": float(sum(relative_layer_norms) / len(relative_layer_norms)) if relative_layer_norms else float("nan"),
    }


def update_singular_values(before: list[torch.Tensor], after: list[torch.nn.Parameter]) -> list[list[float]]:
    return [singular_values(param.detach() - old) for old, param in zip(before, after)]


def update_gradient_alignment(before: list[torch.Tensor], after: list[torch.nn.Parameter]) -> dict:
    inner = 0.0
    grad_fro_sq = 0.0
    update_fro_sq = 0.0
    layer_rows = []
    for old, param in zip(before, after):
        if param.grad is None:
            layer_rows.append(
                {
                    "update_grad_inner": float("nan"),
                    "update_grad_cosine": float("nan"),
                    "update_grad_per_update_norm": float("nan"),
                }
            )
            continue
        grad = param.grad.detach()
        descent_update = old - param.detach()
        layer_inner = float(torch.sum(grad * descent_update).cpu())
        layer_grad_fro = float(torch.linalg.norm(grad).cpu())
        layer_update_fro = float(torch.linalg.norm(descent_update).cpu())
        inner += layer_inner
        grad_fro_sq += layer_grad_fro**2
        update_fro_sq += layer_update_fro**2
        layer_rows.append(
            {
                "update_grad_inner": layer_inner,
                "update_grad_cosine": layer_inner / max(layer_grad_fro * layer_update_fro, 1e-300),
                "update_grad_per_update_norm": layer_inner / max(layer_update_fro, 1e-300),
            }
        )
    grad_fro = grad_fro_sq**0.5
    update_fro = update_fro_sq**0.5
    return {
        "update_grad_inner": inner,
        "update_grad_cosine": inner / max(grad_fro * update_fro, 1e-300),
        "update_grad_per_update_norm": inner / max(update_fro, 1e-300),
        "layer_metrics": layer_rows,
    }


def relative_update_norm(before: list[torch.Tensor], after: list[torch.nn.Parameter]) -> float:
    fro_sq = 0.0
    param_fro_sq = 0.0
    for old, param in zip(before, after):
        diff = param.detach() - old
        fro_sq += float(torch.linalg.norm(diff).cpu()) ** 2
        param_fro_sq += float(torch.linalg.norm(old).cpu()) ** 2
    return (fro_sq**0.5) / max(param_fro_sq**0.5, 1e-300)


def rescale_update(before: list[torch.Tensor], after: list[torch.nn.Parameter], target_relative_update: float) -> None:
    proposed = relative_update_norm(before, after)
    if proposed <= 0.0:
        return
    scale = target_relative_update / proposed
    with torch.no_grad():
        for old, param in zip(before, after):
            param.copy_(old + scale * (param.detach() - old))


def append_step_diagnostics(
    *,
    problem: TrainProblem,
    spec: ProblemSpec,
    algo: str,
    seed: int,
    run_id: int,
    step: int,
    started: float,
    step_rows: list[dict],
    layer_rows: list[dict],
) -> tuple[int, int]:
    problem.set_train_step(step)
    loss = problem.loss()
    loss.backward()
    sigma_g, sigma_a = collect_diagnostics(problem)
    derived = derive_step_metrics(sigma_g, sigma_a)
    loss_value = float(loss.detach().cpu())
    row = {
        "run_id": run_id,
        "problem_family": spec.family,
        "setting": spec.setting,
        "kappa": float(spec.kappa),
        "lr": float(spec.lr),
        "algo": algo,
        "seed": int(seed),
        "step": int(step),
        "num_samples": int(spec.num_samples),
        "train_batch_size": effective_train_batch_size(spec),
        "loss": loss_value,
        "delta_loss": float("nan"),
        "recovery_error": problem.recovery_error(),
        "diagnostic_A_definition": problem.diagnostic_a_definition,
        "nrG": derived["nrG"],
        "stA": derived["stA"],
        "condition_score": derived["condition_score"],
        "delta_gd_pred": derived["delta_gd_pred"],
        "delta_spec_pred": derived["delta_spec_pred"],
        "sigma_G": json_dumps_nested(sigma_g),
        "sigma_A": json_dumps_nested(sigma_a),
        "sigma_update": "",
        "nrUpdate": float("nan"),
        "stUpdate": float("nan"),
        "nrUpdateFrac": float("nan"),
        "stUpdateFrac": float("nan"),
        "update_flatness": float("nan"),
        "update_grad_inner": float("nan"),
        "update_grad_cosine": float("nan"),
        "update_grad_per_update_norm": float("nan"),
        "update_fro_norm": float("nan"),
        "update_op_norm": float("nan"),
        "relative_update_fro_norm": float("nan"),
        "mean_relative_layer_update_norm": float("nan"),
        "elapsed_s": time.perf_counter() - started,
    }
    step_rows.append(row)
    layer_start = len(layer_rows)

    for layer_metric in derived["layer_metrics"]:
        layer_rows.append(
            {
                "run_id": run_id,
                "problem_family": spec.family,
                "setting": spec.setting,
                "kappa": float(spec.kappa),
                "lr": float(spec.lr),
                "algo": algo,
                "seed": int(seed),
                "step": int(step),
                "num_samples": int(spec.num_samples),
                "train_batch_size": effective_train_batch_size(spec),
                "loss": loss_value,
                "recovery_error": row["recovery_error"],
                "diagnostic_A_definition": problem.diagnostic_a_definition,
                "sigma_update": "",
                "update_rank_ceiling": float("nan"),
                "nrUpdate": float("nan"),
                "stUpdate": float("nan"),
                "nrUpdateFrac": float("nan"),
                "stUpdateFrac": float("nan"),
                "update_flatness": float("nan"),
                "update_grad_inner": float("nan"),
                "update_grad_cosine": float("nan"),
                "update_grad_per_update_norm": float("nan"),
                **layer_metric,
            }
            )
    return len(step_rows) - 1, layer_start


def apply_update_diagnostics(
    *,
    before: list[torch.Tensor],
    after: list[torch.nn.Parameter],
    step_row: dict,
    layer_rows: list[dict],
    layer_start: int,
) -> None:
    sigma_update = update_singular_values(before, after)
    update_metrics = derive_update_metrics(sigma_update)
    alignment = update_gradient_alignment(before, after)
    step_row.update(
        {
            "sigma_update": json_dumps_nested(sigma_update),
            "nrUpdate": update_metrics["nrUpdate"],
            "stUpdate": update_metrics["stUpdate"],
            "nrUpdateFrac": update_metrics["nrUpdateFrac"],
            "stUpdateFrac": update_metrics["stUpdateFrac"],
            "update_flatness": update_metrics["update_flatness"],
            "update_grad_inner": alignment["update_grad_inner"],
            "update_grad_cosine": alignment["update_grad_cosine"],
            "update_grad_per_update_norm": alignment["update_grad_per_update_norm"],
            **update_norms(before, after),
        }
    )
    for offset, layer_metric in enumerate(update_metrics["layer_metrics"]):
        row = layer_rows[layer_start + offset]
        layer_alignment = alignment["layer_metrics"][offset]
        row["sigma_update"] = json_dumps_nested([sigma_update[offset]])
        row["update_rank_ceiling"] = layer_metric["update_rank_ceiling"]
        row["nrUpdate"] = layer_metric["nrUpdate"]
        row["stUpdate"] = layer_metric["stUpdate"]
        row["nrUpdateFrac"] = layer_metric["nrUpdateFrac"]
        row["stUpdateFrac"] = layer_metric["stUpdateFrac"]
        row["update_flatness"] = layer_metric["update_flatness"]
        row["update_grad_inner"] = layer_alignment["update_grad_inner"]
        row["update_grad_cosine"] = layer_alignment["update_grad_cosine"]
        row["update_grad_per_update_norm"] = layer_alignment["update_grad_per_update_norm"]


def run_single(spec: ProblemSpec, algo: str, seed: int, run_id: int, config: ExperimentConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    device = torch.device(config.device)
    dtype = dtype_from_name(config.dtype)
    problem = build_problem(spec, seed, device=device, dtype=dtype)
    params = problem.parameters()
    optimizer = build_optimizer(algo, params, spec.lr)
    step_rows: list[dict] = []
    layer_rows: list[dict] = []
    started = time.perf_counter()

    for step in range(spec.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        current_row_index, layer_start = append_step_diagnostics(
            problem=problem,
            spec=spec,
            algo=algo,
            seed=seed,
            run_id=run_id,
            step=step,
            started=started,
            step_rows=step_rows,
            layer_rows=layer_rows,
        )

        if step < spec.steps:
            before = snapshot_parameters(params)
            optimizer.step()
            post_update_loss = float(problem.loss().detach().cpu())
            step_rows[current_row_index]["delta_loss"] = step_rows[current_row_index]["loss"] - post_update_loss
            apply_update_diagnostics(
                before=before,
                after=params,
                step_row=step_rows[current_row_index],
                layer_rows=layer_rows,
                layer_start=layer_start,
            )

    return pd.DataFrame(step_rows), pd.DataFrame(layer_rows)


def run_experiment(config: ExperimentConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    step_frames = []
    layer_frames = []
    run_id = 0
    for spec in config.specs:
        for algo in config.algos:
            for seed in config.seeds:
                step_frame, layer_frame = run_single(spec, algo, seed, run_id, config)
                step_frames.append(step_frame)
                layer_frames.append(layer_frame)
                run_id += 1
    steps = pd.concat(step_frames, ignore_index=True)
    layers = pd.concat(layer_frames, ignore_index=True)
    return steps, layers


def run_equal_update_pair(
    spec: ProblemSpec,
    seed: int,
    start_run_id: int,
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    device = torch.device(config.device)
    dtype = dtype_from_name(config.dtype)
    problems = {
        "Adam": build_problem(spec, seed, device=device, dtype=dtype),
        "Muon": build_problem(spec, seed, device=device, dtype=dtype),
    }
    params = {algo: problem.parameters() for algo, problem in problems.items()}
    optimizers = {algo: build_optimizer(algo, params[algo], spec.lr) for algo in ["Adam", "Muon"]}
    run_ids = {"Adam": start_run_id, "Muon": start_run_id + 1}
    step_rows: list[dict] = []
    layer_rows: list[dict] = []
    current_row_indices: dict[str, int] = {"Adam": 0, "Muon": 0}
    current_layer_starts: dict[str, int] = {"Adam": 0, "Muon": 0}
    started = time.perf_counter()

    for step in range(spec.steps + 1):
        for algo in ["Adam", "Muon"]:
            optimizers[algo].zero_grad(set_to_none=True)
            current_row_index, layer_start = append_step_diagnostics(
                problem=problems[algo],
                spec=spec,
                algo=algo,
                seed=seed,
                run_id=run_ids[algo],
                step=step,
                started=started,
                step_rows=step_rows,
                layer_rows=layer_rows,
            )
            current_row_indices[algo] = current_row_index
            current_layer_starts[algo] = layer_start

        if step < spec.steps:
            before = {algo: snapshot_parameters(params[algo]) for algo in ["Adam", "Muon"]}
            for algo in ["Adam", "Muon"]:
                optimizers[algo].step()
            proposed = {algo: relative_update_norm(before[algo], params[algo]) for algo in ["Adam", "Muon"]}
            target = min(value for value in proposed.values() if value > 0.0)
            for algo in ["Adam", "Muon"]:
                rescale_update(before[algo], params[algo], target)
                post_update_loss = float(problems[algo].loss().detach().cpu())
                step_rows[current_row_indices[algo]]["delta_loss"] = (
                    step_rows[current_row_indices[algo]]["loss"] - post_update_loss
                )
                apply_update_diagnostics(
                    before=before[algo],
                    after=params[algo],
                    step_row=step_rows[current_row_indices[algo]],
                    layer_rows=layer_rows,
                    layer_start=current_layer_starts[algo],
                )

    return pd.DataFrame(step_rows), pd.DataFrame(layer_rows)


def run_equal_update_experiment(config: ExperimentConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    step_frames = []
    layer_frames = []
    run_id = 0
    for spec in config.specs:
        for seed in config.seeds:
            step_frame, layer_frame = run_equal_update_pair(spec, seed, run_id, config)
            step_frames.append(step_frame)
            layer_frames.append(layer_frame)
            run_id += 2
    steps = pd.concat(step_frames, ignore_index=True)
    layers = pd.concat(layer_frames, ignore_index=True)
    return steps, layers
