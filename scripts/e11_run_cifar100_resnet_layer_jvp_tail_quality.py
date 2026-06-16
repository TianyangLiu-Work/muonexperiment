from __future__ import annotations

import argparse
import gc
import json
import math
import random
import sys
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.cifar100_resnet_tail import (
    Cifar100ResNetOneStepConfig,
    batch,
    build_cifar_resnet18,
    dtype_from_name,
    full_metrics,
    matrix_named_parameters,
    resolve_device,
    split_long_tail_cifar100,
    train_checkpoint,
)
from e11_condition_geometry.diagnostics import matrix_effective_rank, matrix_view
from e11_condition_geometry.long_tail_digits import make_generator
from e11_condition_geometry.statistics import ci95, corr_ci95, log_ratio_ci95
from e11_condition_geometry.reporting import fmt, markdown_table


DEFAULT_OUTPUT_DIR = Path("results/e11_cifar100_resnet_layer_jvp_tail_quality")
DEFAULT_FIGURE_DIR = Path("figures/e11_cifar100_resnet_layer_jvp_tail_quality")
DEFAULT_DISCUSSION_PATH = Path("discussion/e11_cifar100_resnet_layer_jvp_tail_quality.md")


def _finite_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / max(denominator, 1e-300))


def _tensor_float(value: torch.Tensor) -> float:
    return float(value.detach().cpu())


def layer_profiles(named_params: list[tuple[str, torch.nn.Parameter]]) -> list[dict]:
    profiles = []
    for layer_index, (name, param) in enumerate(named_params, start=1):
        grad = param.grad.detach()
        grad_matrix = matrix_view(grad)
        grad_fro = torch.linalg.norm(grad_matrix)
        grad_fro_value = _tensor_float(grad_fro)
        if grad_fro_value <= 0.0:
            spectral_direction = torch.zeros_like(param)
            singular_values = torch.zeros(1, device=param.device, dtype=param.dtype)
        else:
            u, singular_values, vh = torch.linalg.svd(grad_matrix, full_matrices=False)
            spectral_direction = (u @ vh).reshape_as(param)
        fro_direction = grad / max(grad_fro_value, 1e-300)
        grad_op = _tensor_float(singular_values.max()) if singular_values.numel() else math.nan
        grad_nuclear = _tensor_float(singular_values.sum()) if singular_values.numel() else math.nan
        profiles.append(
            {
                "layer_index": int(layer_index),
                "parameter": name,
                "shape": "x".join(str(dim) for dim in param.shape),
                "param": param,
                "gradient_nuclear_rank": matrix_effective_rank([float(value) for value in singular_values.detach().cpu()]),
                "gradient_fro_norm": grad_fro_value,
                "gradient_op_norm": grad_op,
                "gradient_nuclear_norm": grad_nuclear,
                "frobenius_direction": fro_direction,
                "spectral_direction": spectral_direction,
            }
        )
    return profiles


def direction_for(profile: dict, geometry: str) -> torch.Tensor:
    if geometry == "frobenius":
        return profile["frobenius_direction"]
    if geometry == "spectral":
        return profile["spectral_direction"]
    raise ValueError(f"unknown geometry: {geometry}")


def direction_norms(direction: torch.Tensor) -> tuple[float, float]:
    direction_matrix = matrix_view(direction)
    return (
        _tensor_float(torch.linalg.norm(direction_matrix)),
        _tensor_float(torch.linalg.matrix_norm(direction_matrix, ord=2)),
    )


def finite_difference_tail_jvp(
    model: torch.nn.Module,
    param: torch.nn.Parameter,
    direction: torch.Tensor,
    tail_x: torch.Tensor,
    base_tail_logits: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    old = param.detach().clone()
    with torch.no_grad():
        param.copy_(old + float(epsilon) * direction)
        perturbed = model(tail_x).detach()
        param.copy_(old)
    return (perturbed - base_tail_logits) / float(epsilon)


def apply_layer_update(param: torch.nn.Parameter, direction: torch.Tensor, step_size: float) -> torch.Tensor:
    old = param.detach().clone()
    with torch.no_grad():
        param.copy_(old - float(step_size) * direction)
    return old


def restore_layer(param: torch.nn.Parameter, old: torch.Tensor) -> None:
    with torch.no_grad():
        param.copy_(old)


def run_layer_jvp_probe(
    config: Cifar100ResNetOneStepConfig,
    *,
    jvp_epsilon: float,
    max_matrix_parameters: int | None = None,
    progress: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    random.seed(0)
    torch.manual_seed(0)
    device = resolve_device(config.device)
    dtype = dtype_from_name(config.dtype)
    rows: list[dict] = []

    for seed in config.seeds:
        if progress:
            print(f"[resnet-layer-jvp] seed={seed}: loading data", flush=True)
        train_x, train_y, test_x, test_y, train_indices, head_train, tail_train, tail_eval = split_long_tail_cifar100(
            config,
            seed=seed,
            device=device,
            dtype=dtype,
        )
        torch.manual_seed(seed + 33000)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed + 33000)
        model = build_cifar_resnet18(device=device, dtype=dtype)
        if progress:
            print(f"[resnet-layer-jvp] seed={seed}: training {config.warmup_steps} warmup steps", flush=True)
        train_checkpoint(model, train_x, train_y, train_indices, config, seed=seed)
        model.eval()

        head_generator = make_generator(seed + 34000, device)
        head_batch = batch(head_train, config.head_batch_size, generator=head_generator)
        head_x = train_x[head_batch]
        head_y = train_y[head_batch]
        tail_x = test_x[tail_eval]
        tail_y = test_y[tail_eval]

        model.zero_grad(set_to_none=True)
        head_loss = F.cross_entropy(model(head_x), head_y)
        head_loss.backward()
        target_gain = config.target_head_gain_fraction * float(head_loss.detach().cpu())
        with torch.no_grad():
            base_tail_logits = model(tail_x).detach()
        base_tail = full_metrics(model, test_x, test_y, tail_eval)
        base_head_loss = float(head_loss.detach().cpu())
        profiles = layer_profiles(matrix_named_parameters(model))
        if max_matrix_parameters is not None:
            profiles = profiles[: int(max_matrix_parameters)]
        if progress:
            print(f"[resnet-layer-jvp] seed={seed}: probing {len(profiles)} matrix layers", flush=True)

        for profile in profiles:
            param = profile["param"]
            for geometry in ("frobenius", "spectral"):
                direction = direction_for(profile, geometry)
                alignment = _tensor_float(torch.sum(param.grad.detach() * direction))
                step_size = target_gain / max(alignment, 1e-300)
                direction_fro, direction_op = direction_norms(direction)
                jvp = finite_difference_tail_jvp(model, param, direction, tail_x, base_tail_logits, jvp_epsilon)
                jvp_fro = torch.linalg.norm(jvp)
                scaled_jvp = -float(step_size) * jvp
                old = apply_layer_update(param, direction, step_size)
                with torch.no_grad():
                    tail_logits = model(tail_x).detach()
                    head_after_loss = float(F.cross_entropy(model(head_x), head_y).detach().cpu())
                tail = full_metrics(model, test_x, test_y, tail_eval)
                tail_delta = tail_logits - base_tail_logits
                residual = tail_delta - scaled_jvp
                scaled_jvp_fro = torch.linalg.norm(scaled_jvp).clamp_min(torch.finfo(scaled_jvp.dtype).eps)
                restore_layer(param, old)

                rows.append(
                    {
                        "seed": int(seed),
                        "geometry": geometry,
                        "dataset": "CIFAR100",
                        "model": "resnet18_cifar_stem",
                        "updated_parameter_subset": "single_matrix_weight",
                        "layer_index": int(profile["layer_index"]),
                        "parameter": str(profile["parameter"]),
                        "shape": str(profile["shape"]),
                        "head_classes": ",".join(str(x) for x in config.head_classes),
                        "tail_classes": ",".join(str(x) for x in config.tail_classes),
                        "head_train_examples": int(head_train.numel()),
                        "tail_train_examples": int(tail_train.numel()),
                        "tail_eval_examples": int(tail_eval.numel()),
                        "warmup_steps": int(config.warmup_steps),
                        "jvp_epsilon": float(jvp_epsilon),
                        "head_loss_before": base_head_loss,
                        "head_loss_after": head_after_loss,
                        "actual_head_loss_decrease": base_head_loss - head_after_loss,
                        "actual_head_gain_relative_error": abs((base_head_loss - head_after_loss) - target_gain)
                        / max(target_gain, 1e-300),
                        "matched_first_order_head_gain": target_gain,
                        "alignment": alignment,
                        "step_size": float(step_size),
                        "gradient_nuclear_rank": float(profile["gradient_nuclear_rank"]),
                        "gradient_fro_norm": float(profile["gradient_fro_norm"]),
                        "gradient_op_norm": float(profile["gradient_op_norm"]),
                        "gradient_nuclear_norm": float(profile["gradient_nuclear_norm"]),
                        "direction_fro_norm": direction_fro,
                        "direction_op_norm": direction_op,
                        "jvp_tail_drift_fro": _tensor_float(jvp_fro),
                        "jvp_tail_drift_sq": _tensor_float(torch.sum(jvp.square())),
                        "scaled_jvp_tail_drift_fro": _tensor_float(scaled_jvp_fro),
                        "scaled_jvp_tail_drift_sq": _tensor_float(torch.sum(scaled_jvp.square())),
                        "tail_output_drift_fro": _tensor_float(torch.linalg.norm(tail_delta)),
                        "tail_output_drift_rms": _tensor_float(torch.sqrt(torch.mean(tail_delta.square()))),
                        "tail_output_linearization_residual_fro": _tensor_float(torch.linalg.norm(residual)),
                        "tail_output_linearization_relative_error": _tensor_float(torch.linalg.norm(residual) / scaled_jvp_fro),
                        "tail_loss_before": base_tail["loss"],
                        "tail_loss_after": tail["loss"],
                        "tail_loss_increase": tail["loss"] - base_tail["loss"],
                        "tail_accuracy_before": base_tail["accuracy"],
                        "tail_accuracy_after": tail["accuracy"],
                        "tail_accuracy_drop": base_tail["accuracy"] - tail["accuracy"],
                        "tail_margin_before": base_tail["mean_margin"],
                        "tail_margin_after": tail["mean_margin"],
                        "tail_margin_drop": base_tail["mean_margin"] - tail["mean_margin"],
                        "device": str(device),
                        "dtype": str(dtype).replace("torch.", ""),
                    }
                )
            if progress:
                print(
                    f"[resnet-layer-jvp] seed={seed}: layer {profile['layer_index']} {profile['parameter']} done",
                    flush=True,
                )

        if progress:
            print(f"[resnet-layer-jvp] seed={seed}: done", flush=True)
        del model, train_x, train_y, test_x, test_y
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    metrics = pd.DataFrame(rows)
    paired = paired_layer_metrics(metrics)
    summary = summarize_layer_jvp(paired)
    overall = summarize_overall_layer_jvp(paired)
    return metrics, paired, summary, overall


def paired_layer_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    paired_rows = []
    for (seed, layer_index, parameter), group in metrics.groupby(
        ["seed", "layer_index", "parameter"],
        observed=True,
        sort=False,
    ):
        by_geometry = group.set_index("geometry")
        fro = by_geometry.loc["frobenius"]
        spectral = by_geometry.loc["spectral"]
        paired_rows.append(
            {
                "seed": int(seed),
                "layer_index": int(layer_index),
                "parameter": str(parameter),
                "shape": str(spectral["shape"]),
                "warmup_steps": int(spectral["warmup_steps"]),
                "gradient_nuclear_rank": float(spectral["gradient_nuclear_rank"]),
                "gradient_fro_norm": float(spectral["gradient_fro_norm"]),
                "gradient_op_norm": float(spectral["gradient_op_norm"]),
                "alignment_ratio_spectral_over_fro": _finite_ratio(float(spectral["alignment"]), float(fro["alignment"])),
                "step_size_ratio_spectral_over_fro": _finite_ratio(float(spectral["step_size"]), float(fro["step_size"])),
                "jvp_tail_drift_sq_ratio_spectral_over_fro": _finite_ratio(
                    float(spectral["jvp_tail_drift_sq"]),
                    float(fro["jvp_tail_drift_sq"]),
                ),
                "scaled_jvp_tail_drift_sq_ratio_spectral_over_fro": _finite_ratio(
                    float(spectral["scaled_jvp_tail_drift_sq"]),
                    float(fro["scaled_jvp_tail_drift_sq"]),
                ),
                "observed_tail_drift_sq_ratio_spectral_over_fro": _finite_ratio(
                    float(spectral["tail_output_drift_fro"]) ** 2,
                    float(fro["tail_output_drift_fro"]) ** 2,
                ),
                "linearization_relative_error_diff_spectral_minus_fro": float(
                    spectral["tail_output_linearization_relative_error"]
                )
                - float(fro["tail_output_linearization_relative_error"]),
                "tail_loss_increase_diff_spectral_minus_fro": float(spectral["tail_loss_increase"])
                - float(fro["tail_loss_increase"]),
                "tail_accuracy_drop_diff_spectral_minus_fro": float(spectral["tail_accuracy_drop"])
                - float(fro["tail_accuracy_drop"]),
                "tail_margin_drop_diff_spectral_minus_fro": float(spectral["tail_margin_drop"])
                - float(fro["tail_margin_drop"]),
                "actual_head_loss_decrease_diff_spectral_minus_fro": float(spectral["actual_head_loss_decrease"])
                - float(fro["actual_head_loss_decrease"]),
                "tail_accuracy_before": float(spectral["tail_accuracy_before"]),
                "tail_loss_before": float(spectral["tail_loss_before"]),
                "spectral_less_jvp_tail_drift": bool(
                    spectral["jvp_tail_drift_sq"] < fro["jvp_tail_drift_sq"]
                ),
                "spectral_less_scaled_jvp_tail_drift": bool(
                    spectral["scaled_jvp_tail_drift_sq"] < fro["scaled_jvp_tail_drift_sq"]
                ),
                "spectral_less_observed_tail_drift": bool(
                    spectral["tail_output_drift_fro"] < fro["tail_output_drift_fro"]
                ),
            }
        )
    return pd.DataFrame(paired_rows)


def summarize_layer_jvp(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (layer_index, parameter), group in paired.groupby(["layer_index", "parameter"], observed=True, sort=True):
        jvp_mean, jvp_low, jvp_high = log_ratio_ci95(group["jvp_tail_drift_sq_ratio_spectral_over_fro"])
        scaled_mean, scaled_low, scaled_high = log_ratio_ci95(
            group["scaled_jvp_tail_drift_sq_ratio_spectral_over_fro"]
        )
        observed_mean, observed_low, observed_high = log_ratio_ci95(
            group["observed_tail_drift_sq_ratio_spectral_over_fro"]
        )
        loss_mean, loss_low, loss_high = ci95(group["tail_loss_increase_diff_spectral_minus_fro"])
        acc_mean, acc_low, acc_high = ci95(group["tail_accuracy_drop_diff_spectral_minus_fro"])
        lin_mean, lin_low, lin_high = ci95(group["linearization_relative_error_diff_spectral_minus_fro"])
        rows.append(
            {
                "layer_index": int(layer_index),
                "parameter": str(parameter),
                "shape": str(group["shape"].iloc[0]),
                "seeds": int(group["seed"].nunique()),
                "mean_gradient_nuclear_rank": float(group["gradient_nuclear_rank"].mean()),
                "mean_alignment_ratio_spectral_over_fro": float(group["alignment_ratio_spectral_over_fro"].mean()),
                "mean_step_size_ratio_spectral_over_fro": float(group["step_size_ratio_spectral_over_fro"].mean()),
                "mean_tail_accuracy_before": float(group["tail_accuracy_before"].mean()),
                "geomean_jvp_tail_drift_sq_ratio_spectral_over_fro": jvp_mean,
                "jvp_tail_drift_sq_ratio_ci95_low": jvp_low,
                "jvp_tail_drift_sq_ratio_ci95_high": jvp_high,
                "spectral_less_jvp_tail_drift_fraction": float(group["spectral_less_jvp_tail_drift"].mean()),
                "geomean_scaled_jvp_tail_drift_sq_ratio_spectral_over_fro": scaled_mean,
                "scaled_jvp_tail_drift_sq_ratio_ci95_low": scaled_low,
                "scaled_jvp_tail_drift_sq_ratio_ci95_high": scaled_high,
                "spectral_less_scaled_jvp_tail_drift_fraction": float(
                    group["spectral_less_scaled_jvp_tail_drift"].mean()
                ),
                "geomean_observed_tail_drift_sq_ratio_spectral_over_fro": observed_mean,
                "observed_tail_drift_sq_ratio_ci95_low": observed_low,
                "observed_tail_drift_sq_ratio_ci95_high": observed_high,
                "spectral_less_observed_tail_drift_fraction": float(group["spectral_less_observed_tail_drift"].mean()),
                "mean_tail_loss_increase_diff_spectral_minus_fro": loss_mean,
                "tail_loss_increase_diff_ci95_low": loss_low,
                "tail_loss_increase_diff_ci95_high": loss_high,
                "mean_tail_accuracy_drop_diff_spectral_minus_fro": acc_mean,
                "tail_accuracy_drop_diff_ci95_low": acc_low,
                "tail_accuracy_drop_diff_ci95_high": acc_high,
                "mean_linearization_relative_error_diff_spectral_minus_fro": lin_mean,
                "linearization_relative_error_diff_ci95_low": lin_low,
                "linearization_relative_error_diff_ci95_high": lin_high,
            }
        )
    return pd.DataFrame(rows)


def summarize_overall_layer_jvp(paired: pd.DataFrame) -> pd.DataFrame:
    jvp_mean, jvp_low, jvp_high = log_ratio_ci95(paired["jvp_tail_drift_sq_ratio_spectral_over_fro"])
    scaled_mean, scaled_low, scaled_high = log_ratio_ci95(paired["scaled_jvp_tail_drift_sq_ratio_spectral_over_fro"])
    observed_mean, observed_low, observed_high = log_ratio_ci95(paired["observed_tail_drift_sq_ratio_spectral_over_fro"])
    loss_mean, loss_low, loss_high = ci95(paired["tail_loss_increase_diff_spectral_minus_fro"])
    acc_mean, acc_low, acc_high = ci95(paired["tail_accuracy_drop_diff_spectral_minus_fro"])
    jvp_observed_corr, corr_low, corr_high, corr_points = corr_ci95(
        paired["scaled_jvp_tail_drift_sq_ratio_spectral_over_fro"],
        paired["observed_tail_drift_sq_ratio_spectral_over_fro"],
    )
    return pd.DataFrame(
        [
            {
                "setting": "all_resnet_matrix_layers",
                "seeds": int(paired["seed"].nunique()),
                "parameters": int(paired["parameter"].nunique()),
                "paired_points": int(len(paired)),
                "warmup_steps": int(paired["warmup_steps"].iloc[0]) if not paired.empty else math.nan,
                "mean_tail_accuracy_before": float(paired["tail_accuracy_before"].mean()),
                "geomean_jvp_tail_drift_sq_ratio_spectral_over_fro": jvp_mean,
                "jvp_tail_drift_sq_ratio_ci95_low": jvp_low,
                "jvp_tail_drift_sq_ratio_ci95_high": jvp_high,
                "spectral_less_jvp_tail_drift_fraction": float(paired["spectral_less_jvp_tail_drift"].mean()),
                "geomean_scaled_jvp_tail_drift_sq_ratio_spectral_over_fro": scaled_mean,
                "scaled_jvp_tail_drift_sq_ratio_ci95_low": scaled_low,
                "scaled_jvp_tail_drift_sq_ratio_ci95_high": scaled_high,
                "spectral_less_scaled_jvp_tail_drift_fraction": float(
                    paired["spectral_less_scaled_jvp_tail_drift"].mean()
                ),
                "geomean_observed_tail_drift_sq_ratio_spectral_over_fro": observed_mean,
                "observed_tail_drift_sq_ratio_ci95_low": observed_low,
                "observed_tail_drift_sq_ratio_ci95_high": observed_high,
                "spectral_less_observed_tail_drift_fraction": float(
                    paired["spectral_less_observed_tail_drift"].mean()
                ),
                "mean_tail_loss_increase_diff_spectral_minus_fro": loss_mean,
                "tail_loss_increase_diff_ci95_low": loss_low,
                "tail_loss_increase_diff_ci95_high": loss_high,
                "mean_tail_accuracy_drop_diff_spectral_minus_fro": acc_mean,
                "tail_accuracy_drop_diff_ci95_low": acc_low,
                "tail_accuracy_drop_diff_ci95_high": acc_high,
                "scaled_jvp_observed_ratio_spearman": jvp_observed_corr,
                "scaled_jvp_observed_ratio_spearman_ci95_low": corr_low,
                "scaled_jvp_observed_ratio_spearman_ci95_high": corr_high,
                "scaled_jvp_observed_ratio_spearman_points": corr_points,
            }
        ]
    )


def write_figure(summary: pd.DataFrame, paired: pd.DataFrame, figure_dir: Path) -> Path:
    figure_dir.mkdir(parents=True, exist_ok=True)
    ordered = summary.sort_values("layer_index")
    labels = ordered["parameter"].tolist()
    y_values = list(range(len(ordered)))
    width = 0.38

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 7.2))
    axes[0].barh(
        [y - width / 2 for y in y_values],
        ordered["geomean_scaled_jvp_tail_drift_sq_ratio_spectral_over_fro"].clip(lower=1e-12),
        height=width,
        color="#0072B2",
        label="scaled JVP",
    )
    axes[0].barh(
        [y + width / 2 for y in y_values],
        ordered["geomean_observed_tail_drift_sq_ratio_spectral_over_fro"].clip(lower=1e-12),
        height=width,
        color="#D55E00",
        label="observed",
    )
    axes[0].axvline(1.0, color="black", linewidth=1.0, linestyle="--")
    axes[0].set_xscale("log")
    axes[0].set_yticks(y_values)
    axes[0].set_yticklabels(labels, fontsize=7)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("spectral / Fro squared tail-drift ratio")
    axes[0].set_title("Layer-only matched-head-gain interventions")
    axes[0].legend(frameon=False)

    axes[1].scatter(
        paired["scaled_jvp_tail_drift_sq_ratio_spectral_over_fro"].clip(lower=1e-12),
        paired["observed_tail_drift_sq_ratio_spectral_over_fro"].clip(lower=1e-12),
        s=28,
        alpha=0.75,
        color="#009E73",
    )
    axes[1].axhline(1.0, color="black", linewidth=1.0, linestyle="--")
    axes[1].axvline(1.0, color="black", linewidth=1.0, linestyle="--")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("scaled finite-difference JVP ratio")
    axes[1].set_ylabel("observed drift ratio")
    axes[1].set_title("JVP agreement check")

    fig.suptitle("CIFAR-100-LT ResNet18 all-layer tail-sensitivity diagnostic")
    fig.tight_layout()
    path = figure_dir / "cifar100_resnet_layer_jvp_tail_quality.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def write_discussion(
    config: Cifar100ResNetOneStepConfig,
    summary: pd.DataFrame,
    overall: pd.DataFrame,
    figure_path: Path,
    output_dir: Path,
    discussion_path: Path,
    *,
    jvp_epsilon: float,
) -> None:
    overall_row = overall.iloc[0]
    worst_observed = summary.loc[summary["observed_tail_drift_sq_ratio_ci95_high"].idxmax()]
    weakest_scaled = summary.loc[summary["scaled_jvp_tail_drift_sq_ratio_ci95_high"].idxmax()]
    supported_layers = int((summary["observed_tail_drift_sq_ratio_ci95_high"] < 1.0).sum())
    table_columns = [
        "layer_index",
        "parameter",
        "seeds",
        "mean_gradient_nuclear_rank",
        "mean_tail_accuracy_before",
        "geomean_scaled_jvp_tail_drift_sq_ratio_spectral_over_fro",
        "scaled_jvp_tail_drift_sq_ratio_ci95_low",
        "scaled_jvp_tail_drift_sq_ratio_ci95_high",
        "geomean_observed_tail_drift_sq_ratio_spectral_over_fro",
        "observed_tail_drift_sq_ratio_ci95_low",
        "observed_tail_drift_sq_ratio_ci95_high",
        "spectral_less_observed_tail_drift_fraction",
    ]
    lines = [
        "# E11 CIFAR-100-LT ResNet18 All-Layer JVP Tail-Quality Diagnostic",
        "",
        "This generated diagnostic probes every convolution and linear matrix weight",
        "in a CIFAR-100-LT ResNet18 checkpoint. Each intervention updates exactly one",
        "matrix parameter, matches the same first-order head-batch gain for",
        "Frobenius/GD and spectral/polar directions, and measures held-out tail-logit",
        "drift. It adds the missing downstream-aware check for layers before the",
        "final classifier, where the tail map is nonlinear and rank-only proxies are",
        "not enough.",
        "",
        f"- Seeds: {len(config.seeds)}",
        f"- Warmup steps: {config.warmup_steps}",
        f"- Head train examples per class: {config.head_train_per_class}",
        f"- Tail train examples per class: {config.tail_train_per_class}",
        f"- Tail eval examples per class: {config.tail_eval_per_class}",
        f"- Target head first-order gain: {config.target_head_gain_fraction} * head-batch loss",
        f"- Finite-difference JVP epsilon: {jvp_epsilon}",
        f"- Device/dtype request: {config.device}/{config.dtype}",
        "",
        f"![CIFAR-100-LT ResNet18 all-layer JVP](../{figure_path.as_posix()})",
        "",
        "## Overall Summary",
        "",
        markdown_table(
            overall,
            [
                "seeds",
                "parameters",
                "paired_points",
                "mean_tail_accuracy_before",
                "geomean_scaled_jvp_tail_drift_sq_ratio_spectral_over_fro",
                "scaled_jvp_tail_drift_sq_ratio_ci95_low",
                "scaled_jvp_tail_drift_sq_ratio_ci95_high",
                "geomean_observed_tail_drift_sq_ratio_spectral_over_fro",
                "observed_tail_drift_sq_ratio_ci95_low",
                "observed_tail_drift_sq_ratio_ci95_high",
                "spectral_less_observed_tail_drift_fraction",
                "scaled_jvp_observed_ratio_spearman",
            ],
        ),
        "",
        "## Per-Layer Summary",
        "",
        markdown_table(summary, table_columns),
        "",
        "## Readout",
        "",
        f"- Overall observed spectral/Fro squared drift ratio: "
        f"{fmt(overall_row['geomean_observed_tail_drift_sq_ratio_spectral_over_fro'])} "
        f"[{fmt(overall_row['observed_tail_drift_sq_ratio_ci95_low'])}, "
        f"{fmt(overall_row['observed_tail_drift_sq_ratio_ci95_high'])}].",
        f"- Overall scaled-JVP squared drift ratio: "
        f"{fmt(overall_row['geomean_scaled_jvp_tail_drift_sq_ratio_spectral_over_fro'])} "
        f"[{fmt(overall_row['scaled_jvp_tail_drift_sq_ratio_ci95_low'])}, "
        f"{fmt(overall_row['scaled_jvp_tail_drift_sq_ratio_ci95_high'])}].",
        f"- Per-parameter observed CI upper endpoints below one: {supported_layers}/{len(summary)}.",
        f"- Weakest observed layer by CI upper endpoint: {worst_observed['parameter']} "
        f"with {fmt(worst_observed['geomean_observed_tail_drift_sq_ratio_spectral_over_fro'])} "
        f"[{fmt(worst_observed['observed_tail_drift_sq_ratio_ci95_low'])}, "
        f"{fmt(worst_observed['observed_tail_drift_sq_ratio_ci95_high'])}].",
        f"- Weakest scaled-JVP layer by CI upper endpoint: {weakest_scaled['parameter']} "
        f"with {fmt(weakest_scaled['geomean_scaled_jvp_tail_drift_sq_ratio_spectral_over_fro'])} "
        f"[{fmt(weakest_scaled['scaled_jvp_tail_drift_sq_ratio_ci95_low'])}, "
        f"{fmt(weakest_scaled['scaled_jvp_tail_drift_sq_ratio_ci95_high'])}].",
        "",
        "Interpretation: this is not a benchmark-training claim. It is a mechanism",
        "check showing whether the local tail-sensitivity calculation and the actual",
        "one-layer intervention agree at a nontrivial ResNet tail-quality checkpoint.",
        "Large residuals or layer CI endpoints above one should be treated as paper",
        "risk rather than hidden.",
        "",
        "Artifacts:",
        f"- [metrics.csv](../{(output_dir / 'metrics.csv').as_posix()})",
        f"- [paired_metrics.csv](../{(output_dir / 'paired_metrics.csv').as_posix()})",
        f"- [summary.csv](../{(output_dir / 'summary.csv').as_posix()})",
        f"- [overall_summary.csv](../{(output_dir / 'overall_summary.csv').as_posix()})",
        f"- [config.json](../{(output_dir / 'config.json').as_posix()})",
    ]
    discussion_path.parent.mkdir(parents=True, exist_ok=True)
    discussion_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--warmup-steps", type=int, default=5000)
    parser.add_argument("--target-head-gain-fraction", type=float, default=0.005)
    parser.add_argument("--head-train-per-class", type=int, default=None)
    parser.add_argument("--tail-train-per-class", type=int, default=None)
    parser.add_argument("--tail-eval-per-class", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--head-batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--jvp-epsilon", type=float, default=1e-4)
    parser.add_argument("--max-matrix-parameters", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--discussion-path", type=Path, default=DEFAULT_DISCUSSION_PATH)
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> Cifar100ResNetOneStepConfig:
    config = Cifar100ResNetOneStepConfig()
    if args.smoke:
        config = replace(
            config,
            seeds=(0,),
            head_classes=tuple(range(5)),
            tail_classes=tuple(range(5, 10)),
            head_train_per_class=20,
            tail_train_per_class=5,
            tail_eval_per_class=5,
            warmup_steps=2,
            warmup_batch_size=16,
            head_batch_size=16,
            target_head_gain_fraction=0.001,
        )
    else:
        config = replace(config, seeds=tuple(range(args.seeds)), warmup_steps=args.warmup_steps)
    if args.device is not None:
        config = replace(config, device=args.device)
    if args.target_head_gain_fraction is not None:
        config = replace(config, target_head_gain_fraction=args.target_head_gain_fraction)
    if args.head_train_per_class is not None:
        config = replace(config, head_train_per_class=args.head_train_per_class)
    if args.tail_train_per_class is not None:
        config = replace(config, tail_train_per_class=args.tail_train_per_class)
    if args.tail_eval_per_class is not None:
        config = replace(config, tail_eval_per_class=args.tail_eval_per_class)
    if args.batch_size is not None:
        config = replace(config, warmup_batch_size=args.batch_size)
    if args.head_batch_size is not None:
        config = replace(config, head_batch_size=args.head_batch_size)
    if args.lr is not None:
        config = replace(config, lr=args.lr)
    if args.download is not None:
        config = replace(config, download=args.download)
    return config


def main() -> None:
    args = parse_args()
    config = config_from_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics, paired, summary, overall = run_layer_jvp_probe(
        config,
        jvp_epsilon=args.jvp_epsilon,
        max_matrix_parameters=args.max_matrix_parameters,
        progress=args.progress,
    )
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    paired.to_csv(args.output_dir / "paired_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    overall.to_csv(args.output_dir / "overall_summary.csv", index=False)
    (args.output_dir / "config.json").write_text(
        json.dumps(
            {
                "base_config": asdict(config),
                "jvp_epsilon": float(args.jvp_epsilon),
                "max_matrix_parameters": args.max_matrix_parameters,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    figure_path = write_figure(summary, paired, args.figure_dir)
    write_discussion(
        config,
        summary,
        overall,
        figure_path,
        args.output_dir,
        args.discussion_path,
        jvp_epsilon=args.jvp_epsilon,
    )
    print(f"saved CIFAR-100-LT ResNet18 all-layer JVP results to {args.output_dir}")
    print(f"metric rows={len(metrics)}, paired rows={len(paired)}, layer summaries={len(summary)}")
    print(f"figure: {figure_path}")
    print(f"discussion: {args.discussion_path}")
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
