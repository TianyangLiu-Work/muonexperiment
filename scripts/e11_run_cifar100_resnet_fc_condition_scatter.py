from __future__ import annotations

import argparse
import gc
import json
import math
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
    resolve_device,
    split_long_tail_cifar100,
    train_checkpoint,
)
from e11_condition_geometry.diagnostics import matrix_effective_rank, singular_values, stable_rank
from e11_condition_geometry.long_tail_digits import make_generator, margins
from e11_condition_geometry.long_tail_one_step import summarize_long_tail_one_step
from e11_condition_geometry.reporting import fmt, markdown_table


DEFAULT_OUTPUT_DIR = Path("results/e11_cifar100_resnet_fc_condition_scatter")
DEFAULT_FIGURE_DIR = Path("figures/e11_cifar100_resnet_fc_condition_scatter")
DEFAULT_DISCUSSION_PATH = Path("discussion/e11_cifar100_resnet_fc_condition_scatter.md")
DEFAULT_WARMUP_STEPS = (250, 500, 1000, 2000)


def parse_warmup_steps(value: str) -> tuple[int, ...]:
    steps = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not steps:
        raise argparse.ArgumentTypeError("expected at least one warmup step")
    if any(step <= 0 for step in steps):
        raise argparse.ArgumentTypeError("warmup steps must be positive")
    return steps


def final_features(model, x: torch.Tensor) -> torch.Tensor:
    z = model.conv1(x)
    z = model.bn1(z)
    z = model.relu(z)
    z = model.maxpool(z)
    z = model.layer1(z)
    z = model.layer2(z)
    z = model.layer3(z)
    z = model.layer4(z)
    z = model.avgpool(z)
    return torch.flatten(z, 1)


def fc_directions(fc_grad: torch.Tensor, geometry: str) -> torch.Tensor:
    if geometry == "frobenius":
        return fc_grad / max(float(torch.linalg.norm(fc_grad).detach().cpu()), 1e-300)
    if geometry == "spectral":
        u, _s, vh = torch.linalg.svd(fc_grad, full_matrices=False)
        return u @ vh
    raise ValueError(f"unknown geometry: {geometry}")


def fc_update_norms(direction: torch.Tensor, step_size: float) -> tuple[float, float]:
    update = float(step_size) * direction
    return (
        float(torch.linalg.norm(update).detach().cpu()),
        float(torch.linalg.matrix_norm(update, ord=2).detach().cpu()),
    )


def apply_fc_direction(model, before: torch.Tensor, direction: torch.Tensor, step_size: float) -> None:
    with torch.no_grad():
        model.fc.weight.copy_(before - float(step_size) * direction)


def run_one_setting(
    base_config: Cifar100ResNetOneStepConfig,
    *,
    warmup_steps: int,
    progress: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = replace(base_config, warmup_steps=warmup_steps)
    device = resolve_device(config.device)
    dtype = dtype_from_name(config.dtype)
    rows: list[dict] = []
    condition_rows: list[dict] = []
    for seed in config.seeds:
        if progress:
            print(f"[resnet-fc-condition] warmup={warmup_steps} seed={seed}: loading data", flush=True)
        train_x, train_y, test_x, test_y, train_indices, head_train, _tail_train, tail_eval = split_long_tail_cifar100(
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
            print(f"[resnet-fc-condition] warmup={warmup_steps} seed={seed}: training", flush=True)
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
        fc_grad = model.fc.weight.grad.detach().clone()
        fc_before = model.fc.weight.detach().clone()
        target_gain = config.target_head_gain_fraction * float(head_loss.detach().cpu())
        fc_gradient_nuclear_rank = matrix_effective_rank(singular_values(fc_grad))
        with torch.no_grad():
            tail_features = final_features(model, tail_x).detach()
            tail_feature_stable_rank = stable_rank(singular_values(tail_features))
            base_tail_logits = model(tail_x).detach()
            base_tail_pred = torch.argmax(base_tail_logits, dim=1)
        base_tail_margins = margins(base_tail_logits, tail_y)
        positive_margin_mask = base_tail_margins > 0
        positive_margin_count = int(positive_margin_mask.sum().detach().cpu())
        positive_margin_fraction = float(positive_margin_mask.to(torch.float64).mean().detach().cpu())
        base_head = full_metrics(model, train_x, train_y, head_batch)
        base_tail = full_metrics(model, test_x, test_y, tail_eval)
        condition_score = fc_gradient_nuclear_rank / max(tail_feature_stable_rank, 1e-300)
        theory_ratio = tail_feature_stable_rank / max(fc_gradient_nuclear_rank, 1e-300)

        condition_rows.append(
            {
                "warmup_steps": int(warmup_steps),
                "seed": int(seed),
                "parameter": "fc.weight",
                "fc_gradient_nuclear_rank": fc_gradient_nuclear_rank,
                "tail_feature_stable_rank": tail_feature_stable_rank,
                "condition_score_nrank_over_tail_srank": condition_score,
                "theory_ratio_tail_srank_over_nrank": theory_ratio,
                "tail_accuracy_before": base_tail["accuracy"],
                "tail_positive_margin_fraction_before": positive_margin_fraction,
                "head_loss_before": float(head_loss.detach().cpu()),
            }
        )

        for geometry in ["frobenius", "spectral"]:
            direction = fc_directions(fc_grad, geometry)
            alignment = float(torch.sum(fc_grad * direction).detach().cpu())
            step_size = target_gain / max(alignment, 1e-300)
            update_fro, update_op = fc_update_norms(direction, step_size)
            apply_fc_direction(model, fc_before, direction, step_size)
            with torch.no_grad():
                after_tail_logits = model(tail_x).detach()
                after_tail_pred = torch.argmax(after_tail_logits, dim=1)
            after_head = full_metrics(model, train_x, train_y, head_batch)
            after_tail = full_metrics(model, test_x, test_y, tail_eval)
            output_delta = after_tail_logits - base_tail_logits
            centered_output_delta = output_delta - output_delta.mean(dim=1, keepdim=True)
            sample_indices = torch.arange(output_delta.shape[0], device=output_delta.device)
            true_logit_delta = output_delta[sample_indices, tail_y]
            base_competitor_logits = base_tail_logits.clone()
            base_competitor_logits[sample_indices, tail_y] = -torch.inf
            base_competitor_indices = torch.argmax(base_competitor_logits, dim=1)
            competitor_logit_delta = output_delta[sample_indices, base_competitor_indices]
            margin_delta = margins(after_tail_logits, tail_y) - base_tail_margins
            per_sample_delta_inf = torch.max(torch.abs(output_delta), dim=1).values
            actual_head_loss_decrease = base_head["loss"] - after_head["loss"]
            actual_head_gain_relative_error = abs(actual_head_loss_decrease - target_gain) / max(target_gain, 1e-300)
            if positive_margin_count > 0:
                certified_positive = per_sample_delta_inf[positive_margin_mask] < 0.5 * base_tail_margins[
                    positive_margin_mask
                ]
                certified_preserved_fraction = float(certified_positive.to(torch.float64).mean().detach().cpu())
                positive_changed = after_tail_pred[positive_margin_mask] != base_tail_pred[positive_margin_mask]
                positive_prediction_changed_fraction = float(positive_changed.to(torch.float64).mean().detach().cpu())
            else:
                certified_preserved_fraction = float("nan")
                positive_prediction_changed_fraction = float("nan")
            rows.append(
                {
                    "seed": int(seed),
                    "geometry": geometry,
                    "dataset": "CIFAR100",
                    "model": "resnet18_cifar_stem",
                    "updated_parameter_subset": "final_linear_weight_only",
                    "warmup_steps": int(warmup_steps),
                    "head_loss_before": base_head["loss"],
                    "head_loss_after": after_head["loss"],
                    "actual_head_loss_decrease": actual_head_loss_decrease,
                    "actual_head_gain_relative_error": actual_head_gain_relative_error,
                    "matched_first_order_head_gain": target_gain,
                    "tail_loss_before": base_tail["loss"],
                    "tail_loss_after": after_tail["loss"],
                    "tail_loss_increase": after_tail["loss"] - base_tail["loss"],
                    "tail_accuracy_before": base_tail["accuracy"],
                    "tail_accuracy_after": after_tail["accuracy"],
                    "tail_accuracy_drop": base_tail["accuracy"] - after_tail["accuracy"],
                    "tail_margin_before": base_tail["mean_margin"],
                    "tail_margin_after": after_tail["mean_margin"],
                    "tail_margin_drop": base_tail["mean_margin"] - after_tail["mean_margin"],
                    "tail_positive_margin_fraction_before": positive_margin_fraction,
                    "tail_margin_certified_preserved_fraction": certified_preserved_fraction,
                    "tail_prediction_changed_fraction": float(
                        (after_tail_pred != base_tail_pred).to(torch.float64).mean().detach().cpu()
                    ),
                    "tail_positive_margin_prediction_changed_fraction": positive_prediction_changed_fraction,
                    "tail_output_drift_fro": float(torch.linalg.norm(output_delta).detach().cpu()),
                    "tail_output_drift_rms": float(torch.sqrt(torch.mean(output_delta.square())).detach().cpu()),
                    "centered_tail_output_drift_fro": float(torch.linalg.norm(centered_output_delta).detach().cpu()),
                    "true_logit_delta_fro": float(torch.linalg.norm(true_logit_delta).detach().cpu()),
                    "competitor_logit_delta_fro": float(torch.linalg.norm(competitor_logit_delta).detach().cpu()),
                    "margin_delta_fro": float(torch.linalg.norm(margin_delta).detach().cpu()),
                    "margin_delta_rms": float(torch.sqrt(torch.mean(margin_delta.square())).detach().cpu()),
                    "tail_output_jvp_fro": float("nan"),
                    "tail_output_linearization_residual_fro": float("nan"),
                    "tail_output_linearization_relative_error": float("nan"),
                    "update_fro_norm": update_fro,
                    "update_op_norm": update_op,
                    "step_size": float(step_size),
                    "alignment": alignment,
                    "nrG": fc_gradient_nuclear_rank,
                    "stA_tail": tail_feature_stable_rank,
                    "condition_score_tail": condition_score,
                    "mean_matrix_update_nuclear_rank": matrix_effective_rank(singular_values(direction)),
                    "device": str(device),
                    "dtype": str(dtype).replace("torch.", ""),
                }
            )
            apply_fc_direction(model, fc_before, torch.zeros_like(fc_before), 0.0)
        if progress:
            print(f"[resnet-fc-condition] warmup={warmup_steps} seed={seed}: done", flush=True)
        del model, train_x, train_y, test_x, test_y
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return pd.DataFrame(rows), pd.DataFrame(condition_rows)


def summarize_by_warmup(step_metrics: pd.DataFrame, condition_metrics: pd.DataFrame) -> pd.DataFrame:
    summaries = []
    for warmup_steps, group in step_metrics.groupby("warmup_steps", sort=True):
        summary = summarize_long_tail_one_step(group.drop(columns=["warmup_steps"]))
        summary.insert(0, "warmup_steps", int(warmup_steps))
        condition_group = condition_metrics[condition_metrics["warmup_steps"].eq(warmup_steps)]
        summary["mean_condition_score_nrank_over_tail_srank"] = condition_group[
            "condition_score_nrank_over_tail_srank"
        ].mean()
        summary["mean_theory_ratio_tail_srank_over_nrank"] = condition_group[
            "theory_ratio_tail_srank_over_nrank"
        ].mean()
        summary["condition_favors_spectral_fraction"] = (
            condition_group["condition_score_nrank_over_tail_srank"] > 1.0
        ).mean()
        summaries.append(summary)
    return pd.concat(summaries, ignore_index=True)


def write_figure(summary: pd.DataFrame, figure_dir: Path) -> Path:
    figure_dir.mkdir(parents=True, exist_ok=True)
    ordered = summary.sort_values("warmup_steps")
    x = ordered["warmup_steps"].to_numpy(dtype=float)
    drift = ordered["geomean_tail_output_drift_sq_ratio_spectral_over_fro"].to_numpy(dtype=float)
    drift_low = ordered["tail_output_drift_sq_ratio_ci95_low"].to_numpy(dtype=float)
    drift_high = ordered["tail_output_drift_sq_ratio_ci95_high"].to_numpy(dtype=float)
    theory = ordered["mean_theory_ratio_tail_srank_over_nrank"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.errorbar(
        x,
        drift,
        yerr=[drift - drift_low, drift_high - drift],
        marker="o",
        capsize=4,
        linewidth=2,
        color="#0072B2",
        label="observed final-layer drift ratio",
    )
    ax.plot(x, theory, marker="s", linewidth=2, color="#D55E00", label="tail srank / nrank")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("warmup checkpoint steps")
    ax.set_ylabel("ratio")
    ax.set_yscale("log")
    ax.set_title("ResNet18 final-layer condition diagnostic")
    ax.legend(frameon=False)
    fig.tight_layout()
    path = figure_dir / "cifar100_resnet_fc_condition_scatter.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def write_discussion(
    summary: pd.DataFrame,
    condition_metrics: pd.DataFrame,
    figure_path: Path,
    discussion_path: Path,
) -> None:
    worst = summary.loc[summary["tail_output_drift_sq_ratio_ci95_high"].idxmax()]
    weakest_condition = summary.loc[summary["mean_condition_score_nrank_over_tail_srank"].idxmin()]
    text = f"""# E11 CIFAR-100-LT ResNet18 Final-Layer Condition Scatter

This generated diagnostic is a downstream-aware ResNet condition check for the
final linear layer. For `fc.weight`, the local tail map is the tail feature
matrix feeding the classifier, so the measured proxy is
`nrank(G_H) / srank(H_T)`. The intervention updates only `fc.weight`, matches
the same first-order head gain for Frobenius/GD and spectral/polar directions,
and measures held-out tail-example logit drift.

![CIFAR-100-LT ResNet18 final-layer condition](../{figure_path.as_posix()})

## Summary

{markdown_table(summary, [
    "warmup_steps",
    "seeds",
    "mean_condition_score_nrank_over_tail_srank",
    "mean_theory_ratio_tail_srank_over_nrank",
    "condition_favors_spectral_fraction",
    "geomean_tail_output_drift_sq_ratio_spectral_over_fro",
    "tail_output_drift_sq_ratio_ci95_low",
    "tail_output_drift_sq_ratio_ci95_high",
    "mean_tail_accuracy_before",
])}

## Readout

- Worst final-layer-only drift-ratio CI upper endpoint:
  {fmt(worst["tail_output_drift_sq_ratio_ci95_high"])} at
  {int(worst["warmup_steps"])} warmup steps.
- Weakest mean condition score:
  {fmt(weakest_condition["mean_condition_score_nrank_over_tail_srank"])} at
  {int(weakest_condition["warmup_steps"])} warmup steps.
- Condition-favors-spectral fraction across all seed/checkpoint points:
  {fmt((condition_metrics["condition_score_nrank_over_tail_srank"] > 1.0).mean())}.

## Interpretation

This is closer to the theorem than the rank-only scatter because it includes a
tail-side sensitivity term for the final layer. It is still not a full
all-layer ResNet proof: earlier convolution blocks have nonlinear downstream
maps and need a separate layerwise JVP or tail-sensitivity approximation.

## Artifacts

- [step_metrics.csv](../results/e11_cifar100_resnet_fc_condition_scatter/step_metrics.csv)
- [summary.csv](../results/e11_cifar100_resnet_fc_condition_scatter/summary.csv)
- [condition_metrics.csv](../results/e11_cifar100_resnet_fc_condition_scatter/condition_metrics.csv)
- [config.json](../results/e11_cifar100_resnet_fc_condition_scatter/config.json)
"""
    discussion_path.parent.mkdir(parents=True, exist_ok=True)
    discussion_path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--warmup-steps", type=parse_warmup_steps, default=DEFAULT_WARMUP_STEPS)
    parser.add_argument("--target-head-gain-fraction", type=float, default=0.005)
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
            warmup_batch_size=16,
            head_batch_size=16,
            target_head_gain_fraction=0.001,
        )
    else:
        config = replace(config, seeds=tuple(range(args.seeds)))
    if args.device is not None:
        config = replace(config, device=args.device)
    if args.target_head_gain_fraction is not None:
        config = replace(config, target_head_gain_fraction=args.target_head_gain_fraction)
    if args.download is not None:
        config = replace(config, download=args.download)
    return config


def main() -> None:
    args = parse_args()
    config = config_from_args(args)
    warmup_steps = (2,) if args.smoke else args.warmup_steps
    step_frames = []
    condition_frames = []
    for warmup_step in warmup_steps:
        step_metrics, condition_metrics = run_one_setting(config, warmup_steps=warmup_step, progress=args.progress)
        step_frames.append(step_metrics)
        condition_frames.append(condition_metrics)
    step_metrics = pd.concat(step_frames, ignore_index=True)
    condition_metrics = pd.concat(condition_frames, ignore_index=True)
    summary = summarize_by_warmup(step_metrics, condition_metrics)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    step_metrics.to_csv(args.output_dir / "step_metrics.csv", index=False)
    condition_metrics.to_csv(args.output_dir / "condition_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    (args.output_dir / "config.json").write_text(
        json.dumps({"base_config": asdict(config), "warmup_steps": list(warmup_steps)}, indent=2) + "\n",
        encoding="utf-8",
    )
    figure_path = write_figure(summary, args.figure_dir)
    write_discussion(summary, condition_metrics, figure_path, args.discussion_path)
    print(f"saved final-layer condition scatter to {args.output_dir}")
    print(f"step rows={len(step_metrics)}, condition rows={len(condition_metrics)}, settings={len(summary)}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
