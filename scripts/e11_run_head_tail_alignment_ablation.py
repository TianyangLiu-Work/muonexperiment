from __future__ import annotations

import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.head_tail_interference import (
    HeadTailConfig,
    default_head_tail_settings,
    matrix_with_singular_values,
    nuclear_rank,
    sandwiched_stable_rank,
)
from e11_condition_geometry.reporting import fmt, markdown_table
from e11_condition_geometry.statistics import ci95, log_ratio_ci95


OUTPUT_DIR = Path("results/e11_head_tail_alignment_ablation")
FIGURE_DIR = Path("figures/e11_head_tail_alignment_ablation")
DISCUSSION_PATH = Path("discussion/e11_head_tail_alignment_ablation.md")
CONFIG = replace(HeadTailConfig(), seeds=tuple(range(500)))


def _dtype_from_name(name: str) -> torch.dtype:
    dtype = getattr(torch, name)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"unknown torch dtype: {name}")
    return dtype


def _directions(head_gradient: torch.Tensor) -> dict[str, torch.Tensor]:
    u, _, vh = torch.linalg.svd(head_gradient, full_matrices=False)
    return {
        "frobenius": head_gradient / torch.linalg.norm(head_gradient),
        "spectral": u @ vh,
    }


def run_alignment_ablation(config: HeadTailConfig = CONFIG) -> tuple[pd.DataFrame, pd.DataFrame]:
    device = torch.device(config.device)
    dtype = _dtype_from_name(config.dtype)
    tail_output_dim = config.output_dim if config.tail_output_dim is None else int(config.tail_output_dim)
    rows: list[dict[str, object]] = []

    for setting in default_head_tail_settings(config.output_dim, config.input_dim):
        for seed in config.seeds:
            generator = torch.Generator(device=device).manual_seed(int(seed) + 51000)
            head_gradient = matrix_with_singular_values(
                config.output_dim,
                config.input_dim,
                setting.gradient_singular_values,
                generator=generator,
                device=device,
                dtype=dtype,
            )
            tail_inputs = matrix_with_singular_values(
                config.input_dim,
                config.tail_batch_size,
                setting.tail_singular_values,
                generator=generator,
                device=device,
                dtype=dtype,
            )
            downstream = matrix_with_singular_values(
                tail_output_dim,
                config.output_dim,
                setting.downstream_singular_values,
                generator=generator,
                device=device,
                dtype=dtype,
            )
            grad_fro = float(torch.linalg.norm(head_gradient).cpu())
            first_order_gain = config.first_order_gain_per_grad_fro * grad_fro
            drift_by_geometry: dict[str, float] = {}
            for geometry, direction in _directions(head_gradient).items():
                alignment = float(torch.sum(head_gradient * direction).cpu())
                step_size = first_order_gain / max(alignment, 1e-300)
                output_delta = -step_size * (downstream @ direction @ tail_inputs)
                drift_by_geometry[geometry] = float(torch.linalg.norm(output_delta).square().cpu())

            tail_ssrank = sandwiched_stable_rank(downstream, tail_inputs)
            grad_nrank = nuclear_rank(head_gradient)
            theory_ratio = tail_ssrank / max(grad_nrank, 1e-300)
            observed_ratio = drift_by_geometry["spectral"] / max(drift_by_geometry["frobenius"], 1e-300)
            rows.append(
                {
                    "setting": setting.name,
                    "seed": int(seed),
                    "head_gradient_nuclear_rank": grad_nrank,
                    "tail_downstream_aware_stable_rank": tail_ssrank,
                    "theory_ratio_spectral_over_fro": theory_ratio,
                    "predicted_spectral_less_drift": bool(theory_ratio < 1.0),
                    "tail_output_drift_sq_frobenius": drift_by_geometry["frobenius"],
                    "tail_output_drift_sq_spectral": drift_by_geometry["spectral"],
                    "tail_output_drift_sq_ratio_spectral_over_fro": observed_ratio,
                    "spectral_less_tail_output_drift": bool(observed_ratio < 1.0),
                }
            )

    step_metrics = pd.DataFrame(rows)
    summary = summarize_alignment_ablation(step_metrics)
    return step_metrics, summary


def summarize_alignment_ablation(step_metrics: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for setting, group in step_metrics.groupby("setting", observed=True, sort=False):
        ratio = group["tail_output_drift_sq_ratio_spectral_over_fro"]
        geomean, low, high = log_ratio_ci95(ratio)
        theory_mean, theory_low, theory_high = ci95(group["theory_ratio_spectral_over_fro"])
        records.append(
            {
                "setting": setting,
                "seeds": int(group["seed"].nunique()),
                "predicted_spectral_less_drift": bool(group["predicted_spectral_less_drift"].iloc[0]),
                "mean_head_gradient_nuclear_rank": float(group["head_gradient_nuclear_rank"].mean()),
                "mean_tail_downstream_aware_stable_rank": float(group["tail_downstream_aware_stable_rank"].mean()),
                "mean_theory_ratio_spectral_over_fro": theory_mean,
                "theory_ratio_ci95_low": theory_low,
                "theory_ratio_ci95_high": theory_high,
                "geomean_tail_output_drift_sq_ratio_spectral_over_fro": geomean,
                "tail_output_drift_sq_ratio_ci95_low": low,
                "tail_output_drift_sq_ratio_ci95_high": high,
                "median_tail_output_drift_sq_ratio_spectral_over_fro": float(ratio.median()),
                "q05_tail_output_drift_sq_ratio_spectral_over_fro": float(ratio.quantile(0.05)),
                "q95_tail_output_drift_sq_ratio_spectral_over_fro": float(ratio.quantile(0.95)),
                "min_tail_output_drift_sq_ratio_spectral_over_fro": float(ratio.min()),
                "max_tail_output_drift_sq_ratio_spectral_over_fro": float(ratio.max()),
                "spectral_less_tail_output_drift_fraction": float(group["spectral_less_tail_output_drift"].mean()),
            }
        )
    return pd.DataFrame(records)


def write_figure(step_metrics: pd.DataFrame, summary: pd.DataFrame) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    ordered_settings = [setting.name for setting in default_head_tail_settings(CONFIG.output_dim, CONFIG.input_dim)]
    label_map = {
        "high_head_rank_low_tail_srank": "rank condition favors spectral",
        "low_head_rank_high_tail_srank": "rank condition favors Fro",
    }
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    rng = np.random.default_rng(0)
    for index, setting in enumerate(ordered_settings):
        rows = step_metrics[step_metrics["setting"].eq(setting)]
        y = rows["tail_output_drift_sq_ratio_spectral_over_fro"].to_numpy(dtype=float)
        x = index + rng.uniform(-0.16, 0.16, size=len(y))
        ax.scatter(x, y, s=12, alpha=0.24, color="#0072B2" if index == 0 else "#D55E00", linewidths=0)
        row = summary[summary["setting"].eq(setting)].iloc[0]
        ax.errorbar(
            [index],
            [row["geomean_tail_output_drift_sq_ratio_spectral_over_fro"]],
            yerr=[
                [
                    row["geomean_tail_output_drift_sq_ratio_spectral_over_fro"]
                    - row["tail_output_drift_sq_ratio_ci95_low"]
                ],
                [
                    row["tail_output_drift_sq_ratio_ci95_high"]
                    - row["geomean_tail_output_drift_sq_ratio_spectral_over_fro"]
                ],
            ],
            marker="D",
            markersize=7,
            capsize=4,
            color="black",
        )
        ax.hlines(
            row["mean_theory_ratio_spectral_over_fro"],
            index - 0.32,
            index + 0.32,
            colors="black",
            linestyles=":",
            linewidth=2,
        )
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_yscale("log")
    ax.set_xticks(range(len(ordered_settings)))
    ax.set_xticklabels([label_map[setting] for setting in ordered_settings], rotation=10, ha="right")
    ax.set_ylabel("spectral / Fro squared drift ratio")
    ax.set_title("Random singular-vector alignment at fixed spectra")
    fig.tight_layout()
    path = FIGURE_DIR / "head_tail_alignment_ablation.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def write_discussion(summary: pd.DataFrame, figure_path: Path) -> None:
    positive = summary[summary["setting"].eq("high_head_rank_low_tail_srank")].iloc[0]
    table_columns = [
        "setting",
        "seeds",
        "predicted_spectral_less_drift",
        "mean_theory_ratio_spectral_over_fro",
        "geomean_tail_output_drift_sq_ratio_spectral_over_fro",
        "tail_output_drift_sq_ratio_ci95_low",
        "tail_output_drift_sq_ratio_ci95_high",
        "median_tail_output_drift_sq_ratio_spectral_over_fro",
        "q05_tail_output_drift_sq_ratio_spectral_over_fro",
        "q95_tail_output_drift_sq_ratio_spectral_over_fro",
        "spectral_less_tail_output_drift_fraction",
    ]
    text = f"""# E11 Head-to-Tail Alignment Ablation

This ablation keeps the singular values from the synthetic head-to-tail
boundary diagnostic, but randomizes the singular-vector alignment of the head
gradient, tail activation, and downstream tail map. The purpose is to separate
the worst-case rank condition from the realized drift of a particular polar
direction.

![Alignment ablation](../{figure_path.as_posix()})

## Summary

{markdown_table(summary, table_columns)}

## Readout

In the positive-spectrum setting, the worst-case bound ratio still favors
spectral geometry: the mean theory ratio is
{fmt(positive["mean_theory_ratio_spectral_over_fro"])}. However, after random
singular-vector alignment, the observed squared drift ratio is mixed: the
geometric mean is
{fmt(positive["geomean_tail_output_drift_sq_ratio_spectral_over_fro"])}
[{fmt(positive["tail_output_drift_sq_ratio_ci95_low"])},
{fmt(positive["tail_output_drift_sq_ratio_ci95_high"])}], the median is
{fmt(positive["median_tail_output_drift_sq_ratio_spectral_over_fro"])}, and
spectral has lower drift in only
{fmt(positive["spectral_less_tail_output_drift_fraction"])} of seeds.

This does not contradict the theorem. It shows that
`nrank(G_H) > ssrank(B_T,A_T)` is a bound-ordering condition, while realized
drift also depends on the singular-vector alignment between the polar direction
and the tail perturbation operator.

## Artifacts

- [step_metrics.csv](../results/e11_head_tail_alignment_ablation/step_metrics.csv)
- [summary.csv](../results/e11_head_tail_alignment_ablation/summary.csv)
- [config.json](../results/e11_head_tail_alignment_ablation/config.json)
"""
    DISCUSSION_PATH.write_text(text, encoding="utf-8")


def write_outputs(step_metrics: pd.DataFrame, summary: pd.DataFrame, config: HeadTailConfig = CONFIG) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    step_metrics.to_csv(OUTPUT_DIR / "step_metrics.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)
    (OUTPUT_DIR / "config.json").write_text(json.dumps(config.__dict__, indent=2), encoding="utf-8")
    figure_path = write_figure(step_metrics, summary)
    write_discussion(summary, figure_path)


def main() -> None:
    step_metrics, summary = run_alignment_ablation(CONFIG)
    write_outputs(step_metrics, summary, CONFIG)
    print(f"saved head-to-tail alignment ablation to {OUTPUT_DIR}")
    print(f"step rows={len(step_metrics)}, settings={summary['setting'].nunique()}, seeds={len(CONFIG.seeds)}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
