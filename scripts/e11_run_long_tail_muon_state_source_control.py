from __future__ import annotations

from dataclasses import asdict
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.long_tail_digits import (
    TinyMLP,
    add_noise,
    batch,
    dtype_from_name,
    full_metrics,
    make_generator,
    restore,
    snapshot,
    split_long_tail_digits,
    train_digits_checkpoint,
)
from e11_condition_geometry.long_tail_muon_bridge import (
    LongTailPracticalMuonBridgeConfig,
    apply_practical_muon_step,
    assign_grads,
    collect_gradients,
    direction_cosine,
    evaluate_bridge_direction,
    frobenius_grad_direction,
    momentum_update,
    newton_schulz_directions,
    polar_directions,
    run_long_tail_practical_muon_bridge,
    summarize_long_tail_practical_muon_bridge,
    tensor_fro_norm,
)
from e11_condition_geometry.reporting import fmt, markdown_table


OUTPUT_DIR = Path("results/e11_long_tail_muon_state_source_control")
FIGURE_DIR = Path("figures/e11_long_tail_muon_state_source_control")
DISCUSSION_PATH = Path("discussion/e11_long_tail_muon_state_source_control.md")

STATE_SOURCE_NAMES = {
    "muon_ns_trajectory": "NS-Muon-style trajectory",
    "fro_gd_trajectory": "Fro/GD-style trajectory",
}
DISPLAY_NAMES = {
    "polar_grad": r"polar($G_t$)",
    "polar_momentum": r"polar($M_t$)",
    "ns_momentum": r"NS($M_t$)",
}


def _fmt(value: float) -> str:
    return f"{value:.4g}"


def _collect_loss_and_grads(model: TinyMLP, x_batch: torch.Tensor, y_batch: torch.Tensor) -> tuple[float, list[torch.Tensor]]:
    params = model.parameters()
    for param in params:
        param.grad = None
    loss = F.cross_entropy(model.logits(x_batch), y_batch)
    loss.backward()
    return float(loss.detach().cpu()), [param.grad.detach().clone() for param in params]


def _run_fro_trajectory_source(config: LongTailPracticalMuonBridgeConfig) -> pd.DataFrame:
    device = torch.device(config.device)
    dtype = dtype_from_name(config.dtype)
    rows: list[dict] = []

    for seed in config.seeds:
        x, y, train_indices, head_train, _tail_train, tail_eval = split_long_tail_digits(
            config,
            seed=seed,
            device=device,
            dtype=dtype,
        )
        generator = make_generator(seed + 13000, device)
        model = TinyMLP(
            input_dim=x.shape[1],
            hidden_dim=config.hidden_dim,
            output_dim=10,
            generator=generator,
            device=device,
            dtype=dtype,
        )
        train_digits_checkpoint(model, x, y, train_indices, config, seed=seed, seed_offset=12000)
        params = model.parameters()
        momentum: list[torch.Tensor] = []

        for trajectory_step in range(config.trajectory_steps):
            head_generator = make_generator(seed + 17000 + trajectory_step, device)
            head_batch = batch(head_train, config.head_batch_size, generator=head_generator)
            head_x = add_noise(x[head_batch], config.train_noise_std, generator=head_generator)
            head_loss_value, current_grads = _collect_loss_and_grads(model, head_x, y[head_batch])
            momentum = momentum_update(momentum, current_grads, beta=config.momentum_beta)
            assign_grads(params, current_grads)

            before = snapshot(params)
            base_tail_logits = model.logits(x[tail_eval]).detach()
            base_head = full_metrics(model, x, y, head_batch)
            base_tail = full_metrics(model, x, y, tail_eval)
            target_gain = config.target_head_gain_fraction * head_loss_value

            directions = {
                "frobenius_grad": frobenius_grad_direction(current_grads),
                "polar_grad": polar_directions(current_grads),
                "polar_momentum": polar_directions(momentum),
                "ns_momentum": newton_schulz_directions(momentum, steps=config.newton_schulz_steps),
            }
            polar_grad = directions["polar_grad"]
            grad_norm = tensor_fro_norm(current_grads)
            momentum_norm = tensor_fro_norm(momentum)
            grad_momentum_cosine = direction_cosine(current_grads, momentum)

            for direction_name, direction in directions.items():
                metrics = evaluate_bridge_direction(
                    model,
                    x,
                    y,
                    head_batch,
                    tail_eval,
                    before,
                    base_head,
                    base_tail,
                    base_tail_logits,
                    direction,
                    target_gain=target_gain,
                )
                rows.append(
                    {
                        "state_source": "fro_gd_trajectory",
                        "seed": int(seed),
                        "trajectory_step": int(trajectory_step),
                        "direction": direction_name,
                        "head_classes": ",".join(str(label) for label in config.head_classes),
                        "tail_classes": ",".join(str(label) for label in config.tail_classes),
                        "warmup_steps": int(config.warmup_steps),
                        "hidden_dim": int(config.hidden_dim),
                        "momentum_beta": float(config.momentum_beta),
                        "trajectory_steps": int(config.trajectory_steps),
                        "trajectory_lr": float(config.trajectory_lr),
                        "newton_schulz_steps": int(config.newton_schulz_steps),
                        "head_gradient_fro_norm": grad_norm,
                        "momentum_fro_norm": momentum_norm,
                        "gradient_momentum_cosine": grad_momentum_cosine,
                        "direction_cosine_to_polar_grad": direction_cosine(direction, polar_grad),
                        **metrics,
                    }
                )

            restore(params, before)
            apply_practical_muon_step(params, directions["frobenius_grad"], lr=config.trajectory_lr)

    return pd.DataFrame(rows)


def run_state_source_control(config: LongTailPracticalMuonBridgeConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    muon_steps, _muon_summary = run_long_tail_practical_muon_bridge(config)
    muon_steps = muon_steps.copy()
    muon_steps["state_source"] = "muon_ns_trajectory"
    fro_steps = _run_fro_trajectory_source(config)
    step_metrics = pd.concat([muon_steps, fro_steps], ignore_index=True)

    summaries = []
    for state_source, group in step_metrics.groupby("state_source", observed=True, sort=False):
        summary = summarize_long_tail_practical_muon_bridge(group.drop(columns=["state_source"]))
        summary.insert(0, "state_source", state_source)
        summaries.append(summary)
    return step_metrics, pd.concat(summaries, ignore_index=True)


def write_figure(summary: pd.DataFrame) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    rows = summary[summary["direction"].isin(["polar_momentum", "ns_momentum"])].copy()
    rows["display_source"] = rows["state_source"].map(STATE_SOURCE_NAMES)
    rows["display_direction"] = rows["direction"].map(DISPLAY_NAMES)
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    colors = {"polar_momentum": "#009E73", "ns_momentum": "#CC79A7"}
    offsets = {"polar_momentum": -0.08, "ns_momentum": 0.08}
    sources = ["muon_ns_trajectory", "fro_gd_trajectory"]
    for direction in ["polar_momentum", "ns_momentum"]:
        sub = rows[rows["direction"].eq(direction)].set_index("state_source").loc[sources].reset_index()
        x = [index + offsets[direction] for index in range(len(sub))]
        y = sub["geomean_tail_output_drift_sq_ratio_vs_fro"].to_numpy(dtype=float)
        low = sub["tail_output_drift_sq_ratio_vs_fro_ci95_low"].to_numpy(dtype=float)
        high = sub["tail_output_drift_sq_ratio_vs_fro_ci95_high"].to_numpy(dtype=float)
        ax.errorbar(
            x,
            y,
            yerr=[y - low, high - y],
            marker="o",
            linewidth=2,
            capsize=4,
            color=colors[direction],
            label=DISPLAY_NAMES[direction],
        )
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(range(len(sources)))
    ax.set_xticklabels([STATE_SOURCE_NAMES[source] for source in sources], rotation=12, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("squared tail-example logit drift ratio vs Fro/GD")
    ax.set_title("Muon-style direction compatibility by trajectory state source")
    ax.legend(frameon=False)
    fig.tight_layout()
    path = FIGURE_DIR / "long_tail_muon_state_source_control.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def write_discussion(config: LongTailPracticalMuonBridgeConfig, summary: pd.DataFrame, figure_path: Path) -> None:
    table = summary[summary["direction"].isin(["polar_momentum", "ns_momentum"])].copy()
    table["state_source_label"] = table["state_source"].map(STATE_SOURCE_NAMES)
    table["direction_label"] = table["direction"].map(DISPLAY_NAMES)
    columns = [
        "state_source_label",
        "direction_label",
        "comparisons",
        "geomean_tail_output_drift_sq_ratio_vs_fro",
        "tail_output_drift_sq_ratio_vs_fro_ci95_low",
        "tail_output_drift_sq_ratio_vs_fro_ci95_high",
        "less_tail_drift_than_fro_fraction",
        "mean_gradient_momentum_cosine",
    ]
    muon_ns = summary.set_index(["state_source", "direction"]).loc[("muon_ns_trajectory", "ns_momentum")]
    fro_ns = summary.set_index(["state_source", "direction"]).loc[("fro_gd_trajectory", "ns_momentum")]
    text = f"""# E11 Long-Tailed Muon State-Source Control

This control tests a state-selection concern in the practical Muon-style
trajectory diagnostic. The original trajectory states are generated by
finite-Newton-Schulz Muon-style head-only updates. Here, the same
matched-head-gain diagnostic is also evaluated along a Fro/GD-style trajectory
with the same seeds, batches, trajectory length, and nominal trajectory
learning rate.

![State-source control](../{figure_path.as_posix()})

## Summary

{markdown_table(table, columns)}

## Readout

For NS($M_t$), the original NS-Muon-style trajectory gives squared drift ratio
{fmt(muon_ns["geomean_tail_output_drift_sq_ratio_vs_fro"])}
[{fmt(muon_ns["tail_output_drift_sq_ratio_vs_fro_ci95_low"])},
{fmt(muon_ns["tail_output_drift_sq_ratio_vs_fro_ci95_high"])}]. On the Fro/GD
trajectory states, the same direction family gives squared drift ratio
{fmt(fro_ns["geomean_tail_output_drift_sq_ratio_vs_fro"])}
[{fmt(fro_ns["tail_output_drift_sq_ratio_vs_fro_ci95_low"])},
{fmt(fro_ns["tail_output_drift_sq_ratio_vs_fro_ci95_high"])}].

This control reduces but does not eliminate state-selection risk: it checks one
alternative trajectory source, not every checkpoint distribution or a full
optimizer benchmark.

## Artifacts

- [step_metrics.csv](../results/e11_long_tail_muon_state_source_control/step_metrics.csv)
- [summary.csv](../results/e11_long_tail_muon_state_source_control/summary.csv)
- [config.json](../results/e11_long_tail_muon_state_source_control/config.json)
"""
    DISCUSSION_PATH.write_text(text, encoding="utf-8")


def write_outputs(step_metrics: pd.DataFrame, summary: pd.DataFrame, config: LongTailPracticalMuonBridgeConfig) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    step_metrics.to_csv(OUTPUT_DIR / "step_metrics.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)
    (OUTPUT_DIR / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    figure_path = write_figure(summary)
    write_discussion(config, summary, figure_path)


def main() -> None:
    config = LongTailPracticalMuonBridgeConfig()
    step_metrics, summary = run_state_source_control(config)
    write_outputs(step_metrics, summary, config)
    print(f"saved long-tail Muon state-source control to {OUTPUT_DIR}")
    print(f"step rows={len(step_metrics)}, rows={len(summary)}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
