from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.long_tail_muon_bridge import LongTailMuonBridgeConfig, run_long_tail_muon_bridge


OUTPUT_DIR = Path("results/e11_long_tail_muon_bridge")
FIGURE_DIR = Path("figures/e11_long_tail_muon_bridge")
DISCUSSION_PATH = Path("discussion/e11_long_tail_muon_bridge.md")


DISPLAY_NAMES = {
    "polar_grad": r"polar($G_t$)",
    "polar_momentum": r"polar($M_t$)",
    "ns_momentum": r"NS($M_t$)",
}


def _fmt(value: float) -> str:
    return f"{value:.4g}"


def write_figure(pair_summary: pd.DataFrame) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    order = ["polar_grad", "polar_momentum", "ns_momentum"]
    rows = pair_summary.set_index("direction").loc[order].reset_index()
    x = range(len(rows))
    y = rows["geomean_tail_output_drift_sq_ratio_vs_fro"].to_numpy(dtype=float)
    y_low = rows["tail_output_drift_sq_ratio_vs_fro_ci95_low"].to_numpy(dtype=float)
    y_high = rows["tail_output_drift_sq_ratio_vs_fro_ci95_high"].to_numpy(dtype=float)
    xerr = [y - y_low, y_high - y]

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
    colors = ["#0072B2", "#009E73", "#CC79A7"]
    axes[0].bar(x, y, color=colors, alpha=0.72)
    axes[0].errorbar(x, y, yerr=xerr, fmt="none", color="#222222", capsize=4, linewidth=1.2)
    axes[0].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels([DISPLAY_NAMES[name] for name in order], rotation=15, ha="right")
    axes[0].set_ylabel("squared tail-example logit drift ratio vs Fro/GD")
    axes[0].set_title("Matched-head-gain squared tail drift")

    axes[1].bar(
        x,
        rows["mean_direction_cosine_to_polar_grad"].to_numpy(dtype=float),
        color=colors,
        alpha=0.72,
    )
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels([DISPLAY_NAMES[name] for name in order], rotation=15, ha="right")
    axes[1].set_ylabel(r"cosine with polar($G_t$)")
    axes[1].set_title("Direction compatibility")

    fig.suptitle("Long-tailed digits Muon-style compatibility diagnostic")
    fig.tight_layout()
    path = FIGURE_DIR / "long_tail_muon_bridge.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def write_discussion(config: LongTailMuonBridgeConfig, pair_summary: pd.DataFrame, figure_path: Path) -> None:
    summary = pair_summary.set_index("direction")

    def row(name: str) -> pd.Series:
        return summary.loc[name]

    table_lines = [
        "| direction | squared drift ratio vs Fro/GD | squared drift ratio vs polar(G_t) | cosine to polar(G_t) | alignment ratio to polar(G_t) |",
        "|---|---:|---:|---:|---:|",
    ]
    for direction in ["polar_grad", "polar_momentum", "ns_momentum"]:
        item = row(direction)
        table_lines.append(
            "| "
            + DISPLAY_NAMES[direction]
            + " | "
            + f"{_fmt(item['geomean_tail_output_drift_sq_ratio_vs_fro'])} "
            + f"[{_fmt(item['tail_output_drift_sq_ratio_vs_fro_ci95_low'])}, "
            + f"{_fmt(item['tail_output_drift_sq_ratio_vs_fro_ci95_high'])}]"
            + " | "
            + f"{_fmt(item['geomean_tail_output_drift_sq_ratio_vs_polar_grad'])} "
            + f"[{_fmt(item['tail_output_drift_sq_ratio_vs_polar_grad_ci95_low'])}, "
            + f"{_fmt(item['tail_output_drift_sq_ratio_vs_polar_grad_ci95_high'])}]"
            + " | "
            + f"{_fmt(item['mean_direction_cosine_to_polar_grad'])} "
            + f"[{_fmt(item['direction_cosine_to_polar_grad_ci95_low'])}, "
            + f"{_fmt(item['direction_cosine_to_polar_grad_ci95_high'])}]"
            + " | "
            + f"{_fmt(item['geomean_alignment_ratio_to_polar_grad'])} "
            + f"[{_fmt(item['alignment_ratio_to_polar_grad_ci95_low'])}, "
            + f"{_fmt(item['alignment_ratio_to_polar_grad_ci95_high'])}]"
            + " |"
        )

    polar_momentum = row("polar_momentum")
    ns_momentum = row("ns_momentum")
    text = f"""# E11 Long-Tailed Muon-Style Compatibility Diagnostic

This diagnostic addresses the gap between the paper's clean direction
`polar(G_t)` and a practical Muon-like update that uses a momentum state and an
approximate polar factor. It reuses the long-tailed digits checkpoint and
matched-head-gain protocol, then compares four directions at the same parameter
state:

1. Frobenius-normalized head gradient, used as the GD-style baseline.
2. Exact `polar(G_t)`, the ideal direction analyzed in the paper.
3. Exact `polar(M_t)`, where `M_t` is an exponential moving average of recent
   head-batch gradients plus the current head gradient.
4. Newton-Schulz `NS(M_t)`, a finite-iteration approximation to `polar(M_t)`.

Settings:

- Seeds: {len(config.seeds)}
- Momentum beta: {config.momentum_beta}
- Momentum history steps: {config.momentum_history_steps}
- Newton-Schulz steps: {config.newton_schulz_steps}
- Warmup steps: {config.warmup_steps}
- Hidden dimension: {config.hidden_dim}
- Target head first-order gain: {config.target_head_gain_fraction} * head loss

## Summary

{chr(10).join(table_lines)}

Mean gradient-momentum cosine across the paired seeds is
`{_fmt(polar_momentum['mean_gradient_momentum_cosine'])}`
`[{_fmt(polar_momentum['gradient_momentum_cosine_ci95_low'])}, {_fmt(polar_momentum['gradient_momentum_cosine_ci95_high'])}]`.

## Interpretation

The ideal `polar(G_t)` row is the paper's current clean mechanism target. The
`polar(M_t)` row asks whether replacing the current gradient by a Muon-style
momentum state gives a compatible local drift signal. The `NS(M_t)` row asks
whether a finite Newton-Schulz approximation still stays close enough to the
momentum polar direction.

If `polar(M_t)` and `NS(M_t)` have tail drift below Fro/GD while maintaining high
cosine and alignment with `polar(G_t)`, the paper can describe the sampled
Muon-style directions as compatible with the ideal spectral direction in this
local diagnostic. If these rows degrade, the safe statement remains only about
spectral-gradient/polar geometry.

Current readout: `polar(M_t)` has squared tail-example logit drift ratio
`{_fmt(polar_momentum['geomean_tail_output_drift_sq_ratio_vs_fro'])}` vs Fro/GD,
and `NS(M_t)` has squared drift ratio
`{_fmt(ns_momentum['geomean_tail_output_drift_sq_ratio_vs_fro'])}`. This supports
only a selected-state compatibility check; it still does not prove full Muon
training performance.

Figure: [{figure_path.as_posix()}](../{figure_path.as_posix()})

Artifacts:

- [step_metrics.csv](../results/e11_long_tail_muon_bridge/step_metrics.csv)
- [pair_summary.csv](../results/e11_long_tail_muon_bridge/pair_summary.csv)
- [config.json](../results/e11_long_tail_muon_bridge/config.json)
"""
    DISCUSSION_PATH.write_text(text, encoding="utf-8")


def main() -> None:
    config = LongTailMuonBridgeConfig()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    step_metrics, pair_summary = run_long_tail_muon_bridge(config)
    step_metrics.to_csv(OUTPUT_DIR / "step_metrics.csv", index=False)
    pair_summary.to_csv(OUTPUT_DIR / "pair_summary.csv", index=False)
    (OUTPUT_DIR / "config.json").write_text(json.dumps(config.__dict__, indent=2), encoding="utf-8")
    figure_path = write_figure(pair_summary)
    write_discussion(config, pair_summary, figure_path)
    print(f"saved long-tail Muon bridge results to {OUTPUT_DIR}")
    print(f"step rows={len(step_metrics)}, seeds={len(config.seeds)}")
    print(f"figure: {figure_path}")
    print(f"discussion: {DISCUSSION_PATH}")
    print(pair_summary.to_string(index=False))


if __name__ == "__main__":
    main()
