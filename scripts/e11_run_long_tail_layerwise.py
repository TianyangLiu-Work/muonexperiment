from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.long_tail_layerwise import LongTailLayerwiseConfig, run_long_tail_layerwise


OUTPUT_DIR = Path("results/e11_long_tail_layerwise")
FIGURE_DIR = Path("figures/e11_long_tail_layerwise")
DISCUSSION_PATH = Path("discussion/e11_long_tail_layerwise.md")


def _fmt(value: float) -> str:
    return f"{value:.4g}"


def write_figure(metrics, summary) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    summary = summary.sort_values("layer")
    x = range(len(summary))
    width = 0.35
    axes[0].bar(
        [value - width / 2 for value in x],
        summary["geomean_scaled_jvp_tail_drift_sq_ratio_spectral_over_fro"],
        width=width,
        color="#0072B2",
        label="scaled JVP squared drift ratio",
    )
    axes[0].bar(
        [value + width / 2 for value in x],
        summary["geomean_observed_tail_drift_sq_ratio_spectral_over_fro"],
        width=width,
        color="#D55E00",
        label="observed squared drift ratio",
    )
    axes[0].axhline(1.0, color="black", linewidth=1.0, linestyle="--")
    axes[0].set_yscale("log")
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels([f"layer {int(layer)}" for layer in summary["layer"]])
    axes[0].set_ylabel("spectral / Fro")
    axes[0].set_title("Layerwise squared tail-drift ratios")
    axes[0].legend(frameon=False)

    for geometry, color, label in [("frobenius", "#D55E00", "Fro/GD"), ("spectral", "#0072B2", "Spectral")]:
        subset = metrics[metrics["geometry"] == geometry]
        axes[1].scatter(
            subset["local_operator_condition_score"],
            subset["tail_output_drift_fro"],
            s=28,
            alpha=0.75,
            color=color,
            label=label,
        )
    axes[1].set_xlabel("local operator score nr(G_l) / sr(J_T,l)")
    axes[1].set_ylabel("layer-only tail output drift")
    axes[1].set_title("Downstream-aware score vs observed drift")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    path = FIGURE_DIR / "long_tail_layerwise_drift.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def write_discussion(config: LongTailLayerwiseConfig, summary, figure_path: Path) -> None:
    lines = [
        "# E11 Long-Tailed Layerwise Diagnostic",
        "",
        "This probe uses the same imbalanced sklearn digits checkpoint as the one-step",
        "and head-only forgetting diagnostics. For each layer, it measures the head",
        "gradient rank, tail activation stable rank, downstream-aware tail rank,",
        "finite-difference tail JVP drift, and layer-only tail drift under matched",
        "first-order head gain.",
        "",
        f"- Seeds: {len(config.seeds)}",
        f"- Layers: 2",
        f"- Head classes: {config.head_classes}",
        f"- Tail classes: {config.tail_classes}",
        f"- Warmup steps: {config.warmup_steps}",
        f"- Target head first-order gain: {config.target_head_gain_fraction} * head-batch loss",
        f"- JVP finite-difference epsilon: {config.jvp_epsilon}",
        "",
        "Ratio columns are spectral divided by Frobenius/GD. Values below 1 mean the",
        "spectral layer direction disturbs logits on held-out tail examples less.",
        "",
        "| layer | activation-only score nr(G)/sr(A) | downstream-aware rank | local operator score | unit-JVP squared drift ratio 95% CI | scaled-JVP squared drift ratio 95% CI | observed squared drift ratio 95% CI | spectral lower observed fraction | tail loss diff 95% CI |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.sort_values("layer").to_dict(orient="records"):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(int(row["layer"])),
                    _fmt(row["mean_condition_score"]),
                    (
                        _fmt(row["mean_tail_sandwiched_stable_rank"])
                        if row["mean_tail_sandwiched_stable_rank"] == row["mean_tail_sandwiched_stable_rank"]
                        else "n/a"
                    ),
                    _fmt(row["mean_local_operator_condition_score"]),
                    (
                        f"{_fmt(row['geomean_jvp_tail_drift_sq_ratio_spectral_over_fro'])} "
                        f"[{_fmt(row['jvp_tail_drift_sq_ratio_ci95_low'])}, "
                        f"{_fmt(row['jvp_tail_drift_sq_ratio_ci95_high'])}]"
                    ),
                    (
                        f"{_fmt(row['geomean_scaled_jvp_tail_drift_sq_ratio_spectral_over_fro'])} "
                        f"[{_fmt(row['scaled_jvp_tail_drift_sq_ratio_ci95_low'])}, "
                        f"{_fmt(row['scaled_jvp_tail_drift_sq_ratio_ci95_high'])}]"
                    ),
                    (
                        f"{_fmt(row['geomean_observed_tail_drift_sq_ratio_spectral_over_fro'])} "
                        f"[{_fmt(row['observed_tail_drift_sq_ratio_ci95_low'])}, "
                        f"{_fmt(row['observed_tail_drift_sq_ratio_ci95_high'])}]"
                    ),
                    _fmt(row["spectral_less_observed_tail_drift_fraction"]),
                    (
                        f"{_fmt(row['mean_tail_loss_increase_diff_spectral_minus_fro'])} "
                        f"[{_fmt(row['tail_loss_increase_diff_ci95_low'])}, "
                        f"{_fmt(row['tail_loss_increase_diff_ci95_high'])}]"
                    ),
                ]
            )
            + " |"
        )
    first = summary.iloc[0]
    lines.extend(
        [
            "",
            f"Figure: [{figure_path.as_posix()}](../{figure_path.as_posix()})",
            "",
            "Main readout: this is a layer-level check of the head-to-tail mechanism.",
            "The strongest supported statement is whether finite-difference tail JVP",
            "squared drift ratios, head-gain scaling, and layer-only observed squared drift ratios agree layer",
            "by layer. Unit JVP drift measures tail sensitivity of the direction; scaled",
            "JVP drift also includes the smaller/larger step required to match head gain.",
            "For layer 2, `downstream-aware rank` is the exact sandwich quantity",
            "`ssrank(B_T,A_T)` because the final linear layer has a single downstream",
            "matrix. For layer 1, ReLU gates vary across tail samples, so the layer is",
            "not a single `B_T D A_T` block; the table instead reports the exact stable",
            "rank of the frozen-gate local linear operator `J_{T,1}` through the",
            "`local operator score`.",
            "",
            "Across both layers, the Spearman correlation between local operator score and",
            "the observed spectral/Frobenius squared drift ratio is "
            f"{_fmt(first['local_operator_score_observed_ratio_spearman_all_layers'])} "
            f"[{_fmt(first['local_operator_score_observed_ratio_spearman_ci95_low'])}, "
            f"{_fmt(first['local_operator_score_observed_ratio_spearman_ci95_high'])}].",
            "",
            "Caveats:",
            "- This is still a small MLP diagnostic, not an exact matrix-block theorem check.",
            "- JVP is estimated by finite difference; the epsilon is fixed and reported.",
            "- Per-layer updates are artificial interventions and should not be interpreted as complete optimizer trajectories.",
            "",
            "Artifacts:",
            "- [metrics.csv](../results/e11_long_tail_layerwise/metrics.csv)",
            "- [summary.csv](../results/e11_long_tail_layerwise/summary.csv)",
            "- [config.json](../results/e11_long_tail_layerwise/config.json)",
        ]
    )
    DISCUSSION_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    config = LongTailLayerwiseConfig()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics, summary = run_long_tail_layerwise(config)
    metrics.to_csv(OUTPUT_DIR / "metrics.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)
    (OUTPUT_DIR / "config.json").write_text(json.dumps(config.__dict__, indent=2), encoding="utf-8")
    figure_path = write_figure(metrics, summary)
    write_discussion(config, summary, figure_path)
    print(f"saved long-tail layerwise results to {OUTPUT_DIR}")
    print(f"metric rows={len(metrics)}, seeds={len(config.seeds)}")
    print(f"figure: {figure_path}")
    print(f"discussion: {DISCUSSION_PATH}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
