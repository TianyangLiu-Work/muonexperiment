from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.head_tail_interference import HeadTailConfig, run_head_tail_interference


OUTPUT_DIR = Path("results/e11_head_tail_interference")
FIGURE_DIR = Path("figures/e11_head_tail_interference")
DISCUSSION_PATH = Path("discussion/e11_head_tail_interference.md")


def _fmt(value: float) -> str:
    return f"{value:.4g}"


def write_figure(pair_summary):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plot_data = pair_summary.set_index("setting")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = range(len(plot_data))
    width = 0.35
    ax.bar(
        [value - width / 2 for value in x],
        plot_data["mean_theory_ratio_spectral_over_fro"],
        width=width,
        label="theory ratio",
    )
    ax.bar(
        [value + width / 2 for value in x],
        plot_data["geomean_tail_output_drift_sq_ratio_spectral_over_fro"],
        width=width,
        label="observed squared drift ratio",
    )
    ax.axhline(1.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_yscale("log")
    ax.set_ylabel("spectral / frobenius")
    ax.set_xticks(list(x))
    ax.set_xticklabels(plot_data.index, rotation=15, ha="right")
    ax.set_title("Head-to-tail output drift under matched head gain")
    ax.legend(frameon=False)
    fig.tight_layout()
    path = FIGURE_DIR / "head_tail_drift_ratio.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def write_discussion(config: HeadTailConfig, pair_summary, figure_path: Path) -> None:
    lines = [
        "# E11 Head-to-Tail Interference Probe",
        "",
        "This probe implements a synthetic one-step linear classification experiment for the",
        "head-to-tail interference note. For each seed, it compares a Frobenius-normalized",
        "gradient step and a spectral/polar step scaled to the same first-order head gain.",
        "The tail batch is held out; the reported drift is the change in logits on tail examples after",
        "the head-only step.",
        "",
        f"- Seeds: {len(config.seeds)}",
        f"- Output dimension: {config.output_dim}",
        f"- Input dimension: {config.input_dim}",
        f"- Tail batch size: {config.tail_batch_size}",
        f"- First-order head gain: {config.first_order_gain_per_grad_fro} * ||G_H||_F",
        "",
        "Ratio columns are spectral divided by Frobenius. Values below 1 mean the spectral",
        "step disturbed the tail outputs less.",
        "",
        "| setting | predicted spectral less drift | nrank(G_H) | srank(A_T) | ssrank(B_T,A_T) | theory ratio | observed squared drift ratio 95% CI | spectral less drift fraction | CE increase diff 95% CI | margin drop diff 95% CI |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in pair_summary.to_dict(orient="records"):
        drift_ci = (
            f"{_fmt(row['geomean_tail_output_drift_sq_ratio_spectral_over_fro'])} "
            f"[{_fmt(row['tail_output_drift_sq_ratio_ci95_low'])}, "
            f"{_fmt(row['tail_output_drift_sq_ratio_ci95_high'])}]"
        )
        ce_ci = (
            f"{_fmt(row['mean_tail_ce_increase_diff_spectral_minus_fro'])} "
            f"[{_fmt(row['tail_ce_increase_diff_ci95_low'])}, "
            f"{_fmt(row['tail_ce_increase_diff_ci95_high'])}]"
        )
        margin_ci = (
            f"{_fmt(row['mean_tail_margin_drop_diff_spectral_minus_fro'])} "
            f"[{_fmt(row['tail_margin_drop_diff_ci95_low'])}, "
            f"{_fmt(row['tail_margin_drop_diff_ci95_high'])}]"
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["setting"]),
                    str(bool(row["predicted_spectral_less_drift"])),
                    _fmt(row["mean_head_gradient_nuclear_rank"]),
                    _fmt(row["mean_tail_activation_stable_rank"]),
                    _fmt(row["mean_tail_downstream_aware_stable_rank"]),
                    _fmt(row["mean_theory_ratio_spectral_over_fro"]),
                    drift_ci,
                    _fmt(row["spectral_less_tail_output_drift_fraction"]),
                    ce_ci,
                    margin_ci,
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            f"Figure: [{figure_path.as_posix()}](../{figure_path.as_posix()})",
            "",
            "Main readout: the observed tail-output drift follows the same qualitative boundary",
            "as the theoretical ratio ssrank(B_T,A_T) / nrank(G_H). In the high-head-rank /",
            "low downstream-aware tail-stable-rank setting, the spectral step has lower tail",
            "drift. In the low-head-rank / high downstream-aware tail-stable-rank setting,",
            "it has higher tail drift.",
            "",
            "Caveat: this first probe validates the matched-head-gain drift mechanism, not",
            "real long-tailed generalization. Cross-entropy and margin changes are reported as",
            "secondary diagnostics and can be smaller or noisier than direct output drift.",
            "",
            "Artifacts:",
            "- [step_metrics.csv](../results/e11_head_tail_interference/step_metrics.csv)",
            "- [pair_summary.csv](../results/e11_head_tail_interference/pair_summary.csv)",
            "- [config.json](../results/e11_head_tail_interference/config.json)",
        ]
    )
    DISCUSSION_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    config = HeadTailConfig()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    step_metrics, pair_summary = run_head_tail_interference(config)
    step_metrics.to_csv(OUTPUT_DIR / "step_metrics.csv", index=False)
    pair_summary.to_csv(OUTPUT_DIR / "pair_summary.csv", index=False)
    (OUTPUT_DIR / "config.json").write_text(json.dumps(config.__dict__, indent=2), encoding="utf-8")
    figure_path = write_figure(pair_summary)
    write_discussion(config, pair_summary, figure_path)
    print(f"saved head-to-tail results to {OUTPUT_DIR}")
    print(f"step rows={len(step_metrics)}, settings={pair_summary['setting'].nunique()}, seeds={len(config.seeds)}")
    print(f"figure: {figure_path}")
    print(f"discussion: {DISCUSSION_PATH}")
    print(pair_summary.to_string(index=False))


if __name__ == "__main__":
    main()
