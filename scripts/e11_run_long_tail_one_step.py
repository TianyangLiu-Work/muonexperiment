from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.long_tail_one_step import LongTailOneStepConfig, run_long_tail_one_step


OUTPUT_DIR = Path("results/e11_long_tail_one_step")
FIGURE_DIR = Path("figures/e11_long_tail_one_step")
DISCUSSION_PATH = Path("discussion/e11_long_tail_one_step.md")


def _fmt(value: float) -> str:
    return f"{value:.4g}"


def write_figure(step_metrics, pair_summary) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    order = ["frobenius", "spectral"]
    colors = {"frobenius": "#D55E00", "spectral": "#0072B2"}
    step_metrics = step_metrics.copy()
    step_metrics["tail_output_drift_rms_sq"] = step_metrics["tail_output_drift_rms"] ** 2
    drift_values = [
        step_metrics.loc[step_metrics["geometry"] == geometry, "tail_output_drift_rms_sq"].to_numpy(dtype=float)
        for geometry in order
    ]
    axes[0].boxplot(drift_values, tick_labels=["Fro/GD", "Spectral"], patch_artist=True)
    for patch, geometry in zip(axes[0].artists, order):
        patch.set_facecolor(colors[geometry])
        patch.set_alpha(0.35)
    axes[0].set_ylabel("mean squared tail-example logit drift")
    axes[0].set_title("Held-out tail function movement")

    for seed, group in step_metrics.groupby("seed", observed=True, sort=False):
        group = group.set_index("geometry").loc[order]
        axes[1].plot([0, 1], group["tail_loss_increase"], color="#666666", alpha=0.35, linewidth=1)
        axes[1].scatter(
            [0, 1],
            group["tail_loss_increase"],
            color=[colors["frobenius"], colors["spectral"]],
            s=18,
            zorder=3,
        )
    axes[1].axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(["Fro/GD", "Spectral"])
    axes[1].set_ylabel("tail loss increase")
    axes[1].set_title("Paired one-step tail loss response")

    ratio = pair_summary.iloc[0]["geomean_tail_output_drift_sq_ratio_spectral_over_fro"]
    low = pair_summary.iloc[0]["tail_output_drift_sq_ratio_ci95_low"]
    high = pair_summary.iloc[0]["tail_output_drift_sq_ratio_ci95_high"]
    fig.suptitle(
        "Long-tailed digits one-step diagnostic: "
        f"squared drift ratio={ratio:.3g} [{low:.3g}, {high:.3g}]"
    )
    fig.tight_layout()
    path = FIGURE_DIR / "long_tail_one_step_tail_response.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def write_discussion(config: LongTailOneStepConfig, pair_summary, figure_path: Path) -> None:
    row = pair_summary.iloc[0].to_dict()
    lines = [
        "# E11 Long-Tailed One-Step Diagnostic",
        "",
        "This probe trains a small MLP checkpoint on an imbalanced sklearn digits split,",
        "then forms a head-only gradient from classes 0-4 and evaluates the same",
        "one-step intervention on held-out tail classes 5-9. Frobenius/GD and",
        "spectral/polar directions are both rescaled to the same head first-order",
        "gain before measuring tail function movement.",
        "",
        f"- Seeds: {len(config.seeds)}",
        f"- Head classes: {config.head_classes}",
        f"- Tail classes: {config.tail_classes}",
        f"- Head train examples per class: {config.head_train_per_class}",
        f"- Tail train examples per class: {config.tail_train_per_class}",
        f"- Tail evaluation examples per class: {config.tail_eval_per_class}",
        f"- Warmup steps: {config.warmup_steps}",
        f"- Hidden dimension: {config.hidden_dim}",
        f"- Target head first-order gain: {config.target_head_gain_fraction} * head loss",
        "",
        "Ratio columns are spectral divided by Frobenius/GD. Values below 1 mean the",
        "spectral direction disturbed logits on held-out tail examples less after matching head gain.",
        "",
        "| quantity | value |",
        "|---|---:|",
        f"| squared tail-example logit drift ratio, spectral/Fro | {_fmt(row['geomean_tail_output_drift_sq_ratio_spectral_over_fro'])} [{_fmt(row['tail_output_drift_sq_ratio_ci95_low'])}, {_fmt(row['tail_output_drift_sq_ratio_ci95_high'])}] |",
        f"| centered-logit squared drift ratio, spectral/Fro | {_fmt(row['geomean_centered_tail_output_drift_sq_ratio_spectral_over_fro'])} [{_fmt(row['centered_tail_output_drift_sq_ratio_ci95_low'])}, {_fmt(row['centered_tail_output_drift_sq_ratio_ci95_high'])}] |",
        f"| true-class logit delta squared ratio, spectral/Fro | {_fmt(row['geomean_true_logit_delta_sq_ratio_spectral_over_fro'])} [{_fmt(row['true_logit_delta_sq_ratio_ci95_low'])}, {_fmt(row['true_logit_delta_sq_ratio_ci95_high'])}] |",
        f"| top-competitor logit delta squared ratio, spectral/Fro | {_fmt(row['geomean_competitor_logit_delta_sq_ratio_spectral_over_fro'])} [{_fmt(row['competitor_logit_delta_sq_ratio_ci95_low'])}, {_fmt(row['competitor_logit_delta_sq_ratio_ci95_high'])}] |",
        f"| margin-delta squared ratio, spectral/Fro | {_fmt(row['geomean_margin_delta_sq_ratio_spectral_over_fro'])} [{_fmt(row['margin_delta_sq_ratio_ci95_low'])}, {_fmt(row['margin_delta_sq_ratio_ci95_high'])}] |",
        f"| spectral lower squared tail-example logit drift fraction | {_fmt(row['spectral_less_tail_output_drift_fraction'])} |",
        f"| actual head-gain relative error, Fro/GD | {_fmt(row['mean_actual_head_gain_relative_error_frobenius'])} [{_fmt(row['actual_head_gain_relative_error_frobenius_ci95_low'])}, {_fmt(row['actual_head_gain_relative_error_frobenius_ci95_high'])}] |",
        f"| actual head-gain relative error, spectral | {_fmt(row['mean_actual_head_gain_relative_error_spectral'])} [{_fmt(row['actual_head_gain_relative_error_spectral_ci95_low'])}, {_fmt(row['actual_head_gain_relative_error_spectral_ci95_high'])}] |",
        f"| tail loss increase, Fro/GD | {_fmt(row['mean_tail_loss_increase_frobenius'])} [{_fmt(row['tail_loss_increase_frobenius_ci95_low'])}, {_fmt(row['tail_loss_increase_frobenius_ci95_high'])}] |",
        f"| tail loss increase, spectral | {_fmt(row['mean_tail_loss_increase_spectral'])} [{_fmt(row['tail_loss_increase_spectral_ci95_low'])}, {_fmt(row['tail_loss_increase_spectral_ci95_high'])}] |",
        f"| tail loss increase diff, spectral - Fro | {_fmt(row['mean_tail_loss_increase_diff_spectral_minus_fro'])} [{_fmt(row['tail_loss_increase_diff_ci95_low'])}, {_fmt(row['tail_loss_increase_diff_ci95_high'])}] |",
        f"| tail margin drop, Fro/GD | {_fmt(row['mean_tail_margin_drop_frobenius'])} [{_fmt(row['tail_margin_drop_frobenius_ci95_low'])}, {_fmt(row['tail_margin_drop_frobenius_ci95_high'])}] |",
        f"| tail margin drop, spectral | {_fmt(row['mean_tail_margin_drop_spectral'])} [{_fmt(row['tail_margin_drop_spectral_ci95_low'])}, {_fmt(row['tail_margin_drop_spectral_ci95_high'])}] |",
        f"| tail margin drop diff, spectral - Fro | {_fmt(row['mean_tail_margin_drop_diff_spectral_minus_fro'])} [{_fmt(row['tail_margin_drop_diff_ci95_low'])}, {_fmt(row['tail_margin_drop_diff_ci95_high'])}] |",
        f"| tail accuracy drop, Fro/GD | {_fmt(row['mean_tail_accuracy_drop_frobenius'])} [{_fmt(row['tail_accuracy_drop_frobenius_ci95_low'])}, {_fmt(row['tail_accuracy_drop_frobenius_ci95_high'])}] |",
        f"| tail accuracy drop, spectral | {_fmt(row['mean_tail_accuracy_drop_spectral'])} [{_fmt(row['tail_accuracy_drop_spectral_ci95_low'])}, {_fmt(row['tail_accuracy_drop_spectral_ci95_high'])}] |",
        f"| tail accuracy drop diff, spectral - Fro | {_fmt(row['mean_tail_accuracy_drop_diff_spectral_minus_fro'])} [{_fmt(row['tail_accuracy_drop_diff_ci95_low'])}, {_fmt(row['tail_accuracy_drop_diff_ci95_high'])}] |",
        f"| actual head loss decrease diff, spectral - Fro | {_fmt(row['mean_actual_head_loss_decrease_diff_spectral_minus_fro'])} [{_fmt(row['actual_head_loss_decrease_diff_ci95_low'])}, {_fmt(row['actual_head_loss_decrease_diff_ci95_high'])}] |",
        f"| mean tail diagnostic condition score | {_fmt(row['mean_condition_score_tail'])} |",
        f"| tail CE before | {_fmt(row['mean_tail_loss_before'])} [{_fmt(row['tail_loss_before_ci95_low'])}, {_fmt(row['tail_loss_before_ci95_high'])}] |",
        f"| tail CE after Fro/GD | {_fmt(row['mean_tail_loss_after_frobenius'])} [{_fmt(row['tail_loss_after_frobenius_ci95_low'])}, {_fmt(row['tail_loss_after_frobenius_ci95_high'])}] |",
        f"| tail CE after spectral | {_fmt(row['mean_tail_loss_after_spectral'])} [{_fmt(row['tail_loss_after_spectral_ci95_low'])}, {_fmt(row['tail_loss_after_spectral_ci95_high'])}] |",
        f"| tail margin before | {_fmt(row['mean_tail_margin_before'])} [{_fmt(row['tail_margin_before_ci95_low'])}, {_fmt(row['tail_margin_before_ci95_high'])}] |",
        f"| tail margin after Fro/GD | {_fmt(row['mean_tail_margin_after_frobenius'])} [{_fmt(row['tail_margin_after_frobenius_ci95_low'])}, {_fmt(row['tail_margin_after_frobenius_ci95_high'])}] |",
        f"| tail margin after spectral | {_fmt(row['mean_tail_margin_after_spectral'])} [{_fmt(row['tail_margin_after_spectral_ci95_low'])}, {_fmt(row['tail_margin_after_spectral_ci95_high'])}] |",
        f"| tail accuracy before | {_fmt(row['mean_tail_accuracy_before'])} [{_fmt(row['tail_accuracy_before_ci95_low'])}, {_fmt(row['tail_accuracy_before_ci95_high'])}] |",
        f"| positive-margin tail fraction before | {_fmt(row['mean_tail_positive_margin_fraction_before'])} [{_fmt(row['tail_positive_margin_fraction_ci95_low'])}, {_fmt(row['tail_positive_margin_fraction_ci95_high'])}] |",
        f"| certified preserved fraction, Fro/GD | {_fmt(row['mean_tail_margin_certified_preserved_fraction_frobenius'])} [{_fmt(row['tail_margin_certified_preserved_fraction_frobenius_ci95_low'])}, {_fmt(row['tail_margin_certified_preserved_fraction_frobenius_ci95_high'])}] |",
        f"| certified preserved fraction, spectral | {_fmt(row['mean_tail_margin_certified_preserved_fraction_spectral'])} [{_fmt(row['tail_margin_certified_preserved_fraction_spectral_ci95_low'])}, {_fmt(row['tail_margin_certified_preserved_fraction_spectral_ci95_high'])}] |",
        f"| all-tail prediction changed fraction, Fro/GD | {_fmt(row['mean_tail_prediction_changed_fraction_frobenius'])} [{_fmt(row['tail_prediction_changed_fraction_frobenius_ci95_low'])}, {_fmt(row['tail_prediction_changed_fraction_frobenius_ci95_high'])}] |",
        f"| all-tail prediction changed fraction, spectral | {_fmt(row['mean_tail_prediction_changed_fraction_spectral'])} [{_fmt(row['tail_prediction_changed_fraction_spectral_ci95_low'])}, {_fmt(row['tail_prediction_changed_fraction_spectral_ci95_high'])}] |",
        f"| positive-margin prediction changed fraction, Fro/GD | {_fmt(row['mean_tail_positive_margin_prediction_changed_fraction_frobenius'])} [{_fmt(row['tail_positive_margin_prediction_changed_fraction_frobenius_ci95_low'])}, {_fmt(row['tail_positive_margin_prediction_changed_fraction_frobenius_ci95_high'])}] |",
        f"| positive-margin prediction changed fraction, spectral | {_fmt(row['mean_tail_positive_margin_prediction_changed_fraction_spectral'])} [{_fmt(row['tail_positive_margin_prediction_changed_fraction_spectral_ci95_low'])}, {_fmt(row['tail_positive_margin_prediction_changed_fraction_spectral_ci95_high'])}] |",
        "",
        f"Figure: [{figure_path.as_posix()}](../{figure_path.as_posix()})",
        "",
        "Main readout: this is the first non-synthetic one-step check of the",
        "head-to-tail framing. It does not prove long-tailed generalization, but it",
        "tests the required matched-head-gain protocol on a real data distribution.",
        "The key claim should be stated from the table above, not from optimizer labels",
        "alone.",
        "",
        "Caveats:",
        "- This is sklearn digits, not CIFAR-100-LT/ImageNet-LT/iNaturalist.",
        "- The spectral direction is an exact polar intervention, not a full Muon optimizer state.",
        "- MLP nonlinearities make the closed-form sandwich condition layer-dependent; direct measured tail drift is the primary one-step quantity.",
        "",
        "Artifacts:",
        "- [step_metrics.csv](../results/e11_long_tail_one_step/step_metrics.csv)",
        "- [pair_summary.csv](../results/e11_long_tail_one_step/pair_summary.csv)",
        "- [layer_metrics.csv](../results/e11_long_tail_one_step/layer_metrics.csv)",
        "- [config.json](../results/e11_long_tail_one_step/config.json)",
    ]
    DISCUSSION_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    config = LongTailOneStepConfig()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    step_metrics, pair_summary, layer_metrics = run_long_tail_one_step(config)
    step_metrics.to_csv(OUTPUT_DIR / "step_metrics.csv", index=False)
    pair_summary.to_csv(OUTPUT_DIR / "pair_summary.csv", index=False)
    layer_metrics.to_csv(OUTPUT_DIR / "layer_metrics.csv", index=False)
    (OUTPUT_DIR / "config.json").write_text(json.dumps(config.__dict__, indent=2), encoding="utf-8")
    figure_path = write_figure(step_metrics, pair_summary)
    write_discussion(config, pair_summary, figure_path)
    print(f"saved long-tail one-step results to {OUTPUT_DIR}")
    print(f"step rows={len(step_metrics)}, seeds={len(config.seeds)}")
    print(f"figure: {figure_path}")
    print(f"discussion: {DISCUSSION_PATH}")
    print(pair_summary.to_string(index=False))


if __name__ == "__main__":
    main()
