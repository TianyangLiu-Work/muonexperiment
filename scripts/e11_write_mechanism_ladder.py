from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.reporting import markdown_table, write_markdown


OUTPUT_PATH = Path("discussion/e11_mechanism_ladder.md")
FIGURE_DIR = Path("figures/e11_mechanism_ladder")
SUMMARY_PATH = Path("results/e11_mechanism_ladder/intervention_direction_summary.csv")


def build_direction_summary() -> pd.DataFrame:
    target = pd.read_csv("results/e11_target_update_sweep/target_pair_summary.csv")
    target = target[
        (target["group_type"] == "setting_all_targets")
        & (target["metric"] == "update_grad_inner")
    ].copy()
    target["control_level"] = "global_update_norm_fixed"
    target["setting_label"] = target["base_setting"]
    target["available_interpretation"] = "direction plus layer allocation"

    layer = pd.read_csv("results/e11_mlp_per_layer_control/pair_summary.csv")
    layer = layer[
        (layer["group_type"] == "hidden_all_targets")
        & (layer["metric"] == "update_grad_inner")
    ].copy()
    layer["control_level"] = "per_layer_update_norm_fixed"
    layer["problem_family"] = "SmallMLPDigits"
    layer["base_setting"] = layer["hidden_dim"].map(lambda value: f"Small MLP digits hidden={value}")
    layer["setting_label"] = layer["base_setting"]
    layer["available_interpretation"] = "direction after layer allocation removed"

    common = [
        "control_level",
        "problem_family",
        "base_setting",
        "setting_label",
        "metric",
        "n_pairs",
        "muon_win_rate",
        "geomean_ratio_muon_over_adam",
        "ratio_ci95_low",
        "ratio_ci95_high",
        "ratio_ci95_above_one",
        "ratio_ci95_below_one",
        "available_interpretation",
    ]
    return pd.concat([target[common], layer[common]], ignore_index=True)


def build_ladder_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "level": "Raw optimizer run",
                "controlled": "Nothing beyond same initialization and nominal lr grid.",
                "question_answered": "Does a natural optimizer configuration win in this short horizon?",
                "current_status": "Setting-dependent; raw MLP strongly favors Adam, Matrix Sensing favors Muon.",
            },
            {
                "level": "Best learning rate",
                "controlled": "Nominal learning-rate choice, using oracle best final loss per seed.",
                "question_answered": "Is the conclusion only a single-lr artifact?",
                "current_status": "No. Sweep weakens the lr-artifact concern but does not create a global Muon win.",
            },
            {
                "level": "Equal-update / target global update norm",
                "controlled": "Global relative Frobenius update size.",
                "question_answered": "At the same global step size, is Muon's direction better?",
                "current_status": "Still setting-dependent: Matrix Sensing and MLP hidden=16 favor Muon; MF and MLP hidden=64 favor Adam.",
            },
            {
                "level": "Per-layer update norm",
                "controlled": "Each layer's relative update size in SmallMLP.",
                "question_answered": "Is the MLP width effect only layer allocation?",
                "current_status": "No for hidden=64: Adam advantage survives. Hidden=16 becomes target-scale dependent and near-neutral overall.",
            },
            {
                "level": "Within-layer spectral allocation",
                "controlled": "Gradient singular vectors are fixed; singular-value allocation and norm budget are varied in a one-step probe.",
                "question_answered": "Does the remaining effect come from singular-vector alignment or singular-value allocation inside each layer?",
                "current_status": "Norm geometry matters: GD spectrum wins under Frobenius budget, flat polar wins under operator-norm budget.",
            },
            {
                "level": "Singular-vector and trajectory effects",
                "controlled": "Measured by subspace overlap, polar-vector swaps, and natural update-vector swaps.",
                "question_answered": "Do natural optimizer states differ because their singular vectors and trajectories move to different regions?",
                "current_status": "Yes, but not monotonically. Swaps usually reduce progress overall, while some setting/budget/eval-state cells favor the other update.",
            },
            {
                "level": "Trajectory-level optimizer switch",
                "controlled": "Checkpoint state is fixed; remaining horizon compares preserved, fresh-own, fresh-switched, varied-horizon, and best-continuation-lr continuations.",
                "question_answered": "Does one-step trajectory specialization imply the original optimizer is the better continuation?",
                "current_status": "No. Fresh-switch effects depend on source optimizer, task family, continuation horizon, and continuation tuning.",
            },
        ]
    )


def plot_direction_ladder(summary: pd.DataFrame) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / "mechanism_direction_ladder.png"
    rows = summary.copy()
    settings = [
        "MF input kappa=1e+02",
        "MF input kappa=1e+05",
        "Matrix sensing kappa=1e+02",
        "Matrix sensing kappa=1e+05",
        "Small MLP digits hidden=16",
        "Small MLP digits hidden=64",
    ]
    rows["setting_label"] = pd.Categorical(rows["setting_label"], categories=settings, ordered=True)
    rows = rows.dropna(subset=["setting_label"]).sort_values(["setting_label", "control_level"])
    x_positions = {setting: idx for idx, setting in enumerate(settings)}
    offsets = {"global_update_norm_fixed": -0.13, "per_layer_update_norm_fixed": 0.13}
    colors = {"global_update_norm_fixed": "#0072B2", "per_layer_update_norm_fixed": "#D55E00"}
    labels = {
        "global_update_norm_fixed": "global update norm fixed",
        "per_layer_update_norm_fixed": "per-layer update norm fixed",
    }

    fig, ax = plt.subplots(figsize=(10.5, 4.3))
    for level in ["global_update_norm_fixed", "per_layer_update_norm_fixed"]:
        sub = rows[rows["control_level"] == level]
        x = np.array([x_positions[str(setting)] + offsets[level] for setting in sub["setting_label"]], dtype=float)
        y = sub["geomean_ratio_muon_over_adam"].to_numpy(dtype=float)
        yerr = np.vstack(
            [
                y - sub["ratio_ci95_low"].to_numpy(dtype=float),
                sub["ratio_ci95_high"].to_numpy(dtype=float) - y,
            ]
        )
        ax.errorbar(x, y, yerr=yerr, fmt="o", capsize=3, color=colors[level], label=labels[level])
    ax.axhline(1.0, color="#555555", ls="--", lw=0.9)
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(settings)))
    ax.set_xticklabels(
        ["MF\n1e2", "MF\n1e5", "MS\n1e2", "MS\n1e5", "MLP h16", "MLP h64"],
        rotation=0,
    )
    ax.set_ylabel("first-order ratio: Muon / Adam")
    ax.set_title("Intervention ladder for optimizer direction effects")
    ax.grid(alpha=0.25, axis="y")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    summary = build_direction_summary()
    ladder = build_ladder_table()
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_PATH, index=False)
    figure = plot_direction_ladder(summary)
    text = f"""# E11 Mechanism Ladder

## Purpose

This note consolidates the intervention sequence into one mechanism-level reading. The point is to distinguish claims about optimizer geometry from claims about performance, learning-rate choice, global update scale, layer allocation, and remaining within-layer spectral effects.

## Intervention Ladder

{markdown_table(ladder, ["level", "controlled", "question_answered", "current_status"])}

## Direction Ratios Under Size Controls

Ratios above 1 mean Muon's update direction gives larger one-step first-order progress than Adam's direction under the stated control.

![Mechanism direction ladder](../{figure})

{markdown_table(summary, ["control_level", "problem_family", "base_setting", "n_pairs", "muon_win_rate", "geomean_ratio_muon_over_adam", "ratio_ci95_low", "ratio_ci95_high", "ratio_ci95_above_one", "ratio_ci95_below_one", "available_interpretation"])}

## Current Mechanistic Reading

1. The most stable optimizer-intrinsic fact is update-spectrum shaping: ExactMuon produces flatter, higher-rank update matrices.
2. This spectral shaping does not imply a global optimization advantage.
3. Controlling global update size leaves a real direction effect: Muon is favorable in Matrix Sensing and SmallMLP hidden=16, but unfavorable in MF-with-input and SmallMLP hidden=64.
4. Controlling per-layer update size changes the SmallMLP hidden=16 conclusion from favorable to near-neutral, while hidden=64 remains strongly Adam-favorable.
5. The within-layer spectral allocation probe clarifies the norm geometry: flat/polar allocation loses to GD allocation under a Frobenius budget, but wins under an operator-norm budget.
6. The singular-vector trajectory diagnostic shows that natural Adam/Muon states can also diverge in their gradient and parameter subspaces; this is strong in Matrix Sensing and MLP, but weak in MF-with-input.
7. The singular-vector swap probe gives a one-step causal check: using the other optimizer's matched-state polar singular vectors reduces progress, and in Matrix Sensing often turns the update into ascent.
8. The natural update-vector swap probe is more optimizer-level: overall other/own progress is below 1 across five target scales, but the effect depends on budget and evaluation state, so one-step trajectory specialization is real but not a universal own-update dominance rule.
9. The optimizer-switch probes add an important negative result: one-step specialization does not imply that the original optimizer is the best remaining-horizon continuation. After reset control, horizon sweeps, and a small continuation-LR sweep, practical switch effects remain source-, task-, horizon-, and tuning-dependent.
10. Therefore the current best thesis is conditional: Muon's polar direction helps when the task/layer geometry and norm constraint reward spreading update mass across singular directions, but local update geometry and longer-horizon optimizer performance are distinct claims.

## Remaining Gap

The next clean mechanism test is a broader retuned longer-horizon trajectory intervention that separates optimizer identity, update geometry, accumulated optimizer state, and task-family scale effects beyond these fresh-continuation sweeps.
"""
    write_markdown(OUTPUT_PATH, text)
    print(f"saved mechanism ladder to {OUTPUT_PATH}")
    print(f"summary: {SUMMARY_PATH}")
    print(f"figure: {figure}")


if __name__ == "__main__":
    main()
