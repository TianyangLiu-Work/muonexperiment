from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.long_tail_practical_training import (
    LongTailPracticalTrainingLRSweepConfig,
    run_long_tail_practical_training_lr_sweep,
)


OUTPUT_DIR = Path("results/e11_long_tail_practical_training_lr_sweep")
FIGURE_DIR = Path("figures/e11_long_tail_practical_training_lr_sweep")
DISCUSSION_PATH = Path("discussion/e11_long_tail_practical_training_lr_sweep.md")


def _fmt(value: float) -> str:
    return f"{value:.4g}"


def write_figure(sweep: pd.DataFrame) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    ordered = sweep.sort_values("muon_lr")
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), sharex=True)
    panels = [
        ("geomean_final_train_loss_ratio_muon_over_adam", "train loss ratio"),
        ("geomean_final_tail_eval_loss_ratio_muon_over_adam", "tail eval loss ratio"),
        ("geomean_final_tail_eval_drift_rms_ratio_muon_over_adam", "tail drift RMS ratio"),
    ]
    for axis, (column, ylabel) in zip(axes, panels):
        axis.plot(ordered["muon_lr"], ordered[column], marker="o", color="#0072B2", linewidth=1.8)
        axis.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
        axis.axvline(0.03, color="#D55E00", linestyle=":", linewidth=1.2)
        axis.set_xscale("log")
        axis.set_xlabel("NS-Muon-style learning rate")
        axis.set_ylabel(ylabel)
    fig.suptitle("Practical training LR sensitivity")
    fig.tight_layout()
    path = FIGURE_DIR / "long_tail_practical_training_lr_sweep.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def write_discussion(config: LongTailPracticalTrainingLRSweepConfig, sweep: pd.DataFrame, figure_path: Path) -> None:
    rows = [
        "| muon lr | train loss ratio | head loss ratio | tail eval loss ratio | tail drift RMS ratio | tail acc diff |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in sweep.sort_values("muon_lr").iterrows():
        rows.append(
            "| "
            + f"{_fmt(row['muon_lr'])} | "
            + f"{_fmt(row['geomean_final_train_loss_ratio_muon_over_adam'])} "
            + f"[{_fmt(row['final_train_loss_ratio_ci95_low'])}, {_fmt(row['final_train_loss_ratio_ci95_high'])}] | "
            + f"{_fmt(row['geomean_final_head_loss_ratio_muon_over_adam'])} "
            + f"[{_fmt(row['final_head_loss_ratio_ci95_low'])}, {_fmt(row['final_head_loss_ratio_ci95_high'])}] | "
            + f"{_fmt(row['geomean_final_tail_eval_loss_ratio_muon_over_adam'])} "
            + f"[{_fmt(row['final_tail_eval_loss_ratio_ci95_low'])}, {_fmt(row['final_tail_eval_loss_ratio_ci95_high'])}] | "
            + f"{_fmt(row['geomean_final_tail_eval_drift_rms_ratio_muon_over_adam'])} "
            + f"[{_fmt(row['final_tail_eval_drift_rms_ratio_ci95_low'])}, {_fmt(row['final_tail_eval_drift_rms_ratio_ci95_high'])}] | "
            + f"{_fmt(row['mean_final_tail_eval_accuracy_diff_muon_minus_adam'])} "
            + f"[{_fmt(row['final_tail_eval_accuracy_diff_ci95_low'])}, {_fmt(row['final_tail_eval_accuracy_diff_ci95_high'])}] |"
        )

    best_train = sweep.loc[sweep["geomean_final_train_loss_ratio_muon_over_adam"].idxmin()]
    selected = sweep[sweep["muon_lr"].eq(config.base_config.muon_lr)].iloc[0]
    text = f"""# E11 Practical Training LR Sensitivity

This supplemental check tests whether the practical imbalanced-training result is
only a single learning-rate accident. It reruns the same paired Adam vs finite
Newton-Schulz Muon-style training diagnostic for several `muon_lr` values while
keeping Adam lr fixed at `{config.base_config.adam_lr}`.

{chr(10).join(rows)}

Figure: [{figure_path.as_posix()}](../{figure_path.as_posix()})

Interpretation:

- Very small `muon_lr` under-trains relative to Adam: train/head loss ratios stay above 1.
- The selected `muon_lr={config.base_config.muon_lr}` is the first tested scale where train/head loss, tail loss, and tail drift are all below Adam with paired confidence intervals below 1.
- Larger `muon_lr` can lower train/head loss further but increases tail loss and tail drift, so the practical benefit is not monotone in step size.

Current selected setting:

- train loss ratio: `{_fmt(selected['geomean_final_train_loss_ratio_muon_over_adam'])} [{_fmt(selected['final_train_loss_ratio_ci95_low'])}, {_fmt(selected['final_train_loss_ratio_ci95_high'])}]`
- tail eval loss ratio: `{_fmt(selected['geomean_final_tail_eval_loss_ratio_muon_over_adam'])} [{_fmt(selected['final_tail_eval_loss_ratio_ci95_low'])}, {_fmt(selected['final_tail_eval_loss_ratio_ci95_high'])}]`
- tail drift RMS ratio: `{_fmt(selected['geomean_final_tail_eval_drift_rms_ratio_muon_over_adam'])} [{_fmt(selected['final_tail_eval_drift_rms_ratio_ci95_low'])}, {_fmt(selected['final_tail_eval_drift_rms_ratio_ci95_high'])}]`

Best train-loss setting in this sweep:

- `muon_lr={_fmt(best_train['muon_lr'])}` with train loss ratio `{_fmt(best_train['geomean_final_train_loss_ratio_muon_over_adam'])}`.

Caveat:

This is still a small fixed-grid check on sklearn digits. It reduces the
cherry-picking risk for the selected practical-training setting, but it does not
replace a real long-tailed benchmark with retuned optimizers.

Artifacts:

- [sweep_summary.csv](../results/e11_long_tail_practical_training_lr_sweep/sweep_summary.csv)
- [config.json](../results/e11_long_tail_practical_training_lr_sweep/config.json)
"""
    DISCUSSION_PATH.write_text(text, encoding="utf-8")


def write_outputs(config: LongTailPracticalTrainingLRSweepConfig, sweep: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(OUTPUT_DIR / "sweep_summary.csv", index=False)
    (OUTPUT_DIR / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    figure_path = write_figure(sweep)
    write_discussion(config, sweep, figure_path)
    print(f"saved practical training LR sweep results to {OUTPUT_DIR}")
    print(f"rows={len(sweep)}, figure={figure_path}, discussion={DISCUSSION_PATH}")
    print(sweep.to_string(index=False))


def main() -> None:
    config = LongTailPracticalTrainingLRSweepConfig()
    sweep = run_long_tail_practical_training_lr_sweep(config)
    write_outputs(config, sweep)


if __name__ == "__main__":
    main()
