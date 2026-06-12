from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.reporting import fmt, markdown_table, write_markdown


OUTPUT_DIR = Path("results/e11_boundary_predictor")
DISCUSSION_PATH = Path("discussion/e11_boundary_predictor.md")

INDEX_COLUMNS = ["problem_family", "base_setting", "target_relative_update_norm", "seed", "step"]
TARGETS = ["update_grad_inner_muon_higher", "delta_loss_muon_higher"]

FEATURE_SETS = {
    "family_only": [],
    "state_only": [
        "log10_kappa",
        "log10_target_relative_update_norm",
        "loss_Adam",
        "recovery_error_Adam",
        "nrG_Adam",
        "stA_Adam",
        "condition_score_Adam",
    ],
    "update_spectrum_only": [
        "log10_target_relative_update_norm",
        "nrUpdateFrac_Adam",
        "nrUpdateFrac_Muon",
        "stUpdateFrac_Adam",
        "stUpdateFrac_Muon",
        "update_flatness_Adam",
        "update_flatness_Muon",
        "delta_stUpdateFrac",
        "delta_update_flatness",
    ],
    "state_plus_update_spectrum": [
        "log10_kappa",
        "log10_target_relative_update_norm",
        "loss_Adam",
        "recovery_error_Adam",
        "nrG_Adam",
        "stA_Adam",
        "condition_score_Adam",
        "nrUpdateFrac_Adam",
        "nrUpdateFrac_Muon",
        "stUpdateFrac_Adam",
        "stUpdateFrac_Muon",
        "update_flatness_Adam",
        "update_flatness_Muon",
        "delta_stUpdateFrac",
        "delta_update_flatness",
    ],
}


def binary_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(probabilities)
    y_true = y_true[valid].astype(int)
    probabilities = probabilities[valid]
    if y_true.size == 0:
        return {
            "auc": math.nan,
            "accuracy": math.nan,
            "balanced_accuracy": math.nan,
            "brier": math.nan,
            "baseline_brier": math.nan,
        }
    predictions = probabilities >= 0.5
    accuracy = float((predictions == y_true).mean())
    labels = sorted(set(y_true.tolist()))
    if len(labels) == 2:
        from sklearn.metrics import roc_auc_score

        auc = float(roc_auc_score(y_true, probabilities))
        recalls = []
        for label in [0, 1]:
            mask = y_true == label
            recalls.append(float((predictions[mask] == label).mean()))
        balanced_accuracy = float(np.mean(recalls))
    else:
        auc = math.nan
        balanced_accuracy = math.nan
    brier = float(np.mean((probabilities - y_true) ** 2))
    baseline = float(y_true.mean())
    baseline_brier = float(np.mean((baseline - y_true) ** 2))
    return {
        "auc": auc,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "brier": brier,
        "baseline_brier": baseline_brier,
    }


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, *, target: str, feature_set: str) -> np.ndarray:
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numeric_features = FEATURE_SETS[feature_set]
    categorical_features = ["problem_family"] if feature_set in {"family_only", "state_plus_update_spectrum"} else []
    transformers = []
    if numeric_features:
        transformers.append(("numeric", StandardScaler(), numeric_features))
    if categorical_features:
        transformers.append(("family", OneHotEncoder(handle_unknown="ignore"), categorical_features))
    if not transformers:
        return np.full(len(test), float(train[target].mean()))
    if train[target].nunique() < 2:
        return np.full(len(test), float(train[target].mean()))
    model = Pipeline(
        steps=[
            ("features", ColumnTransformer(transformers=transformers, remainder="drop")),
            ("logistic", LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")),
        ]
    )
    model.fit(train, train[target].astype(int))
    return model.predict_proba(test)[:, 1]


def build_frame() -> pd.DataFrame:
    paired = pd.read_csv("results/e11_target_update_sweep/paired_step_metrics.csv")
    steps = pd.read_csv("results/e11_target_update_sweep/step_metrics.csv")
    feature_columns = [
        "nrG",
        "stA",
        "condition_score",
        "nrUpdateFrac",
        "stUpdateFrac",
        "update_flatness",
    ]
    feature_frame = steps.pivot_table(
        index=INDEX_COLUMNS + ["kappa"],
        columns="algo",
        values=feature_columns,
        aggfunc="mean",
    )
    feature_frame.columns = [f"{feature}_{algo}" for feature, algo in feature_frame.columns]
    feature_frame = feature_frame.reset_index()
    feature_frame["log10_kappa"] = np.log10(feature_frame["kappa"].astype(float).clip(lower=1e-300))
    feature_frame["log10_target_relative_update_norm"] = np.log10(
        feature_frame["target_relative_update_norm"].astype(float).clip(lower=1e-300)
    )
    feature_frame["delta_stUpdateFrac"] = feature_frame["stUpdateFrac_Muon"] - feature_frame["stUpdateFrac_Adam"]
    feature_frame["delta_update_flatness"] = feature_frame["update_flatness_Muon"] - feature_frame["update_flatness_Adam"]
    frame = paired.merge(feature_frame, on=INDEX_COLUMNS, how="inner").replace([np.inf, -np.inf], np.nan)
    required = TARGETS + sorted({feature for features in FEATURE_SETS.values() for feature in features})
    return frame.dropna(subset=required).copy()


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    records = []
    for target in TARGETS:
        for feature_set in FEATURE_SETS:
            probabilities = fit_predict(frame, frame, target=target, feature_set=feature_set)
            records.append(
                {
                    "target": target,
                    "feature_set": feature_set,
                    "evaluation": "in_sample",
                    "held_out_setting": "None",
                    "train_rows": int(len(frame)),
                    "test_rows": int(len(frame)),
                    "test_positive_rate": float(frame[target].mean()),
                    **binary_metrics(frame[target].to_numpy(dtype=int), probabilities),
                }
            )
            for setting in sorted(frame["base_setting"].unique()):
                train = frame[frame["base_setting"] != setting].copy()
                test = frame[frame["base_setting"] == setting].copy()
                probabilities = fit_predict(train, test, target=target, feature_set=feature_set)
                records.append(
                    {
                        "target": target,
                        "feature_set": feature_set,
                        "evaluation": "leave_setting_out",
                        "held_out_setting": setting,
                        "train_rows": int(len(train)),
                        "test_rows": int(len(test)),
                        "test_positive_rate": float(test[target].mean()),
                        **binary_metrics(test[target].to_numpy(dtype=int), probabilities),
                    }
                )
    return pd.DataFrame(records)


def leave_setting_uncertainty(summary: pd.DataFrame, *, bootstrap_samples: int = 10000, seed: int = 0) -> pd.DataFrame:
    leave = summary[summary["evaluation"].eq("leave_setting_out")].copy()
    leave["balanced_accuracy_chance_filled"] = leave["balanced_accuracy"].fillna(0.5)
    leave["brier_improvement_over_base_rate"] = leave["baseline_brier"] - leave["brier"]
    rng = np.random.default_rng(seed)
    records = []
    for (target, feature_set), group in leave.groupby(["target", "feature_set"], sort=True):
        ba = group["balanced_accuracy_chance_filled"].to_numpy(dtype=float)
        brier_improvement = group["brier_improvement_over_base_rate"].to_numpy(dtype=float)
        ba_boot = np.array([rng.choice(ba, size=len(ba), replace=True).mean() for _ in range(bootstrap_samples)])
        brier_boot = np.array(
            [rng.choice(brier_improvement, size=len(brier_improvement), replace=True).mean() for _ in range(bootstrap_samples)]
        )
        records.append(
            {
                "target": target,
                "feature_set": feature_set,
                "evaluation": "leave_setting_out",
                "held_out_settings": int(len(group)),
                "mean_balanced_accuracy_chance_filled": float(ba.mean()),
                "balanced_accuracy_ci95_low": float(np.quantile(ba_boot, 0.025)),
                "balanced_accuracy_ci95_high": float(np.quantile(ba_boot, 0.975)),
                "balanced_accuracy_ci95_above_chance": bool(np.quantile(ba_boot, 0.025) > 0.5),
                "mean_brier_improvement_over_base_rate": float(brier_improvement.mean()),
                "brier_improvement_ci95_low": float(np.quantile(brier_boot, 0.025)),
                "brier_improvement_ci95_high": float(np.quantile(brier_boot, 0.975)),
                "brier_improvement_ci95_above_zero": bool(np.quantile(brier_boot, 0.025) > 0.0),
            }
        )
    return pd.DataFrame(records)


def write_discussion(summary: pd.DataFrame, uncertainty: pd.DataFrame, frame: pd.DataFrame) -> None:
    focus = summary[
        (summary["target"] == "update_grad_inner_muon_higher")
        & (summary["evaluation"].isin(["in_sample", "leave_setting_out"]))
    ].copy()
    leave_setting = focus[focus["evaluation"] == "leave_setting_out"].copy()
    aggregate = leave_setting.groupby("feature_set", as_index=False).agg(
        mean_balanced_accuracy=("balanced_accuracy", "mean"),
        mean_auc=("auc", "mean"),
        mean_brier=("brier", "mean"),
        mean_baseline_brier=("baseline_brier", "mean"),
    )
    best = aggregate.sort_values("mean_balanced_accuracy", ascending=False).iloc[0]
    uncertainty_focus = uncertainty[uncertainty["target"] == "update_grad_inner_muon_higher"].copy()
    best_uncertainty = uncertainty_focus.sort_values("mean_balanced_accuracy_chance_filled", ascending=False).iloc[0]
    text = f"""# E11 Boundary Predictor

This generated analysis tests whether a pre-specified set of local features predicts when Muon has larger one-step positive descent proxy `<G,D>` than Adam in the target-update sweep.

## Setup

- Data source: `results/e11_target_update_sweep/paired_step_metrics.csv` plus matched step diagnostics.
- Target: `update_grad_inner_muon_higher`.
- Main evaluation: leave-one-base-setting-out logistic regression.
- Feature sets are fixed in the script before fitting: family-only, state-only, update-spectrum-only, and state-plus-update-spectrum.

## Result

Best leave-setting-out mean balanced accuracy is `{fmt(best['mean_balanced_accuracy'])}` from `{best['feature_set']}` when undefined degenerate settings are skipped. With undefined balanced accuracy filled by chance, the best mean is `{fmt(best_uncertainty['mean_balanced_accuracy_chance_filled'])}` with bootstrap CI `[{fmt(best_uncertainty['balanced_accuracy_ci95_low'])}, {fmt(best_uncertainty['balanced_accuracy_ci95_high'])}]` from `{best_uncertainty['feature_set']}`. This is an exploratory baseline, not yet a publishable predictive law.

## Aggregate Leave-Setting-Out Metrics

{markdown_table(aggregate, ["feature_set", "mean_balanced_accuracy", "mean_auc", "mean_brier", "mean_baseline_brier"])}

## Leave-Setting-Out Uncertainty

{markdown_table(uncertainty_focus, ["feature_set", "held_out_settings", "mean_balanced_accuracy_chance_filled", "balanced_accuracy_ci95_low", "balanced_accuracy_ci95_high", "balanced_accuracy_ci95_above_chance", "mean_brier_improvement_over_base_rate", "brier_improvement_ci95_low", "brier_improvement_ci95_high", "brier_improvement_ci95_above_zero"])}

## Per-Setting Results For One-Step First-Order Wins

{markdown_table(leave_setting, ["feature_set", "held_out_setting", "test_rows", "test_positive_rate", "auc", "balanced_accuracy", "brier", "baseline_brier"])}

## Interpretation

The current data can support a descriptive boundary map more strongly than a predictive boundary law. If a simple feature set fails leave-setting-out, the paper should keep the boundary claim descriptive. If the best feature set performs well, the next step is to pre-register that rule and test it on a new neural benchmark or new problem family.

## Sources

- [boundary predictor rows](../results/e11_boundary_predictor/boundary_predictor_rows.csv)
- [boundary predictor summary](../results/e11_boundary_predictor/boundary_predictor_summary.csv)
- [target-update paired steps](../results/e11_target_update_sweep/paired_step_metrics.csv)
"""
    write_markdown(DISCUSSION_PATH, text)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = build_frame()
    summary = summarize(frame)
    uncertainty = leave_setting_uncertainty(summary)
    frame.to_csv(OUTPUT_DIR / "boundary_predictor_rows.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "boundary_predictor_summary.csv", index=False)
    uncertainty.to_csv(OUTPUT_DIR / "boundary_predictor_uncertainty.csv", index=False)
    write_discussion(summary, uncertainty, frame)
    print(f"saved boundary predictor rows to {OUTPUT_DIR}")
    print(f"saved boundary predictor discussion to {DISCUSSION_PATH}")


if __name__ == "__main__":
    main()
