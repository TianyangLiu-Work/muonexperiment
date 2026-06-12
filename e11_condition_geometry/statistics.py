from __future__ import annotations

import math

import numpy as np
import pandas as pd


def ci95(values: pd.Series) -> tuple[float, float, float]:
    arr = values.to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return math.nan, math.nan, math.nan
    mean = float(arr.mean())
    if arr.size == 1:
        return mean, mean, mean
    multiplier = 2.776 if arr.size <= 5 else 1.96
    half_width = multiplier * float(arr.std(ddof=1)) / math.sqrt(arr.size)
    return mean, mean - half_width, mean + half_width


def _t_critical(n: int) -> float:
    if n <= 1:
        return 0.0
    try:
        from scipy.stats import t

        return float(t.ppf(0.975, n - 1))
    except Exception:
        return 2.776 if n <= 5 else 1.96


def log_ratio_ci95(ratios: pd.Series) -> tuple[float, float, float]:
    arr = ratios.to_numpy(dtype=float)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if arr.size == 0:
        return math.nan, math.nan, math.nan
    logs = np.log(arr)
    mean = float(logs.mean())
    if logs.size == 1:
        return math.exp(mean), math.exp(mean), math.exp(mean)
    half_width = _t_critical(logs.size) * float(logs.std(ddof=1)) / math.sqrt(logs.size)
    return math.exp(mean), math.exp(mean - half_width), math.exp(mean + half_width)


def corr_ci95(x: pd.Series, y: pd.Series, *, method: str = "spearman") -> tuple[float, float, float, int]:
    frame = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 4:
        return math.nan, math.nan, math.nan, int(len(frame))
    x_values = frame["x"].to_numpy(dtype=float)
    y_values = frame["y"].to_numpy(dtype=float)
    if np.nanmax(x_values) - np.nanmin(x_values) <= 1e-10 or np.nanmax(y_values) - np.nanmin(y_values) <= 1e-14:
        return math.nan, math.nan, math.nan, int(len(frame))
    if method == "spearman":
        frame = frame.rank(method="average")
    if len(frame) < 4 or frame["x"].nunique() < 2 or frame["y"].nunique() < 2:
        return math.nan, math.nan, math.nan, int(len(frame))
    r = float(frame["x"].corr(frame["y"], method="pearson"))
    r = float(np.clip(r, -0.999999, 0.999999))
    z = math.atanh(r)
    half_width = _t_critical(len(frame) - 2) / math.sqrt(len(frame) - 3)
    return r, math.tanh(z - half_width), math.tanh(z + half_width), int(len(frame))


def final_performance_summary(steps: pd.DataFrame) -> pd.DataFrame:
    final = steps.sort_values("step").groupby("run_id", observed=True).tail(1)
    update = steps[np.isfinite(steps["relative_update_fro_norm"])].groupby(
        ["problem_family", "setting", "kappa", "lr", "algo"], as_index=False, observed=True
    ).agg(
        median_relative_update_fro_norm=("relative_update_fro_norm", "median"),
        median_mean_relative_layer_update_norm=("mean_relative_layer_update_norm", "median"),
    )
    delta = steps[np.isfinite(steps["delta_loss"])].groupby(
        ["problem_family", "setting", "kappa", "lr", "algo"], as_index=False, observed=True
    ).agg(median_delta_loss=("delta_loss", "median"))
    summary = final.groupby(["problem_family", "setting", "kappa", "lr", "algo"], as_index=False, observed=True).agg(
        runs=("run_id", "nunique"),
        median_final_loss=("loss", "median"),
        median_recovery=("recovery_error", "median"),
        median_nrG=("nrG", "median"),
        median_stA=("stA", "median"),
        median_condition_score=("condition_score", "median"),
    )
    return summary.merge(update, on=["problem_family", "setting", "kappa", "lr", "algo"], how="left").merge(
        delta, on=["problem_family", "setting", "kappa", "lr", "algo"], how="left"
    )


def prediction_summary(steps: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (family, setting, kappa, lr, algo), group in steps[np.isfinite(steps["delta_loss"])].groupby(
        ["problem_family", "setting", "kappa", "lr", "algo"], observed=True, sort=False
    ):
        for predictor in ["delta_gd_pred", "delta_spec_pred", "update_grad_inner"]:
            finite = group[[predictor, "delta_loss"]].replace([np.inf, -np.inf], np.nan).dropna()
            positive = finite[finite["delta_loss"] > 0]
            spearman = finite[predictor].corr(finite["delta_loss"], method="spearman") if len(finite) > 2 else np.nan
            if len(positive) > 2:
                pearson_log_positive = np.corrcoef(
                    np.log10(positive[predictor].clip(lower=1e-300)),
                    np.log10(positive["delta_loss"].clip(lower=1e-300)),
                )[0, 1]
            else:
                pearson_log_positive = np.nan
            rows.append(
                {
                    "problem_family": family,
                    "setting": setting,
                    "kappa": float(kappa),
                    "lr": float(lr),
                    "algo": algo,
                    "predictor": predictor,
                    "points": int(len(finite)),
                    "positive_delta_points": int(len(positive)),
                    "spearman_all_delta_loss": float(spearman) if np.isfinite(spearman) else np.nan,
                    "pearson_log_positive_delta_loss": float(pearson_log_positive) if np.isfinite(pearson_log_positive) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def nearest_centroid_balanced_accuracy(group: pd.DataFrame) -> float:
    rows = group[["algo", "nrG", "stA"]].dropna().copy()
    if set(rows["algo"].astype(str)) != {"Adam", "Muon"}:
        return np.nan
    x = rows[["nrG", "stA"]].to_numpy(dtype=float)
    scale = x.std(axis=0)
    scale[scale == 0] = 1.0
    x = (x - x.mean(axis=0)) / scale
    algos = rows["algo"].astype(str).to_numpy()
    centroids = {algo: x[algos == algo].mean(axis=0) for algo in ["Adam", "Muon"]}
    distances = np.column_stack([np.linalg.norm(x - centroids[algo], axis=1) for algo in ["Adam", "Muon"]])
    pred = np.array(["Adam", "Muon"])[np.argmin(distances, axis=1)]
    return float(np.mean([np.mean(pred[algos == algo] == algo) for algo in ["Adam", "Muon"]]))


def standardized_centroid_distance(group: pd.DataFrame) -> float:
    rows = group[["algo", "nrG", "stA"]].dropna().copy()
    if set(rows["algo"].astype(str)) != {"Adam", "Muon"}:
        return np.nan
    x = rows[["nrG", "stA"]].to_numpy(dtype=float)
    scale = x.std(axis=0)
    scale[scale == 0] = 1.0
    x = (x - x.mean(axis=0)) / scale
    algos = rows["algo"].astype(str).to_numpy()
    return float(np.linalg.norm(x[algos == "Muon"].mean(axis=0) - x[algos == "Adam"].mean(axis=0)))


def geometry_summary(steps: pd.DataFrame) -> pd.DataFrame:
    source = steps[steps["algo"].isin(["Adam", "Muon"])].copy()
    pivot = source.pivot_table(
        index=["problem_family", "setting", "kappa", "lr", "seed", "step"],
        columns="algo",
        values=["nrG", "stA", "loss", "recovery_error"],
        aggfunc="mean",
    )
    pivot.columns = [f"{metric}_{algo}" for metric, algo in pivot.columns]
    paired = pivot.reset_index().dropna().copy()
    paired["delta_nrG"] = paired["nrG_Muon"] - paired["nrG_Adam"]
    paired["delta_stA"] = paired["stA_Muon"] - paired["stA_Adam"]
    paired["delta_loss"] = paired["loss_Muon"] - paired["loss_Adam"]
    paired["delta_recovery"] = paired["recovery_error_Muon"] - paired["recovery_error_Adam"]
    seed_summary = paired.groupby(["problem_family", "setting", "kappa", "lr", "seed"], observed=True).agg(
        mean_delta_nrG=("delta_nrG", "mean"),
        mean_delta_stA=("delta_stA", "mean"),
        mean_delta_loss=("delta_loss", "mean"),
        mean_delta_recovery=("delta_recovery", "mean"),
    ).reset_index()

    records = []
    for (family, setting, kappa, lr), group in seed_summary.groupby(["problem_family", "setting", "kappa", "lr"], observed=True, sort=False):
        point_rows = source[(source["problem_family"] == family) & (source["setting"] == setting)]
        nr_mean, nr_lo, nr_hi = ci95(group["mean_delta_nrG"])
        st_mean, st_lo, st_hi = ci95(group["mean_delta_stA"])
        loss_mean, loss_lo, loss_hi = ci95(group["mean_delta_loss"])
        recovery_mean, recovery_lo, recovery_hi = ci95(group["mean_delta_recovery"])
        records.append(
            {
                "problem_family": family,
                "setting": setting,
                "kappa": float(kappa),
                "lr": float(lr),
                "seed_count": int(group["seed"].nunique()),
                "mean_delta_nrG": nr_mean,
                "delta_nrG_ci95_low": nr_lo,
                "delta_nrG_ci95_high": nr_hi,
                "nrG_muon_higher_seed_count": int((group["mean_delta_nrG"] > 0).sum()),
                "mean_delta_stA": st_mean,
                "delta_stA_ci95_low": st_lo,
                "delta_stA_ci95_high": st_hi,
                "stA_muon_higher_seed_count": int((group["mean_delta_stA"] > 0).sum()),
                "stA_ci95_excludes_zero": bool((st_lo > 0) or (st_hi < 0)),
                "mean_delta_loss": loss_mean,
                "delta_loss_ci95_low": loss_lo,
                "delta_loss_ci95_high": loss_hi,
                "mean_delta_recovery": recovery_mean,
                "delta_recovery_ci95_low": recovery_lo,
                "delta_recovery_ci95_high": recovery_hi,
                "centroid_distance_std": standardized_centroid_distance(point_rows),
                "nearest_centroid_balanced_accuracy": nearest_centroid_balanced_accuracy(point_rows),
            }
        )
    result = pd.DataFrame(records)
    if not result.empty:
        result["delta_stA_95ci"] = result.apply(
            lambda row: f"[{row['delta_stA_ci95_low']:.4g}, {row['delta_stA_ci95_high']:.4g}]",
            axis=1,
        )
    return result


def run_dynamics_summary(steps: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for run_id, group in steps.sort_values(["run_id", "step"]).groupby("run_id", observed=True):
        rec = {
            "run_id": int(run_id),
            "problem_family": group["problem_family"].iloc[0],
            "setting": group["setting"].iloc[0],
            "kappa": float(group["kappa"].iloc[0]),
            "lr": float(group["lr"].iloc[0]),
            "algo": group["algo"].iloc[0],
            "seed": int(group["seed"].iloc[0]),
            "steps": int(group["step"].max()),
        }
        rec["median_relative_update_fro_norm"] = float(np.nanmedian(group["relative_update_fro_norm"].to_numpy(dtype=float)))
        rec["median_mean_relative_layer_update_norm"] = float(
            np.nanmedian(group["mean_relative_layer_update_norm"].to_numpy(dtype=float))
        )
        for metric in [
            "nrG",
            "stA",
            "condition_score",
            "delta_gd_pred",
            "delta_spec_pred",
            "loss",
            "relative_update_fro_norm",
            "mean_relative_layer_update_norm",
        ]:
            values = group[metric].to_numpy(dtype=float)
            diffs = np.diff(values)
            if diffs.size:
                rec[f"{metric}_mean_abs_speed"] = float(np.nanmean(np.abs(diffs)))
                rec[f"{metric}_std_speed"] = float(np.nanstd(diffs))
                rec[f"{metric}_mean_rel_speed"] = float(np.nanmean(np.abs(diffs) / (np.abs(values[:-1]) + 1e-300)))
                rec[f"{metric}_path_length"] = float(np.nansum(np.abs(diffs)))
            else:
                rec[f"{metric}_mean_abs_speed"] = math.nan
                rec[f"{metric}_std_speed"] = math.nan
                rec[f"{metric}_mean_rel_speed"] = math.nan
                rec[f"{metric}_path_length"] = math.nan

        nr = group["nrG"].to_numpy(dtype=float)
        st = group["stA"].to_numpy(dtype=float)
        rank_steps = np.sqrt(np.diff(nr) ** 2 + np.diff(st) ** 2)
        rec["rank_plane_mean_speed"] = float(np.nanmean(rank_steps))
        rec["rank_plane_path_length"] = float(np.nansum(rank_steps))
        rec["rank_plane_net_distance"] = float(np.sqrt((nr[-1] - nr[0]) ** 2 + (st[-1] - st[0]) ** 2))
        rec["rank_plane_tortuosity"] = rec["rank_plane_path_length"] / max(rec["rank_plane_net_distance"], 1e-300)
        rows.append(rec)

    result = pd.DataFrame(rows)
    normalized_rows = []
    for (family, setting), setting_rows in steps.groupby(["problem_family", "setting"], observed=True):
        scale_nr = float(setting_rows["nrG"].std()) or 1.0
        scale_st = float(setting_rows["stA"].std()) or 1.0
        scale_c = float(setting_rows["condition_score"].std()) or 1.0
        for run_id, group in setting_rows.sort_values(["run_id", "step"]).groupby("run_id", observed=True):
            nr = group["nrG"].to_numpy(dtype=float) / max(scale_nr, 1e-300)
            st = group["stA"].to_numpy(dtype=float) / max(scale_st, 1e-300)
            c = group["condition_score"].to_numpy(dtype=float) / max(scale_c, 1e-300)
            updates = group["relative_update_fro_norm"].to_numpy(dtype=float)[:-1]
            rank_step_speed = np.sqrt(np.diff(nr) ** 2 + np.diff(st) ** 2)
            condition_step_speed = np.abs(np.diff(c))
            normalized_rows.append(
                {
                    "run_id": int(run_id),
                    "norm_rank_plane_mean_speed": float(np.nanmean(rank_step_speed)),
                    "norm_rank_plane_path_length": float(np.nansum(rank_step_speed)),
                    "norm_condition_mean_speed": float(np.nanmean(condition_step_speed)),
                    "norm_condition_path_length": float(np.nansum(condition_step_speed)),
                    "per_update_rank_plane_mean_speed": float(np.nanmean(rank_step_speed / (np.abs(updates) + 1e-300))),
                    "per_update_condition_mean_speed": float(np.nanmean(condition_step_speed / (np.abs(updates) + 1e-300))),
                }
            )
    result = result.merge(pd.DataFrame(normalized_rows), on="run_id", how="left")
    result["update_normalized_rank_plane_mean_speed"] = result["norm_rank_plane_mean_speed"] / (
        result["median_relative_update_fro_norm"].abs() + 1e-300
    )
    result["update_normalized_condition_mean_speed"] = result["norm_condition_mean_speed"] / (
        result["median_relative_update_fro_norm"].abs() + 1e-300
    )
    return result


def volatility_summary(run_dynamics: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "norm_rank_plane_mean_speed",
        "norm_condition_mean_speed",
        "condition_score_mean_abs_speed",
        "condition_score_std_speed",
        "delta_gd_pred_mean_rel_speed",
        "delta_spec_pred_mean_rel_speed",
        "loss_mean_rel_speed",
        "median_relative_update_fro_norm",
        "median_mean_relative_layer_update_norm",
        "relative_update_fro_norm_mean_abs_speed",
        "mean_relative_layer_update_norm_mean_abs_speed",
        "update_normalized_rank_plane_mean_speed",
        "update_normalized_condition_mean_speed",
        "per_update_rank_plane_mean_speed",
        "per_update_condition_mean_speed",
    ]
    pivot = run_dynamics.pivot_table(
        index=["problem_family", "setting", "kappa", "lr", "seed"],
        columns="algo",
        values=[metric for metric in metrics if metric in run_dynamics.columns],
        aggfunc="mean",
    )
    pivot.columns = [f"{metric}_{algo}" for metric, algo in pivot.columns]
    paired = pivot.reset_index().dropna().copy()
    records = []
    groupers = [("All", paired)]
    groupers.extend(list(paired.groupby("problem_family", observed=True, sort=False)))
    for metric in metrics:
        muon_col = f"{metric}_Muon"
        adam_col = f"{metric}_Adam"
        if muon_col not in paired or adam_col not in paired:
            continue
        for family, group in groupers:
            ratios = group[muon_col] / (group[adam_col].abs() + 1e-300)
            deltas = group[muon_col] - group[adam_col]
            ratio_mean, ratio_lo, ratio_hi = log_ratio_ci95(ratios)
            delta_mean, delta_lo, delta_hi = ci95(deltas)
            records.append(
                {
                    "metric": metric,
                    "problem_family": family,
                    "n_pairs": int(len(group)),
                    "muon_lower_pairs": int((deltas < 0).sum()),
                    "muon_higher_pairs": int((deltas > 0).sum()),
                    "geomean_ratio_muon_over_adam": ratio_mean,
                    "ratio_ci95_low": ratio_lo,
                    "ratio_ci95_high": ratio_hi,
                    "mean_delta_muon_minus_adam": delta_mean,
                    "delta_ci95_low": delta_lo,
                    "delta_ci95_high": delta_hi,
                    "ratio_ci95_below_one": bool(ratio_hi < 1.0) if np.isfinite(ratio_hi) else False,
                    "delta_ci95_below_zero": bool(delta_hi < 0.0) if np.isfinite(delta_hi) else False,
                }
            )
    return pd.DataFrame(records)


def update_spectrum_summary(steps: pd.DataFrame) -> pd.DataFrame:
    metrics = ["nrUpdate", "stUpdate", "nrUpdateFrac", "stUpdateFrac", "update_flatness"]
    source = steps[np.isfinite(steps["relative_update_fro_norm"])].copy()
    run_level = source.groupby(
        ["problem_family", "setting", "kappa", "lr", "seed", "algo"],
        as_index=False,
        observed=True,
    ).agg({metric: "mean" for metric in metrics})
    pivot = run_level.pivot_table(
        index=["problem_family", "setting", "kappa", "lr", "seed"],
        columns="algo",
        values=metrics,
        aggfunc="mean",
    )
    pivot.columns = [f"{metric}_{algo}" for metric, algo in pivot.columns]
    paired = pivot.reset_index().dropna().copy()
    records = []
    groupers = [("All", paired)]
    groupers.extend(list(paired.groupby("problem_family", observed=True, sort=False)))
    for metric in metrics:
        muon_col = f"{metric}_Muon"
        adam_col = f"{metric}_Adam"
        for family, group in groupers:
            ratios = group[muon_col] / (group[adam_col].abs() + 1e-300)
            deltas = group[muon_col] - group[adam_col]
            ratio_mean, ratio_lo, ratio_hi = log_ratio_ci95(ratios)
            delta_mean, delta_lo, delta_hi = ci95(deltas)
            records.append(
                {
                    "metric": metric,
                    "problem_family": family,
                    "n_pairs": int(len(group)),
                    "muon_higher_pairs": int((deltas > 0).sum()),
                    "muon_lower_pairs": int((deltas < 0).sum()),
                    "geomean_ratio_muon_over_adam": ratio_mean,
                    "ratio_ci95_low": ratio_lo,
                    "ratio_ci95_high": ratio_hi,
                    "mean_delta_muon_minus_adam": delta_mean,
                    "delta_ci95_low": delta_lo,
                    "delta_ci95_high": delta_hi,
                    "ratio_ci95_above_one": bool(ratio_lo > 1.0) if np.isfinite(ratio_lo) else False,
                    "delta_ci95_above_zero": bool(delta_lo > 0.0) if np.isfinite(delta_lo) else False,
                }
            )
    return pd.DataFrame(records)


def update_transmission_summary(steps: pd.DataFrame) -> pd.DataFrame:
    rows = []
    source = steps.sort_values(["run_id", "step"]).copy()
    for _, group in source.groupby("run_id", observed=True, sort=False):
        for column in ["nrG", "stA", "condition_score"]:
            source.loc[group.index, f"next_{column}"] = group[column].shift(-1)

    enriched = []
    for (_, setting), group in source.groupby(["problem_family", "setting"], observed=True):
        group = group.copy()
        scale_nr = float(group["nrG"].std()) or 1.0
        scale_st = float(group["stA"].std()) or 1.0
        scale_c = float(group["condition_score"].std()) or 1.0
        group["norm_rank_movement"] = np.sqrt(
            ((group["next_nrG"] - group["nrG"]) / max(scale_nr, 1e-300)) ** 2
            + ((group["next_stA"] - group["stA"]) / max(scale_st, 1e-300)) ** 2
        )
        group["norm_condition_movement"] = np.abs(group["next_condition_score"] - group["condition_score"]) / max(scale_c, 1e-300)
        enriched.append(group)
    frame = pd.concat(enriched, ignore_index=True)
    frame = frame[np.isfinite(frame["delta_loss"])].copy()
    frame["relative_loss_decrease"] = frame["delta_loss"] / (frame["loss"].abs() + 1e-300)

    predictors = [
        "nrUpdateFrac",
        "stUpdateFrac",
        "update_flatness",
        "relative_update_fro_norm",
        "update_grad_inner",
        "update_grad_cosine",
        "update_grad_per_update_norm",
    ]
    outcomes = ["delta_loss", "relative_loss_decrease", "norm_rank_movement", "norm_condition_movement"]
    groupers = [("All", "All", frame)]
    groupers.extend(("family", family, group) for family, group in frame.groupby("problem_family", observed=True, sort=False))
    groupers.extend(("algo", algo, group) for algo, group in frame.groupby("algo", observed=True, sort=False))
    groupers.extend(
        (f"{family}:{algo}", f"{family}:{algo}", group)
        for (family, algo), group in frame.groupby(["problem_family", "algo"], observed=True, sort=False)
    )
    for group_type, group_name, group in groupers:
        for predictor in predictors:
            for outcome in outcomes:
                corr, lo, hi, n = corr_ci95(group[predictor], group[outcome], method="spearman")
                rows.append(
                    {
                        "group_type": group_type,
                        "group": group_name,
                        "predictor": predictor,
                        "outcome": outcome,
                        "points": n,
                        "spearman": corr,
                        "spearman_ci95_low": lo,
                        "spearman_ci95_high": hi,
                        "ci95_excludes_zero": bool((lo > 0.0) or (hi < 0.0)) if np.isfinite(lo) and np.isfinite(hi) else False,
                    }
                )
    return pd.DataFrame(rows)


def first_order_calibration_summary(steps: pd.DataFrame) -> pd.DataFrame:
    frame = steps[np.isfinite(steps["delta_loss"])].replace([np.inf, -np.inf], np.nan).dropna(
        subset=["delta_loss", "update_grad_inner"]
    )
    groupers = [("All", "All", frame)]
    groupers.extend(("family", family, group) for family, group in frame.groupby("problem_family", observed=True, sort=False))
    groupers.extend(("algo", algo, group) for algo, group in frame.groupby("algo", observed=True, sort=False))
    groupers.extend(
        (f"{family}:{algo}", f"{family}:{algo}", group)
        for (family, algo), group in frame.groupby(["problem_family", "algo"], observed=True, sort=False)
    )
    rows = []
    for group_type, group_name, group in groupers:
        finite = group[["delta_loss", "update_grad_inner"]].replace([np.inf, -np.inf], np.nan).dropna()
        positive = finite[(finite["delta_loss"] > 0) & (finite["update_grad_inner"] > 0)].copy()
        spearman, spearman_lo, spearman_hi, points = corr_ci95(
            finite["update_grad_inner"], finite["delta_loss"], method="spearman"
        )
        if len(positive) >= 4:
            log_pred = np.log10(positive["update_grad_inner"].to_numpy(dtype=float))
            log_obs = np.log10(positive["delta_loss"].to_numpy(dtype=float))
            pearson = float(np.corrcoef(log_pred, log_obs)[0, 1])
            ratio = positive["delta_loss"] / positive["update_grad_inner"]
            ratio_mean, ratio_lo, ratio_hi = log_ratio_ci95(ratio)
            log_abs_error = np.abs(np.log10(ratio.to_numpy(dtype=float)))
            within_factor_2 = float((log_abs_error <= math.log10(2.0)).mean())
            median_abs_log10_ratio = float(np.median(log_abs_error))
            x = positive["update_grad_inner"].to_numpy(dtype=float)
            y = positive["delta_loss"].to_numpy(dtype=float)
            slope_through_origin = float(np.sum(x * y) / max(np.sum(x * x), 1e-300))
        else:
            pearson = math.nan
            ratio_mean = ratio_lo = ratio_hi = math.nan
            within_factor_2 = math.nan
            median_abs_log10_ratio = math.nan
            slope_through_origin = math.nan
        rows.append(
            {
                "group_type": group_type,
                "group": group_name,
                "points": points,
                "positive_points": int(len(positive)),
                "spearman_delta_vs_first_order": spearman,
                "spearman_ci95_low": spearman_lo,
                "spearman_ci95_high": spearman_hi,
                "pearson_log_positive": pearson,
                "geomean_observed_over_first_order": ratio_mean,
                "ratio_ci95_low": ratio_lo,
                "ratio_ci95_high": ratio_hi,
                "median_abs_log10_ratio": median_abs_log10_ratio,
                "within_factor_2": within_factor_2,
                "slope_through_origin": slope_through_origin,
            }
        )
    return pd.DataFrame(rows)


def polar_alignment_summary(layers: pd.DataFrame) -> pd.DataFrame:
    source = layers[
        (layers["algo"] == "Muon")
        & np.isfinite(layers["update_grad_cosine"])
        & np.isfinite(layers["nrG"])
        & np.isfinite(layers["update_rank_ceiling"])
        & (layers["update_rank_ceiling"] > 0)
    ].copy()
    source["cosine_sq"] = source["update_grad_cosine"] ** 2
    source["gradient_rank_fraction"] = source["nrG"] / source["update_rank_ceiling"]
    source["identity_abs_error"] = (source["cosine_sq"] - source["gradient_rank_fraction"]).abs()
    rows = []
    groupers = [("All", "All", source)]
    groupers.extend(("family", family, group) for family, group in source.groupby("problem_family", observed=True, sort=False))
    for group_type, group_name, group in groupers:
        rows.append(
            {
                "group_type": group_type,
                "group": group_name,
                "points": int(len(group)),
                "median_cosine_sq": float(group["cosine_sq"].median()),
                "median_gradient_rank_fraction": float(group["gradient_rank_fraction"].median()),
                "median_abs_identity_error": float(group["identity_abs_error"].median()),
                "max_abs_identity_error": float(group["identity_abs_error"].max()),
                "mean_abs_identity_error": float(group["identity_abs_error"].mean()),
            }
        )
    return pd.DataFrame(rows)


def first_order_pair_summary(steps: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "delta_loss",
        "update_grad_inner",
        "update_grad_cosine",
        "update_grad_per_update_norm",
        "relative_update_fro_norm",
    ]
    source = steps[np.isfinite(steps["delta_loss"])].copy()
    pivot = source.pivot_table(
        index=["problem_family", "setting", "kappa", "lr", "seed", "step"],
        columns="algo",
        values=metrics,
        aggfunc="mean",
    )
    pivot.columns = [f"{metric}_{algo}" for metric, algo in pivot.columns]
    paired = pivot.reset_index().dropna().copy()
    records = []
    groupers = [("All", paired)]
    groupers.extend(list(paired.groupby("problem_family", observed=True, sort=False)))
    for metric in metrics:
        muon_col = f"{metric}_Muon"
        adam_col = f"{metric}_Adam"
        for family, group in groupers:
            ratios = group[muon_col] / (group[adam_col].abs() + 1e-300)
            deltas = group[muon_col] - group[adam_col]
            ratio_mean, ratio_lo, ratio_hi = log_ratio_ci95(ratios)
            delta_mean, delta_lo, delta_hi = ci95(deltas)
            records.append(
                {
                    "metric": metric,
                    "problem_family": family,
                    "n_pairs": int(len(group)),
                    "muon_higher_pairs": int((deltas > 0).sum()),
                    "muon_lower_pairs": int((deltas < 0).sum()),
                    "geomean_ratio_muon_over_adam": ratio_mean,
                    "ratio_ci95_low": ratio_lo,
                    "ratio_ci95_high": ratio_hi,
                    "mean_delta_muon_minus_adam": delta_mean,
                    "delta_ci95_low": delta_lo,
                    "delta_ci95_high": delta_hi,
                    "ratio_ci95_above_one": bool(ratio_lo > 1.0) if np.isfinite(ratio_lo) else False,
                    "delta_ci95_above_zero": bool(delta_lo > 0.0) if np.isfinite(delta_lo) else False,
                }
            )
    return pd.DataFrame(records)


def _paired_step_metrics(steps: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "delta_loss",
        "loss",
        "nrG",
        "stA",
        "condition_score",
        "delta_gd_pred",
        "delta_spec_pred",
        "update_grad_inner",
        "update_grad_cosine",
        "update_grad_per_update_norm",
        "relative_update_fro_norm",
    ]
    source = steps[np.isfinite(steps["delta_loss"])].copy()
    pivot = source.pivot_table(
        index=["problem_family", "setting", "kappa", "lr", "seed", "step"],
        columns="algo",
        values=metrics,
        aggfunc="mean",
    )
    pivot.columns = [f"{metric}_{algo}" for metric, algo in pivot.columns]
    paired = pivot.reset_index().dropna().copy()
    paired["muon_delta_loss_win"] = paired["delta_loss_Muon"] > paired["delta_loss_Adam"]
    paired["muon_first_order_win"] = paired["update_grad_inner_Muon"] > paired["update_grad_inner_Adam"]
    paired["delta_loss_ratio"] = paired["delta_loss_Muon"] / (paired["delta_loss_Adam"].abs() + 1e-300)
    paired["first_order_ratio"] = paired["update_grad_inner_Muon"] / (paired["update_grad_inner_Adam"].abs() + 1e-300)
    paired["delta_condition_score"] = paired["condition_score_Muon"] - paired["condition_score_Adam"]
    paired["delta_nrG"] = paired["nrG_Muon"] - paired["nrG_Adam"]
    paired["delta_stA"] = paired["stA_Muon"] - paired["stA_Adam"]
    return paired


def _step_gradient_rank_fraction(layers: pd.DataFrame) -> pd.DataFrame:
    source = layers[
        np.isfinite(layers["nrG"])
        & np.isfinite(layers["update_rank_ceiling"])
        & (layers["update_rank_ceiling"] > 0)
        & np.isfinite(layers["update_grad_inner"])
    ].copy()
    source["grad_rank_fraction"] = source["nrG"] / source["update_rank_ceiling"]
    grouped = source.groupby(
        ["problem_family", "setting", "kappa", "lr", "seed", "step", "algo"],
        as_index=False,
        observed=True,
    ).agg(
        mean_grad_rank_fraction=("grad_rank_fraction", "mean"),
        min_grad_rank_fraction=("grad_rank_fraction", "min"),
        max_grad_rank_fraction=("grad_rank_fraction", "max"),
    )
    pivot = grouped.pivot_table(
        index=["problem_family", "setting", "kappa", "lr", "seed", "step"],
        columns="algo",
        values=["mean_grad_rank_fraction", "min_grad_rank_fraction", "max_grad_rank_fraction"],
        aggfunc="mean",
    )
    pivot.columns = [f"{metric}_{algo}" for metric, algo in pivot.columns]
    result = pivot.reset_index().dropna().copy()
    result["delta_mean_grad_rank_fraction"] = (
        result["mean_grad_rank_fraction_Muon"] - result["mean_grad_rank_fraction_Adam"]
    )
    return result


def _paired_win_frame(steps: pd.DataFrame, layers: pd.DataFrame) -> pd.DataFrame:
    paired = _paired_step_metrics(steps)
    rank_fraction = _step_gradient_rank_fraction(layers)
    paired = paired.merge(
        rank_fraction,
        on=["problem_family", "setting", "kappa", "lr", "seed", "step"],
        how="left",
    )
    paired["muon_grad_rank_fraction"] = paired["mean_grad_rank_fraction_Muon"]
    paired["adam_grad_rank_fraction"] = paired["mean_grad_rank_fraction_Adam"]
    paired["pair_mean_grad_rank_fraction"] = paired[
        ["mean_grad_rank_fraction_Muon", "mean_grad_rank_fraction_Adam"]
    ].mean(axis=1)
    return paired


def first_order_win_condition_summary(steps: pd.DataFrame, layers: pd.DataFrame) -> pd.DataFrame:
    """Summarize when Muon has larger matched one-step progress than Adam.

    This is intentionally a descriptive diagnostic, not a causal estimator. It
    conditions matched setting/seed/step pairs on current spectral quantities
    and reports Muon's win rate and magnitude ratios.
    """

    paired = _paired_win_frame(steps, layers)

    conditions = [
        ("muon_grad_rank_fraction", "Muon mean nr(G_i)/r_i"),
        ("adam_grad_rank_fraction", "Adam mean nr(G_i)/r_i"),
        ("pair_mean_grad_rank_fraction", "pair mean nr(G_i)/r_i"),
        ("delta_mean_grad_rank_fraction", "Muon - Adam mean nr(G_i)/r_i"),
        ("condition_score_Muon", "Muon condition score"),
        ("delta_condition_score", "Muon - Adam condition score"),
    ]
    groupers = [("All", paired)]
    groupers.extend(list(paired.groupby("problem_family", observed=True, sort=False)))
    records = []
    for condition, label in conditions:
        for family, group in groupers:
            values = group[condition].replace([np.inf, -np.inf], np.nan)
            finite = group[values.notna()].copy()
            if len(finite) < 12 or finite[condition].nunique() < 3:
                continue
            try:
                finite["bin"] = pd.qcut(finite[condition], q=3, labels=["low", "middle", "high"], duplicates="drop")
            except ValueError:
                continue
            for bin_name, bin_rows in finite.groupby("bin", observed=True, sort=False):
                if bin_rows.empty:
                    continue
                first_order_ratio, first_order_lo, first_order_hi = log_ratio_ci95(bin_rows["first_order_ratio"])
                delta_loss_ratio, delta_loss_lo, delta_loss_hi = log_ratio_ci95(bin_rows["delta_loss_ratio"])
                records.append(
                    {
                        "condition": condition,
                        "condition_label": label,
                        "problem_family": family,
                        "bin": str(bin_name),
                        "bin_min": float(bin_rows[condition].min()),
                        "bin_max": float(bin_rows[condition].max()),
                        "n_pairs": int(len(bin_rows)),
                        "muon_first_order_win_rate": float(bin_rows["muon_first_order_win"].mean()),
                        "muon_delta_loss_win_rate": float(bin_rows["muon_delta_loss_win"].mean()),
                        "geomean_first_order_ratio_muon_over_adam": first_order_ratio,
                        "first_order_ratio_ci95_low": first_order_lo,
                        "first_order_ratio_ci95_high": first_order_hi,
                        "geomean_delta_loss_ratio_muon_over_adam": delta_loss_ratio,
                        "delta_loss_ratio_ci95_low": delta_loss_lo,
                        "delta_loss_ratio_ci95_high": delta_loss_hi,
                    }
                )
    return pd.DataFrame(records)


def _binary_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(probabilities)
    y_true = y_true[valid].astype(int)
    probabilities = probabilities[valid]
    if y_true.size == 0:
        return {
            "mean_predicted_probability": math.nan,
            "calibration_error": math.nan,
            "auc": math.nan,
            "accuracy": math.nan,
            "balanced_accuracy": math.nan,
            "brier": math.nan,
            "baseline_brier": math.nan,
        }
    prediction = probabilities >= 0.5
    mean_probability = float(probabilities.mean())
    positive_rate = float(y_true.mean())
    accuracy = float((prediction == y_true).mean())
    class_values = sorted(set(y_true.tolist()))
    if len(class_values) == 2:
        try:
            from sklearn.metrics import roc_auc_score

            auc = float(roc_auc_score(y_true, probabilities))
        except Exception:
            auc = math.nan
        recalls = []
        for label in [0, 1]:
            mask = y_true == label
            recalls.append(float((prediction[mask] == label).mean()))
        balanced_accuracy = float(np.mean(recalls))
    else:
        auc = math.nan
        balanced_accuracy = math.nan
    brier = float(np.mean((probabilities - y_true) ** 2))
    baseline = float(y_true.mean())
    baseline_brier = float(np.mean((baseline - y_true) ** 2))
    return {
        "mean_predicted_probability": mean_probability,
        "calibration_error": mean_probability - positive_rate,
        "auc": auc,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "brier": brier,
        "baseline_brier": baseline_brier,
    }


WIN_NUMERIC_FEATURES = [
    "muon_grad_rank_fraction",
    "adam_grad_rank_fraction",
    "pair_mean_grad_rank_fraction",
    "delta_mean_grad_rank_fraction",
    "condition_score_Muon",
    "condition_score_Adam",
    "delta_condition_score",
    "nrG_Muon",
    "nrG_Adam",
    "delta_nrG",
    "stA_Muon",
    "stA_Adam",
    "delta_stA",
]


def _fit_predict_logistic(train: pd.DataFrame, test: pd.DataFrame, *, target: str, model: str) -> np.ndarray:
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    categorical_features = ["problem_family"]
    if model == "spectrum_only":
        transformers = [("numeric", StandardScaler(), WIN_NUMERIC_FEATURES)]
    elif model == "family_only":
        transformers = [("family", OneHotEncoder(handle_unknown="ignore"), categorical_features)]
    elif model == "spectrum_plus_family":
        transformers = [
            ("numeric", StandardScaler(), WIN_NUMERIC_FEATURES),
            ("family", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    else:
        raise ValueError(f"unknown win-prediction model: {model}")

    if train[target].nunique() < 2:
        return np.full(len(test), float(train[target].mean()))
    pipeline = Pipeline(
        steps=[
            ("features", ColumnTransformer(transformers=transformers, remainder="drop")),
            ("model", LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")),
        ]
    )
    pipeline.fit(train, train[target].astype(int))
    return pipeline.predict_proba(test)[:, 1]


def win_prediction_generalization_summary(steps: pd.DataFrame, layers: pd.DataFrame) -> pd.DataFrame:
    """Test whether rank/spectral diagnostics predict Muon wins across tasks."""

    frame = _paired_win_frame(steps, layers).replace([np.inf, -np.inf], np.nan)
    required = [
        "problem_family",
        "muon_first_order_win",
        "muon_delta_loss_win",
    ] + WIN_NUMERIC_FEATURES
    frame = frame.dropna(subset=required).copy()
    targets = ["muon_first_order_win", "muon_delta_loss_win"]
    models = ["spectrum_only", "family_only", "spectrum_plus_family"]
    records = []
    for target in targets:
        for model in models:
            probabilities = _fit_predict_logistic(frame, frame, target=target, model=model)
            metrics = _binary_metrics(frame[target].to_numpy(dtype=int), probabilities)
            records.append(
                {
                    "target": target,
                    "model": model,
                    "evaluation": "in_sample",
                    "held_out_family": "None",
                    "train_pairs": int(len(frame)),
                    "test_pairs": int(len(frame)),
                    "test_positive_rate": float(frame[target].mean()),
                    **metrics,
                }
            )
            for family in sorted(frame["problem_family"].unique()):
                train = frame[frame["problem_family"] != family].copy()
                test = frame[frame["problem_family"] == family].copy()
                probabilities = _fit_predict_logistic(train, test, target=target, model=model)
                metrics = _binary_metrics(test[target].to_numpy(dtype=int), probabilities)
                records.append(
                    {
                        "target": target,
                        "model": model,
                        "evaluation": "leave_family_out",
                        "held_out_family": family,
                        "train_pairs": int(len(train)),
                        "test_pairs": int(len(test)),
                        "test_positive_rate": float(test[target].mean()),
                        **metrics,
                    }
                )
    return pd.DataFrame(records)


def win_feature_overlap_summary(steps: pd.DataFrame, layers: pd.DataFrame) -> pd.DataFrame:
    """Measure whether held-out task features lie inside the training-task support."""

    frame = _paired_win_frame(steps, layers).replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna(subset=["problem_family"] + WIN_NUMERIC_FEATURES).copy()
    records = []
    for family in sorted(frame["problem_family"].unique()):
        train = frame[frame["problem_family"] != family]
        test = frame[frame["problem_family"] == family]
        train_mean = train[WIN_NUMERIC_FEATURES].mean()
        train_std = train[WIN_NUMERIC_FEATURES].std(ddof=0).replace(0.0, 1.0)
        train_z = ((train[WIN_NUMERIC_FEATURES] - train_mean) / train_std).to_numpy(dtype=float)
        test_z = ((test[WIN_NUMERIC_FEATURES] - train_mean) / train_std).to_numpy(dtype=float)
        if len(train_z) and len(test_z):
            distances = np.sqrt(((test_z[:, None, :] - train_z[None, :, :]) ** 2).sum(axis=2))
            min_distances = distances.min(axis=1)
        else:
            min_distances = np.array([], dtype=float)

        outside_matrix = []
        for feature in WIN_NUMERIC_FEATURES:
            train_min = float(train[feature].min())
            train_max = float(train[feature].max())
            test_values = test[feature].to_numpy(dtype=float)
            outside = (test_values < train_min) | (test_values > train_max)
            outside_matrix.append(outside)
            records.append(
                {
                    "row_type": "feature",
                    "held_out_family": family,
                    "feature": feature,
                    "test_pairs": int(len(test)),
                    "train_min": train_min,
                    "train_max": train_max,
                    "test_min": float(np.nanmin(test_values)),
                    "test_max": float(np.nanmax(test_values)),
                    "frac_test_outside_train_range": float(np.mean(outside)),
                    "mean_feature_outside_fraction": math.nan,
                    "frac_pairs_with_any_feature_outside": math.nan,
                    "median_min_standardized_distance": math.nan,
                    "p90_min_standardized_distance": math.nan,
                }
            )
        if outside_matrix:
            outside_by_pair = np.column_stack(outside_matrix)
            mean_feature_outside = float(outside_by_pair.mean())
            any_outside = float(outside_by_pair.any(axis=1).mean())
        else:
            mean_feature_outside = math.nan
            any_outside = math.nan
        records.append(
            {
                "row_type": "family_summary",
                "held_out_family": family,
                "feature": "ALL",
                "test_pairs": int(len(test)),
                "train_min": math.nan,
                "train_max": math.nan,
                "test_min": math.nan,
                "test_max": math.nan,
                "frac_test_outside_train_range": math.nan,
                "mean_feature_outside_fraction": mean_feature_outside,
                "frac_pairs_with_any_feature_outside": any_outside,
                "median_min_standardized_distance": float(np.median(min_distances)) if len(min_distances) else math.nan,
                "p90_min_standardized_distance": float(np.quantile(min_distances, 0.9)) if len(min_distances) else math.nan,
            }
        )
    return pd.DataFrame(records)
