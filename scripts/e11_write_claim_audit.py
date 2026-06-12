from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.reporting import (
    fmt,
    markdown_table as table,
    require_one as row,
    write_markdown,
)


OUTPUT_PATH = Path("discussion/e11_claim_validity_audit.md")


def main() -> None:
    update_spectrum = pd.read_csv("results/e11_equal_update/update_spectrum_summary.csv")
    first_order = pd.read_csv("results/e11_equal_update/first_order_pair_summary.csv")
    calibration = pd.read_csv("results/e11_equal_update/first_order_calibration_summary.csv")
    generalization = pd.read_csv("results/e11_equal_update/win_prediction_generalization_summary.csv")
    overlap = pd.read_csv("results/e11_equal_update/win_feature_overlap_summary.csv")
    width = pd.read_csv("results/e11_mlp_width_sweep/width_pair_summary.csv")
    hybrid = pd.read_csv("results/e11_mlp_layer_hybrid/hybrid_update_allocation_summary.csv")
    hyper_best = pd.read_csv("results/e11_hyperparam_sweep/best_pair_summary.csv")
    hyper_spectrum = pd.read_csv("results/e11_hyperparam_sweep/update_spectrum_summary.csv")
    target_pair = pd.read_csv("results/e11_target_update_sweep/target_pair_summary.csv")
    target_best = pd.read_csv("results/e11_target_update_sweep/best_target_summary.csv")
    target_spectrum = pd.read_csv("results/e11_target_update_sweep/update_spectrum_summary.csv")
    layer_control = pd.read_csv("results/e11_mlp_per_layer_control/pair_summary.csv")
    layer_control["hidden_dim_numeric"] = pd.to_numeric(layer_control["hidden_dim"], errors="coerce")
    spectral_probe = pd.read_csv("results/e11_spectral_allocation_probe/spectral_allocation_summary.csv")
    subspace_final = pd.read_csv("results/e11_singular_vector_trajectory/subspace_final_summary.csv")
    swap_summary = pd.read_csv("results/e11_singular_vector_swap_probe/swap_ratio_summary.csv")
    natural_swap_summary = pd.read_csv("results/e11_natural_update_swap_probe/natural_update_swap_summary.csv")
    optimizer_switch_summary = pd.read_csv("results/e11_optimizer_switch_probe/optimizer_switch_summary.csv")
    optimizer_switch_reset_summary = pd.read_csv("results/e11_optimizer_switch_reset_control/optimizer_switch_reset_summary.csv")
    optimizer_switch_horizon_summary = pd.read_csv("results/e11_optimizer_switch_horizon_sweep/optimizer_switch_horizon_summary.csv")
    optimizer_switch_lr_summary = pd.read_csv("results/e11_optimizer_switch_lr_sweep/optimizer_switch_lr_summary.csv")

    update_core = update_spectrum[
        (update_spectrum["problem_family"] == "All") & (update_spectrum["metric"].isin(["nrUpdate", "stUpdate", "nrUpdateFrac", "stUpdateFrac"]))
    ].copy()
    first_order_core = first_order[
        first_order["metric"].isin(["delta_loss", "update_grad_inner"]) & first_order["problem_family"].isin(["All", "MatrixFactorizationInput", "MatrixSensing", "SmallMLPDigits"])
    ].copy()
    calibration_core = calibration[calibration["group"].isin(["All", "Adam", "Muon"])].copy()
    leave_family = generalization[
        (generalization["target"] == "muon_first_order_win")
        & (generalization["model"] == "spectrum_only")
        & (generalization["evaluation"] == "leave_family_out")
    ].copy()
    overlap_overall = overlap[overlap["row_type"] == "family_summary"].copy()
    width_core = width[
        (width["group_type"] == "hidden_samples")
        & (width["metric"] == "update_grad_inner")
        & (width["num_samples"].astype(str) == "1024")
    ].copy()
    hybrid_core = hybrid[
        (hybrid["num_samples"] == 1024)
        & (hybrid["layer"] == 2)
        & (hybrid["algo"] == "AdamFirstMuonSecond")
        & (hybrid["hidden_dim"].isin([8, 16, 64, 128]))
    ].copy()
    hyper_best_core = hyper_best[hyper_best["base_setting"] == "All"].copy()
    hyper_spectrum_core = hyper_spectrum[
        (hyper_spectrum["problem_family"] == "All")
        & (hyper_spectrum["metric"].isin(["nrUpdate", "stUpdate", "nrUpdateFrac", "stUpdateFrac"]))
    ].copy()
    target_direction_core = target_pair[
        (target_pair["group_type"] == "setting_all_targets")
        & (target_pair["metric"] == "update_grad_inner")
    ].copy()
    target_best_core = target_best.copy()
    target_spectrum_core = target_spectrum[
        (target_spectrum["problem_family"] == "All")
        & (target_spectrum["metric"].isin(["nrUpdate", "stUpdate", "nrUpdateFrac", "stUpdateFrac"]))
    ].copy()
    layer_control_core = layer_control[
        (layer_control["group_type"] == "hidden_all_targets")
        & (layer_control["metric"] == "update_grad_inner")
    ].copy()
    layer_control_target = layer_control[
        (layer_control["group_type"] == "target_hidden")
        & (layer_control["metric"] == "update_grad_inner")
    ].copy()
    spectral_probe_core = spectral_probe[
        (spectral_probe["group_type"] == "budget_all")
        & (spectral_probe["comparison"] == "flat_polar_over_gd_spectrum")
        & (spectral_probe["metric"] == "update_grad_inner")
    ].copy()
    subspace_core = subspace_final.copy()
    swap_core = swap_summary[
        (swap_summary["group_type"] == "setting_all_eval")
        & (swap_summary["metric"] == "update_grad_inner")
    ].copy()
    natural_swap_core = natural_swap_summary[
        (natural_swap_summary["group_type"] == "setting_target_budget_all_eval")
        & (natural_swap_summary["metric"] == "update_grad_inner")
        & (natural_swap_summary["target_layer_relative_norm"].astype(str) == "0.001")
    ].copy()
    natural_swap_target_core = natural_swap_summary[
        (natural_swap_summary["group_type"].isin(["target_budget_all", "target_all", "all"]))
        & (natural_swap_summary["metric"] == "delta_loss")
    ].copy()
    optimizer_switch_core = optimizer_switch_summary[
        (optimizer_switch_summary["group_type"].isin(["setting_source_all_checkpoints", "source_all", "all"]))
        & (optimizer_switch_summary["metric"] == "total_decrease")
    ].copy()
    optimizer_switch_reset_core = optimizer_switch_reset_summary[
        (optimizer_switch_reset_summary["group_type"].isin(["setting_source_all_checkpoints", "source_all", "all"]))
        & (optimizer_switch_reset_summary["metric"] == "loss_after")
    ].copy()
    optimizer_switch_horizon_core = optimizer_switch_horizon_summary[
        (optimizer_switch_horizon_summary["group_type"].isin(["source_horizon_all_settings", "horizon_all", "all"]))
        & (optimizer_switch_horizon_summary["metric"] == "loss_after")
    ].copy()
    optimizer_switch_lr_core = optimizer_switch_lr_summary[
        (optimizer_switch_lr_summary["group_type"].isin(["setting_source", "source_all", "all"]))
        & (optimizer_switch_lr_summary["metric"] == "loss_after")
    ].copy()

    nr_update = row(update_spectrum, metric="nrUpdate", problem_family="All")
    st_update = row(update_spectrum, metric="stUpdate", problem_family="All")
    all_first = row(first_order, metric="update_grad_inner", problem_family="All")
    matrix_sensing_first = row(first_order, metric="update_grad_inner", problem_family="MatrixSensing")
    mf_first = row(first_order, metric="update_grad_inner", problem_family="MatrixFactorizationInput")
    calibration_all = row(calibration, group="All")
    hidden8 = row(width_core, hidden_dim=8)
    hidden128 = row(width_core, hidden_dim=128)
    hybrid128 = row(hybrid_core, hidden_dim=128)
    hyper_raw_all = row(hyper_best_core, mode="raw")
    hyper_equal_all = row(hyper_best_core, mode="equal_update")
    hyper_raw_nr = row(hyper_spectrum_core, mode="raw", metric="nrUpdate")
    hyper_equal_nr = row(hyper_spectrum_core, mode="equal_update", metric="nrUpdate")
    target_ms = row(target_direction_core, base_setting="Matrix sensing kappa=1e+02")
    target_mlp64 = row(target_direction_core, base_setting="Small MLP digits hidden=64")
    target_nr = row(target_spectrum_core, metric="nrUpdate")
    layer_control_h16 = row(layer_control_core, hidden_dim_numeric=16)
    layer_control_h64 = row(layer_control_core, hidden_dim_numeric=64)
    spectral_fro = row(spectral_probe_core, budget="fro")
    spectral_op = row(spectral_probe_core, budget="op")
    subspace_mf = row(subspace_core, base_setting="MF input kappa=1e+02")
    subspace_ms = row(subspace_core, base_setting="Matrix sensing kappa=1e+02")
    subspace_mlp64 = row(subspace_core, base_setting="Small MLP digits hidden=64")
    swap_all = row(
        swap_summary[
            (swap_summary["group_type"] == "all")
            & (swap_summary["metric"] == "update_grad_inner")
        ],
        eval_algo="All",
    )
    swap_ms = row(swap_core, base_setting="Matrix sensing kappa=1e+02")
    natural_swap_all = row(
        natural_swap_summary[
            (natural_swap_summary["group_type"] == "all")
            & (natural_swap_summary["metric"] == "update_grad_inner")
        ],
        target_layer_relative_norm="All",
        budget="All",
        eval_algo="All",
    )
    natural_swap_delta_all = row(
        natural_swap_summary[
            (natural_swap_summary["group_type"] == "all")
            & (natural_swap_summary["metric"] == "delta_loss")
        ],
        target_layer_relative_norm="All",
        budget="All",
        eval_algo="All",
    )
    natural_swap_fro = row(
        natural_swap_summary[
            (natural_swap_summary["group_type"] == "target_budget_all")
            & (natural_swap_summary["metric"] == "delta_loss")
        ],
        target_layer_relative_norm="0.001",
        budget="fro",
        eval_algo="All",
    )
    natural_swap_op = row(
        natural_swap_summary[
            (natural_swap_summary["group_type"] == "target_budget_all")
            & (natural_swap_summary["metric"] == "delta_loss")
        ],
        target_layer_relative_norm="0.001",
        budget="op",
        eval_algo="All",
    )
    switch_all = row(
        optimizer_switch_summary[
            (optimizer_switch_summary["group_type"] == "all")
            & (optimizer_switch_summary["metric"] == "total_decrease")
        ],
        source_algo="All",
        checkpoint_step="All",
    )
    switch_adam_source = row(
        optimizer_switch_summary[
            (optimizer_switch_summary["group_type"] == "source_all")
            & (optimizer_switch_summary["metric"] == "loss_after")
        ],
        source_algo="Adam",
        checkpoint_step="All",
    )
    switch_muon_source = row(
        optimizer_switch_summary[
            (optimizer_switch_summary["group_type"] == "source_all")
            & (optimizer_switch_summary["metric"] == "loss_after")
        ],
        source_algo="Muon",
        checkpoint_step="All",
    )
    reset_adam_switch = row(
        optimizer_switch_reset_summary[
            (optimizer_switch_reset_summary["group_type"] == "source_all")
            & (optimizer_switch_reset_summary["metric"] == "loss_after")
            & (optimizer_switch_reset_summary["comparison"] == "switched_fresh_over_own_fresh")
        ],
        source_algo="Adam",
    )
    reset_muon_switch = row(
        optimizer_switch_reset_summary[
            (optimizer_switch_reset_summary["group_type"] == "source_all")
            & (optimizer_switch_reset_summary["metric"] == "loss_after")
            & (optimizer_switch_reset_summary["comparison"] == "switched_fresh_over_own_fresh")
        ],
        source_algo="Muon",
    )
    reset_adam_own = row(
        optimizer_switch_reset_summary[
            (optimizer_switch_reset_summary["group_type"] == "source_all")
            & (optimizer_switch_reset_summary["metric"] == "loss_after")
            & (optimizer_switch_reset_summary["comparison"] == "own_fresh_over_own_preserved")
        ],
        source_algo="Adam",
    )
    horizon_all_10 = row(
        optimizer_switch_horizon_summary[
            (optimizer_switch_horizon_summary["group_type"] == "horizon_all")
            & (optimizer_switch_horizon_summary["metric"] == "loss_after")
        ],
        source_algo="All",
        horizon="10",
    )
    horizon_muon_30 = row(
        optimizer_switch_horizon_summary[
            (optimizer_switch_horizon_summary["group_type"] == "source_horizon_all_settings")
            & (optimizer_switch_horizon_summary["metric"] == "loss_after")
        ],
        source_algo="Muon",
        horizon="30",
    )
    horizon_adam_30 = row(
        optimizer_switch_horizon_summary[
            (optimizer_switch_horizon_summary["group_type"] == "source_horizon_all_settings")
            & (optimizer_switch_horizon_summary["metric"] == "loss_after")
        ],
        source_algo="Adam",
        horizon="30",
    )
    lr_all = row(
        optimizer_switch_lr_summary[
            (optimizer_switch_lr_summary["group_type"] == "all")
            & (optimizer_switch_lr_summary["metric"] == "loss_after")
        ],
        source_algo="All",
    )
    lr_adam_source = row(
        optimizer_switch_lr_summary[
            (optimizer_switch_lr_summary["group_type"] == "source_all")
            & (optimizer_switch_lr_summary["metric"] == "loss_after")
        ],
        source_algo="Adam",
    )
    lr_muon_source = row(
        optimizer_switch_lr_summary[
            (optimizer_switch_lr_summary["group_type"] == "source_all")
            & (optimizer_switch_lr_summary["metric"] == "loss_after")
        ],
        source_algo="Muon",
    )

    claim_rows = pd.DataFrame(
        [
            {
                "claim": "Muon has a distinct update-spectrum geometry.",
                "status": "supported",
                "evidence": (
                    f"Equal-update nrUpdate ratio={fmt(nr_update['geomean_ratio_muon_over_adam'])} "
                    f"CI=[{fmt(nr_update['ratio_ci95_low'])},{fmt(nr_update['ratio_ci95_high'])}], "
                    f"stUpdate ratio={fmt(st_update['geomean_ratio_muon_over_adam'])} "
                    f"CI=[{fmt(st_update['ratio_ci95_low'])},{fmt(st_update['ratio_ci95_high'])}]."
                ),
                "main_loophole": "This is partly by construction for exact polar Muon, so it is a mechanism fact, not a surprise performance result.",
            },
            {
                "claim": "Muon is globally better than Adam on short-horizon progress.",
                "status": "not supported",
                "evidence": (
                    f"All-task first-order ratio={fmt(all_first['geomean_ratio_muon_over_adam'])} "
                    f"CI=[{fmt(all_first['ratio_ci95_low'])},{fmt(all_first['ratio_ci95_high'])}]; "
                    f"MF ratio={fmt(mf_first['geomean_ratio_muon_over_adam'])}, "
                    f"Matrix Sensing ratio={fmt(matrix_sensing_first['geomean_ratio_muon_over_adam'])}."
                ),
                "main_loophole": "The result is task-dependent; aggregate ratios hide opposite family-level behavior.",
            },
            {
                "claim": "One-step progress is explained by gradient-update alignment.",
                "status": "supported for these short steps",
                "evidence": (
                    f"All equal-update Spearman(delta_loss, <G,D>)="
                    f"{fmt(calibration_all['spearman_delta_vs_first_order'])} "
                    f"CI=[{fmt(calibration_all['spearman_ci95_low'])},{fmt(calibration_all['spearman_ci95_high'])}], "
                    f"within-factor-2={fmt(calibration_all['within_factor_2'])}."
                ),
                "main_loophole": "This validates a first-order local diagnostic, not long-horizon optimization or generalization.",
            },
            {
                "claim": "A simple cross-task spectral rule predicts when Muon wins.",
                "status": "not yet supported",
                "evidence": "Leave-family-out tests fail and feature-overlap diagnostics show the held-out families often leave the training support.",
                "main_loophole": "Current families are too separated spectrally; a successful in-sample rule may be extrapolating.",
            },
            {
                "claim": "SmallMLP has a real width-dependent transition.",
                "status": "supported in the current digits setup",
                "evidence": (
                    f"samples=1024 hidden=8 first-order ratio={fmt(hidden8['geomean_ratio_muon_over_adam'])} "
                    f"CI=[{fmt(hidden8['ratio_ci95_low'])},{fmt(hidden8['ratio_ci95_high'])}], "
                    f"hidden=128 ratio={fmt(hidden128['geomean_ratio_muon_over_adam'])} "
                    f"CI=[{fmt(hidden128['ratio_ci95_low'])},{fmt(hidden128['ratio_ci95_high'])}]. "
                    f"With per-layer update sizes fixed, hidden=16 becomes near-neutral "
                    f"({fmt(layer_control_h16['geomean_ratio_muon_over_adam'])} "
                    f"CI=[{fmt(layer_control_h16['ratio_ci95_low'])},{fmt(layer_control_h16['ratio_ci95_high'])}]), "
                    f"while hidden=64 remains Adam-favorable "
                    f"({fmt(layer_control_h64['geomean_ratio_muon_over_adam'])} "
                    f"CI=[{fmt(layer_control_h64['ratio_ci95_low'])},{fmt(layer_control_h64['ratio_ci95_high'])}])."
                ),
                "main_loophole": "The wide-network failure is robust in this setup, but the narrow-network Muon advantage depends on target scale and is not purely directional.",
            },
            {
                "claim": "Layerwise Adam/Muon hybrids isolate the bottleneck.",
                "status": "not supported",
                "evidence": (
                    f"At hidden=128/samples=1024, AdamFirstMuonSecond layer-2 update-budget fraction="
                    f"{fmt(hybrid128['fro_sq_fraction'])}, efficiency ratio={fmt(hybrid128['efficiency_ratio_over_adam'])}, "
                    f"inner ratio={fmt(hybrid128['inner_ratio_over_adam'])}."
                ),
                "main_loophole": "Layerwise hybrid changes layer update allocation, so it is not a clean intervention on only the update direction.",
            },
            {
                "claim": "The main conclusions are just a single-learning-rate artifact.",
                "status": "weakened by lr and target-norm sweeps, not fully closed",
                "evidence": (
                    f"Representative best-lr sweep: raw all-setting final-loss ratio={fmt(hyper_raw_all['final_loss_ratio_muon_over_adam'])} "
                    f"CI=[{fmt(hyper_raw_all['final_loss_ratio_ci95_low'])},{fmt(hyper_raw_all['final_loss_ratio_ci95_high'])}], "
                    f"equal-update ratio={fmt(hyper_equal_all['final_loss_ratio_muon_over_adam'])} "
                    f"CI=[{fmt(hyper_equal_all['final_loss_ratio_ci95_low'])},{fmt(hyper_equal_all['final_loss_ratio_ci95_high'])}]; "
                    f"direction-only target sweep gives MatrixSensing kappa=1e2 first-order ratio={fmt(target_ms['geomean_ratio_muon_over_adam'])} "
                    f"but MLP hidden=64 ratio={fmt(target_mlp64['geomean_ratio_muon_over_adam'])}; "
                    f"nrUpdate ratios remain >1 in raw ({fmt(hyper_raw_nr['geomean_ratio_muon_over_adam'])}), "
                    f"equal-update ({fmt(hyper_equal_nr['geomean_ratio_muon_over_adam'])}), "
                    f"and target-norm ({fmt(target_nr['geomean_ratio_muon_over_adam'])})."
                ),
                "main_loophole": "The sweeps are representative and partly oracle-style; per-layer allocation is only controlled for SmallMLP, not for all tasks or broad architectures.",
            },
            {
                "claim": "Flat/polar spectral allocation is universally better than GD allocation.",
                "status": "not supported; norm-geometry dependent",
                "evidence": (
                    f"Within-layer probe: flat_polar/GD first-order ratio under Frobenius budget="
                    f"{fmt(spectral_fro['geomean_ratio'])} CI=[{fmt(spectral_fro['ratio_ci95_low'])},{fmt(spectral_fro['ratio_ci95_high'])}], "
                    f"but under operator-norm budget={fmt(spectral_op['geomean_ratio'])} "
                    f"CI=[{fmt(spectral_op['ratio_ci95_low'])},{fmt(spectral_op['ratio_ci95_high'])}]."
                ),
                "main_loophole": "The probe fixes gradient singular vectors and is one-step artificial; it does not model natural singular-vector trajectories.",
            },
            {
                "claim": "Adam and Muon stay in the same singular-vector geometry.",
                "status": "not generally supported",
                "evidence": (
                    f"Final matched gradient subspace overlap stays high for MF "
                    f"({fmt(subspace_mf['mean_grad_overlap'])}), but drops strongly for Matrix Sensing "
                    f"({fmt(subspace_ms['mean_grad_overlap'])}) and MLP hidden=64 "
                    f"({fmt(subspace_mlp64['mean_grad_overlap'])}). "
                    f"Swap probe mean signed other/own first-order ratio is "
                    f"{fmt(swap_all['mean_signed_other_over_own'])} overall, and "
                    f"{fmt(swap_ms['mean_signed_other_over_own'])} for Matrix Sensing. "
                    f"Natural update-vector swap gives all-budget signed other/own ratio="
                    f"{fmt(natural_swap_all['mean_signed_other_over_own'])} "
                    f"CI=[{fmt(natural_swap_all['signed_ratio_ci95_low'])},{fmt(natural_swap_all['signed_ratio_ci95_high'])}] "
                    f"for first-order progress and observed delta-loss ratio="
                    f"{fmt(natural_swap_delta_all['mean_signed_other_over_own'])} "
                    f"CI=[{fmt(natural_swap_delta_all['signed_ratio_ci95_low'])},{fmt(natural_swap_delta_all['signed_ratio_ci95_high'])}] "
                    f"across five target scales."
                ),
                "main_loophole": "The natural-update swap is still one-step and artificial; some setting/budget/eval-state cells favor the other update, so the effect is not a simple universal specialization claim.",
            },
            {
                "claim": "One-step own-update advantage implies own-optimizer trajectory continuation is best.",
                "status": "not supported",
                "evidence": (
                    f"Trajectory switch probe: switched/own total-decrease ratio={fmt(switch_all['mean_signed_switched_over_own'])} "
                    f"CI=[{fmt(switch_all['signed_ratio_ci95_low'])},{fmt(switch_all['signed_ratio_ci95_high'])}]. "
                    f"Switching from Adam checkpoints increases final loss on average "
                    f"(loss ratio={fmt(switch_adam_source['mean_signed_switched_over_own'])}), "
                    f"while switching from Muon checkpoints to Adam often lowers final loss "
                    f"(loss ratio={fmt(switch_muon_source['mean_signed_switched_over_own'])}). "
                    f"Reset-control gives switched-fresh/own-fresh final-loss ratios "
                    f"{fmt(reset_adam_switch['mean_signed_ratio'])} from Adam checkpoints and "
                    f"{fmt(reset_muon_switch['mean_signed_ratio'])} from Muon checkpoints; "
                    f"Adam own-fresh/own-preserved ratio={fmt(reset_adam_own['mean_signed_ratio'])}. "
                    f"Horizon sweep final-loss ratios are horizon dependent: all-source horizon-10 ratio="
                    f"{fmt(horizon_all_10['mean_signed_switched_over_own'])}, "
                    f"Muon-source horizon-30 ratio={fmt(horizon_muon_30['mean_signed_switched_over_own'])}, "
                    f"Adam-source horizon-30 ratio={fmt(horizon_adam_30['mean_signed_switched_over_own'])}. "
                    f"After a continuation-LR sweep at horizon 30, best switched/own final-loss ratios are "
                    f"{fmt(lr_all['mean_switched_over_own'])} overall, "
                    f"{fmt(lr_adam_source['mean_switched_over_own'])} from Adam checkpoints, and "
                    f"{fmt(lr_muon_source['mean_switched_over_own'])} from Muon checkpoints."
                ),
                "main_loophole": "Reset-control, horizon sweep, and a small continuation-LR sweep reduce specific confounds but still do not constitute retuned full training.",
            },
        ]
    )

    text = f"""# E11 Claim Validity Audit

This audit separates what the current experiments support from what remains a vulnerability. It is generated from the current result CSVs so that the numerical claims are reproducible.

## Claim Status

{table(claim_rows, ["claim", "status", "evidence", "main_loophole"])}

## Evidence Tables

### Equal-Update Spectrum

{table(update_core, ["metric", "problem_family", "n_pairs", "muon_higher_pairs", "geomean_ratio_muon_over_adam", "ratio_ci95_low", "ratio_ci95_high", "ratio_ci95_above_one"])}

### Equal-Update First-Order Progress

{table(first_order_core, ["metric", "problem_family", "n_pairs", "muon_higher_pairs", "geomean_ratio_muon_over_adam", "ratio_ci95_low", "ratio_ci95_high", "ratio_ci95_above_one", "mean_delta_muon_minus_adam"])}

### First-Order Calibration

{table(calibration_core, ["group", "points", "positive_points", "spearman_delta_vs_first_order", "spearman_ci95_low", "spearman_ci95_high", "within_factor_2"])}

### Leave-Family-Out Generalization

{table(leave_family, ["target", "model", "held_out_family", "train_pairs", "test_pairs", "test_positive_rate", "auc", "balanced_accuracy", "brier", "baseline_brier"])}

### Feature-Overlap Failure

{table(overlap_overall, ["held_out_family", "test_pairs", "mean_feature_outside_fraction", "frac_pairs_with_any_feature_outside", "median_min_standardized_distance", "p90_min_standardized_distance"])}

### SmallMLP Width Transition

{table(width_core, ["hidden_dim", "num_samples", "n_pairs", "muon_win_rate", "geomean_ratio_muon_over_adam", "ratio_ci95_low", "ratio_ci95_high", "ratio_ci95_above_one", "ratio_ci95_below_one"])}

### Hybrid Update Allocation

{table(hybrid_core, ["hidden_dim", "num_samples", "layer", "algo", "fro_sq_fraction", "fro_sq_fraction_ratio_over_adam", "efficiency_ratio_over_adam", "inner_ratio_over_adam"])}

### Hyperparameter Sweep Robustness

{table(hyper_best_core, ["mode", "problem_family", "base_setting", "n_seed_pairs", "final_loss_ratio_muon_over_adam", "final_loss_ratio_ci95_low", "final_loss_ratio_ci95_high", "total_decrease_ratio_muon_over_adam", "total_decrease_ratio_ci95_low", "total_decrease_ratio_ci95_high"])}

{table(hyper_spectrum_core, ["mode", "metric", "problem_family", "n_pairs", "geomean_ratio_muon_over_adam", "ratio_ci95_low", "ratio_ci95_high", "ratio_ci95_above_one"])}

### Target-Update-Norm Direction Sweep

{table(target_direction_core, ["problem_family", "base_setting", "n_pairs", "muon_win_rate", "geomean_ratio_muon_over_adam", "ratio_ci95_low", "ratio_ci95_high", "ratio_ci95_above_one", "ratio_ci95_below_one"])}

{table(target_best_core, ["problem_family", "base_setting", "n_seed_pairs", "final_loss_ratio_muon_over_adam", "final_loss_ratio_ci95_low", "final_loss_ratio_ci95_high", "total_decrease_ratio_muon_over_adam", "total_decrease_ratio_ci95_low", "total_decrease_ratio_ci95_high", "mean_best_target_adam", "mean_best_target_muon"])}

{table(target_spectrum_core, ["metric", "problem_family", "n_pairs", "geomean_ratio_muon_over_adam", "ratio_ci95_low", "ratio_ci95_high", "ratio_ci95_above_one"])}

### SmallMLP Per-Layer Update Control

{table(layer_control_core, ["hidden_dim", "n_pairs", "muon_win_rate", "geomean_ratio_muon_over_adam", "ratio_ci95_low", "ratio_ci95_high", "ratio_ci95_above_one", "ratio_ci95_below_one"])}

{table(layer_control_target, ["hidden_dim", "target_layer_relative_update_norm", "n_pairs", "muon_win_rate", "geomean_ratio_muon_over_adam", "ratio_ci95_low", "ratio_ci95_high", "ratio_ci95_above_one", "ratio_ci95_below_one"])}

### Within-Layer Spectral Allocation Probe

{table(spectral_probe_core, ["budget", "comparison", "metric", "n_pairs", "geomean_ratio", "ratio_ci95_low", "ratio_ci95_high", "ratio_ci95_above_one", "ratio_ci95_below_one"])}

### Singular-Vector Trajectory Diagnostic

{table(subspace_core, ["problem_family", "base_setting", "step", "initial_grad_overlap", "mean_grad_overlap", "grad_overlap_drop", "mean_param_overlap", "mean_adam_loss", "mean_muon_loss"])}

### Singular-Vector Swap Probe

{table(swap_core, ["problem_family", "base_setting", "eval_algo", "n_pairs", "other_positive_pairs", "other_better_pairs", "mean_signed_other_over_own", "signed_ratio_ci95_low", "signed_ratio_ci95_high"])}

### Natural Update-Vector Swap Probe

Reference target `1e-3`, first-order metric:

{table(natural_swap_core, ["problem_family", "base_setting", "target_layer_relative_norm", "budget", "eval_algo", "n_pairs", "other_positive_pairs", "other_better_pairs", "mean_signed_other_over_own", "signed_ratio_ci95_low", "signed_ratio_ci95_high"])}

Observed delta-loss target sweep:

{table(natural_swap_target_core, ["group_type", "target_layer_relative_norm", "budget", "eval_algo", "n_pairs", "other_positive_pairs", "other_better_pairs", "mean_signed_other_over_own", "signed_ratio_ci95_low", "signed_ratio_ci95_high"])}

At reference target `1e-3`, observed Frobenius-budget signed other/own ratio={fmt(natural_swap_fro['mean_signed_other_over_own'])} CI=[{fmt(natural_swap_fro['signed_ratio_ci95_low'])},{fmt(natural_swap_fro['signed_ratio_ci95_high'])}], operator-budget ratio={fmt(natural_swap_op['mean_signed_other_over_own'])} CI=[{fmt(natural_swap_op['signed_ratio_ci95_low'])},{fmt(natural_swap_op['signed_ratio_ci95_high'])}].

### Optimizer Switch Probe

{table(optimizer_switch_core, ["group_type", "problem_family", "base_setting", "source_algo", "checkpoint_step", "n_pairs", "switched_better_pairs", "mean_signed_switched_over_own", "signed_ratio_ci95_low", "signed_ratio_ci95_high"])}

### Optimizer Switch Reset-Control Probe

{table(optimizer_switch_reset_core, ["group_type", "problem_family", "base_setting", "source_algo", "comparison", "n_pairs", "numerator_better_pairs", "mean_signed_ratio", "signed_ratio_ci95_low", "signed_ratio_ci95_high"])}

### Optimizer Switch Horizon Sweep

{table(optimizer_switch_horizon_core, ["group_type", "source_algo", "horizon", "n_pairs", "switched_better_pairs", "mean_signed_switched_over_own", "signed_ratio_ci95_low", "signed_ratio_ci95_high"])}

### Optimizer Switch Continuation-LR Sweep

{table(optimizer_switch_lr_core, ["group_type", "problem_family", "base_setting", "source_algo", "n_pairs", "switched_better_pairs", "mean_switched_over_own", "ratio_ci95_low", "ratio_ci95_high", "mean_best_lr_own", "mean_best_lr_switched"])}

## Current Valid Formulation

The defensible formulation is: **Muon is an update-spectrum shaping optimizer; its polar-style update consistently changes the singular-value geometry of the update matrices, but whether that geometry improves one-step progress depends on task and layer conditions.**

The current data do **not** justify saying that Muon is generally more stable, generally better, or that a single rank statistic explains all task families.

The intervention sequence behind this formulation is summarized in [E11 mechanism ladder](e11_mechanism_ladder.md).

## Main Loopholes To Close

1. Hyperparameters and target update norms are not exhaustively searched; the representative sweeps reduce but do not eliminate this concern.
2. The MLP result is a small sklearn digits benchmark, not a broad neural-network result.
3. The cross-task predictor currently fails under leave-family-out evaluation, partly because spectral supports do not overlap enough.
4. Layerwise hybrids are not clean causal interventions because they alter layer update-budget allocation; the per-layer control partially addresses this for SmallMLP but not for other tasks.
5. ExactMuon's update-spectrum flatness is partly algorithmic by construction; the spectral-allocation probe shows that this construction is favorable under an operator-norm budget but not under a Frobenius budget.
6. Singular-vector trajectory divergence is supported by polar-vector and natural-update-vector one-step swap probes.
7. The optimizer-switch, reset-control, horizon-sweep, and continuation-LR probes do not support a simple own-optimizer trajectory-specialization story; practical continuation effects depend on source optimizer, task family, continuation horizon, and tuning.
8. Most evidence is short-horizon one-step or 5-10 step behavior; final training performance remains a separate question.

## Next Experiments With Highest Value

1. Extend the representative hyperparameter and target-update-norm sweeps to a larger grid.
2. Add matched-support task settings where MF, Matrix Sensing, and MLP occupy overlapping `nr(G)/r`, `st(A)`, and update-alignment ranges.
3. Broaden the natural update-vector swap probe across more settings and target scales.
4. Extend the MLP benchmark to at least one external image dataset and one deeper architecture.
5. Report all main claims as paired ratios with confidence intervals, not as visual separations only.
"""
    write_markdown(OUTPUT_PATH, text)
    print(f"saved claim audit to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
