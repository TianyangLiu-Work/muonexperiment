from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "_No rows available._"
    display = frame[[column for column in columns if column in frame.columns]].copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: "n/a" if not np.isfinite(value) else f"{float(value):.4g}")
    header = "| " + " | ".join(display.columns) + " |"
    separator = "|" + "|".join(["---"] * len(display.columns)) + "|"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in display.itertuples(index=False, name=None)]
    return "\n".join([header, separator] + rows)


def geometry_takeaway(geometry: pd.DataFrame) -> str:
    parts = []
    for family, rows in geometry.groupby("problem_family", observed=True, sort=False):
        nr_positive = bool((rows["mean_delta_nrG"] > 0).all())
        st_positive = bool((rows["mean_delta_stA"] > 0).all())
        acc_min = float(rows["nearest_centroid_balanced_accuracy"].min())
        st_ci_count = int(rows["stA_ci95_excludes_zero"].sum())
        direction = []
        if nr_positive:
            direction.append("higher matched mean `nrG`")
        if st_positive:
            direction.append("higher matched mean `stA`")
        if not direction:
            direction.append("a non-uniform coordinate-wise difference")
        parts.append(
            f"{family}: Muon shows {' and '.join(direction)}; "
            f"`stA` CI excludes zero in {st_ci_count}/{len(rows)} settings; "
            f"minimum geometry-only balanced accuracy is {acc_min:.3f}."
        )
    return " ".join(parts)


def write_discussion(
    path: Path,
    *,
    performance: pd.DataFrame,
    geometry: pd.DataFrame,
    prediction: pd.DataFrame,
    volatility: pd.DataFrame,
    update_spectrum: pd.DataFrame,
    update_transmission: pd.DataFrame,
    first_order_calibration: pd.DataFrame,
    polar_alignment: pd.DataFrame,
    first_order_pair: pd.DataFrame,
    win_condition: pd.DataFrame | None = None,
    win_generalization: pd.DataFrame | None = None,
    win_overlap: pd.DataFrame | None = None,
    equal_update_volatility: pd.DataFrame | None = None,
    equal_update_spectrum: pd.DataFrame | None = None,
    equal_update_transmission: pd.DataFrame | None = None,
    equal_update_first_order_calibration: pd.DataFrame | None = None,
    equal_update_polar_alignment: pd.DataFrame | None = None,
    equal_update_first_order_pair: pd.DataFrame | None = None,
    equal_update_win_condition: pd.DataFrame | None = None,
    equal_update_win_generalization: pd.DataFrame | None = None,
    equal_update_win_overlap: pd.DataFrame | None = None,
    figure_paths: dict[str, Path],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    performance_table = markdown_table(
        performance,
        [
            "problem_family",
            "setting",
            "algo",
            "runs",
            "median_final_loss",
            "median_recovery",
            "median_nrG",
            "median_stA",
            "median_condition_score",
            "median_delta_loss",
        ],
    )
    geometry_table = markdown_table(
        geometry,
        [
            "problem_family",
            "setting",
            "seed_count",
            "mean_delta_nrG",
            "nrG_muon_higher_seed_count",
            "mean_delta_stA",
            "delta_stA_95ci",
            "stA_ci95_excludes_zero",
            "mean_delta_loss",
            "mean_delta_recovery",
            "centroid_distance_std",
            "nearest_centroid_balanced_accuracy",
        ],
    )
    prediction_table = markdown_table(
        prediction,
        [
            "problem_family",
            "setting",
            "algo",
            "predictor",
            "points",
            "positive_delta_points",
            "spearman_all_delta_loss",
            "pearson_log_positive_delta_loss",
        ],
    )
    volatility_table = markdown_table(
        volatility[
            volatility["metric"].isin(
                [
                    "norm_rank_plane_mean_speed",
                    "norm_condition_mean_speed",
                    "median_relative_update_fro_norm",
                    "per_update_rank_plane_mean_speed",
                    "per_update_condition_mean_speed",
                    "condition_score_std_speed",
                    "delta_gd_pred_mean_rel_speed",
                    "delta_spec_pred_mean_rel_speed",
                    "loss_mean_rel_speed",
                ]
            )
        ],
        [
            "metric",
            "problem_family",
            "n_pairs",
            "muon_lower_pairs",
            "geomean_ratio_muon_over_adam",
            "ratio_ci95_low",
            "ratio_ci95_high",
            "ratio_ci95_below_one",
            "mean_delta_muon_minus_adam",
            "delta_ci95_low",
            "delta_ci95_high",
        ],
    )
    equal_update_table = markdown_table(
        equal_update_volatility[
            equal_update_volatility["metric"].isin(
                [
                    "norm_rank_plane_mean_speed",
                    "norm_condition_mean_speed",
                    "median_relative_update_fro_norm",
                    "per_update_rank_plane_mean_speed",
                    "per_update_condition_mean_speed",
                    "condition_score_std_speed",
                    "delta_spec_pred_mean_rel_speed",
                    "loss_mean_rel_speed",
                ]
            )
        ]
        if equal_update_volatility is not None
        else pd.DataFrame(),
        [
            "metric",
            "problem_family",
            "n_pairs",
            "muon_lower_pairs",
            "geomean_ratio_muon_over_adam",
            "ratio_ci95_low",
            "ratio_ci95_high",
            "ratio_ci95_below_one",
        ],
    )
    update_spectrum_table = markdown_table(
        update_spectrum,
        [
            "metric",
            "problem_family",
            "n_pairs",
            "muon_higher_pairs",
            "geomean_ratio_muon_over_adam",
            "ratio_ci95_low",
            "ratio_ci95_high",
            "ratio_ci95_above_one",
            "mean_delta_muon_minus_adam",
            "delta_ci95_low",
            "delta_ci95_high",
        ],
    )
    equal_update_spectrum_table = markdown_table(
        equal_update_spectrum if equal_update_spectrum is not None else pd.DataFrame(),
        [
            "metric",
            "problem_family",
            "n_pairs",
            "muon_higher_pairs",
            "geomean_ratio_muon_over_adam",
            "ratio_ci95_low",
            "ratio_ci95_high",
            "ratio_ci95_above_one",
        ],
    )
    transmission_filter = (
        update_transmission["group"].isin(["All", "Adam", "Muon", "MatrixFactorizationInput", "MatrixSensing", "SmallMLPDigits"])
        & update_transmission["predictor"].isin(
            [
                "stUpdateFrac",
                "update_flatness",
                "relative_update_fro_norm",
                "update_grad_inner",
                "update_grad_cosine",
                "update_grad_per_update_norm",
            ]
        )
        & update_transmission["outcome"].isin(["relative_loss_decrease", "norm_rank_movement", "norm_condition_movement"])
    )
    transmission_table = markdown_table(
        update_transmission[transmission_filter],
        [
            "group",
            "predictor",
            "outcome",
            "points",
            "spearman",
            "spearman_ci95_low",
            "spearman_ci95_high",
            "ci95_excludes_zero",
        ],
    )
    equal_transmission_filter = (
        equal_update_transmission["group"].isin(["All", "Adam", "Muon", "MatrixFactorizationInput", "MatrixSensing", "SmallMLPDigits"])
        & equal_update_transmission["predictor"].isin(
            [
                "stUpdateFrac",
                "update_flatness",
                "relative_update_fro_norm",
                "update_grad_inner",
                "update_grad_cosine",
                "update_grad_per_update_norm",
            ]
        )
        & equal_update_transmission["outcome"].isin(["relative_loss_decrease", "norm_rank_movement", "norm_condition_movement"])
        if equal_update_transmission is not None
        else pd.Series(dtype=bool)
    )
    equal_transmission_table = markdown_table(
        equal_update_transmission[equal_transmission_filter] if equal_update_transmission is not None else pd.DataFrame(),
        [
            "group",
            "predictor",
            "outcome",
            "points",
            "spearman",
            "spearman_ci95_low",
            "spearman_ci95_high",
            "ci95_excludes_zero",
        ],
    )
    calibration_table = markdown_table(
        first_order_calibration[
            first_order_calibration["group"].isin(["All", "Adam", "Muon", "MatrixFactorizationInput", "MatrixSensing", "SmallMLPDigits"])
        ],
        [
            "group",
            "points",
            "positive_points",
            "spearman_delta_vs_first_order",
            "spearman_ci95_low",
            "spearman_ci95_high",
            "pearson_log_positive",
            "geomean_observed_over_first_order",
            "ratio_ci95_low",
            "ratio_ci95_high",
            "within_factor_2",
        ],
    )
    equal_calibration_table = markdown_table(
        equal_update_first_order_calibration[
            equal_update_first_order_calibration["group"].isin(
                ["All", "Adam", "Muon", "MatrixFactorizationInput", "MatrixSensing", "SmallMLPDigits"]
            )
        ]
        if equal_update_first_order_calibration is not None
        else pd.DataFrame(),
        [
            "group",
            "points",
            "positive_points",
            "spearman_delta_vs_first_order",
            "spearman_ci95_low",
            "spearman_ci95_high",
            "pearson_log_positive",
            "geomean_observed_over_first_order",
            "ratio_ci95_low",
            "ratio_ci95_high",
            "within_factor_2",
        ],
    )
    polar_alignment_table = markdown_table(
        polar_alignment,
        [
            "group",
            "points",
            "median_cosine_sq",
            "median_gradient_rank_fraction",
            "median_abs_identity_error",
            "max_abs_identity_error",
            "mean_abs_identity_error",
        ],
    )
    equal_polar_alignment_table = markdown_table(
        equal_update_polar_alignment if equal_update_polar_alignment is not None else pd.DataFrame(),
        [
            "group",
            "points",
            "median_cosine_sq",
            "median_gradient_rank_fraction",
            "median_abs_identity_error",
            "max_abs_identity_error",
            "mean_abs_identity_error",
        ],
    )
    first_order_pair_table = markdown_table(
        first_order_pair[
            first_order_pair["metric"].isin(
                [
                    "update_grad_inner",
                    "update_grad_cosine",
                    "update_grad_per_update_norm",
                    "delta_loss",
                    "relative_update_fro_norm",
                ]
            )
        ],
        [
            "metric",
            "problem_family",
            "n_pairs",
            "muon_higher_pairs",
            "geomean_ratio_muon_over_adam",
            "ratio_ci95_low",
            "ratio_ci95_high",
            "ratio_ci95_above_one",
        ],
    )
    equal_first_order_pair_table = markdown_table(
        equal_update_first_order_pair[
            equal_update_first_order_pair["metric"].isin(
                [
                    "update_grad_inner",
                    "update_grad_cosine",
                    "update_grad_per_update_norm",
                    "delta_loss",
                    "relative_update_fro_norm",
                ]
            )
        ]
        if equal_update_first_order_pair is not None
        else pd.DataFrame(),
        [
            "metric",
            "problem_family",
            "n_pairs",
            "muon_higher_pairs",
            "geomean_ratio_muon_over_adam",
            "ratio_ci95_low",
            "ratio_ci95_high",
            "ratio_ci95_above_one",
        ],
    )
    win_condition_source = equal_update_win_condition if equal_update_win_condition is not None else win_condition
    win_condition_table = markdown_table(
        win_condition_source[
            win_condition_source["condition"].isin(["muon_grad_rank_fraction", "delta_mean_grad_rank_fraction"])
            & win_condition_source["problem_family"].isin(["All", "MatrixFactorizationInput", "MatrixSensing", "SmallMLPDigits"])
        ]
        if win_condition_source is not None
        else pd.DataFrame(),
        [
            "condition",
            "problem_family",
            "bin",
            "bin_min",
            "bin_max",
            "n_pairs",
            "muon_first_order_win_rate",
            "muon_delta_loss_win_rate",
            "geomean_first_order_ratio_muon_over_adam",
            "first_order_ratio_ci95_low",
            "first_order_ratio_ci95_high",
        ],
    )
    win_generalization_source = equal_update_win_generalization if equal_update_win_generalization is not None else win_generalization
    win_generalization_table = markdown_table(
        win_generalization_source[
            (win_generalization_source["evaluation"] == "leave_family_out")
            & win_generalization_source["model"].isin(["spectrum_only", "spectrum_plus_family"])
        ]
        if win_generalization_source is not None
        else pd.DataFrame(),
        [
            "target",
            "model",
            "held_out_family",
            "test_pairs",
            "test_positive_rate",
            "mean_predicted_probability",
            "calibration_error",
            "auc",
            "accuracy",
            "brier",
            "baseline_brier",
        ],
    )
    win_overlap_source = equal_update_win_overlap if equal_update_win_overlap is not None else win_overlap
    win_overlap_table = markdown_table(
        win_overlap_source[
            (win_overlap_source["row_type"] == "family_summary")
            & win_overlap_source["held_out_family"].isin(["MatrixFactorizationInput", "MatrixSensing", "SmallMLPDigits"])
        ]
        if win_overlap_source is not None
        else pd.DataFrame(),
        [
            "held_out_family",
            "test_pairs",
            "mean_feature_outside_fraction",
            "frac_pairs_with_any_feature_outside",
            "median_min_standardized_distance",
            "p90_min_standardized_distance",
        ],
    )
    rel = lambda key: "../" + str(figure_paths[key])
    text = f"""# result for discussion

## Research Question

Does Muon act as a geometry-shaping optimizer that induces a quantitatively different spectral/rank geometry from Adam? When does that geometry explain one-step loss decrease or actual optimization progress?

For a stricter paper-facing summary of the current claims, evidence, and caveats, see [E11 research synthesis](e11_research_synthesis.md). That generated note is the recommended entry point before using the longer exploratory tables below.

## Experiment Scope

The evidence uses three problem families, treated as parallel experiments rather than primary/auxiliary cases:

1. **MF-with-input:** 10-factor matrix factorization with explicit input matrix, `kappa = 10 ... 1e5`, 5 seeds, 10 steps.
2. **Matrix Sensing:** direct matrix variable, same `kappa` sweep, 5 seeds, 5 steps.
3. **Small torch MLP:** sklearn digits classifier, 5 seeds, 10 steps.

For every trainable matrix or linear layer, \\(G_i = \\nabla_{{W_i}}L\\). For MF-with-input, \\(A_i = W_{{i+1}}\\cdots W_{{10}}Z\\). For Matrix Sensing, `A` is the measurement-operator proxy. For the MLP, training gradients are computed on a strict mini-batch (`train_batch_size < num_samples`), while `A` is the full-dataset input activation to each linear layer. Therefore MF-with-input is the strict theory-aligned activation-product case; the other two are problem-specific spectral diagnostics.

`delta_loss` is recorded as the same-batch pre/post-update decrease: the loss before the optimizer step minus the loss after applying that step, evaluated on the same training batch used for the gradient. This keeps the one-step decrease calibration aligned with `update_grad_inner` rather than with independent mini-batch noise.

The rank quantities are:

$$
nr(G_i)=\\frac{{\\|G_i\\|_*^2}}{{\\|G_i\\|_F^2}},\\qquad
sr(A_i)=\\frac{{\\|A_i\\|_F^2}}{{\\|A_i\\|_{{op}}^2}},\\qquad
C_i=\\frac{{nr(G_i)}}{{sr(A_i)}}.
$$

## Finding 1: Geometry Separation

**Finding:** Muon and Adam occupy measurably different rank-geometry regions in `(nrG, stA)`, but the direction and implication differ by problem family.

**Interpretation:** This supports the geometry-shaping claim more directly than an optimization-superiority claim.

**Reasoning:** Geometry-only separation is visible when paired `Muon - Adam` differences and nearest-centroid balanced accuracy stay away from zero/chance, but performance must be checked separately.

![Geometry separation]({rel("geometry_separation")})

{geometry_table}

{geometry_takeaway(geometry)}

## Finding 2: One-Step Decrease Prediction

**Finding:** Paper-style predicted decrease quantities are compared against observed \\(\\Delta L = L_t - L_{{t+1}}\\) in all three problem families.

**Interpretation:** The correlations test trend alignment, not exact equality, because Adam and Muon are not identical to the theoretical GD/Spec updates and `A` has problem-specific definitions outside MF.

**Reasoning:** If a predicted decrease is informative, larger predicted values should rank-order larger observed one-step loss decreases.

![Predicted decrease vs observed]({rel("predicted_decrease_vs_observed")})

{prediction_table}

## Finding 3: Geometry Volatility

**Finding:** Across the current lr sweep, Muon has lower step-to-step volatility in rank/condition diagnostics than Adam.

**Interpretation:** The raw cross-task optimizer signature is not a single direction such as "higher `nrG`" or "higher `stA`"; it is smoother movement in the diagnostic geometry.

**Reasoning:** The comparison is paired by problem setting, learning rate, and seed. The main statistic is the geometric mean ratio `Muon / Adam`; a 95% confidence interval entirely below 1 means Muon moves less in that diagnostic. The per-update rows divide each step's geometry movement by that same step's relative parameter-update size before averaging within a run.

![Volatility robustness]({rel("volatility_robustness")})

{volatility_table}

This result should still be read as a trajectory diagnostic, not a performance claim. The equal-update control below tests whether this volatility gap survives after Adam and Muon are forced to have the same global relative update norm at every matched step.

### Equal-Update Control

**Finding:** After per-step relative update norms are matched, the all-family Muon-vs-Adam rank/condition speed advantage disappears.

**Interpretation:** The strongest current evidence is that Muon changes the realized optimization path partly through update-scale control; a task-independent intrinsic rank-geometry smoothing claim is not yet supported.

**Reasoning:** In the control, Adam and Muon start from the same initialization for each setting/seed. Each step first computes the optimizer's proposed update direction, then rescales both optimizers to the smaller proposed global relative update norm. Therefore `median_relative_update_fro_norm` should have ratio 1, and remaining differences reflect direction/geometry rather than update size.

![Equal-update volatility robustness]({rel("equal_update_volatility_robustness")})

{equal_update_table}

## Finding 4: Update-Matrix Spectrum

**Finding:** The most optimizer-intrinsic spectral difference is in the update matrices themselves: Muon updates have higher effective rank, higher stable rank, and flatter singular spectra than Adam updates across the current tasks.

**Interpretation:** This is a stronger geometry-shaping statement than the raw trajectory-volatility claim, but it is partly by construction: ExactMuon applies the polar direction \\(UV^\\top\\), whose nonzero singular values are equal.

**Reasoning:** For every non-final step and layer, the code records singular values of \\(\\Delta W_i\\). The summary compares run-mean `nrUpdate`, `stUpdate`, their ceiling-normalized forms `nrUpdateFrac = nrUpdate / min(shape)` and `stUpdateFrac = stUpdate / min(shape)`, and `update_flatness = stUpdate / nrUpdate`. Ratios above 1 mean the Muon update matrix is spectrally more spread or closer to full-rank flatness.

![Update spectrum robustness]({rel("update_spectrum_robustness")})

{update_spectrum_table}

The equal-update control preserves this update-spectrum diagnostic because rescaling an update changes singular values by a scalar but does not change effective rank, stable rank, rank fractions, or flatness. The substantive takeaway is therefore not that the update spectrum is surprising, but that this construction-level update geometry does not automatically imply smoother state trajectories or better short-horizon performance.

![Equal-update update spectrum robustness]({rel("equal_update_update_spectrum_robustness")})

{equal_update_spectrum_table}

## Finding 5: Transmission From Update Geometry

**Finding:** The full-rank flat update spectrum is easy to detect, but its transmission to next-step loss decrease or state-geometry movement is not uniformly monotone across tasks and optimizers.

**Interpretation:** This narrows the open research question: Muon has a construction-level update geometry, but we still need conditions under which that update geometry becomes useful optimization progress.

**Reasoning:** The transmission table correlates current-step update diagnostics with next-step outcomes: relative loss decrease, normalized rank-plane movement, and normalized condition-score movement. It includes both spectrum-only diagnostics and first-order descent diagnostics such as `update_grad_inner = <G, W_t-W_{{t+1}}>`. Undefined correlations are expected when a predictor is nearly constant, as with Muon's rank fractions under exact polar updates.

![Update transmission heatmap]({rel("update_transmission_heatmap")})

{transmission_table}

In the current results, Muon's update-spectrum fractions and flatness are effectively constant within the optimizer, so they cannot explain within-Muon variation in progress. Relative update norm and the first-order gradient-update inner product are the more directly predictive step-level quantities. The same diagnostic under equal-update control checks whether these relations survive after update scale is matched.

![Equal-update transmission heatmap]({rel("equal_update_update_transmission_heatmap")})

{equal_transmission_table}

### First-Order Calibration

**Finding:** The gradient-update inner product is a strong one-step progress predictor, especially under equal-update control.

**Interpretation:** The immediate mechanism is not update flatness by itself; it is how the chosen update direction converts the current gradient into descent.

**Reasoning:** The scatter compares observed \\(\\Delta L\\) with \\(\\langle G, W_t-W_{{t+1}}\\rangle\\). The table reports rank correlation, log-log correlation on positive points, and how close the observed decrease is to the first-order term.

![First-order calibration]({rel("first_order_calibration")})

{calibration_table}

The equal-update version isolates direction/alignment from global update scale.

![Equal-update first-order calibration]({rel("equal_update_first_order_calibration")})

{equal_calibration_table}

### Polar Alignment Identity

**Finding:** For ExactMuon, the gradient-update cosine is determined by the gradient spectrum: \\(\\cos^2(G_i, \\Delta W_i) = nr(G_i)/r_i\\), up to numerical error.

**Interpretation:** This connects the construction-level flat update to the first-order progress mechanism. Muon does not merely make a flat update; its alignment with the gradient is controlled by the gradient effective-rank fraction.

**Reasoning:** ExactMuon uses the polar direction \\(UV^\\top\\). Therefore \\(\\langle G, UV^\\top\\rangle = \\|G\\|_*\\), \\(\\|UV^\\top\\|_F=\\sqrt{{r}}\\), and the squared cosine is \\(\\|G\\|_*^2/(\\|G\\|_F^2 r)=nr(G)/r\\).

![Polar alignment identity]({rel("polar_alignment_identity")})

{polar_alignment_table}

The equal-update control leaves this identity unchanged.

![Equal-update polar alignment identity]({rel("equal_update_polar_alignment_identity")})

{equal_polar_alignment_table}

### Matched First-Order Comparison

**Finding:** Under equal-update control, Muon does not uniformly dominate Adam in first-order descent; the comparison is task-dependent.

**Interpretation:** Polar geometry changes the alignment formula, but whether that produces a larger one-step descent than Adam depends on the gradient spectrum and the competing Adam direction.

**Reasoning:** The table compares matched setting/seed/step pairs. Ratios above 1 mean Muon has a larger first-order term, cosine, observed loss decrease, or update norm than Adam.

![First-order pair comparison]({rel("first_order_pair_comparison")})

{first_order_pair_table}

The equal-update version removes global update-scale differences, so remaining differences are directional.

![Equal-update first-order pair comparison]({rel("equal_update_first_order_pair_comparison")})

{equal_first_order_pair_table}

### Conditional Win Analysis

**Finding:** The current equal-update evidence does not support a task-independent rule such as "larger Muon gradient rank fraction always means larger one-step progress."

**Interpretation:** The more defensible research direction is conditional: Muon's polar update creates a clean spectral-alignment mechanism, but task structure determines whether that mechanism beats Adam's direction.

**Reasoning:** The table bins matched setting/seed/step pairs by gradient-rank fraction diagnostics. The reported win rates ask whether Muon has larger \\(\\langle G, W_t-W_{{t+1}}\\rangle\\) or observed \\(\\Delta L\\) than Adam within the same bin.

![Equal-update win-condition summary]({rel("equal_update_win_condition_summary")})

{win_condition_table}

The leave-family-out check asks whether a simple rule learned from two problem families predicts Muon wins in the held-out family. This is a direct stress test for cross-task generalization of the rank/spectral explanation.

![Equal-update win-prediction generalization]({rel("equal_update_win_prediction_generalization")})

{win_generalization_table}

The feature-overlap diagnostic checks whether the held-out family is interpolation or extrapolation relative to the two training families. Large outside-range fractions mean that poor leave-family-out prediction should be read as lack of current cross-task support, not as a fitted rule failing under interpolation.

![Equal-update feature support overlap]({rel("equal_update_win_feature_overlap")})

{win_overlap_table}

## Finding 6: Optimization Performance Is Separate

**Finding:** Geometry separation does not automatically imply better final loss, recovery error, or classification error.

**Interpretation:** The current claim should be that Muon changes spectral/rank geometry; whether that helps performance depends on problem family and short-horizon setting.

**Reasoning:** The performance table directly measures final outcomes, while the geometry table measures the path taken through diagnostics.

![Loss curves]({rel("loss_curves")})

{performance_table}

## Trajectory Evidence

![Condition score trajectories]({rel("condition_score_trajectories")})

![Mean 3D condition-loss dynamics]({rel("mean_3d_condition_loss")})

![Layerwise 3D condition-loss dynamics]({rel("layerwise_3d_condition_loss")})

## Next Experiment Candidate Scan

The current cross-task failure is largely an extrapolation problem: the three problem families occupy separated spectral supports. A lightweight initialization scan therefore searches for intermediate settings whose `nrG`, `stA`, condition score, and mean `nr(G_i)/r_i` sit inside the current global support. The scan output is kept separate from the main experiment because it is a candidate-selection diagnostic, not a trained-result claim.

![Initial geometry candidate scan]({rel("candidate_initial_geometry_support")})

See [E11 candidate overlap scan](e11_candidate_overlap_scan.md) for the ranked candidate settings.

An initial selected follow-up has also been run with equal-update control on a small number of overlap settings. See [E11 overlap follow-up](e11_overlap_followup.md). The short version is that Matrix Sensing still gives Muon a small first-order advantage, SmallMLPDigits becomes Muon-favorable after moving to a smaller hidden layer, and MF-with-input remains Adam-favorable in first-order descent.

The SmallMLP flip has been isolated with a width sweep. See [E11 SmallMLP width sweep](e11_mlp_width_sweep.md). In that sweep, hidden widths 8 and 16 are Muon-favorable, while widths 32, 64, and 128 are Adam-favorable under equal-update control. The layerwise diagnostic points to a first-layer bottleneck: the second layer stays mildly Muon-favorable, but as width grows Muon's first-layer `nr(G_i)/r_i` and gradient-update cosine fall sharply while Adam's first-layer cosine stays comparatively flat.

A layerwise hybrid test shows that this bottleneck is not fixed by simply assigning Adam to the first layer and Muon to the second. See [E11 SmallMLP layerwise hybrid test](e11_mlp_layer_hybrid.md). The hybrid results suggest that pure Muon uses a coupled two-layer update geometry in narrow networks; `AdamFirstMuonSecond` keeps the first layer closer to Adam but starves the second layer of equalized update budget, while `MuonFirstAdamSecond` allocates most update budget to the second layer but loses too much first-layer contribution. This makes the mechanism a layer-coupling effect, not a single-layer replacement rule.

A representative hyperparameter sweep has been run to check whether the current conclusions are artifacts of one learning-rate choice. See [E11 hyperparameter sweep](e11_hyperparam_sweep.md). The sweep keeps the update-spectrum claim intact under both raw and equal-update modes, but it still does not support a global optimization-superiority claim for Muon after best-lr selection. It reduces the learning-rate-artifact concern, but it is not an exhaustive tuning protocol.

A target-update-norm sweep has also been run to decouple update direction from update magnitude. See [E11 target-update-norm sweep](e11_target_update_sweep.md). At fixed global relative update norm, Muon's direction remains favorable for Matrix Sensing and SmallMLPDigits with hidden width 16, but unfavorable for MF-with-input and SmallMLPDigits with hidden width 64. This is the cleanest current evidence that the optimizer difference is not only a learning-rate artifact: the direction itself is task- and layer-condition dependent.

A SmallMLP per-layer update-control follow-up fixes each layer's relative update norm separately. See [E11 SmallMLP per-layer update control](e11_mlp_per_layer_control.md). This makes the hidden-64 Adam advantage robust to both global update-size and layer-allocation controls. The hidden-16 Muon advantage is more fragile: it appears at small per-layer targets, becomes near-neutral overall, and reverses at larger targets. This means the narrow-network advantage should be described as direction-plus-scale dependent, not as a pure direction-only effect.

The current intervention sequence is consolidated in [E11 mechanism ladder](e11_mechanism_ladder.md). The ladder separates raw performance, best-lr robustness, global update-size control, per-layer update-size control, within-layer spectral allocation, and the still-open singular-vector/trajectory question.

A within-layer spectral-allocation probe has now been run. See [E11 spectral allocation probe](e11_spectral_allocation_probe.md). It fixes gradient singular vectors and changes only singular-value allocation under either Frobenius or operator-norm budgets. The result is sharp: flat/polar allocation loses to GD-spectrum allocation under a fixed Frobenius budget, but wins under a fixed operator-norm budget. This makes the Muon mechanism more precise: polar updates are not universally better directions; they are favorable under an operator-norm-like geometry when spreading update mass across singular directions is useful.

A singular-vector trajectory diagnostic has also been run. See [E11 singular-vector trajectory diagnostic](e11_singular_vector_trajectory.md). MF-with-input keeps high Adam/Muon gradient-subspace overlap through the short horizon, so MF is closer to a shared-geometry singular-value-allocation story. Matrix Sensing and SmallMLP show much stronger gradient/parameter subspace divergence, so their optimizer difference includes trajectory-level singular-vector geometry.

A singular-vector swap probe strengthens that diagnostic. See [E11 singular-vector swap probe](e11_singular_vector_swap_probe.md). At a fixed Adam or Muon state, replacing the state's own gradient polar singular vectors with the other optimizer's matched-state singular vectors systematically reduces one-step progress; in Matrix Sensing, the swapped vectors often become ascent directions. This suggests the trajectory-level singular-vector geometry is relevant to local descent, not just a visualization artifact.

A natural update-vector swap probe has also been run. See [E11 natural update-vector swap probe](e11_natural_update_swap_probe.md). This probe extracts each optimizer's actual proposed update vector by temporarily stepping and restoring the optimizer state, then evaluates own-update vs other-update directions at the same state under matched per-layer Frobenius or operator-norm budgets across five target scales. Overall, the other update gives lower one-step progress, including for observed `delta_loss`, but the result is not monotone across settings, budgets, and evaluation states. This makes the interpretation sharper: Adam/Muon update vectors are locally consequential, but the evidence does not support a universal statement that each optimizer's own update is always best.

A trajectory-level optimizer switch probe gives an important negative control. See [E11 optimizer switch probe](e11_optimizer_switch_probe.md). After taking Adam or Muon to a checkpoint, the remaining short horizon is continued either with the original optimizer state or by switching to the other optimizer. This does not support a simple own-optimizer continuation story: switching from Muon checkpoints to Adam often improves the remaining-horizon outcome, while switching from Adam checkpoints to Muon usually hurts.

A reset-control version of the switch probe reduces the optimizer-state-reset confound. See [E11 optimizer switch reset-control probe](e11_optimizer_switch_reset_control.md). It compares `own_preserved`, `own_fresh`, and `switched_fresh` from the same checkpoint. Adam state reset does hurt Adam continuations, but it does not explain away the qualitative pattern: from Muon checkpoints, fresh Adam often still gives lower final loss than fresh Muon. The current thesis should therefore separate local update-vector geometry from longer-horizon optimizer performance.

A fresh-continuation horizon sweep further sharpens the trajectory-level caveat. See [E11 optimizer switch horizon sweep](e11_optimizer_switch_horizon_sweep.md). Both continuations use fresh optimizer state, and the continuation horizon is varied over 1, 3, 10, and 30 steps. The switch effect is not horizon-invariant: aggregate switching is worse at horizon 10, while Muon-source checkpoints become Adam-favorable again at horizon 30. This means the trajectory-level story is source-, task-, and horizon-dependent rather than a direct consequence of the one-step update geometry.

A continuation-learning-rate sweep checks whether the horizon-30 result is just a fixed-LR artifact. See [E11 optimizer switch continuation-LR sweep](e11_optimizer_switch_lr_sweep.md). Each fresh continuation gets its best final loss over a small LR grid. The result remains non-monotone: Matrix Sensing and SmallMLP prefer opposite switch directions, and aggregate ratios are sensitive to task-family scale. Thus the trajectory-level evidence should be used mainly as a warning against overclaiming from one-step geometry, not as a positive universal switching rule.

## Caveats

1. MF-with-input has the strict theory-aligned activation product; Matrix Sensing and MLP use problem-specific `A` diagnostics.
2. The horizons are short: 10 steps for MF and MLP, 5 steps for Matrix Sensing.
3. Larger `nrG` or `stA` should not be interpreted as better optimization unless it is tied to loss, recovery, or classification metrics.
4. The MLP experiment is a small torch sanity benchmark on sklearn digits, not a broad neural-network benchmark.
5. The hyperparameter sweep is representative and oracle-style, not exhaustive.
6. The target-update-norm sweep controls global update size, not per-layer update allocation.
7. The per-layer update-control follow-up only covers two SmallMLP widths and does not control within-layer spectral allocation.
8. The spectral-allocation probe is an artificial one-step probe that fixes gradient singular vectors.
9. The singular-vector and natural-update swap probes are artificial one-step interventions.
10. The optimizer-switch probes are trajectory-level but still use fresh-continuation interventions; the horizon and continuation-LR sweeps show the result itself is horizon-, task-, and tuning-dependent.
"""
    path.write_text(text.rstrip() + "\n", encoding="utf-8")
