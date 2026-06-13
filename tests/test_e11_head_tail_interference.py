from e11_condition_geometry.head_tail_interference import HeadTailConfig, run_head_tail_interference


def test_head_tail_interference_boundary_smoke():
    _, summary = run_head_tail_interference(
        HeadTailConfig(seeds=(0, 1), output_dim=6, input_dim=10, tail_batch_size=20)
    )
    by_setting = summary.set_index("setting")

    positive = by_setting.loc["high_head_rank_low_tail_srank"]
    assert positive["predicted_spectral_less_drift"]
    assert positive["mean_tail_downstream_aware_stable_rank"] < positive["mean_head_gradient_nuclear_rank"]
    assert positive["mean_theory_ratio_spectral_over_fro"] < 1.0
    assert positive["mean_tail_output_drift_sq_ratio_spectral_over_fro"] < 1.0
    assert positive["tail_output_drift_sq_ratio_ci95_high"] < 1.0
    assert positive["spectral_less_tail_output_drift_fraction"] == 1.0

    negative = by_setting.loc["low_head_rank_high_tail_srank"]
    assert not negative["predicted_spectral_less_drift"]
    assert negative["mean_tail_downstream_aware_stable_rank"] > negative["mean_head_gradient_nuclear_rank"]
    assert negative["mean_theory_ratio_spectral_over_fro"] > 1.0
    assert negative["mean_tail_output_drift_sq_ratio_spectral_over_fro"] > 1.0
    assert negative["tail_output_drift_sq_ratio_ci95_low"] > 1.0
    assert negative["spectral_less_tail_output_drift_fraction"] == 0.0
