from pathlib import Path

from e11_condition_geometry.paper_stats import load_paper_stats


def test_load_paper_stats_exposes_current_head_tail_anchors() -> None:
    root = Path(__file__).resolve().parents[1]
    stats = load_paper_stats(root)

    assert bool(stats.synthetic_positive["predicted_spectral_less_drift"])
    assert not bool(stats.synthetic_negative["predicted_spectral_less_drift"])
    assert (
        stats.synthetic_positive["mean_tail_downstream_aware_stable_rank"]
        < stats.synthetic_positive["mean_head_gradient_nuclear_rank"]
    )
    assert (
        stats.synthetic_negative["mean_tail_downstream_aware_stable_rank"]
        > stats.synthetic_negative["mean_head_gradient_nuclear_rank"]
    )
    assert stats.synthetic_positive["tail_output_drift_sq_ratio_ci95_high"] < 1.0
    assert stats.synthetic_negative["tail_output_drift_sq_ratio_ci95_low"] > 1.0

    assert int(stats.long_tail_one_step["seeds"]) == 20
    assert stats.long_tail_one_step["tail_output_drift_sq_ratio_ci95_high"] < 1.0
    assert stats.long_tail_one_step["spectral_less_tail_output_drift_fraction"] == 1.0
    assert stats.long_tail_one_step["mean_tail_loss_increase_diff_spectral_minus_fro"] > 0.0

    assert stats.muon_bridge_polar_momentum["tail_output_drift_sq_ratio_vs_fro_ci95_high"] < 1.0
    assert stats.muon_bridge_polar_momentum["mean_direction_cosine_to_polar_grad"] < 1.0
    assert stats.muon_bridge_ns_momentum["geomean_tail_output_drift_sq_ratio_vs_fro"] < 1.0
    assert stats.muon_bridge_ns_momentum["tail_output_drift_sq_ratio_vs_fro_ci95_high"] > 1.0

    assert int(stats.practical_muon_bridge_polar_momentum["comparisons"]) == 120
    assert stats.practical_muon_bridge_polar_momentum["tail_output_drift_sq_ratio_vs_fro_ci95_high"] < 1.0
    assert stats.practical_muon_bridge_ns_momentum["tail_output_drift_sq_ratio_vs_fro_ci95_high"] < 1.0

    assert int(stats.practical_training["train_steps"]) == 80
    assert stats.practical_training["final_train_loss_ratio_ci95_high"] < 1.0
    assert stats.practical_training["final_head_loss_ratio_ci95_high"] < 1.0
    assert stats.practical_training["final_tail_eval_loss_ratio_ci95_high"] < 1.0
    assert stats.practical_training["final_tail_eval_drift_rms_ratio_ci95_high"] < 1.0
    assert stats.practical_training["mean_final_tail_eval_accuracy_diff_muon_minus_adam"] == 0.0

    assert int(stats.long_tail_forgetting["head_only_steps"]) == 8
    assert stats.long_tail_forgetting["final_tail_output_drift_sq_ratio_ci95_high"] < 1.0
    assert stats.long_tail_forgetting["tail_output_drift_area_ratio_ci95_high"] < 1.0
    assert stats.long_tail_forgetting["final_tail_loss_increase_diff_ci95_low"] < 0.0
    assert stats.long_tail_forgetting["final_tail_loss_increase_diff_ci95_high"] > 0.0

    for layer in [stats.layer_one, stats.layer_two]:
        assert layer["jvp_tail_drift_sq_ratio_ci95_low"] > 1.0
        assert layer["scaled_jvp_tail_drift_sq_ratio_ci95_high"] < 1.0
        assert layer["observed_tail_drift_sq_ratio_ci95_high"] < 1.0
