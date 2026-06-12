from pathlib import Path

from e11_condition_geometry.paper_stats import load_paper_stats


def test_load_paper_stats_exposes_paper_facing_anchors() -> None:
    root = Path(__file__).resolve().parents[1]
    stats = load_paper_stats(root)

    assert stats.nr_update["metric"] == "nrUpdate"
    assert stats.st_update["metric"] == "stUpdate"
    assert stats.nr_update["geomean_ratio_muon_over_adam"] > 1.0
    assert stats.st_update["geomean_ratio_muon_over_adam"] > 1.0
    assert stats.spectral_fro["geomean_ratio"] < 1.0
    assert stats.spectral_op["geomean_ratio"] > 1.0
    assert stats.patch_nr["geomean_ratio_muon_over_adam"] > 1.0
    assert stats.patch_st["geomean_ratio_muon_over_adam"] > 1.0
    assert stats.patch_first["geomean_ratio_muon_over_adam"] < 1.0
    assert stats.conv_nr["geomean_ratio_muon_over_adam"] > 1.0
    assert stats.conv_st["geomean_ratio_muon_over_adam"] > 1.0
    assert stats.conv_first["geomean_ratio_muon_over_adam"] < 1.0
    assert stats.boundary_counts.muon_flat_favorable == 4
    assert stats.boundary_counts.own_update_positive_control == 1
    assert stats.boundary_counts.favorable == 5
    assert stats.boundary_counts.unfavorable == 8
    assert stats.boundary_counts.mixed == 1
    assert stats.predictor_best["feature_set"] == "state_plus_update_spectrum"
    assert stats.predictor_uncertainty_best["balanced_accuracy_ci95_low"] <= 0.5
    assert not bool(stats.predictor_uncertainty_best["balanced_accuracy_ci95_above_chance"])
