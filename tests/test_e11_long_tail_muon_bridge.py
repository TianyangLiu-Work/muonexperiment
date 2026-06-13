from e11_condition_geometry.long_tail_muon_bridge import (
    LongTailMuonBridgeConfig,
    LongTailPracticalMuonBridgeConfig,
    run_long_tail_muon_bridge,
    run_long_tail_practical_muon_bridge,
)


def test_long_tail_muon_bridge_smoke():
    steps, summary = run_long_tail_muon_bridge(
        LongTailMuonBridgeConfig(
            seeds=(0, 1),
            warmup_steps=5,
            head_train_per_class=80,
            tail_train_per_class=10,
            tail_eval_per_class=20,
            head_batch_size=32,
            warmup_batch_size=32,
            hidden_dim=16,
            momentum_history_steps=3,
            newton_schulz_steps=4,
        )
    )

    assert len(steps) == 8
    assert set(steps["direction"]) == {"frobenius_grad", "polar_grad", "polar_momentum", "ns_momentum"}
    assert set(summary["direction"]) == {"polar_grad", "polar_momentum", "ns_momentum"}
    assert int(summary["seeds"].iloc[0]) == 2
    assert (summary["geomean_tail_output_drift_sq_ratio_vs_fro"] > 0).all()
    assert (summary["mean_direction_cosine_to_polar_grad"] <= 1.0 + 1e-8).all()
    assert (summary["geomean_alignment_ratio_to_polar_grad"] > 0).all()


def test_long_tail_practical_muon_bridge_smoke():
    steps, summary = run_long_tail_practical_muon_bridge(
        LongTailPracticalMuonBridgeConfig(
            seeds=(0, 1),
            warmup_steps=5,
            head_train_per_class=80,
            tail_train_per_class=10,
            tail_eval_per_class=20,
            head_batch_size=32,
            warmup_batch_size=32,
            hidden_dim=16,
            trajectory_steps=3,
            trajectory_lr=5e-4,
            newton_schulz_steps=4,
        )
    )

    assert len(steps) == 24
    assert set(steps["direction"]) == {"frobenius_grad", "polar_grad", "polar_momentum", "ns_momentum"}
    assert set(steps["trajectory_step"]) == {0, 1, 2}
    assert set(summary["direction"]) == {"polar_grad", "polar_momentum", "ns_momentum"}
    assert int(summary["seeds"].iloc[0]) == 2
    assert int(summary["trajectory_steps"].iloc[0]) == 3
    assert int(summary["comparisons"].iloc[0]) == 6
    assert (summary["geomean_tail_output_drift_sq_ratio_vs_fro"] > 0).all()
    assert (summary["mean_direction_cosine_to_polar_grad"] <= 1.0 + 1e-8).all()
