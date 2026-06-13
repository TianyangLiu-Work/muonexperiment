from e11_condition_geometry.long_tail_forgetting import LongTailForgettingConfig, run_long_tail_forgetting


def test_long_tail_forgetting_smoke():
    steps, summary = run_long_tail_forgetting(
        LongTailForgettingConfig(
            seeds=(0, 1),
            warmup_steps=5,
            head_only_steps=3,
            head_train_per_class=80,
            tail_train_per_class=10,
            tail_eval_per_class=20,
            head_batch_size=32,
            warmup_batch_size=32,
            hidden_dim=16,
        )
    )

    assert len(steps) == 16
    assert set(steps["geometry"]) == {"frobenius", "spectral"}
    assert set(steps["step"]) == {0, 1, 2, 3}
    assert int(summary.iloc[0]["seeds"]) == 2
    assert int(summary.iloc[0]["head_only_steps"]) == 3
    assert summary.iloc[0]["geomean_final_tail_output_drift_sq_ratio_spectral_over_fro"] < 1.0
    assert summary.iloc[0]["geomean_tail_output_drift_area_ratio_spectral_over_fro"] < 1.0
