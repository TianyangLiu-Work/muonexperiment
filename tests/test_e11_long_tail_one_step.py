from e11_condition_geometry.long_tail_one_step import LongTailOneStepConfig, run_long_tail_one_step


def test_long_tail_one_step_smoke():
    steps, summary, layers = run_long_tail_one_step(
        LongTailOneStepConfig(
            seeds=(0, 1),
            warmup_steps=5,
            head_train_per_class=80,
            tail_train_per_class=10,
            tail_eval_per_class=20,
            head_batch_size=32,
            warmup_batch_size=32,
            hidden_dim=16,
        )
    )

    assert len(steps) == 4
    assert set(steps["geometry"]) == {"frobenius", "spectral"}
    assert len(layers) == 8
    assert int(summary.iloc[0]["seeds"]) == 2
    assert summary.iloc[0]["spectral_less_tail_output_drift_fraction"] == 1.0
    assert summary.iloc[0]["geomean_tail_output_drift_sq_ratio_spectral_over_fro"] < 1.0
    assert steps["matched_first_order_head_gain"].nunique() == 2
    assert steps["tail_eval_examples"].iloc[0] == 100
