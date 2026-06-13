from e11_condition_geometry.long_tail_layerwise import LongTailLayerwiseConfig, run_long_tail_layerwise


def test_long_tail_layerwise_smoke():
    metrics, summary = run_long_tail_layerwise(
        LongTailLayerwiseConfig(
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

    assert len(metrics) == 8
    assert set(metrics["geometry"]) == {"frobenius", "spectral"}
    assert set(metrics["layer"]) == {1, 2}
    assert len(summary) == 2
    assert (summary["geomean_scaled_jvp_tail_drift_sq_ratio_spectral_over_fro"] < 1.0).all()
    assert (summary["geomean_observed_tail_drift_sq_ratio_spectral_over_fro"] < 1.0).all()
