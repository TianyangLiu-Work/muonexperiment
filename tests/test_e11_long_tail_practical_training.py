from e11_condition_geometry.long_tail_practical_training import (
    LongTailPracticalTrainingConfig,
    LongTailPracticalTrainingLRSweepConfig,
    run_long_tail_practical_training,
    run_long_tail_practical_training_lr_sweep,
)


def test_long_tail_practical_training_smoke():
    steps, summary = run_long_tail_practical_training(
        LongTailPracticalTrainingConfig(
            seeds=(0, 1),
            head_train_per_class=80,
            tail_train_per_class=10,
            tail_eval_per_class=20,
            hidden_dim=16,
            train_steps=4,
            train_batch_size=32,
            adam_lr=1e-2,
            muon_lr=5e-4,
            newton_schulz_steps=4,
        )
    )

    assert len(steps) == 20
    assert set(steps["optimizer"]) == {"adam", "ns_muon"}
    assert set(steps["step"]) == {0, 1, 2, 3, 4}
    assert len(summary) == 1
    row = summary.iloc[0]
    assert int(row["seeds"]) == 2
    assert int(row["train_steps"]) == 4
    assert row["geomean_final_train_loss_ratio_muon_over_adam"] > 0
    assert row["geomean_final_tail_eval_loss_ratio_muon_over_adam"] > 0


def test_long_tail_practical_training_lr_sweep_smoke():
    sweep = run_long_tail_practical_training_lr_sweep(
        LongTailPracticalTrainingLRSweepConfig(
            muon_lrs=(1e-3, 1e-2),
            base_config=LongTailPracticalTrainingConfig(
                seeds=(0,),
                head_train_per_class=80,
                tail_train_per_class=10,
                tail_eval_per_class=20,
                hidden_dim=16,
                train_steps=2,
                train_batch_size=32,
                newton_schulz_steps=4,
            ),
        )
    )

    assert list(sweep["muon_lr"]) == [1e-3, 1e-2]
    assert (sweep["seeds"] == 1).all()
    assert (sweep["train_steps"] == 2).all()
    assert (sweep["geomean_final_train_loss_ratio_muon_over_adam"] > 0).all()
