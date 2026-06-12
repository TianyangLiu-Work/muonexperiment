from dataclasses import replace

from e11_condition_geometry.config import ExperimentConfig, default_config
from e11_condition_geometry.runner import run_experiment


def test_runner_smoke_all_problem_families():
    base = default_config()
    specs = []
    seen = set()
    for spec in base.specs:
        if spec.family in seen:
            continue
        seen.add(spec.family)
        if spec.family == "SmallMLPDigits":
            specs.append(replace(spec, steps=1, hidden_dim=8, num_samples=64))
        else:
            specs.append(replace(spec, steps=1, d=12, rank=3))
    config = ExperimentConfig(
        seeds=(0,),
        algos=("Adam", "Muon"),
        dtype="float64",
        device="cpu",
        output_dir=base.output_dir,
        figure_dir=base.figure_dir,
        discussion_path=base.discussion_path,
        specs=tuple(specs),
    )
    steps, layers = run_experiment(config)
    assert set(steps["problem_family"]) == {"MatrixFactorizationInput", "MatrixSensing", "SmallMLPDigits"}
    assert steps["run_id"].nunique() == 6
    assert set(["loss", "delta_loss", "recovery_error", "train_batch_size", "nrG", "stA", "condition_score"]).issubset(steps.columns)
    assert not steps[["loss", "recovery_error", "nrG", "stA", "condition_score"]].isna().any().any()
    mlp_steps = steps[steps["problem_family"] == "SmallMLPDigits"]
    assert set(mlp_steps["train_batch_size"]) == {64}
    non_mlp_steps = steps[steps["problem_family"] != "SmallMLPDigits"]
    assert (non_mlp_steps["train_batch_size"] == non_mlp_steps["num_samples"]).all()
    nonfinal = steps[steps["delta_loss"].notna()]
    assert not nonfinal[["nrUpdate", "stUpdate", "nrUpdateFrac", "stUpdateFrac", "update_flatness"]].isna().any().any()
    assert not nonfinal[["update_grad_inner", "update_grad_cosine", "update_grad_per_update_norm"]].isna().any().any()
    assert (nonfinal["sigma_update"].fillna("") != "").all()
    assert not layers.empty
