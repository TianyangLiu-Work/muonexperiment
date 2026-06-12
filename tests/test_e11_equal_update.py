from dataclasses import replace

import numpy as np

from e11_condition_geometry.config import ExperimentConfig, ProblemSpec, default_config
from e11_condition_geometry.runner import run_equal_update_experiment


def test_equal_update_control_matches_relative_update_norms() -> None:
    base = default_config()
    spec = ProblemSpec(
        family="MatrixFactorizationInput",
        setting="mf equal update smoke",
        steps=2,
        d=12,
        rank=3,
        kappa=10.0,
        num_factors=3,
        lr=1e-2,
    )
    config = ExperimentConfig(
        seeds=(0,),
        algos=("Adam", "Muon"),
        dtype=base.dtype,
        device=base.device,
        output_dir=base.output_dir,
        figure_dir=base.figure_dir,
        discussion_path=base.discussion_path,
        specs=(spec,),
    )
    steps, _ = run_equal_update_experiment(config)
    nonfinal = steps[np.isfinite(steps["relative_update_fro_norm"])]
    pivot = nonfinal.pivot_table(
        index=["setting", "seed", "step"],
        columns="algo",
        values="relative_update_fro_norm",
        aggfunc="mean",
    )
    assert set(pivot.columns) == {"Adam", "Muon"}
    assert np.allclose(pivot["Adam"], pivot["Muon"], rtol=1e-9, atol=1e-12)
