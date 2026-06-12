from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProblemSpec:
    family: str
    setting: str
    steps: int
    d: int = 60
    rank: int = 5
    kappa: float = 1.0
    num_factors: int = 10
    input_columns_multiplier: int = 10
    measurement_multiplier: float = 2.0
    input_dim: int = 64
    hidden_dim: int = 64
    output_dim: int = 10
    num_samples: int = 1024
    batch_size: int | None = 128
    lr: float = 1e-2


@dataclass(frozen=True)
class ExperimentConfig:
    seeds: tuple[int, ...]
    algos: tuple[str, ...]
    dtype: str
    device: str
    output_dir: Path
    figure_dir: Path
    discussion_path: Path
    specs: tuple[ProblemSpec, ...]

    def to_dict(self) -> dict:
        data = asdict(self)
        data["output_dir"] = str(self.output_dir)
        data["figure_dir"] = str(self.figure_dir)
        data["discussion_path"] = str(self.discussion_path)
        return data


def default_config() -> ExperimentConfig:
    kappas = tuple(10.0**power for power in range(1, 6))
    learning_rates = (3e-3, 1e-2, 3e-2)
    mf_specs = tuple(
        ProblemSpec(
            family="MatrixFactorizationInput",
            setting=f"MF input kappa={kappa:.0e} lr={lr:.0e}",
            steps=10,
            d=60,
            rank=5,
            kappa=kappa,
            num_factors=10,
            input_columns_multiplier=10,
            lr=lr,
        )
        for kappa in kappas
        for lr in learning_rates
    )
    ms_specs = tuple(
        ProblemSpec(
            family="MatrixSensing",
            setting=f"Matrix sensing kappa={kappa:.0e} lr={lr:.0e}",
            steps=5,
            d=60,
            rank=5,
            kappa=kappa,
            measurement_multiplier=2.0,
            lr=lr,
        )
        for kappa in kappas
        for lr in learning_rates
    )
    mlp_specs = (
        *(
            ProblemSpec(
                family="SmallMLPDigits",
                setting=f"Small MLP digits lr={lr:.0e}",
                steps=10,
                d=64,
                rank=10,
                kappa=1.0,
                hidden_dim=64,
                num_samples=1024,
                lr=lr,
            )
            for lr in learning_rates
        ),
    )
    return ExperimentConfig(
        seeds=tuple(range(5)),
        algos=("Adam", "Muon"),
        dtype="float64",
        device="cpu",
        output_dir=Path("results/e11"),
        figure_dir=Path("figures/e11"),
        discussion_path=Path("discussion/e11_result_for_discussion.md"),
        specs=mf_specs + ms_specs + mlp_specs,
    )
