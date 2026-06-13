from __future__ import annotations

import shutil
from pathlib import Path


PAPER_FIGURE_SOURCES: tuple[tuple[Path, Path], ...] = (
    (
        Path("figures/e11_head_tail_interference/head_tail_drift_ratio.png"),
        Path("paper/specgrad_activation_paper/figures/head_tail_drift_ratio.png"),
    ),
    (
        Path("figures/e11_long_tail_one_step/long_tail_one_step_tail_response.png"),
        Path("paper/specgrad_activation_paper/figures/long_tail_one_step_tail_response.png"),
    ),
    (
        Path("figures/e11_long_tail_muon_bridge/long_tail_muon_bridge.png"),
        Path("paper/specgrad_activation_paper/figures/long_tail_muon_bridge.png"),
    ),
    (
        Path("figures/e11_long_tail_practical_muon_bridge/long_tail_practical_muon_bridge.png"),
        Path("paper/specgrad_activation_paper/figures/long_tail_practical_muon_bridge.png"),
    ),
    (
        Path("figures/e11_long_tail_practical_training/long_tail_practical_training.png"),
        Path("paper/specgrad_activation_paper/figures/long_tail_practical_training.png"),
    ),
    (
        Path("figures/e11_long_tail_forgetting/long_tail_head_only_forgetting.png"),
        Path("paper/specgrad_activation_paper/figures/long_tail_head_only_forgetting.png"),
    ),
    (
        Path("figures/e11_long_tail_layerwise/long_tail_layerwise_drift.png"),
        Path("paper/specgrad_activation_paper/figures/long_tail_layerwise_drift.png"),
    ),
)


def main() -> None:
    for source, destination in PAPER_FIGURE_SOURCES:
        if not source.exists():
            raise FileNotFoundError(f"missing paper figure source: {source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    print(f"saved {len(PAPER_FIGURE_SOURCES)} paper-local figures")


if __name__ == "__main__":
    main()
