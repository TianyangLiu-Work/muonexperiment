from __future__ import annotations

import argparse
import gc
import json
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.cifar100_resnet_tail import (
    Cifar100ResNetOneStepConfig,
    run_cifar100_resnet_one_step,
)
from e11_condition_geometry.reporting import markdown_table, write_markdown


RESULT_DIR = Path("results/e11_natural_negative_search_protocol")
DEFAULT_SEEDS = tuple(range(5))
PHASE1_SEARCH_IDS = (
    "NNS-P1-cifar100lt-resnet18-new-partitions",
    "NNS-P1-cifar10lt-resnet18-cross-partitions",
    "NNS-P1-tail-quality-controls",
)
SEARCH_OUTPUT_PREFIXES = {
    "NNS-P1-cifar100lt-resnet18-new-partitions": RESULT_DIR / "phase1_cifar100lt_resnet18",
    "NNS-P1-cifar10lt-resnet18-cross-partitions": RESULT_DIR / "phase1_cifar10lt_resnet18",
    "NNS-P1-tail-quality-controls": RESULT_DIR / "phase1_tail_quality_controls",
}
MULTIPLICITY_FAMILY = "NNS-P1-fresh-primary-family"
PROTOCOL_VERSION = "nns_phase1_runner_v1"


@dataclass(frozen=True)
class NaturalSearchSetting:
    search_id: str
    setting_id: str
    phase: str
    dataset_label: str
    dataset_name: str
    model_arch: str
    partition_id: str
    head_classes: tuple[int, ...]
    tail_classes: tuple[int, ...]
    head_train_per_class: int
    tail_train_per_class: int
    tail_eval_per_class: int
    warmup_steps: int
    target_head_gain_fraction: float
    freshness_rule: str
    planned_artifact_prefix: Path


def rho_tag(value: float) -> str:
    return f"{value:.4g}".replace(".", "p")


def class_csv(values: tuple[int, ...]) -> str:
    return ",".join(str(value) for value in values)


def by_mod(modulus: int, residues: set[int], *, num_classes: int = 100) -> tuple[int, ...]:
    return tuple(label for label in range(num_classes) if label % modulus in residues)


def cifar100_decade_parity(parity: int) -> tuple[int, ...]:
    return tuple(label for label in range(100) if (label // 10) % 2 == parity)


def make_setting(
    *,
    search_id: str,
    dataset_label: str,
    dataset_name: str,
    partition_id: str,
    head_classes: tuple[int, ...],
    tail_classes: tuple[int, ...],
    head_train_per_class: int,
    tail_train_per_class: int,
    tail_eval_per_class: int,
    warmup_steps: int,
    target_head_gain_fraction: float,
    freshness_rule: str,
) -> NaturalSearchSetting:
    setting_id = (
        f"{partition_id}_w{warmup_steps}_rho{rho_tag(target_head_gain_fraction)}"
        f"_tail{tail_train_per_class}"
    )
    return NaturalSearchSetting(
        search_id=search_id,
        setting_id=setting_id,
        phase="phase1_fresh_primary_search",
        dataset_label=dataset_label,
        dataset_name=dataset_name,
        model_arch="resnet18",
        partition_id=partition_id,
        head_classes=head_classes,
        tail_classes=tail_classes,
        head_train_per_class=head_train_per_class,
        tail_train_per_class=tail_train_per_class,
        tail_eval_per_class=tail_eval_per_class,
        warmup_steps=warmup_steps,
        target_head_gain_fraction=target_head_gain_fraction,
        freshness_rule=freshness_rule,
        planned_artifact_prefix=SEARCH_OUTPUT_PREFIXES[search_id],
    )


def cifar100_fresh_partition_settings() -> list[NaturalSearchSetting]:
    search_id = "NNS-P1-cifar100lt-resnet18-new-partitions"
    partitions = [
        (
            "cifar100_mod5_01_vs_34",
            by_mod(5, {0, 1}),
            by_mod(5, {3, 4}),
        ),
        (
            "cifar100_decade_even_vs_odd",
            cifar100_decade_parity(0),
            cifar100_decade_parity(1),
        ),
    ]
    return [
        make_setting(
            search_id=search_id,
            dataset_label="CIFAR-100-LT",
            dataset_name="CIFAR100",
            partition_id=partition_id,
            head_classes=head_classes,
            tail_classes=tail_classes,
            head_train_per_class=300,
            tail_train_per_class=30,
            tail_eval_per_class=40,
            warmup_steps=warmup_steps,
            target_head_gain_fraction=rho,
            freshness_rule=(
                "exclude every source_id and exact setting_id in "
                "results/e11_natural_head_tail_boundary/primary_drift_scan.csv"
            ),
        )
        for partition_id, head_classes, tail_classes in partitions
        for warmup_steps in (500, 2000, 5000)
        for rho in (0.002, 0.005)
    ]


def cifar10_cross_partition_settings() -> list[NaturalSearchSetting]:
    search_id = "NNS-P1-cifar10lt-resnet18-cross-partitions"
    partitions = [
        ("cifar10_cross_a", (0, 2, 5, 8, 9), (1, 3, 4, 6, 7)),
        ("cifar10_cross_b", (1, 4, 5, 7, 9), (0, 2, 3, 6, 8)),
    ]
    return [
        make_setting(
            search_id=search_id,
            dataset_label="CIFAR-10-LT",
            dataset_name="CIFAR10",
            partition_id=partition_id,
            head_classes=head_classes,
            tail_classes=tail_classes,
            head_train_per_class=300,
            tail_train_per_class=30,
            tail_eval_per_class=40,
            warmup_steps=warmup_steps,
            target_head_gain_fraction=rho,
            freshness_rule=(
                "do not reuse v2/v3/v4/v5 condition-score final partitions for "
                "natural-negative selection"
            ),
        )
        for partition_id, head_classes, tail_classes in partitions
        for warmup_steps in (500, 5000)
        for rho in (0.002, 0.005)
    ]


def tail_quality_control_settings() -> list[NaturalSearchSetting]:
    search_id = "NNS-P1-tail-quality-controls"
    return [
        make_setting(
            search_id=search_id,
            dataset_label="CIFAR-100",
            dataset_name="CIFAR100",
            partition_id="cifar100_tail_quality_mod5_12_vs_04",
            head_classes=by_mod(5, {1, 2}),
            tail_classes=by_mod(5, {0, 4}),
            head_train_per_class=300,
            tail_train_per_class=300,
            tail_eval_per_class=40,
            warmup_steps=warmup_steps,
            target_head_gain_fraction=rho,
            freshness_rule="exclude committed tail_quality_control warmup/partition rows",
        )
        for warmup_steps in (2000, 5000, 10000)
        for rho in (0.002, 0.005)
    ]


def all_settings() -> list[NaturalSearchSetting]:
    settings = (
        cifar100_fresh_partition_settings()
        + cifar10_cross_partition_settings()
        + tail_quality_control_settings()
    )
    setting_ids = [setting.setting_id for setting in settings]
    if len(setting_ids) != len(set(setting_ids)):
        raise AssertionError("natural search phase1 setting ids must be unique")
    return settings


def settings_for_search(search_id: str) -> list[NaturalSearchSetting]:
    settings = all_settings()
    if search_id == "all_phase1":
        return settings
    selected = [setting for setting in settings if setting.search_id == search_id]
    if not selected:
        raise ValueError(f"unknown phase1 search id: {search_id}")
    return selected


def planned_phase1_family_size() -> int:
    return len(all_settings())


def settings_registry(settings: list[NaturalSearchSetting], *, seeds: tuple[int, ...]) -> pd.DataFrame:
    family_size = planned_phase1_family_size()
    rows = []
    for index, setting in enumerate(settings):
        rows.append(
            {
                "protocol_version": PROTOCOL_VERSION,
                "search_id": setting.search_id,
                "setting_id": setting.setting_id,
                "setting_index": index,
                "phase": setting.phase,
                "multiplicity_family": MULTIPLICITY_FAMILY,
                "planned_family_size": family_size,
                "dataset": setting.dataset_label,
                "dataset_name": setting.dataset_name,
                "architecture": "ResNet18 CIFAR stem",
                "model_arch": setting.model_arch,
                "partition_id": setting.partition_id,
                "head_classes": class_csv(setting.head_classes),
                "tail_classes": class_csv(setting.tail_classes),
                "head_class_count": len(setting.head_classes),
                "tail_class_count": len(setting.tail_classes),
                "head_train_per_class": setting.head_train_per_class,
                "tail_train_per_class": setting.tail_train_per_class,
                "tail_eval_per_class": setting.tail_eval_per_class,
                "warmup_steps": setting.warmup_steps,
                "target_head_gain_fraction": setting.target_head_gain_fraction,
                "seeds": class_csv(seeds),
                "seed_count": len(seeds),
                "freshness_rule": setting.freshness_rule,
                "planned_artifact_prefix": setting.planned_artifact_prefix.as_posix(),
                "entrypoint": "scripts/slurm/e11_natural_negative_search_phase1.sbatch",
                "runner": "scripts/e11_run_natural_negative_search_phase1.py",
            }
        )
    return pd.DataFrame(rows)


def parse_seeds(value: str) -> tuple[int, ...]:
    parts = tuple(part.strip() for part in value.split(",") if part.strip())
    if not parts:
        raise argparse.ArgumentTypeError("expected at least one seed or a positive seed count")
    if len(parts) == 1:
        count = int(parts[0])
        if count <= 0:
            raise argparse.ArgumentTypeError("seed count must be positive")
        return tuple(range(count))
    seeds = tuple(int(part) for part in parts)
    if len(seeds) != len(set(seeds)):
        raise argparse.ArgumentTypeError("seeds must be unique")
    return seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-id", choices=("all_phase1", *PHASE1_SEARCH_IDS), default="all_phase1")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seeds", type=parse_seeds, default=DEFAULT_SEEDS)
    parser.add_argument("--max-settings", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--list-settings", action="store_true")
    parser.add_argument("--settings-only", action="store_true")
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def config_for_setting(
    setting: NaturalSearchSetting,
    *,
    seeds: tuple[int, ...],
    args: argparse.Namespace,
) -> Cifar100ResNetOneStepConfig:
    config = Cifar100ResNetOneStepConfig(
        seeds=seeds,
        dataset_name=setting.dataset_name,
        model_arch=setting.model_arch,
        head_classes=setting.head_classes,
        tail_classes=setting.tail_classes,
        head_train_per_class=setting.head_train_per_class,
        tail_train_per_class=setting.tail_train_per_class,
        tail_eval_per_class=setting.tail_eval_per_class,
        warmup_steps=setting.warmup_steps,
        target_head_gain_fraction=setting.target_head_gain_fraction,
    )
    if args.smoke:
        config = replace(
            config,
            seeds=(0,),
            head_train_per_class=min(setting.head_train_per_class, 20),
            tail_train_per_class=min(setting.tail_train_per_class, 5),
            tail_eval_per_class=min(setting.tail_eval_per_class, 5),
            warmup_steps=2,
            warmup_batch_size=16,
            head_batch_size=16,
            target_head_gain_fraction=0.001,
        )
    if args.device is not None:
        config = replace(config, device=args.device)
    if args.download is not None:
        config = replace(config, download=args.download)
    return config


def output_dir_from_args(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir
    if args.search_id in SEARCH_OUTPUT_PREFIXES:
        return SEARCH_OUTPUT_PREFIXES[args.search_id]
    return RESULT_DIR / "phase1_all"


def write_settings_only(settings: list[NaturalSearchSetting], *, seeds: tuple[int, ...], args: argparse.Namespace) -> None:
    if args.output_dir is not None:
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        settings_registry(settings, seeds=seeds).to_csv(output_dir / "settings_registry.csv", index=False)
        print(f"saved natural negative-search phase1 settings registry to {output_dir}")
        return

    for search_id in sorted({setting.search_id for setting in settings}):
        selected = [setting for setting in settings if setting.search_id == search_id]
        output_dir = SEARCH_OUTPUT_PREFIXES[search_id]
        output_dir.mkdir(parents=True, exist_ok=True)
        settings_registry(selected, seeds=seeds).to_csv(output_dir / "settings_registry.csv", index=False)
        print(f"saved natural negative-search phase1 settings registry to {output_dir}")


def annotate_frame(
    frame: pd.DataFrame,
    setting: NaturalSearchSetting,
    *,
    setting_index: int,
) -> pd.DataFrame:
    annotated = frame.copy()
    family_size = planned_phase1_family_size()
    metadata = {
        "protocol_version": PROTOCOL_VERSION,
        "search_id": setting.search_id,
        "setting_id": setting.setting_id,
        "setting_index": setting_index,
        "phase": setting.phase,
        "multiplicity_family": MULTIPLICITY_FAMILY,
        "planned_family_size": family_size,
        "partition_id": setting.partition_id,
        "head_train_per_class": setting.head_train_per_class,
        "tail_train_per_class": setting.tail_train_per_class,
        "tail_eval_per_class": setting.tail_eval_per_class,
        "warmup_steps": setting.warmup_steps,
        "target_head_gain_fraction_registered": setting.target_head_gain_fraction,
        "fresh_protocol_id": "discussion/e11_natural_negative_search_protocol.md",
        "protocol_status": "fresh_phase1_output",
    }
    for key, value in metadata.items():
        annotated[key] = value
    return annotated


def build_decision_template(pair_summary: pd.DataFrame) -> pd.DataFrame:
    decision = pair_summary[
        [
            "search_id",
            "setting_id",
            "phase",
            "multiplicity_family",
            "planned_family_size",
            "dataset",
            "model",
            "partition_id",
            "warmup_steps",
            "target_head_gain_fraction_registered",
            "geomean_tail_output_drift_sq_ratio_spectral_over_fro",
            "tail_output_drift_sq_ratio_ci95_low",
            "tail_output_drift_sq_ratio_ci95_high",
            "mean_tail_accuracy_before",
            "mean_actual_head_gain_relative_error_frobenius",
            "mean_actual_head_gain_relative_error_spectral",
        ]
    ].copy()
    decision.insert(0, "metric_id", "primary_tail_output_drift_ratio")
    decision["raw_worse_ci95_low_above_one"] = decision["tail_output_drift_sq_ratio_ci95_low"] > 1.0
    decision["adjustment_method"] = "pending_multiplicity_evaluator"
    decision["adjusted_ci95_low"] = float("nan")
    decision["adjusted_ci95_high"] = float("nan")
    decision["adjusted_primary_decision"] = "pending"
    decision["claim_boundary"] = "no primary natural-negative claim until adjusted evaluator passes"
    return decision


def write_discussion(
    *,
    output_dir: Path,
    registry: pd.DataFrame,
    pair_summary: pd.DataFrame | None,
    decision_template: pd.DataFrame | None,
    discussion_path: Path,
) -> None:
    lines = [
        "# E11 Natural Negative Search Phase1 Outputs",
        "",
        "This generated artifact is the executable phase1 output boundary for the",
        "pre-registered natural negative-search protocol. It preserves every selected",
        "setting in the registry before any multiplicity-adjusted discovery decision.",
        "",
        "## Settings",
        "",
        markdown_table(
            registry,
            [
                "search_id",
                "setting_id",
                "dataset",
                "partition_id",
                "warmup_steps",
                "target_head_gain_fraction",
                "seed_count",
                "planned_family_size",
            ],
        ),
        "",
    ]
    if pair_summary is not None and not pair_summary.empty:
        lines.extend(
            [
                "## Raw Primary Readout",
                "",
                markdown_table(
                    pair_summary,
                    [
                        "search_id",
                        "setting_id",
                        "geomean_tail_output_drift_sq_ratio_spectral_over_fro",
                        "tail_output_drift_sq_ratio_ci95_low",
                        "tail_output_drift_sq_ratio_ci95_high",
                        "mean_tail_accuracy_before",
                    ],
                ),
                "",
            ]
        )
    if decision_template is not None and not decision_template.empty:
        lines.extend(
            [
                "## Decision Boundary",
                "",
                "The table below is intentionally incomplete: adjusted confidence endpoints",
                "must be added by the registered multiplicity evaluator before any natural",
                "primary counterexample or finite-null claim is made.",
                "",
                markdown_table(
                    decision_template,
                    [
                        "search_id",
                        "setting_id",
                        "metric_id",
                        "raw_worse_ci95_low_above_one",
                        "adjustment_method",
                        "adjusted_primary_decision",
                    ],
                ),
                "",
            ]
        )
    lines.extend(
        [
            "Artifacts:",
            f"- [settings_registry.csv](../{(output_dir / 'settings_registry.csv').as_posix()})",
            f"- [step_metrics.csv](../{(output_dir / 'step_metrics.csv').as_posix()})",
            f"- [pair_summary.csv](../{(output_dir / 'pair_summary.csv').as_posix()})",
            f"- [layer_metrics.csv](../{(output_dir / 'layer_metrics.csv').as_posix()})",
            f"- [decision_template.csv](../{(output_dir / 'decision_template.csv').as_posix()})",
            f"- [config.json](../{(output_dir / 'config.json').as_posix()})",
        ]
    )
    write_markdown(discussion_path, "\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    settings = settings_for_search(args.search_id)
    if args.max_settings is not None:
        if args.max_settings <= 0:
            raise ValueError("--max-settings must be positive")
        settings = settings[: args.max_settings]
    run_seeds = (0,) if args.smoke else args.seeds
    registry = settings_registry(settings, seeds=run_seeds)
    if args.list_settings:
        print(registry.to_csv(index=False), end="")
        return
    if args.settings_only:
        write_settings_only(settings, seeds=run_seeds, args=args)
        return

    output_dir = output_dir_from_args(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    registry.to_csv(output_dir / "settings_registry.csv", index=False)

    step_frames: list[pd.DataFrame] = []
    pair_frames: list[pd.DataFrame] = []
    layer_frames: list[pd.DataFrame] = []
    config_snapshots = []
    for setting_index, setting in enumerate(settings):
        config = config_for_setting(setting, seeds=run_seeds, args=args)
        if args.progress:
            print(
                f"[natural-negative-phase1] {setting.search_id} {setting.setting_id} "
                f"seeds={','.join(str(seed) for seed in config.seeds)}",
                flush=True,
            )
        step_metrics, pair_summary, layer_metrics = run_cifar100_resnet_one_step(config, progress=args.progress)
        step_frames.append(annotate_frame(step_metrics, setting, setting_index=setting_index))
        pair_frames.append(annotate_frame(pair_summary, setting, setting_index=setting_index))
        layer_frames.append(annotate_frame(layer_metrics, setting, setting_index=setting_index))
        config_snapshots.append({"setting_id": setting.setting_id, "config": asdict(config)})
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    step_metrics = pd.concat(step_frames, ignore_index=True)
    pair_summary = pd.concat(pair_frames, ignore_index=True)
    layer_metrics = pd.concat(layer_frames, ignore_index=True)
    decision_template = build_decision_template(pair_summary)
    step_metrics.to_csv(output_dir / "step_metrics.csv", index=False)
    pair_summary.to_csv(output_dir / "pair_summary.csv", index=False)
    layer_metrics.to_csv(output_dir / "layer_metrics.csv", index=False)
    decision_template.to_csv(output_dir / "decision_template.csv", index=False)
    (output_dir / "config.json").write_text(
        json.dumps(
            {
                "protocol_version": PROTOCOL_VERSION,
                "search_id": args.search_id,
                "settings": config_snapshots,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    discussion_path = Path("discussion") / f"e11_natural_negative_search_phase1_{args.search_id}.md"
    write_discussion(
        output_dir=output_dir,
        registry=registry,
        pair_summary=pair_summary,
        decision_template=decision_template,
        discussion_path=discussion_path,
    )
    print(f"saved natural negative-search phase1 results to {output_dir}")
    print(f"settings={len(settings)}, seeds_per_setting={len(run_seeds)}")
    print(f"discussion: {discussion_path}")


if __name__ == "__main__":
    main()
