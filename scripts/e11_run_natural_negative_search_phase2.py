from __future__ import annotations

import argparse
import gc
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.cifar100_resnet_tail import (
    Cifar100ResNetOneStepConfig,
    model_display_name,
    run_cifar100_resnet_one_step,
)
from e11_condition_geometry.reporting import markdown_table, write_markdown
from scripts.e11_run_natural_negative_search_phase1 import (
    NaturalSearchSetting,
    by_mod,
    cifar100_decade_parity,
    class_csv,
    parse_seeds,
    rho_tag,
)


RESULT_DIR = Path("results/e11_natural_negative_search_protocol")
PHASE2_SEARCH_ID = "NNS-P2-heldout-architecture-boundary"
PHASE2_OUTPUT_PREFIX = RESULT_DIR / "phase2_heldout_architecture"
PHASE2_DISCUSSION_PATH = Path("discussion") / f"e11_natural_negative_search_phase2_{PHASE2_SEARCH_ID}.md"
DEFAULT_PHASE2_SEEDS = (0, 1, 2)
PHASE2_MULTIPLICITY_FAMILY = "NNS-P2-heldout-architecture-family"
PROTOCOL_VERSION = "nns_phase2_heldout_architecture_runner_v1"


def make_phase2_setting(
    *,
    partition_id: str,
    head_classes: tuple[int, ...],
    tail_classes: tuple[int, ...],
    warmup_steps: int,
    target_head_gain_fraction: float,
) -> NaturalSearchSetting:
    setting_id = (
        f"resnet34_{partition_id}_w{warmup_steps}_rho{rho_tag(target_head_gain_fraction)}"
        "_tail30"
    )
    return NaturalSearchSetting(
        search_id=PHASE2_SEARCH_ID,
        setting_id=setting_id,
        phase="phase2_fresh_generality_search",
        dataset_label="CIFAR-100-LT",
        dataset_name="CIFAR100",
        model_arch="resnet34",
        partition_id=partition_id,
        head_classes=head_classes,
        tail_classes=tail_classes,
        head_train_per_class=300,
        tail_train_per_class=30,
        tail_eval_per_class=40,
        warmup_steps=warmup_steps,
        target_head_gain_fraction=target_head_gain_fraction,
        freshness_rule=(
            "architecture family fixed after complete phase1 archive; partitions are phase1-declared "
            "and not selected from phase1 outcomes"
        ),
        planned_artifact_prefix=PHASE2_OUTPUT_PREFIX,
    )


def phase2_settings() -> list[NaturalSearchSetting]:
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
        make_phase2_setting(
            partition_id=partition_id,
            head_classes=head_classes,
            tail_classes=tail_classes,
            warmup_steps=warmup_steps,
            target_head_gain_fraction=rho,
        )
        for partition_id, head_classes, tail_classes in partitions
        for warmup_steps in (2000, 5000)
        for rho in (0.002, 0.005)
    ]


def settings_registry(settings: list[NaturalSearchSetting], *, seeds: tuple[int, ...]) -> pd.DataFrame:
    family_size = len(settings)
    rows = []
    for index, setting in enumerate(settings):
        rows.append(
            {
                "protocol_version": PROTOCOL_VERSION,
                "search_id": setting.search_id,
                "setting_id": setting.setting_id,
                "setting_index": index,
                "phase": setting.phase,
                "multiplicity_family": PHASE2_MULTIPLICITY_FAMILY,
                "planned_family_size": family_size,
                "dataset": setting.dataset_label,
                "dataset_name": setting.dataset_name,
                "architecture": f"{model_display_name(setting.model_arch)} CIFAR stem",
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
                "entrypoint": "scripts/slurm/e11_natural_negative_search_phase2.sbatch",
                "runner": "scripts/e11_run_natural_negative_search_phase2.py",
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-id", choices=(PHASE2_SEARCH_ID,), default=PHASE2_SEARCH_ID)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seeds", type=parse_seeds, default=DEFAULT_PHASE2_SEEDS)
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
        config = Cifar100ResNetOneStepConfig(
            seeds=(0,),
            dataset_name=setting.dataset_name,
            model_arch=setting.model_arch,
            head_classes=setting.head_classes,
            tail_classes=setting.tail_classes,
            head_train_per_class=min(setting.head_train_per_class, 20),
            tail_train_per_class=min(setting.tail_train_per_class, 5),
            tail_eval_per_class=min(setting.tail_eval_per_class, 5),
            warmup_steps=2,
            warmup_batch_size=16,
            head_batch_size=16,
            target_head_gain_fraction=0.001,
        )
    if args.device is not None:
        config = Cifar100ResNetOneStepConfig(**{**asdict(config), "device": args.device})
    if args.download is not None:
        config = Cifar100ResNetOneStepConfig(**{**asdict(config), "download": args.download})
    return config


def annotate_frame(frame: pd.DataFrame, setting: NaturalSearchSetting, *, setting_index: int) -> pd.DataFrame:
    annotated = frame.copy()
    metadata = {
        "protocol_version": PROTOCOL_VERSION,
        "search_id": setting.search_id,
        "setting_id": setting.setting_id,
        "setting_index": setting_index,
        "phase": setting.phase,
        "multiplicity_family": PHASE2_MULTIPLICITY_FAMILY,
        "planned_family_size": len(phase2_settings()),
        "partition_id": setting.partition_id,
        "head_train_per_class": setting.head_train_per_class,
        "tail_train_per_class": setting.tail_train_per_class,
        "tail_eval_per_class": setting.tail_eval_per_class,
        "warmup_steps": setting.warmup_steps,
        "target_head_gain_fraction_registered": setting.target_head_gain_fraction,
        "fresh_protocol_id": "discussion/e11_natural_negative_search_protocol.md",
        "protocol_status": "fresh_phase2_output",
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
    decision["adjustment_method"] = "pending_phase2_multiplicity_evaluator"
    decision["adjusted_primary_decision"] = "pending"
    decision["claim_boundary"] = "phase2 can replicate or bound phase1, not retroactively select phase1"
    return decision


def write_discussion(
    *,
    output_dir: Path,
    registry: pd.DataFrame,
    pair_summary: pd.DataFrame | None,
    decision_template: pd.DataFrame | None,
) -> None:
    lines = [
        "# E11 Natural Negative Search Phase2 Held-Out Architecture Outputs",
        "",
        "This generated artifact freezes the phase2 held-out architecture family for",
        "the registered natural negative-search protocol. It uses ResNet34 as the",
        "architecture transport check after the complete phase1 ResNet18 family and",
        "keeps the phase1 partitions fixed before any phase2 metric output is used.",
        "",
        "## Settings",
        "",
        markdown_table(
            registry,
            [
                "search_id",
                "setting_id",
                "dataset",
                "architecture",
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
                        "model",
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
                "Phase2 decisions are claim-boundary evidence only until every declared",
                "phase2 row is evaluated with the same paired per-seed primary rule.",
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
        ]
    )
    if pair_summary is None or pair_summary.empty:
        lines.extend(
            [
                "",
                "No phase2 metric rows are present in this settings-only freeze.",
                "The Slurm run will write `step_metrics.csv`, `pair_summary.csv`,",
                "`layer_metrics.csv`, `decision_template.csv`, and `config.json`.",
            ]
        )
    else:
        lines.extend(
            [
                f"- [step_metrics.csv](../{(output_dir / 'step_metrics.csv').as_posix()})",
                f"- [pair_summary.csv](../{(output_dir / 'pair_summary.csv').as_posix()})",
                f"- [layer_metrics.csv](../{(output_dir / 'layer_metrics.csv').as_posix()})",
                f"- [decision_template.csv](../{(output_dir / 'decision_template.csv').as_posix()})",
                f"- [config.json](../{(output_dir / 'config.json').as_posix()})",
            ]
        )
    write_markdown(PHASE2_DISCUSSION_PATH, "\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    settings = phase2_settings()
    if args.max_settings is not None:
        if args.max_settings <= 0:
            raise ValueError("--max-settings must be positive")
        settings = settings[: args.max_settings]
    run_seeds = (0,) if args.smoke else args.seeds
    registry = settings_registry(settings, seeds=run_seeds)

    if args.list_settings:
        print(registry.to_csv(index=False), end="")
        return

    output_dir = args.output_dir or PHASE2_OUTPUT_PREFIX
    output_dir.mkdir(parents=True, exist_ok=True)
    registry.to_csv(output_dir / "settings_registry.csv", index=False)
    if args.settings_only:
        write_discussion(output_dir=output_dir, registry=registry, pair_summary=None, decision_template=None)
        print(f"saved natural negative-search phase2 settings registry to {output_dir}")
        print(f"discussion: {PHASE2_DISCUSSION_PATH}")
        return

    step_frames: list[pd.DataFrame] = []
    pair_frames: list[pd.DataFrame] = []
    layer_frames: list[pd.DataFrame] = []
    config_snapshots = []
    for setting_index, setting in enumerate(settings):
        config = config_for_setting(setting, seeds=run_seeds, args=args)
        if args.progress:
            print(
                f"[natural-negative-phase2] {setting.search_id} {setting.setting_id} "
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
                "multiplicity_family": PHASE2_MULTIPLICITY_FAMILY,
                "settings": config_snapshots,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_discussion(
        output_dir=output_dir,
        registry=registry,
        pair_summary=pair_summary,
        decision_template=decision_template,
    )
    print(f"saved natural negative-search phase2 results to {output_dir}")
    print(f"settings={len(settings)}, seeds_per_setting={len(run_seeds)}")
    print(f"discussion: {PHASE2_DISCUSSION_PATH}")


if __name__ == "__main__":
    main()
