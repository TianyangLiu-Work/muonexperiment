from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import e11_run_cifar100_resnet_lt_recipe_benchmark as recipe_benchmark


RESULT_ROOT = Path("results/e11_cifar100_resnet_lt_tuned_benchmark")
FIGURE_ROOT = Path("figures/e11_cifar100_resnet_lt_tuned_benchmark")
DISCUSSION_ROOT = Path("discussion/e11_cifar100_resnet_lt_tuned_benchmark")
PROTOCOL_PATH = Path("discussion/e11_cifar100_resnet_lt_tuned_benchmark_protocol.md")


FLOAT_GRIDS = {
    "adamw_lr": [("1e-4", 1e-4), ("3e-4", 3e-4), ("1e-3", 1e-3)],
    "sgd_lr": [("0p03", 0.03), ("0p1", 0.1), ("0p3", 0.3)],
    "muon_lr": [("1e-5", 1e-5), ("3e-5", 3e-5), ("1e-4", 1e-4), ("3e-4", 3e-4)],
    "muon_cb_lr": [("1e-5", 1e-5), ("3e-5", 3e-5), ("1e-4", 1e-4)],
    "wd_adamw": [("1e-4", 1e-4), ("5e-4", 5e-4)],
    "wd_sgd": [("5e-4", 5e-4), ("1e-3", 1e-3)],
    "beta": [("0p999", 0.999), ("0p9999", 0.9999)],
}


@dataclass(frozen=True)
class TunedBenchmarkSetting:
    setting_id: str
    phase: str
    split_id: str
    seed_set: str
    recipe_family: str
    recipe_name: str
    optimizer: str
    lr: float
    other_lr: float
    weight_decay: float
    momentum: float
    nesterov: bool
    class_balanced_loss: bool
    class_balanced_sampler: bool
    class_balance_beta: float | None
    warmup_steps: int
    newton_schulz_steps: int
    protocol_reference: str
    planned_output_dir: Path
    planned_figure_dir: Path
    planned_discussion_path: Path


def _setting_id(family: str, parts: list[str]) -> str:
    return "TBV-" + family + "-" + "-".join(parts)


def _make_setting(
    *,
    recipe_family: str,
    recipe_name: str,
    optimizer: str,
    lr: float,
    weight_decay: float,
    warmup_steps: int,
    other_lr: float = 3e-4,
    momentum: float = 0.0,
    nesterov: bool = False,
    class_balanced_loss: bool = False,
    class_balanced_sampler: bool = False,
    class_balance_beta: float | None = None,
    newton_schulz_steps: int = 5,
    id_parts: list[str],
) -> TunedBenchmarkSetting:
    setting_id = _setting_id(recipe_family, id_parts)
    return TunedBenchmarkSetting(
        setting_id=setting_id,
        phase="validation_tuning",
        split_id="validation_tuning",
        seed_set="10..14",
        recipe_family=recipe_family,
        recipe_name=recipe_name,
        optimizer=optimizer,
        lr=float(lr),
        other_lr=float(other_lr),
        weight_decay=float(weight_decay),
        momentum=float(momentum),
        nesterov=bool(nesterov),
        class_balanced_loss=bool(class_balanced_loss),
        class_balanced_sampler=bool(class_balanced_sampler),
        class_balance_beta=None if class_balance_beta is None else float(class_balance_beta),
        warmup_steps=int(warmup_steps),
        newton_schulz_steps=int(newton_schulz_steps),
        protocol_reference=PROTOCOL_PATH.as_posix(),
        planned_output_dir=RESULT_ROOT / "validation_tuning" / setting_id,
        planned_figure_dir=FIGURE_ROOT / "validation_tuning" / setting_id,
        planned_discussion_path=DISCUSSION_ROOT / f"{setting_id}.md",
    )


def all_settings() -> list[TunedBenchmarkSetting]:
    settings: list[TunedBenchmarkSetting] = []
    for lr_tag, lr in FLOAT_GRIDS["adamw_lr"]:
        for wd_tag, wd in FLOAT_GRIDS["wd_adamw"]:
            for warmup in (0, 500):
                family = "adamw_ce_tuned"
                recipe_name = f"{family}_lr{lr_tag}_wd{wd_tag}_warm{warmup}"
                settings.append(
                    _make_setting(
                        recipe_family=family,
                        recipe_name=recipe_name,
                        optimizer="adamw",
                        lr=lr,
                        weight_decay=wd,
                        warmup_steps=warmup,
                        id_parts=[f"lr{lr_tag}", f"wd{wd_tag}", f"warm{warmup}"],
                    )
                )
    for lr_tag, lr in FLOAT_GRIDS["sgd_lr"]:
        for wd_tag, wd in FLOAT_GRIDS["wd_sgd"]:
            for warmup in (0, 500):
                family = "sgd_momentum_ce_tuned"
                recipe_name = f"{family}_lr{lr_tag}_wd{wd_tag}_warm{warmup}"
                settings.append(
                    _make_setting(
                        recipe_family=family,
                        recipe_name=recipe_name,
                        optimizer="sgd",
                        lr=lr,
                        weight_decay=wd,
                        momentum=0.9,
                        nesterov=True,
                        warmup_steps=warmup,
                        id_parts=[f"lr{lr_tag}", f"wd{wd_tag}", f"warm{warmup}"],
                    )
                )
    for beta_tag, beta in FLOAT_GRIDS["beta"]:
        for lr_tag, lr in FLOAT_GRIDS["adamw_lr"]:
            for wd_tag, wd in FLOAT_GRIDS["wd_adamw"]:
                for warmup in (0, 500):
                    family = "adamw_cb_loss_tuned"
                    recipe_name = f"{family}_beta{beta_tag}_lr{lr_tag}_wd{wd_tag}_warm{warmup}"
                    settings.append(
                        _make_setting(
                            recipe_family=family,
                            recipe_name=recipe_name,
                            optimizer="adamw",
                            lr=lr,
                            weight_decay=wd,
                            class_balanced_loss=True,
                            class_balance_beta=beta,
                            warmup_steps=warmup,
                            id_parts=[f"beta{beta_tag}", f"lr{lr_tag}", f"wd{wd_tag}", f"warm{warmup}"],
                        )
                    )
    for lr_tag, lr in FLOAT_GRIDS["adamw_lr"][:2]:
        for wd_tag, wd in FLOAT_GRIDS["wd_adamw"]:
            for warmup in (0, 500):
                family = "adamw_cb_sampler_tuned"
                recipe_name = f"{family}_lr{lr_tag}_wd{wd_tag}_warm{warmup}"
                settings.append(
                    _make_setting(
                        recipe_family=family,
                        recipe_name=recipe_name,
                        optimizer="adamw",
                        lr=lr,
                        weight_decay=wd,
                        class_balanced_sampler=True,
                        warmup_steps=warmup,
                        id_parts=[f"lr{lr_tag}", f"wd{wd_tag}", f"warm{warmup}"],
                    )
                )
    for lr_tag, lr in FLOAT_GRIDS["muon_lr"]:
        for wd_tag, wd in FLOAT_GRIDS["wd_adamw"]:
            for warmup in (0, 500, 1000):
                for ns_steps in (3, 5, 7):
                    family = "ns_muon_matrix_tuned"
                    recipe_name = f"{family}_lr{lr_tag}_wd{wd_tag}_warm{warmup}_ns{ns_steps}"
                    settings.append(
                        _make_setting(
                            recipe_family=family,
                            recipe_name=recipe_name,
                            optimizer="ns_muon",
                            lr=lr,
                            weight_decay=wd,
                            momentum=0.9,
                            warmup_steps=warmup,
                            newton_schulz_steps=ns_steps,
                            id_parts=[f"lr{lr_tag}", f"wd{wd_tag}", f"warm{warmup}", f"ns{ns_steps}"],
                        )
                    )
    for lr_tag, lr in FLOAT_GRIDS["muon_cb_lr"]:
        for wd_tag, wd in FLOAT_GRIDS["wd_adamw"]:
            for warmup in (500, 1000):
                for ns_steps in (3, 5, 7):
                    family = "ns_muon_cb_tuned"
                    recipe_name = f"{family}_beta0p9999_lr{lr_tag}_wd{wd_tag}_warm{warmup}_ns{ns_steps}"
                    settings.append(
                        _make_setting(
                            recipe_family=family,
                            recipe_name=recipe_name,
                            optimizer="ns_muon",
                            lr=lr,
                            weight_decay=wd,
                            momentum=0.9,
                            class_balanced_loss=True,
                            class_balance_beta=0.9999,
                            warmup_steps=warmup,
                            newton_schulz_steps=ns_steps,
                            id_parts=[f"beta0p9999", f"lr{lr_tag}", f"wd{wd_tag}", f"warm{warmup}", f"ns{ns_steps}"],
                        )
                    )
    setting_ids = [setting.setting_id for setting in settings]
    if len(setting_ids) != len(set(setting_ids)):
        raise AssertionError("tuned benchmark validation setting ids must be unique")
    return settings


def settings_frame(settings: list[TunedBenchmarkSetting]) -> pd.DataFrame:
    rows = []
    for index, setting in enumerate(settings):
        row = asdict(setting)
        row["array_index"] = int(index)
        for key in ("planned_output_dir", "planned_figure_dir", "planned_discussion_path"):
            row[key] = Path(row[key]).as_posix()
        rows.append(row)
    return pd.DataFrame(rows)


def write_settings_registry(settings: list[TunedBenchmarkSetting]) -> Path:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    registry = settings_frame(settings)
    path = RESULT_ROOT / "settings_registry.csv"
    registry.to_csv(path, index=False)
    status = pd.DataFrame(
        [
            {
                "registry_id": "cifar100lt_resnet18_tuned_validation_grid",
                "settings": len(settings),
                "phase": "validation_tuning",
                "status": "settings_registered",
                "claim_status": "not_ready_until_validation_and_final_claim_seeds_complete",
            }
        ]
    )
    status.to_csv(RESULT_ROOT / "execution_status.csv", index=False)
    return path


def recipe_from_setting(setting: TunedBenchmarkSetting) -> recipe_benchmark.Recipe:
    return recipe_benchmark.Recipe(
        name=setting.recipe_name,
        optimizer=setting.optimizer,
        lr=setting.lr,
        other_lr=setting.other_lr,
        weight_decay=setting.weight_decay,
        momentum=setting.momentum,
        nesterov=setting.nesterov,
        class_balanced_loss=setting.class_balanced_loss,
        class_balanced_sampler=setting.class_balanced_sampler,
        class_balance_beta=setting.class_balance_beta,
        augmentation=True,
        cosine_lr=True,
        warmup_steps=setting.warmup_steps,
        newton_schulz_steps=setting.newton_schulz_steps,
    )


def patch_recipe_lookup(settings: list[TunedBenchmarkSetting]) -> None:
    original = recipe_benchmark.recipe_from_name
    recipes = {setting.recipe_name: recipe_from_setting(setting) for setting in settings}

    def lookup(name: str) -> recipe_benchmark.Recipe:
        if name in recipes:
            return recipes[name]
        return original(name)

    recipe_benchmark.recipe_from_name = lookup


def seeds_from_setting(setting: TunedBenchmarkSetting) -> tuple[int, ...]:
    start_text, end_text = setting.seed_set.split("..", maxsplit=1)
    start = int(start_text)
    end = int(end_text)
    return tuple(range(start, end + 1))


def select_setting(
    settings: list[TunedBenchmarkSetting],
    *,
    setting_id: str | None,
    array_index: int | None,
) -> TunedBenchmarkSetting | None:
    if setting_id is None and array_index is None:
        return None
    if setting_id is not None:
        matches = [setting for setting in settings if setting.setting_id == setting_id]
        if len(matches) != 1:
            raise ValueError(f"unknown tuned benchmark setting_id: {setting_id}")
        return matches[0]
    if array_index is None or not (0 <= int(array_index) < len(settings)):
        raise ValueError(f"array index must be in [0, {len(settings) - 1}]")
    return settings[int(array_index)]


def run_setting(
    setting: TunedBenchmarkSetting,
    *,
    device: str,
    download: bool,
    smoke: bool,
    progress: bool,
) -> None:
    patch_recipe_lookup([setting])
    if smoke:
        config = recipe_benchmark.RecipeBenchmarkConfig(
            seeds=(0,),
            recipe_names=(setting.recipe_name,),
            baseline_recipe=setting.recipe_name,
            num_classes=10,
            max_train_count=10,
            imbalance_factor=10.0,
            train_steps=2,
            train_batch_size=32,
            eval_interval=1,
            device=device,
            download=download,
        )
    else:
        config = recipe_benchmark.RecipeBenchmarkConfig(
            seeds=seeds_from_setting(setting),
            recipe_names=(setting.recipe_name,),
            baseline_recipe=setting.recipe_name,
            num_classes=100,
            max_train_count=500,
            imbalance_factor=100.0,
            many_threshold=100,
            medium_threshold=20,
            train_steps=10000,
            train_batch_size=256,
            eval_interval=2000,
            device=device,
            download=download,
        )
    trace, class_metrics, group_metrics, summary, pair_summary = recipe_benchmark.run_recipe_benchmark(
        config,
        progress=progress,
    )
    output_dir = setting.planned_output_dir
    figure_dir = setting.planned_figure_dir
    discussion_path = setting.planned_discussion_path
    output_dir.mkdir(parents=True, exist_ok=True)
    trace.to_csv(output_dir / "train_trace.csv", index=False)
    class_metrics.to_csv(output_dir / "class_metrics.csv", index=False)
    group_metrics.to_csv(output_dir / "group_metrics.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    pair_summary.to_csv(output_dir / "pair_summary.csv", index=False)
    config_payload = asdict(config)
    config_payload["recipe"] = asdict(recipe_from_setting(setting))
    config_payload["setting"] = asdict(setting)
    for key in ("planned_output_dir", "planned_figure_dir", "planned_discussion_path"):
        config_payload["setting"][key] = Path(config_payload["setting"][key]).as_posix()
    (output_dir / "config.json").write_text(json.dumps(config_payload, indent=2, sort_keys=True) + "\n")
    pd.DataFrame([config_payload["setting"]]).to_csv(output_dir / "setting_metadata.csv", index=False)
    figure_path = recipe_benchmark.write_figure(summary, pair_summary, figure_dir)
    recipe_benchmark.write_discussion(config, summary, pair_summary, figure_path, output_dir, discussion_path)
    print(f"saved tuned benchmark setting {setting.setting_id} to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting-id", default=None)
    parser.add_argument("--array-index", type=int, default=None)
    parser.add_argument("--max-settings", type=int, default=None)
    parser.add_argument("--settings-only", action="store_true")
    parser.add_argument("--list-settings", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = all_settings()
    if args.max_settings is not None:
        if args.max_settings <= 0:
            raise ValueError("--max-settings must be positive")
        settings = settings[: args.max_settings]
    registry_path = write_settings_registry(settings)
    if args.list_settings:
        print(settings_frame(settings)[["array_index", "setting_id", "recipe_family", "recipe_name"]].to_string(index=False))
    selected = select_setting(settings, setting_id=args.setting_id, array_index=args.array_index)
    if args.settings_only or selected is None:
        print(f"saved tuned benchmark settings registry to {registry_path}")
        return
    run_setting(
        selected,
        device=args.device,
        download=False if args.download is None else bool(args.download),
        smoke=bool(args.smoke),
        progress=bool(args.progress),
    )


if __name__ == "__main__":
    main()
