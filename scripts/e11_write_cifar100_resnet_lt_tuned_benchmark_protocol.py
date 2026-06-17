from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.reporting import fmt, markdown_table, write_markdown


RESULT_DIR = Path("results/e11_cifar100_resnet_lt_tuned_benchmark_protocol")
DISCUSSION_PATH = Path("discussion/e11_cifar100_resnet_lt_tuned_benchmark_protocol.md")

STANDARD_DIR = Path("results/e11_cifar100_resnet_lt_standard_eval")
RECIPE_DIR = Path("results/e11_cifar100_resnet_lt_recipe_benchmark")
MUON_DIR = Path("results/e11_cifar100_resnet_lt_muon_final_benchmark")


def _summary_row(path: Path, *, frequency_group: str, recipe: str | None = None) -> pd.Series:
    frame = pd.read_csv(path)
    mask = frame["frequency_group"].eq(frequency_group)
    if recipe is not None:
        mask = mask & frame["recipe"].eq(recipe)
    selected = frame[mask]
    if len(selected) != 1:
        raise ValueError(f"expected one row in {path} for recipe={recipe!r}, frequency_group={frequency_group!r}")
    return selected.iloc[0]


def _pair_row(path: Path, *, recipe: str, frequency_group: str) -> pd.Series:
    frame = pd.read_csv(path)
    selected = frame[frame["recipe"].eq(recipe) & frame["frequency_group"].eq(frequency_group)]
    if len(selected) != 1:
        raise ValueError(f"expected one paired row in {path} for recipe={recipe!r}, frequency_group={frequency_group!r}")
    return selected.iloc[0]


def _metric_context(
    *,
    context_id: str,
    source: str,
    recipe: str,
    frequency_group: str,
    metric: str,
    estimate: float,
    low: float,
    high: float,
    interpretation: str,
    allowed_use: str,
) -> dict[str, object]:
    return {
        "context_id": context_id,
        "source": source,
        "recipe": recipe,
        "frequency_group": frequency_group,
        "metric": metric,
        "estimate": float(estimate),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "interpretation": interpretation,
        "allowed_use": allowed_use,
    }


def build_pilot_context() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group in ("all", "few"):
        row = _summary_row(STANDARD_DIR / "summary.csv", frequency_group=group)
        rows.append(
            _metric_context(
                context_id=f"standard_adamw_{group}",
                source="results/e11_cifar100_resnet_lt_standard_eval/summary.csv",
                recipe="adamw_no_aug_ce",
                frequency_group=group,
                metric="balanced_accuracy",
                estimate=row["mean_balanced_accuracy"],
                low=row["balanced_accuracy_ci95_low"],
                high=row["balanced_accuracy_ci95_high"],
                interpretation="standard reporting baseline is weaker than augmented pilots",
                allowed_use="context only; not a tuned benchmark comparator",
            )
        )

    recipe_pairs = RECIPE_DIR / "pair_summary.csv"
    for recipe, group, interpretation in (
        ("sgd_aug_ce", "all", "SGD augmentation pilot improves all-class balanced accuracy over AdamW augmentation"),
        ("sgd_aug_ce", "few", "SGD augmentation pilot improves few-class balanced accuracy over AdamW augmentation"),
        ("adamw_aug_cb_loss", "few", "class-balanced loss hurts few-class balanced accuracy in this pilot"),
    ):
        row = _pair_row(recipe_pairs, recipe=recipe, frequency_group=group)
        rows.append(
            _metric_context(
                context_id=f"recipe_pilot_{recipe}_{group}",
                source="results/e11_cifar100_resnet_lt_recipe_benchmark/pair_summary.csv",
                recipe=recipe,
                frequency_group=group,
                metric="paired_balanced_accuracy_diff_vs_adamw_aug_ce",
                estimate=row["mean_balanced_accuracy_diff"],
                low=row["balanced_accuracy_diff_ci95_low"],
                high=row["balanced_accuracy_diff_ci95_high"],
                interpretation=interpretation,
                allowed_use="pilot context only; cannot choose final hyperparameters",
            )
        )

    muon_pairs = MUON_DIR / "pair_summary.csv"
    for recipe in ("ns_muon_aug_lr3e-5", "ns_muon_aug_lr1e-4"):
        for group in ("all", "few"):
            row = _pair_row(muon_pairs, recipe=recipe, frequency_group=group)
            rows.append(
                _metric_context(
                    context_id=f"muon_pilot_{recipe}_{group}",
                    source="results/e11_cifar100_resnet_lt_muon_final_benchmark/pair_summary.csv",
                    recipe=recipe,
                    frequency_group=group,
                    metric="paired_balanced_accuracy_diff_vs_adamw_aug_ce",
                    estimate=row["mean_balanced_accuracy_diff"],
                    low=row["balanced_accuracy_diff_ci95_low"],
                    high=row["balanced_accuracy_diff_ci95_high"],
                    interpretation="tested finite-NS-Muon final-training recipe is below AdamW augmentation",
                    allowed_use="negative pilot boundary only; does not rule out the tuned Muon grid",
                )
            )
    return pd.DataFrame(rows)


def build_benchmark_scope() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "scope_id": "primary_cifar100lt_resnet18_if100",
                "dataset": "CIFAR-100-LT",
                "imbalance_factor": 100,
                "architecture": "ResNet18",
                "primary_metric": "few balanced accuracy",
                "secondary_metrics": "all/many/medium balanced accuracy; loss; margin",
                "claim_scope": "CIFAR-100-LT ResNet18 final-performance claim only",
                "status": "protocol_registered",
            },
            {
                "scope_id": "broad_long_tail_optimizer_claim",
                "dataset": "multiple long-tail datasets",
                "imbalance_factor": "dataset-specific",
                "architecture": "multiple architectures",
                "primary_metric": "not covered",
                "secondary_metrics": "not covered",
                "claim_scope": "broad optimizer-performance claim remains forbidden",
                "status": "not_covered_by_this_protocol",
            },
        ]
    )


def build_seed_split_contract() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "split_id": "spent_pilot_context",
                "seed_set": "0..9",
                "role": "pilot_context",
                "tuning_allowed": "no for final claims",
                "outputs": "standard, recipe, and NS-Muon pilot result directories",
            },
            {
                "split_id": "validation_tuning",
                "seed_set": "10..14",
                "role": "hyperparameter_selection",
                "tuning_allowed": "yes before final unblinding",
                "outputs": "results/e11_cifar100_resnet_lt_tuned_benchmark/validation_*",
            },
            {
                "split_id": "final_claim",
                "seed_set": "20..29",
                "role": "primary_final_evaluation",
                "tuning_allowed": "no",
                "outputs": "results/e11_cifar100_resnet_lt_tuned_benchmark/final_cifar100lt_resnet18",
            },
            {
                "split_id": "optional_stability_rerun",
                "seed_set": "30..34",
                "role": "post_claim_stability_check",
                "tuning_allowed": "no",
                "outputs": "optional appendix only",
            },
        ]
    )


def build_recipe_grid() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "recipe_family": "adamw_ce_tuned",
                "optimizer": "adamw",
                "matrix_update": "standard adaptive first order",
                "lr_grid": "1e-4;3e-4;1e-3",
                "weight_decay_grid": "1e-4;5e-4",
                "class_reweighting": "none",
                "sampler": "long-tail natural sampler",
                "scheduler": "cosine with warmup grid",
                "warmup_steps_grid": "0;500",
                "newton_schulz_steps_grid": "not_applicable",
                "claim_role": "tuned AdamW baseline",
            },
            {
                "recipe_family": "sgd_momentum_ce_tuned",
                "optimizer": "sgd",
                "matrix_update": "momentum SGD",
                "lr_grid": "0.03;0.1;0.3",
                "weight_decay_grid": "5e-4;1e-3",
                "class_reweighting": "none",
                "sampler": "long-tail natural sampler",
                "scheduler": "cosine with warmup grid",
                "warmup_steps_grid": "0;500",
                "newton_schulz_steps_grid": "not_applicable",
                "claim_role": "tuned SGD baseline",
            },
            {
                "recipe_family": "adamw_cb_loss_tuned",
                "optimizer": "adamw",
                "matrix_update": "standard adaptive first order",
                "lr_grid": "1e-4;3e-4;1e-3",
                "weight_decay_grid": "1e-4;5e-4",
                "class_reweighting": "effective-number class-balanced loss beta=0.999;0.9999",
                "sampler": "long-tail natural sampler",
                "scheduler": "cosine with warmup grid",
                "warmup_steps_grid": "0;500",
                "newton_schulz_steps_grid": "not_applicable",
                "claim_role": "class-balanced AdamW baseline",
            },
            {
                "recipe_family": "adamw_cb_sampler_tuned",
                "optimizer": "adamw",
                "matrix_update": "standard adaptive first order",
                "lr_grid": "1e-4;3e-4",
                "weight_decay_grid": "1e-4;5e-4",
                "class_reweighting": "cross entropy",
                "sampler": "class-balanced sampler",
                "scheduler": "cosine with warmup grid",
                "warmup_steps_grid": "0;500",
                "newton_schulz_steps_grid": "not_applicable",
                "claim_role": "sampler baseline required before any practical claim",
            },
            {
                "recipe_family": "ns_muon_matrix_tuned",
                "optimizer": "ns_muon",
                "matrix_update": "finite Newton-Schulz matrix direction on Conv/Linear weights",
                "lr_grid": "1e-5;3e-5;1e-4;3e-4",
                "weight_decay_grid": "1e-4;5e-4",
                "class_reweighting": "none",
                "sampler": "long-tail natural sampler",
                "scheduler": "cosine with warmup grid",
                "warmup_steps_grid": "0;500;1000",
                "newton_schulz_steps_grid": "3;5;7",
                "claim_role": "candidate practical Muon recipe",
            },
            {
                "recipe_family": "ns_muon_cb_tuned",
                "optimizer": "ns_muon",
                "matrix_update": "finite Newton-Schulz matrix direction on Conv/Linear weights",
                "lr_grid": "1e-5;3e-5;1e-4",
                "weight_decay_grid": "1e-4;5e-4",
                "class_reweighting": "effective-number class-balanced loss beta=0.9999",
                "sampler": "long-tail natural sampler",
                "scheduler": "cosine with warmup grid",
                "warmup_steps_grid": "500;1000",
                "newton_schulz_steps_grid": "3;5;7",
                "claim_role": "candidate practical Muon class-balanced recipe",
            },
        ]
    )


def build_selection_rules() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "rule_id": "SEL-1-validation-only-selection",
                "rule": "Choose one recipe per family using validation_tuning seeds only.",
                "forbidden_action": "Selecting hyperparameters from final_claim seeds or from the published pilot deltas.",
            },
            {
                "rule_id": "SEL-2-primary-metric",
                "rule": "Primary validation objective is few balanced accuracy; ties break by all balanced accuracy.",
                "forbidden_action": "Switching the primary metric after seeing final results.",
            },
            {
                "rule_id": "SEL-3-paired-final",
                "rule": "Run final_claim seeds pairwise for every selected recipe and tuned baseline.",
                "forbidden_action": "Reporting unpaired recipe seeds or dropping failed seeds.",
            },
            {
                "rule_id": "SEL-4-familywise-error",
                "rule": "Treat final comparisons to tuned AdamW and tuned SGD as one family and report Holm-adjusted decisions.",
                "forbidden_action": "Claiming the best-looking pair without multiplicity adjustment.",
            },
            {
                "rule_id": "SEL-5-negative-results",
                "rule": "Publish the tuned result as a boundary if Muon fails tuned baselines.",
                "forbidden_action": "Suppressing negative tuned Muon outcomes while keeping local-drift motivation.",
            },
        ]
    )


def build_acceptance_gates() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "gate_id": "TB-1-complete-final-family",
                "claim_unblocked": "Any CIFAR-100-LT ResNet18 final-performance statement.",
                "pass_rule": "All selected recipes and tuned baselines have 10 paired final seeds with many/medium/few/all metrics.",
                "failure_claim": "not_ready",
            },
            {
                "gate_id": "TB-2-primary-few-accuracy",
                "claim_unblocked": "Practical Muon improves few-class long-tail performance.",
                "pass_rule": "Muon few balanced-accuracy paired CI lower endpoint exceeds both tuned AdamW and tuned SGD by more than zero with Holm-adjusted final comparisons.",
                "failure_claim": "no practical Muon performance advantage",
            },
            {
                "gate_id": "TB-3-no-all-class-collapse",
                "claim_unblocked": "Few-class gain is not bought by broad collapse.",
                "pass_rule": "Muon all-class balanced-accuracy paired CI lower endpoint is no worse than -0.01 against the best tuned baseline.",
                "failure_claim": "tradeoff only; no clean performance claim",
            },
            {
                "gate_id": "TB-4-full-reporting-surface",
                "claim_unblocked": "Benchmark table can enter the main paper.",
                "pass_rule": "Report many, medium, few, all balanced accuracy, loss, margin, per-seed rows, and paired differences.",
                "failure_claim": "appendix-only pilot context",
            },
            {
                "gate_id": "TB-5-scope-control",
                "claim_unblocked": "CIFAR-100-LT ResNet18 benchmark claim.",
                "pass_rule": "Wording names CIFAR-100-LT ResNet18 only unless additional datasets and architectures are run under a separate preregistered protocol.",
                "failure_claim": "broad optimizer claim forbidden",
            },
            {
                "gate_id": "TB-6-local-drift-separation",
                "claim_unblocked": "Mechanism and final-performance evidence can coexist without contradiction.",
                "pass_rule": "Paper separates matched-head-gain local drift from long-horizon final training, including negative tuned outcomes.",
                "failure_claim": "mechanism-only paper path remains",
            },
        ]
    )


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    frames = {
        "pilot_context": build_pilot_context(),
        "benchmark_scope": build_benchmark_scope(),
        "seed_split_contract": build_seed_split_contract(),
        "recipe_grid": build_recipe_grid(),
        "selection_rules": build_selection_rules(),
        "acceptance_gates": build_acceptance_gates(),
    }
    for name, frame in frames.items():
        frame.to_csv(RESULT_DIR / f"{name}.csv", index=False)

    pilot = frames["pilot_context"].copy()
    pilot_display = pilot[
        ["context_id", "frequency_group", "metric", "estimate", "ci95_low", "ci95_high", "allowed_use"]
    ].copy()
    for column in ("estimate", "ci95_low", "ci95_high"):
        pilot_display[column] = pilot_display[column].map(fmt)

    text = f"""# E11 CIFAR-100-LT Tuned Benchmark Protocol

This generated protocol is not a new final benchmark result. It turns the existing CIFAR-100-LT ResNet18 standard baseline, augmented recipe pilot, and negative NS-Muon pilot into a clean preregistered route for a tuned final-performance claim.

## Pilot Context Quarantine

The existing pilots are useful for risk assessment, but they cannot select final hyperparameters or justify a practical optimizer claim. Their allowed role is fixed here before the tuned final seeds are run.

{markdown_table(pilot_display, ["context_id", "frequency_group", "metric", "estimate", "ci95_low", "ci95_high", "allowed_use"])}

## Benchmark Scope

{markdown_table(frames["benchmark_scope"], ["scope_id", "dataset", "imbalance_factor", "architecture", "primary_metric", "secondary_metrics", "claim_scope", "status"])}

## Seed And Split Contract

{markdown_table(frames["seed_split_contract"], ["split_id", "seed_set", "role", "tuning_allowed", "outputs"])}

## Recipe Grid

{markdown_table(frames["recipe_grid"], ["recipe_family", "optimizer", "lr_grid", "weight_decay_grid", "class_reweighting", "sampler", "warmup_steps_grid", "newton_schulz_steps_grid", "claim_role"])}

## Selection Rules

{markdown_table(frames["selection_rules"], ["rule_id", "rule", "forbidden_action"])}

## Executable Validation Registry

The validation grid is materialized by `scripts/e11_run_cifar100_resnet_lt_tuned_benchmark.py --settings-only`, which writes `results/e11_cifar100_resnet_lt_tuned_benchmark/settings_registry.csv` and `execution_status.csv`. GPU validation cells are submitted with `scripts/slurm/e11_cifar100_resnet_lt_tuned_benchmark_validation.sbatch`; each Slurm array cell runs one registered validation setting on seeds `10..14`. The final claim split `20..29` remains untouched until validation selects recipes.

The frozen selection rule is materialized by `scripts/e11_write_cifar100_resnet_lt_tuned_benchmark_selection.py`. It writes `results/e11_cifar100_resnet_lt_tuned_benchmark/validation_selection/*` and `discussion/e11_cifar100_resnet_lt_tuned_benchmark_selection.md`, selecting one recipe per family only after every registered validation setting in that family has a summary.

## Acceptance Gates

{markdown_table(frames["acceptance_gates"], ["gate_id", "claim_unblocked", "pass_rule", "failure_claim"])}

## Current Claim Status

The current state remains `not_ready` for any tuned final-performance claim. A top-tier mechanism paper can still use the existing pilots as context, but any CIFAR-100-LT ResNet18 optimizer-performance statement must wait for the validation/final split contract above. A broad long-tail optimizer claim remains forbidden by this protocol.
"""
    write_markdown(DISCUSSION_PATH, text)
    print(f"saved tuned benchmark protocol to {RESULT_DIR} and {DISCUSSION_PATH}")


if __name__ == "__main__":
    main()
