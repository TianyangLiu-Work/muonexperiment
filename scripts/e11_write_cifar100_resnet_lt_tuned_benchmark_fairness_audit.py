from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.reporting import markdown_table, write_markdown


RESULT_ROOT = Path("results/e11_cifar100_resnet_lt_tuned_benchmark")
PROTOCOL_DIR = Path("results/e11_cifar100_resnet_lt_tuned_benchmark_protocol")
SELECTION_DIR = RESULT_ROOT / "validation_selection"
OUTPUT_DIR = RESULT_ROOT / "validation_fairness_audit"
DISCUSSION_PATH = Path("discussion/e11_cifar100_resnet_lt_tuned_benchmark_fairness_audit.md")


def _count_grid_items(value: object) -> int:
    if pd.isna(value) or str(value).strip() in {"", "not_applicable"}:
        return 0
    return len([part for part in str(value).split(";") if part.strip()])


def _bool_text(value: bool) -> str:
    return "yes" if value else "no"


def build_family_budget_matrix(settings: pd.DataFrame, recipe_grid: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    recipe_lookup = recipe_grid.set_index("recipe_family")
    total_settings = int(len(settings))
    for family, family_rows in settings.groupby("recipe_family", sort=True):
        recipe = recipe_lookup.loc[family]
        setting_count = int(len(family_rows))
        lr_values = sorted(str(value) for value in family_rows["lr"].dropna().unique())
        wd_values = sorted(str(value) for value in family_rows["weight_decay"].dropna().unique())
        warm_values = sorted(str(value) for value in family_rows["warmup_steps"].dropna().unique())
        ns_values = sorted(str(value) for value in family_rows["newton_schulz_steps"].dropna().unique())
        rows.append(
            {
                "recipe_family": family,
                "claim_role": recipe["claim_role"],
                "optimizer": recipe["optimizer"],
                "setting_count": setting_count,
                "setting_fraction": setting_count / total_settings,
                "lr_count": len(lr_values),
                "weight_decay_count": len(wd_values),
                "warmup_count": len(warm_values),
                "newton_schulz_count": len([value for value in ns_values if value != ""]),
                "class_reweighting": recipe["class_reweighting"],
                "sampler": recipe["sampler"],
                "validation_seed_set": ";".join(sorted(family_rows["seed_set"].astype(str).unique())),
                "budget_interpretation": (
                    "tuned_baseline_surface"
                    if "baseline" in str(recipe["claim_role"]).lower()
                    else "candidate_search_budget_disclose_not_final_advantage"
                ),
            }
        )
    return pd.DataFrame(rows)


def build_seed_metric_parity(
    settings: pd.DataFrame,
    seed_split: pd.DataFrame,
    selection_rules: pd.DataFrame,
    acceptance_gates: pd.DataFrame,
) -> pd.DataFrame:
    rule_text = " ".join(selection_rules.astype(str).to_numpy().ravel())
    gate_text = " ".join(acceptance_gates.astype(str).to_numpy().ravel())
    final_seed_rows = seed_split[seed_split["split_id"].eq("final_claim")]
    return pd.DataFrame(
        [
            {
                "parity_id": "TBF-P1-validation-seed-parity",
                "status": "pass" if set(settings["seed_set"].astype(str)) == {"10..14"} else "fail",
                "evidence": "all validation rows use seed_set 10..14",
                "forbidden_asymmetry": "changing validation seeds by optimizer family",
            },
            {
                "parity_id": "TBF-P2-final-seed-parity",
                "status": "pass"
                if len(final_seed_rows) == 1
                and str(final_seed_rows.iloc[0]["seed_set"]) == "20..29"
                and str(final_seed_rows.iloc[0]["tuning_allowed"]) == "no"
                else "fail",
                "evidence": "final_claim seed_set 20..29 tuning_allowed=no",
                "forbidden_asymmetry": "letting one optimizer family tune on final seeds",
            },
            {
                "parity_id": "TBF-P3-primary-metric-parity",
                "status": "pass" if "few balanced accuracy" in rule_text else "fail",
                "evidence": "selection_rules.csv fixes few balanced accuracy for every family",
                "forbidden_asymmetry": "using a different validation objective for Muon and baselines",
            },
            {
                "parity_id": "TBF-P4-multiplicity-parity",
                "status": "pass" if "Holm-adjusted" in rule_text and "tuned AdamW and tuned SGD" in gate_text else "fail",
                "evidence": "final comparisons to tuned AdamW and tuned SGD are one Holm-adjusted family",
                "forbidden_asymmetry": "claiming only the best-looking unadjusted comparison",
            },
            {
                "parity_id": "TBF-P5-reporting-surface-parity",
                "status": "pass"
                if all(phrase in gate_text for phrase in ["many", "medium", "few", "all balanced accuracy"])
                else "fail",
                "evidence": "acceptance gates require many/medium/few/all reporting",
                "forbidden_asymmetry": "reporting tail-only gains while hiding all-class collapse",
            },
        ]
    )


def build_fairness_gate_matrix(
    settings: pd.DataFrame,
    family_budget: pd.DataFrame,
    seed_metric_parity: pd.DataFrame,
    gate_report: pd.DataFrame,
) -> pd.DataFrame:
    families = set(settings["recipe_family"].astype(str))
    baseline_families = {"adamw_ce_tuned", "sgd_momentum_ce_tuned", "adamw_cb_loss_tuned", "adamw_cb_sampler_tuned"}
    muon_families = {"ns_muon_matrix_tuned", "ns_muon_cb_tuned"}
    baseline_budget = int(
        family_budget[family_budget["recipe_family"].isin(baseline_families)]["setting_count"].sum()
    )
    muon_budget = int(family_budget[family_budget["recipe_family"].isin(muon_families)]["setting_count"].sum())
    ce_baseline_counts = family_budget.set_index("recipe_family")["setting_count"].astype(int).to_dict()
    gate_lookup = gate_report.set_index("gate_id")["status"].astype(str).to_dict()
    parity_pass = bool(seed_metric_parity["status"].eq("pass").all())
    return pd.DataFrame(
        [
            {
                "gate_id": "TBF-1-registered-family-coverage",
                "status": "pass" if families == baseline_families.union(muon_families) and len(settings) == 164 else "fail",
                "evidence": f"{len(families)} recipe families; {len(settings)} registered settings",
                "claim_effect": "benchmark scope names every tuned family rather than a hand-picked subset",
            },
            {
                "gate_id": "TBF-2-baseline-tuning-surface",
                "status": "pass"
                if ce_baseline_counts.get("adamw_ce_tuned", 0) >= 12
                and ce_baseline_counts.get("sgd_momentum_ce_tuned", 0) >= 12
                and baseline_families.issubset(families)
                else "fail",
                "evidence": (
                    f"AdamW CE={ce_baseline_counts.get('adamw_ce_tuned', 0)}; "
                    f"SGD CE={ce_baseline_counts.get('sgd_momentum_ce_tuned', 0)}; "
                    "class-balanced loss and sampler baselines present"
                ),
                "claim_effect": "Muon cannot be compared only to undertuned vanilla baselines",
            },
            {
                "gate_id": "TBF-3-candidate-budget-disclosure",
                "status": "pass_with_disclosure" if muon_budget > baseline_budget else "pass",
                "evidence": f"Muon candidate settings={muon_budget}; baseline settings={baseline_budget}",
                "claim_effect": "larger Muon validation budget must be disclosed and cannot count as final evidence",
            },
            {
                "gate_id": "TBF-4-seed-metric-reporting-parity",
                "status": "pass" if parity_pass else "fail",
                "evidence": "validation seeds, final seeds, primary metric, Holm family, and reporting surface are shared",
                "claim_effect": "final paired comparison uses one rule set for every optimizer family",
            },
            {
                "gate_id": "TBF-5-final-gates-still-blocked",
                "status": "pass"
                if gate_lookup.get("TVS-1-validation-grid-complete") != "pass"
                and gate_lookup.get("TVS-3-final-seed-quarantine") == "pass"
                else "fail",
                "evidence": (
                    f"TVS-1={gate_lookup.get('TVS-1-validation-grid-complete')}; "
                    f"TVS-3={gate_lookup.get('TVS-3-final-seed-quarantine')}"
                ),
                "claim_effect": "fairness audit does not authorize final-performance wording from partial validation",
            },
        ]
    )


def build_budget_disclosure_contract(family_budget: pd.DataFrame) -> pd.DataFrame:
    total_baseline = int(
        family_budget[family_budget["budget_interpretation"].eq("tuned_baseline_surface")]["setting_count"].sum()
    )
    total_muon = int(
        family_budget[
            family_budget["budget_interpretation"].eq("candidate_search_budget_disclose_not_final_advantage")
        ]["setting_count"].sum()
    )
    return pd.DataFrame(
        [
            {
                "contract_id": "TBF-C1-validation-budget-disclosure",
                "status": "active",
                "required_wording": (
                    f"report baseline validation budget={total_baseline} settings and "
                    f"Muon candidate validation budget={total_muon} settings"
                ),
                "forbidden_wording": "describing validation leaderboard rank as final benchmark performance",
            },
            {
                "contract_id": "TBF-C2-final-evidence-parity",
                "status": "active",
                "required_wording": "final claims require paired seed set 20..29 for every selected family",
                "forbidden_wording": "using a larger validation search budget as evidence of a final optimizer advantage",
            },
            {
                "contract_id": "TBF-C3-baseline-strength-boundary",
                "status": "active",
                "required_wording": "compare against tuned AdamW, tuned SGD, and class-balanced AdamW baselines",
                "forbidden_wording": "claiming practical Muon superiority against only the weak reporting baseline or pilots",
            },
        ]
    )


def write_discussion(
    family_budget: pd.DataFrame,
    seed_metric_parity: pd.DataFrame,
    fairness_gates: pd.DataFrame,
    budget_contract: pd.DataFrame,
) -> None:
    text = f"""# E11 CIFAR-100-LT Tuned Benchmark Fairness Audit

This generated audit checks whether the tuned CIFAR-100-LT ResNet18 benchmark is reviewer-fair before any final-performance claim. It is a validation-budget and comparison-parity audit: it permits budget disclosure and progress accounting, but it does not convert partial validation rows into benchmark evidence.

## Family Budget Matrix

{markdown_table(family_budget, ["recipe_family", "claim_role", "optimizer", "setting_count", "setting_fraction", "lr_count", "weight_decay_count", "warmup_count", "newton_schulz_count", "class_reweighting", "sampler", "validation_seed_set", "budget_interpretation"])}

## Seed And Metric Parity

{markdown_table(seed_metric_parity, ["parity_id", "status", "evidence", "forbidden_asymmetry"])}

## Fairness Gate Matrix

{markdown_table(fairness_gates, ["gate_id", "status", "evidence", "claim_effect"])}

## Budget Disclosure Contract

{markdown_table(budget_contract, ["contract_id", "status", "required_wording", "forbidden_wording"])}

## Operating Rule

A larger Muon validation search budget is allowed only as a disclosed candidate-search budget. It is not final evidence. Any practical-performance wording must wait for complete validation selection, shared final seed set `20..29`, Holm-adjusted final comparisons against tuned AdamW and tuned SGD, all-class guardrails, and complete many/medium/few/all reporting for every selected recipe.
"""
    write_markdown(DISCUSSION_PATH, text)


def main() -> None:
    settings = pd.read_csv(RESULT_ROOT / "settings_registry.csv")
    recipe_grid = pd.read_csv(PROTOCOL_DIR / "recipe_grid.csv")
    seed_split = pd.read_csv(PROTOCOL_DIR / "seed_split_contract.csv")
    selection_rules = pd.read_csv(PROTOCOL_DIR / "selection_rules.csv")
    acceptance_gates = pd.read_csv(PROTOCOL_DIR / "acceptance_gates.csv")
    gate_report = pd.read_csv(SELECTION_DIR / "gate_report.csv")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    family_budget = build_family_budget_matrix(settings, recipe_grid)
    seed_metric_parity = build_seed_metric_parity(settings, seed_split, selection_rules, acceptance_gates)
    fairness_gates = build_fairness_gate_matrix(settings, family_budget, seed_metric_parity, gate_report)
    budget_contract = build_budget_disclosure_contract(family_budget)

    family_budget.to_csv(OUTPUT_DIR / "family_budget_matrix.csv", index=False)
    seed_metric_parity.to_csv(OUTPUT_DIR / "seed_metric_parity.csv", index=False)
    fairness_gates.to_csv(OUTPUT_DIR / "fairness_gate_matrix.csv", index=False)
    budget_contract.to_csv(OUTPUT_DIR / "budget_disclosure_contract.csv", index=False)
    config = {
        "scope": "validation_budget_and_comparison_fairness",
        "claim_authority": "fairness_disclosure_only_until_final_gates_pass",
        "settings_registry": (RESULT_ROOT / "settings_registry.csv").as_posix(),
        "recipe_grid": (PROTOCOL_DIR / "recipe_grid.csv").as_posix(),
        "selection_rules": (PROTOCOL_DIR / "selection_rules.csv").as_posix(),
        "gate_report": (SELECTION_DIR / "gate_report.csv").as_posix(),
        "registered_settings": int(len(settings)),
    }
    (OUTPUT_DIR / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    write_discussion(family_budget, seed_metric_parity, fairness_gates, budget_contract)
    print(f"saved tuned benchmark fairness audit to {OUTPUT_DIR} and {DISCUSSION_PATH}")


if __name__ == "__main__":
    main()
