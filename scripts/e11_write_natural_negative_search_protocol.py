from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.reporting import fmt, markdown_table, write_markdown


RESULT_DIR = Path("results/e11_natural_negative_search_protocol")
DISCUSSION_PATH = Path("discussion/e11_natural_negative_search_protocol.md")
AUDIT_SUMMARY_PATH = Path("results/e11_natural_head_tail_boundary/boundary_summary.csv")


def load_audit_baseline() -> pd.DataFrame:
    summary = pd.read_csv(AUDIT_SUMMARY_PATH)
    lookup = summary.set_index("summary_id")
    primary = lookup.loc["primary_tail_output_drift"]
    component = lookup.loc["component_ratio_metrics"]
    secondary = lookup.loc["secondary_tail_outcomes"]
    return pd.DataFrame(
        [
            {
                "baseline_id": "committed_natural_primary_full_drift_scan",
                "source_artifact": "results/e11_natural_head_tail_boundary/primary_drift_scan.csv",
                "row_count": int(primary["row_count"]),
                "source_count": int(primary["source_count"]),
                "strict_worse_count": int(primary["spectral_worse_count"]),
                "max_ci95_high": float(primary["max_ci95_high"]),
                "status": str(primary["claim_status"]),
                "claim_use": "baseline null scan only; not a fresh negative-search result",
            },
            {
                "baseline_id": "committed_component_ratio_boundaries",
                "source_artifact": "results/e11_natural_head_tail_boundary/candidate_negative_cases.csv",
                "row_count": int(component["row_count"]),
                "source_count": int(component["source_count"]),
                "strict_worse_count": int(component["spectral_worse_count"]),
                "max_ci95_high": float(component["max_ci95_high"]),
                "status": str(component["claim_status"]),
                "claim_use": "claim-boundary evidence for true-logit/margin components, not primary full-drift failures",
            },
            {
                "baseline_id": "committed_secondary_outcome_tradeoffs",
                "source_artifact": "results/e11_natural_head_tail_boundary/secondary_outcome_scan.csv",
                "row_count": int(secondary["row_count"]),
                "source_count": int(secondary["source_count"]),
                "strict_worse_count": int(secondary["spectral_worse_count"]),
                "max_ci95_high": float(secondary["max_ci95_high"]),
                "status": str(secondary["claim_status"]),
                "claim_use": "claim-boundary evidence for loss, margin, and accuracy outcomes",
            },
        ]
    )


def build_search_space_registry() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "search_id": "NNS-P1-cifar100lt-resnet18-new-partitions",
                "phase": "phase1_fresh_primary_search",
                "dataset": "CIFAR-100-LT",
                "architecture": "ResNet18 CIFAR stem",
                "partition_family": "unspent modulo and semantic class partitions not used in committed natural audit",
                "checkpoint_steps": "500/2000/5000",
                "target_head_gain_fraction": "0.002/0.005",
                "seeds_per_setting": "at least 5",
                "max_settings": 12,
                "compute_mode": "GPU via Slurm",
                "freshness_rule": "exclude every source_id and exact setting_id in results/e11_natural_head_tail_boundary/primary_drift_scan.csv",
                "planned_artifact_prefix": "results/e11_natural_negative_search_protocol/phase1_cifar100lt_resnet18",
                "entrypoint": "scripts/slurm/e11_natural_negative_search_phase1.sbatch",
                "entrypoint_status": "implemented_sbatch",
            },
            {
                "search_id": "NNS-P1-cifar10lt-resnet18-cross-partitions",
                "phase": "phase1_fresh_primary_search",
                "dataset": "CIFAR-10-LT",
                "architecture": "ResNet18 CIFAR stem",
                "partition_family": "unspent head/tail cross-partitions disjoint from original, alternate, mixed, and cross v5 final partitions",
                "checkpoint_steps": "500/5000",
                "target_head_gain_fraction": "0.002/0.005",
                "seeds_per_setting": "at least 5",
                "max_settings": 8,
                "compute_mode": "GPU via Slurm",
                "freshness_rule": "do not reuse v2/v3/v4/v5 condition-score final partitions for natural-negative selection",
                "planned_artifact_prefix": "results/e11_natural_negative_search_protocol/phase1_cifar10lt_resnet18",
                "entrypoint": "scripts/slurm/e11_natural_negative_search_phase1.sbatch",
                "entrypoint_status": "implemented_sbatch",
            },
            {
                "search_id": "NNS-P1-tail-quality-controls",
                "phase": "phase1_fresh_primary_search",
                "dataset": "CIFAR-100",
                "architecture": "ResNet18 CIFAR stem",
                "partition_family": "tail-rich controls with new tail class subsets and tail_train_per_class >= 300",
                "checkpoint_steps": "2000/5000/10000",
                "target_head_gain_fraction": "0.002/0.005",
                "seeds_per_setting": "at least 5",
                "max_settings": 6,
                "compute_mode": "GPU via Slurm",
                "freshness_rule": "exclude committed tail_quality_control warmup/partition rows",
                "planned_artifact_prefix": "results/e11_natural_negative_search_protocol/phase1_tail_quality_controls",
                "entrypoint": "scripts/slurm/e11_natural_negative_search_phase1.sbatch",
                "entrypoint_status": "implemented_sbatch",
            },
            {
                "search_id": "NNS-P2-heldout-architecture-boundary",
                "phase": "phase2_fresh_generality_search",
                "dataset": "CIFAR-100-LT",
                "architecture": "ResNet34 CIFAR stem",
                "partition_family": "reuse only the phase1-declared class partitions, not outcome-selected partitions",
                "checkpoint_steps": "2000/5000",
                "target_head_gain_fraction": "0.002/0.005",
                "seeds_per_setting": "3",
                "max_settings": 8,
                "compute_mode": "GPU via Slurm",
                "freshness_rule": "architecture fixed after complete phase1 archive; partitions are phase1-declared and not outcome-selected",
                "planned_artifact_prefix": "results/e11_natural_negative_search_protocol/phase2_heldout_architecture",
                "entrypoint": "scripts/slurm/e11_natural_negative_search_phase2.sbatch",
                "entrypoint_status": "implemented_sbatch",
            },
        ]
    )


def build_metric_contract() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "metric_id": "primary_tail_output_drift_ratio",
                "claim_role": "primary natural counterexample",
                "estimator": "geomean_tail_output_drift_sq_ratio_spectral_over_fro",
                "worse_rule": "simultaneous or Holm-adjusted 95% lower confidence endpoint is above 1",
                "multiplicity_family": "all fresh primary settings within a phase",
                "required_report": "per-seed paired log-ratio rows, estimate, raw CI, adjusted CI or adjusted one-sided p-value, pre-update tail quality, head-gain error",
                "claim_boundary": "only this metric can support a natural primary full-drift counterexample",
            },
            {
                "metric_id": "centered_tail_output_drift_ratio",
                "claim_role": "primary robustness check",
                "estimator": "geomean_centered_tail_output_drift_sq_ratio_spectral_over_fro",
                "worse_rule": "same adjusted CI rule as primary, but interpreted as robustness not primary discovery",
                "multiplicity_family": "reported alongside primary family",
                "required_report": "estimate and adjusted decision for every fresh setting",
                "claim_boundary": "cannot replace the full tail-output drift metric",
            },
            {
                "metric_id": "true_logit_and_margin_components",
                "claim_role": "component boundary",
                "estimator": "true_logit_delta, competitor_logit_delta, and margin_delta spectral/Frobenius ratios",
                "worse_rule": "raw CI lower endpoint above 1, labeled component-only unless primary also fails",
                "multiplicity_family": "component exploratory family",
                "required_report": "all component rows, including settings where primary drift remains below 1",
                "claim_boundary": "constrains stronger logit-component or margin claims",
            },
            {
                "metric_id": "tail_loss_margin_accuracy_diffs",
                "claim_role": "secondary outcome tradeoff",
                "estimator": "mean spectral-minus-Frobenius tail loss increase, margin drop, and accuracy drop",
                "worse_rule": "raw CI lower endpoint above 0, labeled secondary-only unless primary also fails",
                "multiplicity_family": "secondary exploratory family",
                "required_report": "loss, margin, accuracy, and prediction-change rows for every fresh setting",
                "claim_boundary": "cannot support a primary drift counterexample or final optimizer-performance claim",
            },
            {
                "metric_id": "head_gain_and_quality_controls",
                "claim_role": "validity control",
                "estimator": "actual head-gain relative error, pre-update tail accuracy, and checkpoint quality",
                "worse_rule": "setting is excluded from primary claim if matched head-gain or quality gates fail",
                "multiplicity_family": "not a discovery metric",
                "required_report": "head-gain relative error for both directions and pre-update many/medium/few or tail metrics",
                "claim_boundary": "prevents degenerate negative cases from mismatched head progress or unusable checkpoints",
            },
        ]
    )


def build_stopping_rules() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "rule_id": "NNS-S1-freeze-before-fresh-runs",
                "phase": "all",
                "rule": "Search space, metric contract, multiplicity procedure, and exclusion list are committed before any fresh outputs are generated.",
                "decision": "outputs generated before this protocol are audit baselines only and cannot support fresh natural-negative claims",
            },
            {
                "rule_id": "NNS-S2-complete-phase-before-discovery",
                "phase": "phase1",
                "rule": "Run every declared phase1 setting, except for logged infrastructure failures, before declaring a natural primary counterexample.",
                "decision": "no early stopping on the first negative-looking setting",
            },
            {
                "rule_id": "NNS-S3-primary-success",
                "phase": "phase1_or_phase2",
                "rule": "At least one fresh primary_tail_output_drift_ratio row has adjusted CI lower endpoint above 1 and passes head-gain and quality controls.",
                "decision": "label as fresh natural primary boundary candidate; require replication or held-out architecture before broad claim",
            },
            {
                "rule_id": "NNS-S4-finite-null",
                "phase": "phase1_or_phase2",
                "rule": "All declared settings run, no adjusted primary lower endpoint exceeds 1, and all rows are reported.",
                "decision": "report finite null search; do not claim absence of natural counterexamples outside the registered search space",
            },
            {
                "rule_id": "NNS-S5-component-only",
                "phase": "phase1_or_phase2",
                "rule": "Component or secondary metrics are worse but primary full-drift metric does not pass NNS-S3.",
                "decision": "report as claim-boundary evidence only",
            },
            {
                "rule_id": "NNS-S6-phase2-trigger",
                "phase": "phase2",
                "rule": "Phase2 architecture search is allowed only after phase1 results are archived without choosing architectures from phase1 outcomes.",
                "decision": "phase2 can support generality or replication, not retroactive phase1 selection",
            },
        ]
    )


def build_acceptance_gates() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "gate_id": "NNS-1-protocol-freeze",
                "scope": "fresh natural negative search",
                "requirement": "Protocol tables and discussion define the search and claim boundary before fresh outputs are used for any claim.",
                "pass_condition": "protocol_status.csv records metric_outputs_not_run before jobs or partial_metric_outputs/metric_outputs_complete after jobs; natural claims stay blocked until the evaluator completeness gate passes",
            },
            {
                "gate_id": "NNS-2-freshness-exclusion",
                "scope": "search space",
                "requirement": "No committed-audit setting_id or spent condition-score final partition is reused for selection.",
                "pass_condition": "fresh evaluator writes an exclusion audit with no overlap against the baseline audit and spent-final registries",
            },
            {
                "gate_id": "NNS-3-multiplicity",
                "scope": "statistical decision",
                "requirement": "Primary discovery uses simultaneous confidence intervals or Holm-adjusted one-sided bootstrap tests over the phase family.",
                "pass_condition": "every primary decision row includes raw CI, adjusted CI or adjusted p-value, and phase family size",
            },
            {
                "gate_id": "NNS-4-full-reporting",
                "scope": "artifact reporting",
                "requirement": "All declared settings are reported, including nulls, component-only reversals, quality failures, and infrastructure failures.",
                "pass_condition": "registry row count equals the declared search-space count minus logged infrastructure failures",
            },
            {
                "gate_id": "NNS-5-quality-controls",
                "scope": "claim validity",
                "requirement": "Primary negative candidates pass matched head-gain and pre-update tail-quality controls.",
                "pass_condition": "candidate rows include head-gain relative error within the registered tolerance and nondegenerate pre-update tail metrics",
            },
            {
                "gate_id": "NNS-6-claim-boundary",
                "scope": "paper claim",
                "requirement": "Component and secondary tradeoffs are not described as primary full-drift counterexamples.",
                "pass_condition": "claim ledger separates primary, component, secondary, and final-performance claims",
            },
        ]
    )


def phase1_gate_status() -> dict[str, str]:
    gate_path = RESULT_DIR / "phase1_multiplicity_evaluation" / "gate_report.csv"
    if not gate_path.exists():
        return {}
    gates = pd.read_csv(gate_path)
    return gates.set_index("gate_id")["status"].astype(str).to_dict()


def build_claim_ladder() -> pd.DataFrame:
    gate_status = phase1_gate_status()
    finite_null_candidate = (
        gate_status.get("NNS-E2-phase1-output-completeness") == "pass"
        and gate_status.get("NNS-E3-primary-multiplicity") == "pass"
        and gate_status.get("NNS-E4-natural-primary-claim") == "finite_null_candidate"
    )
    primary_counterexample_status = "not_found_in_phase1" if finite_null_candidate else "not_ready"
    finite_null_status = "finite_null_candidate" if finite_null_candidate else "not_ready"
    return pd.DataFrame(
        [
            {
                "claim_id": "fresh_natural_primary_counterexample",
                "current_status": primary_counterexample_status,
                "unlock_condition": "NNS-S3 passes on a fresh phase and NNS-1 through NNS-6 pass",
                "blocked_if": "only component or secondary metrics reverse",
                "paper_wording_if_unlocked": "a registered natural setting where spectral/polar increases full tail-output drift at matched head gain",
            },
            {
                "claim_id": "finite_natural_null_search",
                "current_status": finite_null_status,
                "unlock_condition": "NNS-S4 passes after all declared fresh phase settings run",
                "blocked_if": "fresh outputs are incomplete, selected post hoc, or used outside the registered phase1 space",
                "paper_wording_if_unlocked": "a finite pre-registered natural search found no primary full-drift counterexample in the declared space",
            },
            {
                "claim_id": "component_boundary_cases",
                "current_status": "supported_by_committed_audit",
                "unlock_condition": "already supported as claim-boundary evidence by results/e11_natural_head_tail_boundary/candidate_negative_cases.csv",
                "blocked_if": "used as a primary drift or final-performance counterexample",
                "paper_wording_if_unlocked": "true-logit and secondary outcome metrics can move against the full-drift direction in natural diagnostics",
            },
            {
                "claim_id": "local_primary_full_drift_mechanism",
                "current_status": "unchanged",
                "unlock_condition": "existing positive diagnostics plus current caveats",
                "blocked_if": "paper claims final loss, accuracy, or universal optimizer superiority from local drift",
                "paper_wording_if_unlocked": "spectral/polar can lower local full tail-output drift at matched head gain in the tested diagnostics",
            },
            {
                "claim_id": "practical_optimizer_performance",
                "current_status": "blocked",
                "unlock_condition": "separate tuned benchmark with final metrics",
                "blocked_if": "only local negative-search evidence exists",
                "paper_wording_if_unlocked": "not claimable from this protocol",
            },
        ]
    )


def phase1_output_status(search_space: pd.DataFrame) -> pd.DataFrame:
    rows = []
    phase1 = search_space[search_space["phase"].eq("phase1_fresh_primary_search")]
    for search in phase1.itertuples():
        prefix = Path(search.planned_artifact_prefix)
        files = {
            "settings_registry": prefix / "settings_registry.csv",
            "step_metrics": prefix / "step_metrics.csv",
            "pair_summary": prefix / "pair_summary.csv",
            "layer_metrics": prefix / "layer_metrics.csv",
            "decision_template": prefix / "decision_template.csv",
            "config": prefix / "config.json",
        }
        required_metric_files = ["step_metrics", "pair_summary", "layer_metrics", "decision_template", "config"]
        settings_rows = len(pd.read_csv(files["settings_registry"])) if files["settings_registry"].exists() else 0
        pair_rows = len(pd.read_csv(files["pair_summary"])) if files["pair_summary"].exists() else 0
        metric_files_present = all(files[name].exists() for name in required_metric_files)
        if metric_files_present and pair_rows == int(search.max_settings):
            status = "complete"
        elif files["settings_registry"].exists() and not any(files[name].exists() for name in required_metric_files):
            status = "settings_only_no_metrics"
        elif any(path.exists() for path in files.values()):
            status = "incomplete_metrics"
        else:
            status = "not_run"
        rows.append(
            {
                "search_id": search.search_id,
                "expected_settings": int(search.max_settings),
                "settings_registry_rows": int(settings_rows),
                "primary_metric_rows": int(pair_rows),
                "output_status": status,
            }
        )
    return pd.DataFrame(rows)


def build_protocol_status(audit_baseline: pd.DataFrame, search_space: pd.DataFrame) -> pd.DataFrame:
    primary = audit_baseline.set_index("baseline_id").loc["committed_natural_primary_full_drift_scan"]
    phase1_status = phase1_output_status(search_space)
    phase2 = search_space[search_space["phase"].eq("phase2_fresh_generality_search")]
    phase2_settings_rows = 0
    phase2_expected_rows = int(phase2["max_settings"].sum()) if not phase2.empty else 0
    for search in phase2.itertuples():
        registry_path = Path(search.planned_artifact_prefix) / "settings_registry.csv"
        if registry_path.exists():
            phase2_settings_rows += len(pd.read_csv(registry_path))
    gate_status = phase1_gate_status()
    expected_metric_rows = int(phase1_status["expected_settings"].sum())
    observed_metric_rows = int(phase1_status["primary_metric_rows"].sum())
    status_fragments = [
        f"{row.search_id}={row.output_status} ({int(row.primary_metric_rows)}/{int(row.expected_settings)} rows)"
        for row in phase1_status.itertuples()
    ]
    if observed_metric_rows == 0:
        output_status = "metric_outputs_not_run"
        output_evidence = "phase1 metric files are absent until the submitted GPU jobs finish"
        evaluator_status = "implemented_pending_outputs"
        claim_evidence = "no fresh search outputs exist under this protocol"
    elif observed_metric_rows == expected_metric_rows and phase1_status["output_status"].eq("complete").all():
        output_status = "metric_outputs_complete"
        output_evidence = "; ".join(status_fragments)
        evaluator_status = "implemented_complete_outputs"
        if gate_status.get("NNS-E4-natural-primary-claim") == "finite_null_candidate":
            claim_status = "finite_null_candidate"
            claim_evidence = (
                f"{observed_metric_rows}/{expected_metric_rows} fresh metric rows exist; "
                "NNS-E4-natural-primary-claim=finite_null_candidate with no adjusted primary worse row"
            )
        else:
            claim_status = "pending_evaluator_decision"
            claim_evidence = "fresh outputs are complete, but claims require the multiplicity evaluator decision table"
    else:
        output_status = "partial_metric_outputs"
        output_evidence = f"{observed_metric_rows}/{expected_metric_rows} phase1 settings have primary metric rows; " + "; ".join(status_fragments)
        evaluator_status = "implemented_partial_outputs"
        claim_status = "not_ready"
        claim_evidence = f"{observed_metric_rows}/{expected_metric_rows} fresh metric rows exist, so natural claims remain blocked until the family is complete"
    if observed_metric_rows == 0:
        claim_status = "not_ready"
    return pd.DataFrame(
        [
            {
                "item": "committed natural audit baseline",
                "status": "loaded",
                "evidence": (
                    f"{int(primary['row_count'])} primary rows; strict worse count "
                    f"{int(primary['strict_worse_count'])}; max CI high {fmt(primary['max_ci95_high'])}"
                ),
                "blocks_stronger_claim_if_missing": "yes",
            },
            {
                "item": "fresh natural search protocol",
                "status": "generated",
                "evidence": "search space, metric contract, stopping rules, gates, and claim ladder written",
                "blocks_stronger_claim_if_missing": "yes",
            },
            {
                "item": "fresh natural search entrypoints",
                "status": "implemented",
                "evidence": (
                    "scripts/e11_run_natural_negative_search_phase1.py, "
                    "scripts/slurm/e11_natural_negative_search_phase1.sbatch, "
                    "scripts/e11_run_natural_negative_search_phase2.py, and "
                    "scripts/slurm/e11_natural_negative_search_phase2.sbatch are registered; "
                    "scripts/e11_evaluate_natural_negative_search_phase2.py fixes the phase2 decision gate; "
                    "scripts/e11_write_natural_negative_phase2_power_audit.py fixes the phase2 detectable-effect boundary"
                ),
                "blocks_stronger_claim_if_missing": "yes",
            },
            {
                "item": "fresh natural search settings registries",
                "status": "locked",
                "evidence": (
                    "phase1_cifar100lt_resnet18, phase1_cifar10lt_resnet18, and "
                    "phase1_tail_quality_controls settings_registry.csv files declare 26 total settings; "
                    f"phase2_heldout_architecture settings_registry.csv declares {phase2_settings_rows}/"
                    f"{phase2_expected_rows} ResNet34 held-out architecture settings"
                ),
                "blocks_stronger_claim_if_missing": "yes",
            },
            {
                "item": "fresh natural search outputs",
                "status": output_status,
                "evidence": output_evidence,
                "blocks_stronger_claim_if_missing": "yes",
            },
            {
                "item": "multiplicity-adjusted evaluator",
                "status": evaluator_status,
                "evidence": (
                    "scripts/e11_evaluate_natural_negative_search_phase1.py writes per-seed log-ratio rows "
                    "and Holm-adjusted decision rows under results/e11_natural_negative_search_protocol/phase1_multiplicity_evaluation"
                ),
                "blocks_stronger_claim_if_missing": "yes",
            },
            {
                "item": "natural negative claim",
                "status": claim_status,
                "evidence": claim_evidence,
                "blocks_stronger_claim_if_missing": "yes",
            },
        ]
    )


def write_outputs(
    audit_baseline: pd.DataFrame,
    search_space: pd.DataFrame,
    metric_contract: pd.DataFrame,
    stopping_rules: pd.DataFrame,
    gates: pd.DataFrame,
    claim_ladder: pd.DataFrame,
    status: pd.DataFrame,
) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    audit_baseline.to_csv(RESULT_DIR / "audit_baseline.csv", index=False)
    search_space.to_csv(RESULT_DIR / "search_space_registry.csv", index=False)
    metric_contract.to_csv(RESULT_DIR / "metric_contract.csv", index=False)
    stopping_rules.to_csv(RESULT_DIR / "stopping_rules.csv", index=False)
    gates.to_csv(RESULT_DIR / "acceptance_gates.csv", index=False)
    claim_ladder.to_csv(RESULT_DIR / "claim_ladder.csv", index=False)
    status.to_csv(RESULT_DIR / "protocol_status.csv", index=False)

    natural_claim_status = status.set_index("item").loc["natural negative claim", "status"]
    if natural_claim_status == "finite_null_candidate":
        allowed_now = (
            "cite the committed natural-boundary audit and report the registered 26-setting "
            "phase1 primary family as a finite-null candidate with detectable-effect and "
            "tail-quality caveats."
        )
        blocked_now = (
            "claiming a fresh natural primary counterexample, claiming absence of natural "
            "counterexamples outside the registered phase1 space, or using quality-failed rows "
            "as mechanism validation."
        )
        phase1_boundary = (
            "The phase1 Slurm outputs are complete and the multiplicity evaluator reports no "
            "adjusted primary worse row. This unlocks only finite registered phase1 wording; "
            "it does not prove a universal natural null."
        )
    else:
        allowed_now = (
            "cite the committed natural-boundary audit as a finite baseline null for primary "
            "full-drift rows and as component/outcome claim-boundary evidence."
        )
        blocked_now = (
            "claiming a fresh natural primary counterexample, a finite pre-registered null "
            "search, or a practical optimizer-performance result from this protocol."
        )
        phase1_boundary = (
            "The phase1 Slurm entrypoint and multiplicity evaluator are implemented, but "
            "those claims still require complete fresh metric outputs and Holm-adjusted "
            "decisions from paired per-seed log-ratio tests that satisfy the acceptance gates above."
        )
    phase2_registry = RESULT_DIR / "phase2_heldout_architecture" / "settings_registry.csv"
    if phase2_registry.exists():
        phase2_boundary = (
            "The ResNet34 held-out architecture settings registry is frozen at "
            "`results/e11_natural_negative_search_protocol/phase2_heldout_architecture/settings_registry.csv`; "
            "the phase2 evaluator is frozen at "
            "`results/e11_natural_negative_search_protocol/phase2_multiplicity_evaluation/*`; "
            "the phase2 detectable-effect audit is frozen at "
            "`results/e11_natural_negative_search_protocol/phase2_power_audit/*`; "
            "no phase2 metric rows are used until the Slurm job completes."
        )
    else:
        phase2_boundary = (
            "The phase2 ResNet34 held-out architecture entrypoint is implemented, but the settings registry has not been written."
        )

    text = f"""# E11 Natural Negative Search Protocol

This generated protocol upgrades the natural-boundary work from a committed
audit to a pre-registered fresh search. It does not claim a new natural counterexample.
Its purpose is to prevent post-hoc selection when searching for natural settings
where the matched-head-gain spectral/polar direction is worse than Frobenius on
the primary full tail-output drift metric.

## Audit Baseline

{markdown_table(audit_baseline, ["baseline_id", "row_count", "source_count", "strict_worse_count", "max_ci95_high", "status", "claim_use"])}

## Fresh Search Space

{markdown_table(search_space, ["search_id", "phase", "dataset", "architecture", "checkpoint_steps", "target_head_gain_fraction", "max_settings", "compute_mode", "entrypoint", "entrypoint_status"])}

## Metric Contract

{markdown_table(metric_contract, ["metric_id", "claim_role", "worse_rule", "multiplicity_family", "claim_boundary"])}

## Stopping Rules

{markdown_table(stopping_rules, ["rule_id", "phase", "rule", "decision"])}

## Acceptance Gates

{markdown_table(gates, ["gate_id", "scope", "requirement", "pass_condition"])}

## Claim Ladder

{markdown_table(claim_ladder, ["claim_id", "current_status", "unlock_condition", "blocked_if"])}

## Protocol Status

{markdown_table(status, ["item", "status", "evidence", "blocks_stronger_claim_if_missing"])}

## Claim Boundary

Allowed now: {allowed_now}

Blocked now: {blocked_now}

Phase1 boundary: {phase1_boundary}

Phase2 boundary: {phase2_boundary}

Generated tables:

- [audit_baseline.csv](../results/e11_natural_negative_search_protocol/audit_baseline.csv)
- [search_space_registry.csv](../results/e11_natural_negative_search_protocol/search_space_registry.csv)
- [metric_contract.csv](../results/e11_natural_negative_search_protocol/metric_contract.csv)
- [stopping_rules.csv](../results/e11_natural_negative_search_protocol/stopping_rules.csv)
- [acceptance_gates.csv](../results/e11_natural_negative_search_protocol/acceptance_gates.csv)
- [claim_ladder.csv](../results/e11_natural_negative_search_protocol/claim_ladder.csv)
- [protocol_status.csv](../results/e11_natural_negative_search_protocol/protocol_status.csv)
"""
    write_markdown(DISCUSSION_PATH, text)


def main() -> None:
    audit_baseline = load_audit_baseline()
    search_space = build_search_space_registry()
    metric_contract = build_metric_contract()
    stopping_rules = build_stopping_rules()
    gates = build_acceptance_gates()
    claim_ladder = build_claim_ladder()
    status = build_protocol_status(audit_baseline, search_space)
    write_outputs(audit_baseline, search_space, metric_contract, stopping_rules, gates, claim_ladder, status)
    print(f"saved natural negative search protocol to {DISCUSSION_PATH} and {RESULT_DIR}")


if __name__ == "__main__":
    main()
