from __future__ import annotations

import hashlib
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
REFRESH_DIR = RESULT_ROOT / "validation_refresh_firewall"
OUTPUT_DIR = RESULT_ROOT / "validation_protocol_seal"
DISCUSSION_PATH = Path("discussion/e11_cifar100_resnet_lt_tuned_benchmark_protocol_seal.md")

FROZEN_ARTIFACTS: tuple[dict[str, str], ...] = (
    {
        "path": "results/e11_cifar100_resnet_lt_tuned_benchmark_protocol/benchmark_scope.csv",
        "artifact_class": "locked_protocol_contract",
        "role": "benchmark scope boundary",
        "post_exposure_change_policy": "fresh_preregistered_protocol_required",
    },
    {
        "path": "results/e11_cifar100_resnet_lt_tuned_benchmark_protocol/pilot_context.csv",
        "artifact_class": "locked_protocol_contract",
        "role": "spent pilot quarantine",
        "post_exposure_change_policy": "fresh_preregistered_protocol_required",
    },
    {
        "path": "results/e11_cifar100_resnet_lt_tuned_benchmark_protocol/recipe_grid.csv",
        "artifact_class": "locked_protocol_contract",
        "role": "candidate recipe family grid",
        "post_exposure_change_policy": "fresh_preregistered_protocol_required",
    },
    {
        "path": "results/e11_cifar100_resnet_lt_tuned_benchmark_protocol/seed_split_contract.csv",
        "artifact_class": "locked_protocol_contract",
        "role": "validation/final seed split",
        "post_exposure_change_policy": "fresh_preregistered_protocol_required",
    },
    {
        "path": "results/e11_cifar100_resnet_lt_tuned_benchmark_protocol/selection_rules.csv",
        "artifact_class": "locked_protocol_contract",
        "role": "validation objective and final multiplicity rule",
        "post_exposure_change_policy": "fresh_preregistered_protocol_required",
    },
    {
        "path": "results/e11_cifar100_resnet_lt_tuned_benchmark_protocol/acceptance_gates.csv",
        "artifact_class": "locked_protocol_contract",
        "role": "benchmark acceptance gates",
        "post_exposure_change_policy": "fresh_preregistered_protocol_required",
    },
    {
        "path": "results/e11_cifar100_resnet_lt_tuned_benchmark/settings_registry.csv",
        "artifact_class": "locked_validation_registry",
        "role": "164-setting validation array registry",
        "post_exposure_change_policy": "fresh_preregistered_protocol_required",
    },
    {
        "path": "scripts/e11_run_cifar100_resnet_lt_tuned_benchmark.py",
        "artifact_class": "locked_runner",
        "role": "validation/final runner and settings materializer",
        "post_exposure_change_policy": "fresh_preregistered_protocol_required_for_performance_claims",
    },
    {
        "path": "scripts/slurm/e11_cifar100_resnet_lt_tuned_benchmark_validation.sbatch",
        "artifact_class": "locked_runner",
        "role": "validation Slurm wrapper",
        "post_exposure_change_policy": "fresh_preregistered_protocol_required_for_performance_claims",
    },
    {
        "path": "scripts/slurm/e11_cifar100_resnet_lt_tuned_benchmark_final.sbatch",
        "artifact_class": "locked_runner",
        "role": "final-claim Slurm wrapper",
        "post_exposure_change_policy": "fresh_preregistered_protocol_required_for_performance_claims",
    },
    {
        "path": "scripts/e11_write_cifar100_resnet_lt_tuned_benchmark_selection.py",
        "artifact_class": "locked_selection_audit",
        "role": "validation-only family selection",
        "post_exposure_change_policy": "fresh_preregistered_protocol_required_for_performance_claims",
    },
    {
        "path": "scripts/e11_write_cifar100_resnet_lt_tuned_benchmark_interim_audit.py",
        "artifact_class": "progress_audit",
        "role": "partial validation readout without selection authority",
        "post_exposure_change_policy": "allowed_if_claim_authority_stays_progress_accounting_only",
    },
    {
        "path": "scripts/e11_write_cifar100_resnet_lt_tuned_benchmark_leakage_audit.py",
        "artifact_class": "progress_audit",
        "role": "selection leakage and optional-stopping audit",
        "post_exposure_change_policy": "allowed_if_claim_authority_stays_progress_accounting_only",
    },
    {
        "path": "scripts/e11_write_cifar100_resnet_lt_tuned_benchmark_refresh_firewall.py",
        "artifact_class": "progress_audit",
        "role": "sequential refresh firewall",
        "post_exposure_change_policy": "allowed_if_claim_authority_stays_progress_accounting_only",
    },
    {
        "path": "scripts/e11_write_cifar100_resnet_lt_tuned_benchmark_final_analysis_plan.py",
        "artifact_class": "locked_final_gate",
        "role": "final analysis plan",
        "post_exposure_change_policy": "fresh_preregistered_protocol_required_for_performance_claims",
    },
    {
        "path": "scripts/e11_write_cifar100_resnet_lt_tuned_benchmark_final_execution_plan.py",
        "artifact_class": "locked_final_gate",
        "role": "final execution plan",
        "post_exposure_change_policy": "fresh_preregistered_protocol_required_for_performance_claims",
    },
    {
        "path": "scripts/e11_evaluate_cifar100_resnet_lt_tuned_benchmark_final.py",
        "artifact_class": "locked_final_gate",
        "role": "final evaluator",
        "post_exposure_change_policy": "fresh_preregistered_protocol_required_for_performance_claims",
    },
    {
        "path": "scripts/e11_submit_cifar100_resnet_lt_tuned_benchmark_final.py",
        "artifact_class": "locked_final_gate",
        "role": "final launch gate",
        "post_exposure_change_policy": "fresh_preregistered_protocol_required_for_performance_claims",
    },
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def final_outputs() -> list[str]:
    final_claim_dir = RESULT_ROOT / "final_claim"
    if not final_claim_dir.exists():
        return []
    return sorted(path.as_posix() for path in final_claim_dir.rglob("*") if path.is_file())


def build_hash_manifest() -> pd.DataFrame:
    rows: list[dict[str, str | int]] = []
    for item in FROZEN_ARTIFACTS:
        path = Path(item["path"])
        exists = path.exists()
        rows.append(
            {
                **item,
                "exists": "yes" if exists else "no",
                "size_bytes": path.stat().st_size if exists else 0,
                "sha256": sha256_file(path) if exists else "",
            }
        )
    return pd.DataFrame(rows)


def build_exposure_boundary(run_registry: pd.DataFrame, refresh_state: pd.DataFrame) -> pd.DataFrame:
    complete = run_registry[run_registry["validation_status"].eq("complete")].copy()
    completed_indices = sorted(complete["array_index"].astype(int).tolist())
    prefix_is_contiguous = completed_indices == list(range(len(completed_indices)))
    missing_indices = sorted(set(run_registry["array_index"].astype(int)).difference(completed_indices))
    final_files = final_outputs()
    refresh = refresh_state.iloc[0]
    completed_span = f"0..{completed_indices[-1]}" if completed_indices else "none"
    return pd.DataFrame(
        [
            {
                "boundary_id": "TPS-current-post-exposure-boundary",
                "status": "post_partial_validation_exposure_sealed",
                "completed_validation_settings": int(len(complete)),
                "total_validation_settings": int(len(run_registry)),
                "completed_occupancy_traces": int(run_registry["occupancy_status"].eq("complete").sum()),
                "completed_array_prefix": completed_span,
                "next_missing_array_index": str(missing_indices[0]) if missing_indices else "none",
                "prefix_contiguous": "yes" if prefix_is_contiguous else "no",
                "refresh_status": str(refresh["status"]),
                "final_output_files": int(len(final_files)),
                "claim_authority": "progress_accounting_only_until_full_validation_and_final_gates_pass",
            }
        ]
    )


def build_immutability_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "immutability_id": "TPS-IMM-1-protocol-csvs",
                "sealed_surface": "benchmark_scope, pilot_context, recipe_grid, seed_split_contract, selection_rules, acceptance_gates",
                "allowed_change_after_partial_exposure": "no",
                "required_response_if_changed": "discard current validation/final split for performance claims and open a fresh preregistered protocol",
                "claim_authority_if_changed": "no benchmark-performance claim",
            },
            {
                "immutability_id": "TPS-IMM-2-validation-registry",
                "sealed_surface": "settings_registry.csv array order, recipe families, hyperparameter grid, validation seed set 10..14",
                "allowed_change_after_partial_exposure": "no",
                "required_response_if_changed": "rerun only as a fresh validation protocol with fresh final seeds",
                "claim_authority_if_changed": "progress accounting only for old grid",
            },
            {
                "immutability_id": "TPS-IMM-3-final-seed-contract",
                "sealed_surface": "final_claim seed set 20..29 and tuning_allowed=no",
                "allowed_change_after_partial_exposure": "no",
                "required_response_if_changed": "treat any touched final rows as contaminated and allocate fresh final seeds",
                "claim_authority_if_changed": "no final-performance claim",
            },
            {
                "immutability_id": "TPS-IMM-4-selection-and-final-gate-code",
                "sealed_surface": "selection, final analysis, final execution, final evaluator, and final launch scripts",
                "allowed_change_after_partial_exposure": "only bug fixes that do not change objective, grid, seed split, metric family, or claim ladder",
                "required_response_if_changed": "document the diff, rerun the protocol seal, and downgrade to no benchmark claim if semantics changed",
                "claim_authority_if_changed": "blocked until semantic audit passes",
            },
            {
                "immutability_id": "TPS-IMM-5-progress-audit-code",
                "sealed_surface": "interim, leakage, refresh firewall, and protocol seal audits",
                "allowed_change_after_partial_exposure": "yes for stricter progress accounting",
                "required_response_if_changed": "must preserve progress-only authority and forbid stronger wording from partial validation rows",
                "claim_authority_if_changed": "progress accounting only",
            },
        ]
    )


def build_seal_gate_matrix(
    hash_manifest: pd.DataFrame,
    exposure_boundary: pd.DataFrame,
    settings_registry: pd.DataFrame,
    selection_rules: pd.DataFrame,
    seed_split: pd.DataFrame,
    gate_report: pd.DataFrame,
) -> pd.DataFrame:
    final_claim = seed_split[seed_split["split_id"].eq("final_claim")]
    selection_text = " ".join(selection_rules.astype(str).to_numpy().ravel())
    gate_lookup = gate_report.set_index("gate_id")["status"].astype(str).to_dict()
    final_files = final_outputs()
    rows = [
        {
            "gate_id": "TPS-1-hash-manifest-complete",
            "status": "pass" if hash_manifest["exists"].eq("yes").all() else "fail",
            "evidence": f"{int(hash_manifest['exists'].eq('yes').sum())}/{len(hash_manifest)} sealed artifacts hashed",
            "blocked_failure_mode": "untracked protocol or runner surface",
        },
        {
            "gate_id": "TPS-2-validation-registry-sealed",
            "status": "pass"
            if len(settings_registry) == 164
            and sorted(settings_registry["array_index"].astype(int).tolist()) == list(range(164))
            and set(settings_registry["seed_set"].astype(str)) == {"10..14"}
            else "fail",
            "evidence": f"{len(settings_registry)} registry rows; seed sets={','.join(sorted(settings_registry['seed_set'].astype(str).unique()))}",
            "blocked_failure_mode": "post-exposure grid or array-order drift",
        },
        {
            "gate_id": "TPS-3-selection-rule-sealed",
            "status": "pass" if "Primary validation objective is few balanced accuracy" in selection_text else "fail",
            "evidence": "primary objective anchor present",
            "blocked_failure_mode": "metric-dependent selection objective rewrite",
        },
        {
            "gate_id": "TPS-4-final-seed-quarantine-sealed",
            "status": "pass"
            if len(final_claim) == 1
            and str(final_claim.iloc[0]["seed_set"]) == "20..29"
            and str(final_claim.iloc[0]["tuning_allowed"]) == "no"
            and not final_files
            and gate_lookup.get("TVS-3-final-seed-quarantine") == "pass"
            else "fail",
            "evidence": "final_claim seed_set 20..29 tuning_allowed=no; no final_claim outputs exist",
            "blocked_failure_mode": "final-seed unblinding before TVS/FEP/TFE/FLA gates",
        },
        {
            "gate_id": "TPS-5-post-exposure-claim-authority",
            "status": "pass"
            if str(exposure_boundary.iloc[0]["claim_authority"])
            == "progress_accounting_only_until_full_validation_and_final_gates_pass"
            else "fail",
            "evidence": (
                f"{int(exposure_boundary.iloc[0]['completed_validation_settings'])}/"
                f"{int(exposure_boundary.iloc[0]['total_validation_settings'])} validation settings exposed"
            ),
            "blocked_failure_mode": "benchmark wording from partial validation exposure",
        },
    ]
    return pd.DataFrame(rows)


def write_discussion(
    hash_manifest: pd.DataFrame,
    exposure_boundary: pd.DataFrame,
    immutability_matrix: pd.DataFrame,
    seal_gates: pd.DataFrame,
) -> None:
    text = f"""# E11 CIFAR-100-LT Tuned Benchmark Protocol Seal

This generated protocol seal records the frozen surfaces of the tuned benchmark after partial validation exposure. It is a post-exposure protocol hash seal, not a benchmark result: the visible validation rows may update progress accounting, but they cannot justify changing the registry, selection objective, final seed set, final-gate code, or performance wording inside the same protocol.

## Exposure Boundary

{markdown_table(exposure_boundary, ["boundary_id", "status", "completed_validation_settings", "total_validation_settings", "completed_occupancy_traces", "completed_array_prefix", "next_missing_array_index", "prefix_contiguous", "refresh_status", "final_output_files", "claim_authority"])}

## Seal Gate Matrix

{markdown_table(seal_gates, ["gate_id", "status", "evidence", "blocked_failure_mode"])}

## Immutability Matrix

{markdown_table(immutability_matrix, ["immutability_id", "sealed_surface", "allowed_change_after_partial_exposure", "required_response_if_changed", "claim_authority_if_changed"])}

## Hash Manifest

{markdown_table(hash_manifest, ["path", "artifact_class", "role", "exists", "size_bytes", "sha256", "post_exposure_change_policy"])}

## Operating Rule

If a locked protocol contract, validation registry, final seed contract, selection rule, or final-gate semantic changes after this seal while validation rows are visible, the current split can no longer support a benchmark-performance claim. The only reviewer-safe responses are progress accounting under the sealed protocol or a fresh preregistered protocol with unspent validation and final seeds.
"""
    write_markdown(DISCUSSION_PATH, text)


def main() -> None:
    settings_registry = pd.read_csv(RESULT_ROOT / "settings_registry.csv")
    run_registry = pd.read_csv(SELECTION_DIR / "run_registry.csv")
    gate_report = pd.read_csv(SELECTION_DIR / "gate_report.csv")
    refresh_state = pd.read_csv(REFRESH_DIR / "refresh_state.csv")
    selection_rules = pd.read_csv(PROTOCOL_DIR / "selection_rules.csv")
    seed_split = pd.read_csv(PROTOCOL_DIR / "seed_split_contract.csv")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    hash_manifest = build_hash_manifest()
    exposure_boundary = build_exposure_boundary(run_registry, refresh_state)
    immutability_matrix = build_immutability_matrix()
    seal_gates = build_seal_gate_matrix(
        hash_manifest,
        exposure_boundary,
        settings_registry,
        selection_rules,
        seed_split,
        gate_report,
    )

    hash_manifest.to_csv(OUTPUT_DIR / "hash_manifest.csv", index=False)
    exposure_boundary.to_csv(OUTPUT_DIR / "exposure_boundary.csv", index=False)
    immutability_matrix.to_csv(OUTPUT_DIR / "immutability_matrix.csv", index=False)
    seal_gates.to_csv(OUTPUT_DIR / "seal_gate_matrix.csv", index=False)
    config = {
        "scope": "post_exposure_tuned_protocol_hash_seal",
        "claim_authority": "progress_accounting_only_until_full_validation_and_final_gates_pass",
        "settings_registry": (RESULT_ROOT / "settings_registry.csv").as_posix(),
        "run_registry": (SELECTION_DIR / "run_registry.csv").as_posix(),
        "refresh_state": (REFRESH_DIR / "refresh_state.csv").as_posix(),
        "sealed_artifact_count": int(len(hash_manifest)),
        "completed_validation_settings": int(exposure_boundary.iloc[0]["completed_validation_settings"]),
    }
    (OUTPUT_DIR / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    write_discussion(hash_manifest, exposure_boundary, immutability_matrix, seal_gates)
    print(f"saved tuned benchmark protocol seal to {OUTPUT_DIR} and {DISCUSSION_PATH}")


if __name__ == "__main__":
    main()
