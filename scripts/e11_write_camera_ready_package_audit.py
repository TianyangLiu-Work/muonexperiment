from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.reporting import markdown_table, write_markdown


OUTPUT_DIR = Path("results/e11_camera_ready_package_audit")
DISCUSSION_PATH = Path("discussion/e11_camera_ready_package_audit.md")
SUBMISSION_DIR = Path("results/e11_submission_repro_audit")
ARTIFACT_REVIEW_DIR = Path("results/e11_artifact_review_packet")
PDF_RENDER_DIR = Path("results/e11_pdf_render_boundary_audit")
CLAIM_TRACE_DIR = Path("results/e11_manuscript_claim_trace")
CLAIM_DECISION_DIR = Path("results/e11_top_conference_claim_decision_audit")


def read_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def status_lookup(frame: pd.DataFrame, key_column: str, value_column: str) -> dict[str, str]:
    if frame.empty or key_column not in frame or value_column not in frame:
        return {}
    return frame.set_index(key_column)[value_column].fillna("").astype(str).to_dict()


def has_command(command_matrix: pd.DataFrame, fragment: str) -> bool:
    if command_matrix.empty or "command" not in command_matrix:
        return False
    return command_matrix["command"].astype(str).str.contains(fragment, regex=False).any()


def build_package_item_matrix(
    source_manifest: pd.DataFrame,
    pdf_checks: pd.DataFrame,
    submission_gates: pd.DataFrame,
    artifact_commands: pd.DataFrame,
    artifact_state: pd.DataFrame,
    pdf_render_gates: pd.DataFrame,
    claim_trace: pd.DataFrame,
    blocked_phrases: pd.DataFrame,
    claim_decisions: pd.DataFrame,
) -> pd.DataFrame:
    source_status = status_lookup(source_manifest, "path", "audit_status")
    pdf_status = status_lookup(pdf_checks, "path", "audit_status")
    submission_status = status_lookup(submission_gates, "gate_id", "status")
    local_state = status_lookup(artifact_state, "item", "state")
    render_status = status_lookup(pdf_render_gates, "gate_id", "status")
    claim_decision = status_lookup(claim_decisions, "claim_id", "current_decision")

    required_source_paths = {
        "paper/specgrad_activation_paper/main.tex",
        "paper/specgrad_activation_paper/two_page.tex",
        "paper/specgrad_activation_paper/references.bib",
        "paper/specgrad_activation_paper/Makefile",
        "paper/specgrad_activation_paper/tables/e11_paper_numbers.tex",
        "Makefile",
        "README_E11.md",
    }
    source_pass = required_source_paths.issubset(set(source_status)) and all(
        source_status[path] == "pass" for path in required_source_paths
    )
    pdf_pass = {
        "paper/specgrad_activation_paper/main.pdf",
        "paper/specgrad_activation_paper/two_page.pdf",
    }.issubset(set(pdf_status)) and all(status == "pass" for status in pdf_status.values())
    reviewer_path_pass = all(
        has_command(artifact_commands, command)
        for command in ["e11-full", "e11-paper-pdf", "e11-check", "e11-artifact-review-packet"]
    )
    claim_trace_pass = (
        not claim_trace.empty
        and claim_trace["trace_status"].astype(str).str.len().gt(0).all()
        and claim_trace["missing_anchors"].astype(str).eq("none").all()
        and not blocked_phrases.empty
        and blocked_phrases["audit_status"].astype(str).eq("pass").all()
    )
    final_claim_quarantine = (
        claim_decision.get("TCD-3-predictive-condition-generalization")
        == "blocked_completed_final_failed_boundary"
        and claim_decision.get("TCD-5-optimizer-performance-benchmark") == "blocked_protocol_pending"
    )
    text_metadata_statuses = {
        render_status.get("PRB-4-pdf-text-extraction-tool", "missing"),
        render_status.get("PRB-5-pdf-metadata-tool", "missing"),
    }
    text_metadata_ready = text_metadata_statuses == {"pass"}

    return pd.DataFrame(
        [
            {
                "item_id": "CRP-1-source-bundle",
                "status": "pass" if source_pass else "fail",
                "evidence": "required paper source, root Makefile, and README_E11.md are present in source_package_manifest.csv",
                "camera_ready_effect": "server package source bundle is present",
                "blocked_claim": "do not claim a complete source package if any required source path is absent",
            },
            {
                "item_id": "CRP-2-rendered-pdf-binaries",
                "status": "pass" if pdf_pass else "fail",
                "evidence": "main.pdf and two_page.pdf pass header, size, and hash checks in pdf_artifact_checks.csv",
                "camera_ready_effect": "server package includes rendered PDFs",
                "blocked_claim": "do not submit stale or non-PDF rendered artifacts",
            },
            {
                "item_id": "CRP-3-claim-trace-clean",
                "status": "pass" if claim_trace_pass else "fail",
                "evidence": "claim_trace.csv anchors are present and blocked_phrase_audit.csv has no hits",
                "camera_ready_effect": "main manuscript wording matches the claim-decision contract",
                "blocked_claim": "do not ship if blocked benchmark, predictive-score, or natural-counterexample wording appears",
            },
            {
                "item_id": "CRP-4-reviewer-command-path",
                "status": "pass" if reviewer_path_pass else "fail",
                "evidence": "artifact review command matrix includes e11-full, e11-paper-pdf, e11-check, and packet regeneration",
                "camera_ready_effect": "reviewers have a CPU reproduction path for the current bundle",
                "blocked_claim": "do not replace the strongest local gate with a narrower passing command",
            },
            {
                "item_id": "CRP-5-local-attachment-excluded",
                "status": "pass"
                if local_state.get("serverREADME.md") == "present_untracked_local_file"
                else "fail",
                "evidence": "artifact review local-state contract keeps serverREADME.md outside committed evidence",
                "camera_ready_effect": "local user attachment is excluded from the submission package",
                "blocked_claim": "do not stage or package serverREADME.md as reproducibility evidence",
            },
            {
                "item_id": "CRP-6-preferred-latex-boundary",
                "status": submission_status.get("R3-preferred-latex-toolchain", "missing"),
                "evidence": "mirrors R3-preferred-latex-toolchain from submission reproducibility audit",
                "camera_ready_effect": "preferred venue-toolchain claim remains explicit",
                "blocked_claim": "do not claim pdflatex/bibtex/xelatex clean-checkout reproducibility from Tectonic evidence",
            },
            {
                "item_id": "CRP-7-rendered-text-metadata-boundary",
                "status": "pass" if text_metadata_ready else "not_ready",
                "evidence": (
                    "PRB-4-pdf-text-extraction-tool="
                    + render_status.get("PRB-4-pdf-text-extraction-tool", "missing")
                    + "; PRB-5-pdf-metadata-tool="
                    + render_status.get("PRB-5-pdf-metadata-tool", "missing")
                ),
                "camera_ready_effect": "rendered text-layer and page metadata gate is explicit",
                "blocked_claim": "do not claim rendered-PDF text-layer or metadata verification until PRB-4/PRB-5 pass",
            },
            {
                "item_id": "CRP-8-final-claim-quarantine",
                "status": "pass" if final_claim_quarantine else "fail",
                "evidence": "TCD-3 is a completed failed boundary and TCD-5 is protocol-pending",
                "camera_ready_effect": "camera-ready package cannot imply predictive-score or optimizer-performance upgrades",
                "blocked_claim": "do not add benchmark-performance or successful predictive-condition wording during packaging",
            },
        ]
    )


def build_submission_gate_matrix(items: pd.DataFrame) -> pd.DataFrame:
    item_status = status_lookup(items, "item_id", "status")
    hard_gate_pass = all(
        item_status.get(item_id) == "pass"
        for item_id in [
            "CRP-1-source-bundle",
            "CRP-2-rendered-pdf-binaries",
            "CRP-3-claim-trace-clean",
            "CRP-4-reviewer-command-path",
            "CRP-5-local-attachment-excluded",
            "CRP-8-final-claim-quarantine",
        ]
    )
    venue_ready = (
        item_status.get("CRP-6-preferred-latex-boundary") == "pass"
        and item_status.get("CRP-7-rendered-text-metadata-boundary") == "pass"
    )
    return pd.DataFrame(
        [
            {
                "gate_id": "CRG-1-current-server-package",
                "status": "pass" if hard_gate_pass else "fail",
                "evidence": "source, rendered PDFs, claim trace, reviewer commands, local-file exclusion, and final-claim quarantine are checked",
                "required_next_action": "run make e11-full and record the pushed commit before sharing the current server package",
            },
            {
                "gate_id": "CRG-2-venue-toolchain-package",
                "status": "pass" if venue_ready else "not_ready",
                "evidence": "requires CRP-6 preferred LaTeX and CRP-7 rendered text/metadata gates to pass",
                "required_next_action": "rerun in a clean venue-style environment with pdflatex, bibtex, xelatex, and PDF text/metadata tools",
            },
            {
                "gate_id": "CRG-3-claim-boundary-package",
                "status": "pass"
                if item_status.get("CRP-3-claim-trace-clean") == "pass"
                and item_status.get("CRP-8-final-claim-quarantine") == "pass"
                else "fail",
                "evidence": "main manuscript claim anchors and blocked phrase audit agree with top-conference claim decisions",
                "required_next_action": "regenerate claim-decision and manuscript-trace audits after any manuscript edit",
            },
            {
                "gate_id": "CRG-4-local-file-exclusion",
                "status": item_status.get("CRP-5-local-attachment-excluded", "fail"),
                "evidence": "serverREADME.md is local and untracked",
                "required_next_action": "keep serverREADME.md out of staged submission artifacts",
            },
        ]
    )


def build_checklist(gates: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "step_id": "CRC-1-current-server-share",
                "command_or_action": "make PYTHON=/data/conda_envs/SpatialQuantization/bin/python e11-full",
                "required_gate": "CRG-1-current-server-package=pass",
                "claim_boundary": "server package only; preferred venue-toolchain remains separate if CRG-2 is not_ready",
            },
            {
                "step_id": "CRC-2-venue-clean-checkout",
                "command_or_action": "run e11-full in a clean checkout with pdflatex, bibtex, xelatex, pdftotext/pdfinfo or equivalent",
                "required_gate": "CRG-2-venue-toolchain-package=pass",
                "claim_boundary": "needed before claiming full venue-toolchain reproducibility",
            },
            {
                "step_id": "CRC-3-after-manuscript-edit",
                "command_or_action": "regenerate e11-top-conference-claim-decision-audit and e11-manuscript-claim-trace",
                "required_gate": "CRG-3-claim-boundary-package=pass",
                "claim_boundary": "no new benchmark, predictive-score, or natural-counterexample wording without matching gates",
            },
            {
                "step_id": "CRC-4-before-commit",
                "command_or_action": "git status --short --branch",
                "required_gate": "CRG-4-local-file-exclusion=pass",
                "claim_boundary": "serverREADME.md remains local and untracked",
            },
        ]
    )


def write_outputs(items: pd.DataFrame, gates: pd.DataFrame, checklist: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    items.to_csv(OUTPUT_DIR / "package_item_matrix.csv", index=False)
    gates.to_csv(OUTPUT_DIR / "submission_gate_matrix.csv", index=False)
    checklist.to_csv(OUTPUT_DIR / "camera_ready_checklist.csv", index=False)
    config = {
        "purpose": "camera-ready package boundary audit for the current E11 evidence bundle",
        "new_empirical_results": False,
        "current_server_gate": "CRG-1-current-server-package",
        "venue_toolchain_gate": "CRG-2-venue-toolchain-package",
    }
    (OUTPUT_DIR / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    text = f"""# E11 Camera-Ready Package Audit

This generated audit is a package-readiness boundary, not a new empirical result.
It ties the rendered PDFs, source package, claim trace, artifact-review commands,
local attachment exclusion, and venue-toolchain caveats into one camera-ready
checklist.

## Package Item Matrix

{markdown_table(items, ["item_id", "status", "evidence", "camera_ready_effect", "blocked_claim"])}

## Submission Gate Matrix

{markdown_table(gates, ["gate_id", "status", "evidence", "required_next_action"])}

## Camera-Ready Checklist

{markdown_table(checklist, ["step_id", "command_or_action", "required_gate", "claim_boundary"])}

## Boundary

Allowed now: share the current server evidence package only when
`CRG-1-current-server-package` passes and the final run summary records
`make e11-full`.

Blocked now: claiming full venue-toolchain clean-checkout reproducibility,
rendered-PDF text-layer or metadata verification, or stronger benchmark and
predictive-condition wording unless the corresponding gates pass.

Artifacts:
- [package_item_matrix.csv](../results/e11_camera_ready_package_audit/package_item_matrix.csv)
- [submission_gate_matrix.csv](../results/e11_camera_ready_package_audit/submission_gate_matrix.csv)
- [camera_ready_checklist.csv](../results/e11_camera_ready_package_audit/camera_ready_checklist.csv)
- [config.json](../results/e11_camera_ready_package_audit/config.json)
"""
    write_markdown(DISCUSSION_PATH, text)


def main() -> None:
    items = build_package_item_matrix(
        read_csv(SUBMISSION_DIR / "source_package_manifest.csv"),
        read_csv(SUBMISSION_DIR / "pdf_artifact_checks.csv"),
        read_csv(SUBMISSION_DIR / "build_gate_summary.csv"),
        read_csv(ARTIFACT_REVIEW_DIR / "command_matrix.csv"),
        read_csv(ARTIFACT_REVIEW_DIR / "local_state_contract.csv"),
        read_csv(PDF_RENDER_DIR / "render_boundary_gates.csv"),
        read_csv(CLAIM_TRACE_DIR / "claim_trace.csv"),
        read_csv(CLAIM_TRACE_DIR / "blocked_phrase_audit.csv"),
        read_csv(CLAIM_DECISION_DIR / "claim_decision_matrix.csv"),
    )
    gates = build_submission_gate_matrix(items)
    checklist = build_checklist(gates)
    write_outputs(items, gates, checklist)
    print(f"saved camera-ready package audit to {OUTPUT_DIR} and {DISCUSSION_PATH}")
    print(gates.to_string(index=False))


if __name__ == "__main__":
    main()
