from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.reporting import markdown_table, write_markdown


OUTPUT_DIR = Path("results/e11_artifact_review_packet")
OUTPUT_PATH = Path("discussion/e11_artifact_review_packet.md")
SUBMISSION_REPRO_DIR = Path("results/e11_submission_repro_audit")
PYTHON_CMD = "/data/conda_envs/SpatialQuantization/bin/python"


def git_output(*args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unavailable"
    return result.stdout.strip()


def read_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def status_lookup(frame: pd.DataFrame, key_column: str, value_column: str) -> dict[str, str]:
    if frame.empty or key_column not in frame or value_column not in frame:
        return {}
    return frame.set_index(key_column)[value_column].fillna("").astype(str).to_dict()


def command_matrix() -> pd.DataFrame:
    rows = [
        {
            "command_id": "AR-C1",
            "command": f"make PYTHON={PYTHON_CMD} e11-artifact-review-packet",
            "purpose": "Regenerate the reviewer-facing artifact-review response packet.",
            "compute_mode": "CPU",
            "expected_state": "recreates discussion/e11_artifact_review_packet.md and results/e11_artifact_review_packet/*",
            "claim_boundary": "No empirical claim changes; this only maps existing evidence to reviewer commands.",
        },
        {
            "command_id": "AR-C2",
            "command": f"make PYTHON={PYTHON_CMD} e11-paper-assets",
            "purpose": "Regenerate current head-to-tail paper Markdown, TeX tables, manifest, and review packets.",
            "compute_mode": "CPU",
            "expected_state": "all paper-facing generated artifacts are refreshed from committed result CSVs",
            "claim_boundary": "Does not inspect pending final GPU outputs unless their result files already exist.",
        },
        {
            "command_id": "AR-C3",
            "command": f"make PYTHON={PYTHON_CMD} e11-paper-pdf",
            "purpose": "Rebuild paper/specgrad_activation_paper/main.pdf and two_page.pdf.",
            "compute_mode": "CPU",
            "expected_state": "Tectonic fallback can build the PDFs on this server",
            "claim_boundary": "Preferred pdflatex/bibtex/xelatex clean-checkout reproducibility remains a separate gate.",
        },
        {
            "command_id": "AR-C4",
            "command": f"make PYTHON={PYTHON_CMD} e11-check",
            "purpose": "Validate artifacts, run tests, and run git diff whitespace checks.",
            "compute_mode": "CPU",
            "expected_state": "validator passes, pytest passes, and git diff --check passes",
            "claim_boundary": "This validates the current evidence bundle; it does not authorize stronger claims.",
        },
        {
            "command_id": "AR-C5",
            "command": f"make PYTHON={PYTHON_CMD} e11-full",
            "purpose": "Strongest local reviewer command: regenerate assets, rebuild PDFs, then run e11-check.",
            "compute_mode": "CPU",
            "expected_state": "complete local artifact gate for the current committed evidence bundle",
            "claim_boundary": "Still CPU-side and still separate from the preferred LaTeX clean-checkout gate.",
        },
        {
            "command_id": "AR-G1",
            "command": f"make PYTHON={PYTHON_CMD} e11-cifar-resnet-condition-score-v5-architecture-results",
            "purpose": "Submit the unspent ResNeXt50-32x4d final architecture split.",
            "compute_mode": "GPU via Slurm",
            "expected_state": "pending or running until Slurm writes final architecture layer tables",
            "claim_boundary": "Not required to reproduce current paper claims; required only before a v5 P0 predictive-condition claim.",
        },
        {
            "command_id": "AR-G2",
            "command": f"make PYTHON={PYTHON_CMD} e11-cifar-resnet-condition-score-v5-data-results",
            "purpose": "Submit the unspent CIFAR-10 cross-partition final data split.",
            "compute_mode": "GPU via Slurm",
            "expected_state": "pending or running until Slurm writes final data layer tables",
            "claim_boundary": "Not required to reproduce current paper claims; required only before a v5 P0 predictive-condition claim.",
        },
        {
            "command_id": "AR-G3",
            "command": f"make PYTHON={PYTHON_CMD} e11-natural-negative-search-phase1-results",
            "purpose": "Submit the registered phase1 natural-negative search family.",
            "compute_mode": "GPU via Slurm",
            "expected_state": "current 26-setting phase1 metric files are complete; rerun only for a new registered family",
            "claim_boundary": "Finite registered phase1 null-candidate wording is allowed only with detectable-effect and tail-quality caveats.",
        },
    ]
    return pd.DataFrame(rows)


def gate_matrix(build_gates: pd.DataFrame) -> pd.DataFrame:
    if build_gates.empty:
        return pd.DataFrame(
            [
                {
                    "gate_id": "submission-repro-audit-missing",
                    "status": "fail",
                    "reviewer_status": "missing_required_input",
                    "evidence": "results/e11_submission_repro_audit/build_gate_summary.csv is absent",
                    "reviewer_action": "run make e11-submission-repro-audit before regenerating this packet",
                }
            ]
        )
    frame = build_gates.copy()
    reviewer_status = []
    reviewer_action = []
    for row in frame.to_dict("records"):
        gate_id = str(row["gate_id"])
        status = str(row["status"])
        if gate_id == "R3-preferred-latex-toolchain" and status == "not_ready":
            reviewer_status.append("external_toolchain_required")
            reviewer_action.append("Use Tectonic-backed server evidence now; run pdflatex/bibtex/xelatex in a clean checkout before venue-toolchain claims.")
        elif gate_id == "R2-working-tree-scope":
            reviewer_status.append("local_scope_declared")
            reviewer_action.append("Keep serverREADME.md untracked and stage only generated evidence files.")
        elif status == "pass":
            reviewer_status.append("reproducible_now")
            reviewer_action.append("Record the command output in the run summary.")
        else:
            reviewer_status.append("needs_explicit_followup")
            reviewer_action.append(str(row.get("required_next_action", "")))
    frame["reviewer_status"] = reviewer_status
    frame["reviewer_action"] = reviewer_action
    return frame


def local_state_contract(toolchain: pd.DataFrame, build_gates: pd.DataFrame) -> pd.DataFrame:
    tool_available = status_lookup(toolchain, "tool", "available")
    gate_status = status_lookup(build_gates, "gate_id", "status")
    server_readme_state = "present_untracked_local_file" if Path("serverREADME.md").exists() else "absent"
    final_arch_layer_summary = Path(
        "results/e11_condition_score_v5_protocol/final_architecture_resnext50_32x4d/layer_summary.csv"
    )
    final_data_layer_summary = Path(
        "results/e11_condition_score_v5_protocol/final_data_cifar10_cross/layer_summary.csv"
    )
    rows = [
        {
            "item": "serverREADME.md",
            "state": server_readme_state,
            "evidence": "Local attachment required by the user, intentionally outside the committed evidence bundle.",
            "reviewer_instruction": "Do not stage or commit this file; it is not needed for make e11-full.",
        },
        {
            "item": "Tectonic fallback",
            "state": "available" if tool_available.get("tectonic") == "yes" else "missing",
            "evidence": "R4-tectonic-fallback-toolchain status is " + gate_status.get("R4-tectonic-fallback-toolchain", "unknown"),
            "reviewer_instruction": "Use make e11-paper-pdf on this server to rebuild PDFs.",
        },
        {
            "item": "Preferred LaTeX toolchain",
            "state": gate_status.get("R3-preferred-latex-toolchain", "unknown"),
            "evidence": "pdflatex/bibtex/xelatex availability is tracked in toolchain_status.csv.",
            "reviewer_instruction": "Do not claim preferred clean-checkout reproducibility until this gate passes in a fresh checkout.",
        },
        {
            "item": "Rendered PDFs",
            "state": gate_status.get("R5-rendered-pdfs", "unknown"),
            "evidence": "main.pdf and two_page.pdf PDF headers and sizes are recorded in pdf_artifact_checks.csv.",
            "reviewer_instruction": "Rebuild with make e11-paper-pdf if the hashes need refreshing.",
        },
        {
            "item": "Full artifact validation",
            "state": gate_status.get("R6-full-artifact-validation", "unknown"),
            "evidence": "The strongest local gate is make e11-full; e11-check records validator, pytest, and whitespace status.",
            "reviewer_instruction": "Use make e11-full for artifact-review reproduction of the current bundle.",
        },
        {
            "item": "v5 final layer tables",
            "state": (
                "present"
                if final_arch_layer_summary.exists() and final_data_layer_summary.exists()
                else "pending_not_required_for_current_claims"
            ),
            "evidence": "The frozen final evaluator remains not_run until both unspent final split layer tables exist.",
            "reviewer_instruction": "Do not upgrade the v5 predictive-condition claim until both Slurm jobs finish and the frozen evaluator passes.",
        },
        {
            "item": "GPU dependence",
            "state": "not_required_for_current_artifact_review",
            "evidence": "Current Markdown, TeX, PDF, manifest, and validation artifacts are CPU-regenerated from committed result files.",
            "reviewer_instruction": "Use Slurm targets only for new evidence, not to reproduce the current submitted bundle.",
        },
    ]
    return pd.DataFrame(rows)


def reviewer_response() -> pd.DataFrame:
    rows = [
        {
            "objection": "Can the exact evidence bundle be reproduced?",
            "answer": f"Run make PYTHON={PYTHON_CMD} e11-full from the repo root; this regenerates paper assets, rebuilds PDFs, and runs validation/tests/whitespace checks.",
            "status": "supported_for_current_bundle",
            "forbidden_shortcut": "Do not cite a passing partial command as a stronger full-artifact gate.",
        },
        {
            "objection": "Why is the preferred LaTeX clean-checkout gate not passing?",
            "answer": "The server lacks pdflatex, bibtex, and xelatex; the packet therefore reports Tectonic-backed server evidence and keeps preferred clean-checkout reproducibility as not_ready.",
            "status": "transparent_boundary",
            "forbidden_shortcut": "Do not claim preferred venue-toolchain reproducibility from the Tectonic fallback alone.",
        },
        {
            "objection": "Does serverREADME.md contaminate the artifact package?",
            "answer": "No. serverREADME.md is a user-provided local attachment kept in the working directory but intentionally untracked and uncommitted.",
            "status": "local_file_excluded",
            "forbidden_shortcut": "Do not stage serverREADME.md with evidence artifacts.",
        },
        {
            "objection": "Do reviewers need GPUs to check the current paper?",
            "answer": "No. The current submitted evidence bundle is checked by CPU-side Make targets; GPUs via Slurm are only for pending new evidence.",
            "status": "cpu_review_path",
            "forbidden_shortcut": "Do not use pending GPU jobs to support current paper claims before their frozen evaluators pass.",
        },
        {
            "objection": "Why are v5 final and natural-negative claims still limited?",
            "answer": "The unspent v5 final splits are still pending. The natural-negative phase1 family is complete and supports only a finite registered null candidate with detectable-effect and tail-quality caveats.",
            "status": "claim_boundary_preserved",
            "forbidden_shortcut": "Do not use not_run, not_ready, partial-family, or finite phase1 outputs as broader positive evidence.",
        },
        {
            "objection": "What scientific claim is actually reproducible now?",
            "answer": "The current artifact-review path supports the local matched-head-gain head-to-tail drift mechanism evidence, theorem/proof contract, and conservative claim ledger.",
            "status": "current_claim_only",
            "forbidden_shortcut": "Do not infer broad optimizer-performance, accuracy, or general predictive-condition claims.",
        },
    ]
    return pd.DataFrame(rows)


def write_outputs(
    commands: pd.DataFrame,
    gates: pd.DataFrame,
    state: pd.DataFrame,
    responses: pd.DataFrame,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    commands.to_csv(OUTPUT_DIR / "command_matrix.csv", index=False)
    gates.to_csv(OUTPUT_DIR / "gate_matrix.csv", index=False)
    state.to_csv(OUTPUT_DIR / "local_state_contract.csv", index=False)
    responses.to_csv(OUTPUT_DIR / "reviewer_response.csv", index=False)

    config = {
        "purpose": "Artifact-review response packet for the current E11 submission bundle.",
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_branch": git_output("branch", "--show-current"),
        "python_command": PYTHON_CMD,
        "strongest_local_gate": f"make PYTHON={PYTHON_CMD} e11-full",
        "claim_boundary": "current evidence bundle only; finite phase1 natural-null candidate with caveats; no v5 final or optimizer-performance upgrade",
    }
    (OUTPUT_DIR / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    text = f"""# E11 Artifact Review Packet

This generated packet converts the submission reproducibility audit into an artifact-review response plan. It is intentionally conservative: it supports the current evidence bundle, records the CPU reviewer path, and keeps preferred LaTeX clean-checkout, v5 final, unqualified natural-null, and optimizer-performance claims outside the current artifact boundary.

## Reviewer Command Matrix

{markdown_table(commands, ["command_id", "command", "purpose", "compute_mode", "expected_state", "claim_boundary"])}

## Build Gate Matrix

{markdown_table(gates, ["gate_id", "status", "reviewer_status", "evidence", "reviewer_action"])}

## Local State Contract

{markdown_table(state, ["item", "state", "evidence", "reviewer_instruction"])}

## Reviewer Response Matrix

{markdown_table(responses, ["objection", "answer", "status", "forbidden_shortcut"])}

## Artifact Boundary

Allowed now: artifact reviewers can reproduce the current bundle with `make PYTHON={PYTHON_CMD} e11-full`, or inspect the narrower steps `make PYTHON={PYTHON_CMD} e11-paper-assets`, `make PYTHON={PYTHON_CMD} e11-paper-pdf`, and `make PYTHON={PYTHON_CMD} e11-check`.

Blocked now: preferred pdflatex/bibtex/xelatex clean-checkout reproducibility, any v5 predictive-condition upgrade, any unqualified natural-null or counterexample wording, and any broad optimizer-performance claim.

Machine-readable tables:
- [command_matrix.csv](../results/e11_artifact_review_packet/command_matrix.csv)
- [gate_matrix.csv](../results/e11_artifact_review_packet/gate_matrix.csv)
- [local_state_contract.csv](../results/e11_artifact_review_packet/local_state_contract.csv)
- [reviewer_response.csv](../results/e11_artifact_review_packet/reviewer_response.csv)
- [config.json](../results/e11_artifact_review_packet/config.json)
"""
    write_markdown(OUTPUT_PATH, text)


def main() -> None:
    toolchain = read_csv(SUBMISSION_REPRO_DIR / "toolchain_status.csv")
    build_gates = read_csv(SUBMISSION_REPRO_DIR / "build_gate_summary.csv")
    commands = command_matrix()
    gates = gate_matrix(build_gates)
    state = local_state_contract(toolchain, build_gates)
    responses = reviewer_response()
    write_outputs(commands, gates, state, responses)
    print(f"saved artifact review packet to {OUTPUT_PATH} and {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
