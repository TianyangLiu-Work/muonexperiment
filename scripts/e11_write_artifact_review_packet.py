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
PDF_RENDER_BOUNDARY_DIR = Path("results/e11_pdf_render_boundary_audit")
FINAL_BENCHMARK_DIR = Path("results/e11_cifar100_resnet_lt_tuned_benchmark")
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
            "claim_boundary": "Regenerates from existing result files and does not launch GPU reruns.",
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
            "command_id": "AR-C6",
            "command": f"make PYTHON={PYTHON_CMD} e11-cifar-resnet-lt-tuned-benchmark-leakage-audit",
            "purpose": "Regenerate the tuned-validation leakage and optional-stopping guard matrix.",
            "compute_mode": "CPU",
            "expected_state": "refreshes validation_leakage_audit tables and reviewer-facing leakage discussion from current validation/queue artifacts",
            "claim_boundary": "Only audits partial validation visibility; it does not authorize recipe selection or final-performance wording.",
        },
        {
            "command_id": "AR-C7",
            "command": f"make PYTHON={PYTHON_CMD} e11-clean-worktree-replay-audit",
            "purpose": "Replay e11-check from a detached clean Git worktree.",
            "compute_mode": "CPU",
            "expected_state": "records run_summary.csv, gate_matrix.csv, and command_log_tail.txt for the tracked-source replay",
            "claim_boundary": "Checks tracked-source artifact replay only; preferred LaTeX clean-checkout reproducibility remains separate.",
        },
        {
            "command_id": "AR-C8",
            "command": f"make PYTHON={PYTHON_CMD} e11-pdf-render-boundary-audit",
            "purpose": "Audit rendered PDF header/hash/source-trace coverage and PDF text/metadata inspection tool availability.",
            "compute_mode": "CPU",
            "expected_state": "records PDF inspection tool status plus render-boundary gates without changing empirical claims",
            "claim_boundary": "Supports binary/hash/source-claim trace evidence now; rendered text-layer and metadata verification remain tool-gated.",
        },
        {
            "command_id": "AR-C9",
            "command": f"make PYTHON={PYTHON_CMD} e11-cifar-resnet-lt-tuned-benchmark-final-execution-plan",
            "purpose": "Regenerate the no-side-effect final-claim execution contract and gate-checked Slurm wrapper plan.",
            "compute_mode": "CPU",
            "expected_state": "refreshes final_execution_plan tables and discussion without submitting final jobs",
            "claim_boundary": "Does not submit jobs or inspect final outputs; final submit remains blocked until every FEP gate passes.",
        },
        {
            "command_id": "AR-C10",
            "command": f"make PYTHON={PYTHON_CMD} e11-cifar-resnet-lt-tuned-benchmark-final-eval",
            "purpose": "Regenerate the fixed final evaluator and final-claim gate report.",
            "compute_mode": "CPU",
            "expected_state": "currently reports not_ready until selected final recipe outputs are complete",
            "claim_boundary": "Does not authorize benchmark wording when final rows are absent or final claim gates are not_ready.",
        },
        {
            "command_id": "AR-C11",
            "command": f"make PYTHON={PYTHON_CMD} e11-cifar-resnet-lt-tuned-benchmark-final-launch-audit",
            "purpose": "Compute queue-aware final-claim launch readiness without submitting.",
            "compute_mode": "CPU",
            "expected_state": "currently records blocked_final_launch_gates_not_ready and emits no submit command",
            "claim_boundary": "Dry-run audit only; use e11-cifar-resnet-lt-tuned-benchmark-final-safe-submit only after all final launch gates pass.",
        },
        {
            "command_id": "AR-G1",
            "command": f"make PYTHON={PYTHON_CMD} e11-cifar-resnet-condition-score-v5-architecture-results",
            "purpose": "Regenerate the registered ResNeXt50-32x4d final architecture split if a new run is explicitly needed.",
            "compute_mode": "GPU via Slurm",
            "expected_state": "current final architecture layer tables are already present in the committed evidence bundle",
            "claim_boundary": "Not required to reproduce current paper claims; the completed v5 final already failed its registered P0 gates.",
        },
        {
            "command_id": "AR-G2",
            "command": f"make PYTHON={PYTHON_CMD} e11-cifar-resnet-condition-score-v5-data-results",
            "purpose": "Regenerate the registered CIFAR-10 cross-partition final data split if a new run is explicitly needed.",
            "compute_mode": "GPU via Slurm",
            "expected_state": "current final data layer tables are already present in the committed evidence bundle",
            "claim_boundary": "Not required to reproduce current paper claims; the completed v5 final already failed its registered P0 gates.",
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
    leakage_guards = read_csv(
        Path("results/e11_cifar100_resnet_lt_tuned_benchmark/validation_leakage_audit/leakage_guard_matrix.csv")
    )
    clean_replay = read_csv(Path("results/e11_clean_worktree_replay_audit/run_summary.csv"))
    pdf_render_gates = read_csv(PDF_RENDER_BOUNDARY_DIR / "render_boundary_gates.csv")
    final_execution_gates = read_csv(FINAL_BENCHMARK_DIR / "final_execution_plan/gate_matrix.csv")
    final_eval_gates = read_csv(FINAL_BENCHMARK_DIR / "final_evaluation/claim_gate_report.csv")
    final_launch_gates = read_csv(FINAL_BENCHMARK_DIR / "final_launch_audit/latest_gate_snapshot.csv")
    final_launch_decision = read_csv(FINAL_BENCHMARK_DIR / "final_launch_audit/latest_final_launch_decision.csv")
    leakage_state = "missing"
    leakage_evidence = "validation_leakage_audit/leakage_guard_matrix.csv is absent"
    if not leakage_guards.empty and {"guard_id", "status"}.issubset(leakage_guards.columns):
        status_by_guard = status_lookup(leakage_guards, "guard_id", "status")
        leakage_state = "pass" if set(status_by_guard.values()) == {"pass"} else "check_required"
        leakage_evidence = "; ".join(f"{guard}={status}" for guard, status in status_by_guard.items())
    clean_replay_state = "missing"
    clean_replay_evidence = "results/e11_clean_worktree_replay_audit/run_summary.csv is absent"
    if not clean_replay.empty and {"status", "command", "server_readme_present_in_worktree"}.issubset(
        clean_replay.columns
    ):
        clean_replay_state = str(clean_replay["status"].iloc[0])
        clean_replay_evidence = (
            f"{clean_replay['command'].iloc[0]}; "
            f"serverREADME.md in clean worktree={clean_replay['server_readme_present_in_worktree'].iloc[0]}"
        )
    pdf_render_state = "missing"
    pdf_render_evidence = "results/e11_pdf_render_boundary_audit/render_boundary_gates.csv is absent"
    if not pdf_render_gates.empty and {"gate_id", "status"}.issubset(pdf_render_gates.columns):
        pdf_gate_status = status_lookup(pdf_render_gates, "gate_id", "status")
        pdf_render_state = ",".join(f"{gate}={status}" for gate, status in pdf_gate_status.items())
        pdf_render_evidence = (
            "Rendered PDF binary/hash/source-trace gates pass; text-layer and metadata gates are explicit "
            "pass/not_ready states: " + pdf_render_state
        )
    final_execution_state = "missing"
    final_execution_evidence = "results/e11_cifar100_resnet_lt_tuned_benchmark/final_execution_plan/gate_matrix.csv is absent"
    if not final_execution_gates.empty and {"gate_id", "status"}.issubset(final_execution_gates.columns):
        final_execution_status = status_lookup(final_execution_gates, "gate_id", "status")
        final_execution_state = "ready" if set(final_execution_status.values()) == {"pass"} else "not_ready"
        final_execution_evidence = "; ".join(
            f"{gate}={status}" for gate, status in final_execution_status.items()
        )
    final_eval_state = "missing"
    final_eval_evidence = (
        "results/e11_cifar100_resnet_lt_tuned_benchmark/final_evaluation/claim_gate_report.csv is absent"
    )
    if not final_eval_gates.empty and {"gate_id", "status", "claim_state"}.issubset(final_eval_gates.columns):
        final_eval_status = status_lookup(final_eval_gates, "gate_id", "status")
        final_eval_claim = status_lookup(final_eval_gates, "gate_id", "claim_state")
        final_eval_state = final_eval_claim.get("TFE-6-final-claim-state", "unknown")
        final_eval_evidence = "; ".join(f"{gate}={status}" for gate, status in final_eval_status.items())
    final_launch_state = "missing"
    final_launch_evidence = (
        "results/e11_cifar100_resnet_lt_tuned_benchmark/final_launch_audit/latest_final_launch_decision.csv is absent"
    )
    if not final_launch_decision.empty and "submission_status" in final_launch_decision.columns:
        final_launch_state = str(final_launch_decision["submission_status"].iloc[0])
        launch_status = status_lookup(final_launch_gates, "gate_id", "status")
        submit_command = ""
        if "submit_command" in final_launch_decision.columns:
            submit_command = str(final_launch_decision["submit_command"].fillna("").iloc[0])
        final_launch_evidence = (
            "; ".join(f"{gate}={status}" for gate, status in launch_status.items())
            + f"; submit_command={submit_command or '<empty>'}"
        )
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
            "item": "Tuned validation leakage audit",
            "state": leakage_state,
            "evidence": leakage_evidence,
            "reviewer_instruction": "Use this audit to show partial validation observations cannot change the registry, selection rule, launch order, or final seed quarantine.",
        },
        {
            "item": "Clean worktree replay",
            "state": clean_replay_state,
            "evidence": clean_replay_evidence,
            "reviewer_instruction": "Use this as tracked-source replay evidence that e11-check does not depend on local untracked attachments.",
        },
        {
            "item": "PDF render boundary audit",
            "state": pdf_render_state,
            "evidence": pdf_render_evidence,
            "reviewer_instruction": "Use this as rendered-PDF byte/header/hash and source-claim trace evidence; do not claim text-layer or metadata inspection until PRB-4/PRB-5 pass.",
        },
        {
            "item": "Tuned final execution gates",
            "state": final_execution_state,
            "evidence": final_execution_evidence,
            "reviewer_instruction": "Do not run final-safe-submit until every FEP gate passes; this CPU plan does not submit final jobs.",
        },
        {
            "item": "Tuned final evaluator",
            "state": final_eval_state,
            "evidence": final_eval_evidence,
            "reviewer_instruction": "Use this only after final outputs exist; current not_ready gates forbid final benchmark wording.",
        },
        {
            "item": "Tuned final launch audit",
            "state": final_launch_state,
            "evidence": final_launch_evidence,
            "reviewer_instruction": "Treat an empty submit_command and blocked_final_launch_gates_not_ready state as evidence that artifact review has not launched final claims.",
        },
        {
            "item": "v5 final layer tables",
            "state": (
                "present"
                if final_arch_layer_summary.exists() and final_data_layer_summary.exists()
                else "pending_not_required_for_current_claims"
            ),
            "evidence": "Both frozen final split layer tables are present; the frozen evaluator reports failed registered gates and P0 remains not_ready.",
            "reviewer_instruction": "Do not upgrade the v5 predictive-condition claim; any score repair needs a new unspent protocol.",
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
            "objection": "Does the CPU artifact gate replay from tracked files only?",
            "answer": "Yes. The clean-worktree replay audit creates a detached Git worktree, confirms serverREADME.md is absent, runs make e11-check, observes pytest and git diff --check, and confirms the replay worktree stays clean.",
            "status": "tracked_source_replay_supported",
            "forbidden_shortcut": "Do not treat this as preferred pdflatex/bibtex/xelatex clean-checkout reproducibility.",
        },
        {
            "objection": "Can reviewers verify rendered PDF text-layer and page metadata here?",
            "answer": "Partly. The PDF render boundary audit verifies PDF bytes, headers, hashes, and source-level claim trace now, but rendered text-layer and page-metadata inspection are not_ready on this server until pdftotext, pdfinfo, mutool, or a Python PDF parser is available.",
            "status": "pdf_render_boundary_declared",
            "forbidden_shortcut": "Do not claim rendered-PDF text-layer or metadata verification from source-level claim trace alone.",
        },
        {
            "objection": "Do reviewers need GPUs to check the current paper?",
            "answer": "No. The current submitted evidence bundle, including the completed v5 final boundary readout, is checked by CPU-side Make targets; GPUs via Slurm are only for new reruns or new evidence.",
            "status": "cpu_review_path",
            "forbidden_shortcut": "Do not treat GPU reruns as a way to repair the already observed frozen v5 final failures.",
        },
        {
            "objection": "Can partial tuned validation create optional-stopping leakage?",
            "answer": "The tuned leakage audit records TLA guards for frozen selection rules, validation-only seed inputs, contiguous array order, partial-grid selection blocking, final-output quarantine, and queue-audited launches; partial validation observations are progress accounting only.",
            "status": "partial_validation_leakage_controlled",
            "forbidden_shortcut": "Do not use the partial leaderboard to change the recipe grid, selection rule, launch order, final seed plan, or benchmark wording.",
        },
        {
            "objection": "Can final benchmark jobs or final-performance claims be triggered by artifact review?",
            "answer": "No. The final execution plan, final evaluator, and final launch audit are CPU dry-run/not_ready gates: final outputs are absent, no submit command is emitted, and final seeds 20..29 remain quarantined until selection gates and queue capacity pass.",
            "status": "final_benchmark_blocked",
            "forbidden_shortcut": "Do not run final-safe-submit or claim optimizer-performance from partial validation, dry-run launch audits, or not_ready final evaluator gates.",
        },
        {
            "objection": "Why are v5 final and natural-negative claims still limited?",
            "answer": "The v5 final splits are complete and failed the registered P0 gate family. The natural-negative phase1 family is complete and supports only a finite registered null candidate with detectable-effect and tail-quality caveats.",
            "status": "claim_boundary_preserved",
            "forbidden_shortcut": "Do not use failed v5 finals, not_ready gates, partial-family, or finite phase1 outputs as broader positive evidence.",
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
        "claim_boundary": "current evidence bundle only; finite phase1 natural-null candidate with caveats; no v5 final, optional-stopping, partial-grid, final tuned benchmark, or optimizer-performance upgrade",
    }
    (OUTPUT_DIR / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    text = f"""# E11 Artifact Review Packet

This generated packet converts the submission reproducibility audit into an artifact-review response plan. It is intentionally conservative: it supports the current evidence bundle, records the CPU reviewer path, and keeps preferred LaTeX clean-checkout, v5 final, partial-validation optional-stopping, unqualified natural-null, final tuned benchmark launch/claim gates, and optimizer-performance claims outside the current artifact boundary.

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

Blocked now: preferred pdflatex/bibtex/xelatex clean-checkout reproducibility, any v5 predictive-condition upgrade, partial-validation optional-stopping or recipe-selection authority, final tuned benchmark Slurm submission or final-performance claim until final execution/evaluation/launch gates pass, any unqualified natural-null or counterexample wording, and any broad optimizer-performance claim.

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
