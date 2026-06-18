from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.reporting import markdown_table, write_markdown


OUTPUT_DIR = Path("results/e11_clean_worktree_replay_audit")
DISCUSSION_PATH = Path("discussion/e11_clean_worktree_replay_audit.md")
DEFAULT_PYTHON = "/data/conda_envs/SpatialQuantization/bin/python"


def run_command(
    command: list[str],
    cwd: Path,
    timeout: int | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def git_output(*args: str, cwd: Path = ROOT) -> str:
    result = run_command(["git", *args], cwd=cwd)
    if result.returncode != 0:
        return "unavailable"
    return result.stdout.strip()


def tail_lines(text: str, limit: int = 120) -> str:
    lines = text.splitlines()
    return "\n".join(lines[-limit:]) + ("\n" if lines else "")


def build_gates(
    checkout_ok: bool,
    server_readme_present: bool,
    check_exit_code: int | None,
    check_log: str,
    post_status: str,
    post_untracked: str,
) -> pd.DataFrame:
    check_passed = check_exit_code == 0
    return pd.DataFrame(
        [
            {
                "gate_id": "CWR-1-detached-worktree-created",
                "status": "pass" if checkout_ok else "fail",
                "evidence": "git worktree add --detach completed" if checkout_ok else "worktree checkout failed",
                "claim_effect": "tracked-source replay is available",
            },
            {
                "gate_id": "CWR-2-local-attachment-excluded",
                "status": "pass" if not server_readme_present else "fail",
                "evidence": "serverREADME.md absent from clean worktree"
                if not server_readme_present
                else "serverREADME.md present in clean worktree",
                "claim_effect": "local attachment is not part of the submitted evidence bundle",
            },
            {
                "gate_id": "CWR-3-e11-check-passes",
                "status": "pass" if check_passed else "fail",
                "evidence": f"make e11-check exit_code={check_exit_code}",
                "claim_effect": "validator, pytest, and whitespace checks replay from tracked files",
            },
            {
                "gate_id": "CWR-4-pytest-pass-observed",
                "status": "pass" if "71 passed" in check_log else "fail",
                "evidence": "clean worktree log contains `71 passed`",
                "claim_effect": "test suite pass is visible in replay log",
            },
            {
                "gate_id": "CWR-5-git-diff-check-observed",
                "status": "pass" if "git diff --check" in check_log else "fail",
                "evidence": "clean worktree log contains `git diff --check`",
                "claim_effect": "whitespace check is part of the replayed gate",
            },
            {
                "gate_id": "CWR-6-worktree-clean-after-check",
                "status": "pass" if not post_status.strip() and not post_untracked.strip() else "fail",
                "evidence": "git status --short and untracked list are empty after e11-check"
                if not post_status.strip() and not post_untracked.strip()
                else f"status={post_status!r}; untracked={post_untracked!r}",
                "claim_effect": "the replay check does not mutate tracked evidence files",
            },
        ]
    )


def main() -> None:
    revision_arg = os.environ.get("E11_CLEAN_REPLAY_REVISION", "HEAD")
    python_cmd = os.environ.get("E11_REPLAY_PYTHON", DEFAULT_PYTHON)
    timeout_s = int(os.environ.get("E11_CLEAN_REPLAY_TIMEOUT", "900"))
    replayed_revision = git_output("rev-parse", revision_arg)
    branch = git_output("branch", "--show-current")
    command = ["make", f"PYTHON={python_cmd}", "e11-check"]
    tmp_root = Path(tempfile.mkdtemp(prefix="e11_clean_worktree_"))
    worktree = tmp_root / "worktree"

    started = time.time()
    checkout_log = ""
    check_log = ""
    checkout_ok = False
    check_exit_code: int | None = None
    server_readme_present = False
    pre_status = "not_run"
    pre_untracked = "not_run"
    post_status = "not_run"
    post_untracked = "not_run"
    worktree_path = worktree.as_posix()

    try:
        checkout = run_command(["git", "worktree", "add", "--detach", worktree_path, replayed_revision], cwd=ROOT)
        checkout_log = checkout.stdout + checkout.stderr
        checkout_ok = checkout.returncode == 0
        if checkout_ok:
            server_readme_present = (worktree / "serverREADME.md").exists()
            pre_status = git_output("status", "--short", cwd=worktree)
            pre_untracked = git_output("ls-files", "--others", "--exclude-standard", cwd=worktree)
            check = run_command(command, cwd=worktree, timeout=timeout_s)
            check_exit_code = check.returncode
            check_log = check.stdout + check.stderr
            post_status = git_output("status", "--short", cwd=worktree)
            post_untracked = git_output("ls-files", "--others", "--exclude-standard", cwd=worktree)
    finally:
        if worktree.exists():
            run_command(["git", "worktree", "remove", "--force", worktree_path], cwd=ROOT)
        shutil.rmtree(tmp_root, ignore_errors=True)

    elapsed_s = time.time() - started
    gates = build_gates(
        checkout_ok=checkout_ok,
        server_readme_present=server_readme_present,
        check_exit_code=check_exit_code,
        check_log=check_log,
        post_status=post_status,
        post_untracked=post_untracked,
    )
    output_status = "pass" if gates["status"].eq("pass").all() else "fail"
    run_summary = pd.DataFrame(
        [
            {
                "audit_id": "clean_worktree_e11_check_replay",
                "status": output_status,
                "replayed_revision": replayed_revision,
                "source_branch_at_generation": branch,
                "revision_arg": revision_arg,
                "command": " ".join(command),
                "elapsed_seconds": round(elapsed_s, 3),
                "check_exit_code": check_exit_code if check_exit_code is not None else "not_run",
                "server_readme_present_in_worktree": "yes" if server_readme_present else "no",
                "pre_status_short": pre_status if pre_status else "clean",
                "pre_untracked": pre_untracked if pre_untracked else "none",
                "post_status_short": post_status if post_status else "clean",
                "post_untracked": post_untracked if post_untracked else "none",
            }
        ]
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_summary.to_csv(OUTPUT_DIR / "run_summary.csv", index=False)
    gates.to_csv(OUTPUT_DIR / "gate_matrix.csv", index=False)
    (OUTPUT_DIR / "command_log_tail.txt").write_text(
        tail_lines(checkout_log + "\n" + check_log),
        encoding="utf-8",
    )
    config = {
        "purpose": "clean detached worktree replay of the CPU-side E11 artifact gate",
        "revision_arg": revision_arg,
        "replayed_revision": replayed_revision,
        "python_command": python_cmd,
        "command": " ".join(command),
        "timeout_seconds": timeout_s,
        "new_empirical_results": False,
        "server_readme_policy": "must_be_absent_from_clean_worktree",
        "status": output_status,
    }
    (OUTPUT_DIR / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    text = f"""# E11 Clean Worktree Replay Audit

This generated audit replays the CPU-side artifact gate from a detached clean Git worktree. It is a reproducibility check for tracked files only: `serverREADME.md` is intentionally absent, and the command does not create new empirical evidence.

## Run Summary

{markdown_table(run_summary, ["audit_id", "status", "replayed_revision", "source_branch_at_generation", "command", "elapsed_seconds", "server_readme_present_in_worktree", "post_status_short", "post_untracked"])}

## Gate Matrix

{markdown_table(gates, ["gate_id", "status", "evidence", "claim_effect"])}

## Operating Rule

This audit supports tracked-source CPU replay of `make PYTHON={python_cmd} e11-check`. It does not close the preferred `pdflatex`/`bibtex`/`xelatex` clean-checkout gate and does not authorize stronger scientific claims.

Artifacts:
- [run_summary.csv](../results/e11_clean_worktree_replay_audit/run_summary.csv)
- [gate_matrix.csv](../results/e11_clean_worktree_replay_audit/gate_matrix.csv)
- [command_log_tail.txt](../results/e11_clean_worktree_replay_audit/command_log_tail.txt)
- [config.json](../results/e11_clean_worktree_replay_audit/config.json)
"""
    write_markdown(DISCUSSION_PATH, text)
    if output_status != "pass":
        raise AssertionError(f"clean worktree replay audit failed; see {OUTPUT_DIR}")
    print(f"saved clean worktree replay audit to {OUTPUT_DIR} and {DISCUSSION_PATH}")


if __name__ == "__main__":
    main()
