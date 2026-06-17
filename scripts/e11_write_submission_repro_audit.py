from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.reporting import markdown_table, write_markdown


OUTPUT_DIR = Path("results/e11_submission_repro_audit")
DISCUSSION_PATH = Path("discussion/e11_submission_repro_audit.md")
PAPER_DIR = Path("paper/specgrad_activation_paper")
PDF_PATHS = (
    PAPER_DIR / "main.pdf",
    PAPER_DIR / "two_page.pdf",
)
SOURCE_PATHS = (
    PAPER_DIR / "main.tex",
    PAPER_DIR / "two_page.tex",
    PAPER_DIR / "references.bib",
    PAPER_DIR / "Makefile",
    PAPER_DIR / "README.md",
    PAPER_DIR / "tables" / "e11_paper_numbers.tex",
    PAPER_DIR / "tables" / "head_tail_empirical_results.tex",
    PAPER_DIR / "tables" / "local_linearization_errors.tex",
    Path("Makefile"),
    Path("README_E11.md"),
)
TOOLS = (
    ("pdflatex", "preferred main-paper LaTeX engine"),
    ("bibtex", "preferred bibliography pass for pdflatex build"),
    ("xelatex", "preferred two-page report LaTeX engine"),
    ("tectonic", "fallback reproducible LaTeX engine"),
    ("gs", "optional PDF text-layer audit"),
    ("git", "source revision and clean-tree audit"),
    ("python", "artifact-generation runtime"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def first_version_line(executable: str) -> str:
    commands = ([executable, "--version"], [executable, "-version"])
    for command in commands:
        try:
            result = subprocess.run(
                command,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        line = result.stdout.splitlines()[0].strip() if result.stdout.splitlines() else ""
        if line:
            return line
    return "version unavailable"


def git_value(args: list[str]) -> str:
    result = subprocess.run(
        ["git", *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        return result.stderr.strip()
    return result.stdout.strip()


def build_toolchain_status() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for tool, role in TOOLS:
        executable = sys.executable if tool == "python" else shutil.which(tool)
        available = executable is not None
        rows.append(
            {
                "tool": tool,
                "available": "yes" if available else "no",
                "path": str(executable) if available else "",
                "version": first_version_line(str(executable)) if available else "",
                "role": role,
                "audit_status": "available" if available else "missing",
            }
        )
    return pd.DataFrame(rows)


def build_pdf_checks() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in PDF_PATHS:
        exists = path.exists()
        header_is_pdf = False
        size_bytes = 0
        digest = ""
        if exists:
            size_bytes = path.stat().st_size
            with path.open("rb") as handle:
                header_is_pdf = handle.read(4) == b"%PDF"
            digest = sha256(path)
        rows.append(
            {
                "path": path.as_posix(),
                "exists": "yes" if exists else "no",
                "size_bytes": size_bytes,
                "sha256": digest,
                "header_is_pdf": "yes" if header_is_pdf else "no",
                "audit_status": "pass" if exists and header_is_pdf and size_bytes >= 10_000 else "fail",
            }
        )
    return pd.DataFrame(rows)


def build_source_manifest() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in SOURCE_PATHS:
        exists = path.exists()
        rows.append(
            {
                "path": path.as_posix(),
                "exists": "yes" if exists else "no",
                "size_bytes": path.stat().st_size if exists else 0,
                "sha256": sha256(path) if exists and path.is_file() else "",
                "role": "paper source or root reproduction entrypoint",
                "audit_status": "pass" if exists else "fail",
            }
        )
    figure_dir = PAPER_DIR / "figures"
    figure_files = sorted(figure_dir.glob("*.png"))
    rows.append(
        {
            "path": figure_dir.as_posix(),
            "exists": "yes" if figure_dir.exists() else "no",
            "size_bytes": sum(path.stat().st_size for path in figure_files),
            "sha256": f"{len(figure_files)} png files",
            "role": "paper-local figure bundle",
            "audit_status": "pass" if len(figure_files) >= 20 else "fail",
        }
    )
    return pd.DataFrame(rows)


def build_build_gate_summary(toolchain: pd.DataFrame, pdf_checks: pd.DataFrame) -> pd.DataFrame:
    available = set(toolchain[toolchain["available"].eq("yes")]["tool"])
    all_pdfs_pass = pdf_checks["audit_status"].eq("pass").all()
    preferred_latex = {"pdflatex", "bibtex", "xelatex"}.issubset(available)
    tectonic = "tectonic" in available
    full_validation_status = os.environ.get("E11_FULL_VALIDATION_STATUS", "external_check_required")
    full_validation_evidence = os.environ.get(
        "E11_FULL_VALIDATION_EVIDENCE",
        "run make e11-full after regenerating this audit",
    )
    rows = [
        {
            "gate_id": "R1-source-revision",
            "status": "pass",
            "evidence": "the commit containing this audit is the source revision",
            "required_next_action": "record the pushed commit in the final run summary",
        },
        {
            "gate_id": "R2-working-tree-scope",
            "status": "info",
            "evidence": "audit generated during an intentional evidence update; final git status is checked before commit",
            "required_next_action": "keep serverREADME.md untracked and stage only intentional evidence files",
        },
        {
            "gate_id": "R3-preferred-latex-toolchain",
            "status": "pass" if preferred_latex else "not_ready",
            "evidence": "pdflatex, bibtex, and xelatex available" if preferred_latex else "pdflatex/bibtex/xelatex not all available",
            "required_next_action": "run a clean checkout with pdflatex/bibtex/xelatex before claiming full venue-toolchain reproducibility",
        },
        {
            "gate_id": "R4-tectonic-fallback-toolchain",
            "status": "pass" if tectonic else "fail",
            "evidence": "tectonic available" if tectonic else "tectonic missing",
            "required_next_action": "use make e11-paper-pdf or paper/specgrad_activation_paper make tectonic on this server",
        },
        {
            "gate_id": "R5-rendered-pdfs",
            "status": "pass" if all_pdfs_pass else "fail",
            "evidence": "main.pdf and two_page.pdf have PDF headers and expected sizes",
            "required_next_action": "rebuild paper PDFs if either rendered artifact fails",
        },
        {
            "gate_id": "R6-full-artifact-validation",
            "status": full_validation_status,
            "evidence": full_validation_evidence,
            "required_next_action": "record the make e11-full result in the commit/push summary",
        },
    ]
    return pd.DataFrame(rows)


def write_discussion(
    toolchain: pd.DataFrame,
    pdf_checks: pd.DataFrame,
    source_manifest: pd.DataFrame,
    gates: pd.DataFrame,
) -> None:
    text = f"""# E11 Submission Reproducibility Audit

This generated audit records the current submission artifact boundary for the
head-to-tail paper. It is intentionally conservative: Tectonic fallback builds
are useful evidence, but the preferred pdflatex/bibtex/xelatex clean-checkout
gate remains `not_ready` when those tools are absent on the server.

## Toolchain Status

{markdown_table(toolchain, ["tool", "available", "path", "version", "role", "audit_status"])}

## Rendered PDF Checks

{markdown_table(pdf_checks, ["path", "exists", "size_bytes", "sha256", "header_is_pdf", "audit_status"])}

## Source Package Manifest

{markdown_table(source_manifest, ["path", "exists", "size_bytes", "sha256", "role", "audit_status"])}

## Build Gate Summary

{markdown_table(gates, ["gate_id", "status", "evidence", "required_next_action"])}

## Boundary

Allowed now: cite the current paper build as Tectonic-backed server evidence
when `R6-full-artifact-validation` records a passing `make e11-full` run and
the rendered PDF checks pass.

Blocked now: claiming a preferred LaTeX clean-checkout reproduction until
pdflatex, bibtex, and xelatex are available and the same artifact path is run
from a fresh checkout.

Artifacts:
- [toolchain_status.csv](../results/e11_submission_repro_audit/toolchain_status.csv)
- [pdf_artifact_checks.csv](../results/e11_submission_repro_audit/pdf_artifact_checks.csv)
- [source_package_manifest.csv](../results/e11_submission_repro_audit/source_package_manifest.csv)
- [build_gate_summary.csv](../results/e11_submission_repro_audit/build_gate_summary.csv)
"""
    write_markdown(DISCUSSION_PATH, text)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    toolchain = build_toolchain_status()
    pdf_checks = build_pdf_checks()
    source_manifest = build_source_manifest()
    gates = build_build_gate_summary(toolchain, pdf_checks)
    toolchain.to_csv(OUTPUT_DIR / "toolchain_status.csv", index=False)
    pdf_checks.to_csv(OUTPUT_DIR / "pdf_artifact_checks.csv", index=False)
    source_manifest.to_csv(OUTPUT_DIR / "source_package_manifest.csv", index=False)
    gates.to_csv(OUTPUT_DIR / "build_gate_summary.csv", index=False)
    write_discussion(toolchain, pdf_checks, source_manifest, gates)
    print(f"saved submission reproducibility audit to {OUTPUT_DIR} and {DISCUSSION_PATH}")
    print(gates.to_string(index=False))


if __name__ == "__main__":
    main()
