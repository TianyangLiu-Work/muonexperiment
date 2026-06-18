from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.reporting import markdown_table, write_markdown


OUTPUT_DIR = Path("results/e11_pdf_render_boundary_audit")
DISCUSSION_PATH = Path("discussion/e11_pdf_render_boundary_audit.md")
SUBMISSION_DIR = Path("results/e11_submission_repro_audit")
CLAIM_TRACE_DIR = Path("results/e11_manuscript_claim_trace")


def tool_status() -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for tool in ("pdftotext", "pdfinfo", "mutool", "gs"):
        path = shutil.which(tool)
        rows.append(
            {
                "tool": tool,
                "available": "yes" if path else "no",
                "path": path or "",
                "role": "external PDF text/metadata inspection",
            }
        )
    for module in ("pypdf", "PyPDF2", "fitz", "pdfminer.high_level"):
        try:
            available = importlib.util.find_spec(module) is not None
        except ModuleNotFoundError:
            available = False
        rows.append(
            {
                "tool": f"python:{module}",
                "available": "yes" if available else "no",
                "path": "importable" if available else "",
                "role": "Python PDF text extraction fallback",
            }
        )
    return pd.DataFrame(rows)


def build_gate_matrix(
    pdf_checks: pd.DataFrame,
    build_gates: pd.DataFrame,
    claim_trace: pd.DataFrame,
    blocked_phrase_audit: pd.DataFrame,
    tools: pd.DataFrame,
) -> pd.DataFrame:
    all_pdf_headers = pdf_checks["audit_status"].astype(str).eq("pass").all() and pdf_checks[
        "header_is_pdf"
    ].astype(str).eq("yes").all()
    hashes_recorded = pdf_checks["sha256"].astype(str).str.len().ge(64).all()
    claim_trace_pass = claim_trace["missing_anchors"].astype(str).eq("none").all() and blocked_phrase_audit[
        "audit_status"
    ].astype(str).eq("pass").all()
    tool_lookup = tools.set_index("tool")["available"].astype(str).to_dict()
    text_tool_available = any(
        tool_lookup.get(tool) == "yes"
        for tool in ("pdftotext", "mutool", "python:pypdf", "python:PyPDF2", "python:fitz", "python:pdfminer.high_level")
    )
    metadata_tool_available = tool_lookup.get("pdfinfo") == "yes" or tool_lookup.get("mutool") == "yes"
    submission_gate_lookup = build_gates.set_index("gate_id")["status"].astype(str).to_dict()
    return pd.DataFrame(
        [
            {
                "gate_id": "PRB-1-rendered-pdf-binaries",
                "status": "pass" if all_pdf_headers else "fail",
                "evidence": "main.pdf and two_page.pdf pass existing header/size checks",
                "claim_effect": "rendered PDF binary artifacts are present",
            },
            {
                "gate_id": "PRB-2-pdf-hashes-recorded",
                "status": "pass" if hashes_recorded else "fail",
                "evidence": "sha256 recorded for every rendered PDF",
                "claim_effect": "reviewers can verify exact PDF bytes",
            },
            {
                "gate_id": "PRB-3-source-claim-trace-covered",
                "status": "pass" if claim_trace_pass else "fail",
                "evidence": "manuscript claim trace anchors and blocked-phrase audit pass on main.tex",
                "claim_effect": "source-level claim boundary is verified",
            },
            {
                "gate_id": "PRB-4-pdf-text-extraction-tool",
                "status": "pass" if text_tool_available else "not_ready",
                "evidence": "at least one PDF text extractor is available"
                if text_tool_available
                else "pdftotext/mutool/Python PDF text libraries are unavailable",
                "claim_effect": "rendered-PDF text-layer claim-boundary audit"
                if text_tool_available
                else "rendered-PDF text-layer audit remains a venue-environment gate",
            },
            {
                "gate_id": "PRB-5-pdf-metadata-tool",
                "status": "pass" if metadata_tool_available else "not_ready",
                "evidence": "pdfinfo or mutool is available"
                if metadata_tool_available
                else "pdfinfo and mutool are unavailable",
                "claim_effect": "page count and PDF metadata audit"
                if metadata_tool_available
                else "page-count and metadata audit remains external to this server",
            },
            {
                "gate_id": "PRB-6-preferred-latex-toolchain",
                "status": submission_gate_lookup.get("R3-preferred-latex-toolchain", "missing"),
                "evidence": "mirrors R3-preferred-latex-toolchain from submission reproducibility audit",
                "claim_effect": "preferred venue-style LaTeX clean-checkout boundary",
            },
        ]
    )


def main() -> None:
    pdf_checks = pd.read_csv(SUBMISSION_DIR / "pdf_artifact_checks.csv")
    build_gates = pd.read_csv(SUBMISSION_DIR / "build_gate_summary.csv")
    claim_trace = pd.read_csv(CLAIM_TRACE_DIR / "claim_trace.csv")
    blocked_phrase_audit = pd.read_csv(CLAIM_TRACE_DIR / "blocked_phrase_audit.csv")
    tools = tool_status()
    gates = build_gate_matrix(pdf_checks, build_gates, claim_trace, blocked_phrase_audit, tools)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tools.to_csv(OUTPUT_DIR / "pdf_inspection_tool_status.csv", index=False)
    gates.to_csv(OUTPUT_DIR / "render_boundary_gates.csv", index=False)
    config = {
        "purpose": "rendered PDF inspection boundary audit",
        "new_empirical_results": False,
        "source_claim_trace": (CLAIM_TRACE_DIR / "claim_trace.csv").as_posix(),
        "rendered_pdf_checks": (SUBMISSION_DIR / "pdf_artifact_checks.csv").as_posix(),
        "text_layer_gate": "PRB-4-pdf-text-extraction-tool",
        "metadata_gate": "PRB-5-pdf-metadata-tool",
    }
    (OUTPUT_DIR / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    text = f"""# E11 PDF Render Boundary Audit

This generated audit separates what is currently verified about the rendered paper PDFs from what still requires a venue-style PDF inspection environment. The committed source-level claim trace passes, and the rendered PDF hashes/headers pass, but this server lacks a PDF text-extraction path for checking that every source-level claim-boundary phrase is visible in the rendered text layer.

## PDF Inspection Tool Status

{markdown_table(tools, ["tool", "available", "path", "role"])}

## Render Boundary Gates

{markdown_table(gates, ["gate_id", "status", "evidence", "claim_effect"])}

## Operating Rule

Allowed now: cite rendered PDF byte hashes, PDF headers, Tectonic-backed build evidence, and source-level claim-boundary checks.

Blocked now: claiming rendered-PDF text-layer or page-metadata verification until a clean venue environment provides `pdftotext`, `pdfinfo`, `mutool`, or an importable Python PDF parser and this audit is rerun.

Artifacts:
- [pdf_inspection_tool_status.csv](../results/e11_pdf_render_boundary_audit/pdf_inspection_tool_status.csv)
- [render_boundary_gates.csv](../results/e11_pdf_render_boundary_audit/render_boundary_gates.csv)
- [config.json](../results/e11_pdf_render_boundary_audit/config.json)
"""
    write_markdown(DISCUSSION_PATH, text)
    if not set(gates["status"].astype(str)).issubset({"pass", "not_ready"}):
        raise AssertionError(f"unexpected PDF render boundary gate status: {set(gates['status'].astype(str))}")
    print(f"saved PDF render boundary audit to {OUTPUT_DIR} and {DISCUSSION_PATH}")


if __name__ == "__main__":
    main()
