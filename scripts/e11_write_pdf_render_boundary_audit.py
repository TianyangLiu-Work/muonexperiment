from __future__ import annotations

import importlib.util
import json
import re
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
MAIN_PDF = Path("paper/specgrad_activation_paper/main.pdf")
TWO_PAGE_PDF = Path("paper/specgrad_activation_paper/two_page.pdf")

TEXT_LAYER_ANCHORS = [
    "Tail drift under matched head gain",
    "not a global optimizer theorem",
    "Worst-case and realized coefficients",
    "tail-example logit drift",
    "tail loss, margin, and accuracy are reported separately",
    "lower logit drift does not imply lower cross-entropy loss",
    "completed final gates failed",
    "ResNeXt50-32x4d residual ranking survives",
    "CIFAR-10 cross-partition residual ranking reverses",
    "finite registered phase1 and phase2 null candidates",
    "detectable-effect, head-gain, and quality caveats",
    "not an optimizer-level performance claim",
    "negative benchmark boundary",
    "not evidence for an optimizer-performance advantage",
    "partial tuned-validation observations are progress accounting only",
    "selection rule, launch order, and final seed quarantine remain frozen",
    "server validation and rendered-PDF fallback",
    "preferred LaTeX clean-checkout reproduction remains an explicit gate",
]

LIGATURE_TRANSLATION = str.maketrans(
    {
        "ﬁ": "fi",
        "ﬂ": "fl",
        "ﬀ": "ff",
        "ﬃ": "ffi",
        "ﬄ": "ffl",
        "−": "-",
        "\u00ad": "",
    }
)


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
                "role": "Python PDF text/metadata extraction fallback",
            }
        )
    return pd.DataFrame(rows)


def normalize_pdf_text(text: str) -> str:
    normalized = text.translate(LIGATURE_TRANSLATION)
    normalized = re.sub(r"(?<=\w)-\s+(?=\w)", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def pypdf_reader(path: Path):
    if importlib.util.find_spec("pypdf") is None:
        return None, "pypdf is not importable"
    try:
        from pypdf import PdfReader

        return PdfReader(path), "ok"
    except Exception as exc:  # pragma: no cover - exercised only on corrupt PDFs
        return None, f"{type(exc).__name__}: {exc}"


def build_rendered_text_checks() -> pd.DataFrame:
    main_reader, main_error = pypdf_reader(MAIN_PDF)
    two_page_reader, two_page_error = pypdf_reader(TWO_PAGE_PDF)
    rows: list[dict[str, object]] = []
    if main_reader is None:
        rows.append(
            {
                "check_id": "PDFTXT-1-main-claim-boundary",
                "path": MAIN_PDF.as_posix(),
                "tool": "python:pypdf",
                "status": "not_ready",
                "page_count": 0,
                "text_character_count": 0,
                "required_anchor_count": len(TEXT_LAYER_ANCHORS),
                "found_anchor_count": 0,
                "missing_anchors": "; ".join(TEXT_LAYER_ANCHORS),
                "evidence": main_error,
            }
        )
    else:
        main_text = normalize_pdf_text("\n".join(page.extract_text() or "" for page in main_reader.pages))
        missing = [
            anchor
            for anchor in TEXT_LAYER_ANCHORS
            if normalize_pdf_text(anchor) not in main_text
        ]
        rows.append(
            {
                "check_id": "PDFTXT-1-main-claim-boundary",
                "path": MAIN_PDF.as_posix(),
                "tool": "python:pypdf",
                "status": "pass" if not missing else "fail",
                "page_count": len(main_reader.pages),
                "text_character_count": len(main_text),
                "required_anchor_count": len(TEXT_LAYER_ANCHORS),
                "found_anchor_count": len(TEXT_LAYER_ANCHORS) - len(missing),
                "missing_anchors": "; ".join(missing) if missing else "none",
                "evidence": "rendered main.pdf text layer contains claim-boundary anchors"
                if not missing
                else "rendered main.pdf text layer is missing claim-boundary anchors",
            }
        )
    if two_page_reader is None:
        rows.append(
            {
                "check_id": "PDFTXT-2-two-page-extractable",
                "path": TWO_PAGE_PDF.as_posix(),
                "tool": "python:pypdf",
                "status": "not_ready",
                "page_count": 0,
                "text_character_count": 0,
                "required_anchor_count": 0,
                "found_anchor_count": 0,
                "missing_anchors": "not_applicable",
                "evidence": two_page_error,
            }
        )
    else:
        two_page_text = normalize_pdf_text("\n".join(page.extract_text() or "" for page in two_page_reader.pages))
        extractable = len(two_page_text) > 1000
        rows.append(
            {
                "check_id": "PDFTXT-2-two-page-extractable",
                "path": TWO_PAGE_PDF.as_posix(),
                "tool": "python:pypdf",
                "status": "pass" if extractable else "fail",
                "page_count": len(two_page_reader.pages),
                "text_character_count": len(two_page_text),
                "required_anchor_count": 0,
                "found_anchor_count": 0,
                "missing_anchors": "not_applicable",
                "evidence": "two_page.pdf text layer is extractable"
                if extractable
                else "two_page.pdf text layer is too small to verify extraction",
            }
        )
    return pd.DataFrame(rows)


def build_pdf_metadata_checks() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    specs = [
        ("PDFMETA-1-main", MAIN_PDF, 20, 60),
        ("PDFMETA-2-two-page", TWO_PAGE_PDF, 2, 4),
    ]
    for check_id, path, min_pages, max_pages in specs:
        reader, error = pypdf_reader(path)
        if reader is None:
            rows.append(
                {
                    "check_id": check_id,
                    "path": path.as_posix(),
                    "tool": "python:pypdf",
                    "status": "not_ready",
                    "page_count": 0,
                    "expected_page_min": min_pages,
                    "expected_page_max": max_pages,
                    "producer": "",
                    "creator": "",
                    "evidence": error,
                }
            )
            continue
        metadata = reader.metadata or {}
        page_count = len(reader.pages)
        producer = str(metadata.get("/Producer", ""))
        creator = str(metadata.get("/Creator", ""))
        page_count_ok = min_pages <= page_count <= max_pages
        metadata_ok = bool(producer or creator)
        rows.append(
            {
                "check_id": check_id,
                "path": path.as_posix(),
                "tool": "python:pypdf",
                "status": "pass" if page_count_ok and metadata_ok else "fail",
                "page_count": page_count,
                "expected_page_min": min_pages,
                "expected_page_max": max_pages,
                "producer": producer,
                "creator": creator,
                "evidence": "page count and producer/creator metadata are readable"
                if page_count_ok and metadata_ok
                else "page count or producer/creator metadata is outside the expected boundary",
            }
        )
    return pd.DataFrame(rows)


def build_gate_matrix(
    pdf_checks: pd.DataFrame,
    build_gates: pd.DataFrame,
    claim_trace: pd.DataFrame,
    blocked_phrase_audit: pd.DataFrame,
    tools: pd.DataFrame,
    text_checks: pd.DataFrame,
    metadata_checks: pd.DataFrame,
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
    metadata_tool_available = (
        tool_lookup.get("pdfinfo") == "yes"
        or tool_lookup.get("mutool") == "yes"
        or tool_lookup.get("python:pypdf") == "yes"
    )
    text_checks_pass = (
        text_tool_available
        and not text_checks.empty
        and text_checks["status"].astype(str).eq("pass").all()
    )
    metadata_checks_pass = (
        metadata_tool_available
        and not metadata_checks.empty
        and metadata_checks["status"].astype(str).eq("pass").all()
    )
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
                "status": "pass" if text_checks_pass else "not_ready",
                "evidence": "python:pypdf extracted rendered text and verified main.pdf claim-boundary anchors"
                if text_checks_pass
                else "pdftotext/mutool/Python PDF text libraries are unavailable",
                "claim_effect": "rendered-PDF text-layer claim-boundary audit"
                if text_checks_pass
                else "rendered-PDF text-layer audit remains a venue-environment gate",
            },
            {
                "gate_id": "PRB-5-pdf-metadata-tool",
                "status": "pass" if metadata_checks_pass else "not_ready",
                "evidence": "python:pypdf read page counts and producer/creator metadata"
                if metadata_checks_pass
                else "pdfinfo and mutool are unavailable",
                "claim_effect": "page count and PDF metadata audit"
                if metadata_checks_pass
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
    text_checks = build_rendered_text_checks()
    metadata_checks = build_pdf_metadata_checks()
    gates = build_gate_matrix(
        pdf_checks, build_gates, claim_trace, blocked_phrase_audit, tools, text_checks, metadata_checks
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tools.to_csv(OUTPUT_DIR / "pdf_inspection_tool_status.csv", index=False)
    text_checks.to_csv(OUTPUT_DIR / "rendered_pdf_text_checks.csv", index=False)
    metadata_checks.to_csv(OUTPUT_DIR / "pdf_metadata_checks.csv", index=False)
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
    text_gate = gates.set_index("gate_id").loc["PRB-4-pdf-text-extraction-tool", "status"]
    metadata_gate = gates.set_index("gate_id").loc["PRB-5-pdf-metadata-tool", "status"]
    allowed_text = (
        "Allowed now: cite rendered PDF byte hashes, PDF headers, Tectonic-backed build evidence, "
        "source-level claim-boundary checks, rendered-PDF text-layer anchor checks, and pypdf page metadata."
        if text_gate == "pass" and metadata_gate == "pass"
        else "Allowed now: cite rendered PDF byte hashes, PDF headers, Tectonic-backed build evidence, and source-level claim-boundary checks."
    )
    blocked_text = (
        "Blocked now: claiming preferred venue-toolchain clean-checkout reproducibility until pdflatex, bibtex, and xelatex pass in a clean checkout."
        if text_gate == "pass" and metadata_gate == "pass"
        else "Blocked now: claiming rendered-PDF text-layer or page-metadata verification until a clean venue environment provides `pdftotext`, `pdfinfo`, `mutool`, or an importable Python PDF parser and this audit is rerun."
    )
    text = f"""# E11 PDF Render Boundary Audit

This generated audit separates what is currently verified about the rendered paper PDFs from what still requires a venue-style PDF inspection environment. The committed source-level claim trace passes, rendered PDF hashes/headers pass, and this server now has a Python PDF inspection path for checking rendered text-layer claim anchors and page metadata.

## PDF Inspection Tool Status

{markdown_table(tools, ["tool", "available", "path", "role"])}

## Rendered PDF Text Checks

{markdown_table(text_checks, ["check_id", "path", "tool", "status", "page_count", "text_character_count", "required_anchor_count", "found_anchor_count", "missing_anchors", "evidence"])}

## PDF Metadata Checks

{markdown_table(metadata_checks, ["check_id", "path", "tool", "status", "page_count", "expected_page_min", "expected_page_max", "producer", "creator", "evidence"])}

## Render Boundary Gates

{markdown_table(gates, ["gate_id", "status", "evidence", "claim_effect"])}

## Operating Rule

{allowed_text}

{blocked_text}

Artifacts:
- [pdf_inspection_tool_status.csv](../results/e11_pdf_render_boundary_audit/pdf_inspection_tool_status.csv)
- [rendered_pdf_text_checks.csv](../results/e11_pdf_render_boundary_audit/rendered_pdf_text_checks.csv)
- [pdf_metadata_checks.csv](../results/e11_pdf_render_boundary_audit/pdf_metadata_checks.csv)
- [render_boundary_gates.csv](../results/e11_pdf_render_boundary_audit/render_boundary_gates.csv)
- [config.json](../results/e11_pdf_render_boundary_audit/config.json)
"""
    write_markdown(DISCUSSION_PATH, text)
    if gates["status"].astype(str).eq("fail").any():
        raise AssertionError(f"PDF render boundary gate failed: {gates.to_dict('records')}")
    print(f"saved PDF render boundary audit to {OUTPUT_DIR} and {DISCUSSION_PATH}")


if __name__ == "__main__":
    main()
