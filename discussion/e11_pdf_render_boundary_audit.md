# E11 PDF Render Boundary Audit

This generated audit separates what is currently verified about the rendered paper PDFs from what still requires a venue-style PDF inspection environment. The committed source-level claim trace passes, and the rendered PDF hashes/headers pass, but this server lacks a PDF text-extraction path for checking that every source-level claim-boundary phrase is visible in the rendered text layer.

## PDF Inspection Tool Status

| tool                       | available   | path   | role                                  |
|:---------------------------|:------------|:-------|:--------------------------------------|
| pdftotext                  | no          |        | external PDF text/metadata inspection |
| pdfinfo                    | no          |        | external PDF text/metadata inspection |
| mutool                     | no          |        | external PDF text/metadata inspection |
| gs                         | no          |        | external PDF text/metadata inspection |
| python:pypdf               | no          |        | Python PDF text extraction fallback   |
| python:PyPDF2              | no          |        | Python PDF text extraction fallback   |
| python:fitz                | no          |        | Python PDF text extraction fallback   |
| python:pdfminer.high_level | no          |        | Python PDF text extraction fallback   |

## Render Boundary Gates

| gate_id                          | status    | evidence                                                                   | claim_effect                                                   |
|:---------------------------------|:----------|:---------------------------------------------------------------------------|:---------------------------------------------------------------|
| PRB-1-rendered-pdf-binaries      | pass      | main.pdf and two_page.pdf pass existing header/size checks                 | rendered PDF binary artifacts are present                      |
| PRB-2-pdf-hashes-recorded        | pass      | sha256 recorded for every rendered PDF                                     | reviewers can verify exact PDF bytes                           |
| PRB-3-source-claim-trace-covered | pass      | manuscript claim trace anchors and blocked-phrase audit pass on main.tex   | source-level claim boundary is verified                        |
| PRB-4-pdf-text-extraction-tool   | not_ready | pdftotext/mutool/Python PDF text libraries are unavailable                 | rendered-PDF text-layer audit remains a venue-environment gate |
| PRB-5-pdf-metadata-tool          | not_ready | pdfinfo and mutool are unavailable                                         | page-count and metadata audit remains external to this server  |
| PRB-6-preferred-latex-toolchain  | not_ready | mirrors R3-preferred-latex-toolchain from submission reproducibility audit | preferred venue-style LaTeX clean-checkout boundary            |

## Operating Rule

Allowed now: cite rendered PDF byte hashes, PDF headers, Tectonic-backed build evidence, and source-level claim-boundary checks.

Blocked now: claiming rendered-PDF text-layer or page-metadata verification until a clean venue environment provides `pdftotext`, `pdfinfo`, `mutool`, or an importable Python PDF parser and this audit is rerun.

Artifacts:
- [pdf_inspection_tool_status.csv](../results/e11_pdf_render_boundary_audit/pdf_inspection_tool_status.csv)
- [render_boundary_gates.csv](../results/e11_pdf_render_boundary_audit/render_boundary_gates.csv)
- [config.json](../results/e11_pdf_render_boundary_audit/config.json)
