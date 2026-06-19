# E11 PDF Render Boundary Audit

This generated audit separates what is currently verified about the rendered paper PDFs from what still requires a venue-style PDF inspection environment. The committed source-level claim trace passes, rendered PDF hashes/headers pass, and this server now has a Python PDF inspection path for checking rendered text-layer claim anchors and page metadata.

## PDF Inspection Tool Status

| tool                       | available   | path       | role                                         |
|:---------------------------|:------------|:-----------|:---------------------------------------------|
| pdftotext                  | no          |            | external PDF text/metadata inspection        |
| pdfinfo                    | no          |            | external PDF text/metadata inspection        |
| mutool                     | no          |            | external PDF text/metadata inspection        |
| gs                         | no          |            | external PDF text/metadata inspection        |
| python:pypdf               | yes         | importable | Python PDF text/metadata extraction fallback |
| python:PyPDF2              | no          |            | Python PDF text/metadata extraction fallback |
| python:fitz                | no          |            | Python PDF text/metadata extraction fallback |
| python:pdfminer.high_level | no          |            | Python PDF text/metadata extraction fallback |

## Rendered PDF Text Checks

| check_id                      | path                                         | tool         | status   |   page_count |   text_character_count |   required_anchor_count |   found_anchor_count | missing_anchors   | evidence                                                     |
|:------------------------------|:---------------------------------------------|:-------------|:---------|-------------:|-----------------------:|------------------------:|---------------------:|:------------------|:-------------------------------------------------------------|
| PDFTXT-1-main-claim-boundary  | paper/specgrad_activation_paper/main.pdf     | python:pypdf | pass     |           33 |                  91208 |                      18 |                   18 | none              | rendered main.pdf text layer contains claim-boundary anchors |
| PDFTXT-2-two-page-extractable | paper/specgrad_activation_paper/two_page.pdf | python:pypdf | pass     |            3 |                   9735 |                       0 |                    0 | not_applicable    | two_page.pdf text layer is extractable                       |

## PDF Metadata Checks

| check_id           | path                                         | tool         | status   |   page_count |   expected_page_min |   expected_page_max | producer        | creator             | evidence                                              |
|:-------------------|:---------------------------------------------|:-------------|:---------|-------------:|--------------------:|--------------------:|:----------------|:--------------------|:------------------------------------------------------|
| PDFMETA-1-main     | paper/specgrad_activation_paper/main.pdf     | python:pypdf | pass     |           33 |                  20 |                  60 | xdvipdfmx (0.1) | LaTeX with hyperref | page count and producer/creator metadata are readable |
| PDFMETA-2-two-page | paper/specgrad_activation_paper/two_page.pdf | python:pypdf | pass     |            3 |                   2 |                   4 | xdvipdfmx (0.1) | LaTeX with hyperref | page count and producer/creator metadata are readable |

## Render Boundary Gates

| gate_id                          | status    | evidence                                                                          | claim_effect                                        |
|:---------------------------------|:----------|:----------------------------------------------------------------------------------|:----------------------------------------------------|
| PRB-1-rendered-pdf-binaries      | pass      | main.pdf and two_page.pdf pass existing header/size checks                        | rendered PDF binary artifacts are present           |
| PRB-2-pdf-hashes-recorded        | pass      | sha256 recorded for every rendered PDF                                            | reviewers can verify exact PDF bytes                |
| PRB-3-source-claim-trace-covered | pass      | manuscript claim trace anchors and blocked-phrase audit pass on main.tex          | source-level claim boundary is verified             |
| PRB-4-pdf-text-extraction-tool   | pass      | python:pypdf extracted rendered text and verified main.pdf claim-boundary anchors | rendered-PDF text-layer claim-boundary audit        |
| PRB-5-pdf-metadata-tool          | pass      | python:pypdf read page counts and producer/creator metadata                       | page count and PDF metadata audit                   |
| PRB-6-preferred-latex-toolchain  | not_ready | mirrors R3-preferred-latex-toolchain from submission reproducibility audit        | preferred venue-style LaTeX clean-checkout boundary |

## Operating Rule

Allowed now: cite rendered PDF byte hashes, PDF headers, Tectonic-backed build evidence, source-level claim-boundary checks, rendered-PDF text-layer anchor checks, and pypdf page metadata.

Blocked now: claiming preferred venue-toolchain clean-checkout reproducibility until pdflatex, bibtex, and xelatex pass in a clean checkout.

Artifacts:
- [pdf_inspection_tool_status.csv](../results/e11_pdf_render_boundary_audit/pdf_inspection_tool_status.csv)
- [rendered_pdf_text_checks.csv](../results/e11_pdf_render_boundary_audit/rendered_pdf_text_checks.csv)
- [pdf_metadata_checks.csv](../results/e11_pdf_render_boundary_audit/pdf_metadata_checks.csv)
- [render_boundary_gates.csv](../results/e11_pdf_render_boundary_audit/render_boundary_gates.csv)
- [config.json](../results/e11_pdf_render_boundary_audit/config.json)
