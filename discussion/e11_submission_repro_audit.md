# E11 Submission Reproducibility Audit

This generated audit records the current submission artifact boundary for the
head-to-tail paper. It is intentionally conservative: Tectonic fallback builds
are useful evidence, but the preferred pdflatex/bibtex/xelatex clean-checkout
gate remains `not_ready` when those tools are absent on the server.

## Toolchain Status

| tool     | available   | path                                            | version            | role                                           | audit_status   |
|:---------|:------------|:------------------------------------------------|:-------------------|:-----------------------------------------------|:---------------|
| pdflatex | no          |                                                 |                    | preferred main-paper LaTeX engine              | missing        |
| bibtex   | no          |                                                 |                    | preferred bibliography pass for pdflatex build | missing        |
| xelatex  | no          |                                                 |                    | preferred two-page report LaTeX engine         | missing        |
| tectonic | yes         | /home/tyliu/.local/bin/tectonic                 | Tectonic 0.16.9    | fallback reproducible LaTeX engine             | available      |
| gs       | no          |                                                 |                    | optional PDF text-layer audit                  | missing        |
| git      | yes         | /usr/bin/git                                    | git version 2.34.1 | source revision and clean-tree audit           | available      |
| python   | yes         | /data/conda_envs/SpatialQuantization/bin/python | Python 3.10.20     | artifact-generation runtime                    | available      |

## Rendered PDF Checks

| path                                         | exists   |   size_bytes | sha256                                                           | header_is_pdf   | audit_status   |
|:---------------------------------------------|:---------|-------------:|:-----------------------------------------------------------------|:----------------|:---------------|
| paper/specgrad_activation_paper/main.pdf     | yes      |      2597001 | bcf33e940409990285405f3a25a9195fc23c21092cc148926bff3954a7976100 | yes             | pass           |
| paper/specgrad_activation_paper/two_page.pdf | yes      |        62048 | 4246bf5fbc845be24d7daaddf1aa6075137cabdde095242e23213fdc9144e212 | yes             | pass           |

## Source Package Manifest

| path                                                                   | exists   |   size_bytes | sha256                                                           | role                                         | audit_status   |
|:-----------------------------------------------------------------------|:---------|-------------:|:-----------------------------------------------------------------|:---------------------------------------------|:---------------|
| paper/specgrad_activation_paper/main.tex                               | yes      |       112036 | 80fe1188f25d4e1f7e6108f21df8fb67d61656f5bc4f0f52a5ab8ef8bc7ec0ce | paper source or root reproduction entrypoint | pass           |
| paper/specgrad_activation_paper/two_page.tex                           | yes      |        13228 | f1e3bff2a16a58c6b6ae373307149578e96633d46acf2c819beacb2e8302cca9 | paper source or root reproduction entrypoint | pass           |
| paper/specgrad_activation_paper/references.bib                         | yes      |         4906 | 64c7430cdd52637a545554f05e4e48575e0dd955877bc81db576b05fbcf52f48 | paper source or root reproduction entrypoint | pass           |
| paper/specgrad_activation_paper/Makefile                               | yes      |          817 | b19b4b3fdf5d4fec90db0de6af2d86b107956c8f9c9c7a5341d03c4f103996b2 | paper source or root reproduction entrypoint | pass           |
| paper/specgrad_activation_paper/README.md                              | yes      |         7273 | 5f8308a1d94828c6ff4a164cf9796777057f27a099c7d3197ae9b626d1cc41ad | paper source or root reproduction entrypoint | pass           |
| paper/specgrad_activation_paper/tables/e11_paper_numbers.tex           | yes      |        55974 | 70b0130ae2d7fb538f3316745d01387f714464b53e1597ec6286032aa488f227 | paper source or root reproduction entrypoint | pass           |
| paper/specgrad_activation_paper/tables/head_tail_empirical_results.tex | yes      |         4934 | 1264d8923f3cfff52ae9af90f44b10bd29710c949ed93fc3d2eda5dff7b30471 | paper source or root reproduction entrypoint | pass           |
| paper/specgrad_activation_paper/tables/local_linearization_errors.tex  | yes      |          961 | 12c108dd5bfdfd2d1ad219904a4f82f1d10e01ab6d6de4cd86a9252cb298a5ef | paper source or root reproduction entrypoint | pass           |
| Makefile                                                               | yes      |        19211 | 11e92cd8ccbedcb093596bc34cc4d658b276d73c3c4ae47ec4dc0c74c330d609 | paper source or root reproduction entrypoint | pass           |
| README_E11.md                                                          | yes      |        97160 | 0091d7d5d891a5505e327bdee743fb6e9851089c1d60de3f289dbf3a885cc6b6 | paper source or root reproduction entrypoint | pass           |
| paper/specgrad_activation_paper/figures                                | yes      |      2923280 | 21 png files                                                     | paper-local figure bundle                    | pass           |

## Build Gate Summary

| gate_id                        | status    | evidence                                                                                                                        | required_next_action                                                                                   |
|:-------------------------------|:----------|:--------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------|
| R1-source-revision             | pass      | the commit containing this audit is the source revision                                                                         | record the pushed commit in the final run summary                                                      |
| R2-working-tree-scope          | info      | audit generated during an intentional evidence update; final git status is checked before commit                                | keep serverREADME.md untracked and stage only intentional evidence files                               |
| R3-preferred-latex-toolchain   | not_ready | pdflatex/bibtex/xelatex not all available                                                                                       | run a clean checkout with pdflatex/bibtex/xelatex before claiming full venue-toolchain reproducibility |
| R4-tectonic-fallback-toolchain | pass      | tectonic available                                                                                                              | use make e11-paper-pdf or paper/specgrad_activation_paper make tectonic on this server                 |
| R5-rendered-pdfs               | pass      | main.pdf and two_page.pdf have PDF headers and expected sizes                                                                   | rebuild paper PDFs if either rendered artifact fails                                                   |
| R6-full-artifact-validation    | pass      | make PYTHON=/data/conda_envs/SpatialQuantization/bin/python e11-full passed on 2026-06-17 after bold conjecture register update | record the make e11-full result in the commit/push summary                                             |

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
