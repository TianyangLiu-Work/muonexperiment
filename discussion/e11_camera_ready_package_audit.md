# E11 Camera-Ready Package Audit

This generated audit is a package-readiness boundary, not a new empirical result.
It ties the rendered PDFs, source package, claim trace, artifact-review commands,
local attachment exclusion, and venue-toolchain caveats into one camera-ready
checklist.

## Package Item Matrix

| item_id                               | status    | evidence                                                                                            | camera_ready_effect                                                                  | blocked_claim                                                                                 |
|:--------------------------------------|:----------|:----------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------|
| CRP-1-source-bundle                   | pass      | required paper source, root Makefile, and README_E11.md are present in source_package_manifest.csv  | server package source bundle is present                                              | do not claim a complete source package if any required source path is absent                  |
| CRP-2-rendered-pdf-binaries           | pass      | main.pdf and two_page.pdf pass header, size, and hash checks in pdf_artifact_checks.csv             | server package includes rendered PDFs                                                | do not submit stale or non-PDF rendered artifacts                                             |
| CRP-3-claim-trace-clean               | pass      | claim_trace.csv anchors are present and blocked_phrase_audit.csv has no hits                        | main manuscript wording matches the claim-decision contract                          | do not ship if blocked benchmark, predictive-score, or natural-counterexample wording appears |
| CRP-4-reviewer-command-path           | pass      | artifact review command matrix includes e11-full, e11-paper-pdf, e11-check, and packet regeneration | reviewers have a CPU reproduction path for the current bundle                        | do not replace the strongest local gate with a narrower passing command                       |
| CRP-5-local-attachment-excluded       | pass      | artifact review local-state contract keeps serverREADME.md outside committed evidence               | local user attachment is excluded from the submission package                        | do not stage or package serverREADME.md as reproducibility evidence                           |
| CRP-6-preferred-latex-boundary        | not_ready | mirrors R3-preferred-latex-toolchain from submission reproducibility audit                          | preferred venue-toolchain claim remains explicit                                     | do not claim pdflatex/bibtex/xelatex clean-checkout reproducibility from Tectonic evidence    |
| CRP-7-rendered-text-metadata-boundary | not_ready | PRB-4-pdf-text-extraction-tool=not_ready; PRB-5-pdf-metadata-tool=not_ready                         | rendered text-layer and page metadata gate is explicit                               | do not claim rendered-PDF text-layer or metadata verification until PRB-4/PRB-5 pass          |
| CRP-8-final-claim-quarantine          | pass      | TCD-3 is a completed failed boundary and TCD-5 is protocol-pending                                  | camera-ready package cannot imply predictive-score or optimizer-performance upgrades | do not add benchmark-performance or successful predictive-condition wording during packaging  |

## Submission Gate Matrix

| gate_id                       | status    | evidence                                                                                                            | required_next_action                                                                                 |
|:------------------------------|:----------|:--------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------|
| CRG-1-current-server-package  | pass      | source, rendered PDFs, claim trace, reviewer commands, local-file exclusion, and final-claim quarantine are checked | run make e11-full and record the pushed commit before sharing the current server package             |
| CRG-2-venue-toolchain-package | not_ready | requires CRP-6 preferred LaTeX and CRP-7 rendered text/metadata gates to pass                                       | rerun in a clean venue-style environment with pdflatex, bibtex, xelatex, and PDF text/metadata tools |
| CRG-3-claim-boundary-package  | pass      | main manuscript claim anchors and blocked phrase audit agree with top-conference claim decisions                    | regenerate claim-decision and manuscript-trace audits after any manuscript edit                      |
| CRG-4-local-file-exclusion    | pass      | serverREADME.md is local and untracked                                                                              | keep serverREADME.md out of staged submission artifacts                                              |

## Camera-Ready Checklist

| step_id                     | command_or_action                                                                                | required_gate                      | claim_boundary                                                                               |
|:----------------------------|:-------------------------------------------------------------------------------------------------|:-----------------------------------|:---------------------------------------------------------------------------------------------|
| CRC-1-current-server-share  | make PYTHON=/data/conda_envs/SpatialQuantization/bin/python e11-full                             | CRG-1-current-server-package=pass  | server package only; preferred venue-toolchain remains separate if CRG-2 is not_ready        |
| CRC-2-venue-clean-checkout  | run e11-full in a clean checkout with pdflatex, bibtex, xelatex, pdftotext/pdfinfo or equivalent | CRG-2-venue-toolchain-package=pass | needed before claiming full venue-toolchain reproducibility                                  |
| CRC-3-after-manuscript-edit | regenerate e11-top-conference-claim-decision-audit and e11-manuscript-claim-trace                | CRG-3-claim-boundary-package=pass  | no new benchmark, predictive-score, or natural-counterexample wording without matching gates |
| CRC-4-before-commit         | git status --short --branch                                                                      | CRG-4-local-file-exclusion=pass    | serverREADME.md remains local and untracked                                                  |

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
