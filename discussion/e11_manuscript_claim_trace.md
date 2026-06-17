# E11 Manuscript Claim Trace

This generated trace links the paper-level top-conference claim decisions to explicit `main.tex` anchors. It does not add empirical evidence; it checks that the manuscript presents supportable claims as scoped claims, registered claims as pending tests, and blocked claims only as limitations or protocols.

## Claim Trace

| claim_id                                    | current_decision                        | main_tex_location                                                              | trace_status                                  |   required_anchor_count |   found_anchor_count | anchor_lines                                                                                                                                                                                                                                    | blocked_phrase_audit   |
|:--------------------------------------------|:----------------------------------------|:-------------------------------------------------------------------------------|:----------------------------------------------|------------------------:|---------------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------|
| TCD-1-main-mechanism-theorem                | supportable_main_with_assumptions       | abstract, introduction, theorem, and conclusion                                | present_as_supportable_scoped_claim           |                       4 |                    4 | Tail drift under matched head gain@150; \nrank(G_H)>\ssrank(B_T,A_T)@65; not a global optimizer theorem@309; Worst-case and realized coefficients@164                                                                                           | pass                   |
| TCD-2-natural-drift-diagnostic              | supportable_diagnostic_only             | abstract, experiment overview, digits/ResNet results, and appendix diagnostics | present_as_diagnostic_only                    |                       4 |                    4 | tail-example logit drift@65; tail loss, margin, and accuracy are reported separately@88; lower logit drift does not imply lower cross-entropy loss@253; supportable_diagnostic_only@decision                                                    | pass                   |
| TCD-3-predictive-condition-generalization   | registered_not_ready_wait_for_v5_finals | limitations and appendix condition-score audit                                 | present_as_registered_not_ready               |                       4 |                    4 | v5 validation-only split freezes a transport-normalized candidate@991; this remains validation evidence@991; unspent ResNeXt50-32x4d architecture final and CIFAR-10 cross-partition data final must both pass@991; registered pending test@311 | pass                   |
| TCD-4-natural-counterexample-or-finite-null | blocked_partial_family                  | limitations                                                                    | present_as_blocked_partial_family             |                       4 |                    4 | observed \(20/26\) settings@311; \texttt{raw\_worse\_rows=0}@311; block finite-null wording@311; block a natural counterexample claim@311                                                                                                       | pass                   |
| TCD-5-optimizer-performance-benchmark       | blocked_protocol_pending                | abstract, experiments, limitations, and appendix benchmark pilots              | present_as_blocked_protocol_context           |                       4 |                    4 | not an optimizer-level performance claim@65; negative benchmark boundary@311; not evidence for an optimizer-performance advantage@311; not a complete long-tailed classification optimizer benchmark@319                                        | pass                   |
| TCD-6-artifact-reproducibility              | supportable_with_toolchain_caveat       | limitations and submission reproducibility audit                               | present_as_toolchain_caveated_reproducibility |                       3 |                    3 | server validation and rendered-PDF fallback@311; preferred LaTeX clean-checkout reproduction remains an explicit gate@311; supportable_with_toolchain_caveat@decision                                                                           | pass                   |

## Blocked Phrase Audit

| claim_id                                    | blocked_phrase                                                                              | present_in_main_tex   | audit_status   |
|:--------------------------------------------|:--------------------------------------------------------------------------------------------|:----------------------|:---------------|
| TCD-1-main-mechanism-theorem                | global convergence or optimizer superiority                                                 | no                    | pass           |
| TCD-2-natural-drift-diagnostic              | tail accuracy necessarily improves                                                          | no                    | pass           |
| TCD-3-predictive-condition-generalization   | the v5 score predicts unseen real-task residual risk                                        | no                    | pass           |
| TCD-4-natural-counterexample-or-finite-null | fresh natural primary counterexample                                                        | no                    | pass           |
| TCD-4-natural-counterexample-or-finite-null | finite null over the 26-setting phase1 family                                               | no                    | pass           |
| TCD-5-optimizer-performance-benchmark       | Muon or spectral training is competitive on long-tail benchmarks                            | no                    | pass           |
| TCD-6-artifact-reproducibility              | preferred pdflatex/bibtex/xelatex clean-checkout reproducibility is complete on this server | no                    | pass           |

## Operating Rule

Every `supportable` decision must have a local scoped manuscript anchor. Every `registered_not_ready` or `blocked` decision must have pending, incomplete, negative-boundary, or caveat wording in `main.tex`, and the blocked positive wording from the top-conference claim decision matrix must be absent.

Artifacts:
- [claim_trace.csv](../results/e11_manuscript_claim_trace/claim_trace.csv)
- [blocked_phrase_audit.csv](../results/e11_manuscript_claim_trace/blocked_phrase_audit.csv)
- [config.json](../results/e11_manuscript_claim_trace/config.json)
