# E11 Completion Audit

This audit records the current evidence that the final-report paper package is ready for review. It is conservative: items marked complete are supported by current files or validation output, while remaining risks are explicitly scoped as future work rather than hidden claims.

## Requirement Status

| requirement | evidence | status |
|---|---|---|
| Main paper uses a normal ICLR-style structure with author information and no anonymous review block. | `paper/specgrad_activation_paper/main.tex` uses the ICLR style file, includes `Tianyang Liu`, `UCDavis`, and `tlyliu@ucdavis.edu`, and the PDF text scan rejects `Anonymous authors`, `Paper under double-blind review`, and `Under review as a conference paper`. | complete |
| Two-page final-report version exists and is exactly two pages. | `paper/specgrad_activation_paper/two_page.pdf` builds as `2 pages` according to `two_page.log`. | complete |
| The main thesis is explicit and bounded. | Abstract, Introduction, and Conclusion state the local matched-head-gain function-drift claim and explicitly reject optimizer-level performance claims. | complete |
| Major claims are tied to quantitative evidence. | Main text reports synthetic boundary ratios, one-step squared tail-example logit drift ratio, Muon-style compatibility ratios, CIs, tail-loss caveats, and layerwise scaling evidence; `discussion/e11_claim_validity_audit.md` maps claim to evidence. | complete |
| The paper distinguishes geometry separation, one-step prediction, and actual performance. | The main paper separates worst-case interference coefficients, realized drift diagnostics, Muon-style compatibility checks, and separate tail CE/margin/accuracy outcomes. | complete |
| `G`, `A`, `B`, rank quantities, and squared drift ratios are defined clearly. | Main text and Appendix A define `G_H`, `A_T`, `B_T`, `J_T`, `nrank`, `srank`, `ssrank`, and tail-example logit drift. | complete |
| Muon-style directions are not overclaimed. | Main text labels `polar(M_t)` and `NS(M_t)` as selected-state compatibility checks and uses actual head-alignment normalization for arbitrary directions. | complete |
| Reviewer-risk issues from pasted notes are tracked. | `discussion/e11_pasted_review_audit.md` maps reviewer risks to current evidence and residual risks. | complete |
| Reference metadata and citation boundaries are checked. | `discussion/e11_reference_audit.md` records source pages and conservative use boundaries for each citation; validation checks citation keys and unused BibTeX entries. | complete |
| Reproducibility details are present. | Appendix I specifies data splits, seeds, model, noisy mini-batch warmup, matched-gain scaling, Newton-Schulz approximation, and CI computation. | complete |
| Static figures and generated tables support the current claims. | `discussion/e11_main_paper_package.md` lists the main figure/table package and appendix allocation. | complete |
| PDF/build/test gates pass. | Latest validation ran `make e11-paper-pdf`, `scripts/e11_validate_outputs.py`, `pytest`, `git diff --check`, PDF text artifact scans, and two-page LaTeX box-warning scan successfully. | complete |

## Current Residual Risks

| risk | why it is not a blocker for the current final-report package | future resolution |
|---|---|---|
| The real-data evidence is controlled scikit-learn digits. | The paper frames the result as a mechanism diagnostic and not as a standard long-tailed benchmark result. | Add CIFAR-100-LT, ImageNet-LT, or iNaturalist-style diagnostics. |
| Practical Muon is not fully explained as an optimizer. | The paper only claims selected-state compatibility and explicitly separates this from optimizer-level performance. | Add tuned practical Muon/Adam/SGD baselines and trajectory diagnostics on larger tasks. |
| Lower logit drift does not imply lower CE or higher accuracy. | The paper reports the positive one-step CE tail-loss difference as a caveat next to the drift claim. | Study checkpoints with useful tail predictors and standard tail metrics. |
| The rank boundary is not yet a natural-task predictor. | The synthetic boundary is presented as a controlled mechanism check, and alignment ablation limits the interpretation. | Run held-out boundary-prediction studies across architectures and datasets. |
| Main PDF is longer than an eight-page submission once appendices are included. | The main body ends before the appendix; supporting definitions, proofs, and auxiliary diagnostics are moved to appendices. | For a strict page-limited venue submission, further compress main figures/text and move more material to supplementary. |

## Latest Validation Command

```bash
make e11-paper-pdf
python3 scripts/e11_validate_outputs.py
python3 -m pytest tests -q
git diff --check
```

Additional PDF text-layer scans checked for anonymous-review text, stale terminology, compact CI artifacts, and two-page LaTeX box warnings.
