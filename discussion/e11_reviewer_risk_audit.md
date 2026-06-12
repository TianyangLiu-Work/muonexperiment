# E11 Reviewer Risk Audit

This generated audit lists the likely reviewer objections and the current evidence-backed response. It is meant to keep the manuscript focused on defensible claims.

## Risk Table

| reviewer_objection                                                            | risk_level                                      | current_evidence                                                                                                                                                                                                                                                                              | safe_response                                                                                               | remaining_work                                                                                                      |
|:------------------------------------------------------------------------------|:------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------|
| The update-rank result is trivial because ExactMuon is polar by construction. | medium                                          | Equal-update nrUpdate=2.017 CI=[1.905, 2.136], stUpdate=4.765 CI=[4.582, 4.955]; stateless PolarMuon/GD nrUpdate=2.779 CI=[2.668, 2.895].                                                                                                                                                     | Frame this as the optimizer's controlled spectral bias, not as a surprising performance result.             | Add non-exact Muon variants or approximate polar iterations if the paper targets broader optimizer implementations. |
| Higher-rank updates do not explain optimization progress.                     | low for current claim, high for stronger claims | Stateless PolarMuon/GD update_grad_inner=0.5032 CI=[0.4857, 0.5213]; trajectory total_decrease=0.4656 CI=[0.3686, 0.5882]; Deep MNIST first-order ratio=0.5978 CI=[0.5633, 0.6345] despite nrUpdate=2.262 CI=[2.175, 2.353].                                                                  | Agree; the paper's claim is boundary-dependent update-spectrum shaping, not rank-implies-progress.          | Keep high-rank progress claims out of the abstract and theorem statements.                                          |
| The theory is only local and does not prove optimizer superiority.            | low if framed correctly                         | First-order calibration Spearman=0.9803 CI=[0.9787, 0.9817], Frobenius flat/GD=0.6071 CI=[0.5846, 0.6304], operator flat/GD=1.689 CI=[1.614, 1.768].                                                                                                                                          | State the theorem as a local constrained linearized result and use it only to explain mechanism probes.     | A convergence theorem would be a different paper and is not currently supported.                                    |
| The boundary map is descriptive but not predictive.                           | high for predictive-theory claims               | Best leave-setting-out predictor balanced accuracy=0.6039 from state_plus_update_spectrum when degenerate settings are skipped; chance-filled best=0.5866 CI=[0.5, 0.7446].                                                                                                                   | Explicitly present the predictor as a failed/weak baseline and call the boundary map descriptive.           | Pre-register a predictor and test it on a genuinely new balanced held-out task or architecture.                     |
| The neural evidence is too small or not representative.                       | medium                                          | Deep MNIST MLP preserves update-spectrum shaping with nrUpdate=2.262 CI=[2.175, 2.353], but first-order progress is Adam-favorable: 0.5978 CI=[0.5633, 0.6345]. MNIST ConvNet also preserves nrUpdate=3.718 CI=[3.451, 4.006] while first-order is Adam-favorable: 0.677 CI=[0.6321, 0.7251]. | Use neural experiments as sanity/negative controls, not as broad performance benchmarks.                    | Add modern or longer-horizon neural benchmarks only if the paper claims broad neural relevance.                     |
| Local geometry may not predict longer-horizon behavior.                       | medium                                          | Continuation-LR sweep switched/own final-loss ratio=2.304 CI=[1.573, 3.035].                                                                                                                                                                                                                  | Use continuation/switch probes as negative controls against overclaiming.                                   | Run longer retuned training only if final-performance claims become central.                                        |
| There are too many artifacts and the claim may be hard to follow.             | medium                                          | Paper skeleton, evidence index, theorem bridge, and reviewer audit now provide a claim hierarchy.                                                                                                                                                                                             | Keep the main paper to three claims: update-spectrum signature, local calibration, norm/boundary mechanism. | Before drafting, select 3 main figures and move most ablations to appendix.                                         |

## Claim Decisions

| claim                                                    | decision                   | reason                                                                                          |
|:---------------------------------------------------------|:---------------------------|:------------------------------------------------------------------------------------------------|
| Muon is an update-spectrum shaping optimizer.            | main-paper claim           | Direct, robust, matched-update evidence supports it.                                            |
| Muon's polar direction is locally operator-norm optimal. | main-paper mechanism claim | Narrow theorem plus spectral allocation probe support this exactly.                             |
| Muon is generally better than Adam.                      | do not claim               | Boundary, deep MNIST, stateless trajectory, and continuation evidence contradict a broad claim. |
| Local features predict when Muon wins.                   | future-work claim only     | Current leave-setting-out predictor is weak and imbalanced.                                     |
| The paper gives a convergence theory.                    | do not claim               | Current theorem is local and first-order.                                                       |

## Recommended Manuscript Discipline

1. Put update-spectrum shaping, first-order calibration, and norm-geometry boundary in the main paper.
2. Put neural probes, stateless trajectories, switch controls, and predictor failure modes in supporting/appendix sections.
3. Do not state or imply that higher update rank generally improves progress.
4. Do not present the current boundary predictor as a predictive theory.

## Sources

- [paper skeleton](e11_paper_skeleton.md)
- [mechanism theorem bridge](e11_mechanism_theorem_bridge.md)
- [boundary predictor audit](e11_boundary_predictor_audit.md)
- [Deep MNIST MLP probe](e11_deep_mnist_mlp_probe.md)
- [MNIST ConvNet probe](e11_mnist_conv_probe.md)
- [stateless optimizer trajectory ablation](e11_stateless_optimizer_trajectory.md)
