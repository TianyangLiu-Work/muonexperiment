# E11 Paper Skeleton

This generated skeleton is not a draft paper. It is a maintainable bridge from the current head-to-tail evidence base to the manuscript in `paper/specgrad_activation_paper/main.tex`.

## Working Title

Head-to-Tail Interference in Long-Tailed Small-Batch Training: A Function-Drift View of Spectral Gradient Geometry

## Abstract Sketch

Long-tailed small-batch training creates long stretches of head-only updates between rare tail batches. This paper studies those updates as perturbations to held-out tail functions. We define a head-to-tail interference coefficient and show that, for matrix blocks with local tail map `J_T(D)=B_T D A_T`, spectral geometry has a smaller worst-case matched-head-gain drift bound when `nrank(G_H) > ssrank(B_T,A_T)`. Lightweight diagnostics on a synthetic boundary and long-tailed digits show that idealized spectral-gradient/polar directions reduce tail-example logit drift at matched head gain, while tail loss and margin do not automatically improve. Fixed-checkpoint and short-trajectory Muon-style diagnostics show selected-state compatibility between momentum/NS directions and the local polar mechanism. The paper is therefore a local mechanism study of spectral-gradient/polar geometry, not a broad optimizer-performance claim.

## Core Claims

| claim                                                                                                                   | status                             | evidence                                                                                                                                                                                                       | figure_or_table                                |
|:------------------------------------------------------------------------------------------------------------------------|:-----------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------|
| Head-only updates create a measurable head-to-tail function-drift problem.                                              | paper framing claim                | Long-tail digits probes explicitly separate head batch gain from held-out tail-example logit drift.                                                                                                            | one-step and 8-step tail response figures      |
| Matched-head-gain spectral/polar directions reduce tail-example logit drift in the tested diagnostics.                  | main empirical claim               | One-step squared drift ratio 0.5501 [0.5101, 0.5931]; 8-step final ratio 0.6167 [0.5744, 0.6622].                                                                                                              | long-tail one-step and forgetting figures      |
| nrank(G_H) > ssrank(B_T,A_T) is the matrix-block mechanism boundary.                                                    | theorem-backed mechanism claim     | Positive boundary ratio 0.3403; negative boundary ratio 7.208.                                                                                                                                                 | synthetic boundary figure and paper table      |
| Muon-style momentum polar is compatible with part of the ideal polar drift signal.                                      | selected-state compatibility claim | polar(M_t) squared drift ratio 0.8199 [0.6951, 0.9672]; NS(M_t) squared drift ratio 0.9116 [0.7696, 1.08].                                                                                                     | Muon-style compatibility diagnostic figure     |
| Short practical NS-Muon trajectory states show selected-state compatibility with the local polar mechanism.             | trajectory compatibility claim     | Across 120 state-step comparisons, polar(M_t) squared drift ratio 0.7292 [0.6891, 0.7717]; NS(M_t) squared drift ratio 0.8019 [0.7583, 0.848].                                                                 | practical Muon trajectory compatibility figure |
| A small practical NS-Muon-style training run has lower tail loss and higher measured tail margin in a fixed diagnostic. | practical sanity-check claim       | Final train loss ratio 0.6468 [0.604, 0.6926]; tail eval loss ratio 0.8549 [0.8319, 0.8786]; tail margin diff 1.955 [1.628, 2.281]; tail drift RMS ratio 0.7501 [0.7244, 0.7767]; tail accuracy diff 0 [0, 0]. | practical training diagnostic figure           |
| The observed layerwise mechanism is smaller matched-head-gain step size, not lower unit-direction tail sensitivity.     | mechanism clarification            | Layer 1 unit-JVP/scaled/observed squared drift ratios 1.45/0.4766/0.4767; layer 2 1.476/0.596/0.596.                                                                                                           | layerwise diagnostic figure                    |
| Lower tail-example logit drift does not automatically imply better tail loss, margin, or accuracy.                      | required caveat                    | One-step tail loss-increase diff 0.001399 [0.0002379, 0.002559]; 8-step tail loss diff -0.003538 [-0.007721, 0.0006445].                                                                                       | paper claim-boundary table                     |

## Section Plan

| section      | purpose                                                                                                | must_include                                                                                                                                                                   |
|:-------------|:-------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Introduction | Motivate tail examples being absent for many head-only updates.                                        | The objective is matched-head-gain tail function drift, not optimizer leaderboard performance.                                                                                 |
| Setup        | Define H, T, F_T, J_T, local linearization, and matched-head-gain protocol.                            | State the local assumptions and the distinction between logit drift and tail loss/accuracy.                                                                                    |
| Theory       | Derive the head-to-tail interference coefficient and the matrix-block spectral-vs-Frobenius condition. | K_T,N, I_N(T|H), B_T D A_T, nrank(G_H), ssrank(B_T,A_T), and the worst-case bound.                                                                                             |
| Experiments  | Test the mechanism with seven lightweight diagnostics plus an LR-sensitivity robustness check.         | Synthetic boundary, one-step digits, fixed Muon-style compatibility, trajectory Muon-style compatibility, practical training diagnostic, 8-step forgetting, and layerwise JVP. |
| Discussion   | State what is and is not supported.                                                                    | No broad Muon-training claim, no final-performance claim, and required real long-tail / practical-Muon follow-ups.                                                             |

## Main Figure/Table Plan

| slot     | artifact                                                                        | message                                                                                                                                  |
|:---------|:--------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------|
| Figure 1 | figures/e11_head_tail_interference/head_tail_drift_ratio.png                    | The synthetic boundary flips with nrank(G_H) versus ssrank(B_T,A_T).                                                                     |
| Figure 2 | figures/e11_long_tail_one_step/long_tail_one_step_tail_response.png             | On long-tailed digits, spectral/polar reduces held-out tail-example logit drift at matched head gain.                                    |
| Figure 3 | figures/e11_long_tail_muon_bridge/long_tail_muon_bridge.png                     | Momentum polar gives a selected-state compatibility check for Muon-style state; finite Newton-Schulz is weaker.                          |
| Figure 4 | figures/e11_long_tail_practical_muon_bridge/long_tail_practical_muon_bridge.png | Short practical NS-Muon trajectory states show squared drift ratios compatible with the local matched-head-gain mechanism.               |
| Figure 5 | figures/e11_long_tail_practical_training/long_tail_practical_training.png       | A small practical imbalanced-training sanity check has lower measured drift/loss in this fixed diagnostic, but not better tail accuracy. |
| Figure 6 | figures/e11_long_tail_forgetting/long_tail_head_only_forgetting.png             | The drift reduction persists over an 8-step head-only horizon.                                                                           |
| Figure 7 | figures/e11_long_tail_layerwise/long_tail_layerwise_drift.png                   | Layerwise results show scaled head-gain efficiency rather than lower unit-direction tail sensitivity.                                    |
| Table 1  | paper/specgrad_activation_paper/tables/head_tail_empirical_results.tex          | Quantitative paper table for drift ratios, caveats, and layerwise mechanism.                                                             |

## Current Manuscript Gap

The current evidence is strong enough for a focused theory-and-diagnostic paper about head-to-tail function drift. It is not yet enough for a broad long-tail classification benchmark paper or a full Muon optimizer theory. The next paper-critical items are a real long-tail benchmark and larger-architecture layerwise diagnostics.

## Linked Evidence

- [paper-readiness audit](e11_paper_readiness_audit.md)
- [reviewer risk audit](e11_reviewer_risk_audit.md)
- [head-to-tail interference note](e11_head_tail_interference.md)
- [long-tailed one-step diagnostic](e11_long_tail_one_step.md)
- [long-tailed Muon-style compatibility diagnostic](e11_long_tail_muon_bridge.md)
- [long-tailed practical-Muon trajectory compatibility](e11_long_tail_practical_muon_bridge.md)
- [long-tailed practical training diagnostic](e11_long_tail_practical_training.md)
- [long-tailed practical training LR sensitivity](e11_long_tail_practical_training_lr_sweep.md)
- [head-only forgetting probe](e11_long_tail_forgetting.md)
- [long-tailed layerwise diagnostic](e11_long_tail_layerwise.md)
- [artifact manifest](e11_artifact_manifest.md)
