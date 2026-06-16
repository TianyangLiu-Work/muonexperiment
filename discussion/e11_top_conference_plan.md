# E11 Top-Conference Upgrade Plan

This plan tracks what remains before the head-to-tail interference work can be
defended as an ICLR/NeurIPS/ICML-level mechanism paper. The target claim is not
that Muon is a generally better long-tail optimizer. The target claim is that a
measurable local geometry condition predicts when spectral/polar directions
reduce tail-example function drift at matched head gain.

## Current Evidence That Can Survive Review

| requirement | current evidence | status |
|---|---|---|
| Clear local question | Matched-head-gain protocol isolates head progress from tail drift. | keep |
| Theorem-backed mechanism | Sandwich-block condition `nrank(G_H) > ssrank(B_T,A_T)` gives a reversible synthetic boundary. | keep |
| Small neural diagnostic | Long-tailed digits MLP gives squared drift ratio `0.5501 [0.5101, 0.5931]`. | keep |
| ConvNet architecture check | CIFAR-100-LT ResNet18 gives squared drift ratio `0.5611 [0.5224, 0.6026]` over 10 GPU seeds. | strengthen |
| Step-scale check | CIFAR-100-LT ResNet18 at `rho=0.002 L_H` gives squared drift ratio `0.7761 [0.7585, 0.7941]`. | strengthen |
| Checkpoint sweep | CIFAR-100-LT ResNet18 warmup checkpoints 250/500/1000/2000 all have drift-ratio CI upper endpoint below 1; worst endpoint is `0.6026`. | partial |
| Rank-side proxy scatter | Across 40 ResNet seed/checkpoint points, mean gradient nuclear rank is positively correlated with log drift ratio, Pearson `0.7594 [0.6657, 0.8807]`. | caveat |
| Final-layer downstream-aware condition | Across 40 ResNet final-layer seed/checkpoint points, the weakest mean `nrank(G_H) / srank(H_T)` score is `6.566`, all points favor spectral, and the worst final-layer-only squared drift ratio is `0.2391 [0.2201, 0.2597]`. | partial |
| Tail-quality control | A tail-rich ResNet control with 300 tail-train examples per class reaches best pre-update tail accuracy `0.3739 [0.3454, 0.4024]`; worst squared drift ratio remains `0.7292 [0.6923, 0.768]`. | partial |
| Claim boundary | Tail accuracy remains inconclusive; paper explicitly avoids performance claims. | keep |

## Paper-Critical Missing Evidence

| priority | missing piece | minimum acceptable gate | why it matters |
|---|---|---|---|
| P0 | Standard long-tail benchmark protocol | The tail-rich control now addresses the weakest-tail-function objection, but it is not a tuned CIFAR-100-LT/ImageNet-LT/iNaturalist benchmark with many/medium/few reporting. | Reviewers will still object if the paper implies benchmark-level optimizer performance. |
| P0 | All-layer downstream-aware ResNet condition scatter | The final-layer condition scatter is complete for `fc.weight`, but Conv/Linear blocks still need downstream-aware tail sensitivity, unit JVP, scaled JVP, observed drift, and layer contribution. | The theorem must look predictive beyond the synthetic construction and beyond the classifier layer. |
| P1 | Long-tail imbalance sweep | CIFAR-100-LT imbalance factors or explicit tail-count settings with at least 3 seeds each; keep paired matched-gain diagnostics. | Converts one dataset split into a systematic long-tail experiment. |
| P1 | Practical optimizer bridge on CIFAR-100-LT | Sample Muon-style momentum/NS directions along CIFAR-100-LT ResNet training states, not only digits. | Bridges ideal polar directions to practical Muon without claiming final SOTA. |
| P2 | Long-horizon sanity benchmark | Tuned SGD/AdamW/Muon-style baselines on CIFAR-100-LT with many/medium/few metrics and local drift probes sampled along the trajectory. | Needed only if the paper wants any optimizer-performance discussion. |
| P2 | Larger dataset check | ImageNet-LT or iNaturalist-style matched-head-gain diagnostic on pretrained or partially trained features. | Needed for benchmark-level generality, not for the minimal mechanism paper. |

## Required Figure/Table Upgrades

1. Add a ResNet scale/checkpoint table:
   `rho`, checkpoint step, tail accuracy before update, positive-margin fraction,
   drift ratio, tail-loss diff, margin-drop diff, accuracy-drop diff.
2. Add a condition-scatter figure:
   each point is a layer/checkpoint/seed diagnostic; x-axis is the measurable
   condition score or a stated proxy; y-axis is spectral/Fro squared tail drift.
3. Add a claim-boundary table:
   drift support, tail-loss support, margin support, accuracy support, and what
   remains unsupported.
4. Keep the existing synthetic boundary figure as the theory sanity check.

## Acceptance Gates Before Calling This Top-Tier Ready

- All headline numbers are generated from CSVs and LaTeX macros, not hand typed.
- `scripts/e11_validate_outputs.py` checks the ResNet 10-seed, rho=0.002,
  checkpoint-sweep, rank-proxy, final-layer condition, and tail-quality
  artifacts.
- The paper states one main claim in the abstract and conclusion: local
  matched-head-gain drift, not final accuracy.
- Every ResNet result includes seed count, target gain, checkpoint quality, and
  whether tail accuracy evidence is conclusive.
- The repo has a working test environment with both `torch` and `pytest`.
- The paper PDF builds from a clean checkout with `pdflatex`/`xelatex`.

## Next Implementation Step

The ResNet checkpoint-quality sweep, final-layer condition scatter, and
tail-quality control have been run.

The checkpoint sweep now reduces single-checkpoint risk, the rank-side proxy
scatter makes clear that head-gradient rank alone is not enough, and the
final-layer condition scatter directly measures a downstream-aware classifier
proxy. The tail-rich control weakens the objection that the original ResNet
checkpoints only protected a weak tail function. The next highest-leverage
experiments are therefore:

1. extend the final-layer condition diagnostic to all Conv/Linear blocks with a
   downstream-aware tail sensitivity or finite-difference JVP proxy; or
2. run CIFAR-100-LT practical Muon/AdamW trajectory diagnostics with sampled
   matched-head-gain probes and final class-wise metrics.
3. move the diagnostic to a standard long-tail benchmark protocol with
   many/medium/few metrics.

The completed checkpoint-sweep artifacts are:
`results/e11_cifar100_resnet_checkpoint_sweep/*`,
`figures/e11_cifar100_resnet_checkpoint_sweep/*`, and
`discussion/e11_cifar100_resnet_checkpoint_sweep.md`.
The completed rank-side proxy scatter artifacts are:
`results/e11_cifar100_resnet_condition_proxy_scatter/*`,
`figures/e11_cifar100_resnet_condition_proxy_scatter/*`, and
`discussion/e11_cifar100_resnet_condition_proxy_scatter.md`.
The completed final-layer condition scatter artifacts are:
`results/e11_cifar100_resnet_fc_condition_scatter/*`,
`figures/e11_cifar100_resnet_fc_condition_scatter/*`, and
`discussion/e11_cifar100_resnet_fc_condition_scatter.md`.
The completed tail-quality control artifacts are:
`results/e11_cifar100_resnet_tail_quality_control/*`,
`figures/e11_cifar100_resnet_tail_quality_control/*`, and
`discussion/e11_cifar100_resnet_tail_quality_control.md`.
