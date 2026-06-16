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
| All-layer downstream-aware ResNet condition scatter | The tail-rich ResNet all-layer finite-difference JVP diagnostic covers 21 Conv/Linear weights and 210 paired layer/seed points; observed squared drift ratio is `0.2011 [0.1845, 0.2192]`, scaled-JVP ratio is `0.065 [0.06008, 0.07031]`, and all per-layer observed CI upper endpoints are below 1. | partial |
| Held-out checkpoint-transfer JVP benchmark | Across tail-rich 2000/5000/10000-step ResNet checkpoints, scaled-JVP below-one threshold accuracy is `1`, but held-out layer-risk Spearman is `-0.3203 [-0.3562, -0.2845]`. | boundary/negative |
| Standard long-tail reporting baseline | CIFAR-100-LT IF=100 AdamW ResNet18 over 10 seeds gives many/medium/few balanced accuracy `0.3665 [0.3489, 0.3841]`, `0.1036 [0.09138, 0.1158]`, and `0.0129 [0.009351, 0.01645]`. | reporting baseline |
| Augmented recipe benchmark pilot | CIFAR-100-LT IF=100 ResNet18 with random crop/flip over 5 seeds gives SGD-momentum all/few balanced accuracy `0.4105 [0.4044, 0.4166]` and `0.1047 [0.09389, 0.1156]`; few diff vs AdamW-aug is `0.0194 [0.005581, 0.03322]`, while class-balanced AdamW is worse on few classes. | pilot |
| Practical Muon bridge on CIFAR-100-LT ResNet18 | From the same tail-rich checkpoint, AdamW-sampled states give NS(M_t) squared drift ratio `0.8628 [0.8154, 0.913]`; NS-Muon-sampled states give `0.7247 [0.676, 0.777]` over 30 comparisons per state source. | partial |
| Claim boundary | Tail accuracy remains inconclusive; paper explicitly avoids performance claims. | keep |

## Paper-Critical Missing Evidence

| priority | missing piece | minimum acceptable gate | why it matters |
|---|---|---|---|
| P0 | Standard long-tail benchmark protocol | A first CIFAR-100-LT IF=100 ResNet18 many/medium/few reporting baseline and a 5-seed augmented AdamW/class-balanced/SGD pilot are complete. The remaining gate is a wider tuned multi-optimizer benchmark, including Muon final-performance, on CIFAR-100-LT plus ImageNet-LT/iNaturalist-style protocols if the paper wants benchmark-level performance claims. | Reviewers will still object if the paper implies optimizer-performance superiority from the current pilot. |
| P0 | Predictive downstream-aware condition benchmark | A first held-out checkpoint-transfer benchmark is complete and negative for layer-risk ranking: scaled-JVP preserves the below-one threshold direction but does not predict cross-checkpoint layer ordering. The next gate is a stronger downstream-aware condition score plus held-out architecture/dataset splits. | The theorem must look predictive beyond the synthetic construction and beyond one ResNet checkpoint family. |
| P1 | Long-tail imbalance sweep | CIFAR-100-LT imbalance factors or explicit tail-count settings with at least 3 seeds each; keep paired matched-gain diagnostics. | Converts one dataset split into a systematic long-tail experiment. |
| P1 | Practical optimizer bridge on CIFAR-100-LT | Completed as a local trajectory-state bridge; the remaining gate is final class-wise metrics under tuned practical training if the paper wants optimizer-performance discussion. | Bridges ideal polar directions to practical Muon without claiming final SOTA. |
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
  checkpoint-sweep, rank-proxy, final-layer condition, tail-quality,
  all-layer JVP tail-quality, checkpoint-transfer JVP, standard
  many/medium/few reporting, augmented recipe-pilot, and practical Muon
  trajectory-bridge artifacts.
- The paper states one main claim in the abstract and conclusion: local
  matched-head-gain drift, not final accuracy.
- Every ResNet result includes seed count, target gain, checkpoint quality, and
  whether tail accuracy evidence is conclusive.
- The repo has a working test environment with both `torch` and `pytest`.
- The paper PDF builds from a clean checkout with `pdflatex`/`xelatex`.

## Next Implementation Step

The ResNet checkpoint-quality sweep, final-layer condition scatter,
tail-quality control, all-layer JVP tail-quality diagnostic, standard
CIFAR-100-LT reporting baseline, augmented recipe benchmark pilot, and
practical Muon/AdamW trajectory bridge have been run. The first
checkpoint-transfer JVP benchmark has also been run and is a boundary result
rather than a positive predictor.

The checkpoint sweep now reduces single-checkpoint risk, the rank-side proxy
scatter makes clear that head-gradient rank alone is not enough, and the
final-layer condition scatter directly measures a downstream-aware classifier
proxy. The tail-rich control weakens the objection that the original ResNet
checkpoints only protected a weak tail function. The all-layer JVP diagnostic
adds finite-difference tail sensitivity, scaled-JVP, observed drift, and layer
contribution for every Conv/Linear matrix weight at the tail-rich checkpoint.
The practical bridge adds sampled AdamW and NS-Muon matrix-weight trajectory
states at the same tail-rich ResNet scale, without turning the claim into final
optimizer performance. The checkpoint-transfer benchmark shows that the
current scaled-JVP readout transfers the below-one direction across tail-rich
checkpoints but not the layer-risk ranking, which is exactly the sort of
negative evidence a top-tier version should surface. The standard reporting
baseline gives a benchmark-style classification readout but is still AdamW-only
and untuned. The recipe pilot adds augmentation, class-balanced AdamW, and
SGD-momentum; it improves benchmark context and shows recipe sensitivity, but
it is still not a complete optimizer benchmark.
The next highest-leverage experiments are therefore:

1. improve the downstream-aware condition score and repeat the pre-registered
   predictive benchmark across held-out architectures or datasets;
2. extend the CIFAR-100-LT practical Muon/AdamW trajectory bridge to final
   class-wise metrics only if optimizer-performance discussion becomes central;
3. extend the augmented CIFAR-100-LT recipe pilot into a tuned benchmark with
   more seeds, schedules, class-balanced samplers/losses, and Muon/AdamW final
   comparisons.

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
The completed all-layer JVP tail-quality artifacts are:
`results/e11_cifar100_resnet_layer_jvp_tail_quality/*`,
`figures/e11_cifar100_resnet_layer_jvp_tail_quality/*`, and
`discussion/e11_cifar100_resnet_layer_jvp_tail_quality.md`.
The completed all-layer JVP checkpoint-transfer artifacts are:
`results/e11_cifar100_resnet_layer_jvp_checkpoint_prediction/*`,
`figures/e11_cifar100_resnet_layer_jvp_checkpoint_prediction/*`, and
`discussion/e11_cifar100_resnet_layer_jvp_checkpoint_prediction.md`.
The completed standard CIFAR-100-LT ResNet18 reporting artifacts are:
`results/e11_cifar100_resnet_lt_standard_eval/*`,
`figures/e11_cifar100_resnet_lt_standard_eval/*`, and
`discussion/e11_cifar100_resnet_lt_standard_eval.md`.
The completed augmented CIFAR-100-LT ResNet18 recipe pilot artifacts are:
`results/e11_cifar100_resnet_lt_recipe_benchmark/*`,
`figures/e11_cifar100_resnet_lt_recipe_benchmark/*`, and
`discussion/e11_cifar100_resnet_lt_recipe_benchmark.md`.
The completed ResNet practical Muon bridge artifacts are:
`results/e11_cifar100_resnet_practical_muon_bridge/*`,
`figures/e11_cifar100_resnet_practical_muon_bridge/*`, and
`discussion/e11_cifar100_resnet_practical_muon_bridge.md`.
