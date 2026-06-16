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
| Claim boundary | Tail accuracy remains inconclusive; paper explicitly avoids performance claims. | keep |

## Paper-Critical Missing Evidence

| priority | missing piece | minimum acceptable gate | why it matters |
|---|---|---|---|
| P0 | Higher-quality ResNet tail checkpoint | The completed checkpoint sweep now reduces single-checkpoint risk, but best pre-update tail accuracy is only `0.068`; add checkpoints or data where the tail predictor is meaningfully useful. | Reviewers will object that lower drift may preserve a weak tail function. |
| P0 | Downstream-aware ResNet condition scatter | The rank-side proxy scatter is complete and shows `nrank(G_H)` alone is insufficient; next measure a real downstream-aware tail sensitivity proxy. | The theorem must look predictive beyond the synthetic construction. |
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
- `scripts/e11_validate_outputs.py` checks the ResNet 10-seed, rho=0.002, and
  checkpoint-sweep artifacts.
- The paper states one main claim in the abstract and conclusion: local
  matched-head-gain drift, not final accuracy.
- Every ResNet result includes seed count, target gain, checkpoint quality, and
  whether tail accuracy evidence is conclusive.
- The repo has a working test environment with both `torch` and `pytest`.
- The paper PDF builds from a clean checkout with `pdflatex`/`xelatex`.

## Next Implementation Step

The ResNet checkpoint-quality sweep has been run.

The checkpoint sweep now reduces single-checkpoint risk, and the rank-side
proxy scatter makes clear that head-gradient rank alone is not enough. The next
highest-leverage experiment is therefore a downstream-aware condition scatter
or a higher-quality tail-checkpoint run:

1. record a ResNet layer/checkpoint/seed scatter with a downstream-aware tail
   sensitivity proxy versus observed drift ratio; or
2. train/load checkpoints with substantially higher tail accuracy and rerun the
   matched-head-gain diagnostic.

The completed checkpoint-sweep artifacts are:
`results/e11_cifar100_resnet_checkpoint_sweep/*`,
`figures/e11_cifar100_resnet_checkpoint_sweep/*`, and
`discussion/e11_cifar100_resnet_checkpoint_sweep.md`.
The completed rank-side proxy scatter artifacts are:
`results/e11_cifar100_resnet_condition_proxy_scatter/*`,
`figures/e11_cifar100_resnet_condition_proxy_scatter/*`, and
`discussion/e11_cifar100_resnet_condition_proxy_scatter.md`.
