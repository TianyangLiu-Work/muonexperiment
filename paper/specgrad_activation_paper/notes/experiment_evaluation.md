# Experiment Evaluation for the Head-to-Tail Interference Paper

This note tracks the empirical status of the current draft. The paper is now a
theory plus lightweight-evidence draft, not a full long-tailed optimizer
benchmark.

## Current Thesis

In long-tailed small-batch training, many updates are driven only by head
classes. Those head-only updates can perturb tail predictions before the next
tail batch appears. The paper proposes evaluating update directions by tail
functional drift under matched head gain.

The central coefficient is

```tex
\mathcal I_N(T\mid H)=K_{T,N}^2/\|g_H\|_{N,*}^2.
```

For matrix blocks with `J_T(D)=B_T D A_T`, the proposed spectral-vs-Frobenius
condition is

```tex
\nrank(G_H) > \ssrank(B_T,A_T).
```

## Completed Lightweight Evidence

| experiment | status | strongest supported statement | key caveat |
|---|---|---|---|
| Synthetic head-tail linear model | completed | The sign of `nrank(G_H) > ssrank(B_T,A_T)` matches whether spectral/polar has lower tail drift in controlled positive and negative boundary settings. | Synthetic construction controls singular values directly; it is a boundary sanity check, not a natural-data benchmark. |
| Long-tailed digits one-step diagnostic | completed | On 20 paired seeds, spectral/polar produces lower held-out tail-example logit drift than Fro/GD after matching head first-order gain. | Tail loss and margin do not improve in the same direction; the supported quantity is tail-example logit drift. |
| Long-tailed digits Muon bridge diagnostic | completed | `polar(M_t)` still has lower matched-head-gain tail drift than Fro/GD: squared drift ratio `0.8199 [0.6951, 0.9672]`. | Newton-Schulz `NS(M_t)` is weaker: `0.9116 [0.7696, 1.080]`, so this is a local bridge, not a full practical-Muon training claim. |
| Short practical-Muon trajectory bridge | completed | Across 120 sampled state-step comparisons, `polar(M_t)` and `NS(M_t)` both have lower matched-head-gain squared tail-example logit drift than Fro/GD: ratios `0.7292 [0.6891, 0.7717]` and `0.8019 [0.7583, 0.848]`. | This is still a short local diagnostic on sampled states; it does not establish final tail accuracy, long-horizon training behavior, or hyperparameter robustness. |
| Practical imbalanced-training diagnostic | completed | On the same small long-tailed digits task, NS-Muon-style training has lower final train loss, lower final tail eval loss, and lower tail output drift than Adam at the chosen lightweight hyperparameters. | Tail accuracy does not improve; this is not a tuned optimizer leaderboard and still needs larger long-tail benchmarks. |
| Practical training LR sensitivity | completed | The selected `muon_lr=0.03` is a balanced tested setting: smaller Muon lrs under-train, while `muon_lr=0.1` over-optimizes train/head loss and worsens tail loss/drift. | This reduces cherry-picking risk for the small diagnostic but is still a coarse grid on sklearn digits. |
| 8-step head-only forgetting probe | completed | Lower spectral/polar tail-example logit drift persists over a short head-only horizon and in drift area. | Final tail loss/margin confidence intervals cross zero. |
| Long-tailed layerwise diagnostic | completed | Observed layerwise tail drift matches scaled JVP drift: spectral/polar is not lower-sensitivity per unit direction, but needs a smaller step to reach the same head gain. | Only a two-layer sklearn-digits MLP; layerwise conclusions need testing in larger architectures. |

## Evidence Artifacts

- Synthetic results: `results/e11_head_tail_interference/`.
- One-step long-tail results: `results/e11_long_tail_one_step/`.
- Muon bridge results: `results/e11_long_tail_muon_bridge/`.
- Practical-Muon trajectory bridge results: `results/e11_long_tail_practical_muon_bridge/`.
- Practical training results: `results/e11_long_tail_practical_training/`.
- Practical training LR sensitivity: `results/e11_long_tail_practical_training_lr_sweep/`.
- Head-only forgetting results: `results/e11_long_tail_forgetting/`.
- Layerwise long-tail results: `results/e11_long_tail_layerwise/`.
- Paper table: `paper/specgrad_activation_paper/tables/head_tail_empirical_results.tex`.
- Discussion notes:
  - `discussion/e11_head_tail_interference.md`
  - `discussion/e11_long_tail_one_step.md`
  - `discussion/e11_long_tail_muon_bridge.md`
  - `discussion/e11_long_tail_practical_muon_bridge.md`
  - `discussion/e11_long_tail_practical_training.md`
  - `discussion/e11_long_tail_practical_training_lr_sweep.md`
  - `discussion/e11_long_tail_forgetting.md`
  - `discussion/e11_long_tail_layerwise.md`

## Claims Currently Supported

1. Under matched head gain, spectral/polar directions can reduce held-out tail
   logit drift relative to Fro/GD in the tested synthetic and small long-tail
   digits settings.
2. The relevant theory quantity is head-to-tail function drift, not final
   classification performance.
3. Layerwise evidence supports a scaled-step mechanism: spectral/polar may have
   larger unit tail sensitivity, but its larger head alignment permits a smaller
   step for the same head gain.
4. A local Muon-style bridge exists for momentum polar, and a short practical
   NS-Muon-style trajectory diagnostic now supports the same lower-drift
   direction. The caveat is that this remains local and short-horizon.
5. A small practical imbalanced-training run is consistent with the drift
   mechanism, but it supports tail loss/margin/drift separation rather than a
   broad tail accuracy claim.

## Claims Not Yet Supported

- Do not claim spectral-gradient/polar geometry is generally better for long-tailed classification.
- Do not claim full practical Muon training is explained by the current local
  fixed-checkpoint bridge.
- Do not claim lower tail-example logit drift automatically implies lower tail loss,
  better tail margin, or higher tail accuracy.
- Do not claim the sklearn-digits evidence is sufficient for CIFAR-100-LT,
  ImageNet-LT, or iNaturalist.
- Do not claim the small fixed-hyperparameter practical run is a full Muon
  benchmark.

## Remaining Experiments for a Publishable Empirical Paper

1. **Real long-tail benchmark.** Run the same matched-head-gain diagnostic on
   CIFAR-100-LT, ImageNet-LT, or iNaturalist checkpoints. Required outputs:
   matched head loss decrease, tail-example logit drift, tail loss increase, tail
   margin drop, and paired confidence intervals.
2. **Architecture-level layerwise diagnostic.** Repeat the layerwise JVP and
   scaled-drift check on a modern classifier with convolutional or transformer
   blocks. The current two-layer MLP is useful for mechanism debugging but too
   small for a broad claim.
3. **Full practical Muon training benchmark.** Extend the short bridge to real
   long-horizon practical Muon training with learning-rate schedules, checkpoint
   distributions, final tail metrics, and hyperparameter robustness.
4. **Performance separation.** If the paper wants to claim training benefit,
   add retuned long-horizon runs and report tail accuracy/loss separately from
   the local drift mechanism.

## Decision

The current draft is viable as a focused theory-and-diagnostic paper about
head-to-tail function drift. It is not yet viable as a broad empirical claim
about long-tailed classification performance or Muon's full training behavior.
