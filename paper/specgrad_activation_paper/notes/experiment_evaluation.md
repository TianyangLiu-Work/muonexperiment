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
| CIFAR-100-LT ResNet18 one-step diagnostic | completed | On 10 GPU-rerun seeds, a CIFAR-stem ResNet18 gives lower matched-head-gain squared tail-example logit drift: ratio `0.5611 [0.5224, 0.6026]`, with spectral lower in all seeds. A smaller-head-gain check at `rho=0.002 L_H` gives ratio `0.7761 [0.7585, 0.7941]`. A 250/500/1000/2000-step checkpoint sweep keeps the worst drift-ratio CI endpoint at `0.6026`. | This is still a local one-step diagnostic with fixed BatchNorm state and Conv/Linear matrix-weight interventions; best pre-update tail accuracy in the sweep is only `0.068`, so it is not a full long-tail optimizer benchmark or high-quality tail-predictor preservation result. |
| CIFAR-100-LT ResNet18 rank-proxy scatter | completed | Across 40 seed/checkpoint points, mean matrix-gradient nuclear rank has positive correlation with log squared drift ratio, Pearson `0.7594 [0.6657, 0.8807]`. | This is a useful caveat, not a positive predictor: it shows `nrank(G_H)` alone is not a replacement for downstream-aware tail sensitivity. |
| CIFAR-100-LT ResNet18 final-layer condition scatter | completed | Across 40 seed/checkpoint points for `fc.weight`, weakest mean `nrank(G_H) / srank(H_T)` is `6.566`, all points favor spectral, and the worst final-layer-only squared drift ratio is `0.2391 [0.2201, 0.2597]`. | This is closer to the theorem than the rank-only proxy, but it is only the classifier layer. |
| CIFAR-100 ResNet18 tail-quality control | completed | With 300 tail-train examples per class, best pre-update tail accuracy rises to `0.3739 [0.3454, 0.4024]`, while the worst squared drift ratio remains below 1 at `0.7292 [0.6923, 0.768]`. | This weakens the weak-tail-predictor objection, but it is a tail-rich local control rather than a standard long-tail benchmark or practical optimizer result. |
| CIFAR-100-LT ResNet18 all-layer JVP tail-quality diagnostic | completed | At the tail-rich 5000-step checkpoint, 21 Conv/Linear weights and 210 paired layer/seed points give observed squared drift ratio `0.2011 [0.1845, 0.2192]` and scaled-JVP ratio `0.065 [0.06008, 0.07031]`; every per-layer observed CI upper endpoint is below 1. | This closes the classifier-only mechanism gap locally, but it is still a finite-difference one-step diagnostic rather than a standard long-tail optimizer benchmark or held-out predictor. |
| CIFAR-100-LT ResNet18 all-layer JVP checkpoint transfer | completed | Across tail-rich 2000/5000/10000-step checkpoints, scaled-JVP below-one threshold accuracy is `1`, while held-out layer-risk Spearman is `-0.3203 [-0.3562, -0.2845]`. | This is useful negative evidence: the current local score transfers the below-one direction but is not a positive cross-checkpoint layer-risk predictor. |
| CIFAR-100-LT ResNet18 standard many/medium/few reporting | completed | IF=100 AdamW ResNet18 gives many/medium/few balanced accuracy `0.3665 [0.3489, 0.3841]`, `0.1036 [0.09138, 0.1158]`, and `0.0129 [0.009351, 0.01645]` over 10 seeds. | This is a standard reporting baseline, not a tuned optimizer benchmark, augmentation study, or Muon comparison. |
| CIFAR-100-LT ResNet18 augmented recipe benchmark pilot | completed | With random crop/flip over 5 seeds, SGD-momentum reaches all/few balanced accuracy `0.4105 [0.4044, 0.4166]` and `0.1047 [0.09389, 0.1156]`; few diff vs augmented AdamW is `0.0194 [0.005581, 0.03322]`. | This is useful benchmark context, but it is only a three-recipe pilot and still lacks a tuned Muon final-performance comparison. |
| Long-tailed digits Muon bridge diagnostic | completed | `polar(M_t)` still has lower matched-head-gain tail drift than Fro/GD: squared drift ratio `0.8199 [0.6951, 0.9672]`. | Newton-Schulz `NS(M_t)` is weaker: `0.9116 [0.7696, 1.080]`, so this is a local bridge, not a full practical-Muon training claim. |
| Short practical-Muon trajectory bridge | completed | Across 120 sampled state-step comparisons, `polar(M_t)` and `NS(M_t)` both have lower matched-head-gain squared tail-example logit drift than Fro/GD: ratios `0.7292 [0.6891, 0.7717]` and `0.8019 [0.7583, 0.848]`. | This is still a short local diagnostic on sampled states; it does not establish final tail accuracy, long-horizon training behavior, or hyperparameter robustness. |
| Practical imbalanced-training diagnostic | completed | On the same small long-tailed digits task, NS-Muon-style training has lower final train loss, lower final tail eval loss, and lower tail output drift than Adam at the chosen lightweight hyperparameters. | Tail accuracy does not improve; this is not a tuned optimizer leaderboard and still needs larger long-tail benchmarks. |
| Practical training LR sensitivity | completed | The selected `muon_lr=0.03` is a balanced tested setting: smaller Muon lrs under-train, while `muon_lr=0.1` over-optimizes train/head loss and worsens tail loss/drift. | This reduces cherry-picking risk for the small diagnostic but is still a coarse grid on sklearn digits. |
| 8-step head-only forgetting probe | completed | Lower spectral/polar tail-example logit drift persists over a short head-only horizon and in drift area. | Final tail loss/margin confidence intervals cross zero. |
| Long-tailed layerwise diagnostic | completed | Observed layerwise tail drift matches scaled JVP drift: spectral/polar is not lower-sensitivity per unit direction, but needs a smaller step to reach the same head gain. | Only a two-layer sklearn-digits MLP; layerwise conclusions need testing in larger architectures. |

## Evidence Artifacts

- Synthetic results: `results/e11_head_tail_interference/`.
- One-step long-tail results: `results/e11_long_tail_one_step/`.
- CIFAR-100-LT MLP results: `results/e11_cifar100_lt_one_step/`.
- CIFAR-100-LT ResNet18 results: `results/e11_cifar100_resnet_one_step/`.
- CIFAR-100-LT ResNet18 smaller-head-gain results: `results/e11_cifar100_resnet_one_step_rho002/`.
- CIFAR-100-LT ResNet18 checkpoint sweep results: `results/e11_cifar100_resnet_checkpoint_sweep/`.
- CIFAR-100-LT ResNet18 rank-proxy scatter results: `results/e11_cifar100_resnet_condition_proxy_scatter/`.
- CIFAR-100-LT ResNet18 final-layer condition results: `results/e11_cifar100_resnet_fc_condition_scatter/`.
- CIFAR-100 ResNet18 tail-quality control results: `results/e11_cifar100_resnet_tail_quality_control/`.
- CIFAR-100-LT ResNet18 all-layer JVP tail-quality results: `results/e11_cifar100_resnet_layer_jvp_tail_quality/`.
- CIFAR-100-LT ResNet18 all-layer JVP checkpoint-transfer results: `results/e11_cifar100_resnet_layer_jvp_checkpoint_prediction/`.
- CIFAR-100-LT ResNet18 standard reporting results: `results/e11_cifar100_resnet_lt_standard_eval/`.
- CIFAR-100-LT ResNet18 augmented recipe benchmark results: `results/e11_cifar100_resnet_lt_recipe_benchmark/`.
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
  - `discussion/e11_cifar100_resnet_layer_jvp_checkpoint_prediction.md`
  - `discussion/e11_cifar100_resnet_lt_standard_eval.md`
  - `discussion/e11_cifar100_resnet_lt_recipe_benchmark.md`
  - `discussion/e11_long_tail_muon_bridge.md`
  - `discussion/e11_long_tail_practical_muon_bridge.md`
  - `discussion/e11_long_tail_practical_training.md`
  - `discussion/e11_long_tail_practical_training_lr_sweep.md`
  - `discussion/e11_long_tail_forgetting.md`
  - `discussion/e11_long_tail_layerwise.md`

## Claims Currently Supported

1. Under matched head gain, spectral/polar directions can reduce held-out tail
   logit drift relative to Fro/GD in the tested synthetic, small long-tail
   digits, and CIFAR-100-LT ResNet18 settings.
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
- Do not claim the new CIFAR-100-LT ResNet18 one-step diagnostic or standard
  reporting baseline or augmented recipe pilot is sufficient for ImageNet-LT,
  iNaturalist, or a tuned long-tail optimizer benchmark claim.
- Do not claim the small fixed-hyperparameter practical run is a full Muon
  benchmark.

## Remaining Experiments for a Publishable Empirical Paper

1. **Standard real long-tail benchmark.** A CIFAR-100-LT IF=100 ResNet18
   many/medium/few reporting baseline and a 5-seed augmented
   AdamW/class-balanced/SGD pilot now exist, but the paper still needs a wider
   tuning grid, class-balanced samplers, tuned Muon-style final-performance
   baselines, and ImageNet-LT/iNaturalist-style protocols if it wants
   benchmark-level empirical claims. Required outputs: matched head loss
   decrease, tail-example logit drift, tail loss increase, tail margin drop,
   accuracy, and paired confidence intervals.
2. **Predictive all-layer condition benchmark.** The ResNet all-layer JVP
   diagnostic now covers Conv/Linear weights with unit JVP, scaled JVP,
   observed drift, and layer contribution. The next version should pre-register
   downstream-aware condition scores and test held-out checkpoints,
   architectures, or datasets.
3. **Architecture-level layerwise diagnostic.** Repeat the layerwise JVP and
   scaled-drift check on additional modern classifiers with convolutional or
   transformer blocks. The current ResNet all-layer diagnostic is useful
   mechanism evidence but not yet a general predictor.
4. **Full practical Muon training benchmark.** Extend the short bridge to real
   long-horizon practical Muon training with learning-rate schedules, checkpoint
   distributions, final tail metrics, and hyperparameter robustness.
5. **Performance separation.** If the paper wants to claim training benefit,
   add retuned long-horizon runs and report tail accuracy/loss separately from
   the local drift mechanism.

## Decision

The current draft is viable as a focused theory-and-diagnostic paper about
head-to-tail function drift, now with a CIFAR-100-LT ResNet18 architecture
robustness check, a standard CIFAR-100-LT reporting baseline, and an augmented
recipe benchmark pilot. It is not yet viable as a broad empirical claim about
long-tailed classification performance or Muon's full training behavior.
