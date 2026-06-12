# E11 Research Direction Map

This generated note is the compact paper-direction map for E11. It turns the current evidence into testable claims and explicit next experiments.

## Working Research Direction

Muon should be studied as an **update-spectrum shaping optimizer**. Its robust effect is to produce flatter, higher-rank update matrices than Adam under matched update size. The open scientific question is not whether Muon is universally better, but when this induced update geometry aligns with the local loss geometry strongly enough to explain one-step decrease or optimization progress.

## Claim Map

| question                                                               | claim                                                                                         | status                 | evidence                                                                                                                                                                                                                                                                                                        | interpretation                                                                                                                    | next_test                                                                                                                     |
|:-----------------------------------------------------------------------|:----------------------------------------------------------------------------------------------|:-----------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------|
| What is Muon's optimizer-intrinsic signature?                          | Muon is best framed as an update-spectrum shaping optimizer.                                  | strong core claim      | equal-update nrUpdate Muon/Adam=2.017 CI=[1.905, 2.136]; stUpdate=4.765 CI=[4.582, 4.955]                                                                                                                                                                                                                       | The cross-task effect that survives matched update size is spectral flattening/high-rank updates, not universal loss improvement. | Add momentum-free/state-standardized optimizer variants to isolate polar spectrum from optimizer state.                       |
| Does this signature extend beyond toy matrix problems?                 | The update-spectrum signature appears in MNIST MLP and ConvNet probes too.                    | supporting evidence    | MNIST nrUpdate Muon/Adam=1.957 CI=[1.835, 2.087]; stUpdate=11.55 CI=[9.96, 13.4]. Deep MNIST nrUpdate=2.262 CI=[2.175, 2.353] with first-order ratio=0.5978 CI=[0.5633, 0.6345]. ConvNet nrUpdate=3.718 CI=[3.451, 4.006] with first-order ratio=0.677 CI=[0.6321, 0.7251]                                      | The spectral-shaping effect is not limited to synthetic matrix objectives, but neural probes make the progress caveat sharper.    | Move from short MNIST sanity checks to a modern or longer-horizon benchmark only if neural performance claims become central. |
| Does higher-rank update geometry automatically imply better progress?  | No. Local progress is conditional on task and layer geometry.                                 | strong boundary claim  | MatrixSensing first-order Muon/Adam=1.236 CI=[1.215, 1.258], but MF-with-input=0.3432 CI=[0.3194, 0.3687]; boundary map has 5 favorable and 8 unfavorable rows.                                                                                                                                                 | The same optimizer-induced spectral bias can help or hurt depending on the local geometry.                                        | Replace the descriptive boundary map with a held-out predictive rule, then test on a new task family.                         |
| What local quantity explains observed one-step decrease?               | Observed one-step decrease is well calibrated by gradient-update alignment.                   | paper-ready diagnostic | Spearman(delta_loss, <G,D>)=0.9803 CI=[0.9787, 0.9817], within-factor-2=0.9207                                                                                                                                                                                                                                  | The paper can use <G,D> as the local bridge between update geometry and loss change.                                              | Keep this claim local; validate separately if longer-horizon progress is discussed.                                           |
| Why can a flat/polar spectrum help in some regimes and hurt in others? | The sign flips with the norm budget, even though polar direction reliably raises update rank. | mechanistic evidence   | flat_polar/GD under Frobenius budget=0.6071 CI=[0.5846, 0.6304], but under operator budget=1.689 CI=[1.614, 1.768]; stateless PolarMuon/GD nrUpdate=2.779 CI=[2.668, 2.895] while Frobenius-matched update_grad_inner=0.5032 CI=[0.4857, 0.5213] and short-trajectory total_decrease=0.4656 CI=[0.3686, 0.5882] | Flat spectral allocation is not intrinsically better; it matches operator-norm-like geometry better than Frobenius geometry.      | Formalize this as the main theorem/proposition and add trajectory-level optimizer variants.                                   |
| Do current features predict where Muon wins?                           | Not yet at a publishable level.                                                               | open gap               | best leave-setting-out balanced accuracy=0.6064 from state_plus_update_spectrum when degenerate settings are skipped; chance-filled best=0.5866 CI=[0.5, 0.7446]                                                                                                                                                | The current boundary is descriptive, not a reliable predictive theory for unseen settings.                                        | Use the new benchmark as a held-out test instead of selecting features after seeing all task families.                        |
| Does Muon's geometry guarantee neural performance gains?               | No; MNIST hidden=128 is Adam-favorable in first-order progress.                               | negative control       | MNIST hidden=128 first-order Muon/Adam=0.7546 CI=[0.7387, 0.7708]                                                                                                                                                                                                                                               | The neural evidence supports geometry shaping but warns against claiming optimizer superiority.                                   | Separate local update geometry, final training loss, and classification error in any neural section.                          |

## Paper-Level Hypothesis

The current best hypothesis is:

> Muon imposes a flat/polar update-spectrum bias. This bias is a robust optimizer-level signature, but its optimization benefit is conditional: it helps when the local gradient/norm/task geometry rewards spectrally spread updates, and it hurts or becomes neutral otherwise.

## Current Paper Shape

| role | content |
|:--|:--|
| Core positive result | Muon reliably induces higher-rank, flatter update spectra than Adam under matched update size. |
| Mechanistic bridge | One-step loss decrease is well captured by `<G,D>`. |
| Boundary result | Muon's first-order advantage changes sign across task family, width/layer control, and norm budget. |
| Negative control | Higher rank does not imply lower loss, better final performance, or general stability. |
| Main gap | The current boundary map is descriptive; a stronger paper needs a held-out predictive boundary test and, only for broader neural claims, modern/longer-horizon neural benchmarks. |

## Immediate Next Experiments

1. Add a modern or longer-horizon neural benchmark with per-layer update spectra, first-order alignment, final loss, and classification error if neural performance claims become central.
2. Add optimizer variants that separate polar spectrum shaping from momentum/state details.
3. Turn the descriptive boundary map into a pre-specified predictor and test it on a genuinely held-out task or architecture.

## Sources

- [research synthesis](e11_research_synthesis.md)
- [paper-readiness audit](e11_paper_readiness_audit.md)
- [mechanism boundary map](e11_mechanism_boundary.md)
- [boundary predictor baseline](e11_boundary_predictor.md)
- [MNIST MLP probe](e11_mnist_mlp_probe.md)
- [Deep MNIST MLP probe](e11_deep_mnist_mlp_probe.md)
- [stateless direction ablation](e11_stateless_direction_ablation.md)
- [stateless optimizer trajectory ablation](e11_stateless_optimizer_trajectory.md)
