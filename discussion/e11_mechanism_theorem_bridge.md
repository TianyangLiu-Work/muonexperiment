# E11 Mechanism Theorem Bridge

This generated note links the local theorem to the empirical controls. Its purpose is to make the paper's causal language precise: which statements follow from the theorem and probes, and which statements remain unsupported.

## Theorem-To-Evidence Map

| theory_piece                                                                            | mathematical_content                                                                      | evidence                                                                                                                     | supports                                                                                             | does_not_support                                                                   |
|:----------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|
| First-order local analysis is meaningful.                                               | `delta_loss` should track `<G, D>` for small positive descent updates `D=W-W^+`.          | Spearman(delta_loss, <G, D>)=0.9803 CI=[0.9787, 0.9817], within-factor-2=0.9207.                                             | Using gradient-update alignment as the local progress diagnostic.                                    | Any claim about final loss, convergence, or generalization by itself.              |
| Frobenius-constrained linear objective favors gradient-shaped spectra.                  | `D_F^* = rG/||G||_F` with singular values proportional to those of `G`.                   | flat_polar/GD first-order ratio under Frobenius budget=0.6071 CI=[0.5846, 0.6304].                                           | A flat/polar update can be locally worse than GD-spectrum allocation under a Frobenius budget.       | That Muon is bad in every natural optimizer trajectory.                            |
| Operator-norm-constrained linear objective favors flat/polar spectra.                   | `D_op^* = eta U V^T` with equal active singular values.                                   | flat_polar/GD first-order ratio under operator budget=1.689 CI=[1.614, 1.768].                                               | The precise sense in which Muon's polar-like update matches an operator-norm local optimum.          | That higher rank alone is the right explanatory variable.                          |
| Polar direction alone creates the high-rank update signature.                           | The polar factor has flattened active singular values independent of Adam state.          | Stateless PolarMuon/GD nrUpdate=2.779 CI=[2.668, 2.895], but Frobenius-matched update_grad_inner=0.5032 CI=[0.4857, 0.5213]. | Update-spectrum shaping is a direction-level mechanism, not merely a side effect of optimizer state. | That polar direction alone improves Frobenius-matched local progress.              |
| The one-step mechanism persists across short stateless trajectories.                    | Repeated polar directions retain high-rank updates under matched per-step Frobenius size. | Stateless trajectory PolarMuon/GD mean_nrUpdate=2.704 CI=[2.236, 3.269], total_decrease=0.4656 CI=[0.3686, 0.5882].          | The mechanism is not just a single-step artifact.                                                    | That stateless polar training is a better optimizer under Frobenius-matched steps. |
| Neural negative control: high-rank updates can coexist with worse first-order progress. | The theorem predicts a boundary, not a universal benefit of rank spreading.               | Deep MNIST MLP nrUpdate Muon/Adam=2.262 CI=[2.175, 2.353], but first-order ratio=0.5978 CI=[0.5633, 0.6345].                 | The boundary framing is necessary even on neural probes.                                             | A broad neural-performance claim for Muon.                                         |

## Safe Claim Language

| safe_wording                                                                     | reason                                                                                                           |
|:---------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------|
| Muon is an update-spectrum shaping method.                                       | The update-rank effect is direct, replicated, and survives matched-update controls.                              |
| Muon's polar-like direction matches the local operator-norm constrained optimum. | This is exactly the local proposition and is supported by the operator-budget sign flip.                         |
| The optimization benefit is boundary-dependent.                                  | Frobenius, stateless trajectory, MF, and deep MNIST controls all show high-rank updates without better progress. |
| The current result is local-geometry evidence, not a convergence theorem.        | All core mechanism probes are one-step or short-horizon controls.                                                |

## Language To Avoid

| unsafe_wording                                                | why_wrong                                                                                 |
|:--------------------------------------------------------------|:------------------------------------------------------------------------------------------|
| Muon is better because it has higher rank updates.            | Deep MNIST and stateless controls show higher rank with worse first-order progress.       |
| The theory proves Muon should beat Adam.                      | The theorem compares constrained linearized directions, not full optimizer trajectories.  |
| Operator-norm optimality explains all observed Muon behavior. | Natural optimizers also differ by state, layer allocation, singular vectors, and horizon. |

## One-Sentence Paper Use

The theorem explains Muon's polar update as the local solution to an operator-norm-constrained linearized problem, while the experiments show that this spectral bias is robust but only useful when the task, layer, and norm geometry reward such spectral spreading.

## Sources

- [theory note](e11_theory_note.md)
- [spectral allocation probe](e11_spectral_allocation_probe.md)
- [stateless direction ablation](e11_stateless_direction_ablation.md)
- [stateless optimizer trajectory ablation](e11_stateless_optimizer_trajectory.md)
- [Deep MNIST MLP probe](e11_deep_mnist_mlp_probe.md)
