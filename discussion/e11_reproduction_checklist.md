# E11 Reproduction Checklist

This generated checklist separates the minimal main-paper evidence from appendix controls and generated writing assets. It is not a new experiment; it is the current reproducibility map for the E11 paper direction.

## Make Targets

```bash
make e11-main-results      # core trajectories, equal-update, head-to-tail probes, and spectral-allocation
make e11-cifar-results     # CIFAR-100-LT MLP local matched-head-gain diagnostic
make e11-cifar-resnet-results # submit the CIFAR-100-LT ResNet18 GPU diagnostic via Slurm
make e11-cifar-resnet-rho002-results # submit the CIFAR-100-LT ResNet18 rho=0.002 robustness check via Slurm
make e11-cifar-resnet-checkpoint-sweep-results # submit the top-conference ResNet checkpoint-quality sweep via Slurm
make e11-appendix-results  # current appendix/guardrail probes
make e11-all-results       # main plus appendix/guardrail result generation
make e11-paper-assets      # current head-to-tail Markdown/TeX paper assets
make e11-guardrail-assets  # legacy condition-geometry guardrail notes
make e11-all-assets        # current paper assets plus legacy guardrail notes
make e11-paper-pdf         # rebuild paper/specgrad_activation_paper/main.pdf and two_page.pdf
make e11-check             # validation, tests, and whitespace check
make e11-full              # paper assets, PDF build, and e11-check
```

## Current Head-to-Tail Paper Evidence

| stage                                               | command                                                                    | produces                                                                                                    | paper_role                                                                                                                                                   |
|:----------------------------------------------------|:---------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Head-to-tail interference probe                     | python3 scripts/e11_run_head_tail_interference.py                          | results/e11_head_tail_interference/* and figures/e11_head_tail_interference/*                               | Synthetic matched-head-gain evidence for the head-to-tail interference condition.                                                                            |
| Long-tailed one-step diagnostic                     | python3 scripts/e11_run_long_tail_one_step.py                              | results/e11_long_tail_one_step/* and figures/e11_long_tail_one_step/*                                       | Real-data matched-head-gain check on held-out tail examples.                                                                                                 |
| CIFAR-100-LT MLP one-step diagnostic                | python3 scripts/e11_run_cifar100_lt_one_step.py --device cpu --no-download | results/e11_cifar100_lt_one_step/* and figures/e11_cifar100_lt_one_step/*                                   | Larger visual-data matched-head-gain check using an exactly controlled two-matrix MLP intervention.                                                          |
| CIFAR-100-LT ResNet18 one-step diagnostic           | sbatch scripts/slurm/e11_cifar100_resnet_one_step.sbatch                   | results/e11_cifar100_resnet_one_step/* and figures/e11_cifar100_resnet_one_step/*                           | GPU ResNet18 architecture robustness check for matched-head-gain tail drift at target head gain 0.005 times head loss.                                       |
| CIFAR-100-LT ResNet18 smaller-head-gain check       | sbatch scripts/slurm/e11_cifar100_resnet_one_step_rho002.sbatch            | results/e11_cifar100_resnet_one_step_rho002/* and figures/e11_cifar100_resnet_one_step_rho002/*             | Target-head-gain scale robustness check for the CIFAR-100-LT ResNet18 diagnostic at 0.002 times head loss.                                                   |
| CIFAR-100-LT ResNet18 checkpoint-quality sweep      | sbatch scripts/slurm/e11_cifar100_resnet_checkpoint_sweep.sbatch           | results/e11_cifar100_resnet_checkpoint_sweep/* and figures/e11_cifar100_resnet_checkpoint_sweep/*           | GPU checkpoint sweep across 250, 500, 1000, and 2000 warmup steps; strengthens single-checkpoint robustness while preserving the weak-tail-predictor caveat. |
| Long-tailed Muon-style compatibility diagnostic     | python3 scripts/e11_run_long_tail_muon_bridge.py                           | results/e11_long_tail_muon_bridge/* and figures/e11_long_tail_muon_bridge/*                                 | Fixed-checkpoint compatibility check from ideal polar(G_t) to momentum polar(M_t) and Newton-Schulz Muon-style directions.                                   |
| Long-tailed practical-Muon trajectory compatibility | python3 scripts/e11_run_long_tail_practical_muon_bridge.py                 | results/e11_long_tail_practical_muon_bridge/* and figures/e11_long_tail_practical_muon_bridge/*             | Short trajectory-level check of momentum and Newton-Schulz Muon-style directions at sampled states.                                                          |
| Long-tailed Muon state-source control               | python3 scripts/e11_run_long_tail_muon_state_source_control.py             | results/e11_long_tail_muon_state_source_control/* and figures/e11_long_tail_muon_state_source_control/*     | Checks whether Muon-style direction compatibility persists when sampled states come from a Fro/GD-style trajectory rather than an NS-Muon-style trajectory.  |
| Long-tailed practical training diagnostic           | python3 scripts/e11_run_long_tail_practical_training.py                    | results/e11_long_tail_practical_training/* and figures/e11_long_tail_practical_training/*                   | Practical Adam-vs-NS-Muon-style imbalanced mini-batch training sanity check with final tail metrics.                                                         |
| Long-tailed practical training LR sensitivity       | python3 scripts/e11_run_long_tail_practical_training_lr_sweep.py           | results/e11_long_tail_practical_training_lr_sweep/* and figures/e11_long_tail_practical_training_lr_sweep/* | Learning-rate sensitivity check for the practical NS-Muon-style training diagnostic.                                                                         |
| Head-only forgetting probe                          | python3 scripts/e11_run_long_tail_forgetting.py                            | results/e11_long_tail_forgetting/* and figures/e11_long_tail_forgetting/*                                   | Short-horizon held-out tail drift under matched head-only update schedules.                                                                                  |
| Long-tailed layerwise diagnostic                    | python3 scripts/e11_run_long_tail_layerwise.py                             | results/e11_long_tail_layerwise/* and figures/e11_long_tail_layerwise/*                                     | Layerwise finite-difference tail JVP and matched-head-gain drift evidence.                                                                                   |

## Background / Legacy E11 Evidence

These artifacts remain reproducible because they document the route to the
current head-to-tail framing and provide guardrails against broader optimizer
claims. They are not the main evidence table for the current paper draft.

| stage                                           | command                                                    | produces                                                                                        | paper_role                                                                                                                         |
|:------------------------------------------------|:-----------------------------------------------------------|:------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------|
| Core trajectories                               | python3 scripts/e11_run_experiments.py                     | results/e11/*                                                                                   | Base Adam/Muon trajectories and raw diagnostics.                                                                                   |
| Core figures                                    | python3 scripts/e11_make_figures.py                        | figures/e11/*                                                                                   | Static diagnostic figures for the base run.                                                                                        |
| Matched-update control                          | python3 scripts/e11_run_equal_update_control.py            | results/e11_equal_update/* and figures/e11_equal_update/*                                       | Background E11 optimizer-geometry control; not a current head-to-tail main figure.                                                 |
| Head-to-tail singular-vector alignment ablation | python3 scripts/e11_run_head_tail_alignment_ablation.py    | results/e11_head_tail_alignment_ablation/* and figures/e11_head_tail_alignment_ablation/*       | Checks that the sandwich rank condition is a worst-case bound-ordering condition rather than a realized-drift predictor by itself. |
| Long-tailed imbalance ablation                  | python3 scripts/e11_run_long_tail_imbalance_ablation.py    | results/e11_long_tail_imbalance_ablation/* and figures/e11_long_tail_imbalance_ablation/*       | Checks whether the one-step tail-drift pattern persists as the tail training split becomes rarer.                                  |
| Long-tailed checkpoint sweep                    | python3 scripts/e11_run_long_tail_checkpoint_sweep.py      | results/e11_long_tail_checkpoint_sweep/* and figures/e11_long_tail_checkpoint_sweep/*           | Checks whether the one-step tail-drift pattern depends on selecting a single warmup checkpoint.                                    |
| Long-tailed class-partition sweep               | python3 scripts/e11_run_long_tail_class_partition_sweep.py | results/e11_long_tail_class_partition_sweep/* and figures/e11_long_tail_class_partition_sweep/* | Checks whether the one-step tail-drift pattern depends on the default head/tail digit-class partition.                             |
| Long-tailed matched-gain rho sweep              | python3 scripts/e11_run_long_tail_rho_sweep.py             | results/e11_long_tail_rho_sweep/* and figures/e11_long_tail_rho_sweep/*                         | Checks whether the one-step tail-drift pattern depends on the default target head-gain fraction.                                   |
| Spectral allocation probe                       | python3 scripts/e11_run_spectral_allocation_probe.py       | results/e11_spectral_allocation_probe/* and figures/e11_spectral_allocation_probe/*             | Mechanism background for norm-budget geometry; appendix evidence for the current head-to-tail paper.                               |
| Mechanism boundary map                          | python3 scripts/e11_write_mechanism_boundary.py            | results/e11_mechanism_boundary/mechanism_boundary_map.csv                                       | Legacy E11 boundary summary used as background, not the current head-to-tail result table.                                         |

## Appendix / Guardrail Evidence

| command                                                   | why                                                                       |
|:----------------------------------------------------------|:--------------------------------------------------------------------------|
| python3 scripts/e11_run_overlap_followup.py               | Focused overlap settings used by early boundary and guardrail claims.     |
| python3 scripts/e11_run_hyperparam_sweep.py               | Learning-rate sensitivity and best-over-LR guardrail.                     |
| python3 scripts/e11_run_target_update_sweep.py            | Target update-size boundary control.                                      |
| python3 scripts/e11_run_mlp_width_sweep.py                | MLP width boundary and transition control.                                |
| python3 scripts/e11_run_mlp_per_layer_control.py          | Per-layer matched-update boundary control.                                |
| python3 scripts/e11_run_mlp_layer_hybrid.py               | Layerwise optimizer hybrid control for MLP update allocation.             |
| python3 scripts/e11_run_mnist_mlp_probe.py                | Small neural sanity benchmark.                                            |
| python3 scripts/e11_run_deep_mnist_mlp_probe.py           | Deeper all-matrix MNIST sanity benchmark.                                 |
| python3 scripts/e11_run_mnist_patch_probe.py              | Matrix-only MNIST patch classifier sanity benchmark.                      |
| python3 scripts/e11_run_mnist_conv_probe.py               | Small true Conv2d MNIST sanity benchmark with flattened-kernel Muon view. |
| python3 scripts/e11_run_stateless_direction_ablation.py   | One-step stateless direction mechanism control.                           |
| python3 scripts/e11_run_stateless_optimizer_trajectory.py | Short stateless trajectory mechanism control.                             |
| python3 scripts/e11_run_singular_vector_trajectory.py     | Gradient/update singular-vector overlap trajectory diagnostic.            |
| python3 scripts/e11_run_singular_vector_swap_probe.py     | Singular-vector swap mechanism control.                                   |
| python3 scripts/e11_run_natural_update_swap_probe.py      | Natural update-vector and singular geometry control.                      |
| python3 scripts/e11_run_optimizer_switch_probe.py         | Optimizer-state continuation probe.                                       |
| python3 scripts/e11_run_optimizer_switch_reset_control.py | Reset-state control for optimizer switching.                              |
| python3 scripts/e11_run_optimizer_switch_horizon_sweep.py | Continuation horizon sensitivity control.                                 |
| python3 scripts/e11_run_optimizer_switch_lr_sweep.py      | Continuation negative control against long-horizon overclaiming.          |
| python3 scripts/e11_run_boundary_predictor.py             | Leave-setting-out predictive-boundary baseline.                           |

## Generated Paper-Facing Assets

After the result CSVs and figures exist, regenerate the current head-to-tail paper assets with:

```bash
make e11-paper-assets
```

Regenerate legacy condition-geometry guardrail notes separately with:

```bash
make e11-guardrail-assets
```

| artifact                                                     | role                                                                                 |
|:-------------------------------------------------------------|:-------------------------------------------------------------------------------------|
| discussion/e11_main_paper_package.md                         | Smallest main figure/table package.                                                  |
| discussion/e11_quantitative_claim_ledger.md                  | Allowed wording, forbidden wording, quantitative anchors, and evidence links.        |
| paper/specgrad_activation_paper/tables/e11_paper_numbers.tex | LaTeX macros generated from the current result CSVs and included by the paper draft. |
| discussion/e11_paper_skeleton.md                             | Current section-level paper skeleton and figure/table plan.                          |
| discussion/e11_reviewer_risk_audit.md                        | Known reviewer risks and safe claim decisions.                                       |

## Batch / Activation Contract

The validation gate enforces the current training-versus-diagnostics contract:

- Default core E11 optimization uses noisy mini-batches, so `train_batch_size < num_samples` and `noise_std > 0`.
- Activation diagnostics use the full sampled problem instance. MLP diagnostics must record `diagnostic_A_definition == full_layer_input_activation`.
- `delta_loss` is the same-batch pre/post-update decrease, evaluated on the training batch used for the gradient/update.

## Validation Gate

Before using the numbers in a draft, run:

```bash
make e11-check
```

The stronger local gate is:

```bash
make e11-full
```

`make e11-full` regenerates current paper-facing artifacts, rebuilds the paper PDF, and then runs validation, tests, and whitespace checks. It assumes the longer experiment result CSVs already exist unless their writer script reruns the relevant probe. Legacy guardrail notes are intentionally not part of `e11-full`; use `make e11-guardrail-assets` when those background notes need refreshing.
