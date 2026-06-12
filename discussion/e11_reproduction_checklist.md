# E11 Reproduction Checklist

This generated checklist separates the minimal main-paper evidence from appendix controls and generated writing assets. It is not a new experiment; it is the current reproducibility map for the E11 paper direction.

## Make Targets

```bash
make e11-main-results      # core trajectories, base figures, equal-update control, and spectral-allocation probe
make e11-appendix-results  # current appendix/guardrail probes
make e11-all-results       # main plus appendix/guardrail result generation
make e11-paper-assets      # generated Markdown/TeX paper assets after results exist
make e11-check             # validation, tests, and whitespace check
make e11-full              # paper assets plus e11-check
```

## Minimal Main-Paper Evidence

| stage                     | command                                              | produces                                                                            | paper_role                                          |
|:--------------------------|:-----------------------------------------------------|:------------------------------------------------------------------------------------|:----------------------------------------------------|
| Core trajectories         | python3 scripts/e11_run_experiments.py               | results/e11/*                                                                       | Base Adam/Muon trajectories and raw diagnostics.    |
| Core figures              | python3 scripts/e11_make_figures.py                  | figures/e11/*                                                                       | Static diagnostic figures for the base run.         |
| Matched-update control    | python3 scripts/e11_run_equal_update_control.py      | results/e11_equal_update/* and figures/e11_equal_update/*                           | Main Figure 1 and Figure 2 evidence.                |
| Spectral allocation probe | python3 scripts/e11_run_spectral_allocation_probe.py | results/e11_spectral_allocation_probe/* and figures/e11_spectral_allocation_probe/* | Main Figure 3 evidence.                             |
| Mechanism boundary map    | python3 scripts/e11_write_mechanism_boundary.py      | results/e11_mechanism_boundary/mechanism_boundary_map.csv                           | Main Table 1 evidence after follow-up tables exist. |

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

After the result CSVs and figures exist, regenerate paper-facing assets with:

```bash
make e11-paper-assets
```

| artifact                                    | role                                                                          |
|:--------------------------------------------|:------------------------------------------------------------------------------|
| discussion/e11_main_paper_package.md        | Smallest main figure/table package.                                           |
| discussion/e11_quantitative_claim_ledger.md | Allowed wording, forbidden wording, quantitative anchors, and evidence links. |
| discussion/e11_paper_numbers.tex            | LaTeX macros generated from the current result CSVs.                          |
| discussion/e11_paper_skeleton.md            | Current section-level paper skeleton and figure/table plan.                   |
| discussion/e11_reviewer_risk_audit.md       | Known reviewer risks and safe claim decisions.                                |

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

`make e11-full` regenerates discussion/paper-facing artifacts and then runs validation, tests, and whitespace checks. It assumes the longer experiment result CSVs already exist unless their writer script reruns the relevant probe.
