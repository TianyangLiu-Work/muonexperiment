# E11 Condition Geometry

This directory-level note documents the current E11 experiment suite for the Muon condition-geometry paper project. The goal is to keep the codebase reproducible while the research question evolves.

## Research Question

Does Muon act as a geometry-shaping optimizer, and when does that geometry explain one-step loss decrease or short-horizon optimization progress?

Current working thesis:

> Muon is an update-spectrum shaping optimizer. It consistently produces flatter, higher-rank update spectra than Adam at matched update size, but this spectral bias helps optimization only under specific local task/layer/norm geometry.

## Main Entry Points

Paper draft and experiment triage:

- `paper/specgrad_activation_paper/`: current LaTeX draft for the activation-geometry framing.
- `paper/specgrad_activation_paper/notes/experiment_evaluation.md`: prioritized experiment evaluation for turning the draft into a project paper.

Run the core experiment:

```bash
python3 scripts/e11_run_experiments.py
python3 scripts/e11_make_figures.py
python3 scripts/e11_run_equal_update_control.py
python3 scripts/e11_run_spectral_allocation_probe.py
```

Run the current paper-supporting follow-up experiments:

```bash
python3 scripts/e11_run_overlap_followup.py
python3 scripts/e11_run_hyperparam_sweep.py
python3 scripts/e11_run_target_update_sweep.py
python3 scripts/e11_run_mlp_width_sweep.py
python3 scripts/e11_run_mlp_per_layer_control.py
python3 scripts/e11_run_mlp_layer_hybrid.py
python3 scripts/e11_run_mnist_mlp_probe.py
python3 scripts/e11_run_deep_mnist_mlp_probe.py
python3 scripts/e11_run_mnist_patch_probe.py
python3 scripts/e11_run_mnist_conv_probe.py
python3 scripts/e11_run_stateless_direction_ablation.py
python3 scripts/e11_run_stateless_optimizer_trajectory.py
python3 scripts/e11_run_singular_vector_trajectory.py
python3 scripts/e11_run_singular_vector_swap_probe.py
python3 scripts/e11_run_natural_update_swap_probe.py
python3 scripts/e11_run_optimizer_switch_probe.py
python3 scripts/e11_run_optimizer_switch_reset_control.py
python3 scripts/e11_run_optimizer_switch_horizon_sweep.py
python3 scripts/e11_run_optimizer_switch_lr_sweep.py
python3 scripts/e11_run_boundary_predictor.py
```

The MNIST probe downloads torchvision data into `data/`, which is intentionally git-ignored as a local cache.
Generated animations/videos are also ignored by default; the current E11 evidence set uses static figures and Markdown/CSV artifacts.

Training and diagnostic batch contract:

- MF-with-input and Matrix Sensing are optimized full-batch.
- Neural tasks are optimized with a strict mini-batch (`train_batch_size < num_samples`).
- Neural activation diagnostics use the full sampled dataset for the problem instance. MLP tasks record `diagnostic_A_definition == full_layer_input_activation`; the patch surrogate records `diagnostic_A_definition == full_patch_and_classifier_activation`; the ConvNet probe records `diagnostic_A_definition == full_conv_patch_and_classifier_activation`.
- `delta_loss` is the same-batch pre/post-update decrease, evaluated on the training batch used for the gradient/update.

Regenerate all core discussion artifacts after results exist:

```bash
python3 scripts/e11_write_all_discussion_artifacts.py
```

Validate the current artifact set:

```bash
python3 scripts/e11_validate_outputs.py
python3 -m pytest tests -q
git diff --check
```

Equivalent make targets:

```bash
make e11-main-results      # core trajectories, base figures, equal-update control, and spectral-allocation probe
make e11-appendix-results  # current appendix/guardrail probes
make e11-all-results       # main plus appendix/guardrail result generation
make e11-paper-assets      # regenerate paper-facing Markdown/TeX artifacts after results exist
make e11-check             # validate outputs, run tests, and check whitespace
make e11-full              # regenerate discussion artifacts, then run e11-check
```

## Core Artifacts

Paper-facing synthesis:

- `discussion/e11_paper_skeleton.md`
- `discussion/e11_main_paper_package.md`
- `discussion/e11_main_figure_captions.md`
- `discussion/e11_notation_glossary.md`
- `discussion/e11_quantitative_claim_ledger.md`
- `discussion/e11_paper_numbers.tex`
- `discussion/e11_reproduction_checklist.md`
- `discussion/e11_paper_readiness_audit.md`
- `discussion/e11_reviewer_risk_audit.md`
- `discussion/e11_research_synthesis.md`
- `discussion/e11_evidence_index.md`
- `discussion/e11_artifact_manifest.md`

Mechanism and boundary evidence:

- `discussion/e11_cross_task_signature.md`
- `discussion/e11_mechanism_boundary.md`
- `discussion/e11_theory_note.md`
- `discussion/e11_mechanism_theorem_bridge.md`
- `discussion/e11_optimizer_ablation_map.md`
- `discussion/e11_stateless_direction_ablation.md`
- `discussion/e11_stateless_optimizer_trajectory.md`
- `discussion/e11_boundary_predictor.md`
- `discussion/e11_boundary_predictor_audit.md`
- `discussion/e11_mnist_mlp_probe.md`
- `discussion/e11_deep_mnist_mlp_probe.md`
- `discussion/e11_mnist_patch_probe.md`
- `discussion/e11_mnist_conv_probe.md`
- `discussion/e11_optimizer_invariance_audit.md`
- `discussion/e11_claim_validity_audit.md`

Primary quantitative tables:

- `results/e11_equal_update/update_spectrum_summary.csv`
- `results/e11_equal_update/first_order_calibration_summary.csv`
- `results/e11_cross_task_signature/cross_task_signature_summary.csv`
- `results/e11_mechanism_boundary/mechanism_boundary_map.csv`
- `results/e11_boundary_predictor/boundary_predictor_summary.csv`
- `results/e11_boundary_predictor/boundary_predictor_uncertainty.csv`
- `results/e11_stateless_direction_ablation/stateless_direction_summary.csv`
- `results/e11_stateless_optimizer_trajectory/stateless_optimizer_summary.csv`
- `results/e11_spectral_allocation_probe/spectral_allocation_summary.csv`
- `results/e11_mnist_mlp_probe/pair_summary.csv`
- `results/e11_deep_mnist_mlp_probe/pair_summary.csv`
- `results/e11_mnist_patch_probe/pair_summary.csv`
- `results/e11_mnist_conv_probe/pair_summary.csv`

Primary figures:

- `figures/e11_equal_update/update_spectrum_robustness.png`
- `figures/e11_equal_update/first_order_calibration.png`
- `figures/e11_stateless_direction_ablation/stateless_direction_ratios.png`
- `figures/e11_stateless_optimizer_trajectory/stateless_optimizer_trajectory_ratios.png`
- `figures/e11_spectral_allocation_probe/spectral_allocation_ratios.png`
- `figures/e11_target_update_sweep/target_update_first_order_ratios.png`
- `figures/e11_mnist_mlp_probe/mnist_mlp_first_order_ratios.png`
- `figures/e11_deep_mnist_mlp_probe/deep_mnist_mlp_first_order_ratios.png`
- `figures/e11_mnist_patch_probe/mnist_patch_first_order_ratios.png`
- `figures/e11_mnist_conv_probe/mnist_conv_first_order_ratios.png`

## Claims That Currently Survive

1. Muon reliably changes update spectra.
   - Equal-update `nrUpdate` ratio is about `2.033`.
   - Equal-update `stUpdate` ratio is about `5.215`.
2. One-step loss decrease is locally well explained by `<G,D>`, where `D` is the positive descent update.
   - Spearman correlation is about `0.9206` in the equal-update calibration summary.
3. Flat/polar update allocation has a norm-geometry boundary.
   - It loses under Frobenius budget and wins under operator-norm budget in the spectral-allocation probe.
   - The stateless direction ablation shows polar direction alone raises update rank but still loses to GD under matched Frobenius update size.
   - The stateless optimizer trajectory ablation shows the same pattern across short matched-update trajectories.
   - The local theory note states the matching first-order constrained problem.
   - The optimizer ablation map records which controls isolate update size, layer allocation, stateless direction choice, singular-value allocation, singular-vector geometry, and continuation effects.
4. Muon's local advantage is conditional.
   - The sign flips across Matrix Sensing, MF-with-input, MLP width, and update-size controls.
5. The current boundary map is not yet a strong predictive law.
   - The leave-setting-out boundary predictor is weak and should be treated as a baseline.
6. Neural MNIST sanity checks preserve update-spectrum shaping but remain Adam-favorable for first-order progress in wider/deeper/local models.
   - Deep MNIST, patch/shared-weight, and true ConvNet probes all keep the Muon update-spectrum signature while showing Adam-favorable first-order ratios.
   - This is a neural negative control against claiming high-rank updates directly improve progress.

## Claims To Avoid

Do not claim:

- Muon is generally better than Adam.
- Higher rank/stable rank directly implies lower loss.
- Muon is generally more stable.
- The current boundary map is already a predictive theory for unseen tasks.

## Code Organization

- `e11_condition_geometry/config.py`: default experiment specification.
- `e11_condition_geometry/problems/`: torch problem definitions.
- `e11_condition_geometry/runner.py`: shared training and equal-update execution.
- `e11_condition_geometry/diagnostics.py`: rank, spectrum, and one-step metrics.
- `e11_condition_geometry/statistics.py`: summary tables and confidence intervals.
- `e11_condition_geometry/plots/`: static figure generation.
- `e11_condition_geometry/reporting.py`: shared Markdown/report helpers.
- `scripts/e11_run_*.py`: experiment runners.
- `scripts/e11_write_*.py`: generated discussion and paper-facing artifacts.
- `tests/`: smoke and diagnostic tests.

## Current Publication Gaps

The current evidence supports a focused local-geometry paper. It is not yet enough for a broad optimizer-performance paper.

Most important next steps:

1. Add a modern or longer-horizon neural benchmark with explicit handling of non-matrix parameters only if broader neural-performance claims become central.
2. Define and test a stronger predictive boundary model for when Muon's flat/polar update improves `<G,D>`.
3. Add true trajectory-level optimizer variants that separate polar spectrum shaping from momentum/state details.
4. Turn the current local theory note into a polished theorem/proof section with exact assumptions.
