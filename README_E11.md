# E11 Head-to-Tail Paper Evidence

This directory-level note documents the current E11 evidence package for the head-to-tail interference paper, plus the older condition-geometry experiments that now serve as background guardrails. The goal is to keep the paper evidence, generated artifacts, and legacy diagnostics reproducible without blurring their roles.

## Current Paper Question

Can an idealized spectral/polar update direction reduce head-to-tail function interference in long-tailed small-batch training, once we match the amount of head progress?

Current working thesis:

> Head-only updates can damage logits on tail examples while tail samples are absent. Under a measurable rank/sensitivity condition, an idealized spectral/polar direction causes less tail-example logit drift than a Frobenius/GD-style direction at matched head gain.

Current paper scope:

> The active paper draft is a focused head-to-tail interference paper. Its current defensible claim is that an idealized spectral/polar direction can reduce tail-example logit drift at matched head gain in synthetic, small long-tailed, and CIFAR-100-LT ResNet diagnostics. The older E11 condition-geometry experiments are background evidence and guardrails; they should not be read as a broad claim that Muon is a generally better optimizer.

## Main Entry Points

Paper draft and experiment triage:

- `paper/specgrad_activation_paper/`: current LaTeX draft for the head-to-tail interference framing.
- `paper/specgrad_activation_paper/notes/experiment_evaluation.md`: prioritized experiment evaluation for turning the draft into a project paper.

Run the core experiment:

```bash
python3 scripts/e11_run_experiments.py
python3 scripts/e11_make_figures.py
python3 scripts/e11_run_equal_update_control.py
python3 scripts/e11_run_head_tail_interference.py
python3 scripts/e11_run_head_tail_alignment_ablation.py
python3 scripts/e11_run_long_tail_one_step.py
python3 scripts/e11_run_cifar100_lt_one_step.py --device cpu --no-download
sbatch scripts/slurm/e11_cifar100_resnet_one_step.sbatch
sbatch scripts/slurm/e11_cifar100_resnet_one_step_rho002.sbatch
python3 scripts/e11_run_long_tail_imbalance_ablation.py
python3 scripts/e11_run_long_tail_checkpoint_sweep.py
python3 scripts/e11_run_long_tail_class_partition_sweep.py
python3 scripts/e11_run_long_tail_rho_sweep.py
python3 scripts/e11_run_long_tail_muon_bridge.py
python3 scripts/e11_run_long_tail_practical_muon_bridge.py
python3 scripts/e11_run_long_tail_muon_state_source_control.py
python3 scripts/e11_run_long_tail_practical_training.py
python3 scripts/e11_run_long_tail_practical_training_lr_sweep.py
python3 scripts/e11_run_long_tail_forgetting.py
python3 scripts/e11_run_long_tail_layerwise.py
python3 scripts/e11_run_spectral_allocation_probe.py
```

Run appendix and guardrail follow-up experiments:

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

The MNIST and CIFAR probes download torchvision data into `data/`, which is intentionally git-ignored as a local cache.
Generated animations/videos are also ignored by default; the current E11 evidence set uses static figures and Markdown/CSV artifacts.

Training and diagnostic batch contract:

- The default core E11 settings use noisy mini-batch optimization (`train_batch_size < num_samples`, `noise_std > 0`) for MF-with-input, Matrix Sensing, and SmallMLPDigits.
- Synthetic tasks add fixed observation noise to the sampled training problem; SmallMLPDigits adds step-level input noise to the training mini-batch.
- Activation and spectral diagnostics still use the full sampled problem instance. MLP tasks record `diagnostic_A_definition == full_layer_input_activation`; the patch surrogate records `diagnostic_A_definition == full_patch_and_classifier_activation`; the ConvNet probe records `diagnostic_A_definition == full_conv_patch_and_classifier_activation`.
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
make e11-main-results      # core trajectories, base figures, equal-update, head-tail, Muon-style compatibility, and spectral-allocation probes
make e11-cifar-results     # CIFAR-100-LT MLP local matched-head-gain diagnostic
make e11-cifar-resnet-results # submit the CIFAR-100-LT ResNet18 GPU diagnostic via Slurm
make e11-cifar-resnet-rho002-results # submit the CIFAR-100-LT ResNet18 rho=0.002 robustness check via Slurm
make e11-cifar-resnet-checkpoint-sweep-results # submit the top-conference ResNet checkpoint-quality sweep via Slurm
make e11-appendix-results  # current appendix/guardrail probes
make e11-all-results       # main plus appendix/guardrail result generation
make e11-paper-assets      # regenerate current head-to-tail paper Markdown/TeX artifacts
make e11-guardrail-assets  # regenerate legacy condition-geometry guardrail notes
make e11-all-assets        # regenerate current paper artifacts plus legacy guardrail notes
make e11-paper-pdf         # rebuild paper/specgrad_activation_paper/main.pdf and two_page.pdf
make e11-check             # validate outputs, run tests, and check whitespace
make e11-full              # regenerate paper artifacts, rebuild the PDF, then run e11-check
```

Top-conference upgrade runner:

```bash
make e11-cifar-resnet-checkpoint-sweep-results
```

This submits `scripts/slurm/e11_cifar100_resnet_checkpoint_sweep.sbatch`, which
runs `scripts/e11_run_cifar100_resnet_checkpoint_sweep.py` on GPU across
ResNet18 warmup checkpoints. The current committed result reduces
single-checkpoint risk, but it should not be read as high-quality tail-predictor
evidence because the best pre-update tail accuracy is still low.

## Core Artifacts

Paper-facing synthesis:

- `discussion/e11_paper_skeleton.md`
- `discussion/e11_main_paper_package.md`
- `discussion/e11_main_figure_captions.md`
- `discussion/e11_notation_glossary.md`
- `discussion/e11_quantitative_claim_ledger.md`
- `paper/specgrad_activation_paper/tables/e11_paper_numbers.tex`
- `discussion/e11_paper_numbers.tex` (mirror copy for paper-facing discussion artifacts)
- `discussion/e11_reproduction_checklist.md`
- `discussion/e11_paper_readiness_audit.md`
- `discussion/e11_top_conference_plan.md`
- `discussion/e11_reviewer_risk_audit.md`
- `discussion/e11_pasted_review_audit.md`
- `discussion/e11_end_of_draft_self_review.md`
- `discussion/e11_reference_audit.md`
- `discussion/e11_research_synthesis.md`
- `discussion/e11_evidence_index.md`
- `discussion/e11_artifact_manifest.md`
- `discussion/e11_activation_perturbation.md`
- `discussion/e11_head_tail_interference.md`
- `discussion/e11_head_tail_alignment_ablation.md`
- `discussion/e11_long_tail_one_step.md`
- `discussion/e11_cifar100_lt_one_step.md`
- `discussion/e11_cifar100_resnet_one_step.md`
- `discussion/e11_cifar100_resnet_one_step_rho002.md`
- `discussion/e11_cifar100_resnet_checkpoint_sweep.md`
- `discussion/e11_long_tail_imbalance_ablation.md`
- `discussion/e11_long_tail_checkpoint_sweep.md`
- `discussion/e11_long_tail_class_partition_sweep.md`
- `discussion/e11_long_tail_rho_sweep.md`
- `discussion/e11_long_tail_muon_bridge.md`
- `discussion/e11_long_tail_practical_muon_bridge.md`
- `discussion/e11_long_tail_muon_state_source_control.md`
- `discussion/e11_long_tail_practical_training.md`
- `discussion/e11_long_tail_practical_training_lr_sweep.md`
- `discussion/e11_long_tail_forgetting.md`
- `discussion/e11_long_tail_layerwise.md`

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

Primary paper quantitative tables:

- `results/e11_head_tail_interference/pair_summary.csv`
- `results/e11_head_tail_alignment_ablation/summary.csv`
- `results/e11_long_tail_one_step/pair_summary.csv`
- `results/e11_cifar100_lt_one_step/pair_summary.csv`
- `results/e11_cifar100_resnet_one_step/pair_summary.csv`
- `results/e11_cifar100_resnet_one_step_rho002/pair_summary.csv`
- `results/e11_cifar100_resnet_checkpoint_sweep/pair_summary.csv`
- `results/e11_long_tail_imbalance_ablation/summary.csv`
- `results/e11_long_tail_checkpoint_sweep/summary.csv`
- `results/e11_long_tail_class_partition_sweep/summary.csv`
- `results/e11_long_tail_rho_sweep/summary.csv`
- `results/e11_long_tail_muon_bridge/pair_summary.csv`
- `results/e11_long_tail_practical_muon_bridge/summary.csv`
- `results/e11_long_tail_muon_state_source_control/summary.csv`
- `results/e11_long_tail_practical_training/summary.csv`
- `results/e11_long_tail_practical_training_lr_sweep/sweep_summary.csv`
- `results/e11_long_tail_forgetting/summary.csv`
- `results/e11_long_tail_layerwise/summary.csv`
- `paper/specgrad_activation_paper/tables/head_tail_empirical_results.tex`

Primary paper figures:

- `figures/e11_head_tail_interference/head_tail_drift_ratio.png`
- `figures/e11_head_tail_alignment_ablation/head_tail_alignment_ablation.png`
- `figures/e11_long_tail_one_step/long_tail_one_step_tail_response.png`
- `figures/e11_cifar100_lt_one_step/cifar100_lt_one_step_tail_response.png`
- `figures/e11_cifar100_resnet_one_step/cifar100_resnet_one_step_tail_response.png`
- `figures/e11_cifar100_resnet_one_step_rho002/cifar100_resnet_one_step_tail_response.png`
- `figures/e11_cifar100_resnet_checkpoint_sweep/cifar100_resnet_checkpoint_sweep.png`
- `figures/e11_long_tail_imbalance_ablation/long_tail_imbalance_ablation.png`
- `figures/e11_long_tail_checkpoint_sweep/long_tail_checkpoint_sweep.png`
- `figures/e11_long_tail_class_partition_sweep/long_tail_class_partition_sweep.png`
- `figures/e11_long_tail_rho_sweep/long_tail_rho_sweep.png`
- `figures/e11_long_tail_muon_bridge/long_tail_muon_bridge.png`
- `figures/e11_long_tail_practical_muon_bridge/long_tail_practical_muon_bridge.png`
- `figures/e11_long_tail_muon_state_source_control/long_tail_muon_state_source_control.png`
- `figures/e11_long_tail_practical_training/long_tail_practical_training.png`
- `figures/e11_long_tail_practical_training_lr_sweep/long_tail_practical_training_lr_sweep.png`
- `figures/e11_long_tail_forgetting/long_tail_head_only_forgetting.png`
- `figures/e11_long_tail_layerwise/long_tail_layerwise_drift.png`

## Current Diagnostic Claims

1. The synthetic head-to-tail condition has the expected sign.
   - When `nrank(G_H) > ssrank(B_T,A_T)`, the spectral/Frobenius squared tail-example logit drift ratio is about `0.3403`.
   - When the inequality is reversed, the squared drift ratio is about `7.208`.
   - When the positive singular spectra are kept but singular-vector alignment is randomized, the realized drift ratio is mixed: geomean about `1.171 [1.069, 1.283]`, median about `1.303`, and spectral lower drift fraction about `0.416`. This is a caveat that the condition orders worst-case bounds, not realized drift by itself.
2. The long-tailed one-step diagnostic reports lower tail-example logit drift at matched head gain in the tested digits setting.
   - The spectral/Frobenius squared tail-example logit drift ratio is about `0.5501 [0.5101, 0.5931]`.
   - All 20 paired seeds have lower spectral tail drift.
   - Across `rho/L_H` values from `0.005` to `0.08`, the largest full-drift CI upper endpoint is about `0.5932`; the actual head-gain relative-error upper endpoint grows to about `0.1176`, so this is a local-scale robustness check rather than a large-step claim.
3. The CIFAR-100-LT reruns provide a more appropriate visual-data architecture check.
   - The two-layer CIFAR-100-LT MLP gives squared drift ratio about `0.1944 [0.1848, 0.2044]`, with lower spectral drift in all 5 seeds, but tail loss increase is slightly worse.
   - The GPU ResNet18 CIFAR-100-LT diagnostic gives squared drift ratio about `0.5611 [0.5224, 0.6026]`, with lower spectral drift in all 10 seeds.
   - A smaller-head-gain GPU check at `rho=0.002 L_H` gives squared drift ratio about `0.7761 [0.7585, 0.7941]`.
   - A warmup-checkpoint sweep over 250/500/1000/2000 steps keeps the drift-ratio CI upper endpoint below 1 at every checkpoint; the worst endpoint is about `0.6026`.
   - The same sweep's best pre-update tail accuracy is only about `0.068`, so it reduces checkpoint-selection risk but does not prove preservation of a high-quality tail predictor.
   - In the default ResNet diagnostic, tail-loss increase diff spectral-minus-Fro is about `-0.000421 [-0.000592, -0.000249]`; tail-accuracy-drop diff still crosses zero, so this remains a local drift/loss diagnostic rather than an accuracy claim.
4. The 8-step head-only forgetting diagnostic shows lower measured tail drift across the short horizon.
   - Final squared drift ratio is about `0.6167 [0.5744, 0.6622]`.
   - Drift-area ratio is about `0.7787 [0.7549, 0.8033]`.
5. The layerwise diagnostic identifies the mechanism boundary.
   - Unit-direction spectral JVP is larger than Frobenius in both layers.
   - Matched-head-gain scaled and observed drift are lower in both layers.
   - The current evidence is consistent with a scaled head-gain efficiency mechanism, not a claim that spectral directions are intrinsically less tail-sensitive.
6. The Muon-style compatibility diagnostic connects the clean polar direction to sampled Muon-style state.
   - `polar(M_t)` has squared tail-example logit drift ratio about `0.8199 [0.6951, 0.9672]` relative to Fro/GD at matched head gain.
   - Newton-Schulz `NS(M_t)` has squared drift ratio about `0.9116 [0.7696, 1.08]`, so this finite-iteration approximation is not yet a significant drift-reduction result.
7. The short practical-Muon trajectory compatibility diagnostic extends this check to sampled trajectory states.
   - Across 120 sampled state-step comparisons, `polar(M_t)` has squared drift ratio about `0.7292 [0.6891, 0.7717]`.
   - `NS(M_t)` has squared drift ratio about `0.8019 [0.7583, 0.848]`, while momentum-gradient cosine averages about `0.8589 [0.8328, 0.885]`.
8. The Muon trajectory state-source control reduces the selected-state concern.
   - On Fro/GD-style trajectory states with matched seeds, batches, length, and nominal trajectory learning rate, `polar(M_t)` has squared drift ratio about `0.7255 [0.686, 0.7672]`.
   - On the same Fro/GD-style states, `NS(M_t)` has squared drift ratio about `0.7973 [0.7546, 0.8425]`, so the short-trajectory compatibility signal is not unique to NS-Muon-style generated states.
9. The practical imbalanced-training diagnostic is consistent with the drift story but still not a leaderboard result.
   - At fixed lightweight hyperparameters on long-tailed digits, NS-Muon-style training has final train loss ratio about `0.6468 [0.604, 0.6926]` and tail eval loss ratio about `0.8549 [0.8319, 0.8786]` versus Adam.
   - Tail eval margin difference is about `1.955 [1.628, 2.281]`, and tail eval drift RMS ratio is about `0.7501 [0.7244, 0.7767]`, but tail accuracy difference is `0`, so this is a drift/loss/margin diagnostic rather than a broad accuracy claim.
   - LR sensitivity shows why this is not a monotone optimizer story: smaller `muon_lr` under-trains, while `muon_lr=0.1` lowers train loss further but worsens tail loss and drift.

## Claims To Avoid

Do not claim:

- Muon is generally better than Adam.
- Higher rank/stable rank directly implies lower loss.
- Muon is generally more stable.
- The current boundary map is already a predictive theory for unseen tasks.
- Lower tail-example logit drift automatically improves tail accuracy or final tail loss.
- The idealized polar/spectral direction already explains full Muon optimizer behavior.
- The current small practical NS-Muon-style run is sufficient as a broad optimizer benchmark.

## Code Organization

- `e11_condition_geometry/config.py`: default experiment specification.
- `e11_condition_geometry/problems/`: torch problem definitions.
- `e11_condition_geometry/runner.py`: shared training and equal-update execution.
- `e11_condition_geometry/diagnostics.py`: rank, spectrum, and one-step metrics.
- `e11_condition_geometry/statistics.py`: summary tables and confidence intervals.
- `e11_condition_geometry/long_tail_digits.py`: shared sklearn-digits long-tail data, MLP, update-direction, and layerwise diagnostic helpers.
- `e11_condition_geometry/cifar100_long_tail.py`: CIFAR-100-LT two-layer MLP matched-head-gain diagnostic.
- `e11_condition_geometry/cifar100_resnet_tail.py`: CIFAR-100-LT ResNet18 matched-head-gain diagnostic with Conv/Linear matrix-weight interventions.
- `e11_condition_geometry/plots/`: static figure generation.
- `e11_condition_geometry/reporting.py`: shared Markdown/report helpers.
- `scripts/e11_run_*.py`: experiment runners.
- `scripts/e11_run_cifar100_lt_one_step.py`: CIFAR-100-LT two-layer MLP runner.
- `scripts/e11_run_cifar100_resnet_one_step.py`: CIFAR-100-LT ResNet18 runner called by the Slurm wrapper.
- `scripts/e11_run_cifar100_resnet_checkpoint_sweep.py`: ResNet18 checkpoint-quality sweep for the top-conference upgrade path.
- `scripts/slurm/e11_cifar100_resnet_one_step.sbatch`: GPU/Slurm submission wrapper for the ResNet18 diagnostic.
- `scripts/slurm/e11_cifar100_resnet_one_step_rho002.sbatch`: GPU/Slurm submission wrapper for the smaller-head-gain ResNet18 check.
- `scripts/slurm/e11_cifar100_resnet_checkpoint_sweep.sbatch`: GPU/Slurm submission wrapper for the checkpoint-quality sweep.
- `scripts/e11_write_*.py`: generated discussion and paper-facing artifacts.
- `tests/`: smoke and diagnostic tests.

## Current Publication Gaps

The current evidence is consistent with a focused local-geometry paper. It is not yet enough for a broad optimizer-performance paper.

Most important next steps:

1. Add higher-quality tail checkpoints or a stronger real long-tail dataset; the completed CIFAR-100-LT ResNet checkpoint sweep reduces single-checkpoint risk but still has low pre-update tail accuracy.
2. Extend the current fixed-checkpoint, short-trajectory, and small practical-training Muon diagnostics into a full practical Muon benchmark with schedules, checkpoint distributions, final tail metrics, and hyperparameter robustness.
3. Add larger-architecture layerwise JVP/decomposition diagnostics if the detailed scaled-head-gain mechanism is meant to survive beyond the current small MLP explanation.
4. Keep separating function-drift evidence from tail loss, margin, accuracy, and final optimizer performance.
