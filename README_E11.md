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
sbatch scripts/slurm/e11_cifar100_resnet_checkpoint_sweep.sbatch
python3 scripts/e11_run_cifar100_resnet_condition_proxy_scatter.py
sbatch scripts/slurm/e11_cifar100_resnet_fc_condition_scatter.sbatch
sbatch scripts/slurm/e11_cifar100_resnet_tail_quality_control.sbatch
sbatch scripts/slurm/e11_cifar100_resnet_imbalance_sweep.sbatch
sbatch scripts/slurm/e11_cifar100_resnet_layer_jvp_tail_quality.sbatch
sbatch scripts/slurm/e11_cifar100_resnet_layer_jvp_checkpoint_prediction.sbatch
sbatch scripts/slurm/e11_cifar100_resnet_condition_score_heldout_architecture.sbatch
sbatch scripts/slurm/e11_cifar100_resnet_condition_score_heldout_data.sbatch
python3 scripts/e11_evaluate_cifar100_resnet_condition_score_heldouts.py
sbatch scripts/slurm/e11_cifar100_resnet_lt_standard_eval.sbatch
sbatch scripts/slurm/e11_cifar100_resnet_lt_recipe_benchmark.sbatch
sbatch scripts/slurm/e11_cifar100_resnet_lt_muon_final_benchmark.sbatch
sbatch scripts/slurm/e11_cifar100_resnet_practical_muon_bridge.sbatch
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

On the shared GPU server, the verified test interpreter is
`/data/conda_envs/SpatialQuantization/bin/python`, which has torch/torchvision.
The root `Makefile` defaults to that interpreter when it exists; pass
`PYTHON=/path/to/python` to override it on another machine.
If that environment is missing pytest, install it into the user site with
`/data/conda_envs/SpatialQuantization/bin/python -m pip install --user pytest`
and then run `/data/conda_envs/SpatialQuantization/bin/python -m pytest tests -q`.

Equivalent make targets:

```bash
make e11-main-results      # core trajectories, base figures, equal-update, head-tail, Muon-style compatibility, and spectral-allocation probes
make e11-cifar-results     # CIFAR-100-LT MLP local matched-head-gain diagnostic
make e11-cifar-resnet-results # submit the CIFAR-100-LT ResNet18 GPU diagnostic via Slurm
make e11-cifar-resnet-rho002-results # submit the CIFAR-100-LT ResNet18 rho=0.002 robustness check via Slurm
make e11-cifar-resnet-checkpoint-sweep-results # submit the top-conference ResNet checkpoint-quality sweep via Slurm
make e11-cifar-resnet-condition-proxy-results # regenerate the ResNet rank-proxy scatter from checkpoint-sweep CSVs
make e11-cifar-resnet-fc-condition-results # submit the ResNet final-layer downstream-aware condition diagnostic via Slurm
make e11-cifar-resnet-tail-quality-results # submit the tail-rich ResNet checkpoint-quality control via Slurm
make e11-cifar-resnet-imbalance-sweep-results # submit the CIFAR-100-LT ResNet18 tail-count imbalance sweep via Slurm
make e11-cifar-resnet-layer-jvp-tail-quality-results # submit the all-layer ResNet finite-difference JVP tail-quality diagnostic via Slurm
make e11-cifar-resnet-layer-jvp-checkpoint-prediction-results # submit the all-layer ResNet JVP checkpoint-transfer benchmark via Slurm
make e11-cifar-resnet-condition-score-heldout-architecture-results # submit the registered ResNet34 held-out architecture condition-score split via Slurm
make e11-cifar-resnet-condition-score-heldout-data-results # submit the registered CIFAR-10-LT held-out data condition-score split via Slurm
make e11-cifar-resnet-condition-score-heldout-eval # evaluate frozen condition-score gates after both held-out Slurm jobs finish
make e11-cifar-resnet-lt-standard-eval-results # submit the standard CIFAR-100-LT ResNet18 many/medium/few reporting baseline via Slurm
make e11-cifar-resnet-lt-recipe-benchmark-results # submit the augmented CIFAR-100-LT ResNet18 recipe benchmark pilot via Slurm
make e11-cifar-resnet-lt-muon-final-benchmark-results # submit the CIFAR-100-LT ResNet18 NS-Muon final-training benchmark pilot via Slurm
make e11-cifar-resnet-practical-muon-bridge-results # submit the ResNet practical Muon/AdamW trajectory bridge via Slurm
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
make e11-cifar-resnet-condition-proxy-results
make e11-cifar-resnet-fc-condition-results
make e11-cifar-resnet-tail-quality-results
make e11-cifar-resnet-imbalance-sweep-results
make e11-cifar-resnet-layer-jvp-tail-quality-results
make e11-cifar-resnet-layer-jvp-checkpoint-prediction-results
make e11-cifar-resnet-condition-score-heldout-architecture-results
make e11-cifar-resnet-condition-score-heldout-data-results
make e11-cifar-resnet-condition-score-heldout-eval
make e11-cifar-resnet-lt-standard-eval-results
make e11-cifar-resnet-lt-recipe-benchmark-results
make e11-cifar-resnet-lt-muon-final-benchmark-results
make e11-cifar-resnet-practical-muon-bridge-results
```

This submits `scripts/slurm/e11_cifar100_resnet_checkpoint_sweep.sbatch`, which
runs `scripts/e11_run_cifar100_resnet_checkpoint_sweep.py` on GPU across
ResNet18 warmup checkpoints. The current committed result reduces
single-checkpoint risk, but it should not be read as high-quality tail-predictor
evidence because the best pre-update tail accuracy is still low.

The final-layer condition target submits
`scripts/slurm/e11_cifar100_resnet_fc_condition_scatter.sbatch`, which runs
`scripts/e11_run_cifar100_resnet_fc_condition_scatter.py` on GPU. It measures
the downstream-aware final-layer proxy `nrank(G_H) / srank(H_T)` and compares
matched-head-gain final-layer-only drift.

The tail-quality target submits
`scripts/slurm/e11_cifar100_resnet_tail_quality_control.sbatch`, which reuses
the ResNet checkpoint-sweep runner with 300 tail-train examples per class. It is
a tail-rich control for the weak-tail-predictor objection, not a standard
long-tailed benchmark.

The imbalance-sweep target submits
`scripts/slurm/e11_cifar100_resnet_imbalance_sweep.sbatch`, which runs
`scripts/e11_run_cifar100_resnet_imbalance_sweep.py` on GPU. It keeps the head
split at 300 train examples per class and sweeps tail_train_per_class
10/30/100/300 over 3 seeds with the same matched-head-gain protocol. This is a
local tail-frequency robustness check, not a final optimizer benchmark.

The all-layer JVP target submits
`scripts/slurm/e11_cifar100_resnet_layer_jvp_tail_quality.sbatch`, which runs
`scripts/e11_run_cifar100_resnet_layer_jvp_tail_quality.py` on the tail-rich
5000-step ResNet checkpoint. It probes all 21 Conv/Linear matrix weights with
finite-difference unit JVP, matched-head-gain scaled JVP, and observed
layer-only drift.

The all-layer JVP checkpoint-transfer target submits
`scripts/slurm/e11_cifar100_resnet_layer_jvp_checkpoint_prediction.sbatch`,
which runs
`scripts/e11_run_cifar100_resnet_layer_jvp_checkpoint_prediction.py` over
tail-rich 2000/5000/10000-step ResNet checkpoints. It asks whether source
checkpoint layer scores predict held-out checkpoint observed layer drift. The
current result is a useful predictive boundary condition: source-observed drift
and an early-layer prior do transfer layer-risk ranking, while scaled-JVP
preserves the below-one threshold direction but not the ranking. The residual
version fits source-checkpoint observed drift against early-layer structure and
tests held-out residual risk: source observed residuals still transfer, while
scaled-JVP residuals remain inverted.

The registered condition-score held-out split targets submit
`scripts/slurm/e11_cifar100_resnet_condition_score_heldout_architecture.sbatch`
and `scripts/slurm/e11_cifar100_resnet_condition_score_heldout_data.sbatch`.
They run the same all-layer checkpoint-transfer probe on ResNet34/CIFAR-100-LT
and ResNet18/CIFAR-10-LT, writing raw held-out layer tables under
`results/e11_cifar100_resnet_condition_score_next/heldout_architecture` and
`results/e11_cifar100_resnet_condition_score_next/heldout_data`. These jobs
prepare the registered no-tuning held-out evidence; until the score gates are
evaluated on those tables, the P0 predictive-condition claim remains open.
After both Slurm jobs finish, `make e11-cifar-resnet-condition-score-heldout-eval`
applies the frozen `condition_score_v2_calibrated_residual` coefficients from
`results/e11_cifar100_resnet_condition_score_next/calibration_coefficients.csv`
to those held-out layer tables and writes the held-out gate report.

The standard long-tail reporting target submits
`scripts/slurm/e11_cifar100_resnet_lt_standard_eval.sbatch`, which runs
`scripts/e11_run_cifar100_resnet_lt_standard_eval.py` on an exponential
CIFAR-100-LT split with imbalance factor 100. It trains an AdamW ResNet18 and
reports many/medium/few balanced accuracy. This is a reporting baseline, not a
tuned benchmark, augmentation study, or Muon comparison.

The augmented recipe benchmark target submits
`scripts/slurm/e11_cifar100_resnet_lt_recipe_benchmark.sbatch`, which runs
`scripts/e11_run_cifar100_resnet_lt_recipe_benchmark.py` with 5 seeds and
5000 training steps per recipe. It compares augmented AdamW cross-entropy,
augmented AdamW with class-balanced loss, and augmented SGD-momentum. This is a
benchmark pilot for recipe sensitivity, not a final tuned leaderboard or Muon
comparison.

The NS-Muon final-training benchmark target submits
`scripts/slurm/e11_cifar100_resnet_lt_muon_final_benchmark.sbatch`, which runs
`scripts/e11_run_cifar100_resnet_lt_recipe_benchmark.py` with augmented AdamW
and two finite-Newton-Schulz Muon-style matrix-weight recipes. The current
3-seed result is a negative final-performance boundary for the tested Muon
recipes, not a proof that no Muon recipe can work.

The ResNet practical Muon bridge target submits
`scripts/slurm/e11_cifar100_resnet_practical_muon_bridge.sbatch`, which runs
`scripts/e11_run_cifar100_resnet_practical_muon_bridge.py` from the same
tail-rich 5000-step ResNet checkpoint. It samples AdamW and finite-step
NS-Muon matrix-weight trajectory states and reruns the matched-head-gain
direction diagnostic for `polar(G_t)`, `polar(M_t)`, and `NS(M_t)`.

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
- `discussion/e11_top_conference_gap_register.md`
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
- `discussion/e11_cifar100_resnet_condition_proxy_scatter.md`
- `discussion/e11_cifar100_resnet_fc_condition_scatter.md`
- `discussion/e11_cifar100_resnet_tail_quality_control.md`
- `discussion/e11_cifar100_resnet_imbalance_sweep.md`
- `discussion/e11_cifar100_resnet_layer_jvp_tail_quality.md`
- `discussion/e11_cifar100_resnet_layer_jvp_checkpoint_prediction.md`
- `discussion/e11_cifar100_resnet_condition_score_audit.md`
- `discussion/e11_cifar100_resnet_condition_score_protocol.md`
- `discussion/e11_cifar100_resnet_condition_score_next.md`
- `discussion/e11_cifar100_resnet_lt_standard_eval.md`
- `discussion/e11_cifar100_resnet_lt_recipe_benchmark.md`
- `discussion/e11_cifar100_resnet_lt_muon_final_benchmark.md`
- `discussion/e11_cifar100_resnet_practical_muon_bridge.md`
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
- `results/e11_cifar100_resnet_condition_proxy_scatter/summary.csv`
- `results/e11_cifar100_resnet_fc_condition_scatter/summary.csv`
- `results/e11_cifar100_resnet_fc_condition_scatter/condition_metrics.csv`
- `results/e11_cifar100_resnet_tail_quality_control/pair_summary.csv`
- `results/e11_cifar100_resnet_imbalance_sweep/pair_summary.csv`
- `results/e11_cifar100_resnet_imbalance_sweep/step_metrics.csv`
- `results/e11_cifar100_resnet_imbalance_sweep/layer_metrics.csv`
- `results/e11_cifar100_resnet_layer_jvp_tail_quality/overall_summary.csv`
- `results/e11_cifar100_resnet_layer_jvp_tail_quality/summary.csv`
- `results/e11_cifar100_resnet_layer_jvp_checkpoint_prediction/checkpoint_summary.csv`
- `results/e11_cifar100_resnet_layer_jvp_checkpoint_prediction/prediction_summary.csv`
- `results/e11_cifar100_resnet_condition_score_audit/raw_score_pairs.csv`
- `results/e11_cifar100_resnet_condition_score_audit/raw_score_summary.csv`
- `results/e11_cifar100_resnet_condition_score_audit/residual_score_pairs.csv`
- `results/e11_cifar100_resnet_condition_score_audit/residual_score_summary.csv`
- `results/e11_cifar100_resnet_condition_score_protocol/score_registry.csv`
- `results/e11_cifar100_resnet_condition_score_protocol/split_registry.csv`
- `results/e11_cifar100_resnet_condition_score_protocol/acceptance_gates.csv`
- `results/e11_cifar100_resnet_condition_score_next/score_pairs.csv`
- `results/e11_cifar100_resnet_condition_score_next/score_summary.csv`
- `results/e11_cifar100_resnet_condition_score_next/calibration_coefficients.csv`
- `results/e11_cifar100_resnet_condition_score_next/gate_report.csv`
- `results/e11_top_conference_gap_register/gap_register.csv`
- `results/e11_cifar100_resnet_lt_standard_eval/summary.csv`
- `results/e11_cifar100_resnet_lt_standard_eval/class_summary.csv`
- `results/e11_cifar100_resnet_lt_recipe_benchmark/summary.csv`
- `results/e11_cifar100_resnet_lt_recipe_benchmark/pair_summary.csv`
- `results/e11_cifar100_resnet_lt_muon_final_benchmark/summary.csv`
- `results/e11_cifar100_resnet_lt_muon_final_benchmark/pair_summary.csv`
- `results/e11_cifar100_resnet_practical_muon_bridge/summary.csv`
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
- `figures/e11_cifar100_resnet_condition_proxy_scatter/cifar100_resnet_condition_proxy_scatter.png`
- `figures/e11_cifar100_resnet_fc_condition_scatter/cifar100_resnet_fc_condition_scatter.png`
- `figures/e11_cifar100_resnet_tail_quality_control/cifar100_resnet_checkpoint_sweep.png`
- `figures/e11_cifar100_resnet_imbalance_sweep/cifar100_resnet_imbalance_sweep.png`
- `figures/e11_cifar100_resnet_layer_jvp_tail_quality/cifar100_resnet_layer_jvp_tail_quality.png`
- `figures/e11_cifar100_resnet_layer_jvp_checkpoint_prediction/cifar100_resnet_layer_jvp_checkpoint_prediction.png`
- `figures/e11_cifar100_resnet_condition_score_audit/cifar100_resnet_condition_score_audit.png`
- `figures/e11_cifar100_resnet_lt_standard_eval/cifar100_resnet_lt_standard_eval.png`
- `figures/e11_cifar100_resnet_lt_recipe_benchmark/cifar100_resnet_lt_recipe_benchmark.png`
- `figures/e11_cifar100_resnet_lt_muon_final_benchmark/cifar100_resnet_lt_recipe_benchmark.png`
- `figures/e11_cifar100_resnet_practical_muon_bridge/cifar100_resnet_practical_muon_bridge.png`
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
   - A ResNet rank-side proxy scatter over 40 seed/checkpoint points gives positive correlation between mean gradient nuclear rank and log drift ratio, Pearson about `0.7594 [0.6657, 0.8807]`; this supports the caveat that `nrank(G_H)` alone is not the downstream-aware condition.
   - A ResNet final-layer downstream-aware condition diagnostic over 40 seed/checkpoint points has weakest mean `nrank(G_H) / srank(H_T)` score about `6.566`, all points favoring spectral, and worst final-layer-only squared drift ratio about `0.2391 [0.2201, 0.2597]`.
   - A tail-rich ResNet control with 300 tail-train examples per class reaches best pre-update tail accuracy about `0.3739 [0.3454, 0.4024]` and still keeps the worst squared drift ratio below 1, about `0.7292 [0.6923, 0.768]`.
   - A CIFAR-100-LT ResNet18 imbalance sweep over tail_train_per_class 10/30/100/300 keeps spectral/Frobenius squared drift ratio below 1 in every setting; worst CI upper endpoint is 0.936 at tail_train_per_class=100, and best pre-update tail accuracy is 0.3297 [0.2954, 0.3639] at tail_train_per_class=300. Tail-loss evidence is mixed, so this remains a local drift result.
   - An all-layer ResNet finite-difference JVP tail-quality diagnostic covers 21 Conv/Linear weights and 210 paired layer/seed points; observed squared drift ratio is about `0.2011 [0.1845, 0.2192]`, scaled-JVP ratio is about `0.065 [0.06008, 0.07031]`, and every per-layer observed CI upper endpoint is below 1.
   - An all-layer ResNet JVP checkpoint-transfer benchmark covers 3 tail-rich checkpoints and 6 directed checkpoint-transfer pairs; source-observed positive-control Spearman is about `0.981 [0.9739, 0.988]` and early-layer prior Spearman is about `0.9126 [0.9029, 0.9222]`, but the scaled-JVP predictor has below-one threshold accuracy `1` and held-out layer-risk Spearman about `-0.3203 [-0.3562, -0.2845]`. After source-fit early-layer residualization, observed residual Spearman is about `0.9403 [0.9216, 0.959]`, while scaled-JVP residual Spearman is about `-0.4872 [-0.5373, -0.4372]`, so the current score is not yet a positive layer-ranking predictor even beyond depth structure.
   - A candidate condition-score audit over the same checkpoint-transfer tables confirms this boundary: the best simple source-only composite, early-minus-scaled-JVP, has held-out Spearman about `0.8656 [0.8578, 0.8734]`, below the early-layer prior, and scaled-JVP remains inverted in the residual score audit.
   - A standard CIFAR-100-LT ResNet18 reporting baseline (IF=100, 10 AdamW seeds, no augmentation/tuning) gives many/medium/few balanced accuracy `0.3665 [0.3489, 0.3841]`, `0.1036 [0.09138, 0.1158]`, and `0.0129 [0.009351, 0.01645]`. This supplies a standard classification reporting surface, not a tuned benchmark or Muon comparison.
   - An augmented CIFAR-100-LT ResNet18 recipe benchmark pilot (5 seeds, 5000 steps) gives SGD-momentum all/few balanced accuracy `0.4105 [0.4044, 0.4166]` and `0.1047 [0.09389, 0.1156]`; the few-group diff versus augmented AdamW is `0.0194 [0.005581, 0.03322]`. Class-balanced AdamW is worse in this pilot, with few-group diff `-0.0114 [-0.0215, -0.001304]`.
   - A CIFAR-100-LT ResNet18 NS-Muon final-training pilot (3 seeds, 5000 steps) is negative: lr=1e-4 all/few balanced accuracy `0.1265 [0.1218, 0.1312]` / `0.0008889 [-0.0006533, 0.002431]`; paired all/few diff vs AdamW-aug `-0.2348 [-0.2395, -0.23]` / `-0.08767 [-0.09948, -0.07585]`; lr=3e-5 is worse.
   - A ResNet practical Muon trajectory bridge from the same tail-rich checkpoint gives `NS(M_t)` squared drift ratio about `0.8628 [0.8154, 0.913]` on AdamW-sampled states and `0.7247 [0.676, 0.777]` on NS-Muon-sampled states.
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
- `e11_condition_geometry/cifar100_resnet_tail.py`: CIFAR-100/CIFAR-10 ResNet18/ResNet34 matched-head-gain diagnostic with Conv/Linear matrix-weight interventions.
- `e11_condition_geometry/plots/`: static figure generation.
- `e11_condition_geometry/reporting.py`: shared Markdown/report helpers.
- `scripts/e11_run_*.py`: experiment runners.
- `scripts/e11_run_cifar100_lt_one_step.py`: CIFAR-100-LT two-layer MLP runner.
- `scripts/e11_run_cifar100_resnet_one_step.py`: CIFAR-100-LT ResNet18 runner called by the Slurm wrapper.
- `scripts/e11_run_cifar100_resnet_checkpoint_sweep.py`: ResNet18 checkpoint-quality sweep for the top-conference upgrade path.
- `scripts/e11_run_cifar100_resnet_condition_proxy_scatter.py`: rank-side proxy scatter generated from the ResNet checkpoint-sweep CSVs.
- `scripts/e11_run_cifar100_resnet_fc_condition_scatter.py`: final-layer downstream-aware condition diagnostic for `fc.weight` on ResNet18 checkpoints.
- `scripts/e11_run_cifar100_resnet_imbalance_sweep.py`: CIFAR-100-LT ResNet18 tail-count imbalance sweep for local matched-head-gain drift robustness.
- `scripts/e11_run_cifar100_resnet_layer_jvp_tail_quality.py`: all-layer ResNet finite-difference JVP diagnostic at the tail-rich checkpoint.
- `scripts/e11_run_cifar100_resnet_layer_jvp_checkpoint_prediction.py`: all-layer ResNet JVP checkpoint-transfer benchmark across tail-rich checkpoints.
- `scripts/e11_write_cifar100_resnet_condition_score_audit.py`: offline candidate condition-score audit generated from checkpoint-transfer tables.
- `scripts/e11_evaluate_cifar100_resnet_condition_score_heldouts.py`: frozen-coefficient condition-score evaluator for registered held-out architecture/data splits.
- `scripts/e11_run_cifar100_resnet_lt_standard_eval.py`: standard CIFAR-100-LT ResNet18 many/medium/few reporting baseline.
- `scripts/e11_run_cifar100_resnet_lt_recipe_benchmark.py`: augmented CIFAR-100-LT ResNet18 recipe benchmark pilot with AdamW, class-balanced AdamW, SGD-momentum, and optional NS-Muon final-training recipes.
- `scripts/e11_run_cifar100_resnet_practical_muon_bridge.py`: ResNet practical Muon/AdamW trajectory-state bridge from the tail-rich checkpoint.
- `scripts/slurm/e11_cifar100_resnet_one_step.sbatch`: GPU/Slurm submission wrapper for the ResNet18 diagnostic.
- `scripts/slurm/e11_cifar100_resnet_one_step_rho002.sbatch`: GPU/Slurm submission wrapper for the smaller-head-gain ResNet18 check.
- `scripts/slurm/e11_cifar100_resnet_checkpoint_sweep.sbatch`: GPU/Slurm submission wrapper for the checkpoint-quality sweep.
- `scripts/slurm/e11_cifar100_resnet_fc_condition_scatter.sbatch`: GPU/Slurm submission wrapper for the final-layer condition diagnostic.
- `scripts/slurm/e11_cifar100_resnet_tail_quality_control.sbatch`: GPU/Slurm submission wrapper for the tail-rich checkpoint-quality control.
- `scripts/slurm/e11_cifar100_resnet_imbalance_sweep.sbatch`: GPU/Slurm submission wrapper for the ResNet18 tail-count imbalance sweep.
- `scripts/slurm/e11_cifar100_resnet_layer_jvp_tail_quality.sbatch`: GPU/Slurm submission wrapper for the all-layer ResNet JVP tail-quality diagnostic.
- `scripts/slurm/e11_cifar100_resnet_layer_jvp_checkpoint_prediction.sbatch`: GPU/Slurm submission wrapper for the all-layer ResNet JVP checkpoint-transfer benchmark.
- `scripts/slurm/e11_cifar100_resnet_condition_score_heldout_architecture.sbatch`: GPU/Slurm submission wrapper for the registered ResNet34 held-out architecture condition-score split.
- `scripts/slurm/e11_cifar100_resnet_condition_score_heldout_data.sbatch`: GPU/Slurm submission wrapper for the registered CIFAR-10-LT held-out data condition-score split.
- `scripts/slurm/e11_cifar100_resnet_lt_standard_eval.sbatch`: GPU/Slurm submission wrapper for the standard CIFAR-100-LT ResNet18 reporting baseline.
- `scripts/slurm/e11_cifar100_resnet_lt_recipe_benchmark.sbatch`: GPU/Slurm submission wrapper for the augmented ResNet18 recipe benchmark pilot.
- `scripts/slurm/e11_cifar100_resnet_lt_muon_final_benchmark.sbatch`: GPU/Slurm submission wrapper for the negative NS-Muon final-training benchmark pilot.
- `scripts/e11_write_*.py`: generated discussion and paper-facing artifacts.
- `tests/`: smoke and diagnostic tests.

## Current Publication Gaps

The current evidence is consistent with a focused local-geometry paper. It is not yet enough for a broad optimizer-performance paper.

The generated next-evidence matrix is `discussion/e11_top_conference_gap_register.md`, backed by `results/e11_top_conference_gap_register/gap_register.csv`. The P0 condition-score protocol is pre-registered in `discussion/e11_cifar100_resnet_condition_score_protocol.md`, with its locked ResNet18 checkpoint-split v2 analysis in `discussion/e11_cifar100_resnet_condition_score_next.md`.

Most important next steps:

1. Extend the new standard CIFAR-100-LT ResNet18 many/medium/few reporting baseline, augmented recipe pilot, negative NS-Muon final-training pilot, and local tail-count imbalance sweep into a tuned benchmark protocol with a wider grid, class-balanced samplers, better Muon schedules, and larger long-tail datasets; the current pilots are useful benchmark context, not a competitive optimizer result.
2. Improve the all-layer ResNet JVP predictive condition benchmark: the held-out checkpoint-transfer run now shows source-observed and early-layer controls transfer, and observed residuals transfer after source-fit depth adjustment, but the current scaled-JVP score has negative raw and residual layer-risk ranking transfer. ResNet34 and CIFAR-10-LT held-out Slurm entry points are now registered, but the jobs and score gates still need to run before any P0 predictive-condition claim is defensible.
3. Extend the current fixed-checkpoint, short-trajectory, small practical-training, and negative ResNet final-training Muon diagnostics into a full practical Muon benchmark with schedules, checkpoint distributions, final tail metrics, and hyperparameter robustness.
4. Add larger-architecture layerwise JVP/decomposition diagnostics if the detailed scaled-head-gain mechanism is meant to survive beyond the current small MLP explanation.
5. Keep separating function-drift evidence from tail loss, margin, accuracy, and final optimizer performance.
