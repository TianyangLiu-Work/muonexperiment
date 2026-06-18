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
python3 scripts/e11_evaluate_condition_score_fresh_protocol.py
python3 scripts/e11_write_condition_score_failure_mechanism_audit.py
python3 scripts/e11_write_condition_score_v4_protocol.py
sbatch scripts/slurm/e11_cifar100_resnet_condition_score_v4_validation_cifar100_rotated.sbatch
python3 scripts/e11_freeze_condition_score_v4_validation.py
sbatch scripts/slurm/e11_cifar100_resnet_condition_score_v4_architecture_wide_resnet50_2.sbatch
sbatch scripts/slurm/e11_cifar100_resnet_condition_score_v4_data_cifar10_mixed.sbatch
python3 scripts/e11_evaluate_condition_score_v4_finals.py
python3 scripts/e11_write_condition_score_v4_failure_mechanism_audit.py
python3 scripts/e11_write_matrix_block_theorem_proof.py
python3 scripts/e11_write_matrix_block_tightness_audit.py
python3 scripts/e11_write_theory_proof_obligation_register.py
python3 scripts/e11_write_condition_score_v5_theory_protocol.py
python3 scripts/e11_write_condition_score_v5_theory_to_score_map.py
python3 scripts/e11_write_condition_score_ablation.py
sbatch scripts/slurm/e11_cifar100_resnet_condition_score_v5_validation_mod4_partition.sbatch
python3 scripts/e11_freeze_condition_score_v5_validation.py
sbatch scripts/slurm/e11_cifar100_resnet_condition_score_v5_architecture_resnext50_32x4d.sbatch
sbatch scripts/slurm/e11_cifar100_resnet_condition_score_v5_data_cifar10_cross.sbatch
python3 scripts/e11_evaluate_condition_score_v5_finals.py
python3 scripts/e11_write_condition_score_v5_final_power_audit.py
python3 scripts/e11_write_condition_score_v5_final_interpretation_plan.py
python3 scripts/e11_write_condition_score_v5_reviewer_failure_response.py
python3 scripts/e11_write_condition_score_v5_direction_guardrail_failure_audit.py
python3 scripts/e11_write_natural_negative_phase1_interim_synthesis.py
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
make e11-cifar-resnet-condition-score-fresh-architecture-results # submit the fresh ResNet50 condition-score architecture split via Slurm
make e11-cifar-resnet-condition-score-fresh-data-results # submit the fresh CIFAR-10 alternate-partition condition-score data split via Slurm
make e11-cifar-resnet-condition-score-fresh-eval # evaluate frozen fresh condition-score gates after fresh Slurm jobs finish
make e11-cifar-resnet-condition-score-v4-validation-freeze # freeze or block the v4 scalar aggregation after the validation split
make e11-cifar-resnet-condition-score-v4-final-eval # evaluate frozen v4 final gates after both unspent final Slurm jobs finish
make e11-cifar-resnet-condition-score-v4-failure-audit # localize the frozen v4 CIFAR-10 mixed final failure mechanism
make e11-matrix-block-theorem-proof # write the matched-gain theorem/proof contract and sandwich rank derivation
make e11-matrix-block-tightness-audit # verify theorem tightness, ratio identity, equality boundary, and degeneracy caveats
make e11-theory-proof-obligation-register # map theorem assumptions, claim scope, and proof obligations before broad claims
make e11-cifar-resnet-condition-score-v5-theory-protocol # write the v5 transport-normalized theory/score contract
make e11-cifar-resnet-condition-score-v5-theory-to-score-map # map the v5 theorem terms to score features, leakage boundaries, and falsifiable gates
make e11-condition-score-ablation # write the spent-evidence score-axis ablation separating direction, residual, nuisance, and transport terms
make e11-cifar-resnet-condition-score-v5-validation-results # submit the v5 validation-only CIFAR-100-LT mod-4 partition via Slurm
make e11-cifar-resnet-condition-score-v5-validation-freeze # freeze or block the v5 transport-normalized score after validation
make e11-cifar-resnet-condition-score-v5-architecture-results # submit the v5 ResNeXt50-32x4d final architecture split via Slurm
make e11-cifar-resnet-condition-score-v5-data-results # submit the v5 CIFAR-10 cross-partition final data split via Slurm
make e11-cifar-resnet-condition-score-v5-final-eval # evaluate frozen v5 final gates after both unspent final Slurm jobs finish
make e11-cifar-resnet-condition-score-v5-final-power-audit # pre-output detectable-effect audit for v5 final residual-Spearman gates
make e11-cifar-resnet-condition-score-v5-final-interpretation-plan # lock the v5 final outcome-to-claim state machine before outputs exist
make e11-cifar-resnet-condition-score-v5-reviewer-failure-response # map v5 final pass/fail modes to reviewer-safe claim downgrades
make e11-cifar-resnet-condition-score-v5-direction-guardrail-failure-audit # diagnose completed v5 final boundary failures across residual ranking and direction threshold
make e11-cifar-resnet-lt-standard-eval-results # submit the standard CIFAR-100-LT ResNet18 many/medium/few reporting baseline via Slurm
make e11-cifar-resnet-lt-recipe-benchmark-results # submit the augmented CIFAR-100-LT ResNet18 recipe benchmark pilot via Slurm
make e11-cifar-resnet-lt-muon-final-benchmark-results # submit the CIFAR-100-LT ResNet18 NS-Muon final-training benchmark pilot via Slurm
make e11-cifar-resnet-lt-tuned-benchmark-protocol # register validation/final splits, tuned baselines, Muon grids, and benchmark claim gates
make e11-cifar-resnet-lt-tuned-benchmark-settings # write the executable 164-setting tuned validation grid registry
make e11-cifar-resnet-lt-tuned-benchmark-validation-results # submit the tuned validation grid via Slurm array
make e11-cifar-resnet-lt-tuned-benchmark-selection # select final recipes from completed validation summaries without touching final seeds
make e11-cifar-resnet-lt-tuned-benchmark-power-audit # lock tuned final seed MDE, Holm family, and all-class guardrail before final outputs
make e11-cifar-resnet-lt-tuned-benchmark-variance-prior-audit # calibrate tuned benchmark MDE assumptions from validation and spent-pilot variance
make e11-cifar-resnet-lt-tuned-benchmark-refresh-firewall # freeze legal sequential refresh actions and forbidden metric-dependent shortcuts
make e11-cifar-resnet-lt-tuned-benchmark-protocol-seal # hash-seal frozen tuned benchmark protocol surfaces after partial validation exposure
make e11-cifar-resnet-lt-tuned-benchmark-fairness-audit # audit tuned baseline coverage, Muon candidate-budget disclosure, and shared seed/metric rules
make e11-cifar-resnet-lt-tuned-benchmark-final-analysis-plan # pre-register final paired tests, Holm adjustment, reporting schema, and claim states
make e11-cifar-resnet-lt-tuned-benchmark-final-execution-plan # write a no-side-effect final-claim execution contract and gate-checked Slurm wrapper plan
make e11-cifar-resnet-lt-tuned-benchmark-final-eval # evaluate final paired seeds with fixed Holm tests and claim gates after final outputs exist
make e11-cifar-resnet-lt-tuned-benchmark-final-launch-audit # compute queue-aware final-claim launch readiness without submitting
make e11-cifar-resnet-lt-tuned-benchmark-final-safe-submit # submit final-claim jobs only if every final launch gate passes
make e11-cifar-resnet-lt-tuned-benchmark-slurm-plan # write a chunked no-side-effect Slurm launch plan for the 164-setting validation grid
make e11-cifar-resnet-lt-tuned-benchmark-launch-audit # compute the current queue-aware validation launch decision without submitting
make e11-cifar-resnet-lt-tuned-benchmark-safe-submit # submit the largest safe validation subchunk under MaxSubmitJobsPerUser
make e11-cifar-resnet-practical-muon-bridge-results # submit the ResNet practical Muon/AdamW trajectory bridge via Slurm
make e11-natural-head-tail-boundary-audit # scan committed natural matched-head-gain sweeps for primary drift and secondary boundary cases
make e11-natural-negative-search-protocol # register fresh natural negative-search space, metrics, stopping rules, and claim gates
make e11-natural-negative-search-phase1-power-audit # compute the phase1 detectable-effect and interpretation boundary
make e11-natural-negative-search-phase1-settings # write settings-only registries for all registered phase1 natural negative-search settings
make e11-natural-negative-search-phase1-results # submit the registered phase1 natural negative-search settings via Slurm
make e11-natural-negative-search-phase1-eval # evaluate Holm-adjusted phase1 decisions after fresh metric outputs exist
make e11-natural-negative-search-phase1-interim-synthesis # summarize complete phase1 coverage and caveated finite-null boundaries
make e11-natural-negative-search-phase2-settings # freeze the ResNet34 held-out architecture phase2 settings registry
make e11-natural-negative-search-phase2-results # submit the registered ResNet34 phase2 held-out architecture settings via Slurm
make e11-natural-negative-search-phase2-eval # evaluate the fixed 8-setting phase2 family after metric outputs exist
make e11-natural-negative-search-phase2-power-audit # compute the phase2 detectable-effect and outcome-state boundary
make e11-heldout-generality-audit # separate positive ResNet18 support from caveated held-out architecture/data boundaries
make e11-bold-conjecture-register # generate the bold-conjecture/careful-verification register
make e11-muon-state-distribution-contract # generate the Muon state-distribution/practical-performance boundary contract
make e11-appendix-results  # current appendix/guardrail probes
make e11-all-results       # main plus appendix/guardrail result generation
make e11-paper-assets      # regenerate current head-to-tail paper Markdown/TeX artifacts
make e11-top-conference-claim-decision-audit # generate the paper-level supportable/blocked claim and rebuttal-readiness contract
make e11-manuscript-claim-trace # map top-conference claim decisions to main.tex anchors and blocked-wording checks
make e11-mechanism-referee-audit # regenerate the adversarial mechanism/referee alternative-explanation audit
make e11-guardrail-assets  # regenerate legacy condition-geometry guardrail notes
make e11-all-assets        # regenerate current paper artifacts plus legacy guardrail notes
make e11-paper-pdf         # rebuild paper/specgrad_activation_paper/main.pdf and two_page.pdf
make e11-submission-repro-audit # audit toolchain availability, PDF hashes, source hashes, and clean-checkout gates
make e11-clean-worktree-replay-audit # replay e11-check from a detached clean tracked-source worktree
make e11-pdf-render-boundary-audit # audit rendered PDF header/hash/source-trace coverage and text-extraction tool gaps
make e11-artifact-review-packet # regenerate the artifact-review command, gate, local-state, and reviewer-response packet
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
make e11-cifar-resnet-condition-score-fresh-architecture-results
make e11-cifar-resnet-condition-score-fresh-data-results
make e11-cifar-resnet-condition-score-fresh-eval
make e11-cifar-resnet-condition-score-v4-validation-freeze
make e11-cifar-resnet-condition-score-v4-final-eval
make e11-cifar-resnet-condition-score-v4-failure-audit
make e11-matrix-block-theorem-proof
make e11-matrix-block-tightness-audit
make e11-theory-proof-obligation-register
make e11-cifar-resnet-condition-score-v5-theory-protocol
make e11-cifar-resnet-condition-score-v5-theory-to-score-map
make e11-condition-score-ablation
make e11-cifar-resnet-condition-score-v5-validation-results
make e11-cifar-resnet-condition-score-v5-validation-freeze
make e11-cifar-resnet-condition-score-v5-architecture-results
make e11-cifar-resnet-condition-score-v5-data-results
make e11-cifar-resnet-condition-score-v5-final-eval
make e11-cifar-resnet-condition-score-v5-final-power-audit
make e11-cifar-resnet-condition-score-v5-final-interpretation-plan
make e11-cifar-resnet-condition-score-v5-reviewer-failure-response
make e11-cifar-resnet-lt-standard-eval-results
make e11-cifar-resnet-lt-recipe-benchmark-results
make e11-cifar-resnet-lt-muon-final-benchmark-results
make e11-cifar-resnet-lt-tuned-benchmark-protocol
make e11-cifar-resnet-lt-tuned-benchmark-settings
make e11-cifar-resnet-lt-tuned-benchmark-validation-results
make e11-cifar-resnet-lt-tuned-benchmark-selection
make e11-cifar-resnet-lt-tuned-benchmark-power-audit
make e11-cifar-resnet-lt-tuned-benchmark-variance-prior-audit
make e11-cifar-resnet-lt-tuned-benchmark-final-analysis-plan
make e11-cifar-resnet-lt-tuned-benchmark-final-execution-plan
make e11-cifar-resnet-lt-tuned-benchmark-final-eval
make e11-cifar-resnet-lt-tuned-benchmark-final-launch-audit
make e11-cifar-resnet-lt-tuned-benchmark-final-safe-submit
make e11-cifar-resnet-lt-tuned-benchmark-slurm-plan
make e11-cifar-resnet-lt-tuned-benchmark-launch-audit
make e11-cifar-resnet-lt-tuned-benchmark-safe-submit
make e11-cifar-resnet-practical-muon-bridge-results
make e11-natural-head-tail-boundary-audit
make e11-natural-negative-search-protocol
make e11-natural-negative-search-phase1-power-audit
make e11-natural-negative-search-phase1-settings
make e11-natural-negative-search-phase1-results
make e11-natural-negative-search-phase1-eval
make e11-natural-negative-search-phase2-settings
make e11-natural-negative-search-phase2-results
make e11-natural-negative-search-phase2-eval
make e11-natural-negative-search-phase2-power-audit
make e11-heldout-generality-audit
make e11-bold-conjecture-register
make e11-muon-state-distribution-contract
make e11-camera-ready-package-audit
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
`results/e11_cifar100_resnet_condition_score_next/heldout_data`. ResNet34 and CIFAR-10-LT held-out Slurm entry points are now registered and have produced
the current no-tuning boundary evidence. After both Slurm jobs finish,
`make e11-cifar-resnet-condition-score-heldout-eval`
applies the frozen `condition_score_v2_calibrated_residual` coefficients from
`results/e11_cifar100_resnet_condition_score_next/calibration_coefficients.csv`
to those held-out layer tables and writes the held-out gate report. The
registered held-out condition-score evaluation now fails the P0
residual-ranking gates: ResNet34 held-out architecture primary residual
Spearman is `0.1992 [-0.07515, 0.4735]`, and CIFAR-10-LT held-out data primary
residual Spearman is `-0.6771 [-0.7011, -0.653]`. The below-one threshold
direction still passes on both splits, and the legacy scaled-JVP ratio on
CIFAR-10-LT has residual Spearman `0.6219 [0.6013, 0.6425]`. This supports a
narrower directional guardrail, not a held-out layer-risk ranking claim.

The fresh condition-score protocol is generated at
`discussion/e11_condition_score_fresh_protocol.md`, backed by
`results/e11_condition_score_fresh_protocol/*`. It quarantines the spent
ResNet34 and original CIFAR-10 held-outs, freezes the zero-fit scaled-JVP and
nested JVP-residual candidates, and registers fresh final splits: ResNet50
CIFAR-100-LT and a CIFAR-10 alternate head/tail partition. The runnable Slurm
entry points are
`scripts/slurm/e11_cifar100_resnet_condition_score_fresh_architecture_resnet50.sbatch`
and
`scripts/slurm/e11_cifar100_resnet_condition_score_fresh_data_cifar10_alt.sbatch`.
The frozen evaluator is `scripts/e11_evaluate_condition_score_fresh_protocol.py`;
after the fresh jobs finish it writes the gate report at
`discussion/e11_condition_score_fresh_evaluation.md` and
`results/e11_condition_score_fresh_protocol/fresh_score_evaluation`. The
current fresh v3 result is still `not_ready`: the CIFAR-10 alternate partition
passes residual ranking (`0.642 [0.6339, 0.6501]`), but the ResNet50
architecture split fails and reverses (`-0.8157 [-0.8652, -0.7661]`) despite
passing the below-one threshold direction. The diagnostic obstruction audit is
`discussion/e11_condition_score_failure_mechanism_audit.md`, backed by
`results/e11_condition_score_failure_mechanism_audit/*`; it shows the ResNet50
residual risk concentrates in bottleneck layer2/layer3 terms while the primary
scaled-JVP score ranks stem/classifier/layer4 terms highest.

The next unspent condition-score attempt is registered at
`discussion/e11_condition_score_v4_protocol.md`, backed by
`results/e11_condition_score_v4_protocol/*`. V4 quarantines all v2/v3 final
splits, separates the direction ratio from residual-amplitude and architecture
transport axes, and adds `wide_resnet50_2` as an unspent architecture final
split. Its runnable entry points are
`scripts/slurm/e11_cifar100_resnet_condition_score_v4_validation_cifar100_rotated.sbatch`,
`scripts/slurm/e11_cifar100_resnet_condition_score_v4_architecture_wide_resnet50_2.sbatch`,
and `scripts/slurm/e11_cifar100_resnet_condition_score_v4_data_cifar10_mixed.sbatch`.
The committed validation-freeze boundary is generated by
`scripts/e11_freeze_condition_score_v4_validation.py`, which writes
`discussion/e11_condition_score_v4_validation_freeze.md` and
`results/e11_condition_score_v4_protocol/validation_score_freeze/*`.
The validation split now exists at
`results/e11_condition_score_v4_protocol/validation_cifar100_rotated/*` and
`discussion/e11_condition_score_v4_validation_cifar100_rotated.md`; the freeze
artifact is `pass` and selects
`condition_score_v4_two_axis_amplitude_minus_direction` with validation residual
Spearman `0.3758 [0.2759, 0.4756]` and direction-axis threshold accuracy `1`.
V4 is not a positive result yet; it now has a validation-frozen scalar
aggregation before either unspent final split, and the final evaluation is now
a useful negative boundary rather than a P0 pass.
The WideResNet50-2 final architecture split passes residual ranking with primary Spearman `0.6449 [0.5122, 0.7776]`, but the CIFAR-10 mixed final data split fails with primary Spearman `-0.6937 [-0.7129, -0.6744]`; both direction guardrails pass with threshold accuracy `1`.
The final evaluator is `scripts/e11_evaluate_condition_score_v4_finals.py`;
its current gate report is `not_ready` because both unspent final splits must
pass residual, direction, and reporting gates before any P0 claim.
The diagnostic failure audit is
`discussion/e11_condition_score_v4_failure_mechanism_audit.md`, backed by
`results/e11_condition_score_v4_failure_mechanism_audit/*` and
`figures/e11_condition_score_v4_failure_mechanism_audit/v4_cifar10_mixed_reversal_top5.png`.
It records `V4-O2-data-partition-reversal`: direction is not the failure on
the CIFAR-10 mixed final split, because the direction axis remains positively
ordered (`0.611 [0.5886, 0.6334]`) while the amplitude/depth side reverses
(`-0.6766 [-0.6903, -0.663]`) and drives the frozen scalar aggregation negative.
The matrix-block theorem proof contract is
`discussion/e11_matrix_block_theorem_proof.md`, backed by
`results/e11_matrix_block_theorem_proof/*`. It is the machine-checkable theory
artifact for the matched-head-gain bound, sandwich sensitivity lemma,
Frobenius/spectral coefficient derivation, nondegenerate tail block condition,
and `nrank(G_H) > ssrank(B_T,A_T)` claim boundary already written in the paper
appendix.
The companion tightness audit is
`discussion/e11_matrix_block_tightness_audit.md`, backed by
`results/e11_matrix_block_tightness_audit/*`. It checks exact diagonal
singular-spectrum witnesses for the Frobenius numerator, the spectral sandwich
numerator, the ratio identity
`I_spectral / I_frobenius = ssrank(B_T,A_T) / nrank(G_H)`, the equality
boundary, the Frobenius-favored boundary, and the degenerate-tail caveat.
The theory proof-obligation register is
`discussion/e11_theory_proof_obligation_register.md`, backed by
`results/e11_theory_proof_obligation_register/*`. It is a top-conference claim
scope checklist, not a new empirical result: it maps the local linearization,
matrix-block boundary, Muon approximation scope, v5 transport score, natural
falsification search, and optimizer-performance separation to formal objects,
required assumptions, decisive gates, and forbidden wording.
The next registered theory step is
`discussion/e11_condition_score_v5_theory_protocol.md`, backed by
`results/e11_condition_score_v5_theory_protocol/*`. It is not a positive P0
result; it turns the v4 amplitude/depth data-partition reversal into a
transport-normalized score contract, quarantines every v2/v3/v4 final row from
score fitting and selection, and registers new unspent entrypoints for a
validation-only CIFAR-100-LT mod-4 partition, a ResNeXt50-32x4d architecture
final split, and a CIFAR-10 cross-partition final data split.
The theorem-to-measurement bridge is
`discussion/e11_condition_score_v5_theory_to_score_map.md`, backed by
`results/e11_condition_score_v5_theory_to_score_map/*`. It states the
transport-stable sandwich residual proposition, maps theorem terms to score
features, and records the ablation and falsifiable prediction matrix that the
v5 validation/final splits must satisfy before any predictive-condition claim.
The standalone score-axis ablation audit is
`discussion/e11_condition_score_ablation.md`, backed by
`results/e11_condition_score_ablation/*`. It is CPU-only and uses spent
v2/v3/v4 rows only for diagnostic obstruction: it separates the direction
guardrail, raw-amplitude reversal, early-depth nuisance baseline, and
transport-normalized v5 validation candidate without fitting or reselecting on
any final row.
The executable freeze boundary is
`discussion/e11_condition_score_v5_validation_freeze.md`, backed by
`results/e11_condition_score_v5_protocol/validation_score_freeze/*`. The v5
validation-only mod-4 split is now generated, and the freeze evaluator selects
`condition_score_v5_transport_normalized_amplitude_minus_direction` with
validation residual Spearman `0.645 [0.5898, 0.7002]`. The direction guardrail
passes, no final outputs existed before freeze, and both registered final splits
were later evaluated under the frozen score without refit.
The pre-registered final evaluator is
`discussion/e11_condition_score_v5_final_evaluation.md`, backed by
`results/e11_condition_score_v5_protocol/final_score_evaluation/*`; both
registered final splits are now generated. The ResNeXt50-32x4d final
architecture split passes residual Spearman but fails the direction-threshold
guardrail, while the CIFAR-10 cross-partition split fails residual Spearman but
passes the direction threshold. The P0 gate remains `not_ready`. It applies the
frozen validation-selected score without refitting or reselection.
The pre-output final power audit is
`discussion/e11_condition_score_v5_final_power_audit.md`, backed by
`results/e11_condition_score_v5_protocol/final_power_audit/*`. It fixes the
reference point count, transfer-pair count, Fisher-z resolution, and
mean-Spearman MDE before final rows exist; the observed final failures are
therefore interpreted under the audited detectable-effect scale rather than by
post-hoc threshold movement.
The pre-output interpretation lock is
`discussion/e11_condition_score_v5_final_interpretation_plan.md`, backed by
`results/e11_condition_score_v5_protocol/final_interpretation_plan/*`. It fixes
the outcome-to-claim state machine before the final layer tables existed, including
the allowed positive claim, architecture/data transport-boundary cases,
direction-guardrail failures, baseline-dominance failures, and no-retuning
leakage lock.
The reviewer failure response is
`discussion/e11_condition_score_v5_reviewer_failure_response.md`, backed by
`results/e11_condition_score_v5_protocol/reviewer_failure_response/*`. It is a
completed-final claim-downgrade plan for top-conference review: the active
architecture direction failure, CIFAR-10 residual reversal, and P0-not-ready
state map to allowed wording, forbidden wording, blocking status, and next
evidence without refitting, reselecting, retuning, or repairing the frozen score
after final rows.
The direction-guardrail failure audit is
`discussion/e11_condition_score_v5_direction_guardrail_failure_audit.md`, backed
by
`results/e11_condition_score_v5_protocol/direction_guardrail_failure_audit/*`.
It diagnoses the completed final boundary: ResNeXt50 residual ranking survives
while its below-one direction-threshold guardrail fails, and CIFAR-10 residual
ranking reverses while its direction threshold survives. It preserves the P0
`not_ready` boundary and requires a new unspent protocol for any score repair.

The natural head-to-tail boundary audit is
`discussion/e11_natural_head_tail_boundary.md`, backed by
`results/e11_natural_head_tail_boundary/*` and generated by
`scripts/e11_write_natural_head_tail_boundary_audit.py`. It scans 11 committed
natural matched-head-gain result sources and 37 primary full tail-output drift rows. The current primary status is
`no_strict_natural_primary_counterexample_in_committed_scan`: all primary
full-drift CIs are strictly below 1. The same audit records component true-logit
and secondary tail loss/margin/accuracy tradeoff candidates, so those stronger
claims remain boundary-limited.
The fresh natural negative-search protocol is
`discussion/e11_natural_negative_search_protocol.md`, backed by
`results/e11_natural_negative_search_protocol/*` and generated by
`scripts/e11_write_natural_negative_search_protocol.py`. It does not claim a
new natural counterexample. It freezes phase1/phase2 search-space rows, the
primary full-drift metric contract, multiplicity-adjusted decision rule,
stopping rules, acceptance gates, and claim ladder before fresh search outputs
are interpreted.
The phase1 detectable-effect audit is
`discussion/e11_natural_negative_search_phase1_power_audit.md`; it records the
minimum-detectable primary drift ratios for the 26-setting Holm family and
separates informative finite nulls from underpowered small-effect nulls.
The phase1 GPU entrypoint is now implemented in
`scripts/e11_run_natural_negative_search_phase1.py` and
`scripts/slurm/e11_natural_negative_search_phase1.sbatch`; the registered fresh
output prefixes currently contain a complete 12-setting CIFAR-100-LT ResNet18
phase1 readout in
`discussion/e11_natural_negative_search_phase1_NNS-P1-cifar100lt-resnet18-new-partitions.md`,
and a complete 8-setting CIFAR-10-LT ResNet18 cross-partition phase1 readout in
`discussion/e11_natural_negative_search_phase1_NNS-P1-cifar10lt-resnet18-cross-partitions.md`,
plus the complete 6-setting tail-quality-control readout in
`discussion/e11_natural_negative_search_phase1_NNS-P1-tail-quality-controls.md`.
Together they complete the registered 26-setting phase1 family. The
multiplicity evaluator is `scripts/e11_evaluate_natural_negative_search_phase1.py`;
its current generated artifact
`discussion/e11_natural_negative_search_phase1_evaluation.md` preserves all 26
phase1 settings, uses paired per-seed log-ratio tests for the primary
tail-output drift ratio, and reports 26/26 primary metric rows with
`NNS-E4-natural-primary-claim=finite_null_candidate`.
The interim synthesis
`discussion/e11_natural_negative_search_phase1_interim_synthesis.md` records
the complete 26/26 family coverage, `raw_worse_rows=0`,
`quality_gate_fail_rows=23`, and a finite registered phase1 null candidate with
detectable-effect and tail-quality caveats; it still blocks any fresh natural
primary counterexample wording or broad natural-null wording outside the
registered phase1 space.
The phase2 held-out architecture family is now complete in
`results/e11_natural_negative_search_protocol/phase2_heldout_architecture/settings_registry.csv`
and documented in
`discussion/e11_natural_negative_search_phase2_NNS-P2-heldout-architecture-boundary.md`.
It uses a ResNet34 CIFAR stem, the phase1-declared CIFAR-100-LT partitions,
2 checkpoint depths, 2 matched head-gain fractions, and 3 seeds per setting for
8 registered settings. The executable GPU entrypoint is
`scripts/e11_run_natural_negative_search_phase2.py` with
`scripts/slurm/e11_natural_negative_search_phase2.sbatch`. The phase2 evaluator is
`scripts/e11_evaluate_natural_negative_search_phase2.py`, with current output in
`discussion/e11_natural_negative_search_phase2_evaluation.md` and
`results/e11_natural_negative_search_protocol/phase2_multiplicity_evaluation/*`;
it records 8/8 observed primary rows, 24 seed-level primary rows,
`NNS-P2-E4-heldout-architecture-claim=finite_null_candidate`, and no adjusted
primary worse row. This supports only finite registered phase2 null-candidate
wording: every phase2 row has `head_gain_gate=False` while `tail_quality_gate=True`,
so it does not validate the mechanism or prove a universal natural null.
The phase2 detectable-effect audit is
`discussion/e11_natural_negative_search_phase2_power_audit.md`, backed by
`results/e11_natural_negative_search_protocol/phase2_power_audit/*`. It records
the 3-seed, 8-setting family MDE and outcome state machine, so a complete phase2
null cannot be overstated when the audited effect scale is below the detectable
boundary.

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

The tuned benchmark protocol target runs
`scripts/e11_write_cifar100_resnet_lt_tuned_benchmark_protocol.py`, writing
`discussion/e11_cifar100_resnet_lt_tuned_benchmark_protocol.md` and
`results/e11_cifar100_resnet_lt_tuned_benchmark_protocol/*`. It quarantines the
standard, recipe, and NS-Muon pilots as context. It freezes validation/final seed splits,
tuned AdamW/SGD/class-balanced baselines, finite-NS-Muon candidate grids,
familywise final comparisons, and scope-control gates before any tuned
final-performance result exists.

The tuned benchmark validation-grid target runs
`scripts/e11_run_cifar100_resnet_lt_tuned_benchmark.py --settings-only`, which
writes `results/e11_cifar100_resnet_lt_tuned_benchmark/settings_registry.csv`
and `execution_status.csv`. The GPU validation entrypoint is
`scripts/slurm/e11_cifar100_resnet_lt_tuned_benchmark_validation.sbatch`; it
runs one registered validation setting per Slurm array cell on seeds `10..14`
and leaves final claim seeds `20..29` untouched until validation selects
recipes.

The tuned benchmark selection target runs
`scripts/e11_write_cifar100_resnet_lt_tuned_benchmark_selection.py`, reading
only validation summaries under
`results/e11_cifar100_resnet_lt_tuned_benchmark/validation_tuning/*`. It writes
`results/e11_cifar100_resnet_lt_tuned_benchmark/validation_selection/*` and
`discussion/e11_cifar100_resnet_lt_tuned_benchmark_selection.md`, selecting one
recipe per family by few balanced accuracy with all balanced accuracy as the
tie-breaker. It remains `not_ready` until validation summaries are complete.

The tuned benchmark power-audit target runs
`scripts/e11_write_cifar100_resnet_lt_tuned_benchmark_power_audit.py`, writing
`discussion/e11_cifar100_resnet_lt_tuned_benchmark_power_audit.md` and
`results/e11_cifar100_resnet_lt_tuned_benchmark/final_power_audit/*`. It fixes
the 10-paired-seed final MDE, the 4-comparison Holm primary family, and the
all-class non-inferiority guardrail before any tuned final-performance output
exists.

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
- `discussion/e11_top_conference_claim_decision_audit.md`
- `discussion/e11_manuscript_claim_trace.md`
- `discussion/e11_mechanism_referee_audit.md`
- `discussion/e11_bold_conjecture_register.md`
- `discussion/e11_muon_state_distribution_contract.md`
- `discussion/e11_natural_head_tail_boundary.md`
- `discussion/e11_reviewer_risk_audit.md`
- `discussion/e11_pasted_review_audit.md`
- `discussion/e11_end_of_draft_self_review.md`
- `discussion/e11_reference_audit.md`
- `discussion/e11_submission_repro_audit.md`
- `discussion/e11_artifact_review_packet.md`
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
- `discussion/e11_cifar100_resnet_condition_score_next_heldout_architecture.md`
- `discussion/e11_cifar100_resnet_condition_score_next_heldout_data.md`
- `discussion/e11_cifar100_resnet_condition_score_next_heldout_evaluation.md`
- `discussion/e11_condition_score_heldout_failure_theory_note.md`
- `discussion/e11_condition_score_theory_bridge.md`
- `discussion/e11_condition_score_fresh_protocol.md`
- `discussion/e11_condition_score_fresh_evaluation.md`
- `discussion/e11_condition_score_failure_mechanism_audit.md`
- `discussion/e11_condition_score_v4_protocol.md`
- `discussion/e11_condition_score_v4_validation_cifar100_rotated.md`
- `discussion/e11_condition_score_v4_validation_freeze.md`
- `discussion/e11_condition_score_v4_architecture_wide_resnet50_2.md`
- `discussion/e11_condition_score_v4_data_cifar10_mixed.md`
- `discussion/e11_condition_score_v4_final_evaluation.md`
- `discussion/e11_condition_score_v4_failure_mechanism_audit.md`
- `discussion/e11_matrix_block_theorem_proof.md`
- `discussion/e11_matrix_block_tightness_audit.md`
- `discussion/e11_theory_proof_obligation_register.md`
- `discussion/e11_condition_score_v5_theory_protocol.md`
- `discussion/e11_condition_score_v5_theory_to_score_map.md`
- `discussion/e11_condition_score_ablation.md`
- `discussion/e11_condition_score_v5_validation_freeze.md`
- `discussion/e11_condition_score_v5_final_evaluation.md`
- `discussion/e11_condition_score_v5_final_interpretation_plan.md`
- `discussion/e11_condition_score_v5_reviewer_failure_response.md`
- `discussion/e11_natural_negative_search_protocol.md`
- `discussion/e11_natural_negative_search_phase1_power_audit.md`
- `discussion/e11_natural_negative_search_phase1_NNS-P1-cifar100lt-resnet18-new-partitions.md`
- `discussion/e11_natural_negative_search_phase1_NNS-P1-cifar10lt-resnet18-cross-partitions.md`
- `discussion/e11_natural_negative_search_phase1_NNS-P1-tail-quality-controls.md`
- `discussion/e11_natural_negative_search_phase1_evaluation.md`
- `discussion/e11_natural_negative_search_phase1_interim_synthesis.md`
- `discussion/e11_natural_negative_search_phase2_NNS-P2-heldout-architecture-boundary.md`
- `discussion/e11_natural_negative_search_phase2_evaluation.md`
- `discussion/e11_natural_negative_search_phase2_power_audit.md`
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
- `results/e11_cifar100_resnet_condition_score_next/heldout_architecture/layer_summary.csv`
- `results/e11_cifar100_resnet_condition_score_next/heldout_data/layer_summary.csv`
- `results/e11_cifar100_resnet_condition_score_next/heldout_score_evaluation/heldout_score_summary.csv`
- `results/e11_cifar100_resnet_condition_score_next/heldout_score_evaluation/heldout_gate_report.csv`
- `results/e11_condition_score_theory_bridge/score_target_register.csv`
- `results/e11_condition_score_theory_bridge/fresh_protocol_requirements.csv`
- `results/e11_condition_score_fresh_protocol/quarantine_register.csv`
- `results/e11_condition_score_fresh_protocol/score_freeze_registry.csv`
- `results/e11_condition_score_fresh_protocol/fresh_split_registry.csv`
- `results/e11_condition_score_fresh_protocol/acceptance_gates.csv`
- `results/e11_condition_score_fresh_protocol/protocol_status.csv`
- `results/e11_condition_score_fresh_protocol/fresh_score_evaluation/fresh_gate_report.csv`
- `results/e11_condition_score_failure_mechanism_audit/score_outcome_matrix.csv`
- `results/e11_condition_score_failure_mechanism_audit/split_obstruction_taxonomy.csv`
- `results/e11_condition_score_failure_mechanism_audit/resnet50_stage_reversal.csv`
- `results/e11_condition_score_v4_protocol/spent_split_register.csv`
- `results/e11_condition_score_v4_protocol/score_axis_registry.csv`
- `results/e11_condition_score_v4_protocol/unspent_split_registry.csv`
- `results/e11_condition_score_v4_protocol/acceptance_gates.csv`
- `results/e11_condition_score_v4_protocol/protocol_status.csv`
- `results/e11_condition_score_v4_protocol/validation_cifar100_rotated/layer_summary.csv`
- `results/e11_condition_score_v4_protocol/validation_cifar100_rotated/prediction_summary.csv`
- `results/e11_condition_score_v4_protocol/validation_cifar100_rotated/residual_prediction_summary.csv`
- `results/e11_condition_score_v4_protocol/validation_score_freeze/score_formula_registry.csv`
- `results/e11_condition_score_v4_protocol/validation_score_freeze/freeze_status.csv`
- `results/e11_condition_score_v4_protocol/validation_score_freeze/validation_gate_report.csv`
- `results/e11_condition_score_v4_protocol/final_architecture_wide_resnet50_2/layer_summary.csv`
- `results/e11_condition_score_v4_protocol/final_data_cifar10_mixed/layer_summary.csv`
- `results/e11_condition_score_v4_protocol/final_score_evaluation/final_score_summary.csv`
- `results/e11_condition_score_v4_protocol/final_score_evaluation/final_gate_report.csv`
- `results/e11_condition_score_v4_failure_mechanism_audit/final_outcome_matrix.csv`
- `results/e11_condition_score_v4_failure_mechanism_audit/axis_pair_summary.csv`
- `results/e11_condition_score_v4_failure_mechanism_audit/top5_stage_summary.csv`
- `results/e11_condition_score_v4_failure_mechanism_audit/obstruction_summary.csv`
- `results/e11_matrix_block_theorem_proof/theorem_statement.csv`
- `results/e11_matrix_block_theorem_proof/assumption_ledger.csv`
- `results/e11_matrix_block_theorem_proof/proof_steps.csv`
- `results/e11_matrix_block_theorem_proof/claim_implications.csv`
- `results/e11_matrix_block_theorem_proof/paper_cross_checks.csv`
- `results/e11_matrix_block_tightness_audit/rank_boundary_cases.csv`
- `results/e11_matrix_block_tightness_audit/formula_checks.csv`
- `results/e11_matrix_block_tightness_audit/caveat_checks.csv`
- `results/e11_theory_proof_obligation_register/proof_obligations.csv`
- `results/e11_theory_proof_obligation_register/assumption_stress_tests.csv`
- `results/e11_theory_proof_obligation_register/claim_scope_boundaries.csv`
- `results/e11_theory_proof_obligation_register/theorem_to_experiment_queue.csv`
- `results/e11_condition_score_v5_theory_protocol/theory_term_register.csv`
- `results/e11_condition_score_v5_theory_protocol/score_contract.csv`
- `results/e11_condition_score_v5_theory_protocol/spent_evidence_policy.csv`
- `results/e11_condition_score_v5_theory_protocol/unspent_split_requirements.csv`
- `results/e11_condition_score_v5_theory_protocol/acceptance_gates.csv`
- `results/e11_condition_score_v5_theory_to_score_map/theorem_proxy_map.csv`
- `results/e11_condition_score_v5_theory_to_score_map/score_lineage.csv`
- `results/e11_condition_score_v5_theory_to_score_map/transport_normalization_contract.csv`
- `results/e11_condition_score_v5_theory_to_score_map/falsifiable_predictions.csv`
- `results/e11_condition_score_v5_theory_to_score_map/ablation_matrix.csv`
- `results/e11_condition_score_ablation/score_ablation_summary.csv`
- `results/e11_condition_score_ablation/term_failure_ladder.csv`
- `results/e11_condition_score_ablation/leakage_and_claim_boundary.csv`
- `results/e11_submission_repro_audit/toolchain_status.csv`
- `results/e11_submission_repro_audit/pdf_artifact_checks.csv`
- `results/e11_submission_repro_audit/source_package_manifest.csv`
- `results/e11_submission_repro_audit/build_gate_summary.csv`
- `results/e11_artifact_review_packet/command_matrix.csv`
- `results/e11_artifact_review_packet/gate_matrix.csv`
- `results/e11_artifact_review_packet/local_state_contract.csv`
- `results/e11_artifact_review_packet/reviewer_response.csv`
- `results/e11_condition_score_v5_theory_to_score_map/claim_readiness_ledger.csv`
- `results/e11_condition_score_v5_protocol/validation_score_freeze/score_formula_registry.csv`
- `results/e11_condition_score_v5_protocol/validation_score_freeze/freeze_status.csv`
- `results/e11_condition_score_v5_protocol/validation_score_freeze/validation_gate_report.csv`
- `results/e11_condition_score_v5_protocol/final_score_evaluation/final_score_pairs.csv`
- `results/e11_condition_score_v5_protocol/final_score_evaluation/final_score_summary.csv`
- `results/e11_condition_score_v5_protocol/final_score_evaluation/final_gate_report.csv`
- `results/e11_condition_score_v5_protocol/final_power_audit/split_power_design.csv`
- `results/e11_condition_score_v5_protocol/final_power_audit/fisher_z_resolution.csv`
- `results/e11_condition_score_v5_protocol/final_power_audit/mean_spearman_mde.csv`
- `results/e11_condition_score_v5_protocol/final_power_audit/interpretation_ladder.csv`
- `results/e11_condition_score_v5_protocol/final_power_audit/outcome_state_machine.csv`
- `results/e11_condition_score_v5_protocol/final_interpretation_plan/final_split_status.csv`
- `results/e11_condition_score_v5_protocol/final_interpretation_plan/final_gate_contract.csv`
- `results/e11_condition_score_v5_protocol/final_interpretation_plan/outcome_interpretation_ladder.csv`
- `results/e11_condition_score_v5_protocol/final_interpretation_plan/leakage_lock.csv`
- `results/e11_condition_score_v5_protocol/reviewer_failure_response/final_split_output_status.csv`
- `results/e11_condition_score_v5_protocol/reviewer_failure_response/current_gate_snapshot.csv`
- `results/e11_condition_score_v5_protocol/reviewer_failure_response/active_failure_modes.csv`
- `results/e11_condition_score_v5_protocol/reviewer_failure_response/failure_mode_register.csv`
- `results/e11_condition_score_v5_protocol/reviewer_failure_response/reviewer_objection_map.csv`
- `results/e11_condition_score_v5_protocol/reviewer_failure_response/claim_downgrade_actions.csv`
- `results/e11_condition_score_v5_protocol/reviewer_failure_response/next_evidence_queue.csv`
- `results/e11_condition_score_v5_protocol/direction_guardrail_failure_audit/score_axis_contrast.csv`
- `results/e11_condition_score_v5_protocol/direction_guardrail_failure_audit/gate_boundary_summary.csv`
- `results/e11_condition_score_v5_protocol/direction_guardrail_failure_audit/mechanistic_diagnosis.csv`
- `results/e11_condition_score_v5_protocol/direction_guardrail_failure_audit/next_protocol_requirements.csv`
- `results/e11_natural_head_tail_boundary/search_registry.csv`
- `results/e11_natural_head_tail_boundary/primary_drift_scan.csv`
- `results/e11_natural_head_tail_boundary/secondary_outcome_scan.csv`
- `results/e11_natural_head_tail_boundary/boundary_summary.csv`
- `results/e11_natural_head_tail_boundary/candidate_negative_cases.csv`
- `results/e11_natural_negative_search_protocol/audit_baseline.csv`
- `results/e11_natural_negative_search_protocol/search_space_registry.csv`
- `results/e11_natural_negative_search_protocol/metric_contract.csv`
- `results/e11_natural_negative_search_protocol/stopping_rules.csv`
- `results/e11_natural_negative_search_protocol/acceptance_gates.csv`
- `results/e11_natural_negative_search_protocol/claim_ladder.csv`
- `results/e11_natural_negative_search_protocol/protocol_status.csv`
- `results/e11_natural_negative_search_protocol/phase1_power_audit/power_grid.csv`
- `results/e11_natural_negative_search_protocol/phase1_power_audit/minimum_detectable_effect.csv`
- `results/e11_natural_negative_search_protocol/phase1_power_audit/interpretation_ladder.csv`
- `results/e11_natural_negative_search_protocol/phase1_cifar100lt_resnet18/settings_registry.csv`
- `results/e11_natural_negative_search_protocol/phase1_cifar100lt_resnet18/step_metrics.csv`
- `results/e11_natural_negative_search_protocol/phase1_cifar100lt_resnet18/pair_summary.csv`
- `results/e11_natural_negative_search_protocol/phase1_cifar100lt_resnet18/layer_metrics.csv`
- `results/e11_natural_negative_search_protocol/phase1_cifar100lt_resnet18/decision_template.csv`
- `results/e11_natural_negative_search_protocol/phase1_cifar10lt_resnet18/settings_registry.csv`
- `results/e11_natural_negative_search_protocol/phase1_cifar10lt_resnet18/step_metrics.csv`
- `results/e11_natural_negative_search_protocol/phase1_cifar10lt_resnet18/pair_summary.csv`
- `results/e11_natural_negative_search_protocol/phase1_cifar10lt_resnet18/layer_metrics.csv`
- `results/e11_natural_negative_search_protocol/phase1_cifar10lt_resnet18/decision_template.csv`
- `results/e11_natural_negative_search_protocol/phase1_multiplicity_evaluation/run_registry.csv`
- `results/e11_natural_negative_search_protocol/phase1_multiplicity_evaluation/seed_level_primary_ratios.csv`
- `results/e11_natural_negative_search_protocol/phase1_multiplicity_evaluation/primary_decisions.csv`
- `results/e11_natural_negative_search_protocol/phase1_multiplicity_evaluation/gate_report.csv`
- `results/e11_natural_negative_search_protocol/phase1_interim_synthesis/family_coverage.csv`
- `results/e11_natural_negative_search_protocol/phase1_interim_synthesis/observed_primary_summary.csv`
- `results/e11_natural_negative_search_protocol/phase1_interim_synthesis/claim_boundary.csv`
- `results/e11_natural_negative_search_protocol/phase1_interim_synthesis/remaining_work.csv`
- `results/e11_natural_negative_search_protocol/phase2_heldout_architecture/settings_registry.csv`
- `results/e11_natural_negative_search_protocol/phase2_multiplicity_evaluation/run_registry.csv`
- `results/e11_natural_negative_search_protocol/phase2_multiplicity_evaluation/seed_level_primary_ratios.csv`
- `results/e11_natural_negative_search_protocol/phase2_multiplicity_evaluation/primary_decisions.csv`
- `results/e11_natural_negative_search_protocol/phase2_multiplicity_evaluation/gate_report.csv`
- `results/e11_natural_negative_search_protocol/phase2_power_audit/power_grid.csv`
- `results/e11_natural_negative_search_protocol/phase2_power_audit/minimum_detectable_effect.csv`
- `results/e11_natural_negative_search_protocol/phase2_power_audit/interpretation_ladder.csv`
- `results/e11_natural_negative_search_protocol/phase2_power_audit/outcome_state_machine.csv`
- `results/e11_top_conference_gap_register/gap_register.csv`
- `results/e11_top_conference_claim_decision_audit/claim_decision_matrix.csv`
- `results/e11_top_conference_claim_decision_audit/reviewer_objection_matrix.csv`
- `results/e11_top_conference_claim_decision_audit/rebuttal_response_pack.csv`
- `results/e11_top_conference_claim_decision_audit/manuscript_edit_queue.csv`
- `results/e11_top_conference_claim_decision_audit/paper_sequence.csv`
- `results/e11_top_conference_claim_decision_audit/readiness_summary.csv`
- `results/e11_manuscript_claim_trace/claim_trace.csv`
- `results/e11_manuscript_claim_trace/blocked_phrase_audit.csv`
- `results/e11_mechanism_referee_audit/alternative_explanation_matrix.csv`
- `results/e11_mechanism_referee_audit/theory_measurement_contract.csv`
- `results/e11_mechanism_referee_audit/falsification_trigger_matrix.csv`
- `results/e11_bold_conjecture_register/conjecture_register.csv`
- `results/e11_bold_conjecture_register/stress_test_matrix.csv`
- `results/e11_bold_conjecture_register/claim_upgrade_ladder.csv`
- `results/e11_muon_state_distribution_contract/state_distribution_terms.csv`
- `results/e11_muon_state_distribution_contract/evidence_link_matrix.csv`
- `results/e11_muon_state_distribution_contract/falsification_tests.csv`
- `results/e11_muon_state_distribution_contract/claim_gate_ladder.csv`
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
- `figures/e11_cifar100_resnet_condition_score_next/heldout_score_evaluation/cifar100_resnet_condition_score_heldout_evaluation.png`
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
   - The registered held-out condition-score evaluation now fails the P0 residual-ranking gates. ResNet34 held-out architecture primary residual Spearman is `0.1992 [-0.07515, 0.4735]`, and CIFAR-10-LT held-out data primary residual Spearman is `-0.6771 [-0.7011, -0.653]`. The below-one threshold direction still passes (`0.991` and `0.9841`), while the legacy scaled-JVP ratio on CIFAR-10-LT has residual Spearman `0.6219 [0.6013, 0.6425]`. This turns v2 into a useful obstruction: the checkpoint-fit residual score is not invariant across architecture/data held-outs.
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
- `scripts/e11_evaluate_condition_score_fresh_protocol.py`: frozen evaluator for the fresh condition-score protocol; current generated state is `not_ready` because the fresh ResNet50 residual-ranking gate fails.
- `scripts/e11_write_condition_score_theory_bridge.py`: generated theory/protocol bridge separating threshold-direction and residual-ranking targets after the held-out condition-score failure.
- `scripts/e11_write_condition_score_fresh_protocol.py`: generated fresh condition-score protocol with spent-heldout quarantine, score-freeze registry, fresh split registry, and acceptance gates.
- `scripts/e11_write_condition_score_failure_mechanism_audit.py`: diagnostic-only obstruction audit for the v2/v3 held-out failures and the fresh ResNet50 scaled-JVP reversal.
- `scripts/e11_write_condition_score_v4_protocol.py`: generated v4 protocol that quarantines all spent final splits, separates direction/amplitude/transport score axes, and registers unspent WideResNet50-2 and CIFAR-10 mixed final splits.
- `scripts/e11_freeze_condition_score_v4_validation.py`: validation-freeze boundary that aggregates the v4 Frobenius amplitude axis from `metrics.csv` and freezes the scalar score before final split evaluation.
- `scripts/e11_evaluate_condition_score_v4_finals.py`: frozen-score evaluator for the unspent v4 WideResNet50-2 and CIFAR-10 mixed final splits.
- `scripts/e11_write_condition_score_v4_failure_mechanism_audit.py`: diagnostic-only audit showing that the v4 CIFAR-10 mixed failure is an amplitude/depth scalar-aggregation reversal, not a direction-threshold failure.
- `scripts/e11_write_matrix_block_theorem_proof.py`: machine-checkable theorem/proof contract for the matched-gain bound, sandwich sensitivity lemma, and `nrank`/`ssrank` boundary.
- `scripts/e11_write_matrix_block_tightness_audit.py`: deterministic theorem sanity audit for exact diagonal witnesses, the coefficient-ratio identity, equality/Frobenius-favored cases, and degenerate-tail caveats.
- `scripts/e11_write_theory_proof_obligation_register.py`: theory-facing top-conference checklist mapping theorem claims, assumptions, pending empirical gates, and forbidden wording.
- `scripts/e11_write_condition_score_v5_theory_protocol.py`: theory-facing v5 score contract requiring transport-normalized amplitude or a narrower fixed-partition claim before any new P0 predictive-condition attempt.
- `scripts/e11_write_condition_score_v5_theory_to_score_map.py`: v5 theorem-to-measurement bridge that maps sandwich-tail-drift terms to score features, transport contracts, ablations, and falsifiable validation/final gates.
- `scripts/e11_write_condition_score_ablation.py`: spent-evidence score-axis ablation that separates the direction guardrail, residual-risk ranking, amplitude/depth nuisance, transport validation, and leakage boundary before final rows exist.
- `scripts/e11_freeze_condition_score_v5_validation.py`: v5 validation-freeze evaluator; it now freezes the transport-normalized residual score after the validation split and keeps final splits blocked until run with the frozen score.
- `scripts/e11_evaluate_condition_score_v5_finals.py`: validation-frozen v5 final evaluator for the submitted ResNeXt50-32x4d and CIFAR-10 cross-partition final splits; it reports generated split gates and `not_run` split gates without retuning on final rows.
- `scripts/e11_write_condition_score_v5_final_evaluation.py`: paper-asset wrapper that refreshes the v5 final evaluator outputs without changing the registered scoring rule.
- `scripts/e11_write_condition_score_v5_final_power_audit.py`: pre-output v5 final detectable-effect audit that fixes Fisher-z resolution, mean-Spearman MDE, and underpowered-null versus negative-transport wording before final rows exist.
- `scripts/e11_write_condition_score_v5_final_interpretation_plan.py`: pre-output v5 final interpretation lock that maps pass/fail patterns to allowed claims, boundary interpretations, and forbidden post-hoc retuning actions.
- `scripts/e11_write_condition_score_v5_reviewer_failure_response.py`: completed-final v5 reviewer failure response that maps active final failure modes to claim downgrades, forbidden wording, and next evidence.
- `scripts/e11_write_condition_score_v5_direction_guardrail_failure_audit.py`: completed-final mechanism diagnosis that separates ResNeXt50 residual-ranking survival from direction-threshold failure and CIFAR-10 residual-ranking reversal from direction-threshold survival without repairing the frozen score.
- `scripts/e11_write_natural_negative_search_protocol.py`: pre-registered fresh natural negative-search protocol with search space, metric contract, multiplicity rule, stopping rules, gates, and claim ladder.
- `scripts/e11_write_natural_negative_power_audit.py`: phase1 natural-negative power/MDE audit for interpreting adjusted positive and finite-null outcomes.
- `scripts/e11_run_natural_negative_search_phase1.py`: executable phase1 runner for the registered natural negative-search settings; supports `--list-settings` without launching training and writes fresh outputs only when submitted.
- `scripts/e11_run_natural_negative_search_phase2.py`: executable ResNet34 held-out architecture runner for the registered phase2 natural negative-search settings; `--settings-only` freezes the 8-setting registry before metric rows are generated.
- `scripts/e11_evaluate_natural_negative_search_phase1.py`: Holm-adjusted phase1 evaluator for the natural negative-search protocol; uses paired per-seed log-ratio tests and keeps every registered setting in the complete 26-setting decision family.
- `scripts/e11_evaluate_natural_negative_search_phase2.py`: Holm-adjusted evaluator for the completed registered 8-setting ResNet34 phase2 held-out architecture family; reports finite-null-candidate wording with power, head-gain, and quality caveats.
- `scripts/e11_write_natural_negative_phase2_power_audit.py`: phase2 detectable-effect and outcome-state audit for the 3-seed, 8-setting ResNet34 held-out architecture family.
- `scripts/e11_write_natural_negative_phase1_interim_synthesis.py`: claim-boundary synthesis for complete phase1 coverage, raw-worse counts, quality gates, and finite registered phase1 null-candidate caveats.
- `scripts/e11_write_submission_repro_audit.py`: submission reproducibility audit for LaTeX toolchain availability, rendered PDF hashes, paper source hashes, and clean-checkout gates.
- `scripts/e11_write_clean_worktree_replay_audit.py`: clean detached worktree replay audit for `make e11-check`, excluding local untracked attachments such as `serverREADME.md`.
- `scripts/e11_write_pdf_render_boundary_audit.py`: rendered-PDF inspection boundary audit that separates passing binary/hash/source-claim evidence from text-layer and page-metadata tool gaps.
- `scripts/e11_write_artifact_review_packet.py`: artifact-review command, gate, local-state, and reviewer-response packet for reproducing the current bundle without expanding claims.
- `scripts/e11_write_camera_ready_package_audit.py`: camera-ready package audit that ties source bundle, rendered PDFs, claim trace, reviewer commands, local attachment exclusion, and venue-toolchain boundaries into package gates.
- `scripts/e11_write_mechanism_referee_audit.py`: adversarial mechanism/referee audit that maps alternative explanations, theory-to-measurement contracts, and falsification triggers to current evidence and forbidden wording.
- `scripts/e11_run_cifar100_resnet_lt_standard_eval.py`: standard CIFAR-100-LT ResNet18 many/medium/few reporting baseline.
- `scripts/e11_run_cifar100_resnet_lt_recipe_benchmark.py`: augmented CIFAR-100-LT ResNet18 recipe benchmark pilot with AdamW, class-balanced AdamW, SGD-momentum, and optional NS-Muon final-training recipes.
- `scripts/e11_write_cifar100_resnet_lt_tuned_benchmark_protocol.py`: registered tuned-benchmark protocol that separates spent pilots from validation/final seeds, tuned baselines, candidate Muon grids, and final claim gates.
- `scripts/e11_run_cifar100_resnet_lt_tuned_benchmark.py`: executable tuned-benchmark validation-grid registry and per-setting runner for the registered CIFAR-100-LT ResNet18 tuned benchmark.
- `scripts/e11_write_cifar100_resnet_lt_tuned_benchmark_settings.py`: paper-asset-safe wrapper that regenerates the tuned validation-grid registry without launching GPU training.
- `scripts/e11_write_cifar100_resnet_lt_tuned_benchmark_selection.py`: validation-only selection audit that chooses final recipes after completed validation summaries and keeps final seeds quarantined.
- `scripts/e11_write_cifar100_resnet_lt_tuned_benchmark_power_audit.py`: pre-output tuned final-performance power audit that locks the final MDE, Holm family, all-class guardrail, and outcome state machine before final outputs exist.
- `scripts/e11_write_cifar100_resnet_lt_tuned_benchmark_variance_prior_audit.py`: pre-final variance-prior audit that calibrates tuned benchmark MDE sensitivity from validation-only and spent-pilot variability without touching final seeds.
- `scripts/e11_write_cifar100_resnet_lt_tuned_benchmark_refresh_firewall.py`: validation refresh firewall that freezes legal sequential prefix refreshes and forbidden metric-dependent launch, ordering, selection, and final-submit actions.
- `scripts/e11_write_cifar100_resnet_lt_tuned_benchmark_protocol_seal.py`: post-exposure protocol hash seal for frozen tuned benchmark registry, seed split, selection objective, runner, and final-gate surfaces.
- `scripts/e11_write_cifar100_resnet_lt_tuned_benchmark_fairness_audit.py`: validation-budget and comparison-parity audit for tuned baselines, Muon candidate-budget disclosure, shared seed/metric rules, and final reporting parity.
- `scripts/e11_write_cifar100_resnet_lt_tuned_benchmark_final_analysis_plan.py`: pre-final statistical analysis plan that fixes paired tests, Holm adjustment, reporting schema, and claim states before final outputs exist.
- `scripts/e11_write_cifar100_resnet_lt_tuned_benchmark_final_execution_plan.py`: no-side-effect final-claim execution plan that links the gate-checked final runner, Slurm wrapper, selected recipe families, and seed `20..29` without submitting jobs.
- `scripts/e11_evaluate_cifar100_resnet_lt_tuned_benchmark_final.py`: fixed final evaluator for selected recipe families, paired seed metrics, Holm-adjusted primary comparisons, all-class guardrails, occupancy summaries, and final claim gates.
- `scripts/e11_submit_cifar100_resnet_lt_tuned_benchmark_final.py`: queue-aware final-claim launcher/audit that emits a final Slurm command only after FEP gates pass and records the launch decision.
- `scripts/e11_write_cifar100_resnet_lt_tuned_benchmark_slurm_plan.py`: chunked no-side-effect Slurm launch plan for the registered 164-setting tuned validation grid under the server MaxSubmitJobs policy.
- `scripts/e11_submit_cifar100_resnet_lt_tuned_benchmark_validation.py`: queue-aware Slurm launcher/audit that submits only a safe validation subchunk and records the launch decision.
- `scripts/e11_write_top_conference_claim_decision_audit.py`: paper-level claim decision and rebuttal-readiness contract that separates supportable theorem/diagnostic wording from completed-negative-boundary and blocked predictive, natural-negative, benchmark, and toolchain claims.
- `scripts/e11_write_manuscript_claim_trace.py`: manuscript-level trace that maps each top-conference claim decision to `main.tex` anchors and checks that blocked positive wording is absent.
- `scripts/e11_write_bold_conjecture_register.py`: bold-conjecture/careful-verification ledger that links theory bets to falsifiers, unspent tests, and forbidden claim shortcuts.
- `scripts/e11_write_muon_state_distribution_contract.py`: state-distribution transport contract separating sampled local Muon drift compatibility from blocked practical optimizer-performance claims.
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
- `scripts/slurm/e11_cifar100_resnet_condition_score_fresh_architecture_resnet50.sbatch`: GPU/Slurm submission wrapper for the fresh ResNet50 condition-score architecture split.
- `scripts/slurm/e11_cifar100_resnet_condition_score_fresh_data_cifar10_alt.sbatch`: GPU/Slurm submission wrapper for the fresh CIFAR-10 alternate-partition condition-score data split.
- `scripts/slurm/e11_cifar100_resnet_condition_score_v4_validation_cifar100_rotated.sbatch`: GPU/Slurm submission wrapper for the v4 validation-only CIFAR-100-LT rotated partition.
- `scripts/slurm/e11_cifar100_resnet_condition_score_v4_architecture_wide_resnet50_2.sbatch`: GPU/Slurm submission wrapper for the unspent v4 WideResNet50-2 architecture final split.
- `scripts/slurm/e11_cifar100_resnet_condition_score_v4_data_cifar10_mixed.sbatch`: GPU/Slurm submission wrapper for the unspent v4 CIFAR-10 mixed-partition final split.
- `scripts/slurm/e11_cifar100_resnet_condition_score_v5_validation_mod4_partition.sbatch`: GPU/Slurm submission wrapper for the v5 validation-only CIFAR-100-LT mod-4 partition.
- `scripts/slurm/e11_cifar100_resnet_condition_score_v5_architecture_resnext50_32x4d.sbatch`: GPU/Slurm submission wrapper for the v5 ResNeXt50-32x4d final architecture split.
- `scripts/slurm/e11_cifar100_resnet_condition_score_v5_data_cifar10_cross.sbatch`: GPU/Slurm submission wrapper for the v5 CIFAR-10 cross-partition final data split.
- `scripts/slurm/e11_cifar100_resnet_lt_standard_eval.sbatch`: GPU/Slurm submission wrapper for the standard CIFAR-100-LT ResNet18 reporting baseline.
- `scripts/slurm/e11_cifar100_resnet_lt_recipe_benchmark.sbatch`: GPU/Slurm submission wrapper for the augmented ResNet18 recipe benchmark pilot.
- `scripts/slurm/e11_natural_negative_search_phase1.sbatch`: GPU/Slurm array wrapper for the registered phase1 natural negative-search settings.
- `scripts/slurm/e11_natural_negative_search_phase2.sbatch`: GPU/Slurm wrapper for the registered ResNet34 phase2 held-out architecture natural negative-search settings.
- `scripts/slurm/e11_cifar100_resnet_lt_muon_final_benchmark.sbatch`: GPU/Slurm submission wrapper for the negative NS-Muon final-training benchmark pilot.
- `scripts/e11_write_*.py`: generated discussion and paper-facing artifacts.
- `tests/`: smoke and diagnostic tests.

## Current Publication Gaps

The current evidence is consistent with a focused local-geometry paper. It is not yet enough for a broad optimizer-performance paper.

The generated next-evidence matrix is `discussion/e11_top_conference_gap_register.md`, backed by `results/e11_top_conference_gap_register/gap_register.csv`. The paper-level claim contract is `discussion/e11_top_conference_claim_decision_audit.md`, backed by `results/e11_top_conference_claim_decision_audit/*`; it marks theorem and diagnostic wording as supportable, v5 predictive-condition wording as blocked by completed final failures, and natural-negative, benchmark-performance, and preferred-LaTeX claims as blocked or caveated until their named gates pass. The manuscript claim trace in `discussion/e11_manuscript_claim_trace.md`, backed by `results/e11_manuscript_claim_trace/*`, maps those decisions to `main.tex` anchors and checks that blocked positive wording is absent. The bold-conjecture/careful-verification ledger in `discussion/e11_bold_conjecture_register.md`, backed by `results/e11_bold_conjecture_register/*`, turns endpoint-factorized transport, natural-boundary, optimizer-performance, and layer-risk theory bets into explicit falsifiers and unspent-test gates without upgrading current claims. The Muon state-distribution transport contract in `discussion/e11_muon_state_distribution_contract.md`, backed by `results/e11_muon_state_distribution_contract/*`, explains why sampled local Muon drift compatibility can coexist with the negative final-training pilot and registers occupancy, schedule, terminal-risk, and baseline-dominance falsifiers before any practical performance claim. The P0 condition-score protocol is pre-registered in `discussion/e11_cifar100_resnet_condition_score_protocol.md`, with its locked ResNet18 checkpoint-split v2 analysis in `discussion/e11_cifar100_resnet_condition_score_next.md`, failed registered held-out evaluation in `discussion/e11_cifar100_resnet_condition_score_next_heldout_evaluation.md`, theory-boundary note in `discussion/e11_condition_score_heldout_failure_theory_note.md`, generated score/theory protocol bridge in `discussion/e11_condition_score_theory_bridge.md`, fresh protocol revision in `discussion/e11_condition_score_fresh_protocol.md`, frozen fresh evaluator in `discussion/e11_condition_score_fresh_evaluation.md`, failure-mechanism audit in `discussion/e11_condition_score_failure_mechanism_audit.md`, v4 protocol in `discussion/e11_condition_score_v4_protocol.md`, v4 final evaluation in `discussion/e11_condition_score_v4_final_evaluation.md`, v4 failure mechanism audit in `discussion/e11_condition_score_v4_failure_mechanism_audit.md`, theory proof-obligation register in `discussion/e11_theory_proof_obligation_register.md`, v5 theory protocol in `discussion/e11_condition_score_v5_theory_protocol.md`, v5 theory-to-score map in `discussion/e11_condition_score_v5_theory_to_score_map.md`, v5 validation-freeze boundary in `discussion/e11_condition_score_v5_validation_freeze.md`, v5 final evaluator in `discussion/e11_condition_score_v5_final_evaluation.md`, v5 final detectable-effect audit in `discussion/e11_condition_score_v5_final_power_audit.md`, v5 final interpretation lock in `discussion/e11_condition_score_v5_final_interpretation_plan.md`, v5 reviewer failure response in `discussion/e11_condition_score_v5_reviewer_failure_response.md`, v5 direction-guardrail failure audit in `discussion/e11_condition_score_v5_direction_guardrail_failure_audit.md`, mechanism referee audit in `discussion/e11_mechanism_referee_audit.md`, tuned benchmark protocol in `discussion/e11_cifar100_resnet_lt_tuned_benchmark_protocol.md`, tuned benchmark selection audit in `discussion/e11_cifar100_resnet_lt_tuned_benchmark_selection.md`, tuned benchmark power audit in `discussion/e11_cifar100_resnet_lt_tuned_benchmark_power_audit.md`, validation refresh firewall in `discussion/e11_cifar100_resnet_lt_tuned_benchmark_refresh_firewall.md`, protocol hash seal in `discussion/e11_cifar100_resnet_lt_tuned_benchmark_protocol_seal.md` with `results/e11_cifar100_resnet_lt_tuned_benchmark/validation_protocol_seal/hash_manifest.csv`, fairness audit in `discussion/e11_cifar100_resnet_lt_tuned_benchmark_fairness_audit.md` with `results/e11_cifar100_resnet_lt_tuned_benchmark/validation_fairness_audit/fairness_gate_matrix.csv`, natural boundary audit in `discussion/e11_natural_head_tail_boundary.md`, fresh natural negative-search protocol in `discussion/e11_natural_negative_search_protocol.md`, submission reproducibility audit in `discussion/e11_submission_repro_audit.md`, clean worktree replay audit in `discussion/e11_clean_worktree_replay_audit.md` with `results/e11_clean_worktree_replay_audit/run_summary.csv`, `results/e11_clean_worktree_replay_audit/gate_matrix.csv`, PDF render boundary audit in `discussion/e11_pdf_render_boundary_audit.md` with `results/e11_pdf_render_boundary_audit/pdf_inspection_tool_status.csv` and `results/e11_pdf_render_boundary_audit/render_boundary_gates.csv`, and artifact-review packet in `discussion/e11_artifact_review_packet.md`.

Most important next steps:

1. Run the executable tuned benchmark validation grid from `results/e11_cifar100_resnet_lt_tuned_benchmark/settings_registry.csv`, select recipes on validation seeds only, rerun the tuned benchmark power audit, then run final paired seeds for tuned AdamW/SGD/class-balanced baselines and finite-NS-Muon candidates. Larger long-tail datasets still require a separate preregistered protocol; the current pilots are useful benchmark context, not a competitive optimizer result.
2. Improve the all-layer ResNet JVP predictive condition benchmark: the held-out checkpoint-transfer run shows source-observed and early-layer controls transfer, and observed residuals transfer after source-fit depth adjustment, but the frozen v2 condition score fails the registered ResNet34 and CIFAR-10-LT held-out residual-ranking gates. The fresh v3 zero-fit scaled-JVP score then passes the CIFAR-10 alternate partition but reverses on the fresh ResNet50 architecture split. V4 then freezes a two-axis amplitude-minus-direction score and passes the WideResNet50-2 final architecture split, but fails the CIFAR-10 mixed final data split. The v4 failure audit localizes this data-partition reversal mechanism problem to amplitude/depth scalar aggregation rather than the direction threshold. The proof-obligation register now separates theorem claims, assumptions, and blocked predictive-condition wording. V5 has a frozen transport-normalized validation score, a pre-registered final evaluator, a pre-output power audit, a pre-output interpretation lock, a reviewer failure response, and a completed-final direction/boundary audit. The final result is negative for P0: ResNeXt50 residual ranking survives but direction threshold fails, while CIFAR-10 residual ranking reverses but direction threshold survives. Any next predictive-condition attempt needs new theory terms and a new unspent protocol rather than repair on these final rows.
3. Extend the current fixed-checkpoint, short-trajectory, small practical-training, and negative ResNet final-training Muon diagnostics into a full practical Muon benchmark with schedules, state-distribution occupancy logging, final tail metrics, and hyperparameter robustness.
4. Add larger-architecture layerwise JVP/decomposition diagnostics if the detailed scaled-head-gain mechanism is meant to survive beyond the current small MLP explanation.
5. Treat the current 26-setting phase1 and 8-setting phase2 natural-negative families as complete only within their registered scope; rerun the phase1/phase2 evaluators after any new registered family, and preregister larger-dataset or additional held-out-architecture searches before broadening the finite-null wording.
6. Keep separating function-drift evidence from tail loss, margin, accuracy, and final optimizer performance.
