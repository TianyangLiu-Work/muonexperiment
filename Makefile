.PHONY: e11-main-results e11-cifar-results e11-cifar-resnet-results e11-cifar-resnet-rho002-results e11-cifar-resnet-checkpoint-sweep-results e11-cifar-resnet-condition-proxy-results e11-cifar-resnet-fc-condition-results e11-cifar-resnet-tail-quality-results e11-cifar-resnet-imbalance-sweep-results e11-cifar-resnet-layer-jvp-tail-quality-results e11-cifar-resnet-layer-jvp-checkpoint-prediction-results e11-cifar-resnet-condition-score-heldout-architecture-results e11-cifar-resnet-condition-score-heldout-data-results e11-cifar-resnet-condition-score-heldout-eval e11-cifar-resnet-condition-score-fresh-architecture-results e11-cifar-resnet-condition-score-fresh-data-results e11-cifar-resnet-condition-score-fresh-eval e11-cifar-resnet-condition-score-failure-audit e11-cifar-resnet-condition-score-v4-protocol e11-cifar-resnet-condition-score-v4-validation-results e11-cifar-resnet-condition-score-v4-validation-freeze e11-cifar-resnet-condition-score-v4-architecture-results e11-cifar-resnet-condition-score-v4-data-results e11-cifar-resnet-condition-score-v4-final-eval e11-cifar-resnet-condition-score-v4-failure-audit e11-matrix-block-theorem-proof e11-matrix-block-tightness-audit e11-theory-proof-obligation-register e11-cifar-resnet-condition-score-v5-theory-protocol e11-cifar-resnet-condition-score-v5-theory-to-score-map e11-cifar-resnet-condition-score-v5-validation-results e11-cifar-resnet-condition-score-v5-validation-freeze e11-cifar-resnet-condition-score-v5-architecture-results e11-cifar-resnet-condition-score-v5-data-results e11-cifar-resnet-condition-score-v5-final-eval e11-cifar-resnet-condition-score-v5-final-interpretation-plan e11-cifar-resnet-condition-score-v5-reviewer-failure-response e11-cifar-resnet-lt-standard-eval-results e11-cifar-resnet-lt-recipe-benchmark-results e11-cifar-resnet-lt-muon-final-benchmark-results e11-cifar-resnet-lt-tuned-benchmark-protocol e11-cifar-resnet-lt-tuned-benchmark-settings e11-cifar-resnet-lt-tuned-benchmark-validation-results e11-cifar-resnet-lt-tuned-benchmark-selection e11-cifar-resnet-practical-muon-bridge-results e11-natural-head-tail-boundary-audit e11-natural-negative-search-protocol e11-natural-negative-search-phase1-power-audit e11-natural-negative-search-phase1-settings e11-natural-negative-search-phase1-results e11-natural-negative-search-phase1-eval e11-natural-negative-search-phase1-interim-synthesis e11-submission-repro-audit e11-appendix-results e11-all-results e11-paper-assets e11-guardrail-assets e11-all-assets e11-paper-pdf e11-artifacts e11-validate e11-test e11-check e11-full

DEFAULT_PYTHON := $(shell if [ -x /data/conda_envs/SpatialQuantization/bin/python ]; then echo /data/conda_envs/SpatialQuantization/bin/python; else command -v python3 || echo python3; fi)
PYTHON ?= $(DEFAULT_PYTHON)

e11-main-results:
	$(PYTHON) scripts/e11_run_experiments.py
	$(PYTHON) scripts/e11_make_figures.py
	$(PYTHON) scripts/e11_run_equal_update_control.py
	$(PYTHON) scripts/e11_run_head_tail_interference.py
	$(PYTHON) scripts/e11_run_head_tail_alignment_ablation.py
	$(PYTHON) scripts/e11_run_long_tail_one_step.py
	$(PYTHON) scripts/e11_run_long_tail_imbalance_ablation.py
	$(PYTHON) scripts/e11_run_long_tail_checkpoint_sweep.py
	$(PYTHON) scripts/e11_run_long_tail_class_partition_sweep.py
	$(PYTHON) scripts/e11_run_long_tail_rho_sweep.py
	$(PYTHON) scripts/e11_run_long_tail_muon_bridge.py
	$(PYTHON) scripts/e11_run_long_tail_practical_muon_bridge.py
	$(PYTHON) scripts/e11_run_long_tail_muon_state_source_control.py
	$(PYTHON) scripts/e11_run_long_tail_practical_training.py
	$(PYTHON) scripts/e11_run_long_tail_practical_training_lr_sweep.py
	$(PYTHON) scripts/e11_run_long_tail_forgetting.py
	$(PYTHON) scripts/e11_run_long_tail_layerwise.py
	$(PYTHON) scripts/e11_run_spectral_allocation_probe.py

e11-cifar-results:
	$(PYTHON) scripts/e11_run_cifar100_lt_one_step.py --device cpu --no-download

e11-cifar-resnet-results: scripts/e11_run_cifar100_resnet_one_step.py scripts/slurm/e11_cifar100_resnet_one_step.sbatch
	sbatch scripts/slurm/e11_cifar100_resnet_one_step.sbatch

e11-cifar-resnet-rho002-results: scripts/e11_run_cifar100_resnet_one_step.py scripts/slurm/e11_cifar100_resnet_one_step_rho002.sbatch
	sbatch scripts/slurm/e11_cifar100_resnet_one_step_rho002.sbatch

e11-cifar-resnet-checkpoint-sweep-results: scripts/e11_run_cifar100_resnet_checkpoint_sweep.py scripts/slurm/e11_cifar100_resnet_checkpoint_sweep.sbatch
	sbatch scripts/slurm/e11_cifar100_resnet_checkpoint_sweep.sbatch

e11-cifar-resnet-condition-proxy-results: scripts/e11_run_cifar100_resnet_condition_proxy_scatter.py
	$(PYTHON) scripts/e11_run_cifar100_resnet_condition_proxy_scatter.py

e11-cifar-resnet-fc-condition-results: scripts/e11_run_cifar100_resnet_fc_condition_scatter.py scripts/slurm/e11_cifar100_resnet_fc_condition_scatter.sbatch
	sbatch scripts/slurm/e11_cifar100_resnet_fc_condition_scatter.sbatch

e11-cifar-resnet-tail-quality-results: scripts/e11_run_cifar100_resnet_checkpoint_sweep.py scripts/slurm/e11_cifar100_resnet_tail_quality_control.sbatch
	sbatch scripts/slurm/e11_cifar100_resnet_tail_quality_control.sbatch

e11-cifar-resnet-imbalance-sweep-results: scripts/e11_run_cifar100_resnet_imbalance_sweep.py scripts/slurm/e11_cifar100_resnet_imbalance_sweep.sbatch
	sbatch scripts/slurm/e11_cifar100_resnet_imbalance_sweep.sbatch

e11-cifar-resnet-layer-jvp-tail-quality-results: scripts/e11_run_cifar100_resnet_layer_jvp_tail_quality.py scripts/slurm/e11_cifar100_resnet_layer_jvp_tail_quality.sbatch
	sbatch scripts/slurm/e11_cifar100_resnet_layer_jvp_tail_quality.sbatch

e11-cifar-resnet-layer-jvp-checkpoint-prediction-results: scripts/e11_run_cifar100_resnet_layer_jvp_checkpoint_prediction.py scripts/slurm/e11_cifar100_resnet_layer_jvp_checkpoint_prediction.sbatch
	sbatch scripts/slurm/e11_cifar100_resnet_layer_jvp_checkpoint_prediction.sbatch

e11-cifar-resnet-condition-score-heldout-architecture-results: scripts/e11_run_cifar100_resnet_layer_jvp_checkpoint_prediction.py scripts/slurm/e11_cifar100_resnet_condition_score_heldout_architecture.sbatch
	sbatch scripts/slurm/e11_cifar100_resnet_condition_score_heldout_architecture.sbatch

e11-cifar-resnet-condition-score-heldout-data-results: scripts/e11_run_cifar100_resnet_layer_jvp_checkpoint_prediction.py scripts/slurm/e11_cifar100_resnet_condition_score_heldout_data.sbatch
	sbatch scripts/slurm/e11_cifar100_resnet_condition_score_heldout_data.sbatch

e11-cifar-resnet-condition-score-heldout-eval: scripts/e11_evaluate_cifar100_resnet_condition_score_heldouts.py
	$(PYTHON) scripts/e11_evaluate_cifar100_resnet_condition_score_heldouts.py

e11-cifar-resnet-condition-score-fresh-architecture-results: scripts/e11_run_cifar100_resnet_layer_jvp_checkpoint_prediction.py scripts/slurm/e11_cifar100_resnet_condition_score_fresh_architecture_resnet50.sbatch
	sbatch scripts/slurm/e11_cifar100_resnet_condition_score_fresh_architecture_resnet50.sbatch

e11-cifar-resnet-condition-score-fresh-data-results: scripts/e11_run_cifar100_resnet_layer_jvp_checkpoint_prediction.py scripts/slurm/e11_cifar100_resnet_condition_score_fresh_data_cifar10_alt.sbatch
	sbatch scripts/slurm/e11_cifar100_resnet_condition_score_fresh_data_cifar10_alt.sbatch

e11-cifar-resnet-condition-score-fresh-eval: scripts/e11_evaluate_condition_score_fresh_protocol.py
	$(PYTHON) scripts/e11_evaluate_condition_score_fresh_protocol.py

e11-cifar-resnet-condition-score-failure-audit: scripts/e11_write_condition_score_failure_mechanism_audit.py
	$(PYTHON) scripts/e11_write_condition_score_failure_mechanism_audit.py

e11-cifar-resnet-condition-score-v4-protocol: scripts/e11_write_condition_score_v4_protocol.py
	$(PYTHON) scripts/e11_write_condition_score_v4_protocol.py

e11-cifar-resnet-condition-score-v4-validation-results: scripts/e11_run_cifar100_resnet_layer_jvp_checkpoint_prediction.py scripts/slurm/e11_cifar100_resnet_condition_score_v4_validation_cifar100_rotated.sbatch
	sbatch scripts/slurm/e11_cifar100_resnet_condition_score_v4_validation_cifar100_rotated.sbatch

e11-cifar-resnet-condition-score-v4-validation-freeze: scripts/e11_freeze_condition_score_v4_validation.py
	$(PYTHON) scripts/e11_freeze_condition_score_v4_validation.py

e11-cifar-resnet-condition-score-v4-architecture-results: scripts/e11_run_cifar100_resnet_layer_jvp_checkpoint_prediction.py scripts/slurm/e11_cifar100_resnet_condition_score_v4_architecture_wide_resnet50_2.sbatch
	sbatch scripts/slurm/e11_cifar100_resnet_condition_score_v4_architecture_wide_resnet50_2.sbatch

e11-cifar-resnet-condition-score-v4-data-results: scripts/e11_run_cifar100_resnet_layer_jvp_checkpoint_prediction.py scripts/slurm/e11_cifar100_resnet_condition_score_v4_data_cifar10_mixed.sbatch
	sbatch scripts/slurm/e11_cifar100_resnet_condition_score_v4_data_cifar10_mixed.sbatch

e11-cifar-resnet-condition-score-v4-final-eval: scripts/e11_evaluate_condition_score_v4_finals.py
	$(PYTHON) scripts/e11_evaluate_condition_score_v4_finals.py

e11-cifar-resnet-condition-score-v4-failure-audit: scripts/e11_write_condition_score_v4_failure_mechanism_audit.py
	$(PYTHON) scripts/e11_write_condition_score_v4_failure_mechanism_audit.py

e11-matrix-block-theorem-proof: scripts/e11_write_matrix_block_theorem_proof.py
	$(PYTHON) scripts/e11_write_matrix_block_theorem_proof.py

e11-matrix-block-tightness-audit: scripts/e11_write_matrix_block_tightness_audit.py
	$(PYTHON) scripts/e11_write_matrix_block_tightness_audit.py

e11-theory-proof-obligation-register: scripts/e11_write_theory_proof_obligation_register.py
	$(PYTHON) scripts/e11_write_theory_proof_obligation_register.py

e11-cifar-resnet-condition-score-v5-theory-protocol: scripts/e11_write_condition_score_v5_theory_protocol.py
	$(PYTHON) scripts/e11_write_condition_score_v5_theory_protocol.py

e11-cifar-resnet-condition-score-v5-theory-to-score-map: scripts/e11_write_condition_score_v5_theory_to_score_map.py
	$(PYTHON) scripts/e11_write_condition_score_v5_theory_to_score_map.py

e11-cifar-resnet-condition-score-v5-validation-results: scripts/e11_run_cifar100_resnet_layer_jvp_checkpoint_prediction.py scripts/slurm/e11_cifar100_resnet_condition_score_v5_validation_mod4_partition.sbatch
	sbatch scripts/slurm/e11_cifar100_resnet_condition_score_v5_validation_mod4_partition.sbatch

e11-cifar-resnet-condition-score-v5-validation-freeze: scripts/e11_freeze_condition_score_v5_validation.py
	$(PYTHON) scripts/e11_freeze_condition_score_v5_validation.py

e11-cifar-resnet-condition-score-v5-architecture-results: scripts/e11_run_cifar100_resnet_layer_jvp_checkpoint_prediction.py scripts/slurm/e11_cifar100_resnet_condition_score_v5_architecture_resnext50_32x4d.sbatch
	sbatch scripts/slurm/e11_cifar100_resnet_condition_score_v5_architecture_resnext50_32x4d.sbatch

e11-cifar-resnet-condition-score-v5-data-results: scripts/e11_run_cifar100_resnet_layer_jvp_checkpoint_prediction.py scripts/slurm/e11_cifar100_resnet_condition_score_v5_data_cifar10_cross.sbatch
	sbatch scripts/slurm/e11_cifar100_resnet_condition_score_v5_data_cifar10_cross.sbatch

e11-cifar-resnet-condition-score-v5-final-eval: scripts/e11_evaluate_condition_score_v5_finals.py
	$(PYTHON) scripts/e11_evaluate_condition_score_v5_finals.py

e11-cifar-resnet-condition-score-v5-final-interpretation-plan: scripts/e11_write_condition_score_v5_final_interpretation_plan.py
	$(PYTHON) scripts/e11_write_condition_score_v5_final_interpretation_plan.py

e11-cifar-resnet-condition-score-v5-reviewer-failure-response: scripts/e11_write_condition_score_v5_reviewer_failure_response.py
	$(PYTHON) scripts/e11_write_condition_score_v5_reviewer_failure_response.py

e11-natural-negative-search-phase1-interim-synthesis: scripts/e11_write_natural_negative_phase1_interim_synthesis.py
	$(PYTHON) scripts/e11_write_natural_negative_phase1_interim_synthesis.py

e11-cifar-resnet-lt-standard-eval-results: scripts/e11_run_cifar100_resnet_lt_standard_eval.py scripts/slurm/e11_cifar100_resnet_lt_standard_eval.sbatch
	sbatch scripts/slurm/e11_cifar100_resnet_lt_standard_eval.sbatch

e11-cifar-resnet-lt-recipe-benchmark-results: scripts/e11_run_cifar100_resnet_lt_recipe_benchmark.py scripts/slurm/e11_cifar100_resnet_lt_recipe_benchmark.sbatch
	sbatch scripts/slurm/e11_cifar100_resnet_lt_recipe_benchmark.sbatch

e11-cifar-resnet-lt-muon-final-benchmark-results: scripts/e11_run_cifar100_resnet_lt_recipe_benchmark.py scripts/slurm/e11_cifar100_resnet_lt_muon_final_benchmark.sbatch
	sbatch scripts/slurm/e11_cifar100_resnet_lt_muon_final_benchmark.sbatch

e11-cifar-resnet-practical-muon-bridge-results: scripts/e11_run_cifar100_resnet_practical_muon_bridge.py scripts/slurm/e11_cifar100_resnet_practical_muon_bridge.sbatch
	sbatch scripts/slurm/e11_cifar100_resnet_practical_muon_bridge.sbatch

e11-appendix-results:
	$(PYTHON) scripts/e11_run_overlap_followup.py
	$(PYTHON) scripts/e11_run_hyperparam_sweep.py
	$(PYTHON) scripts/e11_run_target_update_sweep.py
	$(PYTHON) scripts/e11_run_mlp_width_sweep.py
	$(PYTHON) scripts/e11_run_mlp_per_layer_control.py
	$(PYTHON) scripts/e11_run_mlp_layer_hybrid.py
	$(PYTHON) scripts/e11_run_mnist_mlp_probe.py
	$(PYTHON) scripts/e11_run_deep_mnist_mlp_probe.py
	$(PYTHON) scripts/e11_run_mnist_patch_probe.py
	$(PYTHON) scripts/e11_run_mnist_conv_probe.py
	$(PYTHON) scripts/e11_run_stateless_direction_ablation.py
	$(PYTHON) scripts/e11_run_stateless_optimizer_trajectory.py
	$(PYTHON) scripts/e11_run_singular_vector_trajectory.py
	$(PYTHON) scripts/e11_run_singular_vector_swap_probe.py
	$(PYTHON) scripts/e11_run_natural_update_swap_probe.py
	$(PYTHON) scripts/e11_run_optimizer_switch_probe.py
	$(PYTHON) scripts/e11_run_optimizer_switch_reset_control.py
	$(PYTHON) scripts/e11_run_optimizer_switch_horizon_sweep.py
	$(PYTHON) scripts/e11_run_optimizer_switch_lr_sweep.py
	$(PYTHON) scripts/e11_run_boundary_predictor.py

e11-all-results: e11-main-results e11-appendix-results

e11-submission-repro-audit: scripts/e11_write_submission_repro_audit.py
	$(PYTHON) scripts/e11_write_submission_repro_audit.py

e11-cifar-resnet-lt-tuned-benchmark-protocol: scripts/e11_write_cifar100_resnet_lt_tuned_benchmark_protocol.py
	$(PYTHON) scripts/e11_write_cifar100_resnet_lt_tuned_benchmark_protocol.py

e11-cifar-resnet-lt-tuned-benchmark-settings: scripts/e11_run_cifar100_resnet_lt_tuned_benchmark.py
	$(PYTHON) scripts/e11_run_cifar100_resnet_lt_tuned_benchmark.py --settings-only

e11-cifar-resnet-lt-tuned-benchmark-validation-results: scripts/e11_run_cifar100_resnet_lt_tuned_benchmark.py scripts/slurm/e11_cifar100_resnet_lt_tuned_benchmark_validation.sbatch
	sbatch scripts/slurm/e11_cifar100_resnet_lt_tuned_benchmark_validation.sbatch

e11-cifar-resnet-lt-tuned-benchmark-selection: scripts/e11_write_cifar100_resnet_lt_tuned_benchmark_selection.py
	$(PYTHON) scripts/e11_write_cifar100_resnet_lt_tuned_benchmark_selection.py

e11-natural-head-tail-boundary-audit: scripts/e11_write_natural_head_tail_boundary_audit.py
	$(PYTHON) scripts/e11_write_natural_head_tail_boundary_audit.py

e11-natural-negative-search-protocol: scripts/e11_write_natural_negative_search_protocol.py
	$(PYTHON) scripts/e11_write_natural_negative_search_protocol.py

e11-natural-negative-search-phase1-power-audit: scripts/e11_write_natural_negative_power_audit.py
	$(PYTHON) scripts/e11_write_natural_negative_power_audit.py

e11-natural-negative-search-phase1-settings: scripts/e11_run_natural_negative_search_phase1.py
	$(PYTHON) scripts/e11_run_natural_negative_search_phase1.py --settings-only

e11-natural-negative-search-phase1-results: scripts/e11_run_natural_negative_search_phase1.py scripts/slurm/e11_natural_negative_search_phase1.sbatch
	sbatch scripts/slurm/e11_natural_negative_search_phase1.sbatch

e11-natural-negative-search-phase1-eval: scripts/e11_evaluate_natural_negative_search_phase1.py
	$(PYTHON) scripts/e11_evaluate_natural_negative_search_phase1.py

e11-paper-assets:
	$(PYTHON) scripts/e11_write_all_discussion_artifacts.py

e11-guardrail-assets:
	$(PYTHON) scripts/e11_write_legacy_guardrail_artifacts.py

e11-all-assets: e11-paper-assets e11-guardrail-assets

e11-paper-pdf: e11-paper-assets
	$(MAKE) -C paper/specgrad_activation_paper

e11-artifacts: e11-paper-assets

e11-validate:
	$(PYTHON) scripts/e11_validate_outputs.py

e11-test:
	$(PYTHON) -m pytest tests -q

e11-check: e11-validate e11-test
	git diff --check

e11-full: e11-paper-assets e11-paper-pdf e11-check
