.PHONY: e11-main-results e11-cifar-results e11-cifar-resnet-results e11-cifar-resnet-rho002-results e11-cifar-resnet-checkpoint-sweep-results e11-cifar-resnet-condition-proxy-results e11-cifar-resnet-fc-condition-results e11-cifar-resnet-tail-quality-results e11-appendix-results e11-all-results e11-paper-assets e11-guardrail-assets e11-all-assets e11-paper-pdf e11-artifacts e11-validate e11-test e11-check e11-full

PYTHON ?= python3

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
