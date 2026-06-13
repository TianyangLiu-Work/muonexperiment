.PHONY: e11-main-results e11-appendix-results e11-all-results e11-paper-assets e11-guardrail-assets e11-all-assets e11-paper-pdf e11-artifacts e11-validate e11-test e11-check e11-full

PYTHON ?= python3

e11-main-results:
	$(PYTHON) scripts/e11_run_experiments.py
	$(PYTHON) scripts/e11_make_figures.py
	$(PYTHON) scripts/e11_run_equal_update_control.py
	$(PYTHON) scripts/e11_run_head_tail_interference.py
	$(PYTHON) scripts/e11_run_long_tail_one_step.py
	$(PYTHON) scripts/e11_run_long_tail_muon_bridge.py
	$(PYTHON) scripts/e11_run_long_tail_practical_muon_bridge.py
	$(PYTHON) scripts/e11_run_long_tail_practical_training.py
	$(PYTHON) scripts/e11_run_long_tail_practical_training_lr_sweep.py
	$(PYTHON) scripts/e11_run_long_tail_forgetting.py
	$(PYTHON) scripts/e11_run_long_tail_layerwise.py
	$(PYTHON) scripts/e11_run_spectral_allocation_probe.py

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

e11-paper-pdf:
	$(MAKE) -C paper/specgrad_activation_paper

e11-artifacts: e11-paper-assets

e11-validate:
	$(PYTHON) scripts/e11_validate_outputs.py

e11-test:
	$(PYTHON) -m pytest tests -q

e11-check: e11-validate e11-test
	git diff --check

e11-full: e11-paper-assets e11-paper-pdf e11-check
