from __future__ import annotations


ARTIFACT_DIRS: tuple[dict[str, str], ...] = (
    {
        "path": "e11_condition_geometry",
        "role": "Source package for E11 problems, optimizers, diagnostics, reporting, and plotting.",
        "commit_policy": "commit",
    },
    {
        "path": "scripts",
        "role": "Experiment runners, artifact writers, and validation scripts.",
        "commit_policy": "commit E11 scripts",
    },
    {
        "path": "tests",
        "role": "Smoke tests, diagnostics tests, reporting tests, and validator guardrail tests.",
        "commit_policy": "commit",
    },
    {
        "path": "discussion",
        "role": "Generated head-to-tail paper evidence plus legacy condition-geometry guardrail notes.",
        "commit_policy": "commit",
    },
    {
        "path": "results",
        "role": "Generated CSV evidence used by the paper table, discussion artifacts, and validator.",
        "commit_policy": "commit current E11 evidence set",
    },
    {
        "path": "figures",
        "role": "Static figures used by the paper draft and paper-facing Markdown artifacts.",
        "commit_policy": "commit static E11 figures",
    },
    {
        "path": "configs",
        "role": "Experiment configuration snapshots.",
        "commit_policy": "commit",
    },
    {
        "path": "paper",
        "role": "Head-to-tail LaTeX paper draft, generated paper table, and experiment triage notes.",
        "commit_policy": "commit source and selected rendered PDFs",
    },
    {
        "path": "Makefile",
        "role": "Thin reproducibility entrypoint for E11 artifact generation and validation commands.",
        "commit_policy": "commit",
    },
    {
        "path": ".gitignore",
        "role": "Keeps local caches, datasets, and generated videos out of default commits.",
        "commit_policy": "commit",
    },
    {
        "path": ".gitattributes",
        "role": "Marks generated evidence artifacts and binary files for cleaner GitHub review.",
        "commit_policy": "commit",
    },
)


KEY_TABLES: tuple[str, ...] = (
    "results/e11/step_metrics.csv",
    "results/e11/activation_perturbation_summary.csv",
    "results/e11_head_tail_interference/step_metrics.csv",
    "results/e11_head_tail_interference/pair_summary.csv",
    "results/e11_long_tail_one_step/step_metrics.csv",
    "results/e11_long_tail_one_step/pair_summary.csv",
    "results/e11_long_tail_one_step/layer_metrics.csv",
    "results/e11_long_tail_muon_bridge/step_metrics.csv",
    "results/e11_long_tail_muon_bridge/pair_summary.csv",
    "results/e11_long_tail_practical_muon_bridge/step_metrics.csv",
    "results/e11_long_tail_practical_muon_bridge/step_summary.csv",
    "results/e11_long_tail_practical_muon_bridge/summary.csv",
    "results/e11_long_tail_practical_training/step_metrics.csv",
    "results/e11_long_tail_practical_training/summary.csv",
    "results/e11_long_tail_practical_training_lr_sweep/sweep_summary.csv",
    "results/e11_long_tail_forgetting/step_metrics.csv",
    "results/e11_long_tail_forgetting/summary.csv",
    "results/e11_long_tail_layerwise/metrics.csv",
    "results/e11_long_tail_layerwise/summary.csv",
    "results/e11_equal_update/step_metrics.csv",
    "results/e11_equal_update/activation_perturbation_summary.csv",
    "results/e11_overlap_followup/step_metrics.csv",
    "results/e11_hyperparam_sweep/equal_step_metrics.csv",
    "results/e11_target_update_sweep/step_metrics.csv",
    "results/e11_mlp_width_sweep/step_metrics.csv",
    "results/e11_mlp_per_layer_control/step_metrics.csv",
    "results/e11_mlp_layer_hybrid/step_metrics.csv",
    "results/e11_mnist_mlp_probe/step_metrics.csv",
    "results/e11_deep_mnist_mlp_probe/step_metrics.csv",
    "results/e11_mnist_patch_probe/step_metrics.csv",
    "results/e11_mnist_conv_probe/step_metrics.csv",
    "results/e11_stateless_direction_ablation/stateless_direction_rows.csv",
    "results/e11_stateless_optimizer_trajectory/stateless_optimizer_step_metrics.csv",
    "results/e11_spectral_allocation_probe/probe_rows.csv",
    "results/e11_singular_vector_trajectory/update_subspace_rows.csv",
    "results/e11_singular_vector_swap_probe/swap_probe_rows.csv",
    "results/e11_natural_update_swap_probe/natural_update_swap_rows.csv",
    "results/e11_optimizer_switch_probe/optimizer_switch_rows.csv",
    "results/e11_optimizer_switch_reset_control/optimizer_switch_reset_rows.csv",
    "results/e11_optimizer_switch_horizon_sweep/optimizer_switch_horizon_rows.csv",
    "results/e11_optimizer_switch_lr_sweep/optimizer_switch_lr_rows.csv",
    "results/e11_boundary_predictor/boundary_predictor_rows.csv",
    "results/e11_mechanism_boundary/mechanism_boundary_map.csv",
    "results/e11_cross_task_signature/cross_task_signature_summary.csv",
    "results/e11_boundary_predictor/boundary_predictor_uncertainty.csv",
)


IGNORE_POLICY: tuple[dict[str, str], ...] = (
    {
        "path_or_pattern": "data/",
        "reason": "Local torchvision/MNIST cache; downloaded by the MNIST probe and not part of the evidence set.",
    },
    {
        "path_or_pattern": ".pytest_cache/",
        "reason": "Local test runner cache.",
    },
    {
        "path_or_pattern": "*.gif, *.mp4",
        "reason": "Generated animations/videos are not part of the default E11 evidence set; force-add only when explicitly needed.",
    },
    {
        "path_or_pattern": "slidev/",
        "reason": "Local presentation workspace, intentionally excluded from the repo.",
    },
)


MAIN_RESULT_SCRIPTS: tuple[str, ...] = (
    "scripts/e11_run_experiments.py",
    "scripts/e11_make_figures.py",
    "scripts/e11_run_equal_update_control.py",
    "scripts/e11_run_head_tail_interference.py",
    "scripts/e11_run_long_tail_one_step.py",
    "scripts/e11_run_long_tail_muon_bridge.py",
    "scripts/e11_run_long_tail_practical_muon_bridge.py",
    "scripts/e11_run_long_tail_practical_training.py",
    "scripts/e11_run_long_tail_practical_training_lr_sweep.py",
    "scripts/e11_run_long_tail_forgetting.py",
    "scripts/e11_run_long_tail_layerwise.py",
    "scripts/e11_run_spectral_allocation_probe.py",
)


MAIN_EVIDENCE_STAGES: tuple[dict[str, str], ...] = (
    {
        "stage": "Core trajectories",
        "command": "python3 scripts/e11_run_experiments.py",
        "produces": "results/e11/*",
        "paper_role": "Base Adam/Muon trajectories and raw diagnostics.",
    },
    {
        "stage": "Core figures",
        "command": "python3 scripts/e11_make_figures.py",
        "produces": "figures/e11/*",
        "paper_role": "Static diagnostic figures for the base run.",
    },
    {
        "stage": "Matched-update control",
        "command": "python3 scripts/e11_run_equal_update_control.py",
        "produces": "results/e11_equal_update/* and figures/e11_equal_update/*",
        "paper_role": "Background E11 optimizer-geometry control; not a current head-to-tail main figure.",
    },
    {
        "stage": "Head-to-tail interference probe",
        "command": "python3 scripts/e11_run_head_tail_interference.py",
        "produces": "results/e11_head_tail_interference/* and figures/e11_head_tail_interference/*",
        "paper_role": "Synthetic matched-head-gain evidence for the head-to-tail interference condition.",
    },
    {
        "stage": "Long-tailed one-step diagnostic",
        "command": "python3 scripts/e11_run_long_tail_one_step.py",
        "produces": "results/e11_long_tail_one_step/* and figures/e11_long_tail_one_step/*",
        "paper_role": "Real-data matched-head-gain check on held-out tail examples.",
    },
    {
        "stage": "Long-tailed Muon-style compatibility diagnostic",
        "command": "python3 scripts/e11_run_long_tail_muon_bridge.py",
        "produces": "results/e11_long_tail_muon_bridge/* and figures/e11_long_tail_muon_bridge/*",
        "paper_role": "Fixed-checkpoint compatibility check from ideal polar(G_t) to momentum polar(M_t) and Newton-Schulz Muon-style directions.",
    },
    {
        "stage": "Long-tailed practical-Muon trajectory compatibility",
        "command": "python3 scripts/e11_run_long_tail_practical_muon_bridge.py",
        "produces": "results/e11_long_tail_practical_muon_bridge/* and figures/e11_long_tail_practical_muon_bridge/*",
        "paper_role": "Short trajectory-level check of momentum and Newton-Schulz Muon-style directions at sampled states.",
    },
    {
        "stage": "Long-tailed practical training diagnostic",
        "command": "python3 scripts/e11_run_long_tail_practical_training.py",
        "produces": "results/e11_long_tail_practical_training/* and figures/e11_long_tail_practical_training/*",
        "paper_role": "Practical Adam-vs-NS-Muon-style imbalanced mini-batch training sanity check with final tail metrics.",
    },
    {
        "stage": "Long-tailed practical training LR sensitivity",
        "command": "python3 scripts/e11_run_long_tail_practical_training_lr_sweep.py",
        "produces": "results/e11_long_tail_practical_training_lr_sweep/* and figures/e11_long_tail_practical_training_lr_sweep/*",
        "paper_role": "Learning-rate sensitivity check for the practical NS-Muon-style training diagnostic.",
    },
    {
        "stage": "Head-only forgetting probe",
        "command": "python3 scripts/e11_run_long_tail_forgetting.py",
        "produces": "results/e11_long_tail_forgetting/* and figures/e11_long_tail_forgetting/*",
        "paper_role": "Short-horizon held-out tail drift under matched head-only update schedules.",
    },
    {
        "stage": "Long-tailed layerwise diagnostic",
        "command": "python3 scripts/e11_run_long_tail_layerwise.py",
        "produces": "results/e11_long_tail_layerwise/* and figures/e11_long_tail_layerwise/*",
        "paper_role": "Layerwise finite-difference tail JVP and matched-head-gain drift evidence.",
    },
    {
        "stage": "Spectral allocation probe",
        "command": "python3 scripts/e11_run_spectral_allocation_probe.py",
        "produces": "results/e11_spectral_allocation_probe/* and figures/e11_spectral_allocation_probe/*",
        "paper_role": "Mechanism background for norm-budget geometry; appendix evidence for the current head-to-tail paper.",
    },
    {
        "stage": "Mechanism boundary map",
        "command": "python3 scripts/e11_write_mechanism_boundary.py",
        "produces": "results/e11_mechanism_boundary/mechanism_boundary_map.csv",
        "paper_role": "Legacy E11 boundary summary used as background, not the current head-to-tail result table.",
    },
)


APPENDIX_RUNNER_SCRIPTS: tuple[str, ...] = (
    "scripts/e11_run_overlap_followup.py",
    "scripts/e11_run_hyperparam_sweep.py",
    "scripts/e11_run_target_update_sweep.py",
    "scripts/e11_run_mlp_width_sweep.py",
    "scripts/e11_run_mlp_per_layer_control.py",
    "scripts/e11_run_mlp_layer_hybrid.py",
    "scripts/e11_run_mnist_mlp_probe.py",
    "scripts/e11_run_deep_mnist_mlp_probe.py",
    "scripts/e11_run_mnist_patch_probe.py",
    "scripts/e11_run_mnist_conv_probe.py",
    "scripts/e11_run_stateless_direction_ablation.py",
    "scripts/e11_run_stateless_optimizer_trajectory.py",
    "scripts/e11_run_singular_vector_trajectory.py",
    "scripts/e11_run_singular_vector_swap_probe.py",
    "scripts/e11_run_natural_update_swap_probe.py",
    "scripts/e11_run_optimizer_switch_probe.py",
    "scripts/e11_run_optimizer_switch_reset_control.py",
    "scripts/e11_run_optimizer_switch_horizon_sweep.py",
    "scripts/e11_run_optimizer_switch_lr_sweep.py",
    "scripts/e11_run_boundary_predictor.py",
)


APPENDIX_RUNNER_DESCRIPTIONS: dict[str, str] = {
    "scripts/e11_run_overlap_followup.py": "Focused overlap settings used by early boundary and guardrail claims.",
    "scripts/e11_run_hyperparam_sweep.py": "Learning-rate sensitivity and best-over-LR guardrail.",
    "scripts/e11_run_target_update_sweep.py": "Target update-size boundary control.",
    "scripts/e11_run_mlp_width_sweep.py": "MLP width boundary and transition control.",
    "scripts/e11_run_mlp_per_layer_control.py": "Per-layer matched-update boundary control.",
    "scripts/e11_run_mlp_layer_hybrid.py": "Layerwise optimizer hybrid control for MLP update allocation.",
    "scripts/e11_run_mnist_mlp_probe.py": "Small neural sanity benchmark.",
    "scripts/e11_run_deep_mnist_mlp_probe.py": "Deeper all-matrix MNIST sanity benchmark.",
    "scripts/e11_run_mnist_patch_probe.py": "Matrix-only MNIST patch classifier sanity benchmark.",
    "scripts/e11_run_mnist_conv_probe.py": "Small true Conv2d MNIST sanity benchmark with flattened-kernel Muon view.",
    "scripts/e11_run_stateless_direction_ablation.py": "One-step stateless direction mechanism control.",
    "scripts/e11_run_stateless_optimizer_trajectory.py": "Short stateless trajectory mechanism control.",
    "scripts/e11_run_singular_vector_trajectory.py": "Gradient/update singular-vector overlap trajectory diagnostic.",
    "scripts/e11_run_singular_vector_swap_probe.py": "Singular-vector swap mechanism control.",
    "scripts/e11_run_natural_update_swap_probe.py": "Natural update-vector and singular geometry control.",
    "scripts/e11_run_optimizer_switch_probe.py": "Optimizer-state continuation probe.",
    "scripts/e11_run_optimizer_switch_reset_control.py": "Reset-state control for optimizer switching.",
    "scripts/e11_run_optimizer_switch_horizon_sweep.py": "Continuation horizon sensitivity control.",
    "scripts/e11_run_optimizer_switch_lr_sweep.py": "Continuation negative control against long-horizon overclaiming.",
    "scripts/e11_run_boundary_predictor.py": "Leave-setting-out predictive-boundary baseline.",
}


PAPER_ASSET_SCRIPTS: tuple[str, ...] = (
    "scripts/e11_write_discussion_outline.py",
    "scripts/e11_write_head_tail_interference_note.py",
    "scripts/e11_write_long_tail_one_step_note.py",
    "scripts/e11_write_long_tail_muon_bridge_note.py",
    "scripts/e11_write_long_tail_practical_muon_bridge_note.py",
    "scripts/e11_write_long_tail_practical_training_note.py",
    "scripts/e11_write_long_tail_practical_training_lr_sweep_note.py",
    "scripts/e11_write_long_tail_forgetting_note.py",
    "scripts/e11_write_long_tail_layerwise_note.py",
    "scripts/e11_write_head_tail_paper_results.py",
    "scripts/e11_write_claim_audit.py",
    "scripts/e11_write_evidence_index.py",
    "scripts/e11_write_research_synthesis.py",
    "scripts/e11_write_research_direction_map.py",
    "scripts/e11_write_paper_readiness_audit.py",
    "scripts/e11_write_paper_skeleton.py",
    "scripts/e11_write_main_paper_package.py",
    "scripts/e11_write_main_figure_captions.py",
    "scripts/e11_write_notation_glossary.py",
    "scripts/e11_write_quantitative_claim_ledger.py",
    "scripts/e11_write_paper_numbers.py",
    "scripts/e11_write_paper_figures.py",
    "scripts/e11_write_reproduction_checklist.py",
    "scripts/e11_write_reviewer_risk_audit.py",
    "scripts/e11_write_artifact_manifest.py",
)


LEGACY_GUARDRAIL_SCRIPTS: tuple[str, ...] = (
    "scripts/e11_write_activation_perturbation_note.py",
    "scripts/e11_write_mechanism_ladder.py",
    "scripts/e11_write_cross_task_signature.py",
    "scripts/e11_write_mechanism_boundary.py",
    "scripts/e11_write_theory_note.py",
    "scripts/e11_write_mechanism_theorem_bridge.py",
    "scripts/e11_write_boundary_predictor_audit.py",
    "scripts/e11_write_optimizer_ablation_map.py",
    "scripts/e11_write_optimizer_invariance_audit.py",
)
