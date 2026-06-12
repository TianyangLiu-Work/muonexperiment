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
        "role": "Generated paper-facing Markdown evidence and synthesis.",
        "commit_policy": "commit",
    },
    {
        "path": "results",
        "role": "Generated CSV evidence used by discussion artifacts and validator.",
        "commit_policy": "commit current E11 evidence set",
    },
    {
        "path": "figures",
        "role": "Static figures used by paper-facing Markdown artifacts.",
        "commit_policy": "commit static E11 figures",
    },
    {
        "path": "configs",
        "role": "Experiment configuration snapshots.",
        "commit_policy": "commit",
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
    "results/e11_equal_update/step_metrics.csv",
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
        "paper_role": "Main Figure 1 and Figure 2 evidence.",
    },
    {
        "stage": "Spectral allocation probe",
        "command": "python3 scripts/e11_run_spectral_allocation_probe.py",
        "produces": "results/e11_spectral_allocation_probe/* and figures/e11_spectral_allocation_probe/*",
        "paper_role": "Main Figure 3 evidence.",
    },
    {
        "stage": "Mechanism boundary map",
        "command": "python3 scripts/e11_write_mechanism_boundary.py",
        "produces": "results/e11_mechanism_boundary/mechanism_boundary_map.csv",
        "paper_role": "Main Table 1 evidence after follow-up tables exist.",
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
    "scripts/e11_write_discussion.py",
    "scripts/e11_write_claim_audit.py",
    "scripts/e11_write_mechanism_ladder.py",
    "scripts/e11_run_mnist_mlp_probe.py",
    "scripts/e11_run_deep_mnist_mlp_probe.py",
    "scripts/e11_run_mnist_patch_probe.py",
    "scripts/e11_run_mnist_conv_probe.py",
    "scripts/e11_run_stateless_direction_ablation.py",
    "scripts/e11_run_stateless_optimizer_trajectory.py",
    "scripts/e11_run_boundary_predictor.py",
    "scripts/e11_write_cross_task_signature.py",
    "scripts/e11_write_mechanism_boundary.py",
    "scripts/e11_write_theory_note.py",
    "scripts/e11_write_mechanism_theorem_bridge.py",
    "scripts/e11_write_boundary_predictor_audit.py",
    "scripts/e11_write_optimizer_ablation_map.py",
    "scripts/e11_write_optimizer_invariance_audit.py",
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
    "scripts/e11_write_reproduction_checklist.py",
    "scripts/e11_write_reviewer_risk_audit.py",
    "scripts/e11_write_artifact_manifest.py",
)
