from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry import default_config
from e11_condition_geometry.artifacts import (
    APPENDIX_RUNNER_SCRIPTS,
    ARTIFACT_DIRS,
    IGNORE_POLICY,
    KEY_TABLES,
    MAIN_RESULT_SCRIPTS,
)
import pandas as pd


def assert_no_unguarded_overclaims(paths: list[Path]) -> None:
    risky_phrases = [
        "Muon is generally better",
        "Muon is globally better",
        "Muon is generally more stable",
        "universally better",
        "higher rank or stable rank directly implies",
        "already a predictive theory",
    ]
    guardrail_terms = [
        "avoid",
        "but not",
        "conditional",
        "do not",
        "do **not**",
        "does not",
        "does **not**",
        "not ",
        "**not**",
        "not yet",
        "not supported",
        "rather than",
        "reject",
        "should not",
    ]
    violations: list[str] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        lowered = text.lower()
        for phrase in risky_phrases:
            lowered_phrase = phrase.lower()
            start = 0
            while True:
                index = lowered.find(lowered_phrase, start)
                if index == -1:
                    break
                context = lowered[max(0, index - 180) : index + len(lowered_phrase) + 180]
                if not any(term in context for term in guardrail_terms):
                    violations.append(f"{path}: unguarded overclaim phrase `{phrase}`")
                start = index + len(lowered_phrase)
    if violations:
        raise AssertionError("unguarded paper-facing overclaims found: " + "; ".join(violations))


def assert_valid_artifact_manifest(manifest_json: dict) -> None:
    if "validation_command" not in manifest_json:
        raise AssertionError("artifact manifest JSON must include validation_command")
    required_manifest_dirs = {item["path"] for item in ARTIFACT_DIRS}
    manifest_dirs = {item.get("path") for item in manifest_json.get("artifact_dirs", [])}
    if not required_manifest_dirs.issubset(manifest_dirs):
        raise AssertionError(f"artifact manifest missing directories: {required_manifest_dirs - manifest_dirs}")
    required_manifest_tables = set(KEY_TABLES)
    manifest_tables = {item.get("path") for item in manifest_json.get("key_tables", [])}
    if not required_manifest_tables.issubset(manifest_tables):
        raise AssertionError(f"artifact manifest missing key tables: {required_manifest_tables - manifest_tables}")
    bad_manifest_tables = [
        item.get("path")
        for item in manifest_json.get("key_tables", [])
        if item.get("path") in required_manifest_tables and int(item.get("rows", 0)) <= 0
    ]
    if bad_manifest_tables:
        raise AssertionError(f"artifact manifest has non-positive row counts: {bad_manifest_tables}")
    required_ignored = {item["path_or_pattern"] for item in IGNORE_POLICY}
    manifest_ignored = {item.get("path_or_pattern") for item in manifest_json.get("ignore_policy", [])}
    if not required_ignored.issubset(manifest_ignored):
        raise AssertionError(f"artifact manifest missing ignore policy entries: {required_ignored - manifest_ignored}")


def assert_batch_activation_contract(path: Path, frame: pd.DataFrame) -> None:
    """Validate the training-vs-diagnostic batch contract for step metrics."""

    if "problem_family" not in frame.columns:
        return
    required_columns = {"train_batch_size", "num_samples", "diagnostic_A_definition"}
    missing_columns = required_columns - set(frame.columns)
    if missing_columns:
        raise AssertionError(f"{path} missing batch/activation columns: {missing_columns}")
    family = frame["problem_family"].astype(str)
    mlp_rows = frame[family.str.contains("MLP", na=False)]
    patch_rows = frame[family.eq("MNISTPatchClassifier")]
    conv_rows = frame[family.eq("MNISTConvNet")]
    neural_rows = pd.concat([mlp_rows, patch_rows, conv_rows], ignore_index=True)
    non_neural_rows = frame[
        ~family.str.contains("MLP", na=False) & ~family.eq("MNISTPatchClassifier") & ~family.eq("MNISTConvNet")
    ]
    if neural_rows.empty and non_neural_rows.empty:
        return
    if not mlp_rows.empty:
        if (mlp_rows["train_batch_size"] >= mlp_rows["num_samples"]).any():
            raise AssertionError(f"{path} MLP rows must use a strict training mini-batch")
        if set(mlp_rows["diagnostic_A_definition"]) != {"full_layer_input_activation"}:
            raise AssertionError(f"{path} MLP diagnostics must use full_layer_input_activation")
    if not patch_rows.empty:
        if (patch_rows["train_batch_size"] >= patch_rows["num_samples"]).any():
            raise AssertionError(f"{path} MNIST patch rows must use a strict training mini-batch")
        if set(patch_rows["diagnostic_A_definition"]) != {"full_patch_and_classifier_activation"}:
            raise AssertionError(f"{path} MNIST patch diagnostics must use full_patch_and_classifier_activation")
    if not conv_rows.empty:
        if (conv_rows["train_batch_size"] >= conv_rows["num_samples"]).any():
            raise AssertionError(f"{path} MNIST ConvNet rows must use a strict training mini-batch")
        if set(conv_rows["diagnostic_A_definition"]) != {"full_conv_patch_and_classifier_activation"}:
            raise AssertionError(f"{path} MNIST ConvNet diagnostics must use full_conv_patch_and_classifier_activation")
    if not non_neural_rows.empty:
        if (non_neural_rows["train_batch_size"] != non_neural_rows["num_samples"]).any():
            raise AssertionError(f"{path} non-neural rows must use full-batch optimization")


def main() -> None:
    config = default_config()
    equal_output_dir = Path("results/e11_equal_update")
    equal_figure_dir = Path("figures/e11_equal_update")
    required = [
        config.output_dir / "step_metrics.csv",
        config.output_dir / "layer_metrics.csv",
        config.output_dir / "performance_summary.csv",
        config.output_dir / "geometry_summary.csv",
        config.output_dir / "prediction_summary.csv",
        config.output_dir / "run_dynamics_summary.csv",
        config.output_dir / "volatility_summary.csv",
        config.output_dir / "update_spectrum_summary.csv",
        config.output_dir / "update_transmission_summary.csv",
        config.output_dir / "first_order_calibration_summary.csv",
        config.output_dir / "polar_alignment_summary.csv",
        config.output_dir / "first_order_pair_summary.csv",
        config.output_dir / "win_condition_summary.csv",
        config.output_dir / "win_prediction_generalization_summary.csv",
        config.output_dir / "win_feature_overlap_summary.csv",
        config.figure_dir / "loss_curves.png",
        config.figure_dir / "geometry_separation.png",
        config.figure_dir / "predicted_decrease_vs_observed.png",
        config.figure_dir / "condition_score_trajectories.png",
        config.figure_dir / "mean_3d_condition_loss.png",
        config.figure_dir / "layerwise_3d_condition_loss.png",
        config.figure_dir / "volatility_robustness.png",
        config.figure_dir / "update_spectrum_robustness.png",
        config.figure_dir / "update_transmission_heatmap.png",
        config.figure_dir / "first_order_calibration.png",
        config.figure_dir / "polar_alignment_identity.png",
        config.figure_dir / "first_order_pair_comparison.png",
        config.figure_dir / "win_condition_summary.png",
        config.figure_dir / "win_prediction_generalization.png",
        config.figure_dir / "win_feature_overlap.png",
        equal_output_dir / "step_metrics.csv",
        equal_output_dir / "layer_metrics.csv",
        equal_output_dir / "performance_summary.csv",
        equal_output_dir / "geometry_summary.csv",
        equal_output_dir / "prediction_summary.csv",
        equal_output_dir / "run_dynamics_summary.csv",
        equal_output_dir / "volatility_summary.csv",
        equal_output_dir / "update_spectrum_summary.csv",
        equal_output_dir / "update_transmission_summary.csv",
        equal_output_dir / "first_order_calibration_summary.csv",
        equal_output_dir / "polar_alignment_summary.csv",
        equal_output_dir / "first_order_pair_summary.csv",
        equal_output_dir / "win_condition_summary.csv",
        equal_output_dir / "win_prediction_generalization_summary.csv",
        equal_output_dir / "win_feature_overlap_summary.csv",
        equal_figure_dir / "volatility_robustness.png",
        equal_figure_dir / "update_spectrum_robustness.png",
        equal_figure_dir / "update_transmission_heatmap.png",
        equal_figure_dir / "first_order_calibration.png",
        equal_figure_dir / "polar_alignment_identity.png",
        equal_figure_dir / "first_order_pair_comparison.png",
        equal_figure_dir / "win_condition_summary.png",
        equal_figure_dir / "win_prediction_generalization.png",
        equal_figure_dir / "win_feature_overlap.png",
        Path("results/e11_candidate_scan") / "initial_geometry_scan.csv",
        Path("results/e11_candidate_scan") / "candidate_setting_summary.csv",
        Path("results/e11_candidate_scan") / "current_equal_update_support.csv",
        Path("figures/e11_candidate_scan") / "initial_geometry_support.png",
        Path("discussion/e11_candidate_overlap_scan.md"),
        Path("results/e11_overlap_followup") / "step_metrics.csv",
        Path("results/e11_overlap_followup") / "layer_metrics.csv",
        Path("results/e11_overlap_followup") / "first_order_pair_summary.csv",
        Path("results/e11_overlap_followup") / "first_order_calibration_summary.csv",
        Path("figures/e11_overlap_followup") / "first_order_pair_comparison.png",
        Path("figures/e11_overlap_followup") / "first_order_calibration.png",
        Path("figures/e11_overlap_followup") / "loss_curves.png",
        Path("discussion/e11_overlap_followup.md"),
        Path("results/e11_mlp_width_sweep") / "step_metrics.csv",
        Path("results/e11_mlp_width_sweep") / "layer_metrics.csv",
        Path("results/e11_mlp_width_sweep") / "width_pair_summary.csv",
        Path("results/e11_mlp_width_sweep") / "width_layer_geometry_summary.csv",
        Path("results/e11_mlp_width_sweep") / "width_mechanism_coupling_summary.csv",
        Path("results/e11_mlp_width_sweep") / "first_order_calibration_summary.csv",
        Path("figures/e11_mlp_width_sweep") / "width_first_order_transition.png",
        Path("figures/e11_mlp_width_sweep") / "width_layer_mechanism.png",
        Path("figures/e11_mlp_width_sweep") / "width_mechanism_coupling.png",
        Path("discussion/e11_mlp_width_sweep.md"),
        Path("results/e11_mlp_layer_hybrid") / "step_metrics.csv",
        Path("results/e11_mlp_layer_hybrid") / "layer_metrics.csv",
        Path("results/e11_mlp_layer_hybrid") / "hybrid_ratio_summary.csv",
        Path("results/e11_mlp_layer_hybrid") / "hybrid_layer_ratio_summary.csv",
        Path("results/e11_mlp_layer_hybrid") / "hybrid_update_allocation_summary.csv",
        Path("results/e11_mlp_layer_hybrid") / "layer_inner_summary.csv",
        Path("results/e11_mlp_layer_hybrid") / "first_order_calibration_summary.csv",
        Path("figures/e11_mlp_layer_hybrid") / "hybrid_first_order_ratios.png",
        Path("figures/e11_mlp_layer_hybrid") / "hybrid_layer_contributions.png",
        Path("figures/e11_mlp_layer_hybrid") / "hybrid_update_allocation.png",
        Path("discussion/e11_mlp_layer_hybrid.md"),
        Path("results/e11_hyperparam_sweep") / "raw_step_metrics.csv",
        Path("results/e11_hyperparam_sweep") / "raw_layer_metrics.csv",
        Path("results/e11_hyperparam_sweep") / "equal_step_metrics.csv",
        Path("results/e11_hyperparam_sweep") / "equal_layer_metrics.csv",
        Path("results/e11_hyperparam_sweep") / "outcome_metrics.csv",
        Path("results/e11_hyperparam_sweep") / "lr_curve_summary.csv",
        Path("results/e11_hyperparam_sweep") / "best_by_seed.csv",
        Path("results/e11_hyperparam_sweep") / "best_lr_frequency.csv",
        Path("results/e11_hyperparam_sweep") / "best_pair_summary.csv",
        Path("results/e11_hyperparam_sweep") / "first_order_calibration_summary.csv",
        Path("results/e11_hyperparam_sweep") / "update_spectrum_summary.csv",
        Path("figures/e11_hyperparam_sweep") / "hyperparam_best_ratios.png",
        Path("figures/e11_hyperparam_sweep") / "hyperparam_lr_curves.png",
        Path("discussion/e11_hyperparam_sweep.md"),
        Path("results/e11_target_update_sweep") / "step_metrics.csv",
        Path("results/e11_target_update_sweep") / "layer_metrics.csv",
        Path("results/e11_target_update_sweep") / "paired_step_metrics.csv",
        Path("results/e11_target_update_sweep") / "target_pair_summary.csv",
        Path("results/e11_target_update_sweep") / "final_outcomes.csv",
        Path("results/e11_target_update_sweep") / "best_target_summary.csv",
        Path("results/e11_target_update_sweep") / "first_order_calibration_summary.csv",
        Path("results/e11_target_update_sweep") / "update_spectrum_summary.csv",
        Path("figures/e11_target_update_sweep") / "target_update_first_order_ratios.png",
        Path("figures/e11_target_update_sweep") / "target_update_best_ratios.png",
        Path("discussion/e11_target_update_sweep.md"),
        Path("results/e11_mlp_per_layer_control") / "step_metrics.csv",
        Path("results/e11_mlp_per_layer_control") / "layer_metrics.csv",
        Path("results/e11_mlp_per_layer_control") / "paired_step_metrics.csv",
        Path("results/e11_mlp_per_layer_control") / "pair_summary.csv",
        Path("results/e11_mlp_per_layer_control") / "final_outcomes.csv",
        Path("results/e11_mlp_per_layer_control") / "best_target_summary.csv",
        Path("results/e11_mlp_per_layer_control") / "first_order_calibration_summary.csv",
        Path("results/e11_mlp_per_layer_control") / "update_spectrum_summary.csv",
        Path("figures/e11_mlp_per_layer_control") / "per_layer_control_first_order_ratios.png",
        Path("discussion/e11_mlp_per_layer_control.md"),
        Path("results/e11_mnist_mlp_probe") / "step_metrics.csv",
        Path("results/e11_mnist_mlp_probe") / "layer_metrics.csv",
        Path("results/e11_mnist_mlp_probe") / "paired_step_metrics.csv",
        Path("results/e11_mnist_mlp_probe") / "pair_summary.csv",
        Path("results/e11_mnist_mlp_probe") / "first_order_calibration_summary.csv",
        Path("results/e11_mnist_mlp_probe") / "update_spectrum_summary.csv",
        Path("figures/e11_mnist_mlp_probe") / "mnist_mlp_first_order_ratios.png",
        Path("discussion/e11_mnist_mlp_probe.md"),
        Path("results/e11_deep_mnist_mlp_probe") / "step_metrics.csv",
        Path("results/e11_deep_mnist_mlp_probe") / "layer_metrics.csv",
        Path("results/e11_deep_mnist_mlp_probe") / "paired_step_metrics.csv",
        Path("results/e11_deep_mnist_mlp_probe") / "pair_summary.csv",
        Path("results/e11_deep_mnist_mlp_probe") / "first_order_calibration_summary.csv",
        Path("results/e11_deep_mnist_mlp_probe") / "update_spectrum_summary.csv",
        Path("figures/e11_deep_mnist_mlp_probe") / "deep_mnist_mlp_first_order_ratios.png",
        Path("discussion/e11_deep_mnist_mlp_probe.md"),
        Path("results/e11_mnist_patch_probe") / "step_metrics.csv",
        Path("results/e11_mnist_patch_probe") / "layer_metrics.csv",
        Path("results/e11_mnist_patch_probe") / "paired_step_metrics.csv",
        Path("results/e11_mnist_patch_probe") / "pair_summary.csv",
        Path("results/e11_mnist_patch_probe") / "first_order_calibration_summary.csv",
        Path("results/e11_mnist_patch_probe") / "update_spectrum_summary.csv",
        Path("figures/e11_mnist_patch_probe") / "mnist_patch_first_order_ratios.png",
        Path("discussion/e11_mnist_patch_probe.md"),
        Path("results/e11_mnist_conv_probe") / "step_metrics.csv",
        Path("results/e11_mnist_conv_probe") / "layer_metrics.csv",
        Path("results/e11_mnist_conv_probe") / "paired_step_metrics.csv",
        Path("results/e11_mnist_conv_probe") / "pair_summary.csv",
        Path("results/e11_mnist_conv_probe") / "first_order_calibration_summary.csv",
        Path("results/e11_mnist_conv_probe") / "update_spectrum_summary.csv",
        Path("figures/e11_mnist_conv_probe") / "mnist_conv_first_order_ratios.png",
        Path("discussion/e11_mnist_conv_probe.md"),
        Path("results/e11_stateless_direction_ablation") / "stateless_direction_rows.csv",
        Path("results/e11_stateless_direction_ablation") / "stateless_direction_pairs.csv",
        Path("results/e11_stateless_direction_ablation") / "stateless_direction_summary.csv",
        Path("figures/e11_stateless_direction_ablation") / "stateless_direction_ratios.png",
        Path("discussion/e11_stateless_direction_ablation.md"),
        Path("results/e11_stateless_optimizer_trajectory") / "stateless_optimizer_step_metrics.csv",
        Path("results/e11_stateless_optimizer_trajectory") / "stateless_optimizer_layer_metrics.csv",
        Path("results/e11_stateless_optimizer_trajectory") / "stateless_optimizer_outcomes.csv",
        Path("results/e11_stateless_optimizer_trajectory") / "stateless_optimizer_pairs.csv",
        Path("results/e11_stateless_optimizer_trajectory") / "stateless_optimizer_summary.csv",
        Path("figures/e11_stateless_optimizer_trajectory") / "stateless_optimizer_trajectory_ratios.png",
        Path("discussion/e11_stateless_optimizer_trajectory.md"),
        Path("results/e11_mechanism_ladder") / "intervention_direction_summary.csv",
        Path("figures/e11_mechanism_ladder") / "mechanism_direction_ladder.png",
        Path("discussion/e11_mechanism_ladder.md"),
        Path("results/e11_spectral_allocation_probe") / "probe_rows.csv",
        Path("results/e11_spectral_allocation_probe") / "spectral_allocation_summary.csv",
        Path("results/e11_spectral_allocation_probe") / "direction_level_summary.csv",
        Path("figures/e11_spectral_allocation_probe") / "spectral_allocation_ratios.png",
        Path("discussion/e11_spectral_allocation_probe.md"),
        Path("results/e11_singular_vector_trajectory") / "gradient_subspace_rows.csv",
        Path("results/e11_singular_vector_trajectory") / "update_subspace_rows.csv",
        Path("results/e11_singular_vector_trajectory") / "subspace_step_summary.csv",
        Path("results/e11_singular_vector_trajectory") / "subspace_final_summary.csv",
        Path("figures/e11_singular_vector_trajectory") / "singular_vector_overlap_trajectory.png",
        Path("discussion/e11_singular_vector_trajectory.md"),
        Path("results/e11_singular_vector_swap_probe") / "swap_probe_rows.csv",
        Path("results/e11_singular_vector_swap_probe") / "swap_ratio_summary.csv",
        Path("results/e11_singular_vector_swap_probe") / "swap_step_summary.csv",
        Path("figures/e11_singular_vector_swap_probe") / "singular_vector_swap_ratios.png",
        Path("discussion/e11_singular_vector_swap_probe.md"),
        Path("results/e11_natural_update_swap_probe") / "natural_update_swap_rows.csv",
        Path("results/e11_natural_update_swap_probe") / "natural_update_swap_summary.csv",
        Path("results/e11_natural_update_swap_probe") / "natural_update_swap_step_summary.csv",
        Path("figures/e11_natural_update_swap_probe") / "natural_update_swap_ratios.png",
        Path("figures/e11_natural_update_swap_probe") / "natural_update_swap_target_sweep.png",
        Path("discussion/e11_natural_update_swap_probe.md"),
        Path("results/e11_optimizer_switch_probe") / "optimizer_switch_rows.csv",
        Path("results/e11_optimizer_switch_probe") / "optimizer_switch_summary.csv",
        Path("figures/e11_optimizer_switch_probe") / "optimizer_switch_total_decrease.png",
        Path("discussion/e11_optimizer_switch_probe.md"),
        Path("results/e11_optimizer_switch_reset_control") / "optimizer_switch_reset_rows.csv",
        Path("results/e11_optimizer_switch_reset_control") / "optimizer_switch_reset_summary.csv",
        Path("figures/e11_optimizer_switch_reset_control") / "optimizer_switch_reset_control.png",
        Path("discussion/e11_optimizer_switch_reset_control.md"),
        Path("results/e11_optimizer_switch_horizon_sweep") / "optimizer_switch_horizon_rows.csv",
        Path("results/e11_optimizer_switch_horizon_sweep") / "optimizer_switch_horizon_summary.csv",
        Path("figures/e11_optimizer_switch_horizon_sweep") / "optimizer_switch_horizon_sweep.png",
        Path("discussion/e11_optimizer_switch_horizon_sweep.md"),
        Path("results/e11_optimizer_switch_lr_sweep") / "optimizer_switch_lr_rows.csv",
        Path("results/e11_optimizer_switch_lr_sweep") / "optimizer_switch_lr_best.csv",
        Path("results/e11_optimizer_switch_lr_sweep") / "optimizer_switch_lr_summary.csv",
        Path("results/e11_optimizer_switch_lr_sweep") / "optimizer_switch_lr_frequency.csv",
        Path("figures/e11_optimizer_switch_lr_sweep") / "optimizer_switch_lr_sweep.png",
        Path("discussion/e11_optimizer_switch_lr_sweep.md"),
        Path("discussion/e11_evidence_index.md"),
        Path("discussion/e11_optimizer_invariance_audit.md"),
        Path("results/e11_cross_task_signature") / "cross_task_signature_rows.csv",
        Path("results/e11_cross_task_signature") / "cross_task_signature_summary.csv",
        Path("discussion/e11_cross_task_signature.md"),
        Path("results/e11_boundary_predictor") / "boundary_predictor_rows.csv",
        Path("results/e11_boundary_predictor") / "boundary_predictor_summary.csv",
        Path("results/e11_boundary_predictor") / "boundary_predictor_uncertainty.csv",
        Path("discussion/e11_boundary_predictor.md"),
        Path("discussion/e11_boundary_predictor_audit.md"),
        Path("results/e11_mechanism_boundary") / "mechanism_boundary_map.csv",
        Path("discussion/e11_mechanism_boundary.md"),
        Path("discussion/e11_theory_note.md"),
        Path("discussion/e11_mechanism_theorem_bridge.md"),
        Path("discussion/e11_optimizer_ablation_map.md"),
        Path("discussion/e11_research_synthesis.md"),
        Path("discussion/e11_claim_validity_audit.md"),
        Path("discussion/e11_paper_readiness_audit.md"),
        Path("discussion/e11_paper_skeleton.md"),
        Path("discussion/e11_main_paper_package.md"),
        Path("discussion/e11_main_figure_captions.md"),
        Path("discussion/e11_notation_glossary.md"),
        Path("discussion/e11_quantitative_claim_ledger.md"),
        Path("discussion/e11_paper_numbers.tex"),
        Path("discussion/e11_reproduction_checklist.md"),
        Path("discussion/e11_reviewer_risk_audit.md"),
        Path("discussion/e11_artifact_manifest.md"),
        Path("results/e11_artifact_manifest.json"),
        Path("Makefile"),
        Path("README_E11.md"),
        config.discussion_path,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing required outputs: {missing}")
    makefile_text = Path("Makefile").read_text(encoding="utf-8")
    required_makefile_phrases = [
        "e11-all-results: e11-main-results e11-appendix-results",
        *MAIN_RESULT_SCRIPTS,
        *APPENDIX_RUNNER_SCRIPTS,
    ]
    missing_makefile_phrases = [phrase for phrase in required_makefile_phrases if phrase not in makefile_text]
    if missing_makefile_phrases:
        raise AssertionError(f"Makefile missing required E11 reproduction entries: {missing_makefile_phrases}")
    readme_text = Path("README_E11.md").read_text(encoding="utf-8")
    required_readme_phrases = [
        "Equal-update `nrUpdate` ratio is about `2.033`",
        "Equal-update `stUpdate` ratio is about `5.215`",
        "Spearman correlation is about `0.9206`",
        "make e11-all-results",
        "`diagnostic_A_definition == full_layer_input_activation`",
        *MAIN_RESULT_SCRIPTS,
        *APPENDIX_RUNNER_SCRIPTS,
    ]
    missing_readme_phrases = [phrase for phrase in required_readme_phrases if phrase not in readme_text]
    if missing_readme_phrases:
        raise AssertionError(f"README_E11.md missing current paper-facing summary entries: {missing_readme_phrases}")
    step_metric_paths = sorted(Path("results").glob("**/step_metrics.csv"))
    step_metric_paths.extend(
        [
            Path("results/e11_hyperparam_sweep/raw_step_metrics.csv"),
            Path("results/e11_hyperparam_sweep/equal_step_metrics.csv"),
            Path("results/e11_stateless_optimizer_trajectory/stateless_optimizer_step_metrics.csv"),
        ]
    )
    for step_metric_path in step_metric_paths:
        if step_metric_path.exists():
            assert_batch_activation_contract(step_metric_path, pd.read_csv(step_metric_path))
    steps = pd.read_csv(config.output_dir / "step_metrics.csv")
    expected_runs = len(config.specs) * len(config.algos) * len(config.seeds)
    families = set(steps["problem_family"])
    expected_families = {spec.family for spec in config.specs}
    if steps["run_id"].nunique() != expected_runs:
        raise AssertionError(f"run count mismatch: expected {expected_runs}, got {steps['run_id'].nunique()}")
    if families != expected_families:
        raise AssertionError(f"family mismatch: expected {expected_families}, got {families}")
    if steps[["loss", "recovery_error", "nrG", "stA", "condition_score"]].isna().any().any():
        raise AssertionError("unexpected NaN in core step metrics")
    nonfinal = steps[steps["delta_loss"].notna()]
    if nonfinal[["update_fro_norm", "relative_update_fro_norm"]].isna().any().any():
        raise AssertionError("missing update norm on non-final steps")
    if nonfinal[["nrUpdate", "stUpdate", "nrUpdateFrac", "stUpdateFrac", "update_flatness"]].isna().any().any():
        raise AssertionError("missing update spectral metrics on non-final steps")
    if nonfinal[["update_grad_inner", "update_grad_cosine", "update_grad_per_update_norm"]].isna().any().any():
        raise AssertionError("missing gradient-update alignment metrics on non-final steps")
    if (nonfinal["sigma_update"].fillna("") == "").any():
        raise AssertionError("missing update singular values on non-final steps")
    equal_steps = pd.read_csv(equal_output_dir / "step_metrics.csv")
    if equal_steps["run_id"].nunique() != expected_runs:
        raise AssertionError(f"equal-update run count mismatch: expected {expected_runs}, got {equal_steps['run_id'].nunique()}")
    equal_nonfinal = equal_steps[equal_steps["delta_loss"].notna()]
    if equal_nonfinal[["nrUpdate", "stUpdate", "nrUpdateFrac", "stUpdateFrac", "update_flatness"]].isna().any().any():
        raise AssertionError("missing equal-update spectral metrics on non-final steps")
    if equal_nonfinal[["update_grad_inner", "update_grad_cosine", "update_grad_per_update_norm"]].isna().any().any():
        raise AssertionError("missing equal-update gradient-update alignment metrics on non-final steps")
    pivot = equal_nonfinal.pivot_table(
        index=["problem_family", "setting", "kappa", "lr", "seed", "step"],
        columns="algo",
        values="relative_update_fro_norm",
        aggfunc="mean",
    ).dropna()
    max_update_gap = float((pivot["Adam"] - pivot["Muon"]).abs().max())
    if max_update_gap > 1e-10:
        raise AssertionError(f"equal-update control failed: max relative update gap {max_update_gap}")
    overlap_steps = pd.read_csv(Path("results/e11_overlap_followup") / "step_metrics.csv")
    if overlap_steps["run_id"].nunique() != 180:
        raise AssertionError(f"overlap follow-up run count mismatch: expected 180, got {overlap_steps['run_id'].nunique()}")
    overlap_nonfinal = overlap_steps[overlap_steps["delta_loss"].notna()]
    overlap_pivot = overlap_nonfinal.pivot_table(
        index=["problem_family", "setting", "kappa", "lr", "seed", "step"],
        columns="algo",
        values="relative_update_fro_norm",
        aggfunc="mean",
    ).dropna()
    overlap_gap = float((overlap_pivot["Adam"] - overlap_pivot["Muon"]).abs().max())
    if overlap_gap > 1e-10:
        raise AssertionError(f"overlap equal-update control failed: max relative update gap {overlap_gap}")
    width_steps = pd.read_csv(Path("results/e11_mlp_width_sweep") / "step_metrics.csv")
    if width_steps["run_id"].nunique() != 300:
        raise AssertionError(f"MLP width sweep run count mismatch: expected 300, got {width_steps['run_id'].nunique()}")
    width_nonfinal = width_steps[width_steps["delta_loss"].notna()]
    width_pivot = width_nonfinal.pivot_table(
        index=["setting", "lr", "seed", "step"],
        columns="algo",
        values="relative_update_fro_norm",
        aggfunc="mean",
    ).dropna()
    width_gap = float((width_pivot["Adam"] - width_pivot["Muon"]).abs().max())
    if width_gap > 1e-10:
        raise AssertionError(f"MLP width equal-update control failed: max relative update gap {width_gap}")
    hybrid_steps = pd.read_csv(Path("results/e11_mlp_layer_hybrid") / "step_metrics.csv")
    if hybrid_steps["run_id"].nunique() != 200:
        raise AssertionError(f"MLP hybrid run count mismatch: expected 200, got {hybrid_steps['run_id'].nunique()}")
    hybrid_nonfinal = hybrid_steps[hybrid_steps["delta_loss"].notna()]
    hybrid_pivot = hybrid_nonfinal.pivot_table(
        index=["setting", "lr", "seed", "step"],
        columns="algo",
        values="relative_update_fro_norm",
        aggfunc="mean",
    ).dropna()
    hybrid_gap = float(
        (hybrid_pivot.sub(hybrid_pivot["Adam"], axis=0)).abs().max().max()
    )
    if hybrid_gap > 1e-10:
        raise AssertionError(f"MLP hybrid equal-update control failed: max relative update gap {hybrid_gap}")
    hyper_raw_steps = pd.read_csv(Path("results/e11_hyperparam_sweep") / "raw_step_metrics.csv")
    hyper_equal_steps = pd.read_csv(Path("results/e11_hyperparam_sweep") / "equal_step_metrics.csv")
    if hyper_raw_steps["run_id"].nunique() != 360:
        raise AssertionError(
            f"hyperparameter raw run count mismatch: expected 360, got {hyper_raw_steps['run_id'].nunique()}"
        )
    if hyper_equal_steps["run_id"].nunique() != 360:
        raise AssertionError(
            f"hyperparameter equal-update run count mismatch: expected 360, got {hyper_equal_steps['run_id'].nunique()}"
        )
    hyper_equal_nonfinal = hyper_equal_steps[hyper_equal_steps["delta_loss"].notna()]
    hyper_equal_pivot = hyper_equal_nonfinal.pivot_table(
        index=["problem_family", "setting", "kappa", "lr", "seed", "step"],
        columns="algo",
        values="relative_update_fro_norm",
        aggfunc="mean",
    ).dropna()
    hyper_equal_gap = float((hyper_equal_pivot["Adam"] - hyper_equal_pivot["Muon"]).abs().max())
    if hyper_equal_gap > 1e-10:
        raise AssertionError(f"hyperparameter equal-update control failed: max relative update gap {hyper_equal_gap}")
    target_steps = pd.read_csv(Path("results/e11_target_update_sweep") / "step_metrics.csv")
    if target_steps["run_id"].nunique() != 360:
        raise AssertionError(f"target-update sweep run count mismatch: expected 360, got {target_steps['run_id'].nunique()}")
    target_nonfinal = target_steps[target_steps["delta_loss"].notna()].copy()
    target_gap = float(
        (target_nonfinal["relative_update_fro_norm"] - target_nonfinal["target_relative_update_norm"]).abs().max()
    )
    if target_gap > 1e-10:
        raise AssertionError(f"target-update norm control failed: max relative update gap {target_gap}")
    layer_control_steps = pd.read_csv(Path("results/e11_mlp_per_layer_control") / "step_metrics.csv")
    layer_control_layers = pd.read_csv(Path("results/e11_mlp_per_layer_control") / "layer_metrics.csv")
    if layer_control_steps["run_id"].nunique() != 100:
        raise AssertionError(
            f"MLP per-layer control run count mismatch: expected 100, got {layer_control_steps['run_id'].nunique()}"
        )
    layer_control_nonfinal = layer_control_layers[layer_control_layers["layer_relative_update_norm"].notna()].copy()
    layer_control_gap = float(
        (
            layer_control_nonfinal["layer_relative_update_norm"]
            - layer_control_nonfinal["target_layer_relative_update_norm"]
        )
        .abs()
        .max()
    )
    if layer_control_gap > 1e-10:
        raise AssertionError(f"MLP per-layer update control failed: max layer relative update gap {layer_control_gap}")
    mnist_steps = pd.read_csv(Path("results/e11_mnist_mlp_probe") / "step_metrics.csv")
    if mnist_steps["run_id"].nunique() != 36:
        raise AssertionError(f"MNIST MLP probe run count mismatch: expected 36, got {mnist_steps['run_id'].nunique()}")
    mnist_spectrum = pd.read_csv(Path("results/e11_mnist_mlp_probe") / "update_spectrum_summary.csv")
    mnist_nr = mnist_spectrum[
        (mnist_spectrum["problem_family"] == "MNISTMLP") & (mnist_spectrum["metric"] == "nrUpdate")
    ]
    if len(mnist_nr) != 1 or not bool(mnist_nr.iloc[0]["ratio_ci95_above_one"]):
        raise AssertionError("MNIST MLP probe must preserve the update-spectrum shaping signal")
    deep_mnist_steps = pd.read_csv(Path("results/e11_deep_mnist_mlp_probe") / "step_metrics.csv")
    if deep_mnist_steps["run_id"].nunique() != 72:
        raise AssertionError(
            f"Deep MNIST MLP probe run count mismatch: expected 72, got {deep_mnist_steps['run_id'].nunique()}"
        )
    deep_mnist_spectrum = pd.read_csv(Path("results/e11_deep_mnist_mlp_probe") / "update_spectrum_summary.csv")
    deep_mnist_nr = deep_mnist_spectrum[
        (deep_mnist_spectrum["problem_family"] == "DeepMNISTMLP") & (deep_mnist_spectrum["metric"] == "nrUpdate")
    ]
    deep_mnist_st = deep_mnist_spectrum[
        (deep_mnist_spectrum["problem_family"] == "DeepMNISTMLP") & (deep_mnist_spectrum["metric"] == "stUpdate")
    ]
    if len(deep_mnist_nr) != 1 or len(deep_mnist_st) != 1:
        raise AssertionError("Deep MNIST MLP probe must include nrUpdate and stUpdate spectrum summaries")
    if not bool(deep_mnist_nr.iloc[0]["ratio_ci95_above_one"]) or not bool(deep_mnist_st.iloc[0]["ratio_ci95_above_one"]):
        raise AssertionError("Deep MNIST MLP probe must preserve update-spectrum shaping")
    deep_mnist_pair = pd.read_csv(Path("results/e11_deep_mnist_mlp_probe") / "pair_summary.csv")
    if set(deep_mnist_pair["num_factors"].dropna().astype(str)) != {"3", "4", "All"}:
        raise AssertionError("Deep MNIST MLP probe must include both 3- and 4-factor depths")
    patch_steps = pd.read_csv(Path("results/e11_mnist_patch_probe") / "step_metrics.csv")
    if patch_steps["run_id"].nunique() != 48:
        raise AssertionError(f"MNIST patch probe run count mismatch: expected 48, got {patch_steps['run_id'].nunique()}")
    patch_spectrum = pd.read_csv(Path("results/e11_mnist_patch_probe") / "update_spectrum_summary.csv")
    patch_nr = patch_spectrum[
        (patch_spectrum["problem_family"] == "MNISTPatchClassifier") & (patch_spectrum["metric"] == "nrUpdate")
    ]
    patch_st = patch_spectrum[
        (patch_spectrum["problem_family"] == "MNISTPatchClassifier") & (patch_spectrum["metric"] == "stUpdate")
    ]
    if len(patch_nr) != 1 or len(patch_st) != 1:
        raise AssertionError("MNIST patch probe must include nrUpdate and stUpdate spectrum summaries")
    if not bool(patch_nr.iloc[0]["ratio_ci95_above_one"]) or not bool(patch_st.iloc[0]["ratio_ci95_above_one"]):
        raise AssertionError("MNIST patch probe must preserve update-spectrum shaping")
    patch_pair = pd.read_csv(Path("results/e11_mnist_patch_probe") / "pair_summary.csv")
    if set(patch_pair["kernel_size"].dropna().astype(str)) != {"5", "7", "All"}:
        raise AssertionError("MNIST patch probe must include both 5x5 and 7x7 patch settings")
    conv_steps = pd.read_csv(Path("results/e11_mnist_conv_probe") / "step_metrics.csv")
    if conv_steps["run_id"].nunique() != 48:
        raise AssertionError(f"MNIST ConvNet probe run count mismatch: expected 48, got {conv_steps['run_id'].nunique()}")
    conv_spectrum = pd.read_csv(Path("results/e11_mnist_conv_probe") / "update_spectrum_summary.csv")
    conv_nr = conv_spectrum[
        (conv_spectrum["problem_family"] == "MNISTConvNet") & (conv_spectrum["metric"] == "nrUpdate")
    ]
    conv_st = conv_spectrum[
        (conv_spectrum["problem_family"] == "MNISTConvNet") & (conv_spectrum["metric"] == "stUpdate")
    ]
    if len(conv_nr) != 1 or len(conv_st) != 1:
        raise AssertionError("MNIST ConvNet probe must include nrUpdate and stUpdate spectrum summaries")
    if not bool(conv_nr.iloc[0]["ratio_ci95_above_one"]) or not bool(conv_st.iloc[0]["ratio_ci95_above_one"]):
        raise AssertionError("MNIST ConvNet probe must preserve update-spectrum shaping")
    conv_pair = pd.read_csv(Path("results/e11_mnist_conv_probe") / "pair_summary.csv")
    if set(conv_pair["kernel_size"].dropna().astype(str)) != {"5", "7", "All"}:
        raise AssertionError("MNIST ConvNet probe must include both 5x5 and 7x7 kernel settings")
    spectral_probe = pd.read_csv(Path("results/e11_spectral_allocation_probe") / "probe_rows.csv")
    if len(spectral_probe) != 720:
        raise AssertionError(f"spectral allocation probe row count mismatch: expected 720, got {len(spectral_probe)}")
    spectral_summary = pd.read_csv(Path("results/e11_spectral_allocation_probe") / "spectral_allocation_summary.csv")
    required_budgets = {"fro", "op"}
    if set(spectral_summary["budget"]) != required_budgets:
        raise AssertionError(f"spectral allocation budgets mismatch: expected {required_budgets}")
    subspace_rows = pd.read_csv(Path("results/e11_singular_vector_trajectory") / "gradient_subspace_rows.csv")
    update_subspace_rows = pd.read_csv(Path("results/e11_singular_vector_trajectory") / "update_subspace_rows.csv")
    if len(subspace_rows) != 800:
        raise AssertionError(f"singular-vector gradient row count mismatch: expected 800, got {len(subspace_rows)}")
    if len(update_subspace_rows) != 725:
        raise AssertionError(
            f"singular-vector update row count mismatch: expected 725, got {len(update_subspace_rows)}"
        )
    swap_probe = pd.read_csv(Path("results/e11_singular_vector_swap_probe") / "swap_probe_rows.csv")
    if len(swap_probe) != 320:
        raise AssertionError(f"singular-vector swap probe row count mismatch: expected 320, got {len(swap_probe)}")
    natural_swap = pd.read_csv(Path("results/e11_natural_update_swap_probe") / "natural_update_swap_rows.csv")
    if len(natural_swap) != 4800:
        raise AssertionError(f"natural update-vector swap probe row count mismatch: expected 4800, got {len(natural_swap)}")
    expected_targets = {1e-4, 3e-4, 1e-3, 3e-3, 1e-2}
    observed_targets = {round(float(value), 12) for value in natural_swap["target_layer_relative_norm"].unique()}
    if observed_targets != expected_targets:
        raise AssertionError(f"natural update-vector target sweep mismatch: expected {expected_targets}, got {observed_targets}")
    natural_swap_summary = pd.read_csv(Path("results/e11_natural_update_swap_probe") / "natural_update_swap_summary.csv")
    if set(natural_swap_summary["budget"]) != {"fro", "op", "All"}:
        raise AssertionError("natural update-vector swap summary must include fro, op, and All budget groups")
    boundary = pd.read_csv(Path("results/e11_mechanism_boundary") / "mechanism_boundary_map.csv")
    expected_boundary_axes = {
        "task_family",
        "target_update_size",
        "per_layer_update_size",
        "norm_budget",
        "state_specific_direction",
    }
    if set(boundary["boundary_axis"]) != expected_boundary_axes:
        raise AssertionError(f"mechanism boundary axes mismatch: expected {expected_boundary_axes}")
    directions = set(boundary["direction"])
    if "Muon-favorable" not in directions or "Adam/GD-favorable" not in directions or "mixed_or_uncertain" not in directions:
        raise AssertionError("mechanism boundary map must include Muon-favorable, Adam/GD-favorable, and mixed cases")
    if "flat/polar-favorable" not in directions or "GD-spectrum-favorable" not in directions:
        raise AssertionError("mechanism boundary map must include both operator-budget and Frobenius-budget spectral-allocation cases")
    boundary_predictor = pd.read_csv(Path("results/e11_boundary_predictor") / "boundary_predictor_summary.csv")
    predictor_required = {"family_only", "state_only", "update_spectrum_only", "state_plus_update_spectrum"}
    predictor_feature_sets = set(boundary_predictor["feature_set"])
    if not predictor_required.issubset(predictor_feature_sets):
        raise AssertionError(f"boundary predictor feature sets missing: expected {predictor_required}")
    if "leave_setting_out" not in set(boundary_predictor["evaluation"]):
        raise AssertionError("boundary predictor must include leave-setting-out evaluation")
    predictor_uncertainty = pd.read_csv(Path("results/e11_boundary_predictor") / "boundary_predictor_uncertainty.csv")
    required_uncertainty_columns = {
        "mean_balanced_accuracy_chance_filled",
        "balanced_accuracy_ci95_low",
        "balanced_accuracy_ci95_high",
        "balanced_accuracy_ci95_above_chance",
        "mean_brier_improvement_over_base_rate",
    }
    if not required_uncertainty_columns.issubset(predictor_uncertainty.columns):
        raise AssertionError(f"boundary predictor uncertainty missing columns: {required_uncertainty_columns}")
    predictor_focus = predictor_uncertainty[
        predictor_uncertainty["target"].eq("update_grad_inner_muon_higher")
    ].copy()
    if predictor_focus.empty:
        raise AssertionError("boundary predictor uncertainty must include update_grad_inner_muon_higher")
    best_uncertain = predictor_focus.sort_values("mean_balanced_accuracy_chance_filled", ascending=False).iloc[0]
    if bool(best_uncertain["balanced_accuracy_ci95_above_chance"]):
        raise AssertionError("boundary predictor uncertainty should not support an above-chance predictive claim yet")
    boundary_predictor_audit = Path("discussion/e11_boundary_predictor_audit.md").read_text(encoding="utf-8")
    required_predictor_audit_phrases = [
        "In-Sample Versus Leave-Setting-Out Gap",
        "Leave-Setting-Out Uncertainty",
        "Held-Out Setting Difficulty",
        "Minimum Standard For The Next Predictor",
        "Chance-filled",
        "not yet predictive out of sample",
    ]
    missing_predictor_audit = [phrase for phrase in required_predictor_audit_phrases if phrase not in boundary_predictor_audit]
    if missing_predictor_audit:
        raise AssertionError(f"boundary predictor audit missing required content: {missing_predictor_audit}")
    paper_skeleton = Path("discussion/e11_paper_skeleton.md").read_text(encoding="utf-8")
    if "## Core Claims" not in paper_skeleton or "## Main Figure/Table Plan" not in paper_skeleton:
        raise AssertionError("paper skeleton must include core claims and figure/table plan sections")
    main_package = Path("discussion/e11_main_paper_package.md").read_text(encoding="utf-8")
    required_main_package_phrases = [
        "Main Figure/Table Package",
        "Appendix Allocation",
        "Claims To Exclude From Main Text",
        "figures/e11_equal_update/update_spectrum_robustness.png",
        "4 Muon/flat favorable",
        "1 own-update positive-control",
        "8 unfavorable",
        "1 mixed/uncertain",
    ]
    missing_main_package = [phrase for phrase in required_main_package_phrases if phrase not in main_package]
    if missing_main_package:
        raise AssertionError(f"main paper package missing required content: {missing_main_package}")
    main_captions = Path("discussion/e11_main_figure_captions.md").read_text(encoding="utf-8")
    required_caption_phrases = [
        "E11 Main Figure Captions",
        "Figure 1",
        "Figure 2",
        "Figure 3",
        "Caption Discipline",
        "not yet predictive out of sample",
        "own-update positive-control",
    ]
    missing_captions = [phrase for phrase in required_caption_phrases if phrase not in main_captions]
    if missing_captions:
        raise AssertionError(f"main figure captions missing required content: {missing_captions}")
    notation_glossary = Path("discussion/e11_notation_glossary.md").read_text(encoding="utf-8")
    required_notation_phrases = [
        "E11 Notation Glossary",
        r"\(G_i = \nabla_{W_i} L\)",
        r"\(\Delta W_i\)",
        r"\(D_i=-\Delta W_i=W_i-W_i^+\)",
        r"\(\langle G_i, D_i\rangle\)",
        "`update_grad_inner`",
        "strict activation-product definition",
        "same-batch pre/post-update",
        "`train_batch_size < num_samples`",
    ]
    missing_notation = [phrase for phrase in required_notation_phrases if phrase not in notation_glossary]
    if missing_notation:
        raise AssertionError(f"notation glossary missing required content: {missing_notation}")
    claim_ledger = Path("discussion/e11_quantitative_claim_ledger.md").read_text(encoding="utf-8")
    required_claim_ledger_phrases = [
        "Claim Ledger",
        "Writing Priority",
        "C1 -> C2 -> C3 -> C4",
        "not yet predictive out of sample",
        "4 Muon/flat favorable rows",
        "1 own-update positive-control row",
    ]
    missing_claim_ledger = [phrase for phrase in required_claim_ledger_phrases if phrase not in claim_ledger]
    if missing_claim_ledger:
        raise AssertionError(f"quantitative claim ledger missing required content: {missing_claim_ledger}")
    paper_numbers = Path("discussion/e11_paper_numbers.tex").read_text(encoding="utf-8")
    required_number_macros = [
        "\\EelevenNrUpdateRatio",
        "\\EelevenFirstOrderSpearman",
        "\\EelevenFroFlatOverGdRatio",
        "\\EelevenOpFlatOverGdRatio",
        "\\EelevenBoundaryMuonFlatFavorableRows",
        "\\EelevenBoundaryOwnUpdatePositiveControlRows",
        "\\EelevenPatchMnistNrUpdateRatio",
        "\\EelevenPatchMnistFirstOrderRatio",
        "\\EelevenConvMnistNrUpdateRatio",
        "\\EelevenConvMnistFirstOrderRatio",
        "\\EelevenBoundaryPredictorBestBalancedAccuracy",
    ]
    missing_number_macros = [macro for macro in required_number_macros if macro not in paper_numbers]
    if missing_number_macros:
        raise AssertionError(f"paper number macros missing required content: {missing_number_macros}")
    reproduction_checklist = Path("discussion/e11_reproduction_checklist.md").read_text(encoding="utf-8")
    required_reproduction_phrases = [
        "make e11-main-results",
        "make e11-appendix-results",
        "make e11-all-results",
        "make e11-paper-assets",
        "Minimal Main-Paper Evidence",
        "Appendix / Guardrail Evidence",
        "Generated Paper-Facing Assets",
        "make e11-full",
        "Batch / Activation Contract",
        "`diagnostic_A_definition == full_layer_input_activation`",
        "same-batch pre/post-update",
    ]
    missing_reproduction = [phrase for phrase in required_reproduction_phrases if phrase not in reproduction_checklist]
    if missing_reproduction:
        raise AssertionError(f"reproduction checklist missing required content: {missing_reproduction}")
    reviewer_risk = Path("discussion/e11_reviewer_risk_audit.md").read_text(encoding="utf-8")
    required_reviewer_risk_phrases = [
        "Risk Table",
        "Claim Decisions",
        "Do not state or imply that higher update rank generally improves progress",
    ]
    missing_reviewer_risk = [phrase for phrase in required_reviewer_risk_phrases if phrase not in reviewer_risk]
    if missing_reviewer_risk:
        raise AssertionError(f"reviewer risk audit missing required content: {missing_reviewer_risk}")
    theory_note = Path("discussion/e11_theory_note.md").read_text(encoding="utf-8")
    required_theory_phrases = [
        "Under the Frobenius ball",
        "Under the operator-norm ball",
        "von Neumann's trace inequality",
        "operator-norm-constrained spectral allocation",
        "positive descent update",
        "\\(W^+=W-D\\)",
        "optimal first-order decrease is \\(r\\|G\\|_F\\)",
        "optimal first-order decrease is \\(\\eta\\|G\\|_*\\)",
    ]
    missing_theory = [phrase for phrase in required_theory_phrases if phrase not in theory_note]
    if missing_theory:
        raise AssertionError(f"theory note missing required content: {missing_theory}")
    theorem_bridge = Path("discussion/e11_mechanism_theorem_bridge.md").read_text(encoding="utf-8")
    required_bridge_phrases = [
        "Theorem-To-Evidence Map",
        "Safe Claim Language",
        "Language To Avoid",
        "does_not_support",
        "`D_op^* = eta U V^T`",
        "local solution to an operator-norm-constrained linearized problem",
    ]
    missing_bridge = [phrase for phrase in required_bridge_phrases if phrase not in theorem_bridge]
    if missing_bridge:
        raise AssertionError(f"mechanism theorem bridge missing required content: {missing_bridge}")
    ablation_map = Path("discussion/e11_optimizer_ablation_map.md").read_text(encoding="utf-8")
    required_ablation_phrases = [
        "Matched global update size",
        "Matched per-layer update size",
        "Synthetic singular-value allocation",
        "Natural update-vector swap",
        "Still Missing For A Stronger Variant Claim",
    ]
    missing_ablation = [phrase for phrase in required_ablation_phrases if phrase not in ablation_map]
    if missing_ablation:
        raise AssertionError(f"optimizer ablation map missing required content: {missing_ablation}")
    evidence_index = Path("discussion/e11_evidence_index.md").read_text(encoding="utf-8")
    if "e11_optimizer_ablation_map.md" not in evidence_index:
        raise AssertionError("evidence index must link to optimizer ablation map")
    research_synthesis = Path("discussion/e11_research_synthesis.md").read_text(encoding="utf-8")
    if "E11 optimizer ablation map" not in research_synthesis:
        raise AssertionError("research synthesis must reference optimizer ablation map")
    paper_readiness = Path("discussion/e11_paper_readiness_audit.md").read_text(encoding="utf-8")
    required_readiness_phrases = [
        "Next Experiment Checklist",
        "CNN or matrix-only convolutional surrogate",
        "Held-out boundary prediction benchmark",
        "Approximate Muon / polar-iteration ablation",
        "Retuned longer-horizon training grid",
        "Polished local theorem/proof",
        "own-update positive-control row",
        "mixed/uncertain row",
    ]
    missing_readiness = [phrase for phrase in required_readiness_phrases if phrase not in paper_readiness]
    if missing_readiness:
        raise AssertionError(f"paper-readiness audit missing required next-step content: {missing_readiness}")
    readme = Path("README_E11.md").read_text(encoding="utf-8")
    if "## Main Entry Points" not in readme or "## Current Publication Gaps" not in readme:
        raise AssertionError("README_E11.md must document entry points and publication gaps")
    if "make e11-check" not in readme or "make e11-full" not in readme or "make e11-main-results" not in readme:
        raise AssertionError("README_E11.md must document E11 make targets")
    artifact_manifest = Path("discussion/e11_artifact_manifest.md").read_text(encoding="utf-8")
    for phrase in ["Artifact Directories", "Ignored Local Artifacts", "results/e11_artifact_manifest.json"]:
        if phrase not in artifact_manifest:
            raise AssertionError(f"artifact manifest missing required section or reference: {phrase}")
    manifest_json = json.loads(Path("results/e11_artifact_manifest.json").read_text(encoding="utf-8"))
    assert_valid_artifact_manifest(manifest_json)
    assert_no_unguarded_overclaims(
        [
            Path("README_E11.md"),
            Path("discussion/e11_evidence_index.md"),
        Path("discussion/e11_research_synthesis.md"),
        Path("discussion/e11_research_direction_map.md"),
            Path("discussion/e11_paper_readiness_audit.md"),
            Path("discussion/e11_paper_skeleton.md"),
            Path("discussion/e11_main_paper_package.md"),
            Path("discussion/e11_main_figure_captions.md"),
            Path("discussion/e11_notation_glossary.md"),
            Path("discussion/e11_quantitative_claim_ledger.md"),
            Path("discussion/e11_reviewer_risk_audit.md"),
            Path("discussion/e11_theory_note.md"),
            Path("discussion/e11_mechanism_theorem_bridge.md"),
            Path("discussion/e11_optimizer_ablation_map.md"),
            Path("discussion/e11_optimizer_invariance_audit.md"),
            Path("discussion/e11_claim_validity_audit.md"),
        ]
    )
    optimizer_switch = pd.read_csv(Path("results/e11_optimizer_switch_probe") / "optimizer_switch_rows.csv")
    if len(optimizer_switch) != 360:
        raise AssertionError(f"optimizer switch probe row count mismatch: expected 360, got {len(optimizer_switch)}")
    optimizer_switch_reset = pd.read_csv(Path("results/e11_optimizer_switch_reset_control") / "optimizer_switch_reset_rows.csv")
    if len(optimizer_switch_reset) != 540:
        raise AssertionError(
            f"optimizer switch reset-control row count mismatch: expected 540, got {len(optimizer_switch_reset)}"
        )
    optimizer_switch_horizon = pd.read_csv(Path("results/e11_optimizer_switch_horizon_sweep") / "optimizer_switch_horizon_rows.csv")
    if len(optimizer_switch_horizon) != 480:
        raise AssertionError(
            f"optimizer switch horizon sweep row count mismatch: expected 480, got {len(optimizer_switch_horizon)}"
        )
    if not bool(optimizer_switch_horizon["finite"].all()):
        raise AssertionError("optimizer switch horizon sweep has non-finite continuation results")
    optimizer_switch_lr = pd.read_csv(Path("results/e11_optimizer_switch_lr_sweep") / "optimizer_switch_lr_rows.csv")
    optimizer_switch_lr_best = pd.read_csv(Path("results/e11_optimizer_switch_lr_sweep") / "optimizer_switch_lr_best.csv")
    if len(optimizer_switch_lr) != 360:
        raise AssertionError(f"optimizer switch LR sweep row count mismatch: expected 360, got {len(optimizer_switch_lr)}")
    if len(optimizer_switch_lr_best) != 120:
        raise AssertionError(
            f"optimizer switch LR sweep best-row count mismatch: expected 120, got {len(optimizer_switch_lr_best)}"
        )
    if not bool(optimizer_switch_lr["finite"].all()):
        raise AssertionError("optimizer switch LR sweep has non-finite continuation results")
    stateless_rows = pd.read_csv(Path("results/e11_stateless_direction_ablation") / "stateless_direction_rows.csv")
    stateless_pairs = pd.read_csv(Path("results/e11_stateless_direction_ablation") / "stateless_direction_pairs.csv")
    stateless_summary = pd.read_csv(Path("results/e11_stateless_direction_ablation") / "stateless_direction_summary.csv")
    if len(stateless_rows) != 1890:
        raise AssertionError(f"stateless direction ablation row count mismatch: expected 1890, got {len(stateless_rows)}")
    if len(stateless_pairs) != 1890:
        raise AssertionError(f"stateless direction ablation pair count mismatch: expected 1890, got {len(stateless_pairs)}")
    required_candidates = {"GD", "FreshAdamSign", "PolarMuon"}
    if set(stateless_rows["candidate"]) != required_candidates:
        raise AssertionError(f"stateless direction candidates mismatch: expected {required_candidates}")
    required_comparisons = {"PolarMuon/GD", "FreshAdamSign/GD", "PolarMuon/FreshAdamSign"}
    if not required_comparisons.issubset(set(stateless_summary["comparison"])):
        raise AssertionError("stateless direction summary missing required comparisons")
    trajectory_steps = pd.read_csv(Path("results/e11_stateless_optimizer_trajectory") / "stateless_optimizer_step_metrics.csv")
    trajectory_outcomes = pd.read_csv(Path("results/e11_stateless_optimizer_trajectory") / "stateless_optimizer_outcomes.csv")
    trajectory_pairs = pd.read_csv(Path("results/e11_stateless_optimizer_trajectory") / "stateless_optimizer_pairs.csv")
    trajectory_summary = pd.read_csv(Path("results/e11_stateless_optimizer_trajectory") / "stateless_optimizer_summary.csv")
    if len(trajectory_steps) != 840:
        raise AssertionError(f"stateless optimizer trajectory row count mismatch: expected 840, got {len(trajectory_steps)}")
    if len(trajectory_outcomes) != 90 or len(trajectory_pairs) != 90:
        raise AssertionError("stateless optimizer trajectory outcome/pair row count mismatch")
    if set(trajectory_steps["algo"]) != required_candidates:
        raise AssertionError(f"stateless optimizer trajectory candidates mismatch: expected {required_candidates}")
    if not required_comparisons.issubset(set(trajectory_summary["comparison"])):
        raise AssertionError("stateless optimizer trajectory summary missing required comparisons")
    print("E11 outputs validated")
    print(f"runs={steps['run_id'].nunique()}, step_rows={len(steps)}, families={sorted(families)}")
    print(f"equal_update_runs={equal_steps['run_id'].nunique()}, max_relative_update_gap={max_update_gap:.3g}")
    print(f"overlap_followup_runs={overlap_steps['run_id'].nunique()}, max_relative_update_gap={overlap_gap:.3g}")
    print(f"mlp_width_runs={width_steps['run_id'].nunique()}, max_relative_update_gap={width_gap:.3g}")
    print(f"mlp_hybrid_runs={hybrid_steps['run_id'].nunique()}, max_relative_update_gap={hybrid_gap:.3g}")
    print(
        f"hyperparam_raw_runs={hyper_raw_steps['run_id'].nunique()}, "
        f"hyperparam_equal_runs={hyper_equal_steps['run_id'].nunique()}, "
        f"max_relative_update_gap={hyper_equal_gap:.3g}"
    )
    print(f"target_update_runs={target_steps['run_id'].nunique()}, max_target_gap={target_gap:.3g}")
    print(
        f"mlp_per_layer_control_runs={layer_control_steps['run_id'].nunique()}, "
        f"max_layer_target_gap={layer_control_gap:.3g}"
    )
    print(f"spectral_allocation_probe_rows={len(spectral_probe)}")
    print(f"singular_vector_gradient_rows={len(subspace_rows)}, update_rows={len(update_subspace_rows)}")
    print(f"singular_vector_swap_rows={len(swap_probe)}")
    print(f"natural_update_swap_rows={len(natural_swap)}")
    print(f"optimizer_switch_rows={len(optimizer_switch)}")
    print(f"optimizer_switch_reset_rows={len(optimizer_switch_reset)}")
    print(f"optimizer_switch_horizon_rows={len(optimizer_switch_horizon)}")
    print(f"optimizer_switch_lr_rows={len(optimizer_switch_lr)}")
    print(f"stateless_direction_rows={len(stateless_rows)}")
    print(f"stateless_optimizer_trajectory_rows={len(trajectory_steps)}")


if __name__ == "__main__":
    main()
