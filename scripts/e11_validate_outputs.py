from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry import default_config
from e11_condition_geometry.artifacts import (
    APPENDIX_RUNNER_SCRIPTS,
    ARTIFACT_DIRS,
    IGNORE_POLICY,
    KEY_DOCUMENTS,
    KEY_TABLES,
    MAIN_RESULT_SCRIPTS,
)
import pandas as pd


def assert_required_phrases(label: str, text: str, phrases: list[str]) -> None:
    """Validate that a generated paper-facing artifact keeps required anchors."""

    missing = [phrase for phrase in phrases if phrase not in text]
    if missing:
        raise AssertionError(f"{label} missing required content: {missing}")


def assert_forbidden_phrases_absent(label: str, text: str, phrases: list[str]) -> None:
    """Validate that an artifact does not contain known unsafe/stale wording."""

    present = [phrase for phrase in phrases if phrase in text]
    if present:
        raise AssertionError(f"{label} contains forbidden content: {present}")


def assert_includegraphics_files_exist(tex_path: Path) -> None:
    """Validate that local LaTeX figure references resolve from the tex file."""

    text = tex_path.read_text(encoding="utf-8")
    figure_paths = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text)
    missing: list[str] = []
    for figure_path in figure_paths:
        if "://" in figure_path:
            continue
        resolved = (tex_path.parent / figure_path).resolve()
        if not resolved.exists():
            missing.append(f"{figure_path} -> {resolved}")
    if missing:
        raise FileNotFoundError(f"{tex_path} has missing includegraphics files: {missing}")


def assert_latex_log_has_no_serious_warnings(log_path: Path) -> None:
    """Fail on LaTeX issues that affect references, figures, or visible layout."""

    if not log_path.exists():
        return
    serious_patterns = [
        r"^! .*Error",
        r"Emergency stop",
        r"Fatal error",
        r"Underfull \\hbox",
        r"Overfull \\hbox",
        r"LaTeX Warning: Citation `.*' .* undefined",
        r"LaTeX Warning: Reference `.*' .* undefined",
        r"LaTeX Warning: There were undefined references",
        r"LaTeX Warning: Label\\(s\\) may have changed",
        r"Package rerunfilecheck Warning: .*Rerun",
    ]
    violations: list[str] = []
    for lineno, line in enumerate(log_path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        if any(re.search(pattern, line) for pattern in serious_patterns):
            violations.append(f"{log_path}:{lineno}: {line}")
    if violations:
        raise AssertionError("serious LaTeX log issues found: " + "; ".join(violations))


def assert_latex_log_has_no_box_warnings(log_path: Path) -> None:
    """Fail on any overfull or underfull box warning in compact deliverables."""

    if not log_path.exists():
        return
    box_patterns = [
        r"Underfull \\hbox",
        r"Overfull \\hbox",
        r"Underfull \\vbox",
        r"Overfull \\vbox",
    ]
    violations: list[str] = []
    for lineno, line in enumerate(log_path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        if any(re.search(pattern, line) for pattern in box_patterns):
            violations.append(f"{log_path}:{lineno}: {line}")
    if violations:
        raise AssertionError("LaTeX box warnings found: " + "; ".join(violations))


def assert_pdf_artifact_is_valid(pdf_path: Path, min_size_bytes: int = 1000) -> None:
    """Validate that a rendered PDF artifact exists and is not a placeholder."""

    if not pdf_path.exists():
        raise FileNotFoundError(f"missing PDF artifact: {pdf_path}")
    size = pdf_path.stat().st_size
    if size < min_size_bytes:
        raise AssertionError(f"PDF artifact is unexpectedly small: {pdf_path} has {size} bytes")
    with pdf_path.open("rb") as handle:
        header = handle.read(4)
    if header != b"%PDF":
        raise AssertionError(f"PDF artifact does not start with %PDF header: {pdf_path}")


def assert_pdf_text_has_no_known_artifacts(pdf_path: Path) -> None:
    """Use Ghostscript text extraction, when available, to catch copy-text artifacts."""

    gs_path = shutil.which("gs")
    if gs_path is None:
        return
    with tempfile.NamedTemporaryFile(suffix=".txt") as handle:
        result = subprocess.run(
            [
                gs_path,
                "-q",
                "-dNOPAUSE",
                "-dBATCH",
                "-sDEVICE=txtwrite",
                f"-sOutputFile={handle.name}",
                str(pdf_path),
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode != 0:
            raise AssertionError(
                f"Ghostscript text extraction failed for {pdf_path}: {result.stderr.strip()}"
            )
        text = Path(handle.name).read_text(encoding="utf-8", errors="replace")
    bad_fragments = [
        "¿",
        "�",
        "￾",
        "exceedsssrank",
        "Anonymous authors",
        "Paper under double-blind review",
        "Under review as a conference paper",
        "undefined citations",
        "undefined references",
    ]
    present = [fragment for fragment in bad_fragments if fragment in text]
    if present:
        raise AssertionError(f"{pdf_path} text layer contains known artifacts: {present}")
    if re.search(r"(?m)^\s*0{2,3}\s+\S", text):
        raise AssertionError(f"{pdf_path} text layer appears to contain ICLR review line numbers")
    compact_numeric_ci = re.search(r"\[-?\d[\d.eE+-]*,-?\d", text)
    if compact_numeric_ci:
        raise AssertionError(
            f"{pdf_path} text layer contains a numeric CI without comma spacing: "
            f"{compact_numeric_ci.group(0)}"
        )
    table_three_match = re.search(
        r"Table 3: Tail outcomes, margin-relevant drift, and head-gain readouts(?P<table>.*?)(?:Appendix E|Table 4:)",
        text,
        flags=re.S,
    )
    table_three_text = table_three_match.group("table") if table_three_match else ""
    compact_estimate_ci = re.search(r"[-−]?\d[\d.eE+\-−]*\[-?\d", table_three_text)
    if compact_estimate_ci:
        raise AssertionError(
            f"{pdf_path} Table 3 text layer contains an estimate directly attached to a numeric CI: "
            f"{compact_estimate_ci.group(0)}"
        )


def assert_bibtex_citations_are_defined(tex_path: Path, bibliography_path: Path) -> None:
    """Keep paper citations and the ICLR BibTeX bibliography in one-to-one sync."""

    tex_text = tex_path.read_text(encoding="utf-8")
    bibliography_text = bibliography_path.read_text(encoding="utf-8")
    cited_keys: set[str] = set()
    for group in re.findall(r"\\cite[a-zA-Z]*(?:\[[^\]]*\])*\{([^}]+)\}", tex_text):
        cited_keys.update(key.strip() for key in group.split(",") if key.strip())
    bibliography_keys = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)", bibliography_text))
    missing_keys = sorted(cited_keys - bibliography_keys)
    if missing_keys:
        raise AssertionError(f"paper citations are missing from references.bib: missing={missing_keys}")
    unused_keys = sorted(bibliography_keys - cited_keys)
    if unused_keys:
        raise AssertionError(f"references.bib has entries not cited by main.tex: unused={unused_keys}")


def bibtex_entry_body(bibliography_text: str, key: str) -> tuple[str, str]:
    entry = re.search(
        rf"@(?P<kind>\w+)\{{{re.escape(key)},(?P<body>.*?)\n\}}",
        bibliography_text,
        flags=re.DOTALL,
    )
    if entry is None:
        raise AssertionError(f"paper bibliography missing {key} entry")
    return entry.group("kind"), entry.group("body")


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
    required_manifest_documents = set(KEY_DOCUMENTS)
    manifest_documents = {item.get("path") for item in manifest_json.get("key_documents", [])}
    if not required_manifest_documents.issubset(manifest_documents):
        raise AssertionError(
            f"artifact manifest missing key documents: {required_manifest_documents - manifest_documents}"
        )
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
    if (frame["train_batch_size"] > frame["num_samples"]).any():
        raise AssertionError(f"{path} train_batch_size must not exceed num_samples")
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
    if path in {Path("results/e11/step_metrics.csv"), Path("results/e11_equal_update/step_metrics.csv")}:
        if "noise_std" not in frame.columns:
            raise AssertionError(f"{path} default core rows must record noise_std")
        if (frame["train_batch_size"] >= frame["num_samples"]).any():
            raise AssertionError(f"{path} default core rows must use strict noisy mini-batches")
        if (frame["noise_std"] <= 0.0).any():
            raise AssertionError(f"{path} default core rows must use positive observation/input noise")


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
        Path("results/e11_head_tail_interference") / "step_metrics.csv",
        Path("results/e11_head_tail_interference") / "pair_summary.csv",
        Path("figures/e11_head_tail_interference") / "head_tail_drift_ratio.png",
        Path("discussion/e11_head_tail_interference.md"),
        Path("results/e11_head_tail_alignment_ablation") / "step_metrics.csv",
        Path("results/e11_head_tail_alignment_ablation") / "summary.csv",
        Path("figures/e11_head_tail_alignment_ablation") / "head_tail_alignment_ablation.png",
        Path("discussion/e11_head_tail_alignment_ablation.md"),
        Path("results/e11_long_tail_one_step") / "step_metrics.csv",
        Path("results/e11_long_tail_one_step") / "pair_summary.csv",
        Path("results/e11_long_tail_one_step") / "layer_metrics.csv",
        Path("figures/e11_long_tail_one_step") / "long_tail_one_step_tail_response.png",
        Path("discussion/e11_long_tail_one_step.md"),
        Path("results/e11_cifar100_resnet_fc_condition_scatter") / "step_metrics.csv",
        Path("results/e11_cifar100_resnet_fc_condition_scatter") / "summary.csv",
        Path("results/e11_cifar100_resnet_fc_condition_scatter") / "condition_metrics.csv",
        Path("results/e11_cifar100_resnet_fc_condition_scatter") / "config.json",
        Path("figures/e11_cifar100_resnet_fc_condition_scatter") / "cifar100_resnet_fc_condition_scatter.png",
        Path("discussion/e11_cifar100_resnet_fc_condition_scatter.md"),
        Path("results/e11_cifar100_resnet_tail_quality_control") / "step_metrics.csv",
        Path("results/e11_cifar100_resnet_tail_quality_control") / "pair_summary.csv",
        Path("results/e11_cifar100_resnet_tail_quality_control") / "layer_metrics.csv",
        Path("results/e11_cifar100_resnet_tail_quality_control") / "config.json",
        Path("figures/e11_cifar100_resnet_tail_quality_control") / "cifar100_resnet_checkpoint_sweep.png",
        Path("discussion/e11_cifar100_resnet_tail_quality_control.md"),
        Path("results/e11_cifar100_resnet_layer_jvp_tail_quality") / "metrics.csv",
        Path("results/e11_cifar100_resnet_layer_jvp_tail_quality") / "paired_metrics.csv",
        Path("results/e11_cifar100_resnet_layer_jvp_tail_quality") / "summary.csv",
        Path("results/e11_cifar100_resnet_layer_jvp_tail_quality") / "overall_summary.csv",
        Path("results/e11_cifar100_resnet_layer_jvp_tail_quality") / "config.json",
        Path("figures/e11_cifar100_resnet_layer_jvp_tail_quality") / "cifar100_resnet_layer_jvp_tail_quality.png",
        Path("discussion/e11_cifar100_resnet_layer_jvp_tail_quality.md"),
        Path("results/e11_cifar100_resnet_layer_jvp_checkpoint_prediction") / "metrics.csv",
        Path("results/e11_cifar100_resnet_layer_jvp_checkpoint_prediction") / "paired_metrics.csv",
        Path("results/e11_cifar100_resnet_layer_jvp_checkpoint_prediction") / "layer_summary.csv",
        Path("results/e11_cifar100_resnet_layer_jvp_checkpoint_prediction") / "checkpoint_summary.csv",
        Path("results/e11_cifar100_resnet_layer_jvp_checkpoint_prediction") / "prediction_pairs.csv",
        Path("results/e11_cifar100_resnet_layer_jvp_checkpoint_prediction") / "prediction_summary.csv",
        Path("results/e11_cifar100_resnet_layer_jvp_checkpoint_prediction") / "config.json",
        Path("figures/e11_cifar100_resnet_layer_jvp_checkpoint_prediction") / "cifar100_resnet_layer_jvp_checkpoint_prediction.png",
        Path("discussion/e11_cifar100_resnet_layer_jvp_checkpoint_prediction.md"),
        Path("results/e11_cifar100_resnet_lt_standard_eval") / "train_trace.csv",
        Path("results/e11_cifar100_resnet_lt_standard_eval") / "class_metrics.csv",
        Path("results/e11_cifar100_resnet_lt_standard_eval") / "group_metrics.csv",
        Path("results/e11_cifar100_resnet_lt_standard_eval") / "summary.csv",
        Path("results/e11_cifar100_resnet_lt_standard_eval") / "class_summary.csv",
        Path("results/e11_cifar100_resnet_lt_standard_eval") / "config.json",
        Path("figures/e11_cifar100_resnet_lt_standard_eval") / "cifar100_resnet_lt_standard_eval.png",
        Path("discussion/e11_cifar100_resnet_lt_standard_eval.md"),
        Path("results/e11_cifar100_resnet_practical_muon_bridge") / "metrics.csv",
        Path("results/e11_cifar100_resnet_practical_muon_bridge") / "paired_metrics.csv",
        Path("results/e11_cifar100_resnet_practical_muon_bridge") / "summary.csv",
        Path("results/e11_cifar100_resnet_practical_muon_bridge") / "step_summary.csv",
        Path("results/e11_cifar100_resnet_practical_muon_bridge") / "config.json",
        Path("figures/e11_cifar100_resnet_practical_muon_bridge") / "cifar100_resnet_practical_muon_bridge.png",
        Path("discussion/e11_cifar100_resnet_practical_muon_bridge.md"),
        Path("results/e11_long_tail_imbalance_ablation") / "step_metrics.csv",
        Path("results/e11_long_tail_imbalance_ablation") / "summary.csv",
        Path("figures/e11_long_tail_imbalance_ablation") / "long_tail_imbalance_ablation.png",
        Path("discussion/e11_long_tail_imbalance_ablation.md"),
        Path("results/e11_long_tail_checkpoint_sweep") / "step_metrics.csv",
        Path("results/e11_long_tail_checkpoint_sweep") / "summary.csv",
        Path("figures/e11_long_tail_checkpoint_sweep") / "long_tail_checkpoint_sweep.png",
        Path("discussion/e11_long_tail_checkpoint_sweep.md"),
        Path("results/e11_long_tail_class_partition_sweep") / "step_metrics.csv",
        Path("results/e11_long_tail_class_partition_sweep") / "summary.csv",
        Path("figures/e11_long_tail_class_partition_sweep") / "long_tail_class_partition_sweep.png",
        Path("discussion/e11_long_tail_class_partition_sweep.md"),
        Path("results/e11_long_tail_rho_sweep") / "step_metrics.csv",
        Path("results/e11_long_tail_rho_sweep") / "summary.csv",
        Path("figures/e11_long_tail_rho_sweep") / "long_tail_rho_sweep.png",
        Path("discussion/e11_long_tail_rho_sweep.md"),
        Path("results/e11_long_tail_muon_bridge") / "step_metrics.csv",
        Path("results/e11_long_tail_muon_bridge") / "pair_summary.csv",
        Path("results/e11_local_linearization") / "summary.csv",
        Path("figures/e11_long_tail_muon_bridge") / "long_tail_muon_bridge.png",
        Path("discussion/e11_long_tail_muon_bridge.md"),
        Path("results/e11_long_tail_practical_muon_bridge") / "step_metrics.csv",
        Path("results/e11_long_tail_practical_muon_bridge") / "step_summary.csv",
        Path("results/e11_long_tail_practical_muon_bridge") / "summary.csv",
        Path("figures/e11_long_tail_practical_muon_bridge") / "long_tail_practical_muon_bridge.png",
        Path("discussion/e11_long_tail_practical_muon_bridge.md"),
        Path("results/e11_long_tail_muon_state_source_control") / "step_metrics.csv",
        Path("results/e11_long_tail_muon_state_source_control") / "summary.csv",
        Path("figures/e11_long_tail_muon_state_source_control") / "long_tail_muon_state_source_control.png",
        Path("discussion/e11_long_tail_muon_state_source_control.md"),
        Path("results/e11_long_tail_practical_training") / "step_metrics.csv",
        Path("results/e11_long_tail_practical_training") / "summary.csv",
        Path("figures/e11_long_tail_practical_training") / "long_tail_practical_training.png",
        Path("discussion/e11_long_tail_practical_training.md"),
        Path("results/e11_long_tail_practical_training_lr_sweep") / "sweep_summary.csv",
        Path("figures/e11_long_tail_practical_training_lr_sweep") / "long_tail_practical_training_lr_sweep.png",
        Path("discussion/e11_long_tail_practical_training_lr_sweep.md"),
        Path("results/e11_long_tail_forgetting") / "step_metrics.csv",
        Path("results/e11_long_tail_forgetting") / "summary.csv",
        Path("figures/e11_long_tail_forgetting") / "long_tail_head_only_forgetting.png",
        Path("discussion/e11_long_tail_forgetting.md"),
        Path("results/e11_long_tail_layerwise") / "metrics.csv",
        Path("results/e11_long_tail_layerwise") / "summary.csv",
        Path("figures/e11_long_tail_layerwise") / "long_tail_layerwise_drift.png",
        Path("discussion/e11_long_tail_layerwise.md"),
        Path("paper/specgrad_activation_paper/figures") / "head_tail_drift_ratio.png",
        Path("paper/specgrad_activation_paper/figures") / "head_tail_alignment_ablation.png",
        Path("paper/specgrad_activation_paper/figures") / "long_tail_one_step_tail_response.png",
        Path("paper/specgrad_activation_paper/figures") / "long_tail_imbalance_ablation.png",
        Path("paper/specgrad_activation_paper/figures") / "long_tail_checkpoint_sweep.png",
        Path("paper/specgrad_activation_paper/figures") / "long_tail_class_partition_sweep.png",
        Path("paper/specgrad_activation_paper/figures") / "long_tail_rho_sweep.png",
        Path("paper/specgrad_activation_paper/figures") / "long_tail_muon_bridge.png",
        Path("paper/specgrad_activation_paper/figures") / "long_tail_practical_muon_bridge.png",
        Path("paper/specgrad_activation_paper/figures") / "long_tail_muon_state_source_control.png",
        Path("paper/specgrad_activation_paper/figures") / "long_tail_practical_training.png",
        Path("paper/specgrad_activation_paper/figures") / "long_tail_head_only_forgetting.png",
        Path("paper/specgrad_activation_paper/figures") / "long_tail_layerwise_drift.png",
        Path("paper/specgrad_activation_paper/figures") / "cifar100_resnet_layer_jvp_tail_quality.png",
        Path("paper/specgrad_activation_paper/figures") / "cifar100_resnet_layer_jvp_checkpoint_prediction.png",
        Path("paper/specgrad_activation_paper/figures") / "cifar100_resnet_lt_standard_eval.png",
        Path("paper/specgrad_activation_paper/figures") / "cifar100_resnet_practical_muon_bridge.png",
        Path("paper/specgrad_activation_paper/tables") / "head_tail_empirical_results.tex",
        Path("paper/specgrad_activation_paper/tables") / "local_linearization_errors.tex",
        Path("paper/specgrad_activation_paper/tables") / "e11_paper_numbers.tex",
        Path("paper/specgrad_activation_paper/main.pdf"),
        Path("paper/specgrad_activation_paper/two_page.tex"),
        Path("paper/specgrad_activation_paper/two_page.pdf"),
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
        Path("discussion/e11_pasted_review_audit.md"),
        Path("discussion/e11_completion_audit.md"),
        Path("discussion/e11_end_of_draft_self_review.md"),
        Path("discussion/e11_reference_audit.md"),
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
        "e11-paper-pdf:",
        "$(MAKE) -C paper/specgrad_activation_paper",
        "e11-paper-assets:",
        "scripts/e11_write_all_discussion_artifacts.py",
        "e11-guardrail-assets:",
        "scripts/e11_write_legacy_guardrail_artifacts.py",
        "e11-all-assets: e11-paper-assets e11-guardrail-assets",
        "e11-artifacts: e11-paper-assets",
        "e11-full: e11-paper-assets e11-paper-pdf e11-check",
        *MAIN_RESULT_SCRIPTS,
        *APPENDIX_RUNNER_SCRIPTS,
    ]
    missing_makefile_phrases = [phrase for phrase in required_makefile_phrases if phrase not in makefile_text]
    if missing_makefile_phrases:
        raise AssertionError(f"Makefile missing required E11 reproduction entries: {missing_makefile_phrases}")
    gitattributes_text = Path(".gitattributes").read_text(encoding="utf-8")
    required_gitattributes_phrases = [
        "paper/specgrad_activation_paper/tables/head_tail_empirical_results.tex linguist-generated=true",
        "paper/specgrad_activation_paper/tables/e11_paper_numbers.tex linguist-generated=true",
        "paper/specgrad_activation_paper/figures/*.png linguist-generated=true",
        "*.pdf binary",
    ]
    missing_gitattributes_phrases = [
        phrase for phrase in required_gitattributes_phrases if phrase not in gitattributes_text
    ]
    if missing_gitattributes_phrases:
        raise AssertionError(f".gitattributes missing generated/binary review hygiene entries: {missing_gitattributes_phrases}")
    readme_text = Path("README_E11.md").read_text(encoding="utf-8")
    required_readme_phrases = [
        "# E11 Head-to-Tail Paper Evidence",
        "older condition-geometry experiments that now serve as background guardrails",
        "appendix and guardrail follow-up experiments",
        "Current paper scope",
        "focused head-to-tail interference paper",
        "background evidence and guardrails",
        "should not be read as a broad claim",
        "spectral/Frobenius squared tail-example logit drift ratio is about `0.3403`",
        "the squared drift ratio is about `7.208`",
        "The spectral/Frobenius squared tail-example logit drift ratio is about `0.5501 [0.5101, 0.5931]`",
        "`polar(M_t)` has squared tail-example logit drift ratio about `0.8199 [0.6951, 0.9672]`",
        "Newton-Schulz `NS(M_t)` has squared drift ratio about `0.9116 [0.7696, 1.08]`",
        "Across 120 sampled state-step comparisons, `polar(M_t)` has squared drift ratio about `0.7292 [0.6891, 0.7717]`",
        "`NS(M_t)` has squared drift ratio about `0.8019 [0.7583, 0.848]`",
        "On Fro/GD-style trajectory states with matched seeds, batches, length, and nominal trajectory learning rate, `polar(M_t)` has squared drift ratio about `0.7255 [0.686, 0.7672]`",
        "On the same Fro/GD-style states, `NS(M_t)` has squared drift ratio about `0.7973 [0.7546, 0.8425]`",
        "final train loss ratio about `0.6468 [0.604, 0.6926]`",
        "tail eval loss ratio about `0.8549 [0.8319, 0.8786]`",
        "Tail eval margin difference is about `1.955 [1.628, 2.281]`",
        "tail eval drift RMS ratio is about `0.7501 [0.7244, 0.7767]`",
        "LR sensitivity shows why this is not a monotone optimizer story",
        "Final squared drift ratio is about `0.6167 [0.5744, 0.6622]`",
        "Unit-direction spectral JVP is larger than Frobenius in both layers",
        "make e11-cifar-resnet-fc-condition-results # submit the ResNet final-layer downstream-aware condition diagnostic via Slurm",
        "make e11-cifar-resnet-tail-quality-results # submit the tail-rich ResNet checkpoint-quality control via Slurm",
        "make e11-cifar-resnet-layer-jvp-tail-quality-results # submit the all-layer ResNet finite-difference JVP tail-quality diagnostic via Slurm",
        "make e11-cifar-resnet-layer-jvp-checkpoint-prediction-results # submit the all-layer ResNet JVP checkpoint-transfer benchmark via Slurm",
        "make e11-cifar-resnet-lt-standard-eval-results # submit the standard CIFAR-100-LT ResNet18 many/medium/few reporting baseline via Slurm",
        "A ResNet final-layer downstream-aware condition diagnostic over 40 seed/checkpoint points has weakest mean `nrank(G_H) / srank(H_T)` score about `6.566`",
        "A tail-rich ResNet control with 300 tail-train examples per class reaches best pre-update tail accuracy about `0.3739 [0.3454, 0.4024]`",
        "An all-layer ResNet finite-difference JVP tail-quality diagnostic covers 21 Conv/Linear weights and 210 paired layer/seed points",
        "An all-layer ResNet JVP checkpoint-transfer benchmark covers 3 tail-rich checkpoints and 6 directed checkpoint-transfer pairs",
        "A standard CIFAR-100-LT ResNet18 reporting baseline (IF=100, 10 AdamW seeds, no augmentation/tuning) gives many/medium/few balanced accuracy `0.3665 [0.3489, 0.3841]`, `0.1036 [0.09138, 0.1158]`, and `0.0129 [0.009351, 0.01645]`",
        "make e11-all-results",
        "make e11-paper-assets      # regenerate current head-to-tail paper Markdown/TeX artifacts",
        "make e11-guardrail-assets  # regenerate legacy condition-geometry guardrail notes",
        "make e11-all-assets        # regenerate current paper artifacts plus legacy guardrail notes",
        "make e11-paper-pdf         # rebuild paper/specgrad_activation_paper/main.pdf and two_page.pdf",
        "`diagnostic_A_definition == full_layer_input_activation`",
        *MAIN_RESULT_SCRIPTS,
        *APPENDIX_RUNNER_SCRIPTS,
    ]
    missing_readme_phrases = [phrase for phrase in required_readme_phrases if phrase not in readme_text]
    if missing_readme_phrases:
        raise AssertionError(f"README_E11.md missing current paper-facing summary entries: {missing_readme_phrases}")
    manifest_md_text = Path("discussion/e11_artifact_manifest.md").read_text(encoding="utf-8")
    required_manifest_md_phrases = [
        "separates the head-to-tail paper evidence from background condition-geometry guardrails",
        "Generated head-to-tail paper evidence plus legacy condition-geometry guardrail notes",
        "Head-to-tail LaTeX paper draft, generated paper table, and experiment triage notes",
    ]
    missing_manifest_md_phrases = [phrase for phrase in required_manifest_md_phrases if phrase not in manifest_md_text]
    if missing_manifest_md_phrases:
        raise AssertionError(f"artifact manifest Markdown missing paper-scope boundary text: {missing_manifest_md_phrases}")
    paper_readme_text = Path("paper/specgrad_activation_paper/README.md").read_text(encoding="utf-8")
    required_paper_readme_phrases = [
        "## Reproduce From Repo Root",
        "make e11-paper-assets",
        "make e11-paper-pdf",
        "make e11-check",
        "make e11-full",
        "tables/e11_paper_numbers.tex",
        "paper-local `figures/*.png`",
        "## Local Build",
        "both the main paper and the two-page report",
    ]
    assert_required_phrases(
        "paper README root reproduction/build instructions",
        paper_readme_text,
        required_paper_readme_phrases,
    )
    paper_path = Path("paper/specgrad_activation_paper/main.tex")
    paper_text = paper_path.read_text(encoding="utf-8")
    required_head_tail_paper_phrases = [
        "\\section{Introduction}",
        "\\section{Related Work}",
        "\\subsection{Local Geometry of Spectral Gradients and Muon}",
        "\\subsection{Long-Tailed Learning and Optimizer Geometry}",
        "\\subsection{Scope and Relation to Full Muon}",
        "\\section{Problem Setup}",
        "Basic matrix-norm and rank definitions are collected in Appendix \\ref{app:preliminaries}",
        "\\subsection{Head and Tail Batches}",
        "\\subsection{Local Comparison Protocol}",
        "\\section{Theoretical Analysis}",
        "\\section{Preliminaries}",
        "\\label{app:preliminaries}",
        "\\subsection{Notation Summary}",
        "tail activation entering the perturbed matrix block",
        "downstream tail Jacobian after the perturbed block",
        "\\subsection{Matrix Norms}",
        "\\subsection{Spectral Gradient Directions}",
        "\\subsection{Head-to-Tail Interference Coefficient}",
        "\\subsection{General Matched-Gain Drift Bound}",
        "\\subsection{Matrix-Block Spectral Condition}",
        "\\section{Matrix-Block Derivation}",
        "\\label{app:matrix-block-derivation}",
        "Under Frobenius geometry",
        "Under spectral geometry",
        "Spectral geometry has the smaller worst-case matched-gain tail-drift bound",
        "\\section{Tail Drift across Multiple Head-Only Steps}",
        "\\section{Tail Margin Certificate}",
        "\\section{Relation to Existing Layerwise Spectral-Update Theory}",
        "\\citet{davis2025spectral}",
        "Other recent work analyzes Muon through spectral-norm constraints \\citep{chen2025muon}",
        "studies spectral anisotropy or spectral clipping in LLM training \\citep{spectra2026,spectralclipping2026}",
        "examines Muon recipe interactions and gradient spectra in vision transformers \\citep{muonvit2026}",
        "\\citep{cui2019classbalanced}",
        "\\citep{cao2019ldam}",
        "\\citep{liu2019oltr}",
        "\\citep{kang2020decoupling}",
        "\\citep{promo2026}",
        "\\citep{muonmemory2025}",
        "\\citep{pedregosa2011scikit}",
        "fix classes \\(0\\)--\\(4\\) as head classes and classes \\(5\\)--\\(9\\) as tail classes",
        "sample the per-class train/evaluation split independently for each seed",
        "the random seed only changes the within-class sample split, mini-batches, noise draws, and initialization",
        "Head-to-Tail Interference",
        "in Long-Tailed Small-Batch Training",
        "function-drift view",
        "We study this problem through a matched-head-gain diagnostic",
        "We make four contributions: a matched-head-gain formulation",
        "Controlled boundary experiments reverse the spectral/Frobenius squared drift ratio",
        "a spectral/Frobenius squared tail-example logit drift ratio",
        "not an optimizer-level performance claim",
        "do not constitute a theory of complete Muon training",
        "after matching the same head-batch gain, which update perturbs the tail function less",
        "We make four contributions",
        "We formulate a local head-to-tail interference problem",
        "including selected-state compatibility checks for Muon-style momentum and Newton--Schulz directions",
        "The experiments are diagnostic rather than benchmark-driven",
        "tail loss, margin, and accuracy are reported separately",
        "local function-drift reduction",
        "final-layer-only ResNet condition diagnostic",
        "tail-rich checkpoint-quality control",
        "tail-example logit drift means the drift of the full class-logit vector",
        "not restricted to logits of tail classes only",
        "Sandwiched sensitivity",
        "matched-head-gain",
        "worst-case sensitivity upper bound",
        "\\widetilde{\\I}_N(T\\mid H)",
        "\\ssrank(B_T,A_T)",
        "\\sum_i\\sigma_i(B_T)^2\\sigma_i(A_T)^2",
        "observed drift",
        "singular vectors",
        "confidence interval",
        "\\label{assump:local-head-tail}",
        "Under Assumption \\ref{assump:local-head-tail}",
        "\\section{Experiments}",
        "\\subsection{Experiment Protocol Summary}",
        "\\label{sec:experiment-protocol-summary}",
        "All experiments use the same comparison principle unless stated otherwise",
        "Unless otherwise stated, ratios below are squared drift ratios, spectral/polar divided by Frobenius/GD",
        "Because the theory is first-order, we also check the local linearization quality",
        "\\input{tables/local_linearization_errors}",
        "\\subsection{Synthetic Head-to-Tail Linear Model}",
        "Synthetic head-to-tail boundary diagnostic",
        "Synthetic singular-vector alignment ablation",
        "\\subsection{One-Step Diagnostic on Controlled Long-Tailed Digits}",
        "Controlled long-tailed digits one-step diagnostic",
        "\\subsection{Muon-Style Direction Compatibility}",
        "Muon-style direction compatibility diagnostic along a short NS-Muon-style trajectory",
        "\\subsection{Layerwise Diagnostic}",
        "Layerwise tail-drift diagnostic",
        "\\nrank(G_{H,2})/\\ssrank(B_{T,2},A_{T,2})=\\EelevenLayerTwoTheoremConditionScore",
        "For the first ReLU layer, gates vary across tail samples",
        "frozen-gate local-operator score",
        "\\appendix",
        "The appendix provides supporting definitions and checks in the order they are used in the paper",
        "notation and norm preliminaries; proofs and the matrix-block derivation",
        "multi-step drift and margin-certificate consequences",
        "additional discussion of practical Muon-style directions; reproducibility details; auxiliary tables; and auxiliary figures",
        "\\section{Reproducibility Details}",
        "\\label{app:repro}",
        "the head mini-batches and target gains are precomputed once per seed from the initial checkpoint",
        "\\rho_t=0.02\\,\\Loss_H(\\theta_0;B_t)",
        "the intended first-order head-gain budget is matched per seed and step",
        "the realized nonlinear head-loss decrease is measured after each update and is not assumed to be exactly identical",
        "\\section{Auxiliary Diagnostic Figures}",
        "\\label{app:aux-figures}",
        "Fixed-checkpoint Muon-style direction compatibility diagnostic on controlled long-tailed digits",
        "Controlled long-tailed digits practical training diagnostic",
        "Tail forgetting probe under consecutive head-only steps",
        "B_T\\in\\R^{q\\times m}",
        "A_T\\in\\R^{k\\times r}",
        "\\section{Limitations and Discussion}",
        "\\subsection{What the Evidence Establishes}",
        "Table \\ref{tab:claim-boundary} separates the current diagnostic claims from stronger claims",
        "\\label{tab:claim-boundary}",
        "Claim boundary. All reported numbers come from matched-head-gain diagnostics unless explicitly marked as practical training",
        "spectral/Fro head-alignment ratio",
        "matched operator-norm ratio",
        "norm-specific rather than a uniformly smaller parameter step",
        "\\section{Additional Discussion}",
        "\\label{prop:blockwise-polar}",
        "Blockwise polar steepest direction",
        "\\norm{D}_{\\mathrm{maxop}}=\\max_\\ell\\norm{D_\\ell}_{\\op}",
        "\\norm{G}_{*,1}=\\sum_\\ell \\norm{G_\\ell}_*",
        "\\label{app:additional-discussion}",
        "These observations motivate using Muon-style directions as local implementation probes for spectral/polar geometry, while leaving a full theory of Muon training to future work",
        "We therefore treat this experiment as a consistency check for the local mechanism, not as a tuned Adam-vs-Muon comparison",
        "Additional discussion of the gap between ideal spectral gradients and practical Muon-style updates",
        "The current evidence supports a local mechanism claim within the tested matched-head-gain diagnostics",
        "standard long-tailed benchmarks",
        "practical Muon training needs more complete ablation",
        "larger-architecture layerwise diagnostics",
        "\\section{Conclusion}",
        "not a complete long-tailed classification optimizer benchmark",
        "\\label{fig:head-tail-boundary}",
        "\\label{fig:head-tail-alignment-ablation}",
        "\\label{fig:long-tail-one-step}",
        "\\label{fig:long-tail-imbalance-ablation}",
        "\\label{fig:long-tail-checkpoint-sweep}",
        "\\label{fig:long-tail-class-partition-sweep}",
        "\\label{fig:long-tail-rho-sweep}",
        "\\label{tab:one-step-tail-outcomes}",
        "Tail outcomes, margin-relevant drift, and head-gain readouts",
        "Small post-update effects are reported as changes from the pre-update tail state",
        "Pre-update tail CE",
        "Tail CE increase, Fro/GD",
        "Tail CE increase, spectral/polar",
        "Tail CE increase difference, spectral--Fro",
        "Pre-update mean margin",
        "Margin-drop difference, spectral--Fro",
        "Head-alignment denominator, spectral/Fro",
        "Matched update Frobenius norm, spectral/Fro",
        "Matched update operator norm, spectral/Fro",
        "Pre-update tail accuracy",
        "Accuracy-drop difference, spectral--Fro",
        "Certified unchanged-prediction fraction, Fro; spectral",
        "The positive-margin prediction-change rate is zero for both directions",
        "The largest squared drift-ratio upper endpoints are \\(\\EelevenLongTailImbalanceWorstRatioCiHigh\\)",
        "\\EelevenLongTailCheckpointSweepSettings\\) warmup checkpoints",
        "\\EelevenLongTailClassPartitionSweepSettings\\) head/tail class partitions",
        "\\EelevenLongTailRhoSweepSettings\\) target head-gain fractions",
        "does not show preservation of a high-quality tail predictor",
        "\\EelevenLongTailCheckpointSweepTailAccuracyRange",
        "the worst upper endpoint is \\(\\EelevenLongTailCheckpointSweepWorstRatioCiHigh\\)",
        "The weakest full tail-example squared logit drift case is \\(\\EelevenLongTailClassPartitionWorstPartition\\)",
        "The weakest full tail-example squared logit drift case is \\(\\rho/L_H=\\EelevenLongTailRhoSweepWorstFraction\\)",
        "Target head-gain fraction sweep for the controlled long-tailed digits one-step diagnostic",
        "tail training examples per class \\(80,40,20,10\\)",
        "\\label{fig:long-tail-practical-muon-bridge}",
        "\\label{fig:long-tail-muon-state-source-control}",
        "\\label{fig:long-tail-forgetting}",
        "\\label{fig:long-tail-layerwise}",
        "\\label{fig:cifar-resnet-practical-muon-bridge}",
        "figures/head_tail_drift_ratio.png",
        "figures/head_tail_alignment_ablation.png",
        "figures/long_tail_one_step_tail_response.png",
        "figures/long_tail_imbalance_ablation.png",
        "figures/long_tail_checkpoint_sweep.png",
        "figures/long_tail_class_partition_sweep.png",
        "figures/long_tail_rho_sweep.png",
        "figures/long_tail_practical_muon_bridge.png",
        "figures/long_tail_muon_state_source_control.png",
        "figures/long_tail_head_only_forgetting.png",
        "figures/long_tail_layerwise_drift.png",
        "figures/cifar100_resnet_layer_jvp_checkpoint_prediction.png",
        "figures/cifar100_resnet_lt_standard_eval.png",
        "figures/cifar100_resnet_practical_muon_bridge.png",
        "momentum-gradient alignment",
        "Tianyang Liu",
        "UCDavis",
        "tlyliu@ucdavis.edu",
    ]
    missing_head_tail_paper_phrases = [phrase for phrase in required_head_tail_paper_phrases if phrase not in paper_text]
    if missing_head_tail_paper_phrases:
        raise AssertionError(
            "head-to-tail paper draft missing required theory/protocol caveats: "
            f"{missing_head_tail_paper_phrases}"
        )
    if "../../figures/" in paper_text:
        raise AssertionError("paper main tex must use paper-local figures instead of repo-level ../../figures paths")
    if "\\tableofcontents" in paper_text:
        raise AssertionError("paper main tex should follow compact ICLR-style front matter without a table of contents")
    if "ctexart" in paper_text:
        raise AssertionError("paper main tex should use a standard English article class, not ctexart")
    if "\\usepackage{iclr2025_conference,times}" not in paper_text:
        raise AssertionError("paper main tex should use the official ICLR 2025 conference style")
    forbidden_format_overrides = [
        "\\usepackage{geometry}",
        "\\geometry{",
        "a4paper",
        "\\setlength{\\textfloatsep}",
        "\\setlength{\\floatsep}",
        "\\setlength{\\intextsep}",
    ]
    found_format_overrides = [item for item in forbidden_format_overrides if item in paper_text]
    if found_format_overrides:
        raise AssertionError(f"paper main tex should not override ICLR formatting: {found_format_overrides}")
    forbidden_main_paper_phrases = [
        "Anonymous authors",
        "Paper under double-blind review",
        "Under review as a conference paper",
        "\\,\\Eeleven",
        "summary-table",
        "sklearn",
        "paper-facing",
        "make e11",
        "repository root",
        "TODO",
        "TBD",
        "notebook",
        "internal",
        "real-benchmark",
        "over-simple story",
        "fully explained optimizer",
        "should be read only",
        "SpecGrad/polar",
        "ideal polar/SpecGrad",
        "evidence consistent with lower",
        "logit-drift",
        "full tail-example drift ratio",
        "weakest full-drift case",
        "weakest centered-drift case",
        "local function preservation",
        "tail-function preservation between rare tail batches",
        "Tail Preservation across Multiple Head-Only Steps",
        "Tail Margin Preservation",
        "This gives a direct tail-preservation condition",
        "certify label preservation",
        "Certified preserved fraction, Fro; spectral",
    ]
    assert_forbidden_phrases_absent(
        "paper main tex non-paper wording",
        paper_text,
        forbidden_main_paper_phrases,
    )
    for required_style_path in [
        Path("paper/specgrad_activation_paper/iclr2025_conference.sty"),
        Path("paper/specgrad_activation_paper/iclr2025_conference.bst"),
    ]:
        if not required_style_path.exists():
            raise FileNotFoundError(f"missing official ICLR style file: {required_style_path}")
    paper_english_paths = [
        *sorted(Path("paper/specgrad_activation_paper").rglob("*.tex")),
        Path("paper/specgrad_activation_paper/README.md"),
    ]
    for paper_english_path in paper_english_paths:
        paper_english_content = paper_english_path.read_text(encoding="utf-8")
        cjk_chars = [char for char in paper_english_content if "\u4e00" <= char <= "\u9fff"]
        cjk_punctuation = [char for char in paper_english_content if char in "。，；：（）“”、"]
        if cjk_chars or cjk_punctuation:
            raise AssertionError(
                "paper package files should be fully English; "
                f"{paper_english_path} has {len(cjk_chars)} CJK characters and "
                f"{len(cjk_punctuation)} CJK punctuation marks"
            )
    references_index = paper_text.find("\\bibliography{references}")
    appendix_index = paper_text.find("\\appendix")
    if references_index < 0 or appendix_index < 0 or references_index > appendix_index:
        raise AssertionError("paper main tex should place references before appendix for conference-style layout")
    main_text_before_appendix = paper_text[:appendix_index]
    main_sections = re.findall(r"^\\section\{([^}]*)\}", main_text_before_appendix, flags=re.MULTILINE)
    expected_main_sections = [
        "Introduction",
        "Related Work",
        "Problem Setup",
        "Theoretical Analysis",
        "Experiments",
        "Limitations and Discussion",
        "Conclusion",
    ]
    if main_sections != expected_main_sections:
        raise AssertionError(
            "paper main tex should use the requested compact main-section structure; "
            f"found {main_sections}"
        )
    main_experiment_text = paper_text.split("\\section{Limitations and Discussion}", maxsplit=1)[0]
    main_experiment_figure_count = len(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", main_experiment_text))
    if main_experiment_figure_count > 4:
        raise AssertionError(
            "paper main experiment narrative should keep auxiliary diagnostics in the appendix; "
            f"found {main_experiment_figure_count} figures before discussion"
        )
    assert_includegraphics_files_exist(paper_path)
    assert_latex_log_has_no_serious_warnings(paper_path.with_suffix(".log"))
    assert_pdf_artifact_is_valid(paper_path.with_suffix(".pdf"), min_size_bytes=10_000)
    assert_pdf_text_has_no_known_artifacts(paper_path.with_suffix(".pdf"))
    assert_bibtex_citations_are_defined(paper_path, paper_path.parent / "references.bib")
    references_text = (paper_path.parent / "references.bib").read_text(encoding="utf-8")
    required_reference_urls = [
        "https://arxiv.org/abs/2512.04299",
        "https://arxiv.org/abs/2506.15054",
        "https://arxiv.org/abs/2602.11185",
        "https://arxiv.org/abs/2603.14315",
        "https://openreview.net/forum?id=go388T3QjQ",
        "https://arxiv.org/abs/2509.26030",
        "https://www.jmlr.org/papers/v12/pedregosa11a.html",
        "https://openreview.net/forum?id=r1gRTCVFvB",
    ]
    assert_required_phrases("paper bibliography primary-source URLs", references_text, required_reference_urls)
    spectral_clipping_kind, spectral_clipping_body = bibtex_entry_body(references_text, "spectralclipping2026")
    if spectral_clipping_kind != "misc":
        raise AssertionError("spectralclipping2026 should remain an arXiv preprint entry unless proceedings metadata is verified")
    if "ICML 2026" in spectral_clipping_body:
        raise AssertionError("spectralclipping2026 bibliography entry should cite arXiv only until proceedings metadata is verified")
    if "Proceedings of the 43rd International Conference on Machine Learning" in spectral_clipping_body:
        raise AssertionError("spectralclipping2026 must not claim unverified ICML proceedings metadata")
    promo_kind, promo_body = bibtex_entry_body(references_text, "promo2026")
    if promo_kind != "misc":
        raise AssertionError("promo2026 should remain a misc OpenReview submission entry unless its venue status is verified")
    required_promo_phrases = [
        "Submitted to ICLR 2026, OpenReview",
        "https://openreview.net/forum?id=go388T3QjQ",
    ]
    assert_required_phrases("promo2026 conservative submission citation", promo_body, required_promo_phrases)
    forbidden_submission_claims = [
        "Accepted to",
        "accepted to",
        "Published in",
        "published in",
        "International Conference on Learning Representations 2026",
        "ICLR 2026 Conference",
        "Proceedings",
    ]
    assert_forbidden_phrases_absent("promo2026 unverified venue status", promo_body, forbidden_submission_claims)
    for arxiv_key in [
        "davis2025spectral",
        "chen2025muon",
        "spectra2026",
        "muonmemory2025",
        "muonvit2026",
    ]:
        arxiv_kind, arxiv_body = bibtex_entry_body(references_text, arxiv_key)
        if arxiv_kind != "misc":
            raise AssertionError(f"{arxiv_key} should remain a misc arXiv entry unless proceedings metadata is verified")
        assert_required_phrases(f"{arxiv_key} arXiv metadata", arxiv_body, ["archivePrefix = {arXiv}", "note          = {arXiv:"])
    local_linearization_table_text = (paper_path.parent / "tables/local_linearization_errors.tex").read_text(
        encoding="utf-8"
    )
    required_local_linearization_table_phrases = [
        "\\label{tab:local-linearization-error}",
        "Relative error",
        "$\\norm{J_T\\Delta}_F$",
        "Residual norm",
        "Fro/GD",
        "$\\polar(G_t)$",
        "$\\polar(M_t)$",
        "NS$(M_t)$",
    ]
    assert_required_phrases(
        "local linearization error table",
        local_linearization_table_text,
        required_local_linearization_table_phrases,
    )
    two_page_path = paper_path.parent / "two_page.tex"
    two_page_text = two_page_path.read_text(encoding="utf-8")
    required_two_page_phrases = [
        "Head-to-Tail Interference in Long-Tailed Small-Batch Training",
        "\\section*{Abstract}",
        "\\section*{1. Problem}",
        "\\section*{2. Matched-Gain Geometry}",
        "\\section*{3. Matrix-Block Condition}",
        "\\section*{4. Evidence}",
        "\\section*{5. Interpretation}",
        "\\section*{6. Claim Boundary}",
        "\\section*{7. Conclusion}",
        "This report studies a local stability question",
        "change logits on rare tail examples before tail samples reappear",
        "Tail-example logit drift always means drift of the full class-logit vector",
        "not only tail-class logits",
        "scikit-learn \\(8\\times 8\\) digits data",
        "fixed head classes \\(0\\)--\\(4\\)",
        "fixed tail classes \\(5\\)--\\(9\\)",
        "20 random within-class splits",
        "tested split/seed distribution",
        "not generalization guarantees",
        "\\EelevenLongTailOneStepAlignmentRatio",
        "\\EelevenLongTailOneStepUpdateFroNormRatio",
        "\\EelevenLongTailOneStepUpdateOpNormRatio",
        "the scaling effect is norm-specific",
        "direct non-synthetic evidence for lower matched-gain tail-example logit drift",
        "The supported claims are deliberately local",
        "smaller operator-norm step despite a larger Frobenius-norm step",
        "alignment ablation shows that this is a bound-ordering condition",
        "rather than a standalone realized-drift predictor",
        "selected-state compatibility checks",
        "state selection remain separate sources of variation",
        "The mechanism should be evaluated before making optimizer-level claims",
        "The current evidence supports a local mechanism claim",
        "tail-rich ResNet control raises pre-update tail accuracy",
        "\\textbf{References.}",
        "Pedregosa et al.",
        "\\EelevenLocalLinearizationMaxRelativeErrorCiHigh",
        "standard long-tailed benchmarks",
        "Tianyang Liu",
        "UCDavis",
        "tlyliu@ucdavis.edu",
    ]
    missing_two_page_phrases = [phrase for phrase in required_two_page_phrases if phrase not in two_page_text]
    if missing_two_page_phrases:
        raise AssertionError(f"two-page paper summary missing required claim-boundary content: {missing_two_page_phrases}")
    forbidden_two_page_phrases = [
        "A Two-Page Summary",
        "Reading.",
        "over-simple story",
        "change rare tail-class logits",
        "cautious mechanism paper",
        "smaller effective parameter step",
        "real long-tailed benchmarks",
        "sklearn",
        "summary-table",
        "notebook",
        "generated",
        "repository root",
        "make e11",
        "paper-facing",
        "TODO",
        "TBD",
        "evidence consistent with lower",
        "SpecGrad/polar",
        "ideal polar/SpecGrad",
        "tail stable rank",
        "logit-drift",
        "fixed \\(\\polar(M_t)\\) ratio",
        "short-trajectory NS\\((M_t)\\) ratio",
    ]
    assert_forbidden_phrases_absent(
        "two-page paper summary non-paper wording",
        two_page_text,
        forbidden_two_page_phrases,
    )
    assert_latex_log_has_no_serious_warnings(two_page_path.with_suffix(".log"))
    assert_latex_log_has_no_box_warnings(two_page_path.with_suffix(".log"))
    assert_pdf_artifact_is_valid(two_page_path.with_suffix(".pdf"), min_size_bytes=10_000)
    assert_pdf_text_has_no_known_artifacts(two_page_path.with_suffix(".pdf"))
    paper_readme_text = Path("paper/specgrad_activation_paper/README.md").read_text(encoding="utf-8")
    required_paper_readme_phrases = [
        "worst-case sensitivity-bound condition",
        "singular-vector alignment",
        "suppresses the anonymous-author block",
        "the review-status header while keeping the ICLR page geometry",
        "paper-local copies of the paper-facing figures",
        "scripts/e11_write_paper_figures.py",
        "two_page.tex",
        "two_page.pdf",
        "head_tail_drift_ratio.png",
        "head_tail_alignment_ablation.png",
        "long_tail_one_step_tail_response.png",
        "long_tail_imbalance_ablation.png",
        "long_tail_checkpoint_sweep.png",
        "long_tail_class_partition_sweep.png",
        "long_tail_rho_sweep.png",
        "long_tail_muon_bridge.png",
        "long_tail_practical_muon_bridge.png",
        "long_tail_muon_state_source_control.png",
        "long_tail_practical_training.png",
        "long_tail_head_only_forgetting.png",
        "long_tail_layerwise_drift.png",
        "cifar100_resnet_layer_jvp_tail_quality.png",
        "cifar100_resnet_layer_jvp_checkpoint_prediction.png",
        "cifar100_resnet_lt_standard_eval.png",
        "cifar100_resnet_practical_muon_bridge.png",
        "tables/",
        "e11_paper_numbers.tex",
        "head_tail_empirical_results.tex",
        "local_linearization_errors.tex",
        "scripts/e11_write_head_tail_paper_results.py",
        "scripts/e11_write_local_linearization_table.py",
        "make two-page",
    ]
    missing_paper_readme_phrases = [phrase for phrase in required_paper_readme_phrases if phrase not in paper_readme_text]
    if missing_paper_readme_phrases:
        raise AssertionError(f"paper README missing current source inventory: {missing_paper_readme_phrases}")
    if "neutral review header" in paper_readme_text:
        raise AssertionError("paper README must not claim that the local build uses a neutral review header")
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
    head_tail_steps = pd.read_csv(Path("results/e11_head_tail_interference") / "step_metrics.csv")
    head_tail_summary = pd.read_csv(Path("results/e11_head_tail_interference") / "pair_summary.csv")
    if len(head_tail_steps) != 320:
        raise AssertionError(f"head-tail interference row count mismatch: expected 320, got {len(head_tail_steps)}")
    head_tail_by_setting = head_tail_summary.set_index("setting")
    head_tail_positive = head_tail_by_setting.loc["high_head_rank_low_tail_srank"]
    if not bool(head_tail_positive["predicted_spectral_less_drift"]):
        raise AssertionError("head-tail positive setting must predict lower spectral drift")
    if not (
        head_tail_positive["mean_tail_downstream_aware_stable_rank"]
        < head_tail_positive["mean_head_gradient_nuclear_rank"]
    ):
        raise AssertionError("head-tail positive setting must satisfy ssrank(B_T,A_T) < nrank(G_H)")
    if not (
        head_tail_positive["tail_output_drift_sq_ratio_ci95_high"] < 1.0
        and head_tail_positive["spectral_less_tail_output_drift_fraction"] == 1.0
    ):
        raise AssertionError("head-tail positive setting must have CI-bounded lower spectral drift")
    head_tail_negative = head_tail_by_setting.loc["low_head_rank_high_tail_srank"]
    if bool(head_tail_negative["predicted_spectral_less_drift"]):
        raise AssertionError("head-tail negative setting must not predict lower spectral drift")
    if not (
        head_tail_negative["mean_tail_downstream_aware_stable_rank"]
        > head_tail_negative["mean_head_gradient_nuclear_rank"]
    ):
        raise AssertionError("head-tail negative setting must satisfy ssrank(B_T,A_T) > nrank(G_H)")
    if not (
        head_tail_negative["tail_output_drift_sq_ratio_ci95_low"] > 1.0
        and head_tail_negative["spectral_less_tail_output_drift_fraction"] == 0.0
    ):
        raise AssertionError("head-tail negative setting must have CI-bounded higher spectral drift")
    alignment_steps = pd.read_csv(Path("results/e11_head_tail_alignment_ablation") / "step_metrics.csv")
    alignment_summary = pd.read_csv(Path("results/e11_head_tail_alignment_ablation") / "summary.csv")
    if len(alignment_steps) != 1000:
        raise AssertionError(
            f"head-tail alignment ablation row count mismatch: expected 1000, got {len(alignment_steps)}"
        )
    alignment_by_setting = alignment_summary.set_index("setting")
    alignment_positive = alignment_by_setting.loc["high_head_rank_low_tail_srank"]
    if not (
        int(alignment_positive["seeds"]) == 500
        and bool(alignment_positive["predicted_spectral_less_drift"])
        and alignment_positive["mean_theory_ratio_spectral_over_fro"] < 1.0
        and alignment_positive["tail_output_drift_sq_ratio_ci95_low"] > 1.0
        and 0.2 < alignment_positive["spectral_less_tail_output_drift_fraction"] < 0.8
    ):
        raise AssertionError(
            "head-tail alignment ablation must show positive spectra with mixed realized drift under random alignment"
        )
    alignment_negative = alignment_by_setting.loc["low_head_rank_high_tail_srank"]
    if not (
        int(alignment_negative["seeds"]) == 500
        and not bool(alignment_negative["predicted_spectral_less_drift"])
        and alignment_negative["mean_theory_ratio_spectral_over_fro"] > 1.0
        and alignment_negative["tail_output_drift_sq_ratio_ci95_low"] > 1.0
    ):
        raise AssertionError("head-tail alignment ablation must preserve the negative-spectrum boundary")
    long_tail_steps = pd.read_csv(Path("results/e11_long_tail_one_step") / "step_metrics.csv")
    long_tail_summary = pd.read_csv(Path("results/e11_long_tail_one_step") / "pair_summary.csv")
    long_tail_layers = pd.read_csv(Path("results/e11_long_tail_one_step") / "layer_metrics.csv")
    if len(long_tail_steps) != 40:
        raise AssertionError(f"long-tail one-step row count mismatch: expected 40, got {len(long_tail_steps)}")
    if len(long_tail_layers) != 80:
        raise AssertionError(f"long-tail one-step layer row count mismatch: expected 80, got {len(long_tail_layers)}")
    if set(long_tail_steps["geometry"]) != {"frobenius", "spectral"}:
        raise AssertionError("long-tail one-step must compare frobenius and spectral geometries")
    if long_tail_steps["seed"].nunique() != 20:
        raise AssertionError("long-tail one-step must include 20 seeds")
    required_one_step_columns = {
        "tail_positive_margin_fraction_before",
        "tail_margin_certified_preserved_fraction",
        "tail_prediction_changed_fraction",
        "tail_positive_margin_prediction_changed_fraction",
        "centered_tail_output_drift_fro",
        "true_logit_delta_fro",
        "competitor_logit_delta_fro",
        "margin_delta_fro",
        "actual_head_gain_relative_error",
        "alignment",
        "update_fro_norm",
        "update_op_norm",
    }
    missing_one_step_columns = required_one_step_columns - set(long_tail_steps.columns)
    if missing_one_step_columns:
        raise AssertionError(f"long-tail one-step missing tail absolute/certificate columns: {missing_one_step_columns}")
    one_step_fraction_columns = [
        "tail_positive_margin_fraction_before",
        "tail_margin_certified_preserved_fraction",
        "tail_prediction_changed_fraction",
        "tail_positive_margin_prediction_changed_fraction",
    ]
    for column in one_step_fraction_columns:
        if ((long_tail_steps[column] < 0.0) | (long_tail_steps[column] > 1.0)).any():
            raise AssertionError(f"long-tail one-step fraction column out of [0, 1]: {column}")
    head_gain_gap = (
        long_tail_steps.pivot_table(index="seed", columns="geometry", values="matched_first_order_head_gain")
        .diff(axis=1)
        .abs()
        .max()
        .max()
    )
    if float(head_gain_gap) > 1e-12:
        raise AssertionError("long-tail one-step head first-order gains must be exactly matched per seed")
    one_step_wide = long_tail_steps.pivot(index="seed", columns="geometry")
    alignment_ratio = one_step_wide["alignment"]["spectral"] / one_step_wide["alignment"]["frobenius"]
    update_fro_ratio = one_step_wide["update_fro_norm"]["spectral"] / one_step_wide["update_fro_norm"]["frobenius"]
    update_op_ratio = one_step_wide["update_op_norm"]["spectral"] / one_step_wide["update_op_norm"]["frobenius"]
    if not (
        alignment_ratio.min() > 1.0
        and update_fro_ratio.min() > 1.0
        and update_op_ratio.max() < 1.0
    ):
        raise AssertionError(
            "long-tail one-step head-alignment and norm ratios must support the norm-specific scaling statement"
        )
    long_tail_row = long_tail_summary.iloc[0]
    if not (
        long_tail_row["tail_output_drift_sq_ratio_ci95_high"] < 1.0
        and long_tail_row["spectral_less_tail_output_drift_fraction"] == 1.0
    ):
        raise AssertionError("long-tail one-step must have CI-bounded lower spectral tail-example logit drift")
    required_one_step_summary_columns = {
        "geomean_centered_tail_output_drift_sq_ratio_spectral_over_fro",
        "centered_tail_output_drift_sq_ratio_ci95_low",
        "centered_tail_output_drift_sq_ratio_ci95_high",
        "geomean_true_logit_delta_sq_ratio_spectral_over_fro",
        "true_logit_delta_sq_ratio_ci95_low",
        "true_logit_delta_sq_ratio_ci95_high",
        "geomean_competitor_logit_delta_sq_ratio_spectral_over_fro",
        "competitor_logit_delta_sq_ratio_ci95_low",
        "competitor_logit_delta_sq_ratio_ci95_high",
        "geomean_margin_delta_sq_ratio_spectral_over_fro",
        "margin_delta_sq_ratio_ci95_low",
        "margin_delta_sq_ratio_ci95_high",
        "mean_actual_head_gain_relative_error_frobenius",
        "actual_head_gain_relative_error_frobenius_ci95_low",
        "actual_head_gain_relative_error_frobenius_ci95_high",
        "mean_actual_head_gain_relative_error_spectral",
        "actual_head_gain_relative_error_spectral_ci95_low",
        "actual_head_gain_relative_error_spectral_ci95_high",
        "mean_tail_loss_before",
        "mean_tail_loss_after_frobenius",
        "mean_tail_loss_after_spectral",
        "mean_tail_positive_margin_fraction_before",
        "mean_tail_margin_certified_preserved_fraction_frobenius",
        "mean_tail_margin_certified_preserved_fraction_spectral",
        "mean_tail_prediction_changed_fraction_frobenius",
        "mean_tail_prediction_changed_fraction_spectral",
        "mean_tail_positive_margin_prediction_changed_fraction_frobenius",
        "mean_tail_positive_margin_prediction_changed_fraction_spectral",
    }
    missing_one_step_summary_columns = required_one_step_summary_columns - set(long_tail_summary.columns)
    if missing_one_step_summary_columns:
        raise AssertionError(
            f"long-tail one-step summary missing absolute/certificate columns: {missing_one_step_summary_columns}"
        )
    if not (
        long_tail_row["centered_tail_output_drift_sq_ratio_ci95_high"] < 1.0
        and long_tail_row["competitor_logit_delta_sq_ratio_ci95_high"] < 1.0
        and long_tail_row["margin_delta_sq_ratio_ci95_high"] < 1.0
        and long_tail_row["true_logit_delta_sq_ratio_ci95_low"] > 1.0
        and long_tail_row["actual_head_gain_relative_error_frobenius_ci95_high"] < 0.05
        and long_tail_row["actual_head_gain_relative_error_spectral_ci95_high"] < 0.05
    ):
        raise AssertionError(
            "long-tail one-step margin-relevant/head-gain readouts must match the stated narrowed claim"
        )
    if not (
        long_tail_row["mean_tail_loss_before"] > 1.0
        and long_tail_row["mean_tail_margin_certified_preserved_fraction_frobenius"] > 0.95
        and long_tail_row["mean_tail_margin_certified_preserved_fraction_spectral"] > 0.95
        and long_tail_row["mean_tail_positive_margin_prediction_changed_fraction_frobenius"] == 0.0
        and long_tail_row["mean_tail_positive_margin_prediction_changed_fraction_spectral"] == 0.0
    ):
        raise AssertionError("long-tail one-step absolute/certificate readout must support the stated tail caveat")
    cifar_mlp_steps = pd.read_csv(Path("results/e11_cifar100_lt_one_step") / "step_metrics.csv")
    cifar_mlp_summary = pd.read_csv(Path("results/e11_cifar100_lt_one_step") / "pair_summary.csv")
    cifar_mlp_layers = pd.read_csv(Path("results/e11_cifar100_lt_one_step") / "layer_metrics.csv")
    if len(cifar_mlp_steps) != 10 or len(cifar_mlp_layers) != 20:
        raise AssertionError("CIFAR-100-LT MLP one-step diagnostic must contain 5 seeds x 2 geometries")
    if set(cifar_mlp_steps["geometry"]) != {"frobenius", "spectral"} or cifar_mlp_steps["seed"].nunique() != 5:
        raise AssertionError("CIFAR-100-LT MLP one-step diagnostic must compare Fro/GD and spectral on 5 seeds")
    if len(cifar_mlp_summary) != 1:
        raise AssertionError("CIFAR-100-LT MLP one-step summary must contain one paired-summary row")
    cifar_mlp_row = cifar_mlp_summary.iloc[0]
    if not (
        cifar_mlp_row["tail_output_drift_sq_ratio_ci95_high"] < 1.0
        and cifar_mlp_row["spectral_less_tail_output_drift_fraction"] == 1.0
        and cifar_mlp_row["tail_loss_increase_diff_ci95_low"] > 0.0
    ):
        raise AssertionError("CIFAR-100-LT MLP diagnostic must preserve lower drift and the tail-loss caveat")
    cifar_resnet_steps = pd.read_csv(Path("results/e11_cifar100_resnet_one_step") / "step_metrics.csv")
    cifar_resnet_summary = pd.read_csv(Path("results/e11_cifar100_resnet_one_step") / "pair_summary.csv")
    cifar_resnet_layers = pd.read_csv(Path("results/e11_cifar100_resnet_one_step") / "layer_metrics.csv")
    cifar_resnet_config = json.loads((Path("results/e11_cifar100_resnet_one_step") / "config.json").read_text())
    if len(cifar_resnet_steps) != 20 or len(cifar_resnet_layers) != 420:
        raise AssertionError("CIFAR-100-LT ResNet18 one-step diagnostic must contain 10 seeds x 2 geometries")
    if set(cifar_resnet_steps["geometry"]) != {"frobenius", "spectral"} or cifar_resnet_steps["seed"].nunique() != 10:
        raise AssertionError("CIFAR-100-LT ResNet18 one-step diagnostic must compare Fro/GD and spectral on 10 seeds")
    if set(cifar_resnet_steps["updated_parameter_subset"]) != {"conv_and_linear_weights_only"}:
        raise AssertionError("CIFAR-100-LT ResNet18 diagnostic must update only Conv/Linear matrix weights")
    if (
        cifar_resnet_config["device"] != "cuda"
        or cifar_resnet_config["download"]
        or abs(float(cifar_resnet_config["target_head_gain_fraction"]) - 0.005) > 1e-12
    ):
        raise AssertionError("CIFAR-100-LT ResNet18 diagnostic should be the Slurm/GPU no-download run")
    if len(cifar_resnet_summary) != 1:
        raise AssertionError("CIFAR-100-LT ResNet18 one-step summary must contain one paired-summary row")
    cifar_resnet_row = cifar_resnet_summary.iloc[0]
    if not (
        cifar_resnet_row["tail_output_drift_sq_ratio_ci95_high"] < 1.0
        and cifar_resnet_row["centered_tail_output_drift_sq_ratio_ci95_high"] < 1.0
        and cifar_resnet_row["spectral_less_tail_output_drift_fraction"] == 1.0
        and cifar_resnet_row["tail_loss_increase_diff_ci95_high"] < 0.0
        and cifar_resnet_row["tail_accuracy_drop_diff_ci95_low"] < 0.0
        and cifar_resnet_row["tail_accuracy_drop_diff_ci95_high"] > 0.0
    ):
        raise AssertionError("CIFAR-100-LT ResNet18 diagnostic must preserve lower drift, lower tail-loss increase, and inconclusive accuracy")
    cifar_resnet_rho_steps = pd.read_csv(Path("results/e11_cifar100_resnet_one_step_rho002") / "step_metrics.csv")
    cifar_resnet_rho_summary = pd.read_csv(Path("results/e11_cifar100_resnet_one_step_rho002") / "pair_summary.csv")
    cifar_resnet_rho_layers = pd.read_csv(Path("results/e11_cifar100_resnet_one_step_rho002") / "layer_metrics.csv")
    cifar_resnet_rho_config = json.loads(
        (Path("results/e11_cifar100_resnet_one_step_rho002") / "config.json").read_text()
    )
    if len(cifar_resnet_rho_steps) != 20 or len(cifar_resnet_rho_layers) != 420:
        raise AssertionError("CIFAR-100-LT ResNet18 rho=0.002 diagnostic must contain 10 seeds x 2 geometries")
    if set(cifar_resnet_rho_steps["geometry"]) != {"frobenius", "spectral"} or cifar_resnet_rho_steps["seed"].nunique() != 10:
        raise AssertionError("CIFAR-100-LT ResNet18 rho=0.002 diagnostic must compare Fro/GD and spectral on 10 seeds")
    if (
        cifar_resnet_rho_config["device"] != "cuda"
        or cifar_resnet_rho_config["download"]
        or abs(float(cifar_resnet_rho_config["target_head_gain_fraction"]) - 0.002) > 1e-12
    ):
        raise AssertionError("CIFAR-100-LT ResNet18 rho=0.002 diagnostic should be the Slurm/GPU no-download run")
    if len(cifar_resnet_rho_summary) != 1:
        raise AssertionError("CIFAR-100-LT ResNet18 rho=0.002 summary must contain one paired-summary row")
    cifar_resnet_rho_row = cifar_resnet_rho_summary.iloc[0]
    if not (
        cifar_resnet_rho_row["tail_output_drift_sq_ratio_ci95_high"] < 1.0
        and cifar_resnet_rho_row["spectral_less_tail_output_drift_fraction"] == 1.0
        and cifar_resnet_rho_row["tail_loss_increase_diff_ci95_high"] < 0.0
        and cifar_resnet_rho_row["tail_accuracy_drop_diff_ci95_low"] < 0.0
        and cifar_resnet_rho_row["tail_accuracy_drop_diff_ci95_high"] > 0.0
    ):
        raise AssertionError("CIFAR-100-LT ResNet18 rho=0.002 diagnostic must preserve lower drift and inconclusive accuracy")
    cifar_resnet_checkpoint_steps = pd.read_csv(
        Path("results/e11_cifar100_resnet_checkpoint_sweep") / "step_metrics.csv"
    )
    cifar_resnet_checkpoint_summary = pd.read_csv(
        Path("results/e11_cifar100_resnet_checkpoint_sweep") / "pair_summary.csv"
    )
    cifar_resnet_checkpoint_layers = pd.read_csv(
        Path("results/e11_cifar100_resnet_checkpoint_sweep") / "layer_metrics.csv"
    )
    cifar_resnet_checkpoint_config = json.loads(
        (Path("results/e11_cifar100_resnet_checkpoint_sweep") / "config.json").read_text()
    )
    expected_resnet_checkpoint_steps = {250, 500, 1000, 2000}
    if len(cifar_resnet_checkpoint_steps) != 80 or len(cifar_resnet_checkpoint_layers) != 1680:
        raise AssertionError(
            "CIFAR-100-LT ResNet18 checkpoint sweep must contain 4 checkpoints x 10 seeds x 2 geometries"
        )
    if set(cifar_resnet_checkpoint_summary["warmup_steps"]) != expected_resnet_checkpoint_steps:
        raise AssertionError("CIFAR-100-LT ResNet18 checkpoint sweep must cover warmup steps 250, 500, 1000, 2000")
    if set(cifar_resnet_checkpoint_config["warmup_steps"]) != expected_resnet_checkpoint_steps:
        raise AssertionError("CIFAR-100-LT ResNet18 checkpoint sweep config must record the warmup schedule")
    base_resnet_checkpoint_config = cifar_resnet_checkpoint_config["base_config"]
    if (
        base_resnet_checkpoint_config["device"] != "cuda"
        or base_resnet_checkpoint_config["download"]
        or abs(float(base_resnet_checkpoint_config["target_head_gain_fraction"]) - 0.005) > 1e-12
        or len(base_resnet_checkpoint_config["seeds"]) != 10
    ):
        raise AssertionError("CIFAR-100-LT ResNet18 checkpoint sweep should be the Slurm/GPU no-download run")
    if not (
        (cifar_resnet_checkpoint_summary["seeds"] == 10).all()
        and (cifar_resnet_checkpoint_summary["tail_output_drift_sq_ratio_ci95_high"] < 1.0).all()
        and (cifar_resnet_checkpoint_summary["centered_tail_output_drift_sq_ratio_ci95_high"] < 1.0).all()
        and (cifar_resnet_checkpoint_summary["spectral_less_tail_output_drift_fraction"] == 1.0).all()
    ):
        raise AssertionError("CIFAR-100-LT ResNet18 checkpoint sweep must preserve lower full/centered drift")
    early_resnet_checkpoints = cifar_resnet_checkpoint_summary[
        cifar_resnet_checkpoint_summary["warmup_steps"].isin([250, 500])
    ]
    late_resnet_checkpoints = cifar_resnet_checkpoint_summary[
        cifar_resnet_checkpoint_summary["warmup_steps"].isin([1000, 2000])
    ]
    if not (
        (early_resnet_checkpoints["tail_loss_increase_diff_ci95_low"] > 0.0).all()
        and (late_resnet_checkpoints["tail_loss_increase_diff_ci95_high"] < 0.0).all()
        and cifar_resnet_checkpoint_summary["mean_tail_accuracy_before"].max() < 0.1
    ):
        raise AssertionError(
            "CIFAR-100-LT ResNet18 checkpoint sweep must preserve the tail-loss and weak-tail-predictor caveats"
        )
    cifar_resnet_condition_points = pd.read_csv(
        Path("results/e11_cifar100_resnet_condition_proxy_scatter") / "scatter_points.csv"
    )
    cifar_resnet_condition_summary = pd.read_csv(
        Path("results/e11_cifar100_resnet_condition_proxy_scatter") / "summary.csv"
    )
    cifar_resnet_condition_layers = pd.read_csv(
        Path("results/e11_cifar100_resnet_condition_proxy_scatter") / "layer_summary.csv"
    )
    cifar_resnet_condition_config = json.loads(
        (Path("results/e11_cifar100_resnet_condition_proxy_scatter") / "config.json").read_text()
    )
    if len(cifar_resnet_condition_points) != 40 or len(cifar_resnet_condition_summary) != 6:
        raise AssertionError("CIFAR-100-LT ResNet18 condition-proxy scatter must contain 40 points and 6 correlations")
    if len(cifar_resnet_condition_layers) != 84:
        raise AssertionError("CIFAR-100-LT ResNet18 condition-proxy layer summary must contain 4 checkpoints x 21 layers")
    if set(cifar_resnet_condition_points["warmup_steps"]) != expected_resnet_checkpoint_steps:
        raise AssertionError("CIFAR-100-LT ResNet18 condition-proxy scatter must cover the checkpoint-sweep schedule")
    rank_proxy_pearson = cifar_resnet_condition_summary[
        cifar_resnet_condition_summary["comparison"].eq("mean_gradient_nuclear_rank_vs_log_tail_drift_sq_ratio")
        & cifar_resnet_condition_summary["correlation"].eq("pearson")
    ].iloc[0]
    if not (
        (cifar_resnet_condition_points["tail_output_drift_sq_ratio_spectral_over_fro"] < 1.0).all()
        and rank_proxy_pearson["estimate"] > 0.5
        and rank_proxy_pearson["ci95_low"] > 0.0
        and "downstream-aware tail sensitivity is not measured"
        in cifar_resnet_condition_config["claim_boundary"]
    ):
        raise AssertionError(
            "CIFAR-100-LT ResNet18 condition-proxy scatter must preserve the rank-only caveat"
        )
    cifar_resnet_fc_condition_steps = pd.read_csv(
        Path("results/e11_cifar100_resnet_fc_condition_scatter") / "step_metrics.csv"
    )
    cifar_resnet_fc_condition_summary = pd.read_csv(
        Path("results/e11_cifar100_resnet_fc_condition_scatter") / "summary.csv"
    )
    cifar_resnet_fc_condition_metrics = pd.read_csv(
        Path("results/e11_cifar100_resnet_fc_condition_scatter") / "condition_metrics.csv"
    )
    cifar_resnet_fc_condition_config = json.loads(
        (Path("results/e11_cifar100_resnet_fc_condition_scatter") / "config.json").read_text()
    )
    if len(cifar_resnet_fc_condition_steps) != 80 or len(cifar_resnet_fc_condition_metrics) != 40:
        raise AssertionError(
            "CIFAR-100-LT ResNet18 final-layer condition scatter must contain 4 checkpoints x 10 seeds"
        )
    if set(cifar_resnet_fc_condition_summary["warmup_steps"]) != expected_resnet_checkpoint_steps:
        raise AssertionError(
            "CIFAR-100-LT ResNet18 final-layer condition scatter must cover warmup steps 250, 500, 1000, 2000"
        )
    if set(cifar_resnet_fc_condition_config["warmup_steps"]) != expected_resnet_checkpoint_steps:
        raise AssertionError("CIFAR-100-LT ResNet18 final-layer condition config must record the warmup schedule")
    base_fc_condition_config = cifar_resnet_fc_condition_config["base_config"]
    if (
        base_fc_condition_config["device"] != "cuda"
        or base_fc_condition_config["download"]
        or abs(float(base_fc_condition_config["target_head_gain_fraction"]) - 0.005) > 1e-12
        or len(base_fc_condition_config["seeds"]) != 10
    ):
        raise AssertionError("CIFAR-100-LT ResNet18 final-layer condition scatter should be the Slurm/GPU no-download run")
    if set(cifar_resnet_fc_condition_steps["updated_parameter_subset"]) != {"final_linear_weight_only"}:
        raise AssertionError("CIFAR-100-LT ResNet18 final-layer condition scatter must update only fc.weight")
    if not (
        (cifar_resnet_fc_condition_summary["seeds"] == 10).all()
        and (cifar_resnet_fc_condition_summary["tail_output_drift_sq_ratio_ci95_high"] < 1.0).all()
        and (cifar_resnet_fc_condition_summary["condition_favors_spectral_fraction"] == 1.0).all()
        and (cifar_resnet_fc_condition_metrics["condition_score_nrank_over_tail_srank"] > 1.0).all()
        and (cifar_resnet_fc_condition_metrics["theory_ratio_tail_srank_over_nrank"] < 1.0).all()
        and (cifar_resnet_fc_condition_summary["geomean_tail_output_drift_sq_ratio_spectral_over_fro"] < 0.3).all()
    ):
        raise AssertionError(
            "CIFAR-100-LT ResNet18 final-layer condition scatter must preserve the downstream-aware condition readout"
        )
    cifar_resnet_tail_quality_steps = pd.read_csv(
        Path("results/e11_cifar100_resnet_tail_quality_control") / "step_metrics.csv"
    )
    cifar_resnet_tail_quality_summary = pd.read_csv(
        Path("results/e11_cifar100_resnet_tail_quality_control") / "pair_summary.csv"
    )
    cifar_resnet_tail_quality_layers = pd.read_csv(
        Path("results/e11_cifar100_resnet_tail_quality_control") / "layer_metrics.csv"
    )
    cifar_resnet_tail_quality_config = json.loads(
        (Path("results/e11_cifar100_resnet_tail_quality_control") / "config.json").read_text()
    )
    expected_tail_quality_steps = {2000, 5000, 10000}
    if len(cifar_resnet_tail_quality_steps) != 60 or len(cifar_resnet_tail_quality_layers) != 1260:
        raise AssertionError(
            "CIFAR-100 ResNet18 tail-quality control must contain 3 checkpoints x 10 seeds x 2 geometries"
        )
    if set(cifar_resnet_tail_quality_summary["warmup_steps"]) != expected_tail_quality_steps:
        raise AssertionError("CIFAR-100 ResNet18 tail-quality control must cover warmup steps 2000, 5000, 10000")
    if set(cifar_resnet_tail_quality_config["warmup_steps"]) != expected_tail_quality_steps:
        raise AssertionError("CIFAR-100 ResNet18 tail-quality config must record the warmup schedule")
    base_tail_quality_config = cifar_resnet_tail_quality_config["base_config"]
    if (
        base_tail_quality_config["device"] != "cuda"
        or base_tail_quality_config["download"]
        or abs(float(base_tail_quality_config["target_head_gain_fraction"]) - 0.005) > 1e-12
        or len(base_tail_quality_config["seeds"]) != 10
        or int(base_tail_quality_config["head_train_per_class"]) != 300
        or int(base_tail_quality_config["tail_train_per_class"]) != 300
    ):
        raise AssertionError("CIFAR-100 ResNet18 tail-quality control should be the Slurm/GPU tail-rich no-download run")
    if not (
        (cifar_resnet_tail_quality_summary["seeds"] == 10).all()
        and (cifar_resnet_tail_quality_summary["tail_output_drift_sq_ratio_ci95_high"] < 1.0).all()
        and (cifar_resnet_tail_quality_summary["spectral_less_tail_output_drift_fraction"] == 1.0).all()
        and cifar_resnet_tail_quality_summary["mean_tail_accuracy_before"].max() > 0.3
        and cifar_resnet_tail_quality_summary["mean_tail_accuracy_before"].max()
        > 4.0 * cifar_resnet_checkpoint_summary["mean_tail_accuracy_before"].max()
        and (cifar_resnet_tail_quality_summary["tail_loss_increase_diff_ci95_low"] < 0.0).all()
        and (cifar_resnet_tail_quality_summary["tail_loss_increase_diff_ci95_high"] > 0.0).all()
    ):
        raise AssertionError(
            "CIFAR-100 ResNet18 tail-quality control must preserve lower drift and the non-performance caveat"
        )
    cifar_resnet_layer_jvp_metrics = pd.read_csv(
        Path("results/e11_cifar100_resnet_layer_jvp_tail_quality") / "metrics.csv"
    )
    cifar_resnet_layer_jvp_paired = pd.read_csv(
        Path("results/e11_cifar100_resnet_layer_jvp_tail_quality") / "paired_metrics.csv"
    )
    cifar_resnet_layer_jvp_summary = pd.read_csv(
        Path("results/e11_cifar100_resnet_layer_jvp_tail_quality") / "summary.csv"
    )
    cifar_resnet_layer_jvp_overall = pd.read_csv(
        Path("results/e11_cifar100_resnet_layer_jvp_tail_quality") / "overall_summary.csv"
    )
    cifar_resnet_layer_jvp_config = json.loads(
        (Path("results/e11_cifar100_resnet_layer_jvp_tail_quality") / "config.json").read_text()
    )
    base_layer_jvp_config = cifar_resnet_layer_jvp_config["base_config"]
    if (
        len(cifar_resnet_layer_jvp_metrics) != 420
        or len(cifar_resnet_layer_jvp_paired) != 210
        or len(cifar_resnet_layer_jvp_summary) != 21
        or len(cifar_resnet_layer_jvp_overall) != 1
    ):
        raise AssertionError(
            "CIFAR-100-LT ResNet18 all-layer JVP diagnostic must contain 10 seeds x 21 layers x 2 geometries"
        )
    if not (
        set(cifar_resnet_layer_jvp_metrics["geometry"]) == {"frobenius", "spectral"}
        and cifar_resnet_layer_jvp_metrics["seed"].nunique() == 10
        and cifar_resnet_layer_jvp_metrics["parameter"].nunique() == 21
        and cifar_resnet_layer_jvp_paired["parameter"].nunique() == 21
    ):
        raise AssertionError("CIFAR-100-LT ResNet18 all-layer JVP diagnostic must cover 10 seeds and 21 matrix layers")
    if (
        base_layer_jvp_config["device"] != "cuda"
        or base_layer_jvp_config["download"]
        or abs(float(base_layer_jvp_config["target_head_gain_fraction"]) - 0.005) > 1e-12
        or len(base_layer_jvp_config["seeds"]) != 10
        or int(base_layer_jvp_config["warmup_steps"]) != 5000
        or int(base_layer_jvp_config["head_train_per_class"]) != 300
        or int(base_layer_jvp_config["tail_train_per_class"]) != 300
        or abs(float(cifar_resnet_layer_jvp_config["jvp_epsilon"]) - 1e-4) > 1e-12
    ):
        raise AssertionError("CIFAR-100-LT ResNet18 all-layer JVP diagnostic should be the tail-rich Slurm/GPU no-download run")
    layer_jvp_row = cifar_resnet_layer_jvp_overall.iloc[0]
    if not (
        int(layer_jvp_row["parameters"]) == 21
        and int(layer_jvp_row["paired_points"]) == 210
        and layer_jvp_row["mean_tail_accuracy_before"] > 0.3
        and layer_jvp_row["observed_tail_drift_sq_ratio_ci95_high"] < 1.0
        and layer_jvp_row["scaled_jvp_tail_drift_sq_ratio_ci95_high"] < 1.0
        and layer_jvp_row["spectral_less_observed_tail_drift_fraction"] == 1.0
        and layer_jvp_row["spectral_less_scaled_jvp_tail_drift_fraction"] == 1.0
        and (cifar_resnet_layer_jvp_summary["seeds"] == 10).all()
        and (cifar_resnet_layer_jvp_summary["observed_tail_drift_sq_ratio_ci95_high"] < 1.0).all()
        and (cifar_resnet_layer_jvp_summary["scaled_jvp_tail_drift_sq_ratio_ci95_high"] < 1.0).all()
        and (cifar_resnet_layer_jvp_summary["spectral_less_observed_tail_drift_fraction"] == 1.0).all()
    ):
        raise AssertionError(
            "CIFAR-100-LT ResNet18 all-layer JVP diagnostic must preserve lower scaled-JVP and observed drift"
        )
    checkpoint_prediction_dir = Path("results/e11_cifar100_resnet_layer_jvp_checkpoint_prediction")
    cifar_resnet_layer_jvp_checkpoint_metrics = pd.read_csv(checkpoint_prediction_dir / "metrics.csv")
    cifar_resnet_layer_jvp_checkpoint_paired = pd.read_csv(checkpoint_prediction_dir / "paired_metrics.csv")
    cifar_resnet_layer_jvp_checkpoint_layer_summary = pd.read_csv(checkpoint_prediction_dir / "layer_summary.csv")
    cifar_resnet_layer_jvp_checkpoint_summary = pd.read_csv(checkpoint_prediction_dir / "checkpoint_summary.csv")
    cifar_resnet_layer_jvp_checkpoint_prediction_pairs = pd.read_csv(checkpoint_prediction_dir / "prediction_pairs.csv")
    cifar_resnet_layer_jvp_checkpoint_prediction_summary = pd.read_csv(
        checkpoint_prediction_dir / "prediction_summary.csv"
    )
    cifar_resnet_layer_jvp_checkpoint_config = json.loads(
        (checkpoint_prediction_dir / "config.json").read_text()
    )
    expected_checkpoint_prediction_steps = {2000, 5000, 10000}
    expected_checkpoint_prediction_predictors = {
        "source_scaled_jvp_ratio",
        "source_unit_jvp_ratio",
        "source_alignment_ratio",
        "source_gradient_nuclear_rank",
    }
    if (
        len(cifar_resnet_layer_jvp_checkpoint_metrics) != 1260
        or len(cifar_resnet_layer_jvp_checkpoint_paired) != 630
        or len(cifar_resnet_layer_jvp_checkpoint_layer_summary) != 63
        or len(cifar_resnet_layer_jvp_checkpoint_summary) != 3
        or len(cifar_resnet_layer_jvp_checkpoint_prediction_pairs) != 24
        or len(cifar_resnet_layer_jvp_checkpoint_prediction_summary) != 4
    ):
        raise AssertionError(
            "CIFAR-100-LT ResNet18 JVP checkpoint prediction must contain 3 checkpoints x 10 seeds x 21 layers"
        )
    if not (
        set(cifar_resnet_layer_jvp_checkpoint_metrics["warmup_steps"])
        == expected_checkpoint_prediction_steps
        and set(cifar_resnet_layer_jvp_checkpoint_paired["warmup_steps"])
        == expected_checkpoint_prediction_steps
        and set(cifar_resnet_layer_jvp_checkpoint_summary["warmup_steps"])
        == expected_checkpoint_prediction_steps
        and cifar_resnet_layer_jvp_checkpoint_metrics["seed"].nunique() == 10
        and cifar_resnet_layer_jvp_checkpoint_metrics["parameter"].nunique() == 21
        and cifar_resnet_layer_jvp_checkpoint_layer_summary["parameter"].nunique() == 21
        and set(cifar_resnet_layer_jvp_checkpoint_metrics["geometry"]) == {"frobenius", "spectral"}
        and set(cifar_resnet_layer_jvp_checkpoint_prediction_summary["predictor"])
        == expected_checkpoint_prediction_predictors
    ):
        raise AssertionError(
            "CIFAR-100-LT ResNet18 JVP checkpoint prediction must cover 10 seeds, 21 layers, and four pre-registered predictors"
        )
    base_checkpoint_prediction_config = cifar_resnet_layer_jvp_checkpoint_config["base_config"]
    if (
        base_checkpoint_prediction_config["device"] != "cuda"
        or base_checkpoint_prediction_config["download"]
        or abs(float(base_checkpoint_prediction_config["target_head_gain_fraction"]) - 0.005) > 1e-12
        or len(base_checkpoint_prediction_config["seeds"]) != 10
        or int(base_checkpoint_prediction_config["head_train_per_class"]) != 300
        or int(base_checkpoint_prediction_config["tail_train_per_class"]) != 300
        or set(cifar_resnet_layer_jvp_checkpoint_config["warmup_steps"])
        != expected_checkpoint_prediction_steps
        or abs(float(cifar_resnet_layer_jvp_checkpoint_config["jvp_epsilon"]) - 1e-4) > 1e-12
        or cifar_resnet_layer_jvp_checkpoint_config["max_matrix_parameters"] is not None
    ):
        raise AssertionError(
            "CIFAR-100-LT ResNet18 JVP checkpoint prediction should be the full tail-rich Slurm/GPU no-download run"
        )
    if not (
        (cifar_resnet_layer_jvp_checkpoint_summary["parameters"] == 21).all()
        and (cifar_resnet_layer_jvp_checkpoint_summary["paired_points"] == 210).all()
        and (cifar_resnet_layer_jvp_checkpoint_summary["seeds"] == 10).all()
        and (cifar_resnet_layer_jvp_checkpoint_summary["observed_tail_drift_sq_ratio_ci95_high"] < 1.0).all()
        and (cifar_resnet_layer_jvp_checkpoint_summary["scaled_jvp_tail_drift_sq_ratio_ci95_high"] < 1.0).all()
        and (cifar_resnet_layer_jvp_checkpoint_summary["spectral_less_observed_tail_drift_fraction"] == 1.0).all()
        and (cifar_resnet_layer_jvp_checkpoint_summary["spectral_less_scaled_jvp_tail_drift_fraction"] == 1.0).all()
        and (cifar_resnet_layer_jvp_checkpoint_layer_summary["seeds"] == 10).all()
        and (cifar_resnet_layer_jvp_checkpoint_layer_summary["observed_tail_drift_sq_ratio_ci95_high"] < 1.0).all()
    ):
        raise AssertionError(
            "CIFAR-100-LT ResNet18 JVP checkpoint prediction must preserve lower directional drift across checkpoints"
        )
    checkpoint_prediction_by_predictor = cifar_resnet_layer_jvp_checkpoint_prediction_summary.set_index(
        "predictor"
    )
    scaled_checkpoint_prediction = checkpoint_prediction_by_predictor.loc["source_scaled_jvp_ratio"]
    unit_checkpoint_prediction = checkpoint_prediction_by_predictor.loc["source_unit_jvp_ratio"]
    rank_checkpoint_prediction = checkpoint_prediction_by_predictor.loc["source_gradient_nuclear_rank"]
    if not (
        int(scaled_checkpoint_prediction["checkpoint_transfer_pairs"]) == 6
        and abs(float(scaled_checkpoint_prediction["mean_threshold_below_one_accuracy"]) - 1.0) < 1e-12
        and abs(float(scaled_checkpoint_prediction["mean_top5_risk_overlap_fraction"]) - 0.2) < 1e-12
        and float(scaled_checkpoint_prediction["mean_spearman_log_predictor_vs_log_target_observed"]) < 0.0
        and float(scaled_checkpoint_prediction["spearman_ci95_high"]) < 0.0
        and float(unit_checkpoint_prediction["mean_spearman_log_predictor_vs_log_target_observed"])
        < float(scaled_checkpoint_prediction["mean_spearman_log_predictor_vs_log_target_observed"])
        and float(rank_checkpoint_prediction["mean_spearman_log_predictor_vs_log_target_observed"]) < 0.0
    ):
        raise AssertionError(
            "CIFAR-100-LT ResNet18 JVP checkpoint prediction must preserve the current boundary result: threshold transfers, ranking does not"
        )
    lt_standard_dir = Path("results/e11_cifar100_resnet_lt_standard_eval")
    lt_standard_trace = pd.read_csv(lt_standard_dir / "train_trace.csv")
    lt_standard_class_metrics = pd.read_csv(lt_standard_dir / "class_metrics.csv")
    lt_standard_group_metrics = pd.read_csv(lt_standard_dir / "group_metrics.csv")
    lt_standard_summary = pd.read_csv(lt_standard_dir / "summary.csv")
    lt_standard_class_summary = pd.read_csv(lt_standard_dir / "class_summary.csv")
    lt_standard_config = json.loads((lt_standard_dir / "config.json").read_text())
    if (
        len(lt_standard_trace) != 110
        or len(lt_standard_class_metrics) != 1000
        or len(lt_standard_group_metrics) != 40
        or len(lt_standard_summary) != 4
        or len(lt_standard_class_summary) != 100
    ):
        raise AssertionError(
            "CIFAR-100-LT ResNet18 standard reporting run must contain 10 seeds, 100 classes, and four group summaries"
        )
    if not (
        len(lt_standard_config["seeds"]) == 10
        and int(lt_standard_config["num_classes"]) == 100
        and int(lt_standard_config["max_train_count"]) == 500
        and abs(float(lt_standard_config["imbalance_factor"]) - 100.0) < 1e-12
        and int(lt_standard_config["train_steps"]) == 10000
        and int(lt_standard_config["train_batch_size"]) == 256
        and lt_standard_config["device"] == "cuda"
        and not lt_standard_config["download"]
    ):
        raise AssertionError("CIFAR-100-LT ResNet18 standard reporting run should be the full Slurm/GPU no-download run")
    lt_standard_by_group = lt_standard_summary.set_index("frequency_group")
    if set(lt_standard_by_group.index) != {"many", "medium", "few", "all"}:
        raise AssertionError("CIFAR-100-LT ResNet18 standard reporting summary must cover many, medium, few, and all")
    expected_lt_group_counts = {
        "many": (35, 103, 500),
        "medium": (35, 20, 98),
        "few": (30, 5, 19),
        "all": (100, 5, 500),
    }
    for group, (classes, min_count, max_count) in expected_lt_group_counts.items():
        row = lt_standard_by_group.loc[group]
        if not (
            int(row["classes"]) == classes
            and int(row["min_train_count"]) == min_count
            and int(row["max_train_count"]) == max_count
        ):
            raise AssertionError(f"CIFAR-100-LT ResNet18 standard reporting group metadata mismatch for {group}")
    many_balanced_accuracy = float(lt_standard_by_group.loc["many", "mean_balanced_accuracy"])
    medium_balanced_accuracy = float(lt_standard_by_group.loc["medium", "mean_balanced_accuracy"])
    few_balanced_accuracy = float(lt_standard_by_group.loc["few", "mean_balanced_accuracy"])
    all_balanced_accuracy = float(lt_standard_by_group.loc["all", "mean_balanced_accuracy"])
    if not (
        0.34 <= many_balanced_accuracy <= 0.39
        and 0.09 <= medium_balanced_accuracy <= 0.12
        and 0.009 <= few_balanced_accuracy <= 0.017
        and 0.15 <= all_balanced_accuracy <= 0.18
        and many_balanced_accuracy > medium_balanced_accuracy > few_balanced_accuracy
    ):
        raise AssertionError(
            "CIFAR-100-LT ResNet18 standard reporting balanced accuracy should preserve the current many > medium > few result"
        )
    cifar_resnet_practical_metrics = pd.read_csv(
        Path("results/e11_cifar100_resnet_practical_muon_bridge") / "metrics.csv"
    )
    cifar_resnet_practical_paired = pd.read_csv(
        Path("results/e11_cifar100_resnet_practical_muon_bridge") / "paired_metrics.csv"
    )
    cifar_resnet_practical_summary = pd.read_csv(
        Path("results/e11_cifar100_resnet_practical_muon_bridge") / "summary.csv"
    )
    cifar_resnet_practical_step_summary = pd.read_csv(
        Path("results/e11_cifar100_resnet_practical_muon_bridge") / "step_summary.csv"
    )
    cifar_resnet_practical_config = json.loads(
        (Path("results/e11_cifar100_resnet_practical_muon_bridge") / "config.json").read_text()
    )
    base_resnet_practical_config = cifar_resnet_practical_config["base_config"]
    bridge_resnet_practical_config = cifar_resnet_practical_config["bridge_config"]
    expected_resnet_bridge_sources = {"adamw_matrix_trajectory", "ns_muon_matrix_trajectory"}
    expected_resnet_bridge_directions = {"frobenius_grad", "polar_grad", "polar_momentum", "ns_momentum"}
    if (
        len(cifar_resnet_practical_metrics) != 240
        or len(cifar_resnet_practical_paired) != 180
        or len(cifar_resnet_practical_summary) != 6
        or len(cifar_resnet_practical_step_summary) != 18
    ):
        raise AssertionError(
            "CIFAR-100-LT ResNet18 practical Muon bridge must contain 10 seeds x 2 state sources x 3 trajectory steps x 4 directions"
        )
    if not (
        cifar_resnet_practical_metrics["seed"].nunique() == 10
        and cifar_resnet_practical_paired["seed"].nunique() == 10
        and set(cifar_resnet_practical_metrics["state_source"]) == expected_resnet_bridge_sources
        and set(cifar_resnet_practical_metrics["direction"]) == expected_resnet_bridge_directions
        and set(cifar_resnet_practical_summary["state_source"]) == expected_resnet_bridge_sources
        and set(cifar_resnet_practical_summary["direction"]) == {"polar_grad", "polar_momentum", "ns_momentum"}
        and set(cifar_resnet_practical_metrics["updated_parameter_subset"]) == {"conv_and_linear_weights_only"}
        and cifar_resnet_practical_metrics["matrix_parameter_count"].nunique() == 1
        and int(cifar_resnet_practical_metrics["matrix_parameter_count"].iloc[0]) == 21
    ):
        raise AssertionError(
            "CIFAR-100-LT ResNet18 practical Muon bridge must cover both state sources, four directions, and all 21 matrix weights"
        )
    if (
        base_resnet_practical_config["device"] != "cuda"
        or base_resnet_practical_config["download"]
        or abs(float(base_resnet_practical_config["target_head_gain_fraction"]) - 0.005) > 1e-12
        or len(base_resnet_practical_config["seeds"]) != 10
        or int(base_resnet_practical_config["warmup_steps"]) != 5000
        or int(base_resnet_practical_config["head_train_per_class"]) != 300
        or int(base_resnet_practical_config["tail_train_per_class"]) != 300
        or int(bridge_resnet_practical_config["trajectory_steps"]) != 3
        or int(bridge_resnet_practical_config["newton_schulz_steps"]) != 5
        or bridge_resnet_practical_config["max_matrix_parameters"] is not None
    ):
        raise AssertionError(
            "CIFAR-100-LT ResNet18 practical Muon bridge should be the full tail-rich Slurm/GPU no-download run"
        )
    resnet_practical_by_source_direction = cifar_resnet_practical_summary.set_index(["state_source", "direction"])
    for source in expected_resnet_bridge_sources:
        ns_row = resnet_practical_by_source_direction.loc[(source, "ns_momentum")]
        polar_row = resnet_practical_by_source_direction.loc[(source, "polar_momentum")]
        if not (
            int(ns_row["seeds"]) == 10
            and int(ns_row["trajectory_steps"]) == 3
            and int(ns_row["comparisons"]) == 30
            and ns_row["tail_output_drift_sq_ratio_vs_fro_ci95_high"] < 1.0
            and ns_row["geomean_tail_output_drift_sq_ratio_vs_fro"] < 1.0
            and polar_row["tail_output_drift_sq_ratio_vs_fro_ci95_high"] < 1.0
            and polar_row["geomean_tail_output_drift_sq_ratio_vs_fro"] < ns_row["geomean_tail_output_drift_sq_ratio_vs_fro"]
        ):
            raise AssertionError(
                f"CIFAR-100-LT ResNet18 practical Muon bridge must preserve lower local NS(M_t) drift on {source}"
            )
    imbalance_steps = pd.read_csv(Path("results/e11_long_tail_imbalance_ablation") / "step_metrics.csv")
    imbalance_summary = pd.read_csv(Path("results/e11_long_tail_imbalance_ablation") / "summary.csv")
    if len(imbalance_steps) != 160:
        raise AssertionError(f"long-tail imbalance ablation row count mismatch: expected 160, got {len(imbalance_steps)}")
    if set(imbalance_summary["tail_train_per_class"]) != {10, 20, 40, 80}:
        raise AssertionError("long-tail imbalance ablation must cover tail_train_per_class values 80, 40, 20, and 10")
    if (imbalance_summary["seeds"] != 20).any():
        raise AssertionError("long-tail imbalance ablation must use 20 seeds in each split")
    if not (imbalance_summary["tail_output_drift_sq_ratio_ci95_high"] < 1.0).all():
        raise AssertionError("long-tail imbalance ablation must keep spectral/Fro drift-ratio CI upper endpoints below 1")
    if not (imbalance_summary["spectral_less_tail_output_drift_fraction"] == 1.0).all():
        raise AssertionError("long-tail imbalance ablation must have spectral lower drift in every paired seed")
    checkpoint_steps = pd.read_csv(Path("results/e11_long_tail_checkpoint_sweep") / "step_metrics.csv")
    checkpoint_summary = pd.read_csv(Path("results/e11_long_tail_checkpoint_sweep") / "summary.csv")
    if len(checkpoint_steps) != 200:
        raise AssertionError(f"long-tail checkpoint sweep row count mismatch: expected 200, got {len(checkpoint_steps)}")
    if set(checkpoint_summary["warmup_steps"]) != {20, 40, 80, 120, 160}:
        raise AssertionError("long-tail checkpoint sweep must cover warmup steps 20, 40, 80, 120, and 160")
    if (checkpoint_summary["seeds"] != 20).any():
        raise AssertionError("long-tail checkpoint sweep must use 20 seeds at each checkpoint")
    if not (
        (checkpoint_summary["tail_output_drift_sq_ratio_ci95_high"] < 1.0).all()
        and (checkpoint_summary["centered_tail_output_drift_sq_ratio_ci95_high"] < 1.0).all()
    ):
        raise AssertionError(
            "long-tail checkpoint sweep must keep full and centered spectral/Fro CI upper endpoints below 1"
        )
    if not (checkpoint_summary["margin_delta_sq_ratio_ci95_high"].max() > 1.0):
        raise AssertionError("long-tail checkpoint sweep should preserve the stated caveat that margin-delta is mixed")
    class_partition_steps = pd.read_csv(Path("results/e11_long_tail_class_partition_sweep") / "step_metrics.csv")
    class_partition_summary = pd.read_csv(Path("results/e11_long_tail_class_partition_sweep") / "summary.csv")
    if len(class_partition_steps) != 200:
        raise AssertionError(
            f"long-tail class-partition sweep row count mismatch: expected 200, got {len(class_partition_steps)}"
        )
    expected_partitions = {"low_vs_high", "even_vs_odd", "mixed_a", "mixed_b", "mixed_c"}
    if set(class_partition_summary["partition_name"]) != expected_partitions:
        raise AssertionError("long-tail class-partition sweep must cover the expected five head/tail partitions")
    if (class_partition_summary["seeds"] != 20).any():
        raise AssertionError("long-tail class-partition sweep must use 20 seeds in each partition")
    if not (
        (class_partition_summary["tail_output_drift_sq_ratio_ci95_high"] < 1.0).all()
        and (class_partition_summary["centered_tail_output_drift_sq_ratio_ci95_high"] < 1.0).all()
        and (class_partition_summary["spectral_less_tail_output_drift_fraction"] == 1.0).all()
    ):
        raise AssertionError(
            "long-tail class-partition sweep must keep full and centered spectral/Fro CI upper endpoints below 1"
        )
    rho_steps = pd.read_csv(Path("results/e11_long_tail_rho_sweep") / "step_metrics.csv")
    rho_summary = pd.read_csv(Path("results/e11_long_tail_rho_sweep") / "summary.csv")
    if len(rho_steps) != 200:
        raise AssertionError(f"long-tail rho sweep row count mismatch: expected 200, got {len(rho_steps)}")
    expected_rho_values = {0.005, 0.01, 0.02, 0.04, 0.08}
    rounded_rho_values = {round(float(value), 3) for value in rho_summary["target_head_gain_fraction"]}
    if rounded_rho_values != expected_rho_values:
        raise AssertionError("long-tail rho sweep must cover target head-gain fractions 0.005, 0.01, 0.02, 0.04, and 0.08")
    if (rho_summary["seeds"] != 20).any():
        raise AssertionError("long-tail rho sweep must use 20 seeds at each target head-gain fraction")
    if not (
        (rho_summary["tail_output_drift_sq_ratio_ci95_high"] < 1.0).all()
        and (rho_summary["centered_tail_output_drift_sq_ratio_ci95_high"] < 1.0).all()
        and (rho_summary["spectral_less_tail_output_drift_fraction"] == 1.0).all()
    ):
        raise AssertionError("long-tail rho sweep must keep full and centered spectral/Fro CI upper endpoints below 1")
    rho_head_gain_error_high = rho_summary[
        [
            "actual_head_gain_relative_error_frobenius_ci95_high",
            "actual_head_gain_relative_error_spectral_ci95_high",
        ]
    ].max(axis=1)
    if not (rho_head_gain_error_high.max() < 0.15 and rho_head_gain_error_high.max() > 0.05):
        raise AssertionError(
            "long-tail rho sweep should record bounded but visible finite-step head-gain error at larger rho"
        )
    muon_bridge_steps = pd.read_csv(Path("results/e11_long_tail_muon_bridge") / "step_metrics.csv")
    muon_bridge_summary = pd.read_csv(Path("results/e11_long_tail_muon_bridge") / "pair_summary.csv")
    if len(muon_bridge_steps) != 80:
        raise AssertionError(f"long-tail Muon bridge row count mismatch: expected 80, got {len(muon_bridge_steps)}")
    if set(muon_bridge_steps["direction"]) != {"frobenius_grad", "polar_grad", "polar_momentum", "ns_momentum"}:
        raise AssertionError("long-tail Muon bridge must compare Fro/GD, polar(G_t), polar(M_t), and NS(M_t)")
    if muon_bridge_steps["seed"].nunique() != 20:
        raise AssertionError("long-tail Muon bridge must include 20 seeds")
    muon_bridge_gain_gap = (
        muon_bridge_steps.pivot_table(index="seed", columns="direction", values="matched_first_order_head_gain")
        .sub(muon_bridge_steps.groupby("seed", observed=True)["matched_first_order_head_gain"].first(), axis=0)
        .abs()
        .max()
        .max()
    )
    if float(muon_bridge_gain_gap) > 1e-12:
        raise AssertionError("long-tail Muon bridge head first-order gains must be exactly matched per seed")
    muon_bridge_by_direction = muon_bridge_summary.set_index("direction")
    if set(muon_bridge_by_direction.index) != {"polar_grad", "polar_momentum", "ns_momentum"}:
        raise AssertionError("long-tail Muon bridge summary must contain the three non-Fro bridge directions")
    if not (muon_bridge_by_direction.loc["polar_momentum", "tail_output_drift_sq_ratio_vs_fro_ci95_high"] < 1.0):
        raise AssertionError("long-tail Muon bridge must show CI-bounded lower drift for polar(M_t)")
    if not (muon_bridge_by_direction.loc["ns_momentum", "geomean_tail_output_drift_sq_ratio_vs_fro"] < 1.0):
        raise AssertionError("long-tail Muon bridge NS(M_t) mean drift ratio should remain below Fro/GD")
    if not (
        0.0
        < muon_bridge_by_direction.loc["polar_momentum", "mean_direction_cosine_to_polar_grad"]
        < 1.0
    ):
        raise AssertionError("long-tail Muon bridge must record nontrivial momentum-polar deviation from polar(G_t)")
    required_linearization_columns = {
        "tail_output_jvp_fro",
        "tail_output_linearization_residual_fro",
        "tail_output_linearization_relative_error",
    }
    missing_linearization_columns = required_linearization_columns - set(muon_bridge_steps.columns)
    if missing_linearization_columns:
        raise AssertionError(f"long-tail Muon bridge missing local linearization columns: {sorted(missing_linearization_columns)}")
    local_linearization_summary = pd.read_csv(Path("results/e11_local_linearization") / "summary.csv")
    if set(local_linearization_summary["direction"]) != {"frobenius_grad", "polar_grad", "polar_momentum", "ns_momentum"}:
        raise AssertionError("local linearization summary must cover Fro/GD, polar(G_t), polar(M_t), and NS(M_t)")
    if not (local_linearization_summary["relative_error_ci95_high"] < 0.003).all():
        raise AssertionError("local linearization relative-error CI upper bounds should stay below 0.003")
    practical_bridge_steps = pd.read_csv(Path("results/e11_long_tail_practical_muon_bridge") / "step_metrics.csv")
    practical_bridge_step_summary = pd.read_csv(Path("results/e11_long_tail_practical_muon_bridge") / "step_summary.csv")
    practical_bridge_summary = pd.read_csv(Path("results/e11_long_tail_practical_muon_bridge") / "summary.csv")
    if len(practical_bridge_steps) != 480:
        raise AssertionError(
            f"long-tail practical Muon bridge row count mismatch: expected 480, got {len(practical_bridge_steps)}"
        )
    if len(practical_bridge_step_summary) != 18:
        raise AssertionError(
            "long-tail practical Muon bridge step summary must have 6 steps times 3 bridge directions"
        )
    if set(practical_bridge_steps["direction"]) != {"frobenius_grad", "polar_grad", "polar_momentum", "ns_momentum"}:
        raise AssertionError("long-tail practical Muon bridge must compare Fro/GD, polar(G_t), polar(M_t), and NS(M_t)")
    if practical_bridge_steps["seed"].nunique() != 20 or set(practical_bridge_steps["trajectory_step"]) != set(range(6)):
        raise AssertionError("long-tail practical Muon bridge must include 20 seeds and 6 trajectory steps")
    practical_gain_gap = (
        practical_bridge_steps.pivot_table(
            index=["seed", "trajectory_step"],
            columns="direction",
            values="matched_first_order_head_gain",
        )
        .sub(
            practical_bridge_steps.groupby(["seed", "trajectory_step"], observed=True)["matched_first_order_head_gain"].first(),
            axis=0,
        )
        .abs()
        .max()
        .max()
    )
    if float(practical_gain_gap) > 1e-12:
        raise AssertionError("long-tail practical Muon bridge head first-order gains must be matched per seed and step")
    practical_bridge_by_direction = practical_bridge_summary.set_index("direction")
    if set(practical_bridge_by_direction.index) != {"polar_grad", "polar_momentum", "ns_momentum"}:
        raise AssertionError("long-tail practical Muon bridge summary must contain the three non-Fro directions")
    if not (
        practical_bridge_by_direction.loc["polar_momentum", "tail_output_drift_sq_ratio_vs_fro_ci95_high"] < 1.0
        and practical_bridge_by_direction.loc["ns_momentum", "tail_output_drift_sq_ratio_vs_fro_ci95_high"] < 1.0
    ):
        raise AssertionError("long-tail practical Muon bridge must show CI-bounded lower drift for polar(M_t) and NS(M_t)")
    if int(practical_bridge_by_direction.loc["ns_momentum", "comparisons"]) != 120:
        raise AssertionError("long-tail practical Muon bridge must summarize 120 state-step comparisons")
    state_source_steps = pd.read_csv(
        Path("results/e11_long_tail_muon_state_source_control") / "step_metrics.csv"
    )
    state_source_summary = pd.read_csv(
        Path("results/e11_long_tail_muon_state_source_control") / "summary.csv"
    )
    if len(state_source_steps) != 960:
        raise AssertionError(
            "long-tail Muon state-source control row count mismatch: "
            f"expected 960, got {len(state_source_steps)}"
        )
    expected_state_sources = {"muon_ns_trajectory", "fro_gd_trajectory"}
    if set(state_source_steps["state_source"]) != expected_state_sources:
        raise AssertionError("long-tail Muon state-source control must compare NS-Muon and Fro/GD state sources")
    if set(state_source_steps["direction"]) != {"frobenius_grad", "polar_grad", "polar_momentum", "ns_momentum"}:
        raise AssertionError("long-tail Muon state-source control must compare Fro/GD, polar(G_t), polar(M_t), and NS(M_t)")
    if state_source_steps["seed"].nunique() != 20 or set(state_source_steps["trajectory_step"]) != set(range(6)):
        raise AssertionError("long-tail Muon state-source control must include 20 seeds and 6 trajectory steps")
    if len(state_source_summary) != 6:
        raise AssertionError("long-tail Muon state-source control summary must have 2 state sources times 3 directions")
    state_source_by_key = state_source_summary.set_index(["state_source", "direction"])
    expected_summary_keys = {
        ("muon_ns_trajectory", "polar_grad"),
        ("muon_ns_trajectory", "polar_momentum"),
        ("muon_ns_trajectory", "ns_momentum"),
        ("fro_gd_trajectory", "polar_grad"),
        ("fro_gd_trajectory", "polar_momentum"),
        ("fro_gd_trajectory", "ns_momentum"),
    }
    if set(state_source_by_key.index) != expected_summary_keys:
        raise AssertionError("long-tail Muon state-source control summary keys are incomplete")
    if not (
        state_source_by_key.loc[
            ("fro_gd_trajectory", "polar_momentum"),
            "tail_output_drift_sq_ratio_vs_fro_ci95_high",
        ]
        < 1.0
        and state_source_by_key.loc[
            ("fro_gd_trajectory", "ns_momentum"),
            "tail_output_drift_sq_ratio_vs_fro_ci95_high",
        ]
        < 1.0
    ):
        raise AssertionError(
            "long-tail Muon state-source control must show CI-bounded lower drift on Fro/GD-generated states"
        )
    if int(state_source_by_key.loc[("fro_gd_trajectory", "ns_momentum"), "comparisons"]) != 120:
        raise AssertionError("long-tail Muon state-source control must summarize 120 Fro/GD-state comparisons")
    practical_training_steps = pd.read_csv(Path("results/e11_long_tail_practical_training") / "step_metrics.csv")
    practical_training_summary = pd.read_csv(Path("results/e11_long_tail_practical_training") / "summary.csv")
    if len(practical_training_steps) != 3240:
        raise AssertionError(
            f"long-tail practical training row count mismatch: expected 3240, got {len(practical_training_steps)}"
        )
    if set(practical_training_steps["optimizer"]) != {"adam", "ns_muon"}:
        raise AssertionError("long-tail practical training must compare Adam and NS-Muon-style updates")
    if practical_training_steps["seed"].nunique() != 20 or set(practical_training_steps["step"]) != set(range(81)):
        raise AssertionError("long-tail practical training must include 20 seeds and steps 0..80")
    if len(practical_training_summary) != 1:
        raise AssertionError("long-tail practical training summary must contain one paired-summary row")
    practical_training_row = practical_training_summary.iloc[0]
    if not (
        practical_training_row["final_train_loss_ratio_ci95_high"] < 1.0
        and practical_training_row["final_head_loss_ratio_ci95_high"] < 1.0
        and practical_training_row["final_tail_eval_loss_ratio_ci95_high"] < 1.0
        and practical_training_row["final_tail_eval_drift_rms_ratio_ci95_high"] < 1.0
    ):
        raise AssertionError("long-tail practical training must show CI-bounded lower train/head/tail loss and tail drift")
    if abs(float(practical_training_row["mean_final_tail_eval_accuracy_diff_muon_minus_adam"])) > 1e-12:
        raise AssertionError("long-tail practical training should preserve the current no-tail-accuracy-improvement caveat")
    practical_training_lr_sweep = pd.read_csv(
        Path("results/e11_long_tail_practical_training_lr_sweep") / "sweep_summary.csv"
    )
    expected_muon_lrs = {0.003, 0.01, 0.03, 0.1}
    actual_muon_lrs = {round(float(value), 3) for value in practical_training_lr_sweep["muon_lr"]}
    if actual_muon_lrs != expected_muon_lrs:
        raise AssertionError(f"long-tail practical training LR sweep mismatch: {actual_muon_lrs}")
    if len(practical_training_lr_sweep) != 4:
        raise AssertionError("long-tail practical training LR sweep must summarize four Muon learning rates")
    lr_sweep_by_lr = practical_training_lr_sweep.assign(muon_lr_rounded=practical_training_lr_sweep["muon_lr"].round(3)).set_index("muon_lr_rounded")
    selected_lr = lr_sweep_by_lr.loc[0.03]
    large_lr = lr_sweep_by_lr.loc[0.1]
    small_lr = lr_sweep_by_lr.loc[0.003]
    if not (
        selected_lr["final_train_loss_ratio_ci95_high"] < 1.0
        and selected_lr["final_tail_eval_loss_ratio_ci95_high"] < 1.0
        and selected_lr["final_tail_eval_drift_rms_ratio_ci95_high"] < 1.0
    ):
        raise AssertionError("LR sweep must preserve the selected 0.03 practical-training result")
    if not (
        large_lr["final_tail_eval_loss_ratio_ci95_low"] > 1.0
        and large_lr["final_tail_eval_drift_rms_ratio_ci95_low"] > 1.0
    ):
        raise AssertionError("LR sweep must show that too-large Muon lr worsens tail loss and drift")
    if small_lr["final_train_loss_ratio_ci95_low"] <= 1.0:
        raise AssertionError("LR sweep must show that too-small Muon lr under-trains relative to Adam")
    forgetting_steps = pd.read_csv(Path("results/e11_long_tail_forgetting") / "step_metrics.csv")
    forgetting_summary = pd.read_csv(Path("results/e11_long_tail_forgetting") / "summary.csv")
    if len(forgetting_steps) != 360:
        raise AssertionError(f"long-tail forgetting row count mismatch: expected 360, got {len(forgetting_steps)}")
    if set(forgetting_steps["geometry"]) != {"frobenius", "spectral"}:
        raise AssertionError("long-tail forgetting must compare frobenius and spectral geometries")
    if forgetting_steps["seed"].nunique() != 20:
        raise AssertionError("long-tail forgetting must include 20 seeds")
    if set(forgetting_steps["step"]) != set(range(9)):
        raise AssertionError("long-tail forgetting must include baseline plus 8 head-only steps")
    forgetting_gain_gap = (
        forgetting_steps.pivot_table(
            index=["seed", "step"],
            columns="geometry",
            values="target_first_order_head_gain",
        )
        .diff(axis=1)
        .abs()
        .max()
        .max()
    )
    if float(forgetting_gain_gap) > 1e-12:
        raise AssertionError("long-tail forgetting target head-gain schedule must be matched per seed and step")
    forgetting_row = forgetting_summary.iloc[0]
    if int(forgetting_row["head_only_steps"]) != 8:
        raise AssertionError("long-tail forgetting summary must report 8 head-only steps")
    if not (
        forgetting_row["final_tail_output_drift_sq_ratio_ci95_high"] < 1.0
        and forgetting_row["tail_output_drift_area_ratio_ci95_high"] < 1.0
        and forgetting_row["spectral_less_tail_output_drift_area_fraction"] == 1.0
    ):
        raise AssertionError("long-tail forgetting must preserve lower spectral tail drift over the head-only horizon")
    layerwise_metrics = pd.read_csv(Path("results/e11_long_tail_layerwise") / "metrics.csv")
    layerwise_summary = pd.read_csv(Path("results/e11_long_tail_layerwise") / "summary.csv")
    if len(layerwise_metrics) != 80:
        raise AssertionError(f"long-tail layerwise row count mismatch: expected 80, got {len(layerwise_metrics)}")
    if set(layerwise_metrics["geometry"]) != {"frobenius", "spectral"}:
        raise AssertionError("long-tail layerwise must compare frobenius and spectral geometries")
    if set(layerwise_metrics["layer"]) != {1, 2}:
        raise AssertionError("long-tail layerwise must include both MLP matrix layers")
    if layerwise_metrics["seed"].nunique() != 20:
        raise AssertionError("long-tail layerwise must include 20 seeds")
    if len(layerwise_summary) != 2:
        raise AssertionError("long-tail layerwise summary must have one row per layer")
    required_layerwise_metric_columns = {
        "tail_sandwiched_stable_rank",
        "tail_local_operator_stable_rank",
        "theorem_condition_score",
        "local_operator_condition_score",
    }
    missing_layerwise_metric_columns = required_layerwise_metric_columns - set(layerwise_metrics.columns)
    if missing_layerwise_metric_columns:
        raise AssertionError(f"long-tail layerwise metrics missing downstream-aware rank columns: {missing_layerwise_metric_columns}")
    required_layerwise_summary_columns = {
        "mean_tail_sandwiched_stable_rank",
        "mean_tail_local_operator_stable_rank",
        "mean_theorem_condition_score",
        "mean_local_operator_condition_score",
        "local_operator_score_observed_ratio_spearman_all_layers",
    }
    missing_layerwise_summary_columns = required_layerwise_summary_columns - set(layerwise_summary.columns)
    if missing_layerwise_summary_columns:
        raise AssertionError(f"long-tail layerwise summary missing downstream-aware rank columns: {missing_layerwise_summary_columns}")
    layerwise_by_layer = layerwise_summary.set_index("layer")
    if not math.isnan(float(layerwise_by_layer.loc[1, "mean_tail_sandwiched_stable_rank"])):
        raise AssertionError("long-tail layer 1 should not report a single-sandwich ssrank under sample-dependent ReLU gates")
    if not (
        math.isfinite(float(layerwise_by_layer.loc[2, "mean_tail_sandwiched_stable_rank"]))
        and float(layerwise_by_layer.loc[2, "mean_tail_sandwiched_stable_rank"]) > 0.0
        and math.isfinite(float(layerwise_by_layer.loc[2, "mean_theorem_condition_score"]))
    ):
        raise AssertionError("long-tail layer 2 must report a finite exact sandwich ssrank and theorem condition score")
    if not (layerwise_summary["mean_tail_local_operator_stable_rank"] > 0.0).all():
        raise AssertionError("long-tail layerwise must report positive frozen-gate local operator stable ranks")
    if not (layerwise_summary["jvp_tail_drift_sq_ratio_ci95_low"] > 1.0).all():
        raise AssertionError("long-tail layerwise unit-JVP ratio should preserve the sensitivity caveat")
    if not (
        (layerwise_summary["scaled_jvp_tail_drift_sq_ratio_ci95_high"] < 1.0).all()
        and (layerwise_summary["observed_tail_drift_sq_ratio_ci95_high"] < 1.0).all()
        and (layerwise_summary["spectral_less_observed_tail_drift_fraction"] == 1.0).all()
    ):
        raise AssertionError("long-tail layerwise scaled JVP and observed drift must support lower spectral tail drift")
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
    required_skeleton_phrases = [
        "Head-to-Tail Interference in Long-Tailed Small-Batch Training",
        "nrank(G_H) > ssrank(B_T,A_T)",
        "head-to-tail function drift",
        "synthetic boundary",
        "Muon-style compatibility",
        "trajectory Muon-style compatibility",
        "practical training diagnostic",
        "8-step forgetting",
        "layerwise JVP",
    ]
    missing_skeleton_phrases = [phrase for phrase in required_skeleton_phrases if phrase not in paper_skeleton]
    if missing_skeleton_phrases:
        raise AssertionError(f"paper skeleton missing current head-to-tail framing: {missing_skeleton_phrases}")
    main_package = Path("discussion/e11_main_paper_package.md").read_text(encoding="utf-8")
    required_main_package_phrases = [
        "Main Figure/Table Package",
        "Appendix Allocation",
        "Claims To Exclude From Main Text",
        "figures/e11_head_tail_interference/head_tail_drift_ratio.png",
        "figures/e11_long_tail_one_step/long_tail_one_step_tail_response.png",
        "figures/e11_long_tail_muon_bridge/long_tail_muon_bridge.png",
        "figures/e11_long_tail_practical_muon_bridge/long_tail_practical_muon_bridge.png",
        "figures/e11_long_tail_muon_state_source_control/long_tail_muon_state_source_control.png",
        "figures/e11_long_tail_practical_training/long_tail_practical_training.png",
        "figures/e11_long_tail_forgetting/long_tail_head_only_forgetting.png",
        "figures/e11_long_tail_layerwise/long_tail_layerwise_drift.png",
        "figures/e11_cifar100_resnet_layer_jvp_tail_quality/cifar100_resnet_layer_jvp_tail_quality.png",
        "figures/e11_cifar100_resnet_layer_jvp_checkpoint_prediction/cifar100_resnet_layer_jvp_checkpoint_prediction.png",
        "figures/e11_cifar100_resnet_lt_standard_eval/cifar100_resnet_lt_standard_eval.png",
        "figures/e11_cifar100_resnet_practical_muon_bridge/cifar100_resnet_practical_muon_bridge.png",
        "seven figures plus one generated table",
        "paper/specgrad_activation_paper/tables/head_tail_empirical_results.tex",
        "older condition-geometry artifacts",
        "discussion/e11_pasted_review_audit.md",
        "discussion/e11_completion_audit.md",
        "discussion/e11_end_of_draft_self_review.md",
        "discussion/e11_long_tail_practical_training_lr_sweep.md",
        "legacy optimizer-switch and broad LR-sweep figures",
        "This does not exclude the practical-training LR sensitivity note",
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
        "Figure 4",
        "Figure 5",
        "Figure 6",
        "Figure 7",
        "figures/e11_long_tail_practical_training/long_tail_practical_training.png",
        "Caption Discipline",
        "tail-example logit drift",
        "tail loss, margin, accuracy",
        "nrank(G_H)",
        "ssrank(B_T,A_T)",
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
        "head-to-tail function drift at matched head gain",
        "Legacy condition-geometry guardrail and Muon-style compatibility background",
    ]
    missing_notation = [phrase for phrase in required_notation_phrases if phrase not in notation_glossary]
    if missing_notation:
        raise AssertionError(f"notation glossary missing required content: {missing_notation}")
    claim_ledger = Path("discussion/e11_quantitative_claim_ledger.md").read_text(encoding="utf-8")
    required_claim_ledger_phrases = [
        "Claim Ledger",
        "Writing Priority",
        "C1 -> C2 -> C6 -> C7 -> C4 -> C5",
        "head-to-tail interference paper",
        "Drift-squared ratio=0.5501",
        "polar(M_t) squared drift ratio vs Fro/GD=0.8199",
        "short-trajectory NS(M_t) squared drift ratio=0.8019",
        "final train loss ratio Muon/Adam=0.6468",
        "tail eval loss ratio=0.8549",
        "tail margin diff=1.955",
        "tail drift RMS ratio=0.7501",
        "head-alignment ratio=2.083",
        "Frobenius-norm ratio=3.112",
        "operator-norm ratio=0.5543",
        "Final squared drift ratio=0.6167",
        "Layer 1 unit/scaled/observed squared drift ratios=1.45/0.4766/0.4767",
        "smaller operator-norm step despite a larger Frobenius-norm step",
        "complete practical Muon training behavior",
    ]
    missing_claim_ledger = [phrase for phrase in required_claim_ledger_phrases if phrase not in claim_ledger]
    if missing_claim_ledger:
        raise AssertionError(f"quantitative claim ledger missing required content: {missing_claim_ledger}")
    paper_numbers = Path("paper/specgrad_activation_paper/tables/e11_paper_numbers.tex").read_text(encoding="utf-8")
    discussion_paper_numbers = Path("discussion/e11_paper_numbers.tex").read_text(encoding="utf-8")
    if paper_numbers != discussion_paper_numbers:
        raise AssertionError("paper-local and discussion paper-number macro files must match exactly")
    required_number_macros = [
        "\\EelevenHeadTailPositiveNrankG",
        "\\EelevenHeadTailPositiveSsrankBTA",
        "\\EelevenHeadTailPositiveDriftRatio",
        "\\EelevenHeadTailNegativeDriftRatio",
        "\\EelevenHeadTailAlignmentAblationSeeds",
        "\\EelevenHeadTailAlignmentPositiveTheoryRatio",
        "\\EelevenHeadTailAlignmentPositiveDriftRatio",
        "\\EelevenHeadTailAlignmentPositiveMedianDriftRatio",
        "\\EelevenHeadTailAlignmentPositiveSpectralLowerFraction",
        "\\EelevenLongTailOneStepSeeds",
        "\\EelevenLongTailOneStepDriftRatio",
        "\\EelevenLongTailOneStepCenteredDriftRatio",
        "\\EelevenLongTailOneStepTrueLogitDriftRatio",
        "\\EelevenLongTailOneStepCompetitorLogitDriftRatio",
        "\\EelevenLongTailOneStepMarginDeltaRatio",
        "\\EelevenLongTailOneStepTailLossDiff",
        "\\EelevenLongTailOneStepTailLossIncreaseFro",
        "\\EelevenLongTailOneStepTailLossIncreaseSpectral",
        "\\EelevenLongTailOneStepTailMarginDropDiff",
        "\\EelevenLongTailOneStepTailMarginDropFro",
        "\\EelevenLongTailOneStepTailMarginDropSpectral",
        "\\EelevenLongTailOneStepTailAccuracyDropFro",
        "\\EelevenLongTailOneStepTailAccuracyDropSpectral",
        "\\EelevenLongTailOneStepHeadGainErrorFro",
        "\\EelevenLongTailOneStepHeadGainErrorSpectral",
        "\\EelevenLongTailOneStepTailLossBefore",
        "\\EelevenLongTailOneStepTailLossAfterFro",
        "\\EelevenLongTailOneStepTailLossAfterSpectral",
        "\\EelevenLongTailOneStepTailMarginBefore",
        "\\EelevenLongTailOneStepTailMarginAfterFro",
        "\\EelevenLongTailOneStepTailMarginAfterSpectral",
        "\\EelevenLongTailOneStepTailAccuracyBefore",
        "\\EelevenLongTailOneStepTailAccuracyAfterFro",
        "\\EelevenLongTailOneStepTailAccuracyAfterSpectral",
        "\\EelevenLongTailOneStepPositiveMarginFraction",
        "\\EelevenLongTailOneStepCertifiedFractionFro",
        "\\EelevenLongTailOneStepCertifiedFractionSpectral",
        "\\EelevenLongTailOneStepPredictionChangedFro",
        "\\EelevenLongTailOneStepPredictionChangedSpectral",
        "\\EelevenLongTailOneStepPositivePredictionChangedFro",
        "\\EelevenLongTailOneStepPositivePredictionChangedSpectral",
        "\\EelevenCifarResNetOneStepSeeds",
        "\\EelevenCifarResNetOneStepDriftRatio",
        "\\EelevenCifarResNetOneStepCenteredDriftRatio",
        "\\EelevenCifarResNetOneStepMarginDeltaRatio",
        "\\EelevenCifarResNetOneStepTailLossDiff",
        "\\EelevenCifarResNetOneStepTailMarginDropDiff",
        "\\EelevenCifarResNetOneStepTailAccuracyDropDiff",
        "\\EelevenCifarResNetOneStepTailAccuracyBefore",
        "\\EelevenCifarResNetOneStepMeanNrG",
        "\\EelevenCifarResNetRho002DriftRatio",
        "\\EelevenCifarResNetRho002TailLossDiff",
        "\\EelevenCifarResNetCheckpointSweepSettings",
        "\\EelevenCifarResNetCheckpointSweepWorstWarmupSteps",
        "\\EelevenCifarResNetCheckpointSweepWorstDriftRatio",
        "\\EelevenCifarResNetCheckpointSweepBestTailAccuracy",
        "\\EelevenCifarResNetCheckpointSweepTailAccuracyRange",
        "\\EelevenCifarResNetCheckpointSweepPositiveMarginRange",
        "\\EelevenCifarResNetConditionProxyPoints",
        "\\EelevenCifarResNetConditionProxyRankPearson",
        "\\EelevenCifarResNetConditionProxyRankSpearman",
        "\\EelevenCifarResNetConditionProxyTailAccuracySpearman",
        "\\EelevenCifarResNetFcConditionSettings",
        "\\EelevenCifarResNetFcConditionPoints",
        "\\EelevenCifarResNetFcConditionWorstWarmupSteps",
        "\\EelevenCifarResNetFcConditionWorstDriftRatio",
        "\\EelevenCifarResNetFcConditionWeakestMeanConditionScore",
        "\\EelevenCifarResNetFcConditionWeakestPointConditionScore",
        "\\EelevenCifarResNetFcConditionWorstTheoryRatio",
        "\\EelevenCifarResNetFcConditionFavorsSpectralFraction",
        "\\EelevenCifarResNetTailQualitySettings",
        "\\EelevenCifarResNetTailQualityWorstWarmupSteps",
        "\\EelevenCifarResNetTailQualityWorstDriftRatio",
        "\\EelevenCifarResNetTailQualityBestTailAccuracy",
        "\\EelevenCifarResNetTailQualityTailAccuracyRange",
        "\\EelevenCifarResNetLayerJvpSeeds",
        "\\EelevenCifarResNetLayerJvpParameters",
        "\\EelevenCifarResNetLayerJvpPairedPoints",
        "\\EelevenCifarResNetLayerJvpScaledRatio",
        "\\EelevenCifarResNetLayerJvpObservedRatio",
        "\\EelevenCifarResNetLayerJvpSupportedLayers",
        "\\EelevenCifarResNetLayerJvpWorstObservedRatio",
        "\\EelevenCifarResNetLayerJvpWorstScaledRatio",
        "\\EelevenCifarResNetPracticalMuonBridgeStateSources",
        "\\EelevenCifarResNetPracticalMuonBridgeComparisonsPerDirection",
        "\\EelevenCifarResNetPracticalAdamStatePolarMomentumDriftRatio",
        "\\EelevenCifarResNetPracticalAdamStateNsMomentumDriftRatio",
        "\\EelevenCifarResNetPracticalAdamStateNsMomentumCosine",
        "\\EelevenCifarResNetPracticalMuonStatePolarMomentumDriftRatio",
        "\\EelevenCifarResNetPracticalMuonStateNsMomentumDriftRatio",
        "\\EelevenCifarResNetPracticalMuonStateNsMomentumCosine",
        "\\EelevenLocalLinearizationMaxRelativeErrorCiHigh",
        "\\EelevenLocalLinearizationMaxRelativeErrorDirection",
        "\\EelevenLongTailImbalanceAblationSettings",
        "\\EelevenLongTailImbalanceWorstTailTrainPerClass",
        "\\EelevenLongTailImbalanceWorstRatio",
        "\\EelevenLongTailImbalanceDefaultRatio",
        "\\EelevenLongTailImbalanceStrongRatio",
        "\\EelevenLongTailImbalanceStrongPositiveMarginFraction",
        "\\EelevenLongTailCheckpointSweepSettings",
        "\\EelevenLongTailCheckpointSweepWorstWarmupSteps",
        "\\EelevenLongTailCheckpointSweepWorstRatio",
        "\\EelevenLongTailCheckpointSweepWorstMarginWarmupSteps",
        "\\EelevenLongTailCheckpointSweepWorstMarginRatio",
        "\\EelevenLongTailClassPartitionSweepSettings",
        "\\EelevenLongTailClassPartitionWorstPartition",
        "\\EelevenLongTailClassPartitionWorstRatio",
        "\\EelevenLongTailClassPartitionCenteredWorstPartition",
        "\\EelevenLongTailClassPartitionCenteredWorstRatio",
        "\\EelevenLongTailRhoSweepSettings",
        "\\EelevenLongTailRhoSweepWorstFraction",
        "\\EelevenLongTailRhoSweepWorstRatio",
        "\\EelevenLongTailRhoSweepCenteredWorstFraction",
        "\\EelevenLongTailRhoSweepCenteredWorstRatio",
        "\\EelevenLongTailRhoSweepWorstHeadGainErrorFraction",
        "\\EelevenLongTailRhoSweepWorstHeadGainErrorCiHigh",
        "\\EelevenLongTailMuonBridgePolarMomentumDriftRatio",
        "\\EelevenLongTailMuonBridgeNsMomentumDriftRatio",
        "\\EelevenLongTailMuonBridgePolarMomentumCosine",
        "\\EelevenLongTailPracticalMuonBridgeComparisons",
        "\\EelevenLongTailPracticalMuonBridgePolarMomentumDriftRatio",
        "\\EelevenLongTailPracticalMuonBridgeNsMomentumDriftRatio",
        "\\EelevenLongTailPracticalMuonBridgeMomentumCosine",
        "\\EelevenLongTailMuonStateSourceControlComparisons",
        "\\EelevenLongTailMuonStateSourceControlFroPolarMomentumDriftRatio",
        "\\EelevenLongTailMuonStateSourceControlFroNsMomentumDriftRatio",
        "\\EelevenLongTailMuonStateSourceControlMuonNsMomentumDriftRatio",
        "\\EelevenLongTailMuonStateSourceControlFroNsLowerFraction",
        "\\EelevenLongTailPracticalTrainingSteps",
        "\\EelevenLongTailPracticalTrainingAdamLr",
        "\\EelevenLongTailPracticalTrainingMuonLr",
        "\\EelevenLongTailPracticalTrainingTrainLossRatio",
        "\\EelevenLongTailPracticalTrainingHeadLossRatio",
        "\\EelevenLongTailPracticalTrainingTailLossRatio",
        "\\EelevenLongTailPracticalTrainingTailDriftRmsRatio",
        "\\EelevenLongTailPracticalTrainingTailMarginDiff",
        "\\EelevenLongTailPracticalTrainingTailAccuracyDiff",
        "\\EelevenLongTailPracticalTrainingLrSweepSmallTrainLossRatio",
        "\\EelevenLongTailPracticalTrainingLrSweepMediumTrainLossRatio",
        "\\EelevenLongTailPracticalTrainingLrSweepLargeTailLossRatio",
        "\\EelevenLongTailPracticalTrainingLrSweepLargeTailDriftRmsRatio",
        "\\EelevenLongTailForgettingSteps",
        "\\EelevenLongTailForgettingFinalDriftRatio",
        "\\EelevenLongTailForgettingAreaDriftRatio",
        "\\EelevenLongTailForgettingFinalTailLossDiff",
        "\\EelevenLongTailForgettingFinalTailMarginDropDiff",
        "\\EelevenLayerOneUnitJvpDriftRatio",
        "\\EelevenLayerOneScaledJvpDriftRatio",
        "\\EelevenLayerOneObservedDriftRatio",
        "\\EelevenLayerTwoUnitJvpDriftRatio",
        "\\EelevenLayerTwoScaledJvpDriftRatio",
        "\\EelevenLayerTwoObservedDriftRatio",
    ]
    missing_number_macros = [macro for macro in required_number_macros if macro not in paper_numbers]
    if missing_number_macros:
        raise AssertionError(f"paper number macros missing required content: {missing_number_macros}")
    for range_macro in [
        "EelevenLongTailCheckpointSweepTailAccuracyRange",
        "EelevenLongTailCheckpointSweepPositiveMarginRange",
        "EelevenCifarResNetTailQualityTailAccuracyRange",
    ]:
        range_match = re.search(rf"\\newcommand\{{\\{range_macro}\}}\{{(?P<value>[^}}]*(?:\}}[^}}]*)?)\}}", paper_numbers)
        if range_match is None:
            raise AssertionError(f"paper number macro {range_macro} is missing")
        range_value = range_match.group("value")
        if "--" in range_value or r"\text{ to }" not in range_value:
            raise AssertionError(
                f"paper number macro {range_macro} should use math-safe '\\text{{ to }}' range text, "
                f"not a double-minus range: {range_value}"
            )
    paper_tex = Path("paper/specgrad_activation_paper/main.tex").read_text(encoding="utf-8")
    if "Tail CE before; Fro after; spectral after" in paper_tex:
        raise AssertionError("one-step tail outcome table should report small effects as deltas, not rounded after-values")
    required_paper_macro_phrases = [
        r"\input{tables/e11_paper_numbers.tex}",
        r"\EelevenLongTailOneStepDriftRatio",
        r"\EelevenHeadTailAlignmentPositiveDriftRatio",
        r"\EelevenLongTailOneStepTailMarginDropDiff",
        r"\EelevenCifarResNetOneStepDriftRatio",
        r"\EelevenCifarResNetOneStepTailLossDiff",
        r"\EelevenCifarResNetOneStepTailAccuracyDropDiff",
        r"\EelevenCifarResNetRho002DriftRatio",
        r"\EelevenCifarResNetRho002TailLossDiff",
        r"\EelevenCifarResNetCheckpointSweepWorstDriftRatio",
        r"\EelevenCifarResNetCheckpointSweepTailAccuracyRange",
        r"\EelevenCifarResNetConditionProxyRankPearson",
        r"\EelevenCifarResNetFcConditionWorstDriftRatio",
        r"\EelevenCifarResNetFcConditionWeakestMeanConditionScore",
        r"\EelevenCifarResNetFcConditionFavorsSpectralFraction",
        r"\EelevenCifarResNetTailQualityWorstDriftRatio",
        r"\EelevenCifarResNetTailQualityBestTailAccuracy",
        r"\EelevenCifarResNetLayerJvpObservedRatio",
        r"\EelevenCifarResNetLayerJvpScaledRatio",
        r"\EelevenCifarResNetPracticalAdamStateNsMomentumDriftRatio",
        r"\EelevenCifarResNetPracticalMuonStateNsMomentumDriftRatio",
        r"\EelevenLocalLinearizationMaxRelativeErrorCiHigh",
        r"\EelevenLocalLinearizationMaxRelativeErrorDirection",
        r"\EelevenLongTailImbalanceWorstRatioCiHigh",
        r"\EelevenLongTailCheckpointSweepWorstRatioCiHigh",
        r"\EelevenLongTailCheckpointSweepTailAccuracyRange",
        r"\EelevenLongTailCheckpointSweepPositiveMarginRange",
        r"\EelevenLongTailClassPartitionWorstRatioCiHigh",
        r"\EelevenLongTailRhoSweepWorstRatioCiHigh",
        r"\EelevenLongTailRhoSweepWorstHeadGainErrorCiHigh",
        r"\EelevenLongTailOneStepAlignmentRatio",
        r"\EelevenLongTailOneStepUpdateFroNormRatio",
        r"\EelevenLongTailOneStepUpdateOpNormRatio",
        r"\EelevenLongTailMuonStateSourceControlFroNsMomentumDriftRatio",
        r"\EelevenLongTailPracticalTrainingTrainLossRatio",
        r"\EelevenLongTailPracticalTrainingTailMarginDiff",
        r"\EelevenLongTailPracticalTrainingLrSweepLargeTailLossRatio",
        r"\EelevenLongTailForgettingFinalDriftRatio",
        r"\EelevenLongTailForgettingFinalTailMarginDropDiff",
        r"\EelevenLayerOneUnitJvpDriftRatio",
        r"\EelevenLayerTwoObservedDriftRatio",
    ]
    missing_paper_macro_phrases = [phrase for phrase in required_paper_macro_phrases if phrase not in paper_tex]
    if missing_paper_macro_phrases:
        raise AssertionError(f"paper main tex is not using generated paper-number macros: {missing_paper_macro_phrases}")
    forbidden_paper_tex_phrases = [
        "explain why some optimizers have better tail accuracy at the same head accuracy",
        "2.05\\times 10^{-3}",
    ]
    assert_forbidden_phrases_absent(
        "paper main tex over-strong tail-accuracy wording",
        paper_tex,
        forbidden_paper_tex_phrases,
    )
    reproduction_checklist = Path("discussion/e11_reproduction_checklist.md").read_text(encoding="utf-8")
    required_reproduction_phrases = [
        "make e11-main-results",
        "make e11-cifar-resnet-results",
        "make e11-cifar-resnet-rho002-results",
        "make e11-cifar-resnet-checkpoint-sweep-results",
        "make e11-cifar-resnet-condition-proxy-results",
        "make e11-cifar-resnet-fc-condition-results",
        "make e11-cifar-resnet-tail-quality-results",
        "make e11-cifar-resnet-layer-jvp-tail-quality-results",
        "make e11-cifar-resnet-practical-muon-bridge-results",
        "make e11-appendix-results",
        "make e11-all-results",
        "make e11-paper-assets",
        "make e11-guardrail-assets",
        "make e11-all-assets",
        "make e11-paper-pdf",
        "legacy condition-geometry guardrail notes",
        "current paper assets plus legacy guardrail notes",
        "Current Head-to-Tail Paper Evidence",
        "Background / Legacy E11 Evidence",
        "not the main evidence table for the current paper draft",
        "Head-to-tail interference probe",
        "Long-tailed one-step diagnostic",
        "CIFAR-100-LT ResNet18 one-step diagnostic",
        "CIFAR-100-LT ResNet18 smaller-head-gain check",
        "CIFAR-100-LT ResNet18 checkpoint-quality sweep",
        "CIFAR-100-LT ResNet18 condition-proxy scatter",
        "CIFAR-100-LT ResNet18 final-layer condition scatter",
        "CIFAR-100 ResNet18 tail-quality control",
        "CIFAR-100-LT ResNet18 all-layer JVP tail-quality diagnostic",
        "CIFAR-100-LT ResNet18 practical Muon trajectory bridge",
        "Long-tailed Muon-style compatibility diagnostic",
        "Long-tailed practical-Muon trajectory compatibility",
        "Long-tailed practical training diagnostic",
        "Long-tailed practical training LR sensitivity",
        "Head-only forgetting probe",
        "Appendix / Guardrail Evidence",
        "Generated Paper-Facing Assets",
        "make e11-full",
        "Legacy guardrail notes are intentionally not part of `e11-full`",
        "Batch / Activation Contract",
        "`diagnostic_A_definition == full_layer_input_activation`",
        "same-batch pre/post-update",
    ]
    missing_reproduction = [phrase for phrase in required_reproduction_phrases if phrase not in reproduction_checklist]
    if missing_reproduction:
        raise AssertionError(f"reproduction checklist missing required content: {missing_reproduction}")
    current_reproduction_section = reproduction_checklist.split("## Background / Legacy E11 Evidence", 1)[0]
    required_current_reproduction_phrases = [
        "CIFAR-100-LT ResNet18 one-step diagnostic",
        "CIFAR-100-LT ResNet18 smaller-head-gain check",
        "CIFAR-100-LT ResNet18 checkpoint-quality sweep",
        "CIFAR-100-LT ResNet18 condition-proxy scatter",
        "CIFAR-100-LT ResNet18 final-layer condition scatter",
        "CIFAR-100 ResNet18 tail-quality control",
        "CIFAR-100-LT ResNet18 all-layer JVP tail-quality diagnostic",
        "CIFAR-100-LT ResNet18 practical Muon trajectory bridge",
        "Long-tailed practical training diagnostic",
        "Long-tailed practical training LR sensitivity",
    ]
    missing_current_reproduction = [
        phrase for phrase in required_current_reproduction_phrases if phrase not in current_reproduction_section
    ]
    if missing_current_reproduction:
        raise AssertionError(
            "reproduction checklist misclassifies current paper evidence as background: "
            f"{missing_current_reproduction}"
        )
    reviewer_risk = Path("discussion/e11_reviewer_risk_audit.md").read_text(encoding="utf-8")
    required_reviewer_risk_phrases = [
        "Risk Table",
        "Claim Decisions",
        "current head-to-tail interference paper",
        "lower tail-example logit drift",
        "nrank-vs-ssrank condition",
        "final-layer condition scatter",
        "tail-rich ResNet control",
        "all-layer JVP",
        "Treat Muon as motivation",
        "legacy update-spectrum artifacts",
    ]
    missing_reviewer_risk = [phrase for phrase in required_reviewer_risk_phrases if phrase not in reviewer_risk]
    if missing_reviewer_risk:
        raise AssertionError(f"reviewer risk audit missing required content: {missing_reviewer_risk}")
    pasted_review_audit = Path("discussion/e11_pasted_review_audit.md").read_text(encoding="utf-8")
    required_pasted_review_phrases = [
        "E11 Pasted Review Audit",
        "arbitrary-direction matched-gain definition",
        "two-layer MLP blockwise spectral construction",
        "local first-order approximation",
        "Trajectory CIs",
        "Synthetic point CIs",
        "central condition should be written as a direct bound ratio",
        "Layerwise diagnostics should use downstream-aware quantities",
        "Full-logit drift may not be margin-relevant drift",
        "The one-step tail checkpoint may be too weak",
        "protection of a high-quality tail predictor",
        "| 18 |",
        "Squared drift ratio and RMS drift ratio must not be mixed",
        "The supported result is a local, matched-head-gain function-drift mechanism",
        "| 15 |",
    ]
    missing_pasted_review = [
        phrase for phrase in required_pasted_review_phrases if phrase not in pasted_review_audit
    ]
    if missing_pasted_review:
        raise AssertionError(f"pasted review audit missing required content: {missing_pasted_review}")
    completion_audit = Path("discussion/e11_completion_audit.md").read_text(encoding="utf-8")
    required_completion_audit_phrases = [
        "E11 Completion Audit",
        "Requirement Status",
        "Current Residual Risks",
        "Latest Validation Command",
        "Main paper uses a normal ICLR-style structure",
        "Two-page final-report version exists and is exactly two pages",
        "Major claims are tied to quantitative evidence",
        "PDF/build/test gates pass",
    ]
    missing_completion_audit = [
        phrase for phrase in required_completion_audit_phrases if phrase not in completion_audit
    ]
    if missing_completion_audit:
        raise AssertionError(f"completion audit missing required content: {missing_completion_audit}")
    end_self_review = Path("discussion/e11_end_of_draft_self_review.md").read_text(encoding="utf-8")
    required_end_self_review_phrases = [
        "E11 End-of-Draft Self-Review",
        "Five-Dimension Review",
        "Claim-Evidence Map",
        "Required Manuscript Discipline",
        "Current Residual Risks",
        "pass for mechanism paper",
        "not supported",
        "selected-state compatibility",
        "standard long-tail tasks",
    ]
    missing_end_self_review = [
        phrase for phrase in required_end_self_review_phrases if phrase not in end_self_review
    ]
    if missing_end_self_review:
        raise AssertionError(f"end-of-draft self-review missing required content: {missing_end_self_review}")
    reference_audit = Path("discussion/e11_reference_audit.md").read_text(encoding="utf-8")
    required_reference_audit_phrases = [
        "E11 Reference Audit",
        "davis2025spectral",
        "chen2025muon",
        "promo2026",
        "pedregosa2011scikit",
        "cui2019classbalanced",
        "kang2020decoupling",
        "preprint/submission caveat",
        "There are no unused BibTeX entries",
        "newer Muon/spectral papers should be used for positioning",
    ]
    missing_reference_audit = [
        phrase for phrase in required_reference_audit_phrases if phrase not in reference_audit
    ]
    if missing_reference_audit:
        raise AssertionError(f"reference audit missing required content: {missing_reference_audit}")
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
    required_evidence_index_phrases = [
        "current head-to-tail paper claim",
        "figures/e11_head_tail_interference/head_tail_drift_ratio.png",
        "figures/e11_long_tail_one_step/long_tail_one_step_tail_response.png",
        "figures/e11_long_tail_muon_bridge/long_tail_muon_bridge.png",
        "figures/e11_long_tail_practical_muon_bridge/long_tail_practical_muon_bridge.png",
        "figures/e11_long_tail_practical_training/long_tail_practical_training.png",
        "figures/e11_long_tail_practical_training_lr_sweep/long_tail_practical_training_lr_sweep.png",
        "figures/e11_long_tail_forgetting/long_tail_head_only_forgetting.png",
        "figures/e11_long_tail_layerwise/long_tail_layerwise_drift.png",
        "figures/e11_cifar100_resnet_practical_muon_bridge/cifar100_resnet_practical_muon_bridge.png",
        "figures/e11_cifar100_resnet_layer_jvp_checkpoint_prediction/cifar100_resnet_layer_jvp_checkpoint_prediction.png",
        "squared drift ratio=0.5501",
        "head-alignment ratio=2.083",
        "Frobenius-norm ratio=3.112",
        "operator-norm ratio=0.5543",
        "polar(M_t) squared drift ratio=0.8199",
        "trajectory NS(M_t) squared drift ratio=0.8019",
        "scaled-JVP threshold accuracy=1",
        "scaled-JVP held-out Spearman=-0.3203",
        "many balanced accuracy=0.3665",
        "medium=0.1036",
        "few=0.0129",
        "final squared drift ratio=0.6167",
        "broad tail-accuracy or benchmark improvement",
        "narrow tail-loss/margin diagnostic",
        "Older condition-geometry artifacts remain useful guardrails",
    ]
    missing_evidence_index = [phrase for phrase in required_evidence_index_phrases if phrase not in evidence_index]
    if missing_evidence_index:
        raise AssertionError(f"evidence index missing current head-to-tail entries: {missing_evidence_index}")
    assert_forbidden_phrases_absent(
        "evidence index squared-drift wording",
        evidence_index,
        [
            "polar(M_t) drift ratio=0.8199",
            "trajectory NS(M_t) drift ratio=0.8019",
            "| drift ratio=0.5501",
            "| final drift ratio=0.6167",
            "smaller effective step",
        ],
    )
    research_synthesis = Path("discussion/e11_research_synthesis.md").read_text(encoding="utf-8")
    required_research_synthesis_phrases = [
        "current paper-facing synthesis",
        "head-to-tail interference mechanism paper",
        "matched-head-gain protocol",
        "Fixed-checkpoint and short-trajectory Muon-style directions show selected-state compatibility",
        "Muon-style compatibility fixed/trajectory",
        "scaled head-gain efficiency rather than intrinsically lower tail sensitivity",
        "not a claim that full Muon",
    ]
    missing_research_synthesis = [phrase for phrase in required_research_synthesis_phrases if phrase not in research_synthesis]
    if missing_research_synthesis:
        raise AssertionError(f"research synthesis missing current head-to-tail framing: {missing_research_synthesis}")
    research_direction = Path("discussion/e11_research_direction_map.md").read_text(encoding="utf-8")
    required_research_direction_phrases = [
        "head-to-tail interference mechanism paper",
        "matched-head-gain spectral/polar",
        "rank/sensitivity condition",
        "selected-state compatibility",
        "Muon-style momentum/NS states",
        "Older Muon/Adam condition-geometry results remain useful guardrails",
        "not the main paper thesis",
    ]
    missing_research_direction = [phrase for phrase in required_research_direction_phrases if phrase not in research_direction]
    if missing_research_direction:
        raise AssertionError(f"research direction map missing current head-to-tail framing: {missing_research_direction}")
    claim_validity = Path("discussion/e11_claim_validity_audit.md").read_text(encoding="utf-8")
    required_claim_validity_phrases = [
        "current head-to-tail experiments",
        "idealized spectral/polar directions can reduce head-to-tail drift on tail-example logits",
        "small-digits selected-state compatibility check",
        "short-trajectory",
        "not a modern long-tail optimizer benchmark",
        "do **not** justify saying",
        "head-alignment ratio=2.083",
        "operator-norm ratio=0.5543",
        "final-layer ResNet condition scatter",
        "tail-rich control",
        "all-layer ResNet JVP",
    ]
    missing_claim_validity = [phrase for phrase in required_claim_validity_phrases if phrase not in claim_validity]
    if missing_claim_validity:
        raise AssertionError(f"claim validity audit missing current head-to-tail framing: {missing_claim_validity}")
    paper_readiness = Path("discussion/e11_paper_readiness_audit.md").read_text(encoding="utf-8")
    required_readiness_phrases = [
        "current head-to-tail interference paper",
        "Matched-head-gain spectral/polar directions reduce held-out tail-example logit drift",
        "nrank(G_H) > ssrank(B_T,A_T)",
        "Real long-tail benchmark",
        "Real long-tail practical Muon benchmark",
        "Larger-architecture layerwise diagnostic",
        "final-layer condition scatter",
        "tail-rich ResNet control",
        "all-layer ResNet JVP",
        "lower tail-example logit drift automatically improves",
        "Head-to-Tail Interference in Long-Tailed Small-Batch Training",
        "explicit norm-specific scaling readouts",
        "smaller operator norm but larger Frobenius norm",
    ]
    missing_readiness = [phrase for phrase in required_readiness_phrases if phrase not in paper_readiness]
    if missing_readiness:
        raise AssertionError(f"paper-readiness audit missing required next-step content: {missing_readiness}")
    top_conference_plan = Path("discussion/e11_top_conference_plan.md").read_text(encoding="utf-8")
    required_top_conference_phrases = [
        "Top-Conference Upgrade Plan",
        "CIFAR-100-LT ResNet18 gives squared drift ratio",
        "ResNet checkpoint-quality sweep",
        "checkpoint sweep now reduces single-checkpoint risk",
        "Rank-side proxy scatter",
        "Final-layer downstream-aware condition",
        "All-layer downstream-aware ResNet condition scatter",
        "all-layer JVP tail-quality",
        "Tail-quality control",
        "tail-quality control",
        "Long-tail imbalance sweep",
        "Practical optimizer bridge on CIFAR-100-LT",
        "Acceptance Gates",
        "working test environment with both `torch` and `pytest`",
    ]
    missing_top_conference = [
        phrase for phrase in required_top_conference_phrases if phrase not in top_conference_plan
    ]
    if missing_top_conference:
        raise AssertionError(f"top-conference plan missing required gates: {missing_top_conference}")
    readme = Path("README_E11.md").read_text(encoding="utf-8")
    if "## Main Entry Points" not in readme or "## Current Publication Gaps" not in readme:
        raise AssertionError("README_E11.md must document entry points and publication gaps")
    if (
        "make e11-check" not in readme
        or "make e11-full" not in readme
        or "make e11-main-results" not in readme
        or "make e11-cifar-resnet-results" not in readme
        or "make e11-cifar-resnet-rho002-results" not in readme
        or "make e11-cifar-resnet-checkpoint-sweep-results" not in readme
        or "make e11-cifar-resnet-condition-proxy-results" not in readme
        or "make e11-cifar-resnet-layer-jvp-tail-quality-results" not in readme
        or "make e11-cifar-resnet-practical-muon-bridge-results" not in readme
        or "make e11-guardrail-assets" not in readme
        or "make e11-all-assets" not in readme
    ):
        raise AssertionError("README_E11.md must document E11 make targets")
    artifact_manifest = Path("discussion/e11_artifact_manifest.md").read_text(encoding="utf-8")
    for phrase in [
        "Artifact Directories",
        "Key Quantitative Tables",
        "Key Paper Documents",
        "discussion/e11_reference_audit.md",
        "Ignored Local Artifacts",
        "results/e11_artifact_manifest.json",
    ]:
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
