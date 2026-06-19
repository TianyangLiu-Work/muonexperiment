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
    ZERO_ROW_ALLOWED_TABLES,
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
    zero_row_allowed_tables = set(ZERO_ROW_ALLOWED_TABLES)
    manifest_zero_allowed = set(manifest_json.get("zero_row_allowed_tables", []))
    if not zero_row_allowed_tables.issubset(manifest_zero_allowed):
        raise AssertionError(
            f"artifact manifest missing zero-row allowed table policy: {zero_row_allowed_tables - manifest_zero_allowed}"
        )
    bad_manifest_tables = [
        item.get("path")
        for item in manifest_json.get("key_tables", [])
        if item.get("path") in required_manifest_tables
        and item.get("path") not in zero_row_allowed_tables
        and int(item.get("rows", 0)) <= 0
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


def assert_condition_score_protocol(scores: pd.DataFrame, splits: pd.DataFrame, gates: pd.DataFrame) -> None:
    required_score_columns = {
        "score_id",
        "role",
        "uses_observed_source_drift",
        "fit_rule",
        "input_features",
        "heldout_claim_allowed",
        "reason",
    }
    missing_score_columns = required_score_columns - set(scores.columns)
    if missing_score_columns:
        raise AssertionError(f"condition-score protocol score registry missing columns: {missing_score_columns}")
    required_score_ids = {
        "source_observed_drift_positive_control",
        "early_layer_prior",
        "legacy_scaled_jvp_ratio",
        "condition_score_v2_calibrated_residual",
        "theory_sign_composite",
    }
    score_ids = set(scores["score_id"])
    if not required_score_ids.issubset(score_ids):
        raise AssertionError(f"condition-score protocol missing score ids: {required_score_ids - score_ids}")
    primary = scores[scores["score_id"].eq("condition_score_v2_calibrated_residual")]
    if len(primary) != 1:
        raise AssertionError("condition-score protocol must have exactly one primary v2 score row")
    primary_row = primary.iloc[0]
    if primary_row["role"] != "primary_candidate" or primary_row["heldout_claim_allowed"] != "yes, if all primary gates pass":
        raise AssertionError("condition-score protocol primary row must be the only held-out claim candidate")
    if "target" in str(primary_row["fit_rule"]).lower() and "without" not in str(primary_row["fit_rule"]).lower():
        raise AssertionError("condition-score protocol primary fit rule must avoid target leakage")

    required_split_columns = {
        "split_id",
        "role",
        "dataset",
        "architecture",
        "checkpoints",
        "seeds",
        "score_tuning_allowed",
        "target_used_for_tuning",
        "compute_mode",
        "planned_artifact_prefix",
    }
    missing_split_columns = required_split_columns - set(splits.columns)
    if missing_split_columns:
        raise AssertionError(f"condition-score protocol split registry missing columns: {missing_split_columns}")
    required_split_roles = {
        "calibration_only",
        "primary_heldout_checkpoint",
        "primary_heldout_architecture",
        "primary_heldout_data",
    }
    split_roles = set(splits["role"])
    if not required_split_roles.issubset(split_roles):
        raise AssertionError(f"condition-score protocol missing split roles: {required_split_roles - split_roles}")
    heldout = splits[splits["role"].astype(str).str.startswith("primary_heldout")]
    if heldout.empty:
        raise AssertionError("condition-score protocol must include primary held-out splits")
    if not heldout["score_tuning_allowed"].eq("no").all() or not heldout["target_used_for_tuning"].eq("no").all():
        raise AssertionError("condition-score protocol held-out splits must forbid score and target tuning")
    if "GPU via Slurm" not in set(splits["compute_mode"]):
        raise AssertionError("condition-score protocol must specify GPU via Slurm for held-out experiments")

    required_gate_columns = {"gate_id", "scope", "requirement", "pass_condition"}
    missing_gate_columns = required_gate_columns - set(gates.columns)
    if missing_gate_columns:
        raise AssertionError(f"condition-score protocol gates missing columns: {missing_gate_columns}")
    required_gate_ids = {
        "G1-no-target-leakage",
        "G2-primary-residual-prediction",
        "G3-threshold-direction",
        "G4-baseline-comparison",
        "G5-no-performance-overclaim",
        "G6-reporting-completeness",
    }
    gate_ids = set(gates["gate_id"])
    if not required_gate_ids.issubset(gate_ids):
        raise AssertionError(f"condition-score protocol missing gates: {required_gate_ids - gate_ids}")
    combined_gates = " ".join(gates["pass_condition"].astype(str))
    if "CI lower endpoint is above 0" not in combined_gates or "at least 0.8" not in combined_gates:
        raise AssertionError("condition-score protocol gates must encode residual prediction and threshold success")


def assert_top_conference_gap_register(frame: pd.DataFrame) -> None:
    required_columns = {
        "gap_id",
        "priority",
        "claim_unblocked",
        "current_state",
        "required_next_evidence",
        "acceptance_gate",
        "compute_mode",
        "planned_artifacts",
        "risk_if_missing",
    }
    missing_columns = required_columns - set(frame.columns)
    if missing_columns:
        raise AssertionError(f"top-conference gap register missing columns: {missing_columns}")
    required_gap_ids = {
        "P0-PredictiveCondition",
        "P0-StandardBenchmark",
        "P1-HeldOutGenerality",
        "P1-TheoryToScore",
        "P1-PracticalMuonBridge",
        "P2-NaturalBoundaryCases",
        "P2-PackagingRepro",
    }
    gap_ids = set(frame["gap_id"])
    if not required_gap_ids.issubset(gap_ids):
        raise AssertionError(f"top-conference gap register missing gap ids: {required_gap_ids - gap_ids}")
    priorities = set(frame["priority"])
    if not {"P0", "P1", "P2"}.issubset(priorities):
        raise AssertionError(f"top-conference gap register missing priorities: {priorities}")
    if int(frame["priority"].eq("P0").sum()) < 2:
        raise AssertionError("top-conference gap register must keep at least two P0 gates")
    compute_modes = set(frame["compute_mode"])
    if "GPU via Slurm" not in compute_modes or "CPU" not in compute_modes:
        raise AssertionError(f"top-conference gap register missing compute modes: {compute_modes}")
    weak_gate_rows = frame[
        frame["acceptance_gate"].astype(str).str.len().lt(80)
        | frame["acceptance_gate"].astype(str).str.contains("TBD|todo", case=False, regex=True)
    ]
    if not weak_gate_rows.empty:
        raise AssertionError(f"top-conference gap register has weak acceptance gates: {weak_gate_rows['gap_id'].tolist()}")
    missing_result_artifacts = frame[~frame["planned_artifacts"].astype(str).str.contains("results/|discussion/", regex=True)]
    if not missing_result_artifacts.empty:
        raise AssertionError(
            f"top-conference gap register rows must name planned artifacts: "
            f"{missing_result_artifacts['gap_id'].tolist()}"
        )
    packaging_row = frame[frame["gap_id"].eq("P2-PackagingRepro")]
    if packaging_row.empty:
        raise AssertionError("top-conference gap register missing P2-PackagingRepro row")
    packaging_text = " ".join(
        str(packaging_row.iloc[0][column])
        for column in ["current_state", "required_next_evidence", "planned_artifacts"]
    )
    for phrase in [
        "discussion/e11_artifact_review_packet.md",
        "results/e11_artifact_review_packet/*",
        "discussion/e11_camera_ready_package_audit.md",
        "results/e11_camera_ready_package_audit/*",
        "serverREADME.md",
        "GPU-pending boundary",
    ]:
        if phrase not in packaging_text:
            raise AssertionError(f"P2-PackagingRepro missing artifact-review packet boundary: {phrase}")


def assert_top_conference_claim_decision_audit(
    claim_matrix: pd.DataFrame,
    reviewer_objections: pd.DataFrame,
    rebuttal_pack: pd.DataFrame,
    manuscript_queue: pd.DataFrame,
    paper_sequence: pd.DataFrame,
    readiness_summary: pd.DataFrame,
) -> None:
    required_claim_columns = {
        "claim_id",
        "paper_section",
        "current_decision",
        "evidence_status",
        "author_allowed_wording",
        "author_blocked_wording",
        "decisive_gate",
        "required_next_action",
        "source_artifacts",
    }
    missing_claim_columns = required_claim_columns - set(claim_matrix.columns)
    if missing_claim_columns:
        raise AssertionError(f"top-conference claim decision matrix missing columns: {missing_claim_columns}")
    expected_claim_ids = {
        "TCD-1-main-mechanism-theorem",
        "TCD-2-natural-drift-diagnostic",
        "TCD-3-predictive-condition-generalization",
        "TCD-4-natural-counterexample-or-finite-null",
        "TCD-5-optimizer-performance-benchmark",
        "TCD-6-artifact-reproducibility",
    }
    if set(claim_matrix["claim_id"]) != expected_claim_ids:
        raise AssertionError(
            "top-conference claim decision matrix claim ids changed: "
            f"{sorted(set(claim_matrix['claim_id']))}"
        )
    decision_lookup = claim_matrix.set_index("claim_id")["current_decision"].to_dict()
    expected_decisions = {
        "TCD-1-main-mechanism-theorem": "supportable_main_with_assumptions",
        "TCD-2-natural-drift-diagnostic": "supportable_diagnostic_only",
        "TCD-3-predictive-condition-generalization": "blocked_completed_final_failed_boundary",
        "TCD-4-natural-counterexample-or-finite-null": "finite_null_candidate_with_caveats",
        "TCD-5-optimizer-performance-benchmark": "blocked_protocol_pending",
        "TCD-6-artifact-reproducibility": "supportable_with_toolchain_caveat",
    }
    if decision_lookup != expected_decisions:
        raise AssertionError(f"top-conference claim decisions drifted: {decision_lookup}")
    joined_blocked = " ".join(claim_matrix["author_blocked_wording"].astype(str))
    for phrase in [
        "global convergence or optimizer superiority",
        "the v5 score predicts unseen real-task residual risk",
        "fresh natural primary counterexample",
        "unqualified absence of natural counterexamples outside the registered phase1/phase2 spaces",
        "quality-failed or head-gain-failed rows validate the mechanism",
        "Muon or spectral training is competitive on long-tail benchmarks",
        "local Muon drift compatibility implies benchmark superiority",
        "sampled bridge states represent the full training trajectory distribution",
        "using partial validation leaderboard to change selection, launch order, or final seed plan",
        "running final-safe-submit before FEP/TFE/FLA gates pass",
        "claiming final benchmark performance from not_ready final evaluator gates",
        "preferred pdflatex/bibtex/xelatex clean-checkout reproducibility is complete on this server",
    ]:
        if phrase not in joined_blocked:
            raise AssertionError(f"top-conference claim decision audit missing blocked wording: {phrase}")
    evidence_text = " ".join(claim_matrix["evidence_status"].astype(str))
    for phrase in [
        "v5_p0_predictive_condition_claim=not_ready",
        "observed=26/26",
        "raw_worse_rows=0",
        "quality_gate_fail_rows=23",
        "phase2_observed=8/8",
        "phase2_head_gain_gate_fail_rows=8",
        "TVS-1-validation-grid-complete=not_ready",
        "TVS-5-occupancy-logging-complete=not_ready",
        "TLA-1-selection-rule-frozen=pass",
        "TLA-4-partial-grid-selection-block=pass",
        "TLA-6-launch-history-auditable=pass",
        "VRF-current-validation-prefix=partial_grid_no_claim_change",
        "VRF-A1-prefix-result-refresh=pass",
        "VRF-A4-final-submit-handoff=not_ready",
        "VRF-F3-partial-family-selection=active",
        "VRF-F5-validation-leaderboard-performance-claim=active",
        "TPS-current-post-exposure-boundary=post_partial_validation_exposure_sealed",
        "TPS-1-hash-manifest-complete=pass",
        "TPS-5-post-exposure-claim-authority=pass",
        "TBF-1-registered-family-coverage=pass",
        "TBF-3-candidate-budget-disclosure=pass_with_disclosure",
        "validation_budget_baselines=56",
        "validation_budget_muon_candidates=108",
        "RBP-G1-final-output-quarantine=pass",
        "RBP-G3-robustness-tests-registered=pass",
        "RBP-G5-final-execution-readiness=not_ready",
        "robustness_test_rows=16",
        "FEP-1-selection-gates-ready=not_ready",
        "FEP-5-all-families-ready=not_ready",
        "TFE-6-final-claim-state=not_ready",
        "FLA-1-final-execution-gates-ready=not_ready",
        "FLA-5-submit-flag=dry_run",
        "final_launch_status=blocked_final_launch_gates_not_ready",
        "final_submit_command=<empty>",
        "MSD-T1-local-response-integrand=local_integrand_supported_on_sampled_states",
        "MSD-T2-state-occupancy-measure=occupancy_measure_missing_for_final_training",
        "MSD-T4-terminal-risk-functional=final_performance_negative_boundary",
        "MSG-4-top-tier-practical-claim=blocked_until_state_distribution_and_final_gates_pass",
        "R3-preferred-latex-toolchain=not_ready",
    ]:
        if phrase not in evidence_text:
            raise AssertionError(f"top-conference claim decision audit missing evidence status: {phrase}")

    expected_objections = {
        "RO-1-toy-theorem",
        "RO-2-score-cherry-picking",
        "RO-3-no-natural-negative",
        "RO-4-muon-overclaim",
        "RO-5-artifact-reproducibility",
    }
    if set(reviewer_objections["objection_id"]) != expected_objections:
        raise AssertionError("top-conference reviewer objection matrix must cover the fixed objection set")
    forbidden_shortcuts = " ".join(reviewer_objections["forbidden_shortcut"].astype(str))
    for phrase in [
        "using any final row to refit or reselect the score",
        "claiming a universal finite null or natural counterexample without adjusted primary evidence",
        "turning lower local drift into a final tail-accuracy claim",
        "treating sampled bridge states as the full training trajectory distribution",
        "partial validation leaderboard to change selection, launch order, or final seed plan",
        "violating the validation refresh firewall forbidden-action matrix",
        "changing sealed protocol surfaces after partial exposure without a fresh protocol",
        "hiding validation-budget asymmetry",
        "claiming robust final performance without sign-flip/bootstrap sensitivity",
        "running final-safe-submit before FEP/TFE/FLA gates pass",
    ]:
        if phrase not in forbidden_shortcuts:
            raise AssertionError(f"top-conference reviewer objection matrix missing forbidden shortcut: {phrase}")
    if set(rebuttal_pack["objection_id"]) != expected_objections:
        raise AssertionError("top-conference rebuttal response pack must cover every reviewer objection")
    if set(rebuttal_pack["claim_id"]) != expected_claim_ids - {"TCD-2-natural-drift-diagnostic"}:
        raise AssertionError("top-conference rebuttal response pack must map objections to active decision claims")
    rebuttal_text = " ".join(rebuttal_pack.astype(str).to_numpy().ravel())
    for phrase in [
        "answer_now_with_scope_and_real-diagnostic_bridge",
        "do not use final rows for refit or score selection",
        "26/26 phase1 observed, 8/8 phase2 observed",
        "finite-null-candidate wording only with detectable-effect, head-gain, and quality caveats",
        "quarantine benchmark claims",
        "discussion/e11_cifar100_resnet_lt_tuned_benchmark_leakage_audit.md",
        "discussion/e11_cifar100_resnet_lt_tuned_benchmark_refresh_firewall.md",
        "discussion/e11_cifar100_resnet_lt_tuned_benchmark_protocol_seal.md",
        "discussion/e11_cifar100_resnet_lt_tuned_benchmark_fairness_audit.md",
        "discussion/e11_cifar100_resnet_lt_tuned_benchmark_final_robustness_plan.md",
        "discussion/e11_cifar100_resnet_lt_tuned_benchmark_final_execution_plan.md",
        "discussion/e11_cifar100_resnet_lt_tuned_benchmark_final_evaluation.md",
        "discussion/e11_cifar100_resnet_lt_tuned_benchmark_final_launch_audit.md",
        "final execution/evaluation/launch gates",
        "leakage guards",
        "validation refresh firewall transitions",
        "protocol-seal gates",
        "fairness gates",
        "robustness gates",
        "discussion/e11_muon_state_distribution_contract.md",
        "local compatibility can coexist with poor final performance",
        "occupancy logging",
        "preferred-LaTeX clean-checkout completion",
    ]:
        if phrase not in rebuttal_text:
            raise AssertionError(f"top-conference rebuttal response pack missing phrase: {phrase}")
    expected_edits = {
        "MEQ-1-theory-frontload-scope",
        "MEQ-2-diagnostic-caveat-next-to-results",
        "MEQ-3-v5-completed-boundary-language",
        "MEQ-4-natural-negative-complete-finite-family",
        "MEQ-5-performance-benchmark-quarantine",
        "MEQ-6-artifact-review-caveat",
    }
    if set(manuscript_queue["edit_id"]) != expected_edits:
        raise AssertionError("top-conference manuscript edit queue must preserve the fixed edit set")
    manuscript_text = " ".join(manuscript_queue.astype(str).to_numpy().ravel())
    for phrase in [
        "worst-case-vs-realized distinction",
        "No result paragraph may convert matched-head-gain logit drift into tail-accuracy",
        "frozen completed negative boundary",
        "finite registered null candidates with detectable-effect, head-gain, and quality caveats",
        "quarantine all competitive optimizer wording",
        "tuned leakage audit",
        "validation refresh firewall",
        "protocol hash seal",
        "fairness audit",
        "final robustness plan",
        "final execution/evaluator/launch gates",
        "state-distribution contract",
        "state-distribution occupancy summaries, passing leakage guards, passing refresh-firewall transitions, passing protocol-seal gates, disclosed fairness-budget gates, and passing robustness sensitivity gates",
        "passing final execution/evaluator/launch gates before practical-performance wording",
        "preferred-LaTeX clean-checkout gap",
    ]:
        if phrase not in manuscript_text:
            raise AssertionError(f"top-conference manuscript edit queue missing phrase: {phrase}")
    if list(paper_sequence["sequence_step"]) != [1, 2, 3, 4, 5, 6]:
        raise AssertionError("top-conference paper sequence must preserve the six-step claim order")
    summary_lookup = readiness_summary.set_index("current_decision")["claim_count"].astype(int).to_dict()
    if summary_lookup != {value: 1 for value in expected_decisions.values()}:
        raise AssertionError(f"top-conference readiness summary changed: {summary_lookup}")


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
        Path("results/e11_cifar100_resnet_imbalance_sweep") / "step_metrics.csv",
        Path("results/e11_cifar100_resnet_imbalance_sweep") / "pair_summary.csv",
        Path("results/e11_cifar100_resnet_imbalance_sweep") / "layer_metrics.csv",
        Path("results/e11_cifar100_resnet_imbalance_sweep") / "config.json",
        Path("figures/e11_cifar100_resnet_imbalance_sweep") / "cifar100_resnet_imbalance_sweep.png",
        Path("discussion/e11_cifar100_resnet_imbalance_sweep.md"),
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
        Path("results/e11_cifar100_resnet_layer_jvp_checkpoint_prediction") / "residual_prediction_pairs.csv",
        Path("results/e11_cifar100_resnet_layer_jvp_checkpoint_prediction") / "residual_prediction_summary.csv",
        Path("results/e11_cifar100_resnet_layer_jvp_checkpoint_prediction") / "config.json",
        Path("figures/e11_cifar100_resnet_layer_jvp_checkpoint_prediction") / "cifar100_resnet_layer_jvp_checkpoint_prediction.png",
        Path("discussion/e11_cifar100_resnet_layer_jvp_checkpoint_prediction.md"),
        Path("results/e11_cifar100_resnet_condition_score_audit") / "raw_score_pairs.csv",
        Path("results/e11_cifar100_resnet_condition_score_audit") / "raw_score_summary.csv",
        Path("results/e11_cifar100_resnet_condition_score_audit") / "residual_score_pairs.csv",
        Path("results/e11_cifar100_resnet_condition_score_audit") / "residual_score_summary.csv",
        Path("results/e11_cifar100_resnet_condition_score_audit") / "config.json",
        Path("figures/e11_cifar100_resnet_condition_score_audit") / "cifar100_resnet_condition_score_audit.png",
        Path("discussion/e11_cifar100_resnet_condition_score_audit.md"),
        Path("results/e11_cifar100_resnet_condition_score_protocol") / "score_registry.csv",
        Path("results/e11_cifar100_resnet_condition_score_protocol") / "split_registry.csv",
        Path("results/e11_cifar100_resnet_condition_score_protocol") / "acceptance_gates.csv",
        Path("discussion/e11_cifar100_resnet_condition_score_protocol.md"),
        Path("results/e11_cifar100_resnet_condition_score_next") / "score_pairs.csv",
        Path("results/e11_cifar100_resnet_condition_score_next") / "score_summary.csv",
        Path("results/e11_cifar100_resnet_condition_score_next") / "calibration_coefficients.csv",
        Path("results/e11_cifar100_resnet_condition_score_next") / "gate_report.csv",
        Path("results/e11_cifar100_resnet_condition_score_next") / "config.json",
        Path("figures/e11_cifar100_resnet_condition_score_next") / "cifar100_resnet_condition_score_next.png",
        Path("discussion/e11_cifar100_resnet_condition_score_next.md"),
        Path("results/e11_cifar100_resnet_condition_score_next/heldout_architecture") / "metrics.csv",
        Path("results/e11_cifar100_resnet_condition_score_next/heldout_architecture") / "paired_metrics.csv",
        Path("results/e11_cifar100_resnet_condition_score_next/heldout_architecture") / "layer_summary.csv",
        Path("results/e11_cifar100_resnet_condition_score_next/heldout_architecture") / "checkpoint_summary.csv",
        Path("results/e11_cifar100_resnet_condition_score_next/heldout_architecture") / "prediction_pairs.csv",
        Path("results/e11_cifar100_resnet_condition_score_next/heldout_architecture") / "prediction_summary.csv",
        Path("results/e11_cifar100_resnet_condition_score_next/heldout_architecture") / "residual_prediction_pairs.csv",
        Path("results/e11_cifar100_resnet_condition_score_next/heldout_architecture") / "residual_prediction_summary.csv",
        Path("results/e11_cifar100_resnet_condition_score_next/heldout_architecture") / "config.json",
        Path("figures/e11_cifar100_resnet_condition_score_next/heldout_architecture") / "cifar100_resnet_layer_jvp_checkpoint_prediction.png",
        Path("discussion/e11_cifar100_resnet_condition_score_next_heldout_architecture.md"),
        Path("results/e11_cifar100_resnet_condition_score_next/heldout_data") / "metrics.csv",
        Path("results/e11_cifar100_resnet_condition_score_next/heldout_data") / "paired_metrics.csv",
        Path("results/e11_cifar100_resnet_condition_score_next/heldout_data") / "layer_summary.csv",
        Path("results/e11_cifar100_resnet_condition_score_next/heldout_data") / "checkpoint_summary.csv",
        Path("results/e11_cifar100_resnet_condition_score_next/heldout_data") / "prediction_pairs.csv",
        Path("results/e11_cifar100_resnet_condition_score_next/heldout_data") / "prediction_summary.csv",
        Path("results/e11_cifar100_resnet_condition_score_next/heldout_data") / "residual_prediction_pairs.csv",
        Path("results/e11_cifar100_resnet_condition_score_next/heldout_data") / "residual_prediction_summary.csv",
        Path("results/e11_cifar100_resnet_condition_score_next/heldout_data") / "config.json",
        Path("figures/e11_cifar100_resnet_condition_score_next/heldout_data") / "cifar100_resnet_layer_jvp_checkpoint_prediction.png",
        Path("discussion/e11_cifar100_resnet_condition_score_next_heldout_data.md"),
        Path("results/e11_cifar100_resnet_condition_score_next/heldout_score_evaluation") / "heldout_score_pairs.csv",
        Path("results/e11_cifar100_resnet_condition_score_next/heldout_score_evaluation") / "heldout_score_summary.csv",
        Path("results/e11_cifar100_resnet_condition_score_next/heldout_score_evaluation") / "heldout_gate_report.csv",
        Path("results/e11_cifar100_resnet_condition_score_next/heldout_score_evaluation") / "config.json",
        Path("figures/e11_cifar100_resnet_condition_score_next/heldout_score_evaluation") / "cifar100_resnet_condition_score_heldout_evaluation.png",
        Path("discussion/e11_cifar100_resnet_condition_score_next_heldout_evaluation.md"),
        Path("discussion/e11_condition_score_heldout_failure_theory_note.md"),
        Path("results/e11_condition_score_theory_bridge") / "score_target_register.csv",
        Path("results/e11_condition_score_theory_bridge") / "fresh_protocol_requirements.csv",
        Path("discussion/e11_condition_score_theory_bridge.md"),
        Path("results/e11_condition_score_fresh_protocol") / "quarantine_register.csv",
        Path("results/e11_condition_score_fresh_protocol") / "score_freeze_registry.csv",
        Path("results/e11_condition_score_fresh_protocol") / "fresh_split_registry.csv",
        Path("results/e11_condition_score_fresh_protocol") / "acceptance_gates.csv",
        Path("results/e11_condition_score_fresh_protocol") / "protocol_status.csv",
        Path("results/e11_condition_score_fresh_protocol/fresh_score_evaluation") / "fresh_gate_report.csv",
        Path("results/e11_condition_score_fresh_protocol/fresh_score_evaluation") / "config.json",
        Path("discussion/e11_condition_score_fresh_protocol.md"),
        Path("discussion/e11_condition_score_fresh_evaluation.md"),
        Path("scripts/e11_evaluate_condition_score_fresh_protocol.py"),
        Path("scripts/e11_write_condition_score_fresh_protocol.py"),
        Path("scripts/slurm/e11_cifar100_resnet_condition_score_fresh_architecture_resnet50.sbatch"),
        Path("scripts/slurm/e11_cifar100_resnet_condition_score_fresh_data_cifar10_alt.sbatch"),
        Path("results/e11_condition_score_failure_mechanism_audit") / "score_outcome_matrix.csv",
        Path("results/e11_condition_score_failure_mechanism_audit") / "split_obstruction_taxonomy.csv",
        Path("results/e11_condition_score_failure_mechanism_audit") / "resnet50_stage_reversal.csv",
        Path("figures/e11_condition_score_failure_mechanism_audit") / "resnet50_stage_reversal.png",
        Path("discussion/e11_condition_score_failure_mechanism_audit.md"),
        Path("scripts/e11_write_condition_score_failure_mechanism_audit.py"),
        Path("results/e11_condition_score_v4_protocol") / "spent_split_register.csv",
        Path("results/e11_condition_score_v4_protocol") / "score_axis_registry.csv",
        Path("results/e11_condition_score_v4_protocol") / "unspent_split_registry.csv",
        Path("results/e11_condition_score_v4_protocol") / "acceptance_gates.csv",
        Path("results/e11_condition_score_v4_protocol") / "protocol_status.csv",
        Path("results/e11_condition_score_v4_protocol/validation_cifar100_rotated") / "metrics.csv",
        Path("results/e11_condition_score_v4_protocol/validation_cifar100_rotated") / "paired_metrics.csv",
        Path("results/e11_condition_score_v4_protocol/validation_cifar100_rotated") / "layer_summary.csv",
        Path("results/e11_condition_score_v4_protocol/validation_cifar100_rotated") / "checkpoint_summary.csv",
        Path("results/e11_condition_score_v4_protocol/validation_cifar100_rotated") / "prediction_pairs.csv",
        Path("results/e11_condition_score_v4_protocol/validation_cifar100_rotated") / "prediction_summary.csv",
        Path("results/e11_condition_score_v4_protocol/validation_cifar100_rotated") / "residual_prediction_pairs.csv",
        Path("results/e11_condition_score_v4_protocol/validation_cifar100_rotated") / "residual_prediction_summary.csv",
        Path("results/e11_condition_score_v4_protocol/validation_cifar100_rotated") / "config.json",
        Path("figures/e11_condition_score_v4_protocol/validation_cifar100_rotated") / "cifar100_resnet_layer_jvp_checkpoint_prediction.png",
        Path("results/e11_condition_score_v4_protocol/validation_score_freeze") / "score_formula_registry.csv",
        Path("results/e11_condition_score_v4_protocol/validation_score_freeze") / "freeze_status.csv",
        Path("results/e11_condition_score_v4_protocol/validation_score_freeze") / "validation_score_pairs.csv",
        Path("results/e11_condition_score_v4_protocol/validation_score_freeze") / "validation_score_summary.csv",
        Path("results/e11_condition_score_v4_protocol/validation_score_freeze") / "validation_gate_report.csv",
        Path("results/e11_condition_score_v4_protocol/validation_score_freeze") / "config.json",
        Path("results/e11_condition_score_v4_protocol/final_architecture_wide_resnet50_2") / "metrics.csv",
        Path("results/e11_condition_score_v4_protocol/final_architecture_wide_resnet50_2") / "paired_metrics.csv",
        Path("results/e11_condition_score_v4_protocol/final_architecture_wide_resnet50_2") / "layer_summary.csv",
        Path("results/e11_condition_score_v4_protocol/final_architecture_wide_resnet50_2") / "checkpoint_summary.csv",
        Path("results/e11_condition_score_v4_protocol/final_architecture_wide_resnet50_2") / "prediction_pairs.csv",
        Path("results/e11_condition_score_v4_protocol/final_architecture_wide_resnet50_2") / "prediction_summary.csv",
        Path("results/e11_condition_score_v4_protocol/final_architecture_wide_resnet50_2") / "residual_prediction_pairs.csv",
        Path("results/e11_condition_score_v4_protocol/final_architecture_wide_resnet50_2") / "residual_prediction_summary.csv",
        Path("results/e11_condition_score_v4_protocol/final_architecture_wide_resnet50_2") / "config.json",
        Path("figures/e11_condition_score_v4_protocol/final_architecture_wide_resnet50_2") / "cifar100_resnet_layer_jvp_checkpoint_prediction.png",
        Path("results/e11_condition_score_v4_protocol/final_data_cifar10_mixed") / "metrics.csv",
        Path("results/e11_condition_score_v4_protocol/final_data_cifar10_mixed") / "paired_metrics.csv",
        Path("results/e11_condition_score_v4_protocol/final_data_cifar10_mixed") / "layer_summary.csv",
        Path("results/e11_condition_score_v4_protocol/final_data_cifar10_mixed") / "checkpoint_summary.csv",
        Path("results/e11_condition_score_v4_protocol/final_data_cifar10_mixed") / "prediction_pairs.csv",
        Path("results/e11_condition_score_v4_protocol/final_data_cifar10_mixed") / "prediction_summary.csv",
        Path("results/e11_condition_score_v4_protocol/final_data_cifar10_mixed") / "residual_prediction_pairs.csv",
        Path("results/e11_condition_score_v4_protocol/final_data_cifar10_mixed") / "residual_prediction_summary.csv",
        Path("results/e11_condition_score_v4_protocol/final_data_cifar10_mixed") / "config.json",
        Path("figures/e11_condition_score_v4_protocol/final_data_cifar10_mixed") / "cifar100_resnet_layer_jvp_checkpoint_prediction.png",
        Path("results/e11_condition_score_v4_protocol/final_score_evaluation") / "final_score_pairs.csv",
        Path("results/e11_condition_score_v4_protocol/final_score_evaluation") / "final_score_summary.csv",
        Path("results/e11_condition_score_v4_protocol/final_score_evaluation") / "final_gate_report.csv",
        Path("results/e11_condition_score_v4_protocol/final_score_evaluation") / "config.json",
        Path("results/e11_condition_score_v4_failure_mechanism_audit") / "final_outcome_matrix.csv",
        Path("results/e11_condition_score_v4_failure_mechanism_audit") / "axis_all_layer_summary.csv",
        Path("results/e11_condition_score_v4_failure_mechanism_audit") / "axis_transfer_pair_scores.csv",
        Path("results/e11_condition_score_v4_failure_mechanism_audit") / "axis_pair_summary.csv",
        Path("results/e11_condition_score_v4_failure_mechanism_audit") / "top5_stage_summary.csv",
        Path("results/e11_condition_score_v4_failure_mechanism_audit") / "obstruction_summary.csv",
        Path("figures/e11_condition_score_v4_failure_mechanism_audit") / "v4_cifar10_mixed_reversal_top5.png",
        Path("discussion/e11_condition_score_v4_failure_mechanism_audit.md"),
        Path("scripts/e11_write_condition_score_v4_failure_mechanism_audit.py"),
        Path("results/e11_matrix_block_theorem_proof") / "theorem_statement.csv",
        Path("results/e11_matrix_block_theorem_proof") / "assumption_ledger.csv",
        Path("results/e11_matrix_block_theorem_proof") / "proof_steps.csv",
        Path("results/e11_matrix_block_theorem_proof") / "claim_implications.csv",
        Path("results/e11_matrix_block_theorem_proof") / "paper_cross_checks.csv",
        Path("results/e11_matrix_block_theorem_proof") / "config.json",
        Path("discussion/e11_matrix_block_theorem_proof.md"),
        Path("scripts/e11_write_matrix_block_theorem_proof.py"),
        Path("results/e11_matrix_block_tightness_audit") / "rank_boundary_cases.csv",
        Path("results/e11_matrix_block_tightness_audit") / "formula_checks.csv",
        Path("results/e11_matrix_block_tightness_audit") / "caveat_checks.csv",
        Path("results/e11_matrix_block_tightness_audit") / "config.json",
        Path("discussion/e11_matrix_block_tightness_audit.md"),
        Path("scripts/e11_write_matrix_block_tightness_audit.py"),
        Path("results/e11_theory_proof_obligation_register") / "proof_obligations.csv",
        Path("results/e11_theory_proof_obligation_register") / "assumption_stress_tests.csv",
        Path("results/e11_theory_proof_obligation_register") / "claim_scope_boundaries.csv",
        Path("results/e11_theory_proof_obligation_register") / "theorem_to_experiment_queue.csv",
        Path("results/e11_theory_proof_obligation_register") / "config.json",
        Path("discussion/e11_theory_proof_obligation_register.md"),
        Path("scripts/e11_write_theory_proof_obligation_register.py"),
        Path("results/e11_condition_score_v5_theory_protocol") / "theory_term_register.csv",
        Path("results/e11_condition_score_v5_theory_protocol") / "score_contract.csv",
        Path("results/e11_condition_score_v5_theory_protocol") / "spent_evidence_policy.csv",
        Path("results/e11_condition_score_v5_theory_protocol") / "unspent_split_requirements.csv",
        Path("results/e11_condition_score_v5_theory_protocol") / "acceptance_gates.csv",
        Path("discussion/e11_condition_score_v5_theory_protocol.md"),
        Path("scripts/e11_write_condition_score_v5_theory_protocol.py"),
        Path("results/e11_condition_score_v5_theory_to_score_map") / "theorem_proxy_map.csv",
        Path("results/e11_condition_score_v5_theory_to_score_map") / "score_lineage.csv",
        Path("results/e11_condition_score_v5_theory_to_score_map") / "transport_normalization_contract.csv",
        Path("results/e11_condition_score_v5_theory_to_score_map") / "post_final_transport_obligations.csv",
        Path("results/e11_condition_score_v5_theory_to_score_map") / "next_protocol_firewall.csv",
        Path("results/e11_condition_score_v5_theory_to_score_map") / "falsifiable_predictions.csv",
        Path("results/e11_condition_score_v5_theory_to_score_map") / "ablation_matrix.csv",
        Path("results/e11_condition_score_v5_theory_to_score_map") / "claim_readiness_ledger.csv",
        Path("discussion/e11_condition_score_v5_theory_to_score_map.md"),
        Path("scripts/e11_write_condition_score_v5_theory_to_score_map.py"),
        Path("results/e11_condition_score_v5_protocol/validation_cifar100_mod4_partition") / "metrics.csv",
        Path("results/e11_condition_score_v5_protocol/validation_cifar100_mod4_partition") / "paired_metrics.csv",
        Path("results/e11_condition_score_v5_protocol/validation_cifar100_mod4_partition") / "layer_summary.csv",
        Path("results/e11_condition_score_v5_protocol/validation_cifar100_mod4_partition") / "checkpoint_summary.csv",
        Path("results/e11_condition_score_v5_protocol/validation_cifar100_mod4_partition") / "prediction_summary.csv",
        Path("results/e11_condition_score_v5_protocol/validation_cifar100_mod4_partition") / "residual_prediction_summary.csv",
        Path("results/e11_condition_score_v5_protocol/validation_cifar100_mod4_partition") / "config.json",
        Path("figures/e11_condition_score_v5_protocol/validation_cifar100_mod4_partition")
        / "cifar100_resnet_layer_jvp_checkpoint_prediction.png",
        Path("discussion/e11_condition_score_v5_validation_cifar100_mod4_partition.md"),
        Path("results/e11_condition_score_v5_protocol/validation_score_freeze") / "score_formula_registry.csv",
        Path("results/e11_condition_score_v5_protocol/validation_score_freeze") / "freeze_status.csv",
        Path("results/e11_condition_score_v5_protocol/validation_score_freeze") / "validation_score_pairs.csv",
        Path("results/e11_condition_score_v5_protocol/validation_score_freeze") / "validation_score_summary.csv",
        Path("results/e11_condition_score_v5_protocol/validation_score_freeze") / "validation_gate_report.csv",
        Path("results/e11_condition_score_v5_protocol/validation_score_freeze") / "config.json",
        Path("discussion/e11_condition_score_v5_validation_freeze.md"),
        Path("scripts/e11_freeze_condition_score_v5_validation.py"),
        Path("results/e11_condition_score_v5_protocol/final_score_evaluation") / "final_score_pairs.csv",
        Path("results/e11_condition_score_v5_protocol/final_score_evaluation") / "final_score_summary.csv",
        Path("results/e11_condition_score_v5_protocol/final_score_evaluation") / "final_gate_report.csv",
        Path("results/e11_condition_score_v5_protocol/final_score_evaluation") / "config.json",
        Path("discussion/e11_condition_score_v5_final_evaluation.md"),
        Path("scripts/e11_evaluate_condition_score_v5_finals.py"),
        Path("scripts/e11_write_condition_score_v5_final_evaluation.py"),
        Path("results/e11_condition_score_v5_protocol/final_interpretation_plan")
        / "current_interpretation_summary.csv",
        Path("results/e11_condition_score_v5_protocol/final_interpretation_plan") / "final_split_status.csv",
        Path("results/e11_condition_score_v5_protocol/final_interpretation_plan") / "final_gate_contract.csv",
        Path("results/e11_condition_score_v5_protocol/final_interpretation_plan") / "outcome_interpretation_ladder.csv",
        Path("results/e11_condition_score_v5_protocol/final_interpretation_plan") / "leakage_lock.csv",
        Path("results/e11_condition_score_v5_protocol/final_interpretation_plan") / "config.json",
        Path("discussion/e11_condition_score_v5_final_interpretation_plan.md"),
        Path("scripts/e11_write_condition_score_v5_final_interpretation_plan.py"),
        Path("results/e11_condition_score_v5_protocol/reviewer_failure_response") / "final_split_output_status.csv",
        Path("results/e11_condition_score_v5_protocol/reviewer_failure_response") / "current_gate_snapshot.csv",
        Path("results/e11_condition_score_v5_protocol/reviewer_failure_response") / "active_failure_modes.csv",
        Path("results/e11_condition_score_v5_protocol/reviewer_failure_response") / "failure_mode_register.csv",
        Path("results/e11_condition_score_v5_protocol/reviewer_failure_response") / "reviewer_objection_map.csv",
        Path("results/e11_condition_score_v5_protocol/reviewer_failure_response") / "claim_downgrade_actions.csv",
        Path("results/e11_condition_score_v5_protocol/reviewer_failure_response") / "next_evidence_queue.csv",
        Path("results/e11_condition_score_v5_protocol/reviewer_failure_response") / "config.json",
        Path("discussion/e11_condition_score_v5_reviewer_failure_response.md"),
        Path("scripts/e11_write_condition_score_v5_reviewer_failure_response.py"),
        Path("results/e11_condition_score_v5_protocol/direction_guardrail_failure_audit") / "score_axis_contrast.csv",
        Path("results/e11_condition_score_v5_protocol/direction_guardrail_failure_audit") / "gate_boundary_summary.csv",
        Path("results/e11_condition_score_v5_protocol/direction_guardrail_failure_audit") / "mechanistic_diagnosis.csv",
        Path("results/e11_condition_score_v5_protocol/direction_guardrail_failure_audit") / "next_protocol_requirements.csv",
        Path("results/e11_condition_score_v5_protocol/direction_guardrail_failure_audit") / "config.json",
        Path("discussion/e11_condition_score_v5_direction_guardrail_failure_audit.md"),
        Path("scripts/e11_write_condition_score_v5_direction_guardrail_failure_audit.py"),
        Path("results/e11_natural_head_tail_boundary") / "search_registry.csv",
        Path("results/e11_natural_head_tail_boundary") / "primary_drift_scan.csv",
        Path("results/e11_natural_head_tail_boundary") / "secondary_outcome_scan.csv",
        Path("results/e11_natural_head_tail_boundary") / "boundary_summary.csv",
        Path("results/e11_natural_head_tail_boundary") / "candidate_negative_cases.csv",
        Path("discussion/e11_natural_head_tail_boundary.md"),
        Path("scripts/e11_write_natural_head_tail_boundary_audit.py"),
        Path("results/e11_natural_negative_search_protocol") / "audit_baseline.csv",
        Path("results/e11_natural_negative_search_protocol") / "search_space_registry.csv",
        Path("results/e11_natural_negative_search_protocol") / "metric_contract.csv",
        Path("results/e11_natural_negative_search_protocol") / "stopping_rules.csv",
        Path("results/e11_natural_negative_search_protocol") / "acceptance_gates.csv",
        Path("results/e11_natural_negative_search_protocol") / "claim_ladder.csv",
        Path("results/e11_natural_negative_search_protocol") / "protocol_status.csv",
        Path("results/e11_natural_negative_search_protocol/phase1_power_audit") / "power_grid.csv",
        Path("results/e11_natural_negative_search_protocol/phase1_power_audit") / "minimum_detectable_effect.csv",
        Path("results/e11_natural_negative_search_protocol/phase1_power_audit") / "interpretation_ladder.csv",
        Path("results/e11_natural_negative_search_protocol/phase1_cifar100lt_resnet18") / "settings_registry.csv",
        Path("results/e11_natural_negative_search_protocol/phase1_cifar100lt_resnet18") / "step_metrics.csv",
        Path("results/e11_natural_negative_search_protocol/phase1_cifar100lt_resnet18") / "pair_summary.csv",
        Path("results/e11_natural_negative_search_protocol/phase1_cifar100lt_resnet18") / "layer_metrics.csv",
        Path("results/e11_natural_negative_search_protocol/phase1_cifar100lt_resnet18") / "decision_template.csv",
        Path("results/e11_natural_negative_search_protocol/phase1_cifar100lt_resnet18") / "config.json",
        Path("results/e11_natural_negative_search_protocol/phase1_cifar10lt_resnet18") / "settings_registry.csv",
        Path("results/e11_natural_negative_search_protocol/phase1_cifar10lt_resnet18") / "step_metrics.csv",
        Path("results/e11_natural_negative_search_protocol/phase1_cifar10lt_resnet18") / "pair_summary.csv",
        Path("results/e11_natural_negative_search_protocol/phase1_cifar10lt_resnet18") / "layer_metrics.csv",
        Path("results/e11_natural_negative_search_protocol/phase1_cifar10lt_resnet18") / "decision_template.csv",
        Path("results/e11_natural_negative_search_protocol/phase1_cifar10lt_resnet18") / "config.json",
        Path("results/e11_natural_negative_search_protocol/phase1_tail_quality_controls") / "settings_registry.csv",
        Path("results/e11_natural_negative_search_protocol/phase1_multiplicity_evaluation") / "run_registry.csv",
        Path("results/e11_natural_negative_search_protocol/phase1_multiplicity_evaluation") / "seed_level_primary_ratios.csv",
        Path("results/e11_natural_negative_search_protocol/phase1_multiplicity_evaluation") / "primary_decisions.csv",
        Path("results/e11_natural_negative_search_protocol/phase1_multiplicity_evaluation") / "gate_report.csv",
        Path("results/e11_natural_negative_search_protocol/phase1_multiplicity_evaluation") / "config.json",
        Path("results/e11_natural_negative_search_protocol/phase1_interim_synthesis") / "family_coverage.csv",
        Path("results/e11_natural_negative_search_protocol/phase1_interim_synthesis") / "observed_primary_summary.csv",
        Path("results/e11_natural_negative_search_protocol/phase1_interim_synthesis") / "claim_boundary.csv",
        Path("results/e11_natural_negative_search_protocol/phase1_interim_synthesis") / "remaining_work.csv",
        Path("results/e11_natural_negative_search_protocol/phase1_interim_synthesis") / "config.json",
        Path("discussion/e11_natural_negative_search_protocol.md"),
        Path("discussion/e11_natural_negative_search_phase1_power_audit.md"),
        Path("discussion/e11_natural_negative_search_phase1_NNS-P1-cifar100lt-resnet18-new-partitions.md"),
        Path("discussion/e11_natural_negative_search_phase1_NNS-P1-cifar10lt-resnet18-cross-partitions.md"),
        Path("discussion/e11_natural_negative_search_phase1_evaluation.md"),
        Path("discussion/e11_natural_negative_search_phase1_interim_synthesis.md"),
        Path("scripts/e11_write_natural_negative_search_protocol.py"),
        Path("scripts/e11_write_natural_negative_power_audit.py"),
        Path("scripts/e11_run_natural_negative_search_phase1.py"),
        Path("scripts/e11_evaluate_natural_negative_search_phase1.py"),
        Path("scripts/e11_write_natural_negative_phase1_interim_synthesis.py"),
        Path("scripts/slurm/e11_natural_negative_search_phase1.sbatch"),
        Path("discussion/e11_condition_score_v4_protocol.md"),
        Path("discussion/e11_condition_score_v4_validation_cifar100_rotated.md"),
        Path("discussion/e11_condition_score_v4_validation_freeze.md"),
        Path("discussion/e11_condition_score_v4_architecture_wide_resnet50_2.md"),
        Path("discussion/e11_condition_score_v4_data_cifar10_mixed.md"),
        Path("discussion/e11_condition_score_v4_final_evaluation.md"),
        Path("scripts/e11_write_condition_score_v4_protocol.py"),
        Path("scripts/e11_freeze_condition_score_v4_validation.py"),
        Path("scripts/e11_evaluate_condition_score_v4_finals.py"),
        Path("scripts/slurm/e11_cifar100_resnet_condition_score_v4_validation_cifar100_rotated.sbatch"),
        Path("scripts/slurm/e11_cifar100_resnet_condition_score_v4_architecture_wide_resnet50_2.sbatch"),
        Path("scripts/slurm/e11_cifar100_resnet_condition_score_v4_data_cifar10_mixed.sbatch"),
        Path("scripts/slurm/e11_cifar100_resnet_condition_score_v5_validation_mod4_partition.sbatch"),
        Path("scripts/slurm/e11_cifar100_resnet_condition_score_v5_architecture_resnext50_32x4d.sbatch"),
        Path("scripts/slurm/e11_cifar100_resnet_condition_score_v5_data_cifar10_cross.sbatch"),
        Path("results/e11_cifar100_resnet_lt_standard_eval") / "train_trace.csv",
        Path("results/e11_cifar100_resnet_lt_standard_eval") / "class_metrics.csv",
        Path("results/e11_cifar100_resnet_lt_standard_eval") / "group_metrics.csv",
        Path("results/e11_cifar100_resnet_lt_standard_eval") / "summary.csv",
        Path("results/e11_cifar100_resnet_lt_standard_eval") / "class_summary.csv",
        Path("results/e11_cifar100_resnet_lt_standard_eval") / "config.json",
        Path("figures/e11_cifar100_resnet_lt_standard_eval") / "cifar100_resnet_lt_standard_eval.png",
        Path("discussion/e11_cifar100_resnet_lt_standard_eval.md"),
        Path("results/e11_cifar100_resnet_lt_recipe_benchmark") / "train_trace.csv",
        Path("results/e11_cifar100_resnet_lt_recipe_benchmark") / "class_metrics.csv",
        Path("results/e11_cifar100_resnet_lt_recipe_benchmark") / "group_metrics.csv",
        Path("results/e11_cifar100_resnet_lt_recipe_benchmark") / "summary.csv",
        Path("results/e11_cifar100_resnet_lt_recipe_benchmark") / "pair_summary.csv",
        Path("results/e11_cifar100_resnet_lt_recipe_benchmark") / "config.json",
        Path("figures/e11_cifar100_resnet_lt_recipe_benchmark") / "cifar100_resnet_lt_recipe_benchmark.png",
        Path("discussion/e11_cifar100_resnet_lt_recipe_benchmark.md"),
        Path("results/e11_cifar100_resnet_lt_muon_final_benchmark") / "train_trace.csv",
        Path("results/e11_cifar100_resnet_lt_muon_final_benchmark") / "class_metrics.csv",
        Path("results/e11_cifar100_resnet_lt_muon_final_benchmark") / "group_metrics.csv",
        Path("results/e11_cifar100_resnet_lt_muon_final_benchmark") / "summary.csv",
        Path("results/e11_cifar100_resnet_lt_muon_final_benchmark") / "pair_summary.csv",
        Path("results/e11_cifar100_resnet_lt_muon_final_benchmark") / "config.json",
        Path("figures/e11_cifar100_resnet_lt_muon_final_benchmark") / "cifar100_resnet_lt_recipe_benchmark.png",
        Path("discussion/e11_cifar100_resnet_lt_muon_final_benchmark.md"),
        Path("results/e11_cifar100_resnet_lt_tuned_benchmark_protocol") / "pilot_context.csv",
        Path("results/e11_cifar100_resnet_lt_tuned_benchmark_protocol") / "benchmark_scope.csv",
        Path("results/e11_cifar100_resnet_lt_tuned_benchmark_protocol") / "seed_split_contract.csv",
        Path("results/e11_cifar100_resnet_lt_tuned_benchmark_protocol") / "recipe_grid.csv",
        Path("results/e11_cifar100_resnet_lt_tuned_benchmark_protocol") / "selection_rules.csv",
        Path("results/e11_cifar100_resnet_lt_tuned_benchmark_protocol") / "acceptance_gates.csv",
        Path("discussion/e11_cifar100_resnet_lt_tuned_benchmark_protocol.md"),
        Path("scripts/e11_write_cifar100_resnet_lt_tuned_benchmark_protocol.py"),
        Path("results/e11_cifar100_resnet_lt_tuned_benchmark") / "settings_registry.csv",
        Path("results/e11_cifar100_resnet_lt_tuned_benchmark") / "execution_status.csv",
        Path("results/e11_cifar100_resnet_lt_tuned_benchmark/validation_selection") / "run_registry.csv",
        Path("results/e11_cifar100_resnet_lt_tuned_benchmark/validation_selection") / "family_selection.csv",
        Path("results/e11_cifar100_resnet_lt_tuned_benchmark/validation_selection") / "final_claim_plan.csv",
        Path("results/e11_cifar100_resnet_lt_tuned_benchmark/validation_selection") / "gate_report.csv",
        Path("results/e11_cifar100_resnet_lt_tuned_benchmark/validation_selection") / "config.json",
        Path("discussion/e11_cifar100_resnet_lt_tuned_benchmark_selection.md"),
        Path("results/e11_cifar100_resnet_lt_tuned_benchmark/interim_validation_audit")
        / "completed_setting_leaderboard.csv",
        Path("results/e11_cifar100_resnet_lt_tuned_benchmark/interim_validation_audit") / "family_progress.csv",
        Path("results/e11_cifar100_resnet_lt_tuned_benchmark/interim_validation_audit")
        / "occupancy_interim_summary.csv",
        Path("results/e11_cifar100_resnet_lt_tuned_benchmark/interim_validation_audit")
        / "partial_grid_guardrail.csv",
        Path("results/e11_cifar100_resnet_lt_tuned_benchmark/interim_validation_audit")
        / "claim_boundary_gates.csv",
        Path("results/e11_cifar100_resnet_lt_tuned_benchmark/interim_validation_audit") / "config.json",
        Path("discussion/e11_cifar100_resnet_lt_tuned_benchmark_interim_audit.md"),
        Path("results/e11_cifar100_resnet_lt_tuned_benchmark/validation_leakage_audit")
        / "leakage_guard_matrix.csv",
        Path("results/e11_cifar100_resnet_lt_tuned_benchmark/validation_leakage_audit") / "observed_surface.csv",
        Path("results/e11_cifar100_resnet_lt_tuned_benchmark/validation_leakage_audit")
        / "immutability_contract.csv",
        Path("results/e11_cifar100_resnet_lt_tuned_benchmark/validation_leakage_audit") / "config.json",
        Path("discussion/e11_cifar100_resnet_lt_tuned_benchmark_leakage_audit.md"),
        Path("results/e11_cifar100_resnet_lt_tuned_benchmark/validation_refresh_firewall")
        / "refresh_state.csv",
        Path("results/e11_cifar100_resnet_lt_tuned_benchmark/validation_refresh_firewall")
        / "allowed_transition_matrix.csv",
        Path("results/e11_cifar100_resnet_lt_tuned_benchmark/validation_refresh_firewall")
        / "forbidden_action_matrix.csv",
        Path("results/e11_cifar100_resnet_lt_tuned_benchmark/validation_refresh_firewall") / "config.json",
        Path("discussion/e11_cifar100_resnet_lt_tuned_benchmark_refresh_firewall.md"),
        Path("scripts/e11_write_cifar100_resnet_lt_tuned_benchmark_refresh_firewall.py"),
        Path("results/e11_cifar100_resnet_lt_tuned_benchmark/final_power_audit") / "final_family_design.csv",
        Path("results/e11_cifar100_resnet_lt_tuned_benchmark/final_power_audit") / "primary_comparison_plan.csv",
        Path("results/e11_cifar100_resnet_lt_tuned_benchmark/final_power_audit") / "paired_diff_mde.csv",
        Path("results/e11_cifar100_resnet_lt_tuned_benchmark/final_power_audit") / "all_class_guardrail_mde.csv",
        Path("results/e11_cifar100_resnet_lt_tuned_benchmark/final_power_audit") / "interpretation_ladder.csv",
        Path("results/e11_cifar100_resnet_lt_tuned_benchmark/final_power_audit") / "outcome_state_machine.csv",
        Path("results/e11_cifar100_resnet_lt_tuned_benchmark/final_power_audit") / "config.json",
        Path("discussion/e11_cifar100_resnet_lt_tuned_benchmark_power_audit.md"),
        Path("results/e11_cifar100_resnet_lt_tuned_benchmark/variance_prior_audit")
        / "validation_setting_variance.csv",
        Path("results/e11_cifar100_resnet_lt_tuned_benchmark/variance_prior_audit")
        / "spent_pilot_paired_variance.csv",
        Path("results/e11_cifar100_resnet_lt_tuned_benchmark/variance_prior_audit")
        / "variance_prior_summary.csv",
        Path("results/e11_cifar100_resnet_lt_tuned_benchmark/variance_prior_audit")
        / "mde_sensitivity_from_empirical_sd.csv",
        Path("results/e11_cifar100_resnet_lt_tuned_benchmark/variance_prior_audit") / "gate_matrix.csv",
        Path("results/e11_cifar100_resnet_lt_tuned_benchmark/variance_prior_audit") / "config.json",
        Path("discussion/e11_cifar100_resnet_lt_tuned_benchmark_variance_prior_audit.md"),
        Path("scripts/e11_write_cifar100_resnet_lt_tuned_benchmark_variance_prior_audit.py"),
        Path("results/e11_cifar100_resnet_lt_tuned_benchmark/final_analysis_plan")
        / "analysis_input_contract.csv",
        Path("results/e11_cifar100_resnet_lt_tuned_benchmark/final_analysis_plan") / "metric_contract.csv",
        Path("results/e11_cifar100_resnet_lt_tuned_benchmark/final_analysis_plan")
        / "primary_comparison_family.csv",
        Path("results/e11_cifar100_resnet_lt_tuned_benchmark/final_analysis_plan")
        / "multiplicity_and_guardrail_plan.csv",
        Path("results/e11_cifar100_resnet_lt_tuned_benchmark/final_analysis_plan") / "reporting_schema.csv",
        Path("results/e11_cifar100_resnet_lt_tuned_benchmark/final_analysis_plan") / "claim_ladder.csv",
        Path("results/e11_cifar100_resnet_lt_tuned_benchmark/final_analysis_plan") / "gate_matrix.csv",
        Path("results/e11_cifar100_resnet_lt_tuned_benchmark/final_analysis_plan") / "config.json",
        Path("discussion/e11_cifar100_resnet_lt_tuned_benchmark_final_analysis_plan.md"),
        Path("scripts/e11_write_cifar100_resnet_lt_tuned_benchmark_final_analysis_plan.py"),
        Path("results/e11_cifar100_resnet_lt_tuned_benchmark/final_execution_plan") / "gate_matrix.csv",
        Path("results/e11_cifar100_resnet_lt_tuned_benchmark/final_execution_plan")
        / "final_family_run_plan.csv",
        Path("results/e11_cifar100_resnet_lt_tuned_benchmark/final_execution_plan") / "slurm_submit_plan.csv",
        Path("results/e11_cifar100_resnet_lt_tuned_benchmark/final_execution_plan") / "config.json",
        Path("discussion/e11_cifar100_resnet_lt_tuned_benchmark_final_execution_plan.md"),
        Path("scripts/e11_write_cifar100_resnet_lt_tuned_benchmark_final_execution_plan.py"),
        Path("scripts/slurm/e11_cifar100_resnet_lt_tuned_benchmark_final.sbatch"),
        Path("results/e11_cifar100_resnet_lt_tuned_benchmark/final_evaluation") / "run_registry.csv",
        Path("results/e11_cifar100_resnet_lt_tuned_benchmark/final_evaluation")
        / "per_seed_final_metrics.csv",
        Path("results/e11_cifar100_resnet_lt_tuned_benchmark/final_evaluation")
        / "paired_primary_comparisons.csv",
        Path("results/e11_cifar100_resnet_lt_tuned_benchmark/final_evaluation") / "primary_decisions.csv",
        Path("results/e11_cifar100_resnet_lt_tuned_benchmark/final_evaluation") / "final_summary.csv",
        Path("results/e11_cifar100_resnet_lt_tuned_benchmark/final_evaluation") / "occupancy_summary.csv",
        Path("results/e11_cifar100_resnet_lt_tuned_benchmark/final_evaluation") / "claim_gate_report.csv",
        Path("results/e11_cifar100_resnet_lt_tuned_benchmark/final_evaluation") / "config.json",
        Path("discussion/e11_cifar100_resnet_lt_tuned_benchmark_final_evaluation.md"),
        Path("scripts/e11_evaluate_cifar100_resnet_lt_tuned_benchmark_final.py"),
        Path("results/e11_cifar100_resnet_lt_tuned_benchmark/final_launch_audit")
        / "latest_final_launch_decision.csv",
        Path("results/e11_cifar100_resnet_lt_tuned_benchmark/final_launch_audit")
        / "latest_final_family_plan.csv",
        Path("results/e11_cifar100_resnet_lt_tuned_benchmark/final_launch_audit") / "latest_gate_snapshot.csv",
        Path("results/e11_cifar100_resnet_lt_tuned_benchmark/final_launch_audit") / "latest_queue_snapshot.csv",
        Path("results/e11_cifar100_resnet_lt_tuned_benchmark/final_launch_audit") / "final_launch_history.csv",
        Path("results/e11_cifar100_resnet_lt_tuned_benchmark/final_launch_audit") / "config.json",
        Path("discussion/e11_cifar100_resnet_lt_tuned_benchmark_final_launch_audit.md"),
        Path("scripts/e11_submit_cifar100_resnet_lt_tuned_benchmark_final.py"),
        Path("results/e11_cifar100_resnet_lt_tuned_benchmark/slurm_submission_plan") / "chunk_plan.csv",
        Path("results/e11_cifar100_resnet_lt_tuned_benchmark/slurm_submission_plan") / "queue_policy.csv",
        Path("results/e11_cifar100_resnet_lt_tuned_benchmark/slurm_submission_plan") / "config.json",
        Path("discussion/e11_cifar100_resnet_lt_tuned_benchmark_slurm_plan.md"),
        Path("results/e11_cifar100_resnet_lt_tuned_benchmark/slurm_launch_audit") / "latest_launch_decision.csv",
        Path("results/e11_cifar100_resnet_lt_tuned_benchmark/slurm_launch_audit") / "latest_selected_settings.csv",
        Path("results/e11_cifar100_resnet_lt_tuned_benchmark/slurm_launch_audit") / "latest_queue_snapshot.csv",
        Path("results/e11_cifar100_resnet_lt_tuned_benchmark/slurm_launch_audit") / "launch_history.csv",
        Path("results/e11_cifar100_resnet_lt_tuned_benchmark/slurm_launch_audit") / "config.json",
        Path("discussion/e11_cifar100_resnet_lt_tuned_benchmark_slurm_launch_audit.md"),
        Path("scripts/e11_run_cifar100_resnet_lt_tuned_benchmark.py"),
        Path("scripts/e11_write_cifar100_resnet_lt_tuned_benchmark_settings.py"),
        Path("scripts/e11_write_cifar100_resnet_lt_tuned_benchmark_selection.py"),
        Path("scripts/e11_write_cifar100_resnet_lt_tuned_benchmark_interim_audit.py"),
        Path("scripts/e11_write_cifar100_resnet_lt_tuned_benchmark_leakage_audit.py"),
        Path("scripts/e11_write_cifar100_resnet_lt_tuned_benchmark_power_audit.py"),
        Path("scripts/e11_write_cifar100_resnet_lt_tuned_benchmark_slurm_plan.py"),
        Path("scripts/e11_submit_cifar100_resnet_lt_tuned_benchmark_validation.py"),
        Path("scripts/slurm/e11_cifar100_resnet_lt_tuned_benchmark_validation.sbatch"),
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
        Path("paper/specgrad_activation_paper/figures") / "cifar100_resnet_condition_score_audit.png",
        Path("paper/specgrad_activation_paper/figures") / "cifar100_resnet_imbalance_sweep.png",
        Path("paper/specgrad_activation_paper/figures") / "cifar100_resnet_lt_standard_eval.png",
        Path("paper/specgrad_activation_paper/figures") / "cifar100_resnet_lt_recipe_benchmark.png",
        Path("paper/specgrad_activation_paper/figures") / "cifar100_resnet_lt_muon_final_benchmark.png",
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
        Path("results/e11_top_conference_gap_register") / "gap_register.csv",
        Path("discussion/e11_top_conference_gap_register.md"),
        Path("results/e11_top_conference_claim_decision_audit") / "claim_decision_matrix.csv",
        Path("results/e11_top_conference_claim_decision_audit") / "reviewer_objection_matrix.csv",
        Path("results/e11_top_conference_claim_decision_audit") / "rebuttal_response_pack.csv",
        Path("results/e11_top_conference_claim_decision_audit") / "manuscript_edit_queue.csv",
        Path("results/e11_top_conference_claim_decision_audit") / "paper_sequence.csv",
        Path("results/e11_top_conference_claim_decision_audit") / "readiness_summary.csv",
        Path("results/e11_top_conference_claim_decision_audit") / "config.json",
        Path("discussion/e11_top_conference_claim_decision_audit.md"),
        Path("scripts/e11_write_top_conference_claim_decision_audit.py"),
        Path("discussion/e11_manuscript_claim_trace.md"),
        Path("results/e11_manuscript_claim_trace") / "claim_trace.csv",
        Path("results/e11_manuscript_claim_trace") / "blocked_phrase_audit.csv",
        Path("results/e11_manuscript_claim_trace") / "config.json",
        Path("scripts/e11_write_manuscript_claim_trace.py"),
        Path("discussion/e11_clean_worktree_replay_audit.md"),
        Path("results/e11_clean_worktree_replay_audit") / "run_summary.csv",
        Path("results/e11_clean_worktree_replay_audit") / "gate_matrix.csv",
        Path("results/e11_clean_worktree_replay_audit") / "command_log_tail.txt",
        Path("results/e11_clean_worktree_replay_audit") / "config.json",
        Path("scripts/e11_write_clean_worktree_replay_audit.py"),
        Path("discussion/e11_pdf_render_boundary_audit.md"),
        Path("results/e11_pdf_render_boundary_audit") / "pdf_inspection_tool_status.csv",
        Path("results/e11_pdf_render_boundary_audit") / "render_boundary_gates.csv",
        Path("results/e11_pdf_render_boundary_audit") / "config.json",
        Path("scripts/e11_write_pdf_render_boundary_audit.py"),
        Path("discussion/e11_mechanism_referee_audit.md"),
        Path("results/e11_mechanism_referee_audit") / "alternative_explanation_matrix.csv",
        Path("results/e11_mechanism_referee_audit") / "theory_measurement_contract.csv",
        Path("results/e11_mechanism_referee_audit") / "falsification_trigger_matrix.csv",
        Path("results/e11_mechanism_referee_audit") / "config.json",
        Path("scripts/e11_write_mechanism_referee_audit.py"),
        Path("discussion/e11_bold_conjecture_register.md"),
        Path("results/e11_bold_conjecture_register") / "conjecture_register.csv",
        Path("results/e11_bold_conjecture_register") / "stress_test_matrix.csv",
        Path("results/e11_bold_conjecture_register") / "claim_upgrade_ladder.csv",
        Path("results/e11_bold_conjecture_register") / "config.json",
        Path("scripts/e11_write_bold_conjecture_register.py"),
        Path("discussion/e11_muon_state_distribution_contract.md"),
        Path("results/e11_muon_state_distribution_contract") / "state_distribution_terms.csv",
        Path("results/e11_muon_state_distribution_contract") / "evidence_link_matrix.csv",
        Path("results/e11_muon_state_distribution_contract") / "falsification_tests.csv",
        Path("results/e11_muon_state_distribution_contract") / "claim_gate_ladder.csv",
        Path("results/e11_muon_state_distribution_contract") / "config.json",
        Path("scripts/e11_write_muon_state_distribution_contract.py"),
        Path("scripts/e11_write_condition_score_ablation.py"),
        Path("discussion/e11_condition_score_ablation.md"),
        Path("results/e11_condition_score_ablation") / "score_ablation_summary.csv",
        Path("results/e11_condition_score_ablation") / "term_failure_ladder.csv",
        Path("results/e11_condition_score_ablation") / "leakage_and_claim_boundary.csv",
        Path("results/e11_condition_score_ablation") / "config.json",
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
        Path("discussion/e11_submission_repro_audit.md"),
        Path("discussion/e11_artifact_manifest.md"),
        Path("results/e11_artifact_manifest.json"),
        Path("results/e11_submission_repro_audit") / "toolchain_status.csv",
        Path("results/e11_submission_repro_audit") / "pdf_artifact_checks.csv",
        Path("results/e11_submission_repro_audit") / "source_package_manifest.csv",
        Path("results/e11_submission_repro_audit") / "build_gate_summary.csv",
        Path("scripts/e11_write_submission_repro_audit.py"),
        Path("discussion/e11_artifact_review_packet.md"),
        Path("results/e11_artifact_review_packet") / "command_matrix.csv",
        Path("results/e11_artifact_review_packet") / "gate_matrix.csv",
        Path("results/e11_artifact_review_packet") / "local_state_contract.csv",
        Path("results/e11_artifact_review_packet") / "reviewer_response.csv",
        Path("results/e11_artifact_review_packet") / "config.json",
        Path("scripts/e11_write_artifact_review_packet.py"),
        Path("discussion/e11_camera_ready_package_audit.md"),
        Path("results/e11_camera_ready_package_audit") / "package_item_matrix.csv",
        Path("results/e11_camera_ready_package_audit") / "submission_gate_matrix.csv",
        Path("results/e11_camera_ready_package_audit") / "camera_ready_checklist.csv",
        Path("results/e11_camera_ready_package_audit") / "config.json",
        Path("scripts/e11_write_camera_ready_package_audit.py"),
        Path("scripts/e11_write_natural_head_tail_boundary_audit.py"),
        Path("scripts/e11_write_natural_negative_search_protocol.py"),
        Path("scripts/e11_run_natural_negative_search_phase1.py"),
        Path("scripts/e11_evaluate_natural_negative_search_phase1.py"),
        Path("scripts/slurm/e11_natural_negative_search_phase1.sbatch"),
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
        "e11-natural-head-tail-boundary-audit:",
        "scripts/e11_write_natural_head_tail_boundary_audit.py",
        "e11-natural-negative-search-protocol:",
        "scripts/e11_write_natural_negative_search_protocol.py",
        "e11-natural-negative-search-phase1-power-audit:",
        "scripts/e11_write_natural_negative_power_audit.py",
        "e11-natural-negative-search-phase1-settings:",
        "scripts/e11_run_natural_negative_search_phase1.py --settings-only",
        "e11-natural-negative-search-phase1-results:",
        "scripts/e11_run_natural_negative_search_phase1.py",
        "scripts/slurm/e11_natural_negative_search_phase1.sbatch",
        "e11-matrix-block-tightness-audit:",
        "scripts/e11_write_matrix_block_tightness_audit.py",
        "e11-natural-negative-search-phase1-eval:",
        "scripts/e11_evaluate_natural_negative_search_phase1.py",
        "e11-natural-negative-search-phase1-interim-synthesis:",
        "scripts/e11_write_natural_negative_phase1_interim_synthesis.py",
        "e11-submission-repro-audit:",
        "e11-clean-worktree-replay-audit:",
        "scripts/e11_write_clean_worktree_replay_audit.py",
        "e11-pdf-render-boundary-audit:",
        "scripts/e11_write_pdf_render_boundary_audit.py",
        "scripts/e11_write_submission_repro_audit.py",
        "e11-artifact-review-packet:",
        "scripts/e11_write_artifact_review_packet.py",
        "e11-top-conference-claim-decision-audit:",
        "scripts/e11_write_top_conference_claim_decision_audit.py",
        "e11-mechanism-referee-audit:",
        "scripts/e11_write_mechanism_referee_audit.py",
        "e11-cifar-resnet-lt-muon-final-benchmark-results:",
        "e11-cifar-resnet-lt-tuned-benchmark-protocol:",
        "scripts/e11_write_cifar100_resnet_lt_tuned_benchmark_protocol.py",
        "e11-cifar-resnet-lt-tuned-benchmark-settings:",
        "scripts/e11_run_cifar100_resnet_lt_tuned_benchmark.py --settings-only",
        "e11-cifar-resnet-lt-tuned-benchmark-validation-results:",
        "scripts/slurm/e11_cifar100_resnet_lt_tuned_benchmark_validation.sbatch",
        "e11-cifar-resnet-lt-tuned-benchmark-selection:",
        "scripts/e11_write_cifar100_resnet_lt_tuned_benchmark_selection.py",
        "e11-cifar-resnet-lt-tuned-benchmark-interim-audit:",
        "scripts/e11_write_cifar100_resnet_lt_tuned_benchmark_interim_audit.py",
        "e11-cifar-resnet-lt-tuned-benchmark-leakage-audit:",
        "scripts/e11_write_cifar100_resnet_lt_tuned_benchmark_leakage_audit.py",
        "e11-cifar-resnet-lt-tuned-benchmark-power-audit:",
        "scripts/e11_write_cifar100_resnet_lt_tuned_benchmark_power_audit.py",
        "e11-cifar-resnet-lt-tuned-benchmark-variance-prior-audit:",
        "scripts/e11_write_cifar100_resnet_lt_tuned_benchmark_variance_prior_audit.py",
        "e11-cifar-resnet-lt-tuned-benchmark-final-analysis-plan:",
        "scripts/e11_write_cifar100_resnet_lt_tuned_benchmark_final_analysis_plan.py",
        "e11-cifar-resnet-lt-tuned-benchmark-final-execution-plan:",
        "scripts/e11_write_cifar100_resnet_lt_tuned_benchmark_final_execution_plan.py",
        "scripts/slurm/e11_cifar100_resnet_lt_tuned_benchmark_final.sbatch",
        "e11-cifar-resnet-lt-tuned-benchmark-final-eval:",
        "scripts/e11_evaluate_cifar100_resnet_lt_tuned_benchmark_final.py",
        "e11-cifar-resnet-lt-tuned-benchmark-final-launch-audit:",
        "scripts/e11_submit_cifar100_resnet_lt_tuned_benchmark_final.py",
        "e11-cifar-resnet-lt-tuned-benchmark-final-safe-submit:",
        "scripts/e11_submit_cifar100_resnet_lt_tuned_benchmark_final.py --submit",
        "e11-cifar-resnet-imbalance-sweep-results:",
        "e11-cifar-resnet-condition-score-heldout-architecture-results:",
        "scripts/slurm/e11_cifar100_resnet_condition_score_heldout_architecture.sbatch",
        "e11-cifar-resnet-condition-score-heldout-data-results:",
        "scripts/slurm/e11_cifar100_resnet_condition_score_heldout_data.sbatch",
        "e11-cifar-resnet-condition-score-heldout-eval:",
        "scripts/e11_evaluate_cifar100_resnet_condition_score_heldouts.py",
        "e11-cifar-resnet-condition-score-fresh-architecture-results:",
        "scripts/slurm/e11_cifar100_resnet_condition_score_fresh_architecture_resnet50.sbatch",
        "e11-cifar-resnet-condition-score-fresh-data-results:",
        "scripts/slurm/e11_cifar100_resnet_condition_score_fresh_data_cifar10_alt.sbatch",
        "e11-cifar-resnet-condition-score-fresh-eval:",
        "scripts/e11_evaluate_condition_score_fresh_protocol.py",
        "e11-cifar-resnet-condition-score-failure-audit:",
        "scripts/e11_write_condition_score_failure_mechanism_audit.py",
        "e11-cifar-resnet-condition-score-v4-protocol:",
        "scripts/e11_write_condition_score_v4_protocol.py",
        "e11-cifar-resnet-condition-score-v4-validation-results:",
        "scripts/slurm/e11_cifar100_resnet_condition_score_v4_validation_cifar100_rotated.sbatch",
        "e11-cifar-resnet-condition-score-v4-validation-freeze:",
        "scripts/e11_freeze_condition_score_v4_validation.py",
        "e11-cifar-resnet-condition-score-v4-architecture-results:",
        "scripts/slurm/e11_cifar100_resnet_condition_score_v4_architecture_wide_resnet50_2.sbatch",
        "e11-cifar-resnet-condition-score-v4-data-results:",
        "scripts/slurm/e11_cifar100_resnet_condition_score_v4_data_cifar10_mixed.sbatch",
        "e11-cifar-resnet-condition-score-v4-final-eval:",
        "scripts/e11_evaluate_condition_score_v4_finals.py",
        "e11-cifar-resnet-condition-score-v4-failure-audit:",
        "scripts/e11_write_condition_score_v4_failure_mechanism_audit.py",
        "e11-cifar-resnet-condition-score-v5-theory-protocol:",
        "scripts/e11_write_condition_score_v5_theory_protocol.py",
        "e11-cifar-resnet-condition-score-v5-theory-to-score-map:",
        "scripts/e11_write_condition_score_v5_theory_to_score_map.py",
        "e11-cifar-resnet-condition-score-v5-validation-results:",
        "scripts/slurm/e11_cifar100_resnet_condition_score_v5_validation_mod4_partition.sbatch",
        "e11-cifar-resnet-condition-score-v5-validation-freeze:",
        "scripts/e11_freeze_condition_score_v5_validation.py",
        "e11-cifar-resnet-condition-score-v5-architecture-results:",
        "scripts/slurm/e11_cifar100_resnet_condition_score_v5_architecture_resnext50_32x4d.sbatch",
        "e11-cifar-resnet-condition-score-v5-data-results:",
        "scripts/slurm/e11_cifar100_resnet_condition_score_v5_data_cifar10_cross.sbatch",
        "e11-cifar-resnet-condition-score-v5-final-eval:",
        "scripts/e11_evaluate_condition_score_v5_finals.py",
        "e11-cifar-resnet-condition-score-v5-final-power-audit:",
        "scripts/e11_write_condition_score_v5_final_power_audit.py",
        "e11-cifar-resnet-condition-score-v5-final-interpretation-plan:",
        "scripts/e11_write_condition_score_v5_final_interpretation_plan.py",
        "e11-cifar-resnet-condition-score-v5-reviewer-failure-response:",
        "scripts/e11_write_condition_score_v5_reviewer_failure_response.py",
        "e11-cifar-resnet-condition-score-v5-direction-guardrail-failure-audit:",
        "scripts/e11_write_condition_score_v5_direction_guardrail_failure_audit.py",
        "e11-bold-conjecture-register:",
        "scripts/e11_write_bold_conjecture_register.py",
        "e11-muon-state-distribution-contract:",
        "scripts/e11_write_muon_state_distribution_contract.py",
        "e11-camera-ready-package-audit:",
        "scripts/e11_write_camera_ready_package_audit.py",
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
    paper_makefile_text = Path("paper/specgrad_activation_paper/Makefile").read_text(encoding="utf-8")
    required_paper_makefile_phrases = [
        "tectonic: main-tectonic two-page-tectonic",
        "$(TECTONIC) main.tex",
        "$(TECTONIC) two_page.tex",
    ]
    missing_paper_makefile_phrases = [
        phrase for phrase in required_paper_makefile_phrases if phrase not in paper_makefile_text
    ]
    if missing_paper_makefile_phrases:
        raise AssertionError(
            f"paper Makefile missing Tectonic fallback entries: {missing_paper_makefile_phrases}"
        )
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
        "make e11-cifar-resnet-imbalance-sweep-results # submit the CIFAR-100-LT ResNet18 tail-count imbalance sweep via Slurm",
        "make e11-cifar-resnet-layer-jvp-tail-quality-results # submit the all-layer ResNet finite-difference JVP tail-quality diagnostic via Slurm",
        "make e11-cifar-resnet-layer-jvp-checkpoint-prediction-results # submit the all-layer ResNet JVP checkpoint-transfer benchmark via Slurm",
        "make e11-cifar-resnet-condition-score-heldout-architecture-results # submit the registered ResNet34 held-out architecture condition-score split via Slurm",
        "make e11-cifar-resnet-condition-score-heldout-data-results # submit the registered CIFAR-10-LT held-out data condition-score split via Slurm",
        "make e11-cifar-resnet-condition-score-heldout-eval # evaluate frozen condition-score gates after both held-out Slurm jobs finish",
        "make e11-cifar-resnet-condition-score-fresh-architecture-results # submit the fresh ResNet50 condition-score architecture split via Slurm",
        "make e11-cifar-resnet-condition-score-fresh-data-results # submit the fresh CIFAR-10 alternate-partition condition-score data split via Slurm",
        "make e11-cifar-resnet-condition-score-fresh-eval # evaluate frozen fresh condition-score gates after fresh Slurm jobs finish",
        "make e11-cifar-resnet-condition-score-v4-validation-freeze # freeze or block the v4 scalar aggregation after the validation split",
        "make e11-cifar-resnet-condition-score-v4-final-eval # evaluate frozen v4 final gates after both unspent final Slurm jobs finish",
        "make e11-matrix-block-theorem-proof # write the matched-gain theorem/proof contract and sandwich rank derivation",
        "make e11-matrix-block-tightness-audit # verify theorem tightness, ratio identity, equality boundary, and degeneracy caveats",
        "make e11-theory-proof-obligation-register # map theorem assumptions, claim scope, and proof obligations before broad claims",
        "make e11-cifar-resnet-condition-score-v5-theory-protocol # write the v5 transport-normalized theory/score contract",
        "make e11-cifar-resnet-condition-score-v5-theory-to-score-map # map the v5 theorem terms to score features, leakage boundaries, and falsifiable gates",
        "make e11-bold-conjecture-register # generate the bold-conjecture/careful-verification register",
        "make e11-cifar-resnet-condition-score-v5-validation-results # submit the v5 validation-only CIFAR-100-LT mod-4 partition via Slurm",
        "make e11-cifar-resnet-condition-score-v5-validation-freeze # freeze or block the v5 transport-normalized score after validation",
        "make e11-cifar-resnet-condition-score-v5-architecture-results # submit the v5 ResNeXt50-32x4d final architecture split via Slurm",
        "make e11-cifar-resnet-condition-score-v5-data-results # submit the v5 CIFAR-10 cross-partition final data split via Slurm",
        "make e11-cifar-resnet-condition-score-v5-final-eval # evaluate frozen v5 final gates after both unspent final Slurm jobs finish",
        "make e11-cifar-resnet-condition-score-v5-final-power-audit # pre-output detectable-effect audit for v5 final residual-Spearman gates",
        "make e11-cifar-resnet-condition-score-v5-final-interpretation-plan # lock the v5 final outcome-to-claim state machine before outputs exist",
        "make e11-cifar-resnet-condition-score-v5-reviewer-failure-response # map v5 final pass/fail modes to reviewer-safe claim downgrades",
        "make e11-cifar-resnet-condition-score-v5-direction-guardrail-failure-audit # diagnose completed v5 final boundary failures across residual ranking and direction threshold",
        "make e11-cifar-resnet-lt-standard-eval-results # submit the standard CIFAR-100-LT ResNet18 many/medium/few reporting baseline via Slurm",
        "make e11-cifar-resnet-lt-recipe-benchmark-results # submit the augmented CIFAR-100-LT ResNet18 recipe benchmark pilot via Slurm",
        "make e11-cifar-resnet-lt-muon-final-benchmark-results # submit the CIFAR-100-LT ResNet18 NS-Muon final-training benchmark pilot via Slurm",
        "make e11-cifar-resnet-lt-tuned-benchmark-protocol # register validation/final splits, tuned baselines, Muon grids, and benchmark claim gates",
        "make e11-cifar-resnet-lt-tuned-benchmark-settings # write the executable 164-setting tuned validation grid registry",
        "make e11-cifar-resnet-lt-tuned-benchmark-validation-results # submit the tuned validation grid via Slurm array",
        "make e11-cifar-resnet-lt-tuned-benchmark-selection # select final recipes from completed validation summaries without touching final seeds",
        "make e11-cifar-resnet-lt-tuned-benchmark-power-audit # lock tuned final seed MDE, Holm family, and all-class guardrail before final outputs",
        "make e11-cifar-resnet-lt-tuned-benchmark-variance-prior-audit # calibrate tuned benchmark MDE assumptions from validation and spent-pilot variance",
        "make e11-cifar-resnet-lt-tuned-benchmark-final-analysis-plan # pre-register final paired tests, Holm adjustment, reporting schema, and claim states",
        "make e11-cifar-resnet-lt-tuned-benchmark-final-execution-plan # write a no-side-effect final-claim execution contract and gate-checked Slurm wrapper plan",
        "make e11-cifar-resnet-lt-tuned-benchmark-final-eval # evaluate final paired seeds with fixed Holm tests and claim gates after final outputs exist",
        "make e11-cifar-resnet-lt-tuned-benchmark-final-launch-audit # compute queue-aware final-claim launch readiness without submitting",
        "make e11-cifar-resnet-lt-tuned-benchmark-final-safe-submit # submit final-claim jobs only if every final launch gate passes",
        "make e11-cifar-resnet-lt-tuned-benchmark-slurm-plan # write a chunked no-side-effect Slurm launch plan for the 164-setting validation grid",
        "make e11-cifar-resnet-lt-tuned-benchmark-launch-audit # compute the current queue-aware validation launch decision without submitting",
        "make e11-cifar-resnet-lt-tuned-benchmark-safe-submit # submit the largest safe validation subchunk under MaxSubmitJobsPerUser",
        "make e11-muon-state-distribution-contract # generate the Muon state-distribution/practical-performance boundary contract",
        "make e11-natural-head-tail-boundary-audit # scan committed natural matched-head-gain sweeps for primary drift and secondary boundary cases",
        "make e11-natural-negative-search-protocol # register fresh natural negative-search space, metrics, stopping rules, and claim gates",
        "make e11-natural-negative-search-phase1-power-audit # compute the phase1 detectable-effect and interpretation boundary",
        "make e11-natural-negative-search-phase1-settings # write settings-only registries for all registered phase1 natural negative-search settings",
        "make e11-natural-negative-search-phase1-results # submit the registered phase1 natural negative-search settings via Slurm",
        "make e11-natural-negative-search-phase1-eval # evaluate Holm-adjusted phase1 decisions after fresh metric outputs exist",
        "A ResNet final-layer downstream-aware condition diagnostic over 40 seed/checkpoint points has weakest mean `nrank(G_H) / srank(H_T)` score about `6.566`",
        "A tail-rich ResNet control with 300 tail-train examples per class reaches best pre-update tail accuracy about `0.3739 [0.3454, 0.4024]`",
        "A CIFAR-100-LT ResNet18 imbalance sweep over tail_train_per_class 10/30/100/300 keeps spectral/Frobenius squared drift ratio below 1 in every setting",
        "An all-layer ResNet finite-difference JVP tail-quality diagnostic covers 21 Conv/Linear weights and 210 paired layer/seed points",
        "An all-layer ResNet JVP checkpoint-transfer benchmark covers 3 tail-rich checkpoints and 6 directed checkpoint-transfer pairs",
        "ResNet34 and CIFAR-10-LT held-out Slurm entry points are now registered",
        "registered held-out condition-score evaluation now fails the P0",
        "ResNet34 held-out architecture primary residual Spearman is `0.1992 [-0.07515, 0.4735]`",
        "CIFAR-10-LT held-out data primary residual Spearman is `-0.6771 [-0.7011, -0.653]`",
        "legacy scaled-JVP ratio on CIFAR-10-LT has residual Spearman `0.6219 [0.6013, 0.6425]`",
        "discussion/e11_condition_score_heldout_failure_theory_note.md",
        "discussion/e11_condition_score_theory_bridge.md",
        "discussion/e11_condition_score_fresh_protocol.md",
        "discussion/e11_condition_score_fresh_evaluation.md",
        "discussion/e11_condition_score_failure_mechanism_audit.md",
        "discussion/e11_condition_score_v4_protocol.md",
        "discussion/e11_condition_score_v4_validation_cifar100_rotated.md",
        "discussion/e11_condition_score_v4_validation_freeze.md",
        "discussion/e11_condition_score_v4_architecture_wide_resnet50_2.md",
        "discussion/e11_condition_score_v4_data_cifar10_mixed.md",
        "discussion/e11_condition_score_v4_final_evaluation.md",
        "discussion/e11_condition_score_v4_failure_mechanism_audit.md",
        "discussion/e11_condition_score_v5_theory_protocol.md",
        "discussion/e11_condition_score_v5_theory_to_score_map.md",
        "discussion/e11_condition_score_ablation.md",
        "discussion/e11_condition_score_v5_validation_freeze.md",
        "discussion/e11_bold_conjecture_register.md",
        "results/e11_bold_conjecture_register/conjecture_register.csv",
        "results/e11_bold_conjecture_register/stress_test_matrix.csv",
        "bold-conjecture/careful-verification ledger",
        "discussion/e11_muon_state_distribution_contract.md",
        "results/e11_muon_state_distribution_contract/state_distribution_terms.csv",
        "results/e11_muon_state_distribution_contract/falsification_tests.csv",
        "state-distribution transport contract",
        "scripts/e11_evaluate_condition_score_fresh_protocol.py",
        "scripts/e11_write_condition_score_theory_bridge.py",
        "scripts/e11_write_condition_score_fresh_protocol.py",
        "scripts/e11_write_condition_score_failure_mechanism_audit.py",
        "scripts/e11_write_condition_score_v4_protocol.py",
        "scripts/e11_freeze_condition_score_v4_validation.py",
        "scripts/e11_evaluate_condition_score_v4_finals.py",
        "scripts/e11_write_condition_score_v4_failure_mechanism_audit.py",
        "scripts/e11_write_condition_score_v5_theory_protocol.py",
        "scripts/e11_write_condition_score_v5_theory_to_score_map.py",
        "scripts/e11_freeze_condition_score_v5_validation.py",
        "scripts/e11_evaluate_condition_score_v5_finals.py",
        "scripts/e11_write_condition_score_v5_final_evaluation.py",
        "scripts/e11_write_condition_score_v5_final_interpretation_plan.py",
        "scripts/e11_write_condition_score_v5_reviewer_failure_response.py",
        "scripts/slurm/e11_cifar100_resnet_condition_score_fresh_architecture_resnet50.sbatch",
        "scripts/slurm/e11_cifar100_resnet_condition_score_fresh_data_cifar10_alt.sbatch",
        "scripts/slurm/e11_cifar100_resnet_condition_score_v4_architecture_wide_resnet50_2.sbatch",
        "scripts/slurm/e11_cifar100_resnet_condition_score_v5_architecture_resnext50_32x4d.sbatch",
        "registers fresh final splits: ResNet50",
        "CIFAR-100-LT and a CIFAR-10 alternate head/tail partition",
        "current fresh v3 result is still `not_ready`",
        "architecture split fails and reverses",
        "`-0.8157 [-0.8652, -0.7661]`",
        "results/e11_condition_score_failure_mechanism_audit/resnet50_stage_reversal.csv",
        "wide_resnet50_2",
        "V4 is not a positive result yet",
        "validation-frozen scalar",
        "validation-freeze boundary",
        "validation split now exists",
        "condition_score_v4_two_axis_amplitude_minus_direction",
        "0.3758 [0.2759, 0.4756]",
        "committed validation-freeze boundary",
        "final evaluator is `scripts/e11_evaluate_condition_score_v4_finals.py`",
        "current gate report is `not_ready`",
        "WideResNet50-2 final architecture split passes residual ranking",
        "0.6449 [0.5122, 0.7776]",
        "CIFAR-10 mixed final data split fails",
        "-0.6937 [-0.7129, -0.6744]",
        "V4-O2-data-partition-reversal",
        "direction is not the failure",
        "amplitude/depth",
        "transport-normalized score contract",
        "ResNeXt50-32x4d",
        "data-partition reversal mechanism problem",
        "aggregation before either unspent final split",
        "transport-stable sandwich residual proposition",
        "theorem-to-measurement bridge",
        "results/e11_condition_score_v5_protocol/validation_score_freeze/*",
        "transport-normalized residual score",
        "discussion/e11_matrix_block_tightness_audit.md",
        "I_spectral / I_frobenius = ssrank(B_T,A_T) / nrank(G_H)",
        "discussion/e11_natural_head_tail_boundary.md",
        "results/e11_natural_head_tail_boundary/*",
        "scripts/e11_write_natural_head_tail_boundary_audit.py",
        "no_strict_natural_primary_counterexample_in_committed_scan",
        "37 primary full tail-output drift rows",
        "discussion/e11_natural_negative_search_protocol.md",
        "results/e11_natural_negative_search_protocol/*",
        "scripts/e11_write_natural_negative_search_protocol.py",
        "scripts/e11_write_natural_negative_power_audit.py",
        "discussion/e11_natural_negative_search_phase1_power_audit.md",
        "scripts/e11_run_natural_negative_search_phase1.py",
        "scripts/e11_evaluate_natural_negative_search_phase1.py",
        "scripts/slurm/e11_natural_negative_search_phase1.sbatch",
        "discussion/e11_natural_negative_search_phase1_NNS-P1-cifar100lt-resnet18-new-partitions.md",
        "scripts/e11_write_cifar100_resnet_lt_tuned_benchmark_protocol.py",
        "scripts/e11_run_cifar100_resnet_lt_tuned_benchmark.py",
        "scripts/e11_write_cifar100_resnet_lt_tuned_benchmark_selection.py",
        "scripts/e11_write_cifar100_resnet_lt_tuned_benchmark_power_audit.py",
        "scripts/slurm/e11_cifar100_resnet_lt_tuned_benchmark_validation.sbatch",
        "discussion/e11_cifar100_resnet_lt_tuned_benchmark_protocol.md",
        "discussion/e11_cifar100_resnet_lt_tuned_benchmark_selection.md",
        "discussion/e11_cifar100_resnet_lt_tuned_benchmark_refresh_firewall.md",
        "discussion/e11_cifar100_resnet_lt_tuned_benchmark_power_audit.md",
        "results/e11_cifar100_resnet_lt_tuned_benchmark_protocol",
        "results/e11_cifar100_resnet_lt_tuned_benchmark/settings_registry.csv",
        "results/e11_cifar100_resnet_lt_tuned_benchmark/validation_selection",
        "results/e11_cifar100_resnet_lt_tuned_benchmark/final_power_audit",
        "discussion/e11_cifar100_resnet_lt_tuned_benchmark_protocol_seal.md",
        "results/e11_cifar100_resnet_lt_tuned_benchmark/validation_protocol_seal/hash_manifest.csv",
        "discussion/e11_cifar100_resnet_lt_tuned_benchmark_fairness_audit.md",
        "results/e11_cifar100_resnet_lt_tuned_benchmark/validation_fairness_audit/fairness_gate_matrix.csv",
        "protocol hash seal",
        "fairness audit",
        "validation/final seed splits",
        "tuned AdamW/SGD/class-balanced baselines",
        "phase1 GPU entrypoint is now implemented",
        "26/26 primary metric rows",
        "discussion/e11_natural_negative_search_phase1_evaluation.md",
        "discussion/e11_natural_negative_search_phase1_interim_synthesis.md",
        "raw_worse_rows=0",
        "finite registered phase1 null candidate",
        "quality_gate_fail_rows=23",
        "make e11-natural-negative-search-phase1-power-audit",
        "make e11-natural-negative-search-phase1-results",
        "make e11-natural-negative-search-phase1-eval",
        "make e11-natural-negative-search-phase1-interim-synthesis",
        "multiplicity-adjusted decision rule",
        "before fresh search outputs",
        "frozen `condition_score_v2_calibrated_residual` coefficients",
        "source-observed positive-control Spearman",
        "candidate condition-score audit",
        "scaled-JVP residual Spearman",
        "A standard CIFAR-100-LT ResNet18 reporting baseline (IF=100, 10 AdamW seeds, no augmentation/tuning) gives many/medium/few balanced accuracy `0.3665 [0.3489, 0.3841]`, `0.1036 [0.09138, 0.1158]`, and `0.0129 [0.009351, 0.01645]`",
        "An augmented CIFAR-100-LT ResNet18 recipe benchmark pilot (5 seeds, 5000 steps) gives SGD-momentum all/few balanced accuracy `0.4105 [0.4044, 0.4166]` and `0.1047 [0.09389, 0.1156]`",
        "A CIFAR-100-LT ResNet18 NS-Muon final-training pilot (3 seeds, 5000 steps) is negative: lr=1e-4 all/few balanced accuracy `0.1265 [0.1218, 0.1312]` / `0.0008889 [-0.0006533, 0.002431]`",
        "make e11-all-results",
        "make e11-paper-assets      # regenerate current head-to-tail paper Markdown/TeX artifacts",
        "make e11-top-conference-claim-decision-audit",
        "make e11-mechanism-referee-audit # regenerate the adversarial mechanism/referee alternative-explanation audit",
        "make e11-condition-score-ablation",
        "make e11-guardrail-assets  # regenerate legacy condition-geometry guardrail notes",
        "make e11-all-assets        # regenerate current paper artifacts plus legacy guardrail notes",
        "make e11-submission-repro-audit # audit toolchain availability, PDF hashes, source hashes, and clean-checkout gates",
        "make e11-clean-worktree-replay-audit # replay e11-check from a detached clean tracked-source worktree",
        "make e11-pdf-render-boundary-audit # audit rendered PDF header/hash/source-trace coverage and text-extraction tool gaps",
        "make e11-artifact-review-packet # regenerate the artifact-review command, gate, local-state, and reviewer-response packet",
        "make e11-paper-pdf         # rebuild paper/specgrad_activation_paper/main.pdf and two_page.pdf",
        "`diagnostic_A_definition == full_layer_input_activation`",
        *MAIN_RESULT_SCRIPTS,
        *APPENDIX_RUNNER_SCRIPTS,
        "scripts/e11_write_natural_head_tail_boundary_audit.py",
        "scripts/e11_write_natural_negative_search_protocol.py",
        "scripts/e11_write_natural_negative_power_audit.py",
        "scripts/e11_run_natural_negative_search_phase1.py",
        "scripts/e11_evaluate_natural_negative_search_phase1.py",
        "scripts/e11_write_top_conference_claim_decision_audit.py",
        "scripts/e11_write_bold_conjecture_register.py",
        "scripts/e11_write_muon_state_distribution_contract.py",
        "discussion/e11_top_conference_claim_decision_audit.md",
        "results/e11_top_conference_claim_decision_audit/claim_decision_matrix.csv",
        "results/e11_top_conference_claim_decision_audit/rebuttal_response_pack.csv",
        "results/e11_top_conference_claim_decision_audit/manuscript_edit_queue.csv",
        "discussion/e11_clean_worktree_replay_audit.md",
        "results/e11_clean_worktree_replay_audit/run_summary.csv",
        "results/e11_clean_worktree_replay_audit/gate_matrix.csv",
        "scripts/e11_write_clean_worktree_replay_audit.py",
        "discussion/e11_pdf_render_boundary_audit.md",
        "results/e11_pdf_render_boundary_audit/pdf_inspection_tool_status.csv",
        "results/e11_pdf_render_boundary_audit/render_boundary_gates.csv",
        "scripts/e11_write_pdf_render_boundary_audit.py",
        "discussion/e11_mechanism_referee_audit.md",
        "results/e11_mechanism_referee_audit/alternative_explanation_matrix.csv",
        "results/e11_mechanism_referee_audit/theory_measurement_contract.csv",
        "results/e11_mechanism_referee_audit/falsification_trigger_matrix.csv",
        "scripts/e11_write_mechanism_referee_audit.py",
        "discussion/e11_condition_score_ablation.md",
        "results/e11_condition_score_ablation/score_ablation_summary.csv",
        "results/e11_condition_score_ablation/term_failure_ladder.csv",
        "results/e11_condition_score_ablation/leakage_and_claim_boundary.csv",
        "discussion/e11_artifact_review_packet.md",
        "results/e11_artifact_review_packet/command_matrix.csv",
        "results/e11_artifact_review_packet/gate_matrix.csv",
        "results/e11_artifact_review_packet/local_state_contract.csv",
        "results/e11_artifact_review_packet/reviewer_response.csv",
        "scripts/e11_write_artifact_review_packet.py",
        "completed-negative-boundary",
        "supportable theorem",
        "rebuttal-readiness contract",
        "scripts/e11_write_submission_repro_audit.py",
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
        "falls back to Tectonic",
        "make tectonic",
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
        "tail-count sweep",
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
        "\\subsection{Claim Scope and Rebuttal Discipline}",
        "The supported claim is local matched-head-gain tail-example logit drift",
        "The theorem is a local worst-case comparison",
        "not a global optimizer theorem",
        "v5 condition-score program is a registered completed negative boundary",
        "observed \\(26/26\\) settings",
        "observed \\(8/8\\) phase2 settings",
        "\\texttt{raw\\_worse\\_rows=0}",
        "finite registered phase1 and phase2 null candidates",
        "detectable-effect, head-gain, and quality caveats",
        "do not prove that no natural counterexample exists outside the registered phase1/phase2 spaces",
        "negative benchmark boundary, not evidence for an optimizer-performance advantage",
        "preferred LaTeX clean-checkout reproduction remains an explicit gate",
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
        "figures/cifar100_resnet_lt_muon_final_benchmark.png",
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
        "\\label{fig:cifar-resnet-lt-muon-final-benchmark}",
        "\\label{fig:cifar-resnet-imbalance-sweep}",
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
        "figures/cifar100_resnet_condition_score_audit.png",
        "A later score-axis ablation keeps this claim boundary explicit",
        "the direction-axis residual Spearman is positive",
        "the raw Frobenius-amplitude axis reverses",
        "a below-one direction guardrail is not a residual layer-risk ranking claim",
        "completed final gates failed",
        "ResNeXt50-32x4d residual ranking survives but direction threshold fails",
        "CIFAR-10 cross-partition residual ranking reverses but direction threshold survives",
        "figures/cifar100_resnet_imbalance_sweep.png",
        "figures/cifar100_resnet_lt_standard_eval.png",
        "figures/cifar100_resnet_lt_recipe_benchmark.png",
        "figures/cifar100_resnet_lt_muon_final_benchmark.png",
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
        "cifar100_resnet_condition_score_audit.png",
        "cifar100_resnet_imbalance_sweep.png",
        "cifar100_resnet_lt_standard_eval.png",
        "cifar100_resnet_lt_recipe_benchmark.png",
        "cifar100_resnet_lt_muon_final_benchmark.png",
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
    imbalance_sweep_dir = Path("results/e11_cifar100_resnet_imbalance_sweep")
    cifar_resnet_imbalance_steps = pd.read_csv(imbalance_sweep_dir / "step_metrics.csv")
    cifar_resnet_imbalance_summary = pd.read_csv(imbalance_sweep_dir / "pair_summary.csv")
    cifar_resnet_imbalance_layers = pd.read_csv(imbalance_sweep_dir / "layer_metrics.csv")
    cifar_resnet_imbalance_config = json.loads((imbalance_sweep_dir / "config.json").read_text())
    expected_resnet_imbalance_tail_counts = {10, 30, 100, 300}
    if (
        len(cifar_resnet_imbalance_steps) != 24
        or len(cifar_resnet_imbalance_summary) != 4
        or len(cifar_resnet_imbalance_layers) != 504
    ):
        raise AssertionError(
            "CIFAR-100-LT ResNet18 imbalance sweep must contain 4 tail counts x 3 seeds x 2 geometries"
        )
    if not (
        set(cifar_resnet_imbalance_steps["tail_train_per_class"]) == expected_resnet_imbalance_tail_counts
        and set(cifar_resnet_imbalance_summary["tail_train_per_class"]) == expected_resnet_imbalance_tail_counts
        and set(cifar_resnet_imbalance_layers["tail_train_per_class"]) == expected_resnet_imbalance_tail_counts
        and set(cifar_resnet_imbalance_steps["geometry"]) == {"frobenius", "spectral"}
        and cifar_resnet_imbalance_steps["seed"].nunique() == 3
        and (cifar_resnet_imbalance_summary["seeds"] == 3).all()
    ):
        raise AssertionError(
            "CIFAR-100-LT ResNet18 imbalance sweep must cover four tail counts, 3 seeds, and both geometries"
        )
    base_imbalance_config = cifar_resnet_imbalance_config["base_config"]
    if (
        base_imbalance_config["device"] != "cuda"
        or base_imbalance_config["download"]
        or abs(float(base_imbalance_config["target_head_gain_fraction"]) - 0.005) > 1e-12
        or len(base_imbalance_config["seeds"]) != 3
        or int(base_imbalance_config["head_train_per_class"]) != 300
        or int(base_imbalance_config["tail_eval_per_class"]) != 40
        or int(base_imbalance_config["warmup_steps"]) != 1000
        or set(cifar_resnet_imbalance_config["tail_train_per_class"])
        != expected_resnet_imbalance_tail_counts
    ):
        raise AssertionError(
            "CIFAR-100-LT ResNet18 imbalance sweep should be the formal 3-seed Slurm/GPU no-download run"
        )
    imbalance_worst = cifar_resnet_imbalance_summary.loc[
        cifar_resnet_imbalance_summary["tail_output_drift_sq_ratio_ci95_high"].idxmax()
    ]
    imbalance_tail300 = cifar_resnet_imbalance_summary[
        cifar_resnet_imbalance_summary["tail_train_per_class"].eq(300)
    ].iloc[0]
    imbalance_tail10 = cifar_resnet_imbalance_summary[
        cifar_resnet_imbalance_summary["tail_train_per_class"].eq(10)
    ].iloc[0]
    imbalance_tail30 = cifar_resnet_imbalance_summary[
        cifar_resnet_imbalance_summary["tail_train_per_class"].eq(30)
    ].iloc[0]
    if not (
        (cifar_resnet_imbalance_summary["spectral_less_tail_output_drift_fraction"] == 1.0).all()
        and (cifar_resnet_imbalance_summary["tail_output_drift_sq_ratio_ci95_high"] < 1.0).all()
        and 0.90 <= float(imbalance_worst["tail_output_drift_sq_ratio_ci95_high"]) <= 0.95
        and 0.32 <= float(imbalance_tail300["mean_tail_accuracy_before"]) <= 0.34
        and 0.35 <= float(imbalance_tail300["geomean_tail_output_drift_sq_ratio_spectral_over_fro"]) <= 0.45
        and 0.50 <= float(imbalance_tail10["geomean_tail_output_drift_sq_ratio_spectral_over_fro"]) <= 0.56
        and 0.50 <= float(imbalance_tail30["geomean_tail_output_drift_sq_ratio_spectral_over_fro"]) <= 0.56
    ):
        raise AssertionError(
            "CIFAR-100-LT ResNet18 imbalance sweep must preserve lower drift across tail-count settings"
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
    cifar_resnet_layer_jvp_checkpoint_residual_pairs = pd.read_csv(
        checkpoint_prediction_dir / "residual_prediction_pairs.csv"
    )
    cifar_resnet_layer_jvp_checkpoint_residual_summary = pd.read_csv(
        checkpoint_prediction_dir / "residual_prediction_summary.csv"
    )
    cifar_resnet_layer_jvp_checkpoint_config = json.loads(
        (checkpoint_prediction_dir / "config.json").read_text()
    )
    expected_checkpoint_prediction_steps = {2000, 5000, 10000}
    expected_checkpoint_prediction_predictors = {
        "source_observed_drift_ratio",
        "source_scaled_jvp_ratio",
        "source_unit_jvp_ratio",
        "source_alignment_ratio",
        "source_gradient_nuclear_rank",
        "architecture_early_layer_prior",
    }
    expected_checkpoint_prediction_residual_predictors = {
        "source_observed_residual",
        "source_scaled_jvp_residual",
        "source_unit_jvp_residual",
        "source_gradient_nuclear_rank_residual",
    }
    if (
        len(cifar_resnet_layer_jvp_checkpoint_metrics) != 1260
        or len(cifar_resnet_layer_jvp_checkpoint_paired) != 630
        or len(cifar_resnet_layer_jvp_checkpoint_layer_summary) != 63
        or len(cifar_resnet_layer_jvp_checkpoint_summary) != 3
        or len(cifar_resnet_layer_jvp_checkpoint_prediction_pairs) != 36
        or len(cifar_resnet_layer_jvp_checkpoint_prediction_summary) != 6
        or len(cifar_resnet_layer_jvp_checkpoint_residual_pairs) != 24
        or len(cifar_resnet_layer_jvp_checkpoint_residual_summary) != 4
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
        and set(cifar_resnet_layer_jvp_checkpoint_residual_summary["predictor"])
        == expected_checkpoint_prediction_residual_predictors
    ):
        raise AssertionError(
            "CIFAR-100-LT ResNet18 JVP checkpoint prediction must cover 10 seeds, 21 layers, four pre-registered predictors, two positive controls, and four residual predictors"
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
        or set(cifar_resnet_layer_jvp_checkpoint_config["residual_predictors"])
        != expected_checkpoint_prediction_residual_predictors
        or cifar_resnet_layer_jvp_checkpoint_config["residual_adjustment"]
        != "source_checkpoint_log_observed_drift_on_log_early_layer_prior"
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
    observed_checkpoint_prediction = checkpoint_prediction_by_predictor.loc["source_observed_drift_ratio"]
    early_layer_checkpoint_prediction = checkpoint_prediction_by_predictor.loc["architecture_early_layer_prior"]
    residual_prediction_by_predictor = cifar_resnet_layer_jvp_checkpoint_residual_summary.set_index(
        "predictor"
    )
    observed_residual_prediction = residual_prediction_by_predictor.loc["source_observed_residual"]
    scaled_residual_prediction = residual_prediction_by_predictor.loc["source_scaled_jvp_residual"]
    unit_residual_prediction = residual_prediction_by_predictor.loc["source_unit_jvp_residual"]
    rank_residual_prediction = residual_prediction_by_predictor.loc[
        "source_gradient_nuclear_rank_residual"
    ]
    if not (
        int(scaled_checkpoint_prediction["checkpoint_transfer_pairs"]) == 6
        and int(observed_checkpoint_prediction["checkpoint_transfer_pairs"]) == 6
        and abs(float(scaled_checkpoint_prediction["mean_threshold_below_one_accuracy"]) - 1.0) < 1e-12
        and abs(float(scaled_checkpoint_prediction["mean_top5_risk_overlap_fraction"]) - 0.2) < 1e-12
        and abs(float(observed_checkpoint_prediction["mean_top5_risk_overlap_fraction"]) - 1.0) < 1e-12
        and abs(float(early_layer_checkpoint_prediction["mean_top5_risk_overlap_fraction"]) - 1.0) < 1e-12
        and float(observed_checkpoint_prediction["mean_spearman_log_predictor_vs_log_target_observed"]) > 0.95
        and float(observed_checkpoint_prediction["spearman_ci95_low"]) > 0.95
        and float(early_layer_checkpoint_prediction["mean_spearman_log_predictor_vs_log_target_observed"]) > 0.85
        and float(early_layer_checkpoint_prediction["spearman_ci95_low"]) > 0.85
        and float(scaled_checkpoint_prediction["mean_spearman_log_predictor_vs_log_target_observed"]) < 0.0
        and float(scaled_checkpoint_prediction["spearman_ci95_high"]) < 0.0
        and float(unit_checkpoint_prediction["mean_spearman_log_predictor_vs_log_target_observed"])
        < float(scaled_checkpoint_prediction["mean_spearman_log_predictor_vs_log_target_observed"])
        and float(rank_checkpoint_prediction["mean_spearman_log_predictor_vs_log_target_observed"]) < 0.0
        and int(observed_residual_prediction["checkpoint_transfer_pairs"]) == 6
        and float(observed_residual_prediction["mean_spearman_residual_predictor_vs_residual_target_observed"])
        > 0.9
        and float(observed_residual_prediction["spearman_ci95_low"]) > 0.9
        and float(scaled_residual_prediction["mean_spearman_residual_predictor_vs_residual_target_observed"])
        < -0.4
        and float(scaled_residual_prediction["spearman_ci95_high"]) < -0.4
        and float(unit_residual_prediction["mean_spearman_residual_predictor_vs_residual_target_observed"])
        < float(scaled_residual_prediction["mean_spearman_residual_predictor_vs_residual_target_observed"])
        and 0.2
        < float(rank_residual_prediction["mean_spearman_residual_predictor_vs_residual_target_observed"])
        < 0.4
    ):
        raise AssertionError(
            "CIFAR-100-LT ResNet18 JVP checkpoint prediction must preserve the current boundary result: positive controls and observed residual transfer, scaled-JVP ranking does not"
        )
    score_audit_dir = Path("results/e11_cifar100_resnet_condition_score_audit")
    score_audit_raw_pairs = pd.read_csv(score_audit_dir / "raw_score_pairs.csv")
    score_audit_raw_summary = pd.read_csv(score_audit_dir / "raw_score_summary.csv")
    score_audit_residual_pairs = pd.read_csv(score_audit_dir / "residual_score_pairs.csv")
    score_audit_residual_summary = pd.read_csv(score_audit_dir / "residual_score_summary.csv")
    score_audit_config = json.loads((score_audit_dir / "config.json").read_text())
    expected_raw_scores = {
        "source_observed_drift_positive_control",
        "early_layer_prior",
        "gradient_nuclear_rank",
        "alignment_ratio",
        "step_size_ratio",
        "scaled_jvp_ratio",
        "inverse_scaled_jvp_ratio",
        "early_plus_gradient_rank",
        "early_minus_scaled_jvp",
        "early_plus_rank_minus_scaled_jvp",
    }
    expected_residual_scores = {
        "source_observed_residual_positive_control",
        "scaled_jvp_residual",
        "unit_jvp_residual",
        "gradient_nuclear_rank_residual",
        "alignment_ratio_residual",
    }
    if (
        len(score_audit_raw_pairs) != 60
        or len(score_audit_raw_summary) != 10
        or len(score_audit_residual_pairs) != 30
        or len(score_audit_residual_summary) != 5
        or set(score_audit_raw_summary["score"]) != expected_raw_scores
        or set(score_audit_residual_summary["score"]) != expected_residual_scores
        or set(score_audit_config["raw_score_definitions"]) != expected_raw_scores
        or set(score_audit_config["residual_score_definitions"]) != expected_residual_scores
    ):
        raise AssertionError(
            "CIFAR-100-LT ResNet18 condition-score audit must preserve raw and residual score coverage"
        )
    score_audit_raw_by_score = score_audit_raw_summary.set_index("score")
    score_audit_residual_by_score = score_audit_residual_summary.set_index("score")
    score_audit_early = score_audit_raw_by_score.loc["early_layer_prior"]
    score_audit_best_simple = score_audit_raw_by_score.loc["early_minus_scaled_jvp"]
    score_audit_scaled = score_audit_raw_by_score.loc["scaled_jvp_ratio"]
    score_audit_observed_control = score_audit_raw_by_score.loc[
        "source_observed_drift_positive_control"
    ]
    score_audit_observed_residual = score_audit_residual_by_score.loc[
        "source_observed_residual_positive_control"
    ]
    score_audit_scaled_residual = score_audit_residual_by_score.loc["scaled_jvp_residual"]
    if not (
        float(score_audit_observed_control["mean_spearman"]) > 0.95
        and float(score_audit_early["mean_spearman"]) > 0.9
        and float(score_audit_best_simple["mean_spearman"]) < float(score_audit_early["mean_spearman"])
        and float(score_audit_scaled["mean_spearman"]) < -0.3
        and float(score_audit_scaled["spearman_ci95_high"]) < -0.25
        and float(score_audit_observed_residual["mean_spearman"]) > 0.9
        and float(score_audit_scaled_residual["mean_spearman"]) < -0.4
        and float(score_audit_scaled_residual["spearman_ci95_high"]) < -0.4
    ):
        raise AssertionError(
            "CIFAR-100-LT ResNet18 condition-score audit must preserve the negative score-selection guardrail"
        )
    score_protocol_dir = Path("results/e11_cifar100_resnet_condition_score_protocol")
    score_protocol_scores = pd.read_csv(score_protocol_dir / "score_registry.csv")
    score_protocol_splits = pd.read_csv(score_protocol_dir / "split_registry.csv")
    score_protocol_gates = pd.read_csv(score_protocol_dir / "acceptance_gates.csv")
    assert_condition_score_protocol(score_protocol_scores, score_protocol_splits, score_protocol_gates)
    score_next_dir = Path("results/e11_cifar100_resnet_condition_score_next")
    score_next_pairs = pd.read_csv(score_next_dir / "score_pairs.csv")
    score_next_summary = pd.read_csv(score_next_dir / "score_summary.csv")
    score_next_coefficients = pd.read_csv(score_next_dir / "calibration_coefficients.csv")
    score_next_gates = pd.read_csv(score_next_dir / "gate_report.csv")
    score_next_config = json.loads((score_next_dir / "config.json").read_text())
    expected_next_scores = {
        "condition_score_v2_calibrated_residual",
        "early_layer_prior",
        "legacy_scaled_jvp_ratio",
        "theory_sign_composite",
        "source_observed_drift_positive_control",
    }
    if (
        len(score_next_pairs) != 30
        or len(score_next_summary) != 5
        or len(score_next_coefficients) != 27
        or len(score_next_gates) != 6
        or set(score_next_summary["score"]) != expected_next_scores
        or score_next_config.get("analysis_scope")
        != "locked retrospective ResNet18 checkpoint split; registered held-out architecture/data evaluation available separately when heldout_score_evaluation exists"
    ):
        raise AssertionError("condition-score v2 retrospective analysis must preserve registered output shape")
    score_next_by_score = score_next_summary.set_index("score")
    score_next_primary = score_next_by_score.loc["condition_score_v2_calibrated_residual"]
    score_next_early = score_next_by_score.loc["early_layer_prior"]
    score_next_legacy = score_next_by_score.loc["legacy_scaled_jvp_ratio"]
    score_next_gate_status = score_next_gates.set_index("gate_id")["status"].to_dict()
    if not (
        float(score_next_primary["mean_spearman_score_vs_target_residual"]) > 0.5
        and float(score_next_primary["spearman_ci95_low"]) > 0.0
        and float(score_next_primary["mean_spearman_score_vs_target_residual"])
        > float(score_next_early["mean_spearman_score_vs_target_residual"])
        and float(score_next_legacy["mean_spearman_score_vs_target_residual"]) < 0.0
        and score_next_gate_status.get("legacy_checkpoint_residual_spearman") == "pass"
        and score_next_gate_status.get("primary_heldout_architecture") == "fail"
        and score_next_gate_status.get("primary_heldout_data") == "fail"
        and score_next_gate_status.get("p0_predictive_condition_claim") == "not_ready"
    ):
        raise AssertionError(
            "condition-score v2 analysis must show a promising retrospective checkpoint result and the failed held-out P0 boundary"
        )
    heldout_arch_dir = score_next_dir / "heldout_architecture"
    heldout_data_dir = score_next_dir / "heldout_data"
    heldout_eval_dir = score_next_dir / "heldout_score_evaluation"
    heldout_arch_layer_summary = pd.read_csv(heldout_arch_dir / "layer_summary.csv")
    heldout_data_layer_summary = pd.read_csv(heldout_data_dir / "layer_summary.csv")
    heldout_arch_config = json.loads((heldout_arch_dir / "config.json").read_text())
    heldout_data_config = json.loads((heldout_data_dir / "config.json").read_text())
    heldout_score_pairs = pd.read_csv(heldout_eval_dir / "heldout_score_pairs.csv")
    heldout_score_summary = pd.read_csv(heldout_eval_dir / "heldout_score_summary.csv")
    heldout_gate_report = pd.read_csv(heldout_eval_dir / "heldout_gate_report.csv")
    if (
        len(heldout_arch_layer_summary) != 111
        or len(heldout_data_layer_summary) != 63
        or len(heldout_score_pairs) != 90
        or len(heldout_score_summary) != 10
        or len(heldout_gate_report) != 9
        or set(heldout_score_summary["score"]) != expected_next_scores
    ):
        raise AssertionError(
            "registered condition-score held-out evaluation must preserve architecture/data score coverage"
        )
    heldout_arch_base_config = heldout_arch_config["base_config"]
    heldout_data_base_config = heldout_data_config["base_config"]
    if not (
        heldout_arch_base_config["dataset_name"] == "CIFAR100"
        and heldout_arch_base_config["model_arch"] == "resnet34"
        and heldout_arch_base_config["tail_train_per_class"] == 300
        and heldout_arch_config["warmup_steps"] == [2000, 5000, 10000]
        and len(heldout_arch_base_config["seeds"]) == 5
        and heldout_data_base_config["dataset_name"] == "CIFAR10"
        and heldout_data_base_config["model_arch"] == "resnet18"
        and heldout_data_base_config["head_classes"] == [0, 1, 2, 3, 4]
        and heldout_data_base_config["tail_classes"] == [5, 6, 7, 8, 9]
        and heldout_data_base_config["tail_eval_per_class"] == 200
        and heldout_data_config["warmup_steps"] == [2000, 5000, 10000]
        and len(heldout_data_base_config["seeds"]) == 10
    ):
        raise AssertionError(
            "registered condition-score held-out configs must preserve ResNet34 architecture and CIFAR-10-LT data split definitions"
        )
    heldout_summary_by_key = heldout_score_summary.set_index(["split_role", "score"])
    heldout_arch_primary = heldout_summary_by_key.loc[
        ("primary_heldout_architecture", "condition_score_v2_calibrated_residual")
    ]
    heldout_arch_source = heldout_summary_by_key.loc[
        ("primary_heldout_architecture", "source_observed_drift_positive_control")
    ]
    heldout_data_primary = heldout_summary_by_key.loc[
        ("primary_heldout_data", "condition_score_v2_calibrated_residual")
    ]
    heldout_data_legacy = heldout_summary_by_key.loc[
        ("primary_heldout_data", "legacy_scaled_jvp_ratio")
    ]
    heldout_gate_status = heldout_gate_report.set_index("gate_id")["status"].to_dict()
    if not (
        heldout_gate_status.get("primary_heldout_architecture_residual_spearman") == "fail"
        and heldout_gate_status.get("primary_heldout_architecture_threshold_accuracy") == "pass"
        and heldout_gate_status.get("primary_heldout_architecture_early_prior_comparison") == "pass"
        and heldout_gate_status.get("primary_heldout_data_residual_spearman") == "fail"
        and heldout_gate_status.get("primary_heldout_data_threshold_accuracy") == "pass"
        and heldout_gate_status.get("primary_heldout_data_early_prior_comparison") == "pass"
        and heldout_gate_status.get("p0_predictive_condition_heldout_claim") == "not_ready"
        and float(heldout_arch_primary["mean_spearman_score_vs_target_residual"]) > 0.15
        and float(heldout_arch_primary["spearman_ci95_low"]) < 0.0
        and float(heldout_arch_primary["spearman_ci95_high"]) > 0.0
        and float(heldout_arch_primary["mean_threshold_below_one_accuracy"]) > 0.98
        and float(heldout_arch_source["mean_spearman_score_vs_target_residual"]) > 0.5
        and float(heldout_arch_source["spearman_ci95_low"]) > 0.45
        and float(heldout_data_primary["mean_spearman_score_vs_target_residual"]) < -0.6
        and float(heldout_data_primary["spearman_ci95_high"]) < -0.6
        and float(heldout_data_primary["mean_threshold_below_one_accuracy"]) > 0.98
        and float(heldout_data_legacy["mean_spearman_score_vs_target_residual"]) > 0.6
        and float(heldout_data_legacy["spearman_ci95_low"]) > 0.59
        and float(heldout_data_legacy["mean_top5_residual_risk_overlap_fraction"]) > 0.9
        and float(heldout_data_legacy["mean_threshold_below_one_accuracy"]) == 1.0
    ):
        raise AssertionError(
            "registered held-out condition-score evaluation must preserve the current failed residual-ranking boundary and passing threshold-direction readout"
        )
    condition_score_bridge_dir = Path("results/e11_condition_score_theory_bridge")
    score_target_register = pd.read_csv(condition_score_bridge_dir / "score_target_register.csv")
    fresh_protocol_requirements = pd.read_csv(
        condition_score_bridge_dir / "fresh_protocol_requirements.csv"
    )
    expected_target_ids = {
        "direction_threshold",
        "residual_layer_ranking",
        "source_observed_transfer_control",
        "legacy_jvp_counterexample",
    }
    expected_requirement_ids = {
        "R1-target-separation",
        "R2-heldout-quarantine",
        "R3-theory-derived-score",
        "R4-nested-calibration",
        "R5-fresh-heldout-gates",
        "R6-negative-outcome-reporting",
    }
    if (
        len(score_target_register) != 4
        or set(score_target_register["target_id"]) != expected_target_ids
        or len(fresh_protocol_requirements) != 6
        or set(fresh_protocol_requirements["requirement_id"]) != expected_requirement_ids
    ):
        raise AssertionError(
            "condition-score theory bridge must preserve target separation and fresh-protocol requirements"
        )
    score_target_status = score_target_register.set_index("target_id")["status"].to_dict()
    requirement_status = fresh_protocol_requirements.set_index("requirement_id")["current_status"].to_dict()
    if not (
        score_target_status.get("direction_threshold") == "supported_guardrail"
        and score_target_status.get("residual_layer_ranking") == "blocked_by_heldout_failure"
        and score_target_status.get("source_observed_transfer_control") == "architecture_transfer_only"
        and score_target_status.get("legacy_jvp_counterexample") == "v2_not_uniformly_better"
        and requirement_status.get("R2-heldout-quarantine") == "satisfied_by_quarantine_register"
        and requirement_status.get("R3-theory-derived-score")
        == "satisfied_by_fresh_registry_and_v5_freeze"
        and requirement_status.get("R4-nested-calibration")
        == "satisfied_by_zero_fit_primary_and_validation_freeze"
        and requirement_status.get("R5-fresh-heldout-gates") == "evaluated_failed_boundary"
    ):
        raise AssertionError(
            "condition-score theory bridge must keep spent held-outs quarantined, record the later frozen-score protocol, and preserve the evaluated failed boundary"
        )
    fresh_protocol_dir = Path("results/e11_condition_score_fresh_protocol")
    quarantine_register = pd.read_csv(fresh_protocol_dir / "quarantine_register.csv")
    score_freeze_registry = pd.read_csv(fresh_protocol_dir / "score_freeze_registry.csv")
    fresh_split_registry = pd.read_csv(fresh_protocol_dir / "fresh_split_registry.csv")
    fresh_acceptance_gates = pd.read_csv(fresh_protocol_dir / "acceptance_gates.csv")
    fresh_protocol_status = pd.read_csv(fresh_protocol_dir / "protocol_status.csv")
    expected_quarantine_ids = {
        "spent_primary_heldout_architecture_resnet34",
        "spent_primary_heldout_data_cifar10lt_resnet18",
    }
    expected_fresh_score_ids = {
        "condition_score_v3_zero_fit_scaled_jvp",
        "condition_score_v3_nested_jvp_residual",
        "early_layer_prior",
        "source_observed_drift_positive_control",
    }
    expected_fresh_split_roles = {
        "calibration_only",
        "validation_only",
        "fresh_final_heldout_architecture",
        "fresh_final_heldout_data_partition",
    }
    expected_fresh_gate_ids = {
        "F1-quarantine-enforced",
        "F2-score-freeze-before-final",
        "F3-target-separation",
        "F4-residual-ranking-success",
        "F5-threshold-direction-success",
        "F6-baselines-reported",
    }
    if (
        set(quarantine_register["split_id"]) != expected_quarantine_ids
        or set(score_freeze_registry["score_id"]) != expected_fresh_score_ids
        or set(fresh_split_registry["role"]) != expected_fresh_split_roles
        or set(fresh_acceptance_gates["gate_id"]) != expected_fresh_gate_ids
        or len(fresh_protocol_status) != 4
    ):
        raise AssertionError(
            "fresh condition-score protocol must preserve quarantine, score-freeze, split, gate, and status coverage"
        )
    if not (
        quarantine_register["forbidden_use"].astype(str).str.contains("final P0 claim evidence").all()
        and score_freeze_registry.set_index("score_id").loc[
            "condition_score_v3_zero_fit_scaled_jvp", "coefficient_rule"
        ].startswith("zero-fit")
        and score_freeze_registry.set_index("score_id").loc[
            "condition_score_v3_zero_fit_scaled_jvp", "uses_spent_heldouts"
        ]
        == "no"
        and "ResNet50 CIFAR stem"
        in set(fresh_split_registry["architecture"])
        and "head=0,2,4,6,8; tail=1,3,5,7,9"
        in set(fresh_split_registry["class_partition"])
        and fresh_protocol_status.set_index("item").loc["fresh final held-out evidence", "status"]
        in {"missing", "generated_pending_evaluation", "evaluated_not_ready", "evaluated_pass"}
    ):
        raise AssertionError(
            "fresh condition-score protocol must quarantine spent held-outs, freeze the zero-fit scaled-JVP candidate, and track fresh final evidence status"
        )
    fresh_eval_dir = fresh_protocol_dir / "fresh_score_evaluation"
    fresh_gate_report = pd.read_csv(fresh_eval_dir / "fresh_gate_report.csv")
    fresh_score_summary = pd.read_csv(fresh_eval_dir / "fresh_score_summary.csv")
    fresh_eval_config = json.loads((fresh_eval_dir / "config.json").read_text())
    fresh_gate_status = fresh_gate_report.set_index("gate_id")["status"].to_dict()
    fresh_eval_generated = [bool(item["generated"]) for item in fresh_eval_config["fresh_splits"]]
    fresh_evidence_status = fresh_protocol_status.set_index("item").loc["fresh final held-out evidence", "status"]
    if all(fresh_eval_generated):
        expected_fresh_gate_status = {
            "fresh_final_heldout_architecture_residual_spearman": "fail",
            "fresh_final_heldout_architecture_threshold_accuracy": "pass",
            "fresh_final_heldout_architecture_early_prior_comparison": "fail",
            "fresh_final_heldout_architecture_baselines_reported": "pass",
            "fresh_final_heldout_data_partition_residual_spearman": "pass",
            "fresh_final_heldout_data_partition_threshold_accuracy": "pass",
            "fresh_final_heldout_data_partition_early_prior_comparison": "pass",
            "fresh_final_heldout_data_partition_baselines_reported": "pass",
            "fresh_p0_predictive_condition_claim": "not_ready",
        }
        primary_rows = fresh_score_summary[
            fresh_score_summary["score"].eq("condition_score_v3_zero_fit_scaled_jvp")
        ].set_index("split_id")
        if not (
            len(fresh_gate_report) == len(expected_fresh_gate_status)
            and fresh_gate_status == expected_fresh_gate_status
            and fresh_eval_config["primary_score"] == "condition_score_v3_zero_fit_scaled_jvp"
            and fresh_evidence_status == "evaluated_not_ready"
            and set(primary_rows.index)
            == {
                "fresh_final_architecture_resnet50_cifar100lt",
                "fresh_final_data_cifar10lt_alt_partition",
            }
            and float(
                primary_rows.loc[
                    "fresh_final_architecture_resnet50_cifar100lt",
                    "spearman_ci95_high",
                ]
            )
            < 0.0
            and float(
                primary_rows.loc[
                    "fresh_final_data_cifar10lt_alt_partition",
                    "spearman_ci95_low",
                ]
            )
            > 0.0
        ):
            raise AssertionError(
                "fresh condition-score evaluator must preserve the evaluated ResNet50 fail, CIFAR-10 alternate pass, and P0 not_ready state"
            )
    elif not any(fresh_eval_generated):
        if not (
            len(fresh_gate_report) == 3
            and fresh_gate_status.get("fresh_final_heldout_architecture_generated") == "not_run"
            and fresh_gate_status.get("fresh_final_heldout_data_partition_generated") == "not_run"
            and fresh_gate_status.get("fresh_p0_predictive_condition_claim") == "not_ready"
            and fresh_eval_config["primary_score"] == "condition_score_v3_zero_fit_scaled_jvp"
        ):
            raise AssertionError(
                "fresh condition-score evaluator must preserve the not_run/not_ready state until fresh final split outputs exist"
            )
    else:
        raise AssertionError("fresh condition-score evaluator should not be committed with only one fresh final split generated")
    failure_audit_dir = Path("results/e11_condition_score_failure_mechanism_audit")
    score_outcome_matrix = pd.read_csv(failure_audit_dir / "score_outcome_matrix.csv")
    obstruction_taxonomy = pd.read_csv(failure_audit_dir / "split_obstruction_taxonomy.csv")
    resnet50_stage_reversal = pd.read_csv(failure_audit_dir / "resnet50_stage_reversal.csv")
    expected_obstruction_ids = {
        "O1-v2-architecture-weak-transfer",
        "O2-v2-data-family-sign-reversal",
        "O3-v3-resnet50-jvp-reversal",
        "O4-v3-data-pass-is-not-universal",
    }
    if not (
        len(score_outcome_matrix) == 18
        and len(obstruction_taxonomy) == 4
        and set(obstruction_taxonomy["obstruction_id"]) == expected_obstruction_ids
        and len(resnet50_stage_reversal) == 18
    ):
        raise AssertionError("condition-score failure mechanism audit must cover all held-out/fresh score rows and ResNet50 stage terms")
    score_lookup = score_outcome_matrix.set_index(["split_role", "score"])
    resnet50_v3 = score_lookup.loc[
        ("fresh_final_heldout_architecture", "condition_score_v3_zero_fit_scaled_jvp")
    ]
    cifar10_v3 = score_lookup.loc[
        ("fresh_final_heldout_data_partition", "condition_score_v3_zero_fit_scaled_jvp")
    ]
    if not (
        resnet50_v3["residual_gate_status"] == "inverted_residual_ranking"
        and bool(resnet50_v3["threshold_only_success"])
        and float(resnet50_v3["spearman_ci95_high"]) < 0.0
        and cifar10_v3["residual_gate_status"] == "passes_residual_gate"
        and float(cifar10_v3["spearman_ci95_low"]) > 0.0
    ):
        raise AssertionError("condition-score failure mechanism audit must preserve the fresh ResNet50 reversal and CIFAR-10 alternate pass")
    stage_lookup = resnet50_stage_reversal.set_index(["stage", "block_term"])
    if not (
        ("layer3", "conv2") in stage_lookup.index
        and int(stage_lookup.loc[("layer3", "conv2"), "total_top5_target_residual_layers"]) >= 20
        and int(stage_lookup.loc[("layer3", "conv2"), "total_top5_primary_score_layers"]) == 0
        and ("classifier", "classifier") in stage_lookup.index
        and int(stage_lookup.loc[("classifier", "classifier"), "total_top5_primary_score_layers"]) > 0
    ):
        raise AssertionError("condition-score failure mechanism audit must expose the ResNet50 layer2/layer3 residual versus classifier/stem score mismatch")
    v4_protocol_dir = Path("results/e11_condition_score_v4_protocol")
    v4_spent = pd.read_csv(v4_protocol_dir / "spent_split_register.csv")
    v4_scores = pd.read_csv(v4_protocol_dir / "score_axis_registry.csv")
    v4_splits = pd.read_csv(v4_protocol_dir / "unspent_split_registry.csv")
    v4_gates = pd.read_csv(v4_protocol_dir / "acceptance_gates.csv")
    v4_status = pd.read_csv(v4_protocol_dir / "protocol_status.csv")
    expected_v4_spent_ids = {
        "spent_v2_primary_heldout_architecture_resnet34",
        "spent_v2_primary_heldout_data_cifar10lt_resnet18",
        "spent_v3_fresh_architecture_resnet50",
        "spent_v3_fresh_data_cifar10lt_alt_partition",
    }
    expected_v4_score_ids = {
        "condition_score_v4_two_axis_transport_jvp",
        "v4_direction_axis_scaled_jvp_ratio",
        "v4_residual_amplitude_axis_scaled_jvp_fro",
        "v4_architecture_transport_tags",
        "early_layer_prior",
        "source_observed_drift_positive_control",
    }
    expected_v4_roles = {
        "calibration_only",
        "validation_only",
        "fresh_final_heldout_architecture",
        "fresh_final_heldout_data_partition",
    }
    expected_v4_gates = {
        "V4-1-spent-final-quarantine",
        "V4-2-score-axis-separation",
        "V4-3-validation-freeze-before-final",
        "V4-4-residual-ranking-success",
        "V4-5-threshold-direction-guardrail",
        "V4-6-baselines-and-negative-reporting",
        "V4-7-claim-boundary",
    }
    if not (
        set(v4_spent["split_id"]) == expected_v4_spent_ids
        and set(v4_scores["score_id"]) == expected_v4_score_ids
        and set(v4_splits["role"]) == expected_v4_roles
        and set(v4_gates["gate_id"]) == expected_v4_gates
        and len(v4_status) == 4
    ):
        raise AssertionError("condition-score v4 protocol must register spent splits, score axes, unspent split roles, and gates")
    v4_score_lookup = v4_scores.set_index("score_id")
    v4_split_lookup = v4_splits.set_index("split_id")
    v4_status_lookup = v4_status.set_index("item")["status"].to_dict()
    if not (
        v4_spent["forbidden_use"].astype(str).str.contains("validation selection").all()
        and v4_score_lookup.loc[
            "condition_score_v4_two_axis_transport_jvp",
            "coefficient_rule",
        ].startswith("zero-fit axes")
        and v4_score_lookup.loc[
            "condition_score_v4_two_axis_transport_jvp",
            "eligible_for_final_p0_claim",
        ].startswith("not until validation")
        and v4_split_lookup.loc[
            "v4_final_architecture_wide_resnet50_2_cifar100lt",
            "architecture",
        ]
        == "WideResNet50-2 CIFAR stem"
        and v4_split_lookup.loc[
            "v4_final_data_cifar10lt_mixed_partition",
            "class_partition",
        ]
        == "head=0,1,4,7,8; tail=2,3,5,6,9"
        and v4_status_lookup.get("v4 final held-out evidence") == "evaluated_not_ready"
        and v4_status_lookup.get("v4 score-axis registry") == "validation_frozen"
    ):
        raise AssertionError("condition-score v4 protocol must keep v2/v3 final splits quarantined and record the evaluated-not-ready v4 final state")
    v4_freeze_dir = v4_protocol_dir / "validation_score_freeze"
    v4_freeze_formulas = pd.read_csv(v4_freeze_dir / "score_formula_registry.csv")
    v4_freeze_status = pd.read_csv(v4_freeze_dir / "freeze_status.csv")
    v4_freeze_pairs = pd.read_csv(v4_freeze_dir / "validation_score_pairs.csv")
    v4_freeze_summary = pd.read_csv(v4_freeze_dir / "validation_score_summary.csv")
    v4_freeze_gates = pd.read_csv(v4_freeze_dir / "validation_gate_report.csv")
    v4_freeze_config = json.loads((v4_freeze_dir / "config.json").read_text(encoding="utf-8"))
    expected_v4_freeze_scores = {
        "condition_score_v4_direction_axis_scaled_jvp_ratio",
        "condition_score_v4_fro_amplitude_axis",
        "condition_score_v4_two_axis_positive",
        "condition_score_v4_two_axis_amplitude_minus_direction",
        "condition_score_v4_two_axis_transport",
        "early_layer_prior",
        "source_observed_drift_positive_control",
        "condition_score_v4_validation_selected",
    }
    expected_v4_freeze_items = {
        "v4 validation split output",
        "v4 candidate pool",
        "v4 selected residual score",
        "v4 final split outputs",
    }
    expected_v4_freeze_gates = {
        "V4F-1-validation-output",
        "V4F-2-no-final-before-freeze",
        "V4F-3-residual-score-freeze",
        "V4F-4-direction-threshold-guardrail",
        "V4F-5-selected-score-residual-spearman",
        "V4F-6-final-claim-readiness",
    }
    if not (
        expected_v4_freeze_scores.issubset(set(v4_freeze_formulas["score_id"]))
        and set(v4_freeze_status["item"]) == expected_v4_freeze_items
        and set(v4_freeze_gates["gate_id"]) == expected_v4_freeze_gates
        and "log_scaled_jvp_fro_amplitude" in set(v4_freeze_config["axis_features"])
        and "CI lower endpoint >= 0.8" in v4_freeze_config["direction_gate"]
    ):
        raise AssertionError("condition-score v4 validation freeze must preserve candidate formulas, status rows, gates, and Frobenius amplitude axis")
    v4_freeze_status_lookup = v4_freeze_status.set_index("item")["status"].to_dict()
    v4_freeze_evidence_lookup = v4_freeze_status.set_index("item")["evidence"].to_dict()
    v4_freeze_gate_lookup = v4_freeze_gates.set_index("gate_id")["status"].to_dict()
    if not bool(v4_freeze_config["validation_generated"]):
        if not (
            v4_freeze_pairs.empty
            and v4_freeze_summary.empty
            and v4_freeze_status_lookup.get("v4 validation split output") == "not_run"
            and v4_freeze_status_lookup.get("v4 candidate pool") == "registered"
            and v4_freeze_status_lookup.get("v4 selected residual score") == "not_ready"
            and v4_freeze_status_lookup.get("v4 final split outputs") == "not_run"
            and v4_freeze_gate_lookup.get("V4F-1-validation-output") == "not_run"
            and v4_freeze_gate_lookup.get("V4F-2-no-final-before-freeze") == "pass"
            and v4_freeze_gate_lookup.get("V4F-6-final-claim-readiness") == "not_ready"
        ):
            raise AssertionError("condition-score v4 validation freeze must keep final splits blocked while validation output is absent")
    else:
        selected_rows = v4_freeze_formulas[v4_freeze_formulas["selected_for_final_evaluation"].eq("yes")]
        selected_status = v4_freeze_status_lookup.get("v4 selected residual score")
        if v4_freeze_pairs.empty or v4_freeze_summary.empty or selected_status not in {"frozen", "validation_failed"}:
            raise AssertionError("condition-score v4 validation freeze with generated validation data must include score rows and a selected-or-blocking status")
        if selected_status == "frozen" and selected_rows.empty:
            raise AssertionError("condition-score v4 validation freeze must mark the selected final score when validation passes")
        if v4_freeze_status_lookup.get("v4 final split outputs") != "not_run":
            raise AssertionError("condition-score v4 final split outputs must not exist before the validation freeze is committed")
        selected_score_id = "condition_score_v4_two_axis_amplitude_minus_direction"
        selected_summary = v4_freeze_summary[v4_freeze_summary["score"].eq(selected_score_id)]
        direction_summary = v4_freeze_summary[
            v4_freeze_summary["score"].eq("condition_score_v4_direction_axis_scaled_jvp_ratio")
        ]
        if not (
            v4_freeze_evidence_lookup.get("v4 selected residual score") == selected_score_id
            and set(v4_freeze_gate_lookup.values()) == {"pass"}
            and not selected_summary.empty
            and not direction_summary.empty
            and abs(float(selected_summary.iloc[0]["mean_spearman_score_vs_target_residual"]) - 0.375758) < 1e-5
            and float(selected_summary.iloc[0]["spearman_ci95_low"]) > 0.27
            and float(direction_summary.iloc[0]["mean_threshold_below_one_accuracy"]) == 1.0
            and int(v4_freeze_config["validation_generated"]) == 1
        ):
            raise AssertionError("condition-score v4 validation freeze must preserve the committed pass state and selected two-axis score")
    v4_final_eval_dir = v4_protocol_dir / "final_score_evaluation"
    v4_final_pairs = pd.read_csv(v4_final_eval_dir / "final_score_pairs.csv")
    v4_final_summary = pd.read_csv(v4_final_eval_dir / "final_score_summary.csv")
    v4_final_gates = pd.read_csv(v4_final_eval_dir / "final_gate_report.csv")
    v4_final_config = json.loads((v4_final_eval_dir / "config.json").read_text(encoding="utf-8"))
    expected_v4_final_gate_ids = {
        "fresh_final_heldout_architecture_generated",
        "fresh_final_heldout_data_partition_generated",
        "v4_p0_predictive_condition_claim",
    }
    expected_v4_final_splits = {
        "v4_final_architecture_wide_resnet50_2_cifar100lt",
        "v4_final_data_cifar10lt_mixed_partition",
    }
    final_generated = [bool(item["generated"]) for item in v4_final_config["final_splits"]]
    final_split_ids = {item["split_id"] for item in v4_final_config["final_splits"]}
    final_gate_lookup = v4_final_gates.set_index("gate_id")["status"].to_dict()
    if not (
        v4_final_config["primary_score"] == "condition_score_v4_two_axis_amplitude_minus_direction"
        and final_split_ids == expected_v4_final_splits
    ):
        raise AssertionError("condition-score v4 final evaluator must use the validation-frozen selected score and unspent split ids")
    if not any(final_generated):
        if not (
            v4_final_pairs.empty
            and v4_final_summary.empty
            and set(v4_final_gates["gate_id"]) == expected_v4_final_gate_ids
            and final_gate_lookup.get("fresh_final_heldout_architecture_generated") == "not_run"
            and final_gate_lookup.get("fresh_final_heldout_data_partition_generated") == "not_run"
            and final_gate_lookup.get("v4_p0_predictive_condition_claim") == "not_ready"
        ):
            raise AssertionError("condition-score v4 final evaluator must stay not_run/not_ready until final split outputs exist")
    else:
        if v4_final_pairs.empty or v4_final_summary.empty:
            raise AssertionError("condition-score v4 final evaluator with generated final data must include score rows")
        if final_gate_lookup.get("v4_p0_predictive_condition_claim") not in {"pass", "not_ready"}:
            raise AssertionError("condition-score v4 final P0 gate must be pass or not_ready after final outputs exist")
        expected_final_gate_status = {
            "fresh_final_heldout_architecture_residual_spearman": "pass",
            "fresh_final_heldout_architecture_direction_threshold_accuracy": "pass",
            "fresh_final_heldout_architecture_baselines_reported": "pass",
            "fresh_final_heldout_data_partition_residual_spearman": "fail",
            "fresh_final_heldout_data_partition_direction_threshold_accuracy": "pass",
            "fresh_final_heldout_data_partition_baselines_reported": "pass",
            "v4_p0_predictive_condition_claim": "not_ready",
        }
        primary_summary = v4_final_summary[
            v4_final_summary["score"].eq("condition_score_v4_two_axis_amplitude_minus_direction")
        ].set_index("split_id")
        if not (
            final_gate_lookup == expected_final_gate_status
            and "v4_final_architecture_wide_resnet50_2_cifar100lt" in primary_summary.index
            and "v4_final_data_cifar10lt_mixed_partition" in primary_summary.index
            and abs(
                float(
                    primary_summary.loc[
                        "v4_final_architecture_wide_resnet50_2_cifar100lt",
                        "mean_spearman_score_vs_target_residual",
                    ]
                )
                - 0.644893
            )
            < 1e-5
            and float(
                primary_summary.loc[
                    "v4_final_architecture_wide_resnet50_2_cifar100lt",
                    "spearman_ci95_low",
                ]
            )
            > 0.51
            and abs(
                float(
                    primary_summary.loc[
                        "v4_final_data_cifar10lt_mixed_partition",
                        "mean_spearman_score_vs_target_residual",
                    ]
                )
                + 0.693651
            )
            < 1e-5
            and float(
                primary_summary.loc[
                    "v4_final_data_cifar10lt_mixed_partition",
                    "spearman_ci95_high",
                ]
            )
            < 0.0
        ):
            raise AssertionError("condition-score v4 final evaluator must preserve the WideResNet50-2 pass, CIFAR-10 mixed fail, and P0 not_ready state")
        v4_failure_audit_dir = Path("results/e11_condition_score_v4_failure_mechanism_audit")
        v4_failure_outcome = pd.read_csv(v4_failure_audit_dir / "final_outcome_matrix.csv")
        v4_axis_all_layer_summary = pd.read_csv(v4_failure_audit_dir / "axis_all_layer_summary.csv")
        v4_axis_transfer_pair_scores = pd.read_csv(v4_failure_audit_dir / "axis_transfer_pair_scores.csv")
        v4_axis_pair_summary = pd.read_csv(v4_failure_audit_dir / "axis_pair_summary.csv")
        v4_top5_stage_summary = pd.read_csv(v4_failure_audit_dir / "top5_stage_summary.csv")
        v4_obstruction_summary = pd.read_csv(v4_failure_audit_dir / "obstruction_summary.csv")
        expected_v4_obstruction_ids = {
            "V4-O1-architecture-transfer-pass",
            "V4-O2-data-partition-reversal",
            "V4-O3-direction-is-not-the-failure",
            "V4-O4-amplitude-depth-confound",
        }
        if not (
            len(v4_failure_outcome) == 9
            and len(v4_axis_all_layer_summary) == 8
            and len(v4_axis_transfer_pair_scores) == 72
            and len(v4_axis_pair_summary) == 8
            and set(v4_obstruction_summary["obstruction_id"]) == expected_v4_obstruction_ids
            and set(v4_axis_pair_summary["transfer_pairs"]) == {9}
        ):
            raise AssertionError(
                "condition-score v4 failure mechanism audit must preserve final outcome, axis, transfer-pair, and obstruction coverage"
            )
        v4_failure_outcome_lookup = v4_failure_outcome.set_index(["split_role", "score"])
        v4_arch_primary_outcome = v4_failure_outcome_lookup.loc[
            ("fresh_final_heldout_architecture", "condition_score_v4_two_axis_amplitude_minus_direction")
        ]
        v4_data_primary_outcome = v4_failure_outcome_lookup.loc[
            ("fresh_final_heldout_data_partition", "condition_score_v4_two_axis_amplitude_minus_direction")
        ]
        v4_data_direction_outcome = v4_failure_outcome_lookup.loc[
            ("fresh_final_heldout_data_partition", "condition_score_v4_direction_axis_scaled_jvp_ratio")
        ]
        v4_final_gate_outcome = v4_failure_outcome_lookup.loc[
            ("p0_predictive_condition", "condition_score_v4_two_axis_amplitude_minus_direction")
        ]
        v4_axis_lookup = v4_axis_pair_summary.set_index(["split_role", "score_id"])
        v4_data_primary_axis = v4_axis_lookup.loc[
            ("fresh_final_heldout_data_partition", "condition_score_v4_two_axis_amplitude_minus_direction")
        ]
        v4_data_direction_axis = v4_axis_lookup.loc[
            ("fresh_final_heldout_data_partition", "v4_direction_axis_scaled_jvp_ratio")
        ]
        v4_data_amplitude_axis = v4_axis_lookup.loc[
            ("fresh_final_heldout_data_partition", "v4_residual_amplitude_axis_scaled_jvp_fro")
        ]
        v4_data_early_axis = v4_axis_lookup.loc[
            ("fresh_final_heldout_data_partition", "early_layer_prior")
        ]
        v4_failed_top5 = v4_top5_stage_summary[
            v4_top5_stage_summary["split_id"].eq("v4_final_data_cifar10lt_mixed_partition")
        ]
        if not (
            v4_arch_primary_outcome["residual_gate_status"] == "passes_residual_gate"
            and v4_data_primary_outcome["residual_gate_status"] == "inverted_residual_ranking"
            and v4_final_gate_outcome["residual_gate_status"] == "not_ready"
            and float(v4_data_direction_outcome["mean_threshold_below_one_accuracy"]) == 1.0
            and float(v4_data_primary_axis["mean_spearman_score_vs_target_residual"]) < -0.69
            and float(v4_data_primary_axis["spearman_ci95_high"]) < 0.0
            and float(v4_data_direction_axis["mean_spearman_score_vs_target_residual"]) > 0.60
            and float(v4_data_direction_axis["spearman_ci95_low"]) > 0.58
            and float(v4_data_amplitude_axis["mean_spearman_score_vs_target_residual"]) < -0.67
            and float(v4_data_amplitude_axis["spearman_ci95_high"]) < 0.0
            and float(v4_data_early_axis["mean_spearman_score_vs_target_residual"]) < -0.79
            and not v4_failed_top5.empty
            and int(v4_failed_top5["target_residual_top5_count"].sum()) > 0
        ):
            raise AssertionError(
                "condition-score v4 failure mechanism audit must preserve the CIFAR-10 mixed direction-pass and amplitude/depth scalar-reversal diagnosis"
            )
        v4_failure_audit_text = Path("discussion/e11_condition_score_v4_failure_mechanism_audit.md").read_text(
            encoding="utf-8"
        )
        assert_required_phrases(
            "condition-score v4 failure mechanism audit",
            v4_failure_audit_text,
            [
                "E11 Condition-Score V4 Failure Mechanism Audit",
                "V4-O2-data-partition-reversal",
                "V4-O3-direction-is-not-the-failure",
                "The failure is not a direction-threshold failure",
                "Frobenius-amplitude/depth",
                "These final rows are now spent for score fitting.",
            ],
        )
    matrix_proof_dir = Path("results/e11_matrix_block_theorem_proof")
    matrix_theorem = pd.read_csv(matrix_proof_dir / "theorem_statement.csv")
    matrix_assumptions = pd.read_csv(matrix_proof_dir / "assumption_ledger.csv")
    matrix_steps = pd.read_csv(matrix_proof_dir / "proof_steps.csv")
    matrix_claims = pd.read_csv(matrix_proof_dir / "claim_implications.csv")
    matrix_cross_checks = pd.read_csv(matrix_proof_dir / "paper_cross_checks.csv")
    matrix_config = json.loads((matrix_proof_dir / "config.json").read_text(encoding="utf-8"))
    expected_matrix_statements = {
        "MBT-1-general-matched-gain-bound",
        "MBT-2-sandwich-frobenius-coefficient",
        "MBT-3-sandwich-spectral-coefficient",
        "MBT-4-rank-boundary",
    }
    expected_matrix_assumptions = {
        "MBA-1-locality",
        "MBA-2-head-gain-matching",
        "MBA-3-sandwich-block",
        "MBA-4-nondegenerate-tail",
        "MBA-5-worst-case-vs-realized",
    }
    expected_matrix_steps = {
        "MBS-1-dual-steepest-direction",
        "MBS-2-tail-sensitivity-bound",
        "MBS-3-frobenius-unit-ball",
        "MBS-4-spectral-sandwich-lemma",
        "MBS-5-rank-ratio",
        "MBS-6-degeneracy-and-alignment-caveat",
    }
    expected_matrix_claims = {
        "MBC-1-main-mechanism",
        "MBC-2-realized-drift",
        "MBC-3-tail-performance",
        "MBC-4-muon-scope",
    }
    expected_matrix_cross_checks = {
        "assump:local-head-tail",
        "app:proof-details",
        "lem:sandwiched-sensitivity",
        "app:matrix-block-derivation",
        "matched_gain_theorem",
        "rank_condition",
    }
    if not (
        set(matrix_theorem["statement_id"]) == expected_matrix_statements
        and set(matrix_assumptions["assumption_id"]) == expected_matrix_assumptions
        and set(matrix_steps["step_id"]) == expected_matrix_steps
        and set(matrix_claims["claim_id"]) == expected_matrix_claims
        and set(matrix_cross_checks["check_id"]) == expected_matrix_cross_checks
        and matrix_cross_checks["present"].eq("yes").all()
        and matrix_theorem["formal_expression"].astype(str).str.len().gt(20).all()
        and matrix_steps["mathematical_tool"].astype(str).str.len().gt(20).all()
        and matrix_claims["blocked_claim"].astype(str).str.len().gt(30).all()
        and matrix_config["all_paper_cross_checks_present"] is True
        and "no new empirical results" in str(matrix_config["analysis_scope"])
    ):
        raise AssertionError("matrix-block theorem proof contract must preserve theorem statements, assumptions, proof steps, claim boundaries, and paper cross-checks")
    matrix_proof_text = Path("discussion/e11_matrix_block_theorem_proof.md").read_text(
        encoding="utf-8"
    )
    assert_required_phrases(
        "matrix-block theorem proof",
        matrix_proof_text,
        [
            "E11 Matrix-Block Theorem Proof",
            "machine-checkable proof contract",
            "nrank(G_H) > ssrank(B_T,A_T)",
            "Theorem Statements",
            "Assumption Ledger",
            "Proof Steps",
            "Claim Implications",
            "Paper Cross-Checks",
            "sandwich sensitivity lemma",
            "worst-case bound",
        ],
    )
    matrix_tightness_dir = Path("results/e11_matrix_block_tightness_audit")
    matrix_tightness_cases = pd.read_csv(matrix_tightness_dir / "rank_boundary_cases.csv")
    matrix_tightness_checks = pd.read_csv(matrix_tightness_dir / "formula_checks.csv")
    matrix_tightness_caveats = pd.read_csv(matrix_tightness_dir / "caveat_checks.csv")
    matrix_tightness_config = json.loads((matrix_tightness_dir / "config.json").read_text(encoding="utf-8"))
    expected_tightness_cases = {
        "MBTA-1-spectral-favored": "spectral_smaller_worst_case_bound",
        "MBTA-2-equality-boundary": "equal_worst_case_bound",
        "MBTA-3-frobenius-favored": "frobenius_smaller_worst_case_bound",
        "MBTA-4-degenerate-tail": "degenerate_tail_boundary_not_applicable",
    }
    expected_tightness_caveats = {
        "MBTC-1-nondegenerate-tail-required",
        "MBTC-2-strict-inequality-required",
        "MBTC-3-worst-case-not-realized",
    }
    tightness_relation_lookup = matrix_tightness_cases.set_index("case_id")["predicted_relation"].to_dict()
    applicable_tightness_checks = matrix_tightness_checks[
        matrix_tightness_checks["tightness_status"].eq("exact_for_diagonal_singular_witness")
    ]
    if not (
        tightness_relation_lookup == expected_tightness_cases
        and set(matrix_tightness_caveats["caveat_id"]) == expected_tightness_caveats
        and len(applicable_tightness_checks) == 3
        and applicable_tightness_checks["ratio_absolute_error"].fillna(1.0).le(1e-12).all()
        and matrix_tightness_checks["tightness_status"].isin(
            {"exact_for_diagonal_singular_witness", "not_applicable_degenerate_tail"}
        ).all()
        and matrix_tightness_config["all_identity_checks_pass"] is True
        and float(matrix_tightness_config["max_ratio_absolute_error"]) <= 1e-12
        and "no new empirical results" in str(matrix_tightness_config["analysis_scope"])
    ):
        raise AssertionError(
            "matrix-block tightness audit must preserve exact witnesses, ratio identities, boundary cases, and caveats"
        )
    matrix_tightness_text = Path("discussion/e11_matrix_block_tightness_audit.md").read_text(
        encoding="utf-8"
    )
    assert_required_phrases(
        "matrix-block tightness audit",
        matrix_tightness_text,
        [
            "E11 Matrix-Block Tightness Audit",
            "deterministic sanity audit",
            "I_spectral / I_frobenius = ssrank(B_T,A_T) / nrank(G_H)",
            "MBTA-1-spectral-favored",
            "MBTA-2-equality-boundary",
            "MBTA-3-frobenius-favored",
            "MBTA-4-degenerate-tail",
            "not_applicable_degenerate_tail",
            "worst-case tail singular directions",
        ],
    )
    proof_dir = Path("results/e11_theory_proof_obligation_register")
    proof_obligations = pd.read_csv(proof_dir / "proof_obligations.csv")
    assumption_stress = pd.read_csv(proof_dir / "assumption_stress_tests.csv")
    claim_scopes = pd.read_csv(proof_dir / "claim_scope_boundaries.csv")
    theorem_queue = pd.read_csv(proof_dir / "theorem_to_experiment_queue.csv")
    proof_config = json.loads((proof_dir / "config.json").read_text(encoding="utf-8"))
    expected_proof_obligations = {
        "PTO-1-local-linearization",
        "PTO-2-matrix-block-boundary",
        "PTO-3-muon-approximation-scope",
        "PTO-4-v5-transport-score",
        "PTO-5-natural-falsification",
        "PTO-6-final-performance-separation",
    }
    expected_assumption_tests = {
        "AST-1-local-step",
        "AST-2-head-gain-matching",
        "AST-3-tail-quality",
        "AST-4-transport-stability",
        "AST-5-multiplicity-integrity",
    }
    expected_claim_scopes = {
        "main_theorem",
        "natural_drift_diagnostic",
        "predictive_condition",
        "natural_counterexample",
        "optimizer_benchmark",
    }
    if not (
        set(proof_obligations["obligation_id"]) == expected_proof_obligations
        and set(assumption_stress["assumption_id"]) == expected_assumption_tests
        and set(claim_scopes["claim_scope"]) == expected_claim_scopes
        and set(theorem_queue["priority"]).issuperset({"P0", "P1"})
        and proof_obligations["formal_object"].astype(str).str.len().gt(30).all()
        and proof_obligations["forbidden_wording"].astype(str).str.contains("do not", case=False).all()
        and claim_scopes["blocked_claim"].astype(str).str.len().gt(25).all()
        and theorem_queue["artifact_or_command"].astype(str).str.len().gt(20).all()
        and proof_config["primary_score"] == "condition_score_v5_transport_normalized_amplitude_minus_direction"
        and "no new empirical results" in str(proof_config["analysis_scope"])
    ):
        raise AssertionError("theory proof-obligation register must preserve obligations, assumptions, claim boundaries, and queue")
    proof_text = Path("discussion/e11_theory_proof_obligation_register.md").read_text(encoding="utf-8")
    assert_required_phrases(
        "theory proof-obligation register",
        proof_text,
        [
            "E11 Theory Proof-Obligation Register",
            "theory-facing top-conference checklist",
            "formal object",
            "Proof Obligations",
            "Assumption Stress Tests",
            "Claim Scope Boundaries",
            "Theorem-To-Experiment Queue",
            "PTO-4-v5-transport-score",
            "global training-dynamics theorem",
            "optimizer-performance",
        ],
    )
    v5_protocol_dir = Path("results/e11_condition_score_v5_theory_protocol")
    v5_theory_terms = pd.read_csv(v5_protocol_dir / "theory_term_register.csv")
    v5_score_contract = pd.read_csv(v5_protocol_dir / "score_contract.csv")
    v5_spent_policy = pd.read_csv(v5_protocol_dir / "spent_evidence_policy.csv")
    v5_split_requirements = pd.read_csv(v5_protocol_dir / "unspent_split_requirements.csv")
    v5_gates = pd.read_csv(v5_protocol_dir / "acceptance_gates.csv")
    expected_v5_term_ids = {
        "sandwiched_tail_drift",
        "direction_ratio_guardrail",
        "raw_residual_amplitude",
        "partition_transport_defect",
        "architecture_transport_defect",
        "early_depth_nuisance",
    }
    expected_v5_contract_ids = {
        "V5-C1-theorem-reduction",
        "V5-C2-target-separation",
        "V5-C3-transport-normalized-amplitude",
        "V5-C4-spent-final-quarantine",
        "V5-C5-freeze-before-final",
        "V5-C6-narrow-claim-fallback",
    }
    expected_v5_spent_ids = {
        "spent_v2_architecture_resnet34",
        "spent_v2_data_cifar10lt_original",
        "spent_v3_architecture_resnet50",
        "spent_v3_data_cifar10lt_alt",
        "spent_v4_architecture_wide_resnet50_2",
        "spent_v4_data_cifar10lt_mixed",
    }
    expected_v5_split_ids = {
        "v5_validation_cifar100lt_mod4_partition",
        "v5_final_architecture_resnext50_32x4d_cifar100lt",
        "v5_final_data_cifar10lt_cross_partition",
    }
    expected_v5_gate_ids = {
        "V5-1-theory-reduction",
        "V5-2-spent-quarantine",
        "V5-3-transport-before-scalar",
        "V5-4-freeze-before-final",
        "V5-5-final-residual-and-direction",
        "V5-6-narrow-claim-if-not-ready",
    }
    if not (
        set(v5_theory_terms["term_id"]) == expected_v5_term_ids
        and set(v5_score_contract["contract_id"]) == expected_v5_contract_ids
        and set(v5_spent_policy["split_id"]) == expected_v5_spent_ids
        and set(v5_split_requirements["split_id"]) == expected_v5_split_ids
        and set(v5_gates["gate_id"]) == expected_v5_gate_ids
    ):
        raise AssertionError(
            "condition-score v5 theory protocol must preserve theory terms, score contract, spent policy, unspent split requirements, and gates"
        )
    v5_term_lookup = v5_theory_terms.set_index("term_id")
    v5_contract_lookup = v5_score_contract.set_index("contract_id")
    v5_split_lookup = v5_split_requirements.set_index("split_id")
    if not (
        "B_T,l D_l A_T,l" in v5_term_lookup.loc["sandwiched_tail_drift", "mathematical_object"]
        and "positive" in v5_term_lookup.loc["direction_ratio_guardrail", "current_evidence"]
        and "sign reversal" in v5_term_lookup.loc["raw_residual_amplitude", "v5_rule"]
        and "partition-transport" in v5_term_lookup.loc["partition_transport_defect", "v5_rule"]
        and "transport normalization" in v5_contract_lookup.loc[
            "V5-C3-transport-normalized-amplitude", "requirement"
        ]
        and v5_spent_policy["forbidden_use"].astype(str).str.contains("score fitting").all()
        and v5_spent_policy["forbidden_use"].astype(str).str.contains("final P0 evidence").all()
        and set(v5_split_requirements["role"]) == {
            "validation_only",
            "fresh_final_heldout_architecture",
            "fresh_final_heldout_data_partition",
        }
        and v5_split_lookup.loc[
            "v5_final_architecture_resnext50_32x4d_cifar100lt", "architecture"
        ]
        == "ResNeXt50-32x4d CIFAR stem"
        and v5_split_lookup.loc[
            "v5_final_architecture_resnext50_32x4d_cifar100lt", "entrypoint"
        ]
        == "scripts/slurm/e11_cifar100_resnet_condition_score_v5_architecture_resnext50_32x4d.sbatch"
        and v5_split_lookup.loc[
            "v5_final_data_cifar10lt_cross_partition", "class_partition"
        ]
        == "head=0,3,4,6,9; tail=1,2,5,7,8"
    ):
        raise AssertionError(
            "condition-score v5 theory protocol must enforce transport-normalized amplitude, spent-final quarantine, and new unspent split entrypoints"
        )
    v5_protocol_text = Path("discussion/e11_condition_score_v5_theory_protocol.md").read_text(
        encoding="utf-8"
    )
    assert_required_phrases(
        "condition-score v5 theory protocol",
        v5_protocol_text,
        [
            "E11 Condition-Score V5 Theory Protocol",
            "not a positive P0 result",
            "transport-normalized residual-amplitude",
            "sandwiched tail-drift",
            "ResNeXt50-32x4d",
            "Blocked now: fitting, selecting, or thresholding a v5 score on any v2/v3/v4 final row",
        ],
    )
    v5_map_dir = Path("results/e11_condition_score_v5_theory_to_score_map")
    v5_theorem_proxy_map = pd.read_csv(v5_map_dir / "theorem_proxy_map.csv")
    v5_score_lineage = pd.read_csv(v5_map_dir / "score_lineage.csv")
    v5_transport_contract = pd.read_csv(v5_map_dir / "transport_normalization_contract.csv")
    v5_post_final_obligations = pd.read_csv(v5_map_dir / "post_final_transport_obligations.csv")
    v5_next_protocol_firewall = pd.read_csv(v5_map_dir / "next_protocol_firewall.csv")
    v5_predictions = pd.read_csv(v5_map_dir / "falsifiable_predictions.csv")
    v5_ablations = pd.read_csv(v5_map_dir / "ablation_matrix.csv")
    v5_readiness = pd.read_csv(v5_map_dir / "claim_readiness_ledger.csv")
    expected_v5_map_ids = {
        "M1-sandwiched-tail-risk",
        "M2-direction-ratio",
        "M3-source-depth-residual",
        "M4-partition-transport",
        "M5-architecture-transport",
    }
    expected_v5_prediction_ids = {
        "V5-P1-direction-guardrail",
        "V5-P2-residual-validation",
        "V5-P3-architecture-final",
        "V5-P4-data-final",
        "V5-P5-baseline-dominance",
    }
    expected_v5_ablation_ids = {
        "A1-direction-only",
        "A2-raw-amplitude-only",
        "A3-early-depth-only",
        "A4-v4-amplitude-minus-direction",
        "A5-transport-normalized-candidate",
    }
    expected_v5_transport_steps = {
        "T1-source-calibration",
        "T2-transport-tags",
        "T3-validation-freeze",
        "T4-final-evaluation",
        "T5-negative-path",
    }
    expected_v5_post_final_obligations = {
        "PFO-1-endpoint-factorization",
        "PFO-2-architecture-direction-transport",
        "PFO-3-data-partition-residual-transport",
        "PFO-4-post-final-quarantine",
        "PFO-5-negative-boundary-ledger",
    }
    expected_v5_next_protocol_firewall = {
        "NPF-1-unspent-split-reset",
        "NPF-2-endpoint-separated-object",
        "NPF-3-architecture-direction-term",
        "NPF-4-data-partition-residual-term",
        "NPF-5-negative-boundary-retention",
    }
    expected_v5_readiness_items = {
        "theory-to-score map",
        "v5 validation output",
        "v5 residual score freeze",
        "no final before freeze",
        "v5 final gate family",
        "predictive-condition claim",
    }
    expected_v5_score_ids = {
        "condition_score_v5_direction_axis_scaled_jvp_ratio",
        "condition_score_v5_raw_fro_amplitude_axis",
        "condition_score_v5_transport_normalized_amplitude_minus_direction",
        "condition_score_v5_transport_defect_penalty",
        "early_layer_prior",
        "source_observed_drift_positive_control",
        "condition_score_v5_validation_selected",
    }
    if not (
        set(v5_theorem_proxy_map["map_id"]) == expected_v5_map_ids
        and set(v5_predictions["prediction_id"]) == expected_v5_prediction_ids
        and set(v5_ablations["ablation_id"]) == expected_v5_ablation_ids
        and set(v5_transport_contract["step_id"]) == expected_v5_transport_steps
        and set(v5_post_final_obligations["obligation_id"]) == expected_v5_post_final_obligations
        and set(v5_next_protocol_firewall["firewall_id"]) == expected_v5_next_protocol_firewall
        and set(v5_readiness["item"]) == expected_v5_readiness_items
        and set(v5_score_lineage["score_id"]) == expected_v5_score_ids
    ):
        raise AssertionError(
            "condition-score v5 theory-to-score map must preserve theorem maps, predictions, ablations, transport steps, readiness rows, and score lineage"
        )
    v5_readiness_lookup = v5_readiness.set_index("item")["status"].to_dict()
    v5_readiness_evidence = v5_readiness.set_index("item")["evidence"].to_dict()
    if not (
        v5_theorem_proxy_map["claim_boundary"].astype(str).str.contains("claim", case=False).all()
        and v5_score_lineage["leakage_status"].eq("no spent final rows").all()
        and v5_transport_contract["forbidden_inputs"].astype(str).str.contains("final", case=False).all()
        and v5_predictions["pass_rule"].astype(str).str.contains("CI|threshold|beat", case=False, regex=True).all()
        and v5_ablations["required_v5_report"].astype(str).str.len().gt(20).all()
        and v5_readiness_lookup["theory-to-score map"] == "generated"
        and v5_readiness_lookup["v5 validation output"] == "generated"
        and v5_readiness_lookup["v5 residual score freeze"] == "frozen"
        and v5_readiness_lookup["no final before freeze"] == "pass"
        and v5_readiness_lookup["v5 final gate family"] == "completed_failed_boundary"
        and v5_readiness_lookup["predictive-condition claim"] == "not_ready"
        and "completed final gates failed"
        in str(v5_readiness_evidence["predictive-condition claim"])
        and "architecture direction-threshold failed"
        in str(v5_readiness_evidence["predictive-condition claim"])
        and v5_post_final_obligations["forbidden_shortcut"].astype(str).str.len().gt(20).all()
        and v5_post_final_obligations["claim_boundary"].astype(str).str.len().gt(20).all()
        and v5_next_protocol_firewall["forbidden_use_of_v5_failure"].astype(str).str.len().gt(20).all()
        and v5_next_protocol_firewall["pre_registration_gate"].astype(str).str.contains(
            "before|committed|frozen|without", case=False, regex=True
        ).all()
    ):
        raise AssertionError(
            "condition-score v5 theory-to-score map must keep leakage boundaries, quantitative gates, ablation reports, and not_ready P0 claim status"
        )
    firewall_text = " ".join(v5_next_protocol_firewall.astype(str).to_numpy().ravel())
    for phrase in [
        "new validation/final split family",
        "Residual-risk ranking and below-one direction classification",
        "Architecture transport needs a direction-threshold reliability term",
        "Data-partition transport needs a residual-risk term",
        "failed predictive scores are retained as falsifiers",
        "one endpoint pass cannot rescue a failed endpoint",
        "downgrade the claim rather than repair the spent score",
    ]:
        if phrase not in firewall_text:
            raise AssertionError(f"condition-score v5 next-protocol firewall missing phrase: {phrase}")
    v5_map_text = Path("discussion/e11_condition_score_v5_theory_to_score_map.md").read_text(
        encoding="utf-8"
    )
    assert_required_phrases(
        "condition-score v5 theory-to-score map",
        v5_map_text,
        [
            "E11 Condition-Score V5 Theory-to-Score Map",
            "Transport-Stable Sandwich Residual",
            "theory-to-measurement bridge",
            "Theorem Proxy Map",
            "Score Lineage",
            "Transport Normalization Contract",
            "Post-Final Failure Reading",
            "Post-Final Transport Obligations",
            "Next Protocol Firewall",
            "Falsifiable Predictions",
            "Required Ablation Matrix",
            "Claim Readiness Ledger",
            "completed final gates failed under the frozen score",
            "architecture-direction transport and data-partition residual transport",
            "Blocked now: fitting, selecting, thresholding, or reweighting any v5 score on",
        ],
    )
    for stale_v5_phrase in [
        "run the v5 final splits with the",
        "validation is not frozen",
    ]:
        if stale_v5_phrase in v5_map_text:
            raise AssertionError(
                f"condition-score v5 theory-to-score map retained stale wording: {stale_v5_phrase}"
            )
    score_ablation_dir = Path("results/e11_condition_score_ablation")
    score_ablation_summary = pd.read_csv(score_ablation_dir / "score_ablation_summary.csv")
    score_ablation_ladder = pd.read_csv(score_ablation_dir / "term_failure_ladder.csv")
    score_ablation_leakage = pd.read_csv(score_ablation_dir / "leakage_and_claim_boundary.csv")
    score_ablation_config = json.loads((score_ablation_dir / "config.json").read_text(encoding="utf-8"))
    expected_ablation_steps = {
        "L1-direction-guardrail-is-separate",
        "L2-raw-amplitude-needs-transport",
        "L3-depth-is-a-nuisance-baseline",
        "L4-v4-scalar-fails-data-transport",
        "L5-v5-candidate-is-frozen-not-proven",
    }
    expected_ablation_boundaries = {
        "B1-spent-v2-v3-v4",
        "B2-v5-validation",
        "B3-v5-finals",
        "B4-paper-wording",
    }
    score_ablation_lookup = score_ablation_summary.set_index(["generation", "split_role", "score_id"])
    v4_data_direction = score_ablation_lookup.loc[
        (
            "v4 frozen final axis audit",
            "fresh_final_heldout_data_partition",
            "v4_direction_axis_scaled_jvp_ratio",
        )
    ]
    v4_data_amplitude = score_ablation_lookup.loc[
        (
            "v4 frozen final axis audit",
            "fresh_final_heldout_data_partition",
            "v4_residual_amplitude_axis_scaled_jvp_fro",
        )
    ]
    v4_data_primary = score_ablation_lookup.loc[
        (
            "v4 frozen final axis audit",
            "fresh_final_heldout_data_partition",
            "condition_score_v4_two_axis_amplitude_minus_direction",
        )
    ]
    v5_primary = score_ablation_lookup.loc[
        (
            "v5 validation freeze",
            "validation_only",
            "condition_score_v5_transport_normalized_amplitude_minus_direction",
        )
    ]
    v5_direction = score_ablation_lookup.loc[
        (
            "v5 validation freeze",
            "validation_only",
            "condition_score_v5_direction_axis_scaled_jvp_ratio",
        )
    ]
    v5_final_data_primary = score_ablation_lookup.loc[
        (
            "v5 completed final evaluation",
            "v5_final_heldout_data_partition",
            "condition_score_v5_transport_normalized_amplitude_minus_direction",
        )
    ]
    v5_final_arch_direction = score_ablation_lookup.loc[
        (
            "v5 completed final evaluation",
            "v5_final_heldout_architecture",
            "condition_score_v5_direction_axis_scaled_jvp_ratio",
        )
    ]
    score_ablation_boundary_status = score_ablation_leakage.set_index("boundary_id")["claim_status"].to_dict()
    if not (
        set(score_ablation_ladder["ladder_step"]) == expected_ablation_steps
        and set(score_ablation_leakage["boundary_id"]) == expected_ablation_boundaries
        and score_ablation_config["uses_new_gpu_results"] is False
        and score_ablation_config["uses_spent_final_rows_for_tuning"] is False
        and score_ablation_config["claim_status"] == "completed_final_failed_boundary"
        and score_ablation_summary["leakage_status"]
        .isin(
            {
                "spent_final_row_diagnostic_only",
                "validation_only_no_final_rows",
                "completed_final_row_diagnostic_only",
            }
        )
        .all()
        and score_ablation_boundary_status["B3-v5-finals"] == "completed_final_failed_boundary"
        and score_ablation_boundary_status["B4-paper-wording"] == "local_mechanism_only_after_final_failure"
        and 0.60 <= float(v4_data_direction["residual_spearman"]) <= 0.62
        and float(v4_data_direction["spearman_ci95_low"]) > 0.58
        and -0.69 <= float(v4_data_amplitude["residual_spearman"]) <= -0.66
        and float(v4_data_amplitude["spearman_ci95_high"]) < -0.66
        and -0.71 <= float(v4_data_primary["residual_spearman"]) <= -0.67
        and float(v4_data_primary["spearman_ci95_high"]) < -0.67
        and 0.63 <= float(v5_primary["residual_spearman"]) <= 0.66
        and float(v5_primary["spearman_ci95_low"]) > 0.58
        and float(v5_direction["threshold_accuracy"]) == 1.0
        and float(v5_final_data_primary["spearman_ci95_high"]) < -0.65
        and float(v5_final_arch_direction["threshold_accuracy"]) < 0.85
        and str(v5_final_arch_direction["claim_use"]).startswith("completed direction readout")
        and score_ablation_leakage["forbidden_use"].astype(str).str.contains("fitting|changing|turning", regex=True).any()
    ):
        raise AssertionError(
            "condition-score ablation audit must preserve the v4 direction/amplitude reversal, v5 validation freeze, completed final failures, and no-final-row-tuning boundary"
        )
    score_ablation_text = Path("discussion/e11_condition_score_ablation.md").read_text(
        encoding="utf-8"
    )
    assert_required_phrases(
        "condition-score ablation audit",
        score_ablation_text,
        [
            "E11 Condition-Score Ablation Audit",
            "CPU-only theory-to-score ablation",
            "below-one direction guardrail",
            "residual layer-risk ranking",
            "partition/architecture transport",
            "direction axis positive",
            "raw amplitude",
            "transport-normalized",
            "completed v5 final evaluator",
            "completed final failures",
            "completed_final_failed_boundary",
            "Score-Axis Summary",
            "Term Failure Ladder",
            "Leakage and Claim Boundary",
            "Blocked now: using any v2/v3/v4 final row",
        ],
    )
    v5_validation_dir = Path("results/e11_condition_score_v5_protocol/validation_cifar100_mod4_partition")
    v5_validation_metrics = pd.read_csv(v5_validation_dir / "metrics.csv")
    v5_validation_paired = pd.read_csv(v5_validation_dir / "paired_metrics.csv")
    v5_validation_layer = pd.read_csv(v5_validation_dir / "layer_summary.csv")
    v5_validation_checkpoint = pd.read_csv(v5_validation_dir / "checkpoint_summary.csv")
    v5_validation_prediction = pd.read_csv(v5_validation_dir / "prediction_summary.csv")
    v5_validation_residual = pd.read_csv(v5_validation_dir / "residual_prediction_summary.csv")
    v5_validation_config = json.loads((v5_validation_dir / "config.json").read_text(encoding="utf-8"))
    if not (
        len(v5_validation_metrics) == 1260
        and len(v5_validation_paired) == 630
        and len(v5_validation_layer) == 63
        and len(v5_validation_checkpoint) == 3
        and len(v5_validation_prediction) == 6
        and len(v5_validation_residual) == 4
        and len(v5_validation_config["base_config"]["seeds"]) == 10
        and v5_validation_config["base_config"]["device"] == "cuda"
        and not bool(v5_validation_config["base_config"]["download"])
    ):
        raise AssertionError("condition-score v5 validation split must contain the full 10-seed GPU validation output")
    v5_validation_checkpoint_by_step = v5_validation_checkpoint.set_index("warmup_steps")
    if set(v5_validation_checkpoint_by_step.index) != {2000, 5000, 10000}:
        raise AssertionError("condition-score v5 validation split must cover warmup steps 2000/5000/10000")
    if not (
        0.22 <= float(
            v5_validation_checkpoint_by_step.loc[10000, "geomean_observed_tail_drift_sq_ratio_spectral_over_fro"]
        )
        <= 0.25
        and bool(v5_validation_layer["spectral_less_observed_tail_drift_fraction"].eq(1.0).all())
        and float(
            v5_validation_prediction.set_index("predictor").loc[
                "source_observed_drift_ratio", "mean_top5_risk_overlap_fraction"
            ]
        )
        == 1.0
        and float(
            v5_validation_residual.set_index("predictor").loc[
                "source_observed_residual", "mean_spearman_residual_predictor_vs_residual_target_observed"
            ]
        )
        > 0.9
    ):
        raise AssertionError("condition-score v5 validation split must preserve the positive validation drift/control readout")
    v5_freeze_dir = Path("results/e11_condition_score_v5_protocol/validation_score_freeze")
    v5_freeze_formulas = pd.read_csv(v5_freeze_dir / "score_formula_registry.csv")
    v5_freeze_status = pd.read_csv(v5_freeze_dir / "freeze_status.csv")
    v5_freeze_pairs = pd.read_csv(v5_freeze_dir / "validation_score_pairs.csv")
    v5_freeze_summary = pd.read_csv(v5_freeze_dir / "validation_score_summary.csv")
    v5_freeze_gates = pd.read_csv(v5_freeze_dir / "validation_gate_report.csv")
    v5_freeze_config = json.loads((v5_freeze_dir / "config.json").read_text(encoding="utf-8"))
    expected_v5_freeze_scores = {
        "condition_score_v5_direction_axis_scaled_jvp_ratio",
        "condition_score_v5_raw_fro_amplitude_axis",
        "condition_score_v5_transport_normalized_amplitude_minus_direction",
        "condition_score_v5_transport_defect_penalty",
        "early_layer_prior",
        "source_observed_drift_positive_control",
        "condition_score_v5_validation_selected",
    }
    expected_v5_freeze_items = {
        "v5 validation split output",
        "v5 candidate pool",
        "v5 transport-normalized residual score",
        "v5 final split outputs",
        "v5 spent-final quarantine",
    }
    expected_v5_freeze_gates = {
        "V5F-1-validation-output",
        "V5F-2-no-final-before-freeze",
        "V5F-3-spent-final-quarantine",
        "V5F-4-residual-score-freeze",
        "V5F-5-direction-threshold-guardrail",
        "V5F-6-final-claim-readiness",
    }
    if not (
        set(v5_freeze_formulas["score_id"]) == expected_v5_freeze_scores
        and set(v5_freeze_status["item"]) == expected_v5_freeze_items
        and set(v5_freeze_gates["gate_id"]) == expected_v5_freeze_gates
        and not v5_freeze_formulas["uses_spent_final_rows"].astype(str).str.contains("yes", case=False).any()
    ):
        raise AssertionError(
            "condition-score v5 validation freeze must preserve registered scores, status rows, gates, and spent-final exclusion"
        )
    v5_freeze_status_lookup = v5_freeze_status.set_index("item")["status"].to_dict()
    v5_freeze_evidence_lookup = v5_freeze_status.set_index("item")["evidence"].to_dict()
    v5_freeze_gate_lookup = v5_freeze_gates.set_index("gate_id")["status"].to_dict()
    if not bool(v5_freeze_config["validation_generated"]):
        if not (
            v5_freeze_pairs.empty
            and v5_freeze_summary.empty
            and v5_freeze_status_lookup["v5 validation split output"] == "not_run"
            and v5_freeze_status_lookup["v5 candidate pool"] == "registered"
            and v5_freeze_status_lookup["v5 transport-normalized residual score"] == "not_ready"
            and v5_freeze_evidence_lookup["v5 transport-normalized residual score"] == "pending_validation_output"
            and v5_freeze_status_lookup["v5 final split outputs"] == "not_run"
            and v5_freeze_status_lookup["v5 spent-final quarantine"] == "enforced"
            and v5_freeze_gate_lookup["V5F-1-validation-output"] == "not_run"
            and v5_freeze_gate_lookup["V5F-2-no-final-before-freeze"] == "pass"
            and v5_freeze_gate_lookup["V5F-3-spent-final-quarantine"] == "pass"
            and v5_freeze_gate_lookup["V5F-4-residual-score-freeze"] == "not_ready"
            and v5_freeze_gate_lookup["V5F-5-direction-threshold-guardrail"] == "not_run"
            and v5_freeze_gate_lookup["V5F-6-final-claim-readiness"] == "not_ready"
        ):
            raise AssertionError(
                "condition-score v5 validation freeze must block final claims cleanly before validation output exists"
            )
    else:
        selected_status = v5_freeze_status_lookup["v5 transport-normalized residual score"]
        selected_score = v5_freeze_evidence_lookup["v5 transport-normalized residual score"]
        selected_summary = v5_freeze_summary[v5_freeze_summary["score"].eq(selected_score)]
        if not (
            not v5_freeze_pairs.empty
            and not v5_freeze_summary.empty
            and v5_freeze_status_lookup["v5 validation split output"] == "generated"
            and v5_freeze_status_lookup["v5 final split outputs"] == "not_run"
            and selected_status in {"frozen", "validation_failed"}
            and selected_score == "condition_score_v5_transport_normalized_amplitude_minus_direction"
            and len(selected_summary) == 1
            and 0.63
            <= float(selected_summary.iloc[0]["mean_spearman_score_vs_target_residual"])
            <= 0.66
            and 0.58 <= float(selected_summary.iloc[0]["spearman_ci95_low"]) <= 0.60
            and v5_freeze_gate_lookup["V5F-1-validation-output"] == "pass"
            and v5_freeze_gate_lookup["V5F-2-no-final-before-freeze"] == "pass"
            and v5_freeze_gate_lookup["V5F-3-spent-final-quarantine"] == "pass"
            and v5_freeze_gate_lookup["V5F-5-direction-threshold-guardrail"] in {"pass", "fail"}
            and v5_freeze_gate_lookup["V5F-6-final-claim-readiness"] in {"pass", "not_ready"}
        ):
            raise AssertionError(
                "condition-score v5 validation freeze must either freeze or reject the residual candidate before final outputs exist"
            )
    v5_freeze_text = Path("discussion/e11_condition_score_v5_validation_freeze.md").read_text(
        encoding="utf-8"
    )
    assert_required_phrases(
        "condition-score v5 validation freeze",
        v5_freeze_text,
        [
            "E11 Condition-Score V5 Validation Freeze",
            "Formula Registry",
            "Freeze Status",
            "transport-normalized residual score",
            "V5F-2-no-final-before-freeze",
            "Final evaluation is now unblocked as a run",
            "A P0 claim still requires both final splits",
        ],
    )
    v5_final_dir = Path("results/e11_condition_score_v5_protocol/final_score_evaluation")
    v5_final_pairs = pd.read_csv(v5_final_dir / "final_score_pairs.csv")
    v5_final_summary = pd.read_csv(v5_final_dir / "final_score_summary.csv")
    v5_final_gates = pd.read_csv(v5_final_dir / "final_gate_report.csv")
    v5_final_config = json.loads((v5_final_dir / "config.json").read_text(encoding="utf-8"))
    expected_v5_final_split_ids = {
        "v5_final_architecture_resnext50_32x4d_cifar100lt",
        "v5_final_data_cifar10lt_cross_partition",
    }
    expected_v5_final_roles = {
        "v5_final_heldout_architecture",
        "v5_final_heldout_data_partition",
    }
    expected_v5_final_scores = {
        "condition_score_v5_transport_normalized_amplitude_minus_direction",
        "condition_score_v5_direction_axis_scaled_jvp_ratio",
        "early_layer_prior",
        "source_observed_drift_positive_control",
    }
    v5_final_config_splits = v5_final_config["final_splits"]
    v5_final_generated = {
        str(split["split_id"]): bool(split["generated"]) for split in v5_final_config_splits
    }
    v5_final_gate_lookup = v5_final_gates.set_index("gate_id")["status"].to_dict()
    if not (
        v5_final_config["primary_score"]
        == "condition_score_v5_transport_normalized_amplitude_minus_direction"
        and set(split["split_id"] for split in v5_final_config_splits) == expected_v5_final_split_ids
        and set(split["role"] for split in v5_final_config_splits) == expected_v5_final_roles
        and set(v5_final_generated) == expected_v5_final_split_ids
        and v5_final_gate_lookup["v5_p0_predictive_condition_claim"] in {"pass", "not_ready"}
    ):
        raise AssertionError("condition-score v5 final evaluator must preserve split registry, frozen score, and P0 gate")
    if not any(v5_final_generated.values()):
        expected_preoutput_gates = {
            "v5_final_heldout_architecture_generated",
            "v5_final_heldout_data_partition_generated",
            "v5_p0_predictive_condition_claim",
        }
        if not (
            v5_final_pairs.empty
            and v5_final_summary.empty
            and set(v5_final_gates["gate_id"]) == expected_preoutput_gates
            and v5_final_gate_lookup["v5_final_heldout_architecture_generated"] == "not_run"
            and v5_final_gate_lookup["v5_final_heldout_data_partition_generated"] == "not_run"
            and v5_final_gate_lookup["v5_p0_predictive_condition_claim"] == "not_ready"
        ):
            raise AssertionError("condition-score v5 final evaluator must report clean not_run gates before final outputs exist")
    else:
        if not (
            not v5_final_pairs.empty
            and not v5_final_summary.empty
            and set(v5_final_summary["split_id"]).issubset(expected_v5_final_split_ids)
            and set(v5_final_summary["score"]).issubset(expected_v5_final_scores)
            and v5_final_summary[
                v5_final_summary["score"].eq(
                    "condition_score_v5_transport_normalized_amplitude_minus_direction"
                )
            ]["score_role"].eq("primary_candidate").all()
            and v5_final_gates["status"].isin({"pass", "fail", "not_ready", "not_run"}).all()
        ):
            raise AssertionError("condition-score v5 final evaluator must score generated final outputs with registered scores only")
    v5_final_text = Path("discussion/e11_condition_score_v5_final_evaluation.md").read_text(
        encoding="utf-8"
    )
    assert_required_phrases(
        "condition-score v5 final evaluation",
        v5_final_text,
        [
            "E11 Condition-Score V5 Final Evaluation",
            "condition_score_v5_transport_normalized_amplitude_minus_direction",
            "does not refit",
            "does not reselect",
            "baseline-dominance",
            "`not_run` is the expected state only for final splits",
        ],
    )
    v5_power_dir = Path("results/e11_condition_score_v5_protocol/final_power_audit")
    v5_power_design = pd.read_csv(v5_power_dir / "split_power_design.csv")
    v5_power_fisher = pd.read_csv(v5_power_dir / "fisher_z_resolution.csv")
    v5_power_mde = pd.read_csv(v5_power_dir / "mean_spearman_mde.csv")
    v5_power_ladder = pd.read_csv(v5_power_dir / "interpretation_ladder.csv")
    v5_power_states = pd.read_csv(v5_power_dir / "outcome_state_machine.csv")
    v5_power_config = json.loads((v5_power_dir / "config.json").read_text(encoding="utf-8"))
    expected_v5_power_cases = {
        "V5-PWR-1-pending-outputs",
        "V5-PWR-2-positive-above-zero-ci",
        "V5-PWR-3-null-below-detectable-scale",
        "V5-PWR-4-negative-above-detectable-scale",
        "V5-PWR-5-high-heterogeneity",
    }
    expected_v5_power_states = {
        "V5-PWR-S1-not-run",
        "V5-PWR-S2-partial",
        "V5-PWR-S3-complete-positive",
        "V5-PWR-S4-complete-small-null",
        "V5-PWR-S5-complete-negative",
    }
    fisher_reference = v5_power_fisher[
        v5_power_fisher["points"].eq(int(v5_power_config["reference_points_per_transfer"]))
    ].iloc[0]
    mde_reference = v5_power_mde[
        v5_power_mde["assumed_across_pair_spearman_sd"].eq(0.2)
    ].iloc[0]
    if not (
        set(v5_power_design["split_id"]) == expected_v5_final_split_ids
        and set(v5_power_design["split_role"]) == expected_v5_final_roles
        and set(v5_power_design["primary_residual_gate"]) == {
            "mean Spearman CI lower endpoint above zero"
        }
        and int(v5_power_config["reference_points_per_transfer"]) == 21
        and int(v5_power_config["reference_transfer_pairs"]) == 9
        and v5_power_config["primary_score"]
        == "condition_score_v5_transport_normalized_amplitude_minus_direction"
        and "no final-row tuning" in str(v5_power_config["analysis_scope"])
        and set(v5_power_ladder["case_id"]) == expected_v5_power_cases
        and set(v5_power_states["state_id"]) == expected_v5_power_states
        and 0.42
        < float(fisher_reference["minimum_observed_spearman_for_ci_low_above_zero"])
        < 0.44
        and 0.12
        < float(mde_reference["minimum_mean_spearman_for_ci_low_above_zero"])
        < 0.14
    ):
        raise AssertionError("condition-score v5 final power audit must lock split registry and detectable-effect scales")
    v5_power_text = Path("discussion/e11_condition_score_v5_final_power_audit.md").read_text(
        encoding="utf-8"
    )
    assert_required_phrases(
        "condition-score v5 final power audit",
        v5_power_text,
        [
            "E11 Condition-Score V5 Final Power Audit",
            "pre-output detectable-effect contract",
            "Fisher-z resolution",
            "mean-Spearman MDE",
            "underpowered",
            "negative transport",
            "Outcome State Machine",
        ],
    )
    v5_interpret_dir = Path("results/e11_condition_score_v5_protocol/final_interpretation_plan")
    v5_interpret_current = pd.read_csv(v5_interpret_dir / "current_interpretation_summary.csv")
    v5_interpret_status = pd.read_csv(v5_interpret_dir / "final_split_status.csv")
    v5_interpret_gate_contract = pd.read_csv(v5_interpret_dir / "final_gate_contract.csv")
    v5_interpret_ladder = pd.read_csv(v5_interpret_dir / "outcome_interpretation_ladder.csv")
    v5_interpret_leakage = pd.read_csv(v5_interpret_dir / "leakage_lock.csv")
    v5_interpret_config = json.loads((v5_interpret_dir / "config.json").read_text(encoding="utf-8"))
    expected_v5_interpret_gates = {
        "V5-FINAL-G1-frozen-primary-score",
        "V5-FINAL-G2-output-completeness",
        "V5-FINAL-G3-residual-ranking",
        "V5-FINAL-G4-direction-guardrail",
        "V5-FINAL-G5-baseline-dominance",
        "V5-FINAL-G6-control-reporting",
    }
    expected_v5_interpret_claim_states = {
        "not_ready",
        "p0_claim_eligible",
        "data_transport_boundary",
        "architecture_transport_boundary",
        "direction_guardrail_failure",
        "nuisance_proxy_boundary",
        "local_mechanism_only",
    }
    expected_v5_interpret_locked_items = {
        "primary_score",
        "final_splits",
        "residual_gate",
        "direction_gate",
        "baseline_gate",
    }
    if not (
        len(v5_interpret_current) == 1
        and set(v5_interpret_status["split_id"]) == expected_v5_final_split_ids
        and v5_interpret_status["claim_gate_group"].eq("required_for_p0").all()
        and v5_interpret_status["can_be_replaced"].eq("no").all()
        and v5_interpret_status["can_be_dropped_after_result"].eq("no").all()
        and set(v5_interpret_gate_contract["gate_id"]) == expected_v5_interpret_gates
        and set(v5_interpret_ladder["claim_state"]) == expected_v5_interpret_claim_states
        and set(v5_interpret_leakage["locked_item"]) == expected_v5_interpret_locked_items
        and v5_interpret_config["primary_score"]
        == "condition_score_v5_transport_normalized_amplitude_minus_direction"
        and v5_interpret_config["current_claim_state"] == "completed_final_failed_boundary"
        and "no final-row tuning" in str(v5_interpret_config["analysis_scope"])
        and "post-output" in str(v5_interpret_config["analysis_scope"])
    ):
        raise AssertionError("condition-score v5 final interpretation plan must lock split, gate, outcome, and leakage policies")
    if not (
        v5_interpret_current["current_claim_state"].eq("completed_final_failed_boundary").all()
        and v5_interpret_current["active_ladder_states"]
        .astype(str)
        .str.contains("data_transport_boundary")
        .all()
        and v5_interpret_current["active_ladder_states"]
        .astype(str)
        .str.contains("direction_guardrail_failure")
        .all()
        and v5_interpret_current["active_ladder_states"].astype(str).str.contains("local_mechanism_only").all()
        and v5_interpret_current["blocking_gate_ids"]
        .astype(str)
        .str.contains("v5_final_heldout_architecture_direction_threshold_accuracy")
        .all()
        and v5_interpret_current["blocking_gate_ids"]
        .astype(str)
        .str.contains("v5_final_heldout_data_partition_residual_spearman")
        .all()
        and v5_interpret_current["forbidden_current_interpretation"]
        .astype(str)
        .str.contains("unseen-task predictive condition")
        .all()
        and v5_interpret_ladder["forbidden_interpretation"].astype(str).str.len().gt(20).all()
        and v5_interpret_ladder["required_paper_action"].astype(str).str.len().gt(30).all()
        and v5_interpret_leakage["forbidden_after_final_outputs"]
        .astype(str)
        .str.contains("changing|dropping|using|lowering|omitting", regex=True)
        .all()
    ):
        raise AssertionError(
            "condition-score v5 final interpretation plan must contain the completed final boundary, forbidden actions, and reporting actions"
        )
    v5_interpret_text = Path("discussion/e11_condition_score_v5_final_interpretation_plan.md").read_text(
        encoding="utf-8"
    )
    if "before their layer tables are available" in v5_interpret_text:
        raise AssertionError("condition-score v5 final interpretation plan contains stale pre-output table wording")
    assert_required_phrases(
        "condition-score v5 final interpretation plan",
        v5_interpret_text,
        [
            "E11 Condition-Score V5 Final Interpretation Plan",
            "post-output interpretation lock",
            "Current Interpretation Summary",
            "completed negative boundary",
            "completed_final_failed_boundary",
            "outcome-to-claim state machine",
            "direction-guardrail failures",
            "baseline-dominance failures",
            "Leakage Lock",
            "forbids changing",
        ],
    )
    v5_response_dir = Path("results/e11_condition_score_v5_protocol/reviewer_failure_response")
    v5_response_status = pd.read_csv(v5_response_dir / "final_split_output_status.csv")
    v5_response_gate_snapshot = pd.read_csv(v5_response_dir / "current_gate_snapshot.csv")
    v5_response_active = pd.read_csv(v5_response_dir / "active_failure_modes.csv")
    v5_response_modes = pd.read_csv(v5_response_dir / "failure_mode_register.csv")
    v5_response_objections = pd.read_csv(v5_response_dir / "reviewer_objection_map.csv")
    v5_response_downgrades = pd.read_csv(v5_response_dir / "claim_downgrade_actions.csv")
    v5_response_next = pd.read_csv(v5_response_dir / "next_evidence_queue.csv")
    v5_response_config = json.loads((v5_response_dir / "config.json").read_text(encoding="utf-8"))
    expected_v5_response_modes = {
        "V5-RFR-0-pending-outputs",
        "V5-RFR-1-both-final-splits-pass",
        "V5-RFR-2-data-transport-boundary",
        "V5-RFR-3-architecture-transport-boundary",
        "V5-RFR-4-direction-guardrail-failure",
        "V5-RFR-5-baseline-dominance-failure",
        "V5-RFR-6-control-reporting-failure",
        "V5-RFR-7-both-final-splits-fail",
        "V5-RFR-8-post-final-leakage-pressure",
    }
    expected_v5_response_claim_states = {
        "not_ready",
        "p0_claim_eligible",
        "single-axis-transport-boundary",
        "direction_guardrail_failure",
        "nuisance_proxy_boundary",
        "local_mechanism_only",
    }
    if not (
        set(v5_response_status["split_id"]) == expected_v5_final_split_ids
        and v5_response_status["pre_output_policy"].eq("do_not_change_score_or_split").all()
        and set(v5_response_modes["failure_mode_id"]) == expected_v5_response_modes
        and v5_response_modes["reviewer_objection"].astype(str).str.len().gt(30).all()
        and v5_response_modes["forbidden_claim"].astype(str).str.len().gt(25).all()
        and set(v5_response_modes["blocks_p0"]) == {"yes", "no"}
        and set(v5_response_downgrades["claim_state"]) == expected_v5_response_claim_states
        and v5_response_objections["remaining_evidence"].astype(str).str.len().gt(25).all()
        and set(v5_response_next["priority"]).issuperset({"P0", "P1"})
        and set(v5_response_gate_snapshot["gate_id"]) == set(v5_final_gates["gate_id"])
        and {
            "V5-RFR-2-data-transport-boundary",
            "V5-RFR-4-direction-guardrail-failure",
            "V5-RFR-7-both-final-splits-fail",
            "V5-RFR-current-p0-not-ready",
        }.issubset(
            set(v5_response_active["active_failure_mode_id"])
        )
        and v5_response_active[
            v5_response_active["active_failure_mode_id"].eq("V5-RFR-7-both-final-splits-fail")
        ]["allowed_current_wording"]
        .astype(str)
        .str.contains("completed final negative boundary")
        .all()
        and v5_response_active["forbidden_current_wording"].astype(str).str.len().gt(25).all()
        and v5_response_config["primary_score"]
        == "condition_score_v5_transport_normalized_amplitude_minus_direction"
        and "no final-row tuning" in str(v5_response_config["analysis_scope"])
    ):
        raise AssertionError("condition-score v5 reviewer failure response must preserve frozen score, split registry, and claim downgrades")
    v5_response_text = Path("discussion/e11_condition_score_v5_reviewer_failure_response.md").read_text(
        encoding="utf-8"
    )
    assert_required_phrases(
        "condition-score v5 reviewer failure response",
        v5_response_text,
        [
            "E11 Condition-Score V5 Reviewer Failure Response",
            "top-conference reviewer failure response",
            "claim-downgrade plan",
            "does not refit, reselect, retune",
            "Current Final Gate Snapshot",
            "Current Active Failure Modes",
            "V5-RFR-4-direction-guardrail-failure",
            "V5-RFR-7-both-final-splits-fail",
            "Failure Mode Register",
            "Reviewer Objection Map",
            "Claim Downgrade Actions",
            "Next Evidence Queue",
            "V5-RFR-8-post-final-leakage-pressure",
        ],
    )
    v5_direction_dir = Path("results/e11_condition_score_v5_protocol/direction_guardrail_failure_audit")
    v5_direction_axis = pd.read_csv(v5_direction_dir / "score_axis_contrast.csv")
    v5_direction_gates = pd.read_csv(v5_direction_dir / "gate_boundary_summary.csv")
    v5_direction_diagnosis = pd.read_csv(v5_direction_dir / "mechanistic_diagnosis.csv")
    v5_direction_requirements = pd.read_csv(v5_direction_dir / "next_protocol_requirements.csv")
    v5_direction_config = json.loads((v5_direction_dir / "config.json").read_text(encoding="utf-8"))
    expected_v5_direction_axes = {
        "architecture_primary_residual_ranking",
        "architecture_direction_threshold_guardrail",
        "architecture_early_layer_prior_baseline",
        "architecture_source_observed_positive_control",
        "data_primary_residual_ranking",
        "data_direction_threshold_guardrail",
        "data_early_layer_prior_baseline",
        "data_source_observed_positive_control",
    }
    expected_v5_direction_diagnoses = {
        "V5-DGF-1-architecture-residual-ranking-survives",
        "V5-DGF-2-architecture-direction-threshold-fails",
        "V5-DGF-3-data-residual-ranking-reverses",
        "V5-DGF-4-data-direction-threshold-survives",
        "V5-DGF-5-failure-modes-are-orthogonal",
        "V5-DGF-6-controls-do-not-rescue-p0",
    }
    expected_v5_direction_requirements = {
        "V5-DGF-NP1-separate-endpoints",
        "V5-DGF-NP2-separate-transport-axes",
        "V5-DGF-NP3-no-post-final-repair",
        "V5-DGF-NP4-direction-transport-term",
        "V5-DGF-NP5-preserve-negative-boundaries",
    }
    arch_primary_axis = v5_direction_axis[
        v5_direction_axis["axis_id"].eq("architecture_primary_residual_ranking")
    ].iloc[0]
    arch_direction_axis = v5_direction_axis[
        v5_direction_axis["axis_id"].eq("architecture_direction_threshold_guardrail")
    ].iloc[0]
    data_primary_axis = v5_direction_axis[
        v5_direction_axis["axis_id"].eq("data_primary_residual_ranking")
    ].iloc[0]
    data_direction_axis = v5_direction_axis[
        v5_direction_axis["axis_id"].eq("data_direction_threshold_guardrail")
    ].iloc[0]
    active_direction_diagnoses = set(
        v5_direction_diagnosis[v5_direction_diagnosis["active_failure_mode"].eq("yes")][
            "diagnosis_id"
        ]
    )
    if not (
        set(v5_direction_axis["axis_id"]) == expected_v5_direction_axes
        and arch_primary_axis["gate_status"] == "pass"
        and 0.42 < float(arch_primary_axis["mean_spearman"]) < 0.44
        and float(arch_primary_axis["spearman_ci95_low"]) > 0.0
        and arch_direction_axis["gate_status"] == "fail"
        and 0.83 < float(arch_direction_axis["threshold_accuracy"]) < 0.85
        and 0.68 < float(arch_direction_axis["threshold_ci95_low"]) < 0.69
        and data_primary_axis["gate_status"] == "fail"
        and -0.70 < float(data_primary_axis["mean_spearman"]) < -0.67
        and float(data_primary_axis["spearman_ci95_high"]) < 0.0
        and data_direction_axis["gate_status"] == "pass"
        and float(data_direction_axis["threshold_accuracy"]) == 1.0
        and float(data_direction_axis["threshold_ci95_low"]) == 1.0
        and set(v5_direction_gates["gate_id"]) == set(v5_final_gates["gate_id"])
        and "yes" in set(v5_direction_gates["blocks_p0"])
        and set(v5_direction_diagnosis["diagnosis_id"]) == expected_v5_direction_diagnoses
        and {"V5-DGF-2-architecture-direction-threshold-fails", "V5-DGF-3-data-residual-ranking-reverses"}.issubset(
            active_direction_diagnoses
        )
        and set(v5_direction_requirements["requirement_id"]) == expected_v5_direction_requirements
        and v5_direction_config["current_p0_status"] == "not_ready"
        and set(v5_direction_config["final_splits"]) == expected_v5_final_split_ids
        and "no final-row tuning or score repair" in str(v5_direction_config["analysis_scope"])
    ):
        raise AssertionError("condition-score v5 direction-guardrail failure audit must preserve the completed-final boundary")
    v5_direction_text = Path("discussion/e11_condition_score_v5_direction_guardrail_failure_audit.md").read_text(
        encoding="utf-8"
    )
    assert_required_phrases(
        "condition-score v5 direction-guardrail failure audit",
        v5_direction_text,
        [
            "E11 Condition-Score V5 Direction-Guardrail Failure Audit",
            "completed final boundary diagnosis",
            "residual ranking survives",
            "direction-threshold guardrail fails",
            "residual ranking reverses",
            "orthogonal final failures",
            "P0 remains not_ready",
            "no score repair",
            "new unspent protocol",
        ],
    )
    natural_boundary_dir = Path("results/e11_natural_head_tail_boundary")
    natural_registry = pd.read_csv(natural_boundary_dir / "search_registry.csv")
    natural_primary = pd.read_csv(natural_boundary_dir / "primary_drift_scan.csv")
    natural_secondary = pd.read_csv(natural_boundary_dir / "secondary_outcome_scan.csv")
    natural_summary = pd.read_csv(natural_boundary_dir / "boundary_summary.csv")
    natural_candidates = pd.read_csv(natural_boundary_dir / "candidate_negative_cases.csv")
    expected_natural_sources = {
        "digits_one_step",
        "digits_imbalance_ablation",
        "digits_checkpoint_sweep",
        "digits_class_partition_sweep",
        "digits_rho_sweep",
        "cifar100_resnet_one_step_rho005",
        "cifar100_resnet_one_step_rho002",
        "cifar100_resnet_checkpoint_sweep",
        "cifar100_resnet_tail_quality_control",
        "cifar100_resnet_imbalance_sweep",
        "cifar100_resnet_fc_condition_scatter",
    }
    expected_natural_summary_ids = {
        "primary_tail_output_drift",
        "all_ratio_metrics",
        "component_ratio_metrics",
        "secondary_tail_outcomes",
    }
    natural_primary_tail = natural_primary[natural_primary["metric_id"].eq("tail_output_drift")]
    natural_component_candidates = natural_candidates[
        natural_candidates["claim_boundary"].eq("component_metric_boundary")
    ]
    natural_outcome_candidates = natural_candidates[
        natural_candidates["claim_boundary"].eq("secondary_outcome_tradeoff")
    ]
    natural_summary_lookup = natural_summary.set_index("summary_id")
    if not (
        set(natural_registry["source_id"]) == expected_natural_sources
        and natural_registry["status"].eq("included").all()
        and natural_registry["primary_tail_drift_present"].eq("yes").all()
        and set(natural_summary["summary_id"]) == expected_natural_summary_ids
        and len(natural_primary_tail) == 37
        and natural_primary_tail["source_id"].nunique() == 11
        and natural_primary_tail["boundary_status"].eq("spectral_better").all()
        and natural_primary_tail["ci95_high"].lt(1.0).all()
        and natural_summary_lookup.loc[
            "primary_tail_output_drift", "claim_status"
        ]
        == "no_strict_natural_primary_counterexample_in_committed_scan"
        and int(natural_summary_lookup.loc["primary_tail_output_drift", "spectral_worse_count"]) == 0
        and not natural_component_candidates.empty
        and not natural_outcome_candidates.empty
        and "strict_primary_tail_drift" not in set(natural_candidates["claim_boundary"])
    ):
        raise AssertionError(
            "natural head-to-tail boundary audit must preserve the committed null primary-drift scan while recording component/outcome boundary candidates"
        )
    natural_boundary_text = Path("discussion/e11_natural_head_tail_boundary.md").read_text(
        encoding="utf-8"
    )
    assert_required_phrases(
        "natural head-to-tail boundary audit",
        natural_boundary_text,
        [
            "E11 Natural Head-to-Tail Boundary Audit",
            "Primary natural counterexample",
            "no_strict_natural_primary_counterexample_in_committed_scan",
            "Component and outcome tradeoffs constrain stronger loss, margin",
            "pre-registered natural negative-search experiment",
        ],
    )
    natural_protocol_dir = Path("results/e11_natural_negative_search_protocol")
    natural_protocol_baseline = pd.read_csv(natural_protocol_dir / "audit_baseline.csv")
    natural_protocol_search = pd.read_csv(natural_protocol_dir / "search_space_registry.csv")
    natural_protocol_metrics = pd.read_csv(natural_protocol_dir / "metric_contract.csv")
    natural_protocol_stopping = pd.read_csv(natural_protocol_dir / "stopping_rules.csv")
    natural_protocol_gates = pd.read_csv(natural_protocol_dir / "acceptance_gates.csv")
    natural_protocol_claims = pd.read_csv(natural_protocol_dir / "claim_ladder.csv")
    natural_protocol_status = pd.read_csv(natural_protocol_dir / "protocol_status.csv")
    natural_power_grid = pd.read_csv(natural_protocol_dir / "phase1_power_audit" / "power_grid.csv")
    natural_mde = pd.read_csv(natural_protocol_dir / "phase1_power_audit" / "minimum_detectable_effect.csv")
    natural_power_ladder = pd.read_csv(natural_protocol_dir / "phase1_power_audit" / "interpretation_ladder.csv")
    natural_phase2_power_grid = pd.read_csv(natural_protocol_dir / "phase2_power_audit" / "power_grid.csv")
    natural_phase2_mde = pd.read_csv(natural_protocol_dir / "phase2_power_audit" / "minimum_detectable_effect.csv")
    natural_phase2_power_ladder = pd.read_csv(natural_protocol_dir / "phase2_power_audit" / "interpretation_ladder.csv")
    natural_phase2_outcome_state = pd.read_csv(natural_protocol_dir / "phase2_power_audit" / "outcome_state_machine.csv")
    expected_natural_protocol_baselines = {
        "committed_natural_primary_full_drift_scan",
        "committed_component_ratio_boundaries",
        "committed_secondary_outcome_tradeoffs",
    }
    expected_natural_protocol_search_ids = {
        "NNS-P1-cifar100lt-resnet18-new-partitions",
        "NNS-P1-cifar10lt-resnet18-cross-partitions",
        "NNS-P1-tail-quality-controls",
        "NNS-P2-heldout-architecture-boundary",
    }
    expected_natural_protocol_metrics = {
        "primary_tail_output_drift_ratio",
        "centered_tail_output_drift_ratio",
        "true_logit_and_margin_components",
        "tail_loss_margin_accuracy_diffs",
        "head_gain_and_quality_controls",
    }
    expected_natural_protocol_rules = {
        "NNS-S1-freeze-before-fresh-runs",
        "NNS-S2-complete-phase-before-discovery",
        "NNS-S3-primary-success",
        "NNS-S4-finite-null",
        "NNS-S5-component-only",
        "NNS-S6-phase2-trigger",
    }
    expected_natural_protocol_gates = {
        "NNS-1-protocol-freeze",
        "NNS-2-freshness-exclusion",
        "NNS-3-multiplicity",
        "NNS-4-full-reporting",
        "NNS-5-quality-controls",
        "NNS-6-claim-boundary",
    }
    expected_natural_protocol_claims = {
        "fresh_natural_primary_counterexample",
        "finite_natural_null_search",
        "component_boundary_cases",
        "local_primary_full_drift_mechanism",
        "practical_optimizer_performance",
    }
    expected_natural_protocol_status_items = {
        "committed natural audit baseline",
        "fresh natural search protocol",
        "fresh natural search entrypoints",
        "fresh natural search settings registries",
        "fresh natural search outputs",
        "multiplicity-adjusted evaluator",
        "natural negative claim",
    }
    natural_protocol_status_lookup = natural_protocol_status.set_index("item")["status"].to_dict()
    natural_protocol_claim_lookup = natural_protocol_claims.set_index("claim_id")["current_status"].to_dict()
    natural_primary_metric = natural_protocol_metrics.set_index("metric_id").loc[
        "primary_tail_output_drift_ratio"
    ]
    planned_prefixes = [Path(path) for path in natural_protocol_search["planned_artifact_prefix"]]
    natural_phase1_search = natural_protocol_search[
        natural_protocol_search["phase"] == "phase1_fresh_primary_search"
    ]
    natural_phase2_search = natural_protocol_search[
        natural_protocol_search["phase"] == "phase2_fresh_generality_search"
    ]
    phase1_settings_counts = {
        row.search_id: len(pd.read_csv(Path(row.planned_artifact_prefix) / "settings_registry.csv"))
        for row in natural_phase1_search.itertuples()
    }
    natural_eval_dir = natural_protocol_dir / "phase1_multiplicity_evaluation"
    natural_eval_run_registry = pd.read_csv(natural_eval_dir / "run_registry.csv")
    natural_eval_seed_ratios = pd.read_csv(natural_eval_dir / "seed_level_primary_ratios.csv")
    natural_eval_decisions = pd.read_csv(natural_eval_dir / "primary_decisions.csv")
    natural_eval_gates = pd.read_csv(natural_eval_dir / "gate_report.csv")
    natural_interim_dir = natural_protocol_dir / "phase1_interim_synthesis"
    natural_interim_coverage = pd.read_csv(natural_interim_dir / "family_coverage.csv")
    natural_interim_summary = pd.read_csv(natural_interim_dir / "observed_primary_summary.csv")
    natural_interim_claim_boundary = pd.read_csv(natural_interim_dir / "claim_boundary.csv")
    natural_interim_remaining = pd.read_csv(natural_interim_dir / "remaining_work.csv")
    natural_interim_config = json.loads((natural_interim_dir / "config.json").read_text(encoding="utf-8"))
    natural_eval_gate_lookup = natural_eval_gates.set_index("gate_id")["status"].to_dict()
    natural_eval_run_status = natural_eval_run_registry.set_index("search_id")["output_status"].to_dict()
    natural_eval_run_pair_rows = natural_eval_run_registry.set_index("search_id")["pair_summary_rows"].to_dict()
    natural_eval_observed_count = int(natural_eval_decisions["output_status"].eq("observed").sum())
    natural_eval_not_run_count = int(natural_eval_decisions["output_status"].eq("not_run").sum())
    natural_eval_observed = natural_eval_decisions[natural_eval_decisions["output_status"].eq("observed")]
    natural_eval_not_run = natural_eval_decisions[natural_eval_decisions["output_status"].eq("not_run")]
    natural_phase1_cifar100_dir = natural_protocol_dir / "phase1_cifar100lt_resnet18"
    natural_phase1_cifar100_pair = pd.read_csv(natural_phase1_cifar100_dir / "pair_summary.csv")
    natural_phase1_cifar100_step = pd.read_csv(natural_phase1_cifar100_dir / "step_metrics.csv")
    natural_phase1_cifar100_layer = pd.read_csv(natural_phase1_cifar100_dir / "layer_metrics.csv")
    natural_phase1_cifar100_decision = pd.read_csv(natural_phase1_cifar100_dir / "decision_template.csv")
    natural_phase1_cifar10_dir = natural_protocol_dir / "phase1_cifar10lt_resnet18"
    natural_phase1_cifar10_pair = pd.read_csv(natural_phase1_cifar10_dir / "pair_summary.csv")
    natural_phase1_cifar10_step = pd.read_csv(natural_phase1_cifar10_dir / "step_metrics.csv")
    natural_phase1_cifar10_layer = pd.read_csv(natural_phase1_cifar10_dir / "layer_metrics.csv")
    natural_phase1_cifar10_decision = pd.read_csv(natural_phase1_cifar10_dir / "decision_template.csv")
    natural_phase2_dir = natural_protocol_dir / "phase2_heldout_architecture"
    natural_phase2_registry = pd.read_csv(natural_phase2_dir / "settings_registry.csv")
    natural_phase2_metric_paths = [
        natural_phase2_dir / "step_metrics.csv",
        natural_phase2_dir / "pair_summary.csv",
        natural_phase2_dir / "layer_metrics.csv",
        natural_phase2_dir / "decision_template.csv",
        natural_phase2_dir / "config.json",
    ]
    natural_phase2_step = pd.read_csv(natural_phase2_dir / "step_metrics.csv")
    natural_phase2_pair = pd.read_csv(natural_phase2_dir / "pair_summary.csv")
    natural_phase2_layer = pd.read_csv(natural_phase2_dir / "layer_metrics.csv")
    natural_phase2_decision = pd.read_csv(natural_phase2_dir / "decision_template.csv")
    natural_phase2_eval_dir = natural_protocol_dir / "phase2_multiplicity_evaluation"
    natural_phase2_eval_run_registry = pd.read_csv(natural_phase2_eval_dir / "run_registry.csv")
    natural_phase2_eval_seed_ratios = pd.read_csv(natural_phase2_eval_dir / "seed_level_primary_ratios.csv")
    natural_phase2_eval_decisions = pd.read_csv(natural_phase2_eval_dir / "primary_decisions.csv")
    natural_phase2_eval_gates = pd.read_csv(natural_phase2_eval_dir / "gate_report.csv")
    natural_phase2_eval_config = json.loads((natural_phase2_eval_dir / "config.json").read_text(encoding="utf-8"))
    natural_phase2_eval_gate_lookup = natural_phase2_eval_gates.set_index("gate_id")["status"].to_dict()
    natural_interim_coverage_lookup = natural_interim_coverage.set_index("search_id")[
        "observed_primary_rows"
    ].to_dict()
    natural_interim_all_observed = natural_interim_summary[
        natural_interim_summary["scope"].eq("all_observed")
    ].iloc[0]
    natural_protocol_checks = {
        "entrypoint_columns": {"entrypoint", "entrypoint_status"}.issubset(natural_protocol_search.columns),
        "baseline_ids": set(natural_protocol_baseline["baseline_id"]) == expected_natural_protocol_baselines,
        "search_ids": set(natural_protocol_search["search_id"]) == expected_natural_protocol_search_ids,
        "metric_ids": set(natural_protocol_metrics["metric_id"]) == expected_natural_protocol_metrics,
        "stopping_rules": set(natural_protocol_stopping["rule_id"]) == expected_natural_protocol_rules,
        "acceptance_gates": set(natural_protocol_gates["gate_id"]) == expected_natural_protocol_gates,
        "claim_ids": set(natural_protocol_claims["claim_id"]) == expected_natural_protocol_claims,
        "status_items": set(natural_protocol_status["item"]) == expected_natural_protocol_status_items,
        "baseline_strict_worse_zero": int(
            natural_protocol_baseline.set_index("baseline_id").loc[
                "committed_natural_primary_full_drift_scan", "strict_worse_count"
            ]
        )
        == 0,
        "protocol_generated": natural_protocol_status_lookup["fresh natural search protocol"] == "generated",
        "entrypoints_implemented": natural_protocol_status_lookup["fresh natural search entrypoints"] == "implemented",
        "settings_locked": natural_protocol_status_lookup["fresh natural search settings registries"] == "locked",
        "metric_outputs_complete": natural_protocol_status_lookup["fresh natural search outputs"] == "metric_outputs_complete",
        "evaluator_implemented": natural_protocol_status_lookup["multiplicity-adjusted evaluator"]
        == "implemented_complete_outputs",
        "claim_finite_null_candidate": natural_protocol_status_lookup["natural negative claim"]
        == "finite_null_candidate",
        "phase1_entrypoint": set(natural_phase1_search["entrypoint"])
        == {"scripts/slurm/e11_natural_negative_search_phase1.sbatch"},
        "phase1_entrypoint_status": set(natural_phase1_search["entrypoint_status"]) == {"implemented_sbatch"},
        "phase2_entrypoint": set(natural_phase2_search["entrypoint"])
        == {"scripts/slurm/e11_natural_negative_search_phase2.sbatch"},
        "phase2_entrypoint_status": set(natural_phase2_search["entrypoint_status"])
        == {"implemented_sbatch"},
        "phase1_settings_counts": phase1_settings_counts
        == {
            "NNS-P1-cifar100lt-resnet18-new-partitions": 12,
            "NNS-P1-cifar10lt-resnet18-cross-partitions": 8,
            "NNS-P1-tail-quality-controls": 6,
        },
        "phase1_cifar100_metric_rows": len(natural_phase1_cifar100_pair) == 12,
        "phase1_cifar100_step_rows": len(natural_phase1_cifar100_step) == 120,
        "phase1_cifar100_layer_rows": len(natural_phase1_cifar100_layer) > 0,
        "phase1_cifar100_decision_rows": len(natural_phase1_cifar100_decision) == 12,
        "phase1_cifar10_metric_rows": len(natural_phase1_cifar10_pair) == 8,
        "phase1_cifar10_step_rows": len(natural_phase1_cifar10_step) == 80,
        "phase1_cifar10_layer_rows": len(natural_phase1_cifar10_layer) > 0,
        "phase1_cifar10_decision_rows": len(natural_phase1_cifar10_decision) == 8,
        "phase2_settings_rows": len(natural_phase2_registry) == 8,
        "phase2_architecture": set(natural_phase2_registry["architecture"]) == {"ResNet34 CIFAR stem"},
        "phase2_seed_count": set(natural_phase2_registry["seed_count"].astype(int)) == {3},
        "phase2_metric_files_present": all(path.exists() for path in natural_phase2_metric_paths),
        "phase2_metric_rows": len(natural_phase2_pair) == 8
        and len(natural_phase2_step) == 48
        and len(natural_phase2_layer) > 0
        and len(natural_phase2_decision) == 8,
        "phase2_eval_run_registry": len(natural_phase2_eval_run_registry) == 1
        and set(natural_phase2_eval_run_registry["output_status"]) == {"complete"}
        and int(natural_phase2_eval_run_registry.iloc[0]["expected_settings"]) == 8
        and int(natural_phase2_eval_run_registry.iloc[0]["pair_summary_rows"]) == 8,
        "phase2_eval_decision_rows": len(natural_phase2_eval_decisions) == 8
        and natural_phase2_eval_decisions["output_status"].eq("observed").all()
        and natural_phase2_eval_decisions["adjusted_primary_decision"].eq("not_primary_worse_adjusted").all()
        and natural_phase2_eval_decisions["inference_source"].eq("paired_seed_log_ratio_t_test").all()
        and natural_phase2_eval_decisions["observed_seeds"].astype(int).eq(3).all()
        and natural_phase2_eval_decisions["head_gain_gate"].astype(str).str.lower().eq("false").all()
        and natural_phase2_eval_decisions["tail_quality_gate"].astype(str).str.lower().eq("true").all()
        and natural_phase2_eval_decisions["quality_gate"].astype(str).str.lower().eq("false").all(),
        "phase2_eval_seed_rows": len(natural_phase2_eval_seed_ratios) == 24,
        "phase2_eval_gate_status": natural_phase2_eval_gate_lookup
        == {
            "NNS-P2-E1-evaluator-implemented": "pass",
            "NNS-P2-E2-phase2-output-completeness": "pass",
            "NNS-P2-E3-primary-multiplicity": "pass",
            "NNS-P2-E4-heldout-architecture-claim": "finite_null_candidate",
            "NNS-P2-E5-full-reporting-boundary": "pass",
        },
        "phase2_eval_config": natural_phase2_eval_config["multiplicity_family"]
        == "NNS-P2-heldout-architecture-family"
        and int(natural_phase2_eval_config["planned_family_size"]) == 8,
        "eval_decision_count": len(natural_eval_decisions) == 26,
        "eval_observed_count": natural_eval_observed_count == 26,
        "eval_not_run_count": natural_eval_not_run_count == 0,
        "eval_seed_ratio_rows": len(natural_eval_seed_ratios) == 130,
        "eval_observed_inference_source": bool(
            natural_eval_observed["inference_source"].eq("paired_seed_log_ratio_t_test").all()
        ),
        "eval_not_run_missing_source": bool(natural_eval_not_run["inference_source"].eq("missing_output").all()),
        "eval_observed_adjusted_null": bool(
            natural_eval_observed["adjusted_primary_decision"].eq("not_primary_worse_adjusted").all()
        ),
        "eval_claims_no_counterexample": bool(
            natural_eval_decisions["claim_status"].eq("no_primary_counterexample_for_setting").all()
        ),
        "eval_run_status": natural_eval_run_status
        == {
            "NNS-P1-cifar100lt-resnet18-new-partitions": "complete",
            "NNS-P1-cifar10lt-resnet18-cross-partitions": "complete",
            "NNS-P1-tail-quality-controls": "complete",
        },
        "eval_run_pair_rows": {key: int(value) for key, value in natural_eval_run_pair_rows.items()}
        == {
            "NNS-P1-cifar100lt-resnet18-new-partitions": 12,
            "NNS-P1-cifar10lt-resnet18-cross-partitions": 8,
            "NNS-P1-tail-quality-controls": 6,
        },
        "eval_gate_implemented": natural_eval_gate_lookup["NNS-E1-evaluator-implemented"] == "pass",
        "eval_gate_completeness": natural_eval_gate_lookup["NNS-E2-phase1-output-completeness"] == "pass",
        "eval_gate_multiplicity": natural_eval_gate_lookup["NNS-E3-primary-multiplicity"] == "pass",
        "eval_gate_finite_null": natural_eval_gate_lookup["NNS-E4-natural-primary-claim"]
        == "finite_null_candidate",
        "eval_gate_reporting": natural_eval_gate_lookup["NNS-E5-full-reporting-boundary"] == "pass",
        "interim_coverage_rows": {key: int(value) for key, value in natural_interim_coverage_lookup.items()}
        == {
            "NNS-P1-cifar100lt-resnet18-new-partitions": 12,
            "NNS-P1-cifar10lt-resnet18-cross-partitions": 8,
            "NNS-P1-tail-quality-controls": 6,
        },
        "interim_all_observed": int(natural_interim_all_observed["observed_primary_rows"]) == 26,
        "interim_missing": int(natural_interim_all_observed["missing_primary_rows"]) == 0,
        "interim_raw_worse_zero": int(natural_interim_all_observed["raw_worse_rows"]) == 0,
        "interim_quality_failures_recorded": int(natural_interim_all_observed["quality_gate_fail_rows"]) == 23,
        "interim_claim_boundary_caveated": set(natural_interim_claim_boundary["current_status"])
        == {
            "no_adjusted_primary_counterexample",
            "finite_null_candidate",
            "quality_caveated_complete_family",
        },
        "interim_remaining_empty": len(natural_interim_remaining) == 0,
        "interim_config_finite_null": natural_interim_config["finite_null_candidate"] is True
        and int(natural_interim_config["observed_primary_rows"]) == 26
        and int(natural_interim_config["missing_primary_rows"]) == 0,
        "claim_ladder_no_counterexample": natural_protocol_claim_lookup["fresh_natural_primary_counterexample"]
        == "not_found_in_phase1",
        "claim_ladder_finite_null": natural_protocol_claim_lookup["finite_natural_null_search"]
        == "finite_null_candidate",
        "primary_rule_holm": "Holm-adjusted" in str(natural_primary_metric["worse_rule"]),
        "primary_boundary": "full-drift counterexample" in str(natural_primary_metric["claim_boundary"]),
        "power_grid_shape": len(natural_power_grid) == 72,
        "mde_shape": len(natural_mde) == 36,
        "power_ladder_shape": set(natural_power_ladder["case_id"])
        == {
            "PWR-1-adjusted-positive",
            "PWR-2-complete-null-above-mde",
            "PWR-3-complete-null-below-mde",
            "PWR-4-incomplete-family",
        },
        "adjusted_mde_above_one": natural_mde[
            natural_mde["alpha_scope"].eq("holm_bonferroni_worst_case")
        ]["minimum_detectable_ratio"].gt(1.0).all(),
        "phase2_power_grid_shape": len(natural_phase2_power_grid) == 72,
        "phase2_mde_shape": len(natural_phase2_mde) == 36,
        "phase2_power_family": set(natural_phase2_power_grid["seed_count"].astype(int)) == {3}
        and set(natural_phase2_power_grid["family_size"].astype(int)) == {1, 8},
        "phase2_power_ladder_shape": set(natural_phase2_power_ladder["case_id"])
        == {
            "P2-PWR-1-adjusted-positive",
            "P2-PWR-2-complete-null-above-mde",
            "P2-PWR-3-complete-null-below-mde",
            "P2-PWR-4-phase1-null-phase2-positive",
            "P2-PWR-5-incomplete-family",
        },
        "phase2_outcome_states": set(natural_phase2_outcome_state["state_id"])
        == {
            "P2-S1-not-run",
            "P2-S2-partial",
            "P2-S3-adjusted-positive",
            "P2-S4-complete-null-above-mde",
            "P2-S5-complete-null-below-mde",
            "P2-S6-quality-failure",
            "P2-S7-complete-null-head-gain-caveat",
        },
        "phase2_current_state": natural_phase2_outcome_state[
            natural_phase2_outcome_state["current_match"].astype(str).eq("yes")
        ]["state_id"].tolist()
        == ["P2-S7-complete-null-head-gain-caveat"],
        "phase2_current_evidence": natural_phase2_outcome_state[
            natural_phase2_outcome_state["current_match"].astype(str).eq("yes")
        ]["current_evidence"].astype(str).str.contains(
            "8/8 observed; adjusted_worse_rows=0; head_gain_gate_fail_rows=8; tail_quality_gate_pass_rows=8",
            regex=False,
        ).all(),
        "phase2_adjusted_mde_above_one": natural_phase2_mde[
            natural_phase2_mde["alpha_scope"].eq("holm_bonferroni_worst_case")
        ]["minimum_detectable_ratio"].gt(1.0).all(),
    }
    if not all(natural_protocol_checks.values()):
        failed_checks = [name for name, passed in natural_protocol_checks.items() if not passed]
        raise AssertionError(
            "natural negative-search protocol must preserve frozen search space, complete phase1 and phase2 outputs, implemented evaluators, adjusted primary rules, and caveated claim boundaries; failed checks: "
            f"{failed_checks}"
        )
    natural_protocol_text = Path("discussion/e11_natural_negative_search_protocol.md").read_text(
        encoding="utf-8"
    )
    assert_required_phrases(
        "natural negative-search protocol",
        natural_protocol_text,
        [
            "E11 Natural Negative Search Protocol",
            "pre-registered fresh search",
            "does not claim a new natural counterexample",
            "multiplicity procedure",
            "metric_outputs_complete",
            "26/26 fresh metric rows exist",
            "Phase1 boundary: The phase1 Slurm outputs are complete",
            "Phase2 boundary",
            "ResNet34 CIFAR stem",
            "phase2_heldout_architecture/settings_registry.csv",
            "phase2_multiplicity_evaluation",
            "phase2_power_audit",
            "scripts/e11_evaluate_natural_negative_search_phase2.py",
            "scripts/slurm/e11_natural_negative_search_phase2.sbatch",
            "phase2 family has 8/8 metric rows",
            "head-gain",
            "multiplicity evaluator",
            "finite_null_candidate",
            "Blocked now: claiming a fresh natural primary counterexample",
        ],
    )
    natural_power_text = Path("discussion/e11_natural_negative_search_phase1_power_audit.md").read_text(
        encoding="utf-8"
    )
    assert_required_phrases(
        "natural negative-search phase1 power audit",
        natural_power_text,
        [
            "E11 Natural Negative Search Phase1 Power Audit",
            "detectable-effect boundary",
            "26-setting natural negative-search phase1 family",
            "Bonferroni `0.05 / 26`",
            "Adjusted Minimum Detectable Ratio",
            "Interpretation Ladder",
            "underpowered for small natural negative effects",
        ],
    )
    natural_phase2_power_text = Path("discussion/e11_natural_negative_search_phase2_power_audit.md").read_text(
        encoding="utf-8"
    )
    assert_required_phrases(
        "natural negative-search phase2 power audit",
        natural_phase2_power_text,
        [
            "E11 Natural Negative Search Phase2 Power Audit",
            "detectable-effect and interpretation boundary",
            "8-setting ResNet34 held-out architecture phase2 family",
            "3 seeds per setting",
            "Current Post-Output Reading",
            "P2-S7-complete-null-head-gain-caveat",
            "head-gain",
            "Adjusted Minimum Detectable Ratio",
            "Outcome State Machine",
            "underpowered",
            "P2-S5-complete-null-below-mde",
        ],
    )
    natural_phase1_eval_text = Path("discussion/e11_natural_negative_search_phase1_evaluation.md").read_text(
        encoding="utf-8"
    )
    assert_required_phrases(
        "natural negative-search phase1 evaluation",
        natural_phase1_eval_text,
        [
            "E11 Natural Negative Search Phase1 Evaluation",
            "multiplicity boundary",
            "all 26 declared settings",
            "Holm",
            "paired per-seed log-ratio tests",
            "Current primary metric coverage: 26/26 settings",
            "Current seed-level primary rows: 130",
            "NNS-E2-phase1-output-completeness",
            "finite_null_candidate",
        ],
    )
    natural_interim_text = Path("discussion/e11_natural_negative_search_phase1_interim_synthesis.md").read_text(
        encoding="utf-8"
    )
    assert_required_phrases(
        "natural negative-search phase1 interim synthesis",
        natural_interim_text,
        [
            "E11 Natural Negative Search Phase1 Interim Synthesis",
            "26/26 observed",
            "raw_worse_rows=0",
            "finite-null candidate",
            "tail-quality and detectable-effect caveats",
            "no adjusted primary full-drift counterexample",
            "NNI-1-primary-natural-counterexample",
            "finite_null_candidate",
            "unqualified finite null over all natural settings",
        ],
    )
    natural_phase2_text = Path(
        "discussion/e11_natural_negative_search_phase2_NNS-P2-heldout-architecture-boundary.md"
    ).read_text(encoding="utf-8")
    assert_required_phrases(
        "natural negative-search phase2 held-out architecture settings",
        natural_phase2_text,
        [
            "E11 Natural Negative Search Phase2 Held-Out Architecture Outputs",
            "ResNet34",
            "Raw Primary Readout",
            "Decision Boundary",
            "settings_registry.csv",
            "pair_summary.csv",
        ],
    )
    natural_phase2_eval_text = Path("discussion/e11_natural_negative_search_phase2_evaluation.md").read_text(
        encoding="utf-8"
    )
    assert_required_phrases(
        "natural negative-search phase2 evaluation",
        natural_phase2_eval_text,
        [
            "E11 Natural Negative Search Phase2 Evaluation",
            "8 declared settings",
            "Current primary metric coverage: 8/8 settings",
            "Current seed-level primary rows: 24",
            "head_gain_gate fails in 8/8 rows",
            "NNS-P2-E2-phase2-output-completeness",
            "finite_null_candidate",
            "not_primary_worse_adjusted",
            "paired per-seed log-ratio tests",
        ],
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
    lt_recipe_dir = Path("results/e11_cifar100_resnet_lt_recipe_benchmark")
    lt_recipe_trace = pd.read_csv(lt_recipe_dir / "train_trace.csv")
    lt_recipe_class_metrics = pd.read_csv(lt_recipe_dir / "class_metrics.csv")
    lt_recipe_group_metrics = pd.read_csv(lt_recipe_dir / "group_metrics.csv")
    lt_recipe_summary = pd.read_csv(lt_recipe_dir / "summary.csv")
    lt_recipe_pairs = pd.read_csv(lt_recipe_dir / "pair_summary.csv")
    lt_recipe_config = json.loads((lt_recipe_dir / "config.json").read_text())
    expected_recipe_names = {"adamw_aug_ce", "adamw_aug_cb_loss", "sgd_aug_ce"}
    if (
        len(lt_recipe_trace) != 90
        or len(lt_recipe_class_metrics) != 1500
        or len(lt_recipe_group_metrics) != 60
        or len(lt_recipe_summary) != 12
        or len(lt_recipe_pairs) != 8
    ):
        raise AssertionError(
            "CIFAR-100-LT ResNet18 recipe benchmark must contain 5 seeds x 3 recipes with four group summaries"
        )
    if not (
        len(lt_recipe_config["seeds"]) == 5
        and set(lt_recipe_config["recipe_names"]) == expected_recipe_names
        and lt_recipe_config["baseline_recipe"] == "adamw_aug_ce"
        and int(lt_recipe_config["num_classes"]) == 100
        and int(lt_recipe_config["train_steps"]) == 5000
        and int(lt_recipe_config["train_batch_size"]) == 256
        and lt_recipe_config["device"] == "cuda"
        and not lt_recipe_config["download"]
    ):
        raise AssertionError("CIFAR-100-LT ResNet18 recipe benchmark should be the formal 5-seed Slurm/GPU no-download run")
    if set(lt_recipe_summary["recipe"]) != expected_recipe_names or set(lt_recipe_pairs["recipe"]) != {
        "adamw_aug_cb_loss",
        "sgd_aug_ce",
    }:
        raise AssertionError("CIFAR-100-LT ResNet18 recipe benchmark must cover the expected recipes and paired comparisons")
    lt_recipe_by_group = lt_recipe_summary.set_index(["recipe", "frequency_group"])
    sgd_all = float(lt_recipe_by_group.loc[("sgd_aug_ce", "all"), "mean_balanced_accuracy"])
    sgd_few = float(lt_recipe_by_group.loc[("sgd_aug_ce", "few"), "mean_balanced_accuracy"])
    adamw_aug_all = float(lt_recipe_by_group.loc[("adamw_aug_ce", "all"), "mean_balanced_accuracy"])
    adamw_aug_few = float(lt_recipe_by_group.loc[("adamw_aug_ce", "few"), "mean_balanced_accuracy"])
    lt_recipe_pair_by_group = lt_recipe_pairs.set_index(["recipe", "frequency_group"])
    sgd_few_diff = float(lt_recipe_pair_by_group.loc[("sgd_aug_ce", "few"), "mean_balanced_accuracy_diff"])
    sgd_all_diff = float(lt_recipe_pair_by_group.loc[("sgd_aug_ce", "all"), "mean_balanced_accuracy_diff"])
    cb_few_diff = float(lt_recipe_pair_by_group.loc[("adamw_aug_cb_loss", "few"), "mean_balanced_accuracy_diff"])
    if not (
        0.35 <= adamw_aug_all <= 0.37
        and 0.075 <= adamw_aug_few <= 0.095
        and 0.40 <= sgd_all <= 0.42
        and 0.09 <= sgd_few <= 0.12
        and 0.005 <= sgd_few_diff <= 0.035
        and 0.04 <= sgd_all_diff <= 0.06
        and cb_few_diff < 0.0
        and sgd_all > adamw_aug_all
        and sgd_few > adamw_aug_few
    ):
        raise AssertionError(
            "CIFAR-100-LT ResNet18 recipe benchmark should preserve the current SGD-aug pilot improvement and class-balanced-loss caveat"
        )
    lt_muon_dir = Path("results/e11_cifar100_resnet_lt_muon_final_benchmark")
    lt_muon_trace = pd.read_csv(lt_muon_dir / "train_trace.csv")
    lt_muon_class_metrics = pd.read_csv(lt_muon_dir / "class_metrics.csv")
    lt_muon_group_metrics = pd.read_csv(lt_muon_dir / "group_metrics.csv")
    lt_muon_summary = pd.read_csv(lt_muon_dir / "summary.csv")
    lt_muon_pairs = pd.read_csv(lt_muon_dir / "pair_summary.csv")
    lt_muon_config = json.loads((lt_muon_dir / "config.json").read_text())
    expected_muon_recipe_names = {"adamw_aug_ce", "ns_muon_aug_lr3e-5", "ns_muon_aug_lr1e-4"}
    if (
        len(lt_muon_trace) != 54
        or len(lt_muon_class_metrics) != 900
        or len(lt_muon_group_metrics) != 36
        or len(lt_muon_summary) != 12
        or len(lt_muon_pairs) != 8
    ):
        raise AssertionError(
            "CIFAR-100-LT ResNet18 NS-Muon final benchmark must contain 3 seeds x 3 recipes with four group summaries"
        )
    if not (
        len(lt_muon_config["seeds"]) == 3
        and set(lt_muon_config["recipe_names"]) == expected_muon_recipe_names
        and lt_muon_config["baseline_recipe"] == "adamw_aug_ce"
        and int(lt_muon_config["num_classes"]) == 100
        and int(lt_muon_config["train_steps"]) == 5000
        and int(lt_muon_config["train_batch_size"]) == 256
        and lt_muon_config["device"] == "cuda"
        and not lt_muon_config["download"]
    ):
        raise AssertionError(
            "CIFAR-100-LT ResNet18 NS-Muon final benchmark should be the formal 3-seed Slurm/GPU no-download run"
        )
    if set(lt_muon_summary["recipe"]) != expected_muon_recipe_names or set(lt_muon_pairs["recipe"]) != {
        "ns_muon_aug_lr3e-5",
        "ns_muon_aug_lr1e-4",
    }:
        raise AssertionError("CIFAR-100-LT ResNet18 NS-Muon final benchmark must cover expected recipes and pairs")
    lt_muon_by_group = lt_muon_summary.set_index(["recipe", "frequency_group"])
    muon_pair_by_group = lt_muon_pairs.set_index(["recipe", "frequency_group"])
    muon_adamw_all = float(lt_muon_by_group.loc[("adamw_aug_ce", "all"), "mean_balanced_accuracy"])
    muon_adamw_few = float(lt_muon_by_group.loc[("adamw_aug_ce", "few"), "mean_balanced_accuracy"])
    muon_lr1e4_all = float(lt_muon_by_group.loc[("ns_muon_aug_lr1e-4", "all"), "mean_balanced_accuracy"])
    muon_lr1e4_few = float(lt_muon_by_group.loc[("ns_muon_aug_lr1e-4", "few"), "mean_balanced_accuracy"])
    muon_lr3e5_all = float(lt_muon_by_group.loc[("ns_muon_aug_lr3e-5", "all"), "mean_balanced_accuracy"])
    muon_lr3e5_few = float(lt_muon_by_group.loc[("ns_muon_aug_lr3e-5", "few"), "mean_balanced_accuracy"])
    muon_lr1e4_all_diff = float(
        muon_pair_by_group.loc[("ns_muon_aug_lr1e-4", "all"), "mean_balanced_accuracy_diff"]
    )
    muon_lr1e4_few_diff = float(
        muon_pair_by_group.loc[("ns_muon_aug_lr1e-4", "few"), "mean_balanced_accuracy_diff"]
    )
    muon_lr3e5_all_diff = float(
        muon_pair_by_group.loc[("ns_muon_aug_lr3e-5", "all"), "mean_balanced_accuracy_diff"]
    )
    if not (
        0.35 <= muon_adamw_all <= 0.37
        and 0.075 <= muon_adamw_few <= 0.105
        and 0.12 <= muon_lr1e4_all <= 0.14
        and muon_lr1e4_few <= 0.003
        and 0.07 <= muon_lr3e5_all <= 0.09
        and abs(muon_lr3e5_few) < 1e-12
        and muon_lr1e4_all_diff < -0.2
        and muon_lr1e4_few_diff < -0.07
        and muon_lr3e5_all_diff < -0.27
        and muon_adamw_all > muon_lr1e4_all > muon_lr3e5_all
    ):
        raise AssertionError(
            "CIFAR-100-LT ResNet18 NS-Muon final benchmark should preserve the current negative final-performance boundary"
        )
    tuned_protocol_dir = Path("results/e11_cifar100_resnet_lt_tuned_benchmark_protocol")
    tuned_pilot_context = pd.read_csv(tuned_protocol_dir / "pilot_context.csv")
    tuned_scope = pd.read_csv(tuned_protocol_dir / "benchmark_scope.csv")
    tuned_seed_split = pd.read_csv(tuned_protocol_dir / "seed_split_contract.csv")
    tuned_recipe_grid = pd.read_csv(tuned_protocol_dir / "recipe_grid.csv")
    tuned_selection_rules = pd.read_csv(tuned_protocol_dir / "selection_rules.csv")
    tuned_acceptance_gates = pd.read_csv(tuned_protocol_dir / "acceptance_gates.csv")
    if (
        len(tuned_pilot_context) != 9
        or len(tuned_scope) != 2
        or len(tuned_seed_split) != 4
        or len(tuned_recipe_grid) != 6
        or len(tuned_selection_rules) != 6
        or len(tuned_acceptance_gates) != 7
    ):
        raise AssertionError(
            "CIFAR-100-LT tuned benchmark protocol must register pilot context, scope, seed splits, recipe grid, selection rules, and gates"
        )
    expected_recipe_families = {
        "adamw_ce_tuned",
        "sgd_momentum_ce_tuned",
        "adamw_cb_loss_tuned",
        "adamw_cb_sampler_tuned",
        "ns_muon_matrix_tuned",
        "ns_muon_cb_tuned",
    }
    if set(tuned_recipe_grid["recipe_family"]) != expected_recipe_families:
        raise AssertionError("CIFAR-100-LT tuned benchmark protocol missing expected tuned recipe families")
    if set(tuned_recipe_grid["optimizer"]) != {"adamw", "sgd", "ns_muon"}:
        raise AssertionError("CIFAR-100-LT tuned benchmark protocol must include AdamW, SGD, and NS-Muon grids")
    final_seed = tuned_seed_split[tuned_seed_split["split_id"].eq("final_claim")]
    validation_seed = tuned_seed_split[tuned_seed_split["split_id"].eq("validation_tuning")]
    if (
        len(final_seed) != 1
        or final_seed["seed_set"].iloc[0] != "20..29"
        or final_seed["tuning_allowed"].iloc[0] != "no"
        or len(validation_seed) != 1
        or validation_seed["seed_set"].iloc[0] != "10..14"
    ):
        raise AssertionError("CIFAR-100-LT tuned benchmark protocol must freeze validation and final seed splits")
    broad_scope = tuned_scope[tuned_scope["scope_id"].eq("broad_long_tail_optimizer_claim")]
    if len(broad_scope) != 1 or broad_scope["status"].iloc[0] != "not_covered_by_this_protocol":
        raise AssertionError("CIFAR-100-LT tuned benchmark protocol must keep broad optimizer claims out of scope")
    if "class-balanced sampler" not in " ".join(tuned_recipe_grid["sampler"].astype(str)):
        raise AssertionError("CIFAR-100-LT tuned benchmark protocol must require a class-balanced sampler baseline")
    if "Holm-adjusted" not in " ".join(tuned_acceptance_gates["pass_rule"].astype(str)):
        raise AssertionError("CIFAR-100-LT tuned benchmark protocol must require adjusted final comparisons")
    protocol_contract_text = " ".join(
        [
            " ".join(tuned_selection_rules.astype(str).to_numpy().ravel()),
            " ".join(tuned_acceptance_gates.astype(str).to_numpy().ravel()),
        ]
    )
    for phrase in [
        "SEL-6-state-distribution-logging",
        "TB-7-state-distribution-occupancy",
        "occupancy_trace.csv",
        "gradient-momentum cosine",
        "matched-head-gain NS-vs-Fro local drift ratio",
    ]:
        if phrase not in protocol_contract_text:
            raise AssertionError(f"CIFAR-100-LT tuned benchmark protocol missing occupancy contract: {phrase}")
    tuned_protocol_text = Path("discussion/e11_cifar100_resnet_lt_tuned_benchmark_protocol.md").read_text(
        encoding="utf-8"
    )
    assert_required_phrases(
        "CIFAR-100-LT tuned benchmark protocol",
        tuned_protocol_text,
        [
            "E11 CIFAR-100-LT Tuned Benchmark Protocol",
            "not a new final benchmark result",
            "Pilot Context Quarantine",
            "Seed And Split Contract",
            "Recipe Grid",
            "Executable Validation Registry",
            "scripts/e11_run_cifar100_resnet_lt_tuned_benchmark.py --settings-only",
            "planned_occupancy_trace_path",
            "occupancy_trace.csv",
            "scripts/slurm/e11_cifar100_resnet_lt_tuned_benchmark_validation.sbatch",
            "Acceptance Gates",
            "not_ready",
            "broad long-tail optimizer claim remains forbidden",
        ],
    )
    tuned_registry = pd.read_csv("results/e11_cifar100_resnet_lt_tuned_benchmark/settings_registry.csv")
    tuned_execution_status = pd.read_csv("results/e11_cifar100_resnet_lt_tuned_benchmark/execution_status.csv")
    if len(tuned_registry) != 164:
        raise AssertionError("CIFAR-100-LT tuned benchmark validation registry must contain 164 settings")
    if "planned_occupancy_trace_path" not in tuned_registry.columns:
        raise AssertionError("CIFAR-100-LT tuned benchmark registry must include planned occupancy trace paths")
    if not tuned_registry["planned_occupancy_trace_path"].astype(str).str.endswith("occupancy_trace.csv").all():
        raise AssertionError("CIFAR-100-LT tuned benchmark registry occupancy paths must end with occupancy_trace.csv")
    if sorted(tuned_registry["array_index"].astype(int).tolist()) != list(range(164)):
        raise AssertionError("CIFAR-100-LT tuned benchmark validation registry must have contiguous Slurm array indices")
    if set(tuned_registry["phase"]) != {"validation_tuning"} or set(tuned_registry["seed_set"]) != {"10..14"}:
        raise AssertionError("CIFAR-100-LT tuned benchmark registry must use validation_tuning seeds 10..14")
    if set(tuned_registry["recipe_family"]) != expected_recipe_families:
        raise AssertionError("CIFAR-100-LT tuned benchmark registry missing expected recipe families")
    family_counts = tuned_registry.groupby("recipe_family").size().to_dict()
    expected_family_counts = {
        "adamw_ce_tuned": 12,
        "sgd_momentum_ce_tuned": 12,
        "adamw_cb_loss_tuned": 24,
        "adamw_cb_sampler_tuned": 8,
        "ns_muon_matrix_tuned": 72,
        "ns_muon_cb_tuned": 36,
    }
    if family_counts != expected_family_counts:
        raise AssertionError(f"CIFAR-100-LT tuned benchmark family counts mismatch: {family_counts}")
    if not tuned_registry["class_balanced_sampler"].astype(bool).any():
        raise AssertionError("CIFAR-100-LT tuned benchmark registry must include class-balanced sampler settings")
    if set(tuned_registry["optimizer"]) != {"adamw", "sgd", "ns_muon"}:
        raise AssertionError("CIFAR-100-LT tuned benchmark registry must include AdamW, SGD, and NS-Muon settings")
    if not (
        len(tuned_execution_status) == 1
        and tuned_execution_status["status"].iloc[0] == "settings_registered"
        and "not_ready" in tuned_execution_status["claim_status"].iloc[0]
    ):
        raise AssertionError("CIFAR-100-LT tuned benchmark execution status must remain not_ready after settings registration")
    tuned_selection_dir = Path("results/e11_cifar100_resnet_lt_tuned_benchmark/validation_selection")
    tuned_selection_run_registry = pd.read_csv(tuned_selection_dir / "run_registry.csv")
    tuned_family_selection = pd.read_csv(tuned_selection_dir / "family_selection.csv")
    tuned_final_plan = pd.read_csv(tuned_selection_dir / "final_claim_plan.csv")
    tuned_selection_gates = pd.read_csv(tuned_selection_dir / "gate_report.csv")
    if (
        len(tuned_selection_run_registry) != 164
        or len(tuned_family_selection) != 6
        or len(tuned_final_plan) != 6
        or len(tuned_selection_gates) != 5
    ):
        raise AssertionError("CIFAR-100-LT tuned benchmark selection audit has the wrong row counts")
    validation_statuses = set(tuned_selection_run_registry["validation_status"].astype(str))
    occupancy_statuses = set(tuned_selection_run_registry["occupancy_status"].astype(str))
    if not validation_statuses.issubset({"missing_summary", "complete"}):
        raise AssertionError(f"CIFAR-100-LT tuned benchmark validation status drifted: {validation_statuses}")
    if not occupancy_statuses.issubset({"missing_occupancy_trace", "complete"}):
        raise AssertionError(f"CIFAR-100-LT tuned benchmark occupancy status drifted: {occupancy_statuses}")
    tuned_validation_complete = tuned_selection_run_registry["validation_status"].eq("complete")
    tuned_occupancy_complete = tuned_selection_run_registry["occupancy_status"].eq("complete")
    tuned_validation_complete_count = int(tuned_validation_complete.sum())
    tuned_occupancy_complete_count = int(tuned_occupancy_complete.sum())
    if tuned_validation_complete_count == 164:
        raise AssertionError("CIFAR-100-LT tuned benchmark validator needs an explicit final-selection audit update after 164/164 validation summaries complete")
    completed_tuned_rows = tuned_selection_run_registry[tuned_validation_complete & tuned_occupancy_complete]
    for row in completed_tuned_rows.itertuples(index=False):
        setting_id = str(row.setting_id)
        output_dir = Path(str(row.summary_path)).parent
        completed_required_paths = [
            output_dir / "train_trace.csv",
            output_dir / "class_metrics.csv",
            output_dir / "group_metrics.csv",
            output_dir / "summary.csv",
            output_dir / "pair_summary.csv",
            output_dir / "occupancy_trace.csv",
            output_dir / "setting_metadata.csv",
            output_dir / "config.json",
            Path("figures/e11_cifar100_resnet_lt_tuned_benchmark/validation_tuning")
            / setting_id
            / "cifar100_resnet_lt_recipe_benchmark.png",
            Path("discussion/e11_cifar100_resnet_lt_tuned_benchmark") / f"{setting_id}.md",
        ]
        missing_completed_paths = [path.as_posix() for path in completed_required_paths if not path.exists()]
        if missing_completed_paths:
            raise AssertionError(f"completed tuned validation setting {setting_id} missing artifacts: {missing_completed_paths}")
        if int(row.occupancy_probe_rows) < 30 or int(row.occupancy_seed_count) != 5 or int(row.occupancy_eval_step_count) != 6:
            raise AssertionError(f"completed tuned validation setting {setting_id} has incomplete occupancy summary")
        completed_text = (Path("discussion/e11_cifar100_resnet_lt_tuned_benchmark") / f"{setting_id}.md").read_text(
            encoding="utf-8"
        )
        assert_required_phrases(
            f"completed tuned validation setting {setting_id}",
            completed_text,
            [
                "E11 CIFAR-100-LT ResNet18 Tuned Benchmark Validation Setting",
                "preregistered 164-setting tuned validation grid",
                "Occupancy Trace",
                "final seeds `20..29`",
                "unblock a final-performance or broad optimizer claim",
            ],
        )
    if not tuned_selection_run_registry["occupancy_trace_path"].astype(str).str.endswith("occupancy_trace.csv").all():
        raise AssertionError("CIFAR-100-LT tuned benchmark selection must carry occupancy trace paths")
    complete_by_family = tuned_selection_run_registry.groupby("recipe_family")["validation_status"].apply(
        lambda values: int(values.eq("complete").sum())
    )
    expected_by_family = tuned_selection_run_registry.groupby("recipe_family").size()
    for row in tuned_family_selection.itertuples(index=False):
        family = str(row.recipe_family)
        expected = int(expected_by_family.loc[family])
        complete = int(complete_by_family.loc[family])
        if int(row.expected_settings) != expected or int(row.complete_settings) != complete:
            raise AssertionError(f"CIFAR-100-LT tuned benchmark family-selection counts drifted for {family}")
        expected_status = "selected" if complete == expected else "not_ready"
        if str(row.selection_status) != expected_status:
            raise AssertionError(f"CIFAR-100-LT tuned benchmark family-selection status drifted for {family}")
    final_by_family = tuned_final_plan.set_index("recipe_family")
    for row in tuned_family_selection.itertuples(index=False):
        family = str(row.recipe_family)
        final_row = final_by_family.loc[family]
        family_runs = tuned_selection_run_registry[tuned_selection_run_registry["recipe_family"].eq(family)]
        family_occupancy_complete = bool(family_runs["occupancy_status"].eq("complete").all())
        selected = str(row.selection_status) == "selected"
        expected_final_status = (
            "ready_for_final_run"
            if selected and family_occupancy_complete
            else "not_ready_missing_occupancy"
            if selected
            else "not_ready"
        )
        if str(final_row["final_status"]) != expected_final_status:
            raise AssertionError(f"CIFAR-100-LT tuned benchmark final-plan status drifted for {family}")
        final_seed = "" if pd.isna(final_row["final_seed_set"]) else str(final_row["final_seed_set"])
        if selected and final_seed != "20..29":
            raise AssertionError(f"CIFAR-100-LT tuned benchmark selected family {family} must map to final seeds 20..29")
        if not selected and final_seed not in {"", "nan"}:
            raise AssertionError(f"CIFAR-100-LT tuned benchmark unselected family {family} must keep final seeds blank")
    if set(tuned_selection_gates["gate_id"]) != {
        "TVS-1-validation-grid-complete",
        "TVS-2-family-selection",
        "TVS-3-final-seed-quarantine",
        "TVS-4-final-run-plan",
        "TVS-5-occupancy-logging-complete",
    }:
        raise AssertionError("CIFAR-100-LT tuned benchmark selection gates changed unexpectedly")
    tuned_gate_status = tuned_selection_gates.set_index("gate_id")["status"].astype(str).to_dict()
    expected_gate_status = {
        "TVS-1-validation-grid-complete": "pass" if tuned_validation_complete_count == 164 else "not_ready",
        "TVS-2-family-selection": "pass" if tuned_family_selection["selection_status"].eq("selected").all() else "not_ready",
        "TVS-3-final-seed-quarantine": "pass",
        "TVS-4-final-run-plan": "ready"
        if tuned_final_plan["final_status"].eq("ready_for_final_run").all()
        else "not_ready",
        "TVS-5-occupancy-logging-complete": "pass" if tuned_occupancy_complete_count == 164 else "not_ready",
    }
    if tuned_gate_status != expected_gate_status:
        raise AssertionError(f"CIFAR-100-LT tuned benchmark selection gates drifted: {tuned_gate_status}")
    tuned_selection_text = Path("discussion/e11_cifar100_resnet_lt_tuned_benchmark_selection.md").read_text(
        encoding="utf-8"
    )
    assert_required_phrases(
        "CIFAR-100-LT tuned benchmark selection",
        tuned_selection_text,
        [
            "E11 CIFAR-100-LT Tuned Benchmark Selection",
            "no-peeking bridge",
            f"{tuned_validation_complete_count}/164",
            f"{tuned_occupancy_complete_count}/164",
            "trajectory occupancy traces complete",
            "final claim seed set is always `20..29`",
            "TVS-3-final-seed-quarantine",
            "TVS-5-occupancy-logging-complete",
            "Occupancy Logging Status",
            "The current result is not a final-performance benchmark result",
        ],
    )
    tuned_interim_dir = Path("results/e11_cifar100_resnet_lt_tuned_benchmark/interim_validation_audit")
    tuned_interim_leaderboard = pd.read_csv(tuned_interim_dir / "completed_setting_leaderboard.csv")
    tuned_interim_family = pd.read_csv(tuned_interim_dir / "family_progress.csv")
    tuned_interim_occupancy = pd.read_csv(tuned_interim_dir / "occupancy_interim_summary.csv")
    tuned_interim_guardrail = pd.read_csv(tuned_interim_dir / "partial_grid_guardrail.csv")
    tuned_interim_gates = pd.read_csv(tuned_interim_dir / "claim_boundary_gates.csv")
    tuned_interim_config = json.loads((tuned_interim_dir / "config.json").read_text(encoding="utf-8"))
    if len(tuned_interim_leaderboard) != tuned_validation_complete_count:
        raise AssertionError("CIFAR-100-LT tuned interim leaderboard must contain every completed validation setting")
    if len(tuned_interim_family) != len(expected_recipe_families) or len(tuned_interim_occupancy) != len(
        expected_recipe_families
    ):
        raise AssertionError("CIFAR-100-LT tuned interim audit must summarize every recipe family")
    if set(tuned_interim_family["recipe_family"]) != expected_recipe_families:
        raise AssertionError("CIFAR-100-LT tuned interim family progress missing recipe families")
    if set(tuned_interim_occupancy["recipe_family"]) != expected_recipe_families:
        raise AssertionError("CIFAR-100-LT tuned interim occupancy summary missing recipe families")
    interim_complete_by_family = tuned_interim_family.set_index("recipe_family")["complete_settings"].astype(int)
    if interim_complete_by_family.to_dict() != complete_by_family.astype(int).to_dict():
        raise AssertionError("CIFAR-100-LT tuned interim family completion counts drifted from run_registry")
    if not tuned_interim_leaderboard.empty:
        if sorted(tuned_interim_leaderboard["rank"].astype(int).tolist()) != list(
            range(1, tuned_validation_complete_count + 1)
        ):
            raise AssertionError("CIFAR-100-LT tuned interim leaderboard ranks must be contiguous")
        if set(tuned_interim_leaderboard["selection_allowed"].astype(str)) != {"no_partial_grid_only"}:
            raise AssertionError("CIFAR-100-LT tuned interim leaderboard must forbid partial-grid selection")
        sorted_scores = tuned_interim_leaderboard["few_balanced_accuracy"].astype(float).tolist()
        if sorted_scores != sorted(sorted_scores, reverse=True):
            raise AssertionError("CIFAR-100-LT tuned interim leaderboard must sort by few balanced accuracy")
    expected_guard_ids = {
        "IPG-1-completed-prefix-auditable",
        "IPG-2-family-coverage-incomplete",
        "IPG-3-occupancy-paired-with-validation",
        "IPG-4-missing-work-blocks-final-readout",
        "IPG-5-final-seed-quarantine-mirrored",
    }
    if set(tuned_interim_guardrail["guard_id"]) != expected_guard_ids:
        raise AssertionError("CIFAR-100-LT tuned interim partial-grid guardrail IDs changed unexpectedly")
    complete_rows = tuned_selection_run_registry[
        tuned_selection_run_registry["validation_status"].eq("complete")
    ].copy()
    completed_indices = sorted(complete_rows["array_index"].astype(int).tolist())
    prefix_is_contiguous = completed_indices == list(range(len(completed_indices)))
    guardrail_occupancy = complete_rows[
        complete_rows["occupancy_status"].eq("complete")
        & complete_rows["occupancy_probe_rows"].fillna(0).astype(float).gt(0)
    ]
    expected_guard_status = {
        "IPG-1-completed-prefix-auditable": "pass" if prefix_is_contiguous else "not_ready",
        "IPG-2-family-coverage-incomplete": "ready"
        if tuned_interim_family["family_status"].eq("complete").all()
        else "not_ready",
        "IPG-3-occupancy-paired-with-validation": "pass"
        if len(guardrail_occupancy) == tuned_validation_complete_count
        else "not_ready",
        "IPG-4-missing-work-blocks-final-readout": "ready"
        if tuned_validation_complete_count == len(tuned_selection_run_registry)
        else "not_ready",
        "IPG-5-final-seed-quarantine-mirrored": expected_gate_status["TVS-3-final-seed-quarantine"],
    }
    observed_guard_status = tuned_interim_guardrail.set_index("guard_id")["status"].astype(str).to_dict()
    if observed_guard_status != expected_guard_status:
        raise AssertionError(f"CIFAR-100-LT tuned interim partial-grid guardrail drifted: {observed_guard_status}")
    if not tuned_interim_guardrail["blocked_action"].astype(str).str.contains(
        "selection|claim|final|trajectory|cherry-picking"
    ).all():
        raise AssertionError("CIFAR-100-LT tuned interim partial-grid guardrail must block unsafe actions")
    expected_interim_gates = {
        "IVA-1-validation-readout-scope",
        "IVA-2-partial-grid-blocks-selection",
        "IVA-3-familywise-selection-authority",
        "IVA-4-occupancy-coverage",
        "IVA-5-final-seed-quarantine",
    }
    if set(tuned_interim_gates["gate_id"]) != expected_interim_gates:
        raise AssertionError("CIFAR-100-LT tuned interim claim-boundary gates changed unexpectedly")
    if not tuned_interim_gates["blocked_wording"].astype(str).str.contains("final|selection|claim|trajectory").all():
        raise AssertionError("CIFAR-100-LT tuned interim gates must include blocked wording")
    if tuned_interim_config.get("scope") != "validation_tuning_only" or tuned_interim_config.get(
        "final_seed_status"
    ) != "not_touched":
        raise AssertionError("CIFAR-100-LT tuned interim config must preserve validation-only scope")
    if tuned_interim_config.get("partial_grid_guardrail") != (
        tuned_interim_dir / "partial_grid_guardrail.csv"
    ).as_posix():
        raise AssertionError("CIFAR-100-LT tuned interim config must point to the partial-grid guardrail")
    tuned_interim_text = Path("discussion/e11_cifar100_resnet_lt_tuned_benchmark_interim_audit.md").read_text(
        encoding="utf-8"
    )
    assert_required_phrases(
        "CIFAR-100-LT tuned benchmark interim validation audit",
        tuned_interim_text,
        [
            "E11 CIFAR-100-LT Tuned Benchmark Interim Validation Audit",
            "interim, no-peeking progress readout",
            "does not authorize final seed runs",
            "Claim Boundary Gates",
            "Partial Grid Guardrail",
            "Completed-Setting Leaderboard",
            "Interim Occupancy Summary",
            f"{tuned_validation_complete_count}/{len(tuned_selection_run_registry)}",
            "progress accounting only",
            "TVS-1",
            "TVS-2",
            "TVS-5",
            "final claim seed set `20..29` remains blocked",
        ],
    )
    tuned_leakage_dir = Path("results/e11_cifar100_resnet_lt_tuned_benchmark/validation_leakage_audit")
    tuned_leakage_guards = pd.read_csv(tuned_leakage_dir / "leakage_guard_matrix.csv")
    tuned_leakage_surface = pd.read_csv(tuned_leakage_dir / "observed_surface.csv")
    tuned_leakage_immutable = pd.read_csv(tuned_leakage_dir / "immutability_contract.csv")
    tuned_leakage_config = json.loads((tuned_leakage_dir / "config.json").read_text(encoding="utf-8"))
    expected_leakage_guards = {
        "TLA-1-selection-rule-frozen",
        "TLA-2-validation-only-input-surface",
        "TLA-3-array-order-contiguous",
        "TLA-4-partial-grid-selection-block",
        "TLA-5-final-output-quarantine",
        "TLA-6-launch-history-auditable",
    }
    if set(tuned_leakage_guards["guard_id"]) != expected_leakage_guards:
        raise AssertionError("CIFAR-100-LT tuned leakage audit guard IDs changed unexpectedly")
    leakage_statuses = set(tuned_leakage_guards["status"].astype(str))
    if not leakage_statuses.issubset({"pass", "ready", "not_ready"}):
        raise AssertionError(f"CIFAR-100-LT tuned leakage audit status drifted: {leakage_statuses}")
    required_pass_guards = {
        "TLA-1-selection-rule-frozen",
        "TLA-2-validation-only-input-surface",
        "TLA-3-array-order-contiguous",
        "TLA-5-final-output-quarantine",
    }
    pass_status = tuned_leakage_guards.set_index("guard_id")["status"].astype(str).to_dict()
    if any(pass_status.get(guard_id) != "pass" for guard_id in required_pass_guards):
        raise AssertionError("CIFAR-100-LT tuned leakage audit must pass frozen-rule, validation-only, array, and final quarantine guards")
    if len(tuned_leakage_surface) != 3 or set(tuned_leakage_surface["contains_final_seed_data"].astype(str)) != {"no"}:
        raise AssertionError("CIFAR-100-LT tuned leakage observed-surface audit must exclude final seed data")
    if len(tuned_leakage_immutable) != 4 or not tuned_leakage_immutable["status"].astype(str).eq("pass").all():
        raise AssertionError("CIFAR-100-LT tuned leakage immutability contract must pass all anchors")
    if tuned_leakage_config.get("claim_authority") != "no_new_claims" or tuned_leakage_config.get(
        "validation_surface"
    ) != "validation_tuning_only":
        raise AssertionError("CIFAR-100-LT tuned leakage config must preserve no-new-claims validation scope")
    tuned_leakage_text = Path("discussion/e11_cifar100_resnet_lt_tuned_benchmark_leakage_audit.md").read_text(
        encoding="utf-8"
    )
    assert_required_phrases(
        "CIFAR-100-LT tuned benchmark leakage audit",
        tuned_leakage_text,
        [
            "E11 CIFAR-100-LT Tuned Benchmark Leakage Audit",
            "selection leakage and optional-stopping risk",
            "partial validation observations",
            "Leakage Guard Matrix",
            "Observed Surface",
            "Immutability Contract",
            "does not authorize a new recipe grid",
            "fresh unspent validation/final splits",
        ],
    )
    tuned_refresh_dir = Path("results/e11_cifar100_resnet_lt_tuned_benchmark/validation_refresh_firewall")
    tuned_refresh_state = pd.read_csv(tuned_refresh_dir / "refresh_state.csv")
    tuned_refresh_allowed = pd.read_csv(tuned_refresh_dir / "allowed_transition_matrix.csv")
    tuned_refresh_forbidden = pd.read_csv(tuned_refresh_dir / "forbidden_action_matrix.csv")
    tuned_refresh_config = json.loads((tuned_refresh_dir / "config.json").read_text(encoding="utf-8"))
    expected_refresh_allowed_ids = {
        "VRF-A1-prefix-result-refresh",
        "VRF-A2-nonprefix-result-quarantine",
        "VRF-A3-full-validation-refresh",
        "VRF-A4-final-submit-handoff",
        "VRF-A5-leakage-guard-repair",
    }
    expected_refresh_forbidden_ids = {
        "VRF-F1-change-selection-objective",
        "VRF-F2-metric-dependent-launch-order",
        "VRF-F3-partial-family-selection",
        "VRF-F4-premature-final-submit",
        "VRF-F5-validation-leaderboard-performance-claim",
        "VRF-F6-in-place-protocol-repair-after-violation",
    }
    if (
        len(tuned_refresh_state) != 1
        or set(tuned_refresh_allowed["transition_id"]) != expected_refresh_allowed_ids
        or set(tuned_refresh_forbidden["forbidden_id"]) != expected_refresh_forbidden_ids
    ):
        raise AssertionError("CIFAR-100-LT tuned validation refresh firewall IDs changed unexpectedly")
    refresh_complete_rows = tuned_selection_run_registry[
        tuned_selection_run_registry["validation_status"].eq("complete")
    ].copy()
    refresh_completed_indices = sorted(refresh_complete_rows["array_index"].astype(int).tolist())
    refresh_prefix_is_contiguous = refresh_completed_indices == list(range(len(refresh_completed_indices)))
    refresh_missing_indices = sorted(
        set(tuned_selection_run_registry["array_index"].astype(int).tolist()).difference(refresh_completed_indices)
    )
    refresh_next_missing = str(refresh_missing_indices[0]) if refresh_missing_indices else "none"
    refresh_completed_families = int(
        tuned_selection_run_registry.groupby("recipe_family")["validation_status"]
        .apply(lambda values: bool(values.eq("complete").all()))
        .sum()
    )
    refresh_leakage_ok = bool(tuned_leakage_guards["status"].astype(str).isin({"pass", "ready"}).all())
    refresh_full_validation_ready = (
        tuned_gate_status["TVS-1-validation-grid-complete"] == "pass"
        and tuned_gate_status["TVS-2-family-selection"] == "pass"
        and tuned_gate_status["TVS-5-occupancy-logging-complete"] == "pass"
        and refresh_leakage_ok
    )
    expected_refresh_state_status = (
        "full_grid_ready_for_final_gate_refresh"
        if refresh_full_validation_ready
        else "partial_grid_no_claim_change"
        if refresh_prefix_is_contiguous and refresh_leakage_ok
        else "quarantine_refresh_until_audit_repaired"
    )
    refresh_state_row = tuned_refresh_state.iloc[0]
    if not (
        str(refresh_state_row["state_id"]) == "VRF-current-validation-prefix"
        and str(refresh_state_row["status"]) == expected_refresh_state_status
        and int(refresh_state_row["completed_validation_settings"]) == tuned_validation_complete_count
        and int(refresh_state_row["total_validation_settings"]) == len(tuned_selection_run_registry)
        and int(refresh_state_row["completed_occupancy_traces"]) == tuned_occupancy_complete_count
        and int(refresh_state_row["completed_recipe_families"]) == refresh_completed_families
        and str(refresh_state_row["next_missing_array_index"]) == refresh_next_missing
        and str(refresh_state_row["prefix_contiguous"]) == ("yes" if refresh_prefix_is_contiguous else "no")
    ):
        raise AssertionError("CIFAR-100-LT tuned validation refresh state drifted from selection registry")
    expected_refresh_allowed_status = {
        "VRF-A1-prefix-result-refresh": "pass"
        if refresh_prefix_is_contiguous and refresh_next_missing != "none"
        else "not_ready",
        "VRF-A2-nonprefix-result-quarantine": "not_ready",
        "VRF-A3-full-validation-refresh": "ready"
        if tuned_gate_status["TVS-1-validation-grid-complete"] == "pass"
        and tuned_gate_status["TVS-5-occupancy-logging-complete"] == "pass"
        and refresh_leakage_ok
        else "not_ready",
        "VRF-A4-final-submit-handoff": "ready" if refresh_full_validation_ready else "not_ready",
        "VRF-A5-leakage-guard-repair": "pass" if refresh_leakage_ok else "fail",
    }
    observed_refresh_allowed_status = tuned_refresh_allowed.set_index("transition_id")["status"].astype(str).to_dict()
    if observed_refresh_allowed_status != expected_refresh_allowed_status:
        raise AssertionError(f"CIFAR-100-LT tuned refresh allowed transitions drifted: {observed_refresh_allowed_status}")
    expected_forbidden_status = {
        "VRF-F1-change-selection-objective": "active",
        "VRF-F2-metric-dependent-launch-order": "active",
        "VRF-F3-partial-family-selection": "active"
        if tuned_gate_status["TVS-1-validation-grid-complete"] != "pass"
        else "retire_after_full_grid",
        "VRF-F4-premature-final-submit": "active",
        "VRF-F5-validation-leaderboard-performance-claim": "active",
        "VRF-F6-in-place-protocol-repair-after-violation": "active",
    }
    observed_forbidden_status = tuned_refresh_forbidden.set_index("forbidden_id")["status"].astype(str).to_dict()
    if observed_forbidden_status != expected_forbidden_status:
        raise AssertionError(f"CIFAR-100-LT tuned refresh forbidden-action statuses drifted: {observed_forbidden_status}")
    if tuned_refresh_config.get("scope") != "sequential_validation_refresh_firewall" or tuned_refresh_config.get(
        "claim_authority"
    ) != "progress_accounting_only_until_TVS_FEP_TFE_FLA_pass":
        raise AssertionError("CIFAR-100-LT tuned refresh firewall config must preserve claim-authority scope")
    tuned_refresh_text = Path("discussion/e11_cifar100_resnet_lt_tuned_benchmark_refresh_firewall.md").read_text(
        encoding="utf-8"
    )
    assert_required_phrases(
        "CIFAR-100-LT tuned benchmark validation refresh firewall",
        tuned_refresh_text,
        [
            "E11 CIFAR-100-LT Tuned Benchmark Validation Refresh Firewall",
            "sequential optional-stopping firewall",
            "Refresh State",
            "Allowed Transition Matrix",
            "Forbidden Action Matrix",
            "progress accounting only",
            "fresh preregistered protocol",
            "TVS-1",
            "TVS-2",
            "TVS-5",
        ],
    )
    tuned_seal_dir = Path("results/e11_cifar100_resnet_lt_tuned_benchmark/validation_protocol_seal")
    tuned_seal_hashes = pd.read_csv(tuned_seal_dir / "hash_manifest.csv")
    tuned_seal_boundary = pd.read_csv(tuned_seal_dir / "exposure_boundary.csv")
    tuned_seal_immutability = pd.read_csv(tuned_seal_dir / "immutability_matrix.csv")
    tuned_seal_gates = pd.read_csv(tuned_seal_dir / "seal_gate_matrix.csv")
    tuned_seal_config = json.loads((tuned_seal_dir / "config.json").read_text(encoding="utf-8"))
    expected_seal_gates = {
        "TPS-1-hash-manifest-complete",
        "TPS-2-validation-registry-sealed",
        "TPS-3-selection-rule-sealed",
        "TPS-4-final-seed-quarantine-sealed",
        "TPS-5-post-exposure-claim-authority",
    }
    expected_immutability_ids = {
        "TPS-IMM-1-protocol-csvs",
        "TPS-IMM-2-validation-registry",
        "TPS-IMM-3-final-seed-contract",
        "TPS-IMM-4-selection-and-final-gate-code",
        "TPS-IMM-5-progress-audit-code",
    }
    if (
        len(tuned_seal_hashes) != 18
        or not tuned_seal_hashes["exists"].astype(str).eq("yes").all()
        or not tuned_seal_hashes["sha256"].astype(str).str.fullmatch(r"[0-9a-f]{64}").all()
    ):
        raise AssertionError("CIFAR-100-LT tuned protocol seal must hash every sealed artifact")
    if set(tuned_seal_gates["gate_id"]) != expected_seal_gates or not tuned_seal_gates["status"].astype(str).eq(
        "pass"
    ).all():
        raise AssertionError("CIFAR-100-LT tuned protocol seal gates must all pass")
    if set(tuned_seal_immutability["immutability_id"]) != expected_immutability_ids:
        raise AssertionError("CIFAR-100-LT tuned protocol seal immutability IDs changed unexpectedly")
    if not tuned_seal_immutability["required_response_if_changed"].astype(str).str.contains(
        "fresh|downgrade|progress", regex=True
    ).all():
        raise AssertionError("CIFAR-100-LT tuned protocol seal must define strict change responses")
    if len(tuned_seal_boundary) != 1:
        raise AssertionError("CIFAR-100-LT tuned protocol seal must have one exposure boundary row")
    seal_boundary_row = tuned_seal_boundary.iloc[0]
    if not (
        str(seal_boundary_row["boundary_id"]) == "TPS-current-post-exposure-boundary"
        and str(seal_boundary_row["status"]) == "post_partial_validation_exposure_sealed"
        and int(seal_boundary_row["completed_validation_settings"]) == tuned_validation_complete_count
        and int(seal_boundary_row["total_validation_settings"]) == len(tuned_selection_run_registry)
        and int(seal_boundary_row["completed_occupancy_traces"]) == tuned_occupancy_complete_count
        and str(seal_boundary_row["next_missing_array_index"]) == refresh_next_missing
        and str(seal_boundary_row["prefix_contiguous"]) == ("yes" if refresh_prefix_is_contiguous else "no")
        and int(seal_boundary_row["final_output_files"]) == 0
        and str(seal_boundary_row["claim_authority"])
        == "progress_accounting_only_until_full_validation_and_final_gates_pass"
    ):
        raise AssertionError("CIFAR-100-LT tuned protocol seal exposure boundary drifted from validation state")
    if tuned_seal_config.get("scope") != "post_exposure_tuned_protocol_hash_seal" or tuned_seal_config.get(
        "claim_authority"
    ) != "progress_accounting_only_until_full_validation_and_final_gates_pass":
        raise AssertionError("CIFAR-100-LT tuned protocol seal config must preserve post-exposure scope")
    tuned_seal_text = Path("discussion/e11_cifar100_resnet_lt_tuned_benchmark_protocol_seal.md").read_text(
        encoding="utf-8"
    )
    assert_required_phrases(
        "CIFAR-100-LT tuned benchmark protocol seal",
        tuned_seal_text,
        [
            "E11 CIFAR-100-LT Tuned Benchmark Protocol Seal",
            "post-exposure protocol hash seal",
            "Exposure Boundary",
            "Seal Gate Matrix",
            "Immutability Matrix",
            "Hash Manifest",
            "progress accounting",
            "fresh preregistered protocol",
            "validation rows are visible",
        ],
    )
    tuned_fairness_dir = Path("results/e11_cifar100_resnet_lt_tuned_benchmark/validation_fairness_audit")
    tuned_fairness_budget = pd.read_csv(tuned_fairness_dir / "family_budget_matrix.csv")
    tuned_fairness_parity = pd.read_csv(tuned_fairness_dir / "seed_metric_parity.csv")
    tuned_fairness_gates = pd.read_csv(tuned_fairness_dir / "fairness_gate_matrix.csv")
    tuned_fairness_contract = pd.read_csv(tuned_fairness_dir / "budget_disclosure_contract.csv")
    tuned_fairness_config = json.loads((tuned_fairness_dir / "config.json").read_text(encoding="utf-8"))
    expected_fairness_families = {
        "adamw_ce_tuned",
        "sgd_momentum_ce_tuned",
        "adamw_cb_loss_tuned",
        "adamw_cb_sampler_tuned",
        "ns_muon_matrix_tuned",
        "ns_muon_cb_tuned",
    }
    expected_fairness_parity_ids = {
        "TBF-P1-validation-seed-parity",
        "TBF-P2-final-seed-parity",
        "TBF-P3-primary-metric-parity",
        "TBF-P4-multiplicity-parity",
        "TBF-P5-reporting-surface-parity",
    }
    expected_fairness_gate_ids = {
        "TBF-1-registered-family-coverage",
        "TBF-2-baseline-tuning-surface",
        "TBF-3-candidate-budget-disclosure",
        "TBF-4-seed-metric-reporting-parity",
        "TBF-5-final-gates-still-blocked",
    }
    expected_fairness_contract_ids = {
        "TBF-C1-validation-budget-disclosure",
        "TBF-C2-final-evidence-parity",
        "TBF-C3-baseline-strength-boundary",
    }
    if (
        set(tuned_fairness_budget["recipe_family"]) != expected_fairness_families
        or int(tuned_fairness_budget["setting_count"].sum()) != len(tuned_selection_run_registry)
    ):
        raise AssertionError("CIFAR-100-LT tuned fairness audit must summarize every registered recipe family")
    fairness_budget_by_family = tuned_fairness_budget.set_index("recipe_family")["setting_count"].astype(int)
    expected_fairness_counts = {
        "adamw_ce_tuned": 12,
        "sgd_momentum_ce_tuned": 12,
        "adamw_cb_loss_tuned": 24,
        "adamw_cb_sampler_tuned": 8,
        "ns_muon_matrix_tuned": 72,
        "ns_muon_cb_tuned": 36,
    }
    if fairness_budget_by_family.to_dict() != expected_fairness_counts:
        raise AssertionError(f"CIFAR-100-LT tuned fairness budget counts drifted: {fairness_budget_by_family.to_dict()}")
    if set(tuned_fairness_parity["parity_id"]) != expected_fairness_parity_ids or not tuned_fairness_parity[
        "status"
    ].astype(str).eq("pass").all():
        raise AssertionError("CIFAR-100-LT tuned fairness seed/metric parity must pass")
    observed_fairness_gate_status = tuned_fairness_gates.set_index("gate_id")["status"].astype(str).to_dict()
    expected_fairness_gate_status = {
        "TBF-1-registered-family-coverage": "pass",
        "TBF-2-baseline-tuning-surface": "pass",
        "TBF-3-candidate-budget-disclosure": "pass_with_disclosure",
        "TBF-4-seed-metric-reporting-parity": "pass",
        "TBF-5-final-gates-still-blocked": "pass",
    }
    if set(tuned_fairness_gates["gate_id"]) != expected_fairness_gate_ids or observed_fairness_gate_status != expected_fairness_gate_status:
        raise AssertionError(f"CIFAR-100-LT tuned fairness gates drifted: {observed_fairness_gate_status}")
    if set(tuned_fairness_contract["contract_id"]) != expected_fairness_contract_ids or not tuned_fairness_contract[
        "forbidden_wording"
    ].astype(str).str.contains("validation|final|Muon|baseline", regex=True).all():
        raise AssertionError("CIFAR-100-LT tuned fairness budget disclosure contract must block unfair wording")
    if tuned_fairness_config.get("scope") != "validation_budget_and_comparison_fairness" or tuned_fairness_config.get(
        "claim_authority"
    ) != "fairness_disclosure_only_until_final_gates_pass":
        raise AssertionError("CIFAR-100-LT tuned fairness audit config must preserve claim authority")
    tuned_fairness_text = Path("discussion/e11_cifar100_resnet_lt_tuned_benchmark_fairness_audit.md").read_text(
        encoding="utf-8"
    )
    assert_required_phrases(
        "CIFAR-100-LT tuned benchmark fairness audit",
        tuned_fairness_text,
        [
            "E11 CIFAR-100-LT Tuned Benchmark Fairness Audit",
            "validation-budget and comparison-parity audit",
            "Family Budget Matrix",
            "Seed And Metric Parity",
            "Fairness Gate Matrix",
            "Budget Disclosure Contract",
            "larger Muon validation search budget",
            "not final evidence",
            "tuned AdamW and tuned SGD",
        ],
    )
    tuned_slurm_dir = Path("results/e11_cifar100_resnet_lt_tuned_benchmark/slurm_submission_plan")
    tuned_slurm_plan = pd.read_csv(tuned_slurm_dir / "chunk_plan.csv")
    tuned_slurm_policy = pd.read_csv(tuned_slurm_dir / "queue_policy.csv")
    tuned_slurm_config = json.loads((tuned_slurm_dir / "config.json").read_text(encoding="utf-8"))
    tuned_missing_for_plan = int((~(tuned_validation_complete & tuned_occupancy_complete)).sum())
    tuned_expected_plan_rows = max(1, math.ceil(tuned_missing_for_plan / 20))
    if (
        len(tuned_slurm_plan) != tuned_expected_plan_rows
        or int(tuned_slurm_plan["setting_count"].sum()) != tuned_missing_for_plan
        or int(tuned_slurm_plan["setting_count"].max()) > 20
        or set(tuned_slurm_plan["submission_status"]) != {"not_submitted_static_plan"}
    ):
        raise AssertionError("CIFAR-100-LT tuned benchmark Slurm plan must chunk missing settings without submission")
    if not tuned_slurm_plan["submit_command"].astype(str).str.contains(
        "sbatch --array=.*%1 scripts/slurm/e11_cifar100_resnet_lt_tuned_benchmark_validation.sbatch",
        regex=True,
    ).all():
        raise AssertionError("CIFAR-100-LT tuned benchmark Slurm plan must use bounded sbatch array commands")
    if set(tuned_slurm_policy["policy_id"]) != {
        "SLURM-1-gpu-only-through-slurm",
        "SLURM-2-submit-limit",
        "SLURM-3-concurrency",
        "SLURM-4-no-final-unblinding",
    }:
        raise AssertionError("CIFAR-100-LT tuned benchmark Slurm plan must preserve queue policy rows")
    if not (
        tuned_slurm_config.get("side_effects") is False
        and tuned_slurm_config.get("submits_jobs") is False
        and int(tuned_slurm_config.get("chunk_size", -1)) == 20
        and int(tuned_slurm_config.get("array_concurrency", -1)) == 1
    ):
        raise AssertionError(f"CIFAR-100-LT tuned benchmark Slurm plan config drifted: {tuned_slurm_config}")
    tuned_slurm_text = Path("discussion/e11_cifar100_resnet_lt_tuned_benchmark_slurm_plan.md").read_text(
        encoding="utf-8"
    )
    assert_required_phrases(
        "CIFAR-100-LT tuned benchmark Slurm plan",
        tuned_slurm_text,
        [
            "E11 CIFAR-100-LT Tuned Benchmark Slurm Plan",
            "no-side-effect launch contract",
            "MaxSubmitJobsPerUser",
            "summary.csv",
            "occupancy_trace.csv",
            "sbatch --array=",
            "TVS-1",
            "TVS-2",
            "TVS-5",
        ],
    )
    tuned_launch_dir = Path("results/e11_cifar100_resnet_lt_tuned_benchmark/slurm_launch_audit")
    tuned_launch_decision = pd.read_csv(tuned_launch_dir / "latest_launch_decision.csv")
    tuned_launch_selected = pd.read_csv(tuned_launch_dir / "latest_selected_settings.csv")
    tuned_launch_queue = pd.read_csv(tuned_launch_dir / "latest_queue_snapshot.csv")
    tuned_launch_history = pd.read_csv(tuned_launch_dir / "launch_history.csv")
    tuned_launch_config = json.loads((tuned_launch_dir / "config.json").read_text(encoding="utf-8"))
    if len(tuned_launch_decision) != 1:
        raise AssertionError("CIFAR-100-LT tuned benchmark launch audit must contain exactly one latest decision")
    if len(tuned_launch_history) < 1 or str(tuned_launch_history.iloc[-1]["submission_status"]) != str(
        tuned_launch_decision.iloc[0]["submission_status"]
    ):
        raise AssertionError("CIFAR-100-LT tuned benchmark launch history must end with the latest decision status")
    launch_row = tuned_launch_decision.iloc[0]
    launch_status = str(launch_row["submission_status"])
    if launch_status not in {"dry_run", "submitted", "blocked_no_missing_settings", "blocked_no_queue_capacity", "sbatch_failed"}:
        raise AssertionError(f"CIFAR-100-LT tuned benchmark launch status drifted: {launch_status}")
    if launch_status == "sbatch_failed":
        raise AssertionError("CIFAR-100-LT tuned benchmark launch audit recorded a failed sbatch call")
    launch_planned_count = int(launch_row["planned_setting_count"])
    launch_available = int(launch_row["available_submit_slots_before_submit"])
    if launch_planned_count > min(20, launch_available, tuned_missing_for_plan):
        raise AssertionError("CIFAR-100-LT tuned benchmark launch audit exceeded the guarded queue capacity")
    if (
        int(launch_row["current_queue_elements_before_submit"])
        + launch_planned_count
        + int(launch_row["reserved_submit_slots"])
        > int(launch_row["max_submit_jobs_per_user"])
    ):
        raise AssertionError("CIFAR-100-LT tuned benchmark launch audit violates MaxSubmitJobsPerUser guard")
    if launch_planned_count != len(tuned_launch_selected):
        raise AssertionError("CIFAR-100-LT tuned benchmark launch selected-settings row count mismatch")
    if launch_planned_count:
        if set(tuned_launch_selected["phase"].astype(str)) != {"validation_tuning"}:
            raise AssertionError("CIFAR-100-LT tuned benchmark launch must select validation_tuning rows only")
        if set(tuned_launch_selected["seed_set"].astype(str)) != {"10..14"}:
            raise AssertionError("CIFAR-100-LT tuned benchmark launch must select validation seeds 10..14 only")
        command = str(launch_row["submit_command"])
        if "sbatch --parsable --array=" not in command or "%1 scripts/slurm/e11_cifar100_resnet_lt_tuned_benchmark_validation.sbatch" not in command:
            raise AssertionError(f"CIFAR-100-LT tuned benchmark launch command is not a bounded Slurm array: {command}")
    if launch_status == "submitted" and not re.fullmatch(r"\d+", str(launch_row["slurm_job_id"])):
        raise AssertionError("CIFAR-100-LT tuned benchmark submitted launch must record a numeric Slurm job id")
    if int(tuned_launch_config.get("max_submit_jobs_per_user", -1)) != 30:
        raise AssertionError("CIFAR-100-LT tuned benchmark launch config must preserve MaxSubmitJobsPerUser=30")
    if int(tuned_launch_config.get("reserved_submit_slots", -1)) != 6:
        raise AssertionError("CIFAR-100-LT tuned benchmark launch config must preserve six reserved submit slots")
    if "job_id" not in tuned_launch_queue.columns or "job_name" not in tuned_launch_queue.columns:
        raise AssertionError("CIFAR-100-LT tuned benchmark launch queue snapshot must include job_id and job_name")
    tuned_launch_text = Path("discussion/e11_cifar100_resnet_lt_tuned_benchmark_slurm_launch_audit.md").read_text(
        encoding="utf-8"
    )
    assert_required_phrases(
        "CIFAR-100-LT tuned benchmark Slurm launch audit",
        tuned_launch_text,
        [
            "E11 CIFAR-100-LT Tuned Benchmark Slurm Launch Audit",
            "queue-aware launch decision",
            "MaxSubmitJobsPerUser",
            "validation_tuning",
            "final_claim",
            "TVS-1",
            "TVS-2",
            "TVS-5",
        ],
    )
    tuned_power_dir = Path("results/e11_cifar100_resnet_lt_tuned_benchmark/final_power_audit")
    tuned_power_design = pd.read_csv(tuned_power_dir / "final_family_design.csv")
    tuned_power_comparisons = pd.read_csv(tuned_power_dir / "primary_comparison_plan.csv")
    tuned_power_mde = pd.read_csv(tuned_power_dir / "paired_diff_mde.csv")
    tuned_power_guardrail = pd.read_csv(tuned_power_dir / "all_class_guardrail_mde.csv")
    tuned_power_ladder = pd.read_csv(tuned_power_dir / "interpretation_ladder.csv")
    tuned_power_states = pd.read_csv(tuned_power_dir / "outcome_state_machine.csv")
    tuned_power_config = json.loads((tuned_power_dir / "config.json").read_text(encoding="utf-8"))
    if len(tuned_power_design) != 6 or set(tuned_power_design["recipe_family"]) != expected_recipe_families:
        raise AssertionError("CIFAR-100-LT tuned benchmark power audit must cover the frozen recipe families")
    power_selection_status = tuned_power_design.set_index("recipe_family")["validation_selection_status"].astype(str)
    selection_status = tuned_family_selection.set_index("recipe_family")["selection_status"].astype(str)
    if power_selection_status.to_dict() != selection_status.to_dict():
        raise AssertionError("CIFAR-100-LT tuned benchmark power audit must mirror validation-selection status")
    if set(tuned_power_design["final_output_status"]) != {"absent"}:
        raise AssertionError("CIFAR-100-LT tuned benchmark power audit must not inspect final outputs")
    if int(tuned_power_config.get("final_seed_count", -1)) != 10:
        raise AssertionError("CIFAR-100-LT tuned benchmark power audit must lock 10 final paired seeds")
    if int(tuned_power_config.get("primary_comparison_family_size", -1)) != 4:
        raise AssertionError("CIFAR-100-LT tuned benchmark power audit must lock the 4-comparison Holm family")
    if (
        len(tuned_power_comparisons) != 4
        or set(tuned_power_comparisons["candidate_family"]) != {"ns_muon_matrix_tuned", "ns_muon_cb_tuned"}
        or set(tuned_power_comparisons["baseline_family"]) != {"adamw_ce_tuned", "sgd_momentum_ce_tuned"}
    ):
        raise AssertionError("CIFAR-100-LT tuned benchmark power audit primary comparison family changed")
    holm_sd003 = tuned_power_mde[
        tuned_power_mde["alpha_scope"].eq("holm_bonferroni_worst_case")
        & tuned_power_mde["paired_diff_sd"].eq(0.03)
    ]
    if len(holm_sd003) != 1 or not (0.028 < float(holm_sd003["minimum_detectable_paired_mean_diff"].iloc[0]) < 0.033):
        raise AssertionError("CIFAR-100-LT tuned benchmark Holm MDE changed unexpectedly")
    holm_guardrail_sd003 = tuned_power_guardrail[
        tuned_power_guardrail["alpha_scope"].eq("holm_bonferroni_worst_case")
        & tuned_power_guardrail["paired_diff_sd"].eq(0.03)
    ]
    if (
        len(holm_guardrail_sd003) != 1
        or not (0.018 < float(holm_guardrail_sd003["minimum_mean_all_accuracy_diff_to_pass"].iloc[0]) < 0.023)
    ):
        raise AssertionError("CIFAR-100-LT tuned benchmark all-class guardrail MDE changed unexpectedly")
    if set(tuned_power_ladder["case_id"]) != {
        "TB-PWR-1-validation-incomplete",
        "TB-PWR-2-positive-primary",
        "TB-PWR-3-underpowered-null",
        "TB-PWR-4-negative-at-detectable-scale",
        "TB-PWR-5-tradeoff-only",
        "TB-PWR-6-reporting-incomplete",
    }:
        raise AssertionError("CIFAR-100-LT tuned benchmark power-audit interpretation ladder changed")
    if set(tuned_power_states["state_id"]) != {
        "TB-PWR-S1-not-ready",
        "TB-PWR-S2-final-quarantine-broken",
        "TB-PWR-S3-final-family-complete-positive",
        "TB-PWR-S4-final-family-complete-underpowered",
        "TB-PWR-S5-final-family-complete-negative",
    }:
        raise AssertionError("CIFAR-100-LT tuned benchmark power-audit outcome state machine changed")
    if "no final seed outputs inspected" not in str(tuned_power_config.get("analysis_scope", "")):
        raise AssertionError("CIFAR-100-LT tuned benchmark power audit must state that final outputs are uninspected")
    tuned_power_text = Path("discussion/e11_cifar100_resnet_lt_tuned_benchmark_power_audit.md").read_text(
        encoding="utf-8"
    )
    assert_required_phrases(
        "CIFAR-100-LT tuned benchmark power audit",
        tuned_power_text,
        [
            "E11 CIFAR-100-LT Tuned Benchmark Power Audit",
            "pre-output detectable-effect contract",
            "Holm-adjusted",
            "all-class guardrail",
            "underpowered",
            "negative tuned-performance boundary",
            "Outcome State Machine",
        ],
    )
    tuned_variance_dir = Path("results/e11_cifar100_resnet_lt_tuned_benchmark/variance_prior_audit")
    tuned_validation_variance = pd.read_csv(tuned_variance_dir / "validation_setting_variance.csv")
    tuned_pilot_variance = pd.read_csv(tuned_variance_dir / "spent_pilot_paired_variance.csv")
    tuned_variance_summary = pd.read_csv(tuned_variance_dir / "variance_prior_summary.csv")
    tuned_mde_sensitivity = pd.read_csv(tuned_variance_dir / "mde_sensitivity_from_empirical_sd.csv")
    tuned_variance_gates = pd.read_csv(tuned_variance_dir / "gate_matrix.csv")
    tuned_variance_config = json.loads((tuned_variance_dir / "config.json").read_text(encoding="utf-8"))
    expected_variance_gates = {
        "EVPA-1-final-seed-quarantine",
        "EVPA-2-validation-sd-surface",
        "EVPA-3-spent-pilot-paired-sd-surface",
        "EVPA-4-primary-few-sd-anchor",
        "EVPA-5-mde-sensitivity-grid",
        "EVPA-6-claim-boundary",
    }
    if set(tuned_variance_gates["gate_id"]) != expected_variance_gates:
        raise AssertionError("CIFAR-100-LT tuned variance-prior gate IDs changed unexpectedly")
    if not tuned_variance_gates["status"].astype(str).eq("pass").all():
        raise AssertionError("CIFAR-100-LT tuned variance-prior audit gates must pass")
    if len(tuned_validation_variance) < 68:
        raise AssertionError("CIFAR-100-LT tuned variance-prior audit must cover completed validation frequency groups")
    if not {"few", "all"}.issubset(set(tuned_pilot_variance["frequency_group"])):
        raise AssertionError("CIFAR-100-LT tuned variance-prior audit must include few/all spent-pilot paired rows")
    if set(tuned_variance_summary["source_id"]) != {"validation_within_setting", "spent_pilot_paired_diff"}:
        raise AssertionError("CIFAR-100-LT tuned variance-prior summary must separate validation and spent-pilot priors")
    if not {"p50", "p80", "p95", "max", "assumed_0p03"}.issubset(
        set(tuned_mde_sensitivity["sd_reference"].astype(str))
    ):
        raise AssertionError("CIFAR-100-LT tuned variance-prior MDE sensitivity missing empirical references")
    if bool(tuned_variance_config.get("final_seed_outputs_inspected", True)):
        raise AssertionError("CIFAR-100-LT tuned variance-prior audit must not inspect final seed outputs")
    if tuned_variance_config.get("assumed_paired_diff_sd") != 0.03:
        raise AssertionError("CIFAR-100-LT tuned variance-prior audit must preserve the 0.03 assumed SD reference")
    tuned_variance_text = Path(
        "discussion/e11_cifar100_resnet_lt_tuned_benchmark_variance_prior_audit.md"
    ).read_text(encoding="utf-8")
    assert_required_phrases(
        "CIFAR-100-LT tuned benchmark variance-prior audit",
        tuned_variance_text,
        [
            "E11 CIFAR-100-LT Tuned Benchmark Variance Prior Audit",
            "pre-final evidence",
            "completed validation summaries and spent pilot paired comparisons only",
            "Variance Prior Summary",
            "MDE Sensitivity Snapshot",
            "assumed_0p03",
            "Blocked now: using validation variability, spent pilot variability, or the assumed SD grid",
        ],
    )
    tuned_final_plan_dir = Path("results/e11_cifar100_resnet_lt_tuned_benchmark/final_analysis_plan")
    tuned_final_inputs = pd.read_csv(tuned_final_plan_dir / "analysis_input_contract.csv")
    tuned_final_metrics = pd.read_csv(tuned_final_plan_dir / "metric_contract.csv")
    tuned_final_comparisons = pd.read_csv(tuned_final_plan_dir / "primary_comparison_family.csv")
    tuned_final_adjustments = pd.read_csv(tuned_final_plan_dir / "multiplicity_and_guardrail_plan.csv")
    tuned_final_schema = pd.read_csv(tuned_final_plan_dir / "reporting_schema.csv")
    tuned_final_claims = pd.read_csv(tuned_final_plan_dir / "claim_ladder.csv")
    tuned_final_gates = pd.read_csv(tuned_final_plan_dir / "gate_matrix.csv")
    tuned_final_config = json.loads((tuned_final_plan_dir / "config.json").read_text(encoding="utf-8"))
    expected_final_plan_gates = {
        "FAP-1-final-quarantine",
        "FAP-2-family-selection-readiness",
        "FAP-3-power-audit-linked",
        "FAP-4-variance-prior-linked",
        "FAP-5-primary-family-fixed",
        "FAP-6-reporting-schema-fixed",
    }
    if set(tuned_final_gates["gate_id"]) != expected_final_plan_gates:
        raise AssertionError("CIFAR-100-LT tuned final-analysis plan gate IDs changed unexpectedly")
    if not set(tuned_final_gates["status"].astype(str)).issubset({"pass", "not_ready"}):
        raise AssertionError("CIFAR-100-LT tuned final-analysis plan gates must use pass/not_ready statuses")
    final_gate_lookup = tuned_final_gates.set_index("gate_id")["status"].astype(str).to_dict()
    if final_gate_lookup["FAP-1-final-quarantine"] != "pass":
        raise AssertionError("CIFAR-100-LT tuned final-analysis plan must keep final seeds quarantined")
    if len(tuned_final_inputs) != 6 or set(tuned_final_inputs["recipe_family"]) != expected_recipe_families:
        raise AssertionError("CIFAR-100-LT tuned final-analysis inputs must cover all recipe families")
    if set(tuned_final_metrics["metric_id"]) != {
        "MET-1-primary-few-balanced-accuracy",
        "MET-2-all-balanced-accuracy-guardrail",
        "MET-3-reporting-many-medium",
        "MET-4-state-distribution-occupancy",
    }:
        raise AssertionError("CIFAR-100-LT tuned final-analysis metric contract changed unexpectedly")
    if (
        len(tuned_final_comparisons) != 4
        or set(tuned_final_comparisons["candidate_family"]) != {"ns_muon_matrix_tuned", "ns_muon_cb_tuned"}
        or set(tuned_final_comparisons["baseline_family"]) != {"adamw_ce_tuned", "sgd_momentum_ce_tuned"}
        or "Holm" not in " ".join(tuned_final_comparisons["adjustment"].astype(str))
    ):
        raise AssertionError("CIFAR-100-LT tuned final-analysis comparison family must stay fixed")
    if "ADJ-1-primary-holm" not in set(tuned_final_adjustments["adjustment_id"]):
        raise AssertionError("CIFAR-100-LT tuned final-analysis plan must include Holm adjustment")
    if set(tuned_final_schema["table_id"]) != {
        "TAB-1-per-seed-final-metrics",
        "TAB-2-paired-primary-comparisons",
        "TAB-3-final-summary",
        "TAB-4-occupancy-summary",
    }:
        raise AssertionError("CIFAR-100-LT tuned final-analysis reporting schema changed unexpectedly")
    if "protocol_violation" not in set(tuned_final_claims["claim_state"]):
        raise AssertionError("CIFAR-100-LT tuned final-analysis claim ladder must include protocol violation state")
    if bool(tuned_final_config.get("final_outputs_inspected", True)):
        raise AssertionError("CIFAR-100-LT tuned final-analysis plan must not inspect final outputs")
    if int(tuned_final_config.get("primary_family_size", -1)) != 4:
        raise AssertionError("CIFAR-100-LT tuned final-analysis plan must lock four primary comparisons")
    tuned_final_text = Path(
        "discussion/e11_cifar100_resnet_lt_tuned_benchmark_final_analysis_plan.md"
    ).read_text(encoding="utf-8")
    assert_required_phrases(
        "CIFAR-100-LT tuned benchmark final analysis plan",
        tuned_final_text,
        [
            "E11 CIFAR-100-LT Tuned Benchmark Final Analysis Plan",
            "pre-final statistical analysis plan",
            "Primary Comparison Family",
            "Multiplicity And Guardrail Plan",
            "Reporting Schema",
            "Claim Ladder",
            "Holm step-down",
            "Blocked now: running final analysis, changing the primary comparison family",
        ],
    )
    tuned_robust_dir = Path("results/e11_cifar100_resnet_lt_tuned_benchmark/final_robustness_plan")
    tuned_robust_tests = pd.read_csv(tuned_robust_dir / "robustness_test_matrix.csv")
    tuned_robust_policy = pd.read_csv(tuned_robust_dir / "missing_seed_policy.csv")
    tuned_robust_ladder = pd.read_csv(tuned_robust_dir / "claim_sensitivity_ladder.csv")
    tuned_robust_gates = pd.read_csv(tuned_robust_dir / "gate_matrix.csv")
    tuned_robust_config = json.loads((tuned_robust_dir / "config.json").read_text(encoding="utf-8"))
    expected_robust_gate_ids = {
        "RBP-G1-final-output-quarantine",
        "RBP-G2-primary-family-linked",
        "RBP-G3-robustness-tests-registered",
        "RBP-G4-missing-seed-policy-active",
        "RBP-G5-final-execution-readiness",
    }
    expected_robust_policy_ids = {
        "RBP-1-no-dropped-seed-claim",
        "RBP-2-paired-unit-lock",
        "RBP-3-failed-run-reporting",
        "RBP-4-fixed-resampling-seed",
    }
    if (
        len(tuned_robust_tests) != len(tuned_final_comparisons) * 4
        or set(tuned_robust_tests["comparison_id"]) != set(tuned_final_comparisons["comparison_id"])
        or set(tuned_robust_tests["test_family"])
        != {"primary_parametric", "nonparametric_sensitivity", "interval_sensitivity", "direction_sensitivity"}
    ):
        raise AssertionError("CIFAR-100-LT tuned robustness plan must register four sensitivity rows per primary comparison")
    observed_robust_gates = tuned_robust_gates.set_index("gate_id")["status"].astype(str).to_dict()
    expected_robust_gates = {
        "RBP-G1-final-output-quarantine": "pass",
        "RBP-G2-primary-family-linked": "pass",
        "RBP-G3-robustness-tests-registered": "pass",
        "RBP-G4-missing-seed-policy-active": "pass",
        "RBP-G5-final-execution-readiness": "not_ready",
    }
    if set(tuned_robust_gates["gate_id"]) != expected_robust_gate_ids or observed_robust_gates != expected_robust_gates:
        raise AssertionError(f"CIFAR-100-LT tuned robustness gates drifted: {observed_robust_gates}")
    if set(tuned_robust_policy["policy_id"]) != expected_robust_policy_ids or not tuned_robust_policy[
        "status"
    ].astype(str).eq("active").all():
        raise AssertionError("CIFAR-100-LT tuned robustness missing-seed policy must be active")
    if "parametric_only_positive" not in set(tuned_robust_ladder["sensitivity_state"]):
        raise AssertionError("CIFAR-100-LT tuned robustness ladder must include parametric-only downgrade state")
    if tuned_robust_config.get("scope") != "pre_final_statistical_robustness_plan" or bool(
        tuned_robust_config.get("final_outputs_inspected", True)
    ):
        raise AssertionError("CIFAR-100-LT tuned robustness plan must be pre-final and not inspect final outputs")
    tuned_robust_text = Path(
        "discussion/e11_cifar100_resnet_lt_tuned_benchmark_final_robustness_plan.md"
    ).read_text(encoding="utf-8")
    assert_required_phrases(
        "CIFAR-100-LT tuned benchmark final robustness plan",
        tuned_robust_text,
        [
            "E11 CIFAR-100-LT Tuned Benchmark Final Robustness Plan",
            "pre-registers robustness checks",
            "exact sign-flip",
            "bootstrap",
            "sign-count",
            "Missing Seed Policy",
            "Claim Sensitivity Ladder",
            "parametric gate",
            "downgrade to a sensitivity caveat",
        ],
    )
    tuned_final_exec_dir = Path("results/e11_cifar100_resnet_lt_tuned_benchmark/final_execution_plan")
    tuned_final_exec_gates = pd.read_csv(tuned_final_exec_dir / "gate_matrix.csv")
    tuned_final_exec_runs = pd.read_csv(tuned_final_exec_dir / "final_family_run_plan.csv")
    tuned_final_exec_sbatch = pd.read_csv(tuned_final_exec_dir / "slurm_submit_plan.csv")
    tuned_final_exec_config = json.loads((tuned_final_exec_dir / "config.json").read_text(encoding="utf-8"))
    expected_final_exec_gates = {
        "FEP-1-selection-gates-ready",
        "FEP-2-final-analysis-plan-linked",
        "FEP-3-runner-contract-present",
        "FEP-4-final-output-quarantine",
        "FEP-5-all-families-ready",
    }
    if set(tuned_final_exec_gates["gate_id"]) != expected_final_exec_gates:
        raise AssertionError("CIFAR-100-LT tuned final-execution gate IDs changed unexpectedly")
    if not set(tuned_final_exec_gates["status"].astype(str)).issubset({"pass", "not_ready", "fail"}):
        raise AssertionError("CIFAR-100-LT tuned final-execution gates must use pass/not_ready/fail statuses")
    final_exec_lookup = tuned_final_exec_gates.set_index("gate_id")["status"].astype(str).to_dict()
    if final_exec_lookup["FEP-3-runner-contract-present"] != "pass":
        raise AssertionError("CIFAR-100-LT tuned final-execution runner contract must be present")
    if final_exec_lookup["FEP-4-final-output-quarantine"] != "pass":
        raise AssertionError("CIFAR-100-LT tuned final-execution plan must keep final outputs absent")
    if len(tuned_final_exec_runs) != 6 or set(tuned_final_exec_runs["recipe_family"]) != expected_recipe_families:
        raise AssertionError("CIFAR-100-LT tuned final-execution plan must cover all recipe families")
    allowed_final_seed_sets = {"", "20..29"}
    observed_seed_sets = set(tuned_final_exec_runs["final_seed_set"].fillna("").astype(str))
    if not observed_seed_sets.issubset(allowed_final_seed_sets):
        raise AssertionError("CIFAR-100-LT tuned final-execution plan must use only final seed set 20..29")
    if not tuned_final_exec_runs["tuning_allowed"].astype(str).eq("no").all():
        raise AssertionError("CIFAR-100-LT tuned final-execution plan must forbid tuning")
    if str(tuned_final_exec_sbatch["submits_jobs"].iloc[0]) != "no":
        raise AssertionError("CIFAR-100-LT tuned final-execution plan must be no-side-effect")
    if bool(tuned_final_exec_config.get("side_effects", True)) or bool(
        tuned_final_exec_config.get("submits_jobs", True)
    ) or bool(tuned_final_exec_config.get("inspects_final_outputs", True)):
        raise AssertionError("CIFAR-100-LT tuned final-execution config must not submit or inspect final outputs")
    tuned_runner_text = Path("scripts/e11_run_cifar100_resnet_lt_tuned_benchmark.py").read_text(encoding="utf-8")
    tuned_final_sbatch_text = Path(
        "scripts/slurm/e11_cifar100_resnet_lt_tuned_benchmark_final.sbatch"
    ).read_text(encoding="utf-8")
    assert_required_phrases(
        "CIFAR-100-LT tuned final runner contract",
        tuned_runner_text,
        [
            "--phase",
            "final_claim",
            "--final-index",
            "final_settings_from_selection",
            "validation selection gates pass",
        ],
    )
    assert_required_phrases(
        "CIFAR-100-LT tuned final Slurm wrapper",
        tuned_final_sbatch_text,
        [
            "#SBATCH --job-name=e11-lt-tuned-final",
            "#SBATCH --array=0-5%1",
            "--phase final_claim",
            "--final-index",
            "--device cuda",
            "--no-download",
        ],
    )
    tuned_final_exec_text = Path(
        "discussion/e11_cifar100_resnet_lt_tuned_benchmark_final_execution_plan.md"
    ).read_text(encoding="utf-8")
    assert_required_phrases(
        "CIFAR-100-LT tuned benchmark final execution plan",
        tuned_final_exec_text,
        [
            "E11 CIFAR-100-LT Tuned Benchmark Final Execution Plan",
            "no-side-effect execution contract",
            "Final Family Run Plan",
            "Slurm Submit Plan",
            "premature final submission fails closed",
            "Holm-adjusted final-analysis plan",
        ],
    )
    tuned_final_eval_dir = Path("results/e11_cifar100_resnet_lt_tuned_benchmark/final_evaluation")
    tuned_final_eval_registry = pd.read_csv(tuned_final_eval_dir / "run_registry.csv")
    tuned_final_eval_per_seed = pd.read_csv(tuned_final_eval_dir / "per_seed_final_metrics.csv")
    tuned_final_eval_paired = pd.read_csv(tuned_final_eval_dir / "paired_primary_comparisons.csv")
    tuned_final_eval_decisions = pd.read_csv(tuned_final_eval_dir / "primary_decisions.csv")
    tuned_final_eval_summary = pd.read_csv(tuned_final_eval_dir / "final_summary.csv")
    tuned_final_eval_occupancy = pd.read_csv(tuned_final_eval_dir / "occupancy_summary.csv")
    tuned_final_eval_gates = pd.read_csv(tuned_final_eval_dir / "claim_gate_report.csv")
    tuned_final_eval_config = json.loads((tuned_final_eval_dir / "config.json").read_text(encoding="utf-8"))
    expected_final_eval_gates = {
        "TFE-1-evaluator-implemented",
        "TFE-2-selection-gates-ready",
        "TFE-3-final-output-completeness",
        "TFE-4-primary-holm-family",
        "TFE-5-all-class-guardrail",
        "TFE-6-final-claim-state",
    }
    if set(tuned_final_eval_gates["gate_id"]) != expected_final_eval_gates:
        raise AssertionError("CIFAR-100-LT tuned final evaluator gate IDs changed unexpectedly")
    final_eval_lookup = tuned_final_eval_gates.set_index("gate_id")["status"].astype(str).to_dict()
    if final_eval_lookup["TFE-1-evaluator-implemented"] != "pass":
        raise AssertionError("CIFAR-100-LT tuned final evaluator implementation gate must pass")
    if final_eval_lookup["TFE-6-final-claim-state"] != "not_ready":
        raise AssertionError("CIFAR-100-LT tuned final evaluator must remain not_ready before final outputs exist")
    if len(tuned_final_eval_registry) != 6 or set(tuned_final_eval_registry["recipe_family"]) != expected_recipe_families:
        raise AssertionError("CIFAR-100-LT tuned final evaluator registry must cover all six recipe families")
    if int(tuned_final_eval_registry["selection_status"].eq("selected").sum()) != 1:
        raise AssertionError("CIFAR-100-LT tuned final evaluator should reflect the current partial 1/6 selection state")
    if not tuned_final_eval_registry["final_output_status"].astype(str).eq("missing_or_incomplete").all():
        raise AssertionError("CIFAR-100-LT tuned final evaluator must not see complete final outputs yet")
    if list(tuned_final_eval_per_seed.columns) != [
        "seed",
        "recipe_family",
        "recipe_name",
        "many_bacc",
        "medium_bacc",
        "few_bacc",
        "all_bacc",
        "loss",
        "margin",
    ]:
        raise AssertionError("CIFAR-100-LT tuned final evaluator per-seed schema changed unexpectedly")
    if list(tuned_final_eval_paired.columns) != [
        "comparison_id",
        "seed",
        "candidate_family",
        "baseline_family",
        "few_bacc_diff",
        "all_bacc_diff",
    ]:
        raise AssertionError("CIFAR-100-LT tuned final evaluator paired schema changed unexpectedly")
    if (
        len(tuned_final_eval_decisions) != 4
        or set(tuned_final_eval_decisions["comparison_id"]) != set(tuned_final_comparisons["comparison_id"])
        or not tuned_final_eval_decisions["primary_decision"].astype(str).eq("not_ready").all()
        or not tuned_final_eval_decisions["all_class_guardrail"].astype(str).eq("not_ready").all()
    ):
        raise AssertionError("CIFAR-100-LT tuned final evaluator decisions must keep the fixed not_ready Holm family")
    if not tuned_final_eval_summary.empty or not tuned_final_eval_occupancy.empty:
        raise AssertionError("CIFAR-100-LT tuned final evaluator summary/occupancy tables must be empty before final outputs")
    if (
        tuned_final_eval_config.get("primary_test")
        != "paired one-sided t-test on candidate-minus-baseline few balanced accuracy with Holm adjustment"
        or int(tuned_final_eval_config.get("primary_family_size", -1)) != 4
        or int(tuned_final_eval_config.get("expected_final_seed_count", -1)) != 10
    ):
        raise AssertionError("CIFAR-100-LT tuned final evaluator config changed unexpectedly")
    tuned_final_eval_text = Path(
        "discussion/e11_cifar100_resnet_lt_tuned_benchmark_final_evaluation.md"
    ).read_text(encoding="utf-8")
    assert_required_phrases(
        "CIFAR-100-LT tuned benchmark final evaluator",
        tuned_final_eval_text,
        [
            "E11 CIFAR-100-LT Tuned Benchmark Final Evaluation",
            "fixed post-final analysis path",
            "Holm family",
            "all-class noninferiority guardrail",
            "Current claim state: `not_ready`",
            "Blocked now: benchmark-performance wording remains unavailable",
        ],
    )
    tuned_final_launch_dir = Path("results/e11_cifar100_resnet_lt_tuned_benchmark/final_launch_audit")
    tuned_final_launch_decision = pd.read_csv(tuned_final_launch_dir / "latest_final_launch_decision.csv")
    tuned_final_launch_family = pd.read_csv(tuned_final_launch_dir / "latest_final_family_plan.csv")
    tuned_final_launch_gates = pd.read_csv(tuned_final_launch_dir / "latest_gate_snapshot.csv")
    tuned_final_launch_history = pd.read_csv(tuned_final_launch_dir / "final_launch_history.csv")
    tuned_final_launch_config = json.loads((tuned_final_launch_dir / "config.json").read_text(encoding="utf-8"))
    expected_final_launch_gates = {
        "FLA-1-final-execution-gates-ready",
        "FLA-2-final-evaluator-implemented",
        "FLA-3-no-final-job-duplicate",
        "FLA-4-queue-capacity",
        "FLA-5-submit-flag",
    }
    if set(tuned_final_launch_gates["gate_id"]) != expected_final_launch_gates:
        raise AssertionError("CIFAR-100-LT tuned final-launch audit gate IDs changed unexpectedly")
    final_launch_lookup = tuned_final_launch_gates.set_index("gate_id")["status"].astype(str).to_dict()
    if final_launch_lookup["FLA-1-final-execution-gates-ready"] != "not_ready":
        raise AssertionError("CIFAR-100-LT tuned final launch must stay blocked while FEP gates are not_ready")
    if final_launch_lookup["FLA-2-final-evaluator-implemented"] != "pass":
        raise AssertionError("CIFAR-100-LT tuned final launch must require the final evaluator contract")
    if final_launch_lookup["FLA-5-submit-flag"] != "dry_run":
        raise AssertionError("CIFAR-100-LT tuned final launch audit must be dry-run in committed artifacts")
    if (
        len(tuned_final_launch_decision) != 1
        or str(tuned_final_launch_decision["submission_status"].iloc[0]) != "blocked_final_launch_gates_not_ready"
        or str(tuned_final_launch_decision["submit_command"].fillna("").iloc[0]) != ""
        or int(tuned_final_launch_decision["planned_final_array_elements"].iloc[0]) != 0
        or str(tuned_final_launch_decision["phase_guard"].iloc[0]) != "final_claim_only"
    ):
        raise AssertionError("CIFAR-100-LT tuned final launch decision must fail closed before FEP gates pass")
    if len(tuned_final_launch_family) != 6 or set(tuned_final_launch_family["recipe_family"]) != expected_recipe_families:
        raise AssertionError("CIFAR-100-LT tuned final launch family plan must cover all six recipe families")
    if tuned_final_launch_history.empty:
        raise AssertionError("CIFAR-100-LT tuned final launch audit must append launch history")
    if (
        tuned_final_launch_config.get("sbatch_wrapper")
        != "scripts/slurm/e11_cifar100_resnet_lt_tuned_benchmark_final.sbatch"
        or int(tuned_final_launch_config.get("final_array_elements", -1)) != 6
        or bool(tuned_final_launch_config.get("submit_flag", True))
    ):
        raise AssertionError("CIFAR-100-LT tuned final launch config changed unexpectedly")
    tuned_final_launch_text = Path(
        "discussion/e11_cifar100_resnet_lt_tuned_benchmark_final_launch_audit.md"
    ).read_text(encoding="utf-8")
    assert_required_phrases(
        "CIFAR-100-LT tuned benchmark final launch audit",
        tuned_final_launch_text,
        [
            "E11 CIFAR-100-LT Tuned Benchmark Final Launch Audit",
            "queue-aware launch decision",
            "Launch Gate Snapshot",
            "blocked_final_launch_gates_not_ready",
            "Only `final_claim` seed set `20..29` is eligible",
            "authorize no benchmark-performance wording and no final submission",
        ],
    )
    tuned_final_claim_dir = Path("results/e11_cifar100_resnet_lt_tuned_benchmark/final_claim")
    final_outputs = sorted(tuned_final_claim_dir.rglob("*")) if tuned_final_claim_dir.exists() else []
    if final_outputs:
        raise AssertionError(f"CIFAR-100-LT tuned benchmark final outputs must stay absent before validation: {final_outputs}")
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
    muon_contract_dir = Path("results/e11_muon_state_distribution_contract")
    muon_contract_terms = pd.read_csv(muon_contract_dir / "state_distribution_terms.csv")
    muon_contract_evidence = pd.read_csv(muon_contract_dir / "evidence_link_matrix.csv")
    muon_contract_falsifiers = pd.read_csv(muon_contract_dir / "falsification_tests.csv")
    muon_contract_gates = pd.read_csv(muon_contract_dir / "claim_gate_ladder.csv")
    muon_contract_config = json.loads((muon_contract_dir / "config.json").read_text(encoding="utf-8"))
    if set(muon_contract_terms["term_id"]) != {
        "MSD-T1-local-response-integrand",
        "MSD-T2-state-occupancy-measure",
        "MSD-T3-transition-and-schedule-operator",
        "MSD-T4-terminal-risk-functional",
        "MSD-T5-claim-composition-rule",
    }:
        raise AssertionError("Muon state-distribution contract must preserve the fixed theory terms")
    if set(muon_contract_evidence["evidence_id"]) != {
        "MSE-1-local-adamw-state",
        "MSE-2-local-ns-muon-state",
        "MSE-3-final-pilot-negative",
        "MSE-4-tuned-grid-registered",
        "MSE-5-bold-conjecture-boundary",
    }:
        raise AssertionError("Muon state-distribution contract must preserve the fixed evidence links")
    if set(muon_contract_falsifiers["test_id"]) != {
        "MSF-1-occupancy-logging",
        "MSF-2-schedule-transport",
        "MSF-3-terminal-risk-separation",
        "MSF-4-baseline-dominance",
        "MSF-5-state-distribution-counterexample",
    }:
        raise AssertionError("Muon state-distribution contract must preserve the fixed falsification tests")
    if set(muon_contract_gates["gate_id"]) != {
        "MSG-1-local-compatibility",
        "MSG-2-state-distribution-transport",
        "MSG-3-tuned-final-performance",
        "MSG-4-top-tier-practical-claim",
    }:
        raise AssertionError("Muon state-distribution contract must preserve the fixed claim gates")
    if muon_contract_config != {
        "term_rows": 5,
        "evidence_rows": 5,
        "falsification_rows": 5,
        "claim_gate_rows": 4,
        "current_claim_boundary": "local Muon-style drift compatibility only",
        "performance_upgrade_boundary": "state-distribution plus tuned validation/final evidence required",
    }:
        raise AssertionError(f"Muon state-distribution config drifted: {muon_contract_config}")
    muon_contract_text = " ".join(
        [
            Path("discussion/e11_muon_state_distribution_contract.md").read_text(encoding="utf-8"),
            " ".join(muon_contract_terms.astype(str).to_numpy().ravel()),
            " ".join(muon_contract_evidence.astype(str).to_numpy().ravel()),
            " ".join(muon_contract_falsifiers.astype(str).to_numpy().ravel()),
            " ".join(muon_contract_gates.astype(str).to_numpy().ravel()),
        ]
    )
    assert_required_phrases(
        "Muon state-distribution contract",
        muon_contract_text,
        [
            "state-distribution transport contract",
            "local_integrand_supported_on_sampled_states",
            "occupancy_measure_missing_for_final_training",
            "schedule_transport_registered_not_evaluated",
            "final_performance_negative_boundary",
            "mechanism_only_until_all_components_pass",
            "0.8628 [0.8154, 0.913]",
            "0.7247 [0.676, 0.777]",
            "best_tested_few_diff=-0.08767 [-0.09948, -0.07585]",
            "TVS-1=not_ready",
            "TVS-5=not_ready",
            "occupancy_trace.csv",
            "Local compatibility can coexist with poor final performance",
            "blocked_until_state_distribution_and_final_gates_pass",
        ],
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
        "figures/e11_cifar100_resnet_imbalance_sweep/cifar100_resnet_imbalance_sweep.png",
        "figures/e11_cifar100_resnet_lt_standard_eval/cifar100_resnet_lt_standard_eval.png",
        "figures/e11_cifar100_resnet_lt_recipe_benchmark/cifar100_resnet_lt_recipe_benchmark.png",
        "figures/e11_cifar100_resnet_lt_muon_final_benchmark/cifar100_resnet_lt_recipe_benchmark.png",
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
        "Referee-Falsification Boundaries",
        "E11 Mechanism Referee Audit",
        "unit-JVP ratios are above one while matched-gain observed ratios are below one",
        "v5 final split outputs=not_run",
        "Use only finite registered phase1/phase2 null-candidate wording with detectable-effect, head-gain, and quality caveats",
        "local drift improvements imply final long-tail optimizer superiority",
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
        "\\EelevenCifarResNetRhoZeroZeroTwoDriftRatio",
        "\\EelevenCifarResNetRhoZeroZeroTwoTailLossDiff",
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
        "\\EelevenCifarResNetImbalanceSweepSettings",
        "\\EelevenCifarResNetImbalanceSweepWorstTailTrainPerClass",
        "\\EelevenCifarResNetImbalanceSweepWorstDriftRatio",
        "\\EelevenCifarResNetImbalanceSweepBestTailAccuracy",
        "\\EelevenCifarResNetImbalanceSweepBestTailDriftRatio",
        "\\EelevenCifarResNetLayerJvpSeeds",
        "\\EelevenCifarResNetLayerJvpParameters",
        "\\EelevenCifarResNetLayerJvpPairedPoints",
        "\\EelevenCifarResNetLayerJvpScaledRatio",
        "\\EelevenCifarResNetLayerJvpObservedRatio",
        "\\EelevenCifarResNetLayerJvpSupportedLayers",
        "\\EelevenCifarResNetLayerJvpWorstObservedRatio",
        "\\EelevenCifarResNetLayerJvpWorstScaledRatio",
        "\\EelevenCifarResNetLtMuonFinalBenchmarkSeeds",
        "\\EelevenCifarResNetLtMuonFinalAdamwAllBalancedAccuracy",
        "\\EelevenCifarResNetLtMuonFinalAdamwFewBalancedAccuracy",
        "\\EelevenCifarResNetLtMuonFinalLrOneEMinusFourAllBalancedAccuracy",
        "\\EelevenCifarResNetLtMuonFinalLrOneEMinusFourFewBalancedAccuracy",
        "\\EelevenCifarResNetLtMuonFinalLrThreeEMinusFiveAllBalancedAccuracy",
        "\\EelevenCifarResNetLtMuonFinalLrThreeEMinusFiveFewBalancedAccuracy",
        "\\EelevenCifarResNetLtMuonFinalLrOneEMinusFourAllBalancedAccuracyDiff",
        "\\EelevenCifarResNetLtMuonFinalLrOneEMinusFourFewBalancedAccuracyDiff",
        "\\EelevenCifarResNetLtMuonFinalLrThreeEMinusFiveAllBalancedAccuracyDiff",
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
        r"\EelevenCifarResNetRhoZeroZeroTwoDriftRatio",
        r"\EelevenCifarResNetRhoZeroZeroTwoTailLossDiff",
        r"\EelevenCifarResNetCheckpointSweepWorstDriftRatio",
        r"\EelevenCifarResNetCheckpointSweepTailAccuracyRange",
        r"\EelevenCifarResNetConditionProxyRankPearson",
        r"\EelevenCifarResNetFcConditionWorstDriftRatio",
        r"\EelevenCifarResNetFcConditionWeakestMeanConditionScore",
        r"\EelevenCifarResNetFcConditionFavorsSpectralFraction",
        r"\EelevenCifarResNetTailQualityWorstDriftRatio",
        r"\EelevenCifarResNetTailQualityBestTailAccuracy",
        r"\EelevenCifarResNetImbalanceSweepWorstDriftRatio",
        r"\EelevenCifarResNetImbalanceSweepBestTailAccuracy",
        r"\EelevenCifarResNetLayerJvpObservedRatio",
        r"\EelevenCifarResNetLayerJvpScaledRatio",
        r"\EelevenCifarResNetLtMuonFinalAdamwAllBalancedAccuracy",
        r"\EelevenCifarResNetLtMuonFinalLrOneEMinusFourAllBalancedAccuracyDiff",
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
        "make e11-cifar-resnet-imbalance-sweep-results",
        "make e11-cifar-resnet-layer-jvp-tail-quality-results",
        "make e11-cifar-resnet-practical-muon-bridge-results",
        "make e11-cifar-resnet-lt-muon-final-benchmark-results",
        "make e11-matrix-block-tightness-audit",
        "make e11-natural-negative-search-phase1-power-audit",
        "make e11-natural-negative-search-phase1-results",
        "make e11-natural-negative-search-phase1-eval",
        "make e11-natural-negative-search-phase2-settings",
        "make e11-natural-negative-search-phase2-results",
        "make e11-natural-negative-search-phase2-eval",
        "make e11-natural-negative-search-phase2-power-audit",
        "make e11-appendix-results",
        "make e11-all-results",
        "make e11-paper-assets",
        "make e11-guardrail-assets",
        "make e11-all-assets",
        "make e11-paper-pdf",
        "make e11-artifact-review-packet",
        "make e11-mechanism-referee-audit",
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
        "CIFAR-100-LT ResNet18 imbalance sweep",
        "CIFAR-100-LT ResNet18 all-layer JVP tail-quality diagnostic",
        "CIFAR-100-LT ResNet18 NS-Muon final-training benchmark pilot",
        "CIFAR-100-LT ResNet18 practical Muon trajectory bridge",
        "discussion/e11_matrix_block_tightness_audit.md",
        "Deterministic exact-witness and ratio-identity audit",
        "discussion/e11_natural_negative_search_phase1_NNS-P1-cifar100lt-resnet18-new-partitions.md",
        "26/26 observed primary rows",
        "discussion/e11_natural_negative_search_phase1_interim_synthesis.md",
        "Complete-family claim-boundary synthesis",
        "discussion/e11_natural_negative_search_phase2_NNS-P2-heldout-architecture-boundary.md",
        "Completed ResNet34 held-out architecture phase2 raw readout",
        "discussion/e11_natural_negative_search_phase2_evaluation.md",
        "Completed Holm evaluator",
        "discussion/e11_natural_negative_search_phase2_power_audit.md",
        "Detectable-effect and outcome-state audit",
        "make e11-natural-negative-search-phase1-interim-synthesis",
        "discussion/e11_top_conference_claim_decision_audit.md",
        "Paper-level supportable/completed-negative-boundary/blocked claim and rebuttal-readiness contract",
        "discussion/e11_manuscript_claim_trace.md",
        "Main-tex claim trace that maps every top-conference claim decision to anchors",
        "discussion/e11_artifact_review_packet.md",
        "Artifact-review command, gate, local-state, and reviewer-response packet",
        "discussion/e11_mechanism_referee_audit.md",
        "Adversarial alternative-explanation, theory-measurement, and falsification-trigger audit",
        "make e11-top-conference-claim-decision-audit",
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
        "CIFAR-100-LT ResNet18 imbalance sweep",
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
        "CIFAR-100-LT ResNet18 local diagnostics",
        "negative NS-Muon final-training pilot",
        "completed v5 final gates failed under the registered protocol",
    ]
    missing_completion_audit = [
        phrase for phrase in required_completion_audit_phrases if phrase not in completion_audit
    ]
    if missing_completion_audit:
        raise AssertionError(f"completion audit missing required content: {missing_completion_audit}")
    assert_forbidden_phrases_absent(
        "completion audit stale evidence wording",
        completion_audit,
        [
            "The real-data evidence is controlled scikit-learn digits.",
            "Add CIFAR-100-LT, ImageNet-LT, or iNaturalist-style diagnostics.",
        ],
    )
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
        "larger long-tail datasets",
        "CIFAR-100-LT NS-Muon final-training pilot is negative",
        "registered tuned benchmark protocol is still not_ready",
        "All-layer ResNet18 JVP diagnostics are present",
        "v5 ResNeXt50-32x4d and CIFAR-10 cross-partition final condition-score outputs are complete and failed the P0 gate family",
    ]
    missing_end_self_review = [
        phrase for phrase in required_end_self_review_phrases if phrase not in end_self_review
    ]
    if missing_end_self_review:
        raise AssertionError(f"end-of-draft self-review missing required content: {missing_end_self_review}")
    assert_forbidden_phrases_absent(
        "end-of-draft self-review stale evidence wording",
        end_self_review,
        [
            "long-tail benchmarks are absent",
            "No standard long-tailed benchmark or tuned practical optimizer comparison.",
            "The core real-data evidence is controlled scikit-learn digits.",
            "Layerwise diagnostics are for a two-layer MLP.",
        ],
    )
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
        "figures/e11_cifar100_resnet_condition_score_audit/cifar100_resnet_condition_score_audit.png",
        "figures/e11_cifar100_resnet_imbalance_sweep/cifar100_resnet_imbalance_sweep.png",
        "squared drift ratio=0.5501",
        "head-alignment ratio=2.083",
        "Frobenius-norm ratio=3.112",
        "operator-norm ratio=0.5543",
        "polar(M_t) squared drift ratio=0.8199",
        "trajectory NS(M_t) squared drift ratio=0.8019",
        "scaled-JVP threshold accuracy=1",
        "source-observed positive-control Spearman=0.981",
        "early-layer prior Spearman=0.9126",
        "scaled-JVP held-out Spearman=-0.3203",
        "best simple condition composite early-minus-scaled-JVP Spearman=0.8656",
        "scaled-JVP residual Spearman=-0.4872",
        "CIFAR-100-LT ResNet18 imbalance sweep",
        "worst drift ratio=0.6075",
        "tail_train_per_class=100",
        "best tail accuracy=0.3297",
        "many balanced accuracy=0.3665",
        "medium=0.1036",
        "few=0.0129",
        "SGD-aug all=0.4105",
        "SGD-aug few=0.1047",
        "few diff vs AdamW-aug=0.0194",
        "NS-Muon final-training",
        "all=0.1265",
        "few=0.0008889",
        "all diff vs AdamW-aug=-0.2348",
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
        "CIFAR-100-LT ResNet18 imbalance sweep",
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
        "tail-count imbalance sweep",
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
        "CIFAR-100-LT ResNet18 imbalance sweep",
        "Practical optimizer bridge on CIFAR-100-LT",
        "NS-Muon final-training pilot",
        "Acceptance Gates",
        "working test environment with both `torch` and `pytest`",
    ]
    missing_top_conference = [
        phrase for phrase in required_top_conference_phrases if phrase not in top_conference_plan
    ]
    if missing_top_conference:
        raise AssertionError(f"top-conference plan missing required gates: {missing_top_conference}")
    condition_score_protocol = Path(
        "discussion/e11_cifar100_resnet_condition_score_protocol.md"
    ).read_text(encoding="utf-8")
    required_condition_score_protocol_phrases = [
        "CIFAR-100 ResNet Condition-Score Protocol",
        "condition_score_v2_calibrated_residual",
        "primary_heldout_checkpoint",
        "primary_heldout_architecture",
        "primary_heldout_data",
        "G1-no-target-leakage",
        "G2-primary-residual-prediction",
        "below-one threshold accuracy is at least 0.8",
        "cannot support a final long-tail accuracy claim",
        "legacy checkpoint-transfer tables may describe the failure",
    ]
    missing_condition_score_protocol = [
        phrase for phrase in required_condition_score_protocol_phrases if phrase not in condition_score_protocol
    ]
    if missing_condition_score_protocol:
        raise AssertionError(
            f"condition-score protocol missing required content: {missing_condition_score_protocol}"
        )
    condition_score_next = Path("discussion/e11_cifar100_resnet_condition_score_next.md").read_text(
        encoding="utf-8"
    )
    required_condition_score_next_phrases = [
        "Condition-Score v2 Retrospective Analysis",
        "condition_score_v2_calibrated_residual",
        "legacy scaled-JVP residual score remains negative",
        "Held-Out Evaluation",
        "registered held-out condition-score evaluation now fails",
        "primary_heldout_data_residual_spearman",
        "This is not the P0 predictive-condition result",
        "ResNet34 architecture split and the CIFAR-10-LT data split",
        "p0_predictive_condition_claim",
        "not_ready",
    ]
    missing_condition_score_next = [
        phrase for phrase in required_condition_score_next_phrases if phrase not in condition_score_next
    ]
    if missing_condition_score_next:
        raise AssertionError(
            f"condition-score v2 retrospective analysis missing required content: {missing_condition_score_next}"
        )
    heldout_failure_theory_note = Path(
        "discussion/e11_condition_score_heldout_failure_theory_note.md"
    ).read_text(encoding="utf-8")
    required_heldout_failure_note_phrases = [
        "Held-Out Condition-Score Failure Theory Note",
        "boundary condition for the theory",
        "These held-out splits are now spent",
        "A direction-threshold diagnostic",
        "A residual-ranking diagnostic",
        "does not support the claim that the current v2 condition score predicts held-out layer-risk ranking",
    ]
    assert_required_phrases(
        "held-out condition-score failure theory note",
        heldout_failure_theory_note,
        required_heldout_failure_note_phrases,
    )
    condition_score_theory_bridge = Path("discussion/e11_condition_score_theory_bridge.md").read_text(
        encoding="utf-8"
    )
    required_condition_score_theory_bridge_phrases = [
        "Condition-Score Theory Bridge",
        "Score Target Register",
        "direction_threshold",
        "residual_layer_ranking",
        "blocked_by_heldout_failure",
        "These held-out splits are now spent",
        "fresh P0 predictive-condition attempt",
        "Post-Fresh Protocol Reading",
        "evaluated failed boundary",
        "later frozen v5 transport-normalized score",
        "does not support a claim that `condition_score_v2_calibrated_residual` or the later frozen v5 transport-normalized score predicts held-out residual layer-risk ranking",
    ]
    assert_required_phrases(
        "condition-score theory bridge",
        condition_score_theory_bridge,
        required_condition_score_theory_bridge_phrases,
    )
    condition_score_fresh_protocol = Path("discussion/e11_condition_score_fresh_protocol.md").read_text(
        encoding="utf-8"
    )
    required_condition_score_fresh_protocol_phrases = [
        "Fresh Condition-Score Protocol",
        "Quarantine Register",
        "Score Freeze Registry",
        "fresh_final_heldout_architecture",
        "ResNet50 CIFAR stem",
        "head=0,2,4,6,8; tail=1,3,5,7,9",
        "condition_score_v3_zero_fit_scaled_jvp",
        "spent ResNet34 and original CIFAR-10 held-outs",
        "fresh ResNet50 architecture residual-ranking gate fails",
    ]
    assert_required_phrases(
        "fresh condition-score protocol",
        condition_score_fresh_protocol,
        required_condition_score_fresh_protocol_phrases,
    )
    condition_score_fresh_evaluation = Path("discussion/e11_condition_score_fresh_evaluation.md").read_text(
        encoding="utf-8"
    )
    required_condition_score_fresh_evaluation_phrases = [
        "Fresh Condition-Score Evaluation",
        "condition_score_v3_zero_fit_scaled_jvp",
        "not_run",
        "not_ready",
        "fresh_final_heldout_architecture_residual_spearman",
        "fresh_final_heldout_data_partition_residual_spearman",
        "spent ResNet34/original-CIFAR-10 held-outs out of fitting and final evidence",
    ]
    assert_required_phrases(
        "fresh condition-score evaluation",
        condition_score_fresh_evaluation,
        required_condition_score_fresh_evaluation_phrases,
    )
    condition_score_failure_mechanism_audit = Path(
        "discussion/e11_condition_score_failure_mechanism_audit.md"
    ).read_text(encoding="utf-8")
    required_condition_score_failure_phrases = [
        "Condition-Score Failure Mechanism Audit",
        "diagnostic-only",
        "O3-v3-resnet50-jvp-reversal",
        "ResNet50 Stage Reversal",
        "architecture/parameterization reversal",
        "bottleneck/downsample parameterization",
        "Blocked: tuning a new score on these rows",
    ]
    assert_required_phrases(
        "condition-score failure mechanism audit",
        condition_score_failure_mechanism_audit,
        required_condition_score_failure_phrases,
    )
    condition_score_v4_protocol = Path("discussion/e11_condition_score_v4_protocol.md").read_text(
        encoding="utf-8"
    )
    required_condition_score_v4_phrases = [
        "Condition-Score V4 Protocol",
        "Spent Final Split Register",
        "condition_score_v4_two_axis_transport_jvp",
        "WideResNet50-2 CIFAR stem",
        "head=0,1,4,7,8; tail=2,3,5,6,9",
        "validation_frozen",
        "evaluated_not_ready",
        "CIFAR-10 mixed final data split failed residual ranking",
        "Blocked now: claiming a v4 predictive condition",
    ]
    assert_required_phrases(
        "condition-score v4 protocol",
        condition_score_v4_protocol,
        required_condition_score_v4_phrases,
    )
    condition_score_v5_protocol = Path("discussion/e11_condition_score_v5_theory_protocol.md").read_text(
        encoding="utf-8"
    )
    required_condition_score_v5_phrases = [
        "Condition-Score V5 Theory Protocol",
        "transport-normalized score contract",
        "sandwiched tail-drift",
        "partition/architecture transport normalization",
        "ResNeXt50-32x4d",
        "v5_validation_cifar100lt_mod4_partition",
        "v5_final_data_cifar10lt_cross_partition",
        "Blocked now: fitting, selecting, or thresholding a v5 score on any v2/v3/v4 final row",
    ]
    assert_required_phrases(
        "condition-score v5 theory protocol",
        condition_score_v5_protocol,
        required_condition_score_v5_phrases,
    )
    condition_score_v5_theory_to_score = Path(
        "discussion/e11_condition_score_v5_theory_to_score_map.md"
    ).read_text(encoding="utf-8")
    required_condition_score_v5_theory_to_score_phrases = [
        "Condition-Score V5 Theory-to-Score Map",
        "Transport-Stable Sandwich Residual",
        "theory-to-measurement bridge",
        "Theorem Proxy Map",
        "Score Lineage",
        "Transport Normalization Contract",
        "Post-Final Failure Reading",
        "Post-Final Transport Obligations",
        "Next Protocol Firewall",
        "Falsifiable Predictions",
        "Required Ablation Matrix",
        "Claim Readiness Ledger",
        "not_ready",
        "completed final gates failed under the frozen score",
        "Blocked now: fitting, selecting, thresholding, or reweighting any v5 score on",
    ]
    assert_required_phrases(
        "condition-score v5 theory-to-score map",
        condition_score_v5_theory_to_score,
        required_condition_score_v5_theory_to_score_phrases,
    )
    condition_score_v5_freeze = Path("discussion/e11_condition_score_v5_validation_freeze.md").read_text(
        encoding="utf-8"
    )
    required_condition_score_v5_freeze_phrases = [
        "Condition-Score V5 Validation Freeze",
        "not_run",
        "not_ready",
        "transport-normalized residual score",
        "V5F-1-validation-output",
        "V5F-4-residual-score-freeze",
        "Final evaluation is now unblocked as a run",
        "A P0 claim still requires both final splits",
    ]
    assert_required_phrases(
        "condition-score v5 validation freeze",
        condition_score_v5_freeze,
        required_condition_score_v5_freeze_phrases,
    )
    condition_score_v5_final = Path("discussion/e11_condition_score_v5_final_evaluation.md").read_text(
        encoding="utf-8"
    )
    required_condition_score_v5_final_phrases = [
        "Condition-Score V5 Final Evaluation",
        "condition_score_v5_transport_normalized_amplitude_minus_direction",
        "does not refit",
        "does not reselect",
        "baseline-dominance",
        "v5_p0_predictive_condition_claim",
    ]
    assert_required_phrases(
        "condition-score v5 final evaluation",
        condition_score_v5_final,
        required_condition_score_v5_final_phrases,
    )
    condition_score_v5_power = Path(
        "discussion/e11_condition_score_v5_final_power_audit.md"
    ).read_text(encoding="utf-8")
    required_condition_score_v5_power_phrases = [
        "Condition-Score V5 Final Power Audit",
        "pre-output detectable-effect contract",
        "Fisher-z resolution",
        "mean-Spearman MDE",
        "underpowered",
        "negative transport",
        "Outcome State Machine",
    ]
    assert_required_phrases(
        "condition-score v5 final power audit",
        condition_score_v5_power,
        required_condition_score_v5_power_phrases,
    )
    condition_score_v5_interpret = Path(
        "discussion/e11_condition_score_v5_final_interpretation_plan.md"
    ).read_text(encoding="utf-8")
    required_condition_score_v5_interpret_phrases = [
        "Condition-Score V5 Final Interpretation Plan",
        "post-output interpretation lock",
        "Current Interpretation Summary",
        "completed_final_failed_boundary",
        "outcome-to-claim state machine",
        "Leakage Lock",
        "p0_claim_eligible",
        "data_transport_boundary",
        "architecture_transport_boundary",
    ]
    assert_required_phrases(
        "condition-score v5 final interpretation plan",
        condition_score_v5_interpret,
        required_condition_score_v5_interpret_phrases,
    )
    condition_score_v5_response = Path(
        "discussion/e11_condition_score_v5_reviewer_failure_response.md"
    ).read_text(encoding="utf-8")
    required_condition_score_v5_response_phrases = [
        "Condition-Score V5 Reviewer Failure Response",
        "top-conference reviewer failure response",
        "claim-downgrade plan",
        "Failure Mode Register",
        "Reviewer Objection Map",
        "Claim Downgrade Actions",
        "Next Evidence Queue",
        "post-final-leakage-pressure",
    ]
    assert_required_phrases(
        "condition-score v5 reviewer failure response",
        condition_score_v5_response,
        required_condition_score_v5_response_phrases,
    )
    theory_proof_obligations = Path("discussion/e11_theory_proof_obligation_register.md").read_text(
        encoding="utf-8"
    )
    required_theory_proof_phrases = [
        "Theory Proof-Obligation Register",
        "theory-facing top-conference checklist",
        "Proof Obligations",
        "Assumption Stress Tests",
        "Claim Scope Boundaries",
        "Theorem-To-Experiment Queue",
        "PTO-1-local-linearization",
        "PTO-2-matrix-block-boundary",
        "PTO-4-v5-transport-score",
        "main_theorem_contract_and_tightness_audit_generated",
        "make e11-matrix-block-theorem-proof",
        "make e11-matrix-block-tightness-audit",
        "finite_registered_phase1_phase2_null_candidate_with_caveats",
        "optimizer-performance",
    ]
    assert_required_phrases(
        "theory proof-obligation register",
        theory_proof_obligations,
        required_theory_proof_phrases,
    )
    gap_register_frame = pd.read_csv("results/e11_top_conference_gap_register/gap_register.csv")
    assert_top_conference_gap_register(gap_register_frame)
    claim_decision_frame = pd.read_csv(
        "results/e11_top_conference_claim_decision_audit/claim_decision_matrix.csv"
    )
    reviewer_objection_frame = pd.read_csv(
        "results/e11_top_conference_claim_decision_audit/reviewer_objection_matrix.csv"
    )
    rebuttal_pack_frame = pd.read_csv(
        "results/e11_top_conference_claim_decision_audit/rebuttal_response_pack.csv"
    )
    manuscript_queue_frame = pd.read_csv(
        "results/e11_top_conference_claim_decision_audit/manuscript_edit_queue.csv"
    )
    paper_sequence_frame = pd.read_csv(
        "results/e11_top_conference_claim_decision_audit/paper_sequence.csv"
    )
    readiness_summary_frame = pd.read_csv(
        "results/e11_top_conference_claim_decision_audit/readiness_summary.csv"
    )
    assert_top_conference_claim_decision_audit(
        claim_decision_frame,
        reviewer_objection_frame,
        rebuttal_pack_frame,
        manuscript_queue_frame,
        paper_sequence_frame,
        readiness_summary_frame,
    )
    top_conference_gap_register = Path("discussion/e11_top_conference_gap_register.md").read_text(
        encoding="utf-8"
    )
    required_gap_register_phrases = [
        "Top-Conference Gap Register",
        "Minimum Viable Top-Tier Mechanism Paper",
        "P0-PredictiveCondition",
        "P0-StandardBenchmark",
        "discussion/e11_cifar100_resnet_condition_score_protocol.md",
        "discussion/e11_cifar100_resnet_condition_score_next.md",
        "discussion/e11_cifar100_resnet_condition_score_next_heldout_evaluation.md",
        "discussion/e11_condition_score_fresh_protocol.md",
        "v5 has consumed both frozen final splits",
        "discussion/e11_condition_score_v4_protocol.md",
        "discussion/e11_condition_score_v4_final_evaluation.md",
        "discussion/e11_matrix_block_theorem_proof.md",
        "discussion/e11_matrix_block_tightness_audit.md",
        "matched-gain theorem statement",
        "coefficient-ratio identity",
        "discussion/e11_theory_proof_obligation_register.md",
        "proof-obligation register",
        "discussion/e11_condition_score_v5_theory_protocol.md",
        "discussion/e11_condition_score_v5_theory_to_score_map.md",
        "results/e11_condition_score_v5_theory_to_score_map/next_protocol_firewall.csv",
        "pre-registered next-protocol firewall",
        "unspent split reset",
        "endpoint-separated residual/direction gates",
        "negative-boundary retention",
        "discussion/e11_condition_score_v5_validation_freeze.md",
        "discussion/e11_condition_score_v5_final_evaluation.md",
        "completed frozen final evaluator",
        "discussion/e11_condition_score_v5_final_interpretation_plan.md",
        "outcome-to-claim state machine",
        "current_interpretation_summary.csv",
        "discussion/e11_bold_conjecture_register.md",
        "bold-conjecture",
        "careful-verification",
        "BC-2-endpoint-factorized-transport",
        "discussion/e11_condition_score_v5_reviewer_failure_response.md",
        "claim-downgrade plan",
        "current_gate_snapshot.csv",
        "active_failure_modes.csv",
        "V5-RFR-4-direction-guardrail-failure",
        "discussion/e11_condition_score_v5_direction_guardrail_failure_audit.md",
        "score_axis_contrast.csv",
        "gate_boundary_summary.csv",
        "mechanistic_diagnosis.csv",
        "next_protocol_requirements.csv",
        "discussion/e11_heldout_generality_audit.md",
        "generality_evidence_matrix.csv",
        "bounded_support_with_caveated_heldout_boundaries",
        "head_gain_gate_pass_rows=0",
        "scripts/e11_write_condition_score_v5_direction_guardrail_failure_audit.py",
        "V5-DGF-2-architecture-direction-threshold-fails",
        "V5-DGF-3-data-residual-ranking-reverses",
        "transport-normalized score contract",
        "transport-stable sandwich residual proposition",
        "theorem terms to measurable score features",
        "direction guardrail, raw amplitude, early-depth nuisance, and transport-normalized validation",
        "discussion/e11_mechanism_referee_audit.md",
        "adversarial alternative-explanation matrix",
        "theory-to-measurement contract",
        "falsification-trigger matrix",
        "validation-freeze boundary",
        "validation-only mod-4 CIFAR-100-LT split",
        "condition_score_v5_transport_normalized_amplitude_minus_direction",
        "0.645 [0.5898, 0.7002]",
        "not_ready",
        "ResNeXt50-32x4d",
        "make e11-cifar-resnet-condition-score-v5-final-eval",
        "make e11-cifar-resnet-condition-score-v5-final-power-audit",
        "make e11-cifar-resnet-condition-score-v5-final-interpretation-plan",
        "make e11-cifar-resnet-condition-score-v5-reviewer-failure-response",
        "make e11-cifar-resnet-condition-score-v5-direction-guardrail-failure-audit",
        "CIFAR-10 mixed final data split fails",
        "data-partition reversal mechanism",
        "discussion/e11_natural_head_tail_boundary.md",
        "no strict natural primary full tail-output drift counterexample",
        "component true-logit and secondary loss/margin/accuracy tradeoff candidates",
        "discussion/e11_natural_negative_search_protocol.md",
        "search-space registry, metric contract, multiplicity rule, stopping rule",
        "discussion/e11_natural_negative_search_phase1_NNS-P1-cifar100lt-resnet18-new-partitions.md",
        "discussion/e11_natural_negative_search_phase1_NNS-P1-cifar10lt-resnet18-cross-partitions.md",
        "discussion/e11_natural_negative_search_phase1_NNS-P1-tail-quality-controls.md",
        "26/26 primary metric rows",
        "discussion/e11_natural_negative_search_phase1_interim_synthesis.md",
        "raw_worse_rows=0",
        "finite registered phase1 null candidate",
        "quality_gate_fail_rows=23",
        "scripts/e11_run_natural_negative_search_phase1.py",
        "scripts/slurm/e11_natural_negative_search_phase1.sbatch",
        "scripts/e11_run_natural_negative_search_phase2.py",
        "scripts/slurm/e11_natural_negative_search_phase2.sbatch",
        "scripts/e11_evaluate_natural_negative_search_phase2.py",
        "scripts/e11_write_natural_negative_phase2_power_audit.py",
        "discussion/e11_natural_negative_search_phase2_NNS-P2-heldout-architecture-boundary.md",
        "discussion/e11_natural_negative_search_phase2_evaluation.md",
        "discussion/e11_natural_negative_search_phase2_power_audit.md",
        "results/e11_natural_negative_search_protocol/phase2_heldout_architecture/settings_registry.csv",
        "results/e11_natural_negative_search_protocol/phase2_multiplicity_evaluation",
        "results/e11_natural_negative_search_protocol/phase2_power_audit",
        "8/8 observed rows",
        "head-gain",
        "scripts/e11_evaluate_natural_negative_search_phase1.py",
        "make e11-natural-negative-search-phase1-power-audit",
        "make e11-natural-negative-search-phase1-eval",
        "make e11-natural-negative-search-phase2-settings",
        "make e11-natural-negative-search-phase2-eval",
        "make e11-natural-negative-search-phase2-power-audit",
        "scripts/e11_write_natural_negative_phase1_interim_synthesis.py",
        "multiplicity evaluator",
        "NNS-E4 returning finite_null_candidate",
        "quality_gate_fail_rows=23",
        "held-out architecture",
        "benchmark-level performance claim",
        "discussion/e11_cifar100_resnet_lt_tuned_benchmark_protocol.md",
        "validation/final seed splits",
        "finite-NS-Muon candidate grids",
        "familywise final comparisons",
        "scripts/e11_run_cifar100_resnet_lt_tuned_benchmark.py",
        "scripts/slurm/e11_cifar100_resnet_lt_tuned_benchmark_validation.sbatch",
        "discussion/e11_cifar100_resnet_lt_tuned_benchmark_selection.md",
        "discussion/e11_cifar100_resnet_lt_tuned_benchmark_power_audit.md",
        "discussion/e11_cifar100_resnet_lt_tuned_benchmark_variance_prior_audit.md",
        "discussion/e11_cifar100_resnet_lt_tuned_benchmark_protocol_seal.md",
        "discussion/e11_cifar100_resnet_lt_tuned_benchmark_fairness_audit.md",
        "discussion/e11_cifar100_resnet_lt_tuned_benchmark_final_analysis_plan.md",
        "discussion/e11_cifar100_resnet_lt_tuned_benchmark_final_robustness_plan.md",
        "discussion/e11_cifar100_resnet_lt_tuned_benchmark_final_execution_plan.md",
        "discussion/e11_cifar100_resnet_lt_tuned_benchmark_final_evaluation.md",
        "discussion/e11_cifar100_resnet_lt_tuned_benchmark_final_launch_audit.md",
        "scripts/e11_write_cifar100_resnet_lt_tuned_benchmark_power_audit.py",
        "scripts/e11_write_cifar100_resnet_lt_tuned_benchmark_refresh_firewall.py",
        "scripts/e11_write_cifar100_resnet_lt_tuned_benchmark_protocol_seal.py",
        "scripts/e11_write_cifar100_resnet_lt_tuned_benchmark_fairness_audit.py",
        "scripts/e11_write_cifar100_resnet_lt_tuned_benchmark_variance_prior_audit.py",
        "scripts/e11_write_cifar100_resnet_lt_tuned_benchmark_final_analysis_plan.py",
        "scripts/e11_write_cifar100_resnet_lt_tuned_benchmark_final_robustness_plan.py",
        "scripts/e11_write_cifar100_resnet_lt_tuned_benchmark_final_execution_plan.py",
        "scripts/e11_evaluate_cifar100_resnet_lt_tuned_benchmark_final.py",
        "scripts/e11_submit_cifar100_resnet_lt_tuned_benchmark_final.py",
        "scripts/slurm/e11_cifar100_resnet_lt_tuned_benchmark_final.sbatch",
        "completed validation-only seed variability and spent-pilot paired-diff variability",
        "paired tests, Holm adjustment, reporting schema, and claim states",
        "gate-checked final runner, Slurm wrapper, selected recipe families, and seed 20..29",
        "post-final paired seed metrics, Holm-adjusted primary comparisons, all-class guardrails, occupancy summaries, and claim gates",
        "FEP gates, evaluator readiness, queue capacity, duplicate final-job guards, and whether sbatch was called",
        "final seeds 20..29 quarantined",
        "make e11-cifar-resnet-lt-tuned-benchmark-selection",
        "make e11-cifar-resnet-lt-tuned-benchmark-refresh-firewall",
        "make e11-cifar-resnet-lt-tuned-benchmark-protocol-seal",
        "make e11-cifar-resnet-lt-tuned-benchmark-fairness-audit",
        "make e11-cifar-resnet-lt-tuned-benchmark-power-audit",
        "make e11-cifar-resnet-lt-tuned-benchmark-variance-prior-audit",
        "make e11-cifar-resnet-lt-tuned-benchmark-final-analysis-plan",
        "make e11-cifar-resnet-lt-tuned-benchmark-final-robustness-plan",
        "make e11-cifar-resnet-lt-tuned-benchmark-final-execution-plan",
        "make e11-cifar-resnet-lt-tuned-benchmark-final-eval",
        "make e11-cifar-resnet-lt-tuned-benchmark-final-launch-audit",
        "164-setting validation registry",
        "planned_occupancy_trace_path",
        "occupancy_trace.csv",
        "validation refresh firewall",
        "protocol hash seal",
        "post-partial-exposure boundary",
        "fairness audit",
        "Muon candidate-budget disclosure",
        "shared validation/final seed rules",
        "exact sign-flip, bootstrap CI, sign-count sensitivity checks",
        "robustness-sensitivity guards",
        f"next legal refresh boundary at array index {refresh_next_missing}",
        "forbidden metric-dependent launch/order/selection/final-submit actions",
        "passing leakage, validation-refresh firewall, protocol-seal, fairness-budget, and robustness-sensitivity guards",
        "discussion/e11_muon_state_distribution_contract.md",
        "state-distribution transport contract",
        "MSD-T1 local-response integrand",
        "MSD-T2 state-occupancy measure",
        "MSD-T4 terminal risk",
        "discussion/e11_submission_repro_audit.md",
        "discussion/e11_artifact_review_packet.md",
        "artifact-review packet",
        "GPU-pending boundary",
        "leakage/optional-stopping audit",
        "preferred pdflatex/bibtex/xelatex clean-checkout gate remains not_ready",
        "GPU via Slurm",
        "No row in this register authorizes a stronger paper claim by itself",
    ]
    missing_gap_register = [
        phrase for phrase in required_gap_register_phrases if phrase not in top_conference_gap_register
    ]
    if missing_gap_register:
        raise AssertionError(f"top-conference gap register missing required content: {missing_gap_register}")
    claim_decision_text = Path("discussion/e11_top_conference_claim_decision_audit.md").read_text(
        encoding="utf-8"
    )
    assert_required_phrases(
        "top-conference claim decision audit",
        claim_decision_text,
        [
            "E11 Top-Conference Claim Decision Audit",
            "paper-level claim contract",
            "does not add new empirical results",
            "Claim Decision Matrix",
            "Reviewer Objection Matrix",
            "Rebuttal Response Pack",
            "Manuscript Edit Queue",
            "Paper Sequence",
            "TCD-1-main-mechanism-theorem",
            "supportable_main_with_assumptions",
            "blocked_completed_final_failed_boundary",
            "finite_null_candidate_with_caveats",
            "blocked_protocol_pending",
            "v5_p0_predictive_condition_claim=not_ready",
            "observed=26/26",
            "raw_worse_rows=0",
            "quality_gate_fail_rows=23",
            "using any final row to refit or reselect the score",
            "quarantine benchmark claims",
            "discussion/e11_cifar100_resnet_lt_tuned_benchmark_leakage_audit.md",
            "discussion/e11_cifar100_resnet_lt_tuned_benchmark_refresh_firewall.md",
            "discussion/e11_cifar100_resnet_lt_tuned_benchmark_protocol_seal.md",
            "discussion/e11_cifar100_resnet_lt_tuned_benchmark_final_execution_plan.md",
            "discussion/e11_cifar100_resnet_lt_tuned_benchmark_final_evaluation.md",
            "discussion/e11_cifar100_resnet_lt_tuned_benchmark_final_launch_audit.md",
            "TLA-1-selection-rule-frozen=pass",
            "VRF-current-validation-prefix=partial_grid_no_claim_change",
            "VRF-A1-prefix-result-refresh=pass",
            "VRF-F5-validation-leaderboard-performance-claim=active",
            "TPS-current-post-exposure-boundary=post_partial_validation_exposure_sealed",
            "TPS-1-hash-manifest-complete=pass",
            "TPS-5-post-exposure-claim-authority=pass",
            "TBF-1-registered-family-coverage=pass",
            "TBF-3-candidate-budget-disclosure=pass_with_disclosure",
            "validation_budget_muon_candidates=108",
            "RBP-G1-final-output-quarantine=pass",
            "RBP-G3-robustness-tests-registered=pass",
            "robustness_test_rows=16",
            "using partial validation leaderboard to change selection, launch order, or final seed plan",
            "validation refresh firewall forbidden-action matrix",
            "changing sealed protocol surfaces after partial validation exposure without a fresh preregistered protocol",
            "hiding validation-budget asymmetry",
            "claiming robust final performance without sign-flip/bootstrap sensitivity",
            "FEP-1-selection-gates-ready=not_ready",
            "TFE-6-final-claim-state=not_ready",
            "FLA-5-submit-flag=dry_run",
            "final_launch_status=blocked_final_launch_gates_not_ready",
            "final_submit_command=<empty>",
            "running final-safe-submit before FEP/TFE/FLA gates pass",
            "discussion/e11_muon_state_distribution_contract.md",
            "state-distribution transport contract",
            "MSD-T1-local-response-integrand=local_integrand_supported_on_sampled_states",
            "MSD-T2-state-occupancy-measure=occupancy_measure_missing_for_final_training",
            "MSG-4-top-tier-practical-claim=blocked_until_state_distribution_and_final_gates_pass",
            "local compatibility can coexist with poor final performance",
            "worst-case-vs-realized distinction",
            "finite-null-candidate wording only with detectable-effect, head-gain, and quality caveats",
        ],
    )
    manuscript_trace = pd.read_csv("results/e11_manuscript_claim_trace/claim_trace.csv")
    blocked_phrase_audit = pd.read_csv("results/e11_manuscript_claim_trace/blocked_phrase_audit.csv")
    manuscript_trace_config = json.loads(
        Path("results/e11_manuscript_claim_trace/config.json").read_text(encoding="utf-8")
    )
    expected_trace_claims = {
        "TCD-1-main-mechanism-theorem",
        "TCD-2-natural-drift-diagnostic",
        "TCD-3-predictive-condition-generalization",
        "TCD-4-natural-counterexample-or-finite-null",
        "TCD-5-optimizer-performance-benchmark",
        "TCD-6-artifact-reproducibility",
    }
    if set(manuscript_trace["claim_id"]) != expected_trace_claims:
        raise AssertionError("manuscript claim trace must cover every top-conference claim decision")
    decision_by_claim = claim_decision_frame.set_index("claim_id")["current_decision"].to_dict()
    trace_decision_by_claim = manuscript_trace.set_index("claim_id")["current_decision"].to_dict()
    if trace_decision_by_claim != decision_by_claim:
        raise AssertionError("manuscript claim trace decisions must match the top-conference decision matrix")
    expected_blocked_pairs = {
        (str(row["claim_id"]), phrase.strip())
        for row in claim_decision_frame.to_dict("records")
        for phrase in str(row["author_blocked_wording"]).split(";")
        if phrase.strip()
    }
    observed_blocked_pairs = {
        (str(row["claim_id"]), str(row["blocked_phrase"]))
        for row in blocked_phrase_audit.to_dict("records")
    }
    if observed_blocked_pairs != expected_blocked_pairs:
        raise AssertionError("manuscript blocked-phrase audit must mirror the top-conference blocked wording set")
    if not (
        manuscript_trace["missing_anchors"].eq("none").all()
        and manuscript_trace["blocked_phrase_audit"].eq("pass").all()
        and blocked_phrase_audit["audit_status"].eq("pass").all()
        and bool(manuscript_trace_config.get("all_anchors_present"))
        and bool(manuscript_trace_config.get("blocked_phrases_absent"))
    ):
        raise AssertionError("manuscript claim trace must find all anchors and exclude blocked positive wording")
    manuscript_trace_text = Path("discussion/e11_manuscript_claim_trace.md").read_text(encoding="utf-8")
    assert_required_phrases(
        "manuscript claim trace",
        manuscript_trace_text,
        [
            "E11 Manuscript Claim Trace",
            "top-conference claim decisions",
            "Claim Trace",
            "Blocked Phrase Audit",
            "present_as_supportable_scoped_claim",
            "present_as_completed_negative_boundary",
            "present_as_finite_null_candidate_with_caveats",
            "present_as_blocked_protocol_context",
            "partial tuned-validation observations are progress accounting only",
            "using partial validation leaderboard to change selection, launch order, or final seed plan",
            "preferred pdflatex/bibtex/xelatex clean-checkout reproducibility is complete on this server",
            "Every `supportable` decision must have a local scoped manuscript anchor",
        ],
    )
    clean_replay_dir = Path("results/e11_clean_worktree_replay_audit")
    clean_replay_summary = pd.read_csv(clean_replay_dir / "run_summary.csv")
    clean_replay_gates = pd.read_csv(clean_replay_dir / "gate_matrix.csv")
    clean_replay_config = json.loads((clean_replay_dir / "config.json").read_text(encoding="utf-8"))
    clean_replay_log = (clean_replay_dir / "command_log_tail.txt").read_text(encoding="utf-8")
    expected_clean_replay_gates = {
        "CWR-1-detached-worktree-created",
        "CWR-2-local-attachment-excluded",
        "CWR-3-e11-check-passes",
        "CWR-4-pytest-pass-observed",
        "CWR-5-git-diff-check-observed",
        "CWR-6-worktree-clean-after-check",
    }
    if set(clean_replay_gates["gate_id"]) != expected_clean_replay_gates:
        raise AssertionError("clean worktree replay audit gate IDs changed unexpectedly")
    if not clean_replay_gates["status"].astype(str).eq("pass").all():
        raise AssertionError("clean worktree replay audit gates must all pass")
    if len(clean_replay_summary) != 1:
        raise AssertionError("clean worktree replay audit must contain one run summary row")
    clean_replay_row = clean_replay_summary.iloc[0].astype(str).to_dict()
    if clean_replay_row.get("status") != "pass":
        raise AssertionError("clean worktree replay summary must pass")
    if clean_replay_row.get("server_readme_present_in_worktree") != "no":
        raise AssertionError("clean worktree replay must exclude serverREADME.md")
    if "e11-check" not in clean_replay_row.get("command", ""):
        raise AssertionError("clean worktree replay must run e11-check")
    if clean_replay_row.get("post_status_short") != "clean" or clean_replay_row.get("post_untracked") != "none":
        raise AssertionError("clean worktree replay must leave the replay worktree clean")
    if clean_replay_config.get("status") != "pass" or clean_replay_config.get(
        "server_readme_policy"
    ) != "must_be_absent_from_clean_worktree":
        raise AssertionError("clean worktree replay config must record pass and serverREADME exclusion policy")
    for phrase in ["71 passed", "git diff --check"]:
        if phrase not in clean_replay_log:
            raise AssertionError(f"clean worktree replay log missing phrase: {phrase}")
    clean_replay_text = Path("discussion/e11_clean_worktree_replay_audit.md").read_text(encoding="utf-8")
    assert_required_phrases(
        "clean worktree replay audit",
        clean_replay_text,
        [
            "E11 Clean Worktree Replay Audit",
            "detached clean Git worktree",
            "serverREADME.md",
            "CWR-3-e11-check-passes",
            "CWR-6-worktree-clean-after-check",
            "does not close the preferred `pdflatex`/`bibtex`/`xelatex` clean-checkout gate",
        ],
    )
    pdf_render_dir = Path("results/e11_pdf_render_boundary_audit")
    pdf_tool_status = pd.read_csv(pdf_render_dir / "pdf_inspection_tool_status.csv")
    pdf_render_gates = pd.read_csv(pdf_render_dir / "render_boundary_gates.csv")
    pdf_render_config = json.loads((pdf_render_dir / "config.json").read_text(encoding="utf-8"))
    expected_pdf_tools = {
        "pdftotext",
        "pdfinfo",
        "mutool",
        "gs",
        "python:pypdf",
        "python:PyPDF2",
        "python:fitz",
        "python:pdfminer.high_level",
    }
    expected_pdf_render_gates = {
        "PRB-1-rendered-pdf-binaries",
        "PRB-2-pdf-hashes-recorded",
        "PRB-3-source-claim-trace-covered",
        "PRB-4-pdf-text-extraction-tool",
        "PRB-5-pdf-metadata-tool",
        "PRB-6-preferred-latex-toolchain",
    }
    if set(pdf_tool_status["tool"]) != expected_pdf_tools:
        raise AssertionError("PDF render boundary audit tool-status rows changed unexpectedly")
    if set(pdf_render_gates["gate_id"]) != expected_pdf_render_gates:
        raise AssertionError("PDF render boundary audit gate IDs changed unexpectedly")
    pdf_gate_lookup = pdf_render_gates.set_index("gate_id")["status"].astype(str).to_dict()
    if not set(pdf_gate_lookup.values()).issubset({"pass", "not_ready"}):
        raise AssertionError("PDF render boundary gates must use only pass/not_ready statuses")
    if not (
        pdf_gate_lookup["PRB-1-rendered-pdf-binaries"] == "pass"
        and pdf_gate_lookup["PRB-2-pdf-hashes-recorded"] == "pass"
        and pdf_gate_lookup["PRB-3-source-claim-trace-covered"] == "pass"
        and pdf_gate_lookup["PRB-4-pdf-text-extraction-tool"] in {"pass", "not_ready"}
        and pdf_gate_lookup["PRB-5-pdf-metadata-tool"] in {"pass", "not_ready"}
        and pdf_gate_lookup["PRB-6-preferred-latex-toolchain"] in {"pass", "not_ready"}
    ):
        raise AssertionError("PDF render boundary audit must keep binary/hash/source gates passing and tool gates explicit")
    if pdf_render_config.get("purpose") != "rendered PDF inspection boundary audit":
        raise AssertionError("PDF render boundary audit config purpose changed unexpectedly")
    pdf_render_text = Path("discussion/e11_pdf_render_boundary_audit.md").read_text(encoding="utf-8")
    assert_required_phrases(
        "PDF render boundary audit",
        pdf_render_text,
        [
            "E11 PDF Render Boundary Audit",
            "PDF Inspection Tool Status",
            "Render Boundary Gates",
            "PRB-1-rendered-pdf-binaries",
            "PRB-4-pdf-text-extraction-tool",
            "PRB-5-pdf-metadata-tool",
            "Allowed now: cite rendered PDF byte hashes",
            "Blocked now: claiming rendered-PDF text-layer or page-metadata verification",
        ],
    )
    heldout_generality_dir = Path("results/e11_heldout_generality_audit")
    heldout_generality_evidence = pd.read_csv(heldout_generality_dir / "generality_evidence_matrix.csv")
    heldout_generality_gates = pd.read_csv(heldout_generality_dir / "generality_claim_gate.csv")
    heldout_generality_config = json.loads(
        (heldout_generality_dir / "config.json").read_text(encoding="utf-8")
    )
    expected_heldout_audit_ids = {
        "HGA-1-resnet18-default-support",
        "HGA-2-resnet18-checkpoint-support",
        "HGA-3-resnet18-tail-quality-frequency-support",
        "HGA-4-resnet18-all-layer-support",
        "HGA-5-cifar10-data-family-boundary",
        "HGA-6-resnet34-architecture-boundary",
        "HGA-7-condition-score-heldout-boundary",
    }
    expected_heldout_gate_ids = {
        "HGG-1-positive-resnet18-family",
        "HGG-2-heldout-data-family",
        "HGG-3-heldout-architecture-family",
        "HGG-4-predictive-score-generality",
        "HGG-5-paper-generality-claim",
    }
    if set(heldout_generality_evidence["audit_id"]) != expected_heldout_audit_ids:
        raise AssertionError("held-out generality audit must preserve the fixed evidence matrix")
    if set(heldout_generality_gates["gate_id"]) != expected_heldout_gate_ids:
        raise AssertionError("held-out generality audit must preserve the fixed gate set")
    heldout_gate_status = heldout_generality_gates.set_index("gate_id")["status"].to_dict()
    if heldout_gate_status["HGG-5-paper-generality-claim"] != "bounded_support_with_caveated_heldout_boundaries":
        raise AssertionError("held-out generality paper gate must remain bounded and caveated")
    heldout_evidence_text = " ".join(heldout_generality_evidence.astype(str).agg(" ".join, axis=1).tolist())
    for phrase in [
        "primary_positive_mechanism_support",
        "heldout_data_finite_null_boundary",
        "heldout_architecture_finite_null_boundary",
        "predictive_score_negative_boundary",
        "head_gain_gate_pass_rows=0",
        "not mechanism validation, not a universal natural null, and not final-performance evidence",
        "does not support a successful held-out natural-task predictor",
    ]:
        if phrase not in heldout_evidence_text:
            raise AssertionError(f"held-out generality evidence missing phrase: {phrase}")
    if not (
        heldout_generality_config.get("evidence_rows") == 7
        and heldout_generality_config.get("gate_rows") == 5
        and heldout_generality_config.get("current_status")
        == "bounded_support_with_caveated_heldout_boundaries"
        and heldout_generality_config.get("resnet34_head_gain_caveat")
        == "all phase2 rows have head_gain_gate=False"
    ):
        raise AssertionError("held-out generality config must preserve bounded status and ResNet34 head-gain caveat")
    heldout_generality_text = Path("discussion/e11_heldout_generality_audit.md").read_text(
        encoding="utf-8"
    )
    assert_required_phrases(
        "held-out generality audit",
        heldout_generality_text,
        [
            "E11 Held-Out Generality Audit",
            "positive mechanism support from caveated held-out",
            "Generality Evidence Matrix",
            "Generality Claim Gate",
            "HGG-5-paper-generality-claim",
            "bounded_support_with_caveated_heldout_boundaries",
            "ResNet34 phase2 finite-null result as mechanism validation",
            "all phase2",
            "head_gain_gate=False",
            "successful predictive condition",
        ],
    )
    bold_conjecture_dir = Path("results/e11_bold_conjecture_register")
    bold_conjectures = pd.read_csv(bold_conjecture_dir / "conjecture_register.csv")
    bold_stress_tests = pd.read_csv(bold_conjecture_dir / "stress_test_matrix.csv")
    bold_ladder = pd.read_csv(bold_conjecture_dir / "claim_upgrade_ladder.csv")
    bold_config = json.loads((bold_conjecture_dir / "config.json").read_text(encoding="utf-8"))
    if set(bold_conjectures["conjecture_id"]) != {
        "BC-1-local-sandwich-drift",
        "BC-2-endpoint-factorized-transport",
        "BC-3-natural-counterexamples-are-structured",
        "BC-4-muon-performance-needs-state-distribution-theory",
        "BC-5-layer-risk-is-predictable-but-the-current-proxy-is-incomplete",
    }:
        raise AssertionError("bold conjecture register must preserve the fixed conjecture set")
    if set(bold_stress_tests["stress_id"]) != {
        "BST-1-local-theorem-scope",
        "BST-2-score-transport-scope",
        "BST-3-natural-family-scope",
        "BST-4-optimizer-performance-scope",
        "BST-5-heldout-predictor-scope",
    }:
        raise AssertionError("bold conjecture register must preserve the fixed stress-test set")
    if set(bold_ladder["ladder_id"]) != {
        "BCL-1-current-submission",
        "BCL-2-predictive-condition-upgrade",
        "BCL-3-natural-boundary-upgrade",
        "BCL-4-optimizer-performance-upgrade",
    }:
        raise AssertionError("bold conjecture register must preserve the fixed claim-upgrade ladder")
    if bold_config != {
        "conjecture_rows": 5,
        "stress_test_rows": 5,
        "upgrade_ladder_rows": 4,
        "positive_claim_boundary": "local matched-head-gain mechanism only",
        "score_upgrade_boundary": "new unspent validation/final protocol required",
    }:
        raise AssertionError(f"bold conjecture config drifted: {bold_config}")
    bold_register_text = " ".join(
        [
            Path("discussion/e11_bold_conjecture_register.md").read_text(encoding="utf-8"),
            " ".join(bold_conjectures.astype(str).to_numpy().ravel()),
            " ".join(bold_stress_tests.astype(str).to_numpy().ravel()),
            " ".join(bold_ladder.astype(str).to_numpy().ravel()),
        ]
    )
    assert_required_phrases(
        "bold conjecture register",
        bold_register_text,
        [
            "supportable_as_local_mechanism",
            "conjecture_for_new_protocol_not_current_positive_claim",
            "bounded_support_with_caveated_heldout_boundaries",
            "benchmark_claim_blocked_until_tuned_validation_and_final_seeds",
            "predictive_condition_failed_but_boundary_is_informative",
            "new unspent validation/final protocol required",
            "bold conjecture, careful verification",
            "Current positive wording remains limited to the local matched-head-gain mechanism",
        ],
    )
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
        or "make e11-cifar-resnet-imbalance-sweep-results" not in readme
        or "make e11-cifar-resnet-layer-jvp-tail-quality-results" not in readme
        or "make e11-cifar-resnet-condition-score-heldout-architecture-results" not in readme
        or "make e11-cifar-resnet-condition-score-heldout-data-results" not in readme
        or "make e11-cifar-resnet-condition-score-heldout-eval" not in readme
        or "make e11-cifar-resnet-condition-score-fresh-architecture-results" not in readme
        or "make e11-cifar-resnet-condition-score-fresh-data-results" not in readme
        or "make e11-cifar-resnet-condition-score-fresh-eval" not in readme
        or "make e11-matrix-block-theorem-proof" not in readme
        or "make e11-matrix-block-tightness-audit" not in readme
        or "make e11-theory-proof-obligation-register" not in readme
        or "make e11-cifar-resnet-condition-score-v5-theory-protocol" not in readme
        or "make e11-cifar-resnet-condition-score-v5-theory-to-score-map" not in readme
        or "make e11-cifar-resnet-condition-score-v5-validation-results" not in readme
        or "make e11-cifar-resnet-condition-score-v5-validation-freeze" not in readme
        or "make e11-cifar-resnet-condition-score-v5-architecture-results" not in readme
        or "make e11-cifar-resnet-condition-score-v5-data-results" not in readme
        or "make e11-cifar-resnet-condition-score-v5-final-eval" not in readme
        or "make e11-cifar-resnet-condition-score-v5-final-power-audit" not in readme
        or "make e11-cifar-resnet-condition-score-v5-final-interpretation-plan" not in readme
        or "make e11-cifar-resnet-condition-score-v5-reviewer-failure-response" not in readme
        or "make e11-cifar-resnet-condition-score-v5-direction-guardrail-failure-audit" not in readme
        or "make e11-cifar-resnet-lt-muon-final-benchmark-results" not in readme
        or "make e11-cifar-resnet-lt-tuned-benchmark-protocol" not in readme
        or "make e11-cifar-resnet-lt-tuned-benchmark-settings" not in readme
        or "make e11-cifar-resnet-lt-tuned-benchmark-validation-results" not in readme
        or "make e11-cifar-resnet-lt-tuned-benchmark-selection" not in readme
        or "make e11-cifar-resnet-lt-tuned-benchmark-protocol-seal" not in readme
        or "make e11-cifar-resnet-lt-tuned-benchmark-fairness-audit" not in readme
        or "make e11-cifar-resnet-lt-tuned-benchmark-power-audit" not in readme
        or "make e11-cifar-resnet-lt-tuned-benchmark-variance-prior-audit" not in readme
        or "make e11-cifar-resnet-lt-tuned-benchmark-final-analysis-plan" not in readme
        or "make e11-cifar-resnet-lt-tuned-benchmark-final-robustness-plan" not in readme
        or "make e11-cifar-resnet-lt-tuned-benchmark-final-execution-plan" not in readme
        or "make e11-cifar-resnet-lt-tuned-benchmark-final-eval" not in readme
        or "make e11-cifar-resnet-lt-tuned-benchmark-final-launch-audit" not in readme
        or "make e11-cifar-resnet-lt-tuned-benchmark-final-safe-submit" not in readme
        or "make e11-cifar-resnet-lt-tuned-benchmark-slurm-plan" not in readme
        or "make e11-cifar-resnet-lt-tuned-benchmark-launch-audit" not in readme
        or "make e11-cifar-resnet-lt-tuned-benchmark-safe-submit" not in readme
        or "make e11-cifar-resnet-practical-muon-bridge-results" not in readme
        or "make e11-natural-head-tail-boundary-audit" not in readme
        or "make e11-natural-negative-search-protocol" not in readme
        or "make e11-natural-negative-search-phase1-power-audit" not in readme
        or "make e11-natural-negative-search-phase1-settings" not in readme
        or "make e11-natural-negative-search-phase1-results" not in readme
        or "make e11-natural-negative-search-phase1-eval" not in readme
        or "make e11-natural-negative-search-phase1-interim-synthesis" not in readme
        or "make e11-natural-negative-search-phase2-settings" not in readme
        or "make e11-natural-negative-search-phase2-results" not in readme
        or "make e11-natural-negative-search-phase2-eval" not in readme
        or "make e11-natural-negative-search-phase2-power-audit" not in readme
        or "make e11-heldout-generality-audit" not in readme
        or "make e11-bold-conjecture-register" not in readme
        or "make e11-muon-state-distribution-contract" not in readme
        or "make e11-top-conference-claim-decision-audit" not in readme
        or "make e11-manuscript-claim-trace" not in readme
        or "make e11-mechanism-referee-audit" not in readme
        or "make e11-submission-repro-audit" not in readme
        or "make e11-pdf-render-boundary-audit" not in readme
        or "make e11-artifact-review-packet" not in readme
        or "make e11-camera-ready-package-audit" not in readme
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
        "discussion/e11_submission_repro_audit.md",
        "discussion/e11_artifact_review_packet.md",
        "discussion/e11_camera_ready_package_audit.md",
        "discussion/e11_pdf_render_boundary_audit.md",
        "discussion/e11_cifar100_resnet_lt_tuned_benchmark_variance_prior_audit.md",
        "discussion/e11_cifar100_resnet_lt_tuned_benchmark_final_analysis_plan.md",
        "discussion/e11_cifar100_resnet_lt_tuned_benchmark_final_robustness_plan.md",
        "discussion/e11_cifar100_resnet_lt_tuned_benchmark_final_execution_plan.md",
        "discussion/e11_cifar100_resnet_lt_tuned_benchmark_final_evaluation.md",
        "discussion/e11_cifar100_resnet_lt_tuned_benchmark_final_launch_audit.md",
        "discussion/e11_mechanism_referee_audit.md",
        "discussion/e11_bold_conjecture_register.md",
        "discussion/e11_muon_state_distribution_contract.md",
        "discussion/e11_natural_negative_search_phase2_NNS-P2-heldout-architecture-boundary.md",
        "discussion/e11_natural_negative_search_phase2_evaluation.md",
        "discussion/e11_natural_negative_search_phase2_power_audit.md",
        "discussion/e11_heldout_generality_audit.md",
        "results/e11_natural_negative_search_protocol/phase2_heldout_architecture/settings_registry.csv",
        "results/e11_natural_negative_search_protocol/phase2_multiplicity_evaluation/primary_decisions.csv",
        "results/e11_natural_negative_search_protocol/phase2_power_audit/minimum_detectable_effect.csv",
        "results/e11_heldout_generality_audit/generality_evidence_matrix.csv",
        "results/e11_heldout_generality_audit/generality_claim_gate.csv",
        "results/e11_bold_conjecture_register/conjecture_register.csv",
        "results/e11_bold_conjecture_register/stress_test_matrix.csv",
        "results/e11_muon_state_distribution_contract/state_distribution_terms.csv",
        "results/e11_muon_state_distribution_contract/falsification_tests.csv",
        "results/e11_pdf_render_boundary_audit/pdf_inspection_tool_status.csv",
        "results/e11_pdf_render_boundary_audit/render_boundary_gates.csv",
        "results/e11_camera_ready_package_audit/package_item_matrix.csv",
        "results/e11_camera_ready_package_audit/submission_gate_matrix.csv",
        "results/e11_camera_ready_package_audit/camera_ready_checklist.csv",
        "results/e11_cifar100_resnet_lt_tuned_benchmark/variance_prior_audit/variance_prior_summary.csv",
        "results/e11_cifar100_resnet_lt_tuned_benchmark/variance_prior_audit/mde_sensitivity_from_empirical_sd.csv",
        "results/e11_cifar100_resnet_lt_tuned_benchmark/final_analysis_plan/primary_comparison_family.csv",
        "results/e11_cifar100_resnet_lt_tuned_benchmark/final_analysis_plan/reporting_schema.csv",
        "results/e11_cifar100_resnet_lt_tuned_benchmark/final_robustness_plan/robustness_test_matrix.csv",
        "results/e11_cifar100_resnet_lt_tuned_benchmark/final_robustness_plan/gate_matrix.csv",
        "results/e11_cifar100_resnet_lt_tuned_benchmark/final_execution_plan/gate_matrix.csv",
        "results/e11_cifar100_resnet_lt_tuned_benchmark/final_execution_plan/final_family_run_plan.csv",
        "results/e11_cifar100_resnet_lt_tuned_benchmark/final_evaluation/primary_decisions.csv",
        "results/e11_cifar100_resnet_lt_tuned_benchmark/final_evaluation/claim_gate_report.csv",
        "results/e11_cifar100_resnet_lt_tuned_benchmark/final_launch_audit/latest_final_launch_decision.csv",
        "results/e11_cifar100_resnet_lt_tuned_benchmark/final_launch_audit/latest_gate_snapshot.csv",
        "results/e11_cifar100_resnet_lt_tuned_benchmark/validation_protocol_seal/hash_manifest.csv",
        "results/e11_cifar100_resnet_lt_tuned_benchmark/validation_protocol_seal/seal_gate_matrix.csv",
        "discussion/e11_cifar100_resnet_lt_tuned_benchmark_protocol_seal.md",
        "results/e11_cifar100_resnet_lt_tuned_benchmark/validation_fairness_audit/fairness_gate_matrix.csv",
        "results/e11_cifar100_resnet_lt_tuned_benchmark/validation_fairness_audit/family_budget_matrix.csv",
        "discussion/e11_cifar100_resnet_lt_tuned_benchmark_fairness_audit.md",
        "Ignored Local Artifacts",
        "results/e11_artifact_manifest.json",
    ]:
        if phrase not in artifact_manifest:
            raise AssertionError(f"artifact manifest missing required section or reference: {phrase}")
    manifest_json = json.loads(Path("results/e11_artifact_manifest.json").read_text(encoding="utf-8"))
    assert_valid_artifact_manifest(manifest_json)
    submission_repro_dir = Path("results/e11_submission_repro_audit")
    submission_toolchain = pd.read_csv(submission_repro_dir / "toolchain_status.csv")
    submission_pdf_checks = pd.read_csv(submission_repro_dir / "pdf_artifact_checks.csv")
    submission_source_manifest = pd.read_csv(submission_repro_dir / "source_package_manifest.csv")
    submission_build_gates = pd.read_csv(submission_repro_dir / "build_gate_summary.csv")
    expected_submission_tools = {"pdflatex", "bibtex", "xelatex", "tectonic", "gs", "git", "python"}
    expected_submission_gates = {
        "R1-source-revision",
        "R2-working-tree-scope",
        "R3-preferred-latex-toolchain",
        "R4-tectonic-fallback-toolchain",
        "R5-rendered-pdfs",
        "R6-full-artifact-validation",
    }
    if not (
        set(submission_toolchain["tool"]) == expected_submission_tools
        and set(submission_build_gates["gate_id"]) == expected_submission_gates
        and set(submission_pdf_checks["path"]) == {
            "paper/specgrad_activation_paper/main.pdf",
            "paper/specgrad_activation_paper/two_page.pdf",
        }
        and submission_source_manifest["audit_status"].eq("pass").all()
    ):
        raise AssertionError(
            "submission reproducibility audit must preserve toolchain rows, build gates, rendered PDF checks, and source package manifest"
        )
    submission_gate_lookup = submission_build_gates.set_index("gate_id")["status"].to_dict()
    submission_tool_lookup = submission_toolchain.set_index("tool")["available"].to_dict()
    if not (
        submission_gate_lookup["R1-source-revision"] == "pass"
        and submission_gate_lookup["R2-working-tree-scope"] == "info"
        and submission_gate_lookup["R3-preferred-latex-toolchain"] in {"pass", "not_ready"}
        and submission_gate_lookup["R4-tectonic-fallback-toolchain"] == "pass"
        and submission_gate_lookup["R5-rendered-pdfs"] == "pass"
        and submission_gate_lookup["R6-full-artifact-validation"] in {"pass", "external_check_required"}
        and submission_tool_lookup["tectonic"] == "yes"
        and submission_pdf_checks["audit_status"].eq("pass").all()
        and submission_pdf_checks["header_is_pdf"].eq("yes").all()
        and (submission_pdf_checks["size_bytes"].astype(int) >= 10_000).all()
    ):
        raise AssertionError(
            "submission reproducibility audit must keep Tectonic and rendered-PDF gates passing while clean-checkout/full-validation gates remain explicit"
        )
    submission_repro_text = Path("discussion/e11_submission_repro_audit.md").read_text(encoding="utf-8")
    assert_required_phrases(
        "submission reproducibility audit",
        submission_repro_text,
        [
            "E11 Submission Reproducibility Audit",
            "Toolchain Status",
            "Rendered PDF Checks",
            "Source Package Manifest",
            "Build Gate Summary",
            "pdflatex/bibtex/xelatex",
            "Tectonic-backed server evidence",
            "Blocked now: claiming a preferred LaTeX clean-checkout reproduction",
        ],
    )
    artifact_review_dir = Path("results/e11_artifact_review_packet")
    artifact_command_matrix = pd.read_csv(artifact_review_dir / "command_matrix.csv")
    artifact_gate_matrix = pd.read_csv(artifact_review_dir / "gate_matrix.csv")
    artifact_local_state = pd.read_csv(artifact_review_dir / "local_state_contract.csv")
    artifact_reviewer_response = pd.read_csv(artifact_review_dir / "reviewer_response.csv")
    artifact_config = json.loads((artifact_review_dir / "config.json").read_text(encoding="utf-8"))
    expected_artifact_command_ids = {
        "AR-C1",
        "AR-C2",
        "AR-C3",
        "AR-C4",
        "AR-C5",
        "AR-C6",
        "AR-C7",
        "AR-C8",
        "AR-C9",
        "AR-C10",
        "AR-C11",
        "AR-G1",
        "AR-G2",
        "AR-G3",
    }
    if not expected_artifact_command_ids.issubset(set(artifact_command_matrix["command_id"])):
        raise AssertionError("artifact review packet command matrix missing CPU/GPU reviewer commands")
    artifact_command_text = " ".join(artifact_command_matrix.astype(str).agg(" ".join, axis=1).tolist())
    for phrase in [
        "make PYTHON=/data/conda_envs/SpatialQuantization/bin/python e11-full",
        "make PYTHON=/data/conda_envs/SpatialQuantization/bin/python e11-paper-assets",
        "make PYTHON=/data/conda_envs/SpatialQuantization/bin/python e11-paper-pdf",
        "make PYTHON=/data/conda_envs/SpatialQuantization/bin/python e11-check",
        "make PYTHON=/data/conda_envs/SpatialQuantization/bin/python e11-cifar-resnet-lt-tuned-benchmark-leakage-audit",
        "make PYTHON=/data/conda_envs/SpatialQuantization/bin/python e11-clean-worktree-replay-audit",
        "make PYTHON=/data/conda_envs/SpatialQuantization/bin/python e11-pdf-render-boundary-audit",
        "make PYTHON=/data/conda_envs/SpatialQuantization/bin/python e11-cifar-resnet-lt-tuned-benchmark-final-execution-plan",
        "make PYTHON=/data/conda_envs/SpatialQuantization/bin/python e11-cifar-resnet-lt-tuned-benchmark-final-eval",
        "make PYTHON=/data/conda_envs/SpatialQuantization/bin/python e11-cifar-resnet-lt-tuned-benchmark-final-launch-audit",
        "blocked_final_launch_gates_not_ready",
        "GPU via Slurm",
        "Not required to reproduce current paper claims",
    ]:
        if phrase not in artifact_command_text:
            raise AssertionError(f"artifact review command matrix missing phrase: {phrase}")
    if set(artifact_gate_matrix["gate_id"]) != expected_submission_gates:
        raise AssertionError("artifact review gate matrix must mirror submission reproducibility gates")
    artifact_gate_text = " ".join(artifact_gate_matrix.astype(str).agg(" ".join, axis=1).tolist())
    for phrase in ["external_toolchain_required", "local_scope_declared", "reproducible_now"]:
        if phrase not in artifact_gate_text:
            raise AssertionError(f"artifact review gate matrix missing reviewer status: {phrase}")
    expected_local_items = {
        "serverREADME.md",
        "Tectonic fallback",
        "Preferred LaTeX toolchain",
        "Rendered PDFs",
        "Full artifact validation",
        "Tuned validation leakage audit",
        "Clean worktree replay",
        "PDF render boundary audit",
        "Tuned final execution gates",
        "Tuned final evaluator",
        "Tuned final launch audit",
        "v5 final layer tables",
        "GPU dependence",
    }
    if not expected_local_items.issubset(set(artifact_local_state["item"])):
        raise AssertionError("artifact review local-state contract missing required items")
    artifact_local_text = " ".join(artifact_local_state.astype(str).agg(" ".join, axis=1).tolist())
    for phrase in [
        "Do not stage or commit this file",
        "Both frozen final split layer tables are present",
        "partial validation observations cannot change the registry",
        "tracked-source replay evidence",
        "rendered-PDF byte/header/hash and source-claim trace evidence",
        "Do not run final-safe-submit until every FEP gate passes",
        "current not_ready gates forbid final benchmark wording",
        "empty submit_command and blocked_final_launch_gates_not_ready",
        "not_required_for_current_artifact_review",
    ]:
        if phrase not in artifact_local_text:
            raise AssertionError(f"artifact review local-state contract missing boundary: {phrase}")
    artifact_response_text = " ".join(artifact_reviewer_response.astype(str).agg(" ".join, axis=1).tolist())
    for phrase in [
        "preferred venue-toolchain reproducibility",
        "serverREADME.md",
        "including the completed v5 final boundary readout",
        "detached Git worktree",
        "serverREADME.md is absent",
        "optional-stopping leakage",
        "rendered text-layer and page-metadata inspection are not_ready",
        "Do not claim rendered-PDF text-layer or metadata verification",
        "partial validation observations are progress accounting only",
        "final outputs are absent, no submit command is emitted, and final seeds 20..29 remain quarantined",
        "Do not run final-safe-submit or claim optimizer-performance",
        "Do not use failed v5 finals, not_ready gates, partial-family, or finite phase1 outputs as broader positive evidence",
        "Do not infer broad optimizer-performance, accuracy, or general predictive-condition claims",
    ]:
        if phrase not in artifact_response_text:
            raise AssertionError(f"artifact review response packet missing reviewer answer: {phrase}")
    artifact_review_text = Path("discussion/e11_artifact_review_packet.md").read_text(encoding="utf-8")
    assert_required_phrases(
        "artifact review packet",
        artifact_review_text,
        [
            "E11 Artifact Review Packet",
            "Reviewer Command Matrix",
            "Build Gate Matrix",
            "Local State Contract",
            "Reviewer Response Matrix",
            "make PYTHON=/data/conda_envs/SpatialQuantization/bin/python e11-full",
            "preferred pdflatex/bibtex/xelatex clean-checkout reproducibility",
            "v5 predictive-condition upgrade",
            "partial-validation optional-stopping",
            "rendered text-layer and page-metadata inspection",
            "final tuned benchmark Slurm submission or final-performance claim",
            "unqualified natural-null",
            "broad optimizer-performance claim",
            "Machine-readable tables",
        ],
    )
    if artifact_config.get("strongest_local_gate") != "make PYTHON=/data/conda_envs/SpatialQuantization/bin/python e11-full":
        raise AssertionError("artifact review config must record the strongest local gate")
    camera_ready_dir = Path("results/e11_camera_ready_package_audit")
    camera_items = pd.read_csv(camera_ready_dir / "package_item_matrix.csv")
    camera_gates = pd.read_csv(camera_ready_dir / "submission_gate_matrix.csv")
    camera_checklist = pd.read_csv(camera_ready_dir / "camera_ready_checklist.csv")
    camera_config = json.loads((camera_ready_dir / "config.json").read_text(encoding="utf-8"))
    expected_camera_items = {
        "CRP-1-source-bundle",
        "CRP-2-rendered-pdf-binaries",
        "CRP-3-claim-trace-clean",
        "CRP-4-reviewer-command-path",
        "CRP-5-local-attachment-excluded",
        "CRP-6-preferred-latex-boundary",
        "CRP-7-rendered-text-metadata-boundary",
        "CRP-8-final-claim-quarantine",
    }
    expected_camera_gates = {
        "CRG-1-current-server-package",
        "CRG-2-venue-toolchain-package",
        "CRG-3-claim-boundary-package",
        "CRG-4-local-file-exclusion",
    }
    expected_camera_steps = {
        "CRC-1-current-server-share",
        "CRC-2-venue-clean-checkout",
        "CRC-3-after-manuscript-edit",
        "CRC-4-before-commit",
    }
    if not (
        set(camera_items["item_id"]) == expected_camera_items
        and set(camera_gates["gate_id"]) == expected_camera_gates
        and set(camera_checklist["step_id"]) == expected_camera_steps
    ):
        raise AssertionError("camera-ready package audit must preserve fixed item, gate, and checklist IDs")
    camera_item_status = camera_items.set_index("item_id")["status"].astype(str).to_dict()
    camera_gate_status = camera_gates.set_index("gate_id")["status"].astype(str).to_dict()
    if not (
        camera_item_status["CRP-1-source-bundle"] == "pass"
        and camera_item_status["CRP-2-rendered-pdf-binaries"] == "pass"
        and camera_item_status["CRP-3-claim-trace-clean"] == "pass"
        and camera_item_status["CRP-4-reviewer-command-path"] == "pass"
        and camera_item_status["CRP-5-local-attachment-excluded"] == "pass"
        and camera_item_status["CRP-6-preferred-latex-boundary"] in {"pass", "not_ready"}
        and camera_item_status["CRP-7-rendered-text-metadata-boundary"] in {"pass", "not_ready"}
        and camera_item_status["CRP-8-final-claim-quarantine"] == "pass"
        and camera_gate_status["CRG-1-current-server-package"] == "pass"
        and camera_gate_status["CRG-2-venue-toolchain-package"] in {"pass", "not_ready"}
        and camera_gate_status["CRG-3-claim-boundary-package"] == "pass"
        and camera_gate_status["CRG-4-local-file-exclusion"] == "pass"
    ):
        raise AssertionError("camera-ready package audit gates must preserve current server pass and venue-boundary states")
    camera_text = " ".join(
        camera_items.astype(str).to_numpy().ravel().tolist()
        + camera_gates.astype(str).to_numpy().ravel().tolist()
        + camera_checklist.astype(str).to_numpy().ravel().tolist()
    )
    for phrase in [
        "serverREADME.md",
        "claim_trace.csv anchors are present",
        "blocked_phrase_audit.csv has no hits",
        "pdflatex/bibtex/xelatex clean-checkout reproducibility",
        "rendered-PDF text-layer or metadata verification",
        "final-claim quarantine",
        "make PYTHON=/data/conda_envs/SpatialQuantization/bin/python e11-full",
        "no new benchmark, predictive-score, or natural-counterexample wording",
    ]:
        if phrase not in camera_text:
            raise AssertionError(f"camera-ready package audit missing boundary phrase: {phrase}")
    camera_discussion = Path("discussion/e11_camera_ready_package_audit.md").read_text(encoding="utf-8")
    assert_required_phrases(
        "camera-ready package audit",
        camera_discussion,
        [
            "E11 Camera-Ready Package Audit",
            "package-readiness boundary",
            "Package Item Matrix",
            "Submission Gate Matrix",
            "Camera-Ready Checklist",
            "CRG-1-current-server-package",
            "CRG-2-venue-toolchain-package",
            "Blocked now: claiming full venue-toolchain clean-checkout reproducibility",
            "stronger benchmark and",
        ],
    )
    if (
        camera_config.get("current_server_gate") != "CRG-1-current-server-package"
        or camera_config.get("venue_toolchain_gate") != "CRG-2-venue-toolchain-package"
    ):
        raise AssertionError("camera-ready package audit config must record current-server and venue-toolchain gates")
    mechanism_referee_dir = Path("results/e11_mechanism_referee_audit")
    alternative_matrix = pd.read_csv(mechanism_referee_dir / "alternative_explanation_matrix.csv")
    theory_contract = pd.read_csv(mechanism_referee_dir / "theory_measurement_contract.csv")
    falsification_triggers = pd.read_csv(mechanism_referee_dir / "falsification_trigger_matrix.csv")
    mechanism_config = json.loads((mechanism_referee_dir / "config.json").read_text(encoding="utf-8"))
    expected_alternative_ids = {
        "MEA-1-head-gain-mismatch",
        "MEA-2-tail-unit-sensitivity",
        "MEA-3-weak-tail-checkpoint",
        "MEA-4-synthetic-only",
        "MEA-5-rank-only-predictor",
        "MEA-6-post-hoc-score-tuning",
        "MEA-7-anecdotal-natural-negative",
        "MEA-8-performance-proxy",
        "MEA-9-phase2-power-overread",
    }
    if set(alternative_matrix["audit_id"]) != expected_alternative_ids:
        raise AssertionError("mechanism referee audit must preserve the fixed alternative-explanation set")
    alternative_text = " ".join(alternative_matrix.astype(str).agg(" ".join, axis=1).tolist())
    for phrase in [
        "addressed_for_local_claim",
        "rejected_as_primary_explanation",
        "finite_phase1_and_phase2_null_candidates_with_caveats",
        "completed_phase2_finite_null_with_power_and_quality_caveats",
        "not_claimed",
        "the validator enforces matched-update/head-gain consistency",
        "unit-JVP ratios",
        "tail-rich control",
        "v5 transport-normalized score is frozen",
        "phase2 power audit fixes the Holm worst-case 80% MDE",
        "local drift improvements imply final long-tail optimizer superiority",
    ]:
        if phrase not in alternative_text:
            raise AssertionError(f"mechanism referee alternative matrix missing phrase: {phrase}")
    expected_contract_ids = {
        "TMC-1-local-linearization",
        "TMC-2-matched-head-gain",
        "TMC-3-sandwich-rank-boundary",
        "TMC-4-transport-normalized-score",
        "TMC-5-term-ablation-lineage",
        "TMC-6-natural-negative-registered-boundary",
    }
    if set(theory_contract["contract_id"]) != expected_contract_ids:
        raise AssertionError("mechanism referee audit must preserve the fixed theory-measurement contract set")
    contract_text = " ".join(theory_contract.astype(str).agg(" ".join, axis=1).tolist())
    for phrase in [
        "first-order tail-logit response",
        "head-gain-normalized comparison",
        "nrank(G_H)>ssrank(B_T,A_T)",
        "source-standardized transport residual",
        "finite natural boundary search",
        "global trajectory or convergence theorem",
        "universal natural-null wording outside registered phase1/phase2 families",
    ]:
        if phrase not in contract_text:
            raise AssertionError(f"mechanism referee theory contract missing phrase: {phrase}")
    expected_trigger_ids = {
        "FT-1-v5-final-fails",
        "FT-2-natural-family-incomplete",
        "FT-3-unit-jvp-misread",
        "FT-4-performance-overread",
        "FT-5-clean-checkout-gap",
        "FT-6-phase2-power-overread",
    }
    if set(falsification_triggers["trigger_id"]) != expected_trigger_ids:
        raise AssertionError("mechanism referee audit must preserve the fixed falsification-trigger set")
    trigger_text = " ".join(falsification_triggers.astype(str).agg(" ".join, axis=1).tolist())
    for phrase in [
        "Downgrade predictive-condition wording",
        "Allow only finite registered phase1/phase2 null-candidate wording",
        "finite phase1/phase2 null candidates with detectable-effect, head-gain, and quality caveats",
        "matched-head-gain local mechanism only",
        "mechanism diagnostic, not benchmark claim",
        "toolchain caveat",
        "underpowered phase2 null",
    ]:
        if phrase not in trigger_text:
            raise AssertionError(f"mechanism referee falsification matrix missing phrase: {phrase}")
    mechanism_referee_text = Path("discussion/e11_mechanism_referee_audit.md").read_text(encoding="utf-8")
    assert_required_phrases(
        "mechanism referee audit",
        mechanism_referee_text,
        [
            "E11 Mechanism Referee Audit",
            "Alternative Explanation Matrix",
            "Theory-To-Measurement Contract",
            "Falsification Trigger Matrix",
            "local matched-head-gain mechanism paper",
            "completed caveated phase2 finite-null-candidate wording",
            "Blocked now: broad optimizer-performance claims",
            "alternative_explanation_matrix.csv",
            "theory_measurement_contract.csv",
            "falsification_trigger_matrix.csv",
        ],
    )
    if mechanism_config.get("alternative_explanations") != 9:
        raise AssertionError("mechanism referee config must record nine alternative explanations")
    if mechanism_config.get("theory_measurement_contracts") != 6:
        raise AssertionError("mechanism referee config must record six theory-measurement contracts")
    if mechanism_config.get("falsification_triggers") != 6:
        raise AssertionError("mechanism referee config must record six falsification triggers")
    assert_no_unguarded_overclaims(
        [
            Path("README_E11.md"),
            Path("discussion/e11_evidence_index.md"),
            Path("discussion/e11_research_synthesis.md"),
            Path("discussion/e11_research_direction_map.md"),
            Path("discussion/e11_paper_readiness_audit.md"),
            Path("discussion/e11_cifar100_resnet_condition_score_protocol.md"),
            Path("discussion/e11_cifar100_resnet_condition_score_next.md"),
            Path("discussion/e11_cifar100_resnet_condition_score_next_heldout_evaluation.md"),
            Path("discussion/e11_condition_score_heldout_failure_theory_note.md"),
            Path("discussion/e11_condition_score_theory_bridge.md"),
            Path("discussion/e11_condition_score_fresh_protocol.md"),
            Path("discussion/e11_condition_score_fresh_evaluation.md"),
            Path("discussion/e11_top_conference_gap_register.md"),
            Path("discussion/e11_mechanism_referee_audit.md"),
            Path("discussion/e11_heldout_generality_audit.md"),
            Path("discussion/e11_bold_conjecture_register.md"),
            Path("discussion/e11_muon_state_distribution_contract.md"),
            Path("discussion/e11_natural_head_tail_boundary.md"),
            Path("discussion/e11_natural_negative_search_protocol.md"),
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
