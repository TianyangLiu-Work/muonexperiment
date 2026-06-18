import importlib.util
import re
from pathlib import Path

import pandas as pd
import pytest

from e11_condition_geometry.artifacts import (
    APPENDIX_RUNNER_DESCRIPTIONS,
    APPENDIX_RUNNER_SCRIPTS,
    ARTIFACT_DIRS,
    IGNORE_POLICY,
    KEY_DOCUMENTS,
    KEY_TABLES,
    LEGACY_GUARDRAIL_SCRIPTS,
    MAIN_EVIDENCE_STAGES,
    MAIN_RESULT_SCRIPTS,
    PAPER_ASSET_SCRIPTS,
    ZERO_ROW_ALLOWED_TABLES,
)


ROOT = Path(__file__).resolve().parents[1]


def script_identity_from_command(command: str) -> str | None:
    parts = command.split()
    if not parts:
        return None
    if parts[0] in {"python3", "$(PYTHON)"}:
        return next((part for part in parts[1:] if part.startswith("scripts/e11_") and part.endswith(".py")), None)
    if parts[0] == "sbatch":
        wrapper = ROOT / parts[1]
        wrapper_text = wrapper.read_text(encoding="utf-8")
        match = re.search(r"(?:^|\s)\S*python(?:3)?\s+(scripts/e11_[^\s\\]+\.py)", wrapper_text)
        assert match is not None, f"could not resolve Python runner from {wrapper}"
        return match.group(1)
    return None


def dedupe_preserving_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    unique = []
    for item in items:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def load_validator_module():
    path = ROOT / "scripts" / "e11_validate_outputs.py"
    spec = importlib.util.spec_from_file_location("e11_validate_outputs", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_artifact_manifest_module():
    path = ROOT / "scripts" / "e11_write_artifact_manifest.py"
    spec = importlib.util.spec_from_file_location("e11_write_artifact_manifest", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_artifact_manifest_size_ignores_python_caches(tmp_path: Path) -> None:
    manifest = load_artifact_manifest_module()
    package = tmp_path / "package"
    cache = package / "__pycache__"
    cache.mkdir(parents=True)
    (package / "source.py").write_text("print('ok')\n", encoding="utf-8")
    (cache / "source.cpython.pyc").write_bytes(b"x" * 1024)

    assert manifest.path_size_bytes(package) == (package / "source.py").stat().st_size


def test_overclaim_guard_rejects_unguarded_claim(tmp_path: Path) -> None:
    validator = load_validator_module()
    document = tmp_path / "bad.md"
    document.write_text("Muon is generally better than Adam on these tasks.\n", encoding="utf-8")

    with pytest.raises(AssertionError, match="unguarded paper-facing overclaims"):
        validator.assert_no_unguarded_overclaims([document])


def test_overclaim_guard_allows_explicit_rejection(tmp_path: Path) -> None:
    validator = load_validator_module()
    document = tmp_path / "guarded.md"
    document.write_text(
        "Do not claim Muon is generally better than Adam. "
        "The supported claim is update-spectrum shaping.\n",
        encoding="utf-8",
    )

    validator.assert_no_unguarded_overclaims([document])


def test_overclaim_guard_allows_markdown_emphasis_negation(tmp_path: Path) -> None:
    validator = load_validator_module()
    document = tmp_path / "markdown.md"
    document.write_text(
        "The current evidence does **not** support the broad claim that "
        "Muon is generally more stable in the state geometry.\n",
        encoding="utf-8",
    )

    validator.assert_no_unguarded_overclaims([document])


def test_required_phrase_guard_reports_missing_content() -> None:
    validator = load_validator_module()

    validator.assert_required_phrases("paper README", "make e11-paper-assets\nmake e11-check\n", [
        "make e11-paper-assets",
        "make e11-check",
    ])
    with pytest.raises(AssertionError, match="paper README missing required content"):
        validator.assert_required_phrases("paper README", "make e11-paper-assets\n", [
            "make e11-paper-assets",
            "make e11-check",
        ])


def test_forbidden_phrase_guard_reports_unsafe_paper_wording() -> None:
    validator = load_validator_module()
    unsafe = "可以解释为什么某些优化器在 head accuracy 相同的情况下 tail accuracy 更好"

    validator.assert_forbidden_phrases_absent(
        "paper main tex",
        "这个视角给出一个可检验假设。",
        [unsafe],
    )
    with pytest.raises(AssertionError, match="paper main tex contains forbidden content"):
        validator.assert_forbidden_phrases_absent(
            "paper main tex",
            f"这个视角{unsafe}。",
            [unsafe],
        )


def test_includegraphics_guard_resolves_paths_from_tex_file(tmp_path: Path) -> None:
    validator = load_validator_module()
    figure_dir = tmp_path / "figures"
    figure_dir.mkdir()
    (figure_dir / "plot.png").write_bytes(b"fake png bytes")
    tex_path = tmp_path / "paper" / "main.tex"
    tex_path.parent.mkdir()
    tex_path.write_text(
        r"\includegraphics[width=\linewidth]{../figures/plot.png}" "\n",
        encoding="utf-8",
    )

    validator.assert_includegraphics_files_exist(tex_path)


def test_includegraphics_guard_rejects_missing_local_file(tmp_path: Path) -> None:
    validator = load_validator_module()
    tex_path = tmp_path / "main.tex"
    tex_path.write_text(r"\includegraphics{missing.png}" "\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="missing includegraphics files"):
        validator.assert_includegraphics_files_exist(tex_path)


def test_latex_log_guard_allows_clean_rerunfilecheck_package_line(tmp_path: Path) -> None:
    validator = load_validator_module()
    log_path = tmp_path / "main.log"
    log_path.write_text(
        "Package: rerunfilecheck 2022-07-10 v1.10 Rerun checks for auxiliary files (HO)\n",
        encoding="utf-8",
    )

    validator.assert_latex_log_has_no_serious_warnings(log_path)


def test_latex_log_guard_rejects_serious_warnings(tmp_path: Path) -> None:
    validator = load_validator_module()
    log_path = tmp_path / "main.log"
    log_path.write_text(
        "Underfull \\hbox (badness 2134) in paragraph at lines 1--2\n"
        "Overfull \\hbox (12.0pt too wide) in paragraph at lines 1--2\n"
        "LaTeX Warning: There were undefined references.\n",
        encoding="utf-8",
    )

    with pytest.raises(AssertionError, match="serious LaTeX log issues"):
        validator.assert_latex_log_has_no_serious_warnings(log_path)


def test_latex_box_guard_rejects_vbox_warnings(tmp_path: Path) -> None:
    validator = load_validator_module()
    log_path = tmp_path / "two_page.log"
    log_path.write_text(
        "Overfull \\vbox (0.31131pt too high) has occurred while \\output is active []\n",
        encoding="utf-8",
    )

    with pytest.raises(AssertionError, match="LaTeX box warnings"):
        validator.assert_latex_log_has_no_box_warnings(log_path)


def test_pdf_artifact_guard_accepts_pdf_header_and_size(tmp_path: Path) -> None:
    validator = load_validator_module()
    pdf_path = tmp_path / "main.pdf"
    pdf_path.write_bytes(b"%PDF" + b"x" * 32)

    validator.assert_pdf_artifact_is_valid(pdf_path, min_size_bytes=10)


def test_pdf_artifact_guard_rejects_tiny_or_non_pdf_files(tmp_path: Path) -> None:
    validator = load_validator_module()
    tiny_pdf = tmp_path / "tiny.pdf"
    tiny_pdf.write_bytes(b"%PDF")
    text_file = tmp_path / "not_pdf.pdf"
    text_file.write_bytes(b"not a pdf" + b"x" * 32)

    with pytest.raises(AssertionError, match="unexpectedly small"):
        validator.assert_pdf_artifact_is_valid(tiny_pdf, min_size_bytes=10)
    with pytest.raises(AssertionError, match="does not start with %PDF"):
        validator.assert_pdf_artifact_is_valid(text_file, min_size_bytes=10)


def test_bibtex_citation_guard_accepts_defined_citations(tmp_path: Path) -> None:
    validator = load_validator_module()
    tex_path = tmp_path / "main.tex"
    bib_path = tmp_path / "references.bib"
    tex_path.write_text(r"\citep{alpha,beta}" "\n", encoding="utf-8")
    bib_path.write_text(
        "@misc{alpha,\n  title={Alpha}\n}\n"
        "@inproceedings{beta,\n  title={Beta}\n}\n",
        encoding="utf-8",
    )

    validator.assert_bibtex_citations_are_defined(tex_path, bib_path)


def test_bibtex_citation_guard_rejects_missing_items(tmp_path: Path) -> None:
    validator = load_validator_module()
    tex_path = tmp_path / "main.tex"
    bib_path = tmp_path / "references.bib"
    tex_path.write_text(r"\citet{alpha} and \citep{beta}" "\n", encoding="utf-8")
    bib_path.write_text(
        "@misc{alpha,\n  title={Alpha}\n}\n",
        encoding="utf-8",
    )

    with pytest.raises(AssertionError, match="missing from references.bib"):
        validator.assert_bibtex_citations_are_defined(tex_path, bib_path)


def test_bibtex_citation_guard_rejects_unused_items(tmp_path: Path) -> None:
    validator = load_validator_module()
    tex_path = tmp_path / "main.tex"
    bib_path = tmp_path / "references.bib"
    tex_path.write_text(r"\citep{alpha}" "\n", encoding="utf-8")
    bib_path.write_text(
        "@misc{alpha,\n  title={Alpha}\n}\n"
        "@article{unused,\n  title={Unused}\n}\n",
        encoding="utf-8",
    )

    with pytest.raises(AssertionError, match="entries not cited"):
        validator.assert_bibtex_citations_are_defined(tex_path, bib_path)


def valid_manifest() -> dict:
    return {
        "validation_command": "python3 scripts/e11_validate_outputs.py",
        "artifact_dirs": [{"path": item["path"]} for item in ARTIFACT_DIRS],
        "key_tables": [{"path": path, "rows": 1} for path in KEY_TABLES],
        "key_documents": [{"path": path} for path in KEY_DOCUMENTS],
        "zero_row_allowed_tables": list(ZERO_ROW_ALLOWED_TABLES),
        "ignore_policy": [{"path_or_pattern": item["path_or_pattern"]} for item in IGNORE_POLICY],
    }


def valid_condition_score_protocol() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scores = pd.DataFrame(
        [
            {
                "score_id": "source_observed_drift_positive_control",
                "role": "positive_control",
                "uses_observed_source_drift": "yes",
                "fit_rule": "No fitting.",
                "input_features": "source observed drift",
                "heldout_claim_allowed": "no",
                "reason": "Positive control only.",
            },
            {
                "score_id": "early_layer_prior",
                "role": "architecture_prior_baseline",
                "uses_observed_source_drift": "no",
                "fit_rule": "No fitting.",
                "input_features": "layer_index",
                "heldout_claim_allowed": "baseline only",
                "reason": "Architecture baseline.",
            },
            {
                "score_id": "legacy_scaled_jvp_ratio",
                "role": "locked_boundary_baseline",
                "uses_observed_source_drift": "no",
                "fit_rule": "No fitting.",
                "input_features": "scaled JVP",
                "heldout_claim_allowed": "boundary only",
                "reason": "Boundary baseline.",
            },
            {
                "score_id": "condition_score_v2_calibrated_residual",
                "role": "primary_candidate",
                "uses_observed_source_drift": "calibration only",
                "fit_rule": "Fit on calibration splits only and freeze coefficients before held-out evaluation.",
                "input_features": "source-only JVP and rank features",
                "heldout_claim_allowed": "yes, if all primary gates pass",
                "reason": "Primary candidate.",
            },
            {
                "score_id": "theory_sign_composite",
                "role": "secondary_zero_fit_candidate",
                "uses_observed_source_drift": "no",
                "fit_rule": "No fitting.",
                "input_features": "rank and scaled JVP",
                "heldout_claim_allowed": "secondary only",
                "reason": "Theory-adjacent candidate.",
            },
        ]
    )
    splits = pd.DataFrame(
        [
            {
                "split_id": "calibration",
                "role": "calibration_only",
                "dataset": "CIFAR-100-LT",
                "architecture": "ResNet18",
                "checkpoints": "source folds",
                "seeds": "10",
                "score_tuning_allowed": "yes",
                "target_used_for_tuning": "no",
                "compute_mode": "GPU via Slurm",
                "planned_artifact_prefix": "results/example/calibration",
            },
            {
                "split_id": "heldout_checkpoint",
                "role": "primary_heldout_checkpoint",
                "dataset": "CIFAR-100-LT",
                "architecture": "ResNet18",
                "checkpoints": "held-out",
                "seeds": "10",
                "score_tuning_allowed": "no",
                "target_used_for_tuning": "no",
                "compute_mode": "GPU via Slurm",
                "planned_artifact_prefix": "results/example/checkpoint",
            },
            {
                "split_id": "heldout_architecture",
                "role": "primary_heldout_architecture",
                "dataset": "CIFAR-100-LT",
                "architecture": "ResNet34",
                "checkpoints": "held-out",
                "seeds": "5",
                "score_tuning_allowed": "no",
                "target_used_for_tuning": "no",
                "compute_mode": "GPU via Slurm",
                "planned_artifact_prefix": "results/example/architecture",
            },
            {
                "split_id": "heldout_data",
                "role": "primary_heldout_data",
                "dataset": "CIFAR-10-LT",
                "architecture": "ResNet18",
                "checkpoints": "held-out",
                "seeds": "10",
                "score_tuning_allowed": "no",
                "target_used_for_tuning": "no",
                "compute_mode": "GPU via Slurm",
                "planned_artifact_prefix": "results/example/data",
            },
        ]
    )
    gates = pd.DataFrame(
        [
            {
                "gate_id": "G1-no-target-leakage",
                "scope": "all",
                "requirement": "No target leakage.",
                "pass_condition": "Protocol and coefficients are committed first.",
            },
            {
                "gate_id": "G2-primary-residual-prediction",
                "scope": "P0",
                "requirement": "Predict residual risk.",
                "pass_condition": "Held-out residual Spearman CI lower endpoint is above 0.",
            },
            {
                "gate_id": "G3-threshold-direction",
                "scope": "direction",
                "requirement": "Preserve threshold.",
                "pass_condition": "Below-one threshold accuracy is at least 0.8.",
            },
            {
                "gate_id": "G4-baseline-comparison",
                "scope": "baseline",
                "requirement": "Compare baselines.",
                "pass_condition": "Report early-layer and positive-control baselines.",
            },
            {
                "gate_id": "G5-no-performance-overclaim",
                "scope": "wording",
                "requirement": "No performance claim.",
                "pass_condition": "Keep claim local unless a separate benchmark passes.",
            },
            {
                "gate_id": "G6-reporting-completeness",
                "scope": "artifact review",
                "requirement": "Report all fields.",
                "pass_condition": "Generated artifacts exist and make e11-check validates them.",
            },
        ]
    )
    return scores, splits, gates


def test_condition_score_protocol_schema_accepts_required_entries() -> None:
    validator = load_validator_module()
    scores, splits, gates = valid_condition_score_protocol()

    validator.assert_condition_score_protocol(scores, splits, gates)


def test_condition_score_protocol_schema_rejects_heldout_tuning() -> None:
    validator = load_validator_module()
    scores, splits, gates = valid_condition_score_protocol()
    splits.loc[splits["role"].eq("primary_heldout_data"), "score_tuning_allowed"] = "yes"

    with pytest.raises(AssertionError, match="held-out splits must forbid"):
        validator.assert_condition_score_protocol(scores, splits, gates)


def valid_top_conference_gap_register() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "gap_id": "P0-PredictiveCondition",
                "priority": "P0",
                "claim_unblocked": "Predictive condition claim.",
                "current_state": "Current score is a boundary result.",
                "required_next_evidence": "Run held-out prediction.",
                "acceptance_gate": "Report positive held-out residual prediction with confidence intervals and no target leakage.",
                "compute_mode": "GPU via Slurm",
                "planned_artifacts": "results/example/*; discussion/example.md",
                "risk_if_missing": "Condition claim remains unsupported.",
            },
            {
                "gap_id": "P0-StandardBenchmark",
                "priority": "P0",
                "claim_unblocked": "Performance claim.",
                "current_state": "Only pilots exist.",
                "required_next_evidence": "Run tuned baselines.",
                "acceptance_gate": "Report confidence intervals, paired seeds, and tuned baselines before any performance claim.",
                "compute_mode": "GPU via Slurm",
                "planned_artifacts": "results/example/*; discussion/example.md",
                "risk_if_missing": "No performance claim.",
            },
            {
                "gap_id": "P1-HeldOutGenerality",
                "priority": "P1",
                "claim_unblocked": "Generality claim.",
                "current_state": "One architecture family is strongest.",
                "required_next_evidence": "Run held-out architecture.",
                "acceptance_gate": "Report seed count, checkpoint quality, and drift-ratio CI before broadening the claim.",
                "compute_mode": "GPU via Slurm",
                "planned_artifacts": "results/example/*; discussion/example.md",
                "risk_if_missing": "Generality remains limited.",
            },
            {
                "gap_id": "P1-TheoryToScore",
                "priority": "P1",
                "claim_unblocked": "Theory-score bridge.",
                "current_state": "The theorem and real score are adjacent.",
                "required_next_evidence": "Add theory bridge.",
                "acceptance_gate": "Add a falsifiable score statement, a score-term ablation, and a documented failure mode before treating the score as predictive.",
                "compute_mode": "CPU plus optional GPU via Slurm",
                "planned_artifacts": "results/example/*; discussion/example.md",
                "risk_if_missing": "Mechanism looks loose.",
            },
            {
                "gap_id": "P1-PracticalMuonBridge",
                "priority": "P1",
                "claim_unblocked": "Practical Muon discussion.",
                "current_state": "Final-training pilot is negative.",
                "required_next_evidence": "Tune schedules.",
                "acceptance_gate": "Report final metrics and trajectory-local diagnostics with confidence intervals.",
                "compute_mode": "GPU via Slurm",
                "planned_artifacts": "results/example/*; discussion/example.md",
                "risk_if_missing": "Muon remains motivation only.",
            },
            {
                "gap_id": "P2-NaturalBoundaryCases",
                "priority": "P2",
                "claim_unblocked": "Natural boundary claim.",
                "current_state": "Synthetic negative exists.",
                "required_next_evidence": "Search natural settings.",
                "acceptance_gate": "Report drift, loss, margin, rank, JVP, checkpoint-quality metrics, and the pre-registered stopping rule.",
                "compute_mode": "CPU plus optional GPU via Slurm",
                "planned_artifacts": "results/example/*; discussion/example.md",
                "risk_if_missing": "Falsifiability concern remains.",
            },
            {
                "gap_id": "P2-PackagingRepro",
                "priority": "P2",
                "claim_unblocked": "Artifact-review readiness.",
                "current_state": "Tectonic build works and discussion/e11_artifact_review_packet.md documents the serverREADME.md exclusion and GPU-pending boundary.",
                "required_next_evidence": "Run clean checkout after refreshing discussion/e11_artifact_review_packet.md.",
                "acceptance_gate": "A fresh checkout completes make e11-check, builds both PDFs, and records the exact command environment.",
                "compute_mode": "CPU",
                "planned_artifacts": "results/example/*; discussion/example.md; results/e11_artifact_review_packet/*; discussion/e11_artifact_review_packet.md",
                "risk_if_missing": "Packaging risk remains.",
            },
        ]
    )


def test_top_conference_gap_register_schema_accepts_required_entries() -> None:
    validator = load_validator_module()

    validator.assert_top_conference_gap_register(valid_top_conference_gap_register())


def test_top_conference_gap_register_schema_rejects_missing_p0_gate() -> None:
    validator = load_validator_module()
    register = valid_top_conference_gap_register()
    register = register[~register["gap_id"].eq("P0-StandardBenchmark")]

    with pytest.raises(AssertionError, match="missing gap ids"):
        validator.assert_top_conference_gap_register(register)


def test_artifact_manifest_schema_accepts_required_entries() -> None:
    validator = load_validator_module()

    validator.assert_valid_artifact_manifest(valid_manifest())


def test_artifact_manifest_schema_rejects_missing_key_table() -> None:
    validator = load_validator_module()
    manifest = valid_manifest()
    manifest["key_tables"] = manifest["key_tables"][:-1]

    with pytest.raises(AssertionError, match="missing key tables"):
        validator.assert_valid_artifact_manifest(manifest)


def test_artifact_manifest_schema_rejects_missing_zero_row_policy() -> None:
    validator = load_validator_module()
    manifest = valid_manifest()
    manifest["zero_row_allowed_tables"] = []

    with pytest.raises(AssertionError, match="missing zero-row allowed table policy"):
        validator.assert_valid_artifact_manifest(manifest)


def test_artifact_manifest_schema_rejects_nonpositive_rows() -> None:
    validator = load_validator_module()
    manifest = valid_manifest()
    manifest["key_tables"][0]["rows"] = 0

    with pytest.raises(AssertionError, match="non-positive row counts"):
        validator.assert_valid_artifact_manifest(manifest)


def test_artifact_manifest_schema_allows_registered_zero_row_tables() -> None:
    validator = load_validator_module()
    manifest = valid_manifest()
    by_path = {item["path"]: item for item in manifest["key_tables"]}
    by_path[ZERO_ROW_ALLOWED_TABLES[-1]]["rows"] = 0

    validator.assert_valid_artifact_manifest(manifest)


def test_artifact_registry_has_unique_entries() -> None:
    artifact_paths = [item["path"] for item in ARTIFACT_DIRS]
    ignored_paths = [item["path_or_pattern"] for item in IGNORE_POLICY]

    assert len(artifact_paths) == len(set(artifact_paths))
    assert len(KEY_TABLES) == len(set(KEY_TABLES))
    assert len(KEY_DOCUMENTS) == len(set(KEY_DOCUMENTS))
    assert len(ignored_paths) == len(set(ignored_paths))
    assert len(MAIN_RESULT_SCRIPTS) == len(set(MAIN_RESULT_SCRIPTS))
    assert len(APPENDIX_RUNNER_SCRIPTS) == len(set(APPENDIX_RUNNER_SCRIPTS))
    assert len(PAPER_ASSET_SCRIPTS) == len(set(PAPER_ASSET_SCRIPTS))
    assert len(LEGACY_GUARDRAIL_SCRIPTS) == len(set(LEGACY_GUARDRAIL_SCRIPTS))


def test_artifact_registry_paths_exist() -> None:
    root = Path(__file__).resolve().parents[1]
    script_paths = [
        *MAIN_RESULT_SCRIPTS,
        *APPENDIX_RUNNER_SCRIPTS,
        *PAPER_ASSET_SCRIPTS,
        *LEGACY_GUARDRAIL_SCRIPTS,
    ]

    assert all((root / script).is_file() for script in script_paths)
    assert all((root / table).is_file() for table in KEY_TABLES)
    assert all(table.endswith(".csv") for table in KEY_TABLES)


def test_paper_asset_registry_contains_only_e11_scripts() -> None:
    all_registered_scripts = {
        *MAIN_RESULT_SCRIPTS,
        *APPENDIX_RUNNER_SCRIPTS,
        *PAPER_ASSET_SCRIPTS,
        *LEGACY_GUARDRAIL_SCRIPTS,
    }

    assert all(script.startswith("scripts/e11_") and script.endswith(".py") for script in all_registered_scripts)
    assert all(script.startswith("scripts/e11_write_") for script in PAPER_ASSET_SCRIPTS)
    assert all(script.startswith("scripts/e11_write_") for script in LEGACY_GUARDRAIL_SCRIPTS)
    assert set(PAPER_ASSET_SCRIPTS).isdisjoint(LEGACY_GUARDRAIL_SCRIPTS)


def test_appendix_runner_registry_has_descriptions() -> None:
    assert set(APPENDIX_RUNNER_DESCRIPTIONS) == set(APPENDIX_RUNNER_SCRIPTS)
    assert all(APPENDIX_RUNNER_DESCRIPTIONS[script].strip() for script in APPENDIX_RUNNER_SCRIPTS)


def test_main_evidence_registry_covers_main_result_scripts() -> None:
    main_commands = [item["command"] for item in MAIN_EVIDENCE_STAGES]
    stage_scripts = [
        script
        for item in MAIN_EVIDENCE_STAGES
        for script in [script_identity_from_command(item["command"])]
        if script is not None and script in MAIN_RESULT_SCRIPTS
    ]

    assert list(MAIN_RESULT_SCRIPTS) == dedupe_preserving_order(stage_scripts)
    assert any("scripts/e11_write_mechanism_boundary.py" in command for command in main_commands)
    assert all(item["stage"].strip() and item["produces"].strip() for item in MAIN_EVIDENCE_STAGES)


def test_makefile_result_targets_cover_registry_scripts() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    start = makefile.index("e11-main-results:")
    end = makefile.index("\ne11-appendix-results:", start)
    result_target_block = makefile[start:end]
    commands = [
        line.strip()
        for line in result_target_block.splitlines()
        if line.strip().startswith(("$(PYTHON) ", "sbatch "))
    ]
    scripts = [
        script
        for command in commands
        for script in [script_identity_from_command(command)]
        if script is not None
    ]

    assert set(scripts) == set(MAIN_RESULT_SCRIPTS)
    assert any(command.startswith("sbatch ") for command in commands)


def test_makefile_appendix_target_matches_registry_order() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    start = makefile.index("e11-appendix-results:")
    end = makefile.index("\ne11-all-results:", start)
    target_block = makefile[start:end]
    commands = [
        line.strip().removeprefix("$(PYTHON) ")
        for line in target_block.splitlines()
        if line.strip().startswith("$(PYTHON) ")
    ]

    assert commands == list(APPENDIX_RUNNER_SCRIPTS)


def test_makefile_paper_pdf_target_is_part_of_full_gate() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")

    assert "e11-paper-pdf:" in makefile
    assert "$(MAKE) -C paper/specgrad_activation_paper" in makefile
    assert "e11-full: e11-paper-assets e11-paper-pdf e11-check" in makefile
    assert "e11-guardrail-assets:" in makefile
    assert "$(PYTHON) scripts/e11_write_legacy_guardrail_artifacts.py" in makefile
    paper_makefile = (ROOT / "paper" / "specgrad_activation_paper" / "Makefile").read_text(encoding="utf-8")
    assert "tectonic: main-tectonic two-page-tectonic" in paper_makefile
    assert "$(TECTONIC) main.tex" in paper_makefile
    assert "$(TECTONIC) two_page.tex" in paper_makefile


def test_readme_command_blocks_match_registry_order() -> None:
    readme = (ROOT / "README_E11.md").read_text(encoding="utf-8")

    core_match = re.search(r"Run the core experiment:\n\n```bash\n(?P<commands>.*?)\n```", readme, re.S)
    appendix_match = re.search(
        r"Run appendix and guardrail follow-up experiments:\n\n```bash\n(?P<commands>.*?)\n```",
        readme,
        re.S,
    )
    assert core_match is not None
    assert appendix_match is not None

    def parse_main_scripts(block: str) -> list[str]:
        scripts = [
            script
            for line in block.splitlines()
            for script in [script_identity_from_command(line)]
            if script is not None
        ]
        return dedupe_preserving_order(scripts)

    def parse_appendix_commands(block: str) -> list[str]:
        return [line.removeprefix("python3 ") for line in block.splitlines() if line.startswith("python3 scripts/e11_")]

    assert parse_main_scripts(core_match.group("commands")) == list(MAIN_RESULT_SCRIPTS)
    assert parse_appendix_commands(appendix_match.group("commands")) == list(APPENDIX_RUNNER_SCRIPTS)


def test_paper_asset_registry_order_preserves_dependencies() -> None:
    assert PAPER_ASSET_SCRIPTS.index("scripts/e11_write_discussion_outline.py") < PAPER_ASSET_SCRIPTS.index(
        "scripts/e11_write_research_synthesis.py"
    )
    assert PAPER_ASSET_SCRIPTS.index("scripts/e11_write_paper_readiness_audit.py") < PAPER_ASSET_SCRIPTS.index(
        "scripts/e11_write_paper_skeleton.py"
    )
    assert PAPER_ASSET_SCRIPTS[-1] == "scripts/e11_write_artifact_manifest.py"
    assert "scripts/e11_write_mechanism_boundary.py" in LEGACY_GUARDRAIL_SCRIPTS
    assert "scripts/e11_write_optimizer_invariance_audit.py" in LEGACY_GUARDRAIL_SCRIPTS
