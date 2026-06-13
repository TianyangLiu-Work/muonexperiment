import importlib.util
import re
from pathlib import Path

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
)


def load_validator_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "e11_validate_outputs.py"
    spec = importlib.util.spec_from_file_location("e11_validate_outputs", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


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
        "ignore_policy": [{"path_or_pattern": item["path_or_pattern"]} for item in IGNORE_POLICY],
    }


def test_artifact_manifest_schema_accepts_required_entries() -> None:
    validator = load_validator_module()

    validator.assert_valid_artifact_manifest(valid_manifest())


def test_artifact_manifest_schema_rejects_missing_key_table() -> None:
    validator = load_validator_module()
    manifest = valid_manifest()
    manifest["key_tables"] = manifest["key_tables"][:-1]

    with pytest.raises(AssertionError, match="missing key tables"):
        validator.assert_valid_artifact_manifest(manifest)


def test_artifact_manifest_schema_rejects_nonpositive_rows() -> None:
    validator = load_validator_module()
    manifest = valid_manifest()
    manifest["key_tables"][0]["rows"] = 0

    with pytest.raises(AssertionError, match="non-positive row counts"):
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
    main_commands = [item["command"].removeprefix("python3 ") for item in MAIN_EVIDENCE_STAGES]

    assert list(MAIN_RESULT_SCRIPTS) == main_commands[: len(MAIN_RESULT_SCRIPTS)]
    assert "scripts/e11_write_mechanism_boundary.py" in main_commands
    assert all(item["stage"].strip() and item["produces"].strip() for item in MAIN_EVIDENCE_STAGES)


def test_makefile_main_target_matches_registry_order() -> None:
    makefile = (Path(__file__).resolve().parents[1] / "Makefile").read_text(encoding="utf-8")
    start = makefile.index("e11-main-results:")
    end = makefile.index("\ne11-appendix-results:", start)
    target_block = makefile[start:end]
    commands = [
        line.strip().removeprefix("$(PYTHON) ")
        for line in target_block.splitlines()
        if line.strip().startswith("$(PYTHON) ")
    ]

    assert commands == list(MAIN_RESULT_SCRIPTS)


def test_makefile_appendix_target_matches_registry_order() -> None:
    makefile = (Path(__file__).resolve().parents[1] / "Makefile").read_text(encoding="utf-8")
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
    makefile = (Path(__file__).resolve().parents[1] / "Makefile").read_text(encoding="utf-8")

    assert "e11-paper-pdf:" in makefile
    assert "$(MAKE) -C paper/specgrad_activation_paper" in makefile
    assert "e11-full: e11-paper-assets e11-paper-pdf e11-check" in makefile
    assert "e11-guardrail-assets:" in makefile
    assert "$(PYTHON) scripts/e11_write_legacy_guardrail_artifacts.py" in makefile


def test_readme_command_blocks_match_registry_order() -> None:
    readme = (Path(__file__).resolve().parents[1] / "README_E11.md").read_text(encoding="utf-8")

    core_match = re.search(r"Run the core experiment:\n\n```bash\n(?P<commands>.*?)\n```", readme, re.S)
    appendix_match = re.search(
        r"Run appendix and guardrail follow-up experiments:\n\n```bash\n(?P<commands>.*?)\n```",
        readme,
        re.S,
    )
    assert core_match is not None
    assert appendix_match is not None

    def parse_commands(block: str) -> list[str]:
        return [line.removeprefix("python3 ") for line in block.splitlines() if line.startswith("python3 scripts/e11_")]

    assert parse_commands(core_match.group("commands")) == list(MAIN_RESULT_SCRIPTS)
    assert parse_commands(appendix_match.group("commands")) == list(APPENDIX_RUNNER_SCRIPTS)


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
