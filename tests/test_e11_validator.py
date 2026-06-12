import importlib.util
import re
from pathlib import Path

import pytest

from e11_condition_geometry.artifacts import (
    APPENDIX_RUNNER_DESCRIPTIONS,
    APPENDIX_RUNNER_SCRIPTS,
    ARTIFACT_DIRS,
    IGNORE_POLICY,
    KEY_TABLES,
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


def valid_manifest() -> dict:
    return {
        "validation_command": "python3 scripts/e11_validate_outputs.py",
        "artifact_dirs": [{"path": item["path"]} for item in ARTIFACT_DIRS],
        "key_tables": [{"path": path, "rows": 1} for path in KEY_TABLES],
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
    assert len(ignored_paths) == len(set(ignored_paths))
    assert len(MAIN_RESULT_SCRIPTS) == len(set(MAIN_RESULT_SCRIPTS))
    assert len(APPENDIX_RUNNER_SCRIPTS) == len(set(APPENDIX_RUNNER_SCRIPTS))
    assert len(PAPER_ASSET_SCRIPTS) == len(set(PAPER_ASSET_SCRIPTS))


def test_artifact_registry_paths_exist() -> None:
    root = Path(__file__).resolve().parents[1]
    script_paths = [*MAIN_RESULT_SCRIPTS, *APPENDIX_RUNNER_SCRIPTS, *PAPER_ASSET_SCRIPTS]

    assert all((root / script).is_file() for script in script_paths)
    assert all((root / table).is_file() for table in KEY_TABLES)
    assert all(table.endswith(".csv") for table in KEY_TABLES)


def test_paper_asset_registry_contains_only_e11_scripts() -> None:
    all_registered_scripts = {*MAIN_RESULT_SCRIPTS, *APPENDIX_RUNNER_SCRIPTS, *PAPER_ASSET_SCRIPTS}

    assert all(script.startswith("scripts/e11_") and script.endswith(".py") for script in all_registered_scripts)
    assert all(script.startswith("scripts/e11_write_") or script in APPENDIX_RUNNER_SCRIPTS for script in PAPER_ASSET_SCRIPTS)


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


def test_readme_command_blocks_match_registry_order() -> None:
    readme = (Path(__file__).resolve().parents[1] / "README_E11.md").read_text(encoding="utf-8")

    core_match = re.search(r"Run the core experiment:\n\n```bash\n(?P<commands>.*?)\n```", readme, re.S)
    appendix_match = re.search(
        r"Run the current paper-supporting follow-up experiments:\n\n```bash\n(?P<commands>.*?)\n```",
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
    assert PAPER_ASSET_SCRIPTS.index("scripts/e11_write_discussion.py") < PAPER_ASSET_SCRIPTS.index(
        "scripts/e11_write_research_synthesis.py"
    )
    assert PAPER_ASSET_SCRIPTS.index("scripts/e11_run_boundary_predictor.py") < PAPER_ASSET_SCRIPTS.index(
        "scripts/e11_write_boundary_predictor_audit.py"
    )
    assert PAPER_ASSET_SCRIPTS.index("scripts/e11_write_paper_readiness_audit.py") < PAPER_ASSET_SCRIPTS.index(
        "scripts/e11_write_paper_skeleton.py"
    )
    assert PAPER_ASSET_SCRIPTS[-1] == "scripts/e11_write_artifact_manifest.py"
