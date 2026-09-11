import json

import pytest

from labs.week14_release_audit import (
    AuditError,
    REQUIRED_ARTIFACTS,
    SCHEMA_VERSION,
    safe_relative_path,
    sha256_file,
    validate_manifest,
)


def complete_manifest(project_root):
    artifacts = {}
    for name in REQUIRED_ARTIFACTS:
        path = project_root / f"{name}.txt"
        path.write_text(f"{name} artifact\n", encoding="utf-8")
        artifacts[name] = {
            "path": path.name,
            "sha256": sha256_file(path),
        }
    return {
        "schema": SCHEMA_VERSION,
        "release": {
            "repository": "https://example.org/research/project",
            "commit": "0123456789abcdef0123456789abcdef01234567",
            "version": "final",
            "license": "LICENSE.md",
            "citation": "CITATION.cff",
            "privacy_review": "Completed before release.",
            "known_limitations": ["One site and five training seeds."],
            "research_direction": "research-direction.md",
        },
        "data": {
            "source": "Generated fixture.",
            "version": "v1",
            "access": "synthetic",
            "license": "MIT",
            "split_identity": "seed 705",
            "access_instructions": "Run the generator.",
        },
        "reproduction": {
            "command": "python experiment.py --config final.json",
            "expected_result": "paired difference remains positive",
            "agreement_criterion": "all paired differences are positive",
            "runtime": "under one minute",
            "device": "CPU",
        },
        "artifacts": artifacts,
    }


def test_week14_release_audit_verifies_files_and_detects_tampering(tmp_path):
    manifest = complete_manifest(tmp_path)
    report = validate_manifest(tmp_path, manifest)

    assert report["status"] == "PASS"
    assert report["artifact_count"] == len(REQUIRED_ARTIFACTS)
    assert report["commands_executed"] is False

    result_path = tmp_path / manifest["artifacts"]["result"]["path"]
    result_path.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(AuditError, match="Artifact hash mismatch"):
        validate_manifest(tmp_path, manifest)


def test_week14_release_audit_rejects_unsafe_paths_and_credentials(tmp_path):
    with pytest.raises(AuditError, match="Unsafe artifact path"):
        safe_relative_path(tmp_path, "../private.txt")

    manifest = complete_manifest(tmp_path)
    manifest["reproduction"]["command"] = (
        "python run.py --token=ghp_123456789012345678901234567890"
    )
    with pytest.raises(AuditError, match="credential"):
        validate_manifest(tmp_path, json.loads(json.dumps(manifest)))
