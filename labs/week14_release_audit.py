"""Static final-release audit for CME705 Week 14.

The audit validates declarations, safe relative paths, artifact hashes, and
optional Git identity. It never executes the reproduction command contained in
a student manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA_VERSION = "cme705-final-release-v1"
REQUIRED_ARTIFACTS = (
    "report",
    "presentation",
    "environment",
    "configuration",
    "result",
)
HEX_DIGEST = re.compile(r"[0-9a-fA-F]{64}")
COMMIT_ID = re.compile(r"[0-9a-fA-F]{7,40}")
SENSITIVE_VALUE_PATTERNS = (
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._~-]{16,}"),
)


class AuditError(RuntimeError):
    """Raised when the declared final release fails a static audit."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AuditError(f"{name} must be a JSON object.")
    return value


def require_text(mapping: dict[str, Any], key: str, section: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise AuditError(f"{section}.{key} must be non-empty text.")
    return value.strip()


def safe_relative_path(project_root: Path, value: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise AuditError("Artifact path must be non-empty text.")
    if "\\" in value or ":" in value or value.startswith(("~", "/")):
        raise AuditError(f"Artifact path must be portable and relative: {value}")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or "." == value:
        raise AuditError(f"Unsafe artifact path: {value}")
    root = project_root.resolve()
    candidate = (root / Path(*pure.parts)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise AuditError(f"Artifact leaves the project root: {value}") from error
    return candidate


def assert_no_sensitive_values(value: Any) -> None:
    serialized = json.dumps(value, ensure_ascii=False)
    for pattern in SENSITIVE_VALUE_PATTERNS:
        if pattern.search(serialized):
            raise AuditError("Manifest appears to contain a credential or private key.")


def validate_git(project_root: Path, declared_commit: str) -> dict[str, Any]:
    try:
        head = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip().lower()
        status = subprocess.run(
            ["git", "-C", str(project_root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise AuditError("Git identity could not be verified.") from error
    if not head.startswith(declared_commit.lower()):
        raise AuditError("Declared commit does not match the project HEAD.")
    if status.strip():
        raise AuditError("Working tree is not clean.")
    return {"head": head, "clean": True}


def validate_manifest(
    project_root: Path,
    manifest: dict[str, Any],
    *,
    check_files: bool = True,
    verify_git: bool = False,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    assert_no_sensitive_values(manifest)
    if manifest.get("schema") != SCHEMA_VERSION:
        raise AuditError(f"schema must be {SCHEMA_VERSION!r}.")

    release = require_mapping(manifest.get("release"), "release")
    data = require_mapping(manifest.get("data"), "data")
    reproduction = require_mapping(manifest.get("reproduction"), "reproduction")
    artifacts = require_mapping(manifest.get("artifacts"), "artifacts")

    repository = require_text(release, "repository", "release")
    commit = require_text(release, "commit", "release").lower()
    if not COMMIT_ID.fullmatch(commit):
        raise AuditError("release.commit must contain 7–40 hexadecimal characters.")
    require_text(release, "version", "release")
    require_text(release, "license", "release")
    require_text(release, "citation", "release")
    require_text(release, "privacy_review", "release")
    require_text(release, "research_direction", "release")
    limitations = release.get("known_limitations")
    if not isinstance(limitations, list) or not limitations or not all(
        isinstance(item, str) and item.strip() for item in limitations
    ):
        raise AuditError("release.known_limitations must contain non-empty text.")

    access = require_text(data, "access", "data").lower()
    if access not in {"public", "restricted", "synthetic"}:
        raise AuditError("data.access must be public, restricted, or synthetic.")
    for key in ("source", "version", "license", "split_identity"):
        require_text(data, key, "data")
    if access == "restricted":
        require_text(data, "access_instructions", "data")

    for key in (
        "command",
        "expected_result",
        "agreement_criterion",
        "runtime",
        "device",
    ):
        require_text(reproduction, key, "reproduction")

    checked: list[dict[str, str]] = []
    for name in REQUIRED_ARTIFACTS:
        record = require_mapping(artifacts.get(name), f"artifacts.{name}")
        relative = require_text(record, "path", f"artifacts.{name}")
        declared_hash = require_text(record, "sha256", f"artifacts.{name}").lower()
        if not HEX_DIGEST.fullmatch(declared_hash):
            raise AuditError(f"artifacts.{name}.sha256 must be a SHA-256 digest.")
        path = safe_relative_path(project_root, relative)
        if check_files:
            if not path.is_file():
                raise AuditError(f"Declared artifact is missing: {relative}")
            actual_hash = sha256_file(path)
            if actual_hash != declared_hash:
                raise AuditError(f"Artifact hash mismatch: {relative}")
        checked.append({"name": name, "path": relative, "sha256": declared_hash})

    git_result = None
    if verify_git:
        git_result = validate_git(project_root, commit)

    return {
        "status": "PASS",
        "schema": SCHEMA_VERSION,
        "repository": repository,
        "commit": commit,
        "artifact_count": len(checked),
        "data_access": access,
        "git": git_result,
        "commands_executed": False,
    }


def teaching_manifest() -> dict[str, Any]:
    artifact = {"path": "replace/file.txt", "sha256": "0" * 64}
    return {
        "schema": SCHEMA_VERSION,
        "release": {
            "repository": "https://example.org/research/project",
            "commit": "0123456789abcdef0123456789abcdef01234567",
            "version": "final-course-submission",
            "license": "LICENSE.md",
            "citation": "CITATION.cff",
            "privacy_review": "Completed before release.",
            "known_limitations": ["One site and five training seeds."],
            "research_direction": "research-direction.md",
        },
        "data": {
            "source": "Generated teaching procedure.",
            "version": "v1",
            "access": "synthetic",
            "license": "Course code license.",
            "split_identity": "Deterministic seed 705.",
            "access_instructions": "Run the declared generator.",
        },
        "reproduction": {
            "command": "python experiment.py --config final.json",
            "expected_result": "Mean paired difference is positive.",
            "agreement_criterion": "All paired differences remain positive.",
            "runtime": "Under one minute.",
            "device": "CPU with NumPy.",
        },
        "artifacts": {name: dict(artifact) for name in REQUIRED_ARTIFACTS},
    }


def audit_only() -> None:
    report = validate_manifest(
        Path("."),
        teaching_manifest(),
        check_files=False,
        verify_git=False,
    )
    unsafe_rejected = False
    try:
        safe_relative_path(Path("."), "../private.txt")
    except AuditError:
        unsafe_rejected = True

    credential_rejected = False
    unsafe = teaching_manifest()
    unsafe["reproduction"]["command"] = (
        "python run.py --token=ghp_123456789012345678901234567890"
    )
    try:
        validate_manifest(Path("."), unsafe, check_files=False)
    except AuditError:
        credential_rejected = True

    print("week14_final_release_audit")
    print(f"schema_valid={report['status'] == 'PASS'}")
    print(f"required_artifacts={report['artifact_count']}")
    print(f"path_traversal_rejected={unsafe_rejected}")
    print(f"credential_pattern_rejected={credential_rejected}")
    print("commands_executed=False")
    print("files_written=False")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--manifest", type=Path, default=Path("project-release.json"))
    parser.add_argument("--verify-git", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.audit_only:
        audit_only()
        return

    project_root = args.project_root.resolve()
    manifest_path = args.manifest
    if not manifest_path.is_absolute():
        manifest_path = project_root / manifest_path
    manifest_path = manifest_path.resolve()
    try:
        manifest_path.relative_to(project_root)
    except ValueError as error:
        raise AuditError("Manifest must be inside the project root.") from error
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report = validate_manifest(
        project_root,
        require_mapping(manifest, "manifest"),
        check_files=True,
        verify_git=args.verify_git,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
