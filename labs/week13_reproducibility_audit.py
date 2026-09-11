"""Reproducible experiment bundle and independent verification for Week 13.

The script runs one controlled NumPy comparison on generated data, records every
result needed to support the claim, writes cryptographic file hashes, and then
repeats the experiment from the saved configuration. It uses no external data
and never treats a matching hash as evidence that the scientific claim is valid.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SOURCE_PATH = Path("labs/week13_reproducibility_audit.py")
SCHEMA_VERSION = "cme705-week13-v1"


@dataclass(frozen=True)
class ExperimentConfig:
    data_seed: int = 1307
    split_seed: int = 1313
    training_seeds: tuple[int, ...] = (13, 29, 47, 71, 101)
    sample_count: int = 900
    validation_fraction: float = 0.2
    test_fraction: float = 0.2
    learning_rate: float = 0.08
    epochs: int = 120
    batch_size: int = 64
    l2: float = 0.002


@dataclass(frozen=True)
class Split:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


@dataclass(frozen=True)
class RunRecord:
    seed: int
    condition: str
    best_epoch: int
    validation_loss: float
    test_loss: float
    test_accuracy: float
    seconds: float


@dataclass(frozen=True)
class LogisticModel:
    weights: np.ndarray
    bias: float


def canonical_json(value: Any) -> bytes:
    """Serialize a JSON value deterministically for hashing."""

    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def sigmoid(logit: np.ndarray) -> np.ndarray:
    positive = logit >= 0
    probability = np.empty_like(logit, dtype=float)
    probability[positive] = 1.0 / (1.0 + np.exp(-logit[positive]))
    exp_logit = np.exp(logit[~positive])
    probability[~positive] = exp_logit / (1.0 + exp_logit)
    return probability


def binary_cross_entropy(
    features: np.ndarray,
    target: np.ndarray,
    model: LogisticModel,
    l2: float = 0.0,
) -> float:
    logit = features @ model.weights + model.bias
    data_loss = np.mean(np.logaddexp(0.0, logit) - target * logit)
    return float(data_loss + 0.5 * l2 * np.sum(model.weights**2))


def make_research_data(config: ExperimentConfig) -> tuple[np.ndarray, np.ndarray]:
    """Create a fixed nonlinear binary-classification population."""

    rng = np.random.default_rng(config.data_seed)
    features = rng.normal(size=(config.sample_count, 3))
    logit = (
        0.45 * features[:, 0]
        - 0.35 * features[:, 1]
        + 0.25 * features[:, 2]
        + 1.35 * features[:, 0] * features[:, 1]
        + rng.normal(0.0, 0.45, size=config.sample_count)
    )
    probability = sigmoid(logit)
    target = rng.binomial(1, probability).astype(float)
    return features, target


def stratified_split(target: np.ndarray, config: ExperimentConfig) -> Split:
    """Create deterministic disjoint train, validation, and test partitions."""

    rng = np.random.default_rng(config.split_seed)
    train_parts: list[np.ndarray] = []
    validation_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    for label in (0.0, 1.0):
        indices = np.flatnonzero(target == label)
        rng.shuffle(indices)
        test_size = round(len(indices) * config.test_fraction)
        validation_size = round(len(indices) * config.validation_fraction)
        test_parts.append(indices[:test_size])
        validation_parts.append(indices[test_size : test_size + validation_size])
        train_parts.append(indices[test_size + validation_size :])

    def combine(parts: list[np.ndarray]) -> np.ndarray:
        joined = np.concatenate(parts)
        rng.shuffle(joined)
        return joined

    split = Split(combine(train_parts), combine(validation_parts), combine(test_parts))
    sets = [set(part.tolist()) for part in (split.train, split.validation, split.test)]
    if not (
        sets[0].isdisjoint(sets[1])
        and sets[0].isdisjoint(sets[2])
        and sets[1].isdisjoint(sets[2])
    ):
        raise RuntimeError("Split partitions overlap.")
    return split


def design_matrix(features: np.ndarray, condition: str) -> np.ndarray:
    if condition == "baseline":
        return features.copy()
    if condition == "interaction":
        interaction = (features[:, 0] * features[:, 1]).reshape(-1, 1)
        return np.concatenate([features, interaction], axis=1)
    raise ValueError(f"Unknown condition: {condition}")


def fit_standardizer(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale == 0.0] = 1.0
    return mean, scale


def apply_standardizer(
    features: np.ndarray, mean: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    return (features - mean) / scale


def fit_logistic(
    features: np.ndarray,
    target: np.ndarray,
    split: Split,
    config: ExperimentConfig,
    seed: int,
) -> tuple[LogisticModel, int, float]:
    """Train with mini-batches and restore the minimum-validation-loss state."""

    rng = np.random.default_rng(seed)
    weights = rng.normal(0.0, 0.05, size=features.shape[1])
    bias = 0.0
    best_model = LogisticModel(weights.copy(), bias)
    best_epoch = 0
    best_validation_loss = float("inf")

    for epoch in range(1, config.epochs + 1):
        order = rng.permutation(split.train)
        for start in range(0, len(order), config.batch_size):
            batch = order[start : start + config.batch_size]
            x_batch = features[batch]
            y_batch = target[batch]
            prediction = sigmoid(x_batch @ weights + bias)
            residual = prediction - y_batch
            gradient_weights = x_batch.T @ residual / len(batch) + config.l2 * weights
            gradient_bias = float(residual.mean())
            weights -= config.learning_rate * gradient_weights
            bias -= config.learning_rate * gradient_bias

        model = LogisticModel(weights.copy(), bias)
        validation_loss = binary_cross_entropy(
            features[split.validation], target[split.validation], model, config.l2
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_model = model

    return best_model, best_epoch, best_validation_loss


def run_condition(
    raw_features: np.ndarray,
    target: np.ndarray,
    split: Split,
    config: ExperimentConfig,
    condition: str,
    seed: int,
) -> RunRecord:
    started = time.perf_counter()
    matrix = design_matrix(raw_features, condition)
    mean, scale = fit_standardizer(matrix[split.train])
    standardized = apply_standardizer(matrix, mean, scale)
    model, best_epoch, validation_loss = fit_logistic(
        standardized, target, split, config, seed
    )
    test_x = standardized[split.test]
    test_y = target[split.test]
    test_loss = binary_cross_entropy(test_x, test_y, model)
    test_prediction = (sigmoid(test_x @ model.weights + model.bias) >= 0.5).astype(float)
    test_accuracy = float(np.mean(test_prediction == test_y))
    return RunRecord(
        seed=seed,
        condition=condition,
        best_epoch=best_epoch,
        validation_loss=validation_loss,
        test_loss=test_loss,
        test_accuracy=test_accuracy,
        seconds=time.perf_counter() - started,
    )


def t_critical_975(degrees_of_freedom: int) -> float:
    """Return a two-sided 95% Student-t critical value for common small samples."""

    values = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        11: 2.201,
        12: 2.179,
        13: 2.160,
        14: 2.145,
        15: 2.131,
        16: 2.120,
        17: 2.110,
        18: 2.101,
        19: 2.093,
        20: 2.086,
        24: 2.064,
        29: 2.045,
        30: 2.042,
    }
    if degrees_of_freedom in values:
        return values[degrees_of_freedom]
    if degrees_of_freedom < 1:
        raise ValueError("At least two paired runs are required.")
    return 1.96 if degrees_of_freedom >= 60 else 2.0


def paired_summary(records: Iterable[RunRecord]) -> dict[str, Any]:
    records = list(records)
    by_seed: dict[int, dict[str, RunRecord]] = {}
    for record in records:
        by_seed.setdefault(record.seed, {})[record.condition] = record
    if not by_seed or any(set(pair) != {"baseline", "interaction"} for pair in by_seed.values()):
        raise ValueError("Every seed must contain baseline and interaction records.")
    seeds = sorted(by_seed)
    differences = np.array(
        [
            by_seed[seed]["interaction"].test_accuracy
            - by_seed[seed]["baseline"].test_accuracy
            for seed in seeds
        ]
    )
    mean_difference = float(differences.mean())
    standard_deviation = float(differences.std(ddof=1))
    standard_error = standard_deviation / np.sqrt(len(differences))
    margin = t_critical_975(len(differences) - 1) * standard_error

    def condition_values(name: str) -> np.ndarray:
        return np.array([by_seed[seed][name].test_accuracy for seed in seeds])

    baseline = condition_values("baseline")
    interaction = condition_values("interaction")
    return {
        "seeds": seeds,
        "run_count": len(seeds),
        "baseline_mean_accuracy": float(baseline.mean()),
        "baseline_standard_deviation": float(baseline.std(ddof=1)),
        "interaction_mean_accuracy": float(interaction.mean()),
        "interaction_standard_deviation": float(interaction.std(ddof=1)),
        "paired_accuracy_differences": differences.tolist(),
        "mean_paired_difference": mean_difference,
        "paired_standard_deviation": standard_deviation,
        "paired_95_percent_t_interval": [mean_difference - margin, mean_difference + margin],
    }


def run_experiment(config: ExperimentConfig) -> tuple[list[RunRecord], dict[str, Any], Split]:
    features, target = make_research_data(config)
    split = stratified_split(target, config)
    records: list[RunRecord] = []
    for seed in config.training_seeds:
        for condition in ("baseline", "interaction"):
            records.append(
                run_condition(features, target, split, config, condition, seed)
            )
    summary = paired_summary(records)
    summary["split_sizes"] = {
        "train": len(split.train),
        "validation": len(split.validation),
        "test": len(split.test),
    }
    summary["claim"] = (
        "Adding the prespecified x0*x1 interaction improves locked-test accuracy "
        "under paired training seeds on this generated population."
    )
    return records, summary, split


def git_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return completed.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_json(value))


def records_for_storage(records: Iterable[RunRecord]) -> list[dict[str, Any]]:
    stored = []
    for record in records:
        row = asdict(record)
        row.pop("seconds")
        stored.append(row)
    return stored


def write_runs_csv(path: Path, records: Iterable[RunRecord]) -> None:
    fieldnames = [
        "seed",
        "condition",
        "best_epoch",
        "validation_loss",
        "test_loss",
        "test_accuracy",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records_for_storage(records))


def config_from_dict(value: dict[str, Any]) -> ExperimentConfig:
    converted = dict(value)
    converted["training_seeds"] = tuple(converted["training_seeds"])
    return ExperimentConfig(**converted)


def produce_bundle(output_dir: Path, config: ExperimentConfig) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records, summary, _ = run_experiment(config)
    write_json(output_dir / "config.json", asdict(config))
    write_json(output_dir / "summary.json", summary)
    write_runs_csv(output_dir / "runs.csv", records)
    environment = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "git_commit": git_commit(),
        "entry_command": f"python {SOURCE_PATH.as_posix()} --output-dir {output_dir.as_posix()}",
        "generated_data": True,
        "external_downloads": False,
    }
    write_json(output_dir / "environment.json", environment)
    files = ["config.json", "environment.json", "runs.csv", "summary.json"]
    manifest = {
        "schema": SCHEMA_VERSION,
        "files": {name: file_sha256(output_dir / name) for name in files},
        "source": {
            "path": SOURCE_PATH.as_posix(),
            "sha256": file_sha256(SOURCE_PATH),
        },
        "numeric_tolerance": 1e-10,
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def compare_reproduction(expected: dict[str, Any], actual: dict[str, Any], tolerance: float) -> None:
    keys = [
        "baseline_mean_accuracy",
        "baseline_standard_deviation",
        "interaction_mean_accuracy",
        "interaction_standard_deviation",
        "mean_paired_difference",
    ]
    for key in keys:
        if not np.isclose(expected[key], actual[key], atol=tolerance, rtol=0.0):
            raise RuntimeError(f"Reproduced {key} differs from the recorded value.")
    if not np.allclose(
        expected["paired_accuracy_differences"],
        actual["paired_accuracy_differences"],
        atol=tolerance,
        rtol=0.0,
    ):
        raise RuntimeError("Reproduced paired differences do not match.")
    if expected["split_sizes"] != actual["split_sizes"]:
        raise RuntimeError("Reproduced split sizes do not match.")


def verify_bundle(artifact_dir: Path) -> dict[str, Any]:
    manifest_path = artifact_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA_VERSION:
        raise RuntimeError("Unknown artifact schema.")
    for name, expected_hash in manifest["files"].items():
        path = artifact_dir / name
        if not path.is_file() or file_sha256(path) != expected_hash:
            raise RuntimeError(f"Artifact hash mismatch: {name}")
    source_path = Path(manifest["source"]["path"])
    if not source_path.is_file() or file_sha256(source_path) != manifest["source"]["sha256"]:
        raise RuntimeError("Source file differs from the recorded version.")

    config = config_from_dict(json.loads((artifact_dir / "config.json").read_text(encoding="utf-8")))
    expected = json.loads((artifact_dir / "summary.json").read_text(encoding="utf-8"))
    _, actual, split = run_experiment(config)
    compare_reproduction(expected, actual, float(manifest["numeric_tolerance"]))
    return {
        "status": "PASS",
        "files_verified": len(manifest["files"]),
        "source_verified": True,
        "split_disjoint": (
            set(split.train.tolist()).isdisjoint(split.validation.tolist())
            and set(split.train.tolist()).isdisjoint(split.test.tolist())
            and set(split.validation.tolist()).isdisjoint(split.test.tolist())
        ),
        "mean_paired_difference": actual["mean_paired_difference"],
        "interval": actual["paired_95_percent_t_interval"],
    }


def audit_only() -> None:
    compact = ExperimentConfig(sample_count=300, epochs=20, training_seeds=(13, 29))
    records, summary, split = run_experiment(compact)
    payload = {"b": 2, "a": [1, 3]}
    print("week13_reproducibility_audit")
    print(f"canonical_hash_stable={sha256_bytes(canonical_json(payload)) == sha256_bytes(canonical_json({'a': [1, 3], 'b': 2}))}")
    print(f"records={len(records)} paired_runs={summary['run_count']}")
    print(f"split_sizes={len(split.train)}/{len(split.validation)}/{len(split.test)}")
    print(f"split_disjoint={set(split.train).isdisjoint(split.validation) and set(split.train).isdisjoint(split.test) and set(split.validation).isdisjoint(split.test)}")
    print("files_written=False")
    print("downloads_started=False")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/week13/reference"))
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.audit_only:
        audit_only()
        return
    if args.verify is not None:
        result = verify_bundle(args.verify)
        print("week13_bundle_verification")
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    config = ExperimentConfig()
    started = time.perf_counter()
    manifest = produce_bundle(args.output_dir, config)
    result = verify_bundle(args.output_dir)
    elapsed = time.perf_counter() - started
    print("week13_reproducible_experiment")
    print(f"artifact_directory={args.output_dir}")
    print(f"files_hashed={len(manifest['files'])}")
    print(f"git_commit={git_commit()}")
    print(f"mean_paired_accuracy_difference={result['mean_paired_difference']:.4f}")
    print(f"paired_95_percent_t_interval={result['interval'][0]:.4f},{result['interval'][1]:.4f}")
    print(f"verification_status={result['status']}")
    print(f"elapsed_seconds={elapsed:.2f}")


if __name__ == "__main__":
    main()