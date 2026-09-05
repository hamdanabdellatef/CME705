"""Week 2 lab: split design, group leakage, and training-only preprocessing.

The generated repeated-measurement data are intentionally constructed so that
rows from one participant are very similar. The example demonstrates why a
row-wise split can answer the wrong evaluation question.
"""

from __future__ import annotations

import numpy as np


def _validate_split_fractions(
    train_fraction: float,
    validation_fraction: float,
) -> None:
    if train_fraction <= 0 or validation_fraction <= 0:
        raise ValueError("train and validation fractions must be positive")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("the fractions must leave a non-empty test share")


def random_row_split(
    row_count: int,
    *,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assign individual rows without protecting repeated sources."""
    _validate_split_fractions(train_fraction, validation_fraction)
    if row_count < 3:
        raise ValueError("at least three rows are required")

    shuffled = np.random.default_rng(seed).permutation(row_count)
    train_end = max(1, int(row_count * train_fraction))
    validation_end = max(
        train_end + 1,
        int(row_count * (train_fraction + validation_fraction)),
    )
    if validation_end >= row_count:
        validation_end = row_count - 1
    return (
        shuffled[:train_end],
        shuffled[train_end:validation_end],
        shuffled[validation_end:],
    )


def grouped_split(
    groups: np.ndarray,
    *,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assign whole groups to train, validation, or test partitions."""
    _validate_split_fractions(train_fraction, validation_fraction)
    unique_groups = np.unique(groups)
    if len(unique_groups) < 3:
        raise ValueError("at least three groups are required")

    shuffled = np.random.default_rng(seed).permutation(unique_groups)
    train_end = max(1, int(len(shuffled) * train_fraction))
    validation_end = max(
        train_end + 1,
        int(len(shuffled) * (train_fraction + validation_fraction)),
    )
    if validation_end >= len(shuffled):
        validation_end = len(shuffled) - 1

    train_groups = shuffled[:train_end]
    validation_groups = shuffled[train_end:validation_end]
    test_groups = shuffled[validation_end:]
    return (
        np.flatnonzero(np.isin(groups, train_groups)),
        np.flatnonzero(np.isin(groups, validation_groups)),
        np.flatnonzero(np.isin(groups, test_groups)),
    )


def make_repeated_measurement_data(
    group_count: int = 60,
    rows_per_group: int = 4,
    *,
    seed: int = 705,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate similar rows and one stable binary label per participant."""
    if group_count < 3:
        raise ValueError("group_count must be at least 3")
    if rows_per_group < 2:
        raise ValueError("rows_per_group must be at least 2")

    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(group_count), rows_per_group)

    participant_signature = rng.normal(0.0, 1.0, size=(group_count, 3))
    propensity = (
        0.75 * participant_signature[:, 0]
        - 0.35 * participant_signature[:, 1]
        + rng.normal(0.0, 1.10, size=group_count)
    )
    participant_label = (propensity >= 0.0).astype(int)

    measurement_noise = rng.normal(
        0.0,
        0.10,
        size=(len(groups), participant_signature.shape[1]),
    )
    features = participant_signature[groups] + measurement_noise
    target = participant_label[groups]
    return features, target, groups


def fit_standardizer(x_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Learn centering and scaling parameters from training data only."""
    if x_train.ndim != 2 or len(x_train) == 0:
        raise ValueError("x_train must be a non-empty two-dimensional array")
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    return mean, np.where(scale == 0, 1.0, scale)


def apply_standardizer(
    features: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    """Apply already fitted preprocessing parameters."""
    return (features - mean) / scale


def nearest_neighbor_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_query: np.ndarray,
) -> np.ndarray:
    """Predict the label of the closest training row."""
    if x_train.ndim != 2 or x_query.ndim != 2:
        raise ValueError("feature arrays must be two-dimensional")
    if x_train.shape[1] != x_query.shape[1]:
        raise ValueError("training and query arrays need equal feature counts")
    if len(x_train) != len(y_train) or len(x_train) == 0:
        raise ValueError("training features and labels must be non-empty and aligned")

    squared_distance = np.sum(
        (x_query[:, np.newaxis, :] - x_train[np.newaxis, :, :]) ** 2,
        axis=2,
    )
    nearest_index = np.argmin(squared_distance, axis=1)
    return y_train[nearest_index]


def balanced_accuracy(target: np.ndarray, prediction: np.ndarray) -> float:
    """Average recall across labels represented in the target."""
    if target.shape != prediction.shape:
        raise ValueError("target and prediction must have the same shape")
    recalls = [
        np.mean(prediction[target == label] == label)
        for label in (0, 1)
        if np.any(target == label)
    ]
    if not recalls:
        raise ValueError("target must not be empty")
    return float(np.mean(recalls))


def overlapping_group_count(
    groups: np.ndarray,
    first_index: np.ndarray,
    second_index: np.ndarray,
) -> int:
    """Count sources represented in both selected partitions."""
    first_groups = set(groups[first_index].tolist())
    second_groups = set(groups[second_index].tolist())
    return len(first_groups.intersection(second_groups))


def evaluate_split(
    features: np.ndarray,
    target: np.ndarray,
    train: np.ndarray,
    validation: np.ndarray,
    test: np.ndarray,
) -> tuple[float, float]:
    """Fit preprocessing on training data and evaluate nearest-neighbor scores."""
    mean, scale = fit_standardizer(features[train])
    x_train = apply_standardizer(features[train], mean, scale)
    x_validation = apply_standardizer(features[validation], mean, scale)
    x_test = apply_standardizer(features[test], mean, scale)

    validation_prediction = nearest_neighbor_predict(
        x_train,
        target[train],
        x_validation,
    )
    test_prediction = nearest_neighbor_predict(x_train, target[train], x_test)
    return (
        balanced_accuracy(target[validation], validation_prediction),
        balanced_accuracy(target[test], test_prediction),
    )


def _partition_summary(
    name: str,
    groups: np.ndarray,
    split: tuple[np.ndarray, np.ndarray, np.ndarray],
    validation_score: float,
    test_score: float,
) -> None:
    train, validation, test = split
    print(name)
    print(
        "  rows: "
        f"train={len(train)}, validation={len(validation)}, test={len(test)}"
    )
    print(
        "  groups: "
        f"train={len(np.unique(groups[train]))}, "
        f"validation={len(np.unique(groups[validation]))}, "
        f"test={len(np.unique(groups[test]))}"
    )
    print(
        "  overlapping groups: "
        f"train/validation={overlapping_group_count(groups, train, validation)}, "
        f"train/test={overlapping_group_count(groups, train, test)}"
    )
    print(f"  validation balanced accuracy={validation_score:.3f}")
    print(f"  test balanced accuracy={test_score:.3f}")


def main() -> None:
    features, target, groups = make_repeated_measurement_data()

    row_split = random_row_split(len(target), seed=705)
    group_split = grouped_split(groups, seed=705)

    row_validation, row_test = evaluate_split(
        features,
        target,
        *row_split,
    )
    group_validation, group_test = evaluate_split(
        features,
        target,
        *group_split,
    )

    print("CME705 Week 2: generalization and evaluation")
    print("Synthetic case: repeated measurements from participants")
    print(f"Dataset: {len(np.unique(groups))} participants, {len(target)} rows")
    print("Evaluation claim: performance on participants unseen during fitting")
    print()
    _partition_summary(
        "Row-wise split (invalid for this claim)",
        groups,
        row_split,
        row_validation,
        row_test,
    )
    print()
    _partition_summary(
        "Grouped split (matches this claim)",
        groups,
        group_split,
        group_validation,
        group_test,
    )
    print()
    print(f"Optimism gap in test score={row_test - group_test:.3f}")
    print()
    print("Interpretation")
    print(
        "  The row-wise split lets the nearest-neighbor model match many test "
        "rows to the same participants in training."
    )
    print(
        "  The grouped split removes participant overlap and evaluates the "
        "stated new-participant claim."
    )
    print(
        "  The numerical gap belongs to this simulation; it is not a universal "
        "estimate of leakage."
    )


if __name__ == "__main__":
    main()
