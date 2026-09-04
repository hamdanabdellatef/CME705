"""A leakage-safe split and logistic baseline implemented with NumPy."""

from __future__ import annotations

import numpy as np


def grouped_split(
    groups: np.ndarray,
    *,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assign whole groups to train, validation, or test partitions."""
    if train_fraction <= 0 or validation_fraction <= 0:
        raise ValueError("train and validation fractions must be positive")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("the fractions must leave a non-empty test share")
    unique_groups = np.unique(groups)
    if len(unique_groups) < 3:
        raise ValueError("at least three groups are required")
    shuffled = np.random.default_rng(seed).permutation(unique_groups)
    train_end = max(1, int(len(shuffled) * train_fraction))
    validation_end = max(train_end + 1, int(len(shuffled) * (train_fraction + validation_fraction)))
    train_groups = shuffled[:train_end]
    validation_groups = shuffled[train_end:validation_end]
    test_groups = shuffled[validation_end:]
    return (
        np.flatnonzero(np.isin(groups, train_groups)),
        np.flatnonzero(np.isin(groups, validation_groups)),
        np.flatnonzero(np.isin(groups, test_groups)),
    )


def fit_standardizer(x_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Learn preprocessing parameters from training data only."""
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    return mean, np.where(scale == 0, 1.0, scale)


def sigmoid(logit: np.ndarray) -> np.ndarray:
    logit = np.clip(logit, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-logit))


def fit_logistic(x: np.ndarray, target: np.ndarray, *, epochs: int = 1500, learning_rate: float = 0.1):
    weights = np.zeros(x.shape[1])
    bias = 0.0
    for _ in range(epochs):
        error = sigmoid(x @ weights + bias) - target
        weights -= learning_rate * (x.T @ error) / len(x)
        bias -= learning_rate * float(error.mean())
    return weights, bias


def balanced_accuracy(target: np.ndarray, prediction: np.ndarray) -> float:
    recalls = [np.mean(prediction[target == label] == label) for label in (0, 1) if np.any(target == label)]
    return float(np.mean(recalls))


def main() -> None:
    rng = np.random.default_rng(705)
    groups = np.repeat(np.arange(60), 4)
    group_effect = rng.normal(size=(60, 2))
    x = group_effect[groups] + rng.normal(0, 0.5, (len(groups), 2))
    target = (x[:, 0] - 0.6 * x[:, 1] > 0).astype(int)
    train, validation, test = grouped_split(groups, seed=705)
    mean, scale = fit_standardizer(x[train])
    weights, bias = fit_logistic((x[train] - mean) / scale, target[train])
    for name, index in (("validation", validation), ("test", test)):
        prediction = (sigmoid(((x[index] - mean) / scale) @ weights + bias) >= 0.5).astype(int)
        print(f"{name}_balanced_accuracy={balanced_accuracy(target[index], prediction):.3f}")


if __name__ == "__main__":
    main()
