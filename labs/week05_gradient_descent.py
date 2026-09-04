"""Mini-batch linear regression implemented with NumPy."""

from __future__ import annotations

import numpy as np


def iter_minibatches(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
):
    """Yield every example exactly once in a shuffled epoch."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if len(x) != len(y):
        raise ValueError("x and y must contain the same number of examples")
    order = rng.permutation(len(x))
    for start in range(0, len(order), batch_size):
        index = order[start : start + batch_size]
        yield x[index], y[index], index


def fit_linear_regression(
    x: np.ndarray,
    y: np.ndarray,
    *,
    learning_rate: float = 0.05,
    batch_size: int = 32,
    epochs: int = 200,
    seed: int = 0,
) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    weights = np.zeros(x.shape[1], dtype=float)
    bias = 0.0
    for _ in range(epochs):
        for xb, yb, _ in iter_minibatches(x, y, batch_size, rng):
            error = xb @ weights + bias - yb
            weights -= learning_rate * (2.0 / len(xb)) * (xb.T @ error)
            bias -= learning_rate * 2.0 * float(error.mean())
    return weights, bias


def main() -> None:
    rng = np.random.default_rng(705)
    x = rng.normal(size=(257, 2))
    y = 1.5 * x[:, 0] - 2.0 * x[:, 1] + 0.4 + rng.normal(0, 0.1, len(x))
    train, test = np.arange(220), np.arange(220, len(x))
    mean, std = x[train].mean(0), x[train].std(0)
    x_train = (x[train] - mean) / std
    x_test = (x[test] - mean) / std
    weights, bias = fit_linear_regression(x_train, y[train])
    mse = float(np.mean((x_test @ weights + bias - y[test]) ** 2))
    print(f"test_mse={mse:.4f}")


if __name__ == "__main__":
    main()
