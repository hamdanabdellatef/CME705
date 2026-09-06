"""Gradient descent for linear regression, implemented with NumPy.

The script separates optimization evidence from evaluation evidence. It uses a
training split to fit parameters, a validation split to select a learning rate,
and the test split once for the selected configuration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OptimizationResult:
    """Parameters and diagnostics from one optimization run."""

    weights: np.ndarray
    bias: float
    train_loss: np.ndarray
    updates: int
    examples_seen: int
    diverged: bool


def make_regression_data(
    *, seed: int = 705, examples: int = 320
) -> tuple[np.ndarray, np.ndarray]:
    """Create a deterministic regression problem with three scaled features."""
    if examples < 30:
        raise ValueError("examples must be at least 30")
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(examples, 3)) * np.array([1.0, 4.0, 0.25])
    true_weights = np.array([1.5, -0.55, 3.0])
    target = features @ true_weights + 0.4 + rng.normal(0.0, 0.25, examples)
    return features, target


def split_indices(
    examples: int, *, seed: int = 705
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return disjoint training, validation, and test indices."""
    if examples < 30:
        raise ValueError("examples must be at least 30")
    order = np.random.default_rng(seed).permutation(examples)
    train_end = int(round(examples * 0.6875))
    validation_end = train_end + int(round(examples * 0.15625))
    return order[:train_end], order[train_end:validation_end], order[validation_end:]


def fit_standardizer(x_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit feature scaling on training data only."""
    mean = np.asarray(x_train, dtype=float).mean(axis=0)
    scale = np.asarray(x_train, dtype=float).std(axis=0)
    if np.any(scale == 0):
        raise ValueError("every feature must vary in the training split")
    return mean, scale


def apply_standardizer(
    x: np.ndarray, mean: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    """Apply fixed training-set scaling statistics."""
    return (np.asarray(x, dtype=float) - mean) / scale


def mean_squared_error(
    x: np.ndarray, y: np.ndarray, weights: np.ndarray, bias: float
) -> float:
    """Return mean squared prediction error."""
    error = np.asarray(x, dtype=float) @ weights + bias - np.asarray(y, dtype=float)
    return float(np.mean(error**2))


def linear_gradients(
    x: np.ndarray, y: np.ndarray, weights: np.ndarray, bias: float
) -> tuple[np.ndarray, float]:
    """Return the exact mini-batch gradients of mean squared error."""
    error = np.asarray(x, dtype=float) @ weights + bias - np.asarray(y, dtype=float)
    gradient_weights = (2.0 / len(x)) * (np.asarray(x, dtype=float).T @ error)
    gradient_bias = 2.0 * float(error.mean())
    return gradient_weights, gradient_bias


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
    epochs: int = 120,
    seed: int = 0,
    divergence_limit: float = 1e12,
) -> OptimizationResult:
    """Fit a linear model and record full-training loss after every epoch."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 2 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("x must be 2D and y must be 1D with matching rows")
    if learning_rate <= 0 or epochs <= 0:
        raise ValueError("learning_rate and epochs must be positive")

    rng = np.random.default_rng(seed)
    weights = np.zeros(x.shape[1], dtype=float)
    bias = 0.0
    history = [mean_squared_error(x, y, weights, bias)]
    updates = 0
    examples_seen = 0
    diverged = False

    for _ in range(epochs):
        for xb, yb, _ in iter_minibatches(x, y, batch_size, rng):
            gradient_weights, gradient_bias = linear_gradients(
                xb, yb, weights, bias
            )
            weights -= learning_rate * gradient_weights
            bias -= learning_rate * gradient_bias
            updates += 1
            examples_seen += len(xb)

        loss = mean_squared_error(x, y, weights, bias)
        history.append(loss)
        if not np.isfinite(loss) or loss > divergence_limit:
            diverged = True
            break

    return OptimizationResult(
        weights=weights,
        bias=bias,
        train_loss=np.asarray(history),
        updates=updates,
        examples_seen=examples_seen,
        diverged=diverged,
    )


def select_learning_rate(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    candidates: tuple[float, ...],
    *,
    batch_size: int = 32,
    epochs: int = 120,
    seed: int = 705,
) -> tuple[float, dict[float, float], dict[float, OptimizationResult]]:
    """Select the finite candidate with the lowest validation MSE."""
    if not candidates:
        raise ValueError("at least one learning-rate candidate is required")
    validation_loss: dict[float, float] = {}
    results: dict[float, OptimizationResult] = {}
    for learning_rate in candidates:
        result = fit_linear_regression(
            x_train,
            y_train,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            seed=seed,
        )
        results[learning_rate] = result
        score = mean_squared_error(
            x_validation,
            y_validation,
            result.weights,
            result.bias,
        )
        validation_loss[learning_rate] = score if np.isfinite(score) else np.inf
    selected = min(candidates, key=validation_loss.__getitem__)
    return selected, validation_loss, results


def main() -> None:
    features, target = make_regression_data()
    train, validation, test = split_indices(len(target))
    mean, scale = fit_standardizer(features[train])
    x_train = apply_standardizer(features[train], mean, scale)
    x_validation = apply_standardizer(features[validation], mean, scale)
    x_test = apply_standardizer(features[test], mean, scale)

    print("update_schedule  batch  updates  initial_mse  final_mse")
    for name, batch_size in (
        ("full_batch", len(train)),
        ("stochastic", 1),
        ("mini_batch", 32),
    ):
        result = fit_linear_regression(
            x_train,
            target[train],
            learning_rate=0.05,
            batch_size=batch_size,
            epochs=40,
            seed=705,
        )
        print(
            f"{name:15s} {batch_size:5d} {result.updates:8d} "
            f"{result.train_loss[0]:11.4f} {result.train_loss[-1]:9.4f}"
        )

    candidates = (0.001, 0.01, 0.05, 0.2, 1.2)
    selected, validation_loss, results = select_learning_rate(
        x_train,
        target[train],
        x_validation,
        target[validation],
        candidates,
    )
    print("\nlearning_rate  validation_mse  status")
    for learning_rate in candidates:
        status = "diverged" if results[learning_rate].diverged else "finite"
        print(
            f"{learning_rate:13.3f} {validation_loss[learning_rate]:15.4f}  {status}"
        )

    chosen = results[selected]
    test_mse = mean_squared_error(
        x_test,
        target[test],
        chosen.weights,
        chosen.bias,
    )
    print(f"\nselected_learning_rate={selected:.3f}")
    print(f"selected_updates={chosen.updates}")
    print(f"test_mse={test_mse:.4f}")
    print(f"weights={np.array2string(chosen.weights, precision=3)}")
    print(f"bias={chosen.bias:.3f}")


if __name__ == "__main__":
    main()
