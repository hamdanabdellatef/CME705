"""Correct forward and backward mechanics for inverted dropout."""

from __future__ import annotations

import numpy as np


def inverted_dropout(
    activation: np.ndarray,
    drop_probability: float,
    rng: np.random.Generator,
    *,
    training: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 <= drop_probability < 1.0:
        raise ValueError("drop_probability must satisfy 0 <= p < 1")
    if not training or drop_probability == 0.0:
        return activation.copy(), np.ones_like(activation)
    keep_probability = 1.0 - drop_probability
    mask = (rng.random(activation.shape) < keep_probability) / keep_probability
    return activation * mask, mask


def sigmoid_dropout_backward(
    sigmoid_activation: np.ndarray,
    upstream_gradient: np.ndarray,
    dropout_mask: np.ndarray,
) -> np.ndarray:
    """Differentiate sigmoid using its unscaled activation, then apply mask."""
    return upstream_gradient * dropout_mask * sigmoid_activation * (1.0 - sigmoid_activation)


def main() -> None:
    rng = np.random.default_rng(705)
    activation = np.full(100_000, 0.8)
    dropped, mask = inverted_dropout(activation, 0.2, rng, training=True)
    gradient = sigmoid_dropout_backward(activation, np.ones_like(activation), mask)
    print(f"activation_mean_before={activation.mean():.4f}")
    print(f"activation_mean_after={dropped.mean():.4f}")
    print(f"gradient_mean={gradient.mean():.4f}")


if __name__ == "__main__":
    main()
