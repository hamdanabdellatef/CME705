"""Week 4 lab: a transparent neural-network forward pass with NumPy.

The parameters are fixed for inspection. Optimization and backpropagation are
introduced in Weeks 5 and 6.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


ArrayFunction = Callable[[np.ndarray], np.ndarray]


def sigmoid(value: np.ndarray) -> np.ndarray:
    """Map real-valued logits to the interval from zero to one."""
    clipped = np.clip(value, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def relu(value: np.ndarray) -> np.ndarray:
    """Return the positive part of every value."""
    return np.maximum(value, 0.0)


def softmax(logits: np.ndarray) -> np.ndarray:
    """Calculate stable class probabilities along the final dimension."""
    if logits.ndim != 2:
        raise ValueError("logits must be a two-dimensional batch")
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=1, keepdims=True)


def affine(
    features: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
) -> np.ndarray:
    """Calculate X @ W + b with explicit dimension checks."""
    if features.ndim != 2 or weights.ndim != 2 or bias.ndim != 1:
        raise ValueError("features and weights must be matrices; bias must be a vector")
    if features.shape[1] != weights.shape[0]:
        raise ValueError("the feature count must match the weight input dimension")
    if weights.shape[1] != len(bias):
        raise ValueError("the bias needs one value per output unit")
    return features @ weights + bias


def neuron_output(
    features: np.ndarray,
    weights: np.ndarray,
    bias: float,
    activation: ArrayFunction,
) -> tuple[float, float]:
    """Calculate one neuron's pre-activation and activation."""
    if features.ndim != 1 or weights.shape != features.shape:
        raise ValueError("features and weights must be aligned vectors")
    pre_activation = float(features @ weights + bias)
    activated = float(activation(np.array(pre_activation)))
    return pre_activation, activated


def forward_two_layer(
    features: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    *,
    hidden_activation: ArrayFunction = relu,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run affine, activation, affine, and softmax operations."""
    hidden_pre_activation = affine(features, w1, b1)
    hidden_activation_value = hidden_activation(hidden_pre_activation)
    logits = affine(hidden_activation_value, w2, b2)
    probability = softmax(logits)
    return hidden_pre_activation, hidden_activation_value, logits, probability


def collapse_linear_layers(
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the equivalent parameters for two affine layers without activation."""
    if w1.shape[1] != w2.shape[0]:
        raise ValueError("adjacent layer dimensions must agree")
    if b1.shape != (w1.shape[1],) or b2.shape != (w2.shape[1],):
        raise ValueError("bias dimensions must match their layers")
    equivalent_weight = w1 @ w2
    equivalent_bias = b1 @ w2 + b2
    return equivalent_weight, equivalent_bias


def parameter_count(weights: list[np.ndarray], biases: list[np.ndarray]) -> int:
    """Count scalar trainable parameters."""
    return int(sum(array.size for array in [*weights, *biases]))


def teaching_parameters() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Return a small batch and fixed two-layer network."""
    features = np.array(
        [
            [0.2, -1.0, 0.5],
            [1.2, 0.3, -0.7],
            [-0.4, 0.8, 1.1],
            [0.0, -0.2, 0.9],
        ]
    )
    w1 = np.array(
        [
            [0.8, -0.4, 0.5, 0.2],
            [-0.3, 0.7, 0.1, -0.8],
            [0.6, 0.2, -0.5, 0.9],
        ]
    )
    b1 = np.array([0.1, -0.2, 0.05, 0.3])
    w2 = np.array(
        [
            [0.7, -0.5],
            [-0.2, 0.8],
            [0.4, 0.1],
            [-0.6, 0.9],
        ]
    )
    b2 = np.array([0.05, -0.1])
    return features, w1, b1, w2, b2


def main() -> None:
    features, w1, b1, w2, b2 = teaching_parameters()
    hidden_pre, hidden, logits, probability = forward_two_layer(
        features,
        w1,
        b1,
        w2,
        b2,
    )

    example_x = np.array([2.0, -1.0, 0.5])
    example_w = np.array([0.4, -0.8, 1.2])
    example_z, example_a = neuron_output(
        example_x,
        example_w,
        -0.3,
        sigmoid,
    )

    equivalent_weight, equivalent_bias = collapse_linear_layers(
        w1,
        b1,
        w2,
        b2,
    )
    two_linear_layers = affine(affine(features, w1, b1), w2, b2)
    one_linear_layer = affine(
        features,
        equivalent_weight,
        equivalent_bias,
    )
    collapse_error = float(np.max(np.abs(two_linear_layers - one_linear_layer)))

    print("CME705 Week 4: neural-network computation")
    print()
    print("One neuron")
    print(f"  pre_activation={example_z:.3f}")
    print(f"  sigmoid_output={example_a:.3f}")
    print()
    print("Two-layer batch")
    print(f"  X shape:           {features.shape}")
    print(f"  W1 and b1 shapes:  {w1.shape}, {b1.shape}")
    print(f"  Z1 and A1 shapes:  {hidden_pre.shape}, {hidden.shape}")
    print(f"  W2 and b2 shapes:  {w2.shape}, {b2.shape}")
    print(f"  logits shape:      {logits.shape}")
    print(f"  probability shape: {probability.shape}")
    print(f"  parameter count:   {parameter_count([w1, w2], [b1, b2])}")
    print()
    print("First two probability rows")
    print(np.round(probability[:2], 3))
    print("Probability row sums")
    print(np.round(probability.sum(axis=1), 6))
    print("Predicted classes")
    print(np.argmax(probability, axis=1))
    print()
    print("Linear-layer collapse")
    print(f"  equivalent W shape: {equivalent_weight.shape}")
    print(f"  equivalent b shape: {equivalent_bias.shape}")
    print(f"  maximum difference: {collapse_error:.3e}")
    print()
    print("Interpretation")
    print("  Matrix dimensions determine which transformations are valid.")
    print("  Bias gives every output unit its own offset.")
    print("  Without a nonlinear hidden activation, two affine layers collapse")
    print("  to one affine transformation.")


if __name__ == "__main__":
    main()
