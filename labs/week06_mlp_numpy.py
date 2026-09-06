"""A transparent two-layer NumPy network trained by backpropagation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Parameters = dict[str, np.ndarray]
Gradients = dict[str, np.ndarray]


@dataclass(frozen=True)
class ForwardCache:
    """Intermediate values required by the backward pass."""

    x: np.ndarray
    hidden_pre: np.ndarray
    hidden: np.ndarray
    logits: np.ndarray
    prediction: np.ndarray


@dataclass(frozen=True)
class TrainingResult:
    """Parameters, predictions, and loss history from one full-batch run."""

    parameters: Parameters
    prediction: np.ndarray
    loss_history: np.ndarray


def xor_data() -> tuple[np.ndarray, np.ndarray]:
    """Return the four XOR observations and binary targets."""
    features = np.array(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    )
    target = np.array([[0.0], [1.0], [1.0], [0.0]])
    return features, target


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Compute a stable sigmoid."""
    x = np.clip(np.asarray(x, dtype=float), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))


def binary_cross_entropy(target: np.ndarray, prediction: np.ndarray) -> float:
    """Compute binary cross-entropy from probabilities."""
    probability = np.clip(np.asarray(prediction, dtype=float), 1e-8, 1.0 - 1e-8)
    target = np.asarray(target, dtype=float)
    return float(
        -np.mean(
            target * np.log(probability)
            + (1.0 - target) * np.log(1.0 - probability)
        )
    )


def binary_cross_entropy_from_logits(
    target: np.ndarray, logits: np.ndarray
) -> float:
    """Compute stable binary cross-entropy without taking log(sigmoid(z))."""
    target = np.asarray(target, dtype=float)
    logits = np.asarray(logits, dtype=float)
    return float(np.mean(np.logaddexp(0.0, logits) - target * logits))


def initialize_parameters(
    *, input_features: int = 2, hidden_units: int = 4, seed: int = 7
) -> Parameters:
    """Initialize a two-layer network with small reproducible weights."""
    if input_features <= 0 or hidden_units <= 0:
        raise ValueError("layer widths must be positive")
    rng = np.random.default_rng(seed)
    return {
        "w1": rng.normal(0.0, 0.6, (input_features, hidden_units)),
        "b1": np.zeros((1, hidden_units)),
        "w2": rng.normal(0.0, 0.6, (hidden_units, 1)),
        "b2": np.zeros((1, 1)),
    }


def forward(x: np.ndarray, parameters: Parameters) -> ForwardCache:
    """Run the affine, tanh, affine, sigmoid computation."""
    x = np.asarray(x, dtype=float)
    hidden_pre = x @ parameters["w1"] + parameters["b1"]
    hidden = np.tanh(hidden_pre)
    logits = hidden @ parameters["w2"] + parameters["b2"]
    prediction = sigmoid(logits)
    return ForwardCache(x, hidden_pre, hidden, logits, prediction)


def backward(
    target: np.ndarray, parameters: Parameters, cache: ForwardCache
) -> Gradients:
    """Backpropagate binary cross-entropy through sigmoid and tanh."""
    target = np.asarray(target, dtype=float)
    if target.shape != cache.prediction.shape:
        raise ValueError("target shape must match prediction shape")
    examples = len(target)

    d_logits = (cache.prediction - target) / examples
    d_w2 = cache.hidden.T @ d_logits
    d_b2 = d_logits.sum(axis=0, keepdims=True)

    d_hidden = d_logits @ parameters["w2"].T
    d_hidden_pre = d_hidden * (1.0 - cache.hidden**2)
    d_w1 = cache.x.T @ d_hidden_pre
    d_b1 = d_hidden_pre.sum(axis=0, keepdims=True)

    return {"w1": d_w1, "b1": d_b1, "w2": d_w2, "b2": d_b2}


def apply_update(
    parameters: Parameters, gradients: Gradients, learning_rate: float
) -> Parameters:
    """Return new parameters after one gradient-descent update."""
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    return {
        name: value - learning_rate * gradients[name]
        for name, value in parameters.items()
    }


def gradient_check(
    x: np.ndarray,
    target: np.ndarray,
    parameters: Parameters,
    *,
    epsilon: float = 1e-5,
) -> tuple[float, dict[str, float]]:
    """Compare every analytical gradient with a centered finite difference."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    analytical = backward(target, parameters, forward(x, parameters))
    per_parameter: dict[str, float] = {}

    for name, value in parameters.items():
        numerical = np.zeros_like(value)
        for index in np.ndindex(value.shape):
            plus = {key: item.copy() for key, item in parameters.items()}
            minus = {key: item.copy() for key, item in parameters.items()}
            plus[name][index] += epsilon
            minus[name][index] -= epsilon
            loss_plus = binary_cross_entropy_from_logits(
                target, forward(x, plus).logits
            )
            loss_minus = binary_cross_entropy_from_logits(
                target, forward(x, minus).logits
            )
            numerical[index] = (loss_plus - loss_minus) / (2.0 * epsilon)

        denominator = np.maximum(
            1e-8, np.abs(analytical[name]) + np.abs(numerical)
        )
        per_parameter[name] = float(
            np.max(np.abs(analytical[name] - numerical) / denominator)
        )

    return max(per_parameter.values()), per_parameter


def train_xor_detailed(
    *,
    epochs: int = 4000,
    learning_rate: float = 0.8,
    seed: int = 7,
) -> TrainingResult:
    """Train a 2-to-4-to-1 tanh network on all four XOR observations."""
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    x, target = xor_data()
    parameters = initialize_parameters(seed=seed)
    history = [
        binary_cross_entropy_from_logits(target, forward(x, parameters).logits)
    ]

    for _ in range(epochs):
        cache = forward(x, parameters)
        gradients = backward(target, parameters, cache)
        parameters = apply_update(parameters, gradients, learning_rate)
        history.append(
            binary_cross_entropy_from_logits(
                target, forward(x, parameters).logits
            )
        )

    prediction = forward(x, parameters).prediction
    return TrainingResult(parameters, prediction, np.asarray(history))


def train_xor(
    *, epochs: int = 4000, learning_rate: float = 0.8, seed: int = 7
) -> tuple[np.ndarray, float]:
    """Compatibility wrapper returning predictions and final loss."""
    result = train_xor_detailed(
        epochs=epochs, learning_rate=learning_rate, seed=seed
    )
    return result.prediction, float(result.loss_history[-1])


def train_linear_xor(
    *, epochs: int = 4000, learning_rate: float = 0.8
) -> tuple[np.ndarray, float]:
    """Fit one sigmoid unit, which cannot represent the XOR decision pattern."""
    x, target = xor_data()
    weights = np.zeros((2, 1))
    bias = np.zeros((1, 1))
    for _ in range(epochs):
        logits = x @ weights + bias
        prediction = sigmoid(logits)
        d_logits = (prediction - target) / len(x)
        weights -= learning_rate * (x.T @ d_logits)
        bias -= learning_rate * d_logits.sum(axis=0, keepdims=True)
    logits = x @ weights + bias
    return sigmoid(logits), binary_cross_entropy_from_logits(target, logits)


def classification_accuracy(
    target: np.ndarray, prediction: np.ndarray, *, threshold: float = 0.5
) -> float:
    """Return binary accuracy for a fixed probability threshold."""
    predicted_class = (np.asarray(prediction) >= threshold).astype(int)
    return float(np.mean(predicted_class == np.asarray(target)))


def main() -> None:
    x, target = xor_data()
    initial = initialize_parameters()
    max_error, per_parameter = gradient_check(x, target, initial)
    linear_prediction, linear_loss = train_linear_xor()
    nonlinear = train_xor_detailed()

    print("forward_shapes")
    first_cache = forward(x, initial)
    print(f"x={first_cache.x.shape}")
    print(f"hidden={first_cache.hidden.shape}")
    print(f"logits={first_cache.logits.shape}")
    print(f"prediction={first_cache.prediction.shape}")

    print("\ngradient_check_relative_error")
    for name in ("w1", "b1", "w2", "b2"):
        print(f"{name}={per_parameter[name]:.3e}")
    print(f"maximum={max_error:.3e}")

    print("\nlinear_baseline")
    print(
        "predictions="
        f"{np.array2string(linear_prediction.ravel(), precision=3)}"
    )
    print(f"binary_cross_entropy={linear_loss:.4f}")
    print(
        "accuracy="
        f"{classification_accuracy(target, linear_prediction):.3f}"
    )

    print("\nnonlinear_mlp")
    print(
        f"loss={nonlinear.loss_history[0]:.4f}"
        f" -> {nonlinear.loss_history[-1]:.4f}"
    )
    print(
        "predictions="
        f"{np.array2string(nonlinear.prediction.ravel(), precision=3)}"
    )
    print(
        "classes="
        f"{(nonlinear.prediction.ravel() >= 0.5).astype(int).tolist()}"
    )
    print(
        "accuracy="
        f"{classification_accuracy(target, nonlinear.prediction):.3f}"
    )


if __name__ == "__main__":
    main()
