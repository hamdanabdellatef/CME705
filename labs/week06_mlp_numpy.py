"""A two-layer NumPy network that learns XOR by backpropagation."""

from __future__ import annotations

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))


def binary_cross_entropy(target: np.ndarray, prediction: np.ndarray) -> float:
    prediction = np.clip(prediction, 1e-8, 1.0 - 1e-8)
    return float(-np.mean(target * np.log(prediction) + (1 - target) * np.log(1 - prediction)))


def train_xor(*, epochs: int = 8000, learning_rate: float = 0.8, seed: int = 7):
    x = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    target = np.array([[0.0], [1.0], [1.0], [0.0]])
    rng = np.random.default_rng(seed)
    w1 = rng.normal(0, 0.6, (2, 4))
    b1 = np.zeros((1, 4))
    w2 = rng.normal(0, 0.6, (4, 1))
    b2 = np.zeros((1, 1))

    for _ in range(epochs):
        hidden = np.tanh(x @ w1 + b1)
        prediction = sigmoid(hidden @ w2 + b2)
        output_delta = (prediction - target) / len(x)
        hidden_delta = (output_delta @ w2.T) * (1.0 - hidden**2)
        w2 -= learning_rate * hidden.T @ output_delta
        b2 -= learning_rate * output_delta.sum(axis=0, keepdims=True)
        w1 -= learning_rate * x.T @ hidden_delta
        b1 -= learning_rate * hidden_delta.sum(axis=0, keepdims=True)

    hidden = np.tanh(x @ w1 + b1)
    prediction = sigmoid(hidden @ w2 + b2)
    return prediction, binary_cross_entropy(target, prediction)


def main() -> None:
    prediction, loss = train_xor()
    print("predictions=", np.round(prediction.ravel(), 3).tolist())
    print(f"binary_cross_entropy={loss:.4f}")


if __name__ == "__main__":
    main()
