"""Numerically stable softmax and multiclass cross-entropy."""

from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    if logits.ndim != 2:
        raise ValueError("logits must have shape (examples, classes)")
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def cross_entropy(logits: np.ndarray, class_index: np.ndarray) -> float:
    probability = softmax(logits)
    chosen = probability[np.arange(len(class_index)), class_index]
    return float(-np.log(np.clip(chosen, 1e-12, 1.0)).mean())


def main() -> None:
    logits = np.array([[1002.0, 1001.0, 998.0], [-1000.0, -999.0, -1003.0]])
    target = np.array([0, 1])
    probability = softmax(logits)
    print("probabilities=", np.round(probability, 4).tolist())
    print(f"cross_entropy={cross_entropy(logits, target):.4f}")


if __name__ == "__main__":
    main()
