"""Week 1 diagnostic: arrays, a baseline, and a transparent scoring rule.

The data are generated for teaching. They do not describe real machines.
"""

from __future__ import annotations

import numpy as np

FEATURE_NAMES = ("temperature_c", "vibration_mm_s", "age_years")
CANDIDATE_WEIGHTS = np.array([0.14, 0.95, 0.12])
CANDIDATE_BIAS = 0.0
CENTER = np.array([75.0, 3.0, 6.0])


def make_machine_data(
    sample_count: int = 160,
    *,
    seed: int = 705,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a small binary-classification dataset."""
    if sample_count < 2:
        raise ValueError("sample_count must be at least 2")

    rng = np.random.default_rng(seed)
    temperature = rng.normal(loc=75.0, scale=8.0, size=sample_count)
    vibration = rng.normal(loc=3.0, scale=0.75, size=sample_count)
    age = rng.uniform(low=0.0, high=12.0, size=sample_count)
    features = np.column_stack((temperature, vibration, age))

    underlying_score = (
        0.16 * (temperature - CENTER[0])
        + 1.10 * (vibration - CENTER[1])
        + 0.10 * (age - CENTER[2])
        + rng.normal(loc=0.0, scale=0.70, size=sample_count)
    )
    target = (underlying_score >= 0.0).astype(int)
    return features, target


def majority_baseline(target: np.ndarray) -> np.ndarray:
    """Predict the most frequent class for every observation."""
    labels, counts = np.unique(target, return_counts=True)
    majority_label = labels[np.argmax(counts)]
    return np.full(target.shape, majority_label, dtype=target.dtype)


def linear_scores(
    features: np.ndarray,
    weights: np.ndarray,
    bias: float = 0.0,
) -> np.ndarray:
    """Calculate one score per row using X @ w + b."""
    if features.ndim != 2:
        raise ValueError("features must be a two-dimensional matrix")
    if weights.shape != (features.shape[1],):
        raise ValueError("weights must contain one value per feature")
    return features @ weights + bias


def inspection_rule(
    features: np.ndarray,
    *,
    weights: np.ndarray = CANDIDATE_WEIGHTS,
    bias: float = CANDIDATE_BIAS,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a centered linear score and a threshold at zero."""
    scores = linear_scores(features - CENTER, weights, bias)
    prediction = (scores >= 0.0).astype(int)
    return scores, prediction


def accuracy(target: np.ndarray, prediction: np.ndarray) -> float:
    """Return the fraction of matching labels."""
    if target.shape != prediction.shape:
        raise ValueError("target and prediction must have the same shape")
    return float(np.mean(target == prediction))


def main() -> None:
    features, target = make_machine_data()
    baseline_prediction = majority_baseline(target)
    scores, rule_prediction = inspection_rule(features)

    print("CME705 Week 1: Python and NumPy diagnostic")
    print("Synthetic case: prioritize machines for inspection")
    print(f"Feature matrix shape: {features.shape}")
    print(f"Target vector shape:  {target.shape}")
    print()

    print("Feature summary")
    for column, name in enumerate(FEATURE_NAMES):
        values = features[:, column]
        print(
            f"  {name:18s} mean={values.mean():7.2f} "
            f"std={values.std():6.2f}"
        )

    requiring_inspection = int(target.sum())
    print()
    print(
        "Class counts: "
        f"no inspection={len(target) - requiring_inspection}, "
        f"inspection={requiring_inspection}"
    )
    print(f"Majority baseline accuracy: {accuracy(target, baseline_prediction):.3f}")
    print(f"Linear rule accuracy:       {accuracy(target, rule_prediction):.3f}")

    print()
    print("First five scores and predictions")
    for index in range(5):
        print(
            f"  row={index:2d} score={scores[index]:6.2f} "
            f"prediction={rule_prediction[index]} target={target[index]}"
        )

    print()
    print("Interpretation prompts")
    print("  1. What does one row represent?")
    print("  2. Why compare the rule with the majority baseline?")
    print("  3. What changes when one weight is set to zero?")
    print("  4. What would real deployment evidence need to include?")


if __name__ == "__main__":
    main()
