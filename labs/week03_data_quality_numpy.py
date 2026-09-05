"""Week 3 lab: data quality, imbalance, and controlled model comparison.

The generated data contain a rare positive class and informative missingness.
Every candidate uses the same split, training-only preprocessing, and metric
definitions so that the comparison changes one documented feature decision.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Evaluation:
    threshold: float
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    specificity: float


def sigmoid(value: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid for vectors."""
    clipped = np.clip(value, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def make_rare_event_data(
    sample_count: int = 1000,
    *,
    positive_fraction: float = 0.12,
    seed: int = 705,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate independent observations with one partially missing feature."""
    if sample_count < 30:
        raise ValueError("sample_count must be at least 30")
    if not 0.05 <= positive_fraction <= 0.45:
        raise ValueError("positive_fraction must be between 0.05 and 0.45")

    rng = np.random.default_rng(seed)
    complete_features = rng.normal(0.0, 1.0, size=(sample_count, 3))
    latent_score = (
        1.00 * complete_features[:, 0]
        + 1.35 * complete_features[:, 1]
        - 0.70 * complete_features[:, 2]
        + rng.normal(0.0, 0.75, size=sample_count)
    )
    cutoff = np.quantile(latent_score, 1.0 - positive_fraction)
    target = (latent_score >= cutoff).astype(int)

    missing_probability = sigmoid(-1.10 + 1.25 * complete_features[:, 1])
    missing_feature_one = rng.random(sample_count) < missing_probability
    features = complete_features.copy()
    features[missing_feature_one, 1] = np.nan
    return features, target


def stratified_split(
    target: np.ndarray,
    *,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
    seed: int = 705,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split each represented class independently, then combine partitions."""
    if train_fraction <= 0 or validation_fraction <= 0:
        raise ValueError("train and validation fractions must be positive")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("the fractions must leave a non-empty test share")

    rng = np.random.default_rng(seed)
    partitions: list[list[np.ndarray]] = [[], [], []]
    for label in np.unique(target):
        class_index = np.flatnonzero(target == label)
        if len(class_index) < 3:
            raise ValueError("each class needs at least three observations")
        shuffled = rng.permutation(class_index)
        train_end = max(1, int(len(shuffled) * train_fraction))
        validation_end = max(
            train_end + 1,
            int(len(shuffled) * (train_fraction + validation_fraction)),
        )
        if validation_end >= len(shuffled):
            validation_end = len(shuffled) - 1
        partitions[0].append(shuffled[:train_end])
        partitions[1].append(shuffled[train_end:validation_end])
        partitions[2].append(shuffled[validation_end:])

    return tuple(
        rng.permutation(np.concatenate(parts))
        for parts in partitions
    )


def fit_median_imputer(x_train: np.ndarray) -> np.ndarray:
    """Estimate one median per feature using training rows only."""
    if x_train.ndim != 2 or len(x_train) == 0:
        raise ValueError("x_train must be a non-empty two-dimensional array")
    medians = np.nanmedian(x_train, axis=0)
    if np.isnan(medians).any():
        raise ValueError("a feature is completely missing in training data")
    return medians


def apply_median_imputer(
    features: np.ndarray,
    medians: np.ndarray,
) -> np.ndarray:
    """Replace missing entries with already fitted medians."""
    if features.shape[1] != len(medians):
        raise ValueError("one median is required per feature")
    return np.where(np.isnan(features), medians, features)


def add_missing_indicators(
    imputed: np.ndarray,
    original: np.ndarray,
    indicator_columns: np.ndarray,
) -> np.ndarray:
    """Append binary flags for columns observed missing during training."""
    if not np.array_equal(indicator_columns, indicator_columns.astype(bool)):
        raise ValueError("indicator_columns must be Boolean")
    if indicator_columns.shape != (original.shape[1],):
        raise ValueError("indicator_columns must match the feature count")
    if not np.any(indicator_columns):
        return imputed.copy()
    indicators = np.isnan(original[:, indicator_columns]).astype(float)
    return np.column_stack((imputed, indicators))


def fit_standardizer(x_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Estimate mean and scale from training features."""
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    return mean, np.where(scale == 0, 1.0, scale)


def fit_logistic(
    features: np.ndarray,
    target: np.ndarray,
    *,
    epochs: int = 2500,
    learning_rate: float = 0.08,
) -> tuple[np.ndarray, float]:
    """Fit a small logistic model with full-batch gradient descent."""
    weights = np.zeros(features.shape[1])
    bias = 0.0
    for _ in range(epochs):
        probability = sigmoid(features @ weights + bias)
        error = probability - target
        weights -= learning_rate * (features.T @ error) / len(features)
        bias -= learning_rate * float(error.mean())
    return weights, bias


def confusion_counts(
    target: np.ndarray,
    prediction: np.ndarray,
) -> tuple[int, int, int, int]:
    """Return TP, FN, FP, and TN."""
    if target.shape != prediction.shape:
        raise ValueError("target and prediction must have the same shape")
    tp = int(np.sum((target == 1) & (prediction == 1)))
    fn = int(np.sum((target == 1) & (prediction == 0)))
    fp = int(np.sum((target == 0) & (prediction == 1)))
    tn = int(np.sum((target == 0) & (prediction == 0)))
    return tp, fn, fp, tn


def classification_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, float]:
    """Calculate complementary binary-classification metrics."""
    tp, fn, fp, tn = confusion_counts(target, prediction)
    total = tp + fn + fp + tn
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    return {
        "accuracy": (tp + tn) / total,
        "balanced_accuracy": 0.5 * (recall + specificity),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
    }


def choose_threshold(
    target: np.ndarray,
    probability: np.ndarray,
) -> tuple[float, float]:
    """Choose a threshold on validation data by balanced accuracy."""
    candidates = np.linspace(0.05, 0.75, 71)
    scored = [
        (
            classification_metrics(
                target,
                (probability >= threshold).astype(int),
            )["balanced_accuracy"],
            -abs(float(threshold) - 0.5),
            float(threshold),
        )
        for threshold in candidates
    ]
    best_score, _, best_threshold = max(scored)
    return best_threshold, best_score


def evaluate_logistic_candidate(
    features: np.ndarray,
    target: np.ndarray,
    split: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    use_missing_indicator: bool,
) -> Evaluation:
    """Fit one candidate under a shared split and validation procedure."""
    train, validation, test = split

    medians = fit_median_imputer(features[train])
    indicator_columns = np.any(np.isnan(features[train]), axis=0)

    transformed = []
    for index in (train, validation, test):
        imputed = apply_median_imputer(features[index], medians)
        if use_missing_indicator:
            imputed = add_missing_indicators(
                imputed,
                features[index],
                indicator_columns,
            )
        transformed.append(imputed)

    mean, scale = fit_standardizer(transformed[0])
    x_train, x_validation, x_test = [
        (partition - mean) / scale
        for partition in transformed
    ]

    weights, bias = fit_logistic(x_train, target[train])
    validation_probability = sigmoid(x_validation @ weights + bias)
    threshold, _ = choose_threshold(
        target[validation],
        validation_probability,
    )
    test_probability = sigmoid(x_test @ weights + bias)
    metrics = classification_metrics(
        target[test],
        (test_probability >= threshold).astype(int),
    )
    return Evaluation(threshold=threshold, **metrics)


def majority_baseline(
    train_target: np.ndarray,
    test_target: np.ndarray,
) -> Evaluation:
    """Evaluate the majority class learned from training labels."""
    label, count = np.unique(train_target, return_counts=True)
    majority_label = int(label[np.argmax(count)])
    prediction = np.full(test_target.shape, majority_label, dtype=int)
    metrics = classification_metrics(test_target, prediction)
    return Evaluation(threshold=float("nan"), **metrics)


def print_evaluation(name: str, result: Evaluation) -> None:
    """Print one compact, comparable results row."""
    threshold = "n/a" if np.isnan(result.threshold) else f"{result.threshold:.2f}"
    print(
        f"{name:28s} threshold={threshold:>4s} "
        f"accuracy={result.accuracy:.3f} "
        f"balanced_accuracy={result.balanced_accuracy:.3f} "
        f"precision={result.precision:.3f} "
        f"recall={result.recall:.3f}"
    )


def main() -> None:
    features, target = make_rare_event_data()
    split = stratified_split(target)
    train, validation, test = split

    baseline = majority_baseline(target[train], target[test])
    imputed_only = evaluate_logistic_candidate(
        features,
        target,
        split,
        use_missing_indicator=False,
    )
    with_indicator = evaluate_logistic_candidate(
        features,
        target,
        split,
        use_missing_indicator=True,
    )

    print("CME705 Week 3: data quality and model comparison")
    print("Synthetic case: rare-event screening with one incomplete feature")
    print(
        f"Rows={len(target)}, positive_prevalence={target.mean():.3f}, "
        f"missing_cell_rate={np.isnan(features).mean():.3f}, "
        f"feature_1_missing_rate={np.isnan(features[:, 1]).mean():.3f}"
    )
    print(
        "Split rows: "
        f"train={len(train)}, validation={len(validation)}, test={len(test)}"
    )
    print("All candidates use the same split and training-only preprocessing.")
    print()
    print_evaluation("Majority baseline", baseline)
    print_evaluation("Median-imputed logistic", imputed_only)
    print_evaluation("Imputation + indicator", with_indicator)
    print()
    print("Interpretation")
    print(
        "  Accuracy makes the majority rule look strong because positive events "
        "are rare, but its recall and balanced accuracy expose failure."
    )
    print(
        "  The missing-indicator comparison changes one documented feature "
        "choice under the same evaluation protocol."
    )
    print(
        "  A useful indicator in this simulation does not prove that missingness "
        "will be stable, appropriate, or safe in another dataset."
    )


if __name__ == "__main__":
    main()
