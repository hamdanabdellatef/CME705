"""Stable multiclass learning and evaluation with NumPy."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class DataSplit:
    """Disjoint indices for training, validation, and final testing."""
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


@dataclass(frozen=True)
class LinearSoftmaxModel:
    """Parameters and objective history for multinomial logistic regression."""
    weights: np.ndarray
    bias: np.ndarray
    loss_history: np.ndarray


@dataclass(frozen=True)
class ClassificationReport:
    """Class-aware evaluation evidence."""
    accuracy: float
    macro_recall: float
    cross_entropy: float
    confusion: np.ndarray


def log_softmax(logits: np.ndarray) -> np.ndarray:
    """Return stable row-wise log probabilities."""
    logits = np.asarray(logits, dtype=float)
    if logits.ndim != 2:
        raise ValueError("logits must have shape (examples, classes)")
    if logits.shape[1] < 2:
        raise ValueError("multiclass logits require at least two classes")
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_normalizer = np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    return shifted - log_normalizer


def softmax(logits: np.ndarray) -> np.ndarray:
    """Return stable row-wise class probabilities."""
    return np.exp(log_softmax(logits))


def one_hot(class_index: np.ndarray, number_of_classes: int) -> np.ndarray:
    """Encode integer class indices as one-hot rows."""
    class_index = np.asarray(class_index, dtype=int)
    if class_index.ndim != 1:
        raise ValueError("class_index must be one-dimensional")
    if number_of_classes < 2:
        raise ValueError("number_of_classes must be at least two")
    if len(class_index) and (
        class_index.min() < 0 or class_index.max() >= number_of_classes
    ):
        raise ValueError("class index outside the declared class range")
    encoded = np.zeros((len(class_index), number_of_classes), dtype=float)
    encoded[np.arange(len(class_index)), class_index] = 1.0
    return encoded


def cross_entropy(logits: np.ndarray, class_index: np.ndarray) -> float:
    """Compute mean categorical cross-entropy from class indices."""
    class_index = np.asarray(class_index, dtype=int)
    log_probability = log_softmax(logits)
    if class_index.shape != (len(log_probability),):
        raise ValueError("one class index is required for each example")
    return float(
        -log_probability[np.arange(len(class_index)), class_index].mean()
    )


def cross_entropy_one_hot(
    logits: np.ndarray, encoded_target: np.ndarray
) -> float:
    """Compute the same objective from one-hot targets."""
    log_probability = log_softmax(logits)
    encoded_target = np.asarray(encoded_target, dtype=float)
    if encoded_target.shape != log_probability.shape:
        raise ValueError("one-hot targets must match the logit shape")
    return float(-np.sum(encoded_target * log_probability, axis=1).mean())


def cross_entropy_logit_gradient(
    logits: np.ndarray, class_index: np.ndarray
) -> np.ndarray:
    """Differentiate mean softmax cross-entropy with respect to logits."""
    probability = softmax(logits)
    target = one_hot(class_index, probability.shape[1])
    return (probability - target) / len(probability)


def linear_logits(
    features: np.ndarray, weights: np.ndarray, bias: np.ndarray
) -> np.ndarray:
    """Compute affine class scores."""
    return np.asarray(features, dtype=float) @ weights + bias


def linear_gradients(
    features: np.ndarray,
    class_index: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return analytical weight and bias gradients."""
    gradient_logits = cross_entropy_logit_gradient(
        linear_logits(features, weights, bias), class_index
    )
    return (
        features.T @ gradient_logits,
        gradient_logits.sum(axis=0, keepdims=True),
    )


def gradient_check(
    features: np.ndarray,
    class_index: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    *,
    epsilon: float = 1e-5,
) -> tuple[float, dict[str, float]]:
    """Compare every analytical parameter gradient with centered differences."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    analytical_weights, analytical_bias = linear_gradients(
        features, class_index, weights, bias
    )
    analytical = {"weights": analytical_weights, "bias": analytical_bias}
    parameters = {"weights": weights, "bias": bias}
    errors: dict[str, float] = {}
    for name, value in parameters.items():
        numerical = np.zeros_like(value)
        for index in np.ndindex(value.shape):
            plus_weights, minus_weights = weights.copy(), weights.copy()
            plus_bias, minus_bias = bias.copy(), bias.copy()
            if name == "weights":
                plus_weights[index] += epsilon
                minus_weights[index] -= epsilon
            else:
                plus_bias[index] += epsilon
                minus_bias[index] -= epsilon
            plus_loss = cross_entropy(
                linear_logits(features, plus_weights, plus_bias), class_index
            )
            minus_loss = cross_entropy(
                linear_logits(features, minus_weights, minus_bias), class_index
            )
            numerical[index] = (plus_loss - minus_loss) / (2.0 * epsilon)
        denominator = np.maximum(
            1e-8, np.abs(analytical[name]) + np.abs(numerical)
        )
        errors[name] = float(
            np.max(np.abs(analytical[name] - numerical) / denominator)
        )
    return max(errors.values()), errors


def make_multiclass_data(
    *, seed: int = 705
) -> tuple[np.ndarray, np.ndarray]:
    """Create an imbalanced three-class teaching dataset."""
    rng = np.random.default_rng(seed)
    sizes = (150, 90, 60)
    centers = np.array([[-1.7, -0.3], [1.1, 1.0], [1.2, -1.4]])
    scales = np.array([[1.05, 0.85], [0.95, 1.0], [0.9, 0.9]])
    blocks = [
        rng.normal(center, scale, size=(size, 2))
        for size, center, scale in zip(sizes, centers, scales)
    ]
    features = np.vstack(blocks)
    target = np.concatenate(
        [np.full(size, label) for label, size in enumerate(sizes)]
    )
    return features, target


def stratified_split(
    class_index: np.ndarray, *, seed: int = 705
) -> DataSplit:
    """Split every class into 60% training, 20% validation, and 20% test."""
    class_index = np.asarray(class_index, dtype=int)
    rng = np.random.default_rng(seed)
    partitions = {"train": [], "validation": [], "test": []}
    for label in np.unique(class_index):
        indices = np.flatnonzero(class_index == label)
        rng.shuffle(indices)
        first = int(round(0.6 * len(indices)))
        second = first + int(round(0.2 * len(indices)))
        partitions["train"].append(indices[:first])
        partitions["validation"].append(indices[first:second])
        partitions["test"].append(indices[second:])
    return DataSplit(
        *(np.sort(np.concatenate(partitions[name])) for name in partitions)
    )


def fit_standardizer(
    features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit centering and scaling values on training features only."""
    mean = np.asarray(features, dtype=float).mean(axis=0)
    scale = np.asarray(features, dtype=float).std(axis=0)
    return mean, np.where(scale == 0.0, 1.0, scale)


def apply_standardizer(
    features: np.ndarray, mean: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    """Apply previously fitted standardization values."""
    return (np.asarray(features, dtype=float) - mean) / scale


def train_softmax_regression(
    features: np.ndarray,
    class_index: np.ndarray,
    *,
    number_of_classes: int,
    learning_rate: float,
    epochs: int = 300,
) -> LinearSoftmaxModel:
    """Fit a full-batch linear softmax classifier from zero initialization."""
    if learning_rate <= 0 or epochs <= 0:
        raise ValueError("learning_rate and epochs must be positive")
    weights = np.zeros((features.shape[1], number_of_classes))
    bias = np.zeros((1, number_of_classes))
    history = [
        cross_entropy(linear_logits(features, weights, bias), class_index)
    ]
    for _ in range(epochs):
        gradient_weights, gradient_bias = linear_gradients(
            features, class_index, weights, bias
        )
        weights -= learning_rate * gradient_weights
        bias -= learning_rate * gradient_bias
        history.append(
            cross_entropy(
                linear_logits(features, weights, bias), class_index
            )
        )
    return LinearSoftmaxModel(weights, bias, np.asarray(history))


def predict(
    features: np.ndarray, model: LinearSoftmaxModel
) -> tuple[np.ndarray, np.ndarray]:
    """Return probabilities and maximum-probability classes."""
    probability = softmax(
        linear_logits(features, model.weights, model.bias)
    )
    return probability, probability.argmax(axis=1)


def confusion_matrix(
    target: np.ndarray, prediction: np.ndarray, number_of_classes: int
) -> np.ndarray:
    """Count rows as true classes and columns as predicted classes."""
    matrix = np.zeros((number_of_classes, number_of_classes), dtype=int)
    np.add.at(
        matrix,
        (np.asarray(target, dtype=int), np.asarray(prediction, dtype=int)),
        1,
    )
    return matrix


def classification_report(
    features: np.ndarray,
    target: np.ndarray,
    model: LinearSoftmaxModel,
) -> ClassificationReport:
    """Evaluate loss, accuracy, macro recall, and class-specific counts."""
    probability, prediction = predict(features, model)
    matrix = confusion_matrix(target, prediction, probability.shape[1])
    recall = np.divide(
        np.diag(matrix),
        matrix.sum(axis=1),
        out=np.zeros(probability.shape[1], dtype=float),
        where=matrix.sum(axis=1) != 0,
    )
    return ClassificationReport(
        accuracy=float(np.mean(prediction == target)),
        macro_recall=float(recall.mean()),
        cross_entropy=cross_entropy(
            linear_logits(features, model.weights, model.bias), target
        ),
        confusion=matrix,
    )


def select_learning_rate(
    training_features: np.ndarray,
    training_target: np.ndarray,
    validation_features: np.ndarray,
    validation_target: np.ndarray,
    candidates: tuple[float, ...],
    *,
    epochs: int = 300,
) -> tuple[float, dict[float, LinearSoftmaxModel], dict[float, float]]:
    """Select one rate using validation cross-entropy without reading test data."""
    if not candidates:
        raise ValueError("at least one learning-rate candidate is required")
    number_of_classes = int(training_target.max()) + 1
    models: dict[float, LinearSoftmaxModel] = {}
    validation_loss: dict[float, float] = {}
    for rate in candidates:
        model = train_softmax_regression(
            training_features,
            training_target,
            number_of_classes=number_of_classes,
            learning_rate=rate,
            epochs=epochs,
        )
        models[rate] = model
        validation_loss[rate] = cross_entropy(
            linear_logits(
                validation_features, model.weights, model.bias
            ),
            validation_target,
        )
    selected = min(candidates, key=validation_loss.__getitem__)
    return selected, models, validation_loss


def majority_baseline(
    training_target: np.ndarray, size: int
) -> np.ndarray:
    """Predict the most frequent training class."""
    majority = int(np.bincount(training_target).argmax())
    return np.full(size, majority)


def main() -> None:
    extreme = np.array(
        [[1002.0, 1001.0, 998.0], [-1000.0, -999.0, -1003.0]]
    )
    extreme_target = np.array([0, 1])
    probability = softmax(extreme)
    shifted = extreme + np.array([[5000.0], [-3000.0]])
    shift_difference = np.max(np.abs(probability - softmax(shifted)))

    check_features = np.array(
        [[0.2, -1.0], [1.5, 0.4], [-0.7, 2.0], [0.3, 0.8]]
    )
    check_target = np.array([0, 1, 2, 1])
    check_weights = np.array(
        [[0.2, -0.1, 0.3], [-0.4, 0.5, 0.1]]
    )
    check_bias = np.array([[0.1, -0.2, 0.05]])
    maximum_error, parameter_errors = gradient_check(
        check_features, check_target, check_weights, check_bias
    )

    features, target = make_multiclass_data()
    split = stratified_split(target)
    mean, scale = fit_standardizer(features[split.train])
    standardized = apply_standardizer(features, mean, scale)
    candidates = (0.01, 0.05, 0.2, 0.8)
    selected, models, validation_loss = select_learning_rate(
        standardized[split.train],
        target[split.train],
        standardized[split.validation],
        target[split.validation],
        candidates,
    )
    report = classification_report(
        standardized[split.test], target[split.test], models[selected]
    )
    baseline_prediction = majority_baseline(
        target[split.train], len(split.test)
    )
    baseline_confusion = confusion_matrix(
        target[split.test], baseline_prediction, 3
    )
    baseline_accuracy = float(
        np.mean(baseline_prediction == target[split.test])
    )
    baseline_recall = np.divide(
        np.diag(baseline_confusion),
        baseline_confusion.sum(axis=1),
        out=np.zeros(3, dtype=float),
        where=baseline_confusion.sum(axis=1) != 0,
    ).mean()

    print("stable_softmax")
    print(f"probabilities={np.array2string(probability, precision=4)}")
    print(
        "row_sums="
        f"{np.array2string(probability.sum(axis=1), precision=4)}"
    )
    print(f"largest_shift_difference={shift_difference:.3e}")
    print(f"cross_entropy={cross_entropy(extreme, extreme_target):.4f}")

    print("\ngradient_check_relative_error")
    for name in ("weights", "bias"):
        print(f"{name}={parameter_errors[name]:.3e}")
    print(f"maximum={maximum_error:.3e}")

    print("\nvalidation_selection")
    for rate in candidates:
        print(
            f"learning_rate={rate:.2f} "
            f"validation_loss={validation_loss[rate]:.4f}"
        )
    print(f"selected_learning_rate={selected:.2f}")

    print("\nlocked_test_evaluation")
    print(f"class_counts={np.bincount(target[split.test]).tolist()}")
    print(
        f"majority_accuracy={baseline_accuracy:.3f} "
        f"majority_macro_recall={baseline_recall:.3f}"
    )
    print(
        f"model_accuracy={report.accuracy:.3f} "
        f"model_macro_recall={report.macro_recall:.3f} "
        f"model_cross_entropy={report.cross_entropy:.4f}"
    )
    print("confusion_rows_true_columns_predicted")
    print(report.confusion)


if __name__ == "__main__":
    main()
