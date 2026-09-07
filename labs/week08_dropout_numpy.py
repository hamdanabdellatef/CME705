"""Deep-network generalization and inverted dropout with NumPy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Parameters = dict[str, np.ndarray]
Gradients = dict[str, np.ndarray]


@dataclass(frozen=True)
class GeneralizationData:
    """Training labels include controlled noise; validation and test stay clean."""

    train_x: np.ndarray
    train_y: np.ndarray
    validation_x: np.ndarray
    validation_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray
    flipped_training_labels: np.ndarray


@dataclass(frozen=True)
class ForwardCache:
    """Values and masks required by the backward pass."""

    x: np.ndarray
    z1: np.ndarray
    h1: np.ndarray
    h1_dropped: np.ndarray
    mask1: np.ndarray
    z2: np.ndarray
    h2: np.ndarray
    h2_dropped: np.ndarray
    mask2: np.ndarray
    logits: np.ndarray
    probability: np.ndarray


@dataclass(frozen=True)
class TrainingResult:
    """Best validation checkpoint and complete learning curves."""

    parameters: Parameters
    train_loss: np.ndarray
    validation_loss: np.ndarray
    best_epoch: int
    stopped_epoch: int
    best_train_loss: float
    best_validation_loss: float
    final_validation_loss: float


@dataclass(frozen=True)
class Evaluation:
    """Binary evaluation evidence."""

    loss: float
    accuracy: float
    positive_recall: float
    negative_recall: float
    confusion: np.ndarray


def sigmoid(values: np.ndarray) -> np.ndarray:
    """Compute a stable sigmoid."""
    clipped = np.clip(np.asarray(values, dtype=float), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def binary_cross_entropy_from_logits(
    target: np.ndarray, logits: np.ndarray
) -> float:
    """Compute stable mean binary cross-entropy."""
    target = np.asarray(target, dtype=float)
    logits = np.asarray(logits, dtype=float)
    return float(np.mean(np.logaddexp(0.0, logits) - target * logits))


def relu(pre_activation: np.ndarray) -> np.ndarray:
    """Apply the rectified linear unit."""
    return np.maximum(np.asarray(pre_activation, dtype=float), 0.0)


def relu_backward(
    pre_activation: np.ndarray, upstream_gradient: np.ndarray
) -> np.ndarray:
    """Apply ReLU's local derivative to an upstream gradient."""
    return np.asarray(upstream_gradient) * (np.asarray(pre_activation) > 0.0)


def inverted_dropout(
    activation: np.ndarray,
    drop_probability: float,
    rng: np.random.Generator,
    *,
    training: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply inverted dropout and return both output and reusable mask."""
    if not 0.0 <= drop_probability < 1.0:
        raise ValueError("drop_probability must satisfy 0 <= p < 1")
    activation = np.asarray(activation, dtype=float)
    if not training or drop_probability == 0.0:
        return activation.copy(), np.ones_like(activation)
    keep_probability = 1.0 - drop_probability
    mask = (
        rng.random(activation.shape) < keep_probability
    ) / keep_probability
    return activation * mask, mask


def sigmoid_dropout_backward(
    sigmoid_activation: np.ndarray,
    upstream_gradient: np.ndarray,
    dropout_mask: np.ndarray,
) -> np.ndarray:
    """Differentiate sigmoid before applying the saved dropout mask."""
    return (
        upstream_gradient
        * dropout_mask
        * sigmoid_activation
        * (1.0 - sigmoid_activation)
    )


def make_generalization_data(
    *,
    seed: int = 705,
    training_size: int = 96,
    validation_size: int = 240,
    test_size: int = 240,
    nuisance_features: int = 30,
    training_label_noise: float = 0.20,
) -> GeneralizationData:
    """Create a nonlinear task with nuisance features and noisy training labels."""
    if not 0.0 <= training_label_noise < 0.5:
        raise ValueError("training_label_noise must satisfy 0 <= p < 0.5")
    rng = np.random.default_rng(seed)

    def sample(size: int) -> tuple[np.ndarray, np.ndarray]:
        signal = rng.uniform(-2.0, 2.0, size=(size, 2))
        score = (
            np.sin(1.7 * signal[:, 0])
            + 0.65 * signal[:, 1]
            - 0.15 * signal[:, 0] * signal[:, 1]
        )
        target = (score > 0.0).astype(float).reshape(-1, 1)
        nuisance = rng.normal(0.0, 1.0, size=(size, nuisance_features))
        return np.column_stack([signal, nuisance]), target

    train_x, clean_train_y = sample(training_size)
    validation_x, validation_y = sample(validation_size)
    test_x, test_y = sample(test_size)
    train_y = clean_train_y.copy()
    number_flipped = int(round(training_label_noise * training_size))
    flipped = np.sort(rng.choice(training_size, number_flipped, replace=False))
    train_y[flipped] = 1.0 - train_y[flipped]
    return GeneralizationData(
        train_x,
        train_y,
        validation_x,
        validation_y,
        test_x,
        test_y,
        flipped,
    )


def fit_standardizer(
    features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit feature scaling on training data only."""
    mean = np.asarray(features, dtype=float).mean(axis=0)
    scale = np.asarray(features, dtype=float).std(axis=0)
    return mean, np.where(scale == 0.0, 1.0, scale)


def apply_standardizer(
    features: np.ndarray, mean: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    """Apply fixed training-set scaling values."""
    return (np.asarray(features, dtype=float) - mean) / scale


def initialize_parameters(
    input_features: int,
    *,
    hidden_widths: tuple[int, int] = (64, 32),
    seed: int = 17,
) -> Parameters:
    """Use reproducible He initialization for two ReLU hidden layers."""
    first, second = hidden_widths
    if min(input_features, first, second) <= 0:
        raise ValueError("layer widths must be positive")
    rng = np.random.default_rng(seed)
    return {
        "w1": rng.normal(0.0, np.sqrt(2.0 / input_features), (input_features, first)),
        "b1": np.zeros((1, first)),
        "w2": rng.normal(0.0, np.sqrt(2.0 / first), (first, second)),
        "b2": np.zeros((1, second)),
        "w3": rng.normal(0.0, np.sqrt(2.0 / second), (second, 1)),
        "b3": np.zeros((1, 1)),
    }


def forward(
    features: np.ndarray,
    parameters: Parameters,
    *,
    drop_probability: float = 0.0,
    rng: np.random.Generator | None = None,
    training: bool = False,
) -> ForwardCache:
    """Run two ReLU layers and one sigmoid output."""
    if training and drop_probability > 0.0 and rng is None:
        raise ValueError("training with dropout requires an RNG")
    if rng is None:
        rng = np.random.default_rng(0)
    x = np.asarray(features, dtype=float)
    z1 = x @ parameters["w1"] + parameters["b1"]
    h1 = relu(z1)
    h1_dropped, mask1 = inverted_dropout(
        h1, drop_probability, rng, training=training
    )
    z2 = h1_dropped @ parameters["w2"] + parameters["b2"]
    h2 = relu(z2)
    h2_dropped, mask2 = inverted_dropout(
        h2, drop_probability, rng, training=training
    )
    logits = h2_dropped @ parameters["w3"] + parameters["b3"]
    probability = sigmoid(logits)
    return ForwardCache(
        x,
        z1,
        h1,
        h1_dropped,
        mask1,
        z2,
        h2,
        h2_dropped,
        mask2,
        logits,
        probability,
    )


def backward(
    target: np.ndarray,
    parameters: Parameters,
    cache: ForwardCache,
) -> Gradients:
    """Backpropagate through sigmoid, ReLU, and saved dropout masks."""
    target = np.asarray(target, dtype=float)
    examples = len(target)
    gradient_logits = (cache.probability - target) / examples
    gradient_w3 = cache.h2_dropped.T @ gradient_logits
    gradient_b3 = gradient_logits.sum(axis=0, keepdims=True)

    gradient_h2_dropped = gradient_logits @ parameters["w3"].T
    gradient_h2 = gradient_h2_dropped * cache.mask2
    gradient_z2 = relu_backward(cache.z2, gradient_h2)
    gradient_w2 = cache.h1_dropped.T @ gradient_z2
    gradient_b2 = gradient_z2.sum(axis=0, keepdims=True)

    gradient_h1_dropped = gradient_z2 @ parameters["w2"].T
    gradient_h1 = gradient_h1_dropped * cache.mask1
    gradient_z1 = relu_backward(cache.z1, gradient_h1)
    gradient_w1 = cache.x.T @ gradient_z1
    gradient_b1 = gradient_z1.sum(axis=0, keepdims=True)
    return {
        "w1": gradient_w1,
        "b1": gradient_b1,
        "w2": gradient_w2,
        "b2": gradient_b2,
        "w3": gradient_w3,
        "b3": gradient_b3,
    }


def apply_update(
    parameters: Parameters, gradients: Gradients, learning_rate: float
) -> Parameters:
    """Return a gradient-descent parameter update."""
    return {
        name: value - learning_rate * gradients[name]
        for name, value in parameters.items()
    }


def deterministic_loss(
    features: np.ndarray, target: np.ndarray, parameters: Parameters
) -> float:
    """Evaluate without dropout."""
    return binary_cross_entropy_from_logits(
        target, forward(features, parameters, training=False).logits
    )


def train_mlp(
    training_features: np.ndarray,
    training_target: np.ndarray,
    validation_features: np.ndarray,
    validation_target: np.ndarray,
    *,
    drop_probability: float,
    learning_rate: float = 0.04,
    maximum_epochs: int = 1800,
    patience: int = 180,
    initialization_seed: int = 17,
    dropout_seed: int = 29,
) -> TrainingResult:
    """Train with one dropout rate and restore the best validation checkpoint."""
    parameters = initialize_parameters(
        training_features.shape[1], seed=initialization_seed
    )
    rng = np.random.default_rng(dropout_seed)
    training_curve: list[float] = []
    validation_curve: list[float] = []
    best_validation = np.inf
    best_parameters = {name: value.copy() for name, value in parameters.items()}
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(1, maximum_epochs + 1):
        cache = forward(
            training_features,
            parameters,
            drop_probability=drop_probability,
            rng=rng,
            training=True,
        )
        gradients = backward(training_target, parameters, cache)
        parameters = apply_update(parameters, gradients, learning_rate)
        training_loss = deterministic_loss(
            training_features, training_target, parameters
        )
        validation_loss = deterministic_loss(
            validation_features, validation_target, parameters
        )
        training_curve.append(training_loss)
        validation_curve.append(validation_loss)

        if validation_loss < best_validation - 1e-5:
            best_validation = validation_loss
            best_parameters = {
                name: value.copy() for name, value in parameters.items()
            }
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break

    return TrainingResult(
        parameters=best_parameters,
        train_loss=np.asarray(training_curve),
        validation_loss=np.asarray(validation_curve),
        best_epoch=best_epoch,
        stopped_epoch=len(training_curve),
        best_train_loss=float(training_curve[best_epoch - 1]),
        best_validation_loss=float(best_validation),
        final_validation_loss=float(validation_curve[-1]),
    )


def evaluate(
    features: np.ndarray, target: np.ndarray, parameters: Parameters
) -> Evaluation:
    """Compute deterministic binary loss and class-aware counts."""
    cache = forward(features, parameters, training=False)
    prediction = (cache.probability >= 0.5).astype(int)
    target_int = np.asarray(target, dtype=int)
    confusion = np.zeros((2, 2), dtype=int)
    np.add.at(confusion, (target_int.ravel(), prediction.ravel()), 1)
    negative_recall = confusion[0, 0] / confusion[0].sum()
    positive_recall = confusion[1, 1] / confusion[1].sum()
    return Evaluation(
        loss=binary_cross_entropy_from_logits(target, cache.logits),
        accuracy=float(np.mean(prediction == target_int)),
        positive_recall=float(positive_recall),
        negative_recall=float(negative_recall),
        confusion=confusion,
    )


def run_controlled_comparison(
    *,
    dropout_rates: tuple[float, ...] = (0.0, 0.15, 0.30, 0.45),
) -> tuple[
    GeneralizationData,
    dict[float, TrainingResult],
    float,
    Evaluation,
    Evaluation,
]:
    """Select dropout on validation data, then open the test split once."""
    data = make_generalization_data()
    mean, scale = fit_standardizer(data.train_x)
    training_x = apply_standardizer(data.train_x, mean, scale)
    validation_x = apply_standardizer(data.validation_x, mean, scale)
    test_x = apply_standardizer(data.test_x, mean, scale)

    runs = {
        rate: train_mlp(
            training_x,
            data.train_y,
            validation_x,
            data.validation_y,
            drop_probability=rate,
        )
        for rate in dropout_rates
    }
    selected = min(
        dropout_rates,
        key=lambda rate: runs[rate].best_validation_loss,
    )
    reference_test = evaluate(
        test_x, data.test_y, runs[0.0].parameters
    )
    selected_test = evaluate(
        test_x, data.test_y, runs[selected].parameters
    )
    return data, runs, selected, reference_test, selected_test


def main() -> None:
    mechanism_rng = np.random.default_rng(705)
    activation = np.full(200_000, 0.8)
    dropped, mask = inverted_dropout(
        activation, 0.25, mechanism_rng, training=True
    )
    evaluation_output, evaluation_mask = inverted_dropout(
        activation,
        0.25,
        np.random.default_rng(1),
        training=False,
    )
    upstream = np.ones_like(activation)
    backward_gradient = sigmoid_dropout_backward(
        activation, upstream, mask
    )

    data, runs, selected, reference_test, selected_test = (
        run_controlled_comparison()
    )

    print("inverted_dropout_mechanics")
    print(f"activation_mean_before={activation.mean():.4f}")
    print(f"activation_mean_training={dropped.mean():.4f}")
    print(f"dropped_fraction={np.mean(mask == 0.0):.4f}")
    print(
        "evaluation_max_difference="
        f"{np.max(np.abs(evaluation_output - activation)):.3e}"
    )
    print(
        "evaluation_mask_all_ones="
        f"{bool(np.all(evaluation_mask == 1.0))}"
    )
    print(f"backward_gradient_mean={backward_gradient.mean():.4f}")
    print(
        "dropped_gradient_nonzero="
        f"{int(np.count_nonzero(backward_gradient[mask == 0.0]))}"
    )

    print("\ncontrolled_validation_comparison")
    print(
        f"training_rows={len(data.train_y)} "
        f"flipped_training_labels={len(data.flipped_training_labels)}"
    )
    for rate, result in runs.items():
        gap = result.best_validation_loss - result.best_train_loss
        print(
            f"dropout={rate:.2f} "
            f"best_epoch={result.best_epoch} "
            f"stopped_epoch={result.stopped_epoch} "
            f"train_loss={result.best_train_loss:.4f} "
            f"validation_loss={result.best_validation_loss:.4f} "
            f"gap={gap:.4f} "
            f"last_validation_loss={result.final_validation_loss:.4f}"
        )
    print(f"selected_dropout={selected:.2f}")

    print("\nlocked_test_evaluation")
    print(
        f"reference_dropout=0.00 "
        f"loss={reference_test.loss:.4f} "
        f"accuracy={reference_test.accuracy:.3f}"
    )
    print(
        f"selected_dropout={selected:.2f} "
        f"loss={selected_test.loss:.4f} "
        f"accuracy={selected_test.accuracy:.3f} "
        f"negative_recall={selected_test.negative_recall:.3f} "
        f"positive_recall={selected_test.positive_recall:.3f}"
    )
    print("selected_confusion_rows_true_columns_predicted")
    print(selected_test.confusion)


if __name__ == "__main__":
    main()
