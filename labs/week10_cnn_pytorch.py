"""Week 10: a reproducible PyTorch CNN experiment on MNIST."""

from __future__ import annotations

import argparse
import os
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


CLASS_NAMES = tuple(str(value) for value in range(10))


@dataclass(frozen=True)
class MNISTData:
    train_images: torch.Tensor
    train_labels: torch.Tensor
    validation_images: torch.Tensor
    validation_labels: torch.Tensor
    test_images: torch.Tensor
    test_labels: torch.Tensor


@dataclass(frozen=True)
class EpochRecord:
    epoch: int
    training_loss: float
    validation_loss: float
    validation_accuracy: float


@dataclass(frozen=True)
class TrainingResult:
    history: tuple[EpochRecord, ...]
    best_epoch: int
    stopped_epoch: int
    best_validation_loss: float
    elapsed_seconds: float


@dataclass(frozen=True)
class Evaluation:
    loss: float
    accuracy: float
    confusion: np.ndarray
    recall: np.ndarray
    predictions: torch.Tensor
    confidence: torch.Tensor


def resolve_device(requested: str = "auto") -> torch.device:
    """Select CUDA when available while retaining a portable CPU path."""
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but this PyTorch build cannot access CUDA. "
            "Install a CUDA-enabled build from pytorch.org/get-started/locally/."
        )
    return device


def configure_reproducibility(seed: int = 705) -> None:
    """Seed the checked teaching path and disable nondeterministic choices."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.use_deterministic_algorithms(True)


def stratified_subset_indices(
    labels: torch.Tensor,
    *,
    training_per_class: int | None = 2000,
    validation_per_class: int = 500,
    seed: int = 705,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a repeatable class-balanced subset of the official train split."""
    if labels.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    if validation_per_class <= 0:
        raise ValueError("validation_per_class must be positive")
    if training_per_class is not None and training_per_class <= 0:
        raise ValueError("training_per_class must be positive or None")

    generator = torch.Generator().manual_seed(seed)
    training_parts: list[torch.Tensor] = []
    validation_parts: list[torch.Tensor] = []
    for class_index in range(len(CLASS_NAMES)):
        indices = torch.nonzero(labels == class_index, as_tuple=False).flatten()
        required = validation_per_class + (training_per_class or 1)
        if len(indices) < required:
            raise ValueError(
                f"class {class_index} has too few examples for the split"
            )
        shuffled = indices[
            torch.randperm(len(indices), generator=generator)
        ]
        validation_parts.append(shuffled[:validation_per_class])
        remaining = shuffled[validation_per_class:]
        training_parts.append(
            remaining
            if training_per_class is None
            else remaining[:training_per_class]
        )

    training = torch.cat(training_parts)
    validation = torch.cat(validation_parts)
    training = training[
        torch.randperm(len(training), generator=generator)
    ]
    validation = validation[
        torch.randperm(len(validation), generator=generator)
    ]
    return training, validation


def fit_image_standardizer(images: torch.Tensor) -> tuple[float, float]:
    """Fit scalar normalization from training images only."""
    mean = float(images.mean())
    scale = float(images.std(unbiased=False))
    if scale <= 0:
        raise ValueError("training image scale must be positive")
    return mean, scale


def apply_image_standardizer(
    images: torch.Tensor,
    mean: float,
    scale: float,
) -> torch.Tensor:
    return (images - mean) / scale


def load_mnist_data(
    root: str | Path = "data/raw",
    *,
    download: bool = True,
    training_per_class: int | None = 2000,
    validation_per_class: int = 500,
    seed: int = 705,
) -> tuple[MNISTData, float, float]:
    """Load MNIST, derive validation from train, and keep test locked."""
    try:
        from torchvision.datasets import MNIST
    except ImportError as error:
        raise RuntimeError(
            "TorchVision is required. Install the course requirements first."
        ) from error

    source_train = MNIST(root=root, train=True, download=download)
    source_test = MNIST(root=root, train=False, download=download)
    training_indices, validation_indices = stratified_subset_indices(
        source_train.targets,
        training_per_class=training_per_class,
        validation_per_class=validation_per_class,
        seed=seed,
    )

    source_images = source_train.data.unsqueeze(1).to(torch.float32) / 255.0
    test_images = source_test.data.unsqueeze(1).to(torch.float32) / 255.0
    training_images = source_images[training_indices]
    validation_images = source_images[validation_indices]
    training_labels = source_train.targets[training_indices].to(torch.int64)
    validation_labels = source_train.targets[validation_indices].to(torch.int64)
    test_labels = source_test.targets.to(torch.int64)

    mean, scale = fit_image_standardizer(training_images)
    data = MNISTData(
        apply_image_standardizer(training_images, mean, scale),
        training_labels,
        apply_image_standardizer(validation_images, mean, scale),
        validation_labels,
        apply_image_standardizer(test_images, mean, scale),
        test_labels,
    )
    return data, mean, scale


class SmallCNN(nn.Module):
    """A compact CNN for 28 by 28 one-channel digit images."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.hidden = nn.Linear(32 * 7 * 7, 64)
        self.relu3 = nn.ReLU()
        self.classifier = nn.Linear(64, num_classes)

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        features = self.pool1(self.relu1(self.conv1(images)))
        return self.pool2(self.relu2(self.conv2(features)))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(images)
        hidden = self.relu3(self.hidden(torch.flatten(features, 1)))
        return self.classifier(hidden)


def model_device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def trace_shapes(
    model: SmallCNN,
    images: torch.Tensor,
) -> dict[str, tuple[int, ...]]:
    """Trace the explicit Week 9 to Week 10 tensor path."""
    value = images.to(model_device(model))
    trace: dict[str, tuple[int, ...]] = {"input": tuple(value.shape)}
    value = model.conv1(value)
    trace["conv1"] = tuple(value.shape)
    value = model.relu1(value)
    trace["relu1"] = tuple(value.shape)
    value = model.pool1(value)
    trace["max_pool1"] = tuple(value.shape)
    value = model.conv2(value)
    trace["conv2"] = tuple(value.shape)
    value = model.relu2(value)
    trace["relu2"] = tuple(value.shape)
    value = model.pool2(value)
    trace["max_pool2"] = tuple(value.shape)
    value = torch.flatten(value, 1)
    trace["flatten"] = tuple(value.shape)
    value = model.hidden(value)
    trace["hidden"] = tuple(value.shape)
    value = model.relu3(value)
    trace["relu3"] = tuple(value.shape)
    value = model.classifier(value)
    trace["logits"] = tuple(value.shape)
    return trace


def one_training_step(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    """Run one update while allowing a supplied optimizer to retain state."""
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    device = model_device(model)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = nn.functional.cross_entropy(
        model(images.to(device)), labels.to(device)
    )
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu())


def predict_model(
    model: nn.Module,
    images: torch.Tensor,
    *,
    batch_size: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return CPU predictions and confidence without building a graph."""
    device = model_device(model)
    loader = DataLoader(
        TensorDataset(images),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )
    predictions_parts: list[torch.Tensor] = []
    confidence_parts: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for (batch_images,) in loader:
            logits = model(
                batch_images.to(device, non_blocking=device.type == "cuda")
            )
            confidence, predictions = logits.softmax(dim=1).max(dim=1)
            predictions_parts.append(predictions.cpu())
            confidence_parts.append(confidence.cpu())
    return torch.cat(predictions_parts), torch.cat(confidence_parts)


def evaluate_model(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    batch_size: int = 512,
) -> Evaluation:
    """Evaluate in batches without storing an autograd graph."""
    device = model_device(model)
    loader = DataLoader(
        TensorDataset(images, labels),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )
    predictions_parts: list[torch.Tensor] = []
    confidence_parts: list[torch.Tensor] = []
    total_loss = 0.0
    model.eval()
    with torch.no_grad():
        for batch_images, batch_labels in loader:
            batch_images = batch_images.to(
                device, non_blocking=device.type == "cuda"
            )
            device_labels = batch_labels.to(
                device, non_blocking=device.type == "cuda"
            )
            logits = model(batch_images)
            total_loss += float(
                nn.functional.cross_entropy(
                    logits, device_labels, reduction="sum"
                ).cpu()
            )
            confidence, predictions = logits.softmax(dim=1).max(dim=1)
            predictions_parts.append(predictions.cpu())
            confidence_parts.append(confidence.cpu())

    predictions = torch.cat(predictions_parts)
    confidence = torch.cat(confidence_parts)
    confusion = torch.zeros(
        len(CLASS_NAMES), len(CLASS_NAMES), dtype=torch.int64
    )
    for truth, prediction in zip(labels, predictions, strict=True):
        confusion[int(truth), int(prediction)] += 1
    confusion_array = confusion.numpy()
    support = confusion_array.sum(axis=1)
    recall = np.divide(
        np.diag(confusion_array),
        support,
        out=np.zeros_like(support, dtype=float),
        where=support != 0,
    )
    return Evaluation(
        loss=total_loss / len(labels),
        accuracy=float((predictions == labels).float().mean()),
        confusion=confusion_array,
        recall=recall,
        predictions=predictions,
        confidence=confidence,
    )


def train_model(
    model: SmallCNN,
    train_images: torch.Tensor,
    train_labels: torch.Tensor,
    validation_images: torch.Tensor,
    validation_labels: torch.Tensor,
    *,
    learning_rate: float = 1e-3,
    batch_size: int = 256,
    maximum_epochs: int = 6,
    patience: int = 2,
    minimum_improvement: float = 1e-3,
    seed: int = 705,
) -> TrainingResult:
    """Train with persistent Adam state and restore the best validation model."""
    device = model_device(model)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(train_images, train_labels),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_state = deepcopy(model.state_dict())
    best_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[EpochRecord] = []

    started = time.perf_counter()
    for epoch in range(1, maximum_epochs + 1):
        model.train()
        total_loss = 0.0
        examples = 0
        for images, labels in loader:
            images = images.to(
                device, non_blocking=device.type == "cuda"
            )
            labels = labels.to(
                device, non_blocking=device.type == "cuda"
            )
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(labels)
            examples += len(labels)

        validation = evaluate_model(
            model, validation_images, validation_labels
        )
        history.append(
            EpochRecord(
                epoch,
                total_loss / examples,
                validation.loss,
                validation.accuracy,
            )
        )
        if validation.loss < best_loss - minimum_improvement:
            best_loss = validation.loss
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    model.load_state_dict(best_state)
    return TrainingResult(
        history=tuple(history),
        best_epoch=best_epoch,
        stopped_epoch=history[-1].epoch,
        best_validation_loss=best_loss,
        elapsed_seconds=time.perf_counter() - started,
    )


def gradient_norms(
    model: SmallCNN,
    images: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, float]:
    """Inspect gradients for one declared batch without updating parameters."""
    device = model_device(model)
    model.train()
    model.zero_grad(set_to_none=True)
    loss = nn.functional.cross_entropy(
        model(images.to(device)), labels.to(device)
    )
    loss.backward()
    result = {
        "conv1_weight": float(model.conv1.weight.grad.norm().cpu()),
        "conv2_weight": float(model.conv2.weight.grad.norm().cpu()),
        "hidden_weight": float(model.hidden.weight.grad.norm().cpu()),
        "classifier_weight": float(model.classifier.weight.grad.norm().cpu()),
    }
    model.zero_grad(set_to_none=True)
    return result


def most_active_feature_channel(
    model: SmallCNN,
    image: torch.Tensor,
) -> tuple[int, float]:
    """Return a descriptive activation summary, not an explanation."""
    model.eval()
    with torch.no_grad():
        features = model.forward_features(image.to(model_device(model)))
        strength = features.abs().mean(dim=(0, 2, 3))
    index = int(strength.argmax())
    return index, float(strength[index].cpu())


def translate_right(images: torch.Tensor, amount: int = 2) -> torch.Tensor:
    """Translate images right with zero fill."""
    if amount < 0 or amount >= images.shape[-1]:
        raise ValueError("amount must be within the image width")
    shifted = torch.zeros_like(images)
    if amount == 0:
        shifted.copy_(images)
    else:
        shifted[..., amount:] = images[..., :-amount]
    return shifted


def prediction_consistency(
    model: SmallCNN,
    images: torch.Tensor,
    amount: int = 2,
) -> float:
    """Measure prediction agreement after one declared image translation."""
    original, _ = predict_model(model, images)
    shifted, _ = predict_model(model, translate_right(images, amount))
    return float((original == shifted).float().mean())


def majority_baseline_accuracy(labels: torch.Tensor) -> float:
    counts = torch.bincount(labels, minlength=len(CLASS_NAMES))
    return float(counts.max() / len(labels))


def train_week10_experiment(
    root: str | Path = "data/raw",
    *,
    download: bool = True,
    seed: int = 705,
    device: str = "auto",
    training_per_class: int | None = 2000,
    validation_per_class: int = 500,
    maximum_epochs: int = 6,
) -> tuple[
    MNISTData,
    SmallCNN,
    TrainingResult,
    Evaluation,
    Evaluation,
    Evaluation,
    float,
    float,
]:
    configure_reproducibility(seed)
    selected_device = resolve_device(device)
    data, mean, scale = load_mnist_data(
        root,
        download=download,
        training_per_class=training_per_class,
        validation_per_class=validation_per_class,
        seed=seed,
    )
    model = SmallCNN(num_classes=len(CLASS_NAMES)).to(selected_device)
    initial_validation = evaluate_model(
        model, data.validation_images, data.validation_labels
    )
    training = train_model(
        model,
        data.train_images,
        data.train_labels,
        data.validation_images,
        data.validation_labels,
        maximum_epochs=maximum_epochs,
        seed=seed,
    )
    validation = evaluate_model(
        model, data.validation_images, data.validation_labels
    )
    test = evaluate_model(model, data.test_images, data.test_labels)
    return (
        data,
        model,
        training,
        initial_validation,
        validation,
        test,
        mean,
        scale,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument(
        "--full-training",
        action="store_true",
        help="use every official training example not reserved for validation",
    )
    parser.add_argument("--epochs", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import torchvision
    except ImportError as error:
        raise RuntimeError(
            "TorchVision is required. Install the course requirements first."
        ) from error

    training_per_class = None if args.full_training else 2000
    (
        data,
        model,
        training,
        initial_validation,
        validation,
        test,
        mean,
        scale,
    ) = train_week10_experiment(
        args.data_root,
        download=not args.no_download,
        device=args.device,
        training_per_class=training_per_class,
        maximum_epochs=args.epochs,
    )
    shapes = trace_shapes(model, data.test_images[:4])
    norms = gradient_norms(
        model, data.train_images[:128], data.train_labels[:128]
    )
    feature_channel, feature_strength = most_active_feature_channel(
        model, data.test_images[:1]
    )
    consistency = prediction_consistency(model, data.test_images, amount=2)
    shifted_test = evaluate_model(
        model, translate_right(data.test_images, 2), data.test_labels
    )
    mistakes = torch.nonzero(
        test.predictions != data.test_labels, as_tuple=False
    ).flatten()
    device_name = (
        torch.cuda.get_device_name(model_device(model))
        if model_device(model).type == "cuda"
        else str(model_device(model))
    )

    print("runtime_and_data")
    print(f"torch_version={torch.__version__}")
    print(f"torchvision_version={torchvision.__version__}")
    print("dataset=torchvision.datasets.MNIST")
    print(f"device={model_device(model)}")
    print(f"device_name={device_name}")
    print("seed=705")
    print(
        "split_sizes="
        f"{len(data.train_labels)}/"
        f"{len(data.validation_labels)}/"
        f"{len(data.test_labels)}"
    )
    print(f"training_mean={mean:.4f}")
    print(f"training_scale={scale:.4f}")
    print(
        "test_support="
        f"{torch.bincount(data.test_labels, minlength=10).tolist()}"
    )

    print("\nmodel_audit")
    print(f"parameter_count={parameter_count(model)}")
    for name, value in shapes.items():
        print(f"{name}_shape={value}")

    print("\ntraining_and_selection")
    print(
        "initial_validation="
        f"loss={initial_validation.loss:.4f} "
        f"accuracy={initial_validation.accuracy:.3f}"
    )
    print(
        f"best_epoch={training.best_epoch} "
        f"stopped_epoch={training.stopped_epoch} "
        f"best_validation_loss={training.best_validation_loss:.4f}"
    )
    print(f"training_seconds={training.elapsed_seconds:.2f}")
    print(
        "restored_validation="
        f"loss={validation.loss:.4f} accuracy={validation.accuracy:.3f}"
    )

    print("\nlocked_test_evaluation")
    print(
        "majority_baseline_accuracy="
        f"{majority_baseline_accuracy(data.test_labels):.3f}"
    )
    print(f"loss={test.loss:.4f} accuracy={test.accuracy:.3f}")
    print("confusion_rows_true_columns_predicted")
    print(test.confusion)
    print(
        "recall="
        + ", ".join(
            f"{name}:{value:.3f}"
            for name, value in zip(CLASS_NAMES, test.recall, strict=True)
        )
    )
    print(f"misclassified_count={len(mistakes)}")
    print(f"first_misclassified_indices={mistakes[:5].tolist()}")
    if len(mistakes):
        first = int(mistakes[0])
        print(
            "first_error="
            f"true:{CLASS_NAMES[int(data.test_labels[first])]} "
            f"predicted:{CLASS_NAMES[int(test.predictions[first])]} "
            f"confidence:{float(test.confidence[first]):.3f}"
        )

    print("\ninspection")
    print(
        "gradient_norms="
        + ", ".join(f"{name}:{value:.4f}" for name, value in norms.items())
    )
    print(
        f"most_active_feature_channel={feature_channel} "
        f"mean_absolute_activation={feature_strength:.4f}"
    )
    print(f"translation_right_2_consistency={consistency:.3f}")
    print(f"translation_right_2_accuracy={shifted_test.accuracy:.3f}")


if __name__ == "__main__":
    main()
