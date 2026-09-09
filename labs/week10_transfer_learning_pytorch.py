"""Week 10 extension: transfer ImageNet ConvNeXt-Tiny to CIFAR-10."""

from __future__ import annotations

import argparse
import os
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset


@dataclass(frozen=True)
class Evaluation:
    loss: float
    accuracy: float


@dataclass(frozen=True)
class StageResult:
    name: str
    best_epoch: int
    best_validation_loss: float
    validation_accuracy: float
    elapsed_seconds: float


def resolve_device(requested: str = "auto") -> torch.device:
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
            "CUDA was requested, but this PyTorch installation cannot access it."
        )
    return device


def configure_reproducibility(seed: int = 705) -> None:
    torch.manual_seed(seed)
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
    training_per_class: int,
    validation_per_class: int,
    seed: int = 705,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    training_parts: list[torch.Tensor] = []
    validation_parts: list[torch.Tensor] = []
    for class_index in range(10):
        indices = torch.nonzero(labels == class_index, as_tuple=False).flatten()
        required = training_per_class + validation_per_class
        if len(indices) < required:
            raise ValueError(f"class {class_index} has fewer than {required} images")
        shuffled = indices[torch.randperm(len(indices), generator=generator)]
        validation_parts.append(shuffled[:validation_per_class])
        training_parts.append(
            shuffled[validation_per_class:required]
        )
    training = torch.cat(training_parts)
    validation = torch.cat(validation_parts)
    training = training[torch.randperm(len(training), generator=generator)]
    validation = validation[torch.randperm(len(validation), generator=generator)]
    return training, validation


def build_transfer_model(
    num_classes: int = 10,
    *,
    pretrained: bool = True,
) -> tuple[nn.Module, object | None]:
    from torchvision.models import (
        ConvNeXt_Tiny_Weights,
        convnext_tiny,
    )

    weights = (
        ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
    )
    model = convnext_tiny(weights=weights)
    for parameter in model.parameters():
        parameter.requires_grad = False
    input_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(input_features, num_classes)
    return model, weights


def total_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def unfreeze_final_feature_stage(model: nn.Module) -> None:
    for parameter in model.features[-1].parameters():
        parameter.requires_grad = True


def make_transforms(weights: object) -> tuple[object, object]:
    from torchvision import transforms

    evaluation_transform = weights.transforms()
    training_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                evaluation_transform.crop_size[0],
                scale=(0.80, 1.0),
                interpolation=evaluation_transform.interpolation,
                antialias=True,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=evaluation_transform.mean,
                std=evaluation_transform.std,
            ),
        ]
    )
    return training_transform, evaluation_transform


def load_cifar10(
    root: str | Path,
    weights: object,
    *,
    download: bool,
    training_per_class: int,
    validation_per_class: int,
    batch_size: int,
    seed: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    from torchvision.datasets import CIFAR10

    training_transform, evaluation_transform = make_transforms(weights)
    training_source = CIFAR10(
        root=root,
        train=True,
        transform=training_transform,
        download=download,
    )
    validation_source = CIFAR10(
        root=root,
        train=True,
        transform=evaluation_transform,
        download=download,
    )
    test_source = CIFAR10(
        root=root,
        train=False,
        transform=evaluation_transform,
        download=download,
    )
    labels = torch.tensor(training_source.targets, dtype=torch.int64)
    training_indices, validation_indices = stratified_subset_indices(
        labels,
        training_per_class=training_per_class,
        validation_per_class=validation_per_class,
        seed=seed,
    )
    generator = torch.Generator().manual_seed(seed)
    pin_memory = torch.cuda.is_available()
    common = {
        "batch_size": batch_size,
        "num_workers": 0,
        "pin_memory": pin_memory,
    }
    training_loader = DataLoader(
        Subset(training_source, training_indices.tolist()),
        shuffle=True,
        generator=generator,
        **common,
    )
    validation_loader = DataLoader(
        Subset(validation_source, validation_indices.tolist()),
        shuffle=False,
        **common,
    )
    test_loader = DataLoader(test_source, shuffle=False, **common)
    return training_loader, validation_loader, test_loader


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Evaluation:
    model.eval()
    total_loss = 0.0
    correct = 0
    examples = 0
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=device.type == "cuda")
            labels = labels.to(device, non_blocking=device.type == "cuda")
            logits = model(images)
            total_loss += float(
                nn.functional.cross_entropy(
                    logits,
                    labels,
                    reduction="sum",
                ).cpu()
            )
            correct += int((logits.argmax(dim=1) == labels).sum())
            examples += len(labels)
    return Evaluation(total_loss / examples, correct / examples)


def train_stage(
    name: str,
    model: nn.Module,
    training_loader: DataLoader,
    validation_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    epochs: int,
) -> StageResult:
    initial = evaluate(model, validation_loader, device)
    best_state = deepcopy(model.state_dict())
    best_loss = initial.loss
    best_accuracy = initial.accuracy
    best_epoch = 0
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        if not any(parameter.requires_grad for parameter in model.features.parameters()):
            model.features.eval()
        for images, labels in training_loader:
            images = images.to(device, non_blocking=device.type == "cuda")
            labels = labels.to(device, non_blocking=device.type == "cuda")
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(images), labels)
            loss.backward()
            optimizer.step()

        validation = evaluate(model, validation_loader, device)
        print(
            f"{name}_epoch={epoch} "
            f"validation_loss={validation.loss:.4f} "
            f"validation_accuracy={validation.accuracy:.3f}"
        )
        if validation.loss < best_loss:
            best_loss = validation.loss
            best_accuracy = validation.accuracy
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return StageResult(
        name,
        best_epoch,
        best_loss,
        best_accuracy,
        time.perf_counter() - started,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--training-per-class", type=int, default=200)
    parser.add_argument("--validation-per-class", type=int, default=100)
    parser.add_argument("--head-epochs", type=int, default=2)
    parser.add_argument("--fine-tune-epochs", type=int, default=1)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_reproducibility(705)
    device = resolve_device(args.device)

    if args.audit_only:
        model, _ = build_transfer_model(pretrained=False)
        model = model.to(device)
        with torch.inference_mode():
            output = model(torch.zeros(1, 3, 64, 64, device=device))
        print("transfer_model_audit")
        print("weights=None")
        print(f"device={device}")
        print(f"output_shape={tuple(output.shape)}")
        print(f"total_parameters={total_parameter_count(model)}")
        print(f"trainable_parameters={trainable_parameter_count(model)}")
        return

    model, weights = build_transfer_model(pretrained=True)
    assert weights is not None
    model = model.to(device)
    training_loader, validation_loader, test_loader = load_cifar10(
        args.data_root,
        weights,
        download=not args.no_download,
        training_per_class=args.training_per_class,
        validation_per_class=args.validation_per_class,
        batch_size=args.batch_size,
        seed=705,
    )

    print("transfer_protocol")
    print("source=ImageNet-1K")
    print("target=CIFAR-10")
    print(f"weights={weights}")
    print(f"device={device}")
    print(f"train_validation_test={len(training_loader.dataset)}/"
          f"{len(validation_loader.dataset)}/{len(test_loader.dataset)}")
    print(f"total_parameters={total_parameter_count(model)}")
    print(f"head_trainable_parameters={trainable_parameter_count(model)}")

    head_optimizer = torch.optim.AdamW(
        model.classifier[2].parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )
    head_result = train_stage(
        "head",
        model,
        training_loader,
        validation_loader,
        head_optimizer,
        device,
        epochs=args.head_epochs,
    )
    print(head_result)

    if args.fine_tune_epochs > 0:
        unfreeze_final_feature_stage(model)
        print(
            "fine_tune_trainable_parameters="
            f"{trainable_parameter_count(model)}"
        )
        fine_tune_optimizer = torch.optim.AdamW(
            [
                {"params": model.classifier[2].parameters(), "lr": 1e-4},
                {"params": model.features[-1].parameters(), "lr": 1e-5},
            ],
            weight_decay=1e-4,
        )
        fine_tune_result = train_stage(
            "fine_tune",
            model,
            training_loader,
            validation_loader,
            fine_tune_optimizer,
            device,
            epochs=args.fine_tune_epochs,
        )
        print(fine_tune_result)

    locked_test = evaluate(model, test_loader, device)
    print("locked_test_evaluation")
    print(
        f"loss={locked_test.loss:.4f} "
        f"accuracy={locked_test.accuracy:.3f}"
    )


if __name__ == "__main__":
    main()