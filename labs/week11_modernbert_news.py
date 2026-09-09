"""ModernBERT transfer learning for real AG News topic classification.

The official AG News training split is divided deterministically into balanced
training and validation subsets. The official test split remains locked until
the validation-selected checkpoint has been restored. By default, only the
ModernBERT prediction head and classifier are trained; use --unfreeze-last-n
for a staged fine-tuning extension.

Dataset card: https://huggingface.co/datasets/fancyzhx/ag_news
Model card: https://huggingface.co/answerdotai/ModernBERT-base
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


MODEL_ID = "answerdotai/ModernBERT-base"
DATASET_ID = "fancyzhx/ag_news"
LABELS = ("World", "Sports", "Business", "Sci/Tech")


@dataclass(frozen=True)
class FineTuningConfig:
    seed: int = 705
    train_per_class: int = 500
    validation_per_class: int = 125
    test_per_class: int = 250
    max_length: int = 128
    batch_size: int = 16
    epochs: int = 2
    learning_rate: float = 2e-4
    unfreeze_last_n: int = 0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but this PyTorch build cannot use it.")
    return device


def stratified_indices(
    labels: Sequence[int],
    first_per_class: int,
    second_per_class: int,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Return repeatable, class-balanced, disjoint index sets."""

    label_tensor = torch.as_tensor(labels, dtype=torch.long)
    generator = torch.Generator().manual_seed(seed)
    first: list[int] = []
    second: list[int] = []
    for class_id in range(len(LABELS)):
        candidates = torch.where(label_tensor == class_id)[0]
        required = first_per_class + second_per_class
        if len(candidates) < required:
            raise ValueError(
                f"Class {class_id} has {len(candidates)} examples; {required} required."
            )
        order = torch.randperm(len(candidates), generator=generator)
        selected = candidates[order[:required]].tolist()
        first.extend(selected[:first_per_class])
        second.extend(selected[first_per_class:])
    return first, second


class NewsDataset(Dataset[tuple[str, int]]):
    def __init__(self, texts: Sequence[str], labels: Sequence[int]) -> None:
        self.texts = list(texts)
        self.labels = [int(label) for label in labels]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[str, int]:
        return self.texts[index], self.labels[index]


def make_collate(tokenizer: Any, max_length: int) -> Callable:
    def collate(batch: Sequence[tuple[str, int]]) -> dict[str, torch.Tensor]:
        texts, labels = zip(*batch)
        encoded = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor(labels, dtype=torch.long)
        return encoded

    return collate


def require_optional_dependencies() -> tuple[Any, Any, Any]:
    try:
        from datasets import load_dataset
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except Exception as error:
        raise RuntimeError(
            "Week 11 requires compatible installations of datasets, transformers, "
            "and pandas. Install the pinned course requirements in a fresh environment."
        ) from error
    return load_dataset, AutoTokenizer, AutoModelForSequenceClassification


def configure_trainable_parameters(model: nn.Module, unfreeze_last_n: int) -> int:
    """Train heads and optionally the final ModernBERT encoder blocks."""

    for parameter in model.parameters():
        parameter.requires_grad = False
    for module_name in ("head", "classifier"):
        module = getattr(model, module_name)
        for parameter in module.parameters():
            parameter.requires_grad = True
    if unfreeze_last_n:
        layers = getattr(getattr(model, "model"), "layers")
        if unfreeze_last_n > len(layers):
            raise ValueError("Cannot unfreeze more layers than the encoder contains.")
        for layer in layers[-unfreeze_last_n:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def subset_from_indices(dataset: Any, indices: Sequence[int]) -> NewsDataset:
    selected = dataset.select(list(indices))
    return NewsDataset(selected["text"], selected["label"])


def load_splits(
    config: FineTuningConfig, cache_dir: Path
) -> tuple[NewsDataset, NewsDataset, NewsDataset]:
    load_dataset, _, _ = require_optional_dependencies()
    raw = load_dataset(DATASET_ID, cache_dir=str(cache_dir))
    training_indices, validation_indices = stratified_indices(
        raw["train"]["label"],
        config.train_per_class,
        config.validation_per_class,
        config.seed,
    )
    test_indices, _ = stratified_indices(
        raw["test"]["label"],
        config.test_per_class,
        0,
        config.seed + 1,
    )
    return (
        subset_from_indices(raw["train"], training_indices),
        subset_from_indices(raw["train"], validation_indices),
        subset_from_indices(raw["test"], test_indices),
    )


def move_batch(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {name: value.to(device) for name, value in batch.items()}


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    for batch in loader:
        batch = move_batch(batch, device)
        output = model(**batch)
        size = len(batch["labels"])
        total_loss += float(output.loss) * size
        total_correct += int((output.logits.argmax(dim=1) == batch["labels"]).sum())
        total_examples += size
    return total_loss / total_examples, total_correct / total_examples


@torch.no_grad()
def confusion_report(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return rows-true/columns-predicted counts and per-class recall."""

    model.eval()
    matrix = torch.zeros(len(LABELS), len(LABELS), dtype=torch.long)
    for batch in loader:
        batch = move_batch(batch, device)
        predictions = model(**batch).logits.argmax(dim=1).cpu()
        targets = batch["labels"].cpu()
        for target, prediction in zip(targets.tolist(), predictions.tolist()):
            matrix[target, prediction] += 1
    support = matrix.sum(dim=1)
    recall = matrix.diag().float() / support.clamp_min(1)
    return matrix, recall


def fit(
    model: nn.Module,
    training_loader: DataLoader,
    validation_loader: DataLoader,
    config: FineTuningConfig,
    device: torch.device,
) -> tuple[int, float, float]:
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_loss = float("inf")
    best_accuracy = 0.0
    for epoch in range(1, config.epochs + 1):
        model.train()
        for batch in training_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                loss = model(**batch).loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        validation_loss, validation_accuracy = evaluate(
            model, validation_loader, device
        )
        print(
            f"epoch={epoch} validation_loss={validation_loss:.4f} "
            f"validation_accuracy={validation_accuracy:.3f}"
        )
        if validation_loss < best_loss:
            best_epoch = epoch
            best_loss = validation_loss
            best_accuracy = validation_accuracy
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("No checkpoint was selected.")
    model.load_state_dict(best_state)
    model.to(device)
    return best_epoch, best_loss, best_accuracy


@torch.no_grad()
def classify_examples(model: nn.Module, tokenizer: Any, device: torch.device) -> None:
    examples = [
        "Central bank signals a change in interest-rate policy",
        "Local team wins the championship after extra time",
        "New processor design reduces energy use in data centers",
    ]
    encoded = tokenizer(
        examples,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    ).to(device)
    probabilities = model(**encoded).logits.softmax(dim=1)
    for text, distribution in zip(examples, probabilities):
        prediction = int(distribution.argmax())
        print(
            f"example_prediction={LABELS[prediction]} "
            f"confidence={float(distribution[prediction]):.3f} text={text!r}"
        )


def run(config: FineTuningConfig, device: torch.device, data_root: Path) -> None:
    set_seed(config.seed)
    _, AutoTokenizer, AutoModelForSequenceClassification = (
        require_optional_dependencies()
    )
    training, validation, locked_test = load_splits(config, data_root)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    collate = make_collate(tokenizer, config.max_length)
    training_loader = DataLoader(
        training,
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
        collate_fn=collate,
    )
    validation_loader = DataLoader(
        validation, batch_size=config.batch_size, collate_fn=collate
    )
    test_loader = DataLoader(
        locked_test, batch_size=config.batch_size, collate_fn=collate
    )
    id2label = dict(enumerate(LABELS))
    label2id = {label: index for index, label in id2label.items()}
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        num_labels=len(LABELS),
        id2label=id2label,
        label2id=label2id,
    ).to(device)
    trainable = configure_trainable_parameters(model, config.unfreeze_last_n)

    print("week11_modernbert_ag_news")
    print(f"model={MODEL_ID}")
    print(f"dataset={DATASET_ID}")
    print(f"device={device}")
    if device.type == "cuda":
        print(f"device_name={torch.cuda.get_device_name(device)}")
    print(f"seed={config.seed}")
    print(f"split_sizes={len(training)}/{len(validation)}/{len(locked_test)}")
    print(f"max_length={config.max_length}")
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"total_parameters={total_parameters}")
    print(f"trainable_parameters={trainable}")
    best_epoch, validation_loss, validation_accuracy = fit(
        model, training_loader, validation_loader, config, device
    )
    print(
        f"selected_checkpoint=epoch:{best_epoch} "
        f"validation_loss:{validation_loss:.4f} "
        f"validation_accuracy:{validation_accuracy:.3f}"
    )
    print("locked_test_evaluation")
    test_loss, test_accuracy = evaluate(model, test_loader, device)
    print(f"loss={test_loss:.4f} accuracy={test_accuracy:.3f}")
    matrix, recall = confusion_report(model, test_loader, device)
    print("confusion_matrix_rows_true_columns_predicted")
    for label, row in zip(LABELS, matrix.tolist()):
        print(f"{label}=" + ",".join(str(value) for value in row))
    print(
        "per_class_recall="
        + ",".join(
            f"{label}:{score:.3f}" for label, score in zip(LABELS, recall.tolist())
        )
    )
    classify_examples(model, tokenizer, device)


def audit_only(device: torch.device) -> None:
    labels = [class_id for class_id in range(4) for _ in range(10)]
    training, validation = stratified_indices(labels, 5, 2, seed=705)
    print("week11_modernbert_audit")
    print(f"model={MODEL_ID}")
    print(f"dataset={DATASET_ID}")
    print(f"device={device}")
    print(f"balanced_indices={len(training)}/{len(validation)}")
    print(f"transformers_available={importlib.util.find_spec('transformers') is not None}")
    print(f"datasets_available={importlib.util.find_spec('datasets') is not None}")
    print("downloads_started=False")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/huggingface"))
    parser.add_argument("--train-per-class", type=int, default=500)
    parser.add_argument("--validation-per-class", type=int, default=125)
    parser.add_argument("--test-per-class", type=int, default=250)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--unfreeze-last-n", type=int, default=0)
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    if args.audit_only:
        audit_only(device)
        return
    config = FineTuningConfig(
        train_per_class=args.train_per_class,
        validation_per_class=args.validation_per_class,
        test_per_class=args.test_per_class,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        unfreeze_last_n=args.unfreeze_last_n,
    )
    run(config, device, args.data_root)


if __name__ == "__main__":
    main()
