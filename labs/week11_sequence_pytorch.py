"""RNN/LSTM delayed-memory experiment and scaled dot-product attention audit.

A class cue appears at the first time step, followed by low-amplitude
distractors. Models are trained on length-30 sequences and evaluated on a
matched validation split and a locked length-60 stress split. The longer split
tests sequence-length extrapolation and is never used for model selection.
"""

from __future__ import annotations

import argparse
import copy
import math
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 705
    input_size: int = 4
    hidden_size: int = 32
    num_classes: int = 3
    train_length: int = 30
    stress_length: int = 60
    train_size: int = 1200
    validation_size: int = 400
    test_size: int = 600
    batch_size: int = 64
    epochs: int = 25
    learning_rate: float = 0.003


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
        raise RuntimeError(
            "CUDA was requested, but this PyTorch build reports CUDA unavailable. "
            "Use the official PyTorch installer for a CUDA-enabled build."
        )
    return device


def make_delayed_memory_split(
    size: int,
    sequence_length: int,
    seed: int,
    num_classes: int = 3,
    noise_standard_deviation: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a disjoint split for a first-cue delayed-memory task."""

    generator = torch.Generator().manual_seed(seed)
    targets = torch.randint(num_classes, (size,), generator=generator)
    inputs = torch.randn(
        size,
        sequence_length,
        num_classes + 1,
        generator=generator,
    ) * noise_standard_deviation
    inputs[:, 0, :num_classes] = nn.functional.one_hot(
        targets, num_classes=num_classes
    ).float()
    inputs[:, 0, num_classes] = 1.0
    return inputs, targets


class RNNClassifier(nn.Module):
    """Many-to-one vanilla tanh RNN used as a controlled baseline."""

    def __init__(self, input_size: int, hidden_size: int, num_classes: int) -> None:
        super().__init__()
        self.encoder = nn.RNN(input_size, hidden_size, batch_first=True)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        _, hidden = self.encoder(sequence)
        return self.classifier(hidden[-1])


class LSTMClassifier(nn.Module):
    """Many-to-one LSTM with an explicit positive forget-gate bias."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        forget_bias: float = 2.0,
    ) -> None:
        super().__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.classifier = nn.Linear(hidden_size, num_classes)
        with torch.no_grad():
            # PyTorch gate order is input, forget, candidate, output.
            self.encoder.bias_ih_l0[hidden_size : 2 * hidden_size].fill_(forget_bias)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.encoder(sequence)
        return self.classifier(hidden[-1])


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    allowed: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return attention output and weights; True mask entries are allowed."""

    scores = query @ key.transpose(-2, -1) / math.sqrt(query.shape[-1])
    if allowed is not None:
        scores = scores.masked_fill(~allowed, float("-inf"))
    weights = scores.softmax(dim=-1)
    return weights @ value, weights


@torch.no_grad()
def evaluate(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> tuple[float, float]:
    model.eval()
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=batch_size)
    total_loss = 0.0
    total_correct = 0
    for batch_inputs, batch_targets in loader:
        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)
        logits = model(batch_inputs)
        total_loss += float(
            nn.functional.cross_entropy(logits, batch_targets, reduction="sum")
        )
        total_correct += int((logits.argmax(dim=1) == batch_targets).sum())
    return total_loss / len(targets), total_correct / len(targets)


def fit(
    model: nn.Module,
    training: tuple[torch.Tensor, torch.Tensor],
    validation: tuple[torch.Tensor, torch.Tensor],
    config: ExperimentConfig,
    device: torch.device,
) -> tuple[int, float, float]:
    """Select and restore the checkpoint with minimum validation loss."""

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loader = DataLoader(
        TensorDataset(*training),
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
    )
    best_epoch = 0
    best_loss = float("inf")
    best_accuracy = 0.0
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, config.epochs + 1):
        model.train()
        for batch_inputs, batch_targets in loader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(batch_inputs), batch_targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        validation_loss, validation_accuracy = evaluate(
            model, *validation, device, config.batch_size
        )
        if validation_loss < best_loss:
            best_epoch = epoch
            best_loss = validation_loss
            best_accuracy = validation_accuracy
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("No checkpoint was selected.")
    model.load_state_dict(best_state)
    return best_epoch, best_loss, best_accuracy


def cue_gradient_profile(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
) -> tuple[float, float, float]:
    """Inspect input-gradient magnitude at the cue, middle, and final step."""

    model.eval()
    sample = inputs[:32].clone().to(device).requires_grad_(True)
    sample_targets = targets[:32].to(device)
    nn.functional.cross_entropy(model(sample), sample_targets).backward()
    profile = sample.grad.norm(dim=-1).mean(dim=0).detach().cpu()
    return float(profile[0]), float(profile[len(profile) // 2]), float(profile[-1])


def attention_audit() -> tuple[tuple[int, ...], list[float], float]:
    torch.manual_seed(705)
    query = torch.randn(1, 4, 8)
    key = torch.randn(1, 4, 8)
    value = torch.randn(1, 4, 6)
    causal = torch.ones(4, 4, dtype=torch.bool).tril()
    output, weights = scaled_dot_product_attention(query, key, value, causal)
    row_sums = weights.sum(dim=-1).squeeze(0).tolist()
    masked_maximum = float(weights.masked_select(~causal).max())
    return tuple(output.shape), row_sums, masked_maximum


def run(config: ExperimentConfig, device: torch.device) -> None:
    set_seed(config.seed)
    training = make_delayed_memory_split(
        config.train_size, config.train_length, config.seed
    )
    validation = make_delayed_memory_split(
        config.validation_size, config.train_length, config.seed + 1
    )
    locked_stress_test = make_delayed_memory_split(
        config.test_size, config.stress_length, config.seed + 2
    )
    print("week11_delayed_memory_experiment")
    print(f"device={device}")
    if device.type == "cuda":
        print(f"device_name={torch.cuda.get_device_name(device)}")
    print(f"seed={config.seed}")
    print(
        "split_sizes="
        f"{config.train_size}/{config.validation_size}/{config.test_size}"
    )
    print(
        f"sequence_lengths=train:{config.train_length} "
        f"validation:{config.train_length} locked_stress:{config.stress_length}"
    )
    print(f"majority_baseline={1.0 / config.num_classes:.3f}")

    start = time.perf_counter()
    for name, model_type in (("rnn", RNNClassifier), ("lstm", LSTMClassifier)):
        set_seed(config.seed)
        model = model_type(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_classes=config.num_classes,
        )
        best_epoch, validation_loss, validation_accuracy = fit(
            model, training, validation, config, device
        )
        stress_loss, stress_accuracy = evaluate(
            model, *locked_stress_test, device, config.batch_size
        )
        cue, middle, final = cue_gradient_profile(model, *validation, device)
        parameters = sum(parameter.numel() for parameter in model.parameters())
        print(f"{name}_parameters={parameters}")
        print(
            f"{name}_selection=epoch:{best_epoch} "
            f"validation_loss:{validation_loss:.4f} "
            f"validation_accuracy:{validation_accuracy:.3f}"
        )
        print(
            f"{name}_locked_stress=loss:{stress_loss:.4f} "
            f"accuracy:{stress_accuracy:.3f}"
        )
        print(
            f"{name}_input_gradient=cue:{cue:.6e} "
            f"middle:{middle:.6e} final:{final:.6e}"
        )

    output_shape, row_sums, masked_maximum = attention_audit()
    print(f"attention_output_shape={output_shape}")
    print("attention_row_sums=" + ",".join(f"{value:.3f}" for value in row_sums))
    print(f"causal_masked_weight_max={masked_maximum:.3f}")
    print(f"training_seconds={time.perf_counter() - start:.2f}")


def audit_only(device: torch.device) -> None:
    set_seed(705)
    inputs, targets = make_delayed_memory_split(8, 20, 705)
    for model_type in (RNNClassifier, LSTMClassifier):
        model = model_type(4, 12, 3).to(device)
        logits = model(inputs.to(device))
        nn.functional.cross_entropy(logits, targets.to(device)).backward()
        print(f"{model_type.__name__}_logits_shape={tuple(logits.shape)}")
    output_shape, row_sums, masked_maximum = attention_audit()
    print(f"attention_output_shape={output_shape}")
    print("attention_row_sums=" + ",".join(f"{value:.3f}" for value in row_sums))
    print(f"causal_masked_weight_max={masked_maximum:.3f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    if args.audit_only:
        audit_only(device)
        return
    run(ExperimentConfig(epochs=args.epochs), device)


if __name__ == "__main__":
    main()
