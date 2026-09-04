"""LSTM sequence classifier with generated input."""

from __future__ import annotations

import torch
from torch import nn


class LSTMClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_classes: int) -> None:
        super().__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.encoder(sequence)
        return self.classifier(hidden[-1])


def main() -> None:
    torch.manual_seed(705)
    model = LSTMClassifier(input_size=6, hidden_size=12, num_classes=3)
    sequence = torch.randn(8, 20, 6)
    target = torch.randint(0, 3, (8,))
    logits = model(sequence)
    loss = nn.functional.cross_entropy(logits, target)
    loss.backward()
    print(f"logit_shape={tuple(logits.shape)}")
    print(f"loss={float(loss.detach()):.4f}")


if __name__ == "__main__":
    main()
