"""A compact PyTorch CNN with a generated-data smoke experiment."""

from __future__ import annotations

import torch
from torch import nn


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Linear(16 * 4 * 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        return self.classifier(torch.flatten(features, 1))


def one_training_step(model: nn.Module, images: torch.Tensor, labels: torch.Tensor) -> float:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    loss = nn.functional.cross_entropy(model(images), labels)
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def main() -> None:
    torch.manual_seed(705)
    model = SmallCNN()
    images = torch.randn(16, 1, 28, 28)
    labels = torch.randint(0, 10, (16,))
    loss = one_training_step(model, images, labels)
    print(f"logit_shape={tuple(model(images).shape)}")
    print(f"one_step_loss={loss:.4f}")


if __name__ == "__main__":
    main()
