"""A small autoencoder that exposes its latent representation."""

from __future__ import annotations

import torch
from torch import nn


class Autoencoder(nn.Module):
    def __init__(self, input_size: int = 32, latent_size: int = 4) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_size, 16), nn.ReLU(), nn.Linear(16, latent_size))
        self.decoder = nn.Sequential(nn.Linear(latent_size, 16), nn.ReLU(), nn.Linear(16, input_size))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(x)
        return self.decoder(latent), latent


def main() -> None:
    torch.manual_seed(705)
    model = Autoencoder()
    batch = torch.randn(12, 32)
    reconstruction, latent = model(batch)
    loss = nn.functional.mse_loss(reconstruction, batch)
    loss.backward()
    print(f"reconstruction_shape={tuple(reconstruction.shape)}")
    print(f"latent_shape={tuple(latent.shape)}")
    print(f"loss={float(loss.detach()):.4f}")


if __name__ == "__main__":
    main()
