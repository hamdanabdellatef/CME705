"""Autoencoder and variational-autoencoder lab on Fashion-MNIST.

The official training split supplies deterministic, class-balanced training and
validation subsets. A balanced subset of the official test split stays locked
until the minimum-validation-loss checkpoint is restored. The vector
``Autoencoder`` class is retained as a compact shape-audit model; the runnable
experiment compares convolutional deterministic and variational models.
"""

from __future__ import annotations

import argparse
import copy
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset


ModelKind = Literal["ae", "vae"]


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 705
    latent_dim: int = 16
    train_per_class: int = 1000
    validation_per_class: int = 200
    test_per_class: int = 200
    batch_size: int = 128
    epochs: int = 5
    learning_rate: float = 1e-3
    beta: float = 1.0


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


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def stratified_indices(
    labels: Sequence[int] | torch.Tensor,
    first_per_class: int,
    second_per_class: int,
    seed: int,
    classes: int = 10,
) -> tuple[list[int], list[int]]:
    """Return repeatable, balanced, and disjoint index sets."""

    target = torch.as_tensor(labels, dtype=torch.long)
    generator = torch.Generator().manual_seed(seed)
    first: list[int] = []
    second: list[int] = []
    for class_id in range(classes):
        candidates = torch.where(target == class_id)[0]
        required = first_per_class + second_per_class
        if len(candidates) < required:
            raise ValueError(
                f"Class {class_id} has {len(candidates)} examples; {required} required."
            )
        order = torch.randperm(len(candidates), generator=generator)
        chosen = candidates[order[:required]].tolist()
        first.extend(chosen[:first_per_class])
        second.extend(chosen[first_per_class:])
    return first, second


class Autoencoder(nn.Module):
    """Small vector autoencoder retained for transparent shape tests."""

    def __init__(self, input_size: int = 32, latent_size: int = 4) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 16), nn.ReLU(), nn.Linear(16, latent_size)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 16), nn.ReLU(), nn.Linear(16, input_size)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(x)
        return self.decoder(latent), latent


class ConvEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.features(images).flatten(1)


class ConvDecoder(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.project = nn.Linear(latent_dim, 32 * 7 * 7)
        self.features = nn.Sequential(
            nn.ConvTranspose2d(
                32, 16, kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        hidden = self.project(latent).reshape(-1, 32, 7, 7)
        return torch.sigmoid(self.features(hidden))


class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim: int = 16) -> None:
        super().__init__()
        self.encoder = ConvEncoder()
        self.to_latent = nn.Linear(32 * 7 * 7, latent_dim)
        self.decoder = ConvDecoder(latent_dim)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.to_latent(self.encoder(images))
        return self.decoder(latent), latent


class ConvVAE(nn.Module):
    def __init__(self, latent_dim: int = 16) -> None:
        super().__init__()
        self.encoder = ConvEncoder()
        self.to_mean = nn.Linear(32 * 7 * 7, latent_dim)
        self.to_log_variance = nn.Linear(32 * 7 * 7, latent_dim)
        self.decoder = ConvDecoder(latent_dim)

    @staticmethod
    def reparameterize(
        mean: torch.Tensor,
        log_variance: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(mean)
        return mean + torch.exp(0.5 * log_variance) * noise

    def encode(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(images)
        return self.to_mean(hidden), self.to_log_variance(hidden)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)

    def forward(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_variance = self.encode(images)
        latent = self.reparameterize(mean, log_variance)
        return self.decode(latent), mean, log_variance, latent

    @torch.no_grad()
    def sample(self, count: int, device: torch.device) -> torch.Tensor:
        return self.decode(torch.randn(count, self.to_mean.out_features, device=device))


def vae_terms(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    mean: torch.Tensor,
    log_variance: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return beta-ELBO loss, reconstruction BCE, and analytic KL per image."""

    batch_size = len(target)
    # Probability-space BCE and the Gaussian KL are evaluated in float32.
    # PyTorch intentionally rejects BCELoss inside CUDA autocast, and the
    # exponential in the KL is also more stable outside float16.
    with torch.autocast(device_type=reconstruction.device.type, enabled=False):
        reconstruction_bce = F.binary_cross_entropy(
            reconstruction.float(), target.float(), reduction="sum"
        ) / batch_size
        mean_float = mean.float()
        log_variance_float = log_variance.float()
        kl = -0.5 * torch.sum(
            1.0
            + log_variance_float
            - mean_float.square()
            - log_variance_float.exp()
        ) / batch_size
    return reconstruction_bce + beta * kl, reconstruction_bce, kl


def model_terms(
    model: nn.Module,
    images: torch.Tensor,
    kind: ModelKind,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if kind == "ae":
        reconstruction, latent = model(images)
        with torch.autocast(device_type=reconstruction.device.type, enabled=False):
            bce = F.binary_cross_entropy(
                reconstruction.float(), images.float(), reduction="sum"
            ) / len(images)
        kl = torch.zeros((), device=images.device)
    else:
        reconstruction, mean, log_variance, latent = model(images)
        _, bce, kl = vae_terms(reconstruction, images, mean, log_variance, beta)
    with torch.autocast(device_type=reconstruction.device.type, enabled=False):
        mse = F.mse_loss(
            reconstruction.float(), images.float(), reduction="sum"
        ) / len(images)
    objective = bce + beta * kl
    return objective, bce, kl, mse, latent


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    kind: ModelKind,
    beta: float,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    totals = {"objective": 0.0, "bce": 0.0, "kl": 0.0, "mse": 0.0}
    examples = 0
    for images, _ in loader:
        images = images.to(device)
        objective, bce, kl, mse, _ = model_terms(model, images, kind, beta)
        size = len(images)
        for name, value in zip(totals, (objective, bce, kl, mse)):
            totals[name] += float(value) * size
        examples += size
    return {name: value / examples for name, value in totals.items()}


def fit(
    model: nn.Module,
    training_loader: DataLoader,
    validation_loader: DataLoader,
    kind: ModelKind,
    config: ExperimentConfig,
    device: torch.device,
) -> tuple[int, dict[str, float]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_epoch = 0
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, float] = {}
    for epoch in range(1, config.epochs + 1):
        model.train()
        for images, _ in training_loader:
            images = images.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                objective, _, _, _, _ = model_terms(
                    model, images, kind, config.beta
                )
            scaler.scale(objective).backward()
            scaler.step(optimizer)
            scaler.update()
        metrics = evaluate(
            model, validation_loader, kind, config.beta, device
        )
        print(
            f"{kind}_epoch={epoch} validation_objective={metrics['objective']:.4f} "
            f"bce={metrics['bce']:.4f} kl={metrics['kl']:.4f} mse={metrics['mse']:.4f}"
        )
        if metrics["objective"] < best_loss:
            best_epoch = epoch
            best_loss = metrics["objective"]
            best_metrics = metrics
            best_state = copy.deepcopy(
                {name: value.detach().cpu() for name, value in model.state_dict().items()}
            )
    if best_state is None:
        raise RuntimeError("No checkpoint was selected.")
    model.load_state_dict(best_state)
    model.to(device)
    return best_epoch, best_metrics


def make_loaders(config: ExperimentConfig, data_root: Path, device: torch.device):
    from torchvision import datasets, transforms

    transform = transforms.ToTensor()
    source_train = datasets.FashionMNIST(
        data_root, train=True, download=True, transform=transform
    )
    source_test = datasets.FashionMNIST(
        data_root, train=False, download=True, transform=transform
    )
    training_indices, validation_indices = stratified_indices(
        source_train.targets,
        config.train_per_class,
        config.validation_per_class,
        config.seed,
    )
    test_indices, _ = stratified_indices(
        source_test.targets, config.test_per_class, 0, config.seed + 1
    )
    common = {
        "batch_size": config.batch_size,
        "num_workers": 0,
        "pin_memory": device.type == "cuda",
    }
    training_loader = DataLoader(
        Subset(source_train, training_indices),
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
        **common,
    )
    validation_loader = DataLoader(
        Subset(source_train, validation_indices), shuffle=False, **common
    )
    test_loader = DataLoader(
        Subset(source_test, test_indices), shuffle=False, **common
    )
    return training_loader, validation_loader, test_loader


@torch.no_grad()
def save_evidence(
    autoencoder: ConvAutoencoder | None,
    vae: ConvVAE | None,
    loader: DataLoader,
    output_dir: Path,
    device: torch.device,
) -> None:
    from torchvision.utils import save_image

    output_dir.mkdir(parents=True, exist_ok=True)
    images, _ = next(iter(loader))
    images = images[:16].to(device)
    rows = [images]
    if autoencoder is not None:
        rows.append(autoencoder(images)[0])
    if vae is not None:
        rows.append(vae(images)[0])
        rows.append(vae.sample(16, device))
    save_image(
        torch.cat(rows).cpu(),
        output_dir / "fashion_mnist_reconstruction_and_samples.png",
        nrow=16,
    )
    if autoencoder is not None:
        latent = autoencoder(images[:2])[1]
        weights = torch.linspace(0, 1, 11, device=device).unsqueeze(1)
        interpolation = (1 - weights) * latent[0] + weights * latent[1]
        save_image(
            autoencoder.decoder(interpolation).cpu(),
            output_dir / "autoencoder_latent_interpolation.png",
            nrow=11,
        )


def audit_only(device: torch.device) -> None:
    set_seed(705)
    images = torch.rand(4, 1, 28, 28, device=device)
    autoencoder = ConvAutoencoder(latent_dim=8).to(device)
    vae = ConvVAE(latent_dim=8).to(device)
    ae_reconstruction, ae_latent = autoencoder(images)
    vae_reconstruction, mean, log_variance, vae_latent = vae(images)
    zero_noise = torch.zeros_like(mean)
    deterministic_sample = ConvVAE.reparameterize(mean, log_variance, zero_noise)
    objective, bce, kl = vae_terms(
        vae_reconstruction, images, mean, log_variance, beta=1.0
    )
    labels = torch.arange(10).repeat_interleave(10)
    first, second = stratified_indices(labels, 5, 2, seed=705)
    print("week12_autoencoder_vae_audit")
    print(f"device={device}")
    print(f"ae_shapes={tuple(ae_reconstruction.shape)}/{tuple(ae_latent.shape)}")
    print(
        f"vae_shapes={tuple(vae_reconstruction.shape)}/{tuple(mean.shape)}/"
        f"{tuple(log_variance.shape)}/{tuple(vae_latent.shape)}"
    )
    print(f"ae_parameters={parameter_count(autoencoder)}")
    print(f"vae_parameters={parameter_count(vae)}")
    print(f"zero_noise_returns_mean={torch.equal(deterministic_sample, mean)}")
    print(f"loss_terms={float(objective):.4f}/{float(bce):.4f}/{float(kl):.4f}")
    print(f"balanced_indices={len(first)}/{len(second)}")
    print(f"disjoint_indices={set(first).isdisjoint(second)}")
    print("downloads_started=False")


def run(
    config: ExperimentConfig,
    device: torch.device,
    data_root: Path,
    output_dir: Path,
    selected: str,
) -> None:
    set_seed(config.seed)
    training, validation, locked_test = make_loaders(config, data_root, device)
    models: dict[ModelKind, nn.Module] = {}
    if selected in ("ae", "both"):
        models["ae"] = ConvAutoencoder(config.latent_dim).to(device)
    if selected in ("vae", "both"):
        models["vae"] = ConvVAE(config.latent_dim).to(device)

    print("week12_autoencoder_vae_fashion_mnist")
    print(f"device={device}")
    if device.type == "cuda":
        print(f"device_name={torch.cuda.get_device_name(device)}")
    print(f"seed={config.seed}")
    print(
        "split_sizes="
        f"{10 * config.train_per_class}/"
        f"{10 * config.validation_per_class}/"
        f"{10 * config.test_per_class}"
    )
    print(f"latent_dim={config.latent_dim} beta={config.beta}")
    started = time.perf_counter()
    for kind, model in models.items():
        print(f"{kind}_parameters={parameter_count(model)}")
        best_epoch, validation_metrics = fit(
            model, training, validation, kind, config, device
        )
        print(
            f"{kind}_selected_checkpoint=epoch:{best_epoch} "
            f"validation_objective:{validation_metrics['objective']:.4f}"
        )
        test_metrics = evaluate(model, locked_test, kind, config.beta, device)
        print(
            f"{kind}_locked_test=objective:{test_metrics['objective']:.4f} "
            f"bce:{test_metrics['bce']:.4f} kl:{test_metrics['kl']:.4f} "
            f"mse:{test_metrics['mse']:.4f}"
        )
    save_evidence(
        models.get("ae"), models.get("vae"), locked_test, output_dir, device
    )
    print(f"evidence_directory={output_dir}")
    print(f"training_seconds={time.perf_counter() - started:.2f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--model", choices=("ae", "vae", "both"), default="both")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/week12"))
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--train-per-class", type=int, default=1000)
    parser.add_argument("--validation-per-class", type=int, default=200)
    parser.add_argument("--test-per-class", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    if args.audit_only:
        audit_only(device)
        return
    config = ExperimentConfig(
        latent_dim=args.latent_dim,
        train_per_class=args.train_per_class,
        validation_per_class=args.validation_per_class,
        test_per_class=args.test_per_class,
        batch_size=args.batch_size,
        epochs=args.epochs,
        beta=args.beta,
    )
    run(config, device, args.data_root, args.output_dir, args.model)


if __name__ == "__main__":
    main()
