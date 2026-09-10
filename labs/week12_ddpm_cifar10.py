"""Pretrained DDPM sampling and benchmark audit on CIFAR-10.

The lab uses the public Apache-2.0 ``google/ddpm-cifar10-32`` checkpoint.
Its default DDIM run is a fast course demonstration. The optional FID path
records sample count, scheduler, denoising steps, and real-data reference;
it must not be compared with a paper result unless the full protocol matches.
"""

from __future__ import annotations

import argparse
import importlib.util
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn


MODEL_ID = "google/ddpm-cifar10-32"
MODEL_REVISION = "267b167dc01f0e4e61923ea244e8b988f84deb80"
PAPER_URL = "https://arxiv.org/abs/2006.11239"
MODEL_CARD_URL = "https://huggingface.co/google/ddpm-cifar10-32"
SchedulerName = Literal["ddpm", "ddim"]


@dataclass(frozen=True)
class SamplingConfig:
    seed: int = 705
    sample_count: int = 16
    batch_size: int = 16
    inference_steps: int = 50
    scheduler: SchedulerName = "ddim"
    fid_samples: int = 0
    reference_split: str = "train"


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but this PyTorch build cannot use it.")
    return device


def linear_beta_schedule(
    steps: int = 1000,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return beta, alpha, and cumulative alpha-bar schedules."""

    if steps < 2 or not 0 < beta_start < beta_end < 1:
        raise ValueError("Require steps >= 2 and 0 < beta_start < beta_end < 1.")
    beta = torch.linspace(beta_start, beta_end, steps)
    alpha = 1.0 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    return beta, alpha, alpha_bar


def q_sample(
    clean: torch.Tensor,
    timesteps: torch.Tensor,
    alpha_bar: torch.Tensor,
    noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample q(x_t | x_0) with the closed-form DDPM reparameterization."""

    if noise is None:
        noise = torch.randn_like(clean)
    if timesteps.ndim != 1 or len(timesteps) != len(clean):
        raise ValueError("timesteps must have shape (batch,).")
    selected = alpha_bar.to(clean.device)[timesteps]
    selected = selected.reshape(-1, *([1] * (clean.ndim - 1)))
    noisy = selected.sqrt() * clean + (1.0 - selected).sqrt() * noise
    return noisy, noise


def noise_prediction_loss(
    predicted_noise: torch.Tensor, target_noise: torch.Tensor
) -> torch.Tensor:
    return nn.functional.mse_loss(predicted_noise, target_noise)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def require_diffusion_dependencies() -> tuple[Any, Any, Any, Any]:
    try:
        from diffusers import DDIMScheduler, DDPMPipeline, DDPMScheduler, UNet2DModel
    except Exception as error:
        raise RuntimeError(
            "Install the pinned course requirements for the Week 12 diffusion lab."
        ) from error
    return UNet2DModel, DDPMPipeline, DDPMScheduler, DDIMScheduler


def load_pipeline(
    device: torch.device,
    cache_dir: Path,
    scheduler_name: SchedulerName,
    revision: str,
):
    UNet2DModel, DDPMPipeline, DDPMScheduler, DDIMScheduler = (
        require_diffusion_dependencies()
    )
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    # The official repository uses a legacy flat layout. Loading the safe U-Net
    # and scheduler components explicitly works across current Diffusers releases.
    unet = UNet2DModel.from_pretrained(
        MODEL_ID,
        revision=revision,
        cache_dir=str(cache_dir),
        torch_dtype=dtype,
        use_safetensors=True,
    )
    base_scheduler = DDPMScheduler.from_pretrained(
        MODEL_ID,
        revision=revision,
        cache_dir=str(cache_dir),
    )
    scheduler = (
        base_scheduler
        if scheduler_name == "ddpm"
        else DDIMScheduler.from_config(base_scheduler.config)
    )
    pipeline = DDPMPipeline(unet=unet, scheduler=scheduler)
    pipeline.to(device)
    pipeline.unet.eval()
    return pipeline


@torch.no_grad()
def sample_batch(
    pipeline: Any,
    count: int,
    steps: int,
    seed: int,
    device: torch.device,
    capture_trajectory: bool = False,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Run the scheduler explicitly so students can inspect the reverse process."""

    generator = torch.Generator(device=device).manual_seed(seed)
    size = int(pipeline.unet.config.sample_size)
    channels = int(pipeline.unet.config.in_channels)
    sample = torch.randn(
        count,
        channels,
        size,
        size,
        generator=generator,
        device=device,
        dtype=next(pipeline.unet.parameters()).dtype,
    )
    sample = sample * pipeline.scheduler.init_noise_sigma
    pipeline.scheduler.set_timesteps(steps, device=device)
    trajectory: list[torch.Tensor] = []
    capture = {0, max(0, steps // 4), max(0, steps // 2), max(0, 3 * steps // 4), steps - 1}
    for index, timestep in enumerate(pipeline.scheduler.timesteps):
        predicted_noise = pipeline.unet(sample, timestep).sample
        sample = pipeline.scheduler.step(
            predicted_noise, timestep, sample, generator=generator
        ).prev_sample
        if capture_trajectory and index in capture:
            trajectory.append((sample[:1] / 2 + 0.5).clamp(0, 1).float().cpu())
    images = (sample / 2 + 0.5).clamp(0, 1).float()
    return images, trajectory


def save_evidence(
    images: torch.Tensor,
    trajectory: list[torch.Tensor],
    output_dir: Path,
) -> None:
    from torchvision.utils import save_image

    output_dir.mkdir(parents=True, exist_ok=True)
    columns = max(1, int(len(images) ** 0.5))
    save_image(images.cpu(), output_dir / "ddpm_cifar10_samples.png", nrow=columns)
    if trajectory:
        save_image(
            torch.cat(trajectory).cpu(),
            output_dir / "ddpm_denoising_trajectory.png",
            nrow=len(trajectory),
        )


def require_fid_dependencies():
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
    except Exception as error:
        raise RuntimeError(
            "FID requires torchmetrics and torch-fidelity from the course requirements."
        ) from error
    return FrechetInceptionDistance


def real_cifar_loader(
    data_root: Path,
    split: str,
    batch_size: int,
    sample_count: int,
):
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, transforms

    training = split == "train"
    dataset = datasets.CIFAR10(
        data_root, train=training, download=True, transform=transforms.ToTensor()
    )
    if sample_count > len(dataset):
        raise ValueError(
            f"Requested {sample_count} real samples from a {len(dataset)}-image split."
        )
    generator = torch.Generator().manual_seed(705)
    indices = torch.randperm(len(dataset), generator=generator)[:sample_count].tolist()
    return DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )


@torch.no_grad()
def course_fid(
    pipeline: Any,
    config: SamplingConfig,
    device: torch.device,
    data_root: Path,
) -> float:
    """Compute a declared course FID; small-N values are not paper-comparable."""

    FrechetInceptionDistance = require_fid_dependencies()
    metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    real_loader = real_cifar_loader(
        data_root,
        config.reference_split,
        config.batch_size,
        config.fid_samples,
    )
    observed = 0
    for images, _ in real_loader:
        metric.update(images.to(device), real=True)
        observed += len(images)
    generated = 0
    batch_index = 0
    while generated < config.fid_samples:
        count = min(config.batch_size, config.fid_samples - generated)
        images, _ = sample_batch(
            pipeline,
            count=count,
            steps=config.inference_steps,
            seed=config.seed + 1000 + batch_index,
            device=device,
        )
        metric.update(images, real=False)
        generated += count
        batch_index += 1
        print(f"fid_progress={generated}/{config.fid_samples}")
    if observed != config.fid_samples:
        raise RuntimeError("Real reference count does not match generated count.")
    return float(metric.compute().cpu())


def audit_only(device: torch.device) -> None:
    beta, alpha, alpha_bar = linear_beta_schedule(100)
    clean = torch.ones(4, 3, 8, 8, device=device)
    timesteps = torch.tensor([0, 10, 50, 99], device=device)
    fixed_noise = torch.zeros_like(clean)
    noisy, target = q_sample(clean, timesteps, alpha_bar, fixed_noise)
    loss = noise_prediction_loss(target, torch.zeros_like(target))
    print("week12_ddpm_cifar10_audit")
    print(f"model={MODEL_ID}")
    print(f"default_revision={MODEL_REVISION}")
    print(f"device={device}")
    print(f"schedule_shapes={tuple(beta.shape)}/{tuple(alpha.shape)}/{tuple(alpha_bar.shape)}")
    print(f"alpha_bar_decreases={bool(torch.all(alpha_bar[1:] < alpha_bar[:-1]))}")
    print(f"q_sample_shape={tuple(noisy.shape)}")
    print(f"zero_noise_loss={float(loss):.4f}")
    print(f"diffusers_available={importlib.util.find_spec('diffusers') is not None}")
    print(f"torchmetrics_available={importlib.util.find_spec('torchmetrics') is not None}")
    print("downloads_started=False")


def run(
    config: SamplingConfig,
    device: torch.device,
    cache_dir: Path,
    data_root: Path,
    output_dir: Path,
    revision: str,
) -> None:
    pipeline = load_pipeline(device, cache_dir, config.scheduler, revision)
    print("week12_ddpm_cifar10")
    print(f"model={MODEL_ID}")
    print(f"model_revision={revision}")
    print(f"paper={PAPER_URL}")
    print(f"model_card={MODEL_CARD_URL}")
    print(f"device={device}")
    if device.type == "cuda":
        print(f"device_name={torch.cuda.get_device_name(device)}")
    print(f"unet_parameters={parameter_count(pipeline.unet)}")
    print(f"scheduler={config.scheduler} inference_steps={config.inference_steps}")
    print(f"sample_count={config.sample_count} seed={config.seed}")
    started = time.perf_counter()
    images, trajectory = sample_batch(
        pipeline,
        count=config.sample_count,
        steps=config.inference_steps,
        seed=config.seed,
        device=device,
        capture_trajectory=True,
    )
    seconds = time.perf_counter() - started
    save_evidence(images, trajectory, output_dir)
    print(f"sample_seconds={seconds:.2f}")
    print(f"seconds_per_image={seconds / config.sample_count:.3f}")
    print(f"evidence_directory={output_dir}")
    print("published_ddpm_cifar10_fid=3.17")
    print("published_result_source=DDPM_paper_50000_sample_protocol")
    if config.fid_samples:
        fid = course_fid(pipeline, config, device, data_root)
        print(
            f"course_fid={fid:.3f} samples={config.fid_samples} "
            f"real_reference={config.reference_split} scheduler={config.scheduler} "
            f"steps={config.inference_steps}"
        )
        if config.fid_samples < 50000:
            print("course_fid_comparable_to_published=False")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--scheduler", choices=("ddpm", "ddim"), default="ddim")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--sample-count", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--fid-samples", type=int, default=0)
    parser.add_argument("--reference-split", choices=("train", "test"), default="train")
    parser.add_argument("--seed", type=int, default=705)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/raw/huggingface"))
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/week12"))
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    if args.audit_only:
        audit_only(device)
        return
    if args.steps < 1 or args.sample_count < 1 or args.fid_samples < 0:
        raise ValueError("steps and sample-count must be positive; fid-samples cannot be negative.")
    config = SamplingConfig(
        seed=args.seed,
        sample_count=args.sample_count,
        batch_size=args.batch_size,
        inference_steps=args.steps,
        scheduler=args.scheduler,
        fid_samples=args.fid_samples,
        reference_split=args.reference_split,
    )
    run(
        config,
        device,
        args.cache_dir,
        args.data_root,
        args.output_dir,
        args.revision,
    )


if __name__ == "__main__":
    main()
