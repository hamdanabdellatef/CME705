# Week 12 — Autoencoders and Generative Models

**Guiding question:** How can a model learn a useful representation or distribution without a class label, and how should we evaluate what it generates?

Week 12 begins with deterministic reconstruction, converts the bottleneck into a probabilistic latent-variable model, and then compares explicit-density, implicit, score-based, and energy-based generators. The goal is to understand the mathematics and sampling path of each family before reading benchmark claims.

## Learning outcomes

By the end of the session, students can:

- distinguish representation learning from distribution modeling;
- calculate an autoencoder reconstruction objective;
- factorize an autoregressive likelihood;
- derive and interpret the VAE evidence lower bound;
- calculate the GAN generator and discriminator objectives;
- apply a one-dimensional change of variables;
- calculate a DDPM forward-corruption sample and noise-prediction loss;
- connect denoising with the score of a noisy distribution;
- define an energy model and explain why its partition function is difficult;
- select metrics whose protocol matches the claim;
- run and audit a modern diffusion checkpoint on CIFAR-10;
- propose a controlled generative-model ablation or robustness test.

## Preparation

Read the abstracts and equations in:

- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114);
- [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661);
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239).

Bring one unlabeled or generative research problem. State what an observation is, what a useful generated example would mean, and one harmful failure that a visually attractive sample could hide.

## Three-hour teaching sequence

| Time | Activity | Evidence |
| ---: | --- | --- |
| 0–15 min | Define representation and generation claims | Claim, observation space, and failure are named |
| 15–35 min | Autoencoder bottlenecks | Shapes, parameters, and reconstruction loss are traced |
| 35–55 min | Autoregressive factorization | Joint probability becomes legal conditional factors |
| 55–85 min | VAE derivation | ELBO, KL, and reparameterization are calculated |
| 85–110 min | Lab 12A | AE and VAE protocols are audited before training |
| 110–120 min | Break | — |
| 120–145 min | GANs and adversarial games | Both losses and failure modes are explained |
| 145–165 min | Normalizing flows and EBMs | Jacobian and partition-function costs are identified |
| 165–195 min | Diffusion and score models | Forward noise, reverse prediction, and sampling steps are traced |
| 195–215 min | Lab 12B | DDIM/DDPM sampling is tied to CIFAR-10 protocol |
| 215–235 min | Benchmarks and recent systems | FID claims are qualified by protocol and compute |
| 235–250 min | Research workshop and exit ticket | One falsifiable ablation is written |

## One unifying view

A generative model must specify at least one of these objects:

- a normalized density $p_\theta(x)$;
- a latent joint model $p_\theta(x,z)$;
- a sampler $x=G_\theta(z)$;
- a score $\nabla_x\log p_t(x)$;
- an unnormalized energy $E_\theta(x)$.

Ask four questions for every family: What is trained? What is tractable? How are samples produced? Which failure can the reported metric miss?

## Core equations

### Autoencoder

An encoder and decoder form

$$
z=f_\phi(x), \qquad \hat{x}=g_\theta(z).
$$

For continuous pixels, one reconstruction objective is

$$
L_{\mathrm{AE}}(x)=\lVert x-g_\theta(f_\phi(x))\rVert_2^2.
$$

A low reconstruction loss does not by itself establish a smooth latent distribution or realistic samples from arbitrary $z$.

### Autoregressive model

Choose an ordering and apply the chain rule:

$$
p_\theta(x)=\prod_{i=1}^{D}p_\theta(x_i\mid x_{<i}).
$$

The negative log-likelihood is exact when every conditional is normalized. Training can process known previous tokens in parallel under a causal mask; generation usually emits tokens sequentially.

### Variational autoencoder

Introduce a prior $p(z)$, decoder $p_\theta(x\mid z)$, and approximate posterior $q_\phi(z\mid x)$. The log evidence satisfies

$$
\log p_\theta(x)
\ge
\mathbb{E}_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]
-
D_{\mathrm{KL}}(q_\phi(z\mid x)\Vert p(z)).
$$

For a diagonal Gaussian posterior,

$$
z=\mu_\phi(x)+\sigma_\phi(x)\odot\epsilon,
\qquad \epsilon\sim\mathcal{N}(0,I).
$$

The KL term regularizes the encoded distribution toward the prior; too much pressure can cause posterior collapse.

### Generative adversarial network

A generator $G$ and discriminator $D$ play

$$
\min_G\max_D
\mathbb{E}_{x\sim p_{\mathrm{data}}}[\log D(x)]
+
\mathbb{E}_{z\sim p(z)}[\log(1-D(G(z)))].
$$

The common non-saturating generator instead minimizes $-\mathbb{E}_z\log D(G(z))$. Training can be unstable because each player changes the other player's objective; mode collapse can yield sharp but insufficiently diverse samples.

### Normalizing flow

Let $z=f_\theta(x)$ be invertible. Change of variables gives

$$
\log p_X(x)=\log p_Z(f_\theta(x))
+
\log\left|\det J_{f_\theta}(x)\right|.
$$

Flows offer exact likelihood and exact inversion, while architectural transformations must remain invertible with tractable Jacobian determinants.

### Diffusion and score models

A DDPM forward process adds Gaussian noise. With $\bar{\alpha}_t=\prod_{s=1}^{t}(1-\beta_s)$,

$$
q(x_t\mid x_0)=
\mathcal{N}(\sqrt{\bar{\alpha}_t}x_0,(1-\bar{\alpha}_t)I),
$$

so a training pair can be sampled directly:

$$
x_t=\sqrt{\bar{\alpha}_t}x_0+sqrt{1-\bar{\alpha}_t}\epsilon,
\qquad \epsilon\sim\mathcal{N}(0,I).
$$

A common simplified objective is

$$
L_{\mathrm{simple}}
=
\mathbb{E}_{x_0,t,\epsilon}
\left[\lVert\epsilon-\epsilon_\theta(x_t,t)\rVert_2^2\right].
$$

Score-based models learn $s_\theta(x,t)\approx\nabla_x\log p_t(x)$. DDPM noise prediction and denoising score matching are closely connected, while samplers differ in stochasticity, numerical path, and number of network evaluations.

### Energy-based model

An EBM defines

$$
p_\theta(x)=\frac{\exp[-E_\theta(x)]}{Z_\theta},
\qquad
Z_\theta=\int\exp[-E_\theta(x)]\,dx.
$$

Its likelihood gradient contrasts lower energy on data with lower energy on model samples. The partition function and negative-phase sampling are usually intractable, so training commonly relies on MCMC or approximations.

## Lab 12A — autoencoder and VAE design

Run the download-free audit first:

```bash
python labs/week12_autoencoder_pytorch.py --audit-only
```

Then compare both models on deterministic Fashion-MNIST subsets:

```bash
python labs/week12_autoencoder_pytorch.py
```

The script:

- uses balanced training, validation, and locked official-test subsets;
- selects checkpoints by validation objective;
- reports reconstruction BCE and MSE for both models;
- reports analytic KL for the VAE;
- saves reconstruction, prior-sample, and interpolation evidence under ignored `outputs/week12/`;
- selects CUDA automatically when the installed PyTorch build supports it.

A useful ablation changes only latent dimension or $\beta$. Record at least three seeds before treating a difference as stable.

## Lab 12B — modern DDPM on CIFAR-10

Audit the DDPM equations without downloads:

```bash
python labs/week12_ddpm_cifar10.py --audit-only
```

Generate 16 images with a 50-step DDIM sampler:

```bash
python labs/week12_ddpm_cifar10.py
```

The lab uses the public [google/ddpm-cifar10-32](https://huggingface.co/google/ddpm-cifar10-32) checkpoint pinned to revision `267b167dc01f0e4e61923ea244e8b988f84deb80`, exposes every reverse scheduler step, and saves a sample grid plus denoising trajectory. Compare speed under one fixed seed:

```bash
python labs/week12_ddpm_cifar10.py --scheduler ddpm --steps 250
python labs/week12_ddpm_cifar10.py --scheduler ddim --steps 25
```

An optional small FID audit is educational rather than paper-comparable:

```bash
python labs/week12_ddpm_cifar10.py --fid-samples 1000 --batch-size 64
```

The script prints the sample count, real reference split, scheduler, and steps beside the result. A defensible reproduction requires the same sample count, preprocessing, feature extractor, reference statistics, conditioning, and checkpoint as the cited result.

## Benchmark literacy

The following values are paper-reported CIFAR-10 FID scores. Lower is better. They form a dated research snapshot, not a single controlled leaderboard.

| Year | Model | Family and setting | Reported FID | Sampling evidence | Source |
| ---: | --- | --- | ---: | --- | --- |
| 2020 | StyleGAN2-ADA | GAN, limited-data study | 2.42 | paper protocol | [Karras et al.](https://arxiv.org/abs/2006.06676) |
| 2020 | DDPM | unconditional diffusion | 3.17 | full iterative sampler | [Ho et al.](https://arxiv.org/abs/2006.11239) |
| 2021 | NCSN++ / SDE | unconditional score model | 2.20 | predictor-corrector study | [Song et al.](https://arxiv.org/abs/2011.13456) |
| 2022 | EDM | unconditional diffusion | 1.97 | 35 network evaluations | [Karras et al.](https://arxiv.org/abs/2206.00364) |
| 2023 | Consistency model | one-step generation | 3.55 | one network evaluation | [Song et al.](https://arxiv.org/abs/2303.01469) |

Before comparing two rows, verify unconditional versus conditional generation, training data and augmentation, sample count, image resolution, feature extractor, reference statistics, checkpoint selection, and compute. FID estimates a distance between fitted feature-space Gaussians; it does not test prompt alignment, memorization, subgroup coverage, provenance, or downstream utility.

## Selected recent directions

| Direction | Main change | Research question |
| --- | --- | --- |
| latent diffusion | denoise compressed autoencoder latents | What detail is lost before generation starts? |
| Diffusion Transformers | replace the U-Net backbone with latent-patch Transformers | How does model compute scale with FID? |
| EDM | separate parameterization, preconditioning, schedule, and solver choices | Which design choice improves quality per network evaluation? |
| consistency models | map points on one probability-flow trajectory to a shared clean endpoint | How much quality remains with one or few steps? |
| flow matching | regress a continuous velocity field along chosen probability paths | Which path minimizes solver cost and training variance? |
| modern visual autoregression | predict discrete visual tokens with Transformer decoders | When does exact next-token training offset sequential sampling? |

Read [the Week 12 research guide](../../readings/week12-generative-models.md) for primary papers, metric cautions, and project directions.

## Research milestone

Complete [the generative-model ablation and robustness note](../../research-project/generative-model-ablation.md). Freeze the data split, preprocessing, seed set, evaluation implementation, and compute budget. Change one mechanism, state a directional prediction, and define what result would reject it.

## Package

- [Detailed notes](week12-notes.md)
- [Accessible slide source](week12-slides.md)
- [Student worksheet](week12-worksheet.md)
- [PowerPoint deck](slides/week12-autoencoders-and-generative-models.pptx)
- [Instructor guide](../../instructor-notes/week12.md)
- [Autoencoder/VAE lab](../../labs/week12_autoencoder_pytorch.py)
- [DDPM CIFAR-10 lab](../../labs/week12_ddpm_cifar10.py)
- [Further reading](../../readings/week12-generative-models.md)
- [Research milestone](../../research-project/generative-model-ablation.md)

## Exit ticket

In four sentences:

- distinguish reconstruction from density estimation;
- explain the VAE KL term;
- contrast autoregressive, adversarial, flow, diffusion, and energy sampling;
- name one protocol detail required before comparing two FID values.
