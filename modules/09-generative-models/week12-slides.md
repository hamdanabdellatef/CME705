# Week 12 slides — Autoencoders and Generative Models

This accessible source mirrors the PowerPoint deck. Equations use GitHub-compatible dollar delimiters.

## Slide 1: Autoencoders and generative models

**Guiding question:** How can a model compress observations, assign probability, or generate new examples, and what evidence shows that the learned distribution is useful?

CME705 Machine Learning — Week 12

## Slide 2: Today’s evidence

Students will produce:

1. an autoencoder shape trace;
2. an autoregressive factorization;
3. a VAE ELBO derivation;
4. a GAN game explanation;
5. a flow Jacobian calculation;
6. a DDPM corruption calculation;
7. a score and energy interpretation;
8. a benchmark-qualified ablation.

## Slide 3: Start with the claim

| Claim | Evidence needed |
| --- | --- |
| useful representation | downstream or robustness evidence |
| high likelihood | declared density and preprocessing |
| realistic samples | fidelity plus coverage |
| conditional control | alignment plus quality |
| novelty | training-set similarity audit |

A sample grid demonstrates selected outputs, not distribution coverage.

## Slide 4: Five objects organize the field

- normalized density $p_\theta(x)$;
- latent joint model $p_\theta(x,z)$;
- sampler $x=G_\theta(z)$;
- score $\nabla_x\log p_t(x)$;
- unnormalized energy $E_\theta(x)$.

Ask: what is trained, what is tractable, how are samples produced, and what can the metric miss?

## Slide 5: Deterministic autoencoder

$$
z=f_\phi(x)
$$

$$
\hat{x}=g_\theta(z)
$$

$$
L_{\mathrm{AE}}=\lVert x-\hat{x}\rVert_2^2
$$

The bottleneck preserves information useful for the declared reconstruction loss.

## Slide 6: A bottleneck is a hypothesis

| Too small | Useful constraint | Too large |
| --- | --- | --- |
| loses important detail | preserves stable structure | may copy or memorize |

A low reconstruction loss does not guarantee a smooth prior, semantic factors, novelty, or downstream utility.

## Slide 7: Convolutional shape trace

| Stage | Shape |
| --- | --- |
| image | $(N,1,28,28)$ |
| conv 1 | $(N,16,14,14)$ |
| conv 2 | $(N,32,7,7)$ |
| flatten | $(N,1568)$ |
| latent | $(N,d_z)$ |
| reconstruction | $(N,1,28,28)$ |

Lab 12A tests every shape before downloading Fashion-MNIST.

## Slide 8: Autoregressive chain rule

$$
p_\theta(x)=\prod_{i=1}^{D}p_\theta(x_i\mid x_{<i})
$$

$$
\log p_\theta(x)=\sum_{i=1}^{D}\log p_\theta(x_i\mid x_{<i})
$$

Every conditional is normalized. The ordering defines the prediction tasks.

## Slide 9: Parallel training, serial generation

During training, true previous tokens are available under a causal mask.

During generation:

$$
x_1\rightarrow x_2\rightarrow\cdots\rightarrow x_D
$$

Strength: exact next-token likelihood.

Cost: many sequential sampling decisions and exposure to model-generated histories.

## Slide 10: VAE is a latent probability model

- prior: $p(z)$;
- decoder: $p_\theta(x\mid z)$;
- approximate posterior: $q_\phi(z\mid x)$.

$$
p_\theta(x)=\int p_\theta(x\mid z)p(z)\,dz
$$

The integral and exact posterior are often intractable.

## Slide 11: Evidence lower bound

$$
\log p_\theta(x)
\ge
\mathbb{E}_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]
-
D_{\mathrm{KL}}(q_\phi(z\mid x)\Vert p(z))
$$

Reconstruction term: explain $x$ through $z$.

KL term: keep encoded distributions compatible with the prior.

## Slide 12: Reparameterization carries gradients

$$
q_\phi(z\mid x)=\mathcal{N}(\mu,\operatorname{diag}(\sigma^2))
$$

$$
\epsilon\sim\mathcal{N}(0,I)
$$

$$
z=\mu+\exp(0.5\log\sigma^2)\odot\epsilon
$$

Randomness is isolated in $\epsilon$; the path through $\mu$ and $\log\sigma^2$ remains differentiable.

## Slide 13: Beta controls a tradeoff

$$
L_{\beta\text{-VAE}}
=
-\mathbb{E}_q\log p_\theta(x\mid z)
+
\beta D_{\mathrm{KL}}(q_\phi(z\mid x)\Vert p(z))
$$

Larger $\beta$ can improve prior matching while losing reconstruction detail.

Posterior collapse warning: KL near zero may mean the decoder ignores $z$.

## Slide 14: Lab 12A protocol

- Fashion-MNIST official train and test splits;
- 10,000 balanced training examples;
- 2,000 balanced validation examples;
- 2,000 locked official-test examples;
- validation-selected AE and VAE checkpoints;
- BCE, MSE, KL, samples, and interpolation;
- automatic CUDA selection.

Change one factor: latent dimension or $\beta$.

## Slide 15: GAN is a two-player game

$$
\min_G\max_D
\mathbb{E}_{x\sim p_{\mathrm{data}}}\log D(x)
+
\mathbb{E}_{z\sim p(z)}\log(1-D(G(z)))
$$

The generator produces samples. The discriminator supplies a learned training signal.

A common generator loss is $-\mathbb{E}_z\log D(G(z))$.

## Slide 16: Sharp samples can hide missing modes

| Failure | Observable evidence |
| --- | --- |
| mode collapse | low diversity, weak recall |
| discriminator overfit | train/validation divergence |
| oscillation | unstable samples and losses |
| imbalance | saturated gradients |
| memorization | close training neighbors |

One discriminator loss curve is insufficient.

## Slide 17: Normalizing flow

For invertible $z=f_\theta(x)$:

$$
\log p_X(x)
=
\log p_Z(f_\theta(x))
+
\log|\det J_{f_\theta}(x)|
$$

Strengths: exact likelihood, exact latent inference, direct sampling.

Constraint: every transform needs a tractable inverse and determinant.

## Slide 18: Change-of-variables calculation

Let $z=2x+1$ and $z\sim\mathcal{N}(0,1)$.

$$
\frac{dz}{dx}=2
$$

$$
\log p_X(x)=\log p_Z(2x+1)+\log 2
$$

$$
x=\frac{z-1}{2}
$$

The determinant corrects probability mass for local scale change.

## Slide 19: Energy-based model

$$
p_\theta(x)=\frac{\exp[-E_\theta(x)]}{Z_\theta}
$$

$$
\nabla_\theta\log p_\theta(x)
=
-\nabla_\theta E_\theta(x)
+
\mathbb{E}_{p_\theta}\nabla_\theta E_\theta(x')
$$

Lower data energy; raise sampled negative energy.

Difficulty: partition function and MCMC mixing.

## Slide 20: DDPM forward corruption

$$
q(x_t\mid x_{t-1})
=
\mathcal{N}(\sqrt{1-\beta_t}x_{t-1},\beta_t I)
$$

$$
\bar{\alpha}_t=\prod_{s=1}^{t}(1-\beta_s)
$$

$$
x_t=\sqrt{\bar{\alpha}_t}x_0+sqrt{1-\bar{\alpha}_t}\epsilon
$$

Any noise level can be sampled directly during training.

## Slide 21: Work one noise step

Given:

$$
x_0=0.8, \quad \bar{\alpha}_t=0.64, \quad \epsilon=-0.5
$$

Then:

$$
x_t=0.8(0.8)+0.6(-0.5)=0.34
$$

The network receives $x_t$ and $t$ and learns the injected noise.

## Slide 22: Noise-prediction objective

$$
L_{\mathrm{simple}}
=
\mathbb{E}_{x_0,t,\epsilon}
\left[\lVert\epsilon-\epsilon_\theta(x_t,t)\rVert_2^2\right]
$$

Training samples one clean image, timestep, and noise realization.

The timestep embedding tells the denoiser the corruption level.

## Slide 23: Generation reverses corruption

Start with $x_T\sim\mathcal{N}(0,I)$.

$$
x_T\rightarrow x_{T-1}\rightarrow\cdots\rightarrow x_0
$$

DDPM: stochastic reverse steps.

DDIM: related deterministic or partially stochastic path, often fewer steps.

Report scheduler, steps, seed, precision, and seconds per image.

## Slide 24: Score is a noisy-density direction

$$
s_\theta(x_t,t)\approx\nabla_{x_t}\log p_t(x_t)
$$

For DDPM corruption:

$$
\nabla_{x_t}\log q(x_t\mid x_0)
=-\frac{\epsilon}{\sqrt{1-\bar{\alpha}_t}}
$$

Noise prediction and denoising score matching differ mainly by a known scale.

## Slide 25: Lab 12B — DDPM on CIFAR-10

- public `google/ddpm-cifar10-32` checkpoint;
- approximately 35.7 million U-Net parameters;
- automatic CUDA selection;
- explicit reverse scheduler loop;
- default 50-step DDIM sampler;
- sample grid and five-stage trajectory;
- optional protocol-declared FID audit.

Sixteen samples test the path, not the published benchmark.

## Slide 26: FID compares feature distributions

$$
\operatorname{FID}
=
\lVert\mu_r-\mu_g\rVert_2^2
+
\operatorname{Tr}
(\Sigma_r+\Sigma_g-2(\Sigma_r\Sigma_g)^{1/2})
$$

Lower is better, but state:

- sample count and reference set;
- feature extractor and preprocessing;
- conditioning and guidance;
- checkpoint and random seed.

## Slide 27: CIFAR-10 research snapshot

| Model | Family | FID | Sampling evidence |
| --- | --- | ---: | --- |
| StyleGAN2-ADA | GAN | 2.42 | paper protocol |
| DDPM | diffusion | 3.17 | full iterative sampler |
| NCSN++ / SDE | score | 2.20 | predictor-corrector study |
| EDM | diffusion | 1.97 | 35 evaluations |
| Consistency | direct mapping | 3.55 | one evaluation |

Paper-reported values; verify protocol before comparison.

## Slide 28: Recent directions change the compute path

| Direction | Main idea |
| --- | --- |
| latent diffusion | denoise compressed autoencoder latents |
| Diffusion Transformer | denoise latent patches with a Transformer |
| EDM | redesign preconditioning, schedules, and solvers |
| consistency model | one- or few-step trajectory mapping |

Quality must be paired with training and sampling compute.

## Slide 29: Autoregression and flows continue to evolve

| Direction | Main idea |
| --- | --- |
| flow matching | regress a continuous velocity field |
| LlamaGen | next-token prediction over visual tokens |
| modern flows | expressive invertible or continuous transforms |
| hybrid systems | autoencoder plus diffusion, AR, or flow prior |

Probabilistic family, representation, backbone, and sampler are separate design choices.

## Slide 30: Research direction and exit ticket

Ablation record:

1. one mechanism;
2. fixed controls and seed set;
3. directional prediction;
4. reconstruction, likelihood, or distribution metric;
5. compute and robustness evidence;
6. rejection condition.

Exit ticket: distinguish reconstruction from density, explain the VAE KL term, compare sampling paths, and name one FID protocol detail.
