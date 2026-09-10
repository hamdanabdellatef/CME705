# Instructor guide — Week 12

## Purpose

Week 12 should leave students with one coherent map of generative modeling rather than a list of model names. Emphasize seven distinct training objects:

1. deterministic reconstruction;
2. autoregressive likelihood;
3. variational lower bounds;
4. adversarial games;
5. exact change of variables;
6. denoising or score matching;
7. unnormalized energy.

The central teaching question is: what quantity does the model learn, and what procedure turns that quantity into a sample?

## Preparation checklist

Before class:

- read the lesson, notes, and worksheet;
- verify GitHub renders every equation in the accessible slide source;
- run both download-free lab audits;
- create a clean Python 3.11 or 3.12 environment from `requirements.txt`;
- pre-download Fashion-MNIST for Lab 12A if the network is unreliable;
- pre-download `google/ddpm-cifar10-32` for Lab 12B;
- verify CUDA with the command in `setup.md`;
- generate a 16-image DDIM grid before class;
- keep a CPU audit path available;
- do not show a paper FID beside a small course FID without protocol labels.

The diffusion checkpoint is about 143 MB on the cited Hugging Face repository and its model card reports roughly 35.7 million parameters. The default 50-step DDIM sampler is suitable for classroom inspection on a laptop GPU. CPU sampling may be slow; use the audit path for live explanation when no accelerator is available.

## Suggested timing

| Segment | Minutes | Instructor action |
| --- | ---: | --- |
| claims and model objects | 15 | separate representation, density, and sampling claims |
| autoencoder | 20 | trace shapes and bottleneck assumptions |
| autoregression | 20 | factorize one binary vector |
| VAE | 30 | derive ELBO and reparameterization |
| Lab 12A | 25 | audit protocol, then launch training |
| break | 10 | — |
| GAN | 25 | play both optimization roles |
| flows and EBMs | 20 | calculate determinant and contrast phases |
| diffusion and score | 30 | calculate forward corruption and reverse path |
| Lab 12B | 20 | inspect scheduler, trajectory, and timing |
| benchmark and research | 20 | qualify FID and write one rejection condition |

## Board sequence

### 1. Start from a claim

Write three statements:

- “the representation reconstructs held-out inputs”;
- “the model assigns a normalized likelihood”;
- “the sampler covers the target distribution.”

Ask students what evidence would be required for each. This prevents later confusion between reconstruction loss, likelihood, and sample quality.

### 2. Autoencoder

Draw

$$
x\rightarrow z\rightarrow\hat{x}.
$$

Keep input and reconstruction shapes visible. Ask where information can be lost and what prevents identity copying. Answers may include latent dimension, noise, sparsity, regularization, architecture, or data augmentation.

Do not claim that every bottleneck learns semantics. The learned representation serves the reconstruction objective and architectural constraints.

### 3. Autoregressive factorization

For four variables, derive

$$
p(x_1,x_2,x_3,x_4)
=p(x_1)p(x_2\mid x_1)p(x_3\mid x_1,x_2)p(x_4\mid x_1,x_2,x_3).
$$

Use factor values $0.8$, $0.7$, $0.5$, and $0.9$. The joint value is

$$
0.8\times0.7\times0.5\times0.9=0.252.
$$

Distinguish parallel loss calculation during teacher-forced Transformer training from serial token sampling.

### 4. VAE ELBO

Begin with

$$
\log p_\theta(x)
=
\log\mathbb{E}_{q_\phi(z\mid x)}
\left[
\frac{p_\theta(x,z)}{q_\phi(z\mid x)}
\right].
$$

Apply Jensen's inequality, factor the joint, and arrive at

$$
\mathcal{L}_{\mathrm{ELBO}}
=
\mathbb{E}_q\log p_\theta(x\mid z)
-D_{\mathrm{KL}}(q_\phi(z\mid x)\Vert p(z)).
$$

Name each term by its probabilistic meaning before calling it reconstruction plus regularization. The decoder term is a conditional log likelihood. The KL term also determines how well prior samples reach regions used by the decoder.

For worksheet values $\mu=0.5$ and $\log\sigma^2=0$:

$$
\sigma^2=1,
$$

$$
D_{\mathrm{KL}}=\frac{1}{2}(0.25+1-1-0)=0.125.
$$

With $\epsilon=-1$ and $\sigma=1$:

$$
z=-0.5.
$$

### 5. GAN game

Assign half the room the generator objective and half the discriminator objective. Ask what happens after either side improves. This makes the non-stationary optimization problem concrete.

Use the non-saturating generator objective in practice:

$$
L_G=-\mathbb{E}_z\log D(G(z)).
$$

Do not diagnose training from losses alone. Require fixed latent samples, held-out discriminator evidence, precision and recall or similar coverage evidence, and training-neighbor checks.

### 6. Flow calculation

For worksheet transform $z=3x-2$:

$$
\frac{dz}{dx}=3,
$$

$$
\log p_X(x)=\log p_Z(3x-2)+\log 3,
$$

and

$$
x=\frac{z+2}{3}.
$$

The determinant is a density correction, not an optimization penalty. Connect a high-dimensional triangular coupling Jacobian to a sum of scale outputs.

### 7. EBM phases

Write

$$
\nabla_\theta\log p_\theta(x)
=
-\nabla_\theta E_\theta(x)
+
\mathbb{E}_{p_\theta}\nabla_\theta E_\theta(x').
$$

Lower energy for data and raise it for model samples. The expectation requires samples from the current model; poor MCMC mixing can bias learning.

### 8. DDPM calculation

For the worksheet:

$$
\sqrt{0.64}=0.8,
$$

$$
\sqrt{1-0.64}=0.6,
$$

$$
x_t=0.8(0.8)+0.6(-0.5)=0.34.
$$

If $\hat{\epsilon}=-0.3$ and $\epsilon=-0.5$:

$$
(\epsilon-\hat{\epsilon})^2=(-0.2)^2=0.04.
$$

Then connect noise prediction to the conditional score:

$$
\nabla_{x_t}\log q(x_t\mid x_0)
=-\frac{\epsilon}{\sqrt{1-\bar{\alpha}_t}}.
$$

### 9. Sampling path

Draw noise at the left and a sample at the right. Mark every U-Net evaluation. Compare:

- DDPM stochastic steps;
- DDIM fewer deterministic or partially stochastic steps;
- EDM schedule and solver changes;
- consistency one- or few-step mapping;
- flow-matching ODE integration.

The network backbone and numerical sampler are separate experimental factors.

## Lab 12A facilitation

Start with:

```bash
python labs/week12_autoencoder_pytorch.py --audit-only
```

Expected CPU audit markers include:

```text
week12_autoencoder_vae_audit
ae_shapes=(4, 1, 28, 28)/(4, 8)
vae_shapes=(4, 1, 28, 28)/(4, 8)/(4, 8)/(4, 8)
zero_noise_returns_mean=True
balanced_indices=50/20
disjoint_indices=True
downloads_started=False
```

Then run:

```bash
python labs/week12_autoencoder_pytorch.py
```

Require students to record:

- exact device and package versions;
- split sizes and seed;
- latent dimension and $\beta$;
- model parameter counts;
- validation history and selected epoch;
- one locked-test evaluation;
- reconstruction BCE and MSE;
- VAE KL;
- sample and interpolation files;
- at least ten inspected failures.

The AE and VAE total objectives are not directly equivalent because the VAE includes KL and stochastic sampling. Compare reconstruction evidence, prior sampling, latent activity, compute, and the specific research claim.

## Lab 12B facilitation

Start with:

```bash
python labs/week12_ddpm_cifar10.py --audit-only
```

Expected markers include decreasing $\bar{\alpha}_t$, the correct noisy-batch shape, and `downloads_started=False`.

For the pretrained model:

```bash
python labs/week12_ddpm_cifar10.py --scheduler ddim --steps 50
```

The script saves a grid and five-stage trajectory. Ask students to identify which changes affect the learned model and which only affect the sampler.

A useful controlled comparison is:

```bash
python labs/week12_ddpm_cifar10.py --scheduler ddim --steps 25 --seed 705
python labs/week12_ddpm_cifar10.py --scheduler ddim --steps 100 --seed 705
```

Record seconds per image. Do not infer a stable quality difference from one grid.

## Benchmark discussion

The deck includes these paper-reported CIFAR-10 values:

| Model | FID | Qualifier |
| --- | ---: | --- |
| StyleGAN2-ADA | 2.42 | limited-data GAN study |
| DDPM | 3.17 | unconditional iterative diffusion |
| NCSN++ / SDE | 2.20 | score-SDE predictor-corrector study |
| EDM | 1.97 | unconditional, 35 network evaluations |
| Consistency model | 3.55 | one-step generation |

Use the table to ask what is missing. Students should request sample count, feature code, real statistics, conditioning, augmentation, architecture scale, training compute, and uncertainty before treating rows as a ranking.

A course FID with 1,000 samples demonstrates the metric pipeline and has substantial estimator uncertainty. It must not be presented as a reproduction of a 50,000-sample paper protocol.

## Recent-model framing

Use recent models to show how components combine:

- latent diffusion: autoencoder representation plus diffusion prior;
- DiT: Transformer backbone inside a diffusion system;
- EDM: parameterization, schedule, preconditioning, and solver design;
- consistency: direct trajectory mapping for fewer evaluations;
- flow matching: continuous normalizing flow trained by velocity regression;
- LlamaGen: tokenizer plus autoregressive Transformer.

Avoid describing “diffusion,” “Transformer,” or “autoencoder” as complete systems without naming the other components.

## Research milestone

Students complete `research-project/generative-model-ablation.md`. Acceptable factors include:

- latent dimension;
- VAE $\beta$;
- decoder capacity;
- diffusion scheduler or steps;
- guidance scale in a conditional extension;
- data augmentation;
- corruption level;
- sampling temperature;
- calibration or anomaly threshold.

The selected factor must match the hypothesis. Every other declared control stays fixed. Require at least three seeds for a training comparison when feasible, plus compute and failure evidence.

## Common misconceptions

### “Autoencoders are generative models by default”

A deterministic autoencoder does not define a usable prior over latent codes. Sampling arbitrary latent points may reach unsupported decoder regions.

### “The ELBO is the likelihood”

It is a lower bound. The gap is a KL divergence between the approximate and true posterior.

### “GAN losses should converge to zero”

The two-player losses are coupled and scale-dependent. Sample quality and coverage require separate evidence.

### “Flows can use any neural layer”

The overall map must remain invertible and have a tractable determinant.

### “Diffusion models learn to add noise”

The forward noising process is fixed. The network learns a reverse-relevant prediction, often the injected noise.

### “Low FID proves the generator is safe and novel”

FID does not test memorization, provenance, condition accuracy, subgroup coverage, factuality, or downstream harm.

## Exit-ticket answer guide

A strong answer states that reconstruction learns to preserve information under a loss but does not alone define a density. The VAE KL term makes encoded distributions compatible with a prior and participates in a lower bound. Autoregressive models sample sequentially, GANs and flows can sample in one main pass, diffusion models use an iterative solver, and EBMs often use MCMC. FID comparisons require matching sample count, features, preprocessing, reference statistics, conditioning, and checkpoint protocol.
