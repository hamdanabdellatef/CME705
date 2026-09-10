# Week 12 notes — Learning representations and distributions

## 1. What does a generative model claim?

A classifier estimates a target conditional such as $p(y\mid x)$. A generative model may instead estimate a data density $p(x)$, a joint density $p(x,z)$, a conditional generator $p(x\mid c)$, a sampling mechanism, or an unnormalized energy. These are different claims.

For an image problem, define:

- the observation space and pixel representation;
- whether generation is unconditional or conditioned on a class, text, or another image;
- the training distribution and allowed external data;
- the sample procedure and compute budget;
- the metric and its exact implementation;
- the intended use and a failure that matters for that use.

A visually pleasing grid is evidence that the program can produce some plausible examples. It is not evidence of complete distribution coverage, novelty, calibrated likelihood, factuality, fairness, privacy, or usefulness.

### 1.1 Five mathematical objects

| Object | Example family | Normalized density? | Typical sampling path |
| --- | --- | --- | --- |
| conditional factors | autoregressive model | yes | sequential conditionals |
| latent joint model | VAE | lower bound used | sample latent, then decode |
| implicit sampler | GAN | usually no tractable density | one generator pass |
| invertible map | normalizing flow | yes | transform a base sample |
| score or energy | diffusion, score model, EBM | indirect or unnormalized | iterative solver or MCMC |

A deterministic autoencoder is primarily a representation model. It becomes a generator only after a distribution over useful latent codes is supplied.

## 2. Deterministic autoencoders

An encoder compresses an observation:

$$
z=f_\phi(x), \qquad z\in\mathbb{R}^{d_z}.
$$

A decoder reconstructs it:

$$
\hat{x}=g_\theta(z).
$$

The parameters minimize empirical reconstruction risk:

$$
\min_{\phi,\theta}
\frac{1}{N}\sum_{n=1}^{N}
\ell(x_n,g_\theta(f_\phi(x_n))).
$$

For real-valued data, mean squared error corresponds to the negative log likelihood of an isotropic Gaussian decoder up to constants and a scale factor. For pixels represented in $[0,1]$, binary cross-entropy is sometimes used as a Bernoulli decoder objective. The observation model must be stated; neither loss is automatically correct for every image pipeline.

### 2.1 Why a bottleneck can help

If $d_z$ is smaller than the input dimension or the encoder is otherwise constrained, the network cannot copy every coordinate independently. It must preserve information useful for reconstruction under the chosen model and training distribution.

A bottleneck does not guarantee semantic factors. An over-capacity network may memorize training examples, and an under-capacity network may preserve low-level averages while discarding rare but important structure.

### 2.2 Convolutional shape trace

The Week 12 lab uses Fashion-MNIST images with shape $(N,1,28,28)$:

| Stage | Shape |
| --- | --- |
| input | $(N,1,28,28)$ |
| stride-2 convolution | $(N,16,14,14)$ |
| stride-2 convolution | $(N,32,7,7)$ |
| flatten | $(N,1568)$ |
| latent projection | $(N,d_z)$ |
| decoder projection | $(N,1568)$ |
| two transposed convolutions | $(N,1,28,28)$ |

The code tests these shapes before downloading data.

### 2.3 Representation evidence

Useful evidence may include:

- reconstruction on a locked test split;
- interpolation between encoded examples;
- neighborhood consistency under a declared distance;
- linear-probe performance on frozen representations;
- robustness to corruptions not used for selection;
- sensitivity to latent dimension and regularization;
- nearest-neighbor checks against the training set.

Latent interpolation is descriptive. A smooth-looking path does not establish disentanglement or causal factors.

## 3. Autoregressive models

For a vector $x=(x_1,\ldots,x_D)$, the probability chain rule gives

$$
p(x)=p(x_1)\prod_{i=2}^{D}p(x_i\mid x_1,\ldots,x_{i-1}).
$$

An autoregressive model parameterizes every factor:

$$
p_\theta(x)=\prod_{i=1}^{D}p_\theta(x_i\mid x_{<i}).
$$

The log likelihood is a sum:

$$
\log p_\theta(x)=\sum_{i=1}^{D}\log p_\theta(x_i\mid x_{<i}).
$$

This yields an exact normalized likelihood if each conditional is normalized.

### 3.1 Teacher forcing and sampling

During training, every true previous coordinate or token is known. A causal mask lets a Transformer predict all next-token distributions in parallel. During generation, the model samples $x_1$, feeds it back to sample $x_2$, and continues. The number of serial decisions can make high-dimensional generation slow.

Exposure bias describes the difference between conditioning on true histories during training and model-generated histories during sampling. Sampling temperature, top-$k$, and nucleus truncation change the sampling distribution and must be reported.

### 3.2 Worked binary example

Suppose

$$
p(x_1=1)=0.7,
$$

$$
p(x_2=1\mid x_1=1)=0.8,
$$

and

$$
p(x_3=0\mid x_1=1,x_2=1)=0.6.
$$

Then

$$
p(1,1,0)=0.7\times0.8\times0.6=0.336.
$$

Changing the ordering changes the conditional tasks that the network must learn, although the exact chain rule can represent the same joint distribution.

### 3.3 From pixels to visual tokens

[PixelRNN](https://arxiv.org/abs/1601.06759) models raw image pixels in a spatial order. Modern visual autoregressive systems often first compress images into discrete tokens, then apply next-token Transformers. [LlamaGen](https://arxiv.org/abs/2406.06525) is a 2024 example that studies tokenizer quality, model scaling, and ImageNet generation. Tokenizer reconstruction quality forms an upper bound on what the token generator can reproduce.

## 4. Variational autoencoders

A VAE specifies:

- prior $p(z)$, commonly $\mathcal{N}(0,I)$;
- decoder or likelihood $p_\theta(x\mid z)$;
- approximate posterior $q_\phi(z\mid x)$.

The marginal likelihood is

$$
p_\theta(x)=\int p_\theta(x,z)\,dz.
$$

The posterior $p_\theta(z\mid x)$ and this integral are often intractable.

### 4.1 Deriving the ELBO

Multiply and divide inside the integral by $q_\phi(z\mid x)$:

$$
\log p_\theta(x)
=
\log\int q_\phi(z\mid x)
\frac{p_\theta(x,z)}{q_\phi(z\mid x)}\,dz.
$$

Apply Jensen's inequality:

$$
\log p_\theta(x)
\ge
\mathbb{E}_{q_\phi(z\mid x)}
\left[
\log p_\theta(x,z)-\log q_\phi(z\mid x)
\right].
$$

Factor $p_\theta(x,z)=p_\theta(x\mid z)p(z)$:

$$
\mathcal{L}_{\mathrm{ELBO}}(x)
=
\mathbb{E}_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]
-
D_{\mathrm{KL}}(q_\phi(z\mid x)\Vert p(z)).
$$

The identity

$$
\log p_\theta(x)
=
\mathcal{L}_{\mathrm{ELBO}}(x)
+
D_{\mathrm{KL}}(q_\phi(z\mid x)\Vert p_\theta(z\mid x))
$$

shows why it is a lower bound: KL divergence is nonnegative.

### 4.2 Diagonal Gaussian KL

Let

$$
q_\phi(z\mid x)=\mathcal{N}(\mu,\operatorname{diag}(\sigma^2)),
\qquad p(z)=\mathcal{N}(0,I).
$$

Then

$$
D_{\mathrm{KL}}(q\Vert p)
=
\frac{1}{2}
\sum_{j=1}^{d_z}
\left(mu_j^2+\sigma_j^2-1-\log\sigma_j^2\right).
$$

The implementation usually predicts $\log\sigma^2$ for numerical stability.

### 4.3 Reparameterization

Direct sampling appears to block gradients through random $z$. Move the randomness to a parameter-free source:

$$
\epsilon\sim\mathcal{N}(0,I),
$$

$$
z=\mu_\phi(x)+\exp\left(\frac{1}{2}\log\sigma_\phi^2(x)\right)\odot\epsilon.
$$

For fixed $\epsilon$, $z$ is differentiable with respect to $\mu$ and $\log\sigma^2$.

### 4.4 Beta-VAE objective

A common controlled modification is

$$
L_{\beta\text{-VAE}}
=
-\mathbb{E}_q\log p_\theta(x\mid z)
+
\beta D_{\mathrm{KL}}(q_\phi(z\mid x)\Vert p(z)).
$$

Increasing $\beta$ strengthens prior matching but can reduce reconstruction detail. It does not guarantee interpretable factors.

### 4.5 Posterior collapse

If the decoder can model $x$ while ignoring $z$, the approximate posterior can approach the prior:

$$
q_\phi(z\mid x)\approx p(z),
$$

and the KL term approaches zero. Monitor KL per dimension, active latent units, reconstruction, and controlled decoder-capacity changes. A low total loss can hide an unused latent representation.

## 5. Generative adversarial networks

A generator maps noise to a sample:

$$
\tilde{x}=G_\theta(z), \qquad z\sim p(z).
$$

A discriminator estimates whether an input came from data. The original game is

$$
\min_G\max_D V(D,G)
=
\mathbb{E}_{x\sim p_{\mathrm{data}}}[\log D(x)]
+
\mathbb{E}_{z\sim p(z)}[\log(1-D(G(z)))].
$$

For a fixed generator, the optimal discriminator is

$$
D^*(x)=\frac{p_{\mathrm{data}}(x)}{p_{\mathrm{data}}(x)+p_g(x)}.
$$

Substitution links the idealized objective to Jensen-Shannon divergence. Finite networks, alternating stochastic updates, and disjoint supports make real training harder than this ideal analysis.

### 5.1 Practical losses

The discriminator minimizes binary cross-entropy for real and generated examples. The non-saturating generator loss is

$$
L_G=-\mathbb{E}_{z\sim p(z)}\log D(G(z)).
$$

It produces stronger gradients early in training than minimizing $\log(1-D(G(z)))$.

### 5.2 Failure modes

- mode collapse: many latent codes produce similar samples;
- discriminator overfitting: the discriminator memorizes a limited dataset;
- oscillation: neither player reaches a stable region;
- gradient imbalance: one player becomes too accurate or too weak;
- metric gaming: quality improves while coverage declines.

[StyleGAN2-ADA](https://arxiv.org/abs/2006.06676) adapts discriminator augmentation strength to reduce overfitting with limited data. Augmentation must not leak into generated outputs or change the target distribution.

## 6. Normalizing flows

A normalizing flow composes invertible transformations:

$$
z=f_K\circ f_{K-1}\circ\cdots\circ f_1(x).
$$

For one invertible transform $z=f(x)$, conservation of probability mass gives

$$
p_X(x)=p_Z(f(x))\left|\det J_f(x)\right|.
$$

Taking logarithms gives

$$
\log p_X(x)
=
\log p_Z(f(x))
+
\log\left|\det J_f(x)\right|.
$$

For a composition, log determinants add. Training can maximize exact log likelihood, and sampling applies the inverse map to a base sample.

### 6.1 One-dimensional affine example

Let

$$
z=f(x)=2x+1,
$$

with $z\sim\mathcal{N}(0,1)$. The Jacobian is $dz/dx=2$, so

$$
\log p_X(x)=\log p_Z(2x+1)+\log 2.
$$

Sampling reverses the map:

$$
x=\frac{z-1}{2}.
$$

The absolute determinant corrects for local expansion or contraction. Omitting it would assign the wrong probability mass.

### 6.2 Coupling layers

A common layer partitions $x=(x_a,x_b)$:

$$
y_a=x_a,
$$

$$
y_b=x_b\odot\exp(s(x_a))+t(x_a).
$$

The Jacobian is triangular, so

$$
\log|\det J|=\sum_j s_j(x_a).
$$

The inverse is easy because $x_a=y_a$ and

$$
x_b=(y_b-t(y_a))\odot\exp(-s(y_a)).
$$

[Glow](https://arxiv.org/abs/1807.03039) adds invertible $1\times1$ convolutions to mix channels. [Flow++](https://arxiv.org/abs/1902.00275) studies variational dequantization, expressive coupling transforms, and conditioning networks.

### 6.3 Limits of likelihood

Exact likelihood is valuable, but likelihood can favor low-level statistics that do not match human semantic similarity. Report sample evidence and task-specific utility beside bits per dimension. For discrete pixels, state the dequantization procedure before comparing likelihoods.

## 7. Energy-based models

An EBM assigns a scalar energy:

$$
E_\theta(x)\in\mathbb{R}.
$$

Low energy means high unnormalized compatibility:

$$
p_\theta(x)=\frac{\exp[-E_\theta(x)]}{Z_\theta},
$$

$$
Z_\theta=\int \exp[-E_\theta(x)]\,dx.
$$

The normalizing constant depends on all possible observations and is usually intractable.

### 7.1 Likelihood gradient

For one data example,

$$
\nabla_\theta\log p_\theta(x)
=
-\nabla_\theta E_\theta(x)
+
\mathbb{E}_{x'\sim p_\theta}
[\nabla_\theta E_\theta(x')].
$$

The positive phase lowers energy on observed data. The negative phase raises energy on model samples. Estimating the negative expectation requires samples from the current model.

### 7.2 Langevin sampling

One approximate sampler iterates

$$
x_{k+1}
=
x_k-\frac{\eta}{2}\nabla_x E_\theta(x_k)
+\sqrt{\eta}\,\xi_k,
$$

where $\xi_k\sim\mathcal{N}(0,I)$. Step size, number of steps, initialization, replay buffers, and mixing all affect the result.

[Implicit Generation and Generalization in Energy-Based Models](https://arxiv.org/abs/1903.08689) demonstrates MCMC-trained neural EBMs on CIFAR-10 and other high-dimensional domains. An energy score is relative; it is not a normalized probability until the partition function is handled.

## 8. Denoising diffusion probabilistic models

A DDPM defines a fixed forward Markov chain:

$$
q(x_t\mid x_{t-1})
=
\mathcal{N}(\sqrt{1-\beta_t}x_{t-1},\beta_t I).
$$

Define

$$
\alpha_t=1-\beta_t,
\qquad
\bar{\alpha}_t=\prod_{s=1}^{t}\alpha_s.
$$

Repeated Gaussian transitions have the closed form

$$
q(x_t\mid x_0)
=
\mathcal{N}(\sqrt{\bar{\alpha}_t}x_0,(1-\bar{\alpha}_t)I).
$$

Therefore any noise level can be sampled in one operation:

$$
x_t
=
\sqrt{\bar{\alpha}_t}x_0
+
\sqrt{1-\bar{\alpha}_t}\epsilon,
\qquad
\epsilon\sim\mathcal{N}(0,I).
$$

### 8.1 Worked forward step

Suppose one scalar pixel is $x_0=0.8$, $\bar{\alpha}_t=0.64$, and sampled noise is $\epsilon=-0.5$. Then

$$
x_t
=0.8(0.8)+0.6(-0.5)
=0.34.
$$

The model receives $x_t$ and $t$ and predicts the injected noise, clean sample, velocity, or another equivalent parameterization.

### 8.2 Noise-prediction training

A common objective is

$$
L_{\mathrm{simple}}
=
\mathbb{E}_{x_0,t,\epsilon}
\left[
\lVert\epsilon-\epsilon_\theta(x_t,t)\rVert_2^2
\right].
$$

The timestep embedding tells the network the corruption scale. A U-Net or Transformer provides the denoiser.

### 8.3 Reverse process

Generation begins with approximately Gaussian noise $x_T$. A learned reverse chain uses

$$
p_\theta(x_{t-1}\mid x_t)
=
\mathcal{N}(\mu_\theta(x_t,t),\Sigma_t).
$$

When the network predicts $\epsilon$, one mean parameterization is

$$
\mu_\theta(x_t,t)
=
\frac{1}{\sqrt{\alpha_t}}
\left(
 x_t-
 \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}
 \epsilon_\theta(x_t,t)
\right).
$$

A DDPM sampler adds appropriate noise at intermediate steps. DDIM follows a related non-Markovian construction and can use a deterministic path with fewer steps. Scheduler name, timestep schedule, stochasticity, and network evaluations are part of the model result.

## 9. Score-based models

The score of a density is

$$
s(x)=\nabla_x\log p(x).
$$

For the DDPM conditional Gaussian,

$$
\nabla_{x_t}\log q(x_t\mid x_0)
=
-\frac{x_t-\sqrt{\bar{\alpha}_t}x_0}{1-\bar{\alpha}_t}
=
-\frac{\epsilon}{\sqrt{1-\bar{\alpha}_t}}.
$$

Noise prediction therefore gives a scaled estimate of the conditional score. Denoising score matching trains across noise levels without requiring the data density's normalizing constant.

[Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456) describes a forward SDE that transports data to a simple prior and a reverse-time SDE driven by the learned score. Predictor-corrector solvers and a probability-flow ODE offer different quality, likelihood, and compute tradeoffs.

## 10. How the families differ

| Family | Training signal | Density | Sampling | Main difficulty |
| --- | --- | --- | --- | --- |
| autoencoder | reconstruction | none by default | decode chosen latent | latent space may be irregular |
| autoregressive | exact conditional NLL | tractable | sequential | serial generation |
| VAE | ELBO | lower bound | one latent plus decoder | posterior or decoder mismatch |
| GAN | adversarial classification | implicit | one generator pass | unstable game and mode coverage |
| normalizing flow | exact NLL | tractable | invert base transform | invertibility and Jacobian constraints |
| diffusion / score | denoising or score matching | indirect or computable with extra work | iterative solver | many network evaluations |
| EBM | energy contrast | unnormalized | MCMC or solver | partition function and mixing |

Architecture and probabilistic family are separate choices. A diffusion denoiser can be a U-Net or Transformer. A VAE decoder can be autoregressive. Latent diffusion combines an autoencoder with a diffusion model.

## 11. Evaluation

### 11.1 Reconstruction metrics

MSE, BCE, PSNR, SSIM, and perceptual distances compare a reconstruction with its source. They do not measure unconditional generation quality. Report the observation scale and reduction convention.

### 11.2 Likelihood and bits per dimension

For an image with $D$ dimensions,

$$
\operatorname{bpd}(x)
=-\frac{\log_2 p(x)}{D}.
$$

Compare values only when preprocessing, dequantization, dimension count, and test split match. High likelihood does not ensure semantically convincing samples.

### 11.3 Fréchet Inception Distance

Fit Gaussian summaries to real and generated Inception features:

$$
\operatorname{FID}
=
\lVert\mu_r-\mu_g\rVert_2^2
+
\operatorname{Tr}
\left(
\Sigma_r+\Sigma_g
-2(\Sigma_r\Sigma_g)^{1/2}
\right).
$$

Lower is better. FID depends on:

- sample count;
- feature extractor and software implementation;
- resizing and pixel range;
- real reference statistics;
- random seed and checkpoint;
- conditioning and guidance;
- dataset duplication and contamination.

FID is biased at finite sample sizes. [The TTUR paper](https://arxiv.org/abs/1706.08500) introduced the metric in the GAN evaluation setting.

### 11.4 KID, precision, and recall

Kernel Inception Distance uses a polynomial-kernel maximum mean discrepancy and has an unbiased estimator under its standard construction. Generative precision asks whether samples lie near the data manifold; recall asks whether the generator covers the data distribution. One scalar rarely diagnoses both fidelity and diversity.

### 11.5 Memorization and human evaluation

Use nearest-neighbor and duplicate searches against training data, but do not treat one feature space as proof of novelty. Human evaluation needs a sampling frame, blinded protocol, clear question, sufficient raters, and uncertainty. For text- or condition-guided generation, separate condition alignment from visual quality.

## 12. CIFAR-10 snapshot

These paper-reported FID results illustrate quality and sampling-compute tradeoffs. They are not automatically comparable.

| Model | Family | Setting | FID | Network evaluations or protocol |
| --- | --- | --- | ---: | --- |
| StyleGAN2-ADA | GAN | limited-data CIFAR-10 study | 2.42 | paper protocol |
| DDPM | diffusion | unconditional | 3.17 | original iterative sampler |
| NCSN++ / SDE | score model | unconditional | 2.20 | predictor-corrector study |
| EDM | diffusion design study | unconditional | 1.97 | 35 evaluations |
| Consistency model | direct trajectory mapping | one-step | 3.55 | 1 evaluation |

A fair comparison must verify the dataset version, labels or conditioning, augmentation, architecture scale, training compute, generated-sample count, feature statistics, and implementation. The Week 12 lab's default 16-sample grid is a software and process demonstration, not a FID reproduction.

## 13. Recent generative directions

### 13.1 Latent diffusion

[Latent Diffusion Models](https://arxiv.org/abs/2112.10752) first compress images with an autoencoder and then denoise in latent space. This reduces spatial compute, while reconstruction loss and latent scaling become part of the generation ceiling.

### 13.2 Diffusion Transformers

[DiT](https://arxiv.org/abs/2212.09748) replaces the common U-Net denoiser with a Transformer over latent patches. The paper studies forward-pass compute and reports class-conditional ImageNet-256 FID. The key research question is quality per training and sampling compute, not architecture name alone.

### 13.3 EDM

[Elucidating the Design Space of Diffusion Models](https://arxiv.org/abs/2206.00364) separates data scaling, network preconditioning, noise distributions, schedules, and numerical solvers. It shows that training and sampling choices can improve an existing model family substantially.

### 13.4 Consistency models

[Consistency Models](https://arxiv.org/abs/2303.01469) learn a mapping that is consistent along a probability-flow trajectory. They target one- or few-step generation and can be trained by distillation or independently.

### 13.5 Flow matching

[Flow Matching](https://arxiv.org/abs/2210.02747) regresses a continuous velocity field along chosen probability paths without simulating the full path during training. Sampling integrates an ODE. Path choice and solver tolerance affect speed and accuracy.

### 13.6 Modern visual autoregression

[LlamaGen](https://arxiv.org/abs/2406.06525) applies next-token Transformer scaling to discrete visual tokens. Separate tokenizer reconstruction quality from the autoregressive generator's performance.

## 14. Week 12 lab protocol

### 14.1 Autoencoder and VAE

The lab uses Fashion-MNIST because it is small enough for controlled laptop experiments and contains ten visual categories. It uses balanced subsets:

- 10,000 training images;
- 2,000 validation images;
- 2,000 locked official-test images.

The selected checkpoint minimizes validation objective. The final report includes AE and VAE parameter counts, reconstruction BCE, pixel-summed MSE per image, VAE KL, sample grids, interpolation, seeds, GPU, and elapsed time.

A direct AE-versus-VAE objective comparison needs care: the VAE objective contains a KL term and stochastic encoder. Compare the evidence relevant to the claim, not only one total loss.

### 14.2 DDPM on CIFAR-10

The modern lab uses the public Apache-2.0 `google/ddpm-cifar10-32` checkpoint pinned to revision `267b167dc01f0e4e61923ea244e8b988f84deb80`, which contains a roughly 35.7-million-parameter U-Net. It supports DDPM or DDIM schedulers and records inference steps.

The default 50-step DDIM run tests loading, sampling, timing, and trajectory capture. The optional FID path downloads a declared real CIFAR-10 split and requires `torchmetrics` plus `torch-fidelity`. A 1,000-sample course result has high estimator variance and does not reproduce the DDPM paper's 50,000-sample result.

## 15. Research directions

- latent dimension, rate-distortion, and downstream utility;
- posterior collapse and decoder capacity;
- discrete versus continuous latent representations;
- likelihood versus perceptual sample quality;
- GAN stability under limited or imbalanced data;
- invertible architectures for scientific inverse problems;
- diffusion schedule and solver efficiency;
- one-step and few-step generation;
- flow matching path design;
- conditional generation under distribution shift;
- memorization, privacy, provenance, and opt-out mechanisms;
- synthetic-data utility and amplification of rare errors;
- generative evaluation beyond one feature extractor;
- energy models for anomaly detection and compositional inference.

A useful research question is narrow and falsifiable:

> On a fixed Fashion-MNIST split and seed set, does increasing $\beta$ from 0.5 to 2.0 reduce active latent units and improve prior-sample FID without increasing locked-test reconstruction BCE by more than 5%?

The thresholds, metric implementation, and seed aggregation must be frozen before looking at the locked test result.

## 16. Claims supported by Week 12

The material supports these claims:

- reconstruction can learn a compressed representation without a class label;
- an autoregressive chain rule yields exact conditional likelihoods;
- a VAE optimizes a lower bound using a differentiable sampling estimator;
- GANs train an implicit generator through a two-player objective;
- flows trade architectural freedom for exact inversion and likelihood;
- diffusion and score models learn to reverse controlled corruption;
- EBMs express compatibility through unnormalized energy and require difficult negative sampling;
- generative metrics answer different questions and depend on protocol.

It does not establish that one family is universally best, that low FID proves novelty or safety, that latent coordinates are causal factors, or that a small course run reproduces a research benchmark.