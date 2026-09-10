# Week 12 further reading — Generative models, benchmarks, and research directions

**Research snapshot:** 10 September 2026.

Use this guide to choose primary sources and evaluation protocols. A paper result is evidence only for its named dataset, conditioning, preprocessing, metric implementation, sample count, checkpoint, and compute budget.

## Foundations

### Autoencoders and latent variables

- [Reducing the Dimensionality of Data with Neural Networks](https://www.science.org/doi/10.1126/science.1127647) introduces deep autoencoder pretraining and nonlinear dimensionality reduction.
- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) derives the stochastic variational lower bound and reparameterization estimator.
- [beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2fzU9gl) studies stronger latent-capacity constraints and disentangling claims.
- [Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937) introduces VQ-VAE with discrete latent codes.

Read for: the training objective, latent prior, posterior approximation, and evidence required for a representation claim.

### Autoregressive generation

- [Pixel Recurrent Neural Networks](https://arxiv.org/abs/1601.06759) factorizes image pixels and models them sequentially.
- [Conditional Image Generation with PixelCNN Decoders](https://arxiv.org/abs/1606.05328) develops gated masked convolutions and conditioning.
- [Autoregressive Model Beats Diffusion: Llama for Scalable Image Generation](https://arxiv.org/abs/2406.06525) studies next-token Transformers over visual tokens on ImageNet.
- [Autoregressive Image Generation with Randomized Parallel Decoding](https://arxiv.org/abs/2503.10568) studies random-order parallel decoding and reports ImageNet-256 quality and throughput evidence.

Read for: the ordering, tokenization ceiling, exact likelihood, training parallelism, sampling latency, and decoding policy.

### Adversarial generation

- [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) introduces the minimax generator-discriminator framework.
- [Wasserstein GAN](https://arxiv.org/abs/1701.07875) changes the discrepancy and critic constraints.
- [Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028) studies a gradient penalty.
- [Training Generative Adversarial Networks with Limited Data](https://arxiv.org/abs/2006.06676) introduces adaptive discriminator augmentation and reports CIFAR-10 FID 2.42.

Read for: the two-player objective, discriminator capacity, stability intervention, diversity evidence, and data regime.

### Normalizing flows

- [Variational Inference with Normalizing Flows](https://arxiv.org/abs/1505.05770) uses invertible transformations to enrich variational posteriors.
- [Density Estimation using Real NVP](https://arxiv.org/abs/1605.08803) develops tractable coupling layers.
- [Glow](https://arxiv.org/abs/1807.03039) adds invertible $1\times1$ convolutions.
- [Flow++](https://arxiv.org/abs/1902.00275) studies dequantization and expressive coupling transformations.
- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) trains continuous normalizing flows by conditional vector-field regression.

Read for: invertibility, Jacobian determinants, exact likelihood, dequantization, solver accuracy, and path design.

### Energy-based models

- [A Tutorial on Energy-Based Learning](http://yann.lecun.com/exdb/publis/pdf/lecun-06.pdf) gives a broad formulation of energy-based learning.
- [Implicit Generation and Generalization in Energy-Based Models](https://arxiv.org/abs/1903.08689) scales MCMC-trained neural EBMs to CIFAR-10 and other domains.
- [Your Classifier is Secretly an Energy Based Model and You Should Treat it Like One](https://arxiv.org/abs/1912.03263) connects discriminative logits and joint energy modeling.

Read for: the energy function, partition function, positive and negative phases, MCMC initialization, mixing, and out-of-distribution claims.

### Diffusion and score models

- [Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/abs/1503.03585) introduces a diffusion probabilistic framework.
- [Generative Modeling by Estimating Gradients of the Data Distribution](https://arxiv.org/abs/1907.05600) develops noise-conditional score networks.
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) introduces the widely used DDPM parameterization and reports unconditional CIFAR-10 FID 3.17.
- [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456) unifies forward and reverse SDEs, probability-flow ODEs, and predictor-corrector sampling.
- [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) studies non-Markovian sampling paths and faster generation.

Read for: the corruption process, predicted target, weighting, score parameterization, sampler, and network-evaluation count.

## Recent system directions

### Latent diffusion

[High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) runs diffusion in a pretrained autoencoder's latent space and uses cross-attention for conditioning. Report the autoencoder reconstruction ceiling, latent scaling, denoiser, conditioning data, guidance, and sampler together.

Research prompts:

- Which semantic or high-frequency details are lost by the compressor?
- Does latent compression improve quality per GPU-hour at a fixed resolution?
- How does the first-stage training distribution limit downstream generation?

### Diffusion Transformers

[Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748) replaces a latent U-Net with a Transformer and relates model forward-pass compute to ImageNet FID.

Research prompts:

- Which gain comes from backbone capacity versus training compute?
- How do patch size and latent resolution change network evaluations and memory?
- Does the scaling relation hold outside class-conditional ImageNet?

### EDM

[Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364) separates preconditioning, training noise, network parameterization, schedule, and solver choices. It reports unconditional CIFAR-10 FID 1.97 with 35 network evaluations.

Research prompts:

- Which component transfers to a new domain?
- Does a better schedule reduce latency at fixed quality?
- Are quality gains stable across random seeds and feature implementations?

### Consistency models

[Consistency Models](https://arxiv.org/abs/2303.01469) maps points on a probability-flow trajectory to a shared clean endpoint and supports one- or few-step generation. The paper reports one-step CIFAR-10 FID 3.55.

Research prompts:

- What quality is lost between one and several evaluations?
- How do distillation and independent training differ?
- Which editing capabilities remain reliable under distribution shift?

### Flow matching

[Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) learns continuous velocity fields without simulating the full path during training. It supports diffusion paths and optimal-transport-inspired paths.

Research prompts:

- How does path curvature affect solver steps?
- Which solver tolerance preserves likelihood and sample quality?
- Can domain geometry motivate a better conditional path?

### Modern visual autoregression

[LlamaGen](https://arxiv.org/abs/2406.06525) separates a visual tokenizer from a next-token Transformer and reports ImageNet-256 results across model scales. [Randomized Parallel Decoding](https://arxiv.org/abs/2503.10568) explores non-raster order and parallel queries.

Research prompts:

- What fraction of total error comes from tokenization?
- How do cache size, token count, and decoding order affect latency?
- Can a faster decoding policy preserve distribution coverage?

## Benchmark families

| Claim | Example benchmark | Typical evidence | Main threat |
| --- | --- | --- | --- |
| small-image unconditional generation | CIFAR-10 | FID, KID, precision and recall, NLL when available | implementation and finite-sample bias |
| class-conditional generation | ImageNet | FID, Inception Score, class consistency | guidance and conditioning protocol |
| face synthesis | FFHQ | FID, precision and recall, nearest neighbors | demographic coverage and privacy |
| scene generation | LSUN | FID and sample inspection | category definition and preprocessing |
| text-to-image alignment | COCO or prompt suites | FID plus alignment and human evaluation | prompt leakage and evaluator bias |
| reconstruction | MNIST, Fashion-MNIST, domain data | BCE, MSE, PSNR, SSIM, perceptual distance | pixel loss may miss semantic utility |
| anomaly detection | MVTec AD or domain benchmark | AUROC, AUPRC, localization | synthetic anomalies and test tuning |
| density estimation | CIFAR-10 or ImageNet variants | bits per dimension | dequantization and semantic mismatch |

A benchmark name is not a protocol. Record version, split, resolution, preprocessing, labels, external data, sample count, random seeds, checkpoint rule, and code revision.

## Evaluation sources

- [GANs Trained by a Two Time-Scale Update Rule](https://arxiv.org/abs/1706.08500) introduces FID.
- [Demystifying MMD GANs](https://arxiv.org/abs/1801.01401) discusses kernel methods that motivate KID-style evaluation.
- [Assessing Generative Models via Precision and Recall](https://arxiv.org/abs/1806.00035) separates sample quality and coverage.
- [Improved Precision and Recall Metric for Assessing Generative Models](https://arxiv.org/abs/1904.06991) develops a nonparametric feature-space metric.
- [On the Quantitative Analysis of Decoder-Based Generative Models](https://arxiv.org/abs/1611.04273) cautions against relying on one likelihood or sample estimator.

## How to read a reported result

Create a metric card with:

| Field | Required question |
| --- | --- |
| task | unconditional, class-conditional, text-conditional, reconstruction, or density? |
| data | exact version, split, resolution, filters, duplicates, and external data? |
| model | parameter count, representation, backbone, conditioning, and checkpoint? |
| sampler | schedule, solver, guidance, stochasticity, and network evaluations? |
| metric | implementation, feature extractor, sample count, and real statistics? |
| uncertainty | seeds, confidence interval, or repeat variation? |
| compute | hardware, precision, training cost, latency, and memory? |
| safety | memorization, provenance, subgroup coverage, and misuse analysis? |

If two papers omit different fields, do not manufacture a precise ranking.

## Research directions

### Representation and rate-distortion

Study how latent dimension, quantization, decoder capacity, and perceptual loss trade compression against downstream utility. Evaluate rare classes and high-frequency details separately.

### Probabilistic inference

Measure posterior collapse, importance-weighted likelihood, calibration, and amortization gaps. Compare a simple posterior with a flow-based posterior under a fixed decoder and compute budget.

### Stable adversarial learning

Test discriminator regularization, augmentation, update ratios, or limited-data transfer. Pair FID with coverage and memorization evidence.

### Fast iterative generation

Compare DDPM, DDIM, EDM-style schedules, consistency models, or flow-matching solvers at fixed network evaluations and wall-clock budgets. Include the same checkpoint where possible.

### Conditional reliability

Test alignment, compositional prompts, rare conditions, and robustness to wording or sensor shift. Report failure rates rather than only selected successful examples.

### Scientific and engineering inverse problems

Use priors or scores for reconstruction, imputation, super-resolution, or simulation. Distinguish data fidelity from prior plausibility and evaluate against known ground truth or conservation laws.

### Synthetic-data utility

Train downstream models on real, synthetic, and mixed data under a fixed budget. Measure subgroup performance, calibration, distribution shift, and whether synthetic data amplifies source errors.

### Privacy, memorization, and provenance

Measure nearest-neighbor similarity, membership inference, duplicate generation, and traceable model/data provenance. A low FID cannot answer these questions.

### Evaluation validity

Compare metric rankings across feature extractors, sample counts, and domains. Test whether metric changes predict human judgments or downstream utility.

## Suggested reading paths

### Latent representation

1. Auto-Encoding Variational Bayes
2. beta-VAE
3. VQ-VAE
4. latent diffusion
5. one domain-specific representation benchmark

### Exact density

1. PixelRNN or PixelCNN
2. Real NVP
3. Glow or Flow++
4. flow matching
5. one controlled likelihood protocol

### Adversarial generation

1. original GAN
2. Wasserstein GAN
3. gradient penalty
4. StyleGAN2-ADA
5. FID plus precision and recall

### Diffusion and score

1. DDPM
2. score-SDE
3. DDIM
4. EDM
5. consistency or flow matching

## Deliverable

The Week 12 research note must cite:

- one foundational family paper;
- one recent method paper;
- one dataset or benchmark source;
- one metric source;
- one competing baseline.

For every numerical claim, name the task, metric, direction, sample count when available, conditioning, and source table or section.
