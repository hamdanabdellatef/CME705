# Autoencoders and generative models

**Guiding question:** How can a model compress observations, assign probability, or generate new examples, and what evidence shows that the learned distribution is useful?

## Learning outcomes

Students will:

- distinguish reconstruction, likelihood, adversarial, score, and energy objectives;
- design and evaluate deterministic autoencoders;
- derive the VAE evidence lower bound and reparameterization estimator;
- factorize an autoregressive probability model;
- explain the GAN minimax game and common instability;
- apply the change-of-variables formula for normalizing flows;
- derive DDPM forward corruption and noise-prediction training;
- connect diffusion models with score estimation;
- interpret energy-based models and contrastive gradients;
- compare generative families using task-matched metrics and protocols;
- run a modern pretrained diffusion model on CIFAR-10;
- define a falsifiable ablation or robustness experiment.

## Week 12 package

- [Lesson plan](week12.md)
- [Detailed mathematical notes](week12-notes.md)
- [Accessible slide source](week12-slides.md)
- [Student worksheet](week12-worksheet.md)
- [PowerPoint deck](slides/week12-autoencoders-and-generative-models.pptx)
- [Instructor guide](../../instructor-notes/week12.md)
- [Autoencoder and VAE lab](../../labs/week12_autoencoder_pytorch.py)
- [DDPM CIFAR-10 lab](../../labs/week12_ddpm_cifar10.py)
- [Further reading](../../readings/week12-generative-models.md)
- [Ablation and robustness milestone](../../research-project/generative-model-ablation.md)

## Topics

- deterministic autoencoders and bottlenecks;
- latent interpolation and representation evidence;
- autoregressive factorization and teacher forcing;
- VAEs, ELBOs, KL divergence, and reparameterization;
- GAN generators, discriminators, minimax objectives, and mode collapse;
- normalizing flows and exact change of variables;
- DDPM forward and reverse processes;
- score-based models and stochastic differential equations;
- EBMs, partition functions, and MCMC;
- latent diffusion, Diffusion Transformers, consistency models, flow matching, and modern visual autoregression;
- FID, KID, likelihood, precision and recall, memorization, and human evaluation.

## Evidence of learning

Students submit:

1. a shape- and split-audited autoencoder/VAE experiment;
2. validation-selected reconstruction and ELBO evidence on Fashion-MNIST;
3. generated samples and an interpolation with claim limits;
4. a scheduler-declared DDPM sample and denoising trajectory on CIFAR-10;
5. a metric card that identifies the dataset, conditioning, sample count, feature extractor, and implementation;
6. one ablation or robustness result with a predeclared rejection condition.

## Place in the course

Week 12 combines the representation learning, optimization, evaluation, convolution, and attention ideas developed in Weeks 1–11. Its research milestone asks students to isolate one mechanism and test whether their project conclusion survives a controlled change.
