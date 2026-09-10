# Week 12 worksheet — Representations, probability, and generation

Name: ____________________

Research topic or application: ____________________

## 1. Define the claim

Describe one unlabeled-data or generation problem.

Observation $x$: ________________________________________________

Condition $c$, if any: __________________________________________

Desired output: _________________________________________________

Choose the main claim:

- [ ] useful representation
- [ ] calibrated or comparative likelihood
- [ ] realistic unconditional samples
- [ ] condition-aligned samples
- [ ] anomaly or compatibility score

One harmful failure that a sample grid could hide:

__________________________________________________________________

## 2. Trace an autoencoder

For input shape $(N,1,28,28)$, complete the trace.

| Operation | Channels | Height | Width |
| --- | ---: | ---: | ---: |
| input | 1 | 28 | 28 |
| stride-2 convolution | 16 | | |
| stride-2 convolution | 32 | | |
| flatten | | 1 | |
| latent projection | | 1 | $d_z$ |

Write the encoder and decoder equations:

$$
z=\underline{\hspace{7cm}}
$$

$$
\hat{x}=\underline{\hspace{7cm}}
$$

State one reason that low reconstruction error may not imply a useful representation.

__________________________________________________________________

## 3. Factorize an autoregressive model

For $x=(x_1,x_2,x_3,x_4)$, write the joint probability as conditional factors.

$$
p(x)=\underline{\hspace{13cm}}
$$

If the four factors for one observation are $0.8$, $0.7$, $0.5$, and $0.9$, calculate $p(x)$.

$$
p(x)=\underline{\hspace{5cm}}
$$

Why can training be parallelized under a causal mask while sampling remains sequential?

__________________________________________________________________

## 4. Derive the VAE objective

Start from

$$
\log p_\theta(x)
=
\log\int q_\phi(z\mid x)
\frac{p_\theta(x,z)}{q_\phi(z\mid x)}\,dz.
$$

Name the inequality used to move the logarithm inside an expectation:

__________________________________________________________________

Complete the ELBO:

$$
\mathcal{L}(x)
=
\mathbb{E}_{q_\phi(z\mid x)}
[\underline{\hspace{5cm}}]
-
D_{\mathrm{KL}}(
\underline{\hspace{5cm}}
).
$$

Circle the term that encourages reconstruction. Box the term that encourages compatibility with the prior.

## 5. Calculate a diagonal-Gaussian KL

For one latent dimension with $\mu=0.5$ and $\log\sigma^2=0$, use

$$
D_{\mathrm{KL}}(q\Vert p)
=
\frac{1}{2}(\mu^2+\sigma^2-1-\log\sigma^2).
$$

$$
\sigma^2=\underline{\hspace{3cm}}
$$

$$
D_{\mathrm{KL}}=\underline{\hspace{3cm}}
$$

If $\epsilon=-1$ and $\sigma=1$, calculate

$$
z=\mu+\sigma\epsilon=\underline{\hspace{3cm}}.
$$

Explain why this reparameterization permits gradients with respect to $\mu$ and $\sigma$.

__________________________________________________________________

## 6. Predict the beta-VAE ablation

Complete a directional prediction before running the lab.

> If $\beta$ increases from ______ to ______ while data, architecture, optimizer, epochs, and seed set remain fixed, then __________________ will __________________ because __________________.

What result would reject this prediction?

__________________________________________________________________

## 7. Trace the GAN game

Write the role of each component.

| Component | Input | Output | Objective |
| --- | --- | --- | --- |
| generator $G$ | | | |
| discriminator $D$ | | | |

Complete the non-saturating generator loss:

$$
L_G=-\mathbb{E}_{z\sim p(z)}\log\underline{\hspace{4cm}}.
$$

A generator produces sharp examples from only a few modes. Which evidence exposes the problem?

__________________________________________________________________

## 8. Apply change of variables

Let $z=3x-2$ and let $z$ have known density $p_Z(z)$.

$$
\frac{dz}{dx}=\underline{\hspace{3cm}}
$$

$$
\log p_X(x)
=
\log p_Z(\underline{\hspace{3cm}})
+
\log\underline{\hspace{3cm}}.
$$

Write the inverse sampling transformation.

$$
x=\underline{\hspace{5cm}}
$$

Name one architectural restriction created by exact change of variables.

__________________________________________________________________

## 9. Calculate DDPM corruption

Given

$$
x_0=0.8,\qquad \bar{\alpha}_t=0.64,\qquad \epsilon=-0.5,
$$

calculate

$$
x_t
=
\sqrt{\bar{\alpha}_t}x_0
+
\sqrt{1-\bar{\alpha}_t}\epsilon.
$$

$$
\sqrt{\bar{\alpha}_t}=\underline{\hspace{2cm}}
$$

$$
\sqrt{1-\bar{\alpha}_t}=\underline{\hspace{2cm}}
$$

$$
x_t=\underline{\hspace{3cm}}
$$

If the network predicts $\hat{\epsilon}=-0.3$, calculate the one-element squared error.

$$
(\epsilon-\hat{\epsilon})^2=\underline{\hspace{3cm}}
$$

## 10. Connect noise and score

Complete the DDPM conditional-score relationship:

$$
\nabla_{x_t}\log q(x_t\mid x_0)
=
-\frac{\epsilon}{\underline{\hspace{5cm}}}.
$$

In words, what direction does a score estimate provide?

__________________________________________________________________

## 11. Interpret an energy model

$$
p_\theta(x)=\frac{\exp[-E_\theta(x)]}{Z_\theta}.
$$

Should a likely observation have higher or lower energy? __________________

Why is $Z_\theta$ difficult? __________________________________________

In the likelihood gradient, what are the two phases?

- data or positive phase: __________________________________________
- model or negative phase: ________________________________________

## 12. Audit Lab 12A

Run:

```bash
python labs/week12_autoencoder_pytorch.py --audit-only
```

Record:

| Item | Value |
| --- | --- |
| device | |
| AE reconstruction / latent shapes | |
| VAE reconstruction / mean / log-variance shapes | |
| AE parameters | |
| VAE parameters | |
| zero-noise reparameterization check | |
| balanced and disjoint index checks | |

Before the full run, predict which model will have lower reconstruction BCE and which can sample directly from $\mathcal{N}(0,I)$.

__________________________________________________________________

## 13. Audit Lab 12B

Run:

```bash
python labs/week12_ddpm_cifar10.py --audit-only
```

Record:

| Item | Value |
| --- | --- |
| schedule shapes | |
| does $\bar{\alpha}_t$ decrease? | |
| corrupted batch shape | |
| downloads started? | |
| diffusion dependencies available? | |

For a full run, record scheduler, steps, seed, GPU, parameter count, sample count, total seconds, and seconds per image.

## 14. Read the benchmark table critically

Choose two CIFAR-10 rows from the Week 12 notes.

| Field | Model A | Model B |
| --- | --- | --- |
| family | | |
| unconditional or conditional | | |
| FID | | |
| network evaluations | | |
| sample count | | |
| real reference statistics | | |
| feature implementation | | |
| training data and augmentation | | |
| comparison valid? | | |

Which missing field most threatens your comparison?

__________________________________________________________________

## 15. Design the research ablation

Mechanism changed: ______________________________________________

Fixed controls: __________________________________________________

Seed set: ________________________________________________________

Directional prediction: __________________________________________

Primary metric and implementation: _______________________________

Compute metric: __________________________________________________

Robustness or coverage check: ____________________________________

Rejection condition: _____________________________________________

## Exit ticket

In four sentences:

1. distinguish reconstruction from probability modeling;
2. explain why a VAE includes KL divergence;
3. compare the sampling paths of an autoregressive model, GAN, flow, diffusion model, and EBM;
4. name one FID protocol detail that must be fixed.
