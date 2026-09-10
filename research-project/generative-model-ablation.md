# Generative-model ablation and robustness note

**Due:** Week 12

**Purpose:** isolate one generative-model mechanism and test whether a research conclusion survives a controlled change.

## Deliverable

Submit a concise report, the exact command or configuration, one results table, one evidence figure, and links to the code and environment record. The report should be reproducible from a clean checkout without committing data, model weights, or generated batches.

## 1. Claim and model object

State the research claim in one sentence. Identify which object the model learns:

- reconstruction map;
- normalized likelihood;
- variational lower bound;
- implicit sampler;
- invertible density;
- score field;
- energy function.

Explain why that object can answer the claim.

## 2. Controlled factor

Choose exactly one main factor, such as:

- autoencoder latent dimension;
- VAE $\beta$;
- posterior family;
- decoder capacity;
- GAN augmentation or update ratio;
- flow depth or coupling transform;
- diffusion scheduler or network evaluations;
- score-noise weighting;
- sampling temperature;
- corruption level;
- training-data quantity.

List every value to compare. State why the range is meaningful.

## 3. Fixed controls

Freeze and record:

- Git commit;
- data version and split indices;
- preprocessing and augmentation;
- architecture except the selected factor;
- initialization and seed set;
- optimizer, learning rate, batch size, and epochs;
- checkpoint-selection rule;
- metric implementation and reference statistics;
- compute device and precision;
- evaluation sample count;
- sampler settings unless they are the selected factor.

If a control cannot remain fixed, explain the consequence before running the experiment.

## 4. Directional prediction

Use this structure:

> If __________________ changes from __________________ to __________________ while the declared controls remain fixed, then __________________ will __________________ because __________________.

Define a result that would reject the prediction. Avoid “performance will change.” Name a metric, direction, and practically meaningful threshold.

## 5. Evaluation protocol

Choose evidence that matches the claim.

| Claim | Minimum evidence |
| --- | --- |
| reconstruction | locked BCE/MSE or perceptual metric plus examples |
| latent utility | frozen-representation downstream task plus robustness |
| likelihood | held-out NLL or bits/dim with preprocessing |
| sample fidelity | FID or KID with protocol card |
| distribution coverage | recall or coverage evidence plus rare cases |
| conditional control | condition accuracy or alignment plus quality |
| efficiency | wall time, network evaluations, memory, and hardware |
| anomaly score | AUROC and AUPRC with threshold selection isolated |

Do not report only selected attractive outputs.

## 6. Metric card

Record:

| Field | Value |
| --- | --- |
| dataset and split | |
| unconditional or conditional | |
| image or sequence representation | |
| generated sample count | |
| real reference set | |
| feature extractor and revision | |
| metric library and version | |
| resize and pixel range | |
| checkpoint rule | |
| sampler and steps | |
| guidance or temperature | |
| random seeds | |

## 7. Results table

Use one row per factor value and seed.

| Factor value | Seed | Validation selection metric | Locked primary metric | Coverage or robustness | Seconds / sample | Peak memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| | | | | | | |

Report mean, standard deviation, and number of seeds where appropriate. Do not select the best seed.

## 8. Evidence figure

Create one figure with a predetermined selection rule:

- fixed latent inputs across conditions;
- fixed source images for reconstruction;
- random samples selected by recorded indices;
- denoising trajectories at fixed timesteps;
- interpolation endpoints chosen before results;
- failure cases selected by the primary metric.

Include nearest training examples when memorization is relevant.

## 9. Robustness check

Choose one stress test that was not used for model selection:

- image corruption;
- rare class or subgroup;
- distribution shift;
- longer sampling trajectory;
- reduced network evaluations;
- perturbed condition;
- alternative feature extractor;
- sample-count sensitivity;
- initialization sensitivity.

Explain why the stress test targets the claim.

## 10. Interpretation

State:

1. what the evidence supports;
2. whether the directional prediction survived;
3. the size and uncertainty of the effect;
4. the compute or quality tradeoff;
5. the most important failure;
6. one alternative explanation;
7. the next experiment that would distinguish explanations.

## 11. Reproducibility record

Include:

- commit hash;
- clean environment versions;
- exact commands;
- seeds;
- data acquisition path;
- checkpoint identity;
- output artifact names;
- run time and hardware;
- any deviation from the plan.

Keep datasets, downloaded weights, checkpoints, and generated batches in ignored directories.

## Submission check

- [ ] One model object and one claim are named.
- [ ] One main factor changes.
- [ ] Controls and deviations are explicit.
- [ ] The prediction is directional and falsifiable.
- [ ] Validation controls selection.
- [ ] Locked evidence is used once after selection.
- [ ] Metrics match the claim and include protocol details.
- [ ] Sample selection is predetermined.
- [ ] Coverage, failure, or robustness evidence is included.
- [ ] Compute and uncertainty are reported.
- [ ] Conclusions stay within the evidence.

## Assessment guide

| Criterion | Weight |
| --- | ---: |
| Claim and model-object alignment | 15 |
| Controlled design and reproducibility | 20 |
| Mathematical and implementation correctness | 20 |
| Metric protocol and benchmark literacy | 15 |
| Results, uncertainty, and robustness | 20 |
| Interpretation and next experiment | 10 |
