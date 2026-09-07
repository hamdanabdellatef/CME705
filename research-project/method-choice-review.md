# Week 9 milestone: Method-choice review

## Purpose

Use this review to decide whether the proposed model matches the project’s data, target, evaluation, and computational constraints. The review should justify one next experiment. It may conclude that a convolutional model is unnecessary.

## Submission

Submit a concise review with the sections below. Link to the exact repository revision and the current controlled-experiment results.

## 1. Research question and prediction unit

State the research question in one sentence.

State:

- observation unit;
- prediction unit;
- target;
- decision supported by the prediction; and
- primary metric with direction.

Explain whether the current validation split matches the intended use.

## 2. Input geometry

Record the model input shape and the meaning of every axis.

| Item | Project value |
| --- | --- |
| raw data shape | |
| model input shape | |
| channel meaning | |
| spatial or temporal dimensions | |
| resolution or sampling rate | |
| missing or padded regions | |
| preprocessing fitted on training data | |

If the data are images, state whether orientation, scale, acquisition device, and spatial position have stable meanings.

## 3. Inductive-bias argument

Describe the local pattern the proposed model should detect.

Explain:

- why nearby values should interact;
- why the same local detector should or should not be reused across positions;
- which translations, rotations, scale changes, or other transformations should preserve the target;
- which transformations would change the target; and
- where position-specific information matters.

A statement such as “CNNs work well for images” is insufficient.

## 4. Candidate and simpler alternative

Describe one candidate architecture and one simpler comparison.

| Decision | Candidate | Simpler alternative |
| --- | --- | --- |
| input representation | | |
| number of learned layers | | |
| channel widths | | |
| downsampling | | |
| classifier head | | |
| approximate parameter count | | |
| expected training cost | | |
| main inductive bias | | |

The simpler alternative may be a linear model, small multilayer perceptron, fixed-feature model, or smaller CNN.

## 5. Shape and capacity audit

Trace the shape after every planned layer.

| Stage | Operation | Input shape | Output shape | Parameters |
| ---: | --- | --- | --- | ---: |
| 0 | input | | | $0$ |
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |
| head | | | | |
| total | | | | |

State the receptive field at the final spatial layer.

Estimate the largest activation tensor for one batch. Record the intended batch size and whether the available hardware can support it.

## 6. Data sufficiency and augmentation

Record:

- number of independent training observations;
- class or outcome distribution;
- repeated subjects, sites, devices, or time periods;
- resolution and channel consistency;
- known acquisition artifacts; and
- augmentation candidates.

For every proposed augmentation, explain why the label should remain valid. For example, horizontal flipping may be valid for one object-recognition task and invalid for left/right anatomy.

## 7. Evaluation plan

Preserve the roles established earlier in the course.

| Item | Decision |
| --- | --- |
| training identities | |
| validation identities | |
| locked test identities | |
| grouping or temporal rule | |
| primary validation metric | |
| class or subgroup metrics | |
| early-stopping rule | |
| random seeds | |
| uncertainty summary | |
| selection rule | |

State whether any test result has influenced the architecture. If yes, label the current test evidence provisional and propose an independent final-evaluation plan.

## 8. Controlled next experiment

Write one directional hypothesis.

State exactly one intended change. List all decisions held fixed:

- data revision;
- split identities;
- preprocessing;
- augmentation;
- initialization policy;
- optimizer and learning rate;
- batch size;
- maximum epochs;
- early stopping;
- metric; and
- evaluation code.

If the candidate differs in several coupled architectural details, describe the comparison as a package comparison rather than attributing the result to one component.

## 9. Feasibility check

Run the smallest useful smoke experiment and record:

- command;
- environment;
- repository commit;
- input batch shape;
- output-logit shape;
- loss before and after one update;
- wall-clock time;
- peak memory if available; and
- any warning or failure.

A failed feasibility check is acceptable when it identifies the limiting resource or invalid assumption.

## 10. Decision

Choose one:

- proceed with the candidate;
- proceed with the simpler alternative;
- collect or repair data first;
- revise the evaluation; or
- run one additional feasibility comparison.

Support the decision with current evidence. State one limitation and the next result that could change the decision.

## Review checklist

- [ ] The model follows from the research question and data geometry.
- [ ] Every tensor axis is defined.
- [ ] Weight sharing is justified or rejected explicitly.
- [ ] A simpler alternative is specified.
- [ ] Shapes and parameter counts are checked.
- [ ] Augmentations preserve label semantics.
- [ ] Split roles and grouping remain valid.
- [ ] The test set remains locked or is labeled provisional.
- [ ] The next comparison has one interpretable purpose.
- [ ] The claim stays within current evidence.
