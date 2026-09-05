# Week 2: Generalization and trustworthy evaluation

**Guiding question:** What evidence shows that a model will work on relevant unseen data?

Week 2 turns a fitted model into an evaluation claim. Students define what “unseen” means for an application, assign distinct roles to training, validation, and test data, choose a split that respects dependence between observations, prevent preprocessing leakage, and interpret a model relative to a baseline and an appropriate metric.

The module continues in Week 3 with data quality, missingness, imbalance, feature quality, and model comparison.

## Learning outcomes

By the end of the week, students should be able to:

- explain the difference between fitting the observed sample and generalizing to a target population;
- state the separate roles of training, validation, and test partitions;
- choose between random, stratified, grouped, and chronological splitting;
- identify target leakage, preprocessing leakage, group leakage, and temporal leakage;
- fit preprocessing parameters using training data only;
- calculate accuracy, precision, recall, specificity, and balanced accuracy from a confusion matrix;
- explain how a decision threshold changes errors without retraining the model; and
- write a reproducible evaluation protocol before examining final test performance.

## Week 2 materials

| Resource | Purpose |
| --- | --- |
| [Student notes](notes.md) | Generalization, split design, leakage, metrics, uncertainty, and reproducibility |
| [Lecture source](slides.md) | Accessible text version of the lecture deck |
| [PowerPoint lecture deck](slides/week02-generalization-and-evaluation.pptx) | Classroom presentation with speaker notes |
| [Student worksheet](worksheet.md) | Split-design, leakage, metric, lab, and dataset-feasibility activities |
| [Leakage-safe NumPy lab](../../labs/week02_evaluation_numpy.py) | Repeated-measurement example comparing row-wise and grouped evaluation |
| [Instructor guide](../../instructor-notes/week02.md) | Three-hour lesson plan, prompts, expected evidence, and adaptations |
| [Problem proposal](../../assignments/02-problem-proposal.md) | Week 3 milestone that uses the Week 2 evaluation protocol |

## Preparation

Before class, students should:

1. review the Week 1 problem-formulation canvas;
2. run `python labs/week02_evaluation_numpy.py`;
3. bring the name or URL of one dataset they might use for the research project; and
4. read the linked scikit-learn guidance on common evaluation pitfalls.

The lab uses only NumPy and generated data. No external download is required.

## Learning sequence

1. Evaluate the claim “the model achieved 95% accuracy.”
2. Define the future cases the model is intended to handle.
3. Assign training, validation, and test data distinct roles.
4. Identify the independent sampling unit and possible dependencies.
5. Select a split that simulates the intended use.
6. Locate leakage in targets, features, preprocessing, groups, and time.
7. Calculate several metrics from the same confusion matrix.
8. Run the repeated-measurement lab and explain why two valid-looking splits disagree.
9. Draft a dataset-specific evaluation protocol before model tuning.

## Evidence of learning

Students leave class with:

- a justified split strategy for one repeated-measurement problem;
- a corrected leakage-prone workflow;
- a completed metric calculation and interpretation;
- an executed lab record comparing row-wise and grouped evaluation; and
- a first feasibility screen for a possible research dataset.

## Essential reading

- [scikit-learn: common pitfalls and recommended practices](https://scikit-learn.org/stable/common_pitfalls.html), sections on inconsistent preprocessing, data leakage, and randomness
- [PyTorch reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness.html), overview and sources of nondeterminism
- [Datasheets for Datasets](https://doi.org/10.1145/3458723), abstract and motivation
- [Course research-project brief](../../research-project/brief.md), especially the data and evaluation requirements

## A boundary on interpretation

The synthetic lab intentionally makes repeated observations from the same source highly similar. Its purpose is to expose the direction of group leakage. The numerical gap is not an estimate of leakage in another dataset, and the example does not imply that grouped splitting is always correct. The split must follow the deployment question.
