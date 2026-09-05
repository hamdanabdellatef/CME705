# Data and evaluation — Weeks 2–3

This module establishes the evidence standards used throughout CME705.

| Week | Guiding question | Main product |
| ---: | --- | --- |
| 2 | What evidence shows that a model will work on relevant unseen data? | A leakage-aware evaluation protocol |
| 3 | What must we learn about the data before comparing models? | A data audit and feasible problem proposal |

## Shared learning outcomes

After completing the module, students should be able to:

- define the intended population and unit of observation;
- create train, validation, and test partitions that respect groups and time;
- fit preprocessing only on training data;
- identify target, preprocessing, duplicate, group, and temporal leakage;
- audit provenance, missingness, class balance, duplicates, labels, outliers, and feature availability;
- select metrics and baselines that match the application;
- compare candidates under one controlled evaluation protocol; and
- document a reproducible dataset and experiment plan.

## Week 2: Generalization and trustworthy evaluation

Week 2 turns a fitted model into an evaluation claim. Students define what “unseen” means, give training, validation, and test data distinct roles, choose a split that respects dependence, prevent leakage, and interpret a score through a baseline and error costs.

| Resource | Purpose |
| --- | --- |
| [Week 2 student notes](notes.md) | Generalization, split design, leakage, metrics, uncertainty, and reproducibility |
| [Week 2 lecture source](slides.md) | Accessible text version of the lecture deck |
| [Week 2 PowerPoint](slides/week02-generalization-and-evaluation.pptx) | Classroom presentation with speaker notes |
| [Week 2 worksheet](worksheet.md) | Split-design, leakage, metric, lab, and dataset-feasibility activities |
| [Week 2 NumPy lab](../../labs/week02_evaluation_numpy.py) | Row-wise and grouped evaluation of repeated measurements |
| [Week 2 instructor guide](../../instructor-notes/week02.md) | Timed plan, prompts, expected evidence, and adaptations |

## Week 3: Data quality and model comparison

Week 3 treats a dataset as a measurement process rather than a clean matrix. Students create an audit, examine missingness and rare classes, test feature availability, compare models fairly, and convert the result into a feasible research proposal.

| Resource | Purpose |
| --- | --- |
| [Week 3 student notes](week03-notes.md) | Data auditing, missingness, imbalance, feature quality, baselines, and comparison |
| [Week 3 lecture source](week03-slides.md) | Accessible text version of the lecture deck |
| [Week 3 PowerPoint](slides/week03-data-quality-and-model-comparison.pptx) | Classroom presentation with speaker notes |
| [Week 3 worksheet](week03-worksheet.md) | Audit, missingness, imbalance, comparison, and proposal activities |
| [Week 3 NumPy lab](../../labs/week03_data_quality_numpy.py) | Rare-event screening with training-only imputation |
| [Week 3 instructor guide](../../instructor-notes/week03.md) | Timed plan, prompts, expected evidence, and adaptations |
| [Problem proposal](../../assignments/02-problem-proposal.md) | Research question, dataset card draft, baseline, and evaluation plan |

## Preparation

For Week 2, review the Week 1 problem canvas and run:

```bash
python labs/week02_evaluation_numpy.py
```

For Week 3, bring one candidate dataset and run:

```bash
python labs/week03_data_quality_numpy.py
```

Both labs use generated data and NumPy. They require no external dataset.

## Evidence across the module

Students produce:

- a future-use claim and justified split;
- corrected leakage-prone workflows;
- a metric interpretation grounded in error cost;
- a compact data-quality audit;
- a controlled comparison table;
- a dataset feasibility decision; and
- the Week 3 problem proposal.

## Essential reading

- [scikit-learn: common pitfalls and recommended practices](https://scikit-learn.org/stable/common_pitfalls.html)
- [PyTorch reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness.html)
- [Datasheets for Datasets](https://doi.org/10.1145/3458723)
- [Course research-project brief](../../research-project/brief.md)

## Interpretation boundaries

The Week 2 lab exaggerates source similarity to make group leakage visible. The Week 3 lab creates a rare class and informative missingness so that metric and feature decisions become visible. Their numerical results belong to those simulations. Students must re-establish every assumption for a real dataset.
