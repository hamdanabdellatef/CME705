# CME705 Machine Learning syllabus

**Subtitle:** Neural Networks, Deep Learning, and Research Practice
**Institution:** Karabuk University, Department of Computer Engineering
**Level:** Graduate
**Duration:** 14 weeks

## Course description

CME705 introduces machine learning through generalization, data quality, neural-network computation, optimization, deep learning, and research practice. Students implement essential mechanisms in NumPy before using PyTorch for convolutional, sequence, and generative models. A semester research project connects technical choices to literature, evidence, reproducibility, and a feasible direction for further study.

## Learning outcomes

A successful student can:

1. distinguish classification, regression, clustering, and representation-learning problems;
2. create training, validation, and test procedures suited to the data-generating process;
3. recognize leakage, imbalance, missing data, noise, overfitting, and misleading metrics;
4. derive and implement gradient-based learning for feed-forward neural networks;
5. explain activation functions, losses, regularization, convolution, recurrence, attention, and latent representations;
6. implement small models with NumPy and train deep models with PyTorch;
7. design controlled comparisons, analyze errors, and communicate limitations; and
8. synthesize relevant literature and propose a justified next research question.

## Prerequisites

Students should be comfortable with vectors and matrices, derivatives and the chain rule, basic probability, and introductory Python. [`setup.md`](setup.md) provides the required software instructions. Week 1 includes a diagnostic and directs students to refresh missing prerequisites.

## Learning approach

Each week combines a concept guide, implementation or experiment, and progress on the research project. NumPy examples make the learning mechanisms visible. PyTorch examples introduce maintained tensor and model abstractions. Students record hypotheses, controlled changes, results, and conclusions in every experiment.

## Assessment

| Component | Weight | Evidence |
| --- | ---: | --- |
| Participation and research workshops | 5% | Prepared participation and constructive peer feedback |
| Assignments | 20% | Six assignments listed in [`assignments/`](assignments/README.md) |
| Midterm assessment | 15% | Foundations, optimization, and neural-network reasoning |
| Final assessment | 30% | Deep-learning concepts and interpretation of the student's project |
| Research project | 30% | Milestones, reproducible code, report, and presentation |

The project rubric is in [`research-project/rubric.md`](research-project/rubric.md). Late-work, accessibility, academic-integrity, and approved-accommodation rules should follow the policies announced for the current offering.

## Required project practice

Every project must identify the unit of observation, document all data splits, preserve a final test set, establish a simple baseline, report a metric appropriate to the task, and include an error or limitation analysis. Preprocessing parameters must be learned using training data only. Students must distinguish reproduced results from results reported by prior work.

## Research and academic integrity

Cite sources for ideas, data, figures, and code. Mark adapted code and describe modifications. Do not submit fabricated experiments, citations, or results. Record the role of any automated tool used for writing, coding, or analysis according to the current university policy and the instructor's directions. Human-subject, medical, biometric, or otherwise sensitive data require an appropriate approval and data-governance plan before use.

## Communication

Project writing should be concise, precise, and evidence based. Tables and figures must have readable labels and captions. A reader should be able to reproduce the reported result from the repository and instructions.
