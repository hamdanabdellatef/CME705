# CME705 Machine Learning

## Neural Networks, Deep Learning, and Research Practice

[![Quality checks](https://github.com/hamdanabdellatef/CME705/actions/workflows/quality.yml/badge.svg)](https://github.com/hamdanabdellatef/CME705/actions/workflows/quality.yml)

A complete 14-week graduate course developed for the Department of Computer Engineering at Karabuk University. The course connects machine-learning foundations with implementation and research practice. Students first make core learning mechanisms visible with NumPy, then use PyTorch for modern deep-learning models, and develop a research question through a semester project.

This repository reorganizes material taught in Spring 2022 into a coherent Python course. It includes the full teaching route, assessment structure, reproducible labs, presentation decks, instructor guidance, and a research-project workflow that ends with a reviewable release and a feasible next research direction.

## Start here

### Students

1. Read the [syllabus](syllabus.md) and [14-week schedule](schedule.md).
2. Follow the [Python setup guide](setup.md).
3. Begin with the Week 1 module and [Python/NumPy diagnostic](labs/week01_python_numpy_diagnostic.py).
4. Start the [research project](research-project/brief.md) and follow its [milestones](research-project/milestones.md).

### Instructors

1. Review the [syllabus](syllabus.md), [schedule](schedule.md), and [instructor guidance](instructor-notes/README.md).
2. Set dates and institutional submission channels without changing the stable assignment IDs.
3. Run the validation commands in [setup.md](setup.md).
4. Review the [project rubric](research-project/rubric.md), [presentation rubric](research-project/presentation-rubric.md), and [final-release checklist](research-project/final-release-checklist.md).

## Course at a glance

| Weeks | Theme | Main work |
| ---: | --- | --- |
| 1-3 | Problem formulation, generalization, evaluation, and data quality | NumPy diagnostics, leakage-safe splits, data audit, and problem proposal |
| 4-8 | Neural computation, optimization, backpropagation, and generalization | Forward pass, gradient descent, XOR MLP, softmax, dropout, and controlled experiments |
| 9-10 | Convolutional neural networks | NumPy convolution mechanics, PyTorch MNIST training, inspection, and ConvNeXt transfer learning |
| 11 | Sequence models | RNN and LSTM mathematics, attention, Transformers, delayed-memory experiments, and ModernBERT |
| 12 | Generative models | Autoencoders, VAEs, GANs, flows, diffusion, score models, EBMs, and CIFAR-10 DDPM |
| 13 | Reproducible research and peer review | Artifact identity, uncertainty, independent reproduction, revision, and report preparation |
| 14 | Research presentation and final release | Oral defense, static release audit, final submission, and research-direction planning |

See the [complete schedule](schedule.md) for the lesson, lab, and research milestone assigned to every week.

## Course outcomes

By the end of the course, students should be able to:

- formulate a machine-learning problem from an application need;
- design an evaluation that avoids data leakage and supports a defensible conclusion;
- implement and explain neural-network training with NumPy;
- train and inspect convolutional, sequence, and generative models with PyTorch;
- compare methods through controlled experiments and appropriate metrics;
- read research critically and turn experimental findings into a feasible next research question; and
- release code and results that another researcher can reproduce.

## Weekly teaching packages

The developed weeks include:

- a lesson guide and detailed technical notes;
- an accessible Markdown slide source and PowerPoint deck;
- a student worksheet;
- a runnable or auditable Python lab;
- an instructor delivery guide; and
- a research-project milestone or assessment where appropriate.

The final capstone package is available in [Week 14](modules/10-research-practice/week14.md), including an editable [student presentation template](research-project/presentation-template.md).

## Repository map

- [`modules/`](modules/README.md): lessons, technical notes, worksheets, accessible slide sources, and PowerPoint decks
- [`labs/`](labs/README.md): runnable NumPy and PyTorch examples
- [`assignments/`](assignments/README.md): six semester assignments
- [`research-project/`](research-project/brief.md): project brief, milestones, rubrics, report and presentation templates, and release tools
- [`readings/`](readings/README.md): annotated links to official and primary sources
- [`data/`](data/README.md): dataset and split policy
- [`instructor-notes/`](instructor-notes/README.md): delivery and assessment guidance

## Technology and runtime

The course uses Python 3.11 or 3.12, NumPy, and PyTorch. NumPy makes foundational computations inspectable. PyTorch supports CNNs, sequence models, transfer learning, and generative models. Weeks 10-12 automatically use CUDA when a compatible GPU and CUDA-enabled PyTorch installation are available.

MATLAB material from the historical course is intentionally excluded. Dataset downloads, pretrained weights, checkpoints, and generated outputs remain outside version control.

## Research project

The semester project begins with an application interest and develops through problem formulation, literature comparison, baseline construction, controlled evaluation, error analysis, reproducibility review, and final defense.

Key documents:

- [Project brief](research-project/brief.md)
- [Milestones](research-project/milestones.md)
- [Project rubric](research-project/rubric.md)
- [Report template](research-project/report-template.md)
- [Final submission](assignments/06-final-project-submission.md)
- [Research-direction note](research-project/research-direction-note.md)

A failed experiment can be a valid result when the protocol is sound, the failure is analyzed, and the next experiment is capable of changing the conclusion.

## Release status

Version 0.1.0 is the first public-ready course release. Core NumPy labs use generated data; the PyTorch CNN, sequence, and generative-model labs download declared public datasets or pretrained weights at run time.

Validate the repository from its root:

```bash
python -m compileall labs tests
python -m pytest
python labs/week14_release_audit.py --audit-only
```

## Citation and license

Cite the course using [`CITATION.cff`](CITATION.cff). Course text is licensed under CC BY 4.0 and original code under the MIT License; see [`LICENSE.md`](LICENSE.md). Third-party books, slides, source trees, student records, and student submissions are not included.
