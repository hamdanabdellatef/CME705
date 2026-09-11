# Week 1: Machine-learning foundations

**Guiding question:** When should we learn a pattern from data, and what evidence would make that pattern useful?

Week 1 introduces the course through a synthetic machine-maintenance case. Students translate an application need into a learning task, inspect its representation in NumPy, compare a simple decision rule with a baseline, and begin identifying a research direction.

## Learning outcomes

By the end of the week, students should be able to:

- distinguish artificial intelligence, machine learning, and deep learning;
- explain how a learned solution differs from an explicitly programmed rule;
- identify the unit of observation, features, target, model output, and intended use in a problem statement;
- distinguish classification, regression, clustering, and representation learning;
- inspect a small dataset with Python and NumPy;
- calculate accuracy and explain why a baseline is needed; and
- state a feasible research interest without claiming a solution in advance.

## Week 1 materials

| Resource | Purpose |
| --- | --- |
| [Student notes](notes.md) | Core concepts, vocabulary, examples, and review questions |
| [Lecture source](slides.md) | Accessible text version of the lecture deck |
| [PowerPoint lecture deck](slides/week01-machine-learning-foundations.pptx) | Classroom presentation with speaker notes |
| [Student worksheet](worksheet.md) | Problem-formulation activities, lab record, and exit ticket |
| [Python/NumPy diagnostic](../../labs/week01_python_numpy_diagnostic.py) | Runnable synthetic-data example |
| [Instructor guide](../../instructor-notes/week01.md) | Timed plan, prompts, expected evidence, and adaptations |
| [Research-interest memo](../../assignments/01-research-interest.md) | First research-project milestone |

## Preparation

Before class, students should:

1. complete the environment steps in [setup.md](../../setup.md);
2. confirm that `python --version` reports Python 3.11 or 3.12;
3. run `python -c "import numpy; print(numpy.__version__)"`; and
4. bring one application area they may want to investigate.

No previous machine-learning course is assumed. Students should be comfortable with variables, functions, one-dimensional arrays, and basic algebra. The diagnostic identifies gaps to address before the neural-network modules.

## Learning sequence

1. Examine an application decision: which machines should receive an inspection?
2. Separate the application goal from the prediction task.
3. Identify observations, features, targets, and model outputs.
4. Compare common learning-task types.
5. Express a linear scoring rule as $s = Xw + b$.
6. Run the diagnostic and compare the rule with a majority-class baseline.
7. Record what the result supports and what remains unknown.
8. Apply the same formulation process to a possible research interest.

## Evidence of learning

Students leave class with:

- a completed problem-formulation canvas;
- a recorded diagnostic result and interpretation;
- an exit ticket that distinguishes an application goal from an ML task; and
- a draft topic for the research-interest memo.

## Essential reading

- [NumPy quickstart](https://numpy.org/doc/stable/user/quickstart.html), through array creation, dimensions, and basic operations
- [PyTorch Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html), overview only
- [Course research-project brief](../../research-project/brief.md)

The Python lab uses generated data. It does not represent a real industrial dataset or establish that the example rule would be safe in practice.
