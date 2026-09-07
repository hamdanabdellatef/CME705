# Week 8: Deep networks and generalization

**Guiding question:** When does added capacity learn useful structure, and when does it memorize training detail?

Week 8 studies deeper ReLU networks through the evidence visible in training and validation curves. Students diagnose optimization failure and overfitting separately, implement inverted dropout correctly in training and evaluation modes, compare dropout rates while holding all other decisions fixed, and use early stopping to restore the best validation checkpoint. Weight decay is derived as a second regularization mechanism, while the lab changes only dropout so the causal comparison remains interpretable.

## Learning outcomes

By the end of the week, students can:

- explain how additional layers change representation capacity and optimization paths;
- distinguish training failure, overfitting, and distribution shift from curve evidence;
- compute ReLU and its local derivative;
- describe vanishing and exploding gradients without claiming one activation solves every case;
- interpret the generalization gap as a diagnostic rather than a performance metric;
- derive the gradient contribution of $L_2$ regularization;
- explain the difference between an objective penalty and decoupled weight decay;
- select and restore an early-stopping checkpoint using validation evidence;
- implement inverted dropout with correct scaling;
- reuse the forward dropout mask during backpropagation;
- disable dropout during evaluation;
- design a controlled comparison that changes one factor;
- preserve a locked test set until the comparison is fixed; and
- state a claim no broader than the data and procedure support.

## Teaching package

- [Concept notes](week08-notes.md)
- [Accessible slide text](week08-slides.md)
- [PowerPoint lecture deck](slides/week08-deep-networks-and-generalization.pptx)
- [Student worksheet](week08-worksheet.md)
- [NumPy deep-network and dropout lab](../../labs/week08_dropout_numpy.py)
- [Instructor guide](../../instructor-notes/week08.md)
- [Controlled-experiment assignment](../../assignments/04-controlled-experiment.md)

## Preparation

Students should understand forward and backward passes, validation-controlled selection, binary cross-entropy, and the distinction between class-specific evaluation and training objectives.

Run:

~~~bash
python labs/week08_dropout_numpy.py
~~~

## Three-hour sequence

1. Separate depth, width, and parameter count from model quality.
2. Trace ReLU forward and backward behavior.
3. Distinguish optimization failure from overfitting with learning curves.
4. interpret the generalization gap and its limits.
5. derive $L_2$ regularization and weight shrinkage.
6. define an early-stopping rule before training.
7. derive inverted dropout scaling.
8. reuse the forward mask in the backward pass.
9. separate training-mode randomness from deterministic evaluation.
10. predict and run the controlled dropout comparison.
11. interpret validation selection and locked test evidence.
12. design the project’s one-factor controlled experiment.

## Evidence of learning

Students submit or show:

- an annotated training/validation curve diagnosis;
- a ReLU forward and backward calculation;
- an $L_2$ gradient and update calculation;
- an early-stopping checkpoint record;
- an inverted-dropout expectation calculation;
- a dropout backward trace using the saved mask;
- a training-versus-evaluation mode table;
- a validation comparison in which only dropout changes;
- a locked-test result for the fixed comparison; and
- a controlled-experiment plan with hypothesis, controls, uncertainty, and claim boundary.

## Claim boundary

The lab establishes correct inverted-dropout mechanics, deterministic evaluation behavior, and a validation-selected improvement on one deliberately small synthetic task with nuisance features and controlled training-label noise. It does not establish that dropout always improves performance, that a dropout rate of $0.45$ transfers to another task, or that the model is robust to real distribution shifts.

## Suggested references

- [Deep Learning, Chapter 7: Regularization for Deep Learning](https://www.deeplearningbook.org/contents/regularization.html)
- [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://jmlr.org/papers/v15/srivastava14a.html)
- [PyTorch Dropout documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout.html)

Use these sources to clarify mechanisms. The project comparison must still identify its data, split, fixed controls, random seeds, metric, uncertainty method, and limitations.
