# Week 7: Multiclass learning and consolidation

**Guiding question:** How do $K$ competing scores become stable probabilities and defensible multiclass evidence?

Week 7 extends the binary output-loss pairing from Week 6 to mutually exclusive multiclass classification. Students convert logits to probabilities with stable softmax, connect class-index and one-hot targets to categorical cross-entropy, derive the compact output gradient, and train a linear softmax classifier. The second half consolidates Weeks 1–6 through split roles, confusion matrices, accuracy, macro recall, and a formal evaluation review.

## Learning outcomes

By the end of the week, students can:

- distinguish mutually exclusive multiclass classification from multilabel prediction;
- choose $K$ output logits for a $K$-class problem;
- explain why logits are relative scores rather than probabilities;
- implement row-wise softmax with maximum subtraction;
- prove that adding the same constant to every logit leaves softmax unchanged;
- represent targets as class indices or one-hot vectors;
- compute categorical cross-entropy from stable log probabilities;
- derive $G_Z=(P-Y)/m$ for softmax with categorical cross-entropy;
- check linear-softmax parameter gradients numerically;
- preserve training, validation, and test roles during model selection;
- read a confusion matrix with an explicit orientation;
- compare accuracy with macro recall on imbalanced classes; and
- justify the metric and split used by a research project.

## Teaching package

- [Concept notes](notes.md)
- [Accessible slide text](slides.md)
- [PowerPoint lecture deck](slides/week07-multiclass-learning-and-consolidation.pptx)
- [Student worksheet](worksheet.md)
- [NumPy multiclass lab](../../labs/week07_softmax_numpy.py)
- [Instructor guide](../../instructor-notes/week07.md)
- [Evaluation-review milestone](../../research-project/evaluation-review.md)

## Preparation

Students should understand the Week 6 binary sigmoid and cross-entropy pairing, the meaning of logits, matrix multiplication, one backward pass, and the separation of training, validation, and test data.

Run:

~~~bash
python labs/week07_softmax_numpy.py
~~~

## Three-hour sequence

1. Decide whether a task is binary, mutually exclusive multiclass, or multilabel.
2. Map one observation to $K$ logits and interpret only their relative differences.
3. Derive softmax and its probability constraints.
4. make the implementation stable by subtracting each row maximum.
5. connect class-index and one-hot targets.
6. derive categorical cross-entropy and $G_Z=(P-Y)/m$.
7. trace linear-softmax parameter shapes and verify the gradients.
8. select a learning rate with validation cross-entropy.
9. open the test split once and read the confusion matrix.
10. compare accuracy, macro recall, and cross-entropy.
11. consolidate the evidence chain from Weeks 1–6.
12. complete the project evaluation review.

## Evidence of learning

Students submit or show:

- a correct output design for one multiclass task;
- stable softmax pseudocode and a shift-invariance explanation;
- equivalent class-index and one-hot cross-entropy calculations;
- the softmax-cross-entropy logit gradient with valid shapes;
- a passing finite-difference check;
- a validation-only model-selection record;
- an annotated confusion matrix;
- an accuracy-versus-macro-recall interpretation; and
- an evaluation review that states the metric, averaging, split, leakage check, and test status.

## Claim boundary

The lab establishes numerical stability for the checked extreme logits, agreement between analytical and finite-difference gradients at the checked parameters, and performance on one deterministic synthetic split. It does not establish calibration, robustness to distribution shift, or generalization to a real application.

## Suggested references

- [Deep Learning, Chapter 6: Deep Feedforward Networks](https://www.deeplearningbook.org/contents/mlp.html)
- [CS231n linear classification notes](https://cs231n.github.io/linear-classify/)
- [PyTorch CrossEntropyLoss documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)

Use these sources to clarify mechanisms. Project evidence must still document the dataset, split, metric, command, actual result, and limitations.
## Week 8 continuation

Continue with [Week 8: Deep networks and generalization](week08.md). Its separate teaching package adds ReLU and depth, learning-curve diagnosis, $L_2$ regularization, early stopping, correct inverted dropout, and a controlled validation comparison.

- [Week 8 concept notes](week08-notes.md)
- [Week 8 accessible slide text](week08-slides.md)
- [Week 8 PowerPoint lecture deck](slides/week08-deep-networks-and-generalization.pptx)
- [Week 8 student worksheet](week08-worksheet.md)
- [Week 8 NumPy lab](../../labs/week08_dropout_numpy.py)
- [Week 8 instructor guide](../../instructor-notes/week08.md)
- [Controlled-experiment assignment](../../assignments/04-controlled-experiment.md)
