# Week 7 instructor guide: Multiclass learning and consolidation

## Purpose

Week 7 moves from binary output to mutually exclusive multiclass learning and then consolidates the evidence chain from Weeks 1–6. Students implement stable row-wise softmax, connect class-index and one-hot targets to categorical cross-entropy, derive the output gradient, check every linear-softmax parameter gradient, and evaluate one selected model with class-aware evidence.

The historical material placed multiclass classification after a long Week 7 treatment of backpropagation, momentum, cross-entropy, and regularization. The revised sequence keeps the useful output-design, one-hot, softmax, and contaminated-input ideas. Week 6 now owns backpropagation and binary loss. Week 8 owns deeper networks, overfitting, regularization, and dropout. MATLAB examples are replaced by one explicit NumPy procedure with a locked test split.

## Before class

- Run python labs/week07_softmax_numpy.py.
- Distribute [the Week 7 worksheet](../modules/06-multiclass-and-generalization/worksheet.md).
- Ask students to bring their Week 6 reproducible baseline and split identifiers.
- Write one fixed class order where everyone can see it.
- Keep the lab output hidden until students predict the row sums, shift effect, and majority macro recall.
- Prepare three colors for true classes and a $3\times3$ confusion-matrix grid.

## Learning evidence

Inspect:

1. correct distinction among binary, multiclass, and multilabel targets;
2. $K$ logits and their matrix shapes;
3. stable row-wise softmax pseudocode;
4. shift-invariance reasoning;
5. equivalent class-index and one-hot cross-entropy;
6. the $G_Z=(P-Y)/m$ output gradient;
7. a passing parameter-gradient check;
8. validation-only learning-rate selection;
9. confusion-matrix orientation and class recall;
10. accuracy-versus-macro-recall interpretation; and
11. the project's evaluation review.

## Three-hour sequence

| Time | Activity | Instructor action | Student evidence |
| ---: | --- | --- | --- |
| 0–10 min | Week 6 bridge | Replace one binary logit with $K$ class logits | Output shape |
| 10–25 min | Task semantics | Contrast exclusive classes and independent labels | Task-to-output table |
| 25–45 min | Softmax | Convert scores to one row distribution | Manual probability calculation |
| 45–58 min | Stability | Prove shift invariance and subtract the row maximum | Stable pseudocode |
| 58–75 min | Targets and loss | Align class indices, one-hot rows, and class order | Cross-entropy calculation |
| 75–90 min | Output gradient | Derive or interpret $G_Z=(P-Y)/m$ | Gradient vector and row-sum check |
| 90–100 min | Break |  |  |
| 100–118 min | Parameter gradients | Trace $G_W$ and $G_b$ with shapes | Shape table |
| 118–130 min | Gradient checking | Compare analytical and numerical components | Tolerance and claim boundary |
| 130–148 min | Split roles | Select on validation and lock the test set | Procedure ordering |
| 148–163 min | Class-aware metrics | Read confusion counts, accuracy, and macro recall | Annotated matrix |
| 163–173 min | NumPy lab | Require predictions before execution | Recorded deterministic output |
| 173–178 min | Course consolidation | Connect Weeks 1–7 into one evidence chain | Failure-to-check map |
| 178–180 min | Exit ticket | Collect five concise statements | Completed exit ticket |

## Facilitation notes

### Task semantics

Start with three questions:

1. Can exactly one class be correct?
2. Can several labels be correct at once?
3. Is the target continuous?

For mutually exclusive $K$-class prediction, use $K$ competing logits and one softmax distribution. For multilabel prediction, each label needs an independent decision; one sigmoid per label is a common design.

Do not let students choose an output layer from the number of labels alone. The co-occurrence rules matter.

### Logits

Write one row:

$$
z=[2,1,0].
$$

Ask which class wins before computing softmax. The largest logit wins, but the values are not probabilities.

Emphasize that softmax preserves ordering. Its probability values still affect cross-entropy and any later confidence-based decision.

### Softmax

Expected equation:

$$
p_k=
\frac{e^{z_k}}
{\sum_j e^{z_j}}.
$$

For $z=[2,1,0]$, subtract two and use approximately:

$$
e^0=1,\qquad e^{-1}=0.368,\qquad e^{-2}=0.135.
$$

The probabilities are approximately:

$$
[0.665,0.245,0.090].
$$

Require the sum check.

### Shift invariance

Show:

$$
\frac{e^{z_k+c}}{\sum_j e^{z_j+c}}
=
\frac{e^c e^{z_k}}{e^c\sum_j e^{z_j}}
=
\frac{e^{z_k}}{\sum_j e^{z_j}}.
$$

This proof supports maximum subtraction. It is not an arbitrary software trick.

### Stable implementation

For batch logits $Z:(m,K)$:

1. compute the maximum across classes for each row;
2. subtract that row maximum;
3. exponentiate shifted logits;
4. sum across classes for each row; and
5. divide each row by its own sum.

Normalizing over the entire batch makes one observation's probabilities depend on other observations.

### Class indices and one-hot rows

Use one fixed order. For class index $1$ among three classes:

$$
y=[0,1,0].
$$

The class order belongs in saved metadata. A correct numerical matrix with the wrong class mapping is still an incorrect experiment.

### Cross-entropy

Expected one-hot form:

$$
J=
-\frac{1}{m}
\sum_i\sum_k Y_{ik}\log P_{ik}.
$$

Expected class-index form:

$$
J=
-\frac{1}{m}
\sum_i\log P_{i,t_i}.
$$

For $p=[0.70,0.20,0.10]$ and target index $1$:

$$
\ell=-\log(0.20)\approx1.609.
$$

For target index $0$, the loss is approximately $0.357$.

### Output gradient

Expected result:

$$
G_Z=\frac{P-Y}{m}.
$$

For one row with $p=[0.70,0.20,0.10]$ and $y=[0,1,0]$:

$$
G_z=[0.70,-0.80,0.10].
$$

The target component is negative, asking gradient descent to raise its relative logit. Non-target components are positive, asking it to lower their relative logits. The row sum is zero.

### Parameter gradients

For:

$$
Z=XW+b,
$$

expected gradients are:

$$
G_W=X^TG_Z,
$$

$$
G_b=\operatorname{sum}(G_Z,\text{rows}).
$$

Require $G_W:(d,K)$ and $G_b:(1,K)$.

### Gradient checking

The lab checks every component of a $2\times3$ weight matrix and a $1\times3$ bias. The maximum relative error is approximately $8.086\times10^{-11}$. The teaching tolerance is $10^{-6}$.

A passing result supports the derivative code at the checked parameters. It does not validate the class order, split, preprocessing, or metric.

### Split roles

The lab creates a deterministic stratified split:

- training: 60% of each class;
- validation: 20% of each class;
- test: 20% of each class.

Standardization is fitted on training features. Four rates are compared with validation cross-entropy. The selected stored model is evaluated on test data once.

Ask which values could change after reading validation results. Then ask which values must already be fixed before test evaluation.

### Confusion matrix

State the course convention before displaying values:

- rows = true classes;
- columns = predicted classes.

The lab test matrix is:

$$
\begin{bmatrix}
29 & 1 & 0 \\
0 & 15 & 3 \\
0 & 0 & 12
\end{bmatrix}.
$$

Read it aloud:

- one true class-0 observation is predicted as class 1;
- three true class-1 observations are predicted as class 2;
- all twelve class-2 observations are correct.

The row totals $[30,18,12]$ match the declared test supports.

### Accuracy and macro recall

Expected equations:

$$
\text{accuracy}
=
\frac{\sum_k C_{kk}}{\sum_i\sum_j C_{ij}},
$$

and:

$$
\text{macro recall}
=
\frac{1}{K}\sum_k
\frac{C_{kk}}{\sum_j C_{kj}}.
$$

The majority baseline predicts only class 0. Its accuracy is $0.500$, while its macro recall is $0.333$. Accuracy credits correct observations; macro recall gives each class equal influence.

The trained model achieves $0.933$ for both metrics on this split. The equality is a property of these particular class recalls and counts, not a general rule.

### Week 7 lab

Expected deterministic output:

| Evidence | Value |
| --- | --- |
| first probability row | approximately $[0.7214,0.2654,0.0132]$ |
| second probability row | approximately $[0.2654,0.7214,0.0132]$ |
| row sums | $[1,1]$ |
| largest shift difference | $0$ at printed precision |
| extreme-logit cross-entropy | $0.3266$ |
| maximum gradient error | approximately $8.086\times10^{-11}$ |
| validation losses for rates $0.01,0.05,0.20,0.80$ | $0.5456,0.3222,0.2600,0.2456$ |
| selected learning rate | $0.80$ |
| test class counts | $[30,18,12]$ |
| majority accuracy / macro recall | $0.500 / 0.333$ |
| model accuracy / macro recall | $0.933 / 0.933$ |
| model test cross-entropy | $0.2179$ |
| confusion matrix | $[[29,1,0],[0,15,3],[0,0,12]]$ |

The test result is evidence about one synthetic split. It is not a benchmark for another dataset or deployment setting.

### Consolidation discussion

Give each group one failure and ask which earlier evidence should catch it:

| Failure | Earliest useful evidence |
| --- | --- |
| target does not match the question | Week 1 problem framing |
| repeated subjects cross splits | Week 2 group-aware evaluation |
| imputer sees all data | Week 3 fitted preprocessing audit |
| output width is wrong | Week 4 shape trace |
| optimizer diverges | Week 5 learning curves and validation |
| hidden gradient is wrong | Week 6 numerical checking |
| minority class is ignored | Week 7 confusion matrix and macro recall |

This review prepares students to explain a complete experimental procedure rather than recite isolated formulas.

### Evaluation-review milestone

Require:

- task type and unit of observation;
- output and objective compatibility;
- split identities and roles;
- leakage audit;
- primary metric with direction, averaging, and decision rule;
- a simple reference;
- class or subgroup evidence;
- configuration selection history;
- locked, final-use, or provisional test status;
- error analysis;
- supported claim, unsupported claim, and next experiment.

Do not accept an unlabeled test score copied from a training log.

## Common misconceptions

| Misconception | Response |
| --- | --- |
| Three classes require three independent sigmoids | Only if labels can co-occur; exclusive classes form one softmax distribution |
| Logits are unnormalized probabilities | They are relative scores without probability constraints |
| Softmax can normalize the entire batch | Each observation needs its own distribution over classes |
| Maximum subtraction changes the prediction | Shift invariance proves the probabilities are unchanged |
| One-hot targets are a different task from class indices | They are two encodings of the same exclusive target |
| Cross-entropy equals misclassification count | It uses the probability assigned to the target |
| $P-Y$ applies to every possible output and loss | It follows from this paired softmax and categorical objective |
| A gradient check validates the experiment | It checks derivative calculations at selected values |
| Accuracy is sufficient for imbalanced classes | Class recall and the confusion matrix may reveal hidden failures |
| The test set can select the best rate once | Any selection use makes that result non-independent |
| High confidence means the probability is calibrated | Softmax output alone does not establish calibration |

## Formative feedback language

- “Can more than one label be true?”
- “Which dimension represents classes?”
- “What is the class order?”
- “What common constant can be removed?”
- “Does each row sum to one?”
- “Which probability enters the loss?”
- “Why does this gradient row sum to zero?”
- “Does the gradient shape match the parameter?”
- “Which split informed this choice?”
- “What do the rows and columns represent?”
- “Which class is hardest?”
- “What claim does this procedure support?”

## Adaptations

### Ninety-minute class

Use slides 1–12, 15–19, and 22. Complete one stable softmax calculation, one cross-entropy example, the output gradient, the lab prediction, and confusion-matrix interpretation. Assign the consolidation map and evaluation review afterward.

### Online delivery

Use a shared table with one logit row per group. Ask groups to shift by different constants and compare identical probabilities. For evaluation, assign one confusion-matrix row to each group and combine the class recalls.

### Limited calculus preparation

Treat $P-Y$ as a sensitivity vector first. Use its signs and zero row sum to explain class competition. Present the full derivative as an optional extension, while still requiring shape and numerical-check reasoning.

## After class

- identify students who normalize softmax across observations;
- check projects for changing class order or undocumented label mapping;
- review whether preprocessing used training data only;
- require exact metric averaging and test-status declarations;
- return provisional test results for an independent-evaluation plan; and
- connect evaluation gaps to Week 8 regularization and controlled comparisons.
