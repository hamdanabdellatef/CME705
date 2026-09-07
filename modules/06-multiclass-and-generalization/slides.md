# Week 7 slide source: Multiclass learning and consolidation

This file is the accessible text version of the PowerPoint deck. Speaker prompts and timing are in the presentation and instructor guide.

## Slide 1: CME705 Machine Learning

**Week 7: Multiclass learning and consolidation**

Guiding question: How do $K$ competing scores become stable probabilities and defensible multiclass evidence?

## Slide 2: Today’s evidence

Each student will produce:

1. a task-to-output design;
2. stable softmax pseudocode;
3. a target and loss calculation;
4. a softmax-cross-entropy gradient;
5. a numerical gradient check;
6. an annotated confusion matrix; and
7. a metric-and-split justification.

## Slide 3: From one binary logit to $K$ class logits

Binary classification can use one logit and one sigmoid probability.

Mutually exclusive $K$-class classification uses $K$ logits:

$$
Z=XW+b,
$$

with $Z:(m,K)$.

Exactly one class is correct for each observation.

## Slide 4: Mutually exclusive is not multilabel

**Multiclass:** one outcome among $K$ competing classes; use one softmax distribution.

**Multilabel:** each label may be present independently; commonly use one sigmoid per label.

Output design follows the target semantics.

## Slide 5: Logits are relative scores

For one observation:

$$
z=[z_1,z_2,ldots,z_K].
$$

A logit may be negative or larger than one. Its absolute value is not a probability.

Softmax compares all logits in the same row.

## Slide 6: Softmax creates one probability distribution

$$
p_k=rac{e^{z_k}}{sum_{j=1}^{K}e^{z_j}}.
$$

Properties:

- $0<p_k<1$;
- $sum_k p_k=1$; and
- the largest logit has the largest probability.

Prediction is commonly $argmax_k p_k$.

## Slide 7: Only differences among logits matter

For any constant $c$:

$$
operatorname{softmax}(z+c)=operatorname{softmax}(z).
$$

Adding the same constant to a row changes no probability.

This invariance enables a stable computation.

## Slide 8: Subtract the row maximum

Compute:

$$
	ilde z_k=z_k-max_j z_j.
$$

Then:

$$
p_k=rac{e^{	ilde z_k}}{sum_j e^{	ilde z_j}}.
$$

The largest shifted logit is zero, so no exponent is larger than one.

## Slide 9: Class indices and one-hot targets

For three classes, class index $1$ corresponds to:

$$
y=[0,1,0].
$$

Class indices are compact. One-hot rows expose the algebra.

Class order must remain fixed in data, outputs, metrics, and reports.

## Slide 10: Categorical cross-entropy

For one-hot targets:

$$
J=-rac{1}{m}sum_{i=1}^{m}sum_{k=1}^{K}
Y_{ik}log P_{ik}.
$$

For class indices $t_i$:

$$
J=-rac{1}{m}sum_{i=1}^{m}log P_{i,t_i}.
$$

The two forms describe the same objective.

## Slide 11: Cross-entropy uses the target probability

If:

$$
p=[0.70,0.20,0.10]
$$

and the target class is index $1$, then:

$$
ell=-log(0.20)approx1.609.
$$

Argmax is wrong here, and the loss records how little probability reached the target.

## Slide 12: The output gradient remains compact

For softmax with categorical cross-entropy:

$$
G_Z=rac{P-Y}{m}.
$$

Each row’s gradient sums to zero because increasing one class’s relative score must reduce others’ shares.

This is the multiclass analogue of Week 6.

## Slide 13: Linear softmax classifier shapes

For $d$ features and $K$ classes:

| Quantity | Shape |
| --- | --- |
| $X$ | $(m,d)$ |
| $W$ | $(d,K)$ |
| $b$ | $(1,K)$ |
| $Z,P,Y$ | $(m,K)$ |

The output width equals the class count.

## Slide 14: Parameter gradients

$$
G_W=X^TG_Z,
$$

$$
G_b=operatorname{sum}(G_Z,	ext{rows}).
$$

Shapes:

- $G_W:(d,K)$, matching $W$;
- $G_b:(1,K)$, matching $b$.

## Slide 15: Numerical checking isolates derivative errors

Centered differences compare each analytical component with an independent numerical estimate.

The Week 7 lab checks every component of $W$ and $b$.

A low error supports the derivative implementation at the checked values.

## Slide 16: Selection and final evaluation have different roles

1. Fit parameters and preprocessing on training data.
2. Compare learning rates on validation data.
3. Freeze the selected configuration.
4. Open test data once for final evidence.

A test score used to choose a configuration is no longer independent final evidence.

## Slide 17: Read the confusion matrix orientation first

The Week 7 lab uses:

- rows = true classes;
- columns = predicted classes.

Diagonal cells are correct predictions. Off-diagonal cells name specific confusions.

Every row total equals that true class’s support.

## Slide 18: Accuracy and macro recall answer different questions

$$
	ext{accuracy}=
rac{	ext{all correct}}{	ext{all observations}}.
$$

$$
	ext{macro recall}=
rac{1}{K}sum_{k=1}^{K}
rac{C_{kk}}{sum_j C_{kj}}.
$$

Accuracy weights observations. Macro recall weights classes equally.

## Slide 19: NumPy lab evidence

Run:

~~~bash
python labs/week07_softmax_numpy.py
~~~

Expected evidence:

- stable probabilities for logits near $pm1000$;
- softmax shift difference $0$ at printed precision;
- maximum gradient-check error near $8.1	imes10^{-11}$;
- validation-selected learning rate $0.80$;
- majority accuracy $0.500$ and macro recall $0.333$;
- model accuracy and macro recall $0.933$; and
- a $3	imes3$ test confusion matrix.

## Slide 20: Consolidation: one evidence chain

1. Define the research question and observation unit.
2. inspect data provenance and leakage paths.
3. freeze split roles and fit preprocessing on training data.
4. map features through model computations.
5. optimize with checked gradients.
6. select with validation evidence.
7. report independent test and class-specific evidence.
8. state a claim no broader than the procedure supports.

## Slide 21: Evaluation review

Record:

- task type and prediction unit;
- split identities and leakage checks;
- primary metric, direction, averaging, and decision rule;
- simple reference performance;
- class or subgroup results;
- selection history;
- honest test status;
- errors, supported claim, limitation, and next experiment.

## Slide 22: Exit and next step

Explain:

1. why softmax must be computed row-wise;
2. why subtracting the row maximum changes no probability;
3. why $G_Z=(P-Y)/m$;
4. what the confusion-matrix orientation means; and
5. when macro recall is more informative than accuracy.

Next: deeper networks, generalization gaps, and regularization.
