# Week 4 slide source: Neural-network computation

This file is the accessible text version of the PowerPoint deck. Speaker prompts and timing are in the presentation and instructor guide.

## Slide 1: CME705 Machine Learning

**Week 4: Neural-network computation**

Guiding question: How do weighted transformations and nonlinear activations turn a batch of inputs into useful outputs?

## Slide 2: Today’s evidence

Each student will produce:

1. one correct neuron calculation;
2. a complete tensor-dimension trace;
3. an output-head decision;
4. a parameter count;
5. a numerical test of affine-layer collapse; and
6. one architecture row extracted from a paper.

## Slide 3: Opening calculation

Given $x=[2,-1,0.5]$, $w=[0.4,-0.8,1.2]$, and $b=-0.3$:

$$
z=x^Tw+b=1.9.
$$

- sigmoid output: approximately 0.870;
- ReLU output: 1.9.

The same pre-activation can have different meanings after activation.

## Slide 4: One neuron, two operations

1. Affine transformation: $z=x^Tw+b$
2. Activation: $a=\phi(z)$

Weights control input contributions. Bias shifts the transformation. The activation determines the response shape.

## Slide 5: Weights orient; bias shifts

- positive weight: larger input increases the pre-activation;
- negative weight: larger input decreases the pre-activation;
- zero weight: the input does not affect that unit;
- bias: changes the output even when all inputs are zero.

A dense layer gives every output unit its own weight vector and bias.

## Slide 6: From one vector to a batch

Rows are observations and columns are features.

$$
X:(n,d),\quad W:(d,h),\quad b:(h,)
$$

$$
Z=XW+b:(n,h)
$$

NumPy broadcasts the bias across rows.

## Slide 7: Dimension logic is part of the model

For four observations, three input features, and four hidden units:

- $X:(4,3)$
- $W_1:(3,4)$
- $b_1:(4,)$
- $Z_1:(4,4)$

The inner dimensions must match. The remaining dimensions define the result.

## Slide 8: Activation functions

| Function | Range | Common role | Main caution |
| --- | --- | --- | --- |
| Identity | all real values | regression output | no added nonlinearity |
| Sigmoid | 0 to 1 | binary or multilabel output | saturates |
| Tanh | -1 to 1 | bounded hidden state | saturates |
| ReLU | 0 to infinity | hidden layer | inactive negative region |

## Slide 9: Activation response shapes

Identity preserves magnitude. Sigmoid and tanh compress extremes. ReLU removes negative responses and stays linear for positive values.

The plot shows behavior, not a universal ranking.

## Slide 10: Nonlinearity changes representation

A flat decision boundary cannot separate every arrangement in the original feature space. A nonlinear hidden transformation can create a representation where a later boundary succeeds.

The usefulness of that representation still requires evaluation on unseen data.

## Slide 11: Two affine layers collapse to one

$$
Y=(XW_1+b_1)W_2+b_2
$$

$$
Y=X(W_1W_2)+(b_1W_2+b_2)
$$

Define $W_*=W_1W_2$ and $b_*=b_1W_2+b_2$. Without an intervening nonlinearity, depth adds no nonlinear representational power.

## Slide 12: Two-layer forward pass

$$
Z_1=XW_1+b_1
$$

$$
A_1=\operatorname{ReLU}(Z_1)
$$

$$
Z_2=A_1W_2+b_2
$$

$$
P=\operatorname{softmax}(Z_2)
$$

Every operation changes or preserves a documented shape.

## Slide 13: Logit, probability, prediction

- logit: raw output before sigmoid or softmax;
- probability: transformed output with a probabilistic interpretation;
- prediction: decision after an argmax or selected threshold.

These are different objects and should be reported precisely.

## Slide 14: Match the output head to the task

| Task | Output | Transformation |
| --- | --- | --- |
| Regression | one or more values | identity |
| Binary classification | one logit | sigmoid |
| Single-label multiclass | one logit per class | softmax |
| Multilabel classification | one logit per label | independent sigmoids |

## Slide 15: Stable softmax

$$
p_k=\frac{e^{z_k-\max(z)}}{\sum_j e^{z_j-\max(z)}}
$$

Subtracting the largest logit prevents unnecessary overflow and leaves the probability ratios unchanged.

Check that every value is finite, nonnegative, and that each row sums to one.

## Slide 16: Dimension-tracing activity

For $X:(4,3)$, $W_1:(3,4)$, $b_1:(4,)$, $W_2:(4,2)$, and $b_2:(2,)$, determine the shapes of:

- $Z_1$;
- $A_1$;
- logits; and
- probabilities.

Explain each result through the matrix operation.

## Slide 17: Count parameters before adding capacity

For $d\rightarrow h\rightarrow k$:

$$
(dh+h)+(hk+k)
$$

For $3\rightarrow4\rightarrow2$:

$$
(12+4)+(8+2)=26.
$$

More parameters change capacity, compute, optimization, and evidence needs.

## Slide 18: NumPy forward-pass lab

Predict, run, and record:

- every tensor shape;
- probability row sums;
- predicted classes;
- parameter count; and
- maximum difference between stacked and collapsed affine layers.

Run python labs/week04_forward_pass_numpy.py.

## Slide 19: What the lab establishes

The lab establishes correct shape flow, bias broadcasting, normalized softmax output, parameter counting, and affine-layer collapse for fixed values.

It does not establish learning, accuracy, calibration, generalization, or a benefit from depth.

## Slide 20: Extract architecture evidence

From one paper, record:

- input representation;
- layer sequence, widths, and activations;
- output head and training objective;
- data and split;
- metric and baseline; and
- one limitation.

Report the paper's facts separately from your interpretation.

## Slide 21: Literature comparison and exit

Compare at least five credible sources. State which results are directly comparable, identify the simplest credible baseline, and name one unresolved question your project can investigate.

Exit: one valid matrix expression, why nonlinearity matters, the likely project output head, and one missing architecture fact.

Next: gradient-based optimization.
