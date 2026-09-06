# Week 4 notes: Neural-network computation

## 1. A neuron performs an affine transformation

For an input vector $x$, weight vector $w$, and scalar bias $b$, a neuron first computes

$$
z=x^T w+b.
$$

The value $z$ is the **pre-activation** or **logit**, depending on its role. An activation function $\phi$ then produces

$$
a=\phi(z).
$$

For $x=[2,-1,0.5]$, $w=[0.4,-0.8,1.2]$, and $b=-0.3$:

$$
z=(2)(0.4)+(-1)(-0.8)+(0.5)(1.2)-0.3=1.9.
$$

The sigmoid output is approximately $0.870$, while the ReLU output is $1.9$.

## 2. Weights and bias have different roles

A weight determines how strongly and in which direction one input contributes. Changing a weight changes the orientation of a decision boundary or response surface.

The bias shifts the transformation. Without a bias, a linear boundary must pass through the origin. In a dense layer, every output unit has its own bias.

Weights and biases are parameters because training will estimate them. Week 4 uses fixed values so the computation remains visible.

## 3. A layer computes several neurons together

For one input with $d$ features and a layer with $h$ units:

- $x$ has shape $(d,)$;
- $W$ has shape $(d,h)$;
- $b$ has shape $(h,)$; and
- $z=xW+b$ has shape $(h,)$.

The columns of $W$ contain the weights feeding different output units.

## 4. Batch computation

Place $n$ observations in the rows of $X$:

$$
X\in\mathbb{R}^{n\times d},\quad
W\in\mathbb{R}^{d\times h},\quad
b\in\mathbb{R}^{h}.
$$

Then

$$
Z=XW+b
$$

has shape $(n,h)$. NumPy broadcasts the bias across the $n$ rows. Broadcasting is valid because the final dimension of $Z$ and the length of $b$ are both $h$.

Keep one orientation throughout a project. This course uses rows for observations and columns for features or units.

## 5. Activation functions

### Identity

$$
\phi(z)=z
$$

The identity function preserves any real value. It is commonly used for an unconstrained regression output.

### Sigmoid

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

Sigmoid maps a real number to $(0,1)$. It is useful for a binary probability or for each independent label in multilabel classification. Large positive or negative inputs produce outputs near 1 or 0, where derivatives become small.

### Hyperbolic tangent

$$
\tanh(z)\in(-1,1)
$$

Tanh is zero-centered and saturates at both extremes. It remains common in recurrent mechanisms and in historical neural-network formulations.

### ReLU

$$
\operatorname{ReLU}(z)=\max(0,z)
$$

ReLU is simple and does not saturate for positive inputs. Negative pre-activations become zero. A unit that remains negative for all relevant inputs can stop contributing.

No activation is universally best. The choice depends on the layer's role, optimization behavior, data scale, and evidence from controlled experiments.

## 6. Why nonlinearity is necessary

Consider two affine layers with no activation between them:

$$
H=XW_1+b_1,
$$

$$
Y=HW_2+b_2.
$$

Substitute the first expression into the second:

$$
Y=(XW_1+b_1)W_2+b_2
=X(W_1W_2)+(b_1W_2+b_2).
$$

Define $W_*=W_1W_2$ and $b_*=b_1W_2+b_2$. The two layers are exactly one affine layer:

$$
Y=XW_*+b_*.
$$

Adding more affine layers without nonlinear activations does not add nonlinear representational power. A nonlinear activation prevents this algebraic collapse.

## 7. A two-layer forward pass

For hidden activation $\phi$ and output transformation $g$:

$$
Z_1=XW_1+b_1,
$$

$$
A_1=\phi(Z_1),
$$

$$
Z_2=A_1W_2+b_2,
$$

$$
\widehat{Y}=g(Z_2).
$$

The forward pass evaluates these expressions in order. It does not update a parameter.

For the Week 4 lab:

- $X$: $(4,3)$
- $W_1$: $(3,4)$
- $b_1$: $(4,)$
- $Z_1$ and $A_1$: $(4,4)$
- $W_2$: $(4,2)$
- $b_2$: $(2,)$
- logits and probabilities: $(4,2)$

## 8. Match the output head to the task

| Task | Output units | Transformation | Interpretation |
| --- | ---: | --- | --- |
| Unconstrained regression | One or more | Identity | Predicted continuous value |
| Binary classification | One | Sigmoid | Probability of the positive class |
| Single-label multiclass classification | Number of classes | Softmax | Mutually exclusive class probabilities |
| Multilabel classification | Number of labels | Independent sigmoids | Probability for each nonexclusive label |

The output representation and loss must agree. A two-column softmax and two independent sigmoids answer different questions.

## 9. Stable softmax

For logits $z_1,\ldots,z_K$, softmax gives

$$
p_k=\frac{e^{z_k}}{\sum_{j=1}^{K}e^{z_j}}.
$$

The probabilities are nonnegative and sum to one. For numerical stability, subtract the largest logit before exponentiation:

$$
p_k=\frac{e^{z_k-\max(z)}}{\sum_j e^{z_j-\max(z)}}.
$$

Subtracting the same constant from every logit does not change the ratio.

## 10. Logits, probabilities, and predictions

A **logit** is the raw output before sigmoid or softmax. A **probability** is a normalized or bounded output interpreted under a model. A **prediction** applies a decision rule such as an argmax or threshold.

Do not use these terms interchangeably. A probability can be poorly calibrated, and a threshold selected after viewing test labels invalidates the final evaluation.

## 11. Parameter counting

A dense layer from $d$ inputs to $h$ units has $d\times h+h$ parameters. The first term counts weights and the second counts biases.

A two-layer network with widths $d\rightarrow h\rightarrow k$ has

$$
(dh+h)+(hk+k)
$$

parameters. For $3\rightarrow4\rightarrow2$, the count is

$$
(3\times4+4)+(4\times2+2)=26.
$$

## 12. Width, depth, and evidence

Width is the number of units in a layer. Depth is the number of successive parameterized transformations. Greater width or depth can increase the functions a network can represent, but also changes parameter count, computation, optimization, and overfitting risk.

A larger model is a hypothesis, not evidence of improvement. Compare architectures using the same split, preprocessing, metrics, tuning budget, and final-test policy.

## 13. Reading architectures in papers

When extracting a method from a paper, record:

- input representation and shape;
- layer types and order;
- hidden widths or channel counts;
- activations;
- output head;
- parameter count when reported or reproducible;
- training objective and optimizer;
- data split and evaluation metric; and
- the exact baseline used for comparison.

Two reported scores are directly comparable only when the task, data, split, target, metric, and evaluation procedure align closely enough. Architecture names alone do not make results comparable.

## 14. What the Week 4 lab establishes

The lab verifies that matrix dimensions determine whether a layer is valid, a bias is added to every row, softmax rows sum to one, a parameter count follows from shapes, and two affine layers collapse numerically when no nonlinear activation separates them.

The lab uses fixed teaching parameters and generated inputs. It does not establish successful training, useful accuracy, calibration, or generalization.

## Review questions

1. What changes when a bias changes but all weights remain fixed?
2. Why does $(n,d)(d,h)$ produce $(n,h)$?
3. Why can two affine layers be replaced by one?
4. Which output head suits single-label classification with five classes?
5. How is a logit different from a probability and a prediction?
6. How many parameters are in a $10\rightarrow8\rightarrow3$ dense network?
7. Which facts must align before two papers' accuracy values are directly comparable?
