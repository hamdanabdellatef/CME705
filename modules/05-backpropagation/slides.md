# Week 6 slide source: Backpropagation and losses

This file is the accessible text version of the PowerPoint deck. Speaker prompts and timing are in the presentation and instructor guide.

## Slide 1: CME705 Machine Learning

**Week 6: Backpropagation and losses**

Guiding question: How does the chain rule assign responsibility for a network's output error to every parameter?

## Slide 2: Today’s evidence

Each student will produce:

1. one scalar chain-rule path;
2. a complete backward-pass shape trace;
3. an output-loss explanation;
4. a numerical gradient check;
5. a linear-versus-nonlinear XOR comparison; and
6. a reproducible baseline record.

## Slide 3: Backpropagation supplies the missing gradients

Week 5 gave the update:

$$
\theta_{t+1}=\theta_t-\eta\nabla J(\theta_t).
$$

Week 6 computes $\nabla J$ through nested network transformations.

Backpropagation computes gradients. The optimizer applies updates.

## Slide 4: Hidden units have no supplied targets

The dataset gives a target for the network output.

For each hidden value, ask:

> How would a small change here change the final objective?

The chain rule combines downstream sensitivity with the local derivative.

## Slide 5: Two directions through one graph

Forward:

$$
X\rightarrow Z_1\rightarrow H\rightarrow Z_2\rightarrow P\rightarrow J.
$$

Backward:

$$
G_{Z_2}\rightarrow G_H\rightarrow G_{Z_1}.
$$

Parameter gradients branch from the related pre-activation gradients.

## Slide 6: Cache the forward values

Forward equations:

$$
Z_1=XW_1+b_1,
$$

$$
H=\tanh(Z_1),
$$

$$
Z_2=HW_2+b_2,
$$

$$
P=\sigma(Z_2).
$$

Cache $X$, $Z_1$, $H$, $Z_2$, and $P$ for the backward pass.

## Slide 7: Binary cross-entropy matches a binary output

$$
J=-\frac{1}{m}\sum_i
\left[y_i\log p_i+(1-y_i)\log(1-p_i)\right].
$$

One sigmoid output represents a binary probability.

A stable implementation computes the loss from logits.

## Slide 8: Sigmoid and cross-entropy simplify

For the mean objective:

$$
G_{Z_2}=\frac{P-Y}{m}.
$$

This is the starting error signal for the backward pass.

The result belongs to this output-loss pairing.

## Slide 9: Output-layer gradients

$$
G_{W_2}=H^TG_{Z_2},
$$

$$
G_{b_2}=\operatorname{sum}(G_{Z_2},\text{rows}).
$$

The gradient shapes match $W_2:(h,1)$ and $b_2:(1,1)$.

## Slide 10: Sensitivity moves through the output weights

$$
G_H=G_{Z_2}W_2^T.
$$

The transpose maps one output sensitivity back to $h$ hidden values for every observation.

This is not yet the hidden pre-activation gradient.

## Slide 11: The tanh derivative is local

$$
\frac{d}{dz}\tanh(z)=1-\tanh^2(z).
$$

Because $H=\tanh(Z_1)$:

$$
G_{Z_1}=G_H\odot(1-H^2).
$$

The elementwise product applies the local activation derivative.

## Slide 12: Hidden-layer parameter gradients

$$
G_{W_1}=X^TG_{Z_1},
$$

$$
G_{b_1}=\operatorname{sum}(G_{Z_1},\text{rows}).
$$

The gradient shapes match $W_1:(d,h)$ and $b_1:(1,h)$.

## Slide 13: Complete backward sequence

1. $G_{Z_2}=(P-Y)/m$
2. $G_{W_2}=H^TG_{Z_2}$
3. $G_{b_2}=\operatorname{sum}(G_{Z_2},\text{rows})$
4. $G_H=G_{Z_2}W_2^T$
5. $G_{Z_1}=G_H\odot(1-H^2)$
6. $G_{W_1}=X^TG_{Z_1}$
7. $G_{b_1}=\operatorname{sum}(G_{Z_1},\text{rows})$

## Slide 14: Every gradient must match its parameter

| Parameter | Shape | Gradient |
| --- | --- | --- |
| $W_1$ | $(d,h)$ | $G_{W_1}$ |
| $b_1$ | $(1,h)$ | $G_{b_1}$ |
| $W_2$ | $(h,1)$ | $G_{W_2}$ |
| $b_2$ | $(1,1)$ | $G_{b_2}$ |

Shape agreement is necessary but not sufficient for correctness.

## Slide 15: Compute first; update second

Correct sequence:

1. run one forward pass;
2. compute every gradient from its cache; and
3. update all parameters.

Updating $W_2$ before calculating $G_H$ mixes two parameter states.

## Slide 16: Centered differences check the gradients

$$
G_j^{\text{num}}
\approx
\frac{J(\theta_j+\varepsilon)-J(\theta_j-\varepsilon)}
{2\varepsilon}.
$$

Compare every numerical component with its analytical counterpart.

A low error validates the checked derivative computation.

## Slide 17: Failure patterns reveal where to inspect

- Wrong transpose: shape or responsibility error.
- Missing tanh derivative: hidden local response ignored.
- Wrong bias axis: invalid aggregation.
- Double averaging: gradients too small.
- Early update: mixed parameter states.
- Unstable logarithm: nonfinite loss.
- Stale cache: forward and backward do not match.

## Slide 18: XOR separates representation from optimization

One affine-sigmoid unit predicts $0.5$ for all four observations under the symmetric teaching setup.

A nonlinear hidden layer creates a representation in which the output can separate the XOR classes.

Perfect fit on four training rows is not generalization evidence.

## Slide 19: NumPy lab results

Run:

~~~bash
python labs/week06_mlp_numpy.py
~~~

Expected evidence:

- largest gradient-check error near $3\times10^{-9}$;
- linear loss $0.6931$ and accuracy $0.5$;
- MLP loss approximately $0.7062\rightarrow0.0006$;
- predicted classes $[0,1,1,0]$; and
- MLP training accuracy $1.0$.

## Slide 20: Reproducible baseline

Record:

- repository state and one command;
- environment and compute;
- dataset version and split identities;
- planned and actual configuration;
- training, validation, and test roles;
- first error analysis;
- deviations from the plan; and
- claim, limitation, and next experiment.

## Slide 21: Exit and next step

Explain:

1. why hidden units need the chain rule;
2. why $W_2^T$ moves sensitivity backward;
3. what a low gradient-check error supports; and
4. why XOR training accuracy is not generalization evidence.

Next: multiclass softmax, cross-entropy, and consolidation.
