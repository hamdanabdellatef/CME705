# Week 6 notes: Backpropagation and losses

## 1. Backpropagation supplies the gradients

Week 5 used

$$
\theta_{t+1}=\theta_t-\eta\nabla J(\theta_t).
$$

For linear regression, the gradient followed directly from one affine prediction. A multilayer network contains nested transformations. Backpropagation evaluates the chain rule efficiently so every parameter receives a gradient.

Backpropagation computes gradients. Gradient descent or another optimizer uses those gradients to update parameters.

## 2. Hidden units have no supplied targets

A supervised dataset provides a target for the network output. It does not provide a desired value for each hidden unit.

The hidden-layer question is therefore:

> How would a small change in this hidden value change the final objective?

The chain rule answers this question by combining downstream sensitivity with the local derivative of the hidden transformation.

## 3. Forward notation

For a batch of $m$ observations with $d$ features, a hidden width $h$, and one binary output:

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

Shapes are:

| Quantity | Shape |
| --- | --- |
| $X$ | $(m,d)$ |
| $W_1$ | $(d,h)$ |
| $b_1$ | $(1,h)$ |
| $Z_1,H$ | $(m,h)$ |
| $W_2$ | $(h,1)$ |
| $b_2$ | $(1,1)$ |
| $Z_2,P,Y$ | $(m,1)$ |

The forward pass should cache $X$, $Z_1$, $H$, $Z_2$, and $P$. The backward pass will reuse them.

## 4. Binary cross-entropy

For a binary target $y_i\in\{0,1\}$ and predicted probability $p_i$:

$$
J=-\frac{1}{m}\sum_{i=1}^{m}
\left[
y_i\log p_i+(1-y_i)\log(1-p_i)
\right].
$$

The output transformation and objective must describe the same prediction problem. A single sigmoid output represents a binary probability.

For numerical stability, software can compute binary cross-entropy directly from the logit $z$:

$$
\ell(y,z)=\log(1+e^z)-yz.
$$

The Week 6 lab uses NumPy's stable $\operatorname{logaddexp}$ form rather than taking the logarithm of a rounded probability.

## 5. Sigmoid and binary cross-entropy simplify

For one observation,

$$
p=\sigma(z),
$$

and

$$
\frac{\partial J}{\partial z}
=
\frac{\partial J}{\partial p}
\frac{\partial p}{\partial z}.
$$

After simplification for sigmoid with binary cross-entropy,

$$
\frac{\partial J}{\partial z}=p-y.
$$

For a mean over $m$ observations, define the output error signal

$$
G_{Z_2}=\frac{P-Y}{m}.
$$

This compact result comes from a specific output-loss pairing. It should not be copied unchanged to every activation and objective.

## 6. Output-layer parameter gradients

Because

$$
Z_2=HW_2+b_2,
$$

the gradients are

$$
G_{W_2}=H^T G_{Z_2}
$$

and

$$
G_{b_2}=\sum_{i=1}^{m}G_{Z_2}^{(i)}.
$$

Shape check:

- $H^T$ has shape $(h,m)$;
- $G_{Z_2}$ has shape $(m,1)$;
- $G_{W_2}$ has shape $(h,1)$, matching $W_2$; and
- $G_{b_2}$ has shape $(1,1)$, matching $b_2$.

The bias gradient sums across observations because the same bias affects every row.

## 7. Move sensitivity into the hidden layer

The hidden outputs affect the objective through $Z_2$. Therefore,

$$
G_H=G_{Z_2}W_2^T.
$$

This multiplication distributes downstream sensitivity through the output weights. It is not yet the gradient with respect to the hidden pre-activation.

For tanh,

$$
\frac{d}{dz}\tanh(z)=1-\tanh^2(z).
$$

Because $H=\tanh(Z_1)$,

$$
G_{Z_1}=G_H\odot(1-H^2),
$$

where $\odot$ denotes elementwise multiplication.

For one hidden unit $j$ and one output,

$$
\frac{\partial J}{\partial z_{1j}}
=
\frac{\partial J}{\partial z_2}
\,w_{2j}\,
(1-h_j^2).
$$

This scalar path exposes the three chain-rule factors: downstream output sensitivity, connecting weight, and local activation derivative.

## 8. Hidden-layer parameter gradients

Because

$$
Z_1=XW_1+b_1,
$$

the remaining gradients are

$$
G_{W_1}=X^T G_{Z_1}
$$

and

$$
G_{b_1}=\sum_{i=1}^{m}G_{Z_1}^{(i)}.
$$

Shape check:

- $X^T$ has shape $(d,m)$;
- $G_{Z_1}$ has shape $(m,h)$;
- $G_{W_1}$ has shape $(d,h)$, matching $W_1$; and
- $G_{b_1}$ has shape $(1,h)$, matching $b_1$.

## 9. The complete backward sequence

Starting from cached forward values:

1. $G_{Z_2}=(P-Y)/m$
2. $G_{W_2}=H^TG_{Z_2}$
3. $G_{b_2}=\operatorname{sum}(G_{Z_2},\text{rows})$
4. $G_H=G_{Z_2}W_2^T$
5. $G_{Z_1}=G_H\odot(1-H^2)$
6. $G_{W_1}=X^TG_{Z_1}$
7. $G_{b_1}=\operatorname{sum}(G_{Z_1},\text{rows})$

Backward order reverses the forward dependencies. The cached values prevent unnecessary recomputation.

## 10. Update after computing all gradients

Use the Week 5 rule for each parameter:

$$
W_1\leftarrow W_1-\eta G_{W_1},
$$

$$
b_1\leftarrow b_1-\eta G_{b_1},
$$

$$
W_2\leftarrow W_2-\eta G_{W_2},
$$

$$
b_2\leftarrow b_2-\eta G_{b_2}.
$$

Compute all gradients from one consistent parameter state before applying any update. Updating $W_2$ early and then using the changed matrix to calculate $G_H$ mixes two states and no longer implements the derived backward pass.

## 11. Gradient checking

For parameter component $\theta_j$, a centered finite difference is

$$
G_j^{\text{num}}
\approx
\frac{J(\theta_j+\varepsilon)-J(\theta_j-\varepsilon)}
{2\varepsilon}.
$$

Compare it with the analytical value $G_j^{\text{ana}}$. A useful relative error is

$$
r_j=
\frac{|G_j^{\text{ana}}-G_j^{\text{num}}|}
{\max(\delta,|G_j^{\text{ana}}|+|G_j^{\text{num}}|)}.
$$

Use a small deterministic problem, double precision, and a suitable $\varepsilon$. Disable randomness that changes between the plus and minus evaluations.

A low error supports the gradient implementation at the checked values. It does not validate the data split, metric, optimizer schedule, or scientific claim.

## 12. Common implementation failures

| Failure | Consequence |
| --- | --- |
| Wrong transpose | Gradient shape mismatch or incorrect parameter responsibility |
| Missing tanh derivative | Hidden gradient ignores the local transformation |
| Summing over the wrong axis | Bias gradient has the wrong meaning or shape |
| Averaging more than once | Gradients become unnecessarily small |
| Updating before the backward pass is complete | Gradients mix different parameter states |
| Taking $\log(0)$ | Loss becomes nonfinite |
| Reusing a stale cache | Backward values do not match the current forward pass |
| Leaving randomness active in a gradient check | Numerical differences measure two different computations |

## 13. XOR exposes a representation limit

XOR assigns positive targets to $(0,1)$ and $(1,0)$, and negative targets to $(0,0)$ and $(1,1)$. One affine boundary cannot separate those two groups.

A single sigmoid unit therefore cannot represent XOR. In the teaching lab, symmetric zero initialization leaves its predictions at $0.5$, its binary cross-entropy at approximately $\log 2=0.6931$, and its accuracy at $0.5$.

A hidden nonlinear transformation can create a representation that a later affine output separates. The two-layer tanh network fits all four observations.

This is a representation and training demonstration. Four training observations do not provide independent generalization evidence.

## 14. What the Week 6 lab checks

The script reports:

- forward shapes $(4,2)\rightarrow(4,4)\rightarrow(4,1)$;
- relative gradient-check errors for $W_1$, $b_1$, $W_2$, and $b_2$;
- the linear model's $0.5$ predictions and $0.6931$ loss;
- the MLP objective decreasing from approximately $0.7062$ to $0.0006$;
- predicted classes $[0,1,1,0]$; and
- training accuracy $1.0$.

The largest gradient-check relative error is approximately $3\times10^{-9}$ for the fixed seed. Small differences across NumPy platforms are acceptable when they remain below the declared tolerance.

## 15. From plan to reproducible baseline

The Week 6 project milestone records the actual run:

- repository commit or archive checksum;
- Python, NumPy, PyTorch, operating system, and compute device;
- dataset version or checksum and split identifiers;
- actual preprocessing, objective, optimizer, learning rate, batch size, seed, and stopping rule;
- one command from the repository root;
- training, validation, and test results with separate roles;
- at least three errors or one meaningful subgroup;
- deviations from the Week 5 plan; and
- the supported claim, limitation, and next controlled experiment.

A result can be weak or failed and still be reproducible. Hidden deviations and unrecorded test use make it difficult to interpret.

## Review questions

1. What does backpropagation compute, and what performs the update?
2. Why does a hidden unit not need its own supplied target?
3. Why does sigmoid with binary cross-entropy give $G_{Z_2}=(P-Y)/m$?
4. Why is $W_2^T$ used when sensitivity moves into the hidden layer?
5. Which cached value is needed for the tanh derivative?
6. Why must all gradients be computed before any parameter update?
7. What does a low finite-difference error establish?
8. Why is perfect XOR training accuracy not generalization evidence?
