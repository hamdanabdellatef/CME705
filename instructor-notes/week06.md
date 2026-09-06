# Week 6 instructor guide: Backpropagation and losses

## Purpose

Week 6 provides the gradients that the Week 5 update rule requires for a multilayer network. Students trace one scalar chain-rule path, then express the same dependencies as batched matrix operations. They check all parameter gradients numerically before using them to fit XOR.

The historical material spread single-layer limitations, MATLAB delta-rule code, backpropagation, cross-entropy, momentum, and regularization across two weeks. This version keeps the hidden-responsibility question, XOR, and output-loss relationship. It replaces the code with explicit NumPy and defers momentum, regularization, and multiclass learning so the chain rule remains visible.

## Before class

- Run python labs/week06_mlp_numpy.py.
- Distribute [the Week 6 worksheet](../modules/05-backpropagation/worksheet.md).
- Ask students to bring the actual repository state for their planned baseline.
- Prepare a board or shared graph with forward arrows in one color and backward arrows in another.
- Keep the lab output hidden until students record their XOR predictions.

## Learning evidence

Inspect:

1. the distinction between gradient computation and parameter update;
2. the forward cache and its shapes;
3. the scalar hidden chain-rule path;
4. the complete matrix backward sequence;
5. the finite-difference tolerance and result;
6. the linear-versus-nonlinear XOR comparison; and
7. a baseline record that another student can run from the repository root.

## Three-hour sequence

| Time | Activity | Instructor action | Student evidence |
| ---: | --- | --- | --- |
| 0–10 min | Week 5 bridge | Show the update and name the missing gradients | Backpropagation versus optimizer statement |
| 10–25 min | Hidden responsibility | Ask how a hidden change affects the objective | Sensitivity question |
| 25–45 min | Forward graph and cache | Trace values and shapes | Completed cache table |
| 45–65 min | Binary cross-entropy | Pair target, output, and objective | Output-loss explanation |
| 65–85 min | Output gradients | Derive $G_{Z_2}$, $G_{W_2}$, and $G_{b_2}$ | Equations with shapes |
| 85–95 min | Break |  |  |
| 95–120 min | Hidden chain rule | Move sensitivity through $W_2^T$ and tanh | Scalar path and matrix equations |
| 120–135 min | Full backward sequence | Order seven operations | Ordered dependency trace |
| 135–148 min | Gradient check | Contrast analytical and numerical values | Tolerance and claim boundary |
| 148–170 min | NumPy XOR lab | Require predictions before execution | Gradient and model comparison record |
| 170–177 min | Reproducible baseline | Inspect one command and result file | One complete milestone row |
| 177–180 min | Exit ticket | Collect four concise statements | Completed exit ticket |

## Facilitation notes

### Backpropagation and optimization

Write

$$
\theta_{t+1}=\theta_t-\eta\nabla J(\theta_t).
$$

Circle $\nabla J$. Backpropagation calculates this quantity through the network. The optimizer uses it. This distinction becomes useful when PyTorch later computes gradients automatically while an optimizer object updates parameters.

### Hidden responsibility

Avoid describing a hidden unit as receiving a target. It receives a gradient that states how changing its pre-activation would change the final objective.

Use the question:

> If this hidden value increased slightly while everything else remained locally fixed, how would the final objective change?

### Forward graph and cache

For the teaching network:

- $X:(4,2)$;
- $W_1:(2,4)$ and $b_1:(1,4)$;
- $Z_1,H:(4,4)$;
- $W_2:(4,1)$ and $b_2:(1,1)$; and
- $Z_2,P,Y:(4,1)$.

Cache values from the same parameter state used for backward computation.

### Binary cross-entropy

For one sigmoid output, binary cross-entropy represents the two possible target outcomes. Show the stable logit form used by the lab:

$$
\ell(y,z)=\log(1+e^z)-yz.
$$

Do not present the phrase “cross-entropy is always faster” as a universal empirical claim. The useful derivative simplification follows from this output-loss pairing.

### Output error signal

Starting with sigmoid and binary cross-entropy, derive or state

$$
G_{Z_2}=\frac{P-Y}{m}.
$$

The averaging occurs here. Do not average the downstream parameter gradients a second time.

### Output gradients

Expected equations:

$$
G_{W_2}=H^TG_{Z_2},
$$

$$
G_{b_2}=\operatorname{sum}(G_{Z_2},\text{rows}).
$$

Ask why a transpose appears. The observations dimension is contracted so the remaining dimensions match $W_2$.

### Hidden chain rule

For one hidden unit:

$$
\frac{\partial J}{\partial z_{1j}}
=
\frac{\partial J}{\partial z_2}
\,w_{2j}\,
(1-h_j^2).
$$

Name the factors:

1. downstream output sensitivity;
2. the connecting output weight; and
3. the local tanh derivative.

Then vectorize:

$$
G_H=G_{Z_2}W_2^T,
$$

$$
G_{Z_1}=G_H\odot(1-H^2).
$$

### Hidden parameter gradients

Expected equations:

$$
G_{W_1}=X^TG_{Z_1},
$$

$$
G_{b_1}=\operatorname{sum}(G_{Z_1},\text{rows}).
$$

Require students to produce $(2,4)$ and $(1,4)$ for the teaching network.

### Complete order

The correct order is:

1. $G_{Z_2}$
2. $G_{W_2}$
3. $G_{b_2}$
4. $G_H$
5. $G_{Z_1}$
6. $G_{W_1}$
7. $G_{b_1}$

The output-parameter gradients and $G_H$ all use the current forward cache and current $W_2$.

### Update ordering

A safe implementation performs:

1. forward pass and cache;
2. every backward gradient; and
3. every parameter update.

If $W_2$ changes before $G_H$ is calculated, the hidden gradient no longer corresponds to the forward pass being differentiated.

### Gradient checking

The lab perturbs every component of $W_1$, $b_1$, $W_2$, and $b_2$. For the fixed initial parameters, the maximum relative error is approximately $3.005\times10^{-9}$. The teaching tolerance is $10^{-6}$.

A passing result supports the derivative code at these parameter values. It does not validate the optimizer schedule, split, metric, or generalization claim.

### XOR

A single affine boundary cannot isolate the diagonal pair of positive XOR points from the negative pair. With symmetric zero initialization, the teaching linear model stays at probability $0.5$ for every row.

The two-layer network uses four tanh hidden units. Its learned hidden representation makes the output classes separable.

### Week 6 lab

Expected deterministic output:

| Evidence | Value |
| --- | --- |
| Hidden shape | $(4,4)$ |
| Prediction shape | $(4,1)$ |
| Maximum relative gradient error | approximately $3.005\times10^{-9}$ |
| Linear probabilities | $[0.5,0.5,0.5,0.5]$ |
| Linear binary cross-entropy | $0.6931$ |
| Linear accuracy | $0.500$ |
| MLP loss | $0.7062\rightarrow0.0006$ |
| MLP probabilities | approximately $[0.0001,0.9993,0.9991,0.0009]$ |
| MLP classes | $[0,1,1,0]$ |
| MLP accuracy | $1.000$ |

All four XOR rows are used for training. The final row is not a test result.

### Reproducible baseline milestone

Require:

- repository commit or archive checksum;
- actual environment and one root-level command;
- immutable dataset identity and explicit split files or identifiers;
- planned versus actual configuration;
- separate training, validation, and test results;
- honest disclosure of test use;
- at least three errors or one meaningful subgroup; and
- the next controlled experiment.

A classmate should attempt the command without verbal guidance. The correction belongs in the submitted record.

## Common misconceptions

| Misconception | Response |
| --- | --- |
| Backpropagation changes the weights | It computes gradients; the optimizer applies the change |
| Hidden units need hidden labels | The chain rule assigns sensitivity from the final objective |
| The output error is copied unchanged to the hidden layer | It is weighted by $W_2^T$ and multiplied by the local activation derivative |
| Shape agreement proves the derivative is correct | It is necessary; numerical checking supplies additional evidence |
| Bias gradients use a feature transpose | Bias is shared across observations, so its gradient sums over rows |
| The mean can be applied at every gradient line | Average once according to the defined objective |
| Parameters can update as soon as one gradient is available | All gradients should describe the same parameter state |
| A tiny gradient-check error validates the entire experiment | It validates selected derivative computations |
| Perfect XOR accuracy proves generalization | Every XOR observation was used for training |
| A deeper network automatically improves a project | Architecture changes need controlled validation evidence |

## Formative feedback language

- “Which operation is being differentiated?”
- “What downstream quantity changes if this hidden value changes?”
- “Write the local derivative beside the graph edge.”
- “Does the gradient shape match the parameter?”
- “Which axis represents observations?”
- “Has the mean already been applied?”
- “Did any parameter change before the backward pass finished?”
- “What exactly did the numerical check validate?”
- “Is this training or independent evaluation evidence?”

## Adaptations

### Ninety-minute class

Use slides 1–4, 6–16, and 18–21. Complete the output-loss simplification, one scalar hidden path, the matrix backward sequence, gradient-check interpretation, and the XOR lab. Assign the failure-pattern table and baseline reproduction check afterward.

### Online delivery

Use a shared computational graph. Assign each breakout group one gradient and require its inputs, local derivative, and output shape. Run the deterministic lab centrally if environments differ.

### Limited calculus preparation

Start with verbal sensitivity and one scalar path. Use color to match each backward factor to a forward edge. Move to matrix form only after students can explain the three hidden-layer factors.

## After class

- identify students who copy an output delta directly into the hidden layer;
- check for transpose and bias-axis errors;
- review baseline records for undocumented deviations or test reuse;
- require reproduction corrections before Week 7 evaluation review; and
- connect binary output-loss pairing to multiclass softmax and cross-entropy next week.
