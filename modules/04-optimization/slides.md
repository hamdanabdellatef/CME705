# Week 5 slide source: Gradient-based optimization

This file is the accessible text version of the PowerPoint deck. Speaker prompts and timing are in the presentation and instructor guide.

## Slide 1: CME705 Machine Learning

**Week 5: Gradient-based optimization**

Guiding question: How do gradients, learning rates, and update schedules change model parameters and the evidence used to judge training?

## Slide 2: Today’s evidence

Each student will produce:

1. one correct gradient update;
2. a linear-regression gradient derivation;
3. exact epoch and update counts;
4. a learning-rate diagnosis;
5. a validation-controlled lab record; and
6. a baseline experiment plan.

## Slide 3: From computation to learning

Week 4 evaluated a forward pass with fixed parameters.

Week 5 adds:

- an objective $J(\theta)$;
- a gradient $\nabla J(\theta)$; and
- an update rule for $\theta$.

Multilayer backpropagation begins in Week 6.

## Slide 4: Objective and metric answer different questions

- Training objective: supplies the gradient used to update parameters.
- Evaluation metric: measures behavior relevant to the research question.
- Validation evidence: selects a planned configuration.
- Test evidence: evaluates the final selected procedure.

The objective and metric may differ.

## Slide 5: A one-parameter loss surface

For

$$
J(w)=(w-3)^2,
$$

the minimum is at $w=3$.

The derivative

$$
\frac{dJ}{dw}=2(w-3)
$$

gives the local slope at the current value.

## Slide 6: The negative gradient gives a descent direction

- Negative derivative: move toward larger $w$.
- Positive derivative: move toward smaller $w$.
- Zero derivative: no first-order direction at that point.

A local descent direction does not guarantee the global solution of every objective.

## Slide 7: One update by hand

With $w_0=0$ and $\eta=0.1$:

$$
\frac{dJ}{dw}=-6,
$$

$$
w_1=0-0.1(-6)=0.6.
$$

The objective decreases from $9$ to $5.76$.

## Slide 8: The gradient-descent rule

$$
\theta_{t+1}=\theta_t-\eta\nabla J(\theta_t).
$$

- $\theta_t$: current parameters;
- $\nabla J(\theta_t)$: local sensitivity;
- $\eta$: learning rate; and
- $\theta_{t+1}$: updated parameters.

## Slide 9: Linear regression objective

$$
\widehat{y}=Xw+b,
$$

$$
e=\widehat{y}-y,
$$

$$
J(w,b)=\frac{1}{m}e^Te.
$$

Rows are observations and columns are features.

## Slide 10: Mean-squared-error gradients

$$
\nabla_wJ=\frac{2}{m}X^Te,
$$

$$
\frac{\partial J}{\partial b}=\frac{2}{m}\sum_{i=1}^{m}e_i.
$$

The weight gradient has shape $(d,)$ and the bias gradient is one scalar.

## Slide 11: Three update schedules

| Schedule | Examples per update | Updates per epoch | Main behavior |
| --- | ---: | ---: | --- |
| Full batch | $n$ | 1 | stable, expensive update |
| Stochastic | 1 | $n$ | noisy, many updates |
| Mini-batch | $B$ | $\lceil n/B\rceil$ | vectorized compromise |

The rule stays the same; the gradient estimate changes.

## Slide 12: Epoch, batch, and update

For 220 examples and batch size 32:

$$
\left\lceil\frac{220}{32}\right\rceil=7
$$

updates per epoch.

Six batches contain 32 examples. The final batch contains 28.

## Slide 13: Shuffle once and preserve coverage

A correct epoch permutes the training indices, divides the order into batches, and uses every example exactly once.

Checks:

- no duplicate indices;
- no omitted indices;
- a smaller final batch is allowed; and
- a fixed random seed reproduces the order.

## Slide 14: Learning-rate behavior

- $0.001$: finite but slow within the fixed budget;
- $0.010$ to $0.200$: rapid reduction on the teaching problem;
- $1.200$: unstable and divergent.

A complete diagnosis uses the curve, update count, and finite-value checks.

## Slide 15: Feature scaling changes the geometry

Standardization uses training-only statistics:

$$
x'_{ij}=\frac{x_{ij}-\mu_j}{s_j}.
$$

Scaling can make one learning rate useful across parameter directions. Apply the fitted training transformation unchanged to validation and test data.

## Slide 16: Training, validation, and test roles

| Split | Purpose |
| --- | --- |
| Training | estimate parameters |
| Validation | select learning rate and stopping decisions |
| Test | evaluate the final selected procedure |

Repeated test-guided tuning removes the independence of the test result.

## Slide 17: Stopping and diagnostics

Record:

- initial and final objective;
- full loss history;
- update and example counts;
- validation history or selection score;
- finite-value failures; and
- the stopping reason.

A flat curve can have several causes and needs diagnosis.

## Slide 18: NumPy lab

Run:

~~~bash
python labs/week05_gradient_descent.py
~~~

The lab compares update schedules for 40 epochs, evaluates five planned learning rates with validation MSE, and reports the test result once for the selected rate.

## Slide 19: What the lab establishes

Supported:

- analytical gradients agree with finite differences;
- stable settings reduce the training objective;
- every epoch covers all training examples; and
- validation selects among declared candidates.

Still untested:

- transfer to another dataset;
- universal optimizer ranking; and
- project generalization.

## Slide 20: Baseline experiment plan

Freeze before execution:

- baseline and literature support;
- dataset version and split unit;
- preprocessing;
- objective and validation metric;
- learning-rate candidates, batch size, and stopping rule;
- seeds, command, expected outputs, and compute limit; and
- interpretation rules.

## Slide 21: Exit and next step

Explain:

1. why the negative gradient is used;
2. how an epoch differs from an update;
3. what validation selects; and
4. when test data are used.

Next: chain rule and multilayer backpropagation.
