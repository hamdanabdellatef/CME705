# Week 5 notes: Gradient-based optimization

## 1. Learning requires an objective

Week 4 computed predictions using fixed parameters. Training adds an objective that measures how the current predictions differ from the desired targets.

For a model with parameters $\theta$, write the training objective as

$$
J(\theta).
$$

Optimization searches for parameter values that reduce this objective. An **evaluation metric** answers a scientific or application question, such as balanced accuracy, mean absolute error, or recall. The objective and metric can be different because the objective must also support useful parameter updates.

## 2. A derivative measures local sensitivity

For the one-parameter objective

$$
J(w)=(w-3)^2,
$$

the derivative is

$$
\frac{dJ}{dw}=2(w-3).
$$

At $w=0$, the derivative is $-6$. A small positive change in $w$ should therefore reduce the objective. At $w=5$, the derivative is $4$, so a small negative change should reduce the objective.

The derivative is local information. It describes the slope at the current parameter value; it does not reveal the complete path or guarantee a global solution for every objective.

## 3. Gradient descent updates parameters

For a parameter vector $\theta$, gradient descent uses

$$
\theta_{t+1}=\theta_t-\eta\nabla J(\theta_t),
$$

where $\eta>0$ is the learning rate.

The gradient points toward the steepest local increase. Subtracting it gives a local descent direction. The learning rate scales the step.

For $J(w)=(w-3)^2$, $w_0=0$, and $\eta=0.1$:

$$
\frac{dJ}{dw}\bigg|_{w=0}=-6,
$$

$$
w_1=0-0.1(-6)=0.6,
$$

$$
J(w_0)=9,\qquad J(w_1)=5.76.
$$

One lower objective value supports the claim that this step improved the objective. It does not establish convergence.

## 4. Linear regression and mean squared error

For $m$ observations with rows in $X$:

$$
\widehat{y}=Xw+b.
$$

Define the residual vector

$$
e=\widehat{y}-y.
$$

The mean-squared-error objective is

$$
J(w,b)=\frac{1}{m}\sum_{i=1}^{m}(\widehat{y}_i-y_i)^2
      =\frac{1}{m}e^Te.
$$

The gradients are

$$
\nabla_w J=\frac{2}{m}X^Te
$$

and

$$
\frac{\partial J}{\partial b}=\frac{2}{m}\sum_{i=1}^{m}e_i.
$$

Shape checks are part of the derivation:

- $X$ has shape $(m,d)$;
- $e$ has shape $(m,)$;
- $X^Te$ has shape $(d,)$, matching $w$; and
- the bias gradient is one scalar.

## 5. Why the gradient has this form

For one weight $w_j$:

$$
\frac{\partial J}{\partial w_j}
=\frac{1}{m}\sum_{i=1}^{m}2e_i\frac{\partial e_i}{\partial w_j}.
$$

Because

$$
e_i=\sum_{k=1}^{d}x_{ik}w_k+b-y_i,
$$

we have

$$
\frac{\partial e_i}{\partial w_j}=x_{ij}.
$$

Therefore,

$$
\frac{\partial J}{\partial w_j}
=\frac{2}{m}\sum_{i=1}^{m}x_{ij}e_i.
$$

Stacking all $d$ components gives $\nabla_wJ=(2/m)X^Te$.

## 6. Full-batch, stochastic, and mini-batch estimates

The update rule is the same across the three schedules. The data used to estimate the gradient differ.

| Schedule | Examples per update | Updates per epoch | Typical behavior |
| --- | ---: | ---: | --- |
| Full batch | all $n$ | 1 | stable direction, expensive update |
| Stochastic gradient descent | 1 | $n$ | noisy direction, many updates |
| Mini-batch | $B$ | $\lceil n/B\rceil$ | compromise for vectorized hardware |

An **epoch** is one pass in which every training example is used once. A **batch** is the subset used for one gradient estimate. An **update** is one parameter change.

With 220 training examples and batch size 32:

$$
\left\lceil\frac{220}{32}\right\rceil=7
$$

updates occur per epoch. Six batches contain 32 examples and the final batch contains 28.

## 7. Shuffle without losing coverage

A correct mini-batch epoch:

1. creates one permutation of the training indices;
2. slices that permutation into consecutive batches;
3. uses every index exactly once; and
4. permits a smaller final batch.

Dropping the final remainder silently removes examples. Sampling independently for each batch can duplicate some observations and omit others. Both choices change the meaning of an epoch.

The Week 5 tests verify exact coverage for divisible and remainder dataset sizes.

## 8. The learning rate controls step size

A learning rate that is too small can reduce the loss while making little progress within the available compute budget. A useful learning rate gives substantial reduction without unstable oscillation. A learning rate that is too large can overshoot repeatedly, increase loss, or produce nonfinite values.

Diagnose the complete record:

- initial and final training loss;
- loss curve shape;
- gradient or update magnitude when available;
- number of updates;
- finite-value checks; and
- validation performance for the chosen selection rule.

A decreasing training curve is optimization evidence. It is not test evidence.

## 9. Feature scaling changes the geometry

If one feature varies from $-1$ to $1$ and another from $-1000$ to $1000$, equal-sized parameter changes can have very different effects on the predictions. The objective surface becomes elongated, and one learning rate may be too aggressive in one direction but slow in another.

Standardization uses

$$
x'_{ij}=\frac{x_{ij}-\mu_j}{s_j}.
$$

Fit $\mu_j$ and $s_j$ on the training split only. Apply those fixed values to validation and test data. Fitting them on all observations leaks information across the evaluation boundary.

Scaling changes the optimization path and the numerical parameter values. It does not create evidence that a model generalizes.

## 10. Select with validation; evaluate once with test

Use the splits for different decisions:

| Split | Purpose |
| --- | --- |
| Training | estimate model parameters |
| Validation | select learning rate, stopping point, and other planned choices |
| Test | estimate final performance of the selected procedure |

Repeatedly choosing settings from test results turns the test set into another validation set. Its final score then overstates how independent the evaluation is.

A transparent selection protocol states the candidate learning rates, the validation metric, the tie rule, the stopping budget, and the random seed before examining the results.

## 11. Stopping rules and diagnostics

Possible stopping conditions include:

- a fixed epoch or update budget;
- validation loss that has not improved for a stated number of checks;
- a small gradient norm;
- a small relative objective change; or
- divergence, nonfinite values, or a resource limit.

A small training-loss change can mean convergence, a learning rate that is too small, poor feature scaling, or a flat region. Interpret it with other diagnostics.

## 12. What the Week 5 lab does

The lab creates 320 generated observations with three differently scaled features. It uses disjoint training, validation, and test indices. Scaling statistics are fitted on the training split.

The script then:

1. compares full-batch, stochastic, and mini-batch schedules for the same 40 epochs;
2. reports update counts so the compute exposure is visible;
3. evaluates five predeclared learning rates with validation MSE;
4. identifies an unstable large learning rate;
5. selects the lowest finite validation result; and
6. reports test MSE once for the selected run.

Because the schedules have different numbers of updates in 40 epochs, the comparison explains behavior and cost; it does not rank the schedules as universally superior.

## 13. Numerical gradient checking

An analytical gradient can be checked against a centered finite difference:

$$
\frac{\partial J}{\partial \theta_j}
\approx
\frac{J(\theta_j+\varepsilon)-J(\theta_j-\varepsilon)}{2\varepsilon}.
$$

For a small test problem and suitable $\varepsilon$, the two values should agree closely. This checks the derivative implementation at selected values. It cannot prove that every part of a training system is correct.

## 14. From the lab to the research project

The Week 5 baseline experiment plan should freeze:

- the research question and simplest credible baseline;
- dataset version, observation unit, and split unit;
- preprocessing fitted on training data;
- objective and primary evaluation metric;
- learning-rate candidates and validation selection rule;
- batch size, initialization, seed, and stopping rule;
- command, environment, expected outputs, and compute limit; and
- conditions for learning, instability, baseline credibility, and revision.

Run the baseline in Week 6 only after these decisions are reviewable.

## Review questions

1. Why is the negative gradient used in the update?
2. What does one lower objective value establish?
3. Derive $\partial J/\partial b$ for mean squared error.
4. How many updates occur for 220 examples with batch size 32?
5. Why can two schedule results after the same number of epochs still reflect different compute?
6. How does feature scaling change the range of useful learning rates?
7. Which decisions belong to validation rather than test?
8. What can a numerical gradient check establish?
