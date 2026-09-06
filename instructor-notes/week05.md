# Week 5 instructor guide: Gradient-based optimization

## Purpose

Week 5 connects the fixed forward computation from Week 4 to parameter learning. Students derive a gradient for transparent linear regression before using the same update logic inside deeper networks. The lesson treats training curves as optimization evidence and validation results as selection evidence.

The historical course mixed the delta rule, neural-network review, and MATLAB demonstrations. This version keeps the useful gradient-descent concepts, replaces the examples with tested NumPy, and reserves multilayer backpropagation for Week 6.

## Before class

- Run python labs/week05_gradient_descent.py.
- Distribute [the Week 5 worksheet](../modules/04-optimization/worksheet.md).
- Ask students to bring the baseline chosen in their Week 4 literature comparison.
- Prepare a board or shared document for the one-parameter update.
- Keep the lab output hidden until students record predictions.

## Learning evidence

Inspect:

1. the hand-calculated update for $J(w)=(w-3)^2$;
2. the mean-squared-error weight and bias gradients;
3. exact batch, epoch, and update counts;
4. the diagnosis of small, useful, and unstable learning rates;
5. the schedule and validation tables from the lab; and
6. a baseline experiment plan that freezes the evaluation and training decisions.

## Three-hour sequence

| Time | Activity | Instructor action | Student evidence |
| ---: | --- | --- | --- |
| 0–10 min | Opening distinction | Contrast objective, metric, and split roles | Four classified statements |
| 10–30 min | One-parameter surface | Trace $J(w)$ and its derivative | Slope table |
| 30–45 min | Manual update | Substitute $w_0$ and $\eta$ | $w_1$ and both objective values |
| 45–70 min | Linear-regression gradient | Derive one component, then vectorize | Weight and bias gradients with shapes |
| 70–85 min | Update schedules | Compare gradient estimates and update counts | Completed schedule table |
| 85–95 min | Break |  |  |
| 95–110 min | Exact coverage | Trace a shuffled remainder batch | Three batches and coverage check |
| 110–130 min | Learning rate and scaling | Diagnose histories and elongated geometry | Three diagnoses and scaling rule |
| 130–145 min | Split roles and stopping | Define selection and failure rules | Evidence-role classification |
| 145–168 min | NumPy lab | Require predictions before execution | Schedule, validation, and test record |
| 168–177 min | Baseline plan | Review one concrete project configuration | Completed plan row |
| 177–180 min | Exit ticket | Collect four concise statements | Completed exit ticket |

## Facilitation notes

### Objective and metric

Ask which quantity directly supplies parameter updates. The objective must support gradients for ordinary gradient descent. The evaluation metric should represent the research question and may be nondifferentiable.

Return to the Week 3 rare-event example: a differentiable classification loss may train the model while balanced accuracy or recall communicates the evaluation.

### One-parameter surface

For $J(w)=(w-3)^2$:

| $w$ | $J(w)$ | $dJ/dw$ | Descent direction |
| ---: | ---: | ---: | --- |
| 0 | 9 | -6 | increase $w$ |
| 3 | 0 | 0 | no first-order direction |
| 5 | 4 | 4 | decrease $w$ |

The derivative gives a local slope. Avoid implying that every zero derivative is a global minimum.

### Manual update

With $w_0=0$ and $\eta=0.1$:

$$
w_1=0-0.1(-6)=0.6.
$$

The objective falls from 9 to 5.76. This establishes improvement for one step on this objective. It does not establish convergence or transfer to another learning rate.

### Linear-regression gradient

Begin with one component:

$$
\frac{\partial J}{\partial w_j}
=\frac{2}{m}\sum_{i=1}^{m}x_{ij}e_i.
$$

Then stack components:

$$
\nabla_wJ=\frac{2}{m}X^Te.
$$

The bias gradient is

$$
\frac{\partial J}{\partial b}=\frac{2}{m}\sum_i e_i.
$$

Require the shape check. $X^Te$ has shape $(d,)$, so it can update $w$.

### Update schedules

For 220 examples:

| Schedule | Batch | Updates per epoch | Updates in 40 epochs |
| --- | ---: | ---: | ---: |
| Full batch | 220 | 1 | 40 |
| Stochastic | 1 | 220 | 8,800 |
| Mini-batch | 32 | 7 | 280 |

The same epoch budget does not imply the same number of updates or identical computation. Treat the lab comparison as a mechanism demonstration.

### Exact coverage

For the worksheet order $[7,2,9,0,4,1,8,5,3,6]$ and batch size 4:

- batch 1: $[7,2,9,0]$;
- batch 2: $[4,1,8,5]$; and
- batch 3: $[3,6]$.

Concatenating and sorting the used indices should reproduce $0$ through $9$ exactly.

### Learning-rate diagnosis

Expected diagnoses:

- Run A: stable but slow within the budget; consider a larger planned candidate or longer justified budget.
- Run B: useful reduction; check validation behavior and repeatability.
- Run C: unstable; stop, inspect scaling and gradients, and reduce the rate.

A monotonic training loss is not required for stochastic or mini-batch training. The full-training loss recorded after each epoch can still fluctuate.

### Feature scaling

Use a narrow elongated contour sketch. A single learning rate can overshoot along the steep direction while moving slowly along the flat direction. Training-only standardization often reduces this mismatch.

Do not let students fit scaling on all data. The validation and test transformations must use training means and scales.

### Evidence roles

Use this language consistently:

- training objective: optimization evidence;
- validation metric: configuration-selection evidence;
- test metric: final evaluation evidence.

If a student checks test results after each learning-rate run, the test split has become part of selection.

### Stopping

A fixed budget is acceptable for the teaching lab. Project plans may also specify patience on a validation metric, a gradient threshold, a relative-change threshold, divergence, or a resource limit. The stopping rule must be declared and recorded.

### Week 5 lab

Expected output is deterministic:

| Schedule | Batch | Updates | Initial MSE | Final MSE |
| --- | ---: | ---: | ---: | ---: |
| Full batch | 220 | 40 | 8.5402 | 0.0530 |
| Stochastic | 1 | 8,800 | 8.5402 | 0.0579 |
| Mini-batch | 32 | 280 | 8.5402 | 0.0522 |

Expected validation results:

| Learning rate | Validation MSE | Status |
| ---: | ---: | --- |
| 0.001 | 0.3143 | finite |
| 0.010 | 0.0481 | finite |
| 0.050 | 0.0483 | finite |
| 0.200 | 0.0525 | finite |
| 1.200 | very large | diverged |

The selected rate is 0.010 and the final test MSE is approximately 0.0532. Small numerical differences may occur across NumPy platforms.

The printed standardized weights are not the original raw-feature coefficients. This is a useful prompt about representation and parameter interpretation.

### Numerical gradient check

The course test compares analytical gradients with centered finite differences at selected parameter values. Close agreement supports the derivative implementation. It does not check data splitting, selection logic, or every possible input.

### Baseline experiment plan

Students should name one validation-controlled decision. If they list many search dimensions, ask them to reduce the plan to a feasible baseline. Require the dataset version, split unit, preprocessing boundary, objective, metric, seed, command, and expected outputs.

Interpretation rules written before execution help distinguish learning, instability, an inadequate baseline, and an implementation error.

## Common misconceptions

| Misconception | Response |
| --- | --- |
| The gradient points toward lower loss | It points toward the steepest local increase; the update subtracts it |
| A lower training loss proves generalization | It establishes optimization on training data |
| One epoch means one update | The number of updates depends on batch size |
| The final remainder batch can be discarded silently | That changes coverage and the meaning of an epoch |
| SGD and mini-batch gradient descent use different objectives | They estimate the same objective gradient from different subsets |
| The smallest learning rate is safest and therefore best | It may make too little progress within the available budget |
| A flat curve proves convergence | Scaling, learning rate, saturation, and implementation errors can also flatten a curve |
| Standardization may use all available data | Its statistics must be fitted on training data |
| Test performance can choose the learning rate | Validation selects; test evaluates the selected procedure |
| Equal epochs make schedule comparisons fair | Update counts and computation can differ substantially |

## Formative feedback language

- “Which quantity supplies the update?”
- “What does the derivative sign say at this point?”
- “Substitute every value before simplifying.”
- “Does the gradient shape match the parameter shape?”
- “How many parameter updates occurred?”
- “Show that every index appears exactly once.”
- “Is this training, validation, or test evidence?”
- “What decision was frozen before the run?”

## Adaptations

### Ninety-minute class

Use slides 1–10 and 14–21. Complete the manual update, gradient shape check, learning-rate diagnosis, lab selection record, and one baseline-plan row. Assign the detailed schedule and coverage work afterward.

### Online delivery

Use a shared loss-curve annotation activity. Give breakout groups different batch sizes and ask each to calculate updates per epoch. Run the deterministic NumPy lab centrally if environments differ.

### Limited calculus preparation

Interpret derivatives first as local slopes. Derive one weight component by expanding the sums, then show the vector form. Require shape reasoning even if symbolic manipulation remains supported.

## After class

- identify students who confuse objective, metric, and split roles;
- check project plans for test-guided tuning or preprocessing leakage;
- require one feasible validation-controlled decision;
- return plan feedback before the Week 6 baseline run; and
- carry the scalar update rule into chain rule and backpropagation.
