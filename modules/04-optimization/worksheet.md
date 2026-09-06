# Week 5 worksheet: Gradient-based optimization

Name: ________________________________  Date: __________________

Show each substitution and label the evidence as training, validation, or test evidence.

## 1. Objective and metric

For a regression project, suppose the model trains with mean squared error and the research question values typical absolute error.

- Training objective: _________________________________________________
- Primary evaluation metric: _________________________________________
- Why might these be different? ______________________________________

For each statement, write **objective**, **metric**, or **both**.

| Statement | Classification |
| --- | --- |
| Supplies the gradient used for updates | |
| Should reflect the research question | |
| Can be recorded during an experiment | |
| Must be differentiable for ordinary gradient descent | |

## 2. Interpret a derivative

Use

$$
J(w)=(w-3)^2,\qquad \frac{dJ}{dw}=2(w-3).
$$

Complete:

| Current $w$ | $J(w)$ | $dJ/dw$ | Descent direction |
| ---: | ---: | ---: | --- |
| 0 | | | |
| 3 | | | |
| 5 | | | |

What does the sign of the derivative tell you? ___________________________

What does it not tell you about a general objective? _____________________

## 3. One gradient-descent update

Let $w_0=0$ and $\eta=0.1$.

$$
w_1=w_0-\eta\frac{dJ}{dw}\bigg|_{w=w_0}.
$$

- Gradient at $w_0$: _________________________________________________
- Update term $\eta(dJ/dw)$: ________________________________________
- New parameter $w_1$: ______________________________________________
- $J(w_0)$: _________________________________________________________
- $J(w_1)$: _________________________________________________________

Claim supported by this calculation: ____________________________________

Claim not yet supported: _______________________________________________

## 4. Derive the linear-regression gradients

For

$$
\widehat{y}=Xw+b,\qquad e=\widehat{y}-y,
$$

and

$$
J(w,b)=\frac{1}{m}e^Te,
$$

complete the shapes.

| Quantity | Shape |
| --- | --- |
| $X$ | $(m,d)$ |
| $w$ | |
| $b$ | |
| $\widehat{y}$ | |
| $e$ | |
| $X^Te$ | |

Complete:

$$
\nabla_wJ=\underline{\hspace{7cm}}
$$

$$
\frac{\partial J}{\partial b}=\underline{\hspace{7cm}}
$$

Why must the weight gradient have the same shape as $w$? ________________

## 5. Compare update schedules

There are 220 training examples.

| Schedule | Batch size | Updates per epoch | Examples used in one epoch |
| --- | ---: | ---: | ---: |
| Full batch | 220 | | |
| Stochastic | 1 | | |
| Mini-batch | 32 | | |

For batch size 32, how many examples are in the final batch? _____________

Why does “40 epochs” imply different update counts across these rows? _____

## 6. Verify exact mini-batch coverage

Suppose the shuffled order for 10 examples is

$$
[7,2,9,0,4,1,8,5,3,6].
$$

With batch size 4, write the batches.

- Batch 1: ___________________________________________________________
- Batch 2: ___________________________________________________________
- Batch 3: ___________________________________________________________

Circle the correct statements:

- Every index appears once / some indices repeat.
- The last batch is retained / discarded.
- One epoch contains 10 / 12 examples.

What implementation check would detect omissions or duplicates? __________

## 7. Diagnose learning-rate histories

| Run | Initial training MSE | Epoch 10 | Epoch 40 | Diagnosis |
| --- | ---: | ---: | ---: | --- |
| A | 8.54 | 5.90 | 3.20 | |
| B | 8.54 | 0.08 | 0.05 | |
| C | 8.54 | 120.0 | nonfinite | |

For each run, state one next action.

- A: __________________________________________________________________
- B: __________________________________________________________________
- C: __________________________________________________________________

Why is Run B's training curve insufficient as final project evidence? _____

## 8. Feature scaling

Feature 1 has standard deviation 1.0. Feature 2 has standard deviation 1000.

Explain why one learning rate can behave very differently across the two parameter directions.

__________________________________________________________________________

Write the standardization rule:

$$
x'_{ij}=\underline{\hspace{7cm}}
$$

Which split supplies $\mu_j$ and $s_j$? _________________________________

## 9. Predict and run the NumPy lab

Before running, predict:

- Which schedule has one update per epoch? _____________________________
- Which schedule has the most updates in 40 epochs? ____________________
- Which learning rate is likely to diverge? ____________________________
- Should all stable rates have identical validation MSE? _______________

Run:

~~~bash
python labs/week05_gradient_descent.py
~~~

Record the schedule table.

| Schedule | Batch | Updates | Initial MSE | Final MSE |
| --- | ---: | ---: | ---: | ---: |
| Full batch | | | | |
| Stochastic | | | | |
| Mini-batch | | | | |

Record the validation comparison.

| Learning rate | Validation MSE | Status |
| ---: | ---: | --- |
| 0.001 | | |
| 0.010 | | |
| 0.050 | | |
| 0.200 | | |
| 1.200 | | |

- Selected learning rate: _____________________________________________
- Final test MSE: ____________________________________________________
- Why was test MSE reported only after selection? ______________________

## 10. Numerical gradient check

A centered finite difference is

$$
\frac{\partial J}{\partial \theta_j}
\approx
\frac{J(\theta_j+\varepsilon)-J(\theta_j-\varepsilon)}{2\varepsilon}.
$$

What does close agreement with the analytical gradient support? __________

What can it not prove? __________________________________________________

## 11. Baseline experiment plan

Complete one concrete row for your project.

| Decision | Planned value | Evidence or reason |
| --- | --- | --- |
| Baseline | | |
| Frozen split | | |
| Training objective | | |
| Validation metric | | |
| Learning-rate candidates | | |
| Batch size | | |
| Stopping rule | | |
| Seed and command | | |

One result that would justify a more complex method: ______________________

One result that would trigger debugging: _________________________________

## 12. Exit ticket

1. The negative gradient is used because __________________________________
2. An epoch differs from an update because ________________________________
3. Validation data will choose ___________________________________________
4. Test data will be used when ___________________________________________
