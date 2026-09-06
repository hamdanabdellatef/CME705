# Week 6 worksheet: Backpropagation and losses

Name: ________________________________  Date: __________________

Write the shape beside every gradient. Use training accuracy on XOR only as mechanism evidence.

## 1. What backpropagation supplies

Complete the Week 5 update:

$$
\theta_{t+1}=\theta_t-\eta\underline{\hspace{4cm}}.
$$

- Backpropagation computes: ___________________________________________
- Gradient descent performs: _________________________________________
- Why is the distinction useful? _____________________________________

## 2. Forward pass and cache

For a batch $X:(m,d)$, hidden width $h$, and one output, complete the expressions and shapes.

| Quantity | Expression | Shape | Cache for backward? |
| --- | --- | --- | --- |
| $Z_1$ | $XW_1+b_1$ | | yes / no |
| $H$ | | | yes / no |
| $Z_2$ | | | yes / no |
| $P$ | | | yes / no |

Parameters:

| Parameter | Shape |
| --- | --- |
| $W_1$ | |
| $b_1$ | |
| $W_2$ | |
| $b_2$ | |

Why must the cache come from the same parameter state as the backward pass?

__________________________________________________________________________

## 3. Hidden responsibility

A dataset supplies $Y$ for the network output but does not supply a target for $H$.

Rewrite the hidden-layer question in terms of sensitivity:

__________________________________________________________________________

Which rule connects the hidden value to the final objective? ______________

## 4. Binary cross-entropy and sigmoid

Write the mean binary cross-entropy objective:

$$
J=\underline{\hspace{11cm}}
$$

For sigmoid output with binary cross-entropy, complete:

$$
G_{Z_2}=\underline{\hspace{6cm}}.
$$

Why should this simplified expression not be copied to every output activation and objective?

__________________________________________________________________________

## 5. Output-layer gradients

Starting from $G_{Z_2}$, complete:

$$
G_{W_2}=\underline{\hspace{6cm}}
$$

$$
G_{b_2}=\underline{\hspace{6cm}}
$$

| Gradient | Required shape | Reason |
| --- | --- | --- |
| $G_{W_2}$ | | |
| $G_{b_2}$ | | |

Why is the bias gradient summed over observations? ________________________

## 6. One hidden chain-rule path

For one hidden unit $j$ and one output, complete:

$$
\frac{\partial J}{\partial z_{1j}}
=
\frac{\partial J}{\partial z_2}
\;\underline{\hspace{2.5cm}}\;
\underline{\hspace{3cm}}.
$$

Name the three factors.

1. ___________________________________________________________________
2. ___________________________________________________________________
3. ___________________________________________________________________

If $|h_j|$ is near 1, what happens to $1-h_j^2$? ________________________

## 7. Hidden-layer gradients

Complete the matrix backward pass:

$$
G_H=\underline{\hspace{6cm}}
$$

$$
G_{Z_1}=\underline{\hspace{6cm}}
$$

$$
G_{W_1}=\underline{\hspace{6cm}}
$$

$$
G_{b_1}=\underline{\hspace{6cm}}
$$

| Gradient | Shape |
| --- | --- |
| $G_H$ | |
| $G_{Z_1}$ | |
| $G_{W_1}$ | |
| $G_{b_1}$ | |

## 8. Put the backward pass in order

Number these operations from 1 to 7.

| Order | Operation |
| ---: | --- |
| | $G_{W_1}=X^TG_{Z_1}$ |
| | $G_H=G_{Z_2}W_2^T$ |
| | $G_{Z_2}=(P-Y)/m$ |
| | $G_{b_1}=\operatorname{sum}(G_{Z_1},\text{rows})$ |
| | $G_{W_2}=H^TG_{Z_2}$ |
| | $G_{Z_1}=G_H\odot(1-H^2)$ |
| | $G_{b_2}=\operatorname{sum}(G_{Z_2},\text{rows})$ |

Why does backward order reverse the forward dependencies? _________________

## 9. Update ordering

A student computes $G_{Z_2}$ and $G_{W_2}$, immediately updates $W_2$, and then uses the changed $W_2$ to calculate $G_H$.

What is wrong with this sequence? ________________________________________

Write the safer sequence:

1. ___________________________________________________________________
2. ___________________________________________________________________
3. ___________________________________________________________________

## 10. Numerical gradient check

Complete the centered finite difference:

$$
G_j^{\text{num}}
\approx
\frac{\underline{\hspace{8cm}}}{2\varepsilon}.
$$

A check reports maximum relative error $3.0\times10^{-9}$ with tolerance $10^{-6}$.

- Does it pass? ______________________________________________________
- What implementation claim does it support? __________________________
- What does it not validate? _________________________________________

## 11. XOR prediction

Targets are:

| $x_1$ | $x_2$ | XOR target |
| ---: | ---: | ---: |
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

Sketch or explain why one straight decision boundary cannot separate the classes.

__________________________________________________________________________

Predict before running:

- One sigmoid unit final probabilities: _______________________________
- One sigmoid unit accuracy: _________________________________________
- Two-layer MLP accuracy on the four training rows: ___________________

## 12. Run the NumPy lab

Run:

~~~bash
python labs/week06_mlp_numpy.py
~~~

Record:

| Evidence | Observed value |
| --- | --- |
| Hidden shape | |
| Prediction shape | |
| Maximum gradient-check error | |
| Linear-model loss | |
| Linear-model accuracy | |
| MLP initial loss | |
| MLP final loss | |
| MLP predicted classes | |
| MLP training accuracy | |

Which result establishes a correct scientific evaluation on unseen data?

__________________________________________________________________________

Explain the claim boundary:

__________________________________________________________________________

## 13. Reproducible project baseline

Complete one row for the baseline you ran.

| Record | Actual value |
| --- | --- |
| Commit or archive checksum | |
| Environment file and Python version | |
| Dataset version or checksum | |
| Split identifiers | |
| Exact command | |
| Actual configuration | |
| Validation result | |
| Final test result, if independently used | |
| First error-analysis finding | |
| Deviation from the Week 5 plan | |
| Next controlled experiment | |

Could another student run the baseline without verbal guidance? Explain.

__________________________________________________________________________

## 14. Exit ticket

1. Backpropagation is needed because _____________________________________
2. $W_2^T$ appears in $G_H$ because ____________________________________
3. A low gradient-check error supports __________________________________
4. Perfect XOR training accuracy does not establish ______________________
