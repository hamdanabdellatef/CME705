# Week 8 notes: Deep networks and generalization

## 1. Depth expands the function class

A deep feedforward network composes transformations:

$$
H_\ell=
\phi_\ell(H_{\ell-1}W_\ell+b_\ell).
$$

Additional depth can construct representations that are difficult to express with one shallow transformation. It also adds parameters, nonlinearities, and longer gradient paths.

Depth is a modeling choice, not evidence of quality. More capacity can fit:

- useful structure;
- irrelevant nuisance features;
- measurement artifacts; and
- label noise.

The relevant question is whether the added capacity improves performance under a valid evaluation procedure.

## 2. Separate three failure patterns

### Underfitting or optimization failure

Training and validation losses both remain high. Possible causes include:

- insufficient representation capacity;
- weak or missing features;
- an unsuitable learning rate;
- poor initialization;
- dead activations;
- too few updates; or
- implementation errors.

A learning curve alone may not distinguish these causes.

### Overfitting

Training loss continues to decrease while validation loss stops improving or becomes worse. The model is fitting training-specific detail that does not help the validation distribution.

### Distribution shift

Training and ordinary validation results may look strong while a deployment period, site, subject group, sensor, or subgroup fails. Regularization does not repair a split that fails to represent the intended use.

Do not label every weak validation result “overfitting.” First inspect optimization, data quality, leakage, sample size, and shift.

## 3. ReLU

The rectified linear unit is:

$$
\operatorname{ReLU}(z)=\max(0,z).
$$

Its local derivative is commonly implemented as:

$$
\frac{d}{dz}\operatorname{ReLU}(z)=
\begin{cases}
1,&z>0,\\
0,&z\le0.
\end{cases}
$$

At exactly zero the mathematical derivative is not unique; software chooses a convention. The Week 8 lab uses zero.

For:

$$
z=[-1,0,2],
$$

ReLU gives:

$$
[0,0,2].
$$

With upstream gradient $[3,4,5]$, the backward result is:

$$
[0,0,5].
$$

ReLU avoids sigmoid saturation on its positive side. It can still create inactive units and does not eliminate exploding gradients, poor initialization, or every optimization difficulty.

## 4. Gradients across depth

Backpropagation multiplies local derivatives and weight matrices across layers. A schematic path is:

$$
\frac{\partial J}{\partial H_{\ell-1}}
=
\frac{\partial J}{\partial H_\ell}
\frac{\partial H_\ell}{\partial Z_\ell}
W_\ell^T.
$$

Repeated factors smaller than one can shrink sensitivity as it moves backward. Repeated large factors can amplify it.

Optimization depends on several interacting choices:

- activation functions;
- initialization scale;
- normalization;
- architecture;
- residual or skip connections;
- objective;
- optimizer; and
- data scale.

The course uses ReLU and He initialization in the Week 8 lab. This pairing is reasonable for the teaching network, but it is not a universal guarantee.

## 5. Read training and validation curves together

A learning curve records a quantity such as loss after each epoch or update.

| Pattern | Training loss | Validation loss | Interpretation to investigate |
| --- | --- | --- | --- |
| Both remain high | high or flat | high or flat | optimization or underfitting |
| Training falls, validation falls | improving | improving | useful learning under the split |
| Training falls, validation stalls | improving | flat | growing generalization gap |
| Training falls, validation rises | improving | worsening | overfitting or mismatch |
| Validation is noisy | variable | variable | small sample, randomness, or unstable procedure |

Curves are diagnostic evidence. They do not prove why a pattern occurred.

## 6. Generalization gap

For a loss:

$$
g_t=
J_{\text{validation},t}
-
J_{\text{training},t}.
$$

A growing positive gap can indicate that the model is fitting training detail or that the validation split differs in a meaningful way.

Compare gaps only when:

- training and validation quantities use the same loss definition;
- preprocessing is fixed correctly;
- dropout and similar stochastic mechanisms are disabled for both evaluations;
- checkpoints use the same architecture; and
- sample roles remain unchanged.

The gap is not a replacement for the primary research metric.

## 7. Capacity, data, and regularization

Generalization interventions change what is easy for the model to fit or which checkpoint is retained.

Common interventions include:

- acquiring or improving data;
- reducing model capacity;
- adding data augmentation appropriate to the domain;
- penalizing weights;
- using dropout;
- stopping at a validation-selected checkpoint; and
- improving the split or metric.

Regularization cannot correct target leakage, duplicated subjects across splits, or an irrelevant metric.

## 8. $L_2$ regularization

Add a weight penalty to the data objective:

$$
J_{\text{reg}}
=
J_{\text{data}}
+
\frac{\lambda}{2}
\sum_\ell
\lVert W_\ell\rVert_F^2.
$$

The gradient for a weight matrix becomes:

$$
\nabla_{W_\ell}J_{\text{reg}}
=
\nabla_{W_\ell}J_{\text{data}}
+
\lambda W_\ell.
$$

With ordinary gradient descent:

$$
W_\ell
\leftarrow
W_\ell
-
\eta
\left(
\nabla_{W_\ell}J_{\text{data}}
+
\lambda W_\ell
\right).
$$

Equivalently:

$$
W_\ell
\leftarrow
(1-\eta\lambda)W_\ell
-
\eta\nabla_{W_\ell}J_{\text{data}}.
$$

This derivation describes an $L_2$ objective penalty with ordinary gradient descent. Some optimizers implement decoupled weight decay differently. Name the implementation rather than treating every option called “weight decay” as identical.

Biases and some scale parameters are often excluded by explicit design. The choice must be documented.

## 9. Early stopping

Early stopping selects a checkpoint with validation evidence.

A reproducible rule specifies:

- monitored quantity;
- whether lower or higher is better;
- minimum improvement;
- patience;
- maximum epochs;
- evaluation frequency;
- tie rule; and
- which state is restored.

A typical procedure is:

1. train for one epoch;
2. evaluate validation loss without dropout;
3. save parameters if the improvement exceeds the tolerance;
4. reset patience after improvement;
5. stop after the allowed number of stale epochs; and
6. restore the best saved parameters.

The final in-memory parameters are not automatically the selected model.

## 10. Standard dropout

Let $p$ be the drop probability and:

$$
q=1-p
$$

the keep probability.

For every activation component during training:

$$
M_{ij}\sim\operatorname{Bernoulli}(q).
$$

A standard unscaled dropout output would be:

$$
\widetilde H=M\odot H.
$$

Its expectation is $qH$, so its scale differs from the full network.

## 11. Inverted dropout

Inverted dropout scales kept activations during training:

$$
\widetilde H=
\frac{M}{q}\odot H.
$$

Its expectation is:

$$
\mathbb{E}[\widetilde H]
=
\mathbb{E}\left[\frac{M}{q}\right]\odot H
=
H.
$$

At evaluation time, dropout is disabled:

$$
\widetilde H=H.
$$

No additional evaluation scaling is needed.

The mask contains:

- $0$ for dropped units; and
- $1/q$ for kept units.

## 12. Dropout backward pass

The forward relation is:

$$
\widetilde H=
\frac{M}{q}\odot H.
$$

The backward relation is:

$$
G_H=
G_{\widetilde H}
\odot
\frac{M}{q}.
$$

The same mask must be reused. A unit dropped during the forward computation has no influence on the output for that pass, so it receives zero gradient for that pass.

Sampling a new mask during backward computation differentiates a different graph.

For an activation function, apply both:

1. the saved dropout mask; and
2. the local activation derivative evaluated from the appropriate pre-activation or unmasked activation.

Do not calculate a sigmoid derivative from a value that has already been multiplied by an inverted-dropout mask.

## 13. Training and evaluation modes

| Behavior | Training | Evaluation |
| --- | --- | --- |
| Dropout | stochastic mask | disabled |
| Kept-unit scale | $1/q$ | unchanged |
| Fixed-input output | stochastic | deterministic |
| Gradient computation | enabled | disabled for ordinary inference |
| Batch-statistic behavior | training rule | stored or evaluation rule |
| Purpose | fit parameters | compare or deploy |

A test run with training-mode dropout measures a stochastic subnetwork rather than the intended evaluation model.

## 14. Week 8 mechanism check

The lab begins with $200{,}000$ activation values equal to $0.8$ and drop probability $p=0.25$.

It reports:

- mean before dropout: $0.8000$;
- training-mode mean: approximately $0.8002$;
- dropped fraction: approximately $0.2498$;
- evaluation maximum difference: $0$;
- evaluation mask all ones: true;
- mean sigmoid backward gradient: $0.1600$; and
- nonzero gradients at dropped positions: $0$.

The expected sigmoid derivative at activation $0.8$ is:

$$
0.8(1-0.8)=0.16.
$$

The sample mean after inverted dropout is close to the original mean, not exactly equal for a finite random mask.

## 15. Controlled teaching data

The lab creates a deliberately difficult binary task:

- $96$ training observations;
- $240$ validation observations;
- $240$ test observations;
- two informative nonlinear features;
- thirty nuisance features; and
- $19$ deliberately flipped training labels.

Validation and test labels follow the clean generating rule. This setup creates a visible risk that a high-capacity model will fit training-specific noise.

The data are synthetic and generated deterministically. The result is a mechanism demonstration, not an estimate for a real application.

## 16. One changed factor

The comparison holds fixed:

- generated observations and split;
- training-label noise;
- training-fitted standardization;
- two-hidden-layer ReLU architecture;
- He initialization;
- initialization seed;
- learning rate;
- maximum epochs;
- early-stopping tolerance and patience;
- validation objective; and
- test procedure.

Only the dropout probability changes:

$$
p\in\{0,0.15,0.30,0.45\}.
$$

This design supports an interpretation about dropout under the recorded conditions. Changing dropout, architecture, learning rate, and augmentation together would not isolate one contribution.

## 17. Validation results

The deterministic comparison reports:

| Dropout | Best epoch | Stop epoch | Training loss | Validation loss | Gap | Last validation loss |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00 | 81 | 261 | 0.2679 | 0.7180 | 0.4500 | 0.8509 |
| 0.15 | 119 | 299 | 0.2670 | 0.6476 | 0.3805 | 0.7363 |
| 0.30 | 232 | 412 | 0.2461 | 0.5920 | 0.3459 | 0.6306 |
| 0.45 | 513 | 693 | 0.2429 | 0.5590 | 0.3161 | 0.5899 |

Validation selects $p=0.45$.

The last validation loss is worse than the best validation loss for every run. This is why the best checkpoint is restored rather than returning the last parameter state.

## 18. Locked test evidence

After dropout selection, the lab evaluates the no-dropout reference and the selected candidate on the same locked test observations:

| Model | Test loss | Accuracy |
| --- | ---: | ---: |
| $p=0$ reference | 0.6746 | 0.633 |
| selected $p=0.45$ | 0.5649 | 0.733 |

For the selected model:

- negative-class recall is $0.661$;
- positive-class recall is $0.810$; and
- the confusion matrix is:

$$
\begin{bmatrix}
82 & 42 \\
22 & 94
\end{bmatrix},
$$

with rows as true classes and columns as predicted classes.

The evidence supports the fixed comparison on this split. It does not show that dropout $0.45$ is optimal elsewhere or that dropout always improves generalization.

## 19. Mechanism evidence and outcome evidence

Mechanism checks answer whether the implementation behaves as defined:

- expected scale is preserved;
- evaluation mode is deterministic;
- the saved mask is reused;
- dropped units receive zero gradient.

Validation and test comparisons answer whether the chosen intervention helped under the recorded procedure.

A correct dropout implementation can fail to improve a particular task. A better validation score can also arise from an implementation error. Both kinds of evidence are needed.

## 20. Controlled experiments

A controlled experiment records:

1. a directional hypothesis before results;
2. one intended changed factor;
3. fixed data, split, preprocessing, and metric;
4. fixed training and selection procedures;
5. stochastic sources and seeds;
6. repeated runs or another justified uncertainty estimate;
7. all runs, including failures;
8. class, subgroup, or error-direction effects;
9. honest test status; and
10. a claim bounded by the procedure.

The Week 8 teaching lab uses one deterministic seed to make classroom results identical. Student research should normally use several seeds or another uncertainty method when stochastic variation matters.

## Common misconceptions

| Misconception | Correction |
| --- | --- |
| A deeper model is automatically better | Depth increases capacity; validation evidence determines usefulness |
| Every validation failure is overfitting | Optimization, data quality, leakage, and shift can also cause failure |
| ReLU eliminates vanishing gradients | It changes local derivatives but does not solve every gradient problem |
| The generalization gap is the final metric | It is a diagnostic used alongside task metrics |
| Regularization repairs leakage | Leakage invalidates the evaluation design |
| Weight decay always means the same equation | Objective penalties and decoupled updates differ |
| Early stopping returns the last epoch | It restores the best validation checkpoint |
| Dropout deletes neurons permanently | It samples masks during training passes |
| Evaluation also samples dropout masks | Inverted dropout is disabled for evaluation |
| Backward can sample a fresh mask | It must reuse the forward mask |
| One improved run proves a method is better | Stochastic comparisons need repeated or uncertainty-aware evidence |

## Review questions

1. What curve pattern suggests underfitting or optimization failure?
2. What curve pattern suggests a growing generalization gap?
3. Why is ReLU's positive-side derivative useful?
4. What does $\lambda W$ add to an $L_2$-regularized gradient?
5. Why must early stopping restore the best checkpoint?
6. Why does inverted dropout divide by keep probability?
7. Why is dropout disabled during evaluation?
8. Why must backward reuse the forward mask?
9. Which decisions are fixed in the Week 8 comparison?
10. What claim does the locked test result support?
