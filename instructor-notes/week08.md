# Week 8 instructor guide: Deep networks and generalization

## Purpose

Week 8 uses deeper ReLU networks to connect representation capacity with evaluation discipline. Students distinguish optimization failure from overfitting, interpret training and validation curves, derive $L_2$ regularization, define early stopping, and implement inverted dropout with correct training, evaluation, and backward behavior.

The historical material identified vanishing gradients, overfitting, and computational load, then introduced ReLU, regularization, validation, and dropout through MATLAB examples. This version retains those concepts while correcting the dropout reasoning: the forward mask is saved, the same mask is used in backward computation, derivatives use the appropriate unmasked activation or pre-activation, and evaluation disables dropout. The NumPy lab holds all decisions fixed except dropout probability.

## Before class

- Run python labs/week08_dropout_numpy.py.
- Distribute [the Week 8 worksheet](../modules/06-multiclass-and-generalization/week08-worksheet.md).
- Ask students to bring the Week 7 evaluation review for their project.
- Prepare four unlabeled training/validation curve sketches.
- Keep the lab comparison results hidden until students predict the direction of the gap.
- Confirm students understand that the teaching data include deliberate training-label noise.

## Learning evidence

Inspect:

1. curve-based distinction among optimization failure, overfitting, and shift;
2. ReLU forward and local backward behavior;
3. an $L_2$ penalty gradient and update;
4. a complete early-stopping rule;
5. inverted-dropout expectation and mask values;
6. saved-mask backward computation;
7. deterministic evaluation behavior;
8. identification of the single changed factor;
9. validation-controlled selection;
10. locked-test interpretation; and
11. a project controlled-experiment plan with uncertainty.

## Three-hour sequence

| Time | Activity | Instructor action | Student evidence |
| ---: | --- | --- | --- |
| 0–12 min | Capacity and evidence | Contrast representational capacity with validation evidence | Capacity statement |
| 12–28 min | Failure patterns | Sort curve sketches | Curve diagnosis |
| 28–45 min | ReLU | Trace forward and local derivative | ReLU calculation |
| 45–58 min | Gradient paths | Explain shrinkage and amplification across depth | Two interacting causes |
| 58–75 min | Generalization gap | Compare training and validation checkpoints | Gap calculation |
| 75–90 min | $L_2$ and weight updates | Derive $\lambda W$ contribution | Penalty update |
| 90–100 min | Break |  |  |
| 100–115 min | Early stopping | Specify monitor, tolerance, patience, and restore rule | Checkpoint record |
| 115–135 min | Inverted dropout | Derive mask scaling and expectation | Forward calculation |
| 135–148 min | Dropout backward | Reuse the saved mask | Backward trace |
| 148–158 min | Train/evaluation modes | Contrast stochastic fitting with deterministic inference | Mode table |
| 158–172 min | NumPy comparison | Predict, run, and interpret | Mechanism and outcome table |
| 172–178 min | Controlled experiment | Freeze controls and plan uncertainty | Assignment design |
| 178–180 min | Exit ticket | Collect five concise statements | Completed exit ticket |

## Facilitation notes

### Depth and capacity

Write:

$$
H_\ell=\phi_\ell(H_{\ell-1}W_\ell+b_\ell).
$$

Ask what another composition can represent. Then ask what else it can fit when the dataset is small or noisy.

Avoid treating the number of layers as a performance guarantee. Use validation evidence to judge whether added capacity helps under the intended evaluation.

### Failure patterns

Show curves without labels and ask students to describe observations before naming causes.

Expected distinctions:

- high flat training and validation loss: optimization or underfitting;
- falling training and validation loss: useful learning;
- falling training loss with stalled or rising validation loss: growing gap;
- unstable validation curve: sample size, stochasticity, or procedure.

Ask what additional evidence would separate alternative explanations.

### ReLU

Expected equations:

$$
\operatorname{ReLU}(z)=\max(0,z),
$$

and:

$$
\operatorname{ReLU}'(z)=
\begin{cases}
1,&z>0,\\
0,&z\le0.
\end{cases}
$$

For $z=[-1,0,2]$ and upstream $[3,4,5]$, the backward result is $[0,0,5]$.

ReLU has a direct positive-side gradient path. It can still create inactive units and does not prevent every exploding or vanishing path.

### Gradients across depth

Use the schematic:

$$
\frac{\partial J}{\partial H_{\ell-1}}
=
\frac{\partial J}{\partial H_\ell}
\frac{\partial H_\ell}{\partial Z_\ell}
W_\ell^T.
$$

Repeated local effects interact with weight matrices. Name activation, initialization, normalization, architecture, and optimizer as related decisions.

### Generalization gap

Expected definition:

$$
g_t=
J_{\text{validation},t}
-
J_{\text{training},t}.
$$

A positive growing gap is a warning, not a complete diagnosis. Require the same loss definition and deterministic evaluation mode on both splits.

### $L_2$ regularization

Expected objective:

$$
J_{\text{reg}}
=
J_{\text{data}}
+
\frac{\lambda}{2}
\sum_\ell\lVert W_\ell\rVert_F^2.
$$

Expected gradient:

$$
\nabla_{W_\ell}J_{\text{reg}}
=
\nabla_{W_\ell}J_{\text{data}}
+\lambda W_\ell.
$$

Expected ordinary gradient-descent update:

$$
W_\ell
\leftarrow
(1-\eta\lambda)W_\ell
-
\eta\nabla_{W_\ell}J_{\text{data}}.
$$

State which parameters receive the penalty. Distinguish the objective derivation from optimizer-specific decoupled weight decay.

### Early stopping

Require all eight decisions:

1. monitored quantity;
2. direction;
3. minimum improvement;
4. patience;
5. maximum epochs;
6. evaluation frequency;
7. tie rule; and
8. restored checkpoint.

The lab saves a parameter copy whenever validation loss improves by at least $10^{-5}$ and stops after 180 stale epochs. It returns the best checkpoint.

### Inverted dropout

Let $p$ be the drop probability and $q=1-p$ the keep probability.

The training mask is:

$$
M_{ij}\sim\operatorname{Bernoulli}(q).
$$

The output is:

$$
\widetilde H=
\frac{M}{q}\odot H.
$$

The mask entries are $0$ or $1/q$. Ask students to prove:

$$
\mathbb{E}[\widetilde H]=H.
$$

For $p=0.25$, kept entries are scaled by $4/3$.

### Dropout backward

Expected result:

$$
G_H=
G_{\widetilde H}
\odot
\frac{M}{q}.
$$

Use the same forward mask. A fresh backward mask describes a different sampled computation.

When dropout follows sigmoid, use the unscaled sigmoid activation for $h(1-h)$, then apply the saved mask. When dropout follows ReLU, use the pre-activation sign for the ReLU derivative and the saved dropout mask for the dropout operation.

### Training and evaluation modes

The lab evaluates loss and predictions with dropout disabled. This makes fixed-input output deterministic and matches the full inverted-dropout model.

Ask students what happens if the test loader or model remains in training mode. The measured model becomes random and differs from the intended evaluation procedure.

### Teaching data

The generated task has:

- 96 training observations;
- 240 validation observations;
- 240 test observations;
- two informative nonlinear features;
- thirty nuisance features; and
- 19 flipped training labels.

Training-fitted standardization is applied to validation and test data. Validation and test labels remain clean under the generating rule.

The construction makes overfitting visible. It does not imitate a specific real dataset.

### Controlled variable

The lab holds fixed:

- data and split;
- label-noise realization;
- preprocessing;
- 32-to-64-to-32-to-1 architecture;
- He initialization and initialization seed;
- learning rate $0.04$;
- maximum 1800 epochs;
- patience 180 and minimum improvement $10^{-5}$; and
- validation loss and test procedure.

Only dropout probability changes among $0$, $0.15$, $0.30$, and $0.45$.

### Mechanism check

Expected deterministic output:

| Evidence | Value |
| --- | --- |
| Activation mean before | $0.8000$ |
| Training-mode mean | $0.8002$ |
| Dropped fraction | $0.2498$ |
| Evaluation maximum difference | $0$ |
| Evaluation mask all ones | true |
| Backward-gradient mean | $0.1600$ |
| Nonzero gradients at dropped positions | $0$ |

The finite-mask mean need not be exactly $0.8$.

### Validation comparison

Expected output:

| Dropout | Best epoch | Stop epoch | Training loss | Validation loss | Gap | Last validation loss |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00 | 81 | 261 | 0.2679 | 0.7180 | 0.4500 | 0.8509 |
| 0.15 | 119 | 299 | 0.2670 | 0.6476 | 0.3805 | 0.7363 |
| 0.30 | 232 | 412 | 0.2461 | 0.5920 | 0.3459 | 0.6306 |
| 0.45 | 513 | 693 | 0.2429 | 0.5590 | 0.3161 | 0.5899 |

Validation selects $p=0.45$.

The best epoch precedes the stop epoch because patience allows training to continue while waiting for another improvement.

### Locked test evaluation

Expected output:

| Model | Loss | Accuracy |
| --- | ---: | ---: |
| $p=0$ reference | $0.6746$ | $0.633$ |
| selected $p=0.45$ | $0.5649$ | $0.733$ |

Selected-model recalls are $0.661$ for class 0 and $0.810$ for class 1. The confusion matrix, with true rows and predicted columns, is:

$$
\begin{bmatrix}
82 & 42 \\
22 & 94
\end{bmatrix}.
$$

The result supports this pre-specified comparison on one synthetic split. It does not establish a transferable dropout rate.

### Mechanism and outcome evidence

Ask students to sort claims:

- preserved expected scale: mechanism;
- dropped units receive zero gradient: mechanism;
- deterministic evaluation: mechanism;
- lower validation loss: outcome;
- higher test accuracy: outcome;
- class-specific recall difference: outcome.

A mechanism can be implemented correctly and still fail to improve a task. Outcome gains can also result from errors. Require both.

### Controlled-experiment assignment

A strong submission includes:

- directional pre-run hypothesis;
- one changed factor;
- verified fixed controls;
- exact repository and data state;
- three or more seeds when stochastic variation matters;
- raw machine-readable results;
- table or curve generated from raw results;
- class or subgroup effects;
- honest test status;
- restrained claim; and
- one next controlled change.

## Common misconceptions

| Misconception | Response |
| --- | --- |
| More layers prove better learning | They increase capacity; validation evidence tests usefulness |
| A validation gap names its cause | It identifies a pattern that needs further checks |
| ReLU guarantees gradients remain healthy | It improves some local paths but other factors still matter |
| Regularization fixes an invalid split | Split validity must be repaired directly |
| Larger $\lambda$ is always safer | Excessive regularization can underfit |
| Early stopping is just fewer epochs | It is a predeclared validation-controlled selection rule |
| The last model is the best model | Restore the saved best checkpoint |
| Dropout uses zeros and ones only | Inverted dropout uses zero or $1/q$ |
| A new mask can be used in backward | Backward differentiates the sampled forward graph |
| Test-time dropout improves robustness automatically | It changes the evaluation model and requires a separate declared procedure |
| One seed is sufficient research evidence | Stochastic comparisons usually need repeated or uncertainty-aware results |

## Formative feedback language

- “Describe the curve before naming the cause.”
- “Which evidence separates underfitting from optimization failure?”
- “Which local derivative is active?”
- “What parameter receives the penalty?”
- “Which checkpoint is restored?”
- “What is the keep probability?”
- “Where was this mask created?”
- “Is the model in training or evaluation mode?”
- “Which single factor changed?”
- “Which split selected this value?”
- “How large is the effect relative to seed variation?”
- “What claim remains unsupported?”

## Adaptations

### Ninety-minute class

Use slides 1–8, 11–19, and 22. Complete one curve diagnosis, ReLU derivative, early-stopping rule, inverted-dropout expectation, mask backward trace, and lab interpretation. Assign $L_2$ derivation and the controlled-experiment design afterward.

### Online delivery

Give groups different curve patterns and dropout masks. Require each group to state observations, calculate one forward and backward mask result, and identify the exact evidence needed next.

### Limited calculus preparation

Use ReLU as a gate and dropout as a sampled scaled gate. Require signs, active paths, expected scale, and saved-mask reasoning. Treat the $L_2$ derivative as a guided result.

## After class

- check whether project curves use deterministic evaluation mode;
- reject comparisons that change several intended factors without qualification;
- verify that early stopping restores the saved best state;
- check raw per-seed results rather than only means;
- inspect class or subgroup regressions;
- require provisional labeling after test-driven tuning; and
- connect the need for structured representations to convolution in Week 9.
