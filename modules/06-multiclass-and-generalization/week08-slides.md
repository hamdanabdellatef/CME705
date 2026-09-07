# Week 8 slide source: Deep networks and generalization

This file is the accessible text version of the PowerPoint deck. Speaker prompts and timing are in the presentation and instructor guide.

## Slide 1: CME705 Machine Learning

**Week 8: Deep networks and generalization**

Guiding question: When does added capacity learn useful structure, and when does it memorize training detail?

## Slide 2: Today’s evidence

Each student will produce:

1. a curve-based diagnosis;
2. a ReLU forward/backward calculation;
3. an $L_2$ gradient and update;
4. an early-stopping checkpoint;
5. an inverted-dropout expectation calculation;
6. a saved-mask backward trace; and
7. a one-factor controlled experiment.

## Slide 3: Depth changes capacity, not evidence

A deeper network composes more transformations:

$$
H_\ell=\phi_\ell(H_{\ell-1}W_\ell+b_\ell).
$$

More capacity can represent useful structure and training noise.

Validation evidence determines whether the added capacity helps the intended task.

## Slide 4: Three problems require different responses

| Pattern | Training evidence | Validation evidence | First response |
| --- | --- | --- | --- |
| Underfitting | poor | poor | representation, features, or optimization |
| Overfitting | improves | worsens or stalls | regularization, data, or simpler capacity |
| Distribution shift | may look good | deployment subgroup fails | revise data and evaluation design |

Do not call every weak validation result overfitting.

## Slide 5: ReLU keeps one side active

$$
\operatorname{ReLU}(z)=\max(0,z).
$$

$$
\frac{d}{dz}\operatorname{ReLU}(z)=
\begin{cases}
1,&z>0,\\
0,&z\le0.
\end{cases}
$$

ReLU avoids sigmoid saturation on its positive side. It can still produce inactive units and does not eliminate all optimization problems.

## Slide 6: Gradient problems accumulate through depth

Backpropagation multiplies local derivatives and weights across layers.

Repeated factors smaller than one can shrink gradients. Repeated large factors can amplify them.

Activation, initialization, normalization, architecture, and optimization interact.

## Slide 7: Read training and validation curves together

- Both losses high: possible underfitting or optimization failure.
- Training falls while validation stalls: growing generalization gap.
- Both fall: useful learning under the current split.
- Validation is unstable: inspect sample size, randomness, and subgroups.

A curve suggests where to investigate; it does not identify the cause alone.

## Slide 8: The generalization gap is a diagnostic

For a loss:

$$
g=J_{\text{validation}}-J_{\text{training}}.
$$

A growing positive gap can indicate memorization or a mismatch between the splits.

Compare checkpoints under the same data and metric. The gap itself is not the final research objective.

## Slide 9: $L_2$ regularization penalizes large weights

$$
J_{\text{reg}}=
J_{\text{data}}+\frac{\lambda}{2}\sum_\ell\lVert W_\ell\rVert_F^2.
$$

Then:

$$
\nabla_{W_\ell}J_{\text{reg}}
=
\nabla_{W_\ell}J_{\text{data}}+\lambda W_\ell.
$$

The coefficient $\lambda$ is selected with validation evidence.

## Slide 10: Weight shrinkage changes the update

With ordinary gradient descent:

$$
W\leftarrow
(1-\eta\lambda)W-\eta\nabla_WJ_{\text{data}}.
$$

Biases are often excluded from the penalty by explicit design.

An objective penalty and optimizer-specific decoupled weight decay should be named accurately.

## Slide 11: Early stopping is validation-controlled checkpoint selection

1. Define the monitored validation quantity.
2. Define improvement tolerance and patience.
3. Save parameters when validation improves.
4. Stop after patience is exhausted.
5. Restore the best saved parameters.

The last epoch is not automatically the best model.

## Slide 12: Dropout samples a training-time subnetwork

For keep probability $q=1-p$:

$$
M_{ij}\sim\operatorname{Bernoulli}(q).
$$

$$
\widetilde H=
\frac{M}{q}\odot H.
$$

Dropped activations are zero. Kept activations are scaled by $1/q$.

## Slide 13: Inverted scaling preserves the expectation

$$
\mathbb{E}\left[\frac{M}{q}\odot H\right]=H.
$$

The expected activation scale matches evaluation mode.

The Week 8 lab measures $0.8000$ before dropout and approximately $0.8002$ after a random training mask.

## Slide 14: Backpropagation reuses the same mask

If:

$$
\widetilde H=\frac{M}{q}\odot H,
$$

then:

$$
G_H=
G_{\widetilde H}\odot\frac{M}{q}.
$$

A unit dropped during the forward pass receives zero gradient in that pass.

A newly sampled backward mask would describe a different computation.

## Slide 15: Training and evaluation modes differ

| Behavior | Training | Evaluation |
| --- | --- | --- |
| Dropout mask | sampled | disabled |
| Kept-unit scaling | $1/q$ | none |
| Output for fixed input | stochastic | deterministic |
| Purpose | regularized fitting | stable prediction |

Evaluation-time dropout changes the model being measured.

## Slide 16: One controlled variable

The Week 8 lab holds fixed:

- generated data and split;
- training-label noise;
- standardization;
- architecture and initialization;
- learning rate and maximum epochs;
- early-stopping rule; and
- evaluation metric.

Only the dropout probability changes: $0$, $0.15$, $0.30$, or $0.45$.

## Slide 17: Validation curves select the candidate

| Dropout | Best epoch | Training loss | Validation loss | Gap |
| ---: | ---: | ---: | ---: | ---: |
| 0.00 | 81 | 0.2679 | 0.7180 | 0.4500 |
| 0.15 | 119 | 0.2670 | 0.6476 | 0.3805 |
| 0.30 | 232 | 0.2461 | 0.5920 | 0.3459 |
| 0.45 | 513 | 0.2429 | 0.5590 | 0.3161 |

Validation selects dropout $0.45$ for this deterministic teaching run.

## Slide 18: Open the test split after selection

| Fixed model | Test loss | Accuracy |
| --- | ---: | ---: |
| Dropout $0.00$ reference | 0.6746 | 0.633 |
| Validation-selected dropout $0.45$ | 0.5649 | 0.733 |

The selected model’s negative recall is $0.661$ and positive recall is $0.810$.

The result supports this comparison on one synthetic split.

## Slide 19: Correct dropout mechanics are independently checked

The lab reports:

- activation mean $0.8000$ before dropout;
- training-mode mean approximately $0.8002$;
- dropped fraction approximately $0.2498$ for $p=0.25$;
- zero evaluation-mode difference;
- an all-ones evaluation mask;
- backward-gradient mean $0.1600$; and
- zero nonzero gradients at dropped positions.

Mechanism checks and generalization evidence answer different questions.

## Slide 20: Regularization does not repair invalid evaluation

Dropout, weight decay, and early stopping can reduce fitting to training detail.

They cannot repair:

- target leakage;
- duplicated subjects across splits;
- incorrect class mapping;
- a metric unrelated to the research question; or
- test-driven model selection.

Fix the experimental design before tuning regularization.

## Slide 21: Controlled-experiment assignment

Record:

- one directional hypothesis;
- one changed factor and all fixed controls;
- data and split identity;
- seeds and repeated runs or another uncertainty estimate;
- validation selection rule;
- locked test status;
- class or subgroup evidence;
- implementation checks;
- result table or figure; and
- supported claim, limitation, and next decision.

## Slide 22: Exit and next step

Explain:

1. how underfitting differs from overfitting;
2. why the best checkpoint may precede the last epoch;
3. why inverted dropout divides by keep probability;
4. why backward uses the saved forward mask; and
5. why a controlled comparison changes one factor.

Next: convolution, spatial structure, and learned feature maps.
