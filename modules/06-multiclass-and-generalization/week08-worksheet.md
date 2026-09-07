# Week 8 worksheet: Deep networks and generalization

Name: ________________________________  Date: __________________

Use validation evidence to select regularization or a checkpoint. Keep the test split locked until the comparison is fixed.

## 1. Capacity and evidence

Complete:

A deeper network composes _______________________________________________

More capacity can fit useful structure and ______________________________

Evidence that added capacity helps must come from _______________________

Why is training accuracy alone insufficient? ____________________________

## 2. Diagnose learning curves

For each pattern, choose the best initial diagnosis: underfitting or optimization failure, useful learning, overfitting or mismatch, unstable evaluation.

| Training curve | Validation curve | Initial diagnosis | Next check |
| --- | --- | --- | --- |
| high and flat | high and flat | | |
| decreasing | decreasing | | |
| decreasing | flat | | |
| decreasing | increasing | | |
| smooth | highly variable | | |

Why can a curve suggest a problem without proving its cause?

__________________________________________________________________________

## 3. ReLU forward pass

For:

$$
z=[-2,-0.1,0,0.5,3],
$$

compute:

$$
\operatorname{ReLU}(z)=
\underline{\hspace{10cm}}.
$$

Which components remain active? ________________________________________

At $z=0$, which derivative convention does the Week 8 lab use? __________

## 4. ReLU backward pass

Let:

$$
z=[-1,0,2],
$$

and upstream gradient:

$$
g=[3,4,5].
$$

Complete:

$$
\frac{d\operatorname{ReLU}}{dz}
=
\underline{\hspace{6cm}},
$$

and:

$$
G_z=
g\odot
\frac{d\operatorname{ReLU}}{dz}
=
\underline{\hspace{6cm}}.
$$

Why does ReLU help some gradient paths but not guarantee successful optimization?

__________________________________________________________________________

## 5. Gradients through depth

Complete the schematic backward relation:

$$
\frac{\partial J}{\partial H_{\ell-1}}
=
\frac{\partial J}{\partial H_\ell}
\;\underline{\hspace{4cm}}\;
\underline{\hspace{3cm}}.
$$

Repeated factors below one may cause ___________________________________

Repeated large factors may cause _______________________________________

Name two other choices that affect gradient behavior.

1. ___________________________________________________________________
2. ___________________________________________________________________

## 6. Generalization gap

For loss, define:

$$
g_t=
\underline{\hspace{8cm}}.
$$

At one checkpoint:

- training loss: $0.25$;
- validation loss: $0.62$.

The gap is: __________________

What might a growing positive gap indicate? _____________________________

Why is the gap not a replacement for the primary research metric?

__________________________________________________________________________

## 7. $L_2$ regularization

Complete:

$$
J_{\text{reg}}
=
J_{\text{data}}
+
\underline{\hspace{8cm}}.
$$

Differentiate with respect to one weight matrix:

$$
\nabla_WJ_{\text{reg}}
=
\underline{\hspace{8cm}}.
$$

For $\lambda=0.02$ and:

$$
W=
\begin{bmatrix}
2 & -1 \\
0.5 & 3
\end{bmatrix},
$$

compute the penalty-gradient contribution:

$$
\lambda W=
\underline{\hspace{8cm}}.
$$

## 8. Weight update

Complete the ordinary gradient-descent update:

$$
W\leftarrow
(1-\underline{\hspace{2cm}})W
-
\underline{\hspace{4cm}}.
$$

Which parameters will be penalized in your project? _____________________

Why should the implementation be described rather than only called “weight decay”?

__________________________________________________________________________

## 9. Early stopping rule

Specify a complete rule.

| Decision | Value |
| --- | --- |
| monitored quantity | |
| direction | |
| minimum improvement | |
| patience | |
| maximum epochs | |
| evaluation frequency | |
| tie rule | |
| restored state | |

A run has best validation loss at epoch 81 and stops at epoch 261.

Which parameters should be returned? ___________________________________

Why are the epoch-261 parameters not automatically the selected model?

__________________________________________________________________________

## 10. Inverted dropout mask

Let drop probability be:

$$
p=0.25.
$$

Then keep probability is:

$$
q=\underline{\hspace{2cm}}.
$$

The mask values are:

- dropped: __________________
- kept: _____________________

If:

$$
H=[0.8,0.8,0.8,0.8],
$$

and the Bernoulli outcomes are $[1,0,1,1]$, compute:

$$
\widetilde H=
\frac{M}{q}\odot H
=
\underline{\hspace{8cm}}.
$$

## 11. Expected activation scale

Complete:

$$
\mathbb{E}
\left[
\frac{M}{q}\odot H
\right]
=
\underline{\hspace{5cm}}.
$$

Why is the sample mean after one finite random mask only approximately equal to the original mean?

__________________________________________________________________________

What scaling is applied at evaluation time with inverted dropout? _______

## 12. Dropout backward pass

Complete:

$$
G_H=
G_{\widetilde H}
\odot
\underline{\hspace{4cm}}.
$$

Using the same mask as Question 10 and upstream gradient $[1,1,1,1]$, compute:

$$
G_H=
\underline{\hspace{8cm}}.
$$

Why must the mask be saved from the forward pass? _______________________

What gradient does a dropped unit receive in that pass? _________________

## 13. Training and evaluation modes

Complete:

| Behavior | Training | Evaluation |
| --- | --- | --- |
| dropout mask | | |
| kept-unit scaling | | |
| fixed-input output | | |
| purpose | | |

What happens to experimental validity if test evaluation accidentally uses training mode?

__________________________________________________________________________

## 14. Identify the controlled variable

The Week 8 lab compares dropout rates $0$, $0.15$, $0.30$, and $0.45$.

List five fixed decisions.

1. ___________________________________________________________________
2. ___________________________________________________________________
3. ___________________________________________________________________
4. ___________________________________________________________________
5. ___________________________________________________________________

Changed factor: ________________________________________________________

Why would changing width and learning rate at the same time weaken the interpretation?

__________________________________________________________________________

## 15. Predict before running

The training set has only $96$ rows, thirty nuisance features, and $19$ flipped labels.

Predict:

- no-dropout training loss: high / low
- no-dropout validation gap: small / large
- effect of stronger dropout on the validation gap: _________________
- selected checkpoint relative to the stop epoch: earlier / same / later
- evaluation output for a fixed input: stochastic / deterministic

## 16. Run the NumPy lab

Run:

~~~bash
python labs/week08_dropout_numpy.py
~~~

Record the mechanism evidence.

| Evidence | Observed value |
| --- | --- |
| activation mean before dropout | |
| training-mode mean | |
| dropped fraction | |
| evaluation maximum difference | |
| evaluation mask all ones | |
| backward-gradient mean | |
| nonzero gradients at dropped positions | |

Record the controlled comparison.

| Dropout | Best epoch | Stop epoch | Training loss | Validation loss | Gap | Last validation loss |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00 | | | | | | |
| 0.15 | | | | | | |
| 0.30 | | | | | | |
| 0.45 | | | | | | |

Selected dropout: __________________

## 17. Locked test interpretation

Record:

| Fixed model | Test loss | Accuracy |
| --- | ---: | ---: |
| dropout $0$ reference | | |
| validation-selected dropout | | |

For the selected model:

- negative recall: __________________
- positive recall: __________________
- confusion matrix: ________________

What claim does this result support? ____________________________________

What claim does it not support? ________________________________________

## 18. Mechanism versus outcome evidence

Classify each statement as mechanism evidence or outcome evidence.

| Statement | Evidence type |
| --- | --- |
| dropped positions receive zero gradient | |
| validation loss is lower at $p=0.45$ | |
| evaluation output is deterministic | |
| expected activation scale is preserved | |
| locked test accuracy improves | |
| positive recall exceeds negative recall | |

Why are both evidence types needed? _____________________________________

## 19. Project controlled experiment

Complete:

| Item | Project decision |
| --- | --- |
| directional hypothesis | |
| baseline | |
| one changed factor | |
| fixed data and split | |
| fixed training decisions | |
| primary metric | |
| planned seeds | |
| uncertainty summary | |
| validation selection rule | |
| test status | |
| important class or subgroup | |
| contradicting result | |
| supported-claim template | |

## 20. Exit ticket

1. Underfitting differs from overfitting because _______________________
2. The best checkpoint may precede the last epoch because ______________
3. Inverted dropout divides by $q$ because _____________________________
4. Backward reuses the mask because ___________________________________
5. A controlled comparison changes one factor so that _________________
