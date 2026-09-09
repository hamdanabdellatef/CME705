# Week 11 worksheet — Sequence memory and attention

Name: ____________________
Research topic or application: ____________________

## 1. Define the sequence task

Describe one example from your research interests.

| Item | Your answer |
| --- | --- |
| one observation $x_t$ | |
| ordering variable | |
| target | |
| prediction time | |
| context legally available at prediction time | |
| sequence-to-one, sequence-to-sequence, autoregressive, or forecasting | |
| entity or time unit that must remain disjoint across splits | |

Write one sentence describing a possible future-information leak:

____________________________________________________________________

## 2. Trace a vanilla RNN

For input width $D=5$, hidden width $H=8$, three classes, batch size $N=16$, and sequence length $T=20$, complete the shapes.

| Quantity | Shape |
| --- | --- |
| input batch $X$ | |
| $W_{xh}$ | |
| $W_{hh}$ | |
| hidden sequence returned by PyTorch | |
| final hidden state | |
| classifier logits | |

Write the recurrence:

$$
a_t =
$$

$$
h_t =
$$

Count the parameters, including two recurrent bias vectors and the classification head:

$$
HD + H^2 + 2H + CH + C =
$$

## 3. Explain gradient instability

Complete the path:

$$
\frac{\partial L}{\partial h_t}
=
\frac{\partial L}{\partial h_T}
\prod_{k=___}^{___}
__________________.
$$

Calculate or estimate:

$$
0.9^{50} \approx ________,
\qquad
1.1^{50} \approx ________.
$$

Explain why gradient clipping addresses one side of the problem but not the other:

____________________________________________________________________

## 4. Calculate one LSTM step

Use scalar values:

$$
i_t=0.8, \quad f_t=0.9, \quad g_t=-0.5,
\quad o_t=0.7, \quad c_{t-1}=1.2.
$$

Calculate:

$$
c_t = f_t c_{t-1} + i_t g_t =
$$

$$
h_t = o_t \tanh(c_t) =
$$

Interpret the result.

- retained contribution from old memory: ____________________
- written contribution from candidate memory: ____________________
- visible hidden output: ____________________

If $f=0.95$ for 40 steps, what proportion remains on the direct cell path?

$$
0.95^{40} \approx ________.
$$

## 5. Predict the delayed-memory experiment

Before running the code, predict validation and locked stress behavior.

| Model | Length-30 validation prediction | Length-60 stress prediction | Reason |
| --- | --- | --- | --- |
| vanilla RNN | | | |
| LSTM | | | |

Run:

    python labs/week11_sequence_pytorch.py

Record:

| Evidence | Vanilla RNN | LSTM |
| --- | ---: | ---: |
| parameters | | |
| selected epoch | | |
| validation loss | | |
| validation accuracy | | |
| locked stress loss | | |
| locked stress accuracy | | |
| cue input-gradient magnitude | | |

Does the result support your prediction? State the narrowest defensible claim.

____________________________________________________________________

Name one confound in the comparison:

____________________________________________________________________

Design one controlled extension:

____________________________________________________________________

## 6. Calculate scaled dot-product attention

Let

$$
Q = K =
\begin{bmatrix}
1 & 0 \\
0 & 1
\end{bmatrix},
\qquad
V =
\begin{bmatrix}
2 & 0 \\
0 & 4
\end{bmatrix}.
$$

1. Calculate $QK^\top$.
2. Divide by $\sqrt{2}$.
3. Apply row-wise softmax.
4. Calculate $AV$.

Your score matrix:

$$
S =
$$

Your attention matrix:

$$
A =
$$

Your output:

$$
O =
$$

Check each row sum: ____________________

## 7. Build a causal mask

Fill the allowed-access matrix for four tokens. Use 1 for allowed and 0 for blocked.

$$
M_{allowed}
=
\begin{bmatrix}
_ & _ & _ & _ \\
_ & _ & _ & _ \\
_ & _ & _ & _ \\
_ & _ & _ & _
\end{bmatrix}.
$$

Why must a next-token predictor block future tokens?

____________________________________________________________________

When would bidirectional attention be appropriate?

____________________________________________________________________

## 8. Trace a Transformer encoder block

Complete the common pre-normalization equations:

$$
X' = X + __________________
$$

$$
Y = X' + __________________.
$$

Match each component to its main role.

| Component | Role |
| --- | --- |
| attention | |
| feed-forward network | |
| residual path | |
| layer normalization | |
| position information | |

## 9. Read the benchmark table critically

Choose one LRA result from the slides.

| Field | Record |
| --- | --- |
| paper and table | |
| model | |
| task or average | |
| metric direction | |
| sequence length | |
| model scale | |
| pretraining or augmentation | |
| number of seeds or uncertainty | |
| copied or reproduced baseline | |
| claim the number supports | |
| claim the number does not support | |

## 10. ModernBERT real-application lab

Run the download-free audit first:

    python labs/week11_modernbert_news.py --audit-only

Then, in the prepared environment:

    python labs/week11_modernbert_news.py

Record:

| Item | Evidence |
| --- | --- |
| GPU or CPU | |
| model identifier and revision | |
| data source and access conditions | |
| train / validation / locked test sizes | |
| maximum token length | |
| trainable parameters | |
| selected epoch | |
| validation metric | |
| locked test metric | |
| runtime and peak memory | |

Inspect at least ten errors.

| Error category | Count | Example ID | Hypothesis |
| --- | ---: | --- | --- |
| | | | |
| | | | |
| | | | |

Do not infer that confidence is calibration. Propose one calibration or robustness check:

____________________________________________________________________

## 11. Research-direction note

Write a falsifiable question using this form:

> On ____________________ split sequences, does ____________________ improve
> ____________________ relative to ____________________ when
> ____________________ changes, without exceeding ____________________?

State the result that would make you reject the direction:

____________________________________________________________________

## Exit ticket

In at most 150 words, explain:

- why a recurrent Jacobian product can make early evidence hard to learn;
- how an LSTM changes the memory path;
- how attention changes access to distant evidence;
- one limitation of the Week 11 experiments.
