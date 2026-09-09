# Instructor guide — Week 11

## Purpose

Week 11 should leave students with one connected story:

1. recurrence stores history in a fixed-width state;
2. backpropagation through time creates a long multiplicative path;
3. LSTM gates add a controllable additive memory path;
4. attention provides direct content-dependent access;
5. Transformers organize attention into a scalable block;
6. recent models revisit the tradeoff among access, state, sequence length, and hardware;
7. benchmark claims require common tasks and declared protocols.

Avoid presenting Transformers as a reason to skip recurrent mathematics. The recurrent derivation makes memory, credit assignment, streaming, and newer state-space work easier to understand.

## Preparation checklist

Before class:

- install the pinned requirements in a fresh Python 3.11 or 3.12 environment;
- verify CUDA with the command in setup.md;
- run the delayed-memory lab once on the teaching machine;
- run the ModernBERT audit;
- if using the real-data lab live, pre-download AG News and ModernBERT-base through the script;
- record the exact model revision if the Hugging Face cache is pinned;
- keep the official test result hidden until students freeze the protocol;
- print or distribute the worksheet.

The full ModernBERT base model has about 149 million parameters. The default lab freezes the encoder and trains only the prediction and classification heads. On a 6 GB laptop GPU, keep maximum length 128 and reduce the batch size if memory is insufficient. The optional last-two-layer stage uses batch size 4 in the example command.

## Suggested timing

| Segment | Minutes | Teaching move |
| --- | ---: | --- |
| prediction time and splits | 15 | challenge one online and one offline example |
| vanilla RNN | 25 | trace shapes and shared parameters |
| BPTT | 20 | calculate repeated scalar factors |
| LSTM | 30 | calculate gates before showing code |
| delayed-memory lab | 25 | predict, run, interpret |
| break | 10 | — |
| attention | 25 | calculate one two-token example |
| Transformer | 20 | trace residual paths and masks |
| benchmark and recent models | 15 | qualify one score |
| ModernBERT lab | 20 | audit protocol and start a run |
| research direction | 15 | convert an interest into a falsifiable test |

If the class meets twice, end meeting one after the delayed-memory interpretation.

## Opening questions

Ask students to classify each task.

1. Sentiment after reading a completed review: sequence-to-one; bidirectional context can be legal.
2. Predict the next machine reading during operation: forecasting; future readings are illegal.
3. Label every token in an archived document: sequence-to-sequence; bidirectional context may be legal.
4. Generate a token one step at a time: autoregressive; use a causal mask.

The key answer is the legal context at prediction time, not the model name.

## Board derivation — vanilla RNN

Write:

$$
a_t = W_{xh}x_t + W_{hh}h_{t-1}+b_h,
$$

$$
h_t = \tanh(a_t).
$$

Draw three time steps with the same $W_{xh}$ and $W_{hh}$. Ask students to identify parameter sharing.

Worksheet shape answers for $N=16$, $T=20$, $D=5$, $H=8$, $C=3$:

| Quantity | Shape |
| --- | --- |
| input | $(16,20,5)$ |
| $W_{xh}$ | $(8,5)$ |
| $W_{hh}$ | $(8,8)$ |
| returned hidden sequence | $(16,20,8)$ |
| final hidden tensor | $(1,16,8)$ |
| logits | $(16,3)$ |

Parameter count:

$$
8(5) + 8^2 + 2(8) + 3(8) + 3 = 147.
$$

PyTorch stores recurrent matrices using its own names and includes two bias vectors. Encourage students to reconcile formulas with named_parameters.

## Board derivation — recurrent gradients

Build the chain:

$$
\frac{\partial L}{\partial h_t}
=
\frac{\partial L}{\partial h_T}
\frac{\partial h_T}{\partial h_{T-1}}
\cdots
\frac{\partial h_{t+1}}{\partial h_t}.
$$

Worksheet estimates:

$$
0.9^{50} \approx 0.00515,
\qquad
1.1^{50} \approx 117.39.
$$

Expected explanation: clipping limits a large gradient norm before an optimizer step. It cannot reconstruct an early signal that has already become extremely small.

Misconception to correct: vanishing gradients and short memory are related but not identical. A trained recurrent dynamical system can retain information, and an LSTM can still forget.

## Board derivation — LSTM

Reveal the equations in two stages.

First, calculate the gates:

$$
i_t, f_t, g_t, o_t.
$$

Then update:

$$
c_t = f_t \odot c_{t-1} + i_t \odot g_t,
$$

$$
h_t = o_t \odot \tanh(c_t).
$$

Worksheet scalar answer:

$$
c_t = 0.9(1.2)+0.8(-0.5)=0.68,
$$

$$
h_t = 0.7\tanh(0.68) \approx 0.414.
$$

The retained contribution is 1.08 and the written contribution is -0.40. The visible hidden state is approximately 0.414.

For a fixed forget value:

$$
0.95^{40} \approx 0.129.
$$

A gate close to one can still lose substantial magnitude across many steps.

## Delayed-memory lab

Ask for predictions before running:

    python labs/week11_sequence_pytorch.py

A checked CPU run with seed 705 produced:

| Evidence | RNN | LSTM |
| --- | ---: | ---: |
| parameters | 1,315 | 4,963 |
| selected epoch | 25 | 25 |
| validation loss | 0.0005 | 0.0005 |
| validation accuracy | 1.000 | 1.000 |
| locked length-60 stress accuracy | 0.000 | 1.000 |

The exact result can vary with PyTorch, hardware, and kernels. The intended learning point is that matched-length validation may not expose a memory failure. If both models succeed on the longer split, raise the stress length. If neither succeeds, confirm the seed and default configuration before changing the experiment.

Discuss comparison limits:

- unequal parameter count;
- one seed;
- generated task;
- fixed cue position;
- one optimizer and initialization;
- longer sequences than those used for validation selection.

A strong extension repeats seeds and tests several lengths while leaving model selection unchanged.

## Attention calculation answer

For

$$
Q=K=I_2,
$$

the scaled scores are

$$
S =
\begin{bmatrix}
0.707 & 0 \\
0 & 0.707
\end{bmatrix}.
$$

Row-wise softmax gives approximately

$$
A =
\begin{bmatrix}
0.670 & 0.330 \\
0.330 & 0.670
\end{bmatrix}.
$$

With

$$
V =
\begin{bmatrix}
2 & 0 \\
0 & 4
\end{bmatrix},
$$

the output is

$$
O \approx
\begin{bmatrix}
1.340 & 1.321 \\
0.660 & 2.679
\end{bmatrix}.
$$

Rows sum to one. Do not call the weights explanations without an intervention or another validation method.

## Causal-mask answer

Allowed positions for four tokens are

$$
M_{allowed}
=
\begin{bmatrix}
1&0&0&0\\
1&1&0&0\\
1&1&1&0\\
1&1&1&1
\end{bmatrix}.
$$

A next-token model must not train by reading the answer from a future token. A bidirectional mask is suitable when the complete input exists at prediction time, such as classifying an archived document.

## Transformer block answer

A common pre-normalization block is

$$
X' = X + \operatorname{MHA}(\operatorname{LN}(X)),
$$

$$
Y = X' + \operatorname{FFN}(\operatorname{LN}(X')).
$$

Expected roles:

| Component | Role |
| --- | --- |
| attention | mixes information across positions |
| feed-forward network | transforms each position |
| residual path | supplies a direct update path |
| layer normalization | stabilizes representation scale |
| position information | distinguishes order |

## Benchmark discussion

The LRA slide uses numbers reported in three primary papers. Students should not conclude that Mega, S5, or S4 wins every sequence task. Ask them to identify:

- the task suite;
- the metric direction;
- input length;
- model scale;
- training budget;
- number of seeds;
- whether the baseline was rerun or copied;
- whether the result transfers to their domain.

The 2026 LongBench Pro reading is useful because it distinguishes configured context from effective context and emphasizes realistic long documents.

## ModernBERT lab

Start with:

    python labs/week11_modernbert_news.py --audit-only

The audit starts no downloads. The full run uses:

    python labs/week11_modernbert_news.py

The default data counts are balanced:

- training: 2,000;
- validation: 500;
- locked test: 1,000.

The model is appropriate for a real classification application because it is a bidirectional encoder with a classification head. It is not a generative chatbot exercise.

Ask students to report:

- exact model identifier and revision;
- GPU name and precision;
- trainable and total parameters;
- split construction;
- class support;
- selected checkpoint;
- confusion matrix and per-class recall;
- ten inspected errors;
- runtime and memory;
- data and model licenses or access conditions.

The model card uses Apache 2.0 for ModernBERT. The AG News dataset card reports unknown license metadata and describes research use. No dataset files, model weights, or checkpoints should be committed.

## Research-direction workshop

Reject vague topics such as “use Transformers for health.” Require:

> On participant-disjoint wearable-sensor sequences, does a parameter-matched LSTM improve macro-F1 at delays above 100 steps relative to a vanilla RNN, without exceeding a 20 ms mobile latency budget?

A complete direction states:

- prediction time;
- split unit;
- baseline;
- model change;
- metric;
- stress variable;
- resource constraint;
- rejection condition.

## Exit-ticket criteria

A complete answer says:

- the recurrent gradient contains a product across time;
- repeated factors can shrink or grow;
- the LSTM cell supplies a gated additive path;
- attention creates direct weighted access;
- one Week 11 result is limited by task, seed, parameter count, model revision, dataset conditions, or evaluation scope.
