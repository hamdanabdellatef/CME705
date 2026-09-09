# Week 11 slides — RNNs, LSTMs, Attention, and Transformers

This accessible source mirrors the PowerPoint deck. Equations use GitHub-compatible dollar delimiters.

## Slide 1: RNNs, LSTMs, attention, and Transformers

**Guiding question:** How should a model store, retrieve, and evaluate evidence when order and distance matter?

CME705 Machine Learning — Week 11

Speaker cue: Connect recurrence to the chain rule from Week 6 and gradient flow from Week 8.

## Slide 2: Today’s evidence

Students will produce:

1. a legal-context statement for a sequence task;
2. an unrolled recurrent computation;
3. a recurrent-gradient explanation;
4. a complete LSTM gate calculation;
5. a delayed-memory comparison;
6. a masked attention calculation;
7. a Transformer block trace;
8. a benchmark-qualified research direction.

## Slide 3: Sequence tasks begin with prediction time

| Mapping | Example | Legal context |
| --- | --- | --- |
| sequence-to-one | document classification | complete document |
| sequence-to-sequence | per-step activity label | task dependent |
| autoregressive | next-token prediction | past and current only |
| forecasting | future demand | observations available by forecast origin |

A bidirectional model leaks if the application must predict online.

## Slide 4: Split the process, not arbitrary rows

- keep entities disjoint when one person or machine yields many sequences;
- split chronologically for future forecasting;
- prevent overlapping windows from crossing partitions;
- fit vocabulary and preprocessing on training data;
- state the locked test period or official split.

Question: what future information would be illegal in your research problem?

## Slide 5: Vanilla recurrent state

$$
a_t = W_{xh}x_t + W_{hh}h_{t-1} + b_h
$$

$$
h_t = \tanh(a_t)
$$

$$
z = W_{hy}h_T + b_y
$$

The same transition parameters are reused at every step.

## Slide 6: Unroll the shared computation

$$
h_1 \rightarrow h_2 \rightarrow \cdots \rightarrow h_T
$$

Each state receives the current observation and the previous state. The final state is a fixed-width summary in a many-to-one classifier.

Prompt: which arrows create the path from an early cue to the loss?

## Slide 7: Backpropagation through time

$$
\frac{\partial L}{\partial h_t}
=
\frac{\partial L}{\partial h_T}
\prod_{k=t+1}^{T}
\frac{\partial h_k}{\partial h_{k-1}}
$$

For tanh:

$$
\frac{\partial h_k}{\partial h_{k-1}}
=
\operatorname{diag}(1-h_k^2)W_{hh}.
$$

## Slide 8: Repeated factors change gradient scale

$$
0.8^{30} \approx 0.00124
$$

$$
1.2^{30} \approx 237.4
$$

- norms below one can erase early credit;
- norms above one can create unstable updates;
- clipping controls explosion after it appears;
- gating changes the path through memory.

## Slide 9: LSTM adds an explicit cell state

The hidden state exposes current output. The cell state carries gated memory.

- write through the input gate;
- retain through the forget gate;
- propose content through the candidate;
- reveal through the output gate.

## Slide 10: LSTM gate equations

$$
\begin{aligned}
i_t &= \sigma(W_i x_t + U_i h_{t-1} + b_i), \\
f_t &= \sigma(W_f x_t + U_f h_{t-1} + b_f), \\
g_t &= \tanh(W_g x_t + U_g h_{t-1} + b_g), \\
o_t &= \sigma(W_o x_t + U_o h_{t-1} + b_o).
\end{aligned}
$$

## Slide 11: Cell update and gradient path

$$
c_t = f_t \odot c_{t-1} + i_t \odot g_t
$$

$$
h_t = o_t \odot \tanh(c_t)
$$

Holding the gates fixed locally:

$$
\frac{\partial c_t}{\partial c_{t-1}} = f_t.
$$

A forget gate near one can preserve information; it does not guarantee useful memory.

## Slide 12: PyTorch shapes and parameters

- Input: $(N,T,D)$
- Hidden and cell: $(L,N,H)$
Logits: $(N,C)$

| Model, $D=4$, $H=32$, $C=3$ | Parameters |
| --- | ---: |
| vanilla RNN plus head | 1,315 |
| LSTM plus head | 4,963 |

Compare equal width and parameter-matched variants before generalizing.

## Slide 13: Delayed-memory protocol

- first step: one-hot class cue plus cue flag;
- later steps: low-amplitude noise;
- training: 1,200 sequences of length 30;
- validation: 400 sequences of length 30;
- locked stress test: 600 sequences of length 60;
- identical seed, optimizer, width, batches, epochs, and selection rule.

Majority baseline: 0.333.

## Slide 14: Validation can hide a memory failure

| Model | Validation loss | Validation accuracy |
| --- | ---: | ---: |
| vanilla RNN | 0.0005 | 1.000 |
| LSTM | 0.0005 | 1.000 |

Both models solve the matched-length problem. Model selection alone does not test length extrapolation.

## Slide 15: Locked longer-sequence evidence

| Model | $T=30$ validation | $T=60$ locked stress |
| --- | ---: | ---: |
| vanilla RNN | 1.000 | 0.000 |
| LSTM | 1.000 | 1.000 |
| majority baseline | 0.333 | 0.333 |

Supported claim: the LSTM retained the early cue when the delay doubled in this declared run.

## Slide 16: Attention changes the access path

Recurrence asks a state to compress the past. Attention lets each query directly weight stored values.

$$
Q=XW_Q, \qquad K=XW_K, \qquad V=XW_V
$$

$$
S = \frac{QK^\top}{\sqrt{d_k}}
$$

## Slide 17: Scaled dot-product attention

$$
A = \operatorname{softmax}(S)
$$

$$
O = AV
$$

Each row of $A$ sums to one. The course code returns both $O$ and $A$ so shapes, normalization, and masks can be tested.

## Slide 18: Masks define legal access

$$
A
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}} + M
\right)
$$

- padding mask: block artificial pad tokens;
- causal mask: block future positions;
- bidirectional encoder: permit both directions in a complete input.

The audit confirms every row sums to 1.000 and blocked causal weights are 0.000.

## Slide 19: Multi-head attention

$$
\operatorname{head}_r
=
\operatorname{Attention}
(XW_Q^{(r)},XW_K^{(r)},XW_V^{(r)})
$$

$$
\operatorname{MHA}(X)
=
\operatorname{Concat}(\operatorname{head}_1,\ldots,\operatorname{head}_H)W_O
$$

Several learned relation subspaces are available. A head is not automatically a human concept.

## Slide 20: Transformer encoder block

A common pre-normalization form is

$$
X' = X + \operatorname{MHA}(\operatorname{LN}(X))
$$

$$
Y = X' + \operatorname{FFN}(\operatorname{LN}(X')).
$$

Attention mixes positions. The feed-forward network transforms each position. Residual paths support optimization.

## Slide 21: Position must enter the model

Self-attention alone is permutation-equivariant.

Position choices include:

- fixed sinusoidal encoding;
- learned absolute embeddings;
- relative position bias;
- rotary position embeddings;
- local, recurrent, or state-space structure.

Configured context length is not the same as effective context length.

## Slide 22: Architecture tradeoffs

| Family | Access | Time parallelism | Sequence cost |
| --- | --- | --- | --- |
| RNN | recurrent state | no | linear in length |
| LSTM / GRU | gated state | no | linear in length |
| full attention | all pairs | yes | quadratic attention |
| sparse attention | selected pairs | yes | pattern dependent |
| state space | state plus scan/convolution | often | commonly near-linear |

Measure latency and memory on the target hardware.

## Slide 23: Long Range Arena benchmark

Reported test accuracy under LRA protocols:

| Model | Family | Average | Path-X |
| --- | --- | ---: | ---: |
| Transformer | full attention | 53.66 | failed |
| S4 | structured state space | 86.09 | 96.35 |
| S5 | multi-input state space | 87.46 | 98.58 |
| Mega | gated attention plus moving average | 88.21 | 97.98 |

Sources: S4 Table 4, S5 Table 1, Mega Table 2. Check versions, scale, compute, and uncertainty before reusing a number.

## Slide 24: Selected recent sequence architectures

| Year | Model | Main idea |
| ---: | --- | --- |
| 2023 | Mamba | input-dependent selective state spaces |
| 2024 | xLSTM | exponential gates and scalar or matrix memory |
| 2024 | ModernBERT | modern long-context bidirectional encoder |
| 2025 | Titans | attention with learned long-term neural memory |
| 2026 | LongBench Pro | realistic bilingual long-context evaluation |

This is a dated research map, not one cross-task leaderboard.

## Slide 25: Real application — ModernBERT on AG News

- 149-million-parameter ModernBERT-base;
- World, Sports, Business, and Sci/Tech labels;
- 2,000 balanced training examples;
- 500 balanced validation examples;
- 1,000 locked official-test examples;
- head-only baseline, optional last-layer fine-tuning;
- automatic CUDA and mixed precision;
- checkpoint selected by validation loss.

Record class errors, latency, memory, model revision, and data-use constraints.

## Slide 26: Research direction and exit ticket

Complete a sequence-model research-direction note:

1. prediction time and legal context;
2. baseline and candidate model;
3. benchmark or task metric;
4. length, latency, memory, or shift stress;
5. result that would reject the direction.

Exit ticket: explain recurrent gradient difficulty, the LSTM cell path, attention-based access, and one limitation of today’s evidence.
