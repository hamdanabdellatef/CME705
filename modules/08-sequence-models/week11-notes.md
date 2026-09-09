# Week 11 notes — From recurrent state to attention

## 1. What makes an input a sequence?

A sequence is more than a matrix with a time axis. The order, spacing, and prediction time carry meaning.

Let one example be

$$
X = (x_1, x_2, \ldots, x_T),
\qquad
x_t \in \mathbb{R}^{D}.
$$

A batch-first tensor has shape

$$
(N, T, D),
$$

where $N$ is batch size, $T$ is sequence length, and $D$ is the number of features per step.

Common mappings include:

| Task | Mapping | Example |
| --- | --- | --- |
| sequence-to-one | $(x_1,\ldots,x_T) \rightarrow y$ | classify a review |
| sequence-to-sequence | $(x_1,\ldots,x_T) \rightarrow (y_1,\ldots,y_T)$ | label each sensor step |
| encoder-decoder | $(x_1,\ldots,x_T) \rightarrow (y_1,\ldots,y_U)$ | translation |
| autoregressive | $x_{\le t} \rightarrow x_{t+1}$ | next-token prediction |
| forecasting | $x_{\le t} \rightarrow x_{t+1:t+H}$ | future demand |

Before selecting a model, define the prediction time. A bidirectional encoder can use left and right context for an already-complete document. It leaks future information if the application must predict online at time $t$.

### Split rules

Random row splitting is often invalid for sequences. Use:

- entity-aware splits when several sequences come from one person, machine, site, or document;
- chronological splits for future forecasting;
- no overlapping windows across train and test when they share raw observations;
- training-only vocabulary, scaling, imputation, and feature selection;
- a locked test period or official test split.

## 2. Vanilla recurrent neural networks

A basic recurrent neural network shares one transition across all time steps:

$$
a_t = W_{xh}x_t + W_{hh}h_{t-1} + b_h,
$$

$$
h_t = \tanh(a_t).
$$

For a sequence-to-one classifier,

$$
z = W_{hy}h_T + b_y,
\qquad
p = \operatorname{softmax}(z).
$$

Shapes for input width $D$, hidden width $H$, and $C$ classes are:

| Quantity | Shape |
| --- | --- |
| $x_t$ | $D$ |
| $h_t$ | $H$ |
| $W_{xh}$ | $H \times D$ |
| $W_{hh}$ | $H \times H$ |
| $W_{hy}$ | $C \times H$ |
| logits $z$ | $C$ |

The same matrices are used at every step. Parameter sharing lets one model process different sequence lengths, but the hidden state is a fixed-width bottleneck.

For a one-layer PyTorch RNN with two bias vectors, the parameter count is

$$
HD + H^2 + 2H + CH + C.
$$

With $D=4$, $H=32$, and $C=3$, this gives

$$
32(4) + 32^2 + 2(32) + 3(32) + 3 = 1{,}315.
$$

## 3. Backpropagation through time

Unrolling the recurrence creates a deep computation graph with shared parameters. If a loss at time $T$ depends on $h_T$, an early state receives

$$
\frac{\partial L}{\partial h_t}
=
\frac{\partial L}{\partial h_T}
\prod_{k=t+1}^{T}
\frac{\partial h_k}{\partial h_{k-1}}.
$$

For the tanh recurrence,

$$
\frac{\partial h_k}{\partial h_{k-1}}
=
\operatorname{diag}(1-h_k^2) W_{hh}.
$$

The product contains one factor per time step. If the typical singular values of the factors are below one, the gradient shrinks approximately exponentially. If they are above one, it may grow rapidly.

A scalar illustration makes the effect visible:

$$
0.8^{30} \approx 0.00124,
\qquad
1.2^{30} \approx 237.4.
$$

This is not a proof that every RNN must fail. Activation derivatives, weight directions, normalization, initialization, data, and optimization all matter. It explains why long credit-assignment paths can be difficult.

Common controls include:

- gradient clipping for exploding gradients;
- orthogonal or otherwise controlled recurrent initialization;
- shorter truncation windows, with an explicit loss of long context;
- gated recurrent units such as LSTMs or GRUs;
- residual, convolutional, attention, or state-space alternatives.

Clipping limits an update after a large gradient appears. It does not recover a signal that already vanished.

## 4. LSTM memory

An LSTM carries a hidden state $h_t$ and a cell state $c_t$. Concatenate the current input and previous hidden state conceptually, or write separate input and recurrent matrices:

$$
i_t = \sigma(W_i x_t + U_i h_{t-1} + b_i),
$$

$$
f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f),
$$

$$
g_t = \tanh(W_g x_t + U_g h_{t-1} + b_g),
$$

$$
o_t = \sigma(W_o x_t + U_o h_{t-1} + b_o).
$$

The cell and hidden states update as

$$
c_t = f_t \odot c_{t-1} + i_t \odot g_t,
$$

$$
h_t = o_t \odot \tanh(c_t).
$$

Interpret each component:

- $i_t$ controls how much candidate content is written;
- $f_t$ controls how much previous cell content is retained;
- $g_t$ proposes signed candidate content;
- $o_t$ controls how much cell content becomes visible;
- $c_t$ supplies an additive memory path.

Holding the gates fixed for one local derivative gives

$$
\frac{\partial c_t}{\partial c_{t-1}} = f_t.
$$

Across several steps, the direct cell path includes

$$
\frac{\partial c_T}{\partial c_t}
=
\prod_{k=t+1}^{T} f_k.
$$

When relevant forget-gate values stay near one, this path can preserve information and gradients. It is an opportunity, not a guarantee: a gate near zero deliberately erases memory, and the gates themselves must be learned.

### LSTM parameter count

An LSTM has four affine gate calculations. In PyTorch, a one-layer LSTM with two bias vectors has

$$
4HD + 4H^2 + 8H
$$

encoder parameters. Adding a $C$-class head gives

$$
4HD + 4H^2 + 8H + CH + C.
$$

For $D=4$, $H=32$, and $C=3$:

$$
4(32)(4) + 4(32^2) + 8(32) + 3(32) + 3 = 4{,}963.
$$

The larger capacity is part of any RNN–LSTM comparison. A strong experiment should compare both equal hidden widths and approximately equal parameter budgets.

## 5. Week 11 delayed-memory experiment

The first input step contains:

- a one-hot class cue in three channels;
- a cue-present flag in a fourth channel.

Later steps contain only low-amplitude noise. Training and validation use $T=30$. The locked stress split uses $T=60$.

The procedure fixes:

- seed 705;
- 1,200 training, 400 validation, and 600 locked stress examples;
- hidden width 32;
- batch size 64;
- Adam learning rate 0.003;
- 25 epochs;
- gradient-norm clipping at 1.0;
- minimum validation loss as the checkpoint rule.

A checked CPU run reported:

| Evidence | Vanilla RNN | LSTM |
| --- | ---: | ---: |
| parameters | 1,315 | 4,963 |
| validation loss, $T=30$ | 0.0005 | 0.0005 |
| validation accuracy, $T=30$ | 1.000 | 1.000 |
| locked stress accuracy, $T=60$ | 0.000 | 1.000 |

The matched validation split alone would hide the difference. The length stress test asks whether the learned mechanism still behaves when the credit-assignment path doubles.

The outcome is evidence about this configuration. Useful extensions include:

1. repeat five seeds;
2. match parameter counts;
3. sweep test lengths without using them for selection;
4. raise distractor amplitude;
5. move the cue to a random early position;
6. compare a GRU;
7. measure memory, runtime, and latency.

## 6. Attention as content-addressed access

Recurrence compresses the past into a state. Attention lets a query directly weight a set of stored values.

For one attention head:

$$
Q = XW_Q,
\qquad
K = XW_K,
\qquad
V = XW_V.
$$

The score matrix is

$$
S = \frac{QK^\top}{\sqrt{d_k}}.
$$

Each row becomes a distribution:

$$
A_{ij}
=
\frac{\exp(S_{ij})}
{\sum_m \exp(S_{im})}.
$$

The output is

$$
O = AV.
$$

The scale $\sqrt{d_k}$ prevents dot-product magnitude from growing simply because the key dimension is large.

### Two-token calculation

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

Then

$$
S =
\frac{1}{\sqrt{2}}
\begin{bmatrix}
1 & 0 \\
0 & 1
\end{bmatrix}.
$$

The approximate row-wise weights are

$$
A \approx
\begin{bmatrix}
0.670 & 0.330 \\
0.330 & 0.670
\end{bmatrix}.
$$

Therefore

$$
O \approx
\begin{bmatrix}
1.340 & 1.321 \\
0.660 & 2.679
\end{bmatrix}.
$$

Attention weights are data-dependent mixing coefficients. A high weight does not by itself prove causal importance or a human-interpretable explanation.

## 7. Masks

A mask changes which relations are legal before softmax:

$$
A
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}} + M
\right).
$$

For an allowed relation, $M_{ij}=0$. For a blocked relation, implementations use a large negative value so its softmax probability becomes zero.

A padding mask blocks artificial pad tokens. A causal mask blocks positions $j>i$ so a prediction at step $i$ cannot read the future. An encoder used on a complete document normally permits bidirectional access.

Always test:

- every valid row sums to one;
- blocked entries receive zero weight;
- pad positions do not affect outputs;
- causal training and generation use consistent indexing.

## 8. Multi-head attention

One head uses one set of projections. Multi-head attention computes several heads:

$$
\operatorname{head}_r
=
\operatorname{Attention}
(XW_Q^{(r)}, XW_K^{(r)}, XW_V^{(r)}),
$$

then concatenates and projects them:

$$
\operatorname{MHA}(X)
=
\operatorname{Concat}
(\operatorname{head}_1,\ldots,\operatorname{head}_H)W_O.
$$

Different heads can learn different relation patterns, but the number of heads is not the number of independent human concepts.

For self-attention with sequence length $T$, storing the attention matrix costs approximately $O(T^2)$ per head. The projection and feed-forward terms also matter in real systems. Report measured memory and latency alongside asymptotic notation.

## 9. Transformer encoder blocks

A common pre-normalization encoder block is

$$
X' = X + \operatorname{MHA}(\operatorname{LN}(X)),
$$

$$
Y = X' + \operatorname{FFN}(\operatorname{LN}(X')).
$$

The feed-forward network is applied independently at each position:

$$
\operatorname{FFN}(x)
=
W_2 \phi(W_1x+b_1)+b_2.
$$

Residual paths support optimization, layer normalization stabilizes representations, attention mixes information across positions, and the feed-forward network transforms each position.

Self-attention without position information is permutation-equivariant. A Transformer therefore needs an ordering mechanism, such as:

- fixed sinusoidal encodings;
- learned absolute position embeddings;
- relative position biases;
- rotary position embeddings;
- architecture-specific local or recurrent structure.

Position method, training length, and inference length are experimental variables. A large configured window does not establish effective use of all positions.

## 10. Architecture tradeoffs

Let $T$ be sequence length and $D$ the representation width.

| Family | Main access path | Training parallelism across time | Typical sequence cost | Main limitation |
| --- | --- | --- | --- | --- |
| vanilla RNN | recurrent hidden state | no | $O(TD^2)$ | long credit assignment |
| LSTM / GRU | gated recurrent state | no | $O(TD^2)$ | sequential training and fixed-width state |
| full attention | all pairwise token relations | yes | $O(T^2D)$ plus projections | quadratic attention memory and work |
| local or sparse attention | selected token relations | yes | pattern-dependent | may miss required global relations |
| structured state space | recurrent state or parallel convolution/scan | often yes | commonly near-linear in $T$ | architecture and kernel complexity |
| hybrid memory/attention | direct context plus persistent state | design-dependent | design-dependent | evaluation and implementation complexity |

These expressions hide constants, hardware utilization, batch size, precision, and cache behavior. Measure the quantities that matter to the application.

## 11. A common long-sequence benchmark

[Long Range Arena](https://arxiv.org/abs/2011.04006) was introduced to compare long-context sequence models under a unified suite. It includes six tasks across text, retrieval, images, spatial reasoning, and mathematical expressions, with lengths from 1,024 to 16,384.

Reported results:

| Model | ListOps | Text | Retrieval | Image | Pathfinder | Path-X | Average |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Transformer | 36.37 | 64.27 | 57.46 | 42.44 | 71.40 | failed | 53.66 |
| S4 | 59.60 | 86.82 | 90.90 | 88.65 | 94.20 | 96.35 | 86.09 |
| S5 | 62.15 | 89.31 | 91.40 | 88.00 | 95.33 | 98.58 | 87.46 |
| Mega | 63.14 | 90.43 | 91.25 | 90.44 | 96.01 | 97.98 | 88.21 |

Sources: [S4 Table 4](https://arxiv.org/abs/2111.00396), [S5 Table 1](https://arxiv.org/abs/2208.04933), and [Mega Table 2](https://arxiv.org/abs/2209.10655).

The table is a historical benchmark snapshot. Before citing one number, check:

1. task version and preprocessing;
2. split and metric;
3. model scale;
4. pretraining or augmentation;
5. training budget and hardware;
6. number of seeds and uncertainty;
7. whether the paper reproduced or copied a baseline;
8. publication date and later revisions.

## 12. Selected recent sequence architectures

### Mamba, 2023

[Mamba](https://arxiv.org/abs/2312.00752) makes state-space parameters depend on the input, allowing selective propagation or forgetting. Its authors report linear scaling in sequence length and applications across language, audio, and genomics. The research question is whether a compact recurrent state can retain content-dependent evidence while preserving efficient training and inference.

### xLSTM, 2024

[xLSTM](https://arxiv.org/abs/2405.04517) revisits recurrent language models with exponential gating, scalar-memory sLSTM blocks, and parallelizable matrix-memory mLSTM blocks. It is useful for asking which limits belonged to the original LSTM design and which belonged to historical training scale.

### ModernBERT, 2024

[ModernBERT](https://arxiv.org/abs/2412.13663) is a bidirectional encoder-only Transformer trained on English and code. Its base model has 22 layers, 149 million parameters, and a native context length up to 8,192 tokens. It uses rotary positions, alternating local/global attention, unpadding, and modern feed-forward layers.

The Week 11 lab uses it because many real applications need classification or retrieval rather than open-ended generation. The model card warns that training data are primarily English and code, full-window inference is slower than short-context inference, and representations may reflect data bias.

### Titans, 2025

[Titans](https://arxiv.org/abs/2501.00663) combines attention with a learned long-term neural memory that can be updated during inference. The paper reports experiments in language, common-sense reasoning, genomics, and time series, including needle tasks beyond two million context positions. Treat those claims as paper-reported evidence that needs independent and application-specific evaluation.

## 13. Benchmark families for further work

A sequence model has no single universal score.

| Domain | Example benchmark | Typical metric | Main threat |
| --- | --- | --- | --- |
| language modeling | WikiText-103 | perplexity, lower is better | tokenization and data contamination |
| language understanding | GLUE / SuperGLUE | task-specific aggregate | saturation and task mixture |
| long dependency | LRA | accuracy | synthetic-task transfer |
| long documents | LongBench / LongBench Pro | task-specific aggregate | effective versus configured context |
| forecasting | M4 / Monash collections | scaled forecast error | temporal leakage and horizon choice |
| speech recognition | LibriSpeech | word error rate, lower is better | domain and speaker shift |
| biological sequence | task-specific genomic sets | accuracy, AUROC, AUPRC | homology leakage and population shift |

Choose a benchmark because its dependency structure, split, and metric match the research claim.

## 14. ModernBERT transfer-learning protocol

The real-application lab uses AG News topic classification with four labels: World, Sports, Business, and Sci/Tech.

Default protocol:

- source: fancyzhx/ag_news dataset card;
- official training split: source for balanced training and validation subsets;
- official test split: source for one locked balanced test subset;
- 2,000 training, 500 validation, and 1,000 test examples;
- maximum token length 128;
- frozen ModernBERT encoder;
- trainable prediction head and classifier;
- two epochs;
- validation loss selects the restored checkpoint;
- automatic CUDA and mixed precision on supported GPUs.

A responsible report includes a majority baseline, class support, per-class recall, confusion matrix, error examples, latency, trainable and total parameters, and the exact model revision if frozen.

The dataset card lists its license as unknown and describes academic, non-commercial research use. Do not redistribute the data. Check the original source and institutional requirements.

## 15. Research directions

Strong Week 11 directions include:

- memory retention as sequence length changes;
- recurrent versus attention latency for streaming inference;
- effective context length rather than configured context length;
- local, global, and hybrid attention patterns;
- parameter-matched LSTM, Transformer, and state-space comparisons;
- low-resource or multilingual encoder transfer;
- irregularly sampled time series;
- calibration under temporal or domain shift;
- retrieval and memory-update failures;
- privacy and data provenance in long-context systems;
- interpretability claims for gates, states, and attention weights;
- energy and memory cost under application constraints.

Phrase a direction as a falsifiable question. For example:

> On participant-disjoint wearable-sensor sequences, does a parameter-matched LSTM improve macro-F1 at delays above 100 steps relative to a vanilla RNN, without exceeding the declared mobile latency budget?

The question names the split, baseline, metric, stress variable, and operational constraint.

## 16. Claims supported by Week 11

The material supports these conclusions:

- recurrence creates a shared-state computation over ordered inputs;
- a recurrent Jacobian product explains possible vanishing and exploding gradients;
- LSTM gates create an additive cell-state path that can preserve selected information;
- attention provides direct content-dependent access and requires legal masking;
- Transformers combine attention, position information, residual paths, normalization, and feed-forward transformations;
- benchmark scores are interpretable only with their protocols;
- recent sequence research explores efficient state, selective memory, hybrid access, and realistic long-context evaluation.

It does not establish that one family wins every sequence problem, that attention weights are explanations, or that a configured context window is effectively used.
