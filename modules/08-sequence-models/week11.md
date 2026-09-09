# Week 11 — RNNs, LSTMs, Attention, and Transformers

**Guiding question:** How should a model store, retrieve, and evaluate evidence when the order and distance between observations matter?

Week 11 begins with recurrence and the mathematics of gradient flow, develops the LSTM cell as a gated memory mechanism, and then reframes memory as direct access through attention. The final section connects these foundations to Transformers and selected recent sequence architectures. It includes a common-benchmark comparison rather than a mixed leaderboard.

## Learning outcomes

By the end of the week, students should be able to:

- distinguish sequence-to-one, sequence-to-sequence, and autoregressive tasks;
- write the vanilla RNN recurrence and trace its tensor shapes;
- explain vanishing and exploding gradients through a product of recurrent Jacobians;
- calculate every LSTM gate and describe the role of the cell state;
- implement and evaluate an LSTM using a validation-selected checkpoint;
- calculate scaled dot-product attention and apply padding or causal masks;
- explain multi-head attention, positional information, and a Transformer block;
- compare recurrent, attention-based, and state-space approaches without mixing incompatible benchmarks;
- identify a sequence-model research direction and a falsifiable next experiment.

## Preparation

Review the Week 6 chain rule and the Week 8 discussion of gradient flow. Read the opening sections of:

- [Long Short-Term Memory](https://doi.org/10.1162/neco.1997.9.8.1735);
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762).

Bring one sequence-learning example from your research interests. State the observation, ordering variable, target, and whether future information is available at prediction time.

## Three-hour teaching sequence

| Time | Activity | Evidence |
| ---: | --- | --- |
| 0–15 min | Define sequence tasks, ordering, and leakage | Students identify the prediction time and legal context |
| 15–40 min | Unroll a vanilla RNN | Correct recurrence, shapes, and parameter sharing |
| 40–60 min | Backpropagation through time | A Jacobian product explains gradient instability |
| 60–90 min | Derive the LSTM cell | All gates and state updates are calculated |
| 90–115 min | Lab 1: delayed-memory learning | RNN and LSTM are compared on a locked longer sequence |
| 115–125 min | Break | — |
| 125–150 min | Scaled dot-product and multi-head attention | Weights normalize and masks block illegal access |
| 150–170 min | Transformer blocks and position | Students trace residual paths and tensor shapes |
| 170–185 min | Common benchmark and recent architectures | Claims are tied to protocols and primary sources |
| 185–205 min | Lab 2: ModernBERT on AG News | A real text classifier uses frozen-head transfer learning |
| 205–220 min | Research-direction workshop and exit ticket | One benchmark-aligned, falsifiable experiment is proposed |

The sequence can be divided into two meetings by ending the first meeting after Lab 1.

## Core equations

A vanilla recurrent state is

$$
h_t = \tanh(W_{xh}x_t + W_{hh}h_{t-1} + b_h)
$$

and a sequence-to-one classifier can use

$$
z = W_{hy}h_T + b_y.
$$

The influence of an early hidden state on a later state contains a product:

$$
\frac{\partial h_T}{\partial h_t}
=
\prod_{k=t+1}^{T}
\frac{\partial h_k}{\partial h_{k-1}}.
$$

Repeated factors with norms below one make gradients vanish; repeated factors above one can make them explode. The full derivation and a numerical trace are in [the notes](week11-notes.md).

An LSTM adds gates and an explicit cell state:

$$
\begin{aligned}
i_t &= \sigma(W_i x_t + U_i h_{t-1} + b_i), \\
f_t &= \sigma(W_f x_t + U_f h_{t-1} + b_f), \\
g_t &= \tanh(W_g x_t + U_g h_{t-1} + b_g), \\
o_t &= \sigma(W_o x_t + U_o h_{t-1} + b_o), \\
c_t &= f_t \odot c_{t-1} + i_t \odot g_t, \\
h_t &= o_t \odot \tanh(c_t).
\end{aligned}
$$

Scaled dot-product attention is

$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}\left(
\frac{QK^\top}{\sqrt{d_k}} + M
\right)V,
$$

where a mask entry is zero for an allowed relation and a large negative value for a blocked relation.

## Lab 1 — delayed memory with RNNs and LSTMs

Run:

    python labs/week11_sequence_pytorch.py

For a fast implementation and shape audit:

    python labs/week11_sequence_pytorch.py --audit-only

The generated task presents a three-class cue at the first step, followed by low-amplitude distractors. Training and validation sequences have length 30. The locked stress split has length 60 and is never used for checkpoint selection. The models share the same data, seed, hidden width, optimizer, batch size, and selection rule.

A checked CPU run with seed 705 produced:

| Model | Parameters | Validation accuracy, length 30 | Locked stress accuracy, length 60 |
| --- | ---: | ---: | ---: |
| Vanilla RNN | 1,315 | 1.000 | 0.000 |
| LSTM | 4,963 | 1.000 | 1.000 |
| Majority baseline | — | 0.333 | 0.333 |

This result supports a narrow claim: under the declared task and initialization, both models fit the training-length problem, while the LSTM preserved the cue when the delay doubled. It does not prove that LSTMs always outperform RNNs. Students should rerun with multiple seeds, hidden widths, noise levels, and delays before making a broader claim.

## Lab 2 — a recent model on a real application

[ModernBERT](https://arxiv.org/abs/2412.13663) is a 2024 encoder-only Transformer intended for retrieval and classification. The course lab uses the 149-million-parameter base checkpoint for four-class AG News topic classification:

    python labs/week11_modernbert_news.py

The default configuration:

- retrieves the AG News dataset through its Hugging Face dataset card;
- derives balanced training and validation subsets from the official training split;
- keeps a balanced subset of the official test split locked;
- tokenizes to at most 128 tokens;
- automatically selects CUDA when available;
- freezes the encoder and trains the prediction and classification heads;
- restores the minimum-validation-loss checkpoint before one test evaluation.

For a download-free audit:

    python labs/week11_modernbert_news.py --audit-only

For staged fine-tuning after the head-only baseline:

    python labs/week11_modernbert_news.py --unfreeze-last-n 2 --batch-size 4

Record GPU model, package versions, trainable parameter count, wall time, peak memory if available, validation history, locked test result, and at least ten inspected errors. AG News has unknown license metadata on the cited card and describes academic, non-commercial research use; check institutional requirements before redistribution or other use. No data or model weights belong in Git.

## Benchmark literacy

The Long Range Arena benchmark uses six tasks with sequence lengths from 1,024 to 16,384. The table below transcribes test accuracy reported in the cited S4, S5, and Mega papers.

| Model | Model family | LRA average accuracy | Path-X accuracy | Reported source |
| --- | --- | ---: | ---: | --- |
| Transformer | full attention | 53.66 | failed / chance | [S4 Table 4](https://arxiv.org/abs/2111.00396) |
| S4 | structured state space | 86.09 | 96.35 | [S4 Table 4](https://arxiv.org/abs/2111.00396) |
| S5 | multi-input state space | 87.46 | 98.58 | [S5 Table 1](https://arxiv.org/abs/2208.04933) |
| Mega | moving-average gated attention | 88.21 | 97.98 | [Mega Table 2](https://arxiv.org/abs/2209.10655) |

Use the table as a historical comparison under LRA protocols. Later papers may modify preprocessing, model scale, training budgets, or reporting. A benchmark score is evidence only after the task, split, metric, compute, and version are named.

## Selected recent architectures

This is a dated teaching snapshot checked on 10 September 2026.

| Year | Architecture | Main memory or access idea | Research question it raises |
| ---: | --- | --- | --- |
| 2017 | Transformer | content-addressed self-attention | How far can direct pairwise access scale? |
| 2021 | S4 | structured state-space recurrence and convolution | Can one state model train in parallel and infer recurrently? |
| 2023 | Mamba | input-dependent selective state spaces | What information should a linear-time state retain or forget? |
| 2024 | xLSTM | exponential gates and scalar or matrix memories | Can recurrent language models scale competitively again? |
| 2024 | ModernBERT | alternating local/global attention and modern encoder training | When is an encoder a better application model than a decoder? |
| 2025 | Titans | attention plus learned long-term neural memory | Can a model update persistent memory during inference? |
| 2026 | LongBench Pro | realistic bilingual long-context evaluation | Is a claimed context window an effective context window? |

Read [the Week 11 research guide](../../readings/week11-sequence-models.md) for benchmark families, primary papers, and project directions.

## Research milestone

Complete [the sequence-model research-direction note](../../research-project/sequence-model-research-direction.md). The note must connect a research question to:

1. the prediction time and legal context;
2. a defensible baseline;
3. one sequence architecture;
4. a benchmark or task-specific metric;
5. a length, latency, memory, or robustness stress test;
6. a result that would reject the proposed direction.

## Package

- [Detailed notes](week11-notes.md)
- [Accessible slide source](week11-slides.md)
- [Student worksheet](week11-worksheet.md)
- [PowerPoint deck](slides/week11-rnns-lstms-attention-and-transformers.pptx)
- [Instructor guide](../../instructor-notes/week11.md)
- [RNN/LSTM lab](../../labs/week11_sequence_pytorch.py)
- [ModernBERT AG News lab](../../labs/week11_modernbert_news.py)
- [Further reading and research directions](../../readings/week11-sequence-models.md)
- [Research-direction milestone](../../research-project/sequence-model-research-direction.md)

## Exit ticket

In at most 150 words:

- explain why a recurrent Jacobian product can make early evidence hard to learn;
- state how the LSTM cell changes that path;
- distinguish recurrent memory from attention-based access;
- name one limitation of the Week 11 evidence.
