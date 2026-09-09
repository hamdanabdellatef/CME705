# Week 11 further reading — Sequence benchmarks, recent models, and research directions

This guide is organized by question. Start with the foundational readings, then select one benchmark and one recent architecture relevant to your project. Record the paper version and retrieval date.

## Foundations

### Recurrent computation and memory

- [Learning representations by back-propagating errors](https://www.nature.com/articles/323533a0) established backpropagation as a practical learning procedure for layered networks.
- [Long Short-Term Memory](https://doi.org/10.1162/neco.1997.9.8.1735) introduced the LSTM memory and gating ideas.
- [Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078) introduced the gated recurrent unit and a recurrent encoder-decoder.

Read for: shared recurrence, credit assignment, gates, and the relationship between state and output.

### Attention and Transformers

- [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) is an early neural attention paper.
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) introduced the Transformer architecture based on attention, position information, residual paths, normalization, and feed-forward blocks.

Read for: the access path created by attention and the reason training can be parallelized across positions.

## Long-sequence benchmark lineage

### Long Range Arena

[Long Range Arena](https://arxiv.org/abs/2011.04006) provides six tasks with input lengths from 1,024 to 16,384 across text, retrieval, images, spatial reasoning, and mathematical expressions. It was designed to compare efficient long-range sequence models under a common suite.

Use it when the research question concerns architectural ability to model long dependencies. Do not assume performance transfers automatically to natural long documents.

### LongBench

[LongBench](https://arxiv.org/abs/2308.14508) evaluates long-context language models across 21 datasets and six task categories in English and Chinese. Categories include single-document question answering, multi-document question answering, summarization, few-shot learning, synthetic tasks, and code completion.

Use it when the claim concerns long-document language understanding. Check model context configuration, prompt format, retrieval or compression, and task-specific metrics.

### LongBench Pro

[LongBench Pro](https://arxiv.org/abs/2601.02872), released in 2026, contains 1,500 naturally occurring bilingual samples across 11 primary tasks and input lengths from 8,000 to 256,000 tokens. The authors emphasize that effective context length can be shorter than a configured context window.

Use it to study realistic long-context reasoning, length sensitivity, and cross-lingual behavior. Treat its construction process and model-assisted drafting as part of the dataset methodology to review.

## Other benchmark families

A sequence architecture should be evaluated on a benchmark whose dependency structure matches the claim.

| Question | Example primary source | Typical metric | Check before comparison |
| --- | --- | --- | --- |
| general language modeling | [Pointer Sentinel Mixture Models / WikiText-103](https://arxiv.org/abs/1609.07843) | perplexity, lower is better | tokenizer, vocabulary, context, contamination |
| language understanding | [GLUE](https://arxiv.org/abs/1804.07461) | task-specific aggregate | task mix, fine-tuning protocol, saturation |
| harder language understanding | [SuperGLUE](https://arxiv.org/abs/1905.00537) | task-specific aggregate | task mix, human baseline, test access |
| forecasting across domains | [Monash Time Series Forecasting Archive](https://arxiv.org/abs/2105.06643) | scaled forecast error | horizon, frequency, missingness, temporal split |
| speech recognition | [LibriSpeech](https://www.danielpovey.com/files/2015_icassp_librispeech.pdf) | word error rate | speaker split, language model, domain shift |
| biological sequences | task-specific primary benchmark | AUROC, AUPRC, accuracy | homology, population, and chromosome leakage |

Never combine accuracy, perplexity, BLEU, word error rate, and forecasting errors into one ranking.

## Efficient attention and state-space progression

### S4, 2021

[Efficiently Modeling Long Sequences with Structured State Spaces](https://arxiv.org/abs/2111.00396) represents a state-space model as a recurrence for inference and a convolution for parallel training. Its paper reports an LRA average of 86.09 and a Path-X accuracy of 96.35.

Research prompts:

- Which long dependencies are captured by a structured state?
- How do recurrent and convolutional implementations compare numerically?
- Does a model trained at one sampling rate transfer to another?

### S5, 2022

[Simplified State Space Layers for Sequence Modeling](https://arxiv.org/abs/2208.04933) replaces banks of single-input state-space systems with one multi-input, multi-output system and uses parallel scans. Its paper reports an LRA average of 87.46.

Research prompts:

- What changes when one state mixes several input features?
- Which benefits come from the state parameterization and which from training scale?
- How stable is irregular-sampling behavior?

### Mega, 2022

[Mega: Moving Average Equipped Gated Attention](https://arxiv.org/abs/2209.10655) combines a learned exponential moving average with gated attention. The full model has quadratic attention; a chunked variant offers linear complexity with a reported quality tradeoff. The paper reports an LRA average of 88.21 for the full model.

Research prompts:

- When should local recency bias and global content access be separated?
- How does chunk size affect accuracy, memory, and latency?
- Do gains remain under equal parameter and compute budgets?

## Selected recent models

### Mamba, 2023

[Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) makes state-space parameters input dependent, allowing the model to select what to propagate or forget. The authors report linear scaling, fast inference, and results across language, audio, and genomics.

Study:

- content-dependent selection;
- recurrent inference and parallel scan training;
- effective memory under distractors;
- throughput and memory as sequence length grows.

### xLSTM, 2024

[xLSTM: Extended Long Short-Term Memory](https://arxiv.org/abs/2405.04517) revisits recurrent language modeling with exponential gates, scalar-memory sLSTM blocks, and matrix-memory mLSTM blocks.

Study:

- how modern normalization and residual blocks change recurrent scaling;
- which xLSTM components are responsible for gains;
- whether parallelizable matrix memory changes the original LSTM tradeoff;
- streaming latency versus Transformer caching.

### ModernBERT, 2024

[ModernBERT](https://arxiv.org/abs/2412.13663) is a bidirectional encoder-only Transformer trained on two trillion English and code tokens with a native sequence length up to 8,192. The [model card](https://huggingface.co/answerdotai/ModernBERT-base) describes a 149-million-parameter base model, rotary positions, alternating local/global attention, unpadding, Apache 2.0 model licensing, and limitations.

Study:

- classification or retrieval versus generation requirements;
- local/global attention ablations;
- effective length on real documents;
- English and code training-data bias;
- fixed-feature, partial, and full fine-tuning under equal budgets.

The accompanying lab applies ModernBERT-base to AG News. The [dataset card](https://huggingface.co/datasets/fancyzhx/ag_news) shows 120,000 training and 7,600 test rows, four topic labels, unknown license metadata, and a research-use description. Review the original source and institutional requirements before use.

### Titans, 2025

[Titans: Learning to Memorize at Test Time](https://arxiv.org/abs/2501.00663) combines attention over current context with a learned long-term neural memory updated during inference. The paper reports experiments across language, common-sense reasoning, genomics, time series, and needle tasks beyond two million context positions.

Study:

- what enters persistent memory;
- how memory updates fail or drift;
- privacy and deletion implications;
- order sensitivity and adversarial distractors;
- comparison with retrieval and fixed caches.

## How to read a sequence-model result

Create one record per result.

| Field | Required question |
| --- | --- |
| task | What mapping is learned? |
| context | What observations are legal at prediction time? |
| data | What version, source, population, and license apply? |
| split | Are entities, windows, and time periods disjoint? |
| metric | What does it reward and hide? |
| model | What parameter count and exact revision were used? |
| training | What data, compute, precision, and selection rule were used? |
| length | What trained, configured, and tested lengths apply? |
| efficiency | What latency, throughput, and peak memory were measured? |
| uncertainty | How many seeds, folds, or confidence intervals were reported? |
| baseline | Was it reproduced, copied, or retuned? |
| limitation | Which application claim remains unsupported? |

## Research directions

### Memory and effective context

Measure performance as relevant evidence moves farther from the prediction point. Include adversarial distractors and several sequence lengths. Compare configured context with the length at which evidence remains usable.

### Streaming and online inference

Compare recurrent state, attention caches, and state-space inference under a fixed latency and memory budget. State whether revising earlier outputs is permitted.

### Time-series validity

Study irregular sampling, missingness, forecast horizon, and temporal drift. Preserve chronological evaluation and fit all preprocessing on the past.

### Multilingual and low-resource transfer

Test whether an English-heavy pretrained encoder transfers to Turkish or another target language. Include monolingual baselines, class-aware metrics, and error analysis by language phenomenon.

### Efficient adaptation

Compare head-only training, parameter-efficient adaptation, partial unfreezing, and full fine-tuning under the same data and compute budget.

### Sequence-model reliability

Test calibration, out-of-distribution shift, perturbations, subgroup performance, and abstention. A higher benchmark score does not imply safe confidence.

### Memory privacy and provenance

Ask whether a long-context or test-time memory system retains sensitive content, whether stored information can be deleted, and how provenance is recorded.

### Interpretability

Compare gate values or attention weights with interventions. Treat visualizations as descriptions until they predict a controlled change in output.

## Suggested reading paths

### Recurrent systems

1. LSTM
2. xLSTM
3. S4
4. Mamba
5. one streaming or time-series benchmark

### Attention and long documents

1. Attention Is All You Need
2. Long Range Arena
3. ModernBERT
4. LongBench
5. LongBench Pro

### Hybrid memory

1. LSTM
2. Attention Is All You Need
3. Mega
4. Mamba
5. Titans

## Deliverable

For the Week 11 milestone, compare at least three sources in a matrix:

- one foundational source;
- one benchmark source;
- one recent architecture source.

End with a falsifiable experiment and a result that would make you abandon or revise the direction.
