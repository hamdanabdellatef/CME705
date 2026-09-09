# Sequence models

**Guiding question:** How should a model store, retrieve, and evaluate evidence when the order and distance between observations matter?

## Learning outcomes

Students will:

- define legal context and leakage risks for sequence tasks;
- derive a vanilla RNN and backpropagation through time;
- explain vanishing and exploding recurrent gradients;
- calculate LSTM gates, cell state, and hidden state;
- train and evaluate a PyTorch LSTM on a delayed-memory task;
- calculate scaled dot-product attention and apply masks;
- trace multi-head attention and a Transformer encoder block;
- compare recurrent, attention, and state-space models using a common benchmark;
- use a recent bidirectional Transformer for a real text-classification application;
- propose a benchmark-aware sequence-model research direction.

## Week 11 package

- [Lesson plan](week11.md)
- [Detailed notes](week11-notes.md)
- [Accessible slide source](week11-slides.md)
- [Student worksheet](week11-worksheet.md)
- [PowerPoint deck](slides/week11-rnns-lstms-attention-and-transformers.pptx)
- [Instructor guide](../../instructor-notes/week11.md)
- [RNN/LSTM and attention lab](../../labs/week11_sequence_pytorch.py)
- [ModernBERT AG News lab](../../labs/week11_modernbert_news.py)
- [Further reading](../../readings/week11-sequence-models.md)
- [Research-direction milestone](../../research-project/sequence-model-research-direction.md)

## Topics

- sequence-to-one, sequence-to-sequence, autoregressive, and forecasting tasks;
- prediction time, causal access, and sequence leakage;
- recurrent state and parameter sharing;
- backpropagation through time;
- vanishing and exploding gradients;
- LSTM input, forget, candidate, and output gates;
- delayed memory and length extrapolation;
- scaled dot-product and multi-head attention;
- padding and causal masks;
- Transformer blocks and position information;
- Long Range Arena benchmark literacy;
- S4, S5, Mega, Mamba, xLSTM, ModernBERT, Titans, and LongBench Pro;
- sequence-model research directions.

## Evidence of learning

Students submit:

1. a shape- and gradient-audited recurrent model;
2. a validation-selected RNN/LSTM comparison with a locked length stress test;
3. a verified causal-attention calculation;
4. a ModernBERT transfer-learning record for AG News;
5. a research note whose benchmark, metric, and rejection condition match its claim.

## Place in the course

Week 11 extends the optimization, backpropagation, generalization, and PyTorch experiment design developed in Weeks 5–10. The research milestone prepares students for the Week 12 ablation or robustness experiment.
