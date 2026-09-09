# Sequence-model research-direction note

**Due:** Week 11
**Purpose:** turn a broad interest in sequence modeling into one benchmark-aware, falsifiable next experiment.

## Deliverable

Submit a two-page note plus one comparison matrix. Link to the exact repository state, data source, and primary papers used.

## 1. Problem and prediction time

State:

- one observation and its shape;
- the ordering variable and sampling interval;
- target and prediction horizon;
- sequence-to-one, sequence-to-sequence, autoregressive, or forecasting form;
- information legally available at prediction time;
- the entity or time unit that must remain disjoint across splits.

Write one example of future-information or overlapping-window leakage relevant to the task.

## 2. Evidence map

Compare at least three primary sources:

1. one foundational recurrent or attention source;
2. one benchmark or dataset source;
3. one recent architecture source.

| Source and version | Task and split | Model and scale | Metric | Sequence length | Compute or efficiency | Main result | Limitation for my question |
| --- | --- | --- | --- | --- | --- | --- | --- |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

Do not combine incompatible metrics into a ranking. Mark whether every baseline number was reproduced, copied from the source, or independently reported elsewhere.

## 3. Baseline and candidate

Name a baseline that can answer the research question. Examples include:

- majority, persistence, or seasonal-naive prediction;
- linear or bag-of-features model;
- vanilla RNN;
- LSTM or GRU;
- a small Transformer;
- a published domain baseline.

Describe one candidate change. Keep data, split, metric, preprocessing, and training budget fixed where possible.

## 4. Benchmark and metrics

Choose a benchmark only if its dependency structure matches the claim. Otherwise define an application-specific protocol.

Record:

- dataset version and access conditions;
- training, validation, and locked test construction;
- primary metric and direction;
- class, subgroup, or horizon evidence;
- uncertainty across seeds, folds, or time periods;
- trained, configured, and evaluated sequence lengths;
- latency, throughput, peak memory, or energy if efficiency is part of the claim.

## 5. Stress test

Choose one controlled stress variable:

- sequence length;
- distance between cue and prediction;
- distractor density;
- sampling irregularity;
- missingness;
- temporal or domain shift;
- causal versus bidirectional context;
- memory or latency budget;
- language or subgroup.

State the fixed values and the sweep values before running the test. Do not use the locked test results to select them.

## 6. Falsifiable research question

Use this pattern if helpful:

> On ____________________ split sequences, does ____________________ improve
> ____________________ relative to ____________________ when
> ____________________ changes, without exceeding ____________________?

Then state:

- expected result;
- mechanism that could explain it;
- result that would reject the direction;
- next decision after either outcome.

## 7. Reproducibility plan

Record:

- environment and package versions;
- device and precision;
- seeds;
- data checksums or stable identifiers;
- exact commands;
- model revision;
- checkpoint rule;
- output files;
- expected runtime and storage;
- files that must remain outside Git.

## Submission check

- [ ] Prediction time and legal context are explicit.
- [ ] Split design prevents entity, window, and temporal leakage.
- [ ] Three primary sources are compared.
- [ ] Baseline and candidate answer the same question.
- [ ] Metric and stress test match the claim.
- [ ] Model scale and compute are reported.
- [ ] The rejection condition is specific.
- [ ] No private data, downloaded datasets, weights, or credentials are committed.

## Assessment guide

| Criterion | Points |
| --- | ---: |
| Task, prediction time, and leakage analysis | 20 |
| Primary-source comparison and benchmark literacy | 20 |
| Baseline and controlled candidate | 15 |
| Metrics, uncertainty, and stress test | 20 |
| Falsifiable question and rejection condition | 15 |
| Reproducibility, data conditions, and feasibility | 10 |
| **Total** | **100** |
