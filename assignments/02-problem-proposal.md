# Assignment 2: Problem proposal

## Purpose

Define a feasible machine-learning research question and the evidence needed to investigate it. The proposal connects the Week 1 research interest with the Week 2 evaluation protocol and the Week 3 data audit.

A proposal does not need to predict a positive outcome. It must show that the question can be tested responsibly with available data, time, and compute.

## Deliverables

Submit:

1. a proposal of approximately 1,000–1,400 words, excluding references and the dataset card; and
2. a one-page dataset card draft using the headings below.

Use a clear filename such as `student-id_problem-proposal.pdf`. Include links to the dataset documentation and code repository when available. Do not include private data, credentials, or restricted samples.

## Proposal structure

### 1. Research question and motivation

State one primary question in a form that an experiment can answer.

Include:

- the application or methodological motivation;
- the intended user or scientific audience;
- the decision, prediction, representation, or analysis task; and
- why the result could matter.

Avoid broad goals such as “use deep learning to improve healthcare.” Name the observation, output, setting, and comparison.

### 2. Intended use and claim boundary

Complete:

> We want to estimate performance when the fitted system is applied to ______.

State one use that is outside the proposal’s evidence.

### 3. Dataset and access

Identify:

- creator and source;
- version and retrieval date;
- license or access conditions;
- approximate size;
- collection population and setting; and
- any approval, authentication, or storage requirement.

Confirm that access is practical for this semester. A link alone is not an access plan.

### 4. Observation, target, and dependence

Define:

- one observation;
- input features or representation;
- target or learned structure;
- target timing and label source;
- grouping identifiers;
- repeated, related, or derived observations; and
- any chronological constraint.

### 5. Data-quality audit plan

Describe the first checks for:

- missingness;
- class or target distribution;
- duplicates and near-duplicates;
- label conflicts or uncertainty;
- invalid ranges and outliers;
- feature availability at prediction time;
- subgroup or source coverage; and
- possible leakage.

State at least one condition that would make the dataset unsuitable.

### 6. Baseline and candidate comparison

Specify:

- a simple, credible baseline;
- one candidate method or feature decision;
- the single intended difference between them; and
- why the baseline is appropriate.

The baseline must be implementable before a deep model is required.

### 7. Evaluation protocol

Define before final testing:

- train, validation, and test roles;
- random, stratified, grouped, chronological, or combined split;
- split proportions or cross-validation plan;
- primary metric and one supporting metric;
- threshold-selection procedure if applicable;
- training-only preprocessing;
- random seed and saved split indices; and
- final-test policy.

Explain how the protocol matches the intended use.

### 8. Feasibility, risks, and minimum result

Estimate required time, storage, and compute. Identify the largest technical or data risk.

Define a minimum viable experiment that can:

1. load and audit the data;
2. reproduce the split;
3. run the baseline;
4. calculate the primary metric;
5. compare one controlled change; and
6. report limitations.

State what result would still be useful if the candidate does not outperform the baseline.

## Dataset card draft

Use these headings:

- Motivation
- Composition
- Collection process
- Preprocessing already performed
- Labels and annotation
- Missingness and known quality issues
- Recommended split and grouping
- Permitted and inappropriate uses
- Distribution and access
- Maintenance and versioning
- Known limitations
- Questions still unresolved

Distinguish facts documented by the provider from your own assumptions.

## Sources

Use primary dataset documentation and original research papers when available. Cite every factual claim about collection, labels, population, license, or published performance.

Related work may remain preliminary here; the Week 4 literature-comparison assignment will deepen it.

## Assessment rubric — 30 points

| Criterion | Points | Full-credit evidence |
| --- | ---: | --- |
| Research question and motivation | 5 | Precise, answerable question with observation, output, setting, and value |
| Intended use and boundary | 3 | Future-use claim and explicit excluded claim |
| Dataset provenance and access | 4 | Source, version, permission, population, and practical access |
| Observation, target, and dependence | 4 | Clear definitions, label timing, groups, and time constraints |
| Data-quality audit | 4 | Specific checks and a defensible rejection condition |
| Baseline and comparison | 3 | Feasible baseline and one named controlled difference |
| Evaluation protocol | 5 | Split, metrics, preprocessing, threshold, seed, and final-test policy match the claim |
| Feasibility and reproducibility | 2 | Minimum experiment, resource estimate, risks, and reproducible records |

## Submission check

Before submitting, confirm that:

- [ ] one primary research question is stated;
- [ ] intended use and one excluded use are explicit;
- [ ] dataset access and permitted use are verified;
- [ ] observation, target, groups, and label timing are defined;
- [ ] data-quality checks and a rejection condition are included;
- [ ] the baseline can be implemented immediately;
- [ ] the split and metrics are chosen before final testing;
- [ ] preprocessing is fitted on training data only;
- [ ] the minimum viable experiment fits the semester constraints;
- [ ] provider facts and student assumptions are distinguished; and
- [ ] all sources are cited.
