# Week 3 instructor guide: Data quality and controlled model comparison

## Purpose

Week 3 moves students from evaluation boundaries to evidence about the dataset itself. Students should inspect provenance, missingness, rare classes, suspicious values, label quality, and feature availability before interpreting model rankings.

The lesson ends with a structured problem-proposal workshop. A proposal is feasible when the data and evaluation plan can answer the stated question with available time and resources.

## Before class

- Run `python labs/week03_data_quality_numpy.py`.
- Distribute [the Week 3 worksheet](../modules/02-data-and-evaluation/week03-worksheet.md).
- Ask students to bring a candidate dataset, its source page, and any available documentation.
- Prepare one small data extract containing a missing value, an extreme value, a duplicate identifier, and a label conflict.
- Do not show the lab result table until students predict the majority baseline.

## Learning evidence

Inspect:

1. evidence/interpretation separation in the first-pass audit;
2. missingness questions across class, source, and time;
3. the imbalance calculation;
4. a feature-quality decision;
5. a controlled comparison protocol; and
6. the problem-proposal planner.

## Three-hour sequence

| Time | Activity | Instructor action | Student evidence |
| ---: | --- | --- | --- |
| 0–10 min | Opening claim | Display “1,000 clean rows” without documentation | Five facts the row count cannot establish |
| 10–30 min | Provenance and measurement | Trace collection, selection, labeling, and storage | One supported and unsupported claim |
| 30–50 min | First-pass audit | Separate observation, explanation, and next check | Audit table |
| 50–70 min | Duplicates, outliers, and labels | Challenge automatic deletion | Evidence needed before action |
| 70–85 min | Missingness | Compare rates by class, source, and time | Three mechanism questions |
| 85–95 min | Break | — | — |
| 95–115 min | Imbalance | Calculate the always-negative example | Accuracy, recall, specificity, balanced accuracy |
| 115–130 min | Responses and feature screen | Place resampling and thresholding in the workflow | Feature decision |
| 130–145 min | Baseline ladder and comparison | Fix a protocol before ranking candidates | Controlled-comparison table |
| 145–165 min | NumPy lab | Predict, run, and interpret three candidates | Completed results and claim boundary |
| 165–175 min | Error analysis | Turn errors into one next experiment | Three error categories |
| 175–180 min | Proposal exit | Check feasibility and remaining evidence | Four-line exit ticket |

## Facilitation notes

### Opening

Organize student questions under:

- provenance and permission;
- observation and target;
- completeness and validity;
- dependence and representation; and
- relevance to intended use.

Do not let “clean” become a binary label. Quality is relative to a task and claim.

### First-pass audit

In the illustrative table:

- the missing value needs a source and pattern check;
- 999 may be an extreme measurement, missing-value code, or unit error;
- repeated sample_id S04 suggests duplication but requires export or timestamp evidence; and
- equal features with conflicting labels may reflect ambiguity, entity-level change, or a bad join.

The expected immediate deletion count is zero. Students should propose checks before actions.

### Missingness

Introduce MCAR, MAR, and MNAR as assumption categories rather than diagnoses from a plot. Ask what observed variables predict missingness and what unavailable process might remain.

Every imputation example should follow the split. If students propose mean or median imputation, ask which rows estimate the value.

### Imbalance

For 120 positives and 880 negatives with an always-negative predictor:

- accuracy = 0.880;
- positive recall = 0.000;
- specificity = 1.000; and
- balanced accuracy = 0.500.

Ask whether the model is useful before discussing more complex methods.

### Responses to imbalance

Expected stages:

- define the positive class before splitting;
- stratify while creating a split if observations are independent;
- oversample and apply class weighting inside training;
- select a threshold with validation;
- report counts and class-aware metrics with results.

Explain that duplicated oversampled rows can cross the test boundary if resampling occurs too early.

### Week 3 lab

The lab intentionally creates a rare positive class and missingness related to an underlying value. Students should observe:

- majority accuracy = 0.880 but recall = 0;
- median-imputed logistic balanced accuracy around 0.774; and
- imputation plus indicator balanced accuracy around 0.857.

The controlled difference is the missingness indicator. The result does not prove that indicators always help. Real missingness can shift, encode access patterns, or become unavailable.

### Controlled comparison

Require one sentence:

> Candidate A and candidate B use the same ______ and differ only in ______.

If tuning budgets or preprocessing differ, the comparison cannot attribute the result to the model alone.

### Error analysis

Ask students to define categories before choosing examples. Categories should lead to actions such as relabeling review, collection, robust preprocessing, subgroup evaluation, or a controlled ablation.

### Proposal workshop

A strong proposal includes a rejection condition. Examples include inaccessible data, unusable label timing, insufficient positive cases, impossible grouping, or a baseline that already saturates the metric.

## Common misconceptions

| Misconception | Response |
| --- | --- |
| More rows guarantee better data | Dependence, poor labels, and selection can make added rows misleading |
| Missing values should always be filled | The feature or question may need redesign |
| Outliers should be deleted | First determine whether they are errors or valid rare cases |
| Oversampling fixes evaluation imbalance | It changes training; evaluation should retain relevant prevalence |
| The model with the best validation score is automatically best | Compare uncertainty, costs, resources, and limitations |
| Adding an indicator is harmless | Missingness can encode unstable processes or sensitive access patterns |

## Formative feedback language

- “State the observed fact before its explanation.”
- “What source record would justify deletion?”
- “Which rows fitted that preprocessing value?”
- “Report the rare-class count before the average.”
- “What exactly changed between the candidates?”
- “Name a condition that would make the dataset infeasible.”

## Adaptations

### Ninety-minute class

Use slides 1–10, 11–17, 20, and 21. Complete the first-pass audit, imbalance calculation, and lab. Assign the feature screen and proposal planner afterward.

### Online delivery

Use shared breakout documents for the audit and fair-comparison protocol. Give each group one suspicious data pattern and require a next check rather than an immediate cleaning action.

### Limited Python preparation

Run the lab as a demonstration. Students must still predict the majority result, identify the fixed protocol, and interpret the metric differences.

## After class

- flag proposals with uncertain data access, label timing, or grouping;
- check that each baseline is feasible;
- return feasibility feedback before students invest in complex models; and
- carry unresolved data-quality questions into the Week 4 literature comparison.
