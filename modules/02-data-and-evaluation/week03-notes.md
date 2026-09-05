# Week 3 notes: Data quality and controlled model comparison

## 1. A dataset is a measurement process

A matrix is the end of a process involving selection, measurement, labeling, storage, cleaning, and access. Model performance can reflect any part of that process.

Before fitting a model, ask:

- Who or what could enter the dataset?
- Who or what was excluded?
- How and when was each feature measured?
- Who assigned the target?
- Which transformations already occurred?
- Which conditions will differ when the system is used?

The data do not automatically represent the application. Their provenance defines the claims they can support.

## 2. Start with an audit

A first-pass audit should record:

1. dataset source, version, license, and retrieval date;
2. unit of observation and grouping identifiers;
3. row and feature counts;
4. feature types, units, valid ranges, and timestamps;
5. target definition, timing, and class counts;
6. missingness by feature and important group;
7. duplicates and near-duplicates;
8. conflicting or uncertain labels;
9. possible outliers and impossible values; and
10. exclusions already applied by the provider.

An audit identifies questions. It should not silently delete unusual examples.

## 3. Duplicates, dependence, and conflicts

Exact duplicate rows can arise from repeated export, merging, augmentation, or legitimate repeated measurements. Identical features can also describe separate observations.

Before removal, determine:

- whether identifiers and timestamps match;
- whether duplicates cross partitions;
- whether one original produced several derived samples; and
- whether duplicate inputs have conflicting labels.

Cross-partition duplicates can inflate evaluation. Conflicting labels may indicate annotation disagreement, ambiguous targets, or an incorrect join.

## 4. Missing data are information

A missing entry can mean a sensor failed, a question was skipped, a test was unnecessary, access was denied, or a field did not exist at the time.

Three concepts help organize questions:

- **MCAR:** missingness is independent of observed and unobserved values;
- **MAR:** missingness can be explained using observed information; and
- **MNAR:** missingness depends on an unobserved value or the missing value itself.

These are assumptions about a data-generating process. A missing-value table alone cannot prove the mechanism.

Always inspect missingness by class, group, source, time, and other relevant variables.

## 5. Responses to missingness

Possible responses include:

- correct a data-collection or merge error;
- exclude a feature unavailable at intended use;
- exclude observations using a documented rule;
- impute with a training-set statistic;
- append a missingness indicator;
- use a model that handles missing values; or
- redesign the question.

Deletion is not neutral when missingness is systematic. Imputation does not recreate an unknown true value; it creates a usable representation under assumptions.

Fit imputation on training data only. Apply the fitted transformation unchanged to validation and test data.

## 6. Class imbalance

For binary classification, prevalence is

[
\pi=\frac{\text{number of positive examples}}{\text{number of examples}}.
]

If (pi=0.12), an always-negative classifier has 0.88 accuracy and zero recall for the positive class.

Report class counts before metrics. Accuracy may be suitable when classes and costs are balanced, but it can hide failure on a rare class.

Useful summaries include recall, specificity, precision, balanced accuracy, precision–recall curves, and confusion-matrix counts. Choose through the intended decision.

## 7. Responses to imbalance

Options include:

- collect more rare-class examples;
- use class-weighted loss;
- oversample or undersample training data;
- tune a decision threshold on validation data;
- use stratification when observations are independent;
- report class-aware metrics; and
- reframe the prediction or ranking task.

Resampling belongs inside the training partition. Resampling before splitting can duplicate information across the evaluation boundary.

A method that changes prevalence in training does not change the real prevalence in deployment.

## 8. Noise and label quality

Feature noise can come from calibration, quantization, transcription, or unstable measurement. Label noise can come from annotator disagreement, changing definitions, proxy outcomes, or delayed events.

Useful checks include:

- review a sample of disagreements and extreme errors;
- estimate inter-annotator agreement when multiple labels exist;
- document uncertain labels separately;
- compare label rules across sources and time; and
- test sensitivity to plausible label corrections.

Do not remove every example a model gets wrong. That would make the model define the data.

## 9. Outliers and impossible values

An unusual value may be:

- a measurement error;
- a unit mismatch;
- a data-entry error;
- a rare but valid case; or
- evidence that the target population is broader than expected.

Use domain-valid ranges, plots, robust summaries, and source records. State each correction or exclusion rule before comparing models when possible.

## 10. Feature quality

A feature should be evaluated for:

- **availability:** is it known at prediction time?
- **meaning:** what process generated it?
- **stability:** will its definition or distribution change?
- **coverage:** how often is it missing?
- **scale and encoding:** what transformation is needed?
- **redundancy:** does it duplicate another signal?
- **sensitivity:** does it expose private or protected information?
- **leakage:** does it contain the target or a later consequence?

A feature can improve validation performance and still be unsuitable for deployment.

## 11. A baseline ladder

Build evidence in increasing complexity:

1. prevalence or constant predictor;
2. simple rule based on domain knowledge;
3. linear or logistic model;
4. established conventional method;
5. more flexible or deep model.

Each step should answer whether added complexity changes performance, computation, interpretability, or robustness enough to matter.

## 12. Fair model comparison

A comparison is controlled when candidates use:

- the same train, validation, and test indices;
- transformations fitted under the same rules;
- the same target and eligible features;
- the same metric definitions;
- a documented tuning budget;
- a threshold selected by the same procedure; and
- the same final-test policy.

Change one named factor where possible. If several factors change, describe the comparison as a system comparison rather than attributing the effect to one component.

## 13. Validation selects; test estimates

Use validation data to choose imputation, indicators, features, model settings, and threshold. Lock the procedure before viewing the final test result.

If the test result leads to a redesign, the old test set has become development evidence. A new independent test is required for a fresh final estimate.

## 14. Error analysis

Aggregate metrics should lead to inspection, not end it.

Create mutually useful error categories such as:

- low-quality or missing inputs;
- minority class or subgroup;
- boundary cases;
- source or site shift;
- label ambiguity;
- temporal drift; and
- failure after a specific preprocessing step.

Count categories, review representative examples, and propose the next controlled experiment. Avoid selecting only dramatic errors.

## 15. The Week 3 lab

The synthetic lab contains:

- 1,000 independent observations;
- a positive prevalence of 0.12;
- one partially missing feature;
- a shared stratified split;
- training-only median imputation;
- a logistic baseline; and
- an optional missingness indicator.

The majority rule obtains high accuracy but zero recall. The two logistic candidates use the same split and differ by one feature decision. The comparison demonstrates a procedure, not a universal benefit of missingness indicators.

## 16. From audit to problem proposal

The problem proposal should connect:

- research question;
- intended use;
- dataset provenance and access;
- observation, target, and grouping definitions;
- data-quality risks;
- baseline and candidate method;
- evaluation protocol;
- feasibility limits; and
- a minimum successful result.

A feasible proposal states what can be tested with available data and resources. It does not promise a positive result.

## Review questions

1. Why is an unusual value not automatically an error?
2. What can and cannot be concluded from a missingness pattern?
3. Why can 88% accuracy represent complete failure on a rare class?
4. Where must oversampling occur?
5. What conditions make two model results comparable?
6. Why can a useful missingness indicator still be risky?
7. What evidence would make you reject a candidate dataset?
