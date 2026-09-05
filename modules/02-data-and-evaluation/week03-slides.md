# Week 3 slide source: Data quality and controlled model comparison

This file is the accessible text version of the PowerPoint deck. Speaker prompts and timing are in the presentation and instructor guide.

## Slide 1 — CME705 Machine Learning

**Week 3: Data quality and controlled model comparison**

Guiding question: What must we learn about the data before comparing models?

## Slide 2 — Today’s evidence

Each student will produce:

1. a compact data audit;
2. a missingness and imbalance interpretation;
3. a controlled comparison protocol; and
4. a feasible problem-proposal draft.

## Slide 3 — “The dataset has 1,000 rows”

Row count does not establish:

- independent observations;
- valid measurements;
- reliable targets;
- representative classes or groups;
- deployment-time feature availability; or
- permission to use the data.

## Slide 4 — Data quality has several dimensions

- provenance and permitted use;
- observation and target definition;
- completeness;
- validity and consistency;
- duplicates and dependence;
- label quality;
- representativeness; and
- feature availability and stability.

## Slide 5 — Data are produced, not found

Collection → selection → measurement → labeling → storage → cleaning → analysis

Every stage can shape the patterns a model learns.

## Slide 6 — Audit before modeling

Record source, version, population, observation unit, feature types and ranges, target timing, class counts, missingness, duplicates, conflicting labels, outliers, and prior exclusions.

An audit identifies questions before it prescribes deletion.

## Slide 7 — Suspicious is not the same as wrong

A repeated row can be a duplicate or a valid repeated measurement.

An extreme value can be an error, a unit mismatch, or a rare valid case.

A label conflict can indicate annotation disagreement, ambiguity, or a bad join.

Investigate with identifiers, timestamps, source records, and domain constraints.

## Slide 8 — Missingness is a pattern

A missing value can represent failure, omission, inapplicability, denied access, or unavailable measurement.

Inspect missingness by class, group, source, and time.

## Slide 9 — Missingness mechanisms guide questions

| Concept | Working assumption | Main question |
| --- | --- | --- |
| MCAR | Missingness is unrelated to observed and unobserved values | Could a random process explain it? |
| MAR | Observed information explains missingness | Which recorded variables predict it? |
| MNAR | Missingness depends on an unobserved or missing value | What selection process remains hidden? |

The table cannot prove the mechanism.

## Slide 10 — Repair without crossing the boundary

1. Create the split.
2. Fit imputation on training data.
3. Optionally append documented missingness indicators.
4. Apply the fitted transformation to validation and test data.
5. test stability across groups and time.

Imputation creates a representation; it does not recover truth.

## Slide 11 — Imbalance changes score meaning

With 12% positives, an always-negative model has:

- accuracy = 0.88;
- recall = 0;
- specificity = 1.00; and
- balanced accuracy = 0.50.

A high average can hide complete failure on the rare class.

## Slide 12 — Start with counts

Before selecting a metric, report:

- positive and negative counts;
- confusion-matrix cells;
- group and subgroup counts; and
- collection or sampling prevalence.

Then select recall, precision, specificity, balanced accuracy, or another metric through the decision cost.

## Slide 13 — Responses to imbalance

- collect more rare examples;
- use class-weighted training;
- resample training data only;
- tune a threshold with validation data;
- stratify independent observations; and
- report class-aware metrics.

Changing training prevalence does not change deployment prevalence.

## Slide 14 — Screen every feature

Ask whether the feature is available at prediction time, meaningful, stable, sufficiently complete, correctly encoded, nonredundant, privacy-compatible, and free of target leakage.

A predictive feature can still be unusable.

## Slide 15 — Build a baseline ladder

1. constant or prevalence predictor;
2. simple domain rule;
3. linear or logistic model;
4. established conventional method;
5. flexible or deep model.

Each step must justify added complexity.

## Slide 16 — Fair comparison changes one named factor

Hold constant:

- split indices;
- eligible data;
- preprocessing rules;
- metrics;
- tuning budget;
- threshold procedure; and
- final-test policy.

If several factors change, report a system comparison.

## Slide 17 — Week 3 lab results

| Candidate | Accuracy | Balanced accuracy | Recall |
| --- | ---: | ---: | ---: |
| Majority baseline | 0.880 | 0.500 | 0.000 |
| Median-imputed logistic | 0.665 | 0.774 | 0.917 |
| Imputation + indicator | 0.780 | 0.857 | 0.958 |

The two logistic candidates use the same split and differ by one documented feature decision.

## Slide 18 — Error analysis creates the next experiment

Categorize errors by missing or low-quality input, rare class or subgroup, boundary case, source shift, label ambiguity, temporal drift, or preprocessing failure.

Count categories, review representative examples, and propose one controlled test.

## Slide 19 — A dataset card supports the claim

Record motivation, composition, collection, preprocessing, permitted use, distribution, maintenance, and limitations.

A dataset card is evidence for decisions, not a guarantee of quality.

## Slide 20 — Minimum viable experiment

A feasible project can:

1. load and audit the data;
2. reproduce a fixed split;
3. run a credible baseline;
4. calculate the primary metric;
5. compare one controlled change; and
6. explain limitations.

## Slide 21 — Problem proposal

Submit:

- a precise research question;
- dataset card draft;
- data-quality audit;
- baseline and candidate;
- evaluation protocol;
- risk and feasibility analysis; and
- minimum successful result.

Next: neural-network computation.
