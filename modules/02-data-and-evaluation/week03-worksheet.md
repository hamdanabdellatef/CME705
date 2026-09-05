# Week 3 worksheet: Data quality and controlled comparison

Name: ________________________________  Date: __________________

Record observations separately from decisions. A suspicious value is a question until evidence supports a correction.

## 1. Opening claim

A dataset is described as “clean and ready” because it contains 1,000 rows.

List five facts the row count does not establish.

1. ______________________________________________________________________
2. ______________________________________________________________________
3. ______________________________________________________________________
4. ______________________________________________________________________
5. ______________________________________________________________________

## 2. First-pass audit

Study the illustrative extract.

| sample_id | entity_id | source | feature_a | feature_b | target |
| --- | --- | --- | ---: | ---: | ---: |
| S01 | E01 | north | 0.42 | 12.1 | 0 |
| S02 | E02 | north | missing | 18.5 | 1 |
| S03 | E03 | south | 0.91 | 999.0 | 1 |
| S04 | E04 | north | 0.18 | 9.2 | 0 |
| S04 | E04 | north | 0.18 | 9.2 | 0 |
| S06 | E05 | south | 0.70 | 15.2 | 0 |
| S07 | E05 | south | 0.70 | 15.2 | 1 |

For each observation, distinguish **evidence**, **possible explanation**, and **next check**.

| Observation | Possible explanation | Evidence needed before action |
| --- | --- | --- |
| Missing feature_a | | |
| feature_b = 999.0 | | |
| Repeated sample_id S04 | | |
| Same entity and features with conflicting targets | | |

Which row would you delete immediately? Explain why a defensible answer may be “none yet.”

__________________________________________________________________________

## 3. Provenance questions

Candidate dataset: ________________________________________________________

- Creator and source: _____________________________________________________
- Version and retrieval date: _____________________________________________
- License or access terms: ________________________________________________
- Population or collection setting: _______________________________________
- Inclusion and exclusion process: ________________________________________
- Label source and timing: ________________________________________________
- Transformations already performed: _____________________________________

One claim the provenance currently supports:

__________________________________________________________________________

One claim it does not yet support:

__________________________________________________________________________

## 4. Missingness audit

Choose one incomplete feature.

Feature: __________________________  Overall missing rate: _________________

| Slice | Missing rate | Why this comparison matters |
| --- | ---: | --- |
| Positive class | | |
| Negative class | | |
| Source or site 1 | | |
| Source or site 2 | | |
| Early time period | | |
| Later time period | | |

Possible mechanisms to investigate:

- MCAR question: __________________________________________________________
- MAR question using observed information: _________________________________
- MNAR question involving an unobserved value: _____________________________

Why can this table not prove a missingness mechanism?

__________________________________________________________________________

## 5. Imbalance and the accuracy trap

A binary dataset contains 120 positive and 880 negative observations.

An always-negative model predicts all 1,000 examples as negative.

- Accuracy: _______________________________________________________________
- Recall for the positive class: __________________________________________
- Specificity: ____________________________________________________________
- Balanced accuracy: _____________________________________________________

Why would accuracy alone be inadequate?

__________________________________________________________________________

Name a baseline and primary metric for your possible project.

Baseline: _________________________________________________________________

Primary metric: _______________________ Reason: ____________________________

## 6. Responses to imbalance

Mark whether each action belongs **before splitting**, **training only**, **validation**, or **reporting**.

| Action | Stage | Main risk or purpose |
| --- | --- | --- |
| Define the positive class | | |
| Oversample rare examples | | |
| Apply class-weighted loss | | |
| Select a threshold | | |
| Report class counts and recall | | |
| Create a stratified split for independent rows | | |

What goes wrong if oversampling occurs before splitting?

__________________________________________________________________________

## 7. Feature-quality screen

Evaluate one candidate feature.

Feature: _________________________________________________________________

| Question | Answer |
| --- | --- |
| Is it available at prediction time? | |
| Who or what generates it? | |
| Could its definition change? | |
| How often is it missing? | |
| Does it contain the target or a later consequence? | |
| Does it expose sensitive information? | |
| What transformation is required? | |
| What simpler feature could replace it? | |

Decision: **include / investigate / exclude**

Evidence for the decision: __________________________________________________

## 8. Predict the lab

The Week 3 lab contains a 12% positive class and one incomplete feature.

Predict:

- Majority-rule accuracy: approximately ___________________________________
- Majority-rule positive recall: __________________________________________
- Metric that will expose the majority rule: _______________________________
- Will a missingness indicator help in this simulation? Why? ______________

Run:

```bash
python labs/week03_data_quality_numpy.py
```

Record the test results.

| Candidate | Threshold | Accuracy | Balanced accuracy | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Majority baseline | | | | | |
| Median-imputed logistic | | | | | |
| Imputation + indicator | | | | | |

Which comparison is controlled?

__________________________________________________________________________

What changed between the two logistic candidates?

__________________________________________________________________________

What does the result not establish about a real dataset?

__________________________________________________________________________

## 9. Fair comparison protocol

Complete before comparing two candidates.

| Element | Fixed rule |
| --- | --- |
| Train/validation/test indices | |
| Eligible input features | |
| Fitted preprocessing | |
| Metric definitions | |
| Tuning budget | |
| Threshold-selection rule | |
| Random seeds | |
| Final-test policy | |

Candidate A: __________________________ Candidate B: _______________________

Single intended difference: _______________________________________________

If more than one factor changes, how will you describe the comparison?

__________________________________________________________________________

## 10. Error-analysis plan

Choose three useful error categories for the project.

| Category | Why it matters | What next experiment could test |
| --- | --- | --- |
| | | |
| | | |
| | | |

How will you choose examples without selecting only dramatic failures?

__________________________________________________________________________

## 11. Problem-proposal planner

- Research question: ______________________________________________________
- Intended use: ___________________________________________________________
- Observation and target: _________________________________________________
- Dataset and access: _____________________________________________________
- Main data-quality risk: _________________________________________________
- Grouping or time constraint: ____________________________________________
- Baseline: _______________________________________________________________
- Primary metric: _________________________________________________________
- Minimum successful result: ______________________________________________
- Evidence that would make the project infeasible: _________________________

## 12. Exit ticket

1. The highest-priority data question is __________________________________
2. The first baseline will be _____________________________________________
3. The comparison will hold _____________________________________ constant
4. Before submission, I must verify _______________________________________
