# Week 2 worksheet: Designing evidence for unseen data

Name: ________________________________  Date: __________________

Use this worksheet during the lecture and lab. Record decisions before running an experiment.

## 1. Opening claim

A team reports: “Our classifier achieved 95% accuracy.”

List four questions you need before deciding whether this is strong evidence.

1. ______________________________________________________________________
2. ______________________________________________________________________
3. ______________________________________________________________________
4. ______________________________________________________________________

Complete the evaluation claim:

> We want to estimate performance when the fitted system is applied to
> ________________________________________________________________________.

## 2. Give each partition one role

For each action, write **train**, **validation**, or **test**.

| Action | Partition |
| --- | --- |
| Estimate feature means and standard deviations | |
| Fit model weights | |
| Choose the learning rate | |
| Select a decision threshold | |
| Produce the final performance estimate | |
| Decide whether to add a feature | |

Why does repeated test-set use weaken the final claim?

__________________________________________________________________________

## 3. Identify the independent unit

A wearable study records 20 sensor windows per participant. The goal is to classify activity for participants not seen during development.

- Storage unit: ___________________________________________________________
- Unit that must remain unseen: ___________________________________________
- Grouping variable: ______________________________________________________
- Invalid split: __________________________________________________________
- Defensible split: _______________________________________________________

What would change if the intended use were future windows from the same enrolled participants?

__________________________________________________________________________

## 4. Choose a split

For each study, select **random**, **stratified**, **grouped**, or **chronological** and justify the choice.

| Study | Split | One-sentence justification |
| --- | --- | --- |
| Predict next month’s electricity demand from prior months | | |
| Classify independent manufactured components; defects are rare | | |
| Diagnose images when each patient contributes several images | | |
| Categorize independently sampled documents from one stable archive | | |

## 5. Leakage hunt

For each workflow, identify the leakage and propose a correction.

### A. Scaling

The researcher standardizes the complete dataset, then creates the split.

Leakage: _________________________________________________________________

Correction: ______________________________________________________________

### B. Image augmentation

One original image is cropped into 30 variants. The variants are randomly assigned to training and test sets.

Leakage: _________________________________________________________________

Correction: ______________________________________________________________

### C. Forecasting

Daily records are randomly divided, and future days appear in training while earlier days appear in testing.

Leakage: _________________________________________________________________

Correction: ______________________________________________________________

### D. Target timing

A feature recorded after the outcome is used to predict that outcome.

Leakage: _________________________________________________________________

Correction: ______________________________________________________________

## 6. Read a confusion matrix

A held-out evaluation contains:

|  | Predicted positive | Predicted negative |
| --- | ---: | ---: |
| Actually positive | TP = 24 | FN = 12 |
| Actually negative | FP = 6 | TN = 58 |

Calculate to three decimal places.

- Accuracy: _______________________________________________________________
- Precision: ______________________________________________________________
- Recall: __________________________________________________________________
- Specificity: _____________________________________________________________
- Balanced accuracy: ______________________________________________________

If false negatives are especially costly, which result deserves immediate attention and why?

__________________________________________________________________________

## 7. Predict the lab result

The lab contains four rows per synthetic participant. Rows from the same participant have similar measurements and the same label.

Before running the code, predict:

- Which split will have participant overlap? _______________________________
- Which split will appear more accurate? __________________________________
- Why? ___________________________________________________________________

Run:

```bash
python labs/week02_evaluation_numpy.py
```

Record the output.

| Quantity | Result |
| --- | ---: |
| Participants | |
| Rows | |
| Row-wise split: overlapping groups | |
| Row-wise test balanced accuracy | |
| Grouped split: overlapping groups | |
| Grouped test balanced accuracy | |

Was your prediction supported? Explain the mechanism rather than only reporting the scores.

__________________________________________________________________________

What conclusion does the grouped result support?

__________________________________________________________________________

What does the synthetic result not establish?

__________________________________________________________________________

## 8. Repair the workflow

Number these operations from 1 to 7.

| Operation | Order |
| --- | ---: |
| Transform validation and test data with training statistics | |
| Define the evaluation claim and independent unit | |
| Report final test performance | |
| Fit the model on training data | |
| Select the model and threshold with validation data | |
| Create the split | |
| Fit preprocessing on training data | |

## 9. Dataset feasibility screen

Candidate dataset: ________________________________________________________

Source or URL: ____________________________________________________________

| Question | Current answer | Evidence still needed |
| --- | --- | --- |
| Is access practical and permitted? | | |
| What is one observation? | | |
| What is the target and when is it measured? | | |
| Which observations are dependent? | | |
| Which split matches the intended use? | | |
| Are important classes or groups represented? | | |
| What simple baseline is feasible? | | |
| Can preparation be reproduced? | | |

Decision: **continue / investigate / replace**

One action before Week 3: _________________________________________________

## 10. Exit ticket

1. Write one leakage risk for your possible project.

   ________________________________________________________________________

2. State the split you currently expect to use and why.

   ________________________________________________________________________

3. Name one result that would cause you to reconsider the dataset.

   ________________________________________________________________________
