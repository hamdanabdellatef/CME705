# Week 7 milestone: Evaluation review

Audit the baseline evaluation before changing the model. The review should make the unit of observation, split roles, metric, leakage controls, class-specific behavior, and test status explicit.

## Deliverable

Submit a 700–1,000 word review excluding tables and command output. Link the exact baseline repository state and result files from Week 6.

## 1. Task and prediction unit

- Research question: _________________________________________________
- Unit of observation: _______________________________________________
- Target definition: _________________________________________________
- Task type: binary / mutually exclusive multiclass / multilabel / regression
- Number and names of classes, if applicable: _________________________
- Decision produced by the model: ____________________________________

Explain why the output layer and objective match the task type.

## 2. Split roles

| Split | Observations or groups | Role | Was it used? |
| --- | ---: | --- | --- |
| Training | | fit parameters and fitted preprocessing | |
| Validation | | select configurations or stopping point | |
| Test | | final evaluation after decisions are fixed | |

Record the split file, identifiers, or checksum. If observations share a subject, device, site, sequence, document, or time window, state which unit was kept together.

## 3. Leakage audit

Answer each item with evidence.

- Were duplicates or near duplicates checked across splits?
- Were related observations grouped before splitting?
- Was time order preserved when deployment predicts the future?
- Were imputation, scaling, vocabulary, feature selection, and augmentation fitted using training data only?
- Does any feature directly or indirectly contain the target or a post-outcome measurement?
- Were validation or test examples inspected while changing the model?

Describe the highest remaining leakage risk:

__________________________________________________________________________

## 4. Primary metric specification

| Item | Decision |
| --- | --- |
| Metric name | |
| Formula or implementation | |
| Direction: higher or lower is better | |
| Averaging: micro, macro, weighted, or none | |
| Positive class or class order | |
| Threshold or decision rule | |
| Unit over which uncertainty is estimated | |
| Why this metric matches the research question | |

Do not write only “accuracy” or “F1.” State the exact averaging and decision rule.

## 5. Baseline context

Compare the trained model with a simple reference that uses the same test observations.

| Model | Primary metric | Secondary metric | Cross-entropy or objective | Notes |
| --- | ---: | ---: | ---: | --- |
| Majority, random, persistence, or simple rule | | | | |
| Week 6 baseline | | | | |

Explain what improvement over the reference means in the application. A small numerical gain may be irrelevant if it does not change a decision or reduce an important error.

## 6. Class-specific evidence

For multiclass classification, state the confusion-matrix orientation.

**Rows represent:** __________________  **Columns represent:** __________________

| True class | Support | Correct | Most common confusion | Recall |
| --- | ---: | ---: | --- | ---: |
| | | | | |
| | | | | |
| | | | | |

For another task type, replace this table with meaningful subgroups, ranges, horizons, or outcome strata.

Explain whether the aggregate metric hides weak performance on a class or subgroup.

## 7. Selection history and test status

List every configuration decision informed by validation evidence:

| Decision | Candidates | Validation evidence | Selected value |
| --- | --- | --- | --- |
| | | | |
| | | | |
| | | | |

Then declare one test status:

- [ ] **Locked:** test data have not been opened.
- [ ] **Final use:** test data were opened once after all choices were fixed.
- [ ] **Provisional:** test data influenced debugging, selection, or interpretation.

If provisional, describe how an independent final evaluation will be obtained.

## 8. Error analysis

Inspect at least three errors or one scientifically meaningful subgroup.

| Observation or subgroup | Target | Prediction | Confidence or score | Relevant context | Fact or hypothesis |
| --- | --- | --- | ---: | --- | --- |
| | | | | | |
| | | | | | |
| | | | | | |

Separate observed facts from possible explanations. Do not tune a model on test errors while continuing to call the same test result independent.

## 9. Interpretation and decision

Complete:

- Evidence supporting the evaluation design: __________________________
- Most serious weakness in the evaluation: ____________________________
- Claim supported by the current result: ______________________________
- Claim the result does not support: __________________________________
- One evaluation correction, if needed: _______________________________
- One controlled next experiment: ____________________________________

## Submission check

- [ ] The task type, unit of observation, output, and objective agree.
- [ ] Training, validation, and test roles are explicit.
- [ ] Split identity and leakage checks are recorded.
- [ ] The primary metric includes averaging and decision details.
- [ ] A simple reference uses the same evaluation observations.
- [ ] Class or subgroup behavior is reported.
- [ ] Configuration selection is separated from final testing.
- [ ] Test status is declared honestly.
- [ ] Errors are inspected without turning the test set into training data.
- [ ] The claim, limitation, and next decision follow from the evidence.
