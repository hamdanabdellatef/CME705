# Assignment 4: Controlled experiment

Change one factor in a working, reproducible baseline and determine what the evidence supports. Suitable factors include a learning rate, regularization coefficient, dropout probability, feature representation, architecture component, augmentation rule, or data treatment.

The experiment must keep the comparison interpretable. If several decisions change together, redesign the comparison or describe it as an exploratory package rather than evidence about one factor.

## Deliverables

Submit:

1. runnable code and configuration files;
2. raw per-run results in CSV, JSON, or another machine-readable format;
3. one compact table or figure generated from the recorded results; and
4. a 700–1,000 word analysis excluding tables, figure captions, and command output.

## 1. Research question and directional hypothesis

- Research question: _________________________________________________
- Baseline condition: ________________________________________________
- Changed factor: ____________________________________________________
- Candidate value or condition: ______________________________________
- Primary metric: ____________________________________________________
- Directional hypothesis: ____________________________________________

A useful hypothesis predicts both the direction and the reason. Example: “Adding dropout $p=0.30$ will reduce validation cross-entropy because the current network shows a growing training–validation gap.”

## 2. Controlled design

| Decision | Fixed value or rule | Evidence that it stayed fixed |
| --- | --- | --- |
| Dataset version or checksum | | |
| Training, validation, and test identifiers | | |
| Fitted preprocessing | | |
| Model initialization policy | | |
| Objective | | |
| Optimizer | | |
| Learning rate and schedule | | |
| Batch size | | |
| Maximum updates | | |
| Early-stopping rule | | |
| Metric implementation | | |
| Compute environment | | |

**The one changed factor:** _____________________________________________

If the changed factor requires a related implementation change, state it before running the experiment.

## 3. Split and test policy

Describe the unit kept together across splits:

__________________________________________________________________________

Describe the strongest remaining leakage risk:

__________________________________________________________________________

Choose one test policy:

- [ ] Test remains locked; comparison uses validation evidence only.
- [ ] Test will be opened once after the comparison and selection rule are fixed.
- [ ] Test was already used; results are provisional and need a new final evaluation.

Do not inspect test errors while modifying the candidate and continue to call the same test result independent.

## 4. Randomness and uncertainty

Use at least three random seeds when the method or data procedure is stochastic, unless compute limits require another justified uncertainty estimate.

| Source of randomness | Controlled or varied? | Recorded value |
| --- | --- | --- |
| Data split | | |
| Parameter initialization | | |
| Batch order | | |
| Dropout or augmentation | | |
| Other | | |

Planned seeds: _________________________________________________________

Uncertainty summary: standard deviation / confidence interval / bootstrap / paired differences / other: __________________

Explain why the chosen summary matches the unit of variation.

## 5. Predictions before execution

Complete before running:

- Expected training-curve change: ____________________________________
- Expected validation-curve change: __________________________________
- Expected primary-metric change: ____________________________________
- Possible negative side effect: ______________________________________
- Result that would contradict the hypothesis: ________________________

## 6. Implementation checks

Choose checks relevant to the changed factor.

- [ ] Tensor or array shapes are recorded.
- [ ] Analytical gradients pass a numerical or trusted-reference check.
- [ ] Training-only transformations are fitted on training data.
- [ ] Training and evaluation modes are explicit.
- [ ] Dropout reuses the forward mask during backward computation.
- [ ] Evaluation output is deterministic when expected.
- [ ] Saved checkpoint and stopping epoch are recorded.
- [ ] Configuration files differ only in the intended factor.

List the exact commands used for these checks:

~~~bash
# add commands here
~~~

## 7. Results

Report every run, including failures.

| Condition | Seed | Best epoch | Training metric | Validation metric | Test metric, if authorized | Runtime | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Baseline | | | | | | | |
| Baseline | | | | | | | |
| Baseline | | | | | | | |
| Candidate | | | | | | | |
| Candidate | | | | | | | |
| Candidate | | | | | | | |

Summarize:

| Condition | Runs | Primary metric mean | Variation or interval | Paired difference |
| --- | ---: | ---: | ---: | ---: |
| Baseline | | | | |
| Candidate | | | | |

The summary table must be generated from the submitted raw results.

## 8. Curve and error analysis

Include one learning-curve figure when the method is iterative. Mark the selected checkpoint.

Inspect class, subgroup, range, or error-direction effects:

| Group or error type | Baseline | Candidate | Difference | Interpretation |
| --- | ---: | ---: | ---: | --- |
| | | | | |
| | | | | |
| | | | | |

Separate observed facts from explanations that remain hypotheses.

## 9. Interpretation

Answer:

1. Did the changed factor produce the predicted effect?
2. Is the difference large relative to run-to-run variation?
3. Did training and validation behavior change in the expected way?
4. Did any class or subgroup become worse?
5. What implementation evidence supports the comparison?
6. Which alternative explanation remains plausible?
7. What claim is supported?
8. What claim is not supported?
9. What decision follows for the project?
10. What is the next single controlled change?

## Claim language

Prefer:

> Under the recorded dataset, split, implementation, and seed procedure, the candidate changed the primary validation metric by ___ relative to the baseline.

Avoid:

> This method is better.

## Submission check

- [ ] The hypothesis was written before the results.
- [ ] Exactly one intended factor changed.
- [ ] Data, splits, environment, and fixed controls are recorded.
- [ ] Random seeds or another uncertainty method are explicit.
- [ ] Test status is declared.
- [ ] Raw per-run results are submitted.
- [ ] The table or figure is generated from those results.
- [ ] Failures and negative effects are included.
- [ ] Class, subgroup, or error-direction evidence is discussed.
- [ ] The conclusion matches the scope of the procedure.
- [ ] The next experiment changes one factor.

## Assessment rubric

| Criterion | Weight | Strong evidence |
| --- | ---: | --- |
| Question and hypothesis | 15% | directional, falsifiable, and tied to a mechanism |
| Control of variables | 20% | one intended change with verified fixed controls |
| Reproducibility | 15% | exact state, commands, configurations, seeds, and raw results |
| Evaluation validity | 20% | split roles, leakage control, metric fit, and honest test status |
| Uncertainty and error analysis | 15% | variation is quantified and class or subgroup effects are inspected |
| Interpretation | 15% | claim is proportional to evidence and leads to a justified next decision |
