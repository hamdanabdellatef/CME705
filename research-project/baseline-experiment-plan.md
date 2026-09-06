# Week 5 milestone: Baseline experiment plan

Plan the first experiment before running it. The plan turns the literature decision from Week 4 into a reproducible baseline and protects the final test evidence from repeated tuning.

## Deliverable

Submit a completed plan of 700–1,000 words, excluding tables and references. Include the exact command you expect to run and a compact configuration table. No result is required at this stage.

## 1. Question and baseline

- Research question: _________________________________________________
- Unit of observation: _______________________________________________
- Target or outcome: _________________________________________________
- Simplest credible baseline: ________________________________________
- Literature source supporting the baseline: __________________________
- Claim the baseline experiment can test: _____________________________

Explain why this method is feasible and why a more complex method should have to improve on it.

## 2. Data and frozen split

| Decision | Planned value | Reason and source |
| --- | --- | --- |
| Dataset name and version | | |
| Included population or records | | |
| Unit of splitting | | |
| Training allocation | | |
| Validation allocation | | |
| Test allocation | | |
| Split seed or fixed identifiers | | |
| Leakage check | | |

State what will be fitted only on training data, including imputation, scaling, feature selection, and augmentation statistics.

## 3. Objective and evaluation

The **training objective** determines the gradient used to update parameters. The **evaluation metric** measures the behavior relevant to the research question. They may differ.

| Item | Planned value | Why it fits the task |
| --- | --- | --- |
| Training objective | | |
| Primary validation metric | | |
| Secondary diagnostic | | |
| Direction of improvement | lower / higher | |
| Decision threshold, if any | | |
| Uncertainty estimate | | |

The test split should remain unopened until the baseline configuration is fixed.

## 4. Training configuration

| Configuration item | Planned value |
| --- | --- |
| Parameter initialization | |
| Optimizer or update rule | |
| Learning-rate candidates | |
| Batch size | |
| Maximum epochs or updates | |
| Stopping rule | |
| Random seeds | |
| Hardware or compute limit | |
| Expected run time | |

Name the one decision the validation comparison will select. Hold the other listed decisions constant.

## 5. Expected records

Plan to retain:

- a machine-readable configuration;
- training and validation curves or tables;
- the selected configuration and selection rule;
- the final test metric after selection;
- one error-analysis table or example set; and
- the software environment and exact reproduction command.

Expected command:

~~~bash
python path/to/train_baseline.py --config path/to/baseline.json
~~~

Expected output files: ________________________________________________

## 6. Interpretation rules

Complete these statements before seeing results:

- Evidence that the implementation is learning: ______________________
- Evidence that the learning rate is too small: _______________________
- Evidence that the run is unstable: _________________________________
- Evidence that the baseline is credible: _____________________________
- Result that would justify testing a more complex method: ____________
- Result that would require debugging or revising the protocol: _______

A failed run may be useful evidence when the configuration and diagnostic observations are recorded.

## Submission check

- [ ] The baseline is supported by the Week 4 literature comparison.
- [ ] Dataset version, observation unit, and split unit are explicit.
- [ ] Preprocessing is fitted on training data only.
- [ ] Training objective and evaluation metric are distinguished.
- [ ] One validation-controlled decision is named.
- [ ] Test data remain reserved for the final selected configuration.
- [ ] Seeds, stopping rule, environment, compute, and command are recorded.
- [ ] Expected outputs and interpretation rules are written before execution.
