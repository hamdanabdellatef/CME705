# Week 6 milestone: Reproducible baseline

Run the baseline planned in Week 5 and provide enough evidence for another student to reproduce the result. A failed or weak baseline is acceptable when the configuration, observations, and next decision are documented.

## Deliverable

Submit the repository state or archive containing the runnable baseline, plus a 700–1,000 word record excluding tables and command output.

## 1. Repository state

- Repository location: ________________________________________________
- Commit identifier or archive checksum: ______________________________
- Entry-point command: _______________________________________________
- Configuration file: ________________________________________________
- Result files: ______________________________________________________

The entry-point command should start from a fresh shell in the repository root.

## 2. Environment

Record the actual environment used.

| Item | Actual value |
| --- | --- |
| Operating system | |
| Python version | |
| NumPy version | |
| PyTorch version, if used | |
| CPU or accelerator | |
| Installation command | |
| Environment lock or requirements file | |

Do not include credentials, private dataset copies, or machine-specific secrets.

## 3. Data and split identity

| Item | Actual value |
| --- | --- |
| Dataset name and version | |
| Source or access date | |
| Dataset checksum or immutable identifier | |
| Included observations | |
| Training identifiers or split file | |
| Validation identifiers or split file | |
| Test identifiers or split file | |
| Split seed | |
| Leakage check performed | |

State any difference from the Week 5 plan and explain why it changed.

## 4. Final configuration

| Decision | Planned value | Actual value |
| --- | --- | --- |
| Baseline method | | |
| Input representation | | |
| Fitted preprocessing | | |
| Parameter initialization | | |
| Objective | | |
| Optimizer or update rule | | |
| Learning rate | | |
| Batch size | | |
| Maximum epochs or updates | | |
| Stopping rule | | |
| Random seed | | |

A change is not automatically a problem. An undocumented change makes the result difficult to interpret.

## 5. Reproduction command and output

Command:

~~~bash
python path/to/train_baseline.py --config path/to/baseline.json
~~~

Record the exit status, elapsed time, stopping reason, and output-file locations.

| Record | Value |
| --- | --- |
| Exit status | |
| Elapsed time | |
| Updates completed | |
| Stopping reason | |
| Training history file | |
| Selected configuration record | |
| Final evaluation file | |

## 6. Results

Report training, validation, and test quantities separately.

| Split | Metric | Value | Role in the procedure |
| --- | --- | ---: | --- |
| Training | | | parameter fitting |
| Validation | | | configuration selection |
| Test | | | final evaluation |

State whether test data were used once after the configuration was fixed. If they were used during debugging or selection, label the result as provisional and prepare a new independent evaluation plan.

## 7. First error analysis

Inspect at least three errors or one scientifically meaningful subgroup.

| Observation or subgroup | Target | Prediction | Relevant feature or context | Possible explanation |
| --- | --- | --- | --- | --- |
| | | | | |
| | | | | |
| | | | | |

Separate observed facts from hypotheses. Do not change the model using test errors and keep calling the same test score independent.

## 8. Interpretation

Complete:

- Evidence that the implementation trained as intended: _______________
- Evidence that the baseline is credible or inadequate: _______________
- Claim supported by the current result: ______________________________
- Claim the result does not support: __________________________________
- Most important uncertainty or failure mode: _________________________
- One controlled next experiment: ____________________________________

## Reproduction check

Ask a classmate to run the command without verbal guidance.

- Reproducer: ________________________________________________________
- Date: ______________________________________________________________
- Same environment created: yes / no
- Command completed: yes / no
- Same result within stated tolerance: yes / no
- Missing or ambiguous instruction: __________________________________
- Correction made: __________________________________________________

## Submission check

- [ ] Repository state is identified by commit or checksum.
- [ ] One command reproduces the baseline from the repository root.
- [ ] Environment, dataset version, and split identities are recorded.
- [ ] Actual configuration and deviations from the plan are explicit.
- [ ] Training, validation, and test roles remain separate.
- [ ] Test use is described honestly.
- [ ] At least three errors or one meaningful subgroup are inspected.
- [ ] The claim, limitation, and next controlled experiment are stated.
