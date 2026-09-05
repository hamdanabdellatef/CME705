# Week 2 instructor guide: Generalization and trustworthy evaluation

## Purpose

Students should leave Week 2 able to design an evaluation boundary before selecting a sophisticated model. The central teaching move is to connect the split to a concrete future-use claim rather than present train, validation, and test partitions as fixed percentages.

The three-hour plan includes a short break. The PowerPoint contains matching speaker notes.

## Before class

- Run `python labs/week02_evaluation_numpy.py`.
- Confirm that the row-wise split has overlapping groups and the grouped split has none.
- Print or distribute [the worksheet](../modules/02-data-and-evaluation/worksheet.md).
- Ask students to bring one candidate dataset name or URL.
- Prepare one example from the cohort’s research interests in addition to the wearable repeated-measurement case.
- Avoid showing the lab’s exact scores before students record a prediction.

## Learning evidence

Collect or inspect:

1. the split-design table;
2. one corrected leakage workflow;
3. confusion-matrix calculations;
4. the lab interpretation; and
5. the dataset-feasibility screen.

## Three-hour sequence

| Time | Activity | Instructor action | Student evidence |
| ---: | --- | --- | --- |
| 0–10 min | Opening claim | Display “95% accuracy” and withhold context | Four questions needed to interpret it |
| 10–25 min | Evaluation claim | Contrast new rows, new entities, and future time | Completed future-case sentence |
| 25–45 min | Train/validation/test roles | Sort development actions by partition | Worksheet role table |
| 45–65 min | Independence and split design | Use repeated sensor windows to expose dependence | Sampling unit and grouping variable |
| 65–80 min | Split activity | Compare random, stratified, grouped, and chronological splits | Four justified choices |
| 80–90 min | Break | — | — |
| 90–115 min | Leakage hunt | Trace which information crosses the evaluation boundary | Four corrected workflows |
| 115–135 min | Metrics | Build the confusion matrix before formulas | Calculated and interpreted metrics |
| 135–145 min | Thresholds and baselines | Connect errors to cost and capacity | Metric and baseline choice |
| 145–165 min | NumPy lab | Require a prediction, execute, then explain the mechanism | Row-wise and grouped results |
| 165–175 min | Dataset feasibility | Review candidate datasets in pairs | Continue/investigate/replace decision |
| 175–180 min | Exit ticket | Collect split, leakage risk, and rejection evidence | Three concise responses |

## Facilitation notes

### Opening: a score without a claim

Accept questions about data source, held-out evaluation, class balance, metric definition, split, baseline, and costs. Group them under population, independence, comparison, and consequence. Use this to show that evaluation begins with a claim and sampling design.

### Partition roles

Use examples of choices that students often make implicitly:

- scaling parameters belong to training;
- architecture and learning rate belong to validation;
- a tuned decision threshold belongs to validation; and
- the final performance estimate belongs to test.

If students say the test set is used to “check whether the model is good,” ask what happens after a disappointing check. If they change the model, the test result has entered model selection.

### Independent unit

Ask:

1. What does one stored row represent?
2. Which rows share information?
3. What must be new when the model is used?

Do not declare grouped splitting universally superior. If the system is personalized after observing each user, future windows from known users may require a chronological or within-user design.

### Leakage hunt

Require students to name the information crossing the boundary and why it would be unavailable at prediction or evaluation time.

Expected corrections:

- split before scaling and fit statistics on training only;
- assign all augmentations of one original to one partition;
- train on earlier time and evaluate on later time; and
- remove features recorded after the target time.

### Confusion-matrix activity

For TP = 24, FN = 12, FP = 6, and TN = 58:

- accuracy = 0.820;
- precision = 0.800;
- recall = 0.667;
- specificity = 0.906; and
- balanced accuracy = 0.786.

Ask why the same predictions can support multiple correct summaries. Then ask which summary addresses the application’s cost.

### Lab

The lab intentionally gives a nearest-neighbor baseline access to stable participant signatures. A row-wise split usually places measurements from the same participant in both training and test sets. The grouped split tests new participants.

Students should state the mechanism:

> The row-wise score is optimistic for a new-participant claim because the model can match a test row to another row from the same participant.

A correct interpretation does not claim that the grouped score is universally unbiased. The simulation is small, and another deployment claim could require another split.

### Dataset feasibility

Pairs should challenge one another on access, label timing, dependence, and sample counts. “Publicly available” does not establish permission, documentation quality, or suitability.

The Week 2 product feeds into the [problem proposal](../assignments/02-problem-proposal.md).

## Common misconceptions

| Misconception | Response |
| --- | --- |
| A random split is always objective | Random assignment cannot repair dependence or time travel |
| Validation is optional if training is careful | Model and threshold choices still need evidence independent of fitting |
| Scaling before splitting is harmless | Test distribution statistics have influenced the representation |
| A fixed seed makes the result reliable | It makes one procedure repeatable; robustness needs broader evidence |
| Accuracy is misleading, so never use it | Suitability depends on class balance, costs, and the evaluation question |
| Cross-validation removes all uncertainty | It cannot repair a mismatched population or invalid sampling unit |

## Formative feedback language

- “Name what must remain unseen.”
- “Which rows share a source?”
- “This statistic was fitted before the boundary was created.”
- “Your metric does not yet express the cost you described.”
- “State what decision the threshold supports.”
- “Identify what this test set represents and excludes.”

## Adaptations

### Ninety-minute class

Use slides 1–9, 11–18, and 21. Complete the split and leakage sections in class. Assign metric calculation and dataset feasibility after class.

### Online delivery

Use breakout pairs for split design and dataset screening. Ask each group to post one corrected leakage workflow. Run the lab through screen sharing, pausing before execution for predictions.

### Students with limited Python experience

Provide the command and ask them to interpret overlap counts and scores. Editing code is optional this week; understanding the evaluation mechanism is required.

## After class

- Review whether students named a defensible independent unit.
- Flag datasets whose access, label timing, or group information is uncertain.
- Use those flags to guide Week 3 problem-proposal feedback.
- Record any lab-output change caused by environment or code revisions.
