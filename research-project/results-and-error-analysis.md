# Results and error analysis milestone

**Due:** end of Week 10  
**Purpose:** turn the frozen experiment into a reproducible claim, then use systematic errors to choose the next controlled experiment.

This milestone reports the result of the method selected in Week 9. The test set must remain locked until preprocessing, model configuration, training budget, checkpoint rule, and reported metrics are fixed.

## Required submission

Submit one concise report section plus the files or commands needed to reproduce its tables. Use the headings below.

## 1. Frozen experiment configuration

Record:

- repository commit or release identifier;
- dataset name, version, access date, and permitted local identifier;
- task unit and target definition;
- training, validation, and test construction;
- sample counts and class or outcome distribution by split;
- preprocessing fitted on training data;
- model and baseline configurations;
- objective, optimizer, learning rate, batch size, maximum epochs, and patience;
- seeds, package versions, and device; and
- the exact command that runs the experiment.

State which choices were fixed before the test set was opened.

## 2. Selection history

Report the criterion used for checkpoint or model selection. Include:

| Candidate or epoch | Training result | Validation result | Selected? | Reason |
| --- | ---: | ---: | --- | --- |
| | | | | |
| | | | | |
| | | | | |

For early stopping, report the best epoch, stopped epoch, minimum required improvement, patience, and evidence that the best checkpoint was restored.

Do not select using test performance.

## 3. Locked test results

Provide a baseline and the selected method in the same table.

| Method | Primary metric | Secondary metric | Class/subgroup range | Runtime or cost |
| --- | ---: | ---: | ---: | ---: |
| baseline | | | | |
| selected method | | | | |

Include uncertainty when the dataset and design support it. State the unit over which uncertainty was calculated.

For classification, include a labeled confusion matrix and class-specific precision, recall, or another justified class-aware measure. For regression, include residual summaries across meaningful ranges or groups. For representation or generative work, define the quantitative and qualitative evaluation protocol before viewing outcomes.

## 4. Structured error analysis

Create an error table with at least ten errors, or every error if fewer than ten exist.

| Stable ID | True/expected | Predicted/observed | Confidence or residual | Relevant metadata | Error category |
| --- | --- | --- | ---: | --- | --- |
| | | | | | |

Then aggregate the categories.

| Error category | Count | Fraction of errors | Proposed explanation | Evidence still needed |
| --- | ---: | ---: | --- | --- |
| | | | | |
| | | | | |

Choose categories tied to the research question or data-generating process. Do not infer sensitive attributes from examples. Distinguish an observed pattern from a hypothesized cause.

## 5. One robustness or ablation check

Change one declared factor while holding the rest of the evaluation fixed. Examples include:

- remove one model component;
- vary one regularization value using validation evidence;
- evaluate one plausible input perturbation;
- compare with a simpler baseline; or
- repeat across seeds without changing the test-driven procedure.

Report both the primary metric and any affected class or subgroup evidence. Explain what the check isolates and what remains confounded.

## 6. Claim paragraph

Write 150–250 words with four explicit parts:

1. **Configuration:** identify the final method, baseline, split, and selection rule.
2. **Observation:** report the locked numerical evidence.
3. **Interpretation:** answer the stated research question at the scale supported by the comparison.
4. **Limitation:** identify an important untested dataset, subgroup, intervention, transformation, or source of uncertainty.

Avoid “proves,” “understands,” “explains,” or “robust” unless the design directly tests that claim.

## 7. Next experiment

Choose one next experiment from the error evidence. State:

- hypothesis;
- one factor changed;
- controls held fixed;
- selection metric and split;
- locked final metric;
- expected result if the hypothesis is correct; and
- result that would make you reject or revise the hypothesis.

## Reproducibility checklist

- [ ] The repository state is identified.
- [ ] No private data, credentials, or trained artifacts are committed.
- [ ] Preprocessing is fitted on training data only.
- [ ] Validation controls selection.
- [ ] The test set is used only after the procedure is fixed.
- [ ] The baseline and selected method use the same evaluation unit.
- [ ] Aggregate and class/subgroup evidence are both reported.
- [ ] Error categories include counts, not only examples.
- [ ] The robustness or ablation factor is declared.
- [ ] Commands, versions, seeds, and device are recorded.
- [ ] Claims are bounded by the measured evidence.

## Rubric: 20 points

| Criterion | Points | Full-credit evidence |
| --- | ---: | --- |
| experiment identity and split integrity | 4 | reproducible configuration, training-only preprocessing, locked test |
| selection history | 3 | declared validation criterion and restored selected state |
| result reporting | 4 | comparable baseline, primary metric, class/subgroup evidence, uncertainty when justified |
| structured error analysis | 4 | contextual records, defensible categories, aggregate counts, hypotheses separated from observations |
| robustness or ablation | 2 | one controlled factor with relevant metrics |
| claim and next experiment | 3 | evidence-matched interpretation, limitation, and falsifiable next step |

## Week 10 lab model

The course lab demonstrates the expected structure on a deterministic MNIST classroom subset:

- a declared TorchVision source, derived training and validation subsets, and locked official test partition;
- automatic CUDA selection with the device and hardware recorded;
- a majority baseline, locked test accuracy, and per-digit recall;
- confusion directions and stored truth, prediction, and confidence for each error;
- a two-pixel translation check with consistency and shifted accuracy; and
- explicit limits on claims from a classroom subset, activation summaries, and one transformation.

Use its reporting structure, not its numerical results, for your own project.
