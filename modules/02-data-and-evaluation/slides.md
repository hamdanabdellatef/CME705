# Week 2 slide source: Generalization and trustworthy evaluation

This file provides an accessible text version of the PowerPoint deck. Speaker prompts and timing are in the presentation and instructor guide.

## Slide 1 — CME705 Machine Learning

**Week 2: Generalization and trustworthy evaluation**

Guiding question: What evidence shows that a model will work on relevant unseen data?

## Slide 2 — Today’s evidence

Each student will produce:

1. a justified split strategy;
2. a repaired leakage-prone workflow;
3. a calculated metric interpretation; and
4. a first dataset-feasibility decision.

## Slide 3 — “95% accuracy”

A score alone does not identify:

- which data produced it;
- whether examples were independent;
- which baseline was exceeded;
- which errors matter; or
- which future cases the claim concerns.

Prompt: What would you ask before trusting this result?

## Slide 4 — Fitting and generalization

Training performance measures fit to examples used during learning.

Generalization performance estimates behavior on relevant unseen cases.

A generalization gap is meaningful only when evaluation data are independent and match the intended claim.

## Slide 5 — Complete the evaluation claim

“We want to estimate performance when the fitted system is applied to ______.”

Examples:

- new independent observations;
- new people, machines, sites, or documents;
- future time periods; or
- a new institution or operating condition.

The blank determines the split.

## Slide 6 — Three partitions, three roles

- **Training:** estimate preprocessing and model parameters.
- **Validation:** choose features, hyperparameters, models, stopping rules, and thresholds.
- **Test:** estimate final performance after choices are fixed.

## Slide 7 — Protect the test set

The test set is a measurement instrument.

Do not use it to select features, choose a model, tune a threshold, stop training, or select the most favorable random seed.

## Slide 8 — A row is not always independent

Rows can share a person, machine, video, conversation, site, family, document, or event.

Identify the unit that must remain unseen, then keep all dependent rows in one partition.

## Slide 9 — Match the split to the claim

| Split | Suitable question | Main protection |
| --- | --- | --- |
| Random | Independent observations from one stable process | Representative holdout |
| Stratified | Independent observations with class imbalance | Class proportions |
| Grouped | Repeated rows share a source or entity | Entity leakage |
| Chronological | Future cases must be predicted from the past | Look-ahead leakage |

## Slide 10 — Split-design activity

Choose and justify a split for:

1. next-month electricity demand;
2. rare defects in independent components;
3. several medical images per patient; and
4. independent documents from one archive.

## Slide 11 — Leakage changes the question

Intended flow:

past and available information → model → future outcome

Leaky flow:

future outcome, test statistics, duplicate source, or future time → training process

Leakage lets the system use information unavailable at the claimed prediction time.

## Slide 12 — Four leakage patterns

- target leakage;
- preprocessing leakage;
- group or duplicate leakage; and
- temporal leakage.

For each pattern, identify the information crossing the boundary.

## Slide 13 — Fit preprocessing safely

1. Create the split.
2. Fit mean, scale, imputation, vocabulary, or feature selection on training data.
3. Transform every partition with the fitted training transformation.
4. Fit the model on training data.
5. Use validation data for choices.

## Slide 14 — A baseline gives a score meaning

Compare the model with a credible reference: majority class, mean or median target, persistence forecast, simple linear model, or current operational procedure.

Complexity requires evidence.

## Slide 15 — Confusion matrix

|  | Predicted positive | Predicted negative |
| --- | ---: | ---: |
| Actually positive | TP | FN |
| Actually negative | FP | TN |

The four counts preserve information hidden by a single average.

## Slide 16 — Same predictions, different questions

- Accuracy: overall fraction correct.
- Precision: when the model predicts positive, how often is it right?
- Recall: how many actual positives are detected?
- Specificity: how many actual negatives are rejected?
- Balanced accuracy: mean of recall and specificity.

## Slide 17 — Thresholds trade errors

A lower threshold usually yields more positive predictions, higher recall, and more false positives.

A higher threshold usually yields fewer positive predictions, higher specificity, and more false negatives.

Choose on validation data using the intended cost or capacity.

## Slide 18 — Lab: repeated measurements

The synthetic dataset has four rows per participant.

Compare a random row-wise split that allows participant overlap with a grouped split that keeps participants disjoint.

Predict the performance direction before running the code.

## Slide 19 — One split is one estimate

Performance varies with sampled cases.

Record counts and class composition, uncertainty or repeated estimates, subgroup behavior, split seed and procedure, and the complete selection process.

Do not search for and report only a favorable split.

## Slide 20 — Screen the dataset before modeling

Check access and permitted use, observation and target definitions, dependence and grouping, population coverage, class and subgroup counts, baseline and metric, reproducible preparation, and ethical or privacy constraints.

## Slide 21 — Exit ticket

- What future cases should your project handle?
- Which split simulates that use?
- What leakage risk is most plausible?
- What evidence could make you reject the dataset?

Next: Week 3 examines data quality and model comparison.
