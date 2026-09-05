# Week 2 notes: Generalization and trustworthy evaluation

## 1. From fitting to evidence

A learning procedure uses observed examples to estimate a model. Evaluation asks a different question:

> What does performance on the available data tell us about performance on relevant future cases?

A high training score shows that a model can fit the examples used during optimization. It does not establish that the model will generalize. Generalization is performance on data generated under the conditions represented by the intended use but unavailable during fitting and model selection.

If the deployment population, measurement process, or operating conditions differ from the evaluation data, even a technically correct test score may answer the wrong question.

## 2. Define the evaluation claim first

Complete this sentence:

> We want to estimate performance when the fitted system is applied to ______.

Possible completions include new independent images, new patients, future time periods, new machines at known sites, new sites, or later documents. These claims require different splits. “Unseen rows” is too weak because different rows can describe the same person, machine, conversation, location, or event.

## 3. Empirical performance and the generalization gap

For loss (L), dataset (D), and fitted model (f), empirical loss is

[
\hat{R}_D(f)=\frac{1}{|D|}\sum_{(x_i,y_i)\in D}L(f(x_i),y_i).
]

Training reduces loss on the training sample. The quantity of interest is expected loss under the target process:

[
R(f)=\mathbb{E}_{(x,y)\sim P_{target}}[L(f(x),y)].
]

The observed difference between training and relevant unseen-data performance is a generalization gap. A small gap is reassuring only when the evaluation set is independent, representative, sufficiently large, and untouched by model selection.

## 4. Three partitions, three roles

### Training set

Training data estimate model parameters and learned preprocessing quantities: means, scales, imputation values, vocabulary, feature selection, and network weights.

### Validation set

Validation data support choices such as model family, hyperparameters, features, stopping point, and decision threshold. Repeated validation use adapts the development process to that set.

### Test set

Test data estimate final performance after the procedure is fixed. The test set should not guide feature design, architecture choice, preprocessing, thresholds, or hyperparameters.

The test set is a measurement instrument. Every use for model selection weakens the independence of that measurement.

## 5. The sampling unit determines the split

A row is a storage unit; it is not necessarily an independent observation. Several rows can share information because they come from the same person, machine, signal, video, conversation, site, family, document, or event.

First identify the unit that must remain unseen. Then assign all dependent rows from that unit to one partition.

## 6. Choosing a split strategy

| Strategy | Suitable question | Main protection |
| --- | --- | --- |
| Random | Independent observations from one stable process | Simple representative holdout |
| Stratified | Independent observations with important class imbalance | Similar class proportions |
| Grouped | Repeated or related rows share a source or entity | Entity leakage |
| Chronological | Future observations must be predicted from the past | Look-ahead leakage |
| Nested cross-validation | Tune and estimate performance with limited data | Selection bias |

A split can combine requirements. A grouped split may seek approximate class balance, but group integrity takes priority.

## 7. Leakage

Leakage occurs when information unavailable at the intended prediction time influences training, model selection, or evaluation.

### Target leakage

A feature contains the outcome or a consequence recorded after it.

### Train–test contamination

A test example, duplicate, or derived version appears in training. Augmentations of one original must remain together.

### Preprocessing leakage

Statistics are calculated on all data before splitting. This includes scaling, imputation, feature selection, vocabulary construction, and dimensionality reduction.

### Group leakage

Rows from one source appear in multiple partitions. The model can recognize the source rather than learn the intended relationship.

### Temporal leakage

Future values influence a prediction about the past. Randomly splitting forecasting data often creates this error.

## 8. Fit preprocessing on training data only

For feature (j), standardization is

[
z_{ij}=\frac{x_{ij}-\mu_j}{\sigma_j}.
]

The safe sequence is:

1. create the split;
2. calculate (mu_j) and (sigma_j) from training rows;
3. transform every partition with those training values;
4. fit the model on transformed training data; and
5. make development choices with validation data.

Test data may pass through an already fitted transformation. They must not contribute to fitting it.

## 9. A baseline gives a score meaning

Useful baselines include the majority class, a prevalence-matched predictor, mean or median regression target, persistence forecast, a simple linear model, an established non-neural method, or the current operational procedure.

A complex model that does not improve on a credible baseline has not justified its complexity.

## 10. Confusion matrix and metrics

|  | Predicted positive | Predicted negative |
| --- | ---: | ---: |
| Actually positive | True positive (TP) | False negative (FN) |
| Actually negative | False positive (FP) | True negative (TN) |

[
\text{Accuracy}=\frac{TP+TN}{TP+TN+FP+FN}
]

[
\text{Precision}=\frac{TP}{TP+FP},\qquad
\text{Recall}=\frac{TP}{TP+FN}
]

[
\text{Specificity}=\frac{TN}{TN+FP}
]

[
\text{Balanced accuracy}=\frac{1}{2}(\text{Recall}+\text{Specificity})
]

Accuracy weights every example equally. Balanced accuracy gives the two classes equal influence. Precision asks how often a positive prediction is correct. Recall asks how many actual positives are detected. The metric follows the cost and intended use.

## 11. Thresholds express an operating choice

A fitted probabilistic model can stay fixed while its decision threshold changes. Lowering the threshold usually increases recall and false positives. Raising it usually increases specificity while missing more positives.

Select a threshold with validation data and a stated cost or capacity. Report test performance once with that threshold fixed.

## 12. One split is one estimate

Performance varies with sampled cases. A small test set can be noisy. Useful responses include confidence intervals, repeated cross-validation when appropriate, class-specific counts, subgroup analysis, plausible split-seed sensitivity, and external evaluation.

Trying many splits and reporting the most favorable one is selection.

## 13. Reproducibility record

Record:

- dataset source, version, and retrieval date;
- inclusion and exclusion criteria;
- observation unit and grouping variables;
- split strategy, proportions, seed, and index file;
- preprocessing learned from training;
- baseline, model, and threshold;
- metric definitions;
- software versions and execution command; and
- limitations of the evaluation claim.

A fixed seed improves repeatability. It does not prove robustness.

## 14. Dataset feasibility for the research project

Before choosing a model, ask:

1. Can the dataset be accessed and used lawfully?
2. What does one observation represent?
3. What target is available, and when is it measured?
4. Which observations share a source, site, or event?
5. Does the dataset represent the population in the claim?
6. Are important classes and subgroups large enough?
7. What baseline can be implemented immediately?
8. What split can be defined before modeling?
9. What privacy, consent, or bias issue requires review?
10. Can another student reproduce preparation?

## Review questions

1. Why is a final test set different from validation data?
2. Give an example where a random row split is invalid.
3. Why must imputation and standardization follow splitting?
4. When could chronological and grouped splits answer different questions?
5. Which metric would you prioritize when false negatives are costly?
6. What does a seed establish, and what does it not establish?
7. Define the future cases your research model should handle.
