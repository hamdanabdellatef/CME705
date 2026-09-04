# Week 1 lecture: Machine-learning foundations

This file is the accessible text source for the classroom deck. Slide breaks are marked with horizontal rules.

---

## 1. CME705 Machine Learning

### Neural Networks, Deep Learning, and Research Practice

Week 1: From an application need to a learning task

Karabuk University · Department of Computer Engineering

---

## 2. Today's evidence

By the end of class, you will be able to:

- identify observations, features, targets, and model outputs;
- distinguish four common learning tasks;
- express a linear score with NumPy; and
- formulate a research interest that can be investigated.

Evidence: a problem canvas, a diagnostic result, and a research-direction draft.

---

## 3. Opening case: twenty inspections

A factory operates 200 machines but can inspect only 20 each day.

Available before the decision:

- temperature;
- vibration;
- machine age; and
- previous maintenance records.

Question for discussion: What should the system output, and what action follows?

---

## 4. Application goal and model output

**Application goal**

Reduce unexpected failures while using limited inspection capacity.

**Possible model output**

A probability that each machine will fail within seven days.

The output supports the decision. It does not specify the complete inspection policy.

---

## 5. Programmed rule and learned model

**Programmed rule**

An engineer specifies the relationship between inputs and action.

`temperature > 95 -> inspect`

**Learned model**

A learning procedure estimates parameters from examples.

`examples + objective -> fitted model`

Many real systems combine learned outputs with explicit constraints.

---

## 6. AI, machine learning, and deep learning

Artificial intelligence is the broad field.

Machine learning estimates useful patterns from data or experience.

Deep learning uses multilayer neural networks to learn representations and task outputs.

The narrowest category is not automatically the best method.

---

## 7. The unit of observation

One row must have a precise meaning.

Possible units in the maintenance case:

- one machine at one scheduled reading;
- one five-minute sensor window; or
- one maintenance episode.

Changing the unit changes the features, target, split, and conclusion.

---

## 8. Features, target, and output

For each observation:

- features \(x\) describe information available at decision time;
- target \(y\) records the outcome used for learning;
- model \(f(x)\) produces a score or prediction; and
- metric compares predictions with observed outcomes.

Prompt: Which target could be measured reliably?

---

## 9. Four common learning tasks

| Task | Output | Maintenance example |
| --- | --- | --- |
| Classification | Discrete class | Failure within seven days: yes or no |
| Regression | Number | Remaining useful life in hours |
| Clustering | Groups without target labels | Recurring operating profiles |
| Representation learning | Learned encoding | Compact vector for each sensor window |

---

## 10. Task-identification activity

Classify each problem:

1. Estimate tomorrow's electricity demand.
2. Assign an image to one of four quality levels.
3. Discover recurring patterns in unlabelled logs.
4. Learn an encoding for retrieving similar papers.

Then name the unit of observation for one problem.

---

## 11. Problem-formulation canvas

1. Application goal
2. Unit of observation
3. Available input
4. Target or learned structure
5. Intended user and action
6. Simple baseline
7. Evidence and metric
8. Main risk

If a field remains ambiguous, the experiment remains ambiguous.

---

## 12. A first model expression

Feature matrix:

\[
X \in \mathbb{R}^{n \times d}
\]

Linear score:

\[
s = Xw + b
\]

Binary prediction:

\[
\hat{y}_i = 1 \text{ when } s_i \ge 0
\]

Week 1 inspects chosen weights. Later weeks learn them from data.

---

## 13. NumPy makes structure visible

```python
scores = features @ weights + bias
predictions = (scores >= 0).astype(int)
accuracy = np.mean(predictions == target)
```

Check every shape before interpreting the result.

`(n, d) @ (d,) -> (n,)`

---

## 14. A baseline gives the result context

If class 0 contains most examples, always predicting class 0 can achieve high accuracy.

The majority-class baseline asks:

> Does the candidate method improve on the simplest frequency-based rule?

A stronger model still needs an evaluation suited to its intended use.

---

## 15. Generalization is the central challenge

Fitting observed examples is not the final goal.

We want useful behaviour on new observations from a defined setting.

Next week:

- training, validation, and test data;
- data leakage;
- group-aware splitting; and
- metrics beyond accuracy.

---

## 16. The machine-learning workflow

Define the task

Data and measurement

Baseline and split

Model and training

Controlled experiment

Error and limitation analysis

Revision of the question

---

## 17. Diagnostic lab

Run:

```bash
python labs/week01_python_numpy_diagnostic.py
```

Record:

- array shapes;
- class counts;
- baseline accuracy;
- linear-rule accuracy; and
- one limitation of the demonstration.

Then remove one feature by setting its weight to zero.

---

## 18. What the demonstration can establish

Supported:

- the NumPy computation runs;
- the array shapes agree;
- the rule captures part of the generated pattern; and
- the rule can be compared with a baseline.

Still unknown:

- behaviour on real machines;
- label validity;
- operational cost; and
- performance after conditions change.

---

## 19. The semester research route

Interest

Problem and data

Literature and baseline

Controlled experiments

Reproducibility review

Research-direction note

Each milestone should narrow uncertainty.

---

## 20. From interest to research direction

```text
industrial maintenance
  -> early bearing-fault detection
    -> classify sensor windows
      -> spectral-feature baseline
        -> performance changes with load
          -> test load-robust representations
```

A useful direction identifies the next answerable question.

---

## 21. Week 1 submission

Complete the research-interest memo:

- two possible application or methodological areas;
- a plausible ML task and data source for each;
- one difficulty or risk for each; and
- one area selected for further investigation.

Exit ticket: application goal, learning task, required evidence.
