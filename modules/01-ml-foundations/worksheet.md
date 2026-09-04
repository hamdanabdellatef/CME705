# Week 1 worksheet

Name:

Research interest:

## 1. Opening case

A factory can inspect 20 of its 200 machines each day. Each machine reports temperature, vibration, and operating hours. The maintenance team wants to reduce unexpected failures.

1. What is the application goal?
2. What decision must the team make?
3. What could one observation represent?
4. List two possible targets. Explain how their meanings differ.
5. What information must be available before an inspection decision?
6. Write one direct programmed rule.
7. Give one reason a learned model might help.
8. Give one reason the direct rule might be preferable.

## 2. Identify the task

For each problem, identify the likely task type and justify the answer in one sentence.

| Problem | Classification, regression, clustering, or representation learning? | Reason |
| --- | --- | --- |
| Estimate tomorrow's electricity demand in MWh | | |
| Assign a retinal image to one of four quality levels | | |
| Discover recurring operating profiles in unlabelled sensor logs | | |
| Learn a compact vector for retrieving similar research papers | | |
| Predict whether a component will fail within seven days | | |

Choose one row and describe how the application goal differs from the model output.

## 3. Problem-formulation canvas

Complete the canvas for the opening case or your own application area.

| Field | Your statement |
| --- | --- |
| Application goal | |
| Unit of observation | |
| Available inputs | |
| Target or learned structure | |
| Intended user and action | |
| Simple baseline | |
| Evidence and metric | |
| Main risk or limitation | |

Exchange canvases with a partner. The partner should circle any term that could support two different interpretations.

## 4. NumPy diagnostic record

Before running the lab:

1. Predict the shape of a dataset containing 160 observations and three features.
2. Predict the shape produced by `X @ weights` when `X.shape == (160, 3)` and `weights.shape == (3,)`.
3. Explain what a positive score means in the lab.
4. Predict whether the hand-designed rule will beat the majority baseline.

Run:

```bash
python labs/week01_python_numpy_diagnostic.py
```

Record:

| Result | Value |
| --- | --- |
| Feature-matrix shape | |
| Target-vector shape | |
| Number requiring inspection | |
| Majority baseline accuracy | |
| Linear-rule accuracy | |

Interpretation:

1. What evidence supports the claim that the rule captures part of the generated pattern?
2. What does this demonstration fail to establish?
3. Change one weight to zero. Which feature did you remove, and how did the accuracy change?
4. Did the result match your prediction? Explain.

## 5. Research-direction ladder

Complete each line with increasing precision.

**Application area:**

**Observable problem:**

**Possible ML task:**

**Available or plausible data:**

**Simple baseline:**

**One uncertainty worth testing:**

**Next small experiment:**

## 6. Exit ticket

Submit three sentences.

1. An application goal I care about is...
2. A learning task that could support it is...
3. Before trusting a result, I would need evidence that...
