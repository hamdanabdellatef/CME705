# Week 1 notes: From an application need to a learning task

## 1. The purpose of this course

Machine learning gives us tools for building systems whose behaviour depends on patterns estimated from data. The useful question is rarely “Can we use AI?” A better starting point is:

> What decision or understanding do we need, what evidence is available, and how will we judge whether a learned pattern helps?

This course develops that judgment alongside implementation. We will expose important mechanisms with NumPy, use PyTorch for deep-learning models, and connect every technical choice to a semester research project.

## 2. Programmed rules and learned models

A conventional program receives inputs and follows rules written by a programmer. For example:

```text
if temperature > 95:
    request_inspection()
```

A learned model estimates its parameters from examples:

```text
examples + learning procedure -> fitted model
new observation + fitted model -> output
```

Both approaches contain human choices. People choose the data, representation, objective, model family, evaluation, and operating conditions. A model does not remove the need to understand the problem.

Use a learned model when useful patterns exist in data but writing complete rules would be difficult. Prefer a direct rule when requirements are stable, fully specified, and easy to test. Many practical systems combine both.

## 3. AI, machine learning, and deep learning

**Artificial intelligence** is the broad field of building systems that perform tasks associated with perception, reasoning, planning, language, or decision making.

**Machine learning** is an approach within AI that estimates useful patterns from data or experience.

**Deep learning** is a family of machine-learning methods that uses neural networks with multiple layers to learn representations and task outputs.

The terms describe scope, not a quality ranking. A simple model can be the best choice when it answers the question reliably and can be maintained.

## 4. The parts of a learning problem

Consider a maintenance team deciding which machines should receive an inspection.

| Term | Question | Example |
| --- | --- | --- |
| Unit of observation | What does one row represent? | One machine at one scheduled reading |
| Feature | What information enters the model? | Temperature, vibration, machine age |
| Target | What outcome should the model learn? | Inspection required: yes or no |
| Model | How are inputs mapped to an output? | A weighted score followed by a threshold |
| Parameter | What values does learning estimate? | Feature weights and bias |
| Hyperparameter | What controls learning or model capacity? | Learning rate or number of layers |
| Training | When are parameters estimated? | Using historical labelled examples |
| Inference | When does the fitted model produce an output? | Scoring a new machine reading |
| Metric | How do we compare outputs with observations? | Accuracy, recall, cost, or another task metric |
| Baseline | What simple reference must a model improve on? | Always predict the most common class |

The target must be defined carefully. “Needs inspection” could mean a technician’s judgment, a failure within seven days, or exceeding a safety standard. These labels have different meanings and limitations.

## 5. Common task types

### Classification

The output belongs to a discrete set of classes.

- Does this image contain a defect?
- Which activity produced this sensor window?
- Is a transaction likely to be fraudulent?

### Regression

The output is numerical.

- How much energy will a building use tomorrow?
- What remaining useful life does a component have?
- How long will a process take?

### Clustering

The method groups observations without a supplied target label.

- Which operating profiles appear in machine logs?
- Which documents have similar themes?

Clusters do not arrive with an automatic meaning. A researcher must study whether a grouping is stable, useful, and connected to the application.

### Representation learning

The method learns a useful encoding of the input. The representation may support visualization, retrieval, generation, anomaly detection, or a later supervised task.

An autoencoder, for example, learns a compact latent representation while reconstructing its input. We study this in Week 12.

## 6. A problem-formulation canvas

Write each proposed problem using the following fields.

1. **Application goal:** What decision or understanding should improve?
2. **Unit of observation:** What exactly does one row or example represent?
3. **Available input:** What information would be present when the output is needed?
4. **Target or learned structure:** What should the method predict or discover?
5. **Intended use:** Who will use the output, and what will they do with it?
6. **Baseline:** What simple rule or model provides a credible reference?
7. **Evidence:** What metric and comparison would support a useful conclusion?
8. **Main risk:** What could make the result misleading or harmful?

A weak statement says, “Use deep learning for medical images.”

A more useful statement says:

> For each chest radiograph collected at Hospital A, estimate whether a radiologist will identify a finding that requires follow-up. Compare a simple image baseline with a convolutional model using a patient-separated evaluation. Study false negatives and performance across acquisition devices.

The second statement still needs data access, ethics review, label validation, and a precise intended use. It gives us enough structure to ask those questions.

## 7. A first model expression

The diagnostic lab represents $n$ machines with $d$ features as a matrix:

$$
X \in \mathbb{R}^{n \times d}
$$

Each row is one observation. Each column is one feature. A linear scoring rule assigns each row a score:

$$
s = Xw + b
$$

where $w \in \mathbb{R}^{d}$ contains feature weights and $b$ is a bias. For binary classification, a threshold converts each score into a predicted class:

$$
\hat{y}_i =
\begin{cases}
1 & \text{if } s_i \ge 0 \\
0 & \text{otherwise}
\end{cases}
$$

Week 1 uses a hand-designed weight vector so we can inspect the computation. Weeks 5 and 6 show how a learning procedure estimates parameters from data.

## 8. Baselines and generalization

Suppose 70 percent of examples belong to class 0. A model with 65 percent accuracy performs worse than the rule “always predict class 0.” The majority-class rule provides a basic baseline.

A useful evaluation also asks whether the model works on observations that did not determine its parameters. This is the problem of **generalization**. We study validation, testing, leakage, group-aware splits, and appropriate metrics in Weeks 2 and 3.

For Week 1, remember:

- training performance describes fit to observed examples;
- test performance estimates behaviour on held-out examples from a defined setting;
- neither proves that the model will work after the population, measurement process, or intended use changes.

## 9. The machine-learning workflow

An ML project is an evidence cycle:

1. define the application goal and learning task;
2. inspect data provenance, measurement, labels, and constraints;
3. establish a split and a simple baseline;
4. choose a model and training procedure;
5. run controlled experiments;
6. analyze errors, uncertainty, and limitations;
7. communicate what the evidence supports;
8. revise the question or method.

The workflow often moves backward. A data limitation may force a narrower question. An error analysis may reveal that the target is ambiguous. That revision is part of research.

## 10. Why Python, NumPy, and PyTorch

Python provides the course environment and a large scientific-computing ecosystem.

NumPy makes arrays, shapes, vectorized operations, and numerical computation visible. We will use it to implement optimization and neural-network mechanisms directly.

PyTorch provides tensors, automatic differentiation, neural-network components, and accelerator support. We introduce it after the core computations are clear.

The technology supports the reasoning. A correct library call cannot repair a poorly defined target, a leaked test set, or an unsupported conclusion.

## 11. From course topic to research direction

A research direction develops through narrowing:

```text
application area
    -> observable problem
        -> feasible learning task
            -> credible baseline
                -> unresolved question
                    -> next experiment
```

For example:

```text
industrial maintenance
    -> early detection of bearing faults
        -> classify fixed sensor windows
            -> spectral features with logistic regression
                -> performance drops under changing load
                    -> test a representation robust to load variation
```

The Week 1 research-interest memo asks for two possible directions. It does not require a final thesis topic. Its purpose is to make interests concrete enough to investigate.

## 12. Review questions

1. What is the difference between an application goal and an ML task?
2. In the maintenance example, what does one row represent?
3. Give one target definition that could replace “needs inspection.”
4. Why can accuracy be misleading without a baseline?
5. Which parts of a learned system still depend on human judgment?
6. Convert one research interest into the eight fields of the problem-formulation canvas.

## Key terms

application goal, unit of observation, feature, target, label, model, parameter, hyperparameter, training, inference, prediction, baseline, metric, classification, regression, clustering, representation learning, generalization
