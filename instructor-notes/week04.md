# Week 4 instructor guide: Neural-network computation

## Purpose

Week 4 gives students a precise computational model of a feed-forward neural network before introducing training. Students should leave able to trace values and dimensions, choose an output head, and explain why nonlinear activations matter.

The historical course reached training rules during this week. In the reorganized sequence, Week 4 stops at the forward pass. Gradient descent begins in Week 5 so the distinction between computation and learning remains clear.

## Before class

- Run python labs/week04_forward_pass_numpy.py.
- Distribute [the Week 4 worksheet](../modules/03-neural-network-basics/worksheet.md).
- Ask students to bring one primary paper related to their proposed topic.
- Prepare a board or shared document for shape tracing.
- Keep the lab output hidden until students record predictions.

## Learning evidence

Inspect:

1. the manual neuron calculation;
2. the batch and layer shapes;
3. the explanation of affine-layer collapse;
4. the output-head table;
5. the parameter count and lab record; and
6. one architecture row supported by a location in a paper.

## Three-hour sequence

| Time | Activity | Instructor action | Student evidence |
| ---: | --- | --- | --- |
| 0–10 min | Opening calculation | Give fixed $x$, $w$, and $b$ | Pre-activation and two activation outputs |
| 10–30 min | Neuron mechanism | Separate weighted sum, bias, and activation | Explanation of each term |
| 30–50 min | Batch shapes | Move from $x^Tw$ to $XW$ | Shape trace and broadcasting explanation |
| 50–70 min | Activation functions | Compare identity, sigmoid, tanh, and ReLU | Completed activation table |
| 70–85 min | Why nonlinearity matters | Derive affine-layer collapse | Equivalent $W_*$ and $b_*$ |
| 85–95 min | Break | — | — |
| 95–115 min | Two-layer forward pass | Trace each operation and shape | Complete network trace |
| 115–135 min | Output heads | Contrast regression, binary, multiclass, and multilabel tasks | Justified output decision |
| 135–150 min | Parameters and capacity | Derive the dense-layer count | Counts for two architectures |
| 150–170 min | NumPy lab | Require predictions before execution | Results and claim boundary |
| 170–177 min | Literature extraction | Model one evidence row from a primary paper | One sourced architecture row |
| 177–180 min | Exit ticket | Collect four concise statements | Completed exit ticket |

## Facilitation notes

### Opening calculation

Expected contributions are $0.8$, $0.8$, and $0.6$. Their sum is $2.2$; adding $-0.3$ gives $z=1.9$. Sigmoid gives approximately $0.870$, and ReLU gives $1.9$.

Ask students to label the pre-activation before applying an activation. This vocabulary prevents later confusion between logits and probabilities.

### Weights and bias

Change one term at a time. A weight changes the contribution associated with an input; a bias adds an offset independent of the current input values. Show that $x=0$ still produces $z=b$.

### Batch shapes

Use the course orientation: observations in rows. For $X:(4,3)$ and $W_1:(3,4)$, the shared dimension 3 is consumed and $Z_1:(4,4)$. The bias length matches the four output units and is broadcast across the four observations.

Do not accept a memorized shape without a reason. Ask which dimensions are contracted and which remain.

### Activation functions

At $z=-2,0,2$:

| $z$ | Identity | Sigmoid | Tanh | ReLU |
| ---: | ---: | ---: | ---: | ---: |
| -2 | -2.000 | 0.119 | -0.964 | 0.000 |
| 0 | 0.000 | 0.500 | 0.000 | 0.000 |
| 2 | 2.000 | 0.881 | 0.964 | 2.000 |

Focus on response shape and role. Avoid presenting a hidden activation as universally superior.

### Affine-layer collapse

Derive:

$$
(XW_1+b_1)W_2+b_2=X(W_1W_2)+(b_1W_2+b_2).
$$

Expected answers are $W_*=W_1W_2$ and $b_*=b_1W_2+b_2$. Explain that the lab's difference near machine precision is numerical evidence for the identity, not experimental evidence that nonlinear networks are always better.

### Output heads

Expected worksheet matches:

- temperature: one identity output;
- one fault versus no fault: one sigmoid output;
- one of six fault types: six logits and softmax;
- any subset of eight tags: eight independent sigmoids.

Softmax represents mutually exclusive outcomes whose probabilities sum to one. Independent sigmoids permit several labels at once.

### Parameter count

For $3\rightarrow4\rightarrow2$, the count is 26. For $10\rightarrow8\rightarrow3$, it is

$$
(10\times8+8)+(8\times3+3)=115.
$$

Ask what evidence would justify increasing width. Parameter count alone does not answer whether capacity is useful.

### Week 4 lab

Students should observe:

- one-neuron pre-activation 1.900 and sigmoid output about 0.870;
- shapes $(4,3)\rightarrow(4,4)\rightarrow(4,2)$;
- probability rows summing to 1;
- 26 trainable parameters; and
- affine-collapse difference near $2.22\times10^{-16}$.

The predicted classes depend on the fixed teaching parameters. Shapes, softmax normalization, and the algebraic collapse follow from the operations.

### Literature extraction

Require a page, section, table, or figure location for every reported method fact. When a paper omits a detail, students should write “not reported” rather than infer it.

Ask whether another paper used the same data, split, target, metric definition, and test policy before accepting a direct score comparison.

## Common misconceptions

| Misconception | Response |
| --- | --- |
| A neuron is only a weighted sum | Bias and activation are separate parts of the computation |
| Bias is another input feature | It is a learned offset associated with an output unit |
| More layers always make a more complex function | Affine layers collapse without intervening nonlinearities |
| Softmax is required for every classification problem | Binary and multilabel tasks commonly use sigmoid outputs |
| A probability is the same as a class prediction | A decision rule turns probabilities or logits into predictions |
| A larger parameter count proves a better model | Capacity requires controlled evaluation and sufficient data |
| Architecture names make paper results comparable | Data, split, target, metric, and protocol must also align |

## Formative feedback language

- “Label the pre-activation before applying the function.”
- “Write the shape beside each symbol.”
- “Which dimensions must match?”
- “What does the bias change?”
- “Show the substitution that collapses the layers.”
- “Does this task require exclusive or independent outputs?”
- “Which paper location supports that architecture detail?”

## Adaptations

### Ninety-minute class

Use slides 1–7, 10–18, and 21. Complete the manual calculation, collapse derivation, output-head match, and lab. Assign the activation table and paper extraction afterward.

### Online delivery

Use a shared shape-tracing table. Give each breakout group one prediction task and ask it to defend an output head. Run the NumPy script centrally if environments differ.

### Limited linear-algebra preparation

Begin with one neuron and expand the calculation column by column. Treat shape annotations as required working rather than optional notation.

## After class

- identify students who confuse rows, features, and units;
- flag project output heads that do not match the target structure;
- return literature-search feedback before the Week 5 baseline plan; and
- carry the distinction between forward computation and parameter learning into gradient descent.
