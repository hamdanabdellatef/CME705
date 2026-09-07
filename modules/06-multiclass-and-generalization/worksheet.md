# Week 7 worksheet: Multiclass learning and consolidation

Name: ________________________________  Date: __________________

Use one fixed class order throughout. State the confusion-matrix orientation before interpreting any cell.

## 1. Match the task to the output

Complete the table.

| Task | Target structure | Output design | Typical objective |
| --- | --- | --- | --- |
| Binary | one of two classes | | |
| Mutually exclusive multiclass | one of $K$ classes | | |
| Multilabel | any subset of $K$ labels | | |

Why should one softmax distribution not be used when several labels may be correct at the same time?

__________________________________________________________________________

## 2. Logits and shapes

For $m$ observations, $d$ input features, and $K$ classes:

$$
Z=XW+b.
$$

Complete:

| Quantity | Meaning | Shape |
| --- | --- | --- |
| $X$ | | |
| $W$ | | |
| $b$ | | |
| $Z$ | | |

For $m=32$, $d=10$, and $K=4$:

- $W$ shape: __________________
- $b$ shape: __________________
- $Z$ shape: __________________

Why is a logit not a probability? ________________________________________

## 3. Compute softmax

For one observation:

$$
z=[2,1,0].
$$

Subtract the maximum:

$$
\tilde z=[\underline{\hspace{1cm}},\underline{\hspace{1cm}},\underline{\hspace{1cm}}].
$$

Approximate the exponentials and normalize:

| Class | Shifted logit | Exponential | Probability |
| ---: | ---: | ---: | ---: |
| 0 | | | |
| 1 | | | |
| 2 | | | |

Check:

$$
\sum_k p_k=\underline{\hspace{3cm}}.
$$

Predicted class: __________________

## 4. Explain shift invariance

Complete the cancellation:

$$
\frac{e^{z_k+c}}{\sum_j e^{z_j+c}}
=
\frac{\underline{\hspace{3cm}}e^{z_k}}
{\underline{\hspace{3cm}}\sum_j e^{z_j}}
=
\underline{\hspace{5cm}}.
$$

Would adding $1000$ to every logit change the probabilities? _____________

Why is this property useful in software? _________________________________

## 5. Stable row-wise implementation

Put these operations in order.

| Order | Operation |
| ---: | --- |
| | divide each exponential by its row sum |
| | subtract the maximum within each row |
| | exponentiate shifted logits |
| | receive logits with shape $(m,K)$ |
| | sum exponentials across the class axis |

Which axis represents classes in this course notation? ___________________

What error results from normalizing all observations together? ___________

## 6. Targets

Use the class order:

| Index | Meaning |
| ---: | --- |
| 0 | normal |
| 1 | warning |
| 2 | critical |

Write one-hot targets:

- normal: __________________
- warning: _________________
- critical: ________________

Convert $[2,0,1,2]$ to a one-hot matrix:

$$
Y=
\underline{\hspace{11cm}}
$$

Why must saved predictions include the class order? ______________________

## 7. Categorical cross-entropy

For:

$$
p=[0.70,0.20,0.10],
$$

and target class index $1$:

$$
\ell=-\log(\underline{\hspace{2cm}})
\approx\underline{\hspace{3cm}}.
$$

Now suppose the target is class index $0$. The loss is:

$$
\ell=-\log(\underline{\hspace{2cm}})
\approx\underline{\hspace{3cm}}.
$$

Explain why accuracy and cross-entropy respond differently to these cases.

__________________________________________________________________________

## 8. Output gradient

For one example:

$$
p=[0.70,0.20,0.10],
$$

and:

$$
y=[0,1,0].
$$

Complete:

$$
G_z=p-y=
[\underline{\hspace{1cm}},\underline{\hspace{1cm}},\underline{\hspace{1cm}}].
$$

Check the row sum:

$$
\sum_k G_{z,k}=\underline{\hspace{2cm}}.
$$

What does the sign of each component ask the optimizer to do? ____________

For a mean over $m$ observations, what factor is added? __________________

## 9. Parameter gradients

Complete:

$$
G_W=\underline{\hspace{5cm}},
$$

$$
G_b=\underline{\hspace{5cm}}.
$$

| Gradient | Required shape | Why |
| --- | --- | --- |
| $G_W$ | | |
| $G_b$ | | |

Why is the bias gradient summed across observations? _____________________

## 10. Numerical gradient check

Complete the centered difference:

$$
G_j^{\text{num}}
\approx
\frac{
\underline{\hspace{7cm}}
}
{2\varepsilon}.
$$

The lab reports a maximum relative error near $8.1\times10^{-11}$ with tolerance $10^{-6}$.

- Does it pass? ______________________________________________________
- What does it support? ______________________________________________
- What does it not establish? ________________________________________

## 11. Split roles

Write the permitted use of each split.

| Split | Permitted role | Example decision |
| --- | --- | --- |
| Training | | |
| Validation | | |
| Test | | |

A student chooses the learning rate with test cross-entropy and then reports the same test accuracy as final evidence. What is wrong?

__________________________________________________________________________

How should the result be labelled now? __________________________________

## 12. Read a confusion matrix

Rows are true classes and columns are predicted classes:

$$
C=
\begin{bmatrix}
18 & 2 & 0 \\
3 & 10 & 2 \\
1 & 2 & 5
\end{bmatrix}.
$$

Complete:

- number of true class-0 observations: __________________
- class-0 recall: ______________________________________
- class-1 recall: ______________________________________
- class-2 recall: ______________________________________
- total correct: _______________________________________
- total observations: __________________________________
- accuracy: ____________________________________________
- macro recall: ________________________________________

Which confusion is most frequent? ______________________________________

## 13. Accuracy versus macro recall

A majority classifier predicts class 0 for every test observation. Test class counts are $[30,18,12]$.

- accuracy: __________________
- recall for class 0: ________
- recall for class 1: ________
- recall for class 2: ________
- macro recall: ______________

Which metric makes the failure on smaller classes more visible? __________

## 14. Predict before running the lab

Predict:

- probability row sums: ______________________________________________
- effect of adding a constant to every logit in one row: ______________
- gradient-check result relative to $10^{-6}$: ________________________
- majority macro recall for three classes: ____________________________
- expected location of most errors: diagonal / off-diagonal

## 15. Run the NumPy lab

Run:

~~~bash
python labs/week07_softmax_numpy.py
~~~

Record:

| Evidence | Observed value |
| --- | --- |
| extreme-logit probabilities | |
| row sums | |
| largest shift difference | |
| cross-entropy | |
| maximum gradient-check error | |
| validation losses | |
| selected learning rate | |
| test class counts | |
| majority accuracy | |
| majority macro recall | |
| model accuracy | |
| model macro recall | |
| model cross-entropy | |
| confusion matrix | |

Which values came from validation? ______________________________________

Which values came from the locked test split? ____________________________

## 16. Consolidate Weeks 1–7

For each failure, name the earliest course evidence that could detect it.

| Failure | Evidence or check |
| --- | --- |
| target does not match the research question | |
| repeated subjects cross the split | |
| imputer fitted on all data | |
| output width does not equal class count | |
| gradient transpose is wrong | |
| learning rate selected from test data | |
| minority class never predicted | |
| claim is broader than the evaluation | |

Why is a high score insufficient without the rest of this chain?

__________________________________________________________________________

## 17. Project evaluation review

Complete one row for the current project.

| Review item | Actual decision or evidence |
| --- | --- |
| prediction unit | |
| task type | |
| output and objective | |
| split identity | |
| strongest leakage check | |
| primary metric and averaging | |
| decision rule or threshold | |
| simple reference | |
| weakest class or subgroup | |
| test status | |
| supported claim | |
| unsupported claim | |
| next controlled experiment | |

## 18. Exit ticket

1. Softmax must be row-wise because ____________________________________
2. Maximum subtraction is valid because _______________________________
3. $G_Z=(P-Y)/m$ means ________________________________________________
4. Macro recall is useful when ________________________________________
5. A final test result remains independent only if ____________________
