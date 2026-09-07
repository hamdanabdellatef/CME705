# Week 7 notes: Multiclass learning and consolidation

## 1. From binary to mutually exclusive multiclass prediction

Week 6 used one logit, one sigmoid probability, and a binary target. In a mutually exclusive multiclass problem, one observation belongs to exactly one of $K$ classes.

For a batch of $m$ observations with $d$ features:

$$
Z=XW+b,
$$

where:

- $X:(m,d)$;
- $W:(d,K)$;
- $b:(1,K)$; and
- $Z:(m,K)$.

Each row of $Z$ contains $K$ class scores called logits. The output width equals the number of classes.

This design is different from multilabel prediction. In a multilabel problem, several labels may be present at the same time. Independent sigmoid outputs often match that target structure better than one softmax distribution.

## 2. Logits are scores, not probabilities

A logit can be negative, greater than one, or shifted by a constant. Its absolute value does not have a standalone probability meaning.

For one observation:

$$
z=[z_1,z_2,ldots,z_K].
$$

The relative differences determine the softmax probabilities. If two logits become farther apart, the distribution becomes more concentrated. If all logits are equal, softmax gives equal probabilities.

Prediction is commonly:

$$
hat y=argmax_k z_k.
$$

Because softmax preserves ordering, this equals $argmax_k p_k$. The probabilities still matter for the loss, uncertainty discussion, and decisions that depend on costs.

## 3. Softmax

Softmax converts one row of logits into a probability distribution:

$$
p_k=
rac{e^{z_k}}
{sum_{j=1}^{K}e^{z_j}}.
$$

For every class $k$:

$$
0<p_k<1,
$$

and:

$$
sum_{k=1}^{K}p_k=1.
$$

Every probability depends on every logit in the row. Increasing one logit raises that class's share relative to the others.

Softmax is computed independently for each observation. If $Z$ has shape $(m,K)$, both maximum subtraction and normalization operate across the class axis.

## 4. Shift invariance

For any constant $c$ added to every component of one row:

$$
operatorname{softmax}(z+c)
=
operatorname{softmax}(z).
$$

To see why:

$$
rac{e^{z_k+c}}{sum_j e^{z_j+c}}
=
rac{e^c e^{z_k}}{e^csum_j e^{z_j}}
=
rac{e^{z_k}}{sum_j e^{z_j}}.
$$

Only logit differences matter. This property allows a numerically stable implementation.

## 5. Stable softmax and log-softmax

A direct calculation of $e^{1000}$ overflows in ordinary floating-point arithmetic. Subtract the largest logit in each row:

$$
	ilde z_k=z_k-max_j z_j.
$$

Then compute:

$$
p_k=
rac{e^{	ilde z_k}}
{sum_j e^{	ilde z_j}}.
$$

The largest shifted logit is zero and all others are nonpositive. Therefore, every exponential is at most one.

For cross-entropy, compute log probabilities without taking the logarithm of a rounded probability:

$$
log p_k
=
	ilde z_k
-
logleft(sum_j e^{	ilde z_j}ight).
$$

The Week 7 lab implements this row-wise stable log-softmax.

## 6. Class indices and one-hot targets

Suppose the fixed class order is:

| Index | Class |
| ---: | --- |
| 0 | class A |
| 1 | class B |
| 2 | class C |

A target with class index $1$ can also be written:

$$
y=[0,1,0].
$$

The one-hot vector has one component equal to one and all other components equal to zero.

Class-index targets are compact. One-hot targets make the algebra visible. The mapping must remain fixed across dataset creation, model outputs, confusion matrices, saved predictions, and reports.

## 7. Categorical cross-entropy

For one-hot targets:

$$
J=
-rac{1}{m}
sum_{i=1}^{m}
sum_{k=1}^{K}
Y_{ik}log P_{ik}.
$$

Only the target-class term survives in each row. If $t_i$ is the class index:

$$
J=
-rac{1}{m}
sum_{i=1}^{m}
log P_{i,t_i}.
$$

The two forms define the same objective.

For one example with:

$$
p=[0.70,0.20,0.10],
$$

and target class index $1$:

$$
ell=-log(0.20)approx1.609.
$$

Cross-entropy responds to the probability assigned to the correct class. Accuracy sees only whether the maximum-probability class is correct.

## 8. Softmax with cross-entropy

For softmax output and categorical cross-entropy, differentiation simplifies to:

$$
G_Z=
rac{P-Y}{m}.
$$

This is the gradient of the mean objective with respect to the logits.

For each observation, the row sum is zero:

$$
sum_k G_{Z,ik}=0.
$$

The predicted probabilities sum to one, and the one-hot target also sums to one. Increasing one class's relative score must be balanced against the others.

The formula resembles Week 6:

- sigmoid with binary cross-entropy gives $(P-Y)/m$ for one output;
- softmax with categorical cross-entropy gives $(P-Y)/m$ across $K$ coupled outputs.

The same appearance comes from different output transformations and target structures.

## 9. Linear softmax parameter gradients

For:

$$
Z=XW+b,
$$

the parameter gradients are:

$$
G_W=X^TG_Z,
$$

and:

$$
G_b=sum_{i=1}^{m}G_Z^{(i)}.
$$

Shape check:

- $X^T:(d,m)$;
- $G_Z:(m,K)$;
- $G_W:(d,K)$, matching $W$; and
- $G_b:(1,K)$, matching $b$.

A neural network can replace $X$ with a learned hidden representation. The output-layer reasoning remains the same.

## 10. Numerical gradient checking

For one parameter component $	heta_j$:

$$
G_j^{	ext{num}}
approx
rac{J(	heta_j+arepsilon)-J(	heta_j-arepsilon)}
{2arepsilon}.
$$

Compare numerical and analytical gradients with a relative error:

$$
r_j=
rac{|G_j^{	ext{ana}}-G_j^{	ext{num}}|}
{max(delta,|G_j^{	ext{ana}}|+|G_j^{	ext{num}}|)}.
$$

The Week 7 lab checks every component of $W$ and $b$. Its maximum relative error is approximately $8.1	imes10^{-11}$ with the fixed example.

A low error supports the derivative implementation at the checked values. It does not verify the split, class mapping, metric choice, or scientific interpretation.

## 11. Training, validation, and test roles

The lab uses a stratified 60/20/20 split. Every class appears in every partition.

The procedure is:

1. split the observations;
2. fit the standardization values on training data only;
3. train one model for each candidate learning rate;
4. compare candidates with validation cross-entropy;
5. freeze the selected rate; and
6. evaluate the stored selected model on the test split once.

Validation data can guide a decision. Test data estimate performance after decisions are fixed. Repeatedly choosing configurations from test results converts the test set into another validation set.

## 12. Confusion matrices

For $K$ classes, a confusion matrix $C$ has shape $(K,K)$.

The Week 7 convention is:

- rows represent true classes;
- columns represent predicted classes.

Then:

$$
C_{ij}
=
	ext{number of observations with true class }i
	ext{ and predicted class }j.
$$

Diagonal cells count correct predictions. Off-diagonal cells name specific error directions. Each row sum is the support for one true class.

Always state the orientation. Some software or reports use a different convention.

## 13. Accuracy and macro recall

Accuracy is:

$$
	ext{accuracy}
=
rac{sum_k C_{kk}}
{sum_isum_j C_{ij}}.
$$

It gives equal weight to observations.

Recall for class $k$ is:

$$
R_k=
rac{C_{kk}}{sum_j C_{kj}}.
$$

Macro recall is:

$$
	ext{macro recall}
=
rac{1}{K}sum_{k=1}^{K}R_k.
$$

It gives equal weight to classes. When classes are imbalanced, accuracy can be dominated by the largest class while macro recall reveals whether smaller classes are being recognized.

A majority classifier in the Week 7 lab predicts only class 0. It obtains accuracy $0.500$ because class 0 contains half the test observations, but macro recall is only $0.333$ because two classes have zero recall.

## 14. Loss and decision metrics

Cross-entropy, accuracy, and macro recall answer different questions.

- Cross-entropy examines the target probability and responds to confidence.
- Accuracy examines the final class decision over all observations.
- Macro recall averages the ability to recover each class.

Two models can have equal accuracy and different cross-entropy. A model can also improve overall accuracy while making a small but important class worse.

Choose a primary metric before viewing final test results. State its averaging, class order, decision rule, and direction.

## 15. Week 7 lab evidence

The deterministic script reports:

- stable probabilities for logits near $1000$ and $-1000$;
- row sums equal to one;
- no printed probability change after large row-wise shifts;
- categorical cross-entropy $0.3266$ for the extreme-logit example;
- maximum gradient-check relative error about $8.086	imes10^{-11}$;
- validation loss for four learning-rate candidates;
- selected learning rate $0.80$;
- test class counts $[30,18,12]$;
- majority accuracy $0.500$ and macro recall $0.333$;
- model accuracy $0.933$ and macro recall $0.933$;
- test cross-entropy $0.2179$; and
- the full $3	imes3$ confusion matrix.

The exact test result belongs to one deterministic synthetic dataset and split. It is teaching evidence, not a real-world benchmark.

## 16. Consolidation across Weeks 1–7

| Course decision | Evidence developed |
| --- | --- |
| What is being predicted? | research question, target, and unit of observation |
| Can the data support it? | provenance, quality, missingness, class balance |
| Will evaluation generalize? | group-aware or time-aware split and leakage audit |
| What computation is the model performing? | forward values, shapes, activation roles |
| How are parameters changed? | objective, gradients, optimizer, update schedule |
| Are derivatives implemented correctly? | numerical gradient checks |
| How is a configuration selected? | validation evidence |
| What does the model get wrong? | confusion matrix and error analysis |
| What can be claimed? | procedure-bounded interpretation and limitations |

A model score is only one link in this chain.

## 17. Evaluation-review milestone

The Week 7 project milestone requires:

- task type and prediction unit;
- output and objective compatibility;
- exact split identities and roles;
- leakage checks;
- metric name, formula or implementation, direction, averaging, and decision rule;
- a simple reference using the same evaluation observations;
- class or subgroup evidence;
- selection history;
- an honest test-status declaration;
- error analysis;
- a supported claim, limitation, and next controlled experiment.

The review occurs before adding model complexity. A better model cannot repair an invalid split or a metric that does not match the research question.

## Common implementation failures

| Failure | Consequence |
| --- | --- |
| Applying softmax over all rows at once | observations influence one another's normalization |
| Exponentiating raw extreme logits | overflow or nonfinite values |
| Treating a logit as a probability | invalid interpretation |
| Changing the class order | targets, outputs, and reports no longer align |
| Using one softmax for multilabel targets | forces labels to compete when they may co-occur |
| Taking the logarithm of rounded zero | infinite loss |
| Averaging the gradient twice | gradients become too small |
| Choosing a learning rate from test loss | test evidence is no longer independent |
| Reporting only accuracy | important class failures may remain hidden |
| Omitting confusion orientation | error counts become ambiguous |

## Review questions

1. Why does a $K$-class problem need $K$ logits?
2. When is independent sigmoid output more appropriate than softmax?
3. Why can logits be shifted without changing softmax?
4. Which axis should softmax normalize for a batch?
5. How do class-index and one-hot targets define the same cross-entropy?
6. Why does $G_Z=(P-Y)/m$?
7. Why does each row of $G_Z$ sum to zero?
8. What does a passing gradient check establish?
9. Why can majority accuracy look acceptable while macro recall is poor?
10. What makes a test result independent final evidence?
