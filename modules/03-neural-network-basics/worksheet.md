# Week 4 worksheet: Neural-network computation

Name: ________________________________  Date: __________________

Show shapes beside every matrix expression. Separate a computed fact from a claim about what a trained model might learn.

## 1. One neuron by hand

Use $x=[2,-1,0.5]$, $w=[0.4,-0.8,1.2]$, and $b=-0.3$.

| Input | Weight | Contribution |
| ---: | ---: | ---: |
| 2.0 | 0.4 | |
| -1.0 | -0.8 | |
| 0.5 | 1.2 | |

- Weighted sum before bias: ______________________________________________
- Pre-activation $z=x^Tw+b$: ___________________________________________
- Sigmoid output $\sigma(z)$: __________________________________________
- ReLU output: ____________________________________________________________

How does changing only the bias affect the calculation?

__________________________________________________________________________

## 2. Trace one dense layer

The batch contains four observations with three features. The hidden layer has four units.

| Quantity | Meaning | Shape |
| --- | --- | --- |
| $X$ | Input batch | $(4,3)$ |
| $W_1$ | Hidden weights | |
| $b_1$ | Hidden biases | |
| $Z_1=XW_1+b_1$ | Hidden pre-activations | |
| $A_1=\operatorname{ReLU}(Z_1)$ | Hidden activations | |

Why can NumPy add $b_1$ to every row of $XW_1$?

__________________________________________________________________________

## 3. Complete the network shape trace

The output layer has two units.

| Expression | Shape | Reason |
| --- | --- | --- |
| $W_2$ | | |
| $b_2$ | | |
| $Z_2=A_1W_2+b_2$ | | |
| $P=\operatorname{softmax}(Z_2)$ | | |

For each observation, how many probabilities are produced? _________________

What must each probability row sum to? ____________________________________

## 4. Match the output head

| Task | Number of output units | Transformation | Decision rule or interpretation |
| --- | ---: | --- | --- |
| Predict temperature | | | |
| Detect one fault or no fault | | | |
| Choose one of six fault types | | | |
| Assign any subset of eight tags | | | |

Why are six independent sigmoids different from one six-class softmax?

__________________________________________________________________________

## 5. Why the activation matters

Start with $H=XW_1+b_1$ and $Y=HW_2+b_2$.

Complete the substitution:

$$
Y=(\underline{\hspace{4cm}})W_2+b_2
$$

$$
Y=X(\underline{\hspace{3cm}})+(\underline{\hspace{4cm}}).
$$

Equivalent weight matrix $W_*$: ________________________________________

Equivalent bias vector $b_*$: __________________________________________

One sentence explaining the consequence:

__________________________________________________________________________

## 6. Activation comparison

For $z\in\{-2,0,2\}$, calculate or estimate each output.

| $z$ | Identity | Sigmoid | Tanh | ReLU |
| ---: | ---: | ---: | ---: | ---: |
| -2 | | | | |
| 0 | | | | |
| 2 | | | | |

Which functions saturate at large magnitude? ______________________________

Which function produces exactly zero for negative inputs? _________________

## 7. Parameter count

For a $3\rightarrow4\rightarrow2$ dense network:

- first-layer weights: ___________________________________________________
- first-layer biases: ____________________________________________________
- second-layer weights: __________________________________________________
- second-layer biases: ___________________________________________________
- total trainable parameters: _____________________________________________

Calculate the total for $10\rightarrow8\rightarrow3$: ___________________

## 8. Predict the NumPy lab

Before running the script, predict:

- shape of the hidden activation: _________________________________________
- shape of the logits: ___________________________________________________
- range of a softmax probability: _________________________________________
- maximum difference after collapsing two affine layers: _________________

Run:

~~~bash
python labs/week04_forward_pass_numpy.py
~~~

Record:

| Evidence | Observed value |
| --- | --- |
| Single-neuron pre-activation | |
| Single-neuron sigmoid output | |
| Probability row sums | |
| Predicted classes | |
| Parameter count | |
| Maximum affine-collapse difference | |

Which observations are guaranteed by the mathematics rather than these particular parameter values?

__________________________________________________________________________

What does this fixed-parameter lab not establish?

__________________________________________________________________________

## 9. Read an architecture from a paper

Paper citation: ___________________________________________________________

| Item | Reported fact | Location in paper | Interpretation or missing detail |
| --- | --- | --- | --- |
| Input representation | | | |
| Layer sequence | | | |
| Hidden activation | | | |
| Output head | | | |
| Training objective | | | |
| Dataset and split | | | |
| Baseline | | | |
| Primary result | | | |

Can this result be compared directly with your proposed project result? State the aligned and unaligned conditions.

__________________________________________________________________________

## 10. Exit ticket

1. A valid matrix expression from today is ________________________________
2. A nonlinear activation is needed because ______________________________
3. My project's likely output head is _____________________________________
4. One architecture fact I still need from the literature is ______________
