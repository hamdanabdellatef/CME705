# Week 9 worksheet: Convolutional networks and spatial structure

Name: ________________________________  Date: __________________

Use NCHW for input/output tensors and OIHW for kernels unless a question states otherwise.

## 1. Spatial inductive bias

Complete:

An image contains values and _____________________________________________

Flattening retains the values but removes _______________________________

Local connectivity means _______________________________________________

Weight sharing means ___________________________________________________

Give one task for which position-specific weights may be useful.

__________________________________________________________________________

## 2. Compare layer types

Complete the table.

| Layer | Uses local patches? | Shares weights across positions? | Preserves a spatial output? |
| --- | --- | --- | --- |
| dense | | | |
| locally connected | | | |
| convolutional | | | |

Which assumption gives a convolution fewer parameters than a locally connected layer?

__________________________________________________________________________

## 3. One patch response

Let:

$$
P=
\begin{bmatrix}
1&2\\
0&1
\end{bmatrix},
\qquad
K=
\begin{bmatrix}
1&0\\
-1&2
\end{bmatrix}.
$$

Calculate:

$$
\sum_{u,v}P_{u,v}K_{u,v}
=
\underline{\hspace{8cm}}.
$$

If the bias is $-0.5$, the final response is _____________________________

## 4. Slide the kernel

For:

$$
X=
\begin{bmatrix}
1&2&0\\
0&1&3\\
2&1&0
\end{bmatrix},
\qquad
K=
\begin{bmatrix}
1&0\\
-1&2
\end{bmatrix},
$$

complete the valid stride-one cross-correlation:

$$
Y=
\begin{bmatrix}
\underline{\hspace{1.5cm}}&
\underline{\hspace{1.5cm}}\\
\underline{\hspace{1.5cm}}&
\underline{\hspace{1.5cm}}
\end{bmatrix}.
$$

How many patch operations were performed? ________________________________

## 5. Cross-correlation versus convolution

Which operation flips the stored kernel first? __________________________

Which operation is normally implemented by a deep-learning convolution layer?

__________________________________________________________________________

Why can learned cross-correlation kernels represent the same detector family as learned flipped kernels?

__________________________________________________________________________

Why does the distinction still matter for a fixed Sobel-like filter?

__________________________________________________________________________

## 6. Tensor layouts

Write the meaning of every axis.

$$
X:(N,C_{\text{in}},H,W)
$$

$N$: ___________________  $C_{\text{in}}$: ______________________________

$H$: ___________________  $W$: _________________________________________

$$
K:(C_{\text{out}},C_{\text{in}},K_h,K_w)
$$

Why must each standard output kernel span all input channels?

__________________________________________________________________________

## 7. Output shape

Complete:

$$
L_{\text{out}}
=
\left\lfloor
\frac{
\underline{\hspace{7cm}}
}{S}
\right\rfloor+1.
$$

For $L_{\text{in}}=8$, $K=3$, $S=2$, $P=1$, and $D=1$:

$$
L_{\text{out}}=
\underline{\hspace{3cm}}.
$$

Then:

$$
(2,3,8,8)
\longrightarrow
(2,\underline{\hspace{1cm}},\underline{\hspace{1cm}},\underline{\hspace{1cm}})
$$

when $C_{\text{out}}=4$.

## 8. Effective kernel

Complete:

$$
K_{\text{eff}}=
\underline{\hspace{5cm}}.
$$

For $K=3$ and $D=2$, the effective kernel is ____________________________

Sketch which five input positions a one-dimensional three-value dilated kernel uses.

__________________________________________________________________________

## 9. Interpret geometry choices

| Choice | Effect on geometry | Potential risk |
| --- | --- | --- |
| stride greater than one | | |
| zero padding | | |
| dilation greater than one | | |
| larger kernel | | |

Why does the output-size formula contain a floor? ________________________

## 10. Channels and feature maps

Input shape:

$$
(5,3,32,32)
$$

Kernel shape:

$$
(16,3,3,3)
$$

With stride one and padding one, output shape:

$$
(\underline{\hspace{1cm}},
\underline{\hspace{1cm}},
\underline{\hspace{1cm}},
\underline{\hspace{1cm}})
$$

Number of output feature maps per image: ________________________________

Number of bias values: _________________________________________________

## 11. Parameter count

Complete:

$$
P_{\text{conv}}
=
\underline{\hspace{8cm}}.
$$

For $C_{\text{in}}=3$, $C_{\text{out}}=4$, and $K_h=K_w=3$:

$$
P_{\text{conv}}=
\underline{\hspace{4cm}}.
$$

Why is $H_{\text{out}}W_{\text{out}}$ absent? ____________________________

## 12. Shared versus unshared parameters

The output has $4\times4$ spatial positions and four channels.

- convolutional parameters: $112$;
- locally connected parameters: __________________;
- spatial sharing factor: _________________________.

What assumption justifies reusing one kernel at every position?

__________________________________________________________________________

When might that assumption be weak? ____________________________________

## 13. Feature response

For the kernel:

$$
\begin{bmatrix}
-1&0&1\\
-1&0&1\\
-1&0&1
\end{bmatrix},
$$

what local pattern creates a large positive response? ___________________

What creates a large negative response? _________________________________

Why should a learned feature map not automatically be named from its appearance?

__________________________________________________________________________

## 14. Convolution and activation

Complete:

$$
Z=
\underline{\hspace{8cm}},
$$

$$
H=
\underline{\hspace{8cm}}.
$$

What would happen to representational capacity if several convolutional layers were stacked without nonlinear activations?

__________________________________________________________________________

## 15. Pooling calculation

For:

$$
X=
\begin{bmatrix}
1&3&2&0\\
4&6&5&1\\
0&2&8&7\\
1&3&9&4
\end{bmatrix},
$$

compute non-overlapping $2\times2$ max pooling:

$$
Y_{\max}
=
\underline{\hspace{8cm}}.
$$

Compute average pooling:

$$
Y_{\text{avg}}
=
\underline{\hspace{8cm}}.
$$

Which retains the strongest response? __________________________________

Which retains the local mean? __________________________________________

## 16. Receptive-field schedule

Start with:

$$
r_0=1,\qquad j_0=1.
$$

Complete:

$$
r_\ell=
\underline{\hspace{8cm}},
$$

$$
j_\ell=
\underline{\hspace{8cm}}.
$$

| Layer | Kernel | Stride | Receptive field | Jump |
| --- | ---: | ---: | ---: | ---: |
| input | | | $1$ | $1$ |
| convolution | $3$ | $1$ | | |
| pooling | $2$ | $2$ | | |
| convolution | $3$ | $1$ | | |

## 17. Equivariance or invariance

Classify each statement.

| Statement | Equivariance or invariance? |
| --- | --- |
| shifting an edge shifts its feature map | |
| shifting an object leaves the class score unchanged | |
| the output transforms in the same way as the input | |
| the output remains constant under the transformation | |

Write the translation-equivariance relation:

$$
\underline{\hspace{10cm}}.
$$

Write the invariance relation:

$$
\underline{\hspace{10cm}}.
$$

## 18. Limits of equivariance

The lab measures zero interior equivariance error.

List four conditions or boundaries on that result.

1. ___________________________________________________________________
2. ___________________________________________________________________
3. ___________________________________________________________________
4. ___________________________________________________________________

Why can zero padding create a different response near an image boundary?

__________________________________________________________________________

Why can stride two respond differently to a one-pixel translation?

__________________________________________________________________________

## 19. Shared-kernel gradient

Complete:

$$
\frac{\partial J}{\partial K_{c,u,v}}
=
\sum_n\sum_i\sum_j
\underline{\hspace{7cm}}.
$$

Why is there a sum over spatial positions? ______________________________

What does a numerical gradient check compare? ___________________________

What does one passing check fail to prove? ______________________________

## 20. Run the NumPy lab

Run:

~~~bash
python labs/week09_convolution_numpy.py
~~~

Record:

| Evidence | Observed value |
| --- | --- |
| cross-correlation output | |
| mathematical-convolution output | |
| maximum kernel-gradient error | |
| output shape | |
| convolutional parameters | |
| locally connected parameters | |
| dense parameters | |
| max-pooling output | |
| average-pooling output | |
| edge-response maximum | |
| interior equivariance error | |
| final receptive field | |

What implementation claim is supported? _________________________________

What general performance claim is unsupported? __________________________

## 21. Project method choice

Complete:

| Item | Project decision |
| --- | --- |
| input geometry | |
| channel meaning | |
| observation unit | |
| prediction unit | |
| relevant local pattern | |
| transformation that should preserve meaning | |
| baseline | |
| proposed CNN capacity | |
| computational constraint | |
| validation metric | |
| class or subgroup check | |
| simpler alternative | |

Why does your data geometry support or reject weight sharing?

__________________________________________________________________________

## 22. Exit ticket

1. Local connectivity differs from weight sharing because ______________
2. An output kernel spans all input channels because ___________________
3. Output size uses a floor because ___________________________________
4. Convolution parameters exclude output positions because ____________
5. Equivariance differs from invariance because _______________________
6. A shared kernel sums gradients across positions because ____________
