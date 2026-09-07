# Week 9 accessible slides: Convolutional networks and spatial structure

## Slide 1: CME705 Machine Learning

**Neural Networks, Deep Learning, and Research Practice**

**Week 9 — Convolutional networks and spatial structure**

Guiding question: How do local connectivity and shared weights turn spatial structure into learnable evidence?

## Slide 2: Today’s evidence

The lesson produces:

1. a spatial inductive-bias argument;
2. a hand-calculated patch response;
3. a cross-correlation versus convolution distinction;
4. an NCHW/OIHW shape trace;
5. output-size and parameter arithmetic;
6. pooling and receptive-field calculations;
7. equivariance and shared-gradient checks; and
8. a project method-choice review.

## Slide 3: Preserve spatial structure

An image batch has shape:

$$
X\in\mathbb{R}^{N\times C_{\text{in}}\times H\times W}.
$$

Flattening retains the values but removes explicit neighborhood structure from the layer definition.

A convolutional layer processes local patches while keeping a spatial output.

## Slide 4: Two inductive biases

| Property | Dense | Locally connected | Convolutional |
| --- | --- | --- | --- |
| local patches | no | yes | yes |
| shared weights | no | no | yes |
| spatial output | usually no | yes | yes |

Local connectivity limits which inputs one output initially sees.

Weight sharing applies the same detector at many locations.

## Slide 5: One patch produces one response

For stride one and one channel:

$$
Y_{i,j}
=
b+
\sum_u\sum_v
X_{i+u,j+v}K_{u,v}.
$$

For:

$$
\begin{bmatrix}
1&2\\
0&1
\end{bmatrix}
\odot
\begin{bmatrix}
1&0\\
-1&2
\end{bmatrix},
$$

the response is:

$$
1+0+0+2=3.
$$

## Slide 6: Cross-correlation is the library convention

Cross-correlation slides the kernel as stored.

Mathematical convolution flips the kernel first.

For the lab example:

$$
Y_{\text{corr}}
=
\begin{bmatrix}
3&7\\
0&0
\end{bmatrix},
\qquad
Y_{\text{conv}}
=
\begin{bmatrix}
1&7\\
0&-1
\end{bmatrix}.
$$

Learned kernels make the parameterizations equivalent in capacity, but the distinction matters for hand checks and fixed filters.

## Slide 7: State tensor layouts before tracing shapes

Week 9 and PyTorch use:

- input NCHW: $(N,C_{\text{in}},H,W)$;
- kernels OIHW: $(C_{\text{out}},C_{\text{in}},K_h,K_w)$; and
- output NCHW: $(N,C_{\text{out}},H_{\text{out}},W_{\text{out}})$.

A stored image may use HWC. Do not infer the layout from four numbers alone.

## Slide 8: Output-size arithmetic

For one spatial dimension:

$$
L_{\text{out}}
=
\left\lfloor
\frac{
L_{\text{in}}+2P-D(K-1)-1
}{S}
\right\rfloor+1.
$$

For $L_{\text{in}}=8$, $K=3$, $S=2$, $P=1$, and $D=1$:

$$
L_{\text{out}}=4.
$$

The lab maps $(2,3,8,8)$ through $(4,3,3,3)$ kernels to $(2,4,4,4)$.

## Slide 9: Stride, padding, and dilation change geometry

| Choice | Changes | Main question |
| --- | --- | --- |
| stride | sampling distance and output size | how densely should locations be evaluated? |
| padding | boundary access and output size | what is assumed beyond the observed image? |
| dilation | spacing within the kernel | how wide a context without more kernel values? |

A final partial window is omitted by the floor rule.

## Slide 10: One output channel uses every input channel

For one output channel:

$$
Y_{n,o,i,j}
=
b_o+
\sum_c\sum_u\sum_v
X_{n,c,\ldots}K_{o,c,u,v}.
$$

One kernel has one slice per input channel.

The number of feature maps equals $C_{\text{out}}$.

## Slide 11: Weight sharing changes the parameter count

For a standard convolution:

$$
P_{\text{conv}}
=
C_{\text{out}}
(C_{\text{in}}K_hK_w+1).
$$

Week 9 example:

| Mapping | Parameters |
| --- | ---: |
| convolutional | $112$ |
| locally connected | $1792$ |
| dense | $12352$ |

The local sharing factor is $16$ because the same kernel is used at $4\times4$ positions.

## Slide 12: A feature map is a spatial response

A fixed vertical-edge kernel is:

$$
\begin{bmatrix}
-1&0&1\\
-1&0&1\\
-1&0&1
\end{bmatrix}.
$$

On a dark-to-bright step image, the response reaches $3$ along the transition.

A learned feature map reflects the trained kernel and the current input. Its visual appearance is not a complete explanation.

## Slide 13: Convolution needs a nonlinearity

A common block is:

$$
Z=\operatorname{Corr}(X,K)+b,
$$

$$
H=\operatorname{ReLU}(Z).
$$

Convolution is an affine local transformation. ReLU changes the representable function.

Stacking only linear spatial operations remains linear, subject to boundary details.

## Slide 14: Pooling summarizes each channel locally

Max pooling retains the strongest response.

Average pooling retains the local mean.

Pooling:

- reduces spatial resolution;
- changes the information retained;
- enlarges later receptive fields; and
- may add local tolerance.

It does not guarantee translation invariance.

## Slide 15: Max and average pooling preserve different evidence

For:

$$
\begin{bmatrix}
1&3&2&0\\
4&6&5&1\\
0&2&8&7\\
1&3&9&4
\end{bmatrix},
$$

non-overlapping $2\times2$ pooling gives:

$$
Y_{\max}
=
\begin{bmatrix}
6&5\\
3&9
\end{bmatrix},
\qquad
Y_{\text{avg}}
=
\begin{bmatrix}
3.5&2\\
1.5&7
\end{bmatrix}.
$$

## Slide 16: Receptive fields grow across layers

Track receptive field $r$ and input-coordinate jump $j$:

$$
r_\ell=r_{\ell-1}+(K_\ell-1)j_{\ell-1},
$$

$$
j_\ell=j_{\ell-1}S_\ell.
$$

| Layer | Receptive field | Jump |
| --- | ---: | ---: |
| conv $3$, stride $1$ | $3$ | $1$ |
| pool $2$, stride $2$ | $4$ | $2$ |
| conv $3$, stride $1$ | $8$ | $2$ |

## Slide 17: Equivariance and invariance answer different questions

Translation equivariance:

$$
f(T_\Delta X)=T_\Delta f(X).
$$

The feature response moves when the input feature moves.

Translation invariance:

$$
f(T_\Delta X)=f(X).
$$

The output remains unchanged.

A spatial feature map should often be equivariant; a final class prediction may seek partial invariance.

## Slide 18: Equivariance has boundary conditions

The NumPy lab reports:

$$
\text{interior translation error}=0.
$$

This exact result applies to:

- one-position horizontal translation;
- stride-one valid cross-correlation;
- shared weights; and
- the overlapping interior region.

Padding, stride, pooling, cropping, and borders can change the result.

## Slide 19: Shared uses create a shared gradient

Every spatial use contributes to the same kernel gradient:

$$
\frac{\partial J}{\partial K_{c,u,v}}
=
\sum_n\sum_i\sum_j
\frac{\partial J}{\partial Y_{n,i,j}}
X_{n,c,i+u,j+v}.
$$

The lab’s analytical and central-difference gradients agree with maximum relative error approximately:

$$
4.5\times10^{-11}.
$$

## Slide 20: Architecture history records design ideas

| Architecture | Design lesson |
| --- | --- |
| LeNet-5 | end-to-end learned document features |
| AlexNet | scale, ReLU, GPU training, augmentation, dropout |
| VGG | repeated small kernels |
| Inception | parallel spatial scales with projections |
| ResNet | skip connections for deeper optimization |

Use the ideas to reason about a task. An architecture name is not a method justification.

## Slide 21: Method-choice review

Record:

- input geometry and channel meaning;
- observation and prediction units;
- local patterns that may matter;
- transformation assumptions;
- resolution and sample size;
- simple baseline;
- proposed capacity and compute;
- augmentation assumptions;
- validation metric;
- class or subgroup evidence; and
- a simpler alternative.

The method should follow the data and research question.

## Slide 22: Exit and next step

Explain:

1. how local connectivity differs from weight sharing;
2. why one output kernel spans all input channels;
3. why output spatial size uses a floor;
4. why convolutional parameter count excludes output positions;
5. why equivariance is not invariance; and
6. why shared parameters sum gradient evidence across positions.

Next: implement, train, and inspect a compact CNN with PyTorch.
