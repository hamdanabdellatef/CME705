# Week 9 notes: Convolutional networks and spatial structure

## 1. Preserve the structure of the input

An image is not only a vector of values. Each value has a spatial location, and nearby values often relate more strongly than distant values.

A batch of images is represented in the Week 9 lab as:

$$
X\in\mathbb{R}^{N\times C_{\text{in}}\times H\times W},
$$

where:

- $N$ is the batch size;
- $C_{\text{in}}$ is the number of input channels;
- $H$ is the height; and
- $W$ is the width.

Flattening produces $N\times(C_{\text{in}}HW)$. The values remain available, but the model no longer receives neighborhood structure through its layer definition.

A convolutional layer encodes two assumptions:

1. **local connectivity:** a unit initially uses a nearby patch; and
2. **weight sharing:** the same kernel is applied at many spatial positions.

These are inductive biases. They make some functions easier to learn but do not guarantee that the assumptions match every dataset.

## 2. Dense, locally connected, and convolutional computation

A dense layer connects every input value to every output unit. A locally connected layer limits each output to one patch but uses different weights at every position. A convolutional layer combines local connectivity with shared weights.

| Layer | Local patches | Shared weights | Spatial output |
| --- | --- | --- | --- |
| dense | no | no | usually flattened |
| locally connected | yes | no | yes |
| convolutional | yes | yes | yes |

Weight sharing reduces parameters and makes the same learned detector usable at several locations.

The reduction is not free. A shared kernel assumes that the same local pattern should have similar meaning across positions. Position-specific tasks may need coordinates, padding choices, region-specific features, or another architecture.

## 3. The patch operation

For a single input channel and one kernel, deep-learning cross-correlation is:

$$
Y_{i,j}
=
b+
\sum_{u=0}^{K_h-1}
\sum_{v=0}^{K_w-1}
X_{iS_h+u,\;jS_w+v}
K_{u,v}.
$$

At each output position:

1. select the corresponding input patch;
2. multiply the patch and kernel element by element;
3. sum the products; and
4. add one output-channel bias.

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

the upper-left output is:

$$
1(1)+2(0)+0(-1)+1(2)=3.
$$

Sliding the kernel gives:

$$
Y_{\text{corr}}
=
\begin{bmatrix}
3&7\\
0&0
\end{bmatrix}.
$$

## 4. Cross-correlation versus mathematical convolution

Mathematical convolution flips the kernel horizontally and vertically before sliding it. Cross-correlation does not.

For the asymmetric example above, the Week 9 lab reports:

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

Most deep-learning libraries implement cross-correlation in their convolution layer. Because the kernel values are learned, a separate flipped parameterization can represent the same family of local detectors. The distinction still matters when checking a hand calculation or porting a fixed filter.

## 5. Channels, kernels, and feature maps

A multi-channel kernel spans every input channel:

$$
K\in
\mathbb{R}^{
C_{\text{out}}\times
C_{\text{in}}\times
K_h\times
K_w}.
$$

One output channel uses one kernel with $C_{\text{in}}$ slices. The layer sums across the input-channel and spatial dimensions.

The output is:

$$
Y\in
\mathbb{R}^{
N\times
C_{\text{out}}\times
H_{\text{out}}\times
W_{\text{out}}}.
$$

The number of feature maps equals $C_{\text{out}}$, not $C_{\text{in}}$. A bias is normally shared across every spatial position of one output channel.

### Tensor layouts

The Week 9 lab and PyTorch use:

- input: NCHW;
- kernel: OIHW; and
- output: NCHW.

Other libraries or stored image files may use NHWC or HWC. State the convention before tracing shapes.

## 6. Output-size arithmetic

For one spatial dimension:

$$
L_{\text{out}}
=
\left\lfloor
\frac{
L_{\text{in}}
+2P
-D(K-1)
-1
}{S}
\right\rfloor
+1,
$$

where:

- $K$ is kernel size;
- $S$ is stride;
- $P$ is padding on each side; and
- $D$ is dilation.

The effective kernel size is:

$$
K_{\text{eff}}=D(K-1)+1.
$$

For $L_{\text{in}}=8$, $K=3$, $S=2$, $P=1$, and $D=1$:

$$
L_{\text{out}}
=
\left\lfloor
\frac{8+2-3}{2}
\right\rfloor
+1
=4.
$$

A batch with shape $(2,3,8,8)$ and four $(3,3,3)$ kernels therefore produces $(2,4,4,4)$.

### What each choice changes

- **Stride** controls how far the kernel moves and reduces spatial sampling when greater than one.
- **Padding** controls boundary access and can preserve size.
- **Dilation** spaces kernel elements apart and increases the effective receptive field.
- **Kernel size** controls the local pattern visible to one output.

The floor operation means a final partial window is omitted.

## 7. Parameters and weight sharing

For a standard convolution with a bias per output channel:

$$
P_{\text{conv}}
=
C_{\text{out}}
\left(
C_{\text{in}}K_hK_w+1
\right).
$$

The count does not multiply by $H_{\text{out}}W_{\text{out}}$ because the same parameters are reused at every position.

For the lab configuration:

- input $(3,8,8)$;
- four $3\times3$ kernels;
- stride $2$;
- padding $1$; and
- output $(4,4,4)$,

the convolution has:

$$
4(3\cdot3\cdot3+1)=112
$$

parameters.

Using a different local kernel at every output position would require:

$$
4\cdot4\cdot4\cdot(3\cdot3\cdot3+1)=1792
$$

parameters. The spatial sharing factor is $4\cdot4=16$.

A fully dense mapping from all 192 input values to all 64 output values would require:

$$
(192+1)64=12352
$$

parameters.

Parameter count alone does not determine compute, memory traffic, generalization, or suitability.

## 8. Learned feature maps

A feature map records the spatial response of one learned kernel. A large positive response means the local patch aligns with the current kernel according to the layer’s learned weights.

A fixed vertical-edge kernel can make the mechanism visible:

$$
K=
\begin{bmatrix}
-1&0&1\\
-1&0&1\\
-1&0&1
\end{bmatrix}.
$$

Applied to a simple left-dark/right-bright step image, the lab produces a maximum response of $3$ along the vertical transition.

In a trained CNN, kernels are optimized with the task objective. Calling a map an “edge detector” requires checking its actual responses rather than assigning meaning from appearance alone.

## 9. Activation after convolution

A common block is:

$$
Z=\operatorname{Corr}(X,K)+b,
$$

followed by:

$$
H=\operatorname{ReLU}(Z).
$$

The convolution is an affine local operation. ReLU supplies the nonlinearity. Stacking convolution without nonlinear activation would collapse into another linear spatial transformation, subject to boundary details.

## 10. Pooling

Pooling summarizes a local region independently in each channel.

For one $2\times2$ patch:

$$
\begin{bmatrix}
1&3\\
4&6
\end{bmatrix},
$$

max pooling returns $6$ and average pooling returns $3.5$.

For the lab input:

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
Y_{\max}=
\begin{bmatrix}
6&5\\
3&9
\end{bmatrix},
\qquad
Y_{\text{avg}}=
\begin{bmatrix}
3.5&2\\
1.5&7
\end{bmatrix}.
$$

Pooling reduces spatial resolution and changes the information retained. Max pooling emphasizes the strongest local response. Average pooling retains the local mean.

Pooling can improve tolerance to small changes, but it does not create unlimited translation invariance. Stride, borders, aliasing, and later layers affect the result.

## 11. Receptive fields

The receptive field of an activation is the region of the original input that can affect it.

Track both:

- $r_\ell$: receptive-field size; and
- $j_\ell$: jump between adjacent activations measured in input coordinates.

Starting with $r_0=1$ and $j_0=1$:

$$
r_\ell
=
r_{\ell-1}
+
(K_\ell-1)j_{\ell-1},
$$

$$
j_\ell
=
j_{\ell-1}S_\ell.
$$

For a $3\times3$ stride-one convolution, $2\times2$ stride-two pooling, and another $3\times3$ stride-one convolution:

| Layer | Receptive field | Jump |
| --- | ---: | ---: |
| conv $3$, stride $1$ | $3$ | $1$ |
| pool $2$, stride $2$ | $4$ | $2$ |
| conv $3$, stride $1$ | $8$ | $2$ |

Each final activation depends on an $8\times8$ region of the original input.

## 12. Translation equivariance is not invariance

A mapping $f$ is translation equivariant when translating the input produces a corresponding translation of the output:

$$
f(T_\Delta X)=T_\Delta f(X).
$$

Stride-one valid cross-correlation with shared weights is equivariant away from newly introduced borders. The Week 9 lab shifts a step image one position right and reports zero interior difference after shifting the output correspondingly.

Important limits include:

- zero padding introduces boundary effects;
- stride greater than one changes the sampling phase;
- pooling can discard precise location;
- cropping can change the valid region; and
- a classifier head may convert spatial equivariance into partial invariance.

Invariance would require:

$$
f(T_\Delta X)=f(X).
$$

A feature map should usually move with the feature. A final class prediction may be designed to remain similar.

## 13. Gradients of a shared kernel

For a scalar objective $J$, each spatial use of a shared kernel contributes to the same parameter gradient:

$$
\frac{\partial J}{\partial K_{c,u,v}}
=
\sum_n
\sum_i
\sum_j
\frac{\partial J}{\partial Y_{n,i,j}}
X_{n,c,i+u,j+v}.
$$

The summation across batches and positions is the backward consequence of weight sharing.

The Week 9 lab constructs an upstream gradient and compares the analytical shared-kernel gradient with central finite differences. The maximum relative error is approximately:

$$
4.5\times10^{-11}.
$$

This supports the derivative implementation at the checked values. It is not a proof for every input or configuration.

## 14. A basic CNN evidence chain

A small classifier often follows:

1. image tensor;
2. convolution;
3. activation;
4. pooling or strided convolution;
5. another convolutional block;
6. aggregation or flattening;
7. class logits;
8. loss;
9. validation-controlled selection; and
10. locked test evaluation.

Every transition needs a shape trace. The classifier receives the representation produced by the spatial feature extractor.

Week 10 implements a compact PyTorch version and inspects one training step. Week 9 establishes the computation needed to audit that implementation.

## 15. Architecture history as design evidence

Several well-known architectures illustrate changes in design:

- **LeNet-5:** end-to-end learned convolution for document recognition;
- **AlexNet:** larger-scale image classification using ReLU, GPUs, data augmentation, and dropout;
- **VGG:** repeated small kernels in a simple deep stack;
- **Inception:** parallel paths at several spatial scales with compute-conscious projections;
- **ResNet:** skip connections that improve optimization of deeper networks.

Treat architecture names as evidence of design ideas, not a list to memorize. Modern method choice still depends on dataset size, resolution, compute, target, and baseline evidence.

## 16. Inspecting feature maps carefully

Feature-map inspection can reveal:

- inactive channels;
- repeated responses;
- strong boundary artifacts;
- unexpected sensitivity to nuisance structure; or
- changes across layers.

It cannot by itself establish:

- the causal reason for a prediction;
- human-interpretable semantics;
- robustness;
- fairness; or
- correct generalization.

Pair visual inspection with controlled perturbations, class or subgroup metrics, and reproducible quantitative evidence.

## 17. The NumPy lab evidence

Run:

~~~bash
python labs/week09_convolution_numpy.py
~~~

The deterministic output includes:

- hand-calculated cross-correlation and mathematical convolution;
- a numerical shared-kernel gradient check;
- NCHW and OIHW shape evidence;
- convolutional, locally connected, and dense parameter counts;
- max and average pooling;
- a fixed edge-response example;
- an interior translation-equivariance check; and
- a receptive-field schedule.

The lab uses generated arrays and downloads no data.

## 18. Method choice for a research project

A CNN is plausible when the data have meaningful local structure and when applying a similar detector at several positions is reasonable.

Record:

- input geometry and channel meaning;
- observation and prediction units;
- relevant local patterns;
- whether translation or another transformation should preserve meaning;
- image resolution and sample size;
- baseline model;
- proposed CNN capacity;
- parameter and activation-memory estimates;
- data augmentation assumptions;
- validation metric;
- subgroup or class evidence; and
- a simpler alternative.

A CNN should be selected because its inductive bias matches the problem and its validation evidence improves the research decision, not because image data automatically require a deep architecture.

## Common misconceptions

| Misconception | Correction |
| --- | --- |
| A CNN sees the whole image in its first layer | One activation initially sees one local receptive field |
| Every output channel corresponds to one input channel | Each output kernel usually spans all input channels |
| Convolution parameters multiply by output height and width | Shared weights are counted once |
| Library convolution always flips the kernel | Most deep-learning layers implement cross-correlation |
| Padding adds real observations | Padding is a declared boundary convention |
| Pooling makes a model translation invariant | It may add local tolerance but has important limits |
| A feature map explains a prediction | It is an internal response that needs further evidence |
| More filters always improve the model | Capacity and compute increase; validation must decide |
| Zero equivariance error proves universal equivariance | It supports the checked region and configuration |
| CNN architecture names justify a project choice | Data geometry, baseline, feasibility, and validation do |

## Preparation for Week 10

Students should bring:

- a completed NCHW/OIHW shape trace;
- the Week 9 method-choice review;
- a proposed image normalization rule;
- the project’s baseline metric; and
- one expected failure mode.

Week 10 maps the same shapes and operations to \`torch.nn.Conv2d\`, trains a compact model, and inspects errors and intermediate features.
