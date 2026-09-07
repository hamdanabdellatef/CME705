# Week 9 instructor guide: Convolutional networks and spatial structure

## Purpose

Week 9 makes convolution auditable before students use a framework implementation. Students calculate patch responses, state tensor layouts, trace shapes, compare parameter counts, distinguish equivariance from invariance, and connect shared forward use to the summed kernel gradient.

The historical lecture contained a strong visual sequence for convolution, feature maps, pooling, architecture history, and MNIST shape changes. This version keeps those ideas while replacing platform-specific code with NumPy, distinguishing cross-correlation from mathematical convolution, adding multi-channel notation, treating pooling claims cautiously, and connecting every mechanism to a checkable result.

## Before class

- Run \`python labs/week09_convolution_numpy.py\`.
- Distribute [the Week 9 worksheet](../modules/07-convolutional-networks/week09-worksheet.md).
- Prepare one $3\times3$ image and one $2\times2$ asymmetric kernel on the board.
- Keep the lab output hidden until students predict the patch and pooling results.
- Ask students to bring the Week 8 controlled-experiment plan.
- Prepare cards containing NCHW and OIHW shapes in mixed order.
- Confirm that the project method-choice review is linked from the schedule.

## Learning evidence

Inspect:

1. a correct patch multiplication and sum;
2. a clear cross-correlation versus convolution distinction;
3. an NCHW/OIHW shape trace;
4. correct output-size arithmetic;
5. a correct parameter count that excludes output positions;
6. max and average pooling calculations;
7. receptive-field and jump calculations;
8. a conditional equivariance claim;
9. an explanation of spatial gradient accumulation; and
10. a method choice justified by data geometry and validation needs.

## Three-hour sequence

| Time | Activity | Instructor action | Student evidence |
| ---: | --- | --- | --- |
| 0–12 min | Spatial structure | Compare an image tensor with a flattened vector | Inductive-bias statement |
| 12–25 min | Layer comparison | Contrast dense, local unshared, and convolutional layers | Completed comparison table |
| 25–45 min | Patch calculation | Slide an asymmetric kernel by hand | $2\times2$ output |
| 45–55 min | Naming convention | Flip the kernel and compare results | Correlation/convolution distinction |
| 55–72 min | Tensor layouts | Sort NCHW and OIHW shape cards | Shape trace |
| 72–90 min | Output geometry | Calculate stride, padding, and dilation examples | Output-size calculations |
| 90–100 min | Break |  |  |
| 100–115 min | Channels and parameters | Trace channel summation and sharing | Parameter comparison |
| 115–130 min | Feature response | Interpret the fixed edge example carefully | Bounded feature-map statement |
| 130–145 min | Pooling | Compare max and average summaries | Pooling calculations |
| 145–160 min | Receptive fields | Track $r$ and $j$ across three layers | Receptive-field table |
| 160–170 min | Equivariance | Translate the input and align outputs | Conditional claim |
| 170–176 min | Shared gradient | Connect spatial reuse with gradient summation | Gradient explanation |
| 176–180 min | Method review and exit | Link the mechanism to project data | Six exit statements |

## Facilitation notes

### Begin with the modeling assumption

Ask: “If a vertical edge matters near the left side, should a different detector be learned for the same edge near the right side?”

Use responses to define weight sharing. Then ask whether the assumption remains reasonable for medical images with fixed anatomy, satellite tiles, documents, or sensors with position-specific calibration. The goal is to treat convolution as an assumption about data, not as a synonym for image classification.

### Make local connectivity and sharing separate

Students often combine the two ideas. Use three cases:

- dense: global and unshared;
- locally connected: local and unshared;
- convolutional: local and shared.

Ask which change reduces parameters and which change limits the initial receptive field.

### Use an asymmetric kernel

A symmetric edge filter hides the correlation/convolution distinction because flipping may reproduce the same or a sign-related pattern. Use:

$$
K=
\begin{bmatrix}
1&0\\
-1&2
\end{bmatrix}.
$$

The lab gives:

$$
Y_{\text{corr}}=
\begin{bmatrix}
3&7\\
0&0
\end{bmatrix},
$$

and:

$$
Y_{\text{conv}}=
\begin{bmatrix}
1&7\\
0&-1
\end{bmatrix}.
$$

State that PyTorch’s \`Conv2d\` applies cross-correlation. Avoid turning the naming convention into a dispute; the practical point is to know whether a fixed kernel is flipped.

### Trace channels explicitly

For every example, read the shape aloud:

- $N$: observations;
- $C_{\text{in}}$: input channels;
- $H,W$: spatial dimensions;
- $C_{\text{out}}$: learned kernels and output feature maps.

Draw one output kernel as a stack with one slice for each input channel. All slices contribute to one output map.

### Ask for the formula before substitution

Use:

$$
L_{\text{out}}
=
\left\lfloor
\frac{L_{\text{in}}+2P-D(K-1)-1}{S}
\right\rfloor+1.
$$

Students should first identify $L_{\text{in}},K,S,P,D$, then substitute. Require them to explain the floor as omission of a partial final window.

### Separate parameters from activations

The lab’s shared convolution has $112$ parameters but produces $2\cdot4\cdot4\cdot4=128$ activation values for the batch. These quantities answer different memory and complexity questions.

The locally connected alternative has $1792$ parameters because it has a different local kernel at all $16$ output positions. The dense alternative has $12352$ parameters because all input values connect to all output values.

### Interpret the edge example conservatively

The fixed kernel is deliberately known, so calling it a vertical-edge filter is justified. A trained kernel should be named only after examining responses across many inputs and controlled perturbations.

Ask students what a negative response means: the transition aligns with the opposite orientation.

### Pooling is a lossy design choice

Avoid saying pooling “prevents overfitting” as a universal mechanism. It reduces resolution and may reduce model capacity or sensitivity, but the effect depends on the whole model and data.

Ask which information max and average pooling discard. Discuss strided convolution as another downsampling option to be examined in Week 10.

### Receptive field needs the jump

Students often add kernel sizes directly. Track the input-coordinate jump:

$$
r_\ell=r_{\ell-1}+(K_\ell-1)j_{\ell-1},
$$

$$
j_\ell=j_{\ell-1}S_\ell.
$$

After stride-two pooling, adjacent next-layer activations are two input positions apart.

### State the boundary of equivariance

The lab’s zero error applies to the overlapping interior under stride-one valid cross-correlation. Show why a newly introduced zero boundary changes the first patch. A one-pixel shift under stride two changes the sampling phase.

Ask students to complete: “The result supports equivariance for ______ under ______, excluding ______.”

### Link sharing to backward computation

A shared parameter appears in every patch operation. The chain rule therefore sums all spatial contributions into one kernel gradient. The finite-difference error of approximately $4.5\times10^{-11}$ supports the implementation at the checked values.

### Architecture history

Use architecture names as short design cases:

- LeNet: learned document features;
- AlexNet: scaling training and regularization;
- VGG: repeated small kernels;
- Inception: parallel scales;
- ResNet: skip connections.

Do not assess year or layer-count memorization. Ask which design problem each idea addressed.

## Expected NumPy lab output

The deterministic run should report:

~~~text
patch_operation
cross_correlation=[[3, 7], [0, 0]]
mathematical_convolution=[[1, 7], [0, -1]]
maximum_kernel_gradient_error=4.522e-11

shape_and_parameter_evidence
input_shape=(2, 3, 8, 8)
kernel_shape=(4, 3, 3, 3)
output_shape=(2, 4, 4, 4)
convolution_parameters=112
locally_connected_parameters=1792
dense_parameters=12352
weight_sharing_ratio=16.0

pooling
max_pool=[[6.0, 5.0], [3.0, 9.0]]
average_pool=[[3.5, 2.0], [1.5, 7.0]]

spatial_evidence
edge_response_shape=(1, 1, 5, 5)
edge_response_max=3.0
interior_translation_equivariance_error=0.000e+00
receptive_fields=conv3:3(jump=1), pool2:4(jump=2), conv3:8(jump=2)
~~~

The peak-location list contains two response columns across all five valid rows.

## Formative feedback language

- “Which axis convention are you using?”
- “Show the patch used for this output.”
- “Was the kernel flipped?”
- “Which values are parameters and which are activations?”
- “Where did the output-channel count come from?”
- “Why did the floor remove the final partial window?”
- “Which weights are reused?”
- “What information did pooling discard?”
- “What is the jump in input coordinates?”
- “Does the output move or stay constant?”
- “Which border region did you exclude?”
- “Where do the spatial gradient contributions meet?”
- “What data property justifies this architecture?”

## Common misconceptions

| Misconception | Instructor response |
| --- | --- |
| Convolution means every output sees the full image | Trace one first-layer receptive field |
| The kernel has one slice total | Draw one slice for every input channel |
| Output positions create new kernel parameters | Point to the same kernel moving across the image |
| Padding creates observed information | Label padded values as a boundary convention |
| Pooling guarantees invariance | Translate by one pixel and compare sampling windows |
| CNNs are automatically best for images | Require a baseline and validation comparison |
| Feature maps explain decisions | Ask for perturbation and outcome evidence |
| A gradient check validates the entire experiment | Restrict the claim to the checked computation |

## Adaptations

### Ninety-minute class

Use slides 1–12, 15–19, 21, and 22. Complete one patch calculation, one output shape, the parameter comparison, one pooling result, the receptive-field table, and the equivariance claim. Assign architecture history and the method-choice review afterward.

### Online delivery

Give each group a different stride/padding configuration. Require a shared shape table. Use a second breakout task in which groups classify claims as equivariance, invariance, or unsupported.

### Limited calculus preparation

Treat the shared gradient as accumulation: the same parameter is used repeatedly, so its update gathers evidence from every use. Keep the numerical check and omit the full index derivation.

## After class

- check every project tensor layout;
- verify that proposed augmentations preserve the intended label;
- ask for a simple non-CNN or shallow baseline;
- reject parameter counts that multiply shared kernels by output positions;
- inspect whether grouping or time-based splits are still required for images;
- check that feature-map figures include the source image and layer identity;
- require class or subgroup metrics in Week 10; and
- preserve the locked test set while architecture decisions are selected on validation evidence.
