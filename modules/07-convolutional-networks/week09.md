# Week 9: Convolutional networks and spatial structure

**Guiding question:** How do local connectivity and shared weights turn spatial structure into learnable evidence?

Week 9 develops convolution from the patch operation upward. Students calculate multi-channel feature maps, trace tensor shapes, compare parameter counts, distinguish deep-learning cross-correlation from mathematical convolution, and examine pooling, receptive fields, and translation equivariance. A transparent NumPy lab exposes the computation and checks the shared-kernel gradient before Week 10 moves the same ideas into PyTorch.

## Learning outcomes

By the end of the week, students can:

- explain why flattening an image discards an explicit spatial inductive bias;
- distinguish local connectivity from weight sharing;
- calculate a 2D cross-correlation output by hand;
- explain why deep-learning libraries commonly use cross-correlation while calling it convolution;
- use NCHW input and OIHW kernel conventions;
- calculate spatial output size from input, kernel, stride, padding, and dilation;
- trace input and output channels through a convolutional layer;
- compare convolutional, locally connected, and dense parameter counts;
- interpret a feature map as a spatial response to one learned kernel;
- compute max and average pooling outputs;
- calculate receptive-field growth and sampling jump across layers;
- distinguish translation equivariance from invariance;
- explain how a shared kernel accumulates gradient evidence across positions;
- inspect feature responses without treating them as complete explanations;
- justify whether a project’s data and question support a convolutional model; and
- identify the evidence needed before added CNN complexity is warranted.

## Teaching package

- [Concept notes](week09-notes.md)
- [Accessible slide text](week09-slides.md)
- [PowerPoint lecture deck](slides/week09-convolutional-networks-and-spatial-structure.pptx)
- [Student worksheet](week09-worksheet.md)
- [NumPy convolution lab](../../labs/week09_convolution_numpy.py)
- [Instructor guide](../../instructor-notes/week09.md)
- [Method-choice review](../../research-project/method-choice-review.md)

## Preparation

Students should understand matrix multiplication, ReLU, multiclass logits, backpropagation, parameter counting, and validation-controlled comparison. No image dataset download is required.

Run:

~~~bash
python labs/week09_convolution_numpy.py
~~~

## Three-hour sequence

1. Identify the spatial structure that flattening hides.
2. Contrast dense, locally connected, and shared local computation.
3. Calculate one cross-correlation response from a patch and kernel.
4. Distinguish cross-correlation from flipped mathematical convolution.
5. Trace NCHW and OIHW tensor dimensions.
6. Calculate output sizes for stride, padding, and dilation.
7. Connect input channels, output channels, kernels, and feature maps.
8. Compare parameter counts with and without weight sharing.
9. Calculate max and average pooling outputs.
10. Trace receptive fields and sampling jumps through a small CNN.
11. Test translation equivariance away from borders.
12. Connect weight sharing to the kernel-gradient sum.
13. Run and interpret the NumPy lab.
14. Complete the project method-choice review.

## Evidence of learning

Students submit or show:

- a hand-calculated patch response;
- correct output shapes for two convolution configurations;
- an explicit tensor-layout trace;
- a shared-versus-unshared parameter comparison;
- max and average pooling calculations;
- a receptive-field table;
- an equivariance statement with its boundary conditions;
- a passing shared-kernel gradient check;
- an interpretation of the edge-response example;
- a method-choice review tied to data geometry and project evidence; and
- a claim that stays within the deterministic generated example.

## Claim boundary

The lab establishes agreement between a transparent NumPy implementation and hand calculations, a numerical kernel-gradient check at one parameter setting, exact interior translation equivariance for one stride-one valid-correlation example, and parameter-count arithmetic for declared shapes. It does not establish that CNNs always outperform other models, that a displayed feature map explains a prediction, or that invariance holds at borders, under arbitrary stride, or after every operation.

## Suggested references

- [Deep Learning, Chapter 9: Convolutional Networks](https://www.deeplearningbook.org/contents/convnets.html)
- [A guide to convolution arithmetic for deep learning](https://arxiv.org/abs/1603.07285)
- [Gradient-based learning applied to document recognition](https://doi.org/10.1109/5.726791)
- [PyTorch Conv2d documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)

Use these sources to clarify mechanisms and notation. Project decisions still require documented data geometry, split design, a suitable baseline, computational feasibility, and validation evidence.
