# Convolutional networks

**Guiding question:** How should a model use spatial structure, and what evidence justifies the added architecture?

This module spans Weeks 9–10. Week 9 exposes convolution, pooling, receptive fields, and spatial inductive bias with NumPy. Week 10 maps the same operations to PyTorch, trains a compact CNN on MNIST with automatic CUDA use, and inspects digit errors and intermediate representations.

## Week 9: Convolutional networks and spatial structure

Students calculate the computation before using a framework abstraction.

- [Week 9 course page](week09.md)
- [Concept notes](week09-notes.md)
- [Accessible slide text](week09-slides.md)
- [PowerPoint lecture deck](slides/week09-convolutional-networks-and-spatial-structure.pptx)
- [Student worksheet](week09-worksheet.md)
- [NumPy convolution lab](../../labs/week09_convolution_numpy.py)
- [Instructor guide](../../instructor-notes/week09.md)
- [Method-choice review](../../research-project/method-choice-review.md)

Week 9 evidence includes a hand-calculated patch response, NCHW/OIHW shape tracing, parameter-count comparisons, pooling and receptive-field calculations, an interior translation-equivariance check, and a numerical shared-kernel gradient check.

## Week 10: CNN implementation, training, and inspection

Students turn the audited operations into a complete PyTorch experiment.

- [Week 10 course page](week10.md)
- [Concept notes](week10-notes.md)
- [Accessible slide text](week10-slides.md)
- [PowerPoint lecture deck](slides/week10-cnn-implementation-training-and-inspection.pptx)
- [Student worksheet](week10-worksheet.md)
- [Reproducible PyTorch CNN lab](../../labs/week10_cnn_pytorch.py)
- [Transfer-learning reading](../../readings/week10-transfer-learning.md)
- [ConvNeXt-Tiny transfer example](../../labs/week10_transfer_learning_pytorch.py)
- [Instructor guide](../../instructor-notes/week10.md)
- [Results and error analysis milestone](../../research-project/results-and-error-analysis.md)

Week 10 evidence includes a reproducible MNIST subset, training-only normalization, automatic CUDA selection, exact tensor and parameter audits, persistent Adam state, validation checkpoint restoration, locked digit-level testing, structured error records, gradient and activation inspection, a declared translation check, an ImageNet-1K CNN comparison, and a ConvNeXt-Tiny transfer-learning extension.

## Module outcomes

By the end of Weeks 9–10, students can:

- explain local connectivity and weight sharing;
- calculate convolution and pooling output sizes;
- trace channels and spatial dimensions through a CNN;
- compare parameter and activation costs;
- distinguish equivariance from invariance;
- implement and test convolution mechanics with NumPy;
- train a compact PyTorch CNN;
- inspect class errors and intermediate representations cautiously;
- interpret ImageNet accuracy together with parameters, compute, and protocol;
- design a fixed-feature or staged fine-tuning experiment; and
- justify a model choice using data geometry, baselines, feasibility, and validation evidence.

## Place in the research project

Week 9 produces the method-choice review. Week 10 turns the selected method into results and error analysis. Architecture decisions use validation evidence while the test split remains locked.
