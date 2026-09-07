# Convolutional networks

**Guiding question:** How should a model use spatial structure, and what evidence justifies the added architecture?

This module spans Weeks 9–10. Week 9 exposes convolution, pooling, receptive fields, and spatial inductive bias with NumPy. Week 10 maps the same operations to PyTorch, trains a compact CNN, and inspects errors and intermediate representations.

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

## Week 10 continuation

Week 10 uses [the compact PyTorch CNN lab](../../labs/week10_cnn_pytorch.py) to connect the audited Week 9 computation to framework modules, automatic differentiation, training, error analysis, and feature inspection.

## Module outcomes

By the end of Weeks 9–10, students can:

- explain local connectivity and weight sharing;
- calculate convolution and pooling output sizes;
- trace channels and spatial dimensions through a CNN;
- compare parameter and activation costs;
- distinguish equivariance from invariance;
- implement and test convolution mechanics with NumPy;
- train a compact PyTorch CNN;
- inspect class errors and intermediate representations cautiously; and
- justify a model choice using data geometry, baselines, feasibility, and validation evidence.

## Place in the research project

Week 9 produces the method-choice review. Week 10 turns the selected method into results and error analysis. Architecture decisions use validation evidence while the test split remains locked.
