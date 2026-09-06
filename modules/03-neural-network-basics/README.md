# Week 4: Neural-network computation

**Guiding question:** How do weighted transformations and nonlinear activations turn a batch of inputs into useful outputs?

Week 4 introduces the computation performed by a feed-forward neural network. The class traces one neuron, one affine layer, and a two-layer network before any training algorithm is introduced. Gradient descent and parameter updates begin in Week 5.

## Learning outcomes

By the end of the week, students can:

- calculate a neuron's pre-activation and output;
- explain the distinct roles of weights, bias, and activation functions;
- trace tensor dimensions through a batched forward pass;
- choose an output activation that matches a prediction task;
- explain why stacked affine layers need nonlinear activations;
- calculate the trainable-parameter count of a dense network; and
- extract architecture evidence from a research paper without claiming that incomparable results are equivalent.

## Teaching package

- [Concept notes](notes.md)
- [Accessible slide text](slides.md)
- [PowerPoint lecture deck](slides/week04-neural-network-computation.pptx)
- [Student worksheet](worksheet.md)
- [NumPy forward-pass lab](../../labs/week04_forward_pass_numpy.py)
- [Instructor guide](../../instructor-notes/week04.md)
- [Literature-comparison assignment](../../assignments/03-literature-comparison.md)

## Preparation

Before class, students should be able to multiply a vector by a scalar, interpret a matrix shape, and run a Python script from the repository root. Review the [Python setup](../../setup.md) if needed.

Run:

~~~bash
python labs/week04_forward_pass_numpy.py
~~~

## Three-hour sequence

1. Calculate one neuron by hand.
2. Move from a vector expression to a batched matrix expression.
3. Compare identity, sigmoid, tanh, and ReLU activations.
4. Prove that two affine layers collapse to one when no nonlinear activation separates them.
5. Trace a two-layer network and match output heads to prediction tasks.
6. Predict, run, and interpret the NumPy lab.
7. Extract an architecture and its evidence from one research paper.

## Evidence of learning

Students submit or show:

- a correct manual neuron calculation;
- a complete dimension trace;
- an appropriate output-head choice with justification;
- the parameter count for the teaching network;
- the observed affine-layer collapse error from the lab; and
- one literature-comparison row that separates reported facts from interpretation.

## Claim boundary

The week establishes how a forward pass is computed. It does not show how parameters are learned, whether a network generalizes, or whether a deeper model is justified for a project. Those claims require an optimization procedure and controlled evaluation.

## Suggested references

- [NumPy broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html)
- [PyTorch neural-network tutorial](https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html)
- [Deep Learning, Chapter 6: Deep Feedforward Networks](https://www.deeplearningbook.org/contents/mlp.html)

Use references to clarify mechanisms. Project claims should rely on primary papers and the comparison rules in the assignment.
