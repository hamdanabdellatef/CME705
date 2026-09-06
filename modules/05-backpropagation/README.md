# Week 6: Backpropagation and losses

**Guiding question:** How does the chain rule assign responsibility for a network's output error to every parameter?

Week 6 extends the gradient-descent rule from Week 5 through a two-layer neural network. Students cache the forward pass, derive output- and hidden-layer gradients in reverse order, check every analytical parameter gradient numerically, and train a nonlinear XOR classifier. The lesson pairs sigmoid output with binary cross-entropy and keeps multiclass softmax for Week 7.

## Learning outcomes

By the end of the week, students can:

- represent a two-layer network as a computational graph;
- distinguish forward values from backward gradients;
- explain why hidden units have no supplied target of their own;
- apply the chain rule along one parameter-to-loss path;
- derive matrix gradients for both layers of a tanh network;
- explain the sigmoid and binary-cross-entropy gradient simplification;
- cache the values needed by the backward pass;
- verify analytical gradients with centered finite differences;
- update all parameters only after the gradients are computed;
- explain why a nonlinear hidden layer can represent XOR while one sigmoid unit cannot; and
- turn the Week 5 experiment plan into a reproducible baseline record.

## Teaching package

- [Concept notes](notes.md)
- [Accessible slide text](slides.md)
- [PowerPoint lecture deck](slides/week06-backpropagation-and-losses.pptx)
- [Student worksheet](worksheet.md)
- [NumPy backpropagation lab](../../labs/week06_mlp_numpy.py)
- [Instructor guide](../../instructor-notes/week06.md)
- [Reproducible baseline milestone](../../research-project/reproducible-baseline.md)

## Preparation

Students should be able to trace the Week 4 forward pass, apply the Week 5 gradient-descent update, differentiate a square and a sigmoid or tanh expression, and run a Python script from the repository root.

Run:

~~~bash
python labs/week06_mlp_numpy.py
~~~

## Three-hour sequence

1. Revisit the parameter update and identify the missing network gradients.
2. Trace the forward graph and name the cached values.
3. Pair sigmoid output with binary cross-entropy.
4. Derive the output-layer error signal and parameter gradients.
5. Apply the chain rule through the hidden activation.
6. Complete the matrix-form backward pass and shape trace.
7. Check all analytical parameter gradients with finite differences.
8. Compare a one-unit linear decision model with a nonlinear MLP on XOR.
9. Predict, run, and interpret the NumPy lab.
10. Record the first project baseline so another student can reproduce it.

## Evidence of learning

Students submit or show:

- one complete scalar chain-rule path;
- the six backward equations with valid shapes;
- a table of cached values and their purpose;
- a numerical-gradient result and tolerance;
- a comparison of the linear and nonlinear XOR models;
- an explanation of what the XOR demonstration does not establish; and
- a reproducible baseline record with environment, data identity, command, result, and first error analysis.

## Claim boundary

The lab establishes that its analytical gradients agree with centered finite differences at the checked values, the two-layer tanh network can fit all four XOR observations, and a single affine-sigmoid unit cannot represent that pattern. Because XOR contains only four training observations and no independent test set, the perfect training accuracy is a mechanism demonstration rather than generalization evidence.

## Suggested references

- [Deep Learning, Chapter 6: Deep Feedforward Networks](https://www.deeplearningbook.org/contents/mlp.html)
- [PyTorch automatic differentiation tutorial](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)
- [CS231n backpropagation notes](https://cs231n.github.io/optimization-2/)

Use these sources to clarify mechanisms. A project baseline must also document its dataset, split, metric, command, and actual result.
