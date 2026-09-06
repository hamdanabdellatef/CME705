# Week 5: Gradient-based optimization

**Guiding question:** How do gradients, learning rates, and update schedules change model parameters and the evidence used to judge training?

Week 5 turns the fixed forward computation from Week 4 into a parameter-learning process. Students derive mean-squared-error gradients for linear regression, apply full-batch, stochastic, and mini-batch updates, diagnose learning-rate behavior, and select one configuration with validation evidence. Multilayer backpropagation begins in Week 6.

## Learning outcomes

By the end of the week, students can:

- distinguish a training objective from an evaluation metric;
- interpret a derivative or gradient as local sensitivity;
- calculate one gradient-descent update by hand;
- derive the weight and bias gradients for linear regression with mean squared error;
- compare full-batch, stochastic, and mini-batch update schedules;
- explain epoch, batch, and update counts;
- implement shuffled mini-batches without omissions or duplicates;
- diagnose learning rates that are too small, useful, or unstable;
- explain why feature scaling changes optimization behavior; and
- use validation evidence to select a configuration while reserving test data.

## Teaching package

- [Concept notes](notes.md)
- [Accessible slide text](slides.md)
- [PowerPoint lecture deck](slides/week05-gradient-based-optimization.pptx)
- [Student worksheet](worksheet.md)
- [NumPy gradient-descent lab](../../labs/week05_gradient_descent.py)
- [Instructor guide](../../instructor-notes/week05.md)
- [Baseline experiment plan](../../research-project/baseline-experiment-plan.md)

## Preparation

Students should be able to evaluate a linear prediction, calculate mean squared error, interpret a matrix transpose, and run a Python script from the repository root.

Run:

~~~bash
python labs/week05_gradient_descent.py
~~~

## Three-hour sequence

1. Separate the optimization objective from the evaluation metric.
2. Interpret slope and gradient on a one-parameter loss function.
3. Calculate one update for $J(w)=(w-3)^2$.
4. Derive the mean-squared-error gradients for linear regression.
5. Compare full-batch, stochastic, and mini-batch updates.
6. Trace exact epoch coverage and relate batch size to update count.
7. Diagnose learning-rate and feature-scaling effects.
8. Predict, run, and interpret the NumPy lab.
9. Complete a validation-controlled baseline experiment plan.

## Evidence of learning

Students submit or show:

- a correct one-step update;
- a complete linear-regression gradient derivation;
- exact batch and update counts for one epoch;
- a diagnosis of three learning-rate histories;
- the observed schedule and validation tables from the lab; and
- a baseline plan that names the frozen split, objective, metric, configuration, and stopping rule.

## Claim boundary

The lab establishes that the implemented analytical gradients agree with numerical checks, stable configurations reduce training loss, every mini-batch epoch covers each example once, and validation data can choose among specified learning rates. Its generated linear data do not show that gradient descent will solve every objective, that the selected learning rate transfers to another dataset, or that a trained model generalizes without independent evaluation.

## Suggested references

- [Deep Learning, Chapter 8: Optimization for Training Deep Models](https://www.deeplearningbook.org/contents/optimization.html)
- [PyTorch optimization tutorial](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)
- [NumPy random Generator](https://numpy.org/doc/stable/reference/random/generator.html)

Use these sources for mechanisms and software behavior. Project claims should rely on primary research and a documented experimental protocol.
