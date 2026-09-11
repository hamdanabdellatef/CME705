# Changelog

## Unreleased
- expanded Week 13 into a complete reproducible-research and peer-review package connecting claims, protocols, artifacts, uncertainty, validity, review, and report revision;
- added a NumPy experiment that produces and verifies a hashed artifact bundle, compares paired training seeds, reports a Student-t confidence interval, and detects tampering;
- added a structured peer-reproduction protocol, focused review-comment workflow, student worksheet, instructor guide, and research-report template;
- added primary-source reading on ACM artifact terminology, the NeurIPS checklist, reproducibility evidence, reporting artifacts, and open research directions;

- expanded Week 12 into a complete autoencoder and generative-model package covering deterministic autoencoders, autoregressive models, VAEs, GANs, normalizing flows, diffusion and score models, and energy-based models;
- added detailed derivations for the ELBO, reparameterization, adversarial game, change of variables, DDPM corruption and reversal, score relation, EBM gradient, and FID;
- replaced the vector-only placeholder with a validation-selected convolutional autoencoder and VAE experiment on balanced Fashion-MNIST subsets with locked testing and generated evidence;
- added a modern pretrained DDPM lab on CIFAR-10 with automatic CUDA selection, explicit reverse steps, DDPM/DDIM comparison, timing, trajectory capture, and optional protocol-declared FID;
- added a paper-sourced CIFAR-10 snapshot, recent-system map, primary-source reading guide, and controlled generative-model ablation and robustness milestone;

- expanded Week 11 into a complete sequence-model package covering recurrent computation, backpropagation through time, LSTM gates, scaled dot-product attention, masking, multi-head attention, and Transformer encoder blocks;
- added a controlled PyTorch RNN-versus-LSTM delayed-memory experiment with validation checkpointing, a locked longer-sequence stress test, gradient inspection, and an auditable attention implementation;
- added a ModernBERT transfer-learning lab for balanced AG News topic classification with automatic CUDA selection, frozen-head training, optional staged unfreezing, locked testing, and class-aware error evidence;
- added a common-protocol Long Range Arena comparison, a dated map of recent sequence architectures, primary-source further reading, and a benchmark-aligned research-direction milestone;

- expanded Week 10 into a complete PyTorch CNN implementation, training, and inspection package;
- replaced the one-step CNN example with a TorchVision MNIST experiment covering deterministic stratified subsets, training-only normalization, automatic CUDA selection, tensor and parameter audits, persistent Adam state, validation checkpoint restoration, and locked digit-level testing;
- added structured error records, gradient and activation inspection, a declared translation check, and a results-and-error-analysis research milestone;
- added ImageNet-1K benchmark literacy with a dated CNN accuracy, parameter, and GFLOP comparison;
- added a student transfer-learning reading and auditable ConvNeXt-Tiny example with frozen-head training, optional staged fine-tuning, and locked CIFAR-10 testing;

- expanded Week 9 into a complete convolutional-networks and spatial-structure package;
- added a tested NumPy lab for cross-correlation, output geometry, multi-channel shapes, weight sharing, pooling, receptive fields, translation equivariance, and shared-kernel gradients;
- added a method-choice review linking model inductive bias, tensor shapes, capacity, augmentation validity, feasibility, and evaluation design;

- expanded Week 8 into a complete deep-networks and generalization package with ReLU, learning-curve diagnosis, $L_2$ regularization, early stopping, and correct inverted dropout;
- added a tested NumPy experiment that holds data, architecture, initialization, optimizer, and selection rules fixed while comparing dropout probabilities;
- expanded the controlled-experiment assignment with a directional hypothesis, fixed controls, uncertainty, class-aware evidence, implementation checks, claim limits, and a rubric;

- expanded Week 7 into a complete stable multiclass learning, class-aware evaluation, and course-consolidation package;
- added a tested NumPy linear-softmax lab with stable log probabilities, equivalent target encodings, analytical and numerical gradients, validation-controlled selection, and a locked test evaluation;
- added an evaluation-review milestone covering task semantics, split roles, leakage, metric specification, class or subgroup evidence, selection history, and test status;

- expanded Week 6 into a complete backpropagation and binary-loss teaching package;
- added a tested NumPy MLP with explicit forward caches, analytical gradients, all-parameter finite-difference checking, and a linear-versus-nonlinear XOR comparison;
- added a reproducible-baseline milestone covering repository state, environment, data identity, actual configuration, split-specific results, and first error analysis;

- expanded Week 5 into a complete gradient-based optimization teaching package;
- added a tested NumPy lab for exact MSE gradients, update schedules, validation-controlled learning-rate selection, and a final test result;
- added a baseline experiment plan that freezes splits, preprocessing, objectives, metrics, stopping rules, and expected evidence;

- expanded Week 4 into a complete neural-network computation teaching package;
- added a tested NumPy forward pass covering shapes, output probabilities, parameter counting, and affine-layer collapse;
- expanded the literature comparison with reproducible search, comparability analysis, baseline selection, and a rubric;
- expanded Week 3 into a complete data-quality and controlled-comparison teaching package;
- added a tested NumPy lab for rare classes, training-only imputation, and missingness indicators;
- expanded the problem proposal with a dataset card, evaluation protocol, feasibility test, and rubric;
- expanded Week 2 into a complete teaching package with notes, slides, a worksheet, and an instructor guide;
- upgraded the evaluation lab to demonstrate row-wise group leakage against a disjoint grouped split;
- added a reproducible dataset-feasibility activity connected to the Week 3 proposal;
- expanded Week 1 into a complete teaching package with notes, slides, a worksheet, and an instructor guide;
- added a tested Python/NumPy diagnostic using generated machine-maintenance data; and
- expanded the research-interest memo with a clear process, rubric, and submission check.

## 0.1.0 - 2026-09-05

- established the title *CME705 Machine Learning: Neural Networks, Deep Learning, and Research Practice*;
- organized the agreed 14-week teaching sequence into stable topic modules;
- replaced the historical MATLAB route with Python, NumPy, and PyTorch;
- added runnable examples for optimization, backpropagation, dropout, CNNs, sequence models, and autoencoders;
- added a semester research-project workflow, assignments, rubric, and reproducibility rules;
- excluded private records, student submissions, copied books, third-party decks and code, autosaves, and duplicate historical files; and
- added tests, GitHub Actions, citation metadata, attribution, and licensing.
