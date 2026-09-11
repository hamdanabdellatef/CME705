# Fourteen-week schedule

The schedule uses stable topic modules. An instructor may adjust dates without reorganizing the repository.

| Week | Topic and module | Practical work | Research milestone |
| ---: | --- | --- | --- |
| 1 | [Course purpose and ML foundations](modules/01-ml-foundations/README.md) | [Python/NumPy diagnostic](labs/week01_python_numpy_diagnostic.py) | [Research-interest memo](assignments/01-research-interest.md) |
| 2 | [Generalization and evaluation](modules/02-data-and-evaluation/README.md) | [Leakage-safe evaluation](labs/week02_evaluation_numpy.py) | Identify a feasible dataset |
| 3 | [Data quality and model comparison](modules/02-data-and-evaluation/README.md) | [Rare-event data-quality audit](labs/week03_data_quality_numpy.py) | [Problem proposal](assignments/02-problem-proposal.md) |
| 4 | [Neural-network computation](modules/03-neural-network-basics/README.md) | [NumPy forward pass](labs/week04_forward_pass_numpy.py) | [Literature comparison](assignments/03-literature-comparison.md) |
| 5 | [Gradient-based optimization](modules/04-optimization/README.md) | [Mini-batch gradient descent](labs/week05_gradient_descent.py) | [Baseline experiment plan](research-project/baseline-experiment-plan.md) |
| 6 | [Backpropagation and losses](modules/05-backpropagation/README.md) | [NumPy XOR network](labs/week06_mlp_numpy.py) | [Reproducible baseline](research-project/reproducible-baseline.md) |
| 7 | [Multiclass learning and consolidation](modules/06-multiclass-and-generalization/README.md) | [Stable softmax and cross-entropy](labs/week07_softmax_numpy.py); midterm review | [Evaluation review](research-project/evaluation-review.md) |
| 8 | [Deep networks and generalization](modules/06-multiclass-and-generalization/week08.md) | [Correct dropout mechanics](labs/week08_dropout_numpy.py) | [Controlled experiment](assignments/04-controlled-experiment.md) |
| 9 | [Convolutional networks and spatial structure](modules/07-convolutional-networks/week09.md) | [NumPy convolution mechanics](labs/week09_convolution_numpy.py) | [Method-choice review](research-project/method-choice-review.md) |
| 10 | [CNN implementation, training, and inspection](modules/07-convolutional-networks/week10.md) | [Reproducible PyTorch CNN](labs/week10_cnn_pytorch.py) | [Results and error analysis](research-project/results-and-error-analysis.md) |
| 11 | [RNNs, LSTMs, attention, and Transformers](modules/08-sequence-models/week11.md) | [Delayed-memory RNN/LSTM](labs/week11_sequence_pytorch.py); [ModernBERT on AG News](labs/week11_modernbert_news.py) | [Sequence-model research direction](research-project/sequence-model-research-direction.md) |
| 12 | [Autoencoders and generative models](modules/09-generative-models/week12.md) | [Autoencoder/VAE on Fashion-MNIST](labs/week12_autoencoder_pytorch.py); [DDPM on CIFAR-10](labs/week12_ddpm_cifar10.py) | [Generative-model ablation](research-project/generative-model-ablation.md) |
| 13 | [Reproducible ML research and peer review](modules/10-research-practice/week13.md) | [Reproducibility audit](labs/week13_reproducibility_audit.py) and peer-review workshop | [Reproducibility review](assignments/05-reproducibility-review.md) and [draft report](research-project/report-template.md) |
| 14 | [Research presentation, final release, and next direction](modules/10-research-practice/week14.md) | [Project presentations and static release audit](labs/week14_release_audit.py) | [Final project submission](assignments/06-final-project-submission.md) and [research-direction note](research-project/research-direction-note.md) |

## Standard weekly rhythm

1. State the question and expected evidence.
2. Study the concept and work through a small example.
3. Predict what an experiment will show.
4. Run a controlled experiment and record the configuration.
5. Explain what the result supports and what remains uncertain.
6. Connect the finding to the semester research project.
