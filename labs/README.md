# Labs

The first release contains compact, auditable scripts. Dataset-based labs declare their source, split protocol, download location, and locked-test rule.

| Week | Example | Main evidence |
| ---: | --- | --- |
| 1 | [`week01_python_numpy_diagnostic.py`](week01_python_numpy_diagnostic.py) | Array shapes, a majority baseline, and a transparent linear rule |
| 2 | [`week02_evaluation_numpy.py`](week02_evaluation_numpy.py) | A row-wise split exposes source overlap; a grouped split keeps participants disjoint |
| 3 | [`week03_data_quality_numpy.py`](week03_data_quality_numpy.py) | Class-aware metrics expose a majority baseline; one controlled feature change is compared |
| 4 | [`week04_forward_pass_numpy.py`](week04_forward_pass_numpy.py) | Layer shapes, probability normalization, parameter count, and affine-layer collapse |
| 5 | [`week05_gradient_descent.py`](week05_gradient_descent.py) | Exact MSE gradients, update-schedule comparison, learning-rate selection, and one final test result |
| 6 | [`week06_mlp_numpy.py`](week06_mlp_numpy.py) | Explicit caches and gradients pass a finite-difference check; a nonlinear MLP learns XOR while a linear unit does not |
| 7 | [`week07_softmax_numpy.py`](week07_softmax_numpy.py) | Stable softmax, equivalent target encodings, checked gradients, validation selection, and class-aware test evidence |
| 8 | [`week08_dropout_numpy.py`](week08_dropout_numpy.py) | ReLU and saved-mask dropout mechanics, validation-controlled early stopping, and a one-factor generalization comparison |
| 9 | [`week09_convolution_numpy.py`](week09_convolution_numpy.py) | Patch arithmetic, NCHW/OIHW shapes, shared parameters, pooling, receptive fields, translation equivariance, and a checked kernel gradient |
| 10 | [week10_cnn_pytorch.py](week10_cnn_pytorch.py) | TorchVision MNIST, deterministic stratified training/validation subsets, automatic CUDA use, exact shape and parameter audits, validation checkpointing, digit-level errors, gradients, activations, and translation checks |
| 10 extension | [week10_transfer_learning_pytorch.py](week10_transfer_learning_pytorch.py) | ConvNeXt-Tiny fixed-feature transfer and optional staged fine-tuning on CIFAR-10 |
| 11A | [week11_sequence_pytorch.py](week11_sequence_pytorch.py) | A controlled vanilla-RNN/LSTM delayed-memory comparison, locked length stress test, gradient inspection, and causal-attention audit |
| 11B | [week11_modernbert_news.py](week11_modernbert_news.py) | ModernBERT-base fixed-feature transfer and optional partial fine-tuning for real AG News topic classification |
| 12 | [`week12_autoencoder_pytorch.py`](week12_autoencoder_pytorch.py) | Autoencoder reconstructs vectors through a latent representation |

Run scripts from the repository root. Downloaded datasets and pretrained weights stay in ignored local data or cache directories and are not redistributed by this repository.
