# Python setup

## Supported environment

The first release targets Python 3.11 or 3.12 on Windows, macOS, and Linux. It uses NumPy for transparent implementations and PyTorch for deep-learning models. A GPU is optional for the included examples.

## Create an environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

Install the course dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

PyTorch publishes platform-specific installation commands for GPU acceleration. Use the official [PyTorch installation selector](https://pytorch.org/get-started/locally/) when CUDA or another accelerator is required. An NVIDIA GPU can be present while a CPU-only PyTorch build reports CUDA unavailable.

Verify the selected runtime:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Weeks 10–12 select CUDA automatically when available. Use `--device cpu` only when a CPU comparison or fallback is required. First runs may download MNIST, Fashion-MNIST, CIFAR-10, AG News, or pretrained weights beneath the ignored `data/raw/` directory.

## Verify the release

```bash
python -m compileall labs tests
python -m pytest
python labs/week01_python_numpy_diagnostic.py
python labs/week02_evaluation_numpy.py
python labs/week03_data_quality_numpy.py
python labs/week04_forward_pass_numpy.py
python labs/week05_gradient_descent.py
python labs/week06_mlp_numpy.py
python labs/week07_softmax_numpy.py
python labs/week08_dropout_numpy.py
python labs/week09_convolution_numpy.py
python labs/week10_cnn_pytorch.py
python labs/week10_transfer_learning_pytorch.py --audit-only --device cpu
python labs/week11_sequence_pytorch.py --audit-only
python labs/week11_sequence_pytorch.py
python labs/week11_modernbert_news.py --audit-only
python labs/week12_autoencoder_pytorch.py --audit-only
python labs/week12_autoencoder_pytorch.py
python labs/week12_ddpm_cifar10.py --audit-only
```

Each example uses a fixed seed. Most labs use generated data. Week 10 downloads MNIST through TorchVision. The Week 11 ModernBERT extension downloads AG News and pretrained weights. Week 12 downloads Fashion-MNIST for the AE/VAE experiment; its modern extension downloads a public CIFAR-10 DDPM checkpoint and optionally CIFAR-10 reference images. Dataset-based training labs preserve an official test partition for final evaluation. Successful execution verifies the software path; it does not establish scientific validity for a new dataset.

## Reproducible project checklist

Record the Python and package versions, seed, split-generation procedure, dataset version, preprocessing fitted on training data, model configuration, compute device, and commands used to produce each reported result. Do not commit private data, credentials, trained models, or large generated outputs.
