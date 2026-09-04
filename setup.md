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

PyTorch publishes platform-specific installation commands for GPU acceleration. Use the official [PyTorch installation selector](https://pytorch.org/get-started/locally/) when CUDA or another accelerator is required.

## Verify the release

```bash
python -m compileall labs tests
python -m pytest
python labs/week01_python_numpy_diagnostic.py
python labs/week02_evaluation_numpy.py
python labs/week05_gradient_descent.py
python labs/week06_mlp_numpy.py
python labs/week07_softmax_numpy.py
python labs/week08_dropout_numpy.py
python labs/week10_cnn_pytorch.py
python labs/week11_sequence_pytorch.py
python labs/week12_autoencoder_pytorch.py
```

Each example uses a fixed seed and generated data. Successful execution verifies the software path; it does not establish scientific validity for a new dataset.

## Reproducible project checklist

Record the Python and package versions, seed, split-generation procedure, dataset version, preprocessing fitted on training data, model configuration, compute device, and commands used to produce each reported result. Do not commit private data, credentials, trained models, or large generated outputs.
