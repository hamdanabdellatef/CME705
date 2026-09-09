# Week 10: CNN implementation, training, and inspection

**Guiding question:** How do we turn an understood convolution into a reproducible PyTorch experiment whose claims stay within the evidence?

Week 10 translates the transparent NumPy mechanics from Week 9 into a complete PyTorch workflow on MNIST handwritten digits. Students audit the official data partitions, derive a deterministic training and validation subset without touching the test set, fit normalization on training images, trace every tensor shape, train on CUDA when available, restore the best validation checkpoint, analyze digit-level errors, and probe gradients, activations, and a declared translation.

## Learning outcomes

By the end of the week, students can:

- load MNIST through `torchvision.datasets.MNIST` and describe its official partitions;
- derive validation data only from the official training partition;
- fit preprocessing on training data and reuse it for validation and test images;
- map Week 9 convolution, ReLU, pooling, and affine classification to `torch.nn` modules;
- trace an NCHW batch through a CNN and reconcile every registered parameter;
- select CUDA automatically, move mini-batches to the selected device, and record the hardware;
- distinguish logits from probabilities and use cross-entropy correctly;
- keep a stateful optimizer alive across mini-batches;
- use `model.train()`, `model.eval()`, and `torch.no_grad()` in the correct phases;
- select and restore a checkpoint using validation loss;
- keep the official test partition locked until the procedure is fixed;
- report a confusion matrix, per-digit recall, and structured error records;
- inspect gradient norms and feature activity without overstating their meaning;
- compare prediction consistency with accuracy after a declared translation;
- interpret ImageNet-1K top-1, top-5, parameter, and compute comparisons under a declared weight and evaluation protocol;
- distinguish fixed-feature transfer from staged fine-tuning with a modern CNN; and
- write a results-and-error-analysis section whose claims match the experiment.

## Teaching package

- [Concept notes](week10-notes.md)
- [Accessible slide text](week10-slides.md)
- [PowerPoint lecture deck](slides/week10-cnn-implementation-training-and-inspection.pptx)
- [Student worksheet](week10-worksheet.md)
- [PyTorch MNIST CNN lab](../../labs/week10_cnn_pytorch.py)
- [Transfer-learning reading](../../readings/week10-transfer-learning.md)
- [ConvNeXt-Tiny transfer example](../../labs/week10_transfer_learning_pytorch.py)
- [Instructor guide](../../instructor-notes/week10.md)
- [Results and error analysis milestone](../../research-project/results-and-error-analysis.md)

## Preparation

Install the course dependencies, including TorchVision. The first run downloads MNIST to the ignored `data/raw/` directory.

~~~bash
python -m pip install -r requirements.txt
python labs/week10_cnn_pytorch.py
~~~

The lab selects CUDA automatically when a compatible PyTorch build can access an NVIDIA GPU. Force the portable CPU path with:

~~~bash
python labs/week10_cnn_pytorch.py --device cpu
~~~

Use the [official PyTorch installation selector](https://pytorch.org/get-started/locally/) when a CUDA-enabled installation is required. Check the selected device with:

~~~bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
~~~

## Data protocol

TorchVision exposes 60,000 official training images and 10,000 official test images. To keep the classroom run short and class-balanced, the default lab samples from the official training partition using seed 705:

- 2,000 training images per digit, for 20,000 total;
- 500 validation images per digit, for 5,000 total; and
- all 10,000 official test images, held locked until selection is complete.

Use every remaining official training image after validation with:

~~~bash
python labs/week10_cnn_pytorch.py --full-training
~~~

A classroom-subset result should not be presented as the standard full-training MNIST benchmark.

## Three-hour sequence

1. Inspect MNIST samples, labels, shapes, and official partitions.
2. Derive the stratified training and validation subsets using only training data.
3. Fit scalar normalization from the selected training subset.
4. Select and verify CPU or CUDA execution.
5. Map Week 9 operations to `SmallCNN`.
6. Trace the NCHW shape after every operation.
7. Reconcile 105,866 registered parameters with manual arithmetic.
8. Read a dated ImageNet-1K comparison across accuracy, parameters, and GFLOPs.
9. Connect logits to cross-entropy without adding softmax before the loss.
10. Execute the mini-batch update lifecycle with device transfer.
11. Explain the persistent Adam optimizer and its state.
12. Separate training, validation, and inference modes.
13. Read validation-loss checkpoint selection and restoration.
14. Evaluate the locked official test set against the majority baseline.
15. Identify the weakest digit recalls and largest confusion pairs.
16. Inspect structured errors, gradient norms, and feature activity.
17. Compare prediction consistency and accuracy after a two-pixel shift.
18. Draft the project results-and-error-analysis milestone.

## Reference GPU run

The checked run used seed 705, PyTorch 2.6.0 with CUDA 12.6, TorchVision 0.21.0, and an NVIDIA GeForce RTX 3060 Laptop GPU.

| Evidence | Result |
| --- | ---: |
| training / validation / test sizes | 20,000 / 5,000 / 10,000 |
| training mean / scale | 0.1318 / 0.3092 |
| registered parameters | 105,866 |
| training time | 2.61 seconds |
| best / stopped epoch | 5 / 6 |
| restored validation accuracy | 0.976 |
| majority baseline accuracy | 0.113 |
| locked test accuracy | 0.977 |
| weakest test recall | digit 7: 0.954 |
| misclassified test images | 226 |
| two-pixel translation consistency | 0.944 |
| shifted test accuracy | 0.937 |

Hardware, software versions, and nondeterministic settings can change runtime and the final decimals. The lab records them so a result can be interpreted rather than copied without context.


## CNN evolution on ImageNet-1K

The classroom MNIST model teaches a complete experimental workflow, but it is not intended to represent the scale of contemporary visual recognition. ImageNet-1K provides a useful bridge: its ILSVRC subset contains 1,000 object classes, 1,281,167 training images, and 50,000 validation images according to the [official ImageNet description](https://www.image-net.org/download.php).

The following values are a dated reference snapshot from the [TorchVision 0.21 classification weight table](https://docs.pytorch.org/vision/0.21/models.html). TorchVision reports single-crop ImageNet-1K accuracy for the named weight versions.

| Architecture and weight | Architectural lesson | Top-1 | Top-5 | Parameters | GFLOPs |
| --- | --- | ---: | ---: | ---: | ---: |
| AlexNet `IMAGENET1K_V1` | ReLU, dropout, and GPU training | 56.52% | 79.07% | 61.1M | 0.71 |
| VGG-16 BN `IMAGENET1K_V1` | repeated $3\times3$ convolution blocks | 73.36% | 91.52% | 138.4M | 15.47 |
| ResNet-50 `IMAGENET1K_V2` | residual shortcuts | 80.86% | 95.43% | 25.6M | 4.09 |
| EfficientNet-B0 `IMAGENET1K_V1` | compound depth, width, and resolution scaling | 77.69% | 93.53% | 5.3M | 0.39 |
| ConvNeXt-Tiny `IMAGENET1K_V1` | a modernized pure convolutional network | 82.52% | 96.15% | 28.6M | 4.46 |

Top-1 accuracy requires the highest-scoring class to be correct. Top-5 accuracy counts a prediction as correct when the true class appears among the five highest scores. Parameter count estimates model capacity and storage, while GFLOPs estimate computation for the documented inference input.

Read the columns together. ResNet-50 exceeds VGG-16 BN with far fewer parameters and less computation. EfficientNet-B0 gives up some accuracy for a much smaller compute budget. ConvNeXt-Tiny provides the highest top-1 value among these five reference weights. The table does not isolate architecture alone because weight versions, preprocessing, optimization, augmentation, and training recipes also affect accuracy.

As a research-frontier example, [ConvNeXt V2](https://arxiv.org/abs/2301.00808) reported 88.9% ImageNet top-1 accuracy for a 650-million-parameter Huge model using public training data. Its scale and pretraining protocol differ from the TorchVision reference rows, so it should be discussed separately rather than ranked as if every condition were controlled.

A benchmark score does not establish robustness under distribution shift, calibration, fairness, latency on a target device, energy use, or suitability for a domain. Those limitations can become research questions.

## Transfer learning extension

The [student reading on transfer learning](../../readings/week10-transfer-learning.md) uses ImageNet-pretrained ConvNeXt-Tiny as a fixed feature extractor on a deterministic CIFAR-10 subset, then offers staged fine-tuning of its final feature stage. The accompanying [PyTorch example](../../labs/week10_transfer_learning_pytorch.py) preserves validation-controlled selection and opens the official test partition only after the procedure is fixed.

## Evidence of learning

Students submit or show:

- the official source partitions and derived subset procedure;
- split sizes, class counts, seed, and training-only normalization statistics;
- the selected device and hardware name;
- an exact shape trace through the model;
- a manual parameter count reconciled with PyTorch;
- a correctly ordered mini-batch update including device transfer;
- the best and stopped epochs plus proof that the best checkpoint was restored;
- a locked test confusion matrix and per-digit recall;
- at least ten structured error records grouped by a defensible category;
- gradient norms with a bounded interpretation;
- one descriptive feature-channel observation;
- translation consistency and shifted accuracy for a declared transformation; and
- an interpretation of one ImageNet model comparison that uses accuracy, parameters, compute, and protocol;
- a transfer-learning plan that identifies source weights, frozen and trainable parameters, target data, and fine-tuning criterion; and
- a result paragraph that separates configuration, observation, interpretation, and limitation.

## Claim boundary

The reference run supports claims about one deterministic 20,000-image MNIST training subset, one compact CNN, one validation procedure, one official test partition, and one two-pixel right translation. It does not establish a full-training MNIST benchmark, handwriting recognition outside MNIST, universal translation robustness, or a causal interpretation of feature maps.

## Suggested references

- [TorchVision MNIST dataset](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html)
- [PyTorch datasets and data loaders](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html)
- [PyTorch `nn.Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html)
- [PyTorch `CrossEntropyLoss`](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
- [PyTorch reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness.html)
- [Official PyTorch installation selector](https://pytorch.org/get-started/locally/)
- [Original MNIST distribution page](https://yann.lecun.com/exdb/mnist/)
- [ImageNet data and ILSVRC subset](https://www.image-net.org/download.php)
- [TorchVision 0.21 classification weights](https://docs.pytorch.org/vision/0.21/models.html)
- [PyTorch transfer-learning tutorial](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [ConvNeXt paper](https://arxiv.org/abs/2201.03545)
- [ConvNeXt V2 paper](https://arxiv.org/abs/2301.00808)
