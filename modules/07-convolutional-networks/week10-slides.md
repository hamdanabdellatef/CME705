# Week 10 accessible slides: CNN implementation, training, and inspection

## Slide 1: CME705 Machine Learning

**Neural Networks, Deep Learning, and Research Practice**

**Week 10: CNN implementation, training, and inspection**

Guiding question: How do we turn an understood convolution into a reproducible PyTorch experiment whose claims stay within the evidence?

## Slide 2: Today’s evidence

The lesson produces:

1. an audited MNIST data protocol;
2. a deterministic training and validation subset;
3. training-only normalization;
4. a recorded CPU or CUDA device;
5. exact tensor and parameter audits;
6. a validation-selected checkpoint;
7. a locked digit-level test evaluation; and
8. error, gradient, activation, and translation checks.

## Slide 3: Week 9 computation inside PyTorch

Week 9 exposed patch correlation, ReLU, and pooling.

Week 10 composes registered PyTorch modules, automatic differentiation, an optimizer, device transfer, and evaluation controls.

The framework executes the declared experiment. The researcher remains responsible for split validity and claim scope.

## Slide 4: MNIST and the classroom subset

| Property | Week 10 value |
| --- | --- |
| input | one-channel $28\times28$ digit image |
| classes | digits 0 through 9 |
| TorchVision partitions | 60,000 train / 10,000 test |
| classroom train / validation | 20,000 / 5,000 |
| locked official test | 10,000 |
| majority test baseline | 0.113 |

The training and validation subsets contain equal counts per digit.

## Slide 5: Split before preprocessing

The validation subset comes only from the official training partition.

Using selected training images:

$$
\widetilde{X}
=
\frac{X-\mu_{\mathrm{train}}}
{\sigma_{\mathrm{train}}}.
$$

Checked values:

$$
\mu_{\mathrm{train}}=0.1318,
\qquad
\sigma_{\mathrm{train}}=0.3092.
$$

The same values transform validation and test images.

## Slide 6: Automatic CUDA selection

The default command uses `--device auto`.

~~~text
CUDA available?  use NVIDIA GPU
otherwise MPS?   use Apple GPU
otherwise        use CPU
~~~

Checked device: **NVIDIA GeForce RTX 3060 Laptop GPU**

The model and each mini-batch must share the selected device.

## Slide 7: Compact MNIST architecture

~~~text
N×1×28×28
Conv 1→16, ReLU, MaxPool
N×16×14×14
Conv 16→32, ReLU, MaxPool
N×32×7×7
Flatten 1,568 → hidden 64 → logits 10
~~~

The hidden layer retains spatial arrangement after two pooling operations.

## Slide 8: Explicit tensor trace

| Stage | Shape for batch 4 |
| --- | --- |
| input | `(4, 1, 28, 28)` |
| conv1 / ReLU | `(4, 16, 28, 28)` |
| max pool 1 | `(4, 16, 14, 14)` |
| conv2 / ReLU | `(4, 32, 14, 14)` |
| max pool 2 | `(4, 32, 7, 7)` |
| flatten | `(4, 1568)` |
| hidden / ReLU | `(4, 64)` |
| logits | `(4, 10)` |

## Slide 9: Registered parameters

| Component | Calculation | Parameters |
| --- | ---: | ---: |
| first convolution | $16(1\cdot3\cdot3+1)$ | 160 |
| second convolution | $32(16\cdot3\cdot3+1)$ | 4,640 |
| hidden layer | $1568\cdot64+64$ | 100,416 |
| classifier | $64\cdot10+10$ | 650 |
| **total** | | **105,866** |

Most parameters sit in the hidden affine layer.


## Slide 10: CNN evolution on ImageNet-1K

TorchVision 0.21 reference weights; single-crop ImageNet-1K accuracy:

| Model | Architectural lesson | Top-1 | Top-5 | Parameters | GFLOPs |
| --- | --- | ---: | ---: | ---: | ---: |
| AlexNet V1 | ReLU, dropout, GPU training | 56.52% | 79.07% | 61.1M | 0.71 |
| VGG-16 BN V1 | stacked $3\times3$ blocks | 73.36% | 91.52% | 138.4M | 15.47 |
| ResNet-50 V2 | residual shortcuts | 80.86% | 95.43% | 25.6M | 4.09 |
| EfficientNet-B0 V1 | compound scaling | 77.69% | 93.53% | 5.3M | 0.39 |
| ConvNeXt-Tiny V1 | modernized pure CNN | 82.52% | 96.15% | 28.6M | 4.46 |

Read accuracy with parameters, compute, data, and the training recipe.

Research-frontier context: ConvNeXt V2 Huge reported 88.9% top-1 with 650M parameters under a different scale and pretraining protocol.

Sources: [TorchVision 0.21 models](https://docs.pytorch.org/vision/0.21/models.html), [ImageNet](https://www.image-net.org/download.php), and [ConvNeXt V2](https://arxiv.org/abs/2301.00808).

## Slide 11: Cross-entropy receives logits

The model returns ten unrestricted scores $z$.

$$
p_k=\frac{\exp(z_k)}{\sum_j\exp(z_j)}.
$$

PyTorch cross-entropy combines log-softmax and negative log-likelihood stably.

Pass logits directly during training. Use softmax later for probabilities or confidence.

## Slide 12: Mini-batch update with device transfer

~~~python
images = images.to(device)
labels = labels.to(device)
model.train()
optimizer.zero_grad(set_to_none=True)
logits = model(images)
loss = F.cross_entropy(logits, labels)
loss.backward()
optimizer.step()
~~~

Read gradients after `backward()` and before they are cleared.

## Slide 13: Persistent Adam state

Adam maintains moving gradient statistics:

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,
$$

$$
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2.
$$

Create Adam once before the loops. Recreating it for each mini-batch erases its history.

## Slide 14: Phase and gradient controls

| Mechanism | Layer behavior | Autograd graph |
| --- | --- | --- |
| `model.train()` | training | stored when enabled |
| `model.eval()` | inference | stored when enabled |
| `torch.no_grad()` | unchanged | not stored |

Evaluation uses `eval()` and `no_grad()` together.

## Slide 15: Deterministic GPU execution

The checked CUDA path records:

- Python, PyTorch, TorchVision, and CUDA versions;
- GPU model and selected device;
- random seed and subset indices;
- `CUBLAS_WORKSPACE_CONFIG`;
- deterministic algorithms; and
- training time.

Reproducibility settings can reduce peak speed. Runtime comparisons must report the settings they used.

## Slide 16: Validation checkpoint selection

| Evidence | Checked value |
| --- | ---: |
| initial validation loss | 2.3064 |
| initial validation accuracy | 0.053 |
| best epoch | 5 |
| stopped epoch | 6 |
| restored validation loss | 0.0802 |
| restored validation accuracy | 0.976 |
| GPU training time | 2.61 seconds |

Validation selected the checkpoint. The official test set remained locked.

## Slide 17: Locked test evidence

| Evidence | Result |
| --- | ---: |
| majority baseline accuracy | 0.113 |
| CNN test loss | 0.0682 |
| CNN test accuracy | 0.977 |
| correct predictions | 9,774 |
| errors | 226 |
| recall range | 0.954 to 0.995 |

Digit 7 had the lowest recall. Overall accuracy alone would hide this.

## Slide 18: Frequent confusion directions

| True digit | Predicted digit | Count |
| ---: | ---: | ---: |
| 7 | 2 | 21 |
| 7 | 9 | 17 |
| 4 | 9 | 16 |
| 3 | 9 | 11 |
| 8 | 9 | 11 |

These counts locate errors for inspection. They do not identify the cause.

## Slide 19: Structured error records

One checked error:

~~~text
index = 247
true = 4
predicted = 2
confidence = 0.524
~~~

A useful record also keeps an allowed digit view and relevant metadata.

Confidence is the maximum softmax value. It is not guaranteed correctness or calibration.

## Slide 20: Gradient and activation inspection

| Parameter group | Gradient norm |
| --- | ---: |
| first convolution | 0.0575 |
| second convolution | 0.1733 |
| hidden layer | 0.5231 |
| classifier | 0.6839 |

The first test image’s most-active feature channel was 1 with mean absolute activation 1.3437.

These are descriptive checks, not causal explanations.

## Slide 21: Two-pixel translation check

Shift every test digit two pixels right with zero fill.

| Measure | Result | Question |
| --- | ---: | --- |
| prediction consistency | 0.944 | Did the predicted digit change? |
| shifted accuracy | 0.937 | Was the shifted prediction correct? |

The check covers one direction, displacement, fill rule, model, and dataset.

## Slide 22: Results and error analysis milestone

Report:

1. frozen experiment configuration and device;
2. selection history and restored checkpoint;
3. baseline and locked class-aware test evidence;
4. structured error categories with counts;
5. one controlled robustness or ablation result;
6. observation, interpretation, and limitation; and
7. a falsifiable next experiment.

## Slide 23: Exit ticket and Week 11

Explain:

1. why validation comes from the official training partition;
2. why CUDA hardware also needs a CUDA-enabled PyTorch build; and
3. what 0.944 consistency and 0.937 shifted accuracy answer separately.

**Next:** sequence models introduce ordered dependencies and learned state.
