# Week 10 notes: CNN implementation, training, and inspection

## 1. MNIST makes the workflow recognizable

MNIST contains grayscale images of handwritten digits from 0 through 9. TorchVision exposes an official training partition with 60,000 images and an official test partition with 10,000 images. Each image has 28 rows, 28 columns, and one channel:

$$
X\in\mathbb{R}^{N\times1\times28\times28}.
$$

The target is an integer in $\{0,1,\ldots,9\}$.

The Week 10 default uses a deterministic classroom subset. For every digit, it selects 2,000 training images and 500 validation images from the official training partition. The official 10,000-image test partition remains intact and locked.

| Partition | Source | Role | Size |
| --- | --- | --- | ---: |
| training | official train | fit preprocessing and parameters | 20,000 |
| validation | official train | select checkpoint | 5,000 |
| test | official test | final locked evidence | 10,000 |

A student can run the remaining official training images with `--full-training`. Results from the classroom subset and full training must be labeled separately.

## 2. The split precedes preprocessing

The lab first selects training and validation indices using seed 705. It never uses official test labels during selection. Each class contributes exactly the same number of training and validation examples, while the official test set retains its natural class counts.

The majority test class is digit 1 with 1,135 examples. The majority baseline is therefore:

$$
\text{majority accuracy}
=
\frac{1135}{10000}
=
0.1135.
$$

The lab prints 0.113 after rounding to three decimals.

## 3. Fit normalization on selected training images

Pixel values are first converted from integers in $[0,255]$ to floats in $[0,1]$. The scalar mean and population standard deviation are then fitted only on the selected 20,000 training images:

$$
\mu_{\mathrm{train}}
=
\frac{1}{NCHW}
\sum_{n,c,h,w}X^{(\mathrm{train})}_{n,c,h,w},
$$

$$
\sigma_{\mathrm{train}}
=
\sqrt{
\frac{1}{NCHW}
\sum_{n,c,h,w}
\left(X^{(\mathrm{train})}_{n,c,h,w}-\mu_{\mathrm{train}}\right)^2
}.
$$

Every partition uses the same transformation:

$$
\widetilde{X}
=
\frac{X-\mu_{\mathrm{train}}}{\sigma_{\mathrm{train}}}.
$$

The checked values were $\mu_{\mathrm{train}}=0.1318$ and $\sigma_{\mathrm{train}}=0.3092$.

## 4. Device selection is part of the experiment

The default command uses `--device auto`. The lab selects CUDA when `torch.cuda.is_available()` returns `True`, then falls back to Apple MPS or CPU when needed.

~~~python
device = resolve_device("auto")
model = SmallCNN(num_classes=10).to(device)
...
images = images.to(device, non_blocking=device.type == "cuda")
labels = labels.to(device, non_blocking=device.type == "cuda")
~~~

The dataset remains in host memory. Each mini-batch moves to the selected device. CUDA loaders use pinned host memory to support efficient transfer.

The checked system contained an RTX 3060 but originally had a CPU-only PyTorch build. GPU hardware alone is insufficient. The installed PyTorch wheel must include a compatible CUDA runtime. Use the official PyTorch installation selector rather than inventing a wheel URL.

For deterministic CUDA matrix operations, the lab sets:

~~~python
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
~~~

before importing PyTorch. It also fixes random seeds, disables cuDNN benchmarking, and enables deterministic algorithms. Reproducibility settings can reduce peak speed, so record them with runtime results.

## 5. The compact MNIST CNN

The model is:

~~~text
input: N × 1 × 28 × 28
conv1: 1 to 16 channels, 3 × 3, padding 1
ReLU and 2 × 2 max pooling
conv2: 16 to 32 channels, 3 × 3, padding 1
ReLU and 2 × 2 max pooling
flatten: 32 × 7 × 7 = 1,568 values
hidden: 1,568 to 64, then ReLU
classifier: 64 to 10 logits
~~~

The hidden layer retains spatial arrangement after two pooling operations. This differs from global average pooling, which deliberately discards location within each feature map.

## 6. Trace shapes before training

For four images:

| Operation | Output shape |
| --- | --- |
| input | `(4, 1, 28, 28)` |
| first convolution / ReLU | `(4, 16, 28, 28)` |
| first max pool | `(4, 16, 14, 14)` |
| second convolution / ReLU | `(4, 32, 14, 14)` |
| second max pool | `(4, 32, 7, 7)` |
| flatten | `(4, 1568)` |
| hidden affine / ReLU | `(4, 64)` |
| class logits | `(4, 10)` |

The batch axis stays four. Convolutions change channels. Pooling changes spatial size. Flattening combines channels and spatial positions without combining examples.

## 7. Reconcile 105,866 parameters

For a convolution with one bias per output channel:

$$
P_{\mathrm{conv}}
=
C_{\mathrm{out}}
\left(C_{\mathrm{in}}K_hK_w+1\right).
$$

The two convolutional layers contain:

$$
P_{\mathrm{conv1}}
=
16(1\cdot3\cdot3+1)
=
160,
$$

$$
P_{\mathrm{conv2}}
=
32(16\cdot3\cdot3+1)
=
4640.
$$

The hidden layer and classifier contain:

$$
P_{\mathrm{hidden}}
=
1568\cdot64+64
=
100416,
$$

$$
P_{\mathrm{classifier}}
=
64\cdot10+10
=
650.
$$

Therefore:

$$
P_{\mathrm{total}}
=
160+4640+100416+650
=
105866.
$$

Most parameters sit in the hidden affine layer. This observation motivates a later comparison with global pooling or a smaller hidden width.


## 8. Scale the discussion to ImageNet-1K

The Week 10 CNN is deliberately small enough to audit line by line. ImageNet-1K shows what changes when image classification moves from ten centered digits to large-scale natural images. The official ILSVRC subset contains 1,000 object classes, 1,281,167 training images, and 50,000 validation images.

The benchmark uses two common accuracy measures. Top-1 accuracy requires:

$$
\operatorname*{arg\,max}_k p_k=y.
$$

Top-5 accuracy requires the true class to appear within the five highest-scoring classes. Top-5 is useful when many ImageNet categories are visually or semantically similar, but it should not replace inspection of class-level errors.

The [TorchVision 0.21 model table](https://docs.pytorch.org/vision/0.21/models.html) reports the following single-crop ImageNet-1K results for fixed public weight versions:

| Architecture and weight | Architectural lesson | Top-1 | Top-5 | Parameters | GFLOPs |
| --- | --- | ---: | ---: | ---: | ---: |
| AlexNet `IMAGENET1K_V1` | ReLU, dropout, and GPU training | 56.52% | 79.07% | 61.1M | 0.71 |
| VGG-16 BN `IMAGENET1K_V1` | repeated $3\times3$ convolution blocks | 73.36% | 91.52% | 138.4M | 15.47 |
| ResNet-50 `IMAGENET1K_V2` | residual shortcuts | 80.86% | 95.43% | 25.6M | 4.09 |
| EfficientNet-B0 `IMAGENET1K_V1` | compound depth, width, and resolution scaling | 77.69% | 93.53% | 5.3M | 0.39 |
| ConvNeXt-Tiny `IMAGENET1K_V1` | a modernized pure convolutional network | 82.52% | 96.15% | 28.6M | 4.46 |

Several comparisons motivate research questions:

- ResNet-50 achieves higher accuracy than VGG-16 BN with about one fifth as many parameters and about one quarter of the documented GFLOPs.
- EfficientNet-B0 uses the least computation and fewest parameters in the table, which matters for embedded or high-throughput inference.
- ConvNeXt-Tiny reaches the highest top-1 accuracy among these five reference weights while retaining a purely convolutional design.
- The same architecture can receive different reported accuracy under a new training recipe. The weight identifier is therefore part of the result.

The table is a model-zoo snapshot, not a controlled architecture experiment. Preprocessing, image resolution, augmentation, optimizer, training duration, regularization, distillation, and pretraining data can all change the outcome. A paper comparison should state these conditions instead of copying only the largest accuracy.

The [ConvNeXt V2 paper](https://arxiv.org/abs/2301.00808) offers a research-frontier example. Its Huge model reports 88.9% ImageNet top-1 accuracy with 650 million parameters and public training data. That result belongs in a separate callout because its scale and pretraining protocol differ from the TorchVision rows.

ImageNet performance also leaves open robustness, calibration, fairness, distribution shift, latency, memory, and energy questions. A graduate research direction often begins by choosing one of those missing dimensions and defining an evaluation that can reject the proposed hypothesis.

The transfer-learning reading turns this comparison into a practical extension. Students load ConvNeXt-Tiny `IMAGENET1K_V1` weights, replace the 1,000-class head, train the new head on CIFAR-10, and optionally unfreeze only the last feature stage under a smaller learning rate. The source task, target task, trainable parameter set, preprocessing, and checkpoint rule all remain part of the experimental record.

## 9. Logits enter cross-entropy

The classifier returns ten unrestricted scores:

$$
z_i\in\mathbb{R}^{10}.
$$

Softmax converts them to probabilities:

$$
p_{i,k}
=
\frac{\exp(z_{i,k})}
{\sum_j\exp(z_{i,j})}.
$$

For integer target $y_i$, multiclass cross-entropy is:

$$
\mathcal{L}
=
-\frac{1}{B}
\sum_{i=1}^{B}
\log p_{i,y_i}.
$$

`torch.nn.functional.cross_entropy` accepts logits and applies the stable log-softmax internally. Softmax is used later when reporting confidence.

## 10. The update lifecycle includes device transfer

One CUDA mini-batch follows this order:

~~~python
images = images.to(device, non_blocking=True)
labels = labels.to(device, non_blocking=True)
model.train()
optimizer.zero_grad(set_to_none=True)
logits = model(images)
loss = F.cross_entropy(logits, labels)
loss.backward()
optimizer.step()
~~~

Model parameters, input images, and labels must share a device. Metrics and stored predictions move back to CPU before conversion to NumPy.

Adam is created once before the epoch loop. Recreating it for every batch discards its moment estimates and changes the algorithm.

## 11. Evaluation combines phase and graph controls

A safe evaluation uses both:

~~~python
model.eval()
with torch.no_grad():
    logits = model(images)
~~~

`eval()` changes phase-sensitive modules. `no_grad()` prevents graph construction. These controls solve different problems.

Evaluation runs in batches so the full test partition does not create all intermediate feature maps at once on the GPU.

## 12. Validation selects a checkpoint

The checked run trained for at most six epochs with patience two. Validation loss improved through epoch five. Epoch six did not improve by the declared minimum, so the saved epoch-five parameters were restored at the end.

| Evidence | Checked value |
| --- | ---: |
| initial validation loss | 2.3064 |
| initial validation accuracy | 0.053 |
| best epoch | 5 |
| stopped epoch | 6 |
| best validation loss | 0.0802 |
| restored validation accuracy | 0.976 |
| GPU training time | 2.61 seconds |

The initial accuracy falls below 0.10 because an untrained random network need not predict digits uniformly. The majority baseline is computed from the test labels and answers a different question.

## 13. Locked test evidence

After model configuration and checkpoint selection were fixed, the restored model achieved:

| Evidence | Result |
| --- | ---: |
| majority baseline accuracy | 0.113 |
| CNN test loss | 0.0682 |
| CNN test accuracy | 0.977 |
| misclassified images | 226 of 10,000 |

Per-digit recalls were:

| Digit | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Recall | .985 | .995 | .988 | .980 | .974 | .975 | .977 | .954 | .960 | .983 |

Digit 7 had the lowest recall. The largest off-diagonal counts included 7 predicted as 2 on 21 images, 7 predicted as 9 on 17 images, and 4 predicted as 9 on 16 images. Those counts suggest which errors to inspect; they do not identify causes by themselves.

## 14. An error record preserves context

The first recorded error was:

~~~text
index=247
true=4
predicted=2
confidence=0.524
~~~

A useful record also contains an allowed view of the digit and a hypothesis about the handwriting feature that created ambiguity. Confidence is the maximum softmax probability. It is not guaranteed correctness or calibration.

Review at least ten errors and aggregate categories. Possible descriptive categories include open-top 4, short-stem 9, or crossed 7 only when the visual evidence actually supports the label. Do not force every error into a preconceived category.

## 15. Gradient norms verify a path

The checked batch produced:

| Parameter group | Gradient norm |
| --- | ---: |
| first convolution weights | 0.0575 |
| second convolution weights | 0.1733 |
| hidden weights | 0.5231 |
| classifier weights | 0.6839 |

A finite nonzero value confirms a gradient path for this batch and parameter state. One batch does not diagnose optimization over the entire run.

## 16. Feature activity remains descriptive

For channel $c$:

$$
s_c
=
\frac{1}{HW}
\sum_{h,w}|A_{c,h,w}|.
$$

The first test image had most-active channel 1 with mean absolute activation 1.3437. This ranks channels under one summary. It does not name the feature or explain the prediction causally.

## 17. Translation consistency and accuracy differ

The lab shifts every test image two pixels right with zero fill. Prediction consistency is:

$$
\operatorname{consistency}
=
\frac{1}{N}
\sum_{i=1}^{N}
\mathbf{1}
\left[
\widehat{y}_i
=
\widehat{y}^{\mathrm{shift}}_i
\right].
$$

The checked result was 0.944 consistency and 0.937 shifted accuracy. Consistency asks whether a prediction changed. Shifted accuracy asks whether it remained correct. The result covers one displacement, direction, fill rule, model, and dataset.

## 18. A bounded result paragraph

> Using a seed-705 stratified subset of 20,000 MNIST training images, 5,000 validation images, and the official 10,000-image test partition, the restored epoch-five CNN achieved 0.977 test accuracy versus the 0.113 majority baseline. Per-digit recall ranged from 0.954 for digit 7 to 0.995 for digit 1. The error matrix identified 7-to-2 and 7-to-9 as frequent confusion directions. After a two-pixel right shift, accuracy decreased to 0.937 while prediction consistency was 0.944. These results support the declared CNN workflow on the classroom MNIST subset. They do not establish a full-training benchmark, robustness to arbitrary transformations, or performance on handwriting outside MNIST.
