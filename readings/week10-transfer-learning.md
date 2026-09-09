# Week 10 reading: transfer learning with ConvNeXt-Tiny

## Guiding question

How can an ImageNet-trained convolutional network become a defensible starting point for a smaller target task?

## Learning outcomes

After this reading and example, students can:

- distinguish a source task from a target task;
- explain fixed-feature transfer and staged fine-tuning;
- replace a pretrained classifier without disturbing the feature extractor;
- count total and trainable parameters;
- reuse the preprocessing attached to a named weight version;
- select a transfer strategy with validation evidence;
- keep the target test partition locked; and
- state how domain shift limits a transfer-learning claim.

## 1. Why transfer learning?

Training a modern visual model from random initialization usually requires substantial data and computation. Transfer learning begins with parameters learned from a source task and adapts them to a target task.

In this extension:

| Role | Choice |
| --- | --- |
| source data and task | ImageNet-1K, 1,000-class natural-image classification |
| source model | ConvNeXt-Tiny |
| source weights | `ConvNeXt_Tiny_Weights.IMAGENET1K_V1` |
| target data and task | CIFAR-10, 10-class natural-image classification |
| target classifier | a new 768-to-10 linear layer |
| selection evidence | target validation loss |
| final evidence | locked official CIFAR-10 test partition |

The source weights already encode reusable visual patterns. Transfer can reduce the amount of target data and training needed, but usefulness depends on similarity between source and target domains.

## 2. Two transfer strategies

### Fixed feature extractor

Freeze every pretrained parameter and replace only the source classifier:

~~~python
weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
model = convnext_tiny(weights=weights)

for parameter in model.parameters():
    parameter.requires_grad = False

input_features = model.classifier[2].in_features
model.classifier[2] = nn.Linear(input_features, 10)
~~~

Newly created classifier parameters remain trainable. The backward pass computes gradients for 7,690 head parameters instead of updating the full 28.6-million-parameter network. During this stage, the example also keeps the frozen feature extractor in evaluation mode so stochastic-depth behavior does not change its output.

This is a strong first experiment when the target dataset is small or compute is limited.

### Staged fine-tuning

After the new head has learned a useful decision surface, unfreeze the final feature stage:

~~~python
for parameter in model.features[-1].parameters():
    parameter.requires_grad = True
~~~

Use a smaller learning rate for pretrained features than for the new classifier:

~~~python
optimizer = torch.optim.AdamW(
    [
        {"params": model.classifier[2].parameters(), "lr": 1e-4},
        {"params": model.features[-1].parameters(), "lr": 1e-5},
    ]
)
~~~

Fine-tuning can adapt high-level features to the target domain. It can also overfit or erase useful source representations. Compare it with the frozen-head checkpoint using validation evidence.

## 3. Preprocessing belongs to the weights

The named TorchVision weight includes an inference transform. For ConvNeXt-Tiny V1, TorchVision documents resizing, a 224-pixel crop, conversion to a tensor, and ImageNet mean and standard-deviation normalization.

~~~python
evaluation_transform = weights.transforms()
~~~

The example creates a compatible randomized training crop but uses the weight transform unchanged for validation and testing. This keeps evaluation deterministic and prevents target-test choices from influencing preprocessing.

CIFAR-10 images are only 32 by 32 pixels, so resizing them to the ConvNeXt input is a real domain and resolution mismatch. Record that limitation. A strong result on this exercise would not prove that every low-resolution domain benefits from ImageNet transfer.

## 4. Target-data protocol

TorchVision provides 50,000 official CIFAR-10 training images and 10,000 official test images. The example derives a deterministic, class-balanced classroom subset from the official training partition:

- 200 training images per class, for 2,000 total;
- 100 validation images per class, for 1,000 total; and
- the complete official 10,000-image test partition, opened after model selection.

The small subset makes the comparison feasible during a course. Label it clearly; it is not the standard full-training CIFAR-10 benchmark.

## 5. Run the example

From the repository root:

~~~bash
python labs/week10_transfer_learning_pytorch.py
~~~

The first run downloads the 109 MB ConvNeXt-Tiny weights and CIFAR-10 beneath local cache and `data/raw/` locations. Those generated downloads are excluded from Git.

For a short head-only experiment:

~~~bash
python labs/week10_transfer_learning_pytorch.py --head-epochs 1 --fine-tune-epochs 0
~~~

Audit the architecture and trainable parameter boundary without downloading data or weights:

~~~bash
python labs/week10_transfer_learning_pytorch.py --audit-only --device cpu
~~~

## 6. Interpret the comparison

Record at least:

| Evidence | Frozen head | Final-stage fine-tuning |
| --- | ---: | ---: |
| trainable parameters | | |
| training time | | |
| best validation loss | | |
| validation accuracy | | |
| locked test accuracy | | |

Answer these questions:

1. Did fine-tuning improve validation loss enough to justify the added trainable parameters?
2. Did validation and test move in the same direction?
3. Which errors remain concentrated by class?
4. Would random initialization be a fair additional baseline under the same target split?
5. Which conclusion is supported only for the classroom subset?

## 7. Research directions

Transfer learning can lead to focused graduate research questions:

- How does source-target similarity predict transfer benefit?
- Which layers should be unfrozen under a fixed compute budget?
- Does transfer improve calibration as well as accuracy?
- How robust are transferred features under corruption or distribution shift?
- Does parameter-efficient adaptation match full fine-tuning?
- How do latency, memory, and energy change across candidate backbones?
- When does self-supervised pretraining outperform supervised ImageNet weights?

For each question, identify the comparison, controlled factors, validation rule, rejection condition, and final test evidence.

## Sources

- [PyTorch transfer learning for computer vision](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [TorchVision ConvNeXt-Tiny weights](https://docs.pytorch.org/vision/0.21/models/generated/torchvision.models.convnext_tiny.html)
- [TorchVision CIFAR-10 dataset](https://docs.pytorch.org/vision/0.21/generated/torchvision.datasets.CIFAR10.html)
- [ConvNeXt: A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)
- [Original CIFAR-10 distribution](https://www.cs.toronto.edu/~kriz/cifar.html)