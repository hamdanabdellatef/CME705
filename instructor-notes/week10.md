# Week 10 instructor guide: CNN implementation, training, and inspection

## Purpose

Week 10 connects the transparent Week 9 convolution to a complete PyTorch experiment on MNIST. Students should leave able to audit data partitions, device placement, tensor shapes, parameter counts, optimizer state, validation selection, digit-level errors, and bounded robustness claims.

MNIST makes the workflow recognizable while remaining small enough for an in-class GPU run. The default uses a deterministic class-balanced subset from the official training partition and keeps the official test partition intact.

ImageNet benchmark literacy connects the auditable classroom model to the broader CNN research trajectory. Students compare accuracy, parameters, and GFLOPs, then identify the training-protocol information needed before interpreting a ranking. The optional transfer-learning reading applies a modern ImageNet-pretrained CNN to a new target task.

## Before class

Install the dependencies and run:

~~~bash
python -m pip install -r requirements.txt
python labs/week10_cnn_pytorch.py
~~~

The first run downloads MNIST beneath `data/raw/`, which Git ignores.

Verify the device separately:

~~~bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
~~~

If an NVIDIA GPU is present but CUDA is unavailable, the active PyTorch installation probably lacks CUDA support. Use the official [PyTorch installation selector](https://pytorch.org/get-started/locally/) for the operating system and GPU. Do not ask students to install a wheel copied from another machine.

The checked instructor machine used PyTorch 2.6.0 with CUDA 12.6 on an NVIDIA GeForce RTX 3060 Laptop GPU.

## Teaching plan: 180 minutes

| Time | Activity | Evidence |
| ---: | --- | --- |
| 0–15 min | Inspect MNIST and official partitions | data audit |
| 15–30 min | Derive stratified classroom subsets | reproducible indices and counts |
| 30–45 min | Fit training-only normalization | leakage explanation |
| 45–60 min | Select and verify the device | recorded CUDA or CPU path |
| 60–78 min | Trace `SmallCNN` | exact NCHW path |
| 78–92 min | Reconcile parameters | manual total of 105,866 |
| 92–105 min | Read the ImageNet-1K comparison | accuracy, parameter, and GFLOP tradeoff |
| 105–112 min | Break and retrieval check | logits and loss prompt |
| 112–132 min | Execute the update lifecycle | correctly ordered batch |
| 132–145 min | Audit phase and optimizer controls | safe evaluation block |
| 145–158 min | Read validation checkpoint evidence | restored epoch-five state |
| 158–171 min | Analyze digit errors | recall and confusion directions |
| 171–178 min | Compare translation measures | bounded statement |
| 178–180 min | Exit ticket | three concise explanations |

## Reference run

~~~text
dataset=torchvision.datasets.MNIST
device=cuda:0
device_name=NVIDIA GeForce RTX 3060 Laptop GPU
split_sizes=20000/5000/10000
training_mean=0.1318
training_scale=0.3092
parameter_count=105866
best_epoch=5
stopped_epoch=6
training_seconds=2.61
restored_validation_accuracy=0.976
majority_baseline_accuracy=0.113
test_accuracy=0.977
misclassified_count=226
translation_right_2_consistency=0.944
translation_right_2_accuracy=0.937
~~~

Expect runtime and small numerical differences across hardware and software versions. Require a coherent record rather than exact timing.

## Demonstration sequence

Begin with the source partitions. Show that validation indices come from `source_train.targets` and that `source_test` is loaded without contributing to selection.

Compare the default 20,000-image subset with `--full-training`. Explain that classroom speed is a design choice and must appear in the result claim.

Show device selection before training. Move one tensor to CUDA and ask why a CPU label with a CUDA logit causes an error. Then trace shapes and reconcile parameter arithmetic.

Use the ImageNet table immediately after the parameter audit. Ask students to select a model under a 6-million-parameter and 0.5-GFLOP budget, then compare VGG-16 BN with ResNet-50. Emphasize that the weight name, preprocessing, and training recipe belong to the result. Present ConvNeXt V2 Huge as separate frontier context because its scale and pretraining protocol differ.

Assign the transfer-learning reading after class or as a research extension. Students should run the structural audit first, then train the frozen classifier head. Staged fine-tuning is optional and must use validation evidence before the locked CIFAR-10 test evaluation.

Have students arrange the update lines, including device transfer. Point out that Adam exists outside the loops and that predictions move back to CPU for NumPy analysis.

Before opening the test matrix, ask which evidence controlled the checkpoint. Then identify digit 7 as the weakest recall and locate the 7-to-2 and 7-to-9 cells.

Finish with the right-shift test. Prediction consistency decreased less than accuracy, but both changed. Ask what additional translations would be needed before using a broader robustness claim.

## Key answers

### Shape trace

~~~text
(4, 1, 28, 28)
(4, 16, 28, 28)
(4, 16, 28, 28)
(4, 16, 14, 14)
(4, 32, 14, 14)
(4, 32, 14, 14)
(4, 32, 7, 7)
(4, 1568)
(4, 64)
(4, 64)
(4, 10)
~~~

### Parameter count

- first convolution: $16(1\cdot3\cdot3+1)=160$;
- second convolution: $32(16\cdot3\cdot3+1)=4640$;
- hidden affine layer: $1568\cdot64+64=100416$;
- classifier: $64\cdot10+10=650$; and
- total: $105866$.

### Update order

1. Move images and labels to the selected device.
2. Select training behavior.
3. Clear earlier gradients.
4. Compute logits and loss.
5. Run backward propagation.
6. Inspect gradients when required.
7. Update parameters.

### Test evidence

The weakest recalls were digit 7 at 0.954 and digit 8 at 0.960. Frequent error directions included 7-to-2 with 21 images, 7-to-9 with 17 images, and 4-to-9 with 16 images.

One recorded error was test index 247, true digit 4, predicted digit 2, confidence 0.524.

## Common misconceptions

| Misconception | Response |
| --- | --- |
| “The official training set is also the validation set.” | Reserve validation indices before fitting the model. Do not use official test data for selection. |
| “An RTX card means PyTorch automatically has CUDA.” | PyTorch needs a CUDA-enabled build and a compatible driver. Verify through `torch.cuda.is_available()`. |
| “Move the model to CUDA once and everything follows.” | Each input and target batch must move to the same device. |
| “The GPU changes the model’s scientific meaning.” | Device changes execution and sometimes numerical behavior. Record it, then keep the algorithm and evaluation fixed. |
| “Softmax belongs before cross-entropy.” | Cross-entropy receives logits and applies the stable log-softmax internally. |
| “The final epoch is the selected model.” | Restore the best validation checkpoint under the declared rule. |
| “97.7% means all digits work equally.” | Read per-digit recall and off-diagonal confusion counts. |
| “The highest ImageNet top-1 value is automatically the best model.” | Evaluate accuracy with parameters, GFLOPs, latency, data, and the target use case. |
| “Transfer learning means every pretrained layer must change.” | Begin with a frozen feature extractor; unfreeze a controlled stage only when validation evidence motivates it. |
| “The active channel explains the prediction.” | It is an activity summary. A causal explanation needs stronger evidence. |
| “94.4% consistency proves translation invariance.” | It covers one direction, distance, fill rule, model, and dataset. |

## Formative checks

1. Why is the validation subset class-balanced while the official test partition is not exactly balanced?
2. Which operation changes 28 by 28 into 14 by 14?
3. Why does the hidden layer receive 1,568 values?
4. Which layer owns most parameters?
5. Where must images and labels move before the forward pass?
6. Why does deterministic CUDA require a cuBLAS workspace setting?
7. Which evidence identifies digit 7 as the next analysis target?
8. Why is ResNet-50 versus VGG-16 BN informative, and why is it still not a controlled causal comparison?
9. What changes between fixed-feature transfer and staged fine-tuning?

## Extensions

Students who finish early can:

- run `--full-training` and label the comparison clearly;
- compare CPU and CUDA runtime under the same deterministic settings;
- replace the hidden layer with global average pooling and compare capacity and accuracy;
- examine whether deskewing changes the 7-to-2 errors;
- test several declared translations while reporting accuracy and consistency separately; or
- repeat the training subset with several seeds before changing architecture;
- run the ConvNeXt-Tiny transfer-learning extension on CIFAR-10; or
- choose an ImageNet model under a declared parameter or GFLOP budget and justify the decision.

Each extension must preserve training-only preprocessing and validation-controlled selection.

## Assessment guidance

Award credit for the chain of evidence. A complete submission identifies the MNIST subset, device, preprocessing fit, checkpoint rule, digit-level results, structured errors, and claim boundary.

Reduce credit when a student tunes on the official test partition, omits the device, applies softmax before cross-entropy, recreates Adam in the batch loop, reports only aggregate accuracy, or presents a feature activation as a causal explanation.

## Bridge to Week 11

End by comparing spatial and sequential assumptions. The MNIST CNN shares detectors over image locations. Week 11 introduces observations whose interpretation depends on order and learned state. The same experimental controls continue to apply.
