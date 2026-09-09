# Week 10 worksheet: CNN implementation, training, and inspection

Name: ________________________________  Date: __________________

Run the MNIST lab from the repository root. Keep the official test partition locked until the worksheet says to evaluate it.

## 1. Audit MNIST

| Item | Value |
| --- | --- |
| image shape | |
| number of classes | |
| official training images | |
| official test images | |
| classroom training images | |
| classroom validation images | |
| locked test images | |
| subset seed | |

Why must validation come from the official training partition?

__________________________________________________________________________

Why should the classroom-subset result be labeled separately from a full-training benchmark?

__________________________________________________________________________

## 2. Verify class support

Record the number of selected examples for each digit.

| Digit | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| training | | | | | | | | | | |
| validation | | | | | | | | | | |
| test | | | | | | | | | | |

Which digit defines the majority test baseline? __________

Calculate its accuracy:

$$
\text{baseline accuracy}
=
\frac{\underline{\hspace{2cm}}}{10000}
=
\underline{\hspace{2cm}}.
$$

## 3. Fit normalization safely

Complete:

$$
\widetilde{X}
=
\frac{X-\underline{\hspace{3.2cm}}}
{\underline{\hspace{3.2cm}}}.
$$

Training mean: ______________  Training scale: ______________

Explain why validation and test images use the training values.

__________________________________________________________________________

## 4. Audit the device

Run:

~~~bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
~~~

Record:

| Item | Value |
| --- | --- |
| selected device | |
| hardware name | |
| PyTorch version | |
| TorchVision version | |
| CUDA runtime reported by PyTorch | |
| deterministic setting | |

Why can an NVIDIA GPU be present while `torch.cuda.is_available()` is false?

__________________________________________________________________________

Where must the model, images, and labels be during one update?

__________________________________________________________________________

## 5. Trace the tensors

Complete the NCHW path for four images.

| Stage | Output shape |
| --- | --- |
| input | `(4, 1, 28, 28)` |
| first convolution | |
| first ReLU | |
| first max pool | |
| second convolution | |
| second ReLU | |
| second max pool | |
| flatten | |
| hidden affine | |
| hidden ReLU | |
| logits | |

Why does flattening produce 1,568 values per example?

__________________________________________________________________________

## 6. Count the parameters

Complete:

$$
P_{\mathrm{conv1}}
=
\underline{\hspace{8cm}}
=
\underline{\hspace{2cm}},
$$

$$
P_{\mathrm{conv2}}
=
\underline{\hspace{8cm}}
=
\underline{\hspace{2cm}},
$$

$$
P_{\mathrm{hidden}}
=
\underline{\hspace{8cm}}
=
\underline{\hspace{2cm}},
$$

$$
P_{\mathrm{classifier}}
=
\underline{\hspace{8cm}}
=
\underline{\hspace{2cm}}.
$$

Total: ______________________

Which component contains most parameters? _______________________________

What controlled comparison does this suggest? ___________________________


## 7. Interpret an ImageNet-1K benchmark

Use the benchmark table in the notes.

Which reference weight has the highest top-1 accuracy? ___________________

Which model has the lowest documented GFLOPs? ___________________________

Compare VGG-16 BN and ResNet-50 using top-1 accuracy, parameters, and GFLOPs.

__________________________________________________________________________

A deployment permits at most 6 million parameters and 0.5 GFLOPs. Which listed model satisfies both constraints? What accuracy tradeoff follows?

__________________________________________________________________________

Why is the table insufficient to claim that architecture alone caused every accuracy difference?

__________________________________________________________________________

Choose one missing dimension that could become a research question: robustness, calibration, fairness, latency, memory, energy, or distribution shift.

Research question: ______________________________________________________

Evidence needed: ________________________________________________________

## 8. Logits and loss

A ten-class model output has shape ______________________.

Are its values probabilities? Explain.

__________________________________________________________________________

Write softmax:

$$
p_k=\underline{\hspace{9cm}}.
$$

What should be passed to `cross_entropy`? _______________________________

When is softmax used in this lab? _______________________________________

## 9. Order one CUDA mini-batch

Number the operations.

| Order | Operation |
| ---: | --- |
| | move images and labels to `device` |
| | `loss.backward()` |
| | `model.train()` |
| | `optimizer.step()` |
| | calculate cross-entropy |
| | `optimizer.zero_grad(set_to_none=True)` |
| | compute logits |

When can the current gradients be inspected? ____________________________

Why must Adam persist across mini-batches? ______________________________

## 10. Separate evaluation controls

| Mechanism | Changes layer behavior? | Prevents graph construction? |
| --- | --- | --- |
| `model.train()` | | |
| `model.eval()` | | |
| `torch.no_grad()` | | |

Write a safe batched evaluation block.

~~~python



~~~

Why are predictions moved back to CPU before conversion to NumPy?

__________________________________________________________________________

## 11. Audit checkpoint selection

| Item | Your run |
| --- | ---: |
| initial validation loss | |
| initial validation accuracy | |
| best epoch | |
| stopped epoch | |
| best validation loss | |
| restored validation accuracy | |
| training time | |

Why must the best `state_dict` be copied? ________________________________

Why does the official test partition remain unopened here?

__________________________________________________________________________

## 12. Read digit-level evidence

| Digit | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| recall | | | | | | | | | | |

Test accuracy: __________  Test errors: __________

Weakest recall: digit __________ with recall __________

Record three frequent confusion directions.

| True digit | Predicted digit | Count |
| ---: | ---: | ---: |
| | | |
| | | |
| | | |

Why does overall accuracy fail to show these directions?

__________________________________________________________________________

## 13. Create structured error records

Inspect at least ten errors.

| Index | True | Predicted | Confidence | Visible handwriting feature | Proposed category |
| ---: | ---: | ---: | ---: | --- | --- |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

Aggregate your categories and state what evidence would test the leading hypothesis.

__________________________________________________________________________

## 14. Inspect gradient norms

| Parameter group | Gradient norm |
| --- | ---: |
| first convolution | |
| second convolution | |
| hidden layer | |
| classifier | |

What does one finite nonzero gradient norm show?

__________________________________________________________________________

What does it fail to establish about the whole run?

__________________________________________________________________________

## 15. Bound the activation claim

Most-active channel: __________  Mean absolute activation: ______________

Complete:

For test image __________, channel __________ had the largest ____________
under the declared summary. This describes ______________________________
but does not establish __________________________________________________.

## 16. Test one translation

| Transformation | Prediction consistency | Shifted accuracy |
| --- | ---: | ---: |
| two pixels right, zero fill | | |

Explain why consistency and accuracy can disagree.

__________________________________________________________________________

List the conditions that bound this result.

__________________________________________________________________________

## 17. Write the result

In 150–200 words, include the subset design, device, selected checkpoint, majority baseline, test accuracy, weakest recall, leading confusion pair, and translation check. End with a limitation.

__________________________________________________________________________

__________________________________________________________________________

__________________________________________________________________________

__________________________________________________________________________

## 18. Exit ticket

Why is the CUDA device a recorded experimental condition rather than only a speed setting?

__________________________________________________________________________

What next controlled experiment follows from the digit-7 errors?

__________________________________________________________________________
