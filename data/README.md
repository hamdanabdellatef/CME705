# Data policy

Datasets are not stored in this first release. Every lab or project dataset must provide:

- an official source and version or retrieval date;
- license and access conditions;
- a checksum when redistribution or stable download permits it;
- a data dictionary and unit of observation;
- a deterministic split-generation script or official split description;
- a statement about sensitive attributes, privacy, consent, and intended use; and
- instructions that fit inside a fresh environment.

Fit imputation, scaling, feature selection, augmentation policies, and other learned preprocessing using the training partition only. Keep the final test partition unavailable for tuning. Use group-aware or time-aware splitting when observations are dependent.

Local data paths `data/raw/` and `data/processed/` are ignored by Git.

## MNIST for Week 10

The Week 10 lab retrieves MNIST through [`torchvision.datasets.MNIST`](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html). TorchVision supplies an official training partition of 60,000 images and a test partition of 10,000 images. The course lab derives a deterministic class-balanced teaching subset of 20,000 training and 5,000 validation images from the official training partition, while retaining the official 10,000-image test partition as locked evidence.

The first run downloads the files under `data/raw/MNIST/`. That directory is ignored by Git. Do not commit or redistribute the downloaded files through this repository. Consult the [original MNIST distribution page](https://yann.lecun.com/exdb/mnist/) and your institution's requirements before redistribution.


## CIFAR-10 transfer-learning extension

The optional Week 10 transfer-learning lab retrieves CIFAR-10 through [`torchvision.datasets.CIFAR10`](https://docs.pytorch.org/vision/0.21/generated/torchvision.datasets.CIFAR10.html). CIFAR-10 provides 50,000 official training images and 10,000 official test images across ten classes. The default extension derives 2,000 class-balanced training images and 1,000 validation images from the official training partition, while retaining the full official test partition for one final evaluation.

Files are downloaded under `data/raw/` and ignored by Git. ImageNet-pretrained ConvNeXt-Tiny weights are stored in the user Torch cache. This repository redistributes neither dataset nor model weights. Consult the [original CIFAR-10 page](https://www.cs.toronto.edu/~kriz/cifar.html), the TorchVision weight documentation, and institutional requirements before redistribution.

## AG News for Week 11

The Week 11 ModernBERT lab retrieves fancyzhx/ag_news through the Hugging Face Datasets library. The dataset card exposes 120,000 official training rows and 7,600 official test rows across World, Sports, Business, and Sci/Tech. The default course run derives balanced subsets of 2,000 training and 500 validation examples from the official training split and retains 1,000 balanced official-test examples for one locked evaluation.

Files are cached under data/raw/huggingface/, and pretrained ModernBERT weights remain in the Hugging Face cache. Both locations are excluded from Git. The dataset card lists the license as unknown and describes academic, non-commercial research use. Review the [dataset card](https://huggingface.co/datasets/fancyzhx/ag_news), original source, and institutional requirements before use or redistribution. This repository includes neither the data nor model weights.
