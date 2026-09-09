import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from labs.week10_cnn_pytorch import (
    SmallCNN,
    one_training_step,
    parameter_count,
    resolve_device,
    stratified_subset_indices,
    trace_shapes,
    translate_right,
)
from labs.week10_transfer_learning_pytorch import (
    build_transfer_model,
    total_parameter_count,
    trainable_parameter_count,
    unfreeze_final_feature_stage,
)
from labs.week11_sequence_pytorch import LSTMClassifier
from labs.week12_autoencoder_pytorch import Autoencoder


def test_cnn_output_shapes_parameters_and_update():
    torch.manual_seed(1)
    model = SmallCNN(num_classes=10).to(resolve_device("cpu"))
    images = torch.randn(4, 1, 28, 28)
    labels = torch.tensor([0, 1, 2, 3])
    before = model.classifier.weight.detach().clone()
    loss = one_training_step(model, images, labels)

    assert parameter_count(model) == 105866
    assert trace_shapes(model, images) == {
        "input": (4, 1, 28, 28),
        "conv1": (4, 16, 28, 28),
        "relu1": (4, 16, 28, 28),
        "max_pool1": (4, 16, 14, 14),
        "conv2": (4, 32, 14, 14),
        "relu2": (4, 32, 14, 14),
        "max_pool2": (4, 32, 7, 7),
        "flatten": (4, 1568),
        "hidden": (4, 64),
        "relu3": (4, 64),
        "logits": (4, 10),
    }
    assert loss > 0
    assert not torch.equal(before, model.classifier.weight)


def test_mnist_subset_indices_are_stratified_disjoint_and_repeatable():
    labels = torch.arange(10).repeat_interleave(20)
    training, validation = stratified_subset_indices(
        labels,
        training_per_class=10,
        validation_per_class=5,
        seed=705,
    )
    repeated_training, repeated_validation = stratified_subset_indices(
        labels,
        training_per_class=10,
        validation_per_class=5,
        seed=705,
    )

    assert torch.equal(training, repeated_training)
    assert torch.equal(validation, repeated_validation)
    assert torch.bincount(labels[training], minlength=10).tolist() == [10] * 10
    assert torch.bincount(labels[validation], minlength=10).tolist() == [5] * 10
    assert set(training.tolist()).isdisjoint(validation.tolist())


def test_translation_uses_zero_fill_and_preserves_shape():
    image = torch.arange(9, dtype=torch.float32).reshape(1, 1, 3, 3)
    shifted = translate_right(image, amount=1)
    assert shifted.shape == image.shape
    assert torch.equal(shifted[..., 0], torch.zeros_like(shifted[..., 0]))
    assert torch.equal(shifted[..., 1:], image[..., :-1])



def test_convnext_transfer_head_and_unfreezing_boundary():
    model, weights = build_transfer_model(num_classes=10, pretrained=False)
    assert weights is None
    assert model(torch.zeros(1, 3, 64, 64)).shape == (1, 10)
    assert total_parameter_count(model) == 27827818
    assert trainable_parameter_count(model) == 7690

    unfreeze_final_feature_stage(model)
    assert trainable_parameter_count(model) == 14297098

def test_sequence_classifier_shape():
    model = LSTMClassifier(input_size=6, hidden_size=12, num_classes=3)
    assert model(torch.randn(8, 20, 6)).shape == (8, 3)


def test_autoencoder_shapes():
    model = Autoencoder(input_size=32, latent_size=4)
    reconstruction, latent = model(torch.randn(12, 32))
    assert reconstruction.shape == (12, 32)
    assert latent.shape == (12, 4)
