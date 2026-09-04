import pytest

torch = pytest.importorskip("torch")

from labs.week10_cnn_pytorch import SmallCNN, one_training_step
from labs.week11_sequence_pytorch import LSTMClassifier
from labs.week12_autoencoder_pytorch import Autoencoder


def test_cnn_output_and_update():
    torch.manual_seed(1)
    model = SmallCNN(num_classes=5)
    images = torch.randn(4, 1, 28, 28)
    labels = torch.tensor([0, 1, 2, 3])
    before = model.classifier.weight.detach().clone()
    loss = one_training_step(model, images, labels)
    assert model(images).shape == (4, 5)
    assert loss > 0
    assert not torch.equal(before, model.classifier.weight)


def test_sequence_classifier_shape():
    model = LSTMClassifier(input_size=6, hidden_size=12, num_classes=3)
    assert model(torch.randn(8, 20, 6)).shape == (8, 3)


def test_autoencoder_shapes():
    model = Autoencoder(input_size=32, latent_size=4)
    reconstruction, latent = model(torch.randn(12, 32))
    assert reconstruction.shape == (12, 32)
    assert latent.shape == (12, 4)
