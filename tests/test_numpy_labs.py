import numpy as np

from labs.week02_evaluation_numpy import grouped_split
from labs.week05_gradient_descent import iter_minibatches
from labs.week06_mlp_numpy import train_xor
from labs.week07_softmax_numpy import cross_entropy, softmax
from labs.week08_dropout_numpy import inverted_dropout, sigmoid_dropout_backward



def test_grouped_split_keeps_subjects_disjoint():
    groups = np.repeat(np.arange(30), 3)
    train, validation, test = grouped_split(groups, seed=1)
    partitions = [set(groups[index]) for index in (train, validation, test)]
    assert all(partitions)
    assert partitions[0].isdisjoint(partitions[1])
    assert partitions[0].isdisjoint(partitions[2])
    assert partitions[1].isdisjoint(partitions[2])


def test_minibatches_cover_each_example_once_for_divisible_and_remainder_sizes():
    for size in (64, 65):
        x = np.arange(size).reshape(-1, 1)
        y = np.arange(size)
        batches = list(iter_minibatches(x, y, 32, np.random.default_rng(0)))
        indices = np.concatenate([index for _, _, index in batches])
        assert all(len(xb) > 0 for xb, _, _ in batches)
        assert sorted(indices.tolist()) == list(range(size))


def test_numpy_network_learns_xor():
    prediction, loss = train_xor()
    assert np.array_equal((prediction.ravel() >= 0.5).astype(int), [0, 1, 1, 0])
    assert loss < 0.02



def test_softmax_is_stable_and_normalized():
    logits = np.array([[1002.0, 1001.0, 998.0], [-1000.0, -999.0, -1003.0]])
    probability = softmax(logits)
    assert np.isfinite(probability).all()
    assert np.allclose(probability.sum(axis=1), 1.0)
    assert cross_entropy(logits, np.array([0, 1])) > 0


def test_inverted_dropout_preserves_expectation_and_backward_mask():
    activation = np.full(200_000, 0.8)
    dropped, mask = inverted_dropout(activation, 0.2, np.random.default_rng(1), training=True)
    gradient = sigmoid_dropout_backward(activation, np.ones_like(activation), mask)
    assert np.isclose(dropped.mean(), activation.mean(), atol=0.005)
    assert np.isclose(gradient.mean(), 0.16, atol=0.005)
    assert np.all(gradient[mask == 0] == 0)
