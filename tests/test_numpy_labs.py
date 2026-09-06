import numpy as np

from labs.week01_python_numpy_diagnostic import (
    accuracy,
    inspection_rule,
    make_machine_data,
    majority_baseline,
)
from labs.week02_evaluation_numpy import (
    evaluate_split,
    grouped_split,
    make_repeated_measurement_data,
    overlapping_group_count,
    random_row_split,
)
from labs.week03_data_quality_numpy import (
    apply_median_imputer,
    evaluate_logistic_candidate,
    fit_median_imputer,
    make_rare_event_data,
    majority_baseline as week03_majority_baseline,
    stratified_split,
)
from labs.week04_forward_pass_numpy import (
    affine,
    collapse_linear_layers,
    forward_two_layer,
    neuron_output,
    parameter_count,
    sigmoid,
    teaching_parameters,
)
from labs.week05_gradient_descent import (
    apply_standardizer,
    fit_standardizer,
    iter_minibatches,
    linear_gradients,
    make_regression_data,
    mean_squared_error,
    select_learning_rate,
    split_indices,
)
from labs.week06_mlp_numpy import train_xor
from labs.week07_softmax_numpy import cross_entropy, softmax
from labs.week08_dropout_numpy import inverted_dropout, sigmoid_dropout_backward


def test_week01_rule_beats_the_baseline_on_its_teaching_data():
    features, target = make_machine_data(seed=705)
    repeated_features, repeated_target = make_machine_data(seed=705)
    _, prediction = inspection_rule(features)
    baseline = majority_baseline(target)

    assert features.shape == (160, 3)
    assert target.shape == (160,)
    assert np.array_equal(features, repeated_features)
    assert np.array_equal(target, repeated_target)
    assert accuracy(target, prediction) > accuracy(target, baseline) + 0.15


def test_grouped_split_keeps_subjects_disjoint():
    groups = np.repeat(np.arange(30), 3)
    train, validation, test = grouped_split(groups, seed=1)
    partitions = [set(groups[index]) for index in (train, validation, test)]
    assert all(partitions)
    assert partitions[0].isdisjoint(partitions[1])
    assert partitions[0].isdisjoint(partitions[2])
    assert partitions[1].isdisjoint(partitions[2])


def test_week02_row_split_exposes_group_leakage_in_teaching_data():
    features, target, groups = make_repeated_measurement_data(seed=705)
    row_split = random_row_split(len(target), seed=705)
    group_split = grouped_split(groups, seed=705)

    row_score = evaluate_split(features, target, *row_split)[1]
    group_score = evaluate_split(features, target, *group_split)[1]

    assert overlapping_group_count(groups, row_split[0], row_split[2]) > 0
    assert overlapping_group_count(groups, group_split[0], group_split[2]) == 0
    assert row_score > group_score + 0.25


def test_week03_pipeline_exposes_imbalance_and_compares_one_feature_change():
    features, target = make_rare_event_data(seed=705)
    split = stratified_split(target, seed=705)
    train, validation, test = split

    assert set(train).isdisjoint(validation)
    assert set(train).isdisjoint(test)
    assert set(validation).isdisjoint(test)
    assert len(np.concatenate(split)) == len(target)
    assert all(abs(target[index].mean() - target.mean()) < 0.01 for index in split)

    medians = fit_median_imputer(features[train])
    assert np.allclose(medians, np.nanmedian(features[train], axis=0))
    assert not np.isnan(apply_median_imputer(features[test], medians)).any()

    baseline = week03_majority_baseline(target[train], target[test])
    imputed = evaluate_logistic_candidate(
        features,
        target,
        split,
        use_missing_indicator=False,
    )
    with_indicator = evaluate_logistic_candidate(
        features,
        target,
        split,
        use_missing_indicator=True,
    )

    assert baseline.recall == 0.0
    assert baseline.accuracy > with_indicator.accuracy
    assert with_indicator.balanced_accuracy > baseline.balanced_accuracy + 0.25
    assert with_indicator.balanced_accuracy > imputed.balanced_accuracy + 0.05


def test_week04_forward_pass_shapes_probabilities_and_affine_collapse():
    example_x = np.array([2.0, -1.0, 0.5])
    example_w = np.array([0.4, -0.8, 1.2])
    pre_activation, activated = neuron_output(
        example_x,
        example_w,
        -0.3,
        sigmoid,
    )
    assert np.isclose(pre_activation, 1.9)
    assert np.isclose(activated, 0.8698915256)

    features, w1, b1, w2, b2 = teaching_parameters()
    hidden_pre, hidden, logits, probability = forward_two_layer(
        features,
        w1,
        b1,
        w2,
        b2,
    )
    assert hidden_pre.shape == hidden.shape == (4, 4)
    assert logits.shape == probability.shape == (4, 2)
    assert np.isfinite(probability).all()
    assert (probability >= 0).all()
    assert np.allclose(probability.sum(axis=1), 1.0)
    assert parameter_count([w1, w2], [b1, b2]) == 26

    equivalent_weight, equivalent_bias = collapse_linear_layers(w1, b1, w2, b2)
    stacked = affine(affine(features, w1, b1), w2, b2)
    collapsed = affine(features, equivalent_weight, equivalent_bias)
    assert np.max(np.abs(stacked - collapsed)) < 1e-12


def test_minibatches_cover_each_example_once_for_divisible_and_remainder_sizes():
    for size in (64, 65):
        x = np.arange(size).reshape(-1, 1)
        y = np.arange(size)
        batches = list(iter_minibatches(x, y, 32, np.random.default_rng(0)))
        indices = np.concatenate([index for _, _, index in batches])
        assert all(len(xb) > 0 for xb, _, _ in batches)
        assert sorted(indices.tolist()) == list(range(size))


def test_week05_analytical_gradients_match_finite_differences():
    x = np.array([[0.2, -1.0], [1.5, 0.4], [-0.7, 2.0]])
    y = np.array([0.5, 1.4, -0.8])
    weights = np.array([0.3, -0.2])
    bias = 0.1
    gradient_weights, gradient_bias = linear_gradients(x, y, weights, bias)
    epsilon = 1e-6

    numerical_weights = np.empty_like(weights)
    for index in range(len(weights)):
        plus = weights.copy()
        minus = weights.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        numerical_weights[index] = (
            mean_squared_error(x, y, plus, bias)
            - mean_squared_error(x, y, minus, bias)
        ) / (2 * epsilon)
    numerical_bias = (
        mean_squared_error(x, y, weights, bias + epsilon)
        - mean_squared_error(x, y, weights, bias - epsilon)
    ) / (2 * epsilon)

    assert np.allclose(gradient_weights, numerical_weights, atol=1e-6)
    assert np.isclose(gradient_bias, numerical_bias, atol=1e-6)


def test_week05_validation_selects_a_stable_learning_rate_before_test_use():
    features, target = make_regression_data()
    train, validation, test = split_indices(len(target))
    assert set(train).isdisjoint(validation)
    assert set(train).isdisjoint(test)
    assert set(validation).isdisjoint(test)

    mean, scale = fit_standardizer(features[train])
    x_train = apply_standardizer(features[train], mean, scale)
    x_validation = apply_standardizer(features[validation], mean, scale)
    x_test = apply_standardizer(features[test], mean, scale)
    selected, validation_loss, results = select_learning_rate(
        x_train,
        target[train],
        x_validation,
        target[validation],
        (0.001, 0.01, 0.05, 0.2, 1.2),
    )

    assert results[0.05].train_loss[-1] < results[0.05].train_loss[0] * 0.02
    assert results[1.2].diverged or validation_loss[1.2] > validation_loss[0.05] * 10
    assert selected in (0.01, 0.05, 0.2)
    assert mean_squared_error(
        x_test,
        target[test],
        results[selected].weights,
        results[selected].bias,
    ) < 0.12


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
