import numpy as np
import pytest

from labs.week09_convolution_numpy import (
    average_pool2d,
    cross_correlation2d,
    interior_equivariance_error,
    kernel_gradient_check as week09_kernel_gradient_check,
    max_pool2d,
    output_size as week09_output_size,
    parameter_comparison,
    receptive_field_schedule,
    teaching_edge_example,
)

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
from labs.week06_mlp_numpy import (
    classification_accuracy as week06_accuracy,
    gradient_check,
    initialize_parameters,
    train_linear_xor,
    train_xor,
    xor_data,
)
from labs.week07_softmax_numpy import (
    apply_standardizer as week07_apply_standardizer,
    classification_report as week07_classification_report,
    cross_entropy,
    cross_entropy_one_hot,
    fit_standardizer as week07_fit_standardizer,
    gradient_check as week07_gradient_check,
    make_multiclass_data,
    one_hot,
    select_learning_rate as week07_select_learning_rate,
    softmax,
    stratified_split as week07_stratified_split,
)
from labs.week08_dropout_numpy import (
    inverted_dropout,
    relu,
    relu_backward,
    run_controlled_comparison,
    sigmoid_dropout_backward,
)

from labs.week13_reproducibility_audit import (
    ExperimentConfig,
    canonical_json,
    make_research_data,
    produce_bundle,
    run_experiment,
    sha256_bytes,
    stratified_split as week13_stratified_split,
    verify_bundle,
)


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


def test_week06_backpropagation_gradients_and_xor_behavior():
    features, target = xor_data()
    parameters = initialize_parameters()
    maximum_error, per_parameter = gradient_check(
        features, target, parameters
    )
    assert maximum_error < 1e-6
    assert set(per_parameter) == {"w1", "b1", "w2", "b2"}

    linear_prediction, linear_loss = train_linear_xor()
    nonlinear_prediction, nonlinear_loss = train_xor()
    assert week06_accuracy(target, linear_prediction) == 0.5
    assert np.isclose(linear_loss, np.log(2.0))
    assert np.array_equal(
        (nonlinear_prediction.ravel() >= 0.5).astype(int),
        [0, 1, 1, 0],
    )
    assert nonlinear_loss < 0.002


def test_week07_softmax_gradients_selection_and_class_aware_evaluation():
    logits = np.array(
        [[1002.0, 1001.0, 998.0], [-1000.0, -999.0, -1003.0]]
    )
    class_index = np.array([0, 1])
    probability = softmax(logits)
    assert np.isfinite(probability).all()
    assert np.allclose(probability.sum(axis=1), 1.0)
    assert np.allclose(
        probability,
        softmax(logits + np.array([[5000.0], [-3000.0]])),
    )
    assert np.isclose(
        cross_entropy(logits, class_index),
        cross_entropy_one_hot(logits, one_hot(class_index, 3)),
    )

    check_features = np.array(
        [[0.2, -1.0], [1.5, 0.4], [-0.7, 2.0], [0.3, 0.8]]
    )
    check_target = np.array([0, 1, 2, 1])
    check_weights = np.array(
        [[0.2, -0.1, 0.3], [-0.4, 0.5, 0.1]]
    )
    check_bias = np.array([[0.1, -0.2, 0.05]])
    maximum_error, errors = week07_gradient_check(
        check_features, check_target, check_weights, check_bias
    )
    assert maximum_error < 1e-6
    assert set(errors) == {"weights", "bias"}

    features, target = make_multiclass_data()
    split = week07_stratified_split(target)
    assert set(split.train).isdisjoint(split.validation)
    assert set(split.train).isdisjoint(split.test)
    assert set(split.validation).isdisjoint(split.test)
    assert np.array_equal(np.bincount(target[split.test]), [30, 18, 12])

    mean, scale = week07_fit_standardizer(features[split.train])
    standardized = week07_apply_standardizer(features, mean, scale)
    candidates = (0.01, 0.05, 0.2, 0.8)
    selected, models, validation_loss = week07_select_learning_rate(
        standardized[split.train],
        target[split.train],
        standardized[split.validation],
        target[split.validation],
        candidates,
    )
    assert selected == min(candidates, key=validation_loss.__getitem__)
    assert models[selected].loss_history[-1] < models[selected].loss_history[0]

    report = week07_classification_report(
        standardized[split.test], target[split.test], models[selected]
    )
    assert report.accuracy > 0.85
    assert report.macro_recall > 0.85
    assert np.array_equal(report.confusion.sum(axis=1), [30, 18, 12])


def test_week08_dropout_mechanics_and_controlled_generalization_comparison():
    activation = np.full(200_000, 0.8)
    dropped, mask = inverted_dropout(
        activation,
        0.2,
        np.random.default_rng(1),
        training=True,
    )
    evaluation, evaluation_mask = inverted_dropout(
        activation,
        0.2,
        np.random.default_rng(2),
        training=False,
    )
    gradient = sigmoid_dropout_backward(
        activation, np.ones_like(activation), mask
    )
    assert np.isclose(dropped.mean(), activation.mean(), atol=0.005)
    assert np.isclose(gradient.mean(), 0.16, atol=0.005)
    assert np.all(gradient[mask == 0] == 0)
    assert np.array_equal(evaluation, activation)
    assert np.all(evaluation_mask == 1.0)

    pre_activation = np.array([[-1.0, 0.0, 2.0]])
    upstream = np.array([[3.0, 4.0, 5.0]])
    assert np.array_equal(relu(pre_activation), [[0.0, 0.0, 2.0]])
    assert np.array_equal(
        relu_backward(pre_activation, upstream),
        [[0.0, 0.0, 5.0]],
    )

    data, runs, selected, reference_test, selected_test = (
        run_controlled_comparison()
    )
    assert len(data.flipped_training_labels) == 19
    assert selected == min(
        runs,
        key=lambda rate: runs[rate].best_validation_loss,
    )
    assert selected == 0.45
    assert runs[selected].best_validation_loss < (
        runs[0.0].best_validation_loss - 0.10
    )
    assert all(run.best_epoch < run.stopped_epoch for run in runs.values())
    assert selected_test.accuracy > reference_test.accuracy + 0.05
    assert selected_test.negative_recall > 0.60
    assert selected_test.positive_recall > 0.75
    assert selected_test.confusion.sum() == len(data.test_y)

def test_week09_convolution_shapes_sharing_gradients_and_spatial_behavior():
    patch = np.array(
        [[[[1.0, 2.0, 0.0],
           [0.0, 1.0, 3.0],
           [2.0, 1.0, 0.0]]]]
    )
    kernel = np.array([[[[1.0, 0.0], [-1.0, 2.0]]]])
    output = cross_correlation2d(patch, kernel)
    assert np.array_equal(output[0, 0], [[3.0, 7.0], [0.0, 0.0]])
    assert week09_output_size(8, 3, stride=2, padding=1) == 4

    multi_channel = cross_correlation2d(
        np.zeros((2, 3, 8, 8)),
        np.zeros((4, 3, 3, 3)),
        stride=2,
        padding=1,
    )
    assert multi_channel.shape == (2, 4, 4, 4)
    assert week09_kernel_gradient_check() < 1e-6

    pooling_input = np.array(
        [[[[1.0, 3.0, 2.0, 0.0],
           [4.0, 6.0, 5.0, 1.0],
           [0.0, 2.0, 8.0, 7.0],
           [1.0, 3.0, 9.0, 4.0]]]]
    )
    assert np.array_equal(
        max_pool2d(pooling_input)[0, 0],
        [[6.0, 5.0], [3.0, 9.0]],
    )
    assert np.allclose(
        average_pool2d(pooling_input)[0, 0],
        [[3.5, 2.0], [1.5, 7.0]],
    )

    counts = parameter_comparison(
        (3, 8, 8), 4, 3, stride=2, padding=1
    )
    assert counts.convolution == 112
    assert counts.locally_connected == 1792
    assert counts.dense == 12352
    assert counts.sharing_ratio == 16.0

    image, edge_kernel, response = teaching_edge_example()
    assert response.shape == (1, 1, 5, 5)
    assert response.max() == 3.0
    assert interior_equivariance_error(image, edge_kernel) == 0.0

    schedule = receptive_field_schedule(
        (("conv3", 3, 1), ("pool2", 2, 2), ("conv3", 3, 1))
    )
    assert [step.receptive_field for step in schedule] == [3, 4, 8]
    assert [step.jump for step in schedule] == [1, 2, 2]

def test_week13_protocol_is_deterministic_paired_and_split_safe():
    config = ExperimentConfig()
    features, target = make_research_data(config)
    split = week13_stratified_split(target, config)
    _, summary, repeated_split = run_experiment(config)

    assert set(split.train).isdisjoint(split.validation)
    assert set(split.train).isdisjoint(split.test)
    assert set(split.validation).isdisjoint(split.test)
    assert np.array_equal(split.train, repeated_split.train)
    assert np.array_equal(split.validation, repeated_split.validation)
    assert np.array_equal(split.test, repeated_split.test)
    assert summary["run_count"] == len(config.training_seeds)
    assert summary["mean_paired_difference"] > 0.05
    assert summary["paired_95_percent_t_interval"][0] > 0.0
    assert sha256_bytes(canonical_json({"b": 2, "a": [1, 3]})) == (
        sha256_bytes(canonical_json({"a": [1, 3], "b": 2}))
    )


def test_week13_bundle_round_trip_and_tamper_detection(tmp_path):
    artifact_dir = tmp_path / "week13-bundle"
    config = ExperimentConfig(
        sample_count=300,
        epochs=20,
        training_seeds=(13, 29),
    )
    produce_bundle(artifact_dir, config)
    result = verify_bundle(artifact_dir)

    assert result["status"] == "PASS"
    assert result["files_verified"] == 4
    assert result["source_verified"]
    assert result["split_disjoint"]

    summary_path = artifact_dir / "summary.json"
    summary_path.write_bytes(summary_path.read_bytes() + b"\n")
    with pytest.raises(RuntimeError, match="Artifact hash mismatch"):
        verify_bundle(artifact_dir)
