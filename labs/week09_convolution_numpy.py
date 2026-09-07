"""Week 9: transparent NumPy convolution, pooling, and receptive fields."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class ParameterComparison:
    convolution: int
    locally_connected: int
    dense: int

    @property
    def sharing_ratio(self) -> float:
        return self.locally_connected / self.convolution


@dataclass(frozen=True)
class ReceptiveFieldStep:
    name: str
    receptive_field: int
    jump: int


def _pair(value: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, int):
        return value, value
    if len(value) != 2:
        raise ValueError("expected one integer or a pair")
    return int(value[0]), int(value[1])


def output_size(
    length: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
) -> int:
    """Return one spatial output size using the standard floor rule."""
    if min(length, kernel_size, stride, dilation) <= 0 or padding < 0:
        raise ValueError("length, kernel, stride, and dilation must be positive")
    effective_kernel = dilation * (kernel_size - 1) + 1
    numerator = length + 2 * padding - effective_kernel
    if numerator < 0:
        raise ValueError("effective kernel is larger than the padded input")
    return numerator // stride + 1


def cross_correlation2d(
    images: Array,
    kernels: Array,
    bias: Array | None = None,
    *,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
) -> Array:
    """Apply batched multi-channel 2D cross-correlation in NCHW layout.

    Deep-learning libraries commonly call this operation convolution even
    though the learned kernel is not flipped.
    """
    images = np.asarray(images, dtype=float)
    kernels = np.asarray(kernels, dtype=float)
    if images.ndim != 4 or kernels.ndim != 4:
        raise ValueError("images and kernels must use NCHW and OIHW layouts")
    batch, input_channels, height, width = images.shape
    output_channels, kernel_channels, kernel_height, kernel_width = kernels.shape
    if input_channels != kernel_channels:
        raise ValueError("kernel input channels must match image channels")

    stride_height, stride_width = _pair(stride)
    padding_height, padding_width = _pair(padding)
    dilation_height, dilation_width = _pair(dilation)
    output_height = output_size(
        height, kernel_height, stride_height, padding_height, dilation_height
    )
    output_width = output_size(
        width, kernel_width, stride_width, padding_width, dilation_width
    )
    padded = np.pad(
        images,
        (
            (0, 0),
            (0, 0),
            (padding_height, padding_height),
            (padding_width, padding_width),
        ),
    )
    if bias is None:
        bias_array = np.zeros(output_channels, dtype=float)
    else:
        bias_array = np.asarray(bias, dtype=float)
        if bias_array.shape != (output_channels,):
            raise ValueError("bias must contain one value per output channel")

    output = np.empty(
        (batch, output_channels, output_height, output_width), dtype=float
    )
    row_offsets = np.arange(kernel_height) * dilation_height
    column_offsets = np.arange(kernel_width) * dilation_width
    for sample in range(batch):
        for output_channel in range(output_channels):
            for output_row in range(output_height):
                row_start = output_row * stride_height
                rows = row_start + row_offsets
                for output_column in range(output_width):
                    column_start = output_column * stride_width
                    columns = column_start + column_offsets
                    patch = padded[sample][:, rows[:, None], columns[None, :]]
                    output[sample, output_channel, output_row, output_column] = (
                        np.sum(patch * kernels[output_channel])
                        + bias_array[output_channel]
                    )
    return output


def mathematical_convolution2d(
    images: Array,
    kernels: Array,
    bias: Array | None = None,
    **kwargs: object,
) -> Array:
    """Apply mathematical convolution by flipping each spatial kernel."""
    flipped = np.flip(np.asarray(kernels), axis=(-2, -1))
    return cross_correlation2d(images, flipped, bias, **kwargs)


def kernel_gradient2d(
    images: Array,
    upstream: Array,
    kernel_shape: tuple[int, int, int, int],
    *,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
) -> tuple[Array, Array]:
    """Differentiate a cross-correlation objective with respect to kernels/bias."""
    images = np.asarray(images, dtype=float)
    upstream = np.asarray(upstream, dtype=float)
    output_channels, input_channels, kernel_height, kernel_width = kernel_shape
    if images.ndim != 4 or images.shape[1] != input_channels:
        raise ValueError("image channels do not match kernel shape")

    stride_height, stride_width = _pair(stride)
    padding_height, padding_width = _pair(padding)
    dilation_height, dilation_width = _pair(dilation)
    expected_shape = (
        images.shape[0],
        output_channels,
        output_size(
            images.shape[2],
            kernel_height,
            stride_height,
            padding_height,
            dilation_height,
        ),
        output_size(
            images.shape[3],
            kernel_width,
            stride_width,
            padding_width,
            dilation_width,
        ),
    )
    if upstream.shape != expected_shape:
        raise ValueError(f"upstream shape must be {expected_shape}")

    padded = np.pad(
        images,
        (
            (0, 0),
            (0, 0),
            (padding_height, padding_height),
            (padding_width, padding_width),
        ),
    )
    gradient = np.zeros(kernel_shape, dtype=float)
    row_offsets = np.arange(kernel_height) * dilation_height
    column_offsets = np.arange(kernel_width) * dilation_width
    for sample in range(images.shape[0]):
        for output_channel in range(output_channels):
            for output_row in range(expected_shape[2]):
                rows = output_row * stride_height + row_offsets
                for output_column in range(expected_shape[3]):
                    columns = output_column * stride_width + column_offsets
                    patch = padded[sample][
                        :, rows[:, None], columns[None, :]
                    ]
                    gradient[output_channel] += (
                        upstream[
                            sample, output_channel, output_row, output_column
                        ]
                        * patch
                    )
    return gradient, upstream.sum(axis=(0, 2, 3))


def kernel_gradient_check(epsilon: float = 1e-6) -> float:
    """Check the shared-kernel gradient on a deterministic small example."""
    images = np.array(
        [[[[0.2, -0.4, 0.7, 1.0],
           [0.0, 0.5, -0.3, 0.8],
           [1.1, -0.2, 0.4, -0.6],
           [0.3, 0.9, -0.7, 0.2]]]]
    )
    kernels = np.array([[[[0.3, -0.2], [0.5, 0.1]]]])
    upstream = np.array(
        [[[[0.4, -0.1, 0.2],
           [0.0, 0.6, -0.3],
           [0.5, -0.4, 0.1]]]]
    )
    analytical, _ = kernel_gradient2d(images, upstream, kernels.shape)
    numerical = np.zeros_like(kernels)
    for index in np.ndindex(kernels.shape):
        positive = kernels.copy()
        negative = kernels.copy()
        positive[index] += epsilon
        negative[index] -= epsilon
        positive_loss = np.sum(
            cross_correlation2d(images, positive) * upstream
        )
        negative_loss = np.sum(
            cross_correlation2d(images, negative) * upstream
        )
        numerical[index] = (positive_loss - negative_loss) / (2 * epsilon)
    denominator = np.maximum(1.0, np.abs(analytical) + np.abs(numerical))
    return float(np.max(np.abs(analytical - numerical) / denominator))


def max_pool2d(
    images: Array,
    kernel_size: int | tuple[int, int] = 2,
    stride: int | tuple[int, int] | None = None,
) -> Array:
    """Apply non-overlapping or strided maximum pooling to NCHW input."""
    images = np.asarray(images, dtype=float)
    if images.ndim != 4:
        raise ValueError("images must use NCHW layout")
    kernel_height, kernel_width = _pair(kernel_size)
    stride_height, stride_width = _pair(
        kernel_size if stride is None else stride
    )
    output_height = output_size(
        images.shape[2], kernel_height, stride_height
    )
    output_width = output_size(
        images.shape[3], kernel_width, stride_width
    )
    output = np.empty(
        (images.shape[0], images.shape[1], output_height, output_width)
    )
    for row in range(output_height):
        for column in range(output_width):
            patch = images[
                :,
                :,
                row * stride_height : row * stride_height + kernel_height,
                column * stride_width : column * stride_width + kernel_width,
            ]
            output[:, :, row, column] = patch.max(axis=(-2, -1))
    return output


def average_pool2d(
    images: Array,
    kernel_size: int | tuple[int, int] = 2,
    stride: int | tuple[int, int] | None = None,
) -> Array:
    """Apply average pooling to NCHW input."""
    images = np.asarray(images, dtype=float)
    if images.ndim != 4:
        raise ValueError("images must use NCHW layout")
    kernel_height, kernel_width = _pair(kernel_size)
    stride_height, stride_width = _pair(
        kernel_size if stride is None else stride
    )
    output_height = output_size(
        images.shape[2], kernel_height, stride_height
    )
    output_width = output_size(
        images.shape[3], kernel_width, stride_width
    )
    output = np.empty(
        (images.shape[0], images.shape[1], output_height, output_width)
    )
    for row in range(output_height):
        for column in range(output_width):
            patch = images[
                :,
                :,
                row * stride_height : row * stride_height + kernel_height,
                column * stride_width : column * stride_width + kernel_width,
            ]
            output[:, :, row, column] = patch.mean(axis=(-2, -1))
    return output


def parameter_comparison(
    input_shape: tuple[int, int, int],
    output_channels: int,
    kernel_size: int,
    *,
    stride: int = 1,
    padding: int = 0,
) -> ParameterComparison:
    """Compare shared, locally connected, and dense parameter counts."""
    input_channels, height, width = input_shape
    output_height = output_size(height, kernel_size, stride, padding)
    output_width = output_size(width, kernel_size, stride, padding)
    one_filter = input_channels * kernel_size * kernel_size + 1
    convolution = output_channels * one_filter
    locally_connected = (
        output_height * output_width * output_channels * one_filter
    )
    dense = (
        input_channels * height * width + 1
    ) * (output_channels * output_height * output_width)
    return ParameterComparison(convolution, locally_connected, dense)


def receptive_field_schedule(
    layers: Iterable[tuple[str, int, int]],
) -> list[ReceptiveFieldStep]:
    """Track receptive-field size and sampling jump across square layers."""
    receptive_field = 1
    jump = 1
    result: list[ReceptiveFieldStep] = []
    for name, kernel_size, stride in layers:
        if kernel_size <= 0 or stride <= 0:
            raise ValueError("kernel and stride must be positive")
        receptive_field += (kernel_size - 1) * jump
        jump *= stride
        result.append(ReceptiveFieldStep(name, receptive_field, jump))
    return result


def shift_right(images: Array, amount: int = 1) -> Array:
    """Translate NCHW images right with zero fill."""
    images = np.asarray(images)
    if amount < 0 or amount >= images.shape[-1]:
        raise ValueError("amount must be within the image width")
    shifted = np.zeros_like(images)
    if amount == 0:
        shifted[...] = images
    else:
        shifted[..., amount:] = images[..., :-amount]
    return shifted


def interior_equivariance_error(
    images: Array,
    kernels: Array,
    amount: int = 1,
) -> float:
    """Measure translation equivariance away from newly introduced borders."""
    original = cross_correlation2d(images, kernels)
    translated = cross_correlation2d(shift_right(images, amount), kernels)
    if amount >= original.shape[-1]:
        raise ValueError("translation is too large for the output")
    return float(
        np.max(
            np.abs(
                translated[..., amount:]
                - original[..., :-amount]
            )
        )
    )


def teaching_edge_example() -> tuple[Array, Array, Array]:
    """Return a vertical step image, a vertical-edge kernel, and its response."""
    image = np.zeros((1, 1, 7, 7), dtype=float)
    image[0, 0, :, 4:] = 1.0
    kernel = np.array(
        [[[[-1.0, 0.0, 1.0],
           [-1.0, 0.0, 1.0],
           [-1.0, 0.0, 1.0]]]]
    )
    return image, kernel, cross_correlation2d(image, kernel)


def main() -> None:
    patch = np.array(
        [[[[1.0, 2.0, 0.0],
           [0.0, 1.0, 3.0],
           [2.0, 1.0, 0.0]]]]
    )
    asymmetric_kernel = np.array([[[[1.0, 0.0], [-1.0, 2.0]]]])
    correlated = cross_correlation2d(patch, asymmetric_kernel)
    convolved = mathematical_convolution2d(patch, asymmetric_kernel)

    batch = np.zeros((2, 3, 8, 8))
    kernels = np.zeros((4, 3, 3, 3))
    shaped = cross_correlation2d(
        batch, kernels, stride=2, padding=1
    )
    counts = parameter_comparison(
        (3, 8, 8), 4, 3, stride=2, padding=1
    )

    pooling_input = np.array(
        [[[[1.0, 3.0, 2.0, 0.0],
           [4.0, 6.0, 5.0, 1.0],
           [0.0, 2.0, 8.0, 7.0],
           [1.0, 3.0, 9.0, 4.0]]]]
    )
    maximum = max_pool2d(pooling_input)
    average = average_pool2d(pooling_input)
    image, edge_kernel, edge_response = teaching_edge_example()
    schedule = receptive_field_schedule(
        (("conv3", 3, 1), ("pool2", 2, 2), ("conv3", 3, 1))
    )

    print("patch_operation")
    print(f"cross_correlation={correlated[0, 0].astype(int).tolist()}")
    print(f"mathematical_convolution={convolved[0, 0].astype(int).tolist()}")
    print(f"maximum_kernel_gradient_error={kernel_gradient_check():.3e}")

    print("\nshape_and_parameter_evidence")
    print(f"input_shape={batch.shape}")
    print(f"kernel_shape={kernels.shape}")
    print(f"output_shape={shaped.shape}")
    print(f"convolution_parameters={counts.convolution}")
    print(f"locally_connected_parameters={counts.locally_connected}")
    print(f"dense_parameters={counts.dense}")
    print(f"weight_sharing_ratio={counts.sharing_ratio:.1f}")

    print("\npooling")
    print(f"max_pool={maximum[0, 0].tolist()}")
    print(f"average_pool={average[0, 0].tolist()}")

    print("\nspatial_evidence")
    peak = np.argwhere(edge_response[0, 0] == edge_response.max())
    print(f"edge_response_shape={edge_response.shape}")
    print(f"edge_response_max={edge_response.max():.1f}")
    print(f"edge_peak_locations={peak.tolist()}")
    print(
        "interior_translation_equivariance_error="
        f"{interior_equivariance_error(image, edge_kernel):.3e}"
    )
    print(
        "receptive_fields="
        + ", ".join(
            f"{step.name}:{step.receptive_field}"
            f"(jump={step.jump})"
            for step in schedule
        )
    )


if __name__ == "__main__":
    main()
