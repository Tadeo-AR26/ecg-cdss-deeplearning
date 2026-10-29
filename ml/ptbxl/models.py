import tensorflow as tf
from tensorflow.keras import layers, models
from typing import Tuple, List

from ml.ptbxl.config import INPUT_SHAPE, TARGET_CLASSES


# ============================================================
# 1. INCEPTION-1D ARCHITECTURE (PTB-XL Benchmark / InceptionTime)
# ============================================================

def _inception_module(
    x: tf.Tensor,
    filters: int = 32,
    kernel_sizes: List[int] = [9, 19, 39],
    bottleneck_size: int = 32,
    stride: int = 1,
    module_idx: int = 1,
) -> tf.Tensor:
    """
    Standard InceptionTime module (Fawaz et al., 2019 / Strodthoff et al., 2020):
    - Bottleneck Conv1D(1x1) if input channels > bottleneck_size
    - 3 parallel multi-scale Conv1D branches (kernels 9, 19, 39)
    - 1 MaxPooling1D branch with Conv1D(1x1)
    - Concatenation along channel axis -> BatchNormalization -> ReLU
    """
    in_channels = x.shape[-1]

    # 1. Bottleneck layer to reduce channel dimensionality
    if in_channels is not None and in_channels > 1 and in_channels > bottleneck_size:
        x_bottleneck = layers.Conv1D(
            filters=bottleneck_size,
            kernel_size=1,
            padding="same",
            use_bias=False,
            name=f"inc_{module_idx}_bottleneck",
        )(x)
    else:
        x_bottleneck = x

    # 2. Parallel multi-scale Conv1D branches
    conv_branches = []
    for k_size in kernel_sizes:
        c = layers.Conv1D(
            filters=filters,
            kernel_size=k_size,
            strides=stride,
            padding="same",
            use_bias=False,
            name=f"inc_{module_idx}_conv_k{k_size}",
        )(x_bottleneck)
        conv_branches.append(c)

    # 3. MaxPooling branch with stride
    pool = layers.MaxPooling1D(
        pool_size=3,
        strides=stride,
        padding="same",
        name=f"inc_{module_idx}_maxpool",
    )(x)
    pool_conv = layers.Conv1D(
        filters=filters,
        kernel_size=1,
        padding="same",
        use_bias=False,
        name=f"inc_{module_idx}_pool_conv",
    )(pool)
    conv_branches.append(pool_conv)

    # 4. Concatenate all 4 branches (filters * 4 = 128 channels)
    out = layers.Concatenate(axis=-1, name=f"inc_{module_idx}_concat")(conv_branches)
    out = layers.BatchNormalization(name=f"inc_{module_idx}_bn")(out)
    out = layers.ReLU(name=f"inc_{module_idx}_relu")(out)
    return out


def _inception_block(
    x: tf.Tensor,
    num_modules: int = 3,
    filters: int = 32,
    kernel_sizes: List[int] = [9, 19, 39],
    bottleneck_size: int = 32,
    stride: int = 2,
    block_idx: int = 1,
) -> tf.Tensor:
    """
    Inception Block composed of `num_modules` Inception modules with a
    residual shortcut connection from input to output.
    Applies `stride` on the first module to scale temporal dimension smoothly.
    """
    shortcut = x
    start_idx = (block_idx - 1) * num_modules + 1

    for m in range(num_modules):
        mod_idx = start_idx + m
        # Only the first module of the block applies the downsampling stride
        mod_stride = stride if m == 0 else 1
        x = _inception_module(
            x,
            filters=filters,
            kernel_sizes=kernel_sizes,
            bottleneck_size=bottleneck_size,
            stride=mod_stride,
            module_idx=mod_idx,
        )

    out_channels = x.shape[-1]
    # Match shortcut channels and temporal stride
    if shortcut.shape[-1] != out_channels or stride != 1:
        shortcut = layers.Conv1D(
            filters=out_channels,
            kernel_size=1,
            strides=stride,
            padding="same",
            use_bias=False,
            name=f"inc_block_{block_idx}_shortcut_conv",
        )(shortcut)
        shortcut = layers.BatchNormalization(name=f"inc_block_{block_idx}_shortcut_bn")(shortcut)

    x = layers.Add(name=f"inc_block_{block_idx}_add")([shortcut, x])
    x = layers.ReLU(name=f"inc_block_{block_idx}_relu")(x)
    return x


def build_inception1d(
    input_shape: Tuple[int, int] = INPUT_SHAPE,
    num_classes: int = len(TARGET_CLASSES),
    num_blocks: int = 2,
    modules_per_block: int = 3,
    filters: int = 32,
    kernel_sizes: List[int] = [9, 19, 39],
    bottleneck_size: int = 32,
    block_stride: int = 2,
    dropout_rate: float = 0.2,
    dual_output: bool = False,
) -> tf.keras.Model:
    """
    Constructs the Inception-1D (InceptionTime) architecture,
    the top-performing benchmark architecture on the PTB-XL dataset
    (Strodthoff et al., 2020 / Fawaz et al., 2019).

    Temporal progression:
      - Input: (None, 4096, 12)
      - Block 1 (3 modules, stride=2): -> (None, 2048, 128)
      - Block 2 (3 modules, stride=2): -> (None, 1024, 128)
      - feature_maps (CAM-1D): shape (None, 1024, 128) -> 1024 points (10ms resolution, 64x sharper than Ribeiro)
      - GAP: shape (None, 128)
      - Output: shape (None, num_classes)

    Key architectural benefits:
      - Multi-scale receptive fields (kernels 9, 19, 39) capture both high-frequency
        sharp morphology (WPW delta waves, QRS notches) and low-frequency intervals
        (QT prolongation in LNGQT, ST changes in MI/STTC).
      - Highly memory-efficient: tensors are only ~16MB at batch=32 (preventing system memory thrashing).
      - ~450k parameters vs 6.4M in Ribeiro, dramatically reducing overfitting.
      - GlobalAveragePooling1D provides native, full-resolution Class Activation Maps (1D CAM).

    Outputs:
      - If dual_output=False: probabilities (None, num_classes)
      - If dual_output=True: [probabilities, feature_maps] for dual ONNX / 1D CAM
    """
    inputs = layers.Input(shape=input_shape, name="signal")
    x = inputs

    # 2 Inception blocks x 3 modules = 6 Inception modules total
    # Temporal reduction: 4096 -> 2048 -> 1024
    for b in range(1, num_blocks + 1):
        x = _inception_block(
            x,
            num_modules=modules_per_block,
            filters=filters,
            kernel_sizes=kernel_sizes,
            bottleneck_size=bottleneck_size,
            stride=block_stride,
            block_idx=b,
        )

    # Feature maps layer for 1D CAM (shape: None, 1024, 128)
    feature_maps = layers.Identity(name="feature_maps")(x)

    # Global Average Pooling along the temporal dimension -> (None, 128)
    gap = layers.GlobalAveragePooling1D(name="gap")(feature_maps)

    if dropout_rate > 0.0:
        gap = layers.Dropout(dropout_rate, name="head_dropout")(gap)

    # Multi-label classification head with Sigmoid activation
    probabilities = layers.Dense(
        num_classes, activation="sigmoid", name="probabilities"
    )(gap)

    if dual_output:
        model = models.Model(
            inputs=inputs,
            outputs=[probabilities, feature_maps],
            name="ptbxl_inception1d_dual",
        )
    else:
        model = models.Model(
            inputs=inputs,
            outputs=probabilities,
            name="ptbxl_inception1d_primary",
        )

    return model


# ============================================================
# 2. RIBEIRO ET AL. RESNET-1D (Original Reference Architecture)
# ============================================================

def _residual_block(
    x: tf.Tensor,
    filters: int,
    kernel_size: int = 16,
    stride: int = 4,
    dropout_rate: float = 0.8,
    block_idx: int = 1,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Ribeiro ResNet-1D residual block:
    - Main path: Conv1D(s=1) -> BN -> ReLU -> Dropout -> Conv1D(s=stride)
    - Shortcut path: MaxPooling1D(s=stride) -> Conv1D(1x1, s=1)
    - Merged with Add() -> BN -> ReLU -> Dropout
    """
    shortcut = layers.MaxPooling1D(
        pool_size=stride, strides=stride, padding="same", name=f"max_pooling1d_{block_idx}"
    )(x)
    shortcut = layers.Conv1D(
        filters=filters, kernel_size=1, strides=1, padding="same", name=f"shortcut_conv1d_{block_idx}"
    )(shortcut)

    y = layers.Conv1D(
        filters=filters, kernel_size=kernel_size, strides=1, padding="same", name=f"res_conv1d_{block_idx}_a"
    )(x)
    y = layers.BatchNormalization(name=f"bn_{block_idx}_a")(y)
    y = layers.ReLU(name=f"relu_{block_idx}_a")(y)
    y = layers.Dropout(dropout_rate, name=f"dropout_{block_idx}_a")(y)

    y = layers.Conv1D(
        filters=filters, kernel_size=kernel_size, strides=stride, padding="same", name=f"res_conv1d_{block_idx}_b"
    )(y)

    out = layers.Add(name=f"add_{block_idx}")([y, shortcut])
    out = layers.BatchNormalization(name=f"bn_{block_idx}_b")(out)
    out = layers.ReLU(name=f"relu_{block_idx}_b")(out)
    out = layers.Dropout(dropout_rate, name=f"dropout_{block_idx}_b")(out)

    return out, y


def build_ribeiro_resnet1d(
    input_shape: Tuple[int, int] = INPUT_SHAPE,
    num_classes: int = len(TARGET_CLASSES),
    dropout_rate: float = 0.8,
    dual_output: bool = False,
) -> tf.keras.Model:
    """
    Constructs the exact Ribeiro et al. ResNet-1D architecture.
    """
    inputs = layers.Input(shape=input_shape, name="signal")

    x = layers.Conv1D(filters=64, kernel_size=16, strides=1, padding="same", name="conv1d_init")(inputs)
    x = layers.BatchNormalization(name="bn_init")(x)
    x = layers.ReLU(name="relu_init")(x)

    block_filters = [128, 196, 256, 320]
    for idx, f in enumerate(block_filters, start=1):
        x, _ = _residual_block(
            x, filters=f, kernel_size=16, stride=4, dropout_rate=dropout_rate, block_idx=idx
        )

    feature_maps = layers.Identity(name="feature_maps")(x)
    x_flat = layers.Flatten(name="flatten")(feature_maps)

    probabilities = layers.Dense(
        num_classes, activation="sigmoid", name="probabilities"
    )(x_flat)

    if dual_output:
        model = models.Model(
            inputs=inputs,
            outputs=[probabilities, feature_maps],
            name="ptbxl_resnet1d_dual",
        )
    else:
        model = models.Model(
            inputs=inputs,
            outputs=probabilities,
            name="ptbxl_resnet1d_primary",
        )

    return model


# ============================================================
# 3. DUAL EXPORT HELPER (Supports both Inception-1D and ResNet-1D)
# ============================================================

def build_dual_export_model(trained_model: tf.keras.Model) -> tf.keras.Model:
    """
    Given a trained single-output PTB-XL model (Inception-1D or ResNet-1D),
    wraps it with dual output:
      - Output 0: probabilities (None, num_classes)
      - Output 1: feature_maps (None, time_steps, channels)
    Ready for conversion via tf2onnx to ptbxl_dual.onnx.
    """
    feature_maps_layer = None
    for layer in trained_model.layers:
        if layer.name == "feature_maps":
            feature_maps_layer = layer
            break

    if feature_maps_layer is not None:
        feature_maps_tensor = feature_maps_layer.output
    else:
        pool_layer = next(
            (l for l in trained_model.layers if isinstance(l, (layers.GlobalAveragePooling1D, layers.Flatten))),
            None,
        )
        if pool_layer is not None:
            feature_maps_tensor = pool_layer.input
        else:
            raise ValueError("Could not find feature_maps or pooling layer in model.")

    dual_model = models.Model(
        inputs=trained_model.input,
        outputs=[trained_model.output, feature_maps_tensor],
        name="ptbxl_dual",
    )
    return dual_model
