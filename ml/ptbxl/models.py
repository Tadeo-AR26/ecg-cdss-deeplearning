import tensorflow as tf
from tensorflow.keras import layers, models
from typing import Tuple, List

from ml.ptbxl.config import INPUT_SHAPE, TARGET_CLASSES


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
    # Shortcut path
    shortcut = layers.MaxPooling1D(
        pool_size=stride, strides=stride, padding="same", name=f"max_pooling1d_{block_idx}"
    )(x)
    shortcut = layers.Conv1D(
        filters=filters, kernel_size=1, strides=1, padding="same", name=f"shortcut_conv1d_{block_idx}"
    )(shortcut)

    # Main path
    y = layers.Conv1D(
        filters=filters, kernel_size=kernel_size, strides=1, padding="same", name=f"res_conv1d_{block_idx}_a"
    )(x)
    y = layers.BatchNormalization(name=f"bn_{block_idx}_a")(y)
    y = layers.ReLU(name=f"relu_{block_idx}_a")(y)
    y = layers.Dropout(dropout_rate, name=f"dropout_{block_idx}_a")(y)

    y = layers.Conv1D(
        filters=filters, kernel_size=kernel_size, strides=stride, padding="same", name=f"res_conv1d_{block_idx}_b"
    )(y)

    # Residual addition
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
    
    Structure:
      - Input: (None, 4096, 12)
      - Initial Conv1D (64 filters, k=16) + BN + ReLU
      - 4 Residual blocks: filters [128, 196, 256, 320], each with stride=4 reduction
      - Final activation feature maps: shape (None, 16, 320)
      - Flatten: strictly 16 * 320 = 5120 features (NO GAP!)
      - Dense: (None, num_classes) with Sigmoid activation
      
    Outputs:
      - If dual_output=False: probabilities (None, num_classes)
      - If dual_output=True: [probabilities, feature_maps] for dual ONNX / 1D CAM
    """
    inputs = layers.Input(shape=input_shape, name="signal")

    # Initial Convolutional Block
    x = layers.Conv1D(filters=64, kernel_size=16, strides=1, padding="same", name="conv1d_init")(inputs)
    x = layers.BatchNormalization(name="bn_init")(x)
    x = layers.ReLU(name="relu_init")(x)

    # 4 Residual Blocks (reducing 4096 -> 1024 -> 256 -> 64 -> 16)
    block_filters = [128, 196, 256, 320]
    for idx, f in enumerate(block_filters, start=1):
        x, last_conv_in_branch = _residual_block(
            x, filters=f, kernel_size=16, stride=4, dropout_rate=dropout_rate, block_idx=idx
        )

    # Feature maps layer for 1D CAM (shape: None, 16, 320)
    feature_maps = layers.Identity(name="feature_maps")(x)

    # Flatten layer (strictly 16 * 320 = 5120 features, as required by Ribeiro CAM formulation)
    x_flat = layers.Flatten(name="flatten")(feature_maps)

    # Multi-label Dense classification layer with Sigmoid
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


def build_dual_export_model(trained_model: tf.keras.Model) -> tf.keras.Model:
    """
    Given a trained single-output PTB-XL model, wraps it with dual output:
      - Output 0: probabilities (None, num_classes)
      - Output 1: feature_maps (None, 16, 320)
    Ready for conversion via tf2onnx to ptbxl_dual.onnx.
    """
    # Locate feature_maps layer or the layer preceding Flatten
    flatten_layer = next(l for l in trained_model.layers if isinstance(l, layers.Flatten))
    feature_maps_tensor = flatten_layer.input

    dual_model = models.Model(
        inputs=trained_model.input,
        outputs=[trained_model.output, feature_maps_tensor],
        name="ptbxl_resnet1d_dual",
    )
    return dual_model
