import argparse
import os
import sys
import numpy as np
import tensorflow as tf

# Configure UTF-8 stdout for Windows
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from ml.ptbxl.config import (
    CHECKPOINTS_DIR,
    RUNS_DIR,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    RANDOM_SEED,
    TARGET_CLASSES,
)
from ml.ptbxl.dataset import (
    get_data_splits,
    compute_class_weights,
    apply_augmentations,
)
from ml.ptbxl.models import build_ribeiro_resnet1d


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Sets random seeds for reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PTB-XL Primary ECG Model")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument("--dry-run", action="store_true", help="Quick test run with 2 epochs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed()

    epochs = 2 if args.dry_run else args.epochs
    batch_size = args.batch_size
    lr = args.lr

    print("=" * 60)
    print("  PTB-XL PRIMARY MODEL — TRAINING SETUP")
    print("=" * 60)
    print(f"  Target Classes ({len(TARGET_CLASSES)}): {TARGET_CLASSES}")
    print(f"  Batch size   : {batch_size}")
    print(f"  Epochs       : {epochs}")
    print(f"  Learning rate: {lr}")
    print(f"  Checkpoints  : {CHECKPOINTS_DIR}")
    print(f"  Runs/Logs    : {RUNS_DIR}")
    print("=" * 60)

    # 1. Load Dataset metadata & splits
    print("\n[*] Loading dataset splits...")
    splits = get_data_splits(classes=TARGET_CLASSES, load_waveforms=True)

    df_train, y_train_df, X_train = splits["train"]
    df_val, y_val_df, X_val = splits["val"]
    df_test, y_test_df, X_test = splits["test"]

    y_train = y_train_df.values
    y_val = y_val_df.values
    y_test = y_test_df.values

    print(f"  Train: X={X_train.shape}, y={y_train.shape}")
    print(f"  Val  : X={X_val.shape}, y={y_val.shape}")
    print(f"  Test : X={X_test.shape}, y={y_test.shape}")

    # Inspect class distributions
    print("\n[*] Positive cases in Training set:")
    for cls_name, count in zip(TARGET_CLASSES, np.sum(y_train, axis=0)):
        pct = (count / len(y_train)) * 100
        print(f"  - {cls_name:<8}: {int(count):>5} samples ({pct:5.2f}%)")

    # 2. Class weights for addressing imbalance (e.g., WPW ~0.6%)
    pos_weights = compute_class_weights(y_train)
    print(f"\n[*] Computed pos_weights: {dict(zip(TARGET_CLASSES, np.round(pos_weights, 2)))}")

    # 3. Build Model
    print("\n[*] Building model...")
    model = build_ribeiro_resnet1d(num_classes=len(TARGET_CLASSES))
    model.summary()

    # ============================================================
    # TODO FOR USER:
    # 1. Customize your Loss Function:
    #    Standard binary_crossentropy works as baseline, but for
    #    severe imbalance (WPW, LNGQT), consider:
    #      - Weighted Binary Crossentropy using `pos_weights`
    #      - Focal Loss
    # ============================================================
    loss_fn = tf.keras.losses.BinaryCrossentropy()

    # 2. Customize Optimizer & Metrics:
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    metrics = [
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.AUC(name="auc", multi_label=True),
    ]

    model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

    # 4. Callbacks setup
    checkpoint_path = os.path.join(CHECKPOINTS_DIR, "best_model.keras")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=RUNS_DIR,
            histogram_freq=1,
        ),
    ]

    # ============================================================
    # TODO FOR USER:
    # Execute training (model.fit) and add post-training evaluation:
    #   - Threshold sweep per class on validation set (especially WPW)
    #   - Classification report (F1-score per class, AUROC)
    # ============================================================
    print("\n[*] Starting training...")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    print("\n[+] Training completed.")
    print(f"[+] Best model checkpoint saved to: {checkpoint_path}")

    # TODO: Add your post-training evaluation / threshold tuning here


if __name__ == "__main__":
    main()
