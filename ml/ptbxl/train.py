import argparse
import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    precision_recall_fscore_support,
    average_precision_score,
    multilabel_confusion_matrix,
)

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

class BinaryFocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25, name="binary_focal_loss"):
        super().__init__(name=name)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        # Avoid numeric instability with log(0) and log(1)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

        # p_t: probability assigned to the ground-truth class
        # For positives:  p_t = y_pred
        # For negatives:  p_t = 1 - y_pred  (NOT epsilon — that was a bug)
        p_t = tf.where(tf.equal(y_true, 1.0), y_pred, 1.0 - y_pred)

        # Alpha balance weight (alpha for positives, 1-alpha for negatives)
        alpha_factor = tf.where(tf.equal(y_true, 1.0), self.alpha, 1.0 - self.alpha)

        # Focal modulation term: (1 - p_t)^gamma down-weights easy examples
        focal_weight = alpha_factor * tf.pow(1.0 - p_t, self.gamma)

        # Standard Binary Cross-Entropy over the ground-truth class probability
        bce = -tf.math.log(p_t)
        loss = focal_weight * bce

        return tf.reduce_mean(loss)


def find_optimal_thresholds(y_true, y_pred_probs, class_names, grid_size=91):
    thresholds = np.linspace(0.05, 0.95, grid_size)
    optimal_thresholds = {}

    print("\n" + "=" * 65)
    print("  OPTIMAL THRESHOLD SWEEP (VAL SET - F1-MAXIMIZATION)")
    print("=" * 65)
    print(f"  {'Class':<10} | {'Opt Thresh':<10} | {'Val F1':<8} | {'Precision':<10} | {'Recall':<8}")
    print("  " + "-" * 61)

    for i, cls_name in enumerate(class_names):
        y_t = y_true[:, i]
        y_p = y_pred_probs[:, i]

        best_thresh = 0.5
        best_f1 = -1.0
        best_prec = 0.0
        best_rec = 0.0

        for thresh in thresholds:
            y_pred_bin = (y_p >= thresh).astype(int)
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_t, y_pred_bin, average='binary', zero_division=0
            )
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
                best_prec = prec
                best_rec = rec
        
        optimal_thresholds[cls_name] = float(best_thresh)
        print(f"  {cls_name:<10} | {best_thresh:<10.2f} | {best_f1:<8.4f} | {best_prec:<10.4f} | {best_rec:<8.4f}")

    print("=" * 65)
    return optimal_thresholds

def evaluate_on_test_set(model, X_test, y_test, class_names, thresholds, batch_size=BATCH_SIZE):
    print("\n[*] Predicting on Test set...")
    y_test_probs = model.predict(X_test, batch_size=batch_size, verbose=0)

    y_test_pred_bin = np.zeros_like(y_test_probs)
    for i, cls_name in enumerate(class_names):
        thresh = thresholds[cls_name]
        y_test_pred_bin[:, i] = (y_test_probs[:, i] >= thresh).astype(int)
    
    print("\n" + "=" * 65)
    print("  FINAL TEST SET EVALUATION (OPTIMIZED THRESHOLDS)")
    print("=" * 65)
    print(f"  {'Class':<10} | {'ROC-AUC':<8} | {'PR-AUC':<8} | {'Precision':<10} | {'Recall':<8} | {'F1-Score':<8}")
    print("  " + "-" * 63)

    for i, cls_name in enumerate(class_names):
        y_t = y_test[:, i]
        y_p = y_test_probs[:, i]
        y_b = y_test_pred_bin[:, i]

        roc_auc = roc_auc_score(y_t, y_p) if np.sum(y_t) > 0 else 0.0
        pr_auc = average_precision_score(y_t, y_p) if np.sum(y_t) > 0 else 0.0
        prec, rec, f1, _ = precision_recall_fscore_support(y_t, y_b, average='binary', zero_division=0)
        print(f"  {cls_name:<10} | {roc_auc:<8.4f} | {pr_auc:<8.4f} | {prec:<10.4f} | {rec:<8.4f} | {f1:<8.4f}")
    
    print("=" * 65)

    print("\n[*] Detailed Classification Report (Test Set):")
    print(classification_report(y_test, y_test_pred_bin, target_names=class_names, zero_division=0))

    # Per-class confusion matrix: multilabel_confusion_matrix returns shape (n_classes, 2, 2)
    # Layout per class:
    #   [[TN  FP]
    #    [FN  TP]]
    # Key metrics to look for:
    #   - TP (True Positives):  correctly flagged as positive — want HIGH, especially for WPW/LNGQT
    #   - FN (False Negatives): missed positives (dangerous in clinical context) — want LOW
    #   - FP (False Positives): false alarms — acceptable trade-off to reduce FN
    #   - TN (True Negatives):  correctly rejected — usually high due to class imbalance
    cm_all = multilabel_confusion_matrix(y_test, y_test_pred_bin)

    print("\n" + "=" * 65)
    print("  PER-CLASS CONFUSION MATRIX (Test Set)")
    print("=" * 65)
    for i, cls_name in enumerate(class_names):
        tn, fp, fn, tp = cm_all[i].ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Recall / True Positive Rate
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # True Negative Rate
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0           # Precision / Positive Predictive Value
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0           # Negative Predictive Value
        print(f"  [{cls_name}]")
        print(f"    Matrix:  [[TN={tn:4d}  FP={fp:4d}]]")
        print(f"             [[FN={fn:4d}  TP={tp:4d}]]")
        print(f"    Sensitivity (Recall): {sensitivity:.4f}  |  Specificity: {specificity:.4f}")
        print(f"    PPV (Precision):       {ppv:.4f}  |  NPV:         {npv:.4f}")
        print()

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

    # Standard baseline (no imbalance correction):
    # loss_fn = tf.keras.losses.BinaryCrossentropy()

    # Focal Loss — recommended for severe imbalance (WPW pos_weight ~271, LNGQT ~184)
    loss_fn = BinaryFocalLoss(gamma=2.0, alpha=0.25)

    # Alternative: weighted BCE with per-class pos_weight tensor
    # loss_fn = WeightedBCEWithSigmoid(pos_weights=pos_weights, max_clip=30.0)

    # 5. Configure Optimizer & Metrics:
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    metrics = [
        tf.keras.metrics.BinaryAccuracy(name="accuracy", threshold=0.5),
        tf.keras.metrics.AUC(name="auc", multi_label=True),
        tf.keras.metrics.AUC(name="pr_auc", curve="PR", multi_label=True),
    ]

    model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

    # 6. Callbacks setup
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

    # 7. Post-training evaluation with per-class threshold calibration

    # Load the best checkpoint saved by ModelCheckpoint
    print("\n[*] Loading best model weights for evaluation...")
    best_model = tf.keras.models.load_model(
        checkpoint_path,
        custom_objects={'BinaryFocalLoss': BinaryFocalLoss},
    )

    # Predict on Validation set to find optimal decision thresholds per class
    print("[*] Generating predictions on Validation set...")
    y_val_probs = best_model.predict(X_val, batch_size=batch_size, verbose=0)

    optimal_thresholds = find_optimal_thresholds(
        y_true=y_val,
        y_pred_probs=y_val_probs,
        class_names=TARGET_CLASSES,
    )

    # Final evaluation on Test set using calibrated per-class thresholds
    evaluate_on_test_set(
        model=best_model,
        X_test=X_test,
        y_test=y_test,
        class_names=TARGET_CLASSES,
        thresholds=optimal_thresholds,
        batch_size=batch_size,
    )


if __name__ == "__main__":
    main()
