import os
import sys

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import tensorflow as tf
from ml.ptbxl.config import CHECKPOINTS_DIR, BATCH_SIZE, TARGET_CLASSES
from ml.ptbxl.dataset import PTBXLSequence, get_data_splits
from ml.ptbxl.train import BinaryFocalLoss, find_optimal_thresholds, evaluate_on_test_set

def main():
    model_path = os.path.join(CHECKPOINTS_DIR, "best_model.keras")
    print(f"[*] Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path, compile=False)

    print("[*] Loading PTB-XL splits...")
    splits = get_data_splits(classes=TARGET_CLASSES, load_waveforms=False)
    df_val, y_val_df, _ = splits["val"]
    df_test, y_test_df, _ = splits["test"]

    y_val = y_val_df.values.astype(np.float32)
    y_test = y_test_df.values.astype(np.float32)

    val_gen = PTBXLSequence(df_val, y_val, batch_size=BATCH_SIZE, shuffle=False)
    test_gen = PTBXLSequence(df_test, y_test, batch_size=BATCH_SIZE, shuffle=False)

    print("[*] Generating predictions on Validation set (fold 9)...")
    y_val_probs = model.predict(val_gen, verbose=1)

    optimal_thresholds = find_optimal_thresholds(
        y_true=y_val,
        y_pred_probs=y_val_probs,
        class_names=TARGET_CLASSES,
    )

    print("\n[*] Evaluating on Test set (fold 10) with default threshold 0.5:")
    default_thresholds = {cls: 0.5 for cls in TARGET_CLASSES}
    evaluate_on_test_set(
        model=model,
        test_data=test_gen,
        y_test=y_test,
        class_names=TARGET_CLASSES,
        thresholds=default_thresholds,
        batch_size=BATCH_SIZE,
    )

    print("\n[*] Evaluating on Test set (fold 10) with optimal validation thresholds:")
    evaluate_on_test_set(
        model=model,
        test_data=test_gen,
        y_test=y_test,
        class_names=TARGET_CLASSES,
        thresholds=optimal_thresholds,
        batch_size=BATCH_SIZE,
    )

if __name__ == "__main__":
    main()
