"""
ml/scripts/evaluate_ptbxl_mimic_gradcam.py
==========================================
Stage 2 — Step 3: Comparative Evaluation, Threshold Calibration & 1D Grad-CAM

Loads the best Stage 2 fine-tuned checkpoint and evaluates it on:
  1. PTB-XL Official Test Set (strat_fold == 10) — before/after comparison
  2. MIMIC External Hold-out (X_mimic_test) — inter-hospital generalization
  3. Finds optimal per-class F1-maximizing thresholds on PTB-XL val set
  4. Generates 1D Grad-CAM visualizations for WPW and LNGQT positive cases

For 1D Grad-CAM with Inception-1D (GAP-based):
  - Forward: extracts feature_maps (batch, 1024, 128)
  - Backward: computes gradient of class logit w.r.t. feature_maps
  - CAM = ReLU(sum(grad * activation, axis=-1)) → (1024,)
  - Interpolated to 4096 signal points

Usage:
    python ml/scripts/evaluate_ptbxl_mimic_gradcam.py [--stage2-checkpoint PATH] [--dry-run]
"""

import argparse
import os
import sys

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/Windows compatibility
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import interp1d
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
    multilabel_confusion_matrix,
)

from ml.ptbxl.config import (
    CHECKPOINTS_DIR,
    BATCH_SIZE,
    RANDOM_SEED,
    TARGET_CLASSES,
)
from ml.ptbxl.dataset import PTBXLSequence, get_data_splits
from ml.ptbxl.train import BinaryFocalLoss


# ---------------------------------------------------------------------------
# Custom Loss for Model Loading
# ---------------------------------------------------------------------------

class MaskedBinaryFocalLoss(tf.keras.losses.Loss):
    """
    Binary Focal Loss with per-sample label masking.
    Included here to deserialize Stage 2 checkpoints cleanly without cross-module imports.
    """
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25,
                 name: str = "masked_binary_focal_loss"):
        super().__init__(name=name)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        mask = tf.cast(y_true >= 0.0, tf.float32)
        y_true_clamped = tf.clip_by_value(y_true, 0.0, 1.0)
        eps = tf.keras.backend.epsilon()
        y_pred_clipped = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        p_t = y_true_clamped * y_pred_clipped + (1.0 - y_true_clamped) * (1.0 - y_pred_clipped)
        alpha_factor = y_true_clamped * self.alpha + (1.0 - y_true_clamped) * (1.0 - self.alpha)
        focal_weight = alpha_factor * tf.pow(1.0 - p_t, self.gamma)
        bce = -tf.math.log(p_t)
        element_loss = focal_weight * bce
        masked_loss = element_loss * mask
        n_supervised = tf.reduce_sum(mask) + eps
        return tf.reduce_sum(masked_loss) / n_supervised


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIMIC_DIR        = os.path.join(PROJECT_ROOT, "ml", "data", "mimic_processed_400hz")
STAGE2_CKPT_DIR  = os.path.join(PROJECT_ROOT, "ml", "ptbxl", "checkpoints")
STAGE2_PLOTS_DIR = os.path.join(PROJECT_ROOT, "ml", "ptbxl", "plots", "stage2_gradcam")
STAGE1_CKPT      = os.path.join(CHECKPOINTS_DIR, "best_model.keras")

candidate_stage2_ckpts = [
    os.path.join(STAGE2_CKPT_DIR, "best_model_stage2b.keras"),
    os.path.join(PROJECT_ROOT, "ml", "mimic", "checkpoints", "best_model_stage2b.keras"),
]
DEFAULT_STAGE2_CKPT = next((p for p in candidate_stage2_ckpts if os.path.exists(p)), candidate_stage2_ckpts[0])

os.makedirs(STAGE2_PLOTS_DIR, exist_ok=True)

NUM_CLASSES  = len(TARGET_CLASSES)  # 7
WPW_IDX      = TARGET_CLASSES.index("WPW")    # 4
LNGQT_IDX    = TARGET_CLASSES.index("LNGQT") # 5

# CAM parameters (Inception-1D with GAP)
CAM_FEATURE_LAYER = "feature_maps"   # (batch, 1024, 128) — before GAP
CAM_OUTPUT_LENGTH = 4096             # Interpolated to full signal length

# Clinically expected Grad-CAM activation regions
CAM_CLINICAL_ANNOTATIONS = {
    "WPW":   "Expected: Delta wave / shortened PR (early QRS region)",
    "LNGQT": "Expected: ST segment + T-wave prolongation (late repolarization)",
}

CAM_DISPLAY_LEAD = "II"
PTBXL_LEAD_ORDER = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]


# ---------------------------------------------------------------------------
# Metric Utilities
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred_probs: np.ndarray,
                    thresholds: dict[str, float]) -> pd.DataFrame:
    """
    Compute per-class metrics table: ROC-AUC, PR-AUC, Precision, Recall, F1.
    """
    rows = []
    for i, cls in enumerate(TARGET_CLASSES):
        y_t = y_true[:, i]
        y_p = y_pred_probs[:, i]
        y_b = (y_p >= thresholds.get(cls, 0.5)).astype(int)

        roc = roc_auc_score(y_t, y_p) if y_t.sum() > 0 else np.nan
        pr  = average_precision_score(y_t, y_p) if y_t.sum() > 0 else np.nan
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_t, y_b, average="binary", zero_division=0
        )
        rows.append({
            "Class": cls,
            "ROC-AUC": roc,
            "PR-AUC":  pr,
            "Precision": prec,
            "Recall":    rec,
            "F1":        f1,
            "Threshold": thresholds.get(cls, 0.5),
        })
    return pd.DataFrame(rows).set_index("Class")


def find_optimal_thresholds(y_true: np.ndarray, y_pred_probs: np.ndarray,
                             grid_size: int = 91) -> dict[str, float]:
    """
    Per-class F1-maximizing threshold sweep over validation set.
    """
    thresholds_grid = np.linspace(0.05, 0.95, grid_size)
    optimal = {}

    print("\n" + "=" * 68)
    print("  THRESHOLD SWEEP — F1 Maximization (Validation Set)")
    print("=" * 68)
    print(f"  {'Class':<10} | {'τ*':<6} | {'F1':>6} | {'Precision':>10} | {'Recall':>8}")
    print("  " + "-" * 55)

    for i, cls in enumerate(TARGET_CLASSES):
        y_t, y_p = y_true[:, i], y_pred_probs[:, i]
        best_τ, best_f1, best_p, best_r = 0.5, -1.0, 0.0, 0.0

        for τ in thresholds_grid:
            p, r, f1, _ = precision_recall_fscore_support(
                y_t, (y_p >= τ).astype(int), average="binary", zero_division=0
            )
            if f1 > best_f1:
                best_f1, best_τ, best_p, best_r = f1, τ, p, r

        optimal[cls] = float(best_τ)
        print(f"  {cls:<10} | {best_τ:<6.2f} | {best_f1:>6.4f} | {best_p:>10.4f} | {best_r:>8.4f}")

    print("=" * 68)
    return optimal


def print_metrics_table(df: pd.DataFrame, title: str) -> None:
    """Pretty-print a metrics DataFrame."""
    print(f"\n{'=' * 68}")
    print(f"  {title}")
    print(f"{'=' * 68}")
    print(f"  {'Class':<10} | {'ROC-AUC':>8} | {'PR-AUC':>7} | {'Precision':>10} | {'Recall':>8} | {'F1':>8}")
    print("  " + "-" * 60)
    for cls, row in df.iterrows():
        roc = f"{row['ROC-AUC']:.4f}" if not np.isnan(row["ROC-AUC"]) else "  N/A  "
        pr  = f"{row['PR-AUC']:.4f}"  if not np.isnan(row["PR-AUC"])  else "  N/A  "
        print(f"  {cls:<10} | {roc:>8} | {pr:>7} | {row['Precision']:>10.4f} | "
              f"{row['Recall']:>8.4f} | {row['F1']:>8.4f}")
    print(f"{'=' * 68}")


def print_comparison_table(before: pd.DataFrame, after: pd.DataFrame) -> None:
    """Side-by-side before/after comparison on PTB-XL test set."""
    print(f"\n{'=' * 90}")
    print("  COMPARATIVE TABLE — PTB-XL Test Set: Stage 1 vs Stage 2")
    print(f"{'=' * 90}")
    header = f"  {'Class':<10} | {'ROC-AUC S1':>11} | {'ROC-AUC S2':>11} | {'Δ AUC':>7} | {'Recall S1':>10} | {'Recall S2':>10} | {'Δ Recall':>9}"
    print(header)
    print("  " + "-" * 85)
    for cls in TARGET_CLASSES:
        auc1 = before.loc[cls, "ROC-AUC"]
        auc2 = after.loc[cls, "ROC-AUC"]
        rec1 = before.loc[cls, "Recall"]
        rec2 = after.loc[cls, "Recall"]
        d_auc = auc2 - auc1 if not (np.isnan(auc1) or np.isnan(auc2)) else np.nan
        d_rec = rec2 - rec1

        auc1_s = f"{auc1:.4f}" if not np.isnan(auc1) else "  N/A  "
        auc2_s = f"{auc2:.4f}" if not np.isnan(auc2) else "  N/A  "
        d_s    = f"{d_auc:+.4f}" if not np.isnan(d_auc) else "  N/A  "
        flag = " ↑" if (not np.isnan(d_auc) and d_auc > 0.02) else (" ↓" if (not np.isnan(d_auc) and d_auc < -0.01) else "  ")
        print(f"  {cls:<10} | {auc1_s:>11} | {auc2_s:>11} | {d_s:>7}{flag} | "
              f"{rec1:>10.4f} | {rec2:>10.4f} | {d_rec:>+9.4f}")
    print(f"{'=' * 90}")


# ---------------------------------------------------------------------------
# 1D Grad-CAM (Inception-1D with GAP)
# ---------------------------------------------------------------------------

def compute_gradcam_1d(model: tf.keras.Model,
                        signal_batch: np.ndarray,
                        class_idx: int) -> np.ndarray:
    """
    Computes 1D Grad-CAM for Inception-1D with GlobalAveragePooling (GAP).
    """
    feature_maps_layer = model.get_layer(CAM_FEATURE_LAYER)
    cam_model = tf.keras.Model(
        inputs=model.input,
        outputs=[model.output, feature_maps_layer.output],
    )

    input_tensor = tf.constant(signal_batch, dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(input_tensor)
        probs, feature_maps = cam_model(input_tensor, training=False)
        tape.watch(feature_maps)
        eps = 1e-7
        p_cls = tf.clip_by_value(probs[0, class_idx], eps, 1.0 - eps)
        class_score = tf.math.log(p_cls / (1.0 - p_cls))

    grads = tape.gradient(class_score, feature_maps)   # (1, 1024, 128)

    if grads is None:
        raise RuntimeError(
            f"Gradient is None for class {TARGET_CLASSES[class_idx]}. "
            "Ensure the model is not compiled with run_eagerly=False issues."
        )

    grads_np = grads.numpy()[0]            # (1024, 128)
    fmaps_np = feature_maps.numpy()[0]    # (1024, 128)

    alpha_k = np.mean(grads_np, axis=0)   # (128,)
    cam_raw = np.sum(alpha_k[np.newaxis, :] * fmaps_np, axis=-1)  # (1024,)
    cam_relu = np.maximum(cam_raw, 0.0)

    t_low  = np.linspace(0, 1, len(cam_relu))
    t_full = np.linspace(0, 1, CAM_OUTPUT_LENGTH)
    cam_full = np.interp(t_full, t_low, cam_relu)

    cam_max = cam_full.max()
    if cam_max > 1e-8:
        cam_full /= cam_max

    return cam_full.astype(np.float32)


# ---------------------------------------------------------------------------
# CAM Visualization
# ---------------------------------------------------------------------------

def plot_gradcam_ecg(signal: np.ndarray,
                      cam: np.ndarray,
                      prob: float,
                      class_name: str,
                      title: str,
                      save_path: str,
                      sampling_rate: int = 400) -> None:
    """
    Plots a 12-lead ECG with 1D Grad-CAM heatmap overlay on Lead II.
    """
    t = np.arange(CAM_OUTPUT_LENGTH) / sampling_rate  # Time axis in seconds
    cam_contrast = np.where(cam > 0.25, cam ** 3, 0.0)

    n_leads = len(PTBXL_LEAD_ORDER)
    fig = plt.figure(figsize=(20, 16))
    fig.patch.set_facecolor("#f9f9f9")
    gs = gridspec.GridSpec(n_leads, 1, hspace=0.6)

    lead_ii_idx = PTBXL_LEAD_ORDER.index(CAM_DISPLAY_LEAD)

    for lead_i, lead_name in enumerate(PTBXL_LEAD_ORDER):
        ax = fig.add_subplot(gs[lead_i])
        ecg_signal = signal[:, lead_i]
        ax.plot(t, ecg_signal, color="#1a1a2e", linewidth=0.6, alpha=0.9)

        if lead_i == lead_ii_idx:
            norm_cam = cam_contrast
            for j in range(len(t) - 1):
                intensity = norm_cam[j]
                if intensity > 0.05:
                    r = min(1.0, 0.6 + 0.4 * intensity)
                    g = max(0.0, 0.5 - 0.5 * intensity)
                    b = 0.0
                    ax.fill_between(t[j:j+2],
                                    ecg_signal[j:j+2],
                                    ecg_signal.min() - 0.1,
                                    color=(r, g, b),
                                    alpha=0.5 * intensity,
                                    linewidth=0)

            cam_line = cam_contrast * (ecg_signal.max() - ecg_signal.min()) * 0.4 + ecg_signal.min()
            ax.plot(t, cam_line, color="#e63946", linewidth=1.0, linestyle="--",
                    alpha=0.7, label="Grad-CAM (normalized)")
            ax.legend(fontsize=7, loc="upper right")
            ax.set_facecolor("#fff9f0")

        ax.set_ylabel(lead_name, fontsize=8, rotation=0, labelpad=20)
        ax.set_xlim(t[0], t[-1])
        ax.tick_params(axis="both", labelsize=6)
        ax.set_yticks([])

        ax.grid(True, which="major", linestyle="-", linewidth=0.3, color="#f0a0a0", alpha=0.5)
        ax.set_xticks(np.arange(0, t[-1], 0.2))
        ax.set_xticks(np.arange(0, t[-1], 0.04), minor=True)
        ax.grid(True, which="minor", linestyle=":", linewidth=0.15, color="#f0c0c0", alpha=0.4)

    clinical_note = CAM_CLINICAL_ANNOTATIONS.get(class_name, "")
    fig.suptitle(
        f"{title}\n{class_name} — P(class) = {prob:.3f}\n{clinical_note}",
        fontsize=11, y=0.98, weight="bold"
    )
    plt.savefig(save_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [+] Saved CAM plot: {os.path.basename(save_path)}")


# ---------------------------------------------------------------------------
# Main Evaluation Pipeline
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 2 — Evaluation & 1D Grad-CAM")
    p.add_argument("--stage2-checkpoint", type=str, default=DEFAULT_STAGE2_CKPT,
                   help="Path to Stage 2 fine-tuned checkpoint (.keras)")
    p.add_argument("--n-cam-samples", type=int, default=5,
                   help="Number of WPW and LNGQT cases for Grad-CAM plots")
    p.add_argument("--dry-run", action="store_true",
                   help="Quick validation with small data subset")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    custom_objects = {
        "BinaryFocalLoss":       BinaryFocalLoss,
        "MaskedBinaryFocalLoss": MaskedBinaryFocalLoss,
    }

    # ------------------------------------------------------------------
    # 1. Load both model checkpoints
    # ------------------------------------------------------------------
    if not os.path.exists(STAGE1_CKPT):
        raise FileNotFoundError(f"Stage 1 checkpoint not found: {STAGE1_CKPT}")
    if not os.path.exists(args.stage2_checkpoint):
        raise FileNotFoundError(
            f"Stage 2 checkpoint not found: {args.stage2_checkpoint}\n"
            f"Run fine-tuning first: python ml/scripts/finetune_ptbxl_mimic.py"
        )

    print(f"[*] Loading Stage 1 checkpoint: {STAGE1_CKPT}")
    model_s1 = tf.keras.models.load_model(STAGE1_CKPT, compile=False)

    print(f"[*] Loading Stage 2 checkpoint: {args.stage2_checkpoint}")
    model_s2 = tf.keras.models.load_model(args.stage2_checkpoint, compile=False)

    # ------------------------------------------------------------------
    # 2. Load PTB-XL Val + Test splits (for threshold calibration + eval)
    # ------------------------------------------------------------------
    print("\n[*] Loading PTB-XL splits...")
    splits = get_data_splits(classes=TARGET_CLASSES, load_waveforms=False)
    df_val,  y_val_df,  _ = splits["val"]
    df_test, y_test_df, _ = splits["test"]

    y_val  = y_val_df.values.astype(np.float32)
    y_test = y_test_df.values.astype(np.float32)

    if args.dry_run:
        df_val   = df_val.head(100);   y_val  = y_val[:100]
        df_test  = df_test.head(200);  y_test = y_test[:200]

    val_gen  = PTBXLSequence(df_val,  y_val_df.head(len(df_val)),
                              batch_size=BATCH_SIZE, shuffle=False)
    test_gen = PTBXLSequence(df_test, y_test_df.head(len(df_test)),
                              batch_size=BATCH_SIZE, shuffle=False)

    # ------------------------------------------------------------------
    # 3. Find optimal thresholds on Val set using Stage 2 model
    # ------------------------------------------------------------------
    print("\n[*] Generating Val predictions (Stage 2)...")
    y_val_probs_s2 = model_s2.predict(val_gen, verbose=0)
    optimal_thresholds = find_optimal_thresholds(y_val, y_val_probs_s2)

    print("[*] Generating Val predictions (Stage 1)...")
    y_val_probs_s1 = model_s1.predict(val_gen, verbose=0)
    opt_thresh_s1 = find_optimal_thresholds(y_val, y_val_probs_s1)

    # ------------------------------------------------------------------
    # 4. Evaluate on PTB-XL Test Set — Stage 1 vs Stage 2
    # ------------------------------------------------------------------
    print("\n[*] Generating Test predictions (Stage 1)...")
    y_test_probs_s1 = model_s1.predict(test_gen, verbose=0)

    print("[*] Generating Test predictions (Stage 2)...")
    y_test_probs_s2 = model_s2.predict(test_gen, verbose=0)

    metrics_s1 = compute_metrics(y_test, y_test_probs_s1, opt_thresh_s1)
    metrics_s2 = compute_metrics(y_test, y_test_probs_s2, optimal_thresholds)

    print_metrics_table(metrics_s1, "PTB-XL TEST — Stage 1 (Baseline)")
    print_metrics_table(metrics_s2, "PTB-XL TEST — Stage 2 (Fine-Tuned)")
    print_comparison_table(metrics_s1, metrics_s2)

    comparison = metrics_s1.add_suffix("_S1").join(metrics_s2.add_suffix("_S2"))
    comparison.to_csv(os.path.join(STAGE2_PLOTS_DIR, "metrics_comparison_ptbxl_test.csv"))
    print(f"[+] Metrics saved: metrics_comparison_ptbxl_test.csv")

    # ------------------------------------------------------------------
    # 5. Evaluate on MIMIC External Hold-out
    # ------------------------------------------------------------------
    for fname in ["X_test.npy", "y_test.npy"]:
        if not os.path.exists(os.path.join(MIMIC_DIR, fname)):
            print(f"\n[!] MIMIC test tensors not found — skipping external evaluation.")
            break
    else:
        print("\n[*] Loading MIMIC external hold-out...")
        X_mimic_test = np.load(os.path.join(MIMIC_DIR, "X_test.npy"))
        y_mimic_test = np.load(os.path.join(MIMIC_DIR, "y_test.npy"))

        if args.dry_run:
            X_mimic_test = X_mimic_test[:50]
            y_mimic_test = y_mimic_test[:50]

        y_mimic_test_clipped = np.where(y_mimic_test >= 0, y_mimic_test, np.nan)

        print(f"  MIMIC test: {X_mimic_test.shape}, WPW={int(np.nansum(y_mimic_test[:,WPW_IDX]))}, "
              f"LNGQT={int(np.nansum(y_mimic_test[:,LNGQT_IDX]))}")

        print("[*] Generating MIMIC predictions (Stage 2)...")
        y_mimic_probs_s2 = model_s2.predict(X_mimic_test, batch_size=BATCH_SIZE, verbose=0)

        print("\n" + "=" * 68)
        print("  MIMIC EXTERNAL HOLD-OUT EVALUATION (Stage 2 — WPW & LNGQT only)")
        print("=" * 68)
        for cls_idx, cls in [(WPW_IDX, "WPW"), (LNGQT_IDX, "LNGQT")]:
            valid_mask = ~np.isnan(y_mimic_test_clipped[:, cls_idx])
            y_t_cls = y_mimic_test_clipped[valid_mask, cls_idx]
            y_p_cls = y_mimic_probs_s2[valid_mask, cls_idx]

            if y_t_cls.sum() == 0:
                print(f"  {cls}: No positive examples in test split.")
                continue

            roc = roc_auc_score(y_t_cls, y_p_cls)
            pr  = average_precision_score(y_t_cls, y_p_cls)
            τ   = optimal_thresholds.get(cls, 0.5)
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_t_cls, (y_p_cls >= τ).astype(int), average="binary", zero_division=0
            )
            print(f"  {cls:<10} | ROC-AUC={roc:.4f} | PR-AUC={pr:.4f} | "
                  f"Precision={prec:.4f} | Recall={rec:.4f} | F1={f1:.4f} | τ={τ:.2f}")
        print("=" * 68)

    # ------------------------------------------------------------------
    # 6. 1D Grad-CAM — WPW and LNGQT positive cases
    # ------------------------------------------------------------------
    print(f"\n[*] Generating {args.n_cam_samples} Grad-CAM plots per class...")

    if os.path.exists(os.path.join(MIMIC_DIR, "X_test.npy")):
        X_cam_pool = np.load(os.path.join(MIMIC_DIR, "X_test.npy"))
        y_cam_pool = np.load(os.path.join(MIMIC_DIR, "y_test.npy"))

        for cls_idx, cls_name in [(WPW_IDX, "WPW"), (LNGQT_IDX, "LNGQT")]:
            pos_indices = np.where(y_cam_pool[:, cls_idx] > 0.5)[0]
            if len(pos_indices) == 0:
                print(f"  [!] No {cls_name} positives in MIMIC test for CAM. Skipping.")
                continue

            probs = model_s2.predict(X_cam_pool[pos_indices], batch_size=BATCH_SIZE, verbose=0)

            for k, orig_idx in enumerate(pos_indices[:args.n_cam_samples]):
                signal = X_cam_pool[[orig_idx]]
                prob_cls = float(probs[k, cls_idx])

                try:
                    cam = compute_gradcam_1d(model_s2, signal, cls_idx)
                except Exception as e:
                    print(f"  [!] Grad-CAM failed for {cls_name} sample {k}: {e}")
                    continue

                save_path = os.path.join(
                    STAGE2_PLOTS_DIR,
                    f"gradcam_{cls_name.lower()}_sample_{k+1}.png"
                )
                plot_gradcam_ecg(
                    signal=signal[0],
                    cam=cam,
                    prob=prob_cls,
                    class_name=cls_name,
                    title=f"1D Grad-CAM | {cls_name} | MIMIC Test Sample #{k+1}",
                    save_path=save_path,
                )

    print(f"\n[+] All plots saved to: {STAGE2_PLOTS_DIR}/")
    print(f"[+] Optimal thresholds (Stage 2): {optimal_thresholds}")
    print("\n[+] Evaluation complete.")


if __name__ == "__main__":
    main()
