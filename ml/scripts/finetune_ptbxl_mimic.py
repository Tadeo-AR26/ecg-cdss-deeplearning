"""
ml/scripts/finetune_ptbxl_mimic.py
==================================
Stage 2 — Step 2: Two-Stage Fine-Tuning with Loss Masking

Loads the Stage 1 Inception-1D checkpoint (ml/ptbxl/checkpoints/best_model.keras),
combines PTB-XL training data (folds 1–8) with MIMIC-processed tensors using a
hybrid Keras Sequence, and applies masked BCE/Focal loss so that MIMIC records only
contribute gradients for WPW and LNGQT outputs.

Training strategy:
  Stage A — Frozen backbone, train only the Dense head (probabilities layer).
             LR=1e-3, 15 epochs.
  Stage B — Unfreeze the last Inception block (inc_block_2_*). Lower LR=2e-4,
             20 more epochs. Preserves early feature representations entirely.

Usage:
    python ml/scripts/finetune_ptbxl_mimic.py [--stage {A,B,AB}] [--epochs-a N] [--epochs-b N]
                                              [--batch-size N] [--dry-run]
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
import tensorflow as tf

from ml.ptbxl.config import (
    CHECKPOINTS_DIR,
    RUNS_DIR,
    BATCH_SIZE,
    RANDOM_SEED,
    TARGET_CLASSES,
    INPUT_LENGTH,
)
from ml.ptbxl.dataset import (
    PTBXLSequence,
    get_data_splits,
    compute_class_weights,
)
from ml.ptbxl.train import BinaryFocalLoss


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIMIC_DIR = os.path.join(PROJECT_ROOT, "ml", "data", "mimic_processed_400hz")
STAGE2_CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "ml", "ptbxl", "checkpoints")
STAGE2_RUNS_DIR = os.path.join(PROJECT_ROOT, "ml", "ptbxl", "runs", "stage2")

os.makedirs(STAGE2_CHECKPOINTS_DIR, exist_ok=True)
os.makedirs(STAGE2_RUNS_DIR, exist_ok=True)

NUM_CLASSES = len(TARGET_CLASSES)  # 7

# Loss mask column indices: only WPW and LNGQT are supervised in MIMIC records
WPW_IDX   = TARGET_CLASSES.index("WPW")    # = 4
LNGQT_IDX = TARGET_CLASSES.index("LNGQT") # = 5
MIMIC_ACTIVE_COLS = [WPW_IDX, LNGQT_IDX]

# Layer name pattern for the Inception backbone (Inception-1D in models.py)
# 'inc_block_2_*' = second Inception block: unfreeze in Stage B
STAGE_B_UNFREEZE_PATTERN = "inc_block_2"

# WPW oversampling ratio in the hybrid dataset
WPW_OVERSAMPLE_FACTOR = 3


# ---------------------------------------------------------------------------
# Masked Loss Function
# ---------------------------------------------------------------------------

class MaskedBinaryFocalLoss(tf.keras.losses.Loss):
    """
    Binary Focal Loss with per-sample label masking.

    For MIMIC-sourced samples, only WPW and LNGQT columns (indices 4, 5)
    contribute to the loss. All other classes are silenced via a binary mask.

    Label encoding in y_true:
        -1.0  → unknown / masked out (from MIMIC for superclasses)
         0.0  → confirmed negative
         1.0  → confirmed positive

    Masking logic:
        mask_col = (y_true >= 0.0).astype(float)  # 1 where supervised, 0 where unknown
        effective_loss = focal_loss(y_pred, clip(y_true, 0, 1)) * mask_col
        loss = mean(effective_loss[mask_col > 0])  # Only average over supervised positions
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25,
                 name: str = "masked_binary_focal_loss"):
        super().__init__(name=name)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)

        # Mask: positions where label is known (≥ 0). Unknown = -1.0
        mask = tf.cast(y_true >= 0.0, tf.float32)   # (batch, 7)

        # Clamp y_true to [0, 1] for BCE computation (removes the -1 sentinel)
        y_true_clamped = tf.clip_by_value(y_true, 0.0, 1.0)

        # Focal loss computation
        eps = tf.keras.backend.epsilon()
        y_pred_clipped = tf.clip_by_value(y_pred, eps, 1.0 - eps)

        # p_t: probability assigned to the ground-truth class
        p_t = y_true_clamped * y_pred_clipped + (1.0 - y_true_clamped) * (1.0 - y_pred_clipped)
        alpha_factor = y_true_clamped * self.alpha + (1.0 - y_true_clamped) * (1.0 - self.alpha)
        focal_weight = alpha_factor * tf.pow(1.0 - p_t, self.gamma)
        bce = -tf.math.log(p_t)
        element_loss = focal_weight * bce   # (batch, 7)

        # Apply supervision mask
        masked_loss = element_loss * mask   # zero out unknown positions

        # Normalize by the number of supervised label positions (avoid / 0)
        n_supervised = tf.reduce_sum(mask) + eps
        return tf.reduce_sum(masked_loss) / n_supervised


# ---------------------------------------------------------------------------
# Hybrid Dataset
# ---------------------------------------------------------------------------

class HybridECGSequence(tf.keras.utils.Sequence):
    """
    Combined PTB-XL + MIMIC-IV-ECG Sequence generator.

    PTB-XL records:
        - Labels: full 7-class supervision (y_true >= 0 for all columns)
        - Loaded dynamically from disk via PTBXLSequence logic

    MIMIC records:
        - Labels: -1.0 for all columns except WPW (idx 4) and LNGQT (idx 5)
        - Already preprocessed as tensors in ml/data/mimic_processed_400hz/

    WPW oversampling:
        MIMIC WPW records (is_WPW=1) are repeated WPW_OVERSAMPLE_FACTOR times
        in the index array to compensate for the 1:5 WPW/LNGQT ratio.
    """

    def __init__(
        self,
        ptbxl_df,
        ptbxl_y: np.ndarray,
        mimic_X: np.ndarray,
        mimic_y: np.ndarray,
        batch_size: int = BATCH_SIZE,
        shuffle: bool = True,
        augment: bool = True,
    ):
        super().__init__()
        from ml.ptbxl.config import DATA_DIR, RAW_SAMPLING_RATE

        self.DATA_DIR = DATA_DIR
        self.sampling_rate = RAW_SAMPLING_RATE
        file_col = "filename_lr" if RAW_SAMPLING_RATE == 100 else "filename_hr"
        self.ptbxl_filenames = ptbxl_df[file_col].values
        self.ptbxl_y = ptbxl_y.astype(np.float32)

        # Build MIMIC index with WPW oversampling
        n_mimic = len(mimic_X)
        wpw_mask = mimic_y[:, WPW_IDX] > 0.5

        mimic_indices = list(range(n_mimic))
        wpw_indices = np.where(wpw_mask)[0].tolist()
        # Repeat WPW records to boost representation
        mimic_indices += wpw_indices * (WPW_OVERSAMPLE_FACTOR - 1)
        mimic_indices = np.array(mimic_indices, dtype=np.int32)

        self.mimic_X = mimic_X
        self.mimic_y = mimic_y.astype(np.float32)
        self.mimic_indices = mimic_indices

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.target_length = INPUT_LENGTH  # 4096

        n_ptbxl = len(self.ptbxl_filenames)
        n_mimic_eff = len(self.mimic_indices)

        # Unified index: [0..n_ptbxl) → PTB-XL, [n_ptbxl..) → MIMIC
        self.ptbxl_range = n_ptbxl
        self.total = n_ptbxl + n_mimic_eff
        self.indices = np.arange(self.total)
        self.on_epoch_end()

        print(f"  [HybridSeq] PTB-XL: {n_ptbxl:,}  MIMIC (w/ WPW x{WPW_OVERSAMPLE_FACTOR}): {n_mimic_eff:,}  Total: {self.total:,}")

    def __len__(self) -> int:
        return int(np.ceil(self.total / self.batch_size))

    def __getitem__(self, idx: int):
        import wfdb
        from scipy.signal import resample as scipy_resample

        batch_idx = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = np.empty((len(batch_idx), self.target_length, 12), dtype=np.float32)
        batch_y = np.empty((len(batch_idx), NUM_CLASSES), dtype=np.float32)

        for i, global_idx in enumerate(batch_idx):
            if global_idx < self.ptbxl_range:
                # PTB-XL record — load dynamically from disk
                from ml.ptbxl.config import DATA_DIR
                fname = self.ptbxl_filenames[global_idx]
                filepath = os.path.join(DATA_DIR, fname)
                record = wfdb.rdrecord(filepath)
                sig = record.p_signal.astype(np.float32)
                sig = np.nan_to_num(sig)
                if sig.shape[0] != self.target_length:
                    sig = scipy_resample(sig, self.target_length, axis=0).astype(np.float32)
                if self.augment:
                    sig = self._augment(sig)
                batch_x[i] = sig
                batch_y[i] = self.ptbxl_y[global_idx]
            else:
                # MIMIC record (already preprocessed tensor)
                mimic_i = self.mimic_indices[global_idx - self.ptbxl_range]
                sig = self.mimic_X[mimic_i].copy()
                if self.augment:
                    sig = self._augment_mimic(sig, self.mimic_y[mimic_i])
                batch_x[i] = sig
                batch_y[i] = self.mimic_y[mimic_i]

        return batch_x, batch_y

    @staticmethod
    def _augment(signal: np.ndarray) -> np.ndarray:
        """PTB-XL augmentation: Gaussian noise + amplitude scaling."""
        noise = np.random.normal(0, 0.02, signal.shape).astype(np.float32)
        scale = np.random.uniform(0.9, 1.1)
        return (signal * scale + noise).astype(np.float32)

    @staticmethod
    def _augment_mimic(signal: np.ndarray, label: np.ndarray) -> np.ndarray:
        """
        MIMIC augmentation — same as PTB-XL but gentler (avoid distorting delta wave).
        No time-shifting, no lead swapping.
        """
        noise_sigma = 0.015  # Softer noise for WPW (preserve delta wave morphology)
        noise = np.random.normal(0, noise_sigma, signal.shape).astype(np.float32)
        scale = np.random.uniform(0.92, 1.08)
        return (signal * scale + noise).astype(np.float32)

    def on_epoch_end(self) -> None:
        if self.shuffle:
            np.random.shuffle(self.indices)


# ---------------------------------------------------------------------------
# Freeze / Unfreeze Utilities
# ---------------------------------------------------------------------------

def freeze_backbone(model: tf.keras.Model) -> None:
    """
    Stage A: Freeze all layers except the final Dense classification head.
    The 'probabilities' Dense layer remains trainable.
    """
    for layer in model.layers:
        layer.trainable = False

    # Unfreeze only the Dense head
    head = model.get_layer("probabilities")
    head.trainable = True
    _report_trainable(model, label="Stage A — Only Dense head trainable")


def unfreeze_last_inception_block(model: tf.keras.Model) -> None:
    """
    Stage B: Unfreeze the last Inception block (inc_block_2_*) and the head.
    All earlier layers (inc_block_1_*, initial conv/bn) remain frozen.
    """
    for layer in model.layers:
        if STAGE_B_UNFREEZE_PATTERN in layer.name or layer.name in ("feature_maps", "gap",
                                                                      "head_dropout", "probabilities"):
            layer.trainable = True

    _report_trainable(model, label="Stage B — Last Inception block + head trainable")


def _report_trainable(model: tf.keras.Model, label: str) -> None:
    print(f"  [{label}]")
    print(f"    Trainable params: {sum(np.prod(v.shape) for v in model.trainable_variables):,}")
    print(f"    Frozen   params:  {sum(np.prod(v.shape) for v in model.non_trainable_variables):,}")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def compile_model(model: tf.keras.Model, lr: float) -> None:
    """Compile model with MaskedBinaryFocalLoss and standard metrics."""
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)
    metrics = [
        tf.keras.metrics.BinaryAccuracy(name="accuracy", threshold=0.5),
        tf.keras.metrics.AUC(name="auc", multi_label=True),
        tf.keras.metrics.AUC(name="pr_auc", curve="PR", multi_label=True),
    ]
    model.compile(
        optimizer=optimizer,
        loss=MaskedBinaryFocalLoss(gamma=2.0, alpha=0.25),
        metrics=metrics,
    )


def run_training_stage(
    model: tf.keras.Model,
    train_gen,
    val_gen,
    epochs: int,
    stage_label: str,
    checkpoint_path: str,
    tensorboard_log_dir: str,
) -> tf.keras.callbacks.History:
    """Run a single training stage with callbacks."""
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
            patience=7,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc",
            mode="max",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    try:
        callbacks.append(tf.keras.callbacks.TensorBoard(
            log_dir=tensorboard_log_dir, histogram_freq=0
        ))
    except Exception as e:
        print(f"[!] TensorBoard not available: {e}")

    print(f"\n[*] === {stage_label} ===")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )
    return history


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 2 — Fine-Tuning with MIMIC-IV-ECG")
    p.add_argument("--stage", choices=["A", "B", "AB"], default="AB",
                   help="Which training stage(s) to run (default: AB = both)")
    p.add_argument("--epochs-a", type=int, default=15, help="Epochs for Stage A (frozen backbone)")
    p.add_argument("--epochs-b", type=int, default=20, help="Epochs for Stage B (partial unfreeze)")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--dry-run", action="store_true",
                   help="Quick 2-epoch test with reduced data")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    epochs_a = 2 if args.dry_run else args.epochs_a
    epochs_b = 2 if args.dry_run else args.epochs_b
    batch_size = args.batch_size

    # ------------------------------------------------------------------
    # 1. Load Stage 1 checkpoint
    # ------------------------------------------------------------------
    stage1_ckpt = os.path.join(CHECKPOINTS_DIR, "best_model.keras")
    if not os.path.exists(stage1_ckpt):
        raise FileNotFoundError(
            f"Stage 1 checkpoint not found: {stage1_ckpt}\n"
            f"Train the primary model first with: python -m ml.ptbxl.train"
        )

    print(f"[*] Loading Stage 1 checkpoint: {stage1_ckpt}")
    model = tf.keras.models.load_model(
        stage1_ckpt,
        custom_objects={"BinaryFocalLoss": BinaryFocalLoss},
    )
    print(f"[+] Model loaded. Input: {model.input_shape}  Output: {model.output_shape}")

    # ------------------------------------------------------------------
    # 2. Load PTB-XL training + validation splits (metadata only)
    # ------------------------------------------------------------------
    print("\n[*] Loading PTB-XL splits (metadata only)...")
    splits = get_data_splits(classes=TARGET_CLASSES, load_waveforms=False)
    df_train_ptbxl, y_train_ptbxl_df, _ = splits["train"]
    df_val_ptbxl,   y_val_ptbxl_df,   _ = splits["val"]

    y_train_ptbxl = y_train_ptbxl_df.values.astype(np.float32)
    y_val_ptbxl   = y_val_ptbxl_df.values.astype(np.float32)

    print(f"  PTB-XL train: {len(df_train_ptbxl):,}  val: {len(df_val_ptbxl):,}")

    # ------------------------------------------------------------------
    # 3. Load MIMIC processed tensors
    # ------------------------------------------------------------------
    for fname in ["X_train.npy", "y_train.npy", "X_test.npy", "y_test.npy"]:
        if not os.path.exists(os.path.join(MIMIC_DIR, fname)):
            raise FileNotFoundError(
                f"Missing MIMIC tensor: {os.path.join(MIMIC_DIR, fname)}\n"
                f"Run preprocessing first: python ml/scripts/preprocess_mimic_wpw_lngqt.py"
            )

    print(f"\n[*] Loading MIMIC processed tensors from {MIMIC_DIR}/...")
    X_mimic_train = np.load(os.path.join(MIMIC_DIR, "X_train.npy"))
    y_mimic_train = np.load(os.path.join(MIMIC_DIR, "y_train.npy"))

    # For validation, use PTB-XL val set only (no MIMIC leak into val)
    print(f"  MIMIC train: X={X_mimic_train.shape}  y={y_mimic_train.shape}")

    wpw_count   = int(np.sum(y_mimic_train[:, WPW_IDX] > 0.5))
    lngqt_count = int(np.sum(y_mimic_train[:, LNGQT_IDX] > 0.5))
    print(f"  MIMIC train labels: WPW={wpw_count}, LNGQT={lngqt_count}")

    # Subsample for dry-run
    if args.dry_run:
        df_train_ptbxl = df_train_ptbxl.head(200)
        y_train_ptbxl  = y_train_ptbxl[:200]
        df_val_ptbxl   = df_val_ptbxl.head(100)
        y_val_ptbxl    = y_val_ptbxl[:100]
        X_mimic_train  = X_mimic_train[:50]
        y_mimic_train  = y_mimic_train[:50]
        print("[!] DRY-RUN: Reduced to small subset.")

    # ------------------------------------------------------------------
    # 4. Build Sequence generators
    # ------------------------------------------------------------------
    train_gen = HybridECGSequence(
        df_train_ptbxl, y_train_ptbxl,
        X_mimic_train, y_mimic_train,
        batch_size=batch_size, shuffle=True, augment=True,
    )
    val_gen = tf.keras.utils.Sequence.__new__(tf.keras.utils.Sequence)
    from ml.ptbxl.dataset import PTBXLSequence
    val_gen = PTBXLSequence(df_val_ptbxl, y_val_ptbxl_df,
                             batch_size=batch_size, shuffle=False, augment=False)

    # ------------------------------------------------------------------
    # 5. Stage A — Frozen backbone, train Dense head only
    # ------------------------------------------------------------------
    if args.stage in ("A", "AB"):
        freeze_backbone(model)
        compile_model(model, lr=1e-3)

        ckpt_a = os.path.join(STAGE2_CHECKPOINTS_DIR, "best_model_stage2a.keras")
        log_a  = os.path.join(STAGE2_RUNS_DIR, "stage_a")
        run_training_stage(model, train_gen, val_gen, epochs_a,
                           "STAGE A — Frozen Backbone", ckpt_a, log_a)

        print(f"\n[+] Stage A complete. Best checkpoint: {ckpt_a}")

    # ------------------------------------------------------------------
    # 6. Stage B — Unfreeze last Inception block + head
    # ------------------------------------------------------------------
    if args.stage in ("B", "AB"):
        if args.stage == "AB":
            ckpt_a = os.path.join(STAGE2_CHECKPOINTS_DIR, "best_model_stage2a.keras")
            if os.path.exists(ckpt_a):
                print(f"\n[*] Reloading best Stage A checkpoint: {ckpt_a}")
                model = tf.keras.models.load_model(
                    ckpt_a,
                    custom_objects={
                        "BinaryFocalLoss": BinaryFocalLoss,
                        "MaskedBinaryFocalLoss": MaskedBinaryFocalLoss,
                    },
                )

        unfreeze_last_inception_block(model)
        compile_model(model, lr=2e-4)

        ckpt_b = os.path.join(STAGE2_CHECKPOINTS_DIR, "best_model_stage2b.keras")
        log_b  = os.path.join(STAGE2_RUNS_DIR, "stage_b")
        run_training_stage(model, train_gen, val_gen, epochs_b,
                           "STAGE B — Partial Unfreeze (last Inception block)", ckpt_b, log_b)

        print(f"\n[+] Stage B complete. Best checkpoint: {ckpt_b}")
        print(f"[+] Use this checkpoint for evaluation: python ml/scripts/evaluate_ptbxl_mimic_gradcam.py")


if __name__ == "__main__":
    main()
