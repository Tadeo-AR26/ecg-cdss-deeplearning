import ast
import os
import numpy as np
import pandas as pd
import wfdb
from typing import Tuple, List, Dict, Optional
import tensorflow as tf

from scipy.signal import resample

from ml.ptbxl.config import (
    DATA_DIR,
    RAW_SAMPLING_RATE,
    INPUT_LENGTH,
    TRAIN_FOLDS,
    VAL_FOLDS,
    TEST_FOLDS,
    TARGET_CLASSES,
)


def load_raw_metadata() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads ptbxl_database.csv and scp_statements.csv from DATA_DIR.
    """
    db_path = os.path.join(DATA_DIR, "ptbxl_database.csv")
    scp_path = os.path.join(DATA_DIR, "scp_statements.csv")

    if not os.path.exists(db_path) or not os.path.exists(scp_path):
        raise FileNotFoundError(
            f"Metadata files missing in {DATA_DIR}. Ensure ptbxl_database.csv "
            f"and scp_statements.csv are present."
        )

    df = pd.read_csv(db_path, index_col="ecg_id", engine="python")
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)

    scp = pd.read_csv(scp_path, index_col=0, engine="python")
    return df, scp


def build_multilabel_targets(
    df: pd.DataFrame,
    scp: pd.DataFrame,
    classes: List[str] = TARGET_CLASSES,
    min_likelihood: float = 0.0,
) -> pd.DataFrame:
    """
    Maps SCP codes to target classes (superclasses + specific codes like WPW/LNGQT).
    Returns a binary multi-hot DataFrame with shape (len(df), len(classes)).
    """
    # Identify diagnostic_class column for superclasses (NORM, MI, STTC, CD, HYP)
    super_col = None
    for cand in ["diagnostic_class", "diagnostic_superclass"]:
        if cand in scp.columns:
            super_col = cand
            break

    code_to_super = scp[super_col].dropna().to_dict() if super_col else {}

    target_matrix = np.zeros((len(df), len(classes)), dtype=np.float32)

    for idx, (_, row) in enumerate(df.iterrows()):
        scp_dict = row["scp_codes"]
        assigned = set()

        for code, likelihood in scp_dict.items():
            if likelihood <= min_likelihood:
                continue

            # 1. Direct match with target class (e.g., 'WPW', 'LNGQT', 'MI')
            if code in classes:
                assigned.add(code)

            # 2. Check superclass match (e.g., IMI -> MI, LAFB -> CD)
            if code in code_to_super:
                sup = code_to_super[code]
                if sup in classes:
                    assigned.add(sup)

        for col_idx, cls_name in enumerate(classes):
            if cls_name in assigned:
                target_matrix[idx, col_idx] = 1.0

    return pd.DataFrame(target_matrix, index=df.index, columns=classes)


def load_signals(
    df_subset: pd.DataFrame,
    sampling_rate: int = RAW_SAMPLING_RATE,
    target_length: int = INPUT_LENGTH,
    verbose: bool = True,
) -> np.ndarray:
    """
    Loads 12-lead ECG signals and resamples to target_length (4096 samples)
    to match the Ribeiro ResNet-1D input tensor shape: (N, 4096, 12).
    """
    file_col = "filename_lr" if sampling_rate == 100 else "filename_hr"
    n_records = len(df_subset)

    signals = np.empty((n_records, target_length, 12), dtype=np.float32)

    for i, (_, row) in enumerate(df_subset.iterrows()):
        filepath = os.path.join(DATA_DIR, row[file_col])
        record = wfdb.rdrecord(filepath)
        raw_sig = record.p_signal.astype(np.float32)

        # Resample along time axis (axis=0) if raw_sig length != target_length
        if raw_sig.shape[0] != target_length:
            signals[i] = resample(raw_sig, target_length, axis=0).astype(np.float32)
        else:
            signals[i] = raw_sig

        if verbose and (i + 1) % 5000 == 0:
            print(f"  [+] Loaded {i + 1}/{n_records} waveforms (resampled to {target_length} pts)...")

    return signals


# ============================================================
# TODO FOR USER: Custom Preprocessing & Data Augmentations
# ============================================================

def apply_augmentations(signal: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    TODO: Implement signal augmentations here.
    Especially recommended for minority classes like WPW (~0.6%):
      - Gaussian noise injection (sigma <= 0.02 mV)
      - Amplitude scaling (e.g. 0.9 - 1.1x)
      - Baseline wander simulation
    (Note: Avoid time-shifting or lead reordering for WPW to protect delta wave).
    """
    # Return signal as is by default until you implement augmentations
    return signal


def compute_class_weights(y: np.ndarray) -> np.ndarray:
    """
    TODO: Compute positive weights (pos_weight) for BCEWithLogits or weighted loss.
    pos_weight = N_neg / N_pos for each class.
    Essential to compensate for severe imbalance (e.g., WPW ~120 records).
    """
    n_samples = len(y)
    n_pos = np.sum(y, axis=0)
    n_neg = n_samples - n_pos

    # Safe division avoiding divide-by-zero
    pos_weights = np.where(n_pos > 0, n_neg / np.maximum(n_pos, 1), 1.0)
    return pos_weights.astype(np.float32)


def get_data_splits(
    classes: List[str] = TARGET_CLASSES,
    load_waveforms: bool = False,
) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame, Optional[np.ndarray]]]:
    """
    Convenience loader splitting the dataset into Train, Val, and Test folds.
    Returns:
        {
            'train': (df_train, y_train, X_train or None),
            'val':   (df_val, y_val, X_val or None),
            'test':  (df_test, y_test, X_test or None),
        }
    """
    df, scp = load_raw_metadata()
    targets = build_multilabel_targets(df, scp, classes=classes)

    train_mask = df["strat_fold"].isin(TRAIN_FOLDS)
    val_mask = df["strat_fold"].isin(VAL_FOLDS)
    test_mask = df["strat_fold"].isin(TEST_FOLDS)

    splits = {
        "train": (df[train_mask], targets[train_mask], load_signals(df[train_mask]) if load_waveforms else None),
        "val": (df[val_mask], targets[val_mask], load_signals(df[val_mask]) if load_waveforms else None),
        "test": (df[test_mask], targets[test_mask], load_signals(df[test_mask]) if load_waveforms else None),
    }

    return splits

class PTBXLSequence(tf.keras.utils.Sequence):
    """
    Keras Sequence generator that loads ECG records dynamically in batches
    from disk, preventing out-of-memory errors by never holding the entire
    dataset waveforms in RAM simultaneously.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        targets: np.ndarray,
        batch_size: int = 64,
        sampling_rate: int = RAW_SAMPLING_RATE,
        target_length: int = INPUT_LENGTH,
        shuffle: bool = True,
        augment: bool = False,
    ):
        super().__init__()
        self.df = df
        file_col = "filename_lr" if sampling_rate == 100 else "filename_hr"
        self.filenames = df[file_col].values
        self.targets = targets.values if hasattr(targets, "values") else np.asarray(targets)
        self.batch_size = batch_size
        self.target_length = target_length
        self.shuffle = shuffle
        self.augment = augment
        self.indices = np.arange(len(self.filenames))
        self.on_epoch_end()

    def __len__(self) -> int:
        return int(np.ceil(len(self.filenames) / self.batch_size))

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        batch_indices = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_filenames = self.filenames[batch_indices]
        batch_y = self.targets[batch_indices]

        batch_x = np.empty((len(batch_indices), self.target_length, 12), dtype=np.float32)
        for i, fname in enumerate(batch_filenames):
            sig = self._load_single_signal(fname)
            if self.augment:
                sig = apply_augmentations(sig, batch_y[i])
            batch_x[i] = sig

        return batch_x, batch_y

    def _load_single_signal(self, filename: str) -> np.ndarray:
        filepath = os.path.join(DATA_DIR, filename)
        record = wfdb.rdrecord(filepath)
        raw_sig = record.p_signal.astype(np.float32)
        if np.isnan(raw_sig).any():
            raw_sig = np.nan_to_num(raw_sig)

        if raw_sig.shape[0] != self.target_length:
            return resample(raw_sig, self.target_length, axis=0).astype(np.float32)
        return raw_sig

    def on_epoch_end(self) -> None:
        if self.shuffle:
            np.random.shuffle(self.indices)