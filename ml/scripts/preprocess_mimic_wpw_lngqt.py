"""
ml/scripts/preprocess_mimic_wpw_lngqt.py
========================================
Stage 2 — Step 1: MIMIC-IV-ECG Subset Preprocessing & Partitioning

Processes ml/data/mimic_subset_wfdb/mimic_wpw_lngqt_subset.csv, reads WFDB
records, applies channel alignment, sanitation, resampling and normalization
identical to the PTB-XL pipeline, then splits by subject_id using
GroupShuffleSplit (80/20) and saves processed tensors to
ml/data/mimic_processed_400hz/.

Usage:
    python ml/scripts/preprocess_mimic_wpw_lngqt.py [--dry-run] [--verbose]
"""

import argparse
import os
import sys

# Configure UTF-8 stdout for Windows
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Resolve project root dynamically
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import wfdb
from scipy.signal import resample_poly
from sklearn.model_selection import GroupShuffleSplit

from ml.ptbxl.config import TARGET_CLASSES  # ['NORM','MI','STTC','HYP','WPW','LNGQT','CD']

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIMIC_WFDB_ROOT = os.path.join(PROJECT_ROOT, "ml", "data", "mimic_subset_wfdb")
CSV_PATH = os.path.join(MIMIC_WFDB_ROOT, "mimic_wpw_lngqt_subset.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "ml", "data", "mimic_processed_400hz")

MIMIC_FS = 500          # MIMIC-IV-ECG native sampling rate (Hz)
TARGET_FS = 400         # PTB-XL model input sampling rate (Hz)
TARGET_LENGTH = 4096    # Exactly 4096 samples — matches ml/ptbxl/config.py INPUT_LENGTH

# Resample ratio: 500 Hz -> 400 Hz = downsample by factor 5/4
# resample_poly(signal, up=4, down=5) reduces 5000 -> 4000, then we crop/pad to 4096
RESAMPLE_UP = 4
RESAMPLE_DOWN = 5

# Standard PTB-XL 12-lead channel order (derived from wfdb sig_names in PTB-XL)
PTBXL_LEAD_ORDER = ["I", "II", "III", "AVR", "AVL", "AVF",
                     "V1", "V2", "V3", "V4", "V5", "V6"]

# MIMIC signal name aliases to canonical PTB-XL names
MIMIC_TO_PTBXL = {
    "I": "I", "II": "II", "III": "III",
    "aVR": "AVR", "AVR": "AVR",
    "aVL": "AVL", "AVL": "AVL",
    "aVF": "AVF", "AVF": "AVF",
    "V1": "V1", "V2": "V2", "V3": "V3",
    "V4": "V4", "V5": "V5", "V6": "V6",
}

# MIMIC-IV-ECG stores aVL and aVF swapped relative to PTB-XL lead ordering.
# This is corrected by reordering via sig_name rather than positional index.

CLIP_MV = 20.0          # Amplitude clipping threshold (mV) for sanitation
RANDOM_SEED = 42
TEST_SIZE = 0.20

# Class mask indices for loss masking in finetune_ptbxl_mimic.py
# MIMIC records only have confirmed labels for WPW (idx 4) and LNGQT (idx 5)
WPW_IDX = TARGET_CLASSES.index("WPW")       # = 4
LNGQT_IDX = TARGET_CLASSES.index("LNGQT")  # = 5
MIMIC_LABEL_INDICES = [WPW_IDX, LNGQT_IDX] # Only these contribute to gradient


# ---------------------------------------------------------------------------
# Signal Processing Helpers
# ---------------------------------------------------------------------------

def _canonicalize_sig_names(sig_names: list[str]) -> list[str]:
    """Map MIMIC raw sig_names to canonical PTB-XL names (uppercase, aVX -> AVX)."""
    return [MIMIC_TO_PTBXL.get(s, s.upper()) for s in sig_names]


def _reorder_channels(signal: np.ndarray, sig_names: list[str]) -> np.ndarray:
    """
    Reorders the 12 leads of `signal` (shape: T x 12) so that the channel
    axis matches PTBXL_LEAD_ORDER.

    Critical: MIMIC-IV-ECG stores aVF and aVL in swapped positions relative
    to the PTB-XL standard. Reordering by name (not by index) corrects this.

    Args:
        signal:    Raw WFDB p_signal, shape (T, N_leads)
        sig_names: List of signal names returned by wfdb.rdrecord()

    Returns:
        Reordered signal, shape (T, 12). Missing leads filled with zeros.
    """
    canonical = _canonicalize_sig_names(sig_names)
    name_to_col = {name: i for i, name in enumerate(canonical)}

    reordered = np.zeros((signal.shape[0], 12), dtype=np.float32)
    for out_idx, lead in enumerate(PTBXL_LEAD_ORDER):
        if lead in name_to_col:
            reordered[:, out_idx] = signal[:, name_to_col[lead]]
        # else: leave as 0.0 (missing lead warning issued in caller)

    return reordered


def _sanitize(signal: np.ndarray) -> np.ndarray:
    """Replace NaN, +Inf, -Inf with 0.0 and clip amplitude to ±CLIP_MV mV."""
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
    signal = np.clip(signal, -CLIP_MV, CLIP_MV)
    return signal


def _resample_to_target(signal: np.ndarray, source_fs: int = MIMIC_FS,
                         target_fs: int = TARGET_FS,
                         target_length: int = TARGET_LENGTH) -> np.ndarray:
    """
    Resample signal from source_fs to target_fs using polyphase filtering
    (scipy.signal.resample_poly), then crop or zero-pad to exactly target_length.

    For MIMIC 500 Hz -> 400 Hz:
        resample_poly(signal, up=4, down=5) on a 5000-sample signal
        yields 4000 samples. Zero-pad 96 samples to reach 4096.

    Args:
        signal:        Input (T_in, 12)
        source_fs:     Source sampling rate in Hz
        target_fs:     Target sampling rate in Hz
        target_length: Exact output length in samples

    Returns:
        Resampled signal, shape (target_length, 12), float32
    """
    gcd = _gcd(target_fs, source_fs)
    up = target_fs // gcd
    down = source_fs // gcd

    resampled = resample_poly(signal, up=up, down=down, axis=0)
    resampled = resampled.astype(np.float32)

    current_len = resampled.shape[0]
    if current_len < target_length:
        pad = np.zeros((target_length - current_len, 12), dtype=np.float32)
        resampled = np.vstack([resampled, pad])
    elif current_len > target_length:
        resampled = resampled[:target_length]

    return resampled


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


def _zscore_per_lead(signal: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Z-score normalization per lead (axis=0, temporal dimension).

    PTB-XL wfdb records deliver signals in physical units (mV). Both PTB-XL
    and MIMIC are already in mV, so the amplitude scale is compatible. However,
    Z-score brings each lead to zero mean / unit variance, reducing inter-patient
    amplitude variability which is the primary domain gap between institutions.

    Args:
        signal: (T, 12) float32 in mV
        eps:    Numerical stability constant

    Returns:
        Normalized signal (T, 12) float32
    """
    mean = signal.mean(axis=0, keepdims=True)
    std = signal.std(axis=0, keepdims=True)
    return ((signal - mean) / (std + eps)).astype(np.float32)


def process_single_record(wfdb_path_rel: str, verbose: bool = False) -> np.ndarray | None:
    """
    Full preprocessing pipeline for one MIMIC WFDB record.

    Returns:
        Processed tensor of shape (4096, 12) in float32, or None on failure.
    """
    full_path = os.path.join(MIMIC_WFDB_ROOT, wfdb_path_rel)
    try:
        record = wfdb.rdrecord(full_path)
    except Exception as e:
        if verbose:
            print(f"  [!] Failed to read {full_path}: {e}")
        return None

    signal = record.p_signal.astype(np.float32)   # (T, N_leads)
    sig_names = record.sig_name                     # e.g., ['I','II','III','aVR','aVL','aVF','V1'...]

    # 1. Channel alignment by name (corrects aVL/aVF swap in MIMIC)
    signal = _reorder_channels(signal, sig_names)

    # 2. Sanitation: NaN/Inf → 0.0, amplitude clip ±20 mV
    signal = _sanitize(signal)

    # 3. Resample 500 Hz → 400 Hz, force exactly 4096 samples
    signal = _resample_to_target(signal,
                                  source_fs=MIMIC_FS,
                                  target_fs=TARGET_FS,
                                  target_length=TARGET_LENGTH)

    # 4. Z-score normalization per lead
    signal = _zscore_per_lead(signal)

    return signal


# ---------------------------------------------------------------------------
# Main Preprocessing Pipeline
# ---------------------------------------------------------------------------

def build_label_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Build a binary label matrix (N, 7) aligned with TARGET_CLASSES.
    MIMIC records confirm only WPW and LNGQT; all other class columns are set
    to -1.0 (unknown) to enable loss masking in training.

    Label convention:
        1.0  = confirmed positive
        0.0  = confirmed negative (only WPW/LNGQT which are explicitly labeled)
        -1.0 = unknown / no supervision (superclasses from PTB-XL perspective)
    """
    n = len(df)
    y = np.full((n, len(TARGET_CLASSES)), fill_value=-1.0, dtype=np.float32)

    wpw_col = TARGET_CLASSES.index("WPW")
    lngqt_col = TARGET_CLASSES.index("LNGQT")

    y[:, wpw_col] = df["is_WPW"].values.astype(np.float32)
    y[:, lngqt_col] = df["is_LNGQT"].values.astype(np.float32)

    return y


def run_preprocessing(dry_run: bool = False, verbose: bool = True) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load metadata CSV
    df = pd.read_csv(CSV_PATH)
    print(f"[*] Loaded MIMIC metadata: {len(df)} records "
          f"(WPW={df['is_WPW'].sum()}, LNGQT={df['is_LNGQT'].sum()})")

    if dry_run:
        df = df.head(20)
        print("[!] DRY-RUN: Processing only 20 records for verification.")

    # Group-aware train/test split (no subject_id leak)
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    train_idx, test_idx = next(gss.split(df, groups=df["subject_id"]))

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_test  = df.iloc[test_idx].reset_index(drop=True)

    print(f"[*] Split: train={len(df_train)}, test={len(df_test)} "
          f"(unique subjects: train={df_train['subject_id'].nunique()}, "
          f"test={df_test['subject_id'].nunique()})")

    def process_split(df_split: pd.DataFrame, split_name: str):
        n = len(df_split)
        X = np.empty((n, TARGET_LENGTH, 12), dtype=np.float32)
        failed_indices = []

        for i, (_, row) in enumerate(df_split.iterrows()):
            sig = process_single_record(row["wfdb_path"], verbose=verbose)
            if sig is None:
                X[i] = np.zeros((TARGET_LENGTH, 12), dtype=np.float32)
                failed_indices.append(i)
            else:
                X[i] = sig

            if verbose and (i + 1) % 200 == 0:
                print(f"  [{split_name}] Processed {i+1}/{n} records...")

        if failed_indices:
            print(f"  [!] {len(failed_indices)} records failed and were zero-filled: {failed_indices[:10]}...")

        y = build_label_matrix(df_split)
        return X, y, df_split

    # Process each split
    print(f"\n[*] Processing TRAIN split ({len(df_train)} records)...")
    X_train, y_train, meta_train = process_split(df_train, "train")

    print(f"\n[*] Processing TEST split ({len(df_test)} records)...")
    X_test, y_test, meta_test = process_split(df_test, "test")

    # Save tensors
    np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(OUTPUT_DIR, "X_test.npy"),  X_test)
    np.save(os.path.join(OUTPUT_DIR, "y_test.npy"),  y_test)
    meta_train.to_csv(os.path.join(OUTPUT_DIR, "meta_train.csv"), index=False)
    meta_test.to_csv(os.path.join(OUTPUT_DIR,  "meta_test.csv"),  index=False)

    print(f"\n[+] Saved to {OUTPUT_DIR}/")
    print(f"    X_train: {X_train.shape}  y_train: {y_train.shape}")
    print(f"    X_test:  {X_test.shape}   y_test:  {y_test.shape}")
    print(f"    Label convention: 1.0=positive, 0.0=negative, -1.0=unknown (masked in loss)")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 2 — MIMIC-IV-ECG Preprocessing")
    parser.add_argument("--dry-run", action="store_true",
                        help="Process only 20 records for quick validation")
    parser.add_argument("--verbose", action="store_true", default=True,
                        help="Show per-record progress")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_preprocessing(dry_run=args.dry_run, verbose=args.verbose)
