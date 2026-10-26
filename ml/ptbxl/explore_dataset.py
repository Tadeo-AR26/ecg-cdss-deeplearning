"""
PTB-XL Dataset Explorer
=======================
Performs an initial exploratory data analysis (EDA) of the PTB-XL dataset.

Expected dataset location:
    ml/data/ptbxl/
        records100/         ECG recordings at 100 Hz (WFDB format)
        records500/         ECG recordings at 500 Hz (WFDB format)
        ptbxl_database.csv  Patient metadata and SCP diagnostic labels
        scp_statements.csv  SCP code dictionary (description, diagnostic class)

Download the dataset from PhysioNet:
    https://physionet.org/content/ptb-xl/1.0.3/

Usage (from project root, with .venv active):
    python ml/ptbxl/explore_dataset.py
    python ml/ptbxl/explore_dataset.py --sampling-rate 500
    python ml/ptbxl/explore_dataset.py --save-plots
"""

import argparse
import ast
import os
import sys

# ------------------------------------------------------------
# Windows UTF-8 console fix
# ------------------------------------------------------------
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ------------------------------------------------------------
# Project root resolution
# ------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PLOTS_DIR = os.path.join(PROJECT_ROOT, "ml", "ptbxl", "plots")

# Auto-detect PTB-XL root: supports both flat layout and the wget2 -r recursive
# layout (physionet.org/files/ptb-xl/1.0.3/) created when downloading with:
#   wget2 -r -N -c -np https://physionet.org/files/ptb-xl/1.0.3/
_PTBXL_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "ml", "data", "ptbxl"),
    os.path.join(PROJECT_ROOT, "ml", "data", "ptbxl",
                 "physionet.org", "files", "ptb-xl", "1.0.3"),
]
PTBXL_DIR = next(
    (p for p in _PTBXL_CANDIDATES
     if os.path.exists(os.path.join(p, "ptbxl_database.csv"))),
    _PTBXL_CANDIDATES[0],  # Default shown in error messages
)

# ------------------------------------------------------------
# Imports (non-stdlib)
# ------------------------------------------------------------
try:
    import numpy as np
    import pandas as pd
    import wfdb
    import matplotlib
    matplotlib.use("Agg")  # Headless rendering — safe on Windows without a display
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
except ImportError as e:
    print(f"[ERROR] Missing dependency: {e}")
    print("  Run: pip install -r requirements.txt")
    sys.exit(1)


# ============================================================
# Helpers
# ============================================================

def check_dataset_present() -> bool:
    """Verify that the minimum required PTB-XL files are present."""
    required = [
        os.path.join(PTBXL_DIR, "ptbxl_database.csv"),
        os.path.join(PTBXL_DIR, "scp_statements.csv"),
    ]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        print("\n[!] PTB-XL dataset not found. Missing files:")
        for p in missing:
            print(f"    {p}")
        print(
            "\n    Download from PhysioNet:\n"
            "    https://physionet.org/content/ptb-xl/1.0.3/\n"
            f"    and extract into: {PTBXL_DIR}\n"
        )
        return False
    return True


def load_metadata() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load ptbxl_database.csv and scp_statements.csv."""
    db_path = os.path.join(PTBXL_DIR, "ptbxl_database.csv")
    scp_path = os.path.join(PTBXL_DIR, "scp_statements.csv")

    df = pd.read_csv(db_path, index_col="ecg_id")
    # scp_codes column is stored as a string-encoded dict — parse it
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)

    scp = pd.read_csv(scp_path, index_col=0)
    return df, scp


def decode_labels(df: pd.DataFrame, scp: pd.DataFrame, threshold: float = 0.0) -> pd.DataFrame:
    """
    Expand the scp_codes dict into a one-hot-style indicator column per code.
    Only keeps codes that are flagged as 'diagnostic' in scp_statements.csv.
    """
    diagnostic_codes = scp[scp["diagnostic"] == 1].index.tolist()

    def extract(codes: dict) -> dict:
        return {code: likelihood for code, likelihood in codes.items()
                if code in diagnostic_codes and likelihood > threshold}

    df["diagnostic_labels"] = df["scp_codes"].apply(extract)
    return df


def load_waveform(row: pd.Series, sampling_rate: int = 100) -> tuple[np.ndarray, wfdb.Record]:
    """Load a single ECG waveform using wfdb."""
    # The filename field in the CSV already includes 'records100/' or 'records500/'
    col = "filename_lr" if sampling_rate == 100 else "filename_hr"
    filename = os.path.join(PTBXL_DIR, row[col])
    record = wfdb.rdrecord(filename)
    signal = record.p_signal  # shape: (n_samples, 12)
    return signal.astype(np.float32), record


# ============================================================
# Report sections
# ============================================================

def report_overview(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("  PTB-XL DATASET — OVERVIEW")
    print("=" * 60)
    print(f"  Total records       : {len(df):,}")
    print(f"  Unique patients     : {df['patient_id'].nunique():,}")
    print(f"  Age  — mean / std   : {df['age'].mean():.1f} ± {df['age'].std():.1f} years")
    print(f"  Sex  — Male / Female: {(df['sex'] == 0).sum():,} / {(df['sex'] == 1).sum():,}")
    print(f"  Validation folds    : {sorted(df['strat_fold'].unique())}")
    print(f"  Devices             : {df['device'].nunique()} unique models")
    print()


def report_signal_specs(df: pd.DataFrame, sampling_rate: int) -> None:
    print("=" * 60)
    print("  SIGNAL SPECIFICATIONS")
    print("=" * 60)
    print(f"  Sampling rate requested : {sampling_rate} Hz")
    duration = 10  # PTB-XL records are always 10 seconds
    n_samples = sampling_rate * duration
    print(f"  Duration                : {duration} s")
    print(f"  Expected samples/lead   : {n_samples}")
    print(f"  Leads                   : 12 (standard)")
    print(f"  Input tensor shape      : (N, {n_samples}, 12)")
    print()


def report_diagnostic_distribution(df: pd.DataFrame, scp: pd.DataFrame, top_n: int = 20) -> None:
    print("=" * 60)
    print(f"  DIAGNOSTIC LABEL DISTRIBUTION (top {top_n})")
    print("=" * 60)

    # Flatten all SCP codes across records
    from collections import Counter
    code_counts: Counter = Counter()
    for codes in df["scp_codes"]:
        for code, likelihood in codes.items():
            if likelihood > 0:
                code_counts[code] += 1

    diagnostic_codes = scp[scp["diagnostic"] == 1].index
    diag_counts = {k: v for k, v in code_counts.items() if k in diagnostic_codes}
    diag_counts_sorted = sorted(diag_counts.items(), key=lambda x: x[1], reverse=True)

    print(f"  {'Code':<12} {'Count':>7}  {'% Records':>10}  Description")
    print(f"  {'-'*12} {'-'*7}  {'-'*10}  {'-'*35}")
    for code, count in diag_counts_sorted[:top_n]:
        desc = scp.loc[code, "description"] if code in scp.index else "—"
        pct = count / len(df) * 100
        print(f"  {code:<12} {count:>7,}  {pct:>9.1f}%  {desc}")
    print()


def report_superclass_distribution(df: pd.DataFrame, scp: pd.DataFrame) -> None:
    print("=" * 60)
    print("  SUPERCLASS DISTRIBUTION")
    print("=" * 60)
    from collections import Counter

    superclass_col = None
    for candidate in ["diagnostic_class", "diagnostic_superclass"]:
        if candidate in scp.columns:
            superclass_col = candidate
            break

    if superclass_col is None:
        print("  [!] 'diagnostic_class' / 'diagnostic_superclass' column not found in scp_statements.csv")
        return

    code_to_super = scp[superclass_col].dropna().to_dict()
    super_counts: Counter = Counter()
    for codes in df["scp_codes"]:
        seen = set()
        for code, likelihood in codes.items():
            if likelihood > 0 and code in code_to_super:
                sup = code_to_super[code]
                if sup not in seen:
                    super_counts[sup] += 1
                    seen.add(sup)

    for sup, count in super_counts.most_common():
        pct = count / len(df) * 100
        print(f"  {sup:<25} {count:>7,}  ({pct:.1f}%)")
    print()


def report_sample_waveform(df: pd.DataFrame, sampling_rate: int, save: bool) -> None:
    """Load the first record and display its 12-lead ECG."""
    print("=" * 60)
    print("  SAMPLE WAVEFORM (record #1)")
    print("=" * 60)

    row = df.iloc[0]
    try:
        signal, record = load_waveform(row, sampling_rate)
    except FileNotFoundError:
        print(f"  [!] Waveform files not found in ml/data/ptbxl/records{sampling_rate}/")
        print("      Download the full dataset including the records folders.")
        return

    print(f"  Signal shape   : {signal.shape}  (samples × leads)")
    print(f"  Sampling rate  : {record.fs} Hz")
    print(f"  Lead names     : {record.sig_name}")
    print(f"  Units          : {record.units}")
    print(f"  Amplitude range: [{signal.min():.4f}, {signal.max():.4f}] mV")
    print(f"  NaN values     : {np.isnan(signal).sum()}")

    scp_raw = row["scp_codes"]
    top_codes = sorted(scp_raw.items(), key=lambda x: x[1], reverse=True)[:3]
    print(f"  SCP codes      : {', '.join(f'{c}({v:.0f}%)' for c, v in top_codes)}")

    if save:
        _plot_sample_waveform(signal, record, row)


def _plot_sample_waveform(signal: np.ndarray, record: "wfdb.Record", row: pd.Series) -> None:
    """Save a 12-lead ECG plot to ml/ptbxl/plots/."""
    os.makedirs(PLOTS_DIR, exist_ok=True)

    lead_names = record.sig_name
    n_leads = signal.shape[1]
    t = np.arange(signal.shape[0]) / record.fs  # Time axis in seconds

    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor("#0d1117")
    gs = gridspec.GridSpec(n_leads, 1, hspace=0.08)

    for i, lead in enumerate(lead_names):
        ax = fig.add_subplot(gs[i])
        ax.plot(t, signal[:, i], color="#58a6ff", linewidth=0.8, alpha=0.92)
        ax.set_facecolor("#161b22")
        ax.set_ylabel(lead, color="#c9d1d9", fontsize=8, rotation=0,
                      labelpad=28, va="center")
        ax.tick_params(colors="#8b949e", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")
        ax.set_xlim(t[0], t[-1])
        if i < n_leads - 1:
            ax.set_xticklabels([])

    axes = fig.get_axes()
    axes[-1].set_xlabel("Time (s)", color="#c9d1d9", fontsize=9)

    scp_raw = row["scp_codes"]
    top_label = max(scp_raw, key=scp_raw.get) if scp_raw else "—"
    fig.suptitle(
        f"PTB-XL — Record #{row.name}  |  Lead: {top_label}  |  "
        f"Age: {row.get('age', '?')}  Sex: {'M' if row.get('sex', 0) == 0 else 'F'}",
        color="#e6edf3", fontsize=11, y=0.995
    )

    out_path = os.path.join(PLOTS_DIR, f"sample_record_{row.name}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\n  [+] Plot saved to: {out_path}")


# ============================================================
# Entry point
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="PTB-XL EDA script")
    parser.add_argument(
        "--sampling-rate", type=int, choices=[100, 500], default=100,
        help="Sampling rate to use for waveform loading (default: 100 Hz)"
    )
    parser.add_argument(
        "--save-plots", action="store_true",
        help="Save EDA plots to ml/ptbxl/plots/"
    )
    parser.add_argument(
        "--top-n", type=int, default=20,
        help="Number of top diagnostic codes to display (default: 20)"
    )
    args = parser.parse_args()

    print(f"\n[*] PTB-XL directory : {PTBXL_DIR}")

    if not check_dataset_present():
        sys.exit(1)

    print("[*] Loading metadata...")
    df, scp = load_metadata()
    df = decode_labels(df, scp)

    report_overview(df)
    report_signal_specs(df, args.sampling_rate)
    report_diagnostic_distribution(df, scp, top_n=args.top_n)
    report_superclass_distribution(df, scp)
    report_sample_waveform(df, args.sampling_rate, save=args.save_plots)

    print("=" * 60)
    print("  EDA complete.")
    if not args.save_plots:
        print("  Tip: run with --save-plots to export a 12-lead ECG figure.")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
