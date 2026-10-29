import os
import sys

# Configure UTF-8 stdout for Windows
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# Paths

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

DATA_DIR = os.path.join(PROJECT_ROOT, "ml", "data", "ptbxl")
CHECKPOINTS_DIR = os.path.join(SCRIPT_DIR, "checkpoints")
RUNS_DIR = os.path.join(SCRIPT_DIR, "runs")
PLOTS_DIR = os.path.join(SCRIPT_DIR, "plots")

# Ensure required output directories exist
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
os.makedirs(RUNS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


# Signal & Dataset Settings (aligned with Ribeiro ResNet-1D)

NUM_LEADS = 12
INPUT_LENGTH = 4096       # Exactly 4096 samples to match Ribeiro (16 bins * 320 = 5120 Flatten)
INPUT_SHAPE = (INPUT_LENGTH, NUM_LEADS)  # (4096, 12)
RAW_SAMPLING_RATE = 100   # PTB-XL raw records100 rate (resampled to INPUT_LENGTH)

# Recommended standard PTB-XL split folds
TRAIN_FOLDS = list(range(1, 9))  # 1 to 8
VAL_FOLDS = [9]
TEST_FOLDS = [10]

TARGET_CLASSES = [
    "NORM",
    "MI",
    "STTC",
    "HYP",
    "WPW",
    "LNGQT",
    "CD",
]

BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-3
RANDOM_SEED = 42
