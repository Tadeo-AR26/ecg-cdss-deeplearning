# ECG CDSS — Deep Learning Clinical Decision Support System

## Overview

This project builds a **hierarchical Clinical Decision Support System (CDSS)** for automatic multi-label diagnosis of 12-lead electrocardiograms (ECG) using deep learning, with integrated explainability through **1D Class Activation Mapping (1D CAM / Grad-CAM)**.

The system combines two complementary models operating in a hierarchical gating pipeline:

1. **Primary Model (Inception-1D / InceptionTime):**
   - Trained on **PTB-XL** (21,837 records) for multi-label diagnosis of morphological, structural, and rhythm abnormalities across 7 diagnostic classes (`NORM`, `MI`, `STTC`, `HYP`, `CD`, `WPW`, `LNGQT`).
   - Fine-tuned on a curated subset of **MIMIC-IV-ECG** (2,413 unique patients) using two-stage training with loss masking (`MaskedBinaryFocalLoss`) to resolve class imbalance in rare pathologies (`WPW` and `LNGQT`) without degrading the superclasses.
   - Computes **1D Grad-CAM** via Global Average Pooling (GAP) activations to highlight morphological hallmarks (e.g., delta waves in WPW, ST/T repolarization in LNGQT).
   - Acts as an automatic gatekeeper: if rhythm abnormality thresholds are exceeded, it conditionally triggers the secondary model.

2. **Subordinate Model (Ribeiro et al. ResNet-1D):**
   - Based on the architecture and pre-trained weights from Ribeiro et al. (2020), trained on the **CODE-15%** dataset (~400k records) for fine-grained rhythm diagnosis (`1dAVb`, `RBBB`, `LBBB`, `SB`, `AF`, `ST`).
   - Computes **1D CAM attention vectors** directly from the dense weight matrix, adapted to the network's `Flatten` layer (5120 features reshaped to 16 × 320).

Both models are exported to **dual-output ONNX graphs** for lightweight, production-grade inference in `backend/` and render calibrated **thermal heatmaps** on a medical-grade HTML5 ECG Canvas (25 mm/s, 10 mm/mV).

---

## Reference Models & Datasets

This system integrates three primary datasets and architectures:

### 1. PTB-XL Dataset
- **Reference:** Wagner, P., et al. (2020). *PTB-XL, a large publicly available electrocardiography dataset.* Scientific Data, 7, 154. https://doi.org/10.1038/s41597-020-0495-6
- **Data:** 21,837 clinical 12-lead ECGs (10-second recordings) covering diagnostic superclasses and rare arrhythmic conditions. Standard 10-fold split (`strat_fold` 1–8 train, 9 val, 10 test).
- **Location:** `ml/data/ptbxl/` (records resampled to 400 Hz / 4096 samples).

### 2. MIMIC-IV-ECG Curated Subset (Stage 2 Fine-Tuning)
- **Reference:** Gow, B., et al. (2023). *MIMIC-IV-ECG: Diagnostic Electrocardiogram Matched Subset.* PhysioNet. https://doi.org/10.13026/4nqg-sb35
- **Data:** 2,413 unique patients with confirmed rare conditions:
  - **413 WPW records** (Wolff-Parkinson-White pattern).
  - **2,022 LNGQT records** (Long QT pattern: QTc Bazett $\ge$ 480 ms, QRS $<$ 120 ms).
- **Preprocessing:** Native 500 Hz polyphase resampled to 400 Hz (4096 samples), channel realignment to fix swapped `aVF`/`aVL` leads, per-lead Z-score normalization, and patient-grouped 80/20 train/test split (`GroupShuffleSplit` by `subject_id`).
- **Location:** `ml/data/mimic_subset_wfdb/` $\to$ processed tensors in `ml/data/mimic_processed_400hz/`.

### 3. Ribeiro et al. ResNet-1D & CODE-15% Dataset
- **Reference:** Ribeiro, A. H., et al. (2020). *Automatic diagnosis of the 12-lead ECG using a deep neural network.* Nature Communications, 11, 1760. https://doi.org/10.1038/s41467-020-15432-4
- **Data & Weights:** Trained on ~400k ECGs from the CODE study (`model.hdf5`, `ecg_tracings.hdf5`), distributed by the authors under their own terms (**not included in this repository**).

---

## Diagnostic Classes

The dual-model system outputs independent sigmoid probabilities across two diagnostic tiers:

### Primary Model (PTB-XL + MIMIC-IV-ECG Fine-Tuning)

| Class   | Full Name                         | Focus / Origin                           |
|---------|-----------------------------------|------------------------------------------|
| `NORM`  | Normal ECG                        | Diagnostic superclass (PTB-XL)           |
| `MI`    | Myocardial Infarction             | Structural superclass (PTB-XL)           |
| `STTC`  | ST/T-Change                       | Ischemia/repolarization (PTB-XL)         |
| `HYP`   | Hypertrophy                       | Structural superclass (PTB-XL)           |
| `CD`    | Conduction Disturbance            | Conduction block (gating trigger)        |
| `WPW`   | Wolff-Parkinson-White             | Rare class boosted by MIMIC-IV-ECG       |
| `LNGQT` | Long QT Syndrome                  | Rare class boosted by MIMIC-IV-ECG       |

### Subordinate Model (Ribeiro et al. ResNet-1D)

Triggered conditionally when rhythm-adjacent probabilities from the primary model exceed configured threshold gates ($\tau$):

| Code    | Condition                     |
|---------|-------------------------------|
| `1dAVb` | First-degree AV block         |
| `RBBB`  | Right bundle branch block     |
| `LBBB`  | Left bundle branch block      |
| `SB`    | Sinus bradycardia             |
| `AF`    | Atrial fibrillation           |
| `ST`    | Sinus tachycardia             |

---

## Architecture

```
[ 12-Lead ECG Input (400 Hz, 4096 samples) ]
                    |
                    v
    +-------------------------------+
    |   Primary Model: Inception-1D |
    |   (PTB-XL + MIMIC-IV-ECG)     |
    +-------------------------------+
                    |
          +---------+--------------------+
          v                              v
    [ Diagnoses ]              [ Threshold Gating ]
    NORM, MI, STTC, HYP,       P(CD), P(AFIB), P(SBRAD), P(STACH) > tau ?
    WPW, LNGQT                           |
          |                  +-----------+-----------+
          |                  | T = True              | T = False
          |                  v                       v
          |      +------------------------+      [ Bypass ]
          |      | Ribeiro ResNet-1D ONNX |          |
          |      | 1dAVb, RBBB, LBBB,     |          |
          |      | AF, SB, ST             |          |
          |      +------------------------+          |
          |                  |                       |
          +------------------+-----------------------+
                             |
                             v
               [ Consolidation & XAI Engine ]
            - Primary: 1D Grad-CAM (GAP-based)
            - Ribeiro: 1D CAM (Dense weight reshape)
                             |
                             v
               [ HTML5 Medical Canvas (25 mm/s, 10 mm/mV) ]
```

---

## Repository Structure

```
PG/
+-- backend/
|   +-- models/
|   |   +-- ptbxl_dual.onnx         # Primary model ONNX graph (Inception-1D)
|   |   +-- ptbxl_dense_weights.npy # Primary model dense weights
|   |   +-- ribeiro_dual.onnx       # Subordinate model ONNX graph (ResNet-1D)
|   |   +-- dense_weights.npy       # Ribeiro dense weight matrix (5120, 6)
|   +-- app/                        # [TODO] FastAPI service
+-- docs/
|   +-- SYSTEM_ARCHITECTURE.md      # Dual-model design specification and rules
|   +-- ML_XAI_SPEC.md              # Mathematical specification of ML and XAI
+-- frontend/
|   +-- spikes/
|       +-- test_canvas.html        # Spike: ECG Canvas viewer with thermal CAM
+-- ml/
|   +-- data/
|   |   +-- ptbxl/                  # PTB-XL dataset (WFDB records & CSVs) — GIT IGNORED
|   |   +-- mimic_subset_wfdb/      # MIMIC-IV-ECG subset WFDB records — GIT IGNORED
|   |   +-- mimic_processed_400hz/  # Preprocessed 400 Hz MIMIC tensors — GIT IGNORED
|   |   +-- ecg_tracings.hdf5       # Ribeiro test tracings — GIT IGNORED
|   +-- ptbxl/
|   |   +-- config.py               # PTB-XL & pipeline settings (4096 pts, 7 classes)
|   |   +-- dataset.py              # Sequence generator, loaders & class balancing
|   |   +-- models.py               # Inception-1D & ResNet-1D architectures
|   |   +-- train.py                # Stage 1 training orchestrator (PTB-XL)
|   |   +-- checkpoints/            # Model checkpoints (.keras) — GIT IGNORED
|   +-- raw/
|   |   +-- model.hdf5              # Original Ribeiro Keras weights — GIT IGNORED
|   +-- scripts/
|   |   +-- preprocess_mimic_wpw_lngqt.py   # MIMIC alignment, resample & partition
|   |   +-- finetune_ptbxl_mimic.py         # Two-stage fine-tuning with loss masking
|   |   +-- evaluate_ptbxl_mimic_gradcam.py # Comparative eval & 1D Grad-CAM
|   |   +-- export_ptbxl_dual_onnx.py       # Keras -> Dual ONNX: Primary model
|   |   +-- export_dual_onnx.py             # Keras -> Dual ONNX: Ribeiro model
|   |   +-- export_sample_json.py           # Generates JSON fixtures for frontend
|   +-- spikes/
|       +-- test_inference_cam.py   # Validates ONNX inference and CAM computation
+-- samples/
|   +-- patient_1_lbbb.json         # Test fixture: Patient #1 (LBBB)
+-- requirements.txt
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- **For Subordinate Model:** Original weights (`ml/raw/model.hdf5`) and tracings (`ml/data/ecg_tracings.hdf5`) from Ribeiro et al.
- **For Primary Model (Stage 1):** PTB-XL dataset placed in `ml/data/ptbxl/`
- **For Fine-Tuning (Stage 2):** Curated MIMIC-IV-ECG subset placed in `ml/data/mimic_subset_wfdb/`

### Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Run the ML pipeline

#### 1. Stage 1 — Train Base Primary Model on PTB-XL
```powershell
python -m ml.ptbxl.train --arch inception1d --epochs 50
```

#### 2. Stage 2 — Fine-Tune on MIMIC-IV-ECG (WPW & LNGQT Boost)
```powershell
# A. Preprocess and partition MIMIC-IV-ECG subset to 400 Hz (4096 samples)
python ml/scripts/preprocess_mimic_wpw_lngqt.py

# B. Two-stage fine-tuning with masked loss (Stage A: head only, Stage B: last block)
python ml/scripts/finetune_ptbxl_mimic.py --stage AB --epochs-a 15 --epochs-b 20

# C. Comparative evaluation (PTB-XL test set + MIMIC hold-out) and 1D Grad-CAM
python ml/scripts/evaluate_ptbxl_mimic_gradcam.py
```

#### 3. Compile Production ONNX Graphs
```powershell
# Export primary Inception-1D model (prefers Stage 2 fine-tuned checkpoint)
python ml/scripts/export_ptbxl_dual_onnx.py

# Export subordinate Ribeiro ResNet-1D model
python ml/scripts/export_dual_onnx.py
```

#### 4. Frontend Spike Viewer
```powershell
# Generate test patient JSON fixture
python ml/scripts/export_sample_json.py

# Launch Canvas viewer
python -m http.server 8000
# Open: http://localhost:8000/frontend/spikes/test_canvas.html
```

---

## Technical Notes

- **Production runtime constraint:** `backend/` code uses strictly `onnxruntime` and `numpy`. TensorFlow and Keras are strictly limited to offline training and ONNX export scripts in `ml/scripts/`.
- **Explainability mechanics:**
  - **Primary Model (Inception-1D):** Uses `GlobalAveragePooling1D` (GAP) and **1D Grad-CAM** via backpropagation of class logits into temporal convolutional activations.
  - **Subordinate Model (Ribeiro ResNet-1D):** Uses a `Flatten` layer (not GAP). Dense weight matrix is strictly `(5120, 6)`, reshaped to `(16, 320)` for 1D CAM computation. Never change this logic to standard GAP.
- **MIMIC Loss Masking (`MaskedBinaryFocalLoss`):** MIMIC records only provide confirmed labels for `WPW` and `LNGQT`. Non-supervised superclasses are marked with `-1.0` and excluded from gradient calculation during fine-tuning to prevent representation drift on the 5 PTB-XL superclasses.
- **MIMIC Channel Alignment:** MIMIC-IV-ECG stores `aVF` and `aVL` inverted relative to PTB-XL. Signals are explicitly remapped by `sig_name` to standard lead order (`I, II, III, AVR, AVL, AVF, V1–V6`).
- **ECG Canvas calibration:** Standard medical calibration: 1 mm = 4 px, sweep speed 25 mm/s (0.04 s/small box), voltage gain 10 mm/mV (0.1 mV/small box). Non-linear contrast expansion is applied to CAM vectors to emphasize diagnostic wave deviations.

---

## License

This project is for academic and research purposes. Dataset access and original model weights are governed by their respective licenses:
- **PTB-XL & MIMIC-IV-ECG:** PhysioNet Credentialed / Open Health Data License.
- **Ribeiro et al. Model:** Refer to their [official repository](https://github.com/antonior92/automatic-ecg-diagnosis).
