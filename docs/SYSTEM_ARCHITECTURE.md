# PG: ECG CDSS DEEP-LEARNING

## 1. Project Overview

This project implements a **dual-model, hierarchical clinical decision support system (CDSS)** for 12-lead electrocardiograms (ECG). The pipeline consists of:

1. **Primary model (PTB-XL):** A model trained on the PTB-XL dataset that acts as the main classifier and gatekeeper. It emits direct diagnoses for structural/morphological pathologies and evaluates probabilistic thresholds to decide whether the secondary model is needed.
2. **Subordinate model (Ribeiro et al. ResNet-1D):** Activated conditionally by the primary model when rhythm-related thresholds are exceeded. It refines or confirms rhythm diagnoses and is the exclusive source of 1D CAM explainability maps.

The pipeline decouples the heavy research ecosystem (TensorFlow/Keras) into a lightweight production runtime (ONNX Runtime) and generates visual attention maps via one-dimensional Class Activation Mapping (1D CAM) rendered on an HTML5 Canvas with standardized medical calibration.

---

## 2. Dual-Model Inference Architecture

```
[ ECG 12-Lead Input ]
          │
          ▼
[ Primary Model: PTB-XL ]
          │
    ┌─────┴──────────────────────────────────────────────┐
    ▼                                                    ▼
[ Direct Predictions ]                    [ Threshold Evaluation ]
  • MI    • STTC                            • P(CD)    > τ_CD ?
  • HYP   • WPW / LNGQT                    • P(AFIB)  > τ_AFIB ?
    │                                      • P(SBRAD) > τ_SBRAD ?
    │                                      • P(STACH) > τ_STACH ?
    │                                                  │
    │            ┌─────────────────────────────────────┴──────────────────────────────────┐
    │            ▼ (T = True: any threshold exceeded)                                     ▼ (T = False)
    │  [ ONNX Inference: Ribeiro et al. ]                                      [ Ribeiro Bypass ]
    │            │                                                                         │
    │            ├─ Refines: 1dAVb, LBBB, RBBB                                            │
    │            └─ Refines: AF, SB, ST                                                    │
    │                    │                                                                 │
    └────────────────────┼─────────────────────────────────────────────────────────────────┘
                         ▼
            [ Consolidation Engine ]
                         │
                         ▼
         [ 1D CAM Generation / Visualizer ]
```

### 2.1 Model Responsibilities

| Layer     | Model                     | Architecture     | Output Labels                                          | Trigger     |
|-----------|---------------------------|------------------|--------------------------------------------------------|-------------|
| Primary   | PTB-XL + MIMIC (Stage 2)  | Inception-1D     | NORM, MI, STTC, HYP, WPW, LNGQT, CD (7 classes)       | Always      |
| Secondary | Ribeiro ResNet-1D         | ResNet-1D        | 1dAVb, RBBB, LBBB, AF (AFIB), SB (SBRAD), ST (STACH)  | Conditional |

### 2.2 Threshold Gate Logic

The PTB-XL model produces probabilities for rhythm-adjacent classes (`CD`, `AFIB`, `SBRAD`, `STACH`). If **any** probability exceeds its configured threshold τ, the Ribeiro model is invoked. Otherwise the Ribeiro pass is skipped entirely for efficiency.

```python
T = (P_CD > τ_CD) or (P_AFIB > τ_AFIB) or (P_SBRAD > τ_SBRAD) or (P_STACH > τ_STACH)
```

### 2.3 Class Imbalance Considerations

The PTB-XL dataset presents **significant class imbalance** for certain labels. Notable cases:

| Label | Approx. Records | Approx. % of Total | Imbalance Risk |
|-------|----------------|--------------------|----------------|
| WPW   | ~120           | ~0.6%              | **Very High**  |
| LNGQT | Low            | Low                | High           |
| MI    | Moderate       | Moderate           | Medium         |

**Mandatory mitigations for imbalanced labels (especially WPW):**

- **Weighted loss functions:** Use `BCEWithLogitsLoss` with `pos_weight` or focal loss to penalize missed positives in minority classes.
- **Stratified splits:** Train/val/test splits must be stratified, ensuring all classes appear in every partition.
- **Oversampling:** Apply oversampling of minority class records (e.g., signal-level augmentation with noise injection, time-warping, or amplitude scaling).
- **Class-aware evaluation metrics:** Report F1-score per class, AUROC per class, and normalized confusion matrices. Accuracy alone is misleading for rare classes like WPW.
- **Decision threshold tuning:** The classification threshold for rare classes (especially WPW) must be tuned independently on the validation set rather than defaulting to 0.5.

---

## 3. Repository Structure Map

```text
PG/
+-- backend/
|   +-- models/
|   |   +-- ribeiro_dual.onnx       # Compiled Ribeiro model with dual graph (~25.7 MB)
|   |   +-- ptbxl_dual.onnx         # Compiled PTB-XL model with dual graph
|   |   +-- dense_weights.npy       # Ribeiro dense layer weight matrix (5120, 6)
|   |   +-- ptbxl_dense_weights.npy # PTB-XL dense layer weight matrix (5120, C)
|   +-- app/                        # Future FastAPI service
+-- docs/
|   +-- SYSTEM_ARCHITECTURE.md      # This document (context for AI agents)
|   +-- ML_XAI_SPEC.md              # Mathematical and model specification
|   +-- Arch_Base_schema.txt        # Hierarchical architecture flowchart (simplified)
|   +-- Arch_Subordination.txt      # Detailed subordination diagram
+-- frontend/
|   +-- spikes/
|       +-- test_canvas.html        # Spike 2: ECG Canvas viewer with thermal CAM
+-- ml/
|   +-- data/
|   |   +-- ptbxl/                  # PTB-XL WFDB dataset (records100/500, CSVs)
|   |   +-- ecg_tracings.hdf5       # Ribeiro test dataset (GIT IGNORED)
|   |   +-- mimic_subset_wfdb/      # MIMIC-IV-ECG WFDB subset (500 Hz, GIT IGNORED)
|   |   |   +-- mimic_wpw_lngqt_subset.csv  # 2,413 curated records (WPW=413, LNGQT=2022)
|   |   |   +-- files/              # WFDB .hea/.dat records (p1000/.../sXXX/)
|   |   +-- mimic_processed_400hz/  # Preprocessed MIMIC tensors at 400 Hz (GIT IGNORED)
|   |       +-- X_train.npy         # (N_train, 4096, 12) float32
|   |       +-- y_train.npy         # (N_train, 7) labels: 1/0/-1 (unknown)
|   |       +-- X_test.npy          # (N_test, 4096, 12) float32
|   |       +-- y_test.npy          # (N_test, 7) labels: 1/0/-1
|   |       +-- meta_train.csv / meta_test.csv
|   +-- ptbxl/
|   |   +-- config.py               # PTB-XL configuration & paths (INPUT_LENGTH=4096, 7 classes)
|   |   +-- dataset.py              # Metadata loader, PTBXLSequence, class weights
|   |   +-- models.py               # Inception-1D (InceptionTime) + Ribeiro ResNet-1D
|   |   +-- train.py                # Stage 1 training orchestrator (BinaryFocalLoss)
|   |   +-- checkpoints/
|   |       +-- best_model.keras    # Stage 1 checkpoint (PTB-XL only)
|   |       +-- best_model_stage2b.keras # Stage 2 fine-tuned checkpoint (PTB-XL + MIMIC)
|   +-- raw/
|   |   +-- model.hdf5              # Original Ribeiro Keras weights (GIT IGNORED)
|   +-- scripts/
|   |   +-- preprocess_mimic_wpw_lngqt.py # MIMIC signal alignment, resample, 80/20 partition
|   |   +-- finetune_ptbxl_mimic.py       # Two-stage fine-tuning with MaskedBinaryFocalLoss
|   |   +-- evaluate_ptbxl_mimic_gradcam.py # Comparative eval & 1D Grad-CAM visualization
|   |   +-- export_dual_onnx.py           # Keras -> Dual ONNX: Ribeiro model
|   |   +-- export_ptbxl_dual_onnx.py     # Keras -> Dual ONNX: PTB-XL/MIMIC primary model
|   |   +-- export_sample_json.py         # Static JSON fixture generator
|   +-- spikes/
|       +-- test_inference_cam.py   # Spike 1: ONNX validation and CAM inference
+-- samples/
|   +-- patient_1_lbbb.json         # Test fixture (Patient #1 - LBBB)
+-- .gitignore
+-- requirements.txt                # Direct project dependencies
+-- requirements-lock.txt           # Frozen environment snapshot
```

---

## 4. Critical Development Rules for AI Agents

1. **Relative Path Resolution:**
   - Every Python script must compute the project root dynamically:
     ```python
     PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
     ```
   - Never use hardcoded absolute local paths.

2. **Production vs ML Environment:**
   - Inference and CAM computation in production (`backend/`) **must NOT** import TensorFlow or Keras. Only `onnxruntime` and `numpy` are consumed.
   - TensorFlow and `tf2onnx` are only permitted inside `ml/scripts/export_dual_onnx.py`.

3. **Windows Compatibility:**
   - Every script with console output must include:
     ```python
     if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
         sys.stdout.reconfigure(encoding="utf-8")
     ```

4. **Dual-Model Inference Ordering:**
   - The PTB-XL primary model **must always run first**. The Ribeiro model is **never** invoked independently without PTB-XL's gate decision.
   - The consolidation engine must merge outputs from both models before generating the 1D CAM or returning a final diagnosis.

5. **Ribeiro Flatten / Dense Weight Integrity:**
   - The Ribeiro ResNet-1D uses a `Flatten` layer (not GAP) before the dense classification layer.
   - The dense weight matrix is strictly `(5120, 6)`, reshaped to `(16, 320)` for 1D CAM calculations.
   - **Never** change this logic to standard Global Average Pooling.

6. **PTB-XL Primary Model Architecture (Inception-1D with GAP):**
   - The primary model uses `GlobalAveragePooling1D`, making **1D Grad-CAM** the correct explainability method.
   - Feature maps shape: `(N, 1024, 128)`. Dense head: `(128, 7)`.
   - **Never apply Flatten-based CAM or the (5120,6) dense weight reshape to this model.**

7. **MIMIC-IV-ECG Loss Masking:**
   - MIMIC records set `y_true = -1.0` for all columns except WPW (idx=4) and LNGQT (idx=5).
   - The `MaskedBinaryFocalLoss` in `ml/scripts/finetune_ptbxl_mimic.py` masks out `-1.0` entries from gradient computation.
   - **Never train MIMIC records with standard BCE/Focal loss** without this mask — doing so would corrupt the 5 superclass representations.

8. **MIMIC Channel Alignment:**
   - MIMIC-IV-ECG stores `aVL` and `aVF` in swapped positions relative to PTB-XL.
   - Always reorder leads by `sig_name` (not by position index) using `MIMIC_TO_PTBXL` mapping in `ml/scripts/preprocess_mimic_wpw_lngqt.py`.

---

## 5. Local Execution Pipeline

With the virtual environment active (`.venv`):

### 5a. Ribeiro Subordinate Model (Stage 0 — pre-trained)
1. **Compile Ribeiro model to dual ONNX:**
   ```powershell
   python ml/scripts/export_dual_onnx.py
   ```
2. **Validate Ribeiro ONNX inference and CAM:**
   ```powershell
   python ml/spikes/test_inference_cam.py
   ```
3. **Generate patient JSON fixture:**
   ```powershell
   python ml/scripts/export_sample_json.py
   ```

### 5b. PTB-XL Primary Model — Stage 1 Training (Inception-1D)
4. **Train on PTB-XL (folds 1–9, test=fold 10):**
   ```powershell
   python -m ml.ptbxl.train --arch inception1d --epochs 50
   ```

### 5c. PTB-XL + MIMIC Fine-Tuning — Stage 2 (WPW/LNGQT boost)
5. **Preprocess MIMIC-IV-ECG subset (run once, ~5–10 min):**
   ```powershell
   python ml/scripts/preprocess_mimic_wpw_lngqt.py
   ```
6. **Two-stage fine-tuning with loss masking:**
   ```powershell
   python ml/scripts/finetune_ptbxl_mimic.py --stage AB --epochs-a 15 --epochs-b 20
   ```
7. **Comparative evaluation and 1D Grad-CAM plots:**
   ```powershell
   python ml/scripts/evaluate_ptbxl_mimic_gradcam.py
   ```
8. **Export fine-tuned model to dual ONNX (production):**
   ```powershell
   python ml/scripts/export_ptbxl_dual_onnx.py
   ```

### 5d. Frontend
9. **Launch Canvas viewer:**
   ```powershell
   python -m http.server 8000
   # Open: http://localhost:8000/frontend/spikes/test_canvas.html
   ```

---

## 6. Stage 2 Training Pipeline — MIMIC-IV-ECG Integration

### 6.1 Dataset Overview

| Property | PTB-XL (Stage 1) | MIMIC-IV-ECG Subset (Stage 2) |
|---|---|---|
| Records | 21,837 | 2,413 (413 WPW + 2,022 LNGQT) |
| Sampling rate | 500 Hz (records100: 100 Hz) | 500 Hz |
| Input to model | 400 Hz / 4096 samples | 400 Hz / 4096 samples |
| Lead order | Standard PTB-XL | MIMIC (aVL/aVF swapped → corrected by sig_name) |
| Supervision scope | Full 7 classes | WPW + LNGQT only (-1.0 for all others) |
| Split strategy | strat_fold (1-10) | GroupShuffleSplit by subject_id (80/20) |

### 6.2 New Scripts

| Script | Purpose |
|---|---|
| ml/scripts/preprocess_mimic_wpw_lngqt.py | WFDB reader, channel alignment, resample_poly, Z-score, GroupShuffleSplit |
| ml/scripts/finetune_ptbxl_mimic.py | Two-stage fine-tuning: MaskedBinaryFocalLoss, HybridECGSequence, WPW×3 oversample |
| ml/scripts/evaluate_ptbxl_mimic_gradcam.py | PTB-XL + MIMIC eval, threshold sweep, 1D Grad-CAM plots for WPW/LNGQT |

### 6.3 Execution Checklist

`powershell
# 1. Preprocess MIMIC (run once)
python ml/scripts/preprocess_mimic_wpw_lngqt.py

# 2. Fine-tune Stage A + B
python ml/scripts/finetune_ptbxl_mimic.py --stage AB

# 3. Evaluate & generate CAM plots
python ml/scripts/evaluate_ptbxl_mimic_gradcam.py

# 4. Export to dual ONNX (update MODEL_PATH to stage2b checkpoint)
python ml/scripts/export_ptbxl_dual_onnx.py
`

> [!IMPORTANT]
> The fine-tuned checkpoint ml/ptbxl/checkpoints/best_model_stage2b.keras is the **production-ready primary model**.
> The Stage 1 checkpoint ml/ptbxl/checkpoints/best_model.keras is kept as the baseline for comparison.
> Never commit either checkpoint to git (both are GIT IGNORED via ml/ptbxl/checkpoints/.gitkeep pattern).
