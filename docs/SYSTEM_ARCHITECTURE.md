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

| Layer     | Model              | Output Labels                                          | Trigger     |
|-----------|--------------------|--------------------------------------------------------|-------------|
| Primary   | PTB-XL             | MI, STTC, HYP, WPW, LNGQT                             | Always      |
| Secondary | Ribeiro ResNet-1D  | 1dAVb, RBBB, LBBB, AF (AFIB), SB (SBRAD), ST (STACH)  | Conditional |

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
|   |   +-- ptbxl_primary.onnx      # Compiled PTB-XL primary model (planned)
|   |   +-- dense_weights.npy       # Ribeiro dense layer weight matrix (5120, 6)
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
|   |   +-- ecg_tracings.hdf5       # Test dataset (GIT IGNORED)
|   +-- raw/
|   |   +-- model.hdf5              # Original Ribeiro Keras weights (GIT IGNORED)
|   +-- scripts/
|   |   +-- export_dual_onnx.py     # Keras -> Dual ONNX compilation pipeline (Ribeiro)
|   |   +-- export_sample_json.py   # Static JSON fixture generator
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

---

## 5. Local Execution Pipeline

With the virtual environment active (`.venv`):

1. **Compile the Ribeiro model and extract weights:**
   ```powershell
   python ml/scripts/export_dual_onnx.py
   ```
2. **Validate Ribeiro inference and CAM in the console:**
   ```powershell
   python ml/spikes/test_inference_cam.py
   ```
3. **Generate patient JSON fixture:**
   ```powershell
   python ml/scripts/export_sample_json.py
   ```
4. **Launch Canvas viewer:**
   ```powershell
   python -m http.server 8000
   # Open: http://localhost:8000/frontend/spikes/test_canvas.html
   ```
