# PG: ECG CDSS DEEP-LEARNING

## 1. Project Overview
This project implements a clinical classification and interpretability solution for 12-lead electrocardiograms (ECG) based on the Ribeiro et al. ResNet-1D architecture.
The pipeline decouples the heavy research ecosystem (TensorFlow/Keras) into a lightweight production runtime (ONNX Runtime) and generates visual attention maps via one-dimensional Class Activation Mapping (1D CAM) rendered on an HTML5 Canvas with standardized medical calibration.

---

## 2. Repository Structure Map

```text
PG/
+-- backend/
|   +-- models/
|   |   +-- ribeiro_dual.onnx       # Compiled model with dual graph (~25.7 MB)
|   |   +-- dense_weights.npy       # Dense layer weight matrix (5120, 6)
|   +-- app/                        # Future FastAPI service
+-- docs/
|   +-- SYSTEM_ARCHITECTURE.md      # This document (context for AI agents)
|   +-- ML_XAI_SPEC.md              # Mathematical and model specification
+-- frontend/
|   +-- spikes/
|       +-- test_canvas.html        # Spike 2: ECG Canvas viewer with thermal CAM
+-- ml/
|   +-- data/
|   |   +-- ecg_tracings.hdf5       # Test dataset (GIT IGNORED)
|   +-- raw/
|   |   +-- model.hdf5              # Original Keras weights (GIT IGNORED)
|   +-- scripts/
|   |   +-- export_dual_onnx.py     # Keras -> Dual ONNX compilation pipeline
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

## 3. Critical Development Rules for AI Agents

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

---

## 4. Local Execution Pipeline

With the virtual environment active (`.venv`):

1. **Compile the model and extract weights:**
   ```powershell
   python ml/scripts/export_dual_onnx.py
   ```
2. **Validate inference and CAM in the console:**
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
