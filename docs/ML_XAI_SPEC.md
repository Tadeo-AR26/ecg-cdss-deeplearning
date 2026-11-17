# Machine Learning and XAI Technical Specification

## 0. Dual-Model Pipeline Overview

The system operates a **hierarchical two-model pipeline**. Both models run under `onnxruntime`; neither imports TensorFlow or Keras in production.

| Property              | Primary Model (PTB-XL)                                              | Subordinate Model (Ribeiro ResNet-1D)                |
|-----------------------|---------------------------------------------------------------------|------------------------------------------------------|
| **Training Dataset**  | PTB-XL (21,837 records) + **MIMIC-IV-ECG subset** (2,413 records)  | CODE-15% (≈400k records, 12-lead, 400 Hz)            |
| **Output labels**     | NORM, MI, STTC, HYP, WPW, LNGQT, CD (7 classes)                   | 1dAVb, RBBB, LBBB, AF, SB, ST                       |
| **Role**              | Always runs; direct diagnosis + threshold gating                    | Conditional; refines rhythm when T = True            |
| **Architecture**      | **Inception-1D (InceptionTime)** with GAP → dual ONNX              | Ribeiro ResNet-1D (same backbone, dual ONNX)         |
| **Explainability**    | **1D Grad-CAM** (GAP-based; see §3)                                 | 1D CAM (Flatten-based; see §3)                       |
| **Invocation rule**   | Mandatory on every ECG input                                        | Only when P_CD, P_AFIB, P_SBRAD, or P_STACH > τ     |
| **Training stages**   | Stage 1: PTB-XL only. **Stage 2: Fine-tune with MIMIC-IV-ECG**     | Pre-trained external weights (Ribeiro et al.)        |

### Gate condition (evaluated after PTB-XL inference):

```python
T = (P_CD > τ_CD) or (P_AFIB > τ_AFIB) or (P_SBRAD > τ_SBRAD) or (P_STACH > τ_STACH)
# T = True  → Run Ribeiro ONNX inference
# T = False → Skip Ribeiro; finalize with PTB-XL outputs only
```

---


## 1. PTB-XL Primary Model Architecture (Inception-1D / InceptionTime)

> [!IMPORTANT]
> The PTB-XL primary model uses **Inception-1D (InceptionTime)**, NOT Ribeiro ResNet-1D.
> It uses `GlobalAveragePooling1D` (GAP), making Grad-CAM the correct explainability method (not Flatten-based CAM).
> Never apply the Flatten/dense-weight reshape logic from §3 to this model.

- **Sampling rate:** 400 Hz (PTB-XL records resampled from 500 Hz; MIMIC resampled from 500 Hz via `resample_poly`)
- **Recording duration:** 10.24 seconds (**4096 samples** — `INPUT_LENGTH` in `config.py`)
- **Input channels:** 12 standard ECG leads
- **Input tensor:** Shape `(N, 4096, 12)`
- **Target classes (7, multi-label Sigmoid):**
  1. `NORM` — Normal ECG
  2. `MI` — Myocardial Infarction
  3. `STTC` — ST/T-Change
  4. `HYP` — Hypertrophy
  5. `WPW` — Wolff-Parkinson-White syndrome *(rare; boosted via MIMIC fine-tuning)*
  6. `LNGQT` — Long QT syndrome *(rare; boosted via MIMIC fine-tuning)*
  7. `CD` — Conduction Disturbance / AV block

- **Architecture:** 2 Inception Blocks × 3 Inception Modules each (6 total). Temporal reduction 4096 → 2048 → 1024.
  - Feature maps before GAP: `(N, 1024, 128)` ← **CAM source layer `feature_maps`**
  - GAP output: `(N, 128)`
  - Dense head: `(N, 7)` with Sigmoid

- **Training stages:**
  - **Stage 1 (PTB-XL only):** `ml/ptbxl/train.py` → checkpoint `ml/ptbxl/checkpoints/best_model.keras`
  - **Stage 2 (MIMIC fine-tune):** `ml/scripts/finetune_ptbxl_mimic.py` → checkpoint `ml/ptbxl/checkpoints/best_model_stage2b.keras`

- **Checkpoint files:** `ml/ptbxl/checkpoints/best_model.keras` (Stage 1) · `ml/ptbxl/checkpoints/best_model_stage2b.keras` (Stage 2)

---

## 1b. Ribeiro ResNet-1D Architecture (Subordinate Model)

- **Sampling rate:** 400 Hz
- **Recording duration:** 10.24 seconds (4096 time samples)
- **Input channels:** 12 standard ECG leads
- **Input tensor:** Shape `(N, 4096, 12)`
- **Diagnostic classes (6 independent outputs with Sigmoid activation):**
  1. `1dAVb` (First-degree AV block)
  2. `RBBB` (Right bundle branch block)
  3. `LBBB` (Left bundle branch block)
  4. `SB` (Sinus bradycardia)
  5. `AF` (Atrial fibrillation)
  6. `ST` (Sinus tachycardia)

---

## 2. Dual ONNX Output Files

### 2a. `ribeiro_dual.onnx` (Subordinate Model)

The compiled Ribeiro ONNX graph has one input and two simultaneous outputs:
* **Input:** `(1, 4096, 12)` — Float32 signal tensor.
* **Output 0 (`probabilities`):** `(1, 6)` — Sigmoid classification probabilities.
* **Output 1 (`feature_maps`):** `(1, 16, 320)` — Temporal activations from the last convolutional layer (`conv1d_11`).

### 2b. `ptbxl_dual.onnx` (Primary Model — Inception-1D)

* **Input:** `(1, 4096, 12)` — Float32 signal tensor.
* **Output 0 (`probabilities`):** `(1, 7)` — Sigmoid classification probabilities (NORM, MI, STTC, HYP, WPW, LNGQT, CD).
* **Output 1 (`feature_maps`):** `(1, 1024, 128)` — Temporal activations before GAP (Inception-1D last block output).
* **Dense weights:** `backend/models/ptbxl_dense_weights.npy` shape `(128, 7)` — used for standard CAM if needed.

Export script: `ml/scripts/export_ptbxl_dual_onnx.py`

---

## 3. Explainability: 1D Grad-CAM (Primary) & 1D CAM (Subordinate)

> [!WARNING]
> **The two models use different explainability methods.** Never cross-apply them.
>
> | Model | Architecture | Pooling | XAI Method | Feature Maps Shape |
> |---|---|---|---|---|
> | PTB-XL Primary (Inception-1D) | InceptionTime | **GAP** | **1D Grad-CAM** | `(N, 1024, 128)` |
> | Ribeiro Subordinate (ResNet-1D) | Ribeiro et al. | **Flatten** | **1D CAM (Dense weights)** | `(N, 16, 320)` |

### 3a. 1D Grad-CAM for Inception-1D Primary Model (PTB-XL/MIMIC)

For a target class $c$ (e.g., $c = 4$ for WPW):

1. **Forward pass** with `GradientTape` recording `feature_maps` (shape `(1, 1024, 128)`).

2. **Gradient computation:** Gradient of the class logit $\log\left(\frac{p_c}{1-p_c}\right)$ w.r.t. feature maps:
   $$\frac{\partial \text{logit}_c}{\partial A^{(l)}} \in \mathbb{R}^{1024 \times 128}$$

3. **Global-average-pool gradients** (importance weight per channel):
   $$\alpha_k^c = \frac{1}{T}\sum_{t=1}^{T}\frac{\partial \text{logit}_c}{\partial A^{(l)}[t, k]}, \quad T=1024$$

4. **Weighted sum with ReLU:**
   $$\text{Grad-CAM}[t] = \text{ReLU}\!\left(\sum_{k=1}^{128} \alpha_k^c \cdot A^{(l)}[t, k]\right)$$

5. **Interpolation to 4096 signal points:**
   $$\text{Grad-CAM}_{4096} = \text{interp}(\text{Grad-CAM}_{1024}, 1024 \to 4096)$$

**Clinical validation targets:**
- `WPW`: activation expected over delta wave / shortened PR (early QRS region)
- `LNGQT`: activation expected over ST segment / T-wave prolongation (late repolarization)

### 3b. 1D CAM for Ribeiro ResNet-1D (Subordinate Model — Flatten-based)

> [!CAUTION]
> The Ribeiro network does **NOT** use GAP. It uses a direct `Flatten` layer.
> The dense weight matrix has shape `(5120, 6)` which is reshaped to `(16, 320)`.
> **Never change this logic to Grad-CAM or GAP-CAM.**

For a target class $c$ (e.g., $c = 2$ for LBBB):

1. **Weight vector isolation:**
   $$W_c \in \mathbb{R}^{5120}$$

2. **Temporal reshape:**
   $$W_c \xrightarrow{\text{reshape}} \mathbb{R}^{16 \times 320}$$

3. **Channel-wise multiplication and summation:**
   $$\text{CAM}_{\text{raw}}[t] = \sum_{k=1}^{320} A[t, k] \cdot W_c[t, k], \quad \forall t \in [0, 15]$$

4. **ReLU rectification:**
   $$\text{CAM}_{\text{relu}}[t] = \max(0, \text{CAM}_{\text{raw}}[t])$$

5. **Linear Interpolation to Full Scale:**
   $$\text{CAM}_{4096} = \text{interp}(\text{CAM}_{\text{relu}}, 16 \to 4096)$$

---

## 4. JSON Data Contract (`samples/patient_1_lbbb.json`)

```json
{
  "metadata": {
    "sampling_rate_hz": 400,
    "num_samples": 4096,
    "duration_sec": 10.24,
    "lead_order": ["DI", "DII", "DIII", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
    "voltage_scale": "0.1 mV (1e-4 V)"
  },
  "probabilities": {
    "1dAVb": 0.0012,
    "RBBB": 0.0034,
    "LBBB": 0.3181,
    "SB": 0.0001,
    "AF": 0.0005,
    "ST": 0.0002
  },
  "cam": {
    "target_class": "LBBB",
    "weights": [0.0, 0.05, 0.42, 0.89, 0.95]
  },
  "leads": {
    "DI": [-0.05, -0.04],
    "V1": [0.12, 0.15]
  }
}
```

---

## 5. Medical Canvas Specification

- **Standard Calibration:**
  - Sweep speed: $25\text{ mm/s}$
  - Voltage gain: $10\text{ mm/mV}$
  - Pixel ratio: $1\text{ mm} = 4\text{ px}$
  - Small grid ($1\text{ mm}$): $4\text{ px} \times 4\text{ px}$ ($0.04\text{ s}$ / $0.1\text{ mV}$)
  - Large grid ($5\text{ mm}$): $20\text{ px} \times 20\text{ px}$ ($0.20\text{ s}$ / $0.5\text{ mV}$)
- **CAM Contrast Normalization in UI:**
  To avoid uniform red backgrounds in sustained pathologies across every beat (e.g., bundle branch blocks):
  - Local normalization: $\text{norm} = \frac{\text{cam} - \min(\text{cam})}{\max(\text{cam}) - \min(\text{cam})}$
  - Non-linear contrast: $\text{norm}^3$ (amplifies ventricular peaks and attenuates the baseline)
  - Render threshold: Display thermal shading only if $\text{norm} > 0.25$.

---

## 6. Class Imbalance: PTB-XL Label Distribution and Mitigation Strategies

### 6.1 Known Imbalance in PTB-XL Output Labels (7 Classes)

The PTB-XL primary model covers seven diagnostic classes. Some are severely underrepresented:

| Label | Full Name                    | PTB-XL Records | % of Total | MIMIC Supplement   | Risk Level    |
|-------|------------------------------|----------------|------------|--------------------|---------------|
| NORM  | Normal ECG                   | ~9,500         | ~44%       | --                  | Low           |
| MI    | Myocardial Infarction        | ~5,000         | ~23%       | --                  | Low           |
| STTC  | ST/T-Change                  | ~5,200         | ~24%       | --                  | Low           |
| HYP   | Hypertrophy                  | ~2,500         | ~12%       | --                  | Medium        |
| CD    | Conduction Disturbance       | ~1,200         | ~6%        | --                  | Medium        |
| LNGQT | Long QT Syndrome             | ~200-400       | ~1-2%      | **+2,022 records** | Mitigated  |
| WPW   | Wolff-Parkinson-White        | **~120**       | **~0.6%**  | **+413 records**   | Mitigated  |

> **Note:** Values are approximations based on PTB-XL v1.0.3 superdiagnostic label counts. Always verify against the active dataset split.

### 6.2 MIMIC-IV-ECG Supplementary Dataset (Stage 2 Fine-Tuning)

The MIMIC-IV-ECG subset was curated specifically to address the WPW/LNGQT imbalance:

| Property | Value |
|---|---|
| Total records | 2,413 unique patients |
| WPW records (is_WPW=1) | 413 (confirmed by clinical report keyword matching) |
| LNGQT records (is_LNGQT=1, QTc >= 480 ms, QRS < 120 ms) | 2,022 |
| Native sampling rate | 500 Hz |
| Resampled to | 400 Hz (4096 samples via resample_poly(up=4, down=5) + zero-pad) |
| Channel alignment | aVL/aVF swapped in MIMIC -> corrected by sig_name reordering |
| Data split | 80/20 by subject_id (GroupShuffleSplit) |
| Label encoding | 1.0=positive, 0.0=negative, -1.0=unknown (superclasses masked in loss) |
| Processed tensors location | ml/data/mimic_processed_400hz/ |

**Critical preprocessing steps (see ml/scripts/preprocess_mimic_wpw_lngqt.py):**
1. Read sig_name from WFDB header; map to canonical lead order [I, II, III, AVR, AVL, AVF, V1-V6]
2. nan_to_num -> clip +/-20 mV -> resample_poly(up=4, down=5) -> pad/crop to 4096 -> Z-score per lead
3. Labels: only WPW (idx=4) and LNGQT (idx=5) are confirmed; all other class columns set to -1.0

### 6.3 Mandatory Training Mitigations

The following mitigations **must** be applied when training or fine-tuning the PTB-XL primary model:

1. **Weighted Loss Function:**
   Use binary cross-entropy with per-class positive weights:
   $$\text{pos\_weight}_c = \frac{N_{\text{neg},c}}{N_{\text{pos},c}}$$
   This is critical for WPW, where the ratio can exceed 170:1.

   ```python
   # Example for PyTorch
   pos_weight = torch.tensor([w_MI, w_STTC, w_HYP, w_LNGQT, w_WPW])
   criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
   ```

2. **Stratified Train/Val/Test Splits:**
   Multi-label stratification (e.g., `iterative-stratification` library) must be used. Simple random splits risk entire minority classes (WPW) being absent from validation or test sets.

3. **Signal-Level Data Augmentation for Minority Classes:**
   Apply augmentation only to minority class records during training:
   - Gaussian noise injection ($\sigma$ ≤ 0.02 mV)
   - Amplitude scaling (factor ∈ [0.85, 1.15])
   - Baseline wander (low-frequency sinusoidal drift)
   - **Do NOT** use time-shifting or lead swapping for WPW; they may destroy the delta wave morphology.

4. **Per-Class Decision Threshold Tuning:**
   After training, sweep thresholds on the validation set independently per class:
   $$\hat{\tau}_c = \arg\max_{\tau \in [0,1]} F1_c(\tau)$$
   Default threshold of 0.5 is inappropriate for WPW and LNGQT.

5. **Evaluation Protocol:**
   Report all of the following; do not report only accuracy:
   - Per-class AUROC
   - Per-class F1, Precision, Recall at tuned threshold
   - Normalized confusion matrix (row-normalized to reveal class-level recall)
   - Macro-averaged and weighted-averaged F1

### 6.3 Imbalance in the Ribeiro Subordinate Model

The Ribeiro model (trained on CODE-15%) has its own label distribution. Its labels (1dAVb, RBBB, LBBB, AF, SB, ST) are generally more balanced due to the large dataset size (~400k records). Nevertheless:
- **AF** (Atrial Fibrillation) may still be a minority relative to SB/ST in some partitions.
- Because the Ribeiro model is invoked conditionally (only when T = True), the **effective input distribution during inference** is already pre-filtered toward rhythm-abnormal ECGs, which may shift its operating point. This must be considered when selecting the inference decision thresholds for the Ribeiro model's outputs.

---

## 7. Stage 2 Fine-Tuning Pipeline: Two-Stage Training with Loss Masking

### 7.1 Training Strategy

| Stage | Frozen Layers | Trainable Layers | LR | Max Epochs |
|---|---|---|---|---|
| **Stage A** | Full backbone (all Inception blocks) | Dense head only (`probabilities`, 128→7) | 1e-3 | 15 |
| **Stage B** | First Inception block (`inc_block_1_*`) | Last Inception block (`inc_block_2_*`) + head | 2e-4 | 20 |

### 7.2 Loss Masking Design

MIMIC records only have confirmed labels for WPW and LNGQT. The `MaskedBinaryFocalLoss` applies:

```python
# y_true encoding:
#   -1.0 → unknown (MIMIC superclass columns NORM, MI, STTC, HYP, CD)
#    0.0 → confirmed negative
#    1.0 → confirmed positive

mask = (y_true >= 0.0).astype(float)         # (batch, 7)
y_true_clamped = clip(y_true, 0.0, 1.0)     # Remove -1 sentinel
element_loss = focal_loss(y_pred, y_true_clamped)
masked_loss = element_loss * mask            # Zero out unknown positions
loss = sum(masked_loss) / sum(mask)          # Average over supervised positions only
```

**Critical indices:** `WPW=4`, `LNGQT=5` (per `TARGET_CLASSES` order in `config.py`)

### 7.3 Hybrid Dataset Composition

| Source | Records | Supervision | WPW Oversampling |
|---|---|---|---|
| PTB-XL folds 1–8 | ~17,420 | Full 7-class | No |
| MIMIC train (80%) | ~1,930 | WPW + LNGQT only | ×3 for WPW records |

### 7.4 Execution Order

```powershell
# Step 1: Preprocess MIMIC records (run once, ~5–10 min)
python ml/scripts/preprocess_mimic_wpw_lngqt.py

# Step 2: Fine-tune (Stage A + B, ~1–2 h on GPU)
python ml/scripts/finetune_ptbxl_mimic.py --stage AB --epochs-a 15 --epochs-b 20

# Step 3: Evaluate + generate Grad-CAM plots
python ml/scripts/evaluate_ptbxl_mimic_gradcam.py

# Step 4: Export fine-tuned model to dual ONNX for production
python ml/scripts/export_ptbxl_dual_onnx.py
```

### 7.5 Expected Outcomes (Design Targets)

| Metric | Stage 1 Baseline | Stage 2 Target | Constraint |
|---|---|---|---|
| WPW ROC-AUC | ~0.72–0.80 | ≥ 0.90 | — |
| WPW Recall (τ*) | ~0.30–0.50 | ≥ 0.70 | — |
| LNGQT ROC-AUC | ~0.75–0.85 | ≥ 0.90 | — |
| LNGQT Recall (τ*) | ~0.40–0.60 | ≥ 0.75 | — |
| MI/STTC/HYP/CD/NORM AUC | ~0.90–0.95 | ≥ 0.90 | **Must not degrade** |

> [!WARNING]
> If any of the 5 superclass ROC-AUCs drop by more than 0.02 relative to Stage 1, re-tune the loss mask or increase the backbone freeze depth in Stage B.
