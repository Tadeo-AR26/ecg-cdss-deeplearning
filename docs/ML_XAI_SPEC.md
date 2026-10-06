# Machine Learning and XAI Technical Specification

## 1. Ribeiro ResNet-1D Model Architecture

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

## 2. Dual ONNX Output (`ribeiro_dual.onnx`)

The compiled ONNX graph has one input and two simultaneous outputs:
* **Input:** `(1, 4096, 12)` — Float32 signal tensor.
* **Output 0 (`probabilities`):** `(1, 6)` — Sigmoid classification probabilities.
* **Output 1 (`feature_maps`):** `(1, 16, 320)` — Temporal activations from the last convolutional layer (`conv1d_11`).

---

## 3. 1D CAM Mechanics: Flatten vs Global Average Pooling (GAP)

> **WARNING FOR AI AGENTS:**
> The Ribeiro network does **NOT** use *Global Average Pooling* (GAP) after the convolutional layers. It uses a direct `Flatten` layer.
> The feature maps from the convolutional layer have dimensions $16 \times 320$, producing a flattened vector of:
> $$16 \times 320 = 5120 \text{ features}$$
> Consequently, the final Dense layer weight matrix has shape `(5120, 6)`.

### CAM Attention Vector Formulation:
For a target class $c$ (e.g., $c = 2$ for LBBB):

1. **Weight vector isolation:**
   Column $c$ is extracted from the dense weight matrix:
   $$W_c \in \mathbb{R}^{5120}$$

2. **Temporal reshape:**
   The vector is reshaped to align with the activation maps $A \in \mathbb{R}^{16 \times 320}$:
   $$W_c \xrightarrow{\text{reshape}} \mathbb{R}^{16 \times 320}$$

3. **Channel-wise multiplication and summation:**
   $$\text{CAM}_{\text{raw}}[t] = \sum_{k=1}^{320} A[t, k] \cdot W_c[t, k], \quad \forall t \in [0, 15]$$

4. **ReLU rectification:**
   Only evidence that positively supports the pathology is retained:
   $$\text{CAM}_{\text{relu}}[t] = \max(0, \text{CAM}_{\text{raw}}[t])$$

5. **Linear Interpolation to Full Scale:**
   The 16-bin temporal vector is expanded to the 4096 points of the full signal:
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
