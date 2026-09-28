import json
import os
import sys
import h5py
import numpy as np
import onnxruntime as ort

# Configurar encoding seguro para consola en Windows
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# 1. Definición robusta de rutas absolutas basadas en la raíz de PG
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

DATA_PATH = os.path.join(PROJECT_ROOT, "ml", "data", "ecg_tracings.hdf5")
MODEL_PATH = os.path.join(
    PROJECT_ROOT, "backend", "models", "ribeiro_dual.onnx"
)
WEIGHTS_PATH = os.path.join(
    PROJECT_ROOT, "backend", "models", "dense_weights.npy"
)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "samples")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "patient_1_lbbb.json")

# Validaciones de preexistencia de artefactos
for path, desc in [
    (DATA_PATH, "Dataset HDF5"),
    (MODEL_PATH, "Modelo ONNX"),
    (WEIGHTS_PATH, "Pesos de capa densa"),
]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el archivo de {desc} en: {path}")

# 2. Cargar el registro del paciente #1
print(f"[*] Cargando registro desde {DATA_PATH}...")
with h5py.File(DATA_PATH, "r") as f:
    ecg = f["tracings"][1].astype(np.float32)  # Forma: (4096, 12)

# 3. Inferencia con ONNX Runtime
print(f"[*] Ejecutando inferencia con {MODEL_PATH}...")
session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name
weights = np.load(WEIGHTS_PATH)  # (5120, 6)

ecg_input = np.expand_dims(ecg, axis=0)
probs, feature_maps = session.run(None, {input_name: ecg_input})

DIAGNOSTICOS = ["1dAVb", "RBBB", "LBBB", "SB", "AF", "ST"]
target_class = 2  # LBBB

# 4. Cálculo de CAM 1D adaptado a Flatten
A = feature_maps[0]  # (16, 320)
W_c = weights[:, target_class].reshape(16, 320)
cam_raw = np.maximum(np.sum(A * W_c, axis=-1), 0)

x_low = np.linspace(0, 1, len(cam_raw))
x_full = np.linspace(0, 1, 4096)
cam_vector = np.interp(x_full, x_low, cam_raw)
if np.max(cam_vector) > 0:
    cam_vector = cam_vector / np.max(cam_vector)

# 5. Empaquetar en estructura JSON estándar
derivaciones = [
    "DI",
    "DII",
    "DIII",
    "aVR",
    "aVL",
    "aVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
]

payload = {
    "metadata": {
        "sampling_rate_hz": 400,
        "num_samples": 4096,
        "duration_sec": 10.24,
        "lead_order": derivaciones,
        "voltage_scale": "0.1 mV (1e-4 V)",
    },
    "probabilities": {
        diag: float(np.round(p, 4)) for diag, p in zip(DIAGNOSTICOS, probs[0])
    },
    "cam": {
        "target_class": "LBBB",
        "weights": [float(np.round(val, 4)) for val in cam_vector],
    },
    "leads": {
        derivaciones[i]: [float(np.round(val, 2)) for val in ecg[:, i]]
        for i in range(12)
    },
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)

print(f"[✓] Muestra exportada con éxito en: {OUTPUT_FILE}")
print(f"    Tamaño en disco: {os.path.getsize(OUTPUT_FILE) / 1024:.1f} KB")