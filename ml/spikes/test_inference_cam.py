import os
import sys
import h5py
import numpy as np
import onnxruntime as ort

# Configurar encoding seguro para consola en Windows
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

try:
    import tf_keras as keras
except ImportError:
    try:
        from tensorflow import keras
    except ImportError:
        keras = None

# Raíz del proyecto (PG/)
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

# Rutas de artefactos de producción
WEIGHTS_PATH = os.path.join(
    PROJECT_ROOT, "backend", "models", "dense_weights.npy"
)
ONNX_PATH = os.path.join(PROJECT_ROOT, "backend", "models", "ribeiro_dual.onnx")

# Rutas de insumos pesados
candidate_model_paths = [
    os.path.join(PROJECT_ROOT, "ml", "weights", "model.hdf5"),
    os.path.join(PROJECT_ROOT, "ml", "raw", "model.hdf5"),
    os.path.join(PROJECT_ROOT, "model.hdf5"),
    "model.hdf5",
]
MODEL_PATH = next((p for p in candidate_model_paths if os.path.exists(p)), None)

candidate_tracings_paths = [
    os.path.join(PROJECT_ROOT, "ml", "data", "ecg_tracings.hdf5"),
    os.path.join(PROJECT_ROOT, "ecg_tracings.hdf5"),
    "ecg_tracings.hdf5",
]
TRACINGS_PATH = next(
    (p for p in candidate_tracings_paths if os.path.exists(p)), None
)

# 1. Obtener los pesos de la capa Dense final (5120, 6)
if os.path.exists(WEIGHTS_PATH):
    print(f"[*] Cargando matriz de pesos W existente desde: {WEIGHTS_PATH}")
    weights = np.load(WEIGHTS_PATH)
    print(f"[✓] Pesos cargados con forma {weights.shape}")
elif MODEL_PATH and keras is not None:
    print(f"[*] Extrayendo matriz de pesos W desde {MODEL_PATH}...")
    model = keras.models.load_model(MODEL_PATH, compile=False)
    dense_layer = [
        layer for layer in model.layers if "dense" in layer.name.lower()
    ][-1]
    weights, _ = dense_layer.get_weights()
    os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
    np.save(WEIGHTS_PATH, weights)
    print(f"[✓] Pesos guardados en {WEIGHTS_PATH} con forma {weights.shape}")
else:
    raise FileNotFoundError(
        f"No se encontró ni 'dense_weights.npy' ni un 'model.hdf5' para extraerlos."
    )

# 2. Validar preexistencia de ONNX y tracings
if not os.path.exists(ONNX_PATH):
    raise FileNotFoundError(
        f"No se encontró el modelo ONNX en: {ONNX_PATH}\nEjecuta primero ml/scripts/export_dual_onnx.py."
    )

if not TRACINGS_PATH:
    rutas_buscadas = "\n - ".join(candidate_tracings_paths)
    raise FileNotFoundError(
        f"No se encontró 'ecg_tracings.hdf5'. Rutas verificadas:\n - {rutas_buscadas}"
    )

print(f"[*] Cargando registros reales de prueba desde {TRACINGS_PATH}...")
with h5py.File(TRACINGS_PATH, "r") as f:
    tracings = f["tracings"][:10].astype(np.float32)

print(f"[*] Inicializando sesión de ONNX Runtime con {ONNX_PATH}...")
session = ort.InferenceSession(ONNX_PATH)
input_name = session.get_inputs()[0].name
DIAGNOSTICOS = ["1dAVb", "RBBB", "LBBB", "SB", "AF", "ST"]

# 3. Buscar el primer registro con patología detectable (>5%)
selected_idx = 0
probs_selected = None
feature_maps_selected = None

for i in range(len(tracings)):
    ecg_input = np.expand_dims(tracings[i], axis=0)  # (1, 4096, 12)
    probs, f_maps = session.run(None, {input_name: ecg_input})
    if np.max(probs[0]) > 0.05:
        selected_idx = i
        probs_selected = probs
        feature_maps_selected = f_maps
        break

if probs_selected is None:
    selected_idx = 0
    ecg_input = np.expand_dims(tracings[0], axis=0)
    probs_selected, feature_maps_selected = session.run(
        None, {input_name: ecg_input}
    )

print(f"\n--- Resultados diagnósticos para paciente #{selected_idx} ---")
for diag, prob in zip(DIAGNOSTICOS, probs_selected[0]):
    print(f"{diag:8s}: {prob * 100:6.2f}% (prob={prob:.4f})")

# 4. Cálculo de CAM 1D adaptado a Flatten
target_class = int(np.argmax(probs_selected[0]))
target_name = DIAGNOSTICOS[target_class]

A = feature_maps_selected[0]  # (16, 320)
W_c = weights[:, target_class]  # (5120,)

# Reshape de (5120,) a (16, 320) para alinear tiempo x canales
W_c_2d = W_c.reshape(16, 320)

# Multiplicación y suma sobre los 320 canales
cam_raw = np.sum(A * W_c_2d, axis=-1)  # (16,)
cam_raw = np.maximum(cam_raw, 0)  # ReLU

# Interpolar linealmente de 16 bins a los 4096 puntos temporales originales
x_low = np.linspace(0, 1, len(cam_raw))
x_full = np.linspace(0, 1, 4096)
cam_vector = np.interp(x_full, x_low, cam_raw)

# Normalizar entre 0 y 1 para renderizado en Canvas
if np.max(cam_vector) > 0:
    cam_vector = cam_vector / np.max(cam_vector)

print(
    f"\n[✓] Vector CAM 1D calculado para '{target_name}': longitud {len(cam_vector)}, min={np.min(cam_vector):.2f}, max={np.max(cam_vector):.2f}"
)