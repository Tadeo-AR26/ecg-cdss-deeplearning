import os
import sys
import numpy as np

# Configurar encoding seguro para consola en Windows
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Forzar el backend compatible con grafos heredados de Keras 2
os.environ["TF_USE_LEGACY_KERAS"] = "1"

try:
    import tf_keras as keras
except ImportError:
    from tensorflow import keras

import onnxruntime as ort
import tf2onnx

# Raíz del proyecto
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Detectar ruta de model.hdf5 (ml/raw, raíz del proyecto o directorio actual)
candidate_model_paths = [
    os.path.join(PROJECT_ROOT, "ml", "raw", "model.hdf5"),
    os.path.join(PROJECT_ROOT, "model.hdf5"),
    "model.hdf5",
]
MODEL_PATH = next((p for p in candidate_model_paths if os.path.exists(p)), candidate_model_paths[0])

OUTPUT_ONNX = os.path.join(PROJECT_ROOT, "backend", "models", "ribeiro_dual.onnx")
OUTPUT_WEIGHTS = os.path.join(PROJECT_ROOT, "backend", "models", "dense_weights.npy")

os.makedirs(os.path.dirname(OUTPUT_ONNX), exist_ok=True)

print(f"[*] Cargando {MODEL_PATH}...")
model = keras.models.load_model(MODEL_PATH, compile=False)

# 1. Identificar la última capa convolucional 1D antes del Global Average Pooling
conv_layers = [
    layer
    for layer in model.layers
    if "conv1d" in layer.name.lower() or "conv" in layer.name.lower()
]
if not conv_layers:
    raise RuntimeError(
        "No se encontraron capas convolucionales en la arquitectura."
    )

last_conv = conv_layers[-1]
print(f"[+] Última capa convolucional detectada: {last_conv.name}")
print(f"    Forma de salida de la convolución: {last_conv.output_shape}")
print(f"    Forma de salida del modelo (clases): {model.output_shape}")

# Crear el submodelo con arquitectura de salida dual
# Salida 0: Probabilidades diagnósticas (N, 6)
# Salida 1: Feature maps para CAM 1D (N, longitud_reducida, canales)
dual_model = keras.Model(
    inputs=model.input,
    outputs=[model.output, last_conv.output],
    name="resnet1d_dual",
)

# Conversión a ONNX
print("[*] Convirtiendo a ONNX...")
onnx_model, _ = tf2onnx.convert.from_keras(
    dual_model, opset=13, output_path=OUTPUT_ONNX
)
print(f"[+] Modelo exportado exitosamente en: {OUTPUT_ONNX}")

# Exportación de pesos de la capa densa para cálculo de CAM
dense_layers = [
    layer for layer in model.layers if "dense" in layer.name.lower()
]
if dense_layers:
    dense_weights, _ = dense_layers[-1].get_weights()
    np.save(OUTPUT_WEIGHTS, dense_weights)
    print(f"[+] Pesos de capa densa exportados exitosamente en: {OUTPUT_WEIGHTS}")

# Verificación de inferencia con ONNX Runtime
print("[*] Verificando grafo en ONNX Runtime...")
session = ort.InferenceSession(OUTPUT_ONNX)
dummy_ecg = np.random.randn(1, 4096, 12).astype(np.float32)

inputs = {session.get_inputs()[0].name: dummy_ecg}
outputs = session.run(None, inputs)

print(f"[OK] Salida 0 (Probabilidades diagnósticas): forma {outputs[0].shape}")
print(f"[OK] Salida 1 (Feature maps para CAM 1D):    forma {outputs[1].shape}")

assert outputs[0].shape == (1, 6), f"Salida 0 esperada (1, 6), obtenida: {outputs[0].shape}"
assert len(outputs[1].shape) == 3, f"Salida 1 esperada 3D, obtenida: {outputs[1].shape}"

print("[OK] Pipeline de exportación validado correctamente.")