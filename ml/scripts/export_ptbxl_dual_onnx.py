import os
import sys
import numpy as np

# Configure safe encoding for console in Windows
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

os.environ["TF_USE_LEGACY_KERAS"] = "1"

try:
    import tf_keras as keras
except ImportError:
    from tensorflow import keras

import onnxruntime as ort
import tf2onnx

# Project root (dynamically resolved)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Candidate checkpoint paths for trained PTB-XL model
candidate_model_paths = [
    os.path.join(PROJECT_ROOT, "ml", "ptbxl", "checkpoints", "best_model.keras"),
    os.path.join(PROJECT_ROOT, "ml", "ptbxl", "checkpoints", "best_model.h5"),
]
MODEL_PATH = next((p for p in candidate_model_paths if os.path.exists(p)), candidate_model_paths[0])

OUTPUT_ONNX = os.path.join(PROJECT_ROOT, "backend", "models", "ptbxl_dual.onnx")
OUTPUT_WEIGHTS = os.path.join(PROJECT_ROOT, "backend", "models", "ptbxl_dense_weights.npy")

os.makedirs(os.path.dirname(OUTPUT_ONNX), exist_ok=True)


def export_ptbxl_dual():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Checkpoint no encontrado en: {MODEL_PATH}\n"
            f"Entrena primero el modelo ejecutando: python ml/ptbxl/train.py"
        )

    print(f"[*] Cargando modelo entrenado desde: {MODEL_PATH}...")
    model = keras.models.load_model(MODEL_PATH, compile=False)

    # 1. Identificar capa de feature maps (capa 'feature_maps' o la capa previa a Flatten)
    feature_maps_layer = None
    for layer in model.layers:
        if layer.name == "feature_maps":
            feature_maps_layer = layer
            break

    if feature_maps_layer is None:
        flatten_layer = next(l for l in model.layers if "flatten" in l.name.lower())
        feature_maps_output = flatten_layer.input
    else:
        feature_maps_output = feature_maps_layer.output

    print(f"[+] Feature maps detectados: {feature_maps_output.shape}")
    print(f"    Forma de salida de probabilidades: {model.output_shape}")

    # 2. Crear submodelo con arquitectura de salida dual:
    # Salida 0: Probabilidades diagnósticas (N, num_classes)
    # Salida 1: Feature maps para 1D CAM (N, 16, 320)
    dual_model = keras.Model(
        inputs=model.input,
        outputs=[model.output, feature_maps_output],
        name="ptbxl_dual",
    )

    # 3. Conversión a ONNX
    print(f"[*] Convirtiendo a ONNX -> {OUTPUT_ONNX}...")
    onnx_model, _ = tf2onnx.convert.from_keras(
        dual_model, opset=13, output_path=OUTPUT_ONNX
    )
    print(f"[+] Modelo ONNX exportado exitosamente en: {OUTPUT_ONNX}")

    # 4. Exportar matriz de pesos Dense (5120, num_classes) para 1D CAM
    dense_layers = [layer for layer in model.layers if "dense" in layer.name.lower() or "prob" in layer.name.lower()]
    if dense_layers:
        dense_weights, _ = dense_layers[-1].get_weights()
        np.save(OUTPUT_WEIGHTS, dense_weights)
        print(f"[+] Pesos Dense (5120, num_classes) guardados en: {OUTPUT_WEIGHTS} (forma: {dense_weights.shape})")

    # 5. Validación con ONNX Runtime
    print("[*] Validando grafo en ONNX Runtime...")
    session = ort.InferenceSession(OUTPUT_ONNX)
    dummy_input = np.random.randn(1, 4096, 12).astype(np.float32)

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: dummy_input})

    num_classes = outputs[0].shape[-1]
    print(f"[OK] Salida 0 (Probabilidades): forma {outputs[0].shape} (esperada: 1, {num_classes})")
    print(f"[OK] Salida 1 (Feature maps):    forma {outputs[1].shape}")

    assert outputs[0].shape == (1, num_classes), f"Salida 0 inválida: {outputs[0].shape}"
    assert len(outputs[1].shape) == 3 and outputs[1].shape[0] == 1, f"Salida 1 inválida: {outputs[1].shape}"
    print(f"[OK] PTB-XL dual ONNX validado (canales: {outputs[1].shape[-1]}, resolución temporal: {outputs[1].shape[1]}).")


if __name__ == "__main__":
    export_ptbxl_dual()
