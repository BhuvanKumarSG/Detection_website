import os
import joblib


def _ensure_ckpt(path: str) -> str:
    base, ext = os.path.splitext(path)
    if ext.lower() == ".ckpt":
        return path
    return base + ".ckpt"


def load_model(path):
    path = _ensure_ckpt(path)
    return joblib.load(path)


def predict_file(model, audio_path):
    if isinstance(model, str):
        model = load_model(model)
    # caller should compute features and pass into model
    raise NotImplementedError("Use extract_features to compute features and then call model.predict([...])")
