"""
Local classifier wrapper for sklearn checkpoint files.

This provides a best-effort feature extractor (MFCC mean+std) and uses joblib to load
RandomForestClassifier-like pickled objects. It is intended as a local fallback when
Hugging Face router / inference is not available for your repo.

IMPORTANT: For correct results you must use the same feature extraction used at training.
This implementation is a starting point and may need to be adjusted.
"""
import os
from pathlib import Path
import io
import numpy as np

def _default_paths():
    # Check environment variables first. If not set, default to repository-local
    # paths under backend/models/ so hosts (Render) can include model files in the repo.
    repo_dir = Path(__file__).parent
    default_base = repo_dir / 'models' / 'base_model_full.ckpt'
    default_bio = repo_dir / 'models' / 'bio_model_full.ckpt'
    return {
        'base': os.getenv('LOCAL_BASE_CKPT') or str(default_base),
        'bio': os.getenv('LOCAL_BIO_CKPT') or str(default_bio),
    }

_models = {}

def _load_model(path):
    # Use model.py loader if available
    try:
        from model import load_model as _mdl_load
        return _mdl_load(path)
    except Exception:
        import joblib
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f'Model file not found: {path}')
        return joblib.load(str(p))

def _ensure_model(key):
    if key in _models:
        return _models[key]
    paths = _default_paths()
    if key not in paths or not paths[key]:
        raise ValueError('No local model path configured for key=' + key)
    model = _load_model(paths[key])
    _models[key] = model
    return model

def _extract_features_from_bytes(file_bytes, sr_target=16000):
    import soundfile as sf
    import numpy as np
    bio = io.BytesIO(file_bytes)
    # soundfile can read from file-like objects
    data, sr = sf.read(bio, dtype='float32')
    if isinstance(data, np.ndarray) and data.ndim == 2:
        # to mono
        data = data.mean(axis=1)
    # Try librosa if available for better features; otherwise fall back to simple stats
    try:
        import librosa
        if sr != sr_target:
            data = librosa.resample(data, orig_sr=sr, target_sr=sr_target)
            sr = sr_target
        mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=13)
        feat_mean = mfcc.mean(axis=1)
        feat_std = mfcc.std(axis=1)
        features = np.concatenate([feat_mean, feat_std])
        return features
    except Exception:
        # Simple fallback features: waveform stats + spectral stats
        import scipy.signal as sps
        if sr != sr_target:
            # naive resample using scipy.signal.resample
            num = int(len(data) * sr_target / sr)
            data = sps.resample(data, num)
            sr = sr_target
        # waveform stats
        rms = np.sqrt(np.mean(data**2))
        zcr = ((data[:-1] * data[1:]) < 0).sum() / max(1, len(data)-1)
        mn = float(np.mean(data))
        sd = float(np.std(data))
        mx = float(np.max(data))
        mn_abs = float(np.mean(np.abs(data)))
        # simple spectral features
        f, t, Sxx = sps.spectrogram(data, fs=sr)
        S = np.log1p(Sxx)
        spec_mean = float(np.mean(S))
        spec_std = float(np.std(S))
        features = np.array([rms, zcr, mn, sd, mx, mn_abs, spec_mean, spec_std], dtype=np.float32)
        return features

def predict_from_bytes(file_bytes, key='base'):
    """Return predict/proba from local model for given key ('base' or 'bio').

    This function writes the bytes to a temporary file and calls `extract_features` from
    the `features` module (which matches the training code). If `librosa` is not installed
    the features import will raise ImportError and we return a helpful error message.
    """
    # ensure feature extractor available
    try:
        from features import extract_features
    except Exception as e:
        raise RuntimeError("Feature extractor unavailable: ensure 'librosa' is installed and features.py is present. Original error: " + str(e))

    import tempfile
    tf = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    try:
        tf.write(file_bytes)
        tf.flush()
        tf.close()
        # choose feature set based on key
        feat_set = 'base' if key == 'base' else 'bio'
        feats = extract_features(tf.name, feature_set=feat_set)
    finally:
        try:
            os.remove(tf.name)
        except Exception:
            pass

    model = _ensure_model(key)
    X = np.asarray(feats).reshape(1, -1)
    result = {}
    try:
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X)
            result['predict_proba'] = proba.tolist()
    except Exception as e:
        result['predict_proba_error'] = str(e)
    try:
        if hasattr(model, 'predict'):
            pred = model.predict(X)
            result['predict'] = pred.tolist()
    except Exception as e:
        result['predict_error'] = str(e)
    return result

if __name__ == '__main__':
    print('This module is a helper for local inference; import and call predict_from_bytes()')
