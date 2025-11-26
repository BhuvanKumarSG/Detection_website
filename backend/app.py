from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import requests
from dotenv import load_dotenv
from asgiref.wsgi import WsgiToAsgi
import logging

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
HF_BASE_MODEL = os.getenv("HF_BASE_MODEL")
HF_BIO_MODEL = os.getenv("HF_BIO_MODEL")
# Allow disabling local fallback (set to '0' or 'false' to disable)
ALLOW_LOCAL_FALLBACK = os.getenv("ALLOW_LOCAL_FALLBACK", "1")

app = Flask(__name__)
FRONTEND_URL ="https://detectionwebsite.netlify.app"
CORS(app, resources={r"/predict": {"origins": FRONTEND_URL}})

# Working directory for model paths inside the repo
BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Configure basic logging to stdout so Render captures messages
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('detection-backend')

# Try to preload local models at import time if configured to use local model ids.
def _preload_local_models():
    # Only attempt to preload when LOCAL usage is configured in env vars
    try:
        b = str(HF_BASE_MODEL or '').lower()
        bi = str(HF_BIO_MODEL or '').lower()
        want_local = any(x.startswith('local') for x in (b, bi))
        if not want_local:
            return
        logger.info('Preloading local models as requested by env vars')
        try:
            from local_classifier import _ensure_model
        except Exception:
            try:
                from .local_classifier import _ensure_model
            except Exception as e:
                logger.exception('Failed importing local_classifier for preload: %s', e)
                return

        if b.startswith('local'):
            try:
                _ensure_model('base')
                logger.info('Preloaded local base model')
            except Exception as e:
                logger.exception('Failed to preload base model: %s', e)
        if bi.startswith('local'):
            try:
                _ensure_model('bio')
                logger.info('Preloaded local bio model')
            except Exception as e:
                logger.exception('Failed to preload bio model: %s', e)
    except Exception:
        logger.exception('Unexpected error during local preload')

# Preload now (module import time) so startup logs capture any issues
_preload_local_models()


def call_hf_model(model_id: str, file_bytes: bytes):
    """Call Hugging Face Inference API for a single model id using raw bytes."""
    if not model_id:
        return {"error": "model id not provided"}
    # The old `api-inference.huggingface.co` endpoint is deprecated and returns 410.
    # Use the new router endpoint instead:
    url = f"https://router.huggingface.co/models/{model_id}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/octet-stream"}
    try:
        resp = requests.post(url, headers=headers, data=file_bytes, timeout=60)
    except Exception as e:
        return {"error": str(e)}

    # If the router returns 404 (model not deployed), optionally try local sklearn-based fallback
    if resp.status_code == 404:
        if str(ALLOW_LOCAL_FALLBACK).lower() in ("0", "false"):
            # Explicitly disabled local fallback — return the router response
            return {"status_code": resp.status_code, "result_text": resp.text, "error": "model not deployed on Hugging Face and local fallback disabled"}
        # model_id may be a repo id or keyword; for local fallback accept 'base' or 'bio'
        try:
            from .local_classifier import predict_from_bytes
        except Exception:
            try:
                # import fallback for different import contexts
                from local_classifier import predict_from_bytes
            except Exception as e:
                return {"status_code": resp.status_code, "result_text": resp.text, "local_error": str(e)}

        # map model_id keywords to local keys
        key = 'base' if model_id in ('base', None) or str(model_id).lower().startswith('base') else ('bio' if str(model_id).lower().startswith('bio') else 'base')
        try:
            local_result = predict_from_bytes(file_bytes, key=key)
            return {"status_code": "local", "result": local_result}
        except Exception as e:
            return {"status_code": resp.status_code, "result_text": resp.text, "local_error": str(e)}

    try:
        return {"status_code": resp.status_code, "result": resp.json()}
    except Exception:
        return {"status_code": resp.status_code, "result_text": resp.text}


@app.route("/predict", methods=["POST"])
def predict():
    # Basic checks
    if not HF_TOKEN:
        return jsonify({"error": "HF_TOKEN not configured on server"}), 500

    uploaded = request.files.get('file')
    if uploaded is None:
        return jsonify({"error": "No file uploaded. Send multipart form field 'file'."}), 400

    file_bytes = uploaded.read()

    # models can be sent as multiple 'models' fields or as JSON list in a 'models' field
    models = request.form.getlist('models')
    # fallback: single comma-separated string
    if not models:
        m = request.form.get('models')
        if m:
            models = [x.strip() for x in m.split(',') if x.strip()]

    if not models:
        # default to base model if configured
        if HF_BASE_MODEL:
            models = ['base']
        else:
            return jsonify({"error": "No models provided and HF_BASE_MODEL not configured"}), 400

    results = {}
    for m in models:
        # allow keywords 'base' and 'bio' or full model ids
        if m == 'base':
            model_id = HF_BASE_MODEL
            label = 'base'
        elif m == 'bio':
            model_id = HF_BIO_MODEL
            label = 'bio'
        else:
            model_id = m
            label = m

        if not model_id:
            results[label] = {"error": "model not configured on server"}
            continue

        # If the configured model_id explicitly requests local models, call local classifier directly.
        # Use 'local' or 'local:base' / 'local:bio' to force local usage (backend must have LOCAL_*_CKPT env vars set).
        try:
            if str(model_id).lower().startswith('local'):
                try:
                    from .local_classifier import predict_from_bytes
                except Exception:
                    try:
                        from local_classifier import predict_from_bytes
                    except Exception as e:
                        results[label] = {"error": "local classifier import failed", "detail": str(e)}
                        continue

                # allow local:base or local:bio syntax; default to label mapping
                parts = str(model_id).split(':', 1)
                key = parts[1] if len(parts) > 1 and parts[1] else label
                try:
                    local_result = predict_from_bytes(file_bytes, key=key)
                    results[label] = {"status_code": "local", "result": local_result}
                except Exception as e:
                    results[label] = {"status_code": "local", "local_error": str(e)}
                continue
        except Exception:
            # if anything goes wrong with local detection, fall back to HF call
            pass

        results[label] = call_hf_model(model_id, file_bytes)

    return jsonify(results)


if __name__ == '__main__':
    # For local development only. Use the PORT env var when present (Render provides $PORT).
    # This keeps local default at 5000 but allows the host to override via environment.
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)


@app.route("/", methods=["GET"])
def index():
    return jsonify({"status": "ok", "message": "Detection backend running. POST /predict with 'file' and 'models' fields."})


@app.route('/debug/files', methods=['GET'])
def debug_files():
    """List files under backend/models for debugging deployments (safe to remove later)."""
    try:
        files = []
        if os.path.isdir(MODELS_DIR):
            for fn in os.listdir(MODELS_DIR):
                p = os.path.join(MODELS_DIR, fn)
                files.append({
                    'name': fn,
                    'size': os.path.getsize(p) if os.path.isfile(p) else None,
                    'is_file': os.path.isfile(p)
                })
        else:
            return jsonify({'error': 'models dir missing', 'path': MODELS_DIR}), 404
        return jsonify({'models_dir': MODELS_DIR, 'files': files})
    except Exception as e:
        logger.exception('Error listing models dir: %s', e)
        return jsonify({'error': str(e)}), 500

# Expose an ASGI app so you can run the Flask app with Uvicorn if preferred.
app_asgi = WsgiToAsgi(app)
