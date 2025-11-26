from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import requests
from dotenv import load_dotenv
from asgiref.wsgi import WsgiToAsgi

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
HF_BASE_MODEL = os.getenv("HF_BASE_MODEL")
HF_BIO_MODEL = os.getenv("HF_BIO_MODEL")

app = Flask(__name__)
CORS(app)



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

    # If the router returns 404 (model not deployed), try local sklearn-based fallback
    if resp.status_code == 404:
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

        results[label] = call_hf_model(model_id, file_bytes)

    return jsonify(results)


if __name__ == '__main__':
    # For local development only. Prefer using `flask run` in production/dev.
    app.run(host='0.0.0.0', port=5000, debug=True)


@app.route("/", methods=["GET"])
def index():
    return jsonify({"status": "ok", "message": "Detection backend running. POST /predict with 'file' and 'models' fields."})

# Expose an ASGI app so you can run the Flask app with Uvicorn if preferred.
app_asgi = WsgiToAsgi(app)
