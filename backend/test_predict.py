import os
import sys
import json
import requests

# Test script to POST an audio file to the local /predict endpoint.
# Usage: python test_predict.py [path/to/file]

DEFAULT_FILE = r"D:\Danush\detectionCPU\dataset\asvspoof_full\fake\LA_T_1004407.flac"
URL = os.getenv('PREDICT_URL', 'http://127.0.0.1:5000/predict')

file_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FILE

if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
    sys.exit(2)

models = os.getenv('TEST_MODELS', 'base')
models_list = [m.strip() for m in models.split(',') if m.strip()]

print(f"Posting file: {file_path}\nTo URL: {URL}\nModels: {models_list}")

with open(file_path, 'rb') as f:
    files = {'file': (os.path.basename(file_path), f, 'audio/flac')}
    data = [("models", m) for m in models_list]
    try:
        resp = requests.post(URL, files=files, data=data, timeout=120)
    except Exception as e:
        print('Request failed:', e)
        sys.exit(3)

print('Status:', resp.status_code)
try:
    print(json.dumps(resp.json(), indent=2))
except Exception:
    print(resp.text)
