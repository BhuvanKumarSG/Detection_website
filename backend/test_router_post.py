import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

URL = 'https://router.huggingface.co/models/facebook/wav2vec2-base-960h'
TOKEN = os.environ.get('HF_TOKEN')
FILE = r"D:\Danush\detectionCPU\dataset\asvspoof_full\fake\LA_T_1004407.flac"

if TOKEN is None:
    print('HF_TOKEN not set in environment')
    raise SystemExit(2)

headers = {'Authorization': f'Bearer {TOKEN}', 'Content-Type': 'audio/flac'}

with open(FILE, 'rb') as f:
    try:
        r = requests.post(URL, headers=headers, data=f, timeout=120)
    except Exception as e:
        print('Request failed:', e)
        raise

print('Status:', r.status_code)
print('Headers:', r.headers)
print('\nResponse body:\n')
try:
    print(json.dumps(r.json(), indent=2))
except Exception:
    print(r.text)
