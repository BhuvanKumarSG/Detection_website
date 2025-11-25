# Detection Website (React + Flask)

This project is a single-page React frontend with a Flask backend that forwards audio files to Hugging Face model(s) for inference.

Quick overview
- Frontend: `frontend/` (Vite + React) — drag-and-drop audio upload, model selection checkboxes, displays results.
- Backend: `backend/` (Flask) — `/predict` endpoint that calls the Hugging Face Inference API using `HF_TOKEN`.

Environment
1. Copy `.env.example` to `.env` and fill in `HF_TOKEN`, `HF_BASE_MODEL`, `HF_BIO_MODEL`.

Run backend
PowerShell examples:
```
cd "d:/Danush/Detection website/backend"
python -m venv venv; ; .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:FLASK_APP = 'app.py'; $env:FLASK_ENV = 'development'
flask run --host=0.0.0.0
```

Run frontend
```
cd "d:/Danush/Detection website/frontend"
npm install
npm run dev
```

Notes
- Backend reads `HF_TOKEN`, `HF_BASE_MODEL`, and `HF_BIO_MODEL` from environment (or `.env`).
- Frontend calls backend at `http://localhost:5000/predict` by default.
