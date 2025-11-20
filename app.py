# app.py
# Flask web application for document summarization
# Main server file that handles HTTP requests and serves the web interface

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from pathlib import Path
from dotenv import load_dotenv
import os
import logging
import requests
import json

# Load environment variables from .env file
ENV_PATH = Path(__file__).resolve().parent / ".env"
if not ENV_PATH.exists():
    print(f"Warning: .env file not found at {ENV_PATH}")
load_dotenv(dotenv_path=str(ENV_PATH))

# Application configuration
UPLOAD_FOLDER = "uploads"
ALLOWED_EXT = {"txt", "md"}
MAX_UPLOAD_SIZE = 5 * 1024 * 1024  # 5 MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_SIZE

# Setup logging
logger = logging.getLogger("docsummarizer")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)

# Import summarizer module
try:
    from summarizer import Summarizer, SummarizationError
    SUMMARIZER_AVAILABLE = True
except Exception as ex:
    logger.warning("Could not import summarizer: %s", ex)
    Summarizer = None
    class SummarizationError(Exception): pass
    SUMMARIZER_AVAILABLE = False

# lazy summarizer instance
_summarizer_instance = None
def get_summarizer():
    global _summarizer_instance
    if _summarizer_instance is None:
        if Summarizer is None:
            raise RuntimeError("summarizer.py not found or failed to import.")
        _summarizer_instance = Summarizer()
    return _summarizer_instance

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

# deterministic mock summary for demos when HF fails
def _mock_summary(text: str, style: str):
    words = text.split()
    if style == "bullets":
        count = min(6, max(1, len(words) // 40))
        return "\n".join([f"- Key point {i+1}" for i in range(count)])
    if style == "detailed":
        snippet = " ".join(words[:100])
        return "Detailed (mock) summary: " + (snippet + "..." if len(words) > 100 else snippet)
    # brief
    return " ".join(words[:25]) + ("..." if len(words) > 25 else "")

# -----------------------
# Routes
# -----------------------

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/api/health", methods=["GET"])
def api_health():
    """
    Health check endpoint to verify API connectivity and summarizer availability
    Returns JSON with status information
    """
    imported = SUMMARIZER_AVAILABLE
    hf_ok = False
    note = ""

    hf_key = os.getenv("HF_API_KEY")
    if not hf_key:
        note = "HF_API_KEY missing in .env"
    else:
        # Test connection to Hugging Face API
        test_model = os.getenv("HF_MODEL", "facebook/bart-large-cnn")
        url = f"https://router.huggingface.co/hf-inference/models/{test_model}"
        headers = {"Authorization": f"Bearer {hf_key}", "Content-Type": "application/json"}
        payload = {"inputs": "Hello", "parameters": {"max_length": 10, "min_length": 5}}
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=10)
            hf_ok = (r.status_code == 200)
            note = f"router_status={r.status_code}"
            if r.status_code != 200:
                note += f", response={r.text[:200]}"
        except Exception as ex:
            hf_ok = False
            note = f"router_error:{repr(ex)}"

    return jsonify({"summarizer_imported": imported, "hf_router_ok": hf_ok, "note": note})

@app.route("/api/summarize", methods=["POST"])
def api_summarize():
    """
    Main summarization endpoint
    Accepts text input or file upload (.txt or .md files)
    Returns JSON response with generated summary
    """
    style = request.form.get("style", "brief")
    text_input = (request.form.get("text") or "").strip()
    uploaded = request.files.get("file")

    # validate input
    if not text_input and (uploaded is None or uploaded.filename == ""):
        return jsonify({"error": "No input provided. Paste text or upload a .txt/.md file."}), 400

    if uploaded and uploaded.filename:
        filename = secure_filename(uploaded.filename)
        if not allowed_file(filename):
            return jsonify({"error": "Unsupported file type. Only .txt and .md allowed."}), 400
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        try:
            uploaded.save(filepath)
            with open(filepath, "r", encoding="utf-8", errors="ignore") as fh:
                text_input = fh.read()
        except Exception as e:
            logger.exception("Failed to save/read uploaded file")
            return jsonify({"error": f"Failed to process uploaded file: {str(e)}"}), 500

    if len(text_input) < 10:
        return jsonify({"error": "Input too short to summarize (min 10 characters)."}), 400

    # Try real summarizer if available
    if SUMMARIZER_AVAILABLE:
        try:
            summarizer = get_summarizer()
            summary = summarizer.summarize(text_input, style=style)
            return jsonify({"summary": summary}), 200
        except SummarizationError as se:
            logger.warning("SummarizationError: %s", se)
            mock = _mock_summary(text_input, style)
            return jsonify({"summary": mock, "note": "mock-summary-due-to-summarizer-error", "detail": str(se)}), 200
        except Exception as e:
            logger.exception("Unexpected error in summarizer")
            mock = _mock_summary(text_input, style)
            return jsonify({"summary": mock, "note": "mock-summary-due-to-unexpected-error", "detail": str(e)}), 200

    # If summarizer not available, return mock
    mock = _mock_summary(text_input, style)
    return jsonify({"summary": mock, "note": "mock-summary-summarizer-not-available"}), 200

if __name__ == "__main__":
    # Run the Flask development server
    app.run(host="127.0.0.1", port=5000, debug=False)
