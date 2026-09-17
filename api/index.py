"""
TRUTHSCAN AI - Unified API Handler
Deep Learning-Based Fake News Detection Using CNN and LSTM
Supports both Localhost (server.py) and Vercel Serverless Function execution.
"""

import os
import sys
import json
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import config

# Lazy load / cache predictor instance to prevent reloading weights on every call
_PREDICTOR_INSTANCE = None

def get_predictor():
    global _PREDICTOR_INSTANCE
    if _PREDICTOR_INSTANCE is None:
        try:
            from src.predict import NewsPredictor
            _PREDICTOR_INSTANCE = NewsPredictor()
        except Exception as e:
            print(f"[API] Error initializing NewsPredictor: {e}")
            _PREDICTOR_INSTANCE = None
    return _PREDICTOR_INSTANCE


class ApiHandler(BaseHTTPRequestHandler):
    """
    Unified HTTP Request Handler for TRUTHSCAN AI.
    Handles REST API endpoints (/api/*) and static files on localhost.
    """

    def _set_headers(self, status=200, content_type="application/json"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_OPTIONS(self):
        """Handle CORS pre-flight requests."""
        self._set_headers(status=204)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        # 1. API: System Status
        if path == "/api/status":
            try:
                from src.dataset import check_dataset_exists
                exists, _ = check_dataset_exists()
                cnn_trained = os.path.exists(config.CNN_CHECKPOINT)
                lstm_trained = os.path.exists(config.LSTM_CHECKPOINT)
                device_name = str(config.DEVICE).upper()

                payload = {
                    "dataset_available": exists,
                    "dataset_name": "ISOT Fake and Real News",
                    "compute_device": device_name,
                    "cnn_status": "Trained" if cnn_trained else "Untrained",
                    "lstm_status": "Trained" if lstm_trained else "Untrained",
                    "cnn_checkpoint_exists": cnn_trained,
                    "lstm_checkpoint_exists": lstm_trained,
                }
                self._set_headers(200)
                self.wfile.write(json.dumps(payload).encode("utf-8"))
            except Exception as e:
                self._set_headers(500)
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
            return

        # 2. API: Dataset Statistics
        elif path == "/api/stats":
            try:
                from src.dataset import get_dataset_statistics
                stats = get_dataset_statistics()
                self._set_headers(200)
                self.wfile.write(json.dumps(stats).encode("utf-8"))
            except Exception as e:
                self._set_headers(500)
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
            return

        # 3. API: Model Comparison
        elif path == "/api/comparison":
            try:
                if not os.path.exists(config.MODEL_COMPARISON_CSV):
                    self._set_headers(200)
                    self.wfile.write(json.dumps({
                        "available": False,
                        "records": [],
                        "message": "Models not evaluated yet. Train models first."
                    }).encode("utf-8"))
                    return

                import pandas as pd
                df = pd.read_csv(config.MODEL_COMPARISON_CSV)
                records = df.to_dict(orient="records")

                best_model = None
                best_f1 = 0.0
                if "F1 Score" in df.columns:
                    b_idx = df["F1 Score"].idxmax()
                    best_model = df.loc[b_idx, "Model"]
                    best_f1 = float(df.loc[b_idx, "F1 Score"])

                self._set_headers(200)
                self.wfile.write(json.dumps({
                    "available": True,
                    "records": records,
                    "best_model": best_model,
                    "best_f1": best_f1,
                    "plots": {
                        "comparison": "/api/plots/model_comparison.png",
                        "learning_curves": "/api/plots/loss_accuracy_curves.png",
                        "confusion_matrices": "/api/plots/confusion_matrices.png",
                    }
                }).encode("utf-8"))
            except Exception as e:
                self._set_headers(500)
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
            return

        # 4. API: Matplotlib Plot Images
        elif path.startswith("/api/plots/"):
            filename = os.path.basename(path)
            file_path = os.path.join(config.PLOTS_DIR, filename)
            if os.path.exists(file_path):
                self._set_headers(200, content_type="image/png")
                with open(file_path, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self._set_headers(404)
                self.wfile.write(json.dumps({"error": f"Plot {filename} not found"}).encode("utf-8"))
            return

        # 5. API: Sessions & Histories
        elif path == "/api/sessions":
            try:
                cnn_hist = {}
                if os.path.exists(config.CNN_HISTORY_FILE):
                    with open(config.CNN_HISTORY_FILE, "r", encoding="utf-8") as f:
                        cnn_hist = json.load(f)

                lstm_hist = {}
                if os.path.exists(config.LSTM_HISTORY_FILE):
                    with open(config.LSTM_HISTORY_FILE, "r", encoding="utf-8") as f:
                        lstm_hist = json.load(f)

                specs = {
                    "cnn": {
                        "architecture": "TextCNN (Multi-Kernel 1D Convolution)",
                        "kernels": config.CNN_CONFIG.get("kernel_sizes", [3, 4, 5]),
                        "num_filters": config.CNN_CONFIG.get("num_filters", 100),
                        "embedding_dim": config.CNN_CONFIG.get("embedding_dim", 128),
                        "head": "2-Stage MLP (300 -> 128 -> 1)",
                        "checkpoint": "checkpoints/cnn_best.pth",
                        "exists": os.path.exists(config.CNN_CHECKPOINT),
                    },
                    "lstm": {
                        "architecture": "LSTM (Recurrent Context Network)",
                        "hidden_dim": config.LSTM_CONFIG.get("hidden_dim", 128),
                        "num_layers": config.LSTM_CONFIG.get("num_layers", 1),
                        "embedding_dim": config.LSTM_CONFIG.get("embedding_dim", 128),
                        "dropout": config.LSTM_CONFIG.get("dropout", 0.3),
                        "checkpoint": "checkpoints/lstm_best.pth",
                        "exists": os.path.exists(config.LSTM_CHECKPOINT),
                    }
                }

                self._set_headers(200)
                self.wfile.write(json.dumps({
                    "cnn_history": cnn_hist,
                    "lstm_history": lstm_hist,
                    "specs": specs,
                    "has_curves": os.path.exists(os.path.join(config.PLOTS_DIR, "loss_accuracy_curves.png")),
                    "curves_url": "/api/plots/loss_accuracy_curves.png"
                }).encode("utf-8"))
            except Exception as e:
                self._set_headers(500)
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
            return

        # 6. Static File Serving on Localhost
        else:
            self._serve_static(path)

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        content_len = int(self.headers.get("Content-Length", 0))
        post_body = self.rfile.read(content_len) if content_len > 0 else b"{}"

        try:
            data = json.loads(post_body.decode("utf-8")) if post_body else {}
        except Exception:
            self._set_headers(400)
            self.wfile.write(json.dumps({"error": "Invalid JSON body"}).encode("utf-8"))
            return

        # 1. API: Predict
        if path == "/api/predict":
            title = data.get("title", "").strip()
            text = data.get("text", "").strip()
            model_type = data.get("model", "both").lower()

            if not title and not text:
                self._set_headers(400)
                self.wfile.write(json.dumps({"error": "Please provide a headline or article text to analyze."}).encode("utf-8"))
                return

            # Check if an external ML backend URL is configured (e.g. on serverless Vercel if PyTorch isn't bundled)
            ml_backend_url = os.environ.get("ML_BACKEND_URL")
            if ml_backend_url and not os.path.exists(config.CNN_CHECKPOINT):
                try:
                    req = urllib.request.Request(
                        f"{ml_backend_url.rstrip('/')}/api/predict",
                        data=json.dumps(data).encode("utf-8"),
                        headers={"Content-Type": "application/json"}
                    )
                    with urllib.request.urlopen(req, timeout=30) as resp:
                        res_data = resp.read()
                        self._set_headers(resp.status)
                        self.wfile.write(res_data)
                        return
                except Exception as proxy_err:
                    print(f"[API] Error proxying to ML backend: {proxy_err}")

            predictor = get_predictor()
            if predictor is None:
                self._set_headers(503)
                self.wfile.write(json.dumps({"error": "Predictor could not be initialized. Please verify PyTorch installation and model checkpoints."}).encode("utf-8"))
                return

            target_check = "both" if model_type in ["both", "compare both"] else model_type
            if not predictor.is_model_available(target_check):
                self._set_headers(404)
                self.wfile.write(json.dumps({"error": "Selected model checkpoint is not available. Please train the model first."}).encode("utf-8"))
                return

            try:
                if model_type in ["both", "compare both"]:
                    result = predictor.predict_both(title, text)
                else:
                    result = predictor.predict_single(title, text, model_type=model_type)

                self._set_headers(200)
                self.wfile.write(json.dumps(result).encode("utf-8"))
            except Exception as e:
                self._set_headers(500)
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
            return

        # 2. API: Train Models
        elif path == "/api/train":
            target = data.get("target", "both").lower()
            mode = data.get("mode", "quick").lower()

            if target not in ["cnn", "lstm", "both"]:
                self._set_headers(400)
                self.wfile.write(json.dumps({"error": "Target must be 'cnn', 'lstm', or 'both'."}).encode("utf-8"))
                return

            try:
                from src.train import train_pipeline
                from src.evaluate import evaluate_models

                logs = []
                def log_cb(line):
                    logs.append(line)

                results = train_pipeline(target=target, mode=mode, log_callback=log_cb)
                eval_res = evaluate_models()

                # Invalidate cached predictor so new weights are picked up immediately
                global _PREDICTOR_INSTANCE
                _PREDICTOR_INSTANCE = None

                df_eval = eval_res["comparison_df"].to_dict(orient="records")

                self._set_headers(200)
                self.wfile.write(json.dumps({
                    "success": True,
                    "target": target,
                    "mode": mode,
                    "logs": logs,
                    "comparison": df_eval,
                    "message": f"Successfully trained {target.upper()} in {mode} mode."
                }).encode("utf-8"))
            except Exception as e:
                self._set_headers(500)
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
            return

        else:
            self._set_headers(404)
            self.wfile.write(json.dumps({"error": "Endpoint not found"}).encode("utf-8"))

    def _serve_static(self, path):
        """Serves static files from the public/ directory for localhost runtime."""
        public_dir = os.path.join(BASE_DIR, "public")
        
        # Clean leading slash
        clean_path = path.lstrip("/")
        if not clean_path or clean_path == "/":
            clean_path = "index.html"

        file_path = os.path.join(public_dir, clean_path)

        # Prevent directory traversal
        real_public = os.path.realpath(public_dir)
        real_file = os.path.realpath(file_path)
        if not real_file.startswith(real_public):
            self._set_headers(403, "text/plain")
            self.wfile.write(b"Forbidden")
            return

        if not os.path.exists(real_file) or os.path.isdir(real_file):
            # Fallback to index.html for SPA client routing if file doesn't exist
            real_file = os.path.join(public_dir, "index.html")

        # Determine MIME type
        content_type = "text/html; charset=utf-8"
        if real_file.endswith(".css"):
            content_type = "text/css; charset=utf-8"
        elif real_file.endswith(".js"):
            content_type = "application/javascript; charset=utf-8"
        elif real_file.endswith(".json"):
            content_type = "application/json"
        elif real_file.endswith(".png"):
            content_type = "image/png"
        elif real_file.endswith(".jpg") or real_file.endswith(".jpeg"):
            content_type = "image/jpeg"
        elif real_file.endswith(".svg"):
            content_type = "image/svg+xml"
        elif real_file.endswith(".ico"):
            content_type = "image/x-icon"

        try:
            with open(real_file, "rb") as f:
                content = f.read()
            self._set_headers(200, content_type=content_type)
            self.wfile.write(content)
        except Exception as e:
            self._set_headers(500, "text/plain")
            self.wfile.write(f"Server error: {e}".encode("utf-8"))


# Vercel Serverless Function entry point
handler = ApiHandler
