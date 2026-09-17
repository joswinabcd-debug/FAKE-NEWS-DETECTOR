"""
TRUTHSCAN AI - Localhost Web Application Server
Runs the unified full-stack application on localhost without Streamlit.
"""

import os
import sys
import webbrowser
from http.server import ThreadingHTTPServer

# Add current directory to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from api.index import ApiHandler


def run_server(port: int = None, open_browser: bool = False):
    """Starts the local web server."""
    if port is None:
        port = int(os.environ.get("PORT", 5000))

    host = "0.0.0.0"
    server_address = (host, port)
    
    try:
        httpd = ThreadingHTTPServer(server_address, ApiHandler)
    except OSError:
        # Fallback to port + 1 if busy
        port += 1
        server_address = (host, port)
        httpd = ThreadingHTTPServer(server_address, ApiHandler)

    url = f"http://localhost:{port}"
    print("=" * 60)
    print("  TRUTHSCAN AI - Web Application Server")
    print("  Deep Learning-Based Fake News Detection (CNN & LSTM)")
    print("=" * 60)
    print(f"\n[INFO] Application successfully started.")
    print(f"[INFO] Local URL:   {url}")
    print(f"[INFO] Network URL: http://0.0.0.0:{port}")
    print(f"[INFO] Serving frontend from: ./public")
    print(f"[INFO] API Endpoints ready at: {url}/api/*")
    print("\nPress Ctrl+C to shut down the server.\n")

    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down TRUTHSCAN AI server...")
        httpd.server_close()
        print("[INFO] Server stopped.")


if __name__ == "__main__":
    run_server()
