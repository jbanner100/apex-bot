# app.py
import os, time, threading
from datetime import datetime
from flask import Flask, jsonify

app = Flask(__name__)

def now():
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

# --- tiny heartbeat so you see *something* in logs
_started = False
def _heartbeat():
    while True:
        print(f"[{now()}] 💓 heartbeat alive")
        time.sleep(10)

def _start_once():
    global _started
    if _started: 
        return
    print(f"[{now()}] 🚀 starting heartbeat thread")
    threading.Thread(target=_heartbeat, name="Heartbeat", daemon=True).start()
    _started = True

print(f"[{now()}] 🔧 app.py imported")
_start_once()

@app.before_request
def _ensure():
    _start_once()

@app.route("/", methods=["GET"])
def root():
    return jsonify(ok=True, ts=int(time.time())), 200

@app.route("/ping", methods=["GET"])
def ping():
    return "pong", 200

@app.route("/alive", methods=["GET"])
def alive():
    return "ok", 200

@app.route("/debug/status", methods=["GET"])
def debug_status():
    th = [t.name for t in threading.enumerate()]
    return jsonify(
        threads=th,
        commit=os.getenv("RENDER_GIT_COMMIT", "unknown")[:7],
        python=os.getenv("PYTHON_VERSION", "unk"),
        port=os.getenv("PORT", "unk"),
    ), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5008, threaded=True, use_reloader=False)


