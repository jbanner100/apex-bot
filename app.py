import os
from flask import Flask, request, jsonify

app = Flask(__name__)

# --- Token helpers ------------------------------------------------------------
def get_secret() -> str:
    """Read SECRET_TOKEN from environment each time (avoids stale values)."""
    return os.getenv("1556c05227673418ae208659ab06e6c5", "") or ""

def require_token():
    """
    Allow ?token=... (query) or X-Webhook-Token: ... (header).
    Return a Flask response (403/500) if invalid; return None if OK.
    """
    secret = get_secret()
    token = request.args.get("token") or request.headers.get("X-Webhook-Token")
    if not secret:
        return jsonify({"ok": False, "error": "server not configured with SECRET_TOKEN"}), 500
    if token != secret:
        return jsonify({"ok": False, "error": "forbidden"}), 403
    return None

# --- Basic health -------------------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    return "OK", 200

@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"pong": True}), 200

@app.route("/alive", methods=["GET"])
def alive_simple():
    return jsonify({"ok": True, "service": "apex-bot"}), 200

# Optional duplicate health path if you want it:
@app.route("/__alive__", methods=["GET"])
def alive_dunder():
    return jsonify({"ok": True, "service": "apex-bot"}), 200

# --- Diagnostics (leave these in for now) ------------------------------------
@app.route("/envcheck", methods=["GET"])
def envcheck():
    secret = get_secret()
    return jsonify({"has_secret": bool(secret), "len": len(secret)})

@app.route("/routes", methods=["GET"])
def routes():
    rules = [str(r) for r in app.url_map.iter_rules()]
    return jsonify({"routes": sorted(rules)})

# --- Your secured webhook endpoint -------------------------------------------
@app.route("/webhook_test", methods=["POST", "GET"])
def webhook_test():
    # 🔒 Guard this endpoint with the token
    guard = require_token()
    if guard:  # if not None, return the 403/500 response
        return guard

    if request.method == "GET":
        return jsonify({"ok": True, "hint": "POST JSON like {'message':'hello'}"}), 200

    data = request.get_json(silent=True) or {}
    return jsonify({"received": data, "ok": True}), 200
