import os
from flask import Flask, request, jsonify

app = Flask(__name__)
def get_secret():
    # Read SECRET_TOKEN from the environment (set in Render Settings → Environment)
    return os.getenv("4294356042fd1248ae673af2a498254c", "") or ""


def require_token():
    """
    Allow ?token=... (query) or X-Webhook-Token: ... (header).
    Return a Flask response (403/500) if invalid; return None if ok.
    """
    secret = get_secret()
    token = request.args.get("token") or request.headers.get("X-Webhook-Token")

    if not secret:
        return jsonify({"ok": False, "error": "server not configured with SECRET_TOKEN"}), 500
    if token != secret:
        return jsonify({"ok": False, "error": "forbidden"}), 403
    return None

@app.route("/", methods=["GET"])
def home():
    return "OK", 200

@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"pong": True}), 200

@app.route("/alive", methods=["GET"])
def alive_simple():
    return jsonify({"ok": True, "service": "apex-bot"}), 200

@app.route("/__alive__", methods=["GET"])
def alive_dunder():
    return jsonify({"ok": True, "service": "apex-bot"}), 200

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

@app.route("/routes", methods=["GET"])
def routes():
    rules = [str(r) for r in app.url_map.iter_rules()]
    return jsonify({"routes": sorted(rules)})

# 🔎 NEW: env var check
@app.route("/envcheck", methods=["GET"])
def envcheck():
    secret = get_secret()
    return jsonify({
        "has_secret": bool(secret),
        "len": len(secret) if secret else 0,
        # Do NOT return the secret value itself for safety.
    })

