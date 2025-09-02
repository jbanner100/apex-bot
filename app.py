import os
from flask import Flask, request, jsonify

app = Flask(__name__)

# ---- Basic health/diagnostics ------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    return "OK", 200

@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"pong": True}), 200

@app.route("/envcheck", methods=["GET"])
def envcheck():
    s = os.getenv("SECRET_TOKEN", "") or ""
    return jsonify({"has_secret": bool(s), "len": len(s)})

@app.route("/routes", methods=["GET"])
def routes():
    return jsonify({"routes": sorted(str(r) for r in app.url_map.iter_rules())})

# ---- Secured webhook ---------------------------------------------------------
import os
from flask import Flask, request, jsonify

# ...your existing routes above...

@app.route("/webhook/<path_secret>", methods=["GET", "POST"])
def webhook_url_secret(path_secret):
    # Read SECRET_TOKEN from env (set in Render → Web Service → Settings → Environment)
    secret = os.getenv("SECRET_TOKEN", "") or ""
    if not secret:
        return jsonify({"ok": False, "error": "server not configured with SECRET_TOKEN"}), 500
    if path_secret != secret:
        return jsonify({"ok": False, "error": "forbidden"}), 403

    if request.method == "GET":
        # handy for quick checks from a browser
        return jsonify({"ok": True, "hint": "POST JSON like {'message':'hello'}"}), 200

    data = request.get_json(silent=True) or {}
    print("[TV] /webhook/<secret>", data, flush=True)
    return jsonify({"received": data, "ok": True}), 200



@app.route("/webhook_test", methods=["GET", "POST"])
def webhook_test():
    # Read the secret FRESH on each request (no module-level caching)
    secret = os.getenv("SECRET_TOKEN", "") or ""
    if not secret:
        # Include length so we can diagnose without revealing the value
        return jsonify({"ok": False, "error": "server not configured with SECRET_TOKEN", "len": 0}), 500

    # Accept token via query or header
    token = request.args.get("token") or request.headers.get("X-Webhook-Token")

    if request.method == "GET":
        if token != secret:
            return jsonify({"ok": False, "error": "forbidden", "need_token": True}), 403
        return jsonify({"ok": True, "hint": "POST JSON like {'message':'hello'}"}), 200

    # POST
    if token != secret:
        return jsonify({"ok": False, "error": "forbidden"}), 403

        data = request.get_json(silent=True) or {}
    print("[TV] /webhook_test", data, flush=True)  # <-- add this line
    return jsonify({"received": data, "ok": True}), 200

