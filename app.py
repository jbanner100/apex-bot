import os
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/")
def home():
    return "OK", 200

@app.route("/ping")
def ping():
    return jsonify({"pong": True}), 200

@app.route("/envcheck")
def envcheck():
    present = "SECRET_TOKEN" in os.environ and bool(os.environ.get("SECRET_TOKEN"))
    return jsonify({"has_secret": bool(present), "len": len(os.environ.get("SECRET_TOKEN",""))})

@app.route("/routes")
def routes():
    return jsonify({"routes": sorted(str(r) for r in app.url_map.iter_rules())})

@app.route("/webhook_test", methods=["GET","POST"])
def webhook_test():
    if request.method == "GET":
        return jsonify({"ok": True, "hint": "POST JSON like {'message':'hello'}"}), 200
    data = request.get_json(silent=True) or {}
    return jsonify({"received": data, "ok": True}), 200
