from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/")
def home():
    return "OK", 200

@app.route("/__alive__")
def alive():
    return jsonify({"ok": True, "service": "apex-bot"}), 200

@app.route("/ping")
def ping():
    return jsonify({"pong": True}), 200

@app.route("/webhook_test", methods=["POST", "GET"])
def webhook_test():
    if request.method == "GET":
        return jsonify({"ok": True, "hint": "POST JSON like {'message':'hello'}"}), 200
    data = request.get_json(silent=True) or {}
    return jsonify({"received": data, "ok": True}), 200
