from flask import Flask, request, jsonify

app = Flask(__name__)

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
    if request.method == "GET":
        return jsonify({"ok": True, "hint": "POST JSON like {'message':'hello'}"}), 200
    data = request.get_json(silent=True) or {}
    return jsonify({"received": data, "ok": True}), 200

@app.route("/routes", methods=["GET"])
def routes():
    # Show all active routes so we can debug easily
    rules = [str(r) for r in app.url_map.iter_rules()]
    return jsonify({"routes": sorted(rules)})
