import os
import threading
from queue import Queue, Empty
from time import time
from flask import Flask, request, jsonify

# Use your simple bot hooks
import bot_logic as bot

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024  # 16 KB cap

def get_secret() -> str:
    return os.getenv("SECRET_TOKEN", "") or ""

# ---------------- diagnostics ----------------
@app.route("/", methods=["GET"])
def home():
    return "OK", 200

@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"pong": True}), 200

@app.route("/envcheck", methods=["GET"])
def envcheck():
    s = get_secret()
    return jsonify({"has_secret": bool(s), "len": len(s)})

@app.route("/routes", methods=["GET"])
def routes():
    return jsonify({"routes": sorted(str(r) for r in app.url_map.iter_rules())})

@app.errorhandler(413)
def too_large(_):
    return jsonify({"ok": False, "error": "payload too large"}), 413

@app.after_request
def no_cache(resp):
    resp.headers["Cache-Control"] = "no-store"
    return resp

# --------------- dispatcher -> bot_logic ---------------
def handle_vector(side: str, payload: dict):
    return bot.on_vector(side, payload)

def handle_mf(direction: str, payload: dict):
    return bot.on_mf(direction, payload)

def handle_bias(bias: str, payload: dict):
    return bot.on_bias(bias, payload)

def handle_test_force_entry(side: str, set_bias: str | None, allow_counter: bool, payload: dict):
    return bot.on_force_entry(side, set_bias, allow_counter, payload)

def dispatch_tv_payload(payload: dict):
    msg = str(payload.get("message", "")).upper().strip()
    typ = str(payload.get("type", "")).lower().strip()

    if msg in ("GVC", "RVC"):
        side = "LONG" if msg == "GVC" else "SHORT"
        result = handle_vector(side, payload)
        return {"kind": "vector", "side": side, "result": result}

    if msg in ("MF UP", "MF LONG", "MF DOWN"):
        direction = "UP" if ("UP" in msg or "LONG" in msg) else "DOWN"
        result = handle_mf(direction, payload)
        return {"kind": "mf", "direction": direction, "result": result}

    if typ == "bias":
        bias = (payload.get("bias") or "NEUTRAL").upper()
        result = handle_bias(bias, payload)
        return {"kind": "bias", "bias": bias, "result": result}

    if typ == "test_force_entry":
        side = (payload.get("side") or "LONG").upper()
        set_bias = (payload.get("set_bias") or "").upper() or None
        allow_counter = bool(payload.get("allow_counter", False))
        result = handle_test_force_entry(side, set_bias, allow_counter, payload)
        return {"kind": "test_force_entry", "side": side, "set_bias": set_bias,
                "allow_counter": allow_counter, "result": result}

    print("[DISPATCH] UNKNOWN payload", payload, flush=True)
    return {"kind": "unknown"}

# --------------- fast webhook via queue ---------------
EVENTS: Queue = Queue(maxsize=10000)

def event_consumer():
    print("[QUEUE] consumer thread online", flush=True)
    while True:
        try:
            item = EVENTS.get(timeout=1.0)
        except Empty:
            continue
        try:
            meta = dispatch_tv_payload(item["payload"])
            took = time() - item["ts"]
            print(f"[QUEUE] processed kind={meta.get('kind')} in {took:.3f}s | meta={meta}", flush=True)
        except Exception as e:
            print(f"[QUEUE] ERROR processing item: {e}", flush=True)

threading.Thread(target=event_consumer, daemon=True).start()

# --------------- single URL with URL-secret ---------------
@app.route("/webhook/<path_secret>", methods=["GET", "POST"])
def webhook_url_secret(path_secret):
    secret = get_secret()
    if not secret:
        return jsonify({"ok": False, "error": "server not configured with SECRET_TOKEN"}), 500
    if path_secret != secret:
        return jsonify({"ok": False, "error": "forbidden"}), 403

    # Browser simulator (no JSON/client needed)
    if request.method == "GET":
        sim = (request.args.get("simulate") or "").lower()
        if sim:
            samples = {
                "gvc": {"source":"sim","message":"GVC","symbol":"BTCUSDT","timeframe":"5","price":12345,"timenow":"now"},
                "rvc": {"source":"sim","message":"RVC","symbol":"BTCUSDT","timeframe":"5","price":12345,"timenow":"now"},
                "mfup": {"source":"sim","message":"MF UP","symbol":"BTCUSDT","timeframe":"5","timenow":"now"},
                "mfdown": {"source":"sim","message":"MF DOWN","symbol":"BTCUSDT","timeframe":"5","timenow":"now"},
                "bias_long": {"source":"sim","type":"bias","bias":"LONG","symbol":"BTCUSDT","timenow":"now"},
                "bias_short": {"source":"sim","type":"bias","bias":"SHORT","symbol":"BTCUSDT","timenow":"now"},
                "test_short": {"source":"sim","type":"test_force_entry","side":"SHORT","set_bias":"SHORT","allow_counter":True},
            }
            payload = samples.get(sim)
            if not payload:
                return jsonify({"ok": False, "error": "unknown simulate value"}), 400
            try:
                EVENTS.put_nowait({"payload": payload, "ts": time()})
            except Exception as e:
                print(f"[QUEUE] DROP (sim full) {e}", flush=True)
                return jsonify({"ok": False, "queued": False}), 503
            return jsonify({"ok": True, "simulated": sim, "queued": True}), 200

        # No simulate param → show hint
        return jsonify({"ok": True,
                        "hint": "POST JSON like {'message':'GVC'|'RVC'|'MF UP'|'MF DOWN'} "
                                "or {'type':'bias','bias':'LONG'} / {'type':'test_force_entry',...}. "
                                "Add ?simulate=gvc/rvc/mfup/mfdown/bias_long/bias_short/test_short"}), 200

    # Normal POST path
    payload = request.get_json(silent=True) or {}
    enqueued = False
    try:
        EVENTS.put_nowait({"payload": payload, "ts": time()})
        enqueued = True
    except Exception as e:
        print(f"[QUEUE] DROP (full) {e}", flush=True)

    print("[TV] /webhook/<secret>", payload, flush=True)
    return jsonify({"ok": True, "enqueued": enqueued, "received": payload}), 200

# Local run (Render uses gunicorn)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5008")), debug=False)
