import os
import threading
from queue import Queue, Empty
from time import time, sleep
from flask import Flask, request, jsonify

app = Flask(__name__)

# --- Safety: limit inbound payload size (16 KB is plenty for TV) --------------
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024  # 16 KB

# --- Read SECRET_TOKEN fresh each request -------------------------------------
def get_secret() -> str:
    return os.getenv("SECRET_TOKEN", "") or ""

# --- Diagnostics / basics -----------------------------------------------------
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

# --- Tiny dispatcher (replace stubs with your real bot calls) -----------------
def handle_vector(side: str, payload: dict):
    print(f"[DISPATCH] VECTOR {side} | sym={payload.get('symbol')} "
          f"tf={payload.get('timeframe')} price={payload.get('price')}", flush=True)
    # TODO: call your real vector handler here (e.g., latch flags / place order)

def handle_mf(direction: str, payload: dict):
    print(f"[DISPATCH] MF {direction} | sym={payload.get('symbol')} "
          f"tf={payload.get('timeframe')} t={payload.get('timenow')}", flush=True)
    # TODO: call your real MF handler

def handle_bias(bias: str, payload: dict):
    print(f"[DISPATCH] BIAS {bias} | sym={payload.get('symbol')} t={payload.get('timenow')}", flush=True)
    # TODO: update your bias state here

def handle_test_force_entry(side: str, set_bias: str | None, allow_counter: bool, payload: dict):
    print(f"[DISPATCH] TEST_FORCE side={side} set_bias={set_bias} allow_counter={allow_counter}", flush=True)
    # TODO: trigger your force-entry path here

def dispatch_tv_payload(payload: dict):
    """
    Accepts your existing TradingView messages unchanged.
      message: "GVC" | "RVC" | "MF UP" | "MF LONG" | "MF DOWN"
      type: "bias" (bias=LONG/SHORT/NEUTRAL), "test_force_entry"
    """
    msg = str(payload.get("message", "")).upper().strip()
    typ = str(payload.get("type", "")).lower().strip()

    if msg in ("GVC", "RVC"):  # Vector candles
        side = "LONG" if msg == "GVC" else "SHORT"
        handle_vector(side, payload)
        return {"kind": "vector", "side": side}

    if msg in ("MF UP", "MF LONG", "MF DOWN"):  # Money Flow
        direction = "UP" if ("UP" in msg or "LONG" in msg) else "DOWN"
        handle_mf(direction, payload)
        return {"kind": "mf", "direction": direction}

    if typ == "bias":  # Bias update
        bias = (payload.get("bias") or "NEUTRAL").upper()
        handle_bias(bias, payload)
        return {"kind": "bias", "bias": bias}

    if typ == "test_force_entry":  # Manual test/force
        side = (payload.get("side") or "LONG").upper()
        set_bias = (payload.get("set_bias") or "").upper() or None
        allow_counter = bool(payload.get("allow_counter", False))
        handle_test_force_entry(side, set_bias, allow_counter, payload)
        return {"kind": "test_force_entry", "side": side,
                "set_bias": set_bias, "allow_counter": allow_counter}

    print("[DISPATCH] UNKNOWN payload", payload, flush=True)
    return {"kind": "unknown"}

# --- In-process queue + worker thread (fast webhooks, async handling) ---------
EVENTS: Queue = Queue(maxsize=10000)

def event_consumer():
    print("[QUEUE] consumer thread online", flush=True)
    while True:
        try:
            item = EVENTS.get(timeout=1.0)  # {'payload':..., 'ts':...}
        except Empty:
            continue
        try:
            meta = dispatch_tv_payload(item["payload"])
            print(f"[QUEUE] processed kind={meta.get('kind')} in {time()-item['ts']:.3f}s", flush=True)
        except Exception as e:
            print(f"[QUEUE] ERROR processing item: {e}", flush=True)

# Start the consumer as a daemon thread (Render web dyno)
threading.Thread(target=event_consumer, daemon=True).start()

# --- Single URL with secret (no header/query needed) --------------------------
@app.route("/webhook/<path_secret>", methods=["GET", "POST"])
def webhook_url_secret(path_secret):
    secret = get_secret()
    if not secret:
        return jsonify({"ok": False, "error": "server not configured with SECRET_TOKEN"}), 500
    if path_secret != secret:
        return jsonify({"ok": False, "error": "forbidden"}), 403

    if request.method == "GET":
        return jsonify({"ok": True,
                        "hint": "POST JSON like {'message':'GVC'|'RVC'|'MF UP'|'MF DOWN'} "
                                "or {'type':'bias','bias':'LONG'} / {'type':'test_force_entry',...}"}), 200

    payload = request.get_json(silent=True) or {}
    enqueued = False
    try:
        EVENTS.put_nowait({"payload": payload, "ts": time()})
        enqueued = True
    except Exception as e:
        print(f"[QUEUE] DROP (full) {e}", flush=True)

    print("[TV] /webhook/<secret>", payload, flush=True)
    return jsonify({"ok": True, "enqueued": enqueued, "received": payload}), 200

# --- Local run only (Render uses gunicorn) ------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5008")), debug=False)
