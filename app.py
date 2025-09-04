# app.py — skeleton with threads + stub webhooks (no pandas/ccxt yet)
import os, time, threading, json
from datetime import datetime
from flask import Flask, request, jsonify

app = Flask(__name__)

def now(): return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

# ---- state ----
ENTRY_ENABLED = False  # trading disabled in the skeleton (no SDK yet)
MF_WAIT_SEC   = 3600
MF_LEAD_SEC   = 3600

POSITION_LOCK = threading.Lock()
POSITION = {
    "open": False,
    "side": None,
    "vector_side": None,
    "vector_close_timestamp": None,
}

LONG_FLAGS  = {"vector": False, "vector_accepted": False, "mf": False}
SHORT_FLAGS = {"vector": False, "vector_accepted": False, "mf": False}
LONG_TS  = {"vector": 0, "mf": 0}
SHORT_TS = {"vector": 0, "mf": 0}

# ---- threads ----
_started = False
_start_ts = time.time()

def _heartbeat():
    while True:
        print(f"[{now()}] 💓 heartbeat alive | ENTRY_ENABLED={ENTRY_ENABLED}")
        time.sleep(10)

def _bias_monitor():
    while True:
        # stub: in your full bot, compute bias here
        time.sleep(60)

def _dashboard():
    last = None
    while True:
        snap = (
            POSITION["open"], POSITION["side"], POSITION.get("vector_side"),
            POSITION.get("vector_close_timestamp"),
            tuple(sorted(LONG_FLAGS.items())), tuple(sorted(SHORT_FLAGS.items()))
        )
        if snap != last:
            print(f"[{now()}] 📊 DASHBOARD | open={POSITION['open']} side={POSITION['side']} vside={POSITION.get('vector_side')} vts={POSITION.get('vector_close_timestamp')}")
            print(f"    LONG={LONG_FLAGS}  SHORT={SHORT_FLAGS}")
            last = snap
        time.sleep(2)

def decide_entry():
    # requires both vector_accepted + mf for the same side, and MF within window
    vec_ts = POSITION.get("vector_close_timestamp")
    if not vec_ts:
        return None
    earliest = vec_ts - MF_LEAD_SEC
    latest   = vec_ts + MF_WAIT_SEC
    # check long side
    if LONG_FLAGS["vector_accepted"] and LONG_FLAGS["mf"] and earliest <= LONG_TS["mf"] <= latest:
        return "LONG"
    # check short side
    if SHORT_FLAGS["vector_accepted"] and SHORT_FLAGS["mf"] and earliest <= SHORT_TS["mf"] <= latest:
        return "SHORT"
    return None

def _main_loop():
    while True:
        try:
            side = decide_entry()
            if side and not POSITION["open"]:
                print(f"[{now()}] ✅ Confluence met for {side} (skeleton) — trading disabled here.")
                # in full bot, call your place_initial_position(...)
                # reset latches to avoid spam
                if side == "LONG":
                    SHORT_FLAGS.update({"vector": False, "vector_accepted": False, "mf": False})
                    SHORT_TS.update({"vector": 0, "mf": 0})
                else:
                    LONG_FLAGS.update({"vector": False, "vector_accepted": False, "mf": False})
                    LONG_TS.update({"vector": 0, "mf": 0})
            time.sleep(1)
        except Exception as e:
            print(f"[{now()}] ⚠️ main_loop error: {e}")
            time.sleep(1)

def _start_once():
    global _started
    if _started: return
    print(f"[{now()}] 🚀 starting threads")
    threading.Thread(target=_heartbeat,   name="Heartbeat",   daemon=True).start()
    threading.Thread(target=_main_loop,   name="Main Loop",   daemon=True).start()
    threading.Thread(target=_bias_monitor,name="Bias Monitor",daemon=True).start()
    threading.Thread(target=_dashboard,   name="Dashboard",   daemon=True).start()
    _started = True

print(f"[{now()}] 🔧 app.py imported")
_start_once()  # ensure threads also start under gunicorn

@app.before_request
def _ensure(): _start_once()

# ---- routes ----
@app.route("/", methods=["GET"])
def root(): return jsonify(ok=True), 200

@app.route("/ping", methods=["GET"])
def ping(): return "pong", 200

@app.route("/alive", methods=["GET"])
def alive(): return "ok", 200

@app.route("/debug/status", methods=["GET"])
def debug_status():
    th = [t.name for t in threading.enumerate()]
    return jsonify(
        threads=th,
        uptime_sec=int(time.time()-_start_ts),
        vector_side=POSITION.get("vector_side"),
        vector_close_ts=POSITION.get("vector_close_timestamp"),
        long_flags=LONG_FLAGS,
        short_flags=SHORT_FLAGS,
        entry_enabled=ENTRY_ENABLED,
        commit=os.getenv("RENDER_GIT_COMMIT", "unknown")[:7],
        port=os.getenv("PORT", "unknown"),
    ), 200

@app.route("/webhook_vc", methods=["POST", "GET"])
def webhook_vector():
    if request.method == "GET":
        return jsonify(ok=True, hint='POST {"message":"GVC"|"RVC"}'), 200
    data = request.get_json(silent=True) or {}
    msg = str(data.get("message","")).upper()
    ts  = int(time.time())
    if msg == "GVC":
        with POSITION_LOCK:
            LONG_FLAGS.update({"vector": True, "vector_accepted": True})
            LONG_TS["vector"] = ts
            POSITION["vector_close_timestamp"] = ts
            POSITION["vector_side"] = "LONG"
            SHORT_FLAGS.update({"vector": False, "vector_accepted": False, "mf": False})
            SHORT_TS.update({"vector": 0, "mf": 0})
        print(f"[{now()}] 🟩 GVC latched (skeleton accept)")
        return jsonify(status="success", vector="GVC", accepted=True, vector_ts=ts), 200
    elif msg == "RVC":
        with POSITION_LOCK:
            SHORT_FLAGS.update({"vector": True, "vector_accepted": True})
            SHORT_TS["vector"] = ts
            POSITION["vector_close_timestamp"] = ts
            POSITION["vector_side"] = "SHORT"
            LONG_FLAGS.update({"vector": False, "vector_accepted": False, "mf": False})
            LONG_TS.update({"vector": 0, "mf": 0})
        print(f"[{now()}] 🟥 RVC latched (skeleton accept)")
        return jsonify(status="success", vector="RVC", accepted=True, vector_ts=ts), 200
    return jsonify(status="error", msg="Invalid vector"), 400

@app.route("/webhook_mf", methods=["POST", "GET"])
def webhook_mf():
    if request.method == "GET":
        return jsonify(ok=True, hint='POST {"message":"MF UP"|"MF LONG"|"MF DOWN"}'), 200
    data = request.get_json(silent=True) or {}
    msg = str(data.get("message","")).upper()
    ts  = int(time.time())
    side = "LONG" if msg in ("MF UP","MF LONG") else "SHORT" if msg == "MF DOWN" else None
    if not side:
        return jsonify(status="error", msg="Invalid MF message"), 400
    with POSITION_LOCK:
        if side == "LONG":
            LONG_FLAGS["mf"] = True;  LONG_TS["mf"] = ts
            SHORT_FLAGS["mf"] = False; SHORT_TS["mf"] = 0
        else:
            SHORT_FLAGS["mf"] = True; SHORT_TS["mf"] = ts
            LONG_FLAGS["mf"] = False;  LONG_TS["mf"] = 0
    print(f"[{now()}] 🔔 MF {side} latched (skeleton)")
    return jsonify(status="latched", side=side, mf_ts=ts), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5008, threaded=True, use_reloader=False)

