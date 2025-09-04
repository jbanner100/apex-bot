# ---------------- Part 1: Imports, Config, Utils, Clients, Globals ----------------
import os
import threading
import time
from decimal import Decimal, ROUND_DOWN
from datetime import datetime
from flask import Flask, request, jsonify
import itertools
import pandas as pd
import ccxt

# ========= Flask app =========
app = Flask(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

# ========= Config =========
APEX_SYMBOL = "BTC-USDT"
BINANCE_SYMBOL = "BTC/USDT"      # ccxt spot symbol
CANDLE_INTERVAL = "5m"           # 5-minute candles for vector/EMA
TICK_SIZE = Decimal('1')         # adjust to actual exchange tick if needed (e.g., 0.5 / 0.1)
SIZE_STEP = Decimal('0.001')     # size step
LEVERAGE = Decimal('10')

# --- Position sizing ---
TRADE_BALANCE_PCT = Decimal("0.05")  # 5% of total USDT contract wallet
MIN_ORDER_USDT    = Decimal("5")     # safety floor

# === DCA / TP / SL Variables ===
# NOTE: "0.5" means 0.5%, i.e., Decimal('0.5') == 0.5 percent.
TREND_TP_PERCENT  = Decimal("0.5")
TREND_SL_PERCENT  = Decimal("0.5")
CTREND_TP_PERCENT = Decimal("0.75")
CTREND_SL_PERCENT = Decimal("0.5")
ALLOW_COUNTER_TREND = True  # if False, block entries against BIAS

# DCA ladder config
DCA_MULTIPLIER = Decimal('1.1')
DCA_STEP_PERCENT = Decimal('0.25')       # % spacing between DCA levels (base)
DCA_STEP_MULTIPLIER = Decimal('1.05')    # geometric widening
MAX_DCA_COUNT = 1

# MF timing window relative to vector candle close timestamp
MF_WAIT_SEC  = 3600   # allow MF up to this many seconds AFTER vector
MF_LEAD_SEC  = 3600   # allow MF up to this many seconds BEFORE vector

# EMA / Vector Settings
EMA_PERIOD     = 50
VECTOR_PERIOD  = 25
VECTOR_THRESHOLD = 0.70  # 70% of last VECTOR_PERIOD candles above/below EMA

# Safety / dashboard
ENTRY_ENABLED     = True   # flip to False for kill-switch (auto-disabled if ApeX SDK missing)
DASHBOARD_ENABLED = False  # event-only logs by default (no periodic spam)
PREV_BIAS         = None

# --- Debounced flat cleanup (daemon) ---
CLEANUP_GRACE_SEC  = 180     # don't auto-clean within 3 min of any order activity
ZERO_DEBOUNCE_COUNT = 6      # need 6 consecutive zero-size reads before fallback clean
STATE = {"last_activity_ts": 0}
def mark_activity():
    STATE["last_activity_ts"] = int(time.time())

# ========= Utils =========
def now():
    return f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]"

def round_price_to_tick(price, tick):
    price = Decimal(price); tick = Decimal(tick)
    return (price / tick).to_integral_value(rounding=ROUND_DOWN) * tick

def round_size_to_step(size, step):
    size = Decimal(size); step = Decimal(step)
    if step <= 0: raise ValueError("step must be > 0")
    floored = (size // step) * step
    try:
        floored = floored.quantize(step, rounding=ROUND_DOWN)
    except Exception:
        pass
    if floored < step: return step
    return floored

def fmt_size(size):
    return format(Decimal(size).quantize(Decimal('0.000001')), 'f')

# ========= ApeX SDK (guarded import; app still boots if missing) =========
APEX_SDK_OK = True
try:
    from apexomni.constants import APEX_OMNI_HTTP_MAIN, NETWORKID_OMNI_MAIN_ARB
    from apexomni.http_private_sign import HttpPrivateSign
    from apexomni.http_public import HttpPublic
except Exception as _e:
    print(f"{now()} ⚠️ apexomni import failed: {_e} — running in NO-TRADING mode.")
    APEX_SDK_OK = False
    APEX_OMNI_HTTP_MAIN = NETWORKID_OMNI_MAIN_ARB = None
    HttpPrivateSign = HttpPublic = None
    ENTRY_ENABLED = False  # disable trading if SDK is missing

# === API Credentials (from environment; set in Render dashboard) ===
api_creds = {
    "key": os.getenv("APEX_API_KEY", ""),
    "secret": os.getenv("APEX_API_SECRET", ""),
    "passphrase": os.getenv("APEX_API_PASSPHRASE", ""),
}
zk_seeds = os.getenv("ZK_SEEDS", "")
zk_l2Key = os.getenv("ZK_L2KEY", "")

# ========= Initialize ApeX client (only if SDK & creds present) =========
client = None
http_public = None
if APEX_SDK_OK and api_creds["key"] and api_creds["secret"] and api_creds["passphrase"] and zk_seeds and zk_l2Key:
    try:
        client = HttpPrivateSign(
            APEX_OMNI_HTTP_MAIN,
            network_id=NETWORKID_OMNI_MAIN_ARB,
            api_key_credentials=api_creds,
            zk_seeds=zk_seeds,
            zk_l2Key=zk_l2Key
        )
        client.configs_v3()
        http_public = HttpPublic(APEX_OMNI_HTTP_MAIN)
        print(f"{now()} ✅ ApeX client initialized.")
    except Exception as e:
        print(f"{now()} ❌ ApeX init failed: {e} — NO-TRADING mode.")
        client = None
        http_public = None
        ENTRY_ENABLED = False
else:
    if not APEX_SDK_OK:
        print(f"{now()} ℹ️ ApeX SDK missing; NO-TRADING mode.")
    else:
        print(f"{now()} ℹ️ ApeX creds not fully set; NO-TRADING mode.")
    ENTRY_ENABLED = False

# ========= Binance (ccxt) =========
binance = ccxt.binance({"enableRateLimit": True})

# ========= Global State =========
POSITION = {
    "open": False,
    "side": None,            # "LONG"/"SHORT"
    "entry": None,
    "initial_size": None,
    "size": None,
    "total_cost": None,
    "dca_count": 0,
    "dca_orders": [],        # list of order IDs
    "dca_levels": [],        # list of Decimal prices (same order as dca_orders)
    "tp": None,
    "tp_id": None,
    "sl": None,
    "sl_id": None,
    "vector_side": None,
    "vector_close_timestamp": None
}
POSITION_LOCK = threading.Lock()

LONG_FLAGS  = {"vector": False, "vector_accepted": False, "mf": False}
SHORT_FLAGS = {"vector": False, "vector_accepted": False, "mf": False}
LONG_TIMESTAMPS  = {"vector": 0, "mf": 0}
SHORT_TIMESTAMPS = {"vector": 0, "mf": 0}

# Bias (4h) is source of TP/SL % selection only
BIAS = None
DEBUG_BIAS = globals().get("DEBUG_BIAS", True)
# ---------------- EMA / Vector Helpers ----------------
def fetch_binance_candles(symbol=BINANCE_SYMBOL, interval=CANDLE_INTERVAL, limit=50):
    try:
        ohlcv = binance.fetch_ohlcv(symbol, timeframe=interval, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
        df[['open','high','low','close','volume']] = df[['open','high','low','close','volume']].astype(float)
        return df
    except Exception as e:
        print(f"{now()} ⚠️ Error fetching Binance candles: {e}")
        return pd.DataFrame()

def compute_ema(df, period=EMA_PERIOD):
    if df is None or df.empty:
        return df
    if len(df) < period:
        df = df.copy()
        df['ema'] = df['close']
        return df
    df = df.copy()
    df['ema'] = df['close'].ewm(span=period, adjust=False).mean()
    return df

def ema_stats_line() -> str:
    try:
        df = fetch_binance_candles(symbol=BINANCE_SYMBOL, interval=CANDLE_INTERVAL,
                                   limit=VECTOR_PERIOD + EMA_PERIOD)
        df = compute_ema(df)
        recent = df.tail(VECTOR_PERIOD)
        above_count = int((recent["close"] > recent["ema"]).sum())
        below_count = int((recent["close"] < recent["ema"]).sum())
        total = int(len(recent)) or 1
        above_pct = (above_count / total) * 100
        below_pct = (below_count / total) * 100
        return f"📈 EMA Stats → Above: {above_count} ({above_pct:.1f}%) | Below: {below_count} ({below_pct:.1f}%)"
    except Exception as e:
        return f"📈 EMA Stats → unavailable ({e})"

def log_event(title: str, *lines: str):
    print(f"{now()} {title}")
    for ln in lines:
        if ln:
            print(f"    {ln}")
    print(f"    {ema_stats_line()}")

def vector_accepted_df(df: pd.DataFrame, side: str, threshold: float = VECTOR_THRESHOLD) -> bool:
    """
    Accept only if:
      - LONG (GVC): vector candle closes above EMA AND fraction of prior VECTOR_PERIOD above EMA < threshold
      - SHORT (RVC): vector candle closes below EMA AND fraction of prior VECTOR_PERIOD below EMA < threshold
    """
    if df is None or df.empty or len(df) < VECTOR_PERIOD + 1:
        return False
    if 'ema' not in df.columns:
        df = compute_ema(df)

    cur = df.iloc[-1]
    prev = df.iloc[-VECTOR_PERIOD-1:-1]
    if side == "LONG":
        if not (cur['close'] > cur['ema']):
            return False
        frac_above = (prev['close'] > prev['ema']).mean()
        return frac_above < threshold
    else:
        if not (cur['close'] < cur['ema']):
            return False
        frac_below = (prev['close'] < prev['ema']).mean()
        return frac_below < threshold

# ---------------- Bias (4h) ----------------
ICT_EMA_SLOPE_BARS = 5
ICT_SWING_LOOKBACK = 3
ICT_BOS_BUFFER_PCT = 0.2
ICT_REQUIRE_BOS = False

def compute_bias():
    """
    Bias is computed on 4h Binance Spot BTC:
    - ICT BOS detection + EMA slope + price vs EMA
    - Sets BIAS to 'LONG'/'SHORT'/None
    - Used ONLY to choose TP/SL % per trade (unless ALLOW_COUNTER_TREND=False)
    """
    global BIAS
    try:
        limit = max(EMA_PERIOD + 200, 300)
        df = fetch_binance_candles(symbol=BINANCE_SYMBOL, interval="4h", limit=limit)
        if df is None or df.empty or len(df) < EMA_PERIOD + ICT_EMA_SLOPE_BARS + 10:
            BIAS = None
            if DEBUG_BIAS:
                print(f"{now()} ⚠️ ICT Bias: insufficient data")
            return

        df = compute_ema(df, period=EMA_PERIOD)
        closes = df["close"].values
        highs  = df["high"].values
        lows   = df["low"].values
        emas   = df["ema"].values

        lb = int(ICT_SWING_LOOKBACK)
        swh = [False] * len(df)
        swl = [False] * len(df)
        for i in range(lb, len(df) - lb):
            left_max = max(highs[i - lb:i]); right_max = max(highs[i + 1:i + 1 + lb])
            if highs[i] > left_max and highs[i] >= right_max: swh[i] = True
            left_min = min(lows[i - lb:i]); right_min = min(lows[i + 1:i + 1 + lb])
            if lows[i] < left_min and lows[i] <= right_min:  swl[i] = True

        buffer_frac = float(ICT_BOS_BUFFER_PCT) / 100.0
        bos_events = []
        for i, v in enumerate(swh):
            if v:
                level = highs[i]; thresh = level * (1.0 + buffer_frac)
                j = next((k for k in range(i + 1, len(df)) if closes[k] > thresh), None)
                if j is not None: bos_events.append((j, "UP"))
        for i, v in enumerate(swl):
            if v:
                level = lows[i]; thresh = level * (1.0 - buffer_frac)
                j = next((k for k in range(i + 1, len(df)) if closes[k] < thresh), None)
                if j is not None: bos_events.append((j, "DOWN"))

        last_bos_dir = None
        if bos_events:
            _, last_bos_dir = max(bos_events, key=lambda x: x[0])

        ema_up   = emas[-1] > emas[-1 - ICT_EMA_SLOPE_BARS]
        ema_down = emas[-1] < emas[-1 - ICT_EMA_SLOPE_BARS]
        price_above = closes[-1] > emas[-1]
        price_below = closes[-1] < emas[-1]

        decided = None
        if last_bos_dir == "UP" and ema_up and price_above:
            decided = "LONG"
        elif last_bos_dir == "DOWN" and ema_down and price_below:
            decided = "SHORT"
        else:
            if not ICT_REQUIRE_BOS:
                if ema_up and price_above: decided = "LONG"
                elif ema_down and price_below: decided = "SHORT"
                else: decided = None
            else:
                decided = None

        BIAS = decided
        if DEBUG_BIAS:
            print(
                f"{now()} 🧭 ICT Bias -> {BIAS or 'NEUTRAL'} | "
                f"EMA_slope={'UP' if ema_up else 'DOWN' if ema_down else 'FLAT'} | "
                f"price_vs_EMA={'ABOVE' if price_above else 'BELOW' if price_below else 'AT'} | "
                f"ema{EMA_PERIOD}={float(emas[-1]):.2f} | close={float(closes[-1]):.2f}"
            )
    except Exception as e:
        BIAS = None
        print(f"{now()} ⚠️ ICT Bias error: {e}")

# ==================== Diagnostics / Health ====================
@app.before_request
def _log_req():
    # Do NOT read bodies here; only log method/path to avoid hangs
    try:
        ct  = request.content_type or "-"
        clen = request.content_length if request.content_length is not None else "-"
        print(f"{now()} ⇢ {request.method} {request.path} | CT={ct} | len={clen}")
    except Exception as e:
        print(f"{now()} ⇢ {request.method} {request.path} | <log err: {e}>")

@app.after_request
def _log_resp(resp):
    try:
        print(f"{now()} ⇠ {resp.status} {request.path}")
    except Exception as e:
        print(f"{now()} ⇠ <log err: {e}>")
    return resp

@app.route('/', methods=['GET'])
def _root_ok():
    return jsonify({"ok": True, "ts": int(time.time())}), 200

@app.route('/ping', methods=['GET'])
def _ping():
    return "pong", 200

@app.route('/__alive__', methods=['GET'])
@app.route('/alive', methods=['GET'])
def _alive():
    return "ok", 200
@app.route('/debug/status', methods=['GET'])
def _debug_status():
    import threading as _th
    try:
        thread_names = [t.name for t in _th.enumerate()]
    except Exception:
        thread_names = []
    return jsonify({
        "threads": thread_names,
        "has_main_loop": any(n == "Main Loop" for n in thread_names),
        "has_dca_tp_monitor": any(n == "DCA/TP Monitor" for n in thread_names),
        "has_bias_monitor": any(n == "Bias Monitor" for n in thread_names),
        "entry_enabled": bool(globals().get("ENTRY_ENABLED", True)),
        "position_open": POSITION.get("open"),
        "vector_side": POSITION.get("vector_side"),
        "vector_close_ts": POSITION.get("vector_close_timestamp"),
    }), 200



# ==================== VECTOR & MF WEBHOOKS + DEV FORCE ENTRY ====================
@app.route('/webhook_vc', methods=['POST', 'GET'], strict_slashes=False)
def webhook_vector():
    """
    Vector webhook (GVC/RVC)
    - Only updates state when ACCEPTED.
    - Rejected vectors DO NOT clear any existing accepted latch/window.
    - On accept, sets POSITION['vector_close_timestamp'] and POSITION['vector_side'],
      and clears the opposite side's latches.
    """
    if request.method == 'GET':
        return jsonify({"ok": True, "hint": 'POST JSON {"message":"GVC"|"RVC"}'}), 200

    data = request.json or {}
    msg = str(data.get("message", "")).upper()
    ts = int(time.time())

    # Fetch candles for acceptance
    try:
        need = int(EMA_PERIOD) + int(VECTOR_PERIOD) + 5
        df = fetch_binance_candles(symbol=BINANCE_SYMBOL, interval=CANDLE_INTERVAL, limit=need)
        df = compute_ema(df, period=EMA_PERIOD)
    except Exception as e:
        print(f"{now()} ⚠️ Error fetching EMA for vector: {e}")
        df = pd.DataFrame()

    if msg == "GVC":
        accepted = vector_accepted_df(df, "LONG")
        if accepted:
            with POSITION_LOCK:
                LONG_FLAGS.update({"vector": True, "vector_accepted": True})
                LONG_TIMESTAMPS["vector"] = ts
                POSITION["vector_close_timestamp"] = ts
                POSITION["vector_side"] = "LONG"
                SHORT_FLAGS.update({"vector": False, "vector_accepted": False, "mf": False})
                SHORT_TIMESTAMPS.update({"vector": 0, "mf": 0})
            window_end = ts + int(MF_WAIT_SEC)
            log_event("🟩 GVC received", "Status: ACCEPTED",
                      f"MF valid until: {window_end} (± lead {int(MF_LEAD_SEC)}s)")
        else:
            log_event("🟩 GVC received", "Status: REJECTED",
                      "Existing vector window (if any) remains latched.")
        return jsonify({"status": "success", "vector": "GVC", "accepted": accepted,
                        "vector_ts": ts if accepted else None}), 200

    elif msg == "RVC":
        accepted = vector_accepted_df(df, "SHORT")
        if accepted:
            with POSITION_LOCK:
                SHORT_FLAGS.update({"vector": True, "vector_accepted": True})
                SHORT_TIMESTAMPS["vector"] = ts
                POSITION["vector_close_timestamp"] = ts
                POSITION["vector_side"] = "SHORT"
                LONG_FLAGS.update({"vector": False, "vector_accepted": False, "mf": False})
                LONG_TIMESTAMPS.update({"vector": 0, "mf": 0})
            window_end = ts + int(MF_WAIT_SEC)
            log_event("🟥 RVC received", "Status: ACCEPTED",
                      f"MF valid until: {window_end} (± lead {int(MF_LEAD_SEC)}s)")
        else:
            log_event("🟥 RVC received", "Status: REJECTED",
                      "Existing vector window (if any) remains latched.")
        return jsonify({"status": "success", "vector": "RVC", "accepted": accepted,
                        "vector_ts": ts if accepted else None}), 200

    else:
        print(f"{now()} ⚠️ Invalid vector message: {msg}")
        return jsonify({"status": "error", "msg": "Invalid vector"}), 400

@app.route('/webhook_mf', methods=['POST', 'GET'], strict_slashes=False)
def webhook_mf():
    """
    Money Flow webhook (MF UP / MF LONG / MF DOWN):
      • If no vector_ts yet → latch MF early (wait up to MF_LEAD_SEC for a vector).
      • If vector_ts exists → accept MF only if side matches POSITION['vector_side']
        AND MF is within [vec_ts - MF_LEAD_SEC, vec_ts + MF_WAIT_SEC].
      • Latching MF on one side clears the opposite MF latch.
    """
    if request.method == 'GET':
        return jsonify({"ok": True, "hint": 'POST JSON {"message":"MF UP"|"MF LONG"|"MF DOWN"}'}), 200

    data = request.json or {}
    msg = str(data.get("message", "")).upper()
    now_ts = int(time.time())

    if msg in ("MF UP", "MF LONG"):
        side = "LONG"
    elif msg == "MF DOWN":
        side = "SHORT"
    else:
        return jsonify({"status": "error", "msg": "Invalid MF message"}), 400

    with POSITION_LOCK:
        vector_ts = POSITION.get("vector_close_timestamp")
        active_vector_side = POSITION.get("vector_side")

    def latch_mf(which: str):
        with POSITION_LOCK:
            if which == "LONG":
                LONG_FLAGS["mf"] = True;  LONG_TIMESTAMPS["mf"] = now_ts
                SHORT_FLAGS["mf"] = False; SHORT_TIMESTAMPS["mf"] = 0
            else:
                SHORT_FLAGS["mf"] = True; SHORT_TIMESTAMPS["mf"] = now_ts
                LONG_FLAGS["mf"] = False;  LONG_TIMESTAMPS["mf"] = 0

    if not vector_ts:
        latch_mf(side)
        log_event(f"🔔 MF {side} latched", f"Awaiting Vector ≤ {int(MF_LEAD_SEC)}s")
        return jsonify({"status": "latched", "side": side, "mf_ts": now_ts}), 200

    if active_vector_side and side != active_vector_side:
        log_event(f"⚠️ MF {side} ignored", f"Accepted vector side is {active_vector_side}")
        return jsonify({"status": "ignored", "msg": "MF opposite to accepted vector",
                        "vector_side": active_vector_side, "mf_ts": now_ts,
                        "vector_ts": vector_ts}), 200

    earliest = vector_ts - int(MF_LEAD_SEC)
    latest   = vector_ts + int(MF_WAIT_SEC)

    if earliest <= now_ts <= latest:
        latch_mf(side)
        log_event(f"🔔 MF {side} accepted", f"Within window [{earliest} → {latest}] (vec_ts={vector_ts})")
        return jsonify({"status": "accepted", "side": side, "mf_ts": now_ts, "vector_ts": vector_ts}), 200

    log_event(f"⚠️ MF {side} ignored", f"Outside window [{earliest} → {latest}] (now={now_ts})")
    return jsonify({"status": "ignored", "msg": "MF outside window",
                    "mf_ts": now_ts, "vector_ts": vector_ts,
                    "earliest": earliest, "latest": latest}), 200

@app.route('/test/force_entry', methods=['POST', 'GET'], strict_slashes=False, endpoint='test_force_entry_v1')
def test_force_entry_v1():
    """
    Dev-only: force confluence for LONG or SHORT and let main_loop place the trade.
    GET returns a usage hint. POST body example:
      {
        "side": "LONG" | "SHORT",
        "set_bias": "LONG" | "SHORT" | null,
        "allow_counter": true | false | null
      }
    """
    if request.method == 'GET':
        return jsonify({"ok": True, "hint": 'POST JSON {"side":"LONG|SHORT","set_bias":"LONG|SHORT"?, "allow_counter":true|false?}'}), 200

    global BIAS, ALLOW_COUNTER_TREND, ENTRY_ENABLED
    data = request.json or {}
    side = str(data.get("side", "LONG")).upper()
    if side not in ("LONG", "SHORT"):
        return jsonify({"status": "error", "msg": "side must be LONG/SHORT"}), 400

    set_bias = data.get("set_bias", None)
    if isinstance(set_bias, str) and set_bias.upper() in ("LONG", "SHORT"):
        BIAS = set_bias.upper()
        print(f"{now()} 🧭 TEST: BIAS set to {BIAS}")

    allow_counter = data.get("allow_counter", None)
    if allow_counter is True:
        ALLOW_COUNTER_TREND = True
        print(f"{now()} 🧪 TEST: ALLOW_COUNTER_TREND forced True")

    # even if NO-TRADING mode, we allow the state change; placing order will still be gated later
    ENTRY_ENABLED = True

    now_ts = int(time.time())
    with POSITION_LOCK:
        POSITION["vector_close_timestamp"] = now_ts
        POSITION["vector_side"] = side
        if side == "LONG":
            LONG_FLAGS.update({"vector": True, "vector_accepted": True, "mf": True})
            LONG_TIMESTAMPS.update({"vector": now_ts, "mf": now_ts})
            SHORT_FLAGS.update({"vector": False, "vector_accepted": False, "mf": False})
            SHORT_TIMESTAMPS.update({"vector": 0, "mf": 0})
        else:
            SHORT_FLAGS.update({"vector": True, "vector_accepted": True, "mf": True})
            SHORT_TIMESTAMPS.update({"vector": now_ts, "mf": now_ts})
            LONG_FLAGS.update({"vector": False, "vector_accepted": False, "mf": False})
            LONG_TIMESTAMPS.update({"vector": 0, "mf": 0})

    print(f"{now()} ✅ TEST: Forced confluence for {side}. main_loop should place the trade shortly.")
    return jsonify({"status": "ok", "forced_side": side, "bias": BIAS, "ts": now_ts}), 200
# ---- Vector/MF latch helpers ----
def vector_window_active() -> bool:
    """True while inside the MF window around the last accepted vector."""
    ts = POSITION.get("vector_close_timestamp")
    if not ts: return False
    now_ts = int(time.time())
    return (ts - int(MF_LEAD_SEC)) <= now_ts <= (ts + int(MF_WAIT_SEC))

def expire_vector_if_out_of_window():
    """
    If the Vector window has ended without an entry, clear vector flags & timestamp.
    (Vector stays latched while the window is active.)
    """
    with POSITION_LOCK:
        ts = POSITION.get("vector_close_timestamp")
        if not ts: return
        now_ts = int(time.time())
        in_window = (ts - int(MF_LEAD_SEC)) <= now_ts <= (ts + int(MF_WAIT_SEC))
        if in_window: return
        LONG_FLAGS.update({"vector": False, "vector_accepted": False})
        SHORT_FLAGS.update({"vector": False, "vector_accepted": False})
        POSITION["vector_close_timestamp"] = None
        POSITION["vector_side"] = None
    print(f"{now()} ⏱️ Vector window expired — cleared vector flags.")

def expire_mf_if_stale():
    """
    If MF arrived first but no Vector was accepted within MF_LEAD_SEC, clear MF latch.
    """
    now_ts = int(time.time())
    vec_ts = POSITION.get("vector_close_timestamp")
    if vec_ts is not None: return

    if LONG_FLAGS.get("mf"):
        mf_ts = LONG_TIMESTAMPS.get("mf") or 0
        if now_ts - int(mf_ts) > int(MF_LEAD_SEC):
            LONG_FLAGS["mf"] = False; LONG_TIMESTAMPS["mf"] = 0
            print(f"{now()} ⏱️ MF LONG latch expired — no Vector within {int(MF_LEAD_SEC)}s.")
    if SHORT_FLAGS.get("mf"):
        mf_ts = SHORT_TIMESTAMPS.get("mf") or 0
        if now_ts - int(mf_ts) > int(MF_LEAD_SEC):
            SHORT_FLAGS["mf"] = False; SHORT_TIMESTAMPS["mf"] = 0
            print(f"{now()} ⏱️ MF SHORT latch expired — no Vector within {int(MF_LEAD_SEC)}s.")

# ---- TP/SL % selection (based on bias at entry) ----
def pick_tp_sl_for(entry_side: str) -> tuple[Decimal, Decimal]:
    """
    Returns (tp_percent, sl_percent) as Decimal percentages based on BIAS.
    BIAS only selects the % pair; it does not time entries unless ALLOW_COUNTER_TREND=False.
    """
    if BIAS in ("LONG", "SHORT") and entry_side == BIAS:
        return TREND_TP_PERCENT, TREND_SL_PERCENT
    else:
        return CTREND_TP_PERCENT, CTREND_SL_PERCENT

# ---- Apex account helpers ----
def get_usdt_contract_balance() -> Decimal:
    """
    Returns TOTAL USDT balance from the contract wallet.
    """
    if not client:
        return Decimal("0")
    try:
        acct = client.get_account_v3()
        for w in acct.get("contractWallets", []):
            if w.get("token") == "USDT":
                return Decimal(str(w.get("balance", "0")))
    except Exception as e:
        print(f"{now()} ⚠️ get_usdt_contract_balance error: {e}")
    return Decimal("0")

# ---------------- Part 3: Orders & Entry ----------------
def place_tp_order(close_side: str, trigger_price: Decimal, size: Decimal):
    """
    TAKE_PROFIT_MARKET; reduceOnly=True
    close_side: 'SELL' when LONG position, 'BUY' when SHORT position
    """
    if not client:
        print(f"{now()} ⚠️ TP skipped — NO-TRADING mode.")
        return None
    try:
        sz = round_size_to_step(size, SIZE_STEP)
        tp_price = round_price_to_tick(trigger_price, TICK_SIZE)
        resp = client.create_order_v3(
            symbol=APEX_SYMBOL, side=close_side, type="TAKE_PROFIT_MARKET",
            triggerPrice=str(tp_price), size=fmt_size(sz), price=str(tp_price),
            reduceOnly=True, timestampSeconds=int(time.time())
        )
        tp_id = (resp.get("data") or {}).get("id")
        with POSITION_LOCK:
            POSITION["tp_id"] = tp_id; POSITION["tp"] = tp_price
        print(f"{now()} 🎯 TP placed (TAKE_PROFIT_MARKET) @ {tp_price} | id={tp_id}")
        return tp_id
    except Exception as e:
        print(f"{now()} ⚠️ Failed to place TP: {e}")
        return None

def place_sl_order(entry_side: str, last_dca_price: Decimal, sl_percent: Decimal):
    """
    STOP_MARKET; reduceOnly=True
    SL is x% away from the LAST (furthest) DCA price and never moved.
    """
    if not client:
        print(f"{now()} ⚠️ SL skipped — NO-TRADING mode.")
        return None
    try:
        if entry_side == "LONG":
            sl_target = last_dca_price * (Decimal('1') - sl_percent / Decimal('100'))
            close_side = "SELL"
        else:
            sl_target = last_dca_price * (Decimal('1') + sl_percent / Decimal('100'))
            close_side = "BUY"
        sl_price = round_price_to_tick(sl_target, TICK_SIZE)
        with POSITION_LOCK:
            base_size = POSITION["size"] or Decimal("0")
        sz = round_size_to_step(base_size, SIZE_STEP)
        resp = client.create_order_v3(
            symbol=APEX_SYMBOL, side=close_side, type="STOP_MARKET",
            triggerPrice=str(sl_price), size=fmt_size(sz), price=str(sl_price),
            reduceOnly=True, timestampSeconds=int(time.time())
        )
        sl_id = (resp.get("data") or {}).get("id")
        with POSITION_LOCK:
            POSITION["sl"] = sl_price; POSITION["sl_id"] = sl_id
        print(f"{now()} 🛑 SL placed (STOP_MARKET) @ {sl_price} | id={sl_id}")
        return sl_id
    except Exception as e:
        print(f"{now()} ⚠️ Failed to place SL: {e}")
        return None

def place_initial_position(side, tp_percent=None, sl_percent=None):
    if not (client and http_public):
        print(f"{now()} ❌ Trading disabled (ApeX SDK/creds not available).")
        return False
    try:
        if tp_percent is None or sl_percent is None:
            tp_percent, sl_percent = pick_tp_sl_for(side)

        acct = client.get_account_v3()
        usdt_balance = Decimal('0')
        for w in acct.get("contractWallets", []):
            if w.get("token") == "USDT":
                usdt_balance = Decimal(str(w.get("balance", "0"))); break
        if usdt_balance <= 0:
            print(f"{now()} ❌ USDT balance too low: {usdt_balance}"); return False

        trade_usdt = max(usdt_balance * TRADE_BALANCE_PCT, MIN_ORDER_USDT)
        ticker = http_public.ticker_v3(symbol=APEX_SYMBOL)
        mark_price = Decimal(str(ticker["data"][0]["markPrice"]))
        mark_price_rounded = round_price_to_tick(mark_price, TICK_SIZE)
        raw_size = trade_usdt * LEVERAGE / mark_price_rounded
        initial_size = round_size_to_step(raw_size, SIZE_STEP)
        side_str = "BUY" if side == "LONG" else "SELL"

        entry_resp = client.create_order_v3(
            symbol=APEX_SYMBOL, side=side_str, type="MARKET",
            size=fmt_size(initial_size), timestampSeconds=int(time.time()),
            price=str(mark_price_rounded)
        )
        entry_id = (entry_resp.get("data") or {}).get("id")
        if not entry_id:
            print(f"{now()} ❌ Initial market order failed: {entry_resp}"); return False

        with POSITION_LOCK:
            POSITION["dca_orders"] = []
        dca_prices = []
        prev_price = mark_price_rounded
        for dca_num in range(1, int(MAX_DCA_COUNT) + 1):
            gap_mult = DCA_STEP_MULTIPLIER ** (dca_num - 1)
            if side == "LONG":
                dca_price = prev_price * (Decimal("1") - (DCA_STEP_PERCENT / Decimal("100")) * gap_mult))
            else:
                dca_price = prev_price * (Decimal("1") + (DCA_STEP_PERCENT / Decimal("100")) * gap_mult))
            dca_price_rounded = round_price_to_tick(dca_price, TICK_SIZE)
            dca_qty = round_size_to_step(initial_size * (DCA_MULTIPLIER ** dca_num), SIZE_STEP)
            dca_resp = client.create_order_v3(
                symbol=APEX_SYMBOL, side=side_str, type="LIMIT",
                price=str(dca_price_rounded), size=fmt_size(dca_qty),
                timestampSeconds=int(time.time())
            )
            dca_id = (dca_resp.get("data") or {}).get("id")
            if dca_id:
                with POSITION_LOCK:
                    POSITION["dca_orders"].append(dca_id)
            dca_prices.append(dca_price_rounded)
            prev_price = dca_price_rounded

        furthest_price = (min(dca_prices) if side == "LONG" else max(dca_prices)) if dca_prices else mark_price_rounded
        if side == "LONG":
            sl_trigger = round_price_to_tick(furthest_price * (Decimal("1") - sl_percent/Decimal("100")), TICK_SIZE)
            tp_trigger = round_price_to_tick(mark_price_rounded * (Decimal("1") + tp_percent/Decimal("100")), TICK_SIZE)
            tp_side, sl_side = "SELL", "SELL"
        else:
            sl_trigger = round_price_to_tick(furthest_price * (Decimal("1") + sl_percent/Decimal("100")), TICK_SIZE)
            tp_trigger = round_price_to_tick(mark_price_rounded * (Decimal("1") - tp_percent/Decimal("100")), TICK_SIZE)
            tp_side, sl_side = "BUY", "BUY"

        tp_resp = client.create_order_v3(
            symbol=APEX_SYMBOL, side=tp_side, type="TAKE_PROFIT_MARKET",
            size=fmt_size(initial_size), reduceOnly=True,
            triggerPrice=str(tp_trigger), price=str(tp_trigger),
            timestampSeconds=int(time.time())
        )
        tp_id = (tp_resp.get("data") or {}).get("id")
        print(f"{now()} 🎯 TP placed (TAKE_PROFIT_MARKET) @ {tp_trigger} | id={tp_id}")

        sl_resp = client.create_order_v3(
            symbol=APEX_SYMBOL, side=sl_side, type="STOP_MARKET",
            size=fmt_size(initial_size), reduceOnly=True,
            triggerPrice=str(sl_trigger), price=str(sl_trigger),
            timestampSeconds=int(time.time())
        )
        sl_id = (sl_resp.get("data") or {}).get("id")
        print(f"{now()} 🛑 SL placed (STOP_MARKET) @ {sl_trigger} | id={sl_id}")

        with POSITION_LOCK:
            POSITION.update({
                "open": True, "side": side, "entry": mark_price_rounded,
                "size": initial_size, "total_cost": initial_size * mark_price_rounded,
                "dca_count": 0, "tp": tp_trigger, "tp_id": tp_id,
                "sl": sl_trigger, "sl_id": sl_id, "tp_percent": tp_percent, "sl_percent": sl_percent
            })
        print(f"{now()} ✅ {side} market order placed: {initial_size} BTC @ {mark_price_rounded}")
        return True

    except Exception as e:
        print(f"{now()} ❌ Error placing position: {e}")
        return False

# ---- TP recompute helper (weighted after DCA fills) ----
def compute_avg_entry_and_tp():
    with POSITION_LOCK:
        size = POSITION["size"]; total_cost = POSITION["total_cost"]; side = POSITION["side"]
        tp_percent = POSITION.get("tp_percent", TREND_TP_PERCENT)
    if not size or not total_cost:
        return None, None
    avg = total_cost / size
    if side == "LONG":
        new_tp = round_price_to_tick(avg * (Decimal('1') + tp_percent / Decimal('100')), TICK_SIZE)
    else:
        new_tp = round_price_to_tick(avg * (Decimal('1') - tp_percent / Decimal('100')), TICK_SIZE)
    return avg, new_tp
# ---------------- Part 4: Monitors ----------------
def _status(info) -> str:
    """Upper-cased order status from an ApeX order dict (or '')."""
    return str((info or {}).get("status", "")).upper()

def cancel_order_id(order_id: str, label: str = "") -> bool:
    """
    Best-effort cancel by known order id.
    1) Try delete_order_v3(id=...)  (official)
    2) Fall back to cancel_order_v3(symbol, orderId=...)
    Any 'not found / filled / triggered / conflict' is treated as already closed.
    """
    if not client or not order_id:
        return False
    try:
        client.delete_order_v3(id=str(order_id))
        print(f"{now()} 🧹 Canceled {label or 'order'} via delete_order_v3: {order_id}")
        return True
    except AttributeError:
        pass
    except Exception as e:
        msg = str(e).lower()
        if any(k in msg for k in ("not found", "filled", "triggered", "conflict")):
            print(f"{now()} 🧹 {label or 'order'} {order_id} already not-cancelable (ok): {e}")
            return True
        print(f"{now()} ⚠️ delete_order_v3({order_id}) error: {e}")
    try:
        client.cancel_order_v3(symbol=APEX_SYMBOL, orderId=str(order_id))
        print(f"{now()} 🧹 Canceled {label or 'order'} via cancel_order_v3: {order_id}")
        return True
    except AttributeError:
        print(f"{now()} ⚠️ cancel_order_v3 not available for {order_id}")
        return False
    except Exception as e:
        msg = str(e).lower()
        if any(k in msg for k in ("not found", "filled", "triggered", "conflict")):
            print(f"{now()} 🧹 {label or 'order'} {order_id} already not-cancelable (ok): {e}")
            return True
        print(f"{now()} ⚠️ cancel_order_v3({order_id}) error: {e}")
        return False

def cancel_dcas_local_only():
    with POSITION_LOCK:
        ids = [oid for oid in POSITION.get("dca_orders", []) if oid]
    if not ids: return
    print(f"{now()} 🧹 Cancelling {len(ids)} stored DCA orders...")
    for oid in ids:
        cancel_order_id(oid, label="DCA")
    with POSITION_LOCK:
        POSITION["dca_orders"] = []

def dca_tp_monitor():
    """
    Minimal & robust monitor:
      • Reweights TP when a DCA LIMIT fills (SL untouched).
      • Declares close ONLY when TP or SL actually reaches a terminal fill state.
      • On close: cancel the other protection, cancel all stored DCA IDs, reset state.
    """
    TERMINAL_STATES = {"FILLED", "TRIGGERED", "EXECUTED", "COMPLETED", "DONE"}
    while True:
        try:
            with POSITION_LOCK:
                open_ = POSITION["open"]
            if not open_:
                time.sleep(1); continue

            with POSITION_LOCK:
                side    = POSITION["side"]
                tp_side = "SELL" if side == "LONG" else "BUY"
                dca_ids = list(POSITION.get("dca_orders", []))

            for idx, order_id in enumerate(dca_ids):
                if not client or not order_id:
                    continue
                try:
                    info   = client.get_order_v3(symbol=APEX_SYMBOL, orderId=order_id).get("data") or {}
                    status = _status(info)
                except Exception as e:
                    print(f"{now()} ⚠️ Fetch DCA {order_id} error: {e}")
                    continue
                if status == "FILLED":
                    dca_qty  = Decimal(str(info.get("size") or "0"))
                    fill_px  = Decimal(str(info.get("avgPrice") or info.get("price") or "0"))
                    with POSITION_LOCK:
                        POSITION["size"]       = (POSITION["size"] or Decimal("0")) + dca_qty
                        POSITION["total_cost"] = (POSITION["total_cost"] or Decimal("0")) + (dca_qty * fill_px)
                        POSITION["dca_count"]  = (POSITION["dca_count"] or 0) + 1
                        POSITION["dca_orders"][idx] = None
                        cur_size = POSITION["size"]
                    with POSITION_LOCK:
                        cur_tp_id = POSITION.get("tp_id")
                    if cur_tp_id:
                        cancel_order_id(cur_tp_id, label="old TP")
                        with POSITION_LOCK:
                            POSITION["tp_id"] = None
                    avg_entry, new_tp = compute_avg_entry_and_tp()
                    if new_tp is not None:
                        tp_id = place_tp_order(tp_side, new_tp, cur_size)
                        with POSITION_LOCK:
                            POSITION["tp"] = new_tp
                        print(f"{now()} 🟢 Weighted TP -> avg={avg_entry} | TP={new_tp} | size={cur_size}")

            closed = False; reason = None
            with POSITION_LOCK:
                tp_id = POSITION.get("tp_id"); sl_id = POSITION.get("sl_id")

            if tp_id and client:
                try:
                    tp_info = client.get_order_v3(symbol=APEX_SYMBOL, orderId=tp_id).get("data") or {}
                    if _status(tp_info) in TERMINAL_STATES:
                        closed, reason = True, "TP filled"
                except Exception as e:
                    print(f"{now()} ⚠️ TP status check error: {e}")

            if (not closed) and sl_id and client:
                try:
                    sl_info = client.get_order_v3(symbol=APEX_SYMBOL, orderId=sl_id).get("data") or {}
                    if _status(sl_info) in TERMINAL_STATES:
                        closed, reason = True, "SL filled"
                except Exception as e:
                    print(f"{now()} ⚠️ SL status check error: {e}")

            if closed:
                print(f"{now()} ✅ Position closed ({reason}). Cleaning DCAs/TP/SL and resetting state.")
                if tp_id: cancel_order_id(tp_id, label="TP")
                if sl_id: cancel_order_id(sl_id, label="SL")
                cancel_dcas_local_only()
                with POSITION_LOCK:
                    POSITION.update({
                        "open": False, "side": None, "entry": None,
                        "initial_size": None, "size": None, "total_cost": None,
                        "dca_count": 0, "dca_orders": [], "dca_levels": [],
                        "tp": None, "tp_id": None, "sl": None, "sl_id": None,
                        "tp_percent": None
                    })
                    LONG_FLAGS.update({"vector": False, "vector_accepted": False, "mf": False})
                    SHORT_FLAGS.update({"vector": False, "vector_accepted": False, "mf": False})
                time.sleep(1); continue

            time.sleep(1)
        except Exception as e:
            print(f"{now()} ⚠️ dca_tp_monitor error: {e}")
            time.sleep(1)

def dashboard():
    last = None; last_print_ts = 0.0; MIN_SECS = 3
    while True:
        try:
            with POSITION_LOCK:
                snap = (
                    POSITION["open"], POSITION["side"], POSITION["size"], POSITION["entry"],
                    POSITION.get("sl"), POSITION.get("tp"),
                    BIAS,
                    tuple(sorted(LONG_FLAGS.items())),
                    tuple(sorted(SHORT_FLAGS.items())),
                    POSITION.get("vector_close_timestamp"),
                )
            now_ts = time.time()
            if snap != last and (now_ts - last_print_ts) >= MIN_SECS:
                bias_display = BIAS if BIAS else "NEUTRAL"
                print(f"{now()} 📊 DASHBOARD")
                if snap[0]:
                    print(f"    Side: {snap[1]} | Size: {snap[2]} | Entry: {snap[3]} | SL: {snap[4]} | TP: {snap[5]}")
                else:
                    vts = snap[9]
                    if vts:
                        earliest = vts - int(MF_LEAD_SEC); latest = vts + int(MF_WAIT_SEC)
                        print(f"    No open position | Bias: {bias_display}")
                        print(f"    Vector window: [{earliest} → {latest}]")
                    else:
                        print(f"    No open position | Bias: {bias_display}")
                        print(f"    Vector window: none")
                print(f"    {ema_stats_line()}")
                last = snap; last_print_ts = now_ts
            time.sleep(1)
        except Exception as e:
            print(f"{now()} ⚠️ Dashboard error: {e}")
            time.sleep(2)

def bias_monitor():
    global PREV_BIAS
    while True:
        try:
            old = BIAS
            compute_bias()
            if BIAS != old:
                print(f"{now()} 🧭 Bias changed")
                print(f"    {old or 'None'} → {BIAS or 'None'}")
                print(f"    {ema_stats_line()}")
            PREV_BIAS = BIAS
            time.sleep(60 * 60)
        except Exception as e:
            print(f"{now()} ⚠️ Bias monitor error: {e}")
            time.sleep(60)

def decide_entry():
    """
    Requires BOTH signals for the same side AND the MF timestamp within
    [vector_ts - MF_LEAD_SEC, vector_ts + MF_WAIT_SEC]. Vector-first and MF-first supported.
    """
    long_ready  = bool(LONG_FLAGS.get("vector_accepted")) and bool(LONG_FLAGS.get("mf"))
    short_ready = bool(SHORT_FLAGS.get("vector_accepted")) and bool(SHORT_FLAGS.get("mf"))
    if long_ready == short_ready:
        return None
    proposed = "LONG" if long_ready else "SHORT"
    vec_ts = POSITION.get("vector_close_timestamp")
    if not vec_ts: return None
    mf_ts = int((LONG_TIMESTAMPS if proposed == "LONG" else SHORT_TIMESTAMPS).get("mf") or 0)
    earliest = vec_ts - int(MF_LEAD_SEC); latest = vec_ts + int(MF_WAIT_SEC)
    if mf_ts < earliest or mf_ts > latest:
        return None
    if not ALLOW_COUNTER_TREND and BIAS in ("LONG", "SHORT") and proposed != BIAS:
        return None
    return proposed

def main_loop():
    while True:
        try:
            expire_vector_if_out_of_window()
            expire_mf_if_stale()
            with POSITION_LOCK:
                open_ = POSITION["open"]
            if ENTRY_ENABLED and not open_:
                side = decide_entry()
                if side in ("LONG", "SHORT"):
                    print(f"{now()} ✅ Confluence met for {side} → placing initial position")
                    ok = place_initial_position(side)
                    if ok:
                        with POSITION_LOCK:
                            if side == "LONG":
                                SHORT_FLAGS.update({"vector": False, "vector_accepted": False, "mf": False})
                                SHORT_TIMESTAMPS.update({"vector": 0, "mf": 0})
                            else:
                                LONG_FLAGS.update({"vector": False, "vector_accepted": False, "mf": False})
                                LONG_TIMESTAMPS.update({"vector": 0, "mf": 0})
                    else:
                        time.sleep(2)
            time.sleep(1)
        except Exception as e:
            print(f"{now()} ⚠️ Main loop error: {e}")
            time.sleep(1)

# ---------------- Startup (Render/Gunicorn friendly) ----------------
# ---------------- Startup (Render/Gunicorn friendly) ----------------
_started = False

def _start_daemons_once():
    """Boot the worker threads exactly once (works under Gunicorn and local)."""
    global _started
    if _started:
        return
    print(f"{now()} 🚀 Starting threads...")
    threading.Thread(target=dca_tp_monitor, name="DCA/TP Monitor", daemon=True).start()
    threading.Thread(target=main_loop,      name="Main Loop",      daemon=True).start()
    if DASHBOARD_ENABLED:
        threading.Thread(target=dashboard,  name="Dashboard",      daemon=True).start()
    threading.Thread(target=bias_monitor,   name="Bias Monitor",   daemon=True).start()
    print(f"{now()} ✅ Bot started and awaiting Vector/MF signals... (ENTRY_ENABLED={ENTRY_ENABLED})")
    _started = True

# DEBUG: prove the module is importing and the startup is called
print(f"{now()} 🔧 app.py imported, calling _start_daemons_once()")
_start_daemons_once()
print(f"{now()} 🔧 _start_daemons_once() returned")

# Safety net: ensure threads are running on any request
@app.before_request
def _ensure_threads():
    _start_daemons_once()

# Local run
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5008, threaded=True, use_reloader=False)
