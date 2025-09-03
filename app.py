# ---------------- Part 1: Imports, Config, Utils, Guards ----------------
import os
import threading
import time
from decimal import Decimal, ROUND_DOWN
from datetime import datetime
from flask import Flask, request, jsonify

# ====== CONFIG YOU CAN TWEAK ======
APEX_SYMBOL = "BTC-USDT"
BINANCE_SYMBOL = "BTC/USDT"       # ccxt spot symbol
CANDLE_INTERVAL = "5m"            # 5-minute candles for vector/EMA
TICK_SIZE = Decimal('1')          # adjust to actual exchange tick if needed (e.g., 0.5 / 0.1)
SIZE_STEP = Decimal('0.001')      # size step
LEVERAGE = Decimal('10')

# Position sizing
TRADE_BALANCE_PCT = Decimal("0.05")  # 5% of total USDT contract wallet
MIN_ORDER_USDT    = Decimal("5")     # safety floor

# DCA / TP / SL
TREND_TP_PERCENT  = Decimal("0.75")
TREND_SL_PERCENT  = Decimal("0.5")
CTREND_TP_PERCENT = Decimal("0.5")
CTREND_SL_PERCENT = Decimal("0.5")
ALLOW_COUNTER_TREND = True

DCA_MULTIPLIER = Decimal('1.1')
DCA_STEP_PERCENT = Decimal('0.25')
DCA_STEP_MULTIPLIER = Decimal('1.05')
MAX_DCA_COUNT = 1

# MF timing window relative to vector candle close timestamp
MF_WAIT_SEC = 3600
MF_LEAD_SEC = 3600

# EMA / Vector
EMA_PERIOD = 50
VECTOR_PERIOD = 25
VECTOR_THRESHOLD = 0.70  # 70%

# Safety / dashboard
ENTRY_ENABLED = True          # will be flipped to False automatically if ApeX SDK missing
DASHBOARD_ENABLED = False
PREV_BIAS = None

# Bias (ICT-ish helpers)
ICT_EMA_SLOPE_BARS = 5
ICT_SWING_LOOKBACK = 3
ICT_BOS_BUFFER_PCT = 0.2
ICT_REQUIRE_BOS = False

# ---- Utils ----
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

# --- Try to import ApeX SDK; if missing, run in NO-TRADING mode (app still boots) ---
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

# === API Credentials (from Environment) ===
api_creds = {
    "key": os.getenv("APEX_API_KEY", ""),
    "secret": os.getenv("APEX_API_SECRET", ""),
    "passphrase": os.getenv("APEX_API_PASSPHRASE", ""),
}
zk_seeds = os.getenv("ZK_SEEDS", "")
zk_l2Key = os.getenv("ZK_L2KEY", "")

# If ApeX SDK not present, disable trading
if not APEX_SDK_OK:
    ENTRY_ENABLED = False

# === Flask App ===
app = Flask(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

# === Binance via ccxt (no pandas) ===
import ccxt
binance = ccxt.binance({"enableRateLimit": True})

def fetch_binance_closes(symbol=BINANCE_SYMBOL, interval=CANDLE_INTERVAL, limit=200):
    """
    Returns list of closes (floats). No pandas needed.
    """
    try:
        ohlcv = binance.fetch_ohlcv(symbol, timeframe=interval, limit=limit)
        return [float(row[4]) for row in ohlcv]  # close column
    except Exception as e:
        print(f"{now()} ⚠️ Error fetching Binance closes: {e}")
        return []

def compute_ema_series(values, period=EMA_PERIOD):
    """
    Simple EMA over a list of floats. Returns list of EMA values (same length).
    If not enough length, returns a flat EMA equal to first value.
    """
    if not values:
        return []
    k = 2.0 / (period + 1.0)
    ema_vals = []
    ema_prev = values[0]
    for v in values:
        ema_prev = (v * k) + (ema_prev * (1 - k))
        ema_vals.append(ema_prev)
    return ema_vals
# ----- Stats / logging -----
def ema_stats_line():
    try:
        limit = max(VECTOR_PERIOD + EMA_PERIOD, 60)
        closes = fetch_binance_closes(BINANCE_SYMBOL, CANDLE_INTERVAL, limit)
        if len(closes) < VECTOR_PERIOD + 1:
            return "📈 EMA Stats → unavailable (insufficient data)"
        ema_vals = compute_ema_series(closes, EMA_PERIOD)
        recent_closes = closes[-VECTOR_PERIOD:]
        recent_ema    = ema_vals[-VECTOR_PERIOD:]
        above = sum(1 for c,e in zip(recent_closes, recent_ema) if c > e)
        below = sum(1 for c,e in zip(recent_closes, recent_ema) if c < e)
        total = max(len(recent_closes), 1)
        return f"📈 EMA Stats → Above: {above} ({above/total*100:.1f}%) | Below: {below} ({below/total*100:.1f}%)"
    except Exception as e:
        return f"📈 EMA Stats → unavailable ({e})"

def log_event(title: str, *lines: str):
    print(f"{now()} {title}")
    for ln in lines:
        if ln: print(f"    {ln}")
    print(f"    {ema_stats_line()}")

# ----- Vector acceptance (no pandas) -----
def vector_accepted(side: str, threshold: float = VECTOR_THRESHOLD) -> bool:
    """
    LONG accept if current close > EMA and fraction of prior period above EMA < threshold.
    SHORT accept if current close < EMA and fraction of prior period below EMA < threshold.
    """
    need = EMA_PERIOD + VECTOR_PERIOD + 5
    closes = fetch_binance_closes(BINANCE_SYMBOL, CANDLE_INTERVAL, need)
    if len(closes) < VECTOR_PERIOD + 1:
        return False
    ema_vals = compute_ema_series(closes, EMA_PERIOD)
    cur_close, cur_ema = closes[-1], ema_vals[-1]
    prev_closes = closes[-VECTOR_PERIOD-1:-1]
    prev_emas   = ema_vals[-VECTOR_PERIOD-1:-1]

    if side == "LONG":
        if not (cur_close > cur_ema): return False
        frac_above = sum(1 for c,e in zip(prev_closes, prev_emas) if c > e) / max(len(prev_closes),1)
        return frac_above < threshold
    else:
        if not (cur_close < cur_ema): return False
        frac_below = sum(1 for c,e in zip(prev_closes, prev_emas) if c < e) / max(len(prev_closes),1)
        return frac_below < threshold

# ----- Bias (4h) simplified (EMA slope + price vs EMA; optional BOS) -----
BIAS = None
DEBUG_BIAS = globals().get("DEBUG_BIAS", True)

def compute_bias():
    """
    Sets global BIAS to 'LONG'/'SHORT'/None on 4h data.
    We keep this lightweight: EMA slope + price vs EMA, with optional BOS gate.
    """
    global BIAS
    try:
        limit = max(EMA_PERIOD + 60, 120)
        closes = fetch_binance_closes(BINANCE_SYMBOL, "4h", limit)
        if len(closes) < EMA_PERIOD + ICT_EMA_SLOPE_BARS + 5:
            BIAS = None
            if DEBUG_BIAS: print(f"{now()} ⚠️ ICT Bias: insufficient data")
            return
        emas = compute_ema_series(closes, EMA_PERIOD)
        ema_up   = emas[-1] > emas[-1 - ICT_EMA_SLOPE_BARS]
        ema_down = emas[-1] < emas[-1 - ICT_EMA_SLOPE_BARS]
        price_above = closes[-1] > emas[-1]
        price_below = closes[-1] < emas[-1]

        decided = None
        if ema_up and price_above:
            decided = "LONG"
        elif ema_down and price_below:
            decided = "SHORT"
        else:
            decided = None

        BIAS = decided
        if DEBUG_BIAS:
            print(f"{now()} 🧭 ICT Bias -> {BIAS or 'NEUTRAL'} | ema{EMA_PERIOD}={emas[-1]:.2f} | close={closes[-1]:.2f}")
    except Exception as e:
        BIAS = None
        print(f"{now()} ⚠️ ICT Bias error: {e}")

# ==================== Diagnostics / Health ====================
@app.before_request
def _log_req():
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
def _alive():
    return "ok", 200

# ============ Global State for vector/MF ============
POSITION_LOCK = threading.Lock()
POSITION = {
    "open": False,
    "side": None,
    "entry": None,
    "initial_size": None,
    "size": None,
    "total_cost": None,
    "dca_count": 0,
    "dca_orders": [],
    "dca_levels": [],
    "tp": None,
    "tp_id": None,
    "sl": None,
    "sl_id": None,
    "vector_side": None,
    "vector_close_timestamp": None
}
LONG_FLAGS  = {"vector": False, "vector_accepted": False, "mf": False}
SHORT_FLAGS = {"vector": False, "vector_accepted": False, "mf": False}
LONG_TIMESTAMPS  = {"vector": 0, "mf": 0}
SHORT_TIMESTAMPS = {"vector": 0, "mf": 0}

# ==================== VECTOR & MF WEBHOOKS + DEV FORCE ENTRY ====================
@app.route('/webhook_vc', methods=['POST', 'GET'], strict_slashes=False)
def webhook_vector():
    if request.method == 'GET':
        return jsonify({"ok": True, "hint": 'POST JSON {"message":"GVC"|"RVC"}'}), 200

    data = request.json or {}
    msg = str(data.get("message", "")).upper()
    ts = int(time.time())

    if msg == "GVC":
        accepted = vector_accepted("LONG")
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
        accepted = vector_accepted("SHORT")
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
    if request.method == 'GET':
        return jsonify({"ok": True, "hint": 'POST JSON {"message":"MF UP"|"MF LONG"|"MF DOWN"}'}), 200

    data = request.json or {}
    msg = str(data.get("message", "")).upper()
    now_ts = int(time.time())

    if msg in ("MF UP", "MF LONG"): side = "LONG"
    elif msg == "MF DOWN":          side = "SHORT"
    else:
        return jsonify({"status": "error", "msg": "Invalid MF message"}), 400

    with POSITION_LOCK:
        vector_ts = POSITION.get("vector_close_timestamp")
        active_vector_side = POSITION.get("vector_side")

    def latch_mf(which: str):
        with POSITION_LOCK:
            if which == "LONG":
                LONG_FLAGS["mf"] = True; LONG_TIMESTAMPS["mf"] = now_ts
                SHORT_FLAGS["mf"] = False; SHORT_TIMESTAMPS["mf"] = 0
            else:
                SHORT_FLAGS["mf"] = True; SHORT_TIMESTAMPS["mf"] = now_ts
                LONG_FLAGS["mf"] = False; LONG_TIMESTAMPS["mf"] = 0

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
# ---------------- Orders & Entry (ApeX) ----------------
if APEX_SDK_OK:
    # Initialize ApeX client only if SDK present
    client = HttpPrivateSign(
        APEX_OMNI_HTTP_MAIN,
        network_id=NETWORKID_OMNI_MAIN_ARB,
        api_key_credentials=api_creds,
        zk_seeds=zk_seeds,
        zk_l2Key=zk_l2Key
    )
    client.configs_v3()
    http_public = HttpPublic(APEX_OMNI_HTTP_MAIN)
else:
    client = None
    http_public = None

def get_usdt_contract_balance() -> Decimal:
    if not (APEX_SDK_OK and client):
        return Decimal("0")
    try:
        acct = client.get_account_v3()
        for w in acct.get("contractWallets", []):
            if w.get("token") == "USDT":
                return Decimal(str(w.get("balance", "0")))
    except Exception as e:
        print(f"{now()} ⚠️ get_usdt_contract_balance error: {e}")
    return Decimal("0")

def pick_tp_sl_for(entry_side: str) -> tuple[Decimal, Decimal]:
    if BIAS in ("LONG", "SHORT") and entry_side == BIAS:
        return TREND_TP_PERCENT, TREND_SL_PERCENT
    else:
        return CTREND_TP_PERCENT, CTREND_SL_PERCENT

def place_tp_order(close_side: str, trigger_price: Decimal, size: Decimal):
    if not (APEX_SDK_OK and client): return None
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
    if not (APEX_SDK_OK and client): return None
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
    if not (APEX_SDK_OK and client and http_public):
        print(f"{now()} ❌ Trading disabled (ApeX SDK not available).")
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
                dca_price = prev_price * (Decimal("1") - (DCA_STEP_PERCENT / Decimal("100")) * gap_mult)
            else:
                dca_price = prev_price * (Decimal("1") + (DCA_STEP_PERCENT / Decimal("100")) * gap_mult)
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

def _status(info) -> str:
    return str((info or {}).get("status", "")).upper()

def cancel_order_id(order_id: str, label: str = "") -> bool:
    if not (APEX_SDK_OK and client) or not order_id:
        return False
    try:
        client.delete_order_v3(id=str(order_id))
        print(f"{now()} 🧹 Canceled {label or 'order'} via delete_order_v3: {order_id}")
        return True
    except Exception as e:
        msg = str(e).lower()
        if any(k in msg for k in ("not found", "filled", "triggered", "conflict")):
            print(f"{now()} 🧹 {label or 'order'} {order_id} already not-cancelable (ok): {e}")
            return True
        try:
            client.cancel_order_v3(symbol=APEX_SYMBOL, orderId=str(order_id))
            print(f"{now()} 🧹 Canceled {label or 'order'} via cancel_order_v3: {order_id}")
            return True
        except Exception as e2:
            msg2 = str(e2).lower()
            if any(k in msg2 for k in ("not found", "filled", "triggered", "conflict")):
                print(f"{now()} 🧹 {label or 'order'} {order_id} already not-cancelable (ok): {e2}")
                return True
            print(f"{now()} ⚠️ cancel_order_v3({order_id}) error: {e2}")
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
                if not (APEX_SDK_OK and client): break
                if not order_id: continue
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

            # Detect closure via TP/SL terminal states
            closed = False; reason = None
            with POSITION_LOCK:
                tp_id = POSITION.get("tp_id"); sl_id = POSITION.get("sl_id")

            if tp_id and (APEX_SDK_OK and client):
                try:
                    tp_info = client.get_order_v3(symbol=APEX_SYMBOL, orderId=tp_id).get("data") or {}
                    if _status(tp_info) in TERMINAL_STATES:
                        closed, reason = True, "TP filled"
                except Exception as e:
                    print(f"{now()} ⚠️ TP status check error: {e}")

            if (not closed) and sl_id and (APEX_SDK_OK and client):
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
def vector_window_active() -> bool:
    ts = POSITION.get("vector_close_timestamp")
    if not ts: return False
    now_ts = int(time.time())
    return (ts - int(MF_LEAD_SEC)) <= now_ts <= (ts + int(MF_WAIT_SEC))

def expire_vector_if_out_of_window():
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

def decide_entry():
    long_ready  = bool(LONG_FLAGS.get("vector_accepted")) and bool(LONG_FLAGS.get("mf"))
    short_ready = bool(SHORT_FLAGS.get("vector_accepted")) and bool(SHORT_FLAGS.get("mf"))
    if long_ready == short_ready:
        return None
    proposed = "LONG" if long_ready else "SHORT"
    vec_ts = POSITION.get("vector_close_timestamp")
    if not vec_ts:
        return None
    mf_ts = int((LONG_TIMESTAMPS if proposed == "LONG" else SHORT_TIMESTAMPS).get("mf") or 0)
    earliest = vec_ts - int(MF_LEAD_SEC)
    latest   = vec_ts + int(MF_WAIT_SEC)
    if mf_ts < earliest or mf_ts > latest:
        return None
    if not ALLOW_COUNTER_TREND and BIAS in ("LONG", "SHORT") and proposed != BIAS:
        return None
    return proposed

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
                    POSITION["vector_close_timestamp"],
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
SECRET = os.getenv("SECRET_TOKEN", "")
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
    _started = True

# Start threads immediately on import (safe; idempotent)
_start_daemons_once()

@app.before_first_request
def _boot_threads():
    _start_daemons_once()

# --- Secret-guard wrappers so TV hits these (no headers/queries)  ---
def _check_secret(path_secret: str):
    if not SECRET or path_secret != SECRET:
        return jsonify({"ok": False, "error": "forbidden"}), 403
    return None

@app.route('/webhook/<path_secret>/vc', methods=['POST', 'GET'], strict_slashes=False)
def webhook_vector_secure(path_secret):
    bad = _check_secret(path_secret)
    if bad: return bad
    return webhook_vector()

@app.route('/webhook/<path_secret>/mf', methods=['POST', 'GET'], strict_slashes=False)
def webhook_mf_secure(path_secret):
    bad = _check_secret(path_secret)
    if bad: return bad
    return webhook_mf()

@app.route('/webhook/<path_secret>/force', methods=['POST', 'GET'], strict_slashes=False)
def test_force_entry_secure(path_secret):
    bad = _check_secret(path_secret)
    if bad: return bad
    return jsonify({"ok": True, "hint":
        'POST JSON {"side":"LONG|SHORT","set_bias":"LONG|SHORT"?, "allow_counter":true|false?} to /test/force_entry'})

# Local run (Render uses gunicorn with -w 1)
if __name__ == "__main__":
    _start_daemons_once()
    print(f"{now()} ✅ Bot started and awaiting Vector/MF signals... (ENTRY_ENABLED={ENTRY_ENABLED})")
    app.run(host="0.0.0.0", port=5008, threaded=True, use_reloader=False)
