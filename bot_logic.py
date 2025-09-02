import os
from datetime import datetime, timezone

# Toggle trading safely via env var
DRY_RUN = (os.getenv("DRY_RUN", "true").lower() in ("1","true","yes"))

def log(msg: str):
    now = datetime.now(timezone.utc).isoformat()
    print(f"[BOT] {now} | {msg}", flush=True)

# --- Replace these with your *real* implementations later --------------------
def on_vector(side: str, payload: dict):
    """
    side: 'LONG' or 'SHORT'
    payload: raw TradingView JSON
    """
    log(f"VECTOR {side} | sym={payload.get('symbol')} tf={payload.get('timeframe')} price={payload.get('price')}")
    if DRY_RUN:
        log("DRY_RUN=on (no orders placed)")
        return {"placed": False}
    # TODO: place/adjust orders here
    return {"placed": True}

def on_mf(direction: str, payload: dict):
    """
    direction: 'UP' or 'DOWN'
    """
    log(f"MF {direction} | sym={payload.get('symbol')} tf={payload.get('timeframe')}")
    return {"ok": True}

def on_bias(bias: str, payload: dict):
    """
    bias: 'LONG' | 'SHORT' | 'NEUTRAL'
    """
    log(f"BIAS -> {bias}")
    return {"ok": True}

def on_force_entry(side: str, set_bias: str | None, allow_counter: bool, payload: dict):
    log(f"FORCE ENTRY side={side} set_bias={set_bias} allow_counter={allow_counter}")
    if DRY_RUN:
        log("DRY_RUN=on (no orders placed)")
        return {"placed": False}
    # TODO: force an entry here
    return {"placed": True}
