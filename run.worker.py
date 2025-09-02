import time
from datetime import datetime, timezone

print("Worker starting…", flush=True)

while True:
    now = datetime.now(timezone.utc).isoformat()
    print(f"[{now}] heartbeat ok", flush=True)
    time.sleep(60)
