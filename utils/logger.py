# utils/logger.py
import os, csv, datetime
from pathlib import Path

LOG_PATH = Path("data/prompt_log.csv")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

def log_prompt(prompt, handler="unknown", success=True, latency_ms=None, extra=None):
    new_file = not LOG_PATH.exists()
    row = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "prompt": prompt.replace("\n", " "),
        "handler": handler,
        "success": success,
        "latency_ms": latency_ms or "",
        "extra_info": str(extra or {})
    }
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if new_file:
            writer.writeheader()
        writer.writerow(row)
