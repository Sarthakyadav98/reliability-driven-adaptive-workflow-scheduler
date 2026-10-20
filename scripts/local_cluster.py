"""Start the worker agents from configs/workers.json as local processes (no Docker needed).

    python scripts/local_cluster.py              # runs until Ctrl+C
    python scripts/local_cluster.py --config configs/workers.json

Then, in another terminal:
    python -m rdas.live --compare eft-retry,adaptive --chaos 1@8:10
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from rdas.backends.http import wait_until_healthy


def start_cluster(config: str = "configs/workers.json", seed: int = 0) -> tuple[list[subprocess.Popen], list[str]]:
    spec = json.loads(Path(config).read_text(encoding="utf-8"))["workers"]
    procs, urls = [], []
    for w in spec:
        cmd = [sys.executable, "-m", "rdas.workers.agent", "--id", w["id"], "--port", str(w["port"]),
               "--speed", str(w["speed"]), "--failure-rate", str(w.get("failure_rate", 0.0)),
               "--seed", str(seed), "--enable-admin"]
        procs.append(subprocess.Popen(cmd))
        urls.append(f"http://127.0.0.1:{w['port']}")
    try:
        wait_until_healthy(urls, timeout=30)
    except Exception:
        stop_cluster(procs)
        raise
    return procs, urls


def stop_cluster(procs: list[subprocess.Popen]) -> None:
    for p in procs:
        p.terminate()
    for p in procs:
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/workers.json")
    a = ap.parse_args()
    procs, urls = start_cluster(a.config)
    print("workers healthy:", ", ".join(urls), "(Ctrl+C to stop)")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        stop_cluster(procs)


if __name__ == "__main__":
    main()
