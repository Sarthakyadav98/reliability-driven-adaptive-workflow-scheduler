"""HTTP worker agent: one per worker process or container. Standard library only.

    python -m rdas.workers.agent --id worker-1 --port 8001 --speed 1.0 --failure-rate 0.05

Endpoints
---------
GET  /health                 200 {"worker_id", "status": "ok", "busy"} | 503 while unavailable
GET  /capabilities           worker id, speed factor, slots, supported task types
POST /execute                run one attempt synchronously and return its result record
POST /cancel/<attempt_id>    best-effort cancel of a running attempt
GET  /tasks/<attempt_id>     status of a recent attempt
POST /admin/fail             {"duration_s": 5} simulate a crash: abort the running attempt and
                             report unavailable for that long (only with --enable-admin)

Result record: task_id, attempt_id, worker_id, status (succeeded | failed |
timed_out | cancelled | worker_unavailable), category, error, started_at /
ended_at (UTC ISO-8601), duration_s, output.

Configured failures (``--failure-rate``) are controlled experimental settings,
seeded per (task, attempt) for reproducibility. They make no claim about
real machine reliability.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import threading
import time
from collections import OrderedDict
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .task_registry import (MAX_PAYLOAD_BYTES, TASK_TYPES, Cancelled, InjectedFailure, TaskContext, TaskError,
                            TimedOut, Unavailable)

SLOTS = 1  # single-slot workers, matching the engine's model


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class WorkerAgent:
    def __init__(self, worker_id: str, speed: float = 1.0, failure_rate: float = 0.0, seed: int = 0,
                 enable_admin: bool = False):
        if speed <= 0:
            raise ValueError("speed must be positive")
        if not 0 <= failure_rate < 1:
            raise ValueError("failure_rate must be in [0, 1)")
        self.worker_id = worker_id
        self.speed = speed
        self.failure_rate = failure_rate
        self.seed = seed
        self.enable_admin = enable_admin
        self._lock = threading.Lock()
        self._running: TaskContext | None = None
        self._running_id: str | None = None
        self._unavailable_until = 0.0
        self._records: OrderedDict[str, dict] = OrderedDict()

    # ------------------------------------------------------------------ state
    @property
    def available(self) -> bool:
        return time.monotonic() >= self._unavailable_until

    def capabilities(self) -> dict:
        return {"worker_id": self.worker_id, "speed": self.speed, "slots": SLOTS,
                "task_types": sorted(TASK_TYPES), "configured_failure_rate": self.failure_rate}

    def health(self) -> tuple[int, dict]:
        if not self.available:
            return 503, {"worker_id": self.worker_id, "status": "unavailable"}
        return 200, {"worker_id": self.worker_id, "status": "ok", "busy": self._running is not None}

    def _remember(self, rec: dict) -> None:
        self._records[rec["attempt_id"]] = rec
        while len(self._records) > 500:
            self._records.popitem(last=False)

    def crash(self, duration_s: float) -> None:
        """Simulated crash: abort the running attempt and refuse work for ``duration_s``."""
        with self._lock:
            self._unavailable_until = time.monotonic() + duration_s
            if self._running is not None:
                self._running.abort.set()

    def cancel(self, attempt_id: str) -> bool:
        with self._lock:
            if self._running_id == attempt_id and self._running is not None:
                self._running.cancel.set()
                return True
        return False

    # ------------------------------------------------------------------ execute
    def execute(self, req: dict) -> tuple[int, dict]:
        try:
            task_id = str(req["task_id"])
            attempt_id = str(req["attempt_id"])
            task_type = req["task_type"]
            work = float(req["work"])
            spu = float(req.get("seconds_per_unit", 0.1))
            timeout_s = float(req.get("timeout_s", 30.0))
            attempt_number = int(req.get("attempt_number", 1))
            payload = req.get("payload") or {}
        except (KeyError, TypeError, ValueError) as e:
            return 400, {"error": f"bad request: {e}"}
        if task_type not in TASK_TYPES:
            return 400, {"error": f"unknown task type {task_type!r}", "allowed": sorted(TASK_TYPES)}
        if not (work > 0 and 0 < spu <= 10 and 0 < timeout_s <= 3600) or not isinstance(payload, dict):
            return 400, {"error": "work, seconds_per_unit, timeout_s or payload out of range"}
        if not self.available:
            return 503, {"error": "worker unavailable", "worker_id": self.worker_id, "status": "worker_unavailable"}

        ctx = TaskContext(work, self.speed, spu, payload, attempt_number, time.monotonic() + timeout_s)
        with self._lock:
            if self._running is not None:
                return 409, {"error": "worker busy", "worker_id": self.worker_id}
            self._running, self._running_id = ctx, attempt_id
        rec = {"task_id": task_id, "attempt_id": attempt_id, "worker_id": self.worker_id,
               "task_type": task_type, "status": "running", "category": None, "error": None,
               "started_at": _utc(), "ended_at": None, "duration_s": None, "output": None}
        self._remember(rec)
        t0 = time.monotonic()
        code = 200
        try:
            # Seeded per (task, attempt): reproducible, and independent of request timing.
            rng = random.Random(f"{self.seed}:{self.worker_id}:{task_id}:{attempt_number}")
            inject_at = rng.random() if rng.random() < self.failure_rate else None
            if inject_at is not None:
                ctx.wait(ctx.target_seconds * inject_at)
                raise InjectedFailure(f"configured failure (rate {self.failure_rate})")
            rec["output"] = TASK_TYPES[task_type](ctx)
            rec["status"] = "succeeded"
        except InjectedFailure as e:
            rec.update(status="failed", category="injected", error=str(e))
        except TimedOut as e:
            rec.update(status="timed_out", category="timeout", error=str(e))
        except Cancelled as e:
            rec.update(status="cancelled", error=str(e))
        except Unavailable as e:
            rec.update(status="worker_unavailable", category="worker_unavailable", error=str(e))
            code = 503
        except TaskError as e:
            rec.update(status="failed", category="task_error", error=str(e))
        except Exception as e:  # a bug in a task implementation is a task error, not a worker fault
            rec.update(status="failed", category="task_error", error=f"{type(e).__name__}: {e}")
        finally:
            with self._lock:
                self._running, self._running_id = None, None
        rec["ended_at"] = _utc()
        rec["duration_s"] = round(time.monotonic() - t0, 4)
        return code, rec


def make_handler(agent: WorkerAgent):
    class Handler(BaseHTTPRequestHandler):
        server_version = "rdas-worker/1.0"

        def log_message(self, fmt, *args):  # keep the console quiet; the orchestrator logs events
            pass

        def _send(self, code: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _body(self) -> dict | None:
            n = int(self.headers.get("Content-Length") or 0)
            if n > MAX_PAYLOAD_BYTES + 4096:
                return None
            try:
                return json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                return None

        def do_GET(self):
            if self.path == "/health":
                self._send(*agent.health())
            elif self.path == "/capabilities":
                self._send(200, agent.capabilities())
            elif self.path.startswith("/tasks/"):
                rec = agent._records.get(self.path.split("/", 2)[2])
                self._send(200, rec) if rec else self._send(404, {"error": "unknown attempt"})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            body = self._body()
            if body is None:
                self._send(400, {"error": "invalid or oversized JSON body"})
            elif self.path == "/execute":
                self._send(*agent.execute(body))
            elif self.path.startswith("/cancel/"):
                self._send(200, {"cancelled": agent.cancel(self.path.split("/", 2)[2])})
            elif self.path == "/admin/fail" and agent.enable_admin:
                duration = float(body.get("duration_s", 5.0))
                agent.crash(min(max(duration, 0.0), 600.0))
                self._send(200, {"worker_id": agent.worker_id, "unavailable_for_s": duration})
            else:
                self._send(404, {"error": "not found"})

    return Handler


def serve(agent: WorkerAgent, host: str = "127.0.0.1", port: int = 8000) -> ThreadingHTTPServer:
    """Create (not start) the HTTP server. ``port=0`` picks a free port."""
    server = ThreadingHTTPServer((host, port), make_handler(agent))
    server.daemon_threads = True
    return server


def main(argv: list[str] | None = None) -> None:
    env = os.environ.get
    ap = argparse.ArgumentParser(description="RDAS worker agent")
    ap.add_argument("--id", default=env("RDAS_WORKER_ID", "worker"))
    ap.add_argument("--host", default=env("RDAS_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(env("RDAS_PORT", "8000")))
    ap.add_argument("--speed", type=float, default=float(env("RDAS_SPEED", "1.0")))
    ap.add_argument("--failure-rate", type=float, default=float(env("RDAS_FAILURE_RATE", "0.0")))
    ap.add_argument("--seed", type=int, default=int(env("RDAS_SEED", "0")))
    ap.add_argument("--enable-admin", action="store_true", default=env("RDAS_ENABLE_ADMIN", "0") == "1")
    a = ap.parse_args(argv)
    agent = WorkerAgent(a.id, a.speed, a.failure_rate, a.seed, a.enable_admin)
    server = serve(agent, a.host, a.port)
    print(f"[{a.id}] listening on {a.host}:{server.server_address[1]} speed={a.speed} "
          f"failure_rate={a.failure_rate} admin={'on' if a.enable_admin else 'off'}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
