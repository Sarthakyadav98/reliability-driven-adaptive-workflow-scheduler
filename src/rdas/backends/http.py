"""Real-execution backend: attempts run on HTTP worker agents (local processes or Docker
containers) and come back as the same normalized events the simulator emits.

* Engine time is wall-clock seconds since the run started divided by
  ``seconds_per_unit``, so a task of ``work`` units takes about ``work / speed``
  engine time units, just like in simulation.
* One dispatcher thread per running attempt (workers are single-slot); a health
  thread polls ``/health``. A worker that stops answering, or answers 503, is
  reported as ``worker_unavailable``. When it answers again it is ``worker_available``.
* Outcomes map to the failure taxonomy: ``injected``, ``timeout``, ``task_error``,
  ``worker_unavailable``, ``dispatch_error``.
"""

from __future__ import annotations

import json
import queue
import threading
import time
import urllib.error
import urllib.request

from ..models import DISPATCH_ERROR, TIMEOUT, WorkerSpec
from .base import FAILED, SUCCEEDED, WORKER_AVAILABLE, WORKER_UNAVAILABLE, Event, ExecutionBackend


def _request(url: str, body: dict | None = None, timeout: float = 5.0) -> tuple[int, dict]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method="POST" if body is not None else "GET",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read() or b"{}")
        except json.JSONDecodeError:
            return e.code, {}


def discover_workers(urls: list[str], timeout: float = 3.0) -> list[WorkerSpec]:
    """Ask every agent for its capabilities. Speeds come from the agents themselves."""
    specs = []
    for i, url in enumerate(urls):
        code, caps = _request(url.rstrip("/") + "/capabilities", timeout=timeout)
        if code != 200:
            raise RuntimeError(f"{url}: /capabilities returned {code}")
        specs.append(WorkerSpec(id=i, speed=float(caps["speed"]), label=caps.get("worker_id", f"w{i}")))
    return specs


def wait_until_healthy(urls: list[str], timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    pending = list(urls)
    while pending:
        if time.monotonic() > deadline:
            raise TimeoutError(f"workers not healthy: {pending}")
        for url in list(pending):
            try:
                if _request(url.rstrip("/") + "/health", timeout=1.0)[0] == 200:
                    pending.remove(url)
            except OSError:
                pass
        time.sleep(0.2)


class HttpBackend(ExecutionBackend):
    def __init__(self, urls: list[str], seconds_per_unit: float = 0.1, timeout_factor: float = 4.0,
                 min_timeout_s: float = 2.0, health_interval: float = 0.25):
        self.urls = [u.rstrip("/") for u in urls]
        self.spu = seconds_per_unit
        self.timeout_factor = timeout_factor
        self.min_timeout_s = min_timeout_s
        self.health_interval = health_interval
        self._events: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._down: set[int] = set()
        self._stop = threading.Event()
        self._inflight = 0
        self._t0 = time.monotonic()

    # ------------------------------------------------------------------ clock
    def now(self) -> float:
        return (time.monotonic() - self._t0) / self.spu

    def bind(self, engine) -> None:
        super().bind(engine)
        self._t0 = time.monotonic()
        self._health = threading.Thread(target=self._health_loop, daemon=True)
        self._health.start()

    # ------------------------------------------------------------------ worker state
    def _mark_down(self, w: int, error: str) -> None:
        with self._lock:
            if w in self._down:
                return
            self._down.add(w)
        self._events.put(Event(WORKER_UNAVAILABLE, worker=w, info={"error": error}))

    def _mark_up(self, w: int) -> None:
        with self._lock:
            if w not in self._down:
                return
            self._down.discard(w)
        self._events.put(Event(WORKER_AVAILABLE, worker=w))

    def _health_loop(self) -> None:
        while not self._stop.is_set():
            for w, url in enumerate(self.urls):
                try:
                    code, _ = _request(url + "/health", timeout=max(self.health_interval * 2, 0.5))
                    ok = code == 200
                except OSError as e:
                    ok, code = False, str(e)
                if ok:
                    self._mark_up(w)
                else:
                    self._mark_down(w, f"health check failed ({code})")
            self._stop.wait(self.health_interval)

    # ------------------------------------------------------------------ attempts
    def start(self, attempt) -> None:
        eng = self.engine
        task = eng.workflow.tasks[attempt.task]
        # attempt_number lets deterministic test tasks (fail_once) know which try this is.
        number = sum(1 for a in eng.attempts.values() if a.task == attempt.task and a.start is not None)
        timeout_s = max(self.min_timeout_s, attempt.expected_duration * self.spu * self.timeout_factor)
        body = {"task_id": task.label, "attempt_id": str(attempt.id), "task_type": task.type,
                "work": task.work, "payload": task.payload, "seconds_per_unit": self.spu,
                "timeout_s": timeout_s, "attempt_number": number}
        with self._lock:
            self._inflight += 1
        threading.Thread(target=self._run_attempt, args=(attempt.id, attempt.worker, body, timeout_s),
                         daemon=True).start()

    def _run_attempt(self, aid: int, w: int, body: dict, timeout_s: float) -> None:
        try:
            try:
                code, rec = _request(self.urls[w] + "/execute", body, timeout=timeout_s + 5.0)
            except OSError as e:  # connection refused / reset: the worker process is gone
                self._mark_down(w, f"dispatch failed: {e}")
                return
            status = rec.get("status")
            if code == 200 and status == "succeeded":
                self._events.put(Event(SUCCEEDED, attempt=aid, worker=w, info={"record": rec}))
            elif status == "cancelled":
                pass  # the engine cancelled it (a losing replica); nothing to report
            elif code == 503 or status == "worker_unavailable":
                self._mark_down(w, rec.get("error") or "worker unavailable")
            elif status == "timed_out":
                self._events.put(Event(FAILED, attempt=aid, worker=w, category=TIMEOUT,
                                       info={"error": rec.get("error", ""), "record": rec}))
            elif status == "failed":
                self._events.put(Event(FAILED, attempt=aid, worker=w, category=rec.get("category") or "task_error",
                                       info={"error": rec.get("error", ""), "record": rec}))
            else:  # 4xx / unexpected answer
                self._events.put(Event(FAILED, attempt=aid, worker=w, category=DISPATCH_ERROR,
                                       info={"error": f"HTTP {code}: {rec.get('error', rec)}"}))
        finally:
            with self._lock:
                self._inflight -= 1

    def cancel(self, attempt) -> None:
        url = f"{self.urls[attempt.worker]}/cancel/{attempt.id}"
        threading.Thread(target=lambda: _safe(lambda: _request(url, {}, timeout=2.0)), daemon=True).start()

    def next_events(self):
        try:
            first = self._events.get(timeout=0.25)
        except queue.Empty:
            return self.now(), []  # periodic wake-up: lets the engine re-decide and check limits
        events = [first]
        while True:
            try:
                events.append(self._events.get_nowait())
            except queue.Empty:
                break
        return self.now(), events

    def close(self) -> None:
        self._stop.set()


def _safe(fn) -> None:
    try:
        fn()
    except OSError:
        pass


def crash_worker(url: str, duration_s: float) -> dict:
    """Ask an agent (started with --enable-admin) to simulate a crash for ``duration_s``."""
    return _request(url.rstrip("/") + "/admin/fail", {"duration_s": duration_s})[1]
