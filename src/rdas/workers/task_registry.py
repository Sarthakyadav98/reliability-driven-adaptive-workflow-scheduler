"""Allowlisted demonstration task types for the worker agents.

Payloads are data, never code: there is no way to run arbitrary Python or shell
commands. Every type validates its input and is bounded in time and size.

Each task's target duration is ``work * seconds_per_unit / worker_speed``, which
emulates heterogeneous worker speed on a single host. The task does real work
(sleep, CPU hashing, data hashing) for that long and can be cancelled or timed
out at any point.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from threading import Event

MAX_PAYLOAD_BYTES = 64 * 1024
MAX_TASK_SECONDS = 300.0


class TaskError(Exception):
    """The task itself failed (bad input or bug). Not the worker's fault."""


class InjectedFailure(Exception):
    """Deliberate, configured failure used to test recovery (worker-related)."""


class Cancelled(Exception):
    pass


class TimedOut(Exception):
    pass


class Unavailable(Exception):
    """The worker became unavailable mid-task (admin 'crash')."""


@dataclass
class TaskContext:
    work: float
    speed: float
    seconds_per_unit: float
    payload: dict
    attempt_number: int  # 1 for the first attempt of this logical task, 2 for its first retry, ...
    deadline: float  # monotonic time at which the attempt times out
    cancel: Event = field(default_factory=Event)
    abort: Event = field(default_factory=Event)  # worker going down

    @property
    def target_seconds(self) -> float:
        return min(self.work * self.seconds_per_unit / self.speed, MAX_TASK_SECONDS)

    def check(self) -> None:
        if self.abort.is_set():
            raise Unavailable("worker became unavailable")
        if self.cancel.is_set():
            raise Cancelled("cancelled by orchestrator")
        if time.monotonic() > self.deadline:
            raise TimedOut("attempt exceeded its timeout")

    def wait(self, seconds: float) -> None:
        """Sleep in small steps so cancellation, timeouts and crashes are noticed quickly."""
        end = time.monotonic() + seconds
        while True:
            self.check()
            left = end - time.monotonic()
            if left <= 0:
                return
            time.sleep(min(left, 0.02))


def _sleep(ctx: TaskContext) -> dict:
    ctx.wait(ctx.target_seconds)
    return {"slept_s": round(ctx.target_seconds, 4)}


def _cpu_benchmark(ctx: TaskContext) -> dict:
    """Bounded CPU work: repeated SHA-256 until the target duration has elapsed."""
    end = time.monotonic() + ctx.target_seconds
    h = hashlib.sha256(b"rdas")
    rounds = 0
    while time.monotonic() < end:
        for _ in range(2000):
            h = hashlib.sha256(h.digest())
        rounds += 2000
        ctx.check()
    return {"rounds": rounds, "digest": h.hexdigest()[:16]}


def _hash_payload(ctx: TaskContext) -> dict:
    """Process a small validated chunk of data: iterated SHA-256 of payload['data']."""
    data = ctx.payload.get("data", "")
    rounds = ctx.payload.get("rounds", 1000)
    if not isinstance(data, str) or len(data.encode()) > MAX_PAYLOAD_BYTES:
        raise TaskError(f"'data' must be a string of at most {MAX_PAYLOAD_BYTES} bytes")
    if not isinstance(rounds, int) or not 1 <= rounds <= 100_000:
        raise TaskError("'rounds' must be an integer in [1, 100000]")
    start = time.monotonic()
    h = data.encode()
    for i in range(rounds):
        h = hashlib.sha256(h).digest()
        if i % 1000 == 0:
            ctx.check()
    ctx.wait(max(0.0, ctx.target_seconds - (time.monotonic() - start)))
    return {"digest": h.hex()[:16], "rounds": rounds}


def _fail_once(ctx: TaskContext) -> dict:
    """Deterministic failure for reproducible recovery demos: fails halfway through the
    attempt numbered payload['fail_on_attempt'] (default 1, i.e. the first try)."""
    fail_on = ctx.payload.get("fail_on_attempt", 1)
    if not isinstance(fail_on, int) or fail_on < 1:
        raise TaskError("'fail_on_attempt' must be a positive integer")
    if ctx.attempt_number == fail_on:
        ctx.wait(ctx.target_seconds / 2)
        raise InjectedFailure(f"fail_once: injected failure on attempt {fail_on}")
    ctx.wait(ctx.target_seconds)
    return {"attempt": ctx.attempt_number}


def _slow_task(ctx: TaskContext) -> dict:
    """Runs payload['factor'] times longer than expected (default 3): exercises timeouts."""
    factor = ctx.payload.get("factor", 3.0)
    if not isinstance(factor, (int, float)) or not 1 <= factor <= 20:
        raise TaskError("'factor' must be a number in [1, 20]")
    ctx.wait(ctx.target_seconds * factor)
    return {"factor": factor}


TASK_TYPES = {
    "sleep": _sleep,
    "cpu_benchmark": _cpu_benchmark,
    "hash_payload": _hash_payload,
    "fail_once": _fail_once,
    "slow_task": _slow_task,
}
