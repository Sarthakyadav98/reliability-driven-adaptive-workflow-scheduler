"""Virtual-time backend: attempt durations and worker crashes come from a FaultTrace."""

from __future__ import annotations

import heapq
import itertools
import math

from ..faults import FaultTrace
from .base import SUCCEEDED, WORKER_AVAILABLE, WORKER_UNAVAILABLE, Event, ExecutionBackend


class SimulatedBackend(ExecutionBackend):
    def __init__(self, trace: FaultTrace):
        self.trace = trace
        self._heap: list = []
        self._seq = itertools.count()

    def _push(self, t: float, ev: Event) -> None:
        heapq.heappush(self._heap, (t, next(self._seq), ev))

    def _schedule_crash(self, w: int, k: int) -> None:
        crash, recover = self.trace.crash_event(w, k)
        if not math.isinf(crash):
            self._push(crash, Event(WORKER_UNAVAILABLE, worker=w, info={"k": k, "until": recover}))

    def bind(self, engine) -> None:
        super().bind(engine)
        for w in range(len(engine.workers)):
            self._schedule_crash(w, 0)

    def start(self, attempt) -> None:
        end = self.trace.finish_time(attempt.worker, self.engine.now, attempt.expected_duration)
        self._push(end, Event(SUCCEEDED, attempt=attempt.id, worker=attempt.worker))

    def next_events(self):
        if not self._heap:
            return None
        t = self._heap[0][0]
        events = []
        while self._heap and self._heap[0][0] == t:
            _, _, ev = heapq.heappop(self._heap)
            # The crash process is part of the backend: a crash schedules its own
            # recovery, and a recovery schedules the next crash.
            if ev.kind == WORKER_UNAVAILABLE:
                self._push(ev.info["until"], Event(WORKER_AVAILABLE, worker=ev.worker, info={"k": ev.info["k"]}))
            elif ev.kind == WORKER_AVAILABLE:
                self._schedule_crash(ev.worker, ev.info["k"] + 1)
            events.append(ev)
        return t, events
