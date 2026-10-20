"""Execution engine shared by every scheduler and every execution backend.

Semantics
---------
* Logical task state (WAITING -> READY -> ASSIGNED -> DONE | FAILED) is tracked
  separately from attempts. A replica is another *attempt* of the same logical
  task, never another DAG node.
* Each worker runs one attempt at a time and has a priority queue of assigned
  attempts. Idle, up workers immediately start their best queued attempt.
* A task becomes READY when all predecessors succeeded and it has no live copy.
  The scheduler's ``decide`` runs after every batch of events and assigns READY
  tasks to worker queues. Assigning a task that already has a live copy creates a
  replica; the first copy to succeed wins and the others are cancelled.
* Events come from an ``ExecutionBackend``: the virtual-time simulator or real
  HTTP worker agents. Both speak the same normalized events. Events for an
  attempt that is already finished are ignored (idempotent processing), so a
  late replica failure cannot undo a logical task that has already succeeded.
* When every copy of a task has failed, the task is retried (READY again) up to
  ``max_retries`` times. Beyond that it FAILS, and so does the workflow.
* The fault & resource monitor feeds uptime, worker-related failures, recoveries
  and attempt outcomes into the reliability estimator. Task-level errors are
  retried but do not count against the worker.
"""

from __future__ import annotations

import heapq
import itertools
import math
import time as _time
from dataclasses import dataclass

from .backends.base import FAILED, SUCCEEDED, WORKER_AVAILABLE, WORKER_UNAVAILABLE, Event, ExecutionBackend
from .backends.simulation import SimulatedBackend
from .faults import FaultConfig, FaultTrace
from .models import (FAILURE_CATEGORIES, WORKER_RELATED, Attempt, RunResult, WorkerSpec, Workflow)
from .models import WORKER_UNAVAILABLE as CAT_WORKER_UNAVAILABLE
from .reliability import GammaPoissonEstimator, _EstimatorBase
from .schedulers.base import Decision, Scheduler

WAITING, READY, ASSIGNED, DONE, FAILED_TASK = "waiting", "ready", "assigned", "done", "failed"

DEFAULT_MAX_RETRIES = 5


@dataclass
class _WorkerState:
    up: bool = True
    down_since: float = 0.0
    running: Attempt | None = None


def default_ref_duration(workflow: Workflow, workers: list[WorkerSpec]) -> float:
    mean_speed = sum(w.speed for w in workers) / len(workers)
    return (workflow.total_work / len(workflow)) / mean_speed


def default_estimator(workers: list[WorkerSpec], ref_duration: float) -> GammaPoissonEstimator:
    # Operator's nominal prior: a typical task fails ~10% of the time. Deliberately
    # independent of the scenario's true failure rate.
    prior_rate = -math.log(0.9) / ref_duration
    return GammaPoissonEstimator(len(workers), prior_rate=prior_rate,
                                 prior_exposure=8 * ref_duration, half_life=25 * ref_duration)


class Engine:
    def __init__(
        self,
        workflow: Workflow,
        workers: list[WorkerSpec],
        scheduler: Scheduler,
        backend: ExecutionBackend,
        estimator: _EstimatorBase | None = None,
        ref_duration: float | None = None,
        max_time: float | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        record: bool = False,
        on_event=None,
    ):
        self.workflow = workflow
        self.workers = workers
        self.scheduler = scheduler
        self.backend = backend
        self.ref_duration = ref_duration or default_ref_duration(workflow, workers)
        self.estimator = estimator or default_estimator(workers, self.ref_duration)
        lower_bound = workflow.total_work / sum(w.speed for w in workers)
        self.max_time = max_time or 200.0 * max(lower_bound, self.ref_duration)
        self.max_retries = max_retries
        self.record = record
        self.on_event = on_event  # optional callback(time, message) for live progress

        n = len(workflow)
        self.now = 0.0
        self._seq = itertools.count()
        self._attempt_ids = itertools.count()
        self.ws = [_WorkerState() for _ in workers]
        self.queues: list[list] = [[] for _ in workers]  # heaps of (priority, seq, attempt)
        self.status = [WAITING] * n
        self.remaining_preds = [len(t.preds) for t in workflow.tasks]
        self.live: list[set[int]] = [set() for _ in range(n)]
        self.attempts: dict[int, Attempt] = {}
        self.ready_since = [math.inf] * n
        self.finish = [math.inf] * n
        self.failed_on: list[set[int]] = [set() for _ in range(n)]
        self.n_task_failures = [0] * n  # failed attempts (any copy)
        self.retries = [0] * n  # logical re-queues after all copies failed
        self.first_loss = [math.inf] * n
        self.workflow_failed = False

        self.busy = self.useful = self.wasted = 0.0
        self.n_crashes = self.n_revoked = self.n_replicas = self.n_cancelled = 0
        self.failures_by_category = {c: 0 for c in FAILURE_CATEGORIES}
        self.decision_time = 0.0
        self.n_decisions = 0
        self.log: list[tuple[float, str]] = []
        self.reliability_timeline: list[tuple[float, list[dict]]] = []
        self.down_intervals: list[tuple[int, float, float]] = []

    # ------------------------------------------------------------------ utils
    def _note(self, msg: str) -> None:
        if self.record:
            self.log.append((self.now, msg))
        if self.on_event is not None:
            self.on_event(self.now, msg)

    def _snapshot(self) -> None:
        if self.record:
            self.reliability_timeline.append((self.now, self.estimator.snapshot(self.ref_duration)))

    def _make_ready(self, tid: int) -> None:
        self.status[tid] = READY
        self.ready_since[tid] = self.now

    def _name(self, tid: int) -> str:
        return self.workflow.tasks[tid].label

    def _wname(self, w: int) -> str:
        return f"W{w}" + (f" ({self.workers[w].label})" if self.workers[w].label else "")

    # ------------------------------------------------------------------ run
    def run(self) -> RunResult:
        self.scheduler.setup(SimView(self))
        self.backend.bind(self)
        for tid in self.workflow.entry_tasks:
            self._make_ready(tid)
        self._snapshot()
        self._decide()
        self._dispatch()

        n_done = 0
        total = len(self.workflow)
        try:
            while n_done < total and not self.workflow_failed:
                nxt = self.backend.next_events()
                if nxt is None:
                    break
                t, events = nxt
                if t > self.max_time:
                    self._note("time limit reached")
                    break
                self._advance(t)
                for ev in events:
                    n_done += self._handle(ev)
                if n_done < total and not self.workflow_failed:
                    self._decide()
                    self._dispatch()
        finally:
            self.backend.close()
        return self._result(n_done == total)

    def _advance(self, t: float) -> None:
        dt = t - self.now
        if dt > 0:
            for w, st in enumerate(self.ws):
                if st.up:
                    self.estimator.observe_uptime(w, dt)
            self.now = t

    def _handle(self, ev: Event) -> int:
        if ev.kind == SUCCEEDED:
            return self._on_succeeded(ev.attempt)
        if ev.kind == FAILED:
            self._on_failed(ev.attempt, ev.category or "task_error", ev.info.get("error", ""))
        elif ev.kind == WORKER_UNAVAILABLE:
            self._on_worker_down(ev.worker, ev.info.get("until"), ev.info.get("error", ""))
        elif ev.kind == WORKER_AVAILABLE:
            self._on_worker_up(ev.worker)
        return 0

    # ------------------------------------------------------------------ events
    def _live_attempt(self, aid: int | None) -> Attempt | None:
        a = self.attempts.get(aid) if aid is not None else None
        if a is None or a.outcome is not None or a.start is None:
            return None  # unknown, already finished/cancelled (late duplicate), or never started
        return a

    def _on_succeeded(self, aid: int) -> int:
        a = self._live_attempt(aid)
        if a is None:
            return 0
        w = a.worker
        a.end, a.outcome = self.now, "success"
        dur = a.end - a.start
        self.busy += dur
        self.useful += dur
        self.ws[w].running = None
        self.estimator.observe_attempt(w, a.expected_duration, dur, True)
        tid = a.task
        self.live[tid].discard(aid)
        self.status[tid] = DONE
        self.finish[tid] = self.now
        self._note(f"{self._name(tid)} succeeded on W{w} (took {dur:.1f})")
        for other in list(self.live[tid]):
            self._cancel(self.attempts[other], "cancelled")
        for s in self.workflow.tasks[tid].succs:
            self.remaining_preds[s] -= 1
            if self.remaining_preds[s] == 0:
                self._make_ready(s)
        self._snapshot()
        return 1

    def _end_attempt_failed(self, a: Attempt, category: str) -> None:
        a.end, a.outcome, a.category = self.now, "failed", category
        spent = a.end - a.start
        self.busy += spent
        self.wasted += spent
        self.ws[a.worker].running = None
        self.failures_by_category[category] = self.failures_by_category.get(category, 0) + 1
        if category in WORKER_RELATED:
            self.estimator.observe_attempt(a.worker, a.expected_duration, spent, False)
        self._task_lost(a)

    def _task_lost(self, a: Attempt) -> None:
        tid = a.task
        self.live[tid].discard(a.id)
        self.failed_on[tid].add(a.worker)
        self.n_task_failures[tid] += 1
        if self.live[tid] or self.status[tid] == DONE:
            return  # another copy is still alive
        self.first_loss[tid] = min(self.first_loss[tid], self.now)
        if self.retries[tid] >= self.max_retries:
            self.status[tid] = FAILED_TASK
            self.workflow_failed = True
            self._note(f"{self._name(tid)} FAILED permanently after {self.retries[tid]} retries")
            return
        self.retries[tid] += 1
        self._make_ready(tid)

    def _on_failed(self, aid: int, category: str, error: str) -> None:
        a = self._live_attempt(aid)
        if a is None:
            return
        if category in WORKER_RELATED:
            self.estimator.observe_failure(a.worker)
        self._note(f"{self._name(a.task)} FAILED on W{a.worker} [{category}]" + (f": {error}" if error else ""))
        self._end_attempt_failed(a, category)
        self._snapshot()

    def _cancel(self, a: Attempt, outcome: str) -> None:
        """Withdraw a live copy: running copies are killed, queued copies removed."""
        w = a.worker
        if a.start is not None:  # running
            a.end = self.now
            spent = a.end - a.start
            self.busy += spent
            self.wasted += spent
            self.ws[w].running = None
            self.backend.cancel(a)
        else:
            self.queues[w] = [e for e in self.queues[w] if e[2] is not a]
            heapq.heapify(self.queues[w])
        a.outcome = outcome
        self.live[a.task].discard(a.id)
        if outcome == "cancelled":
            self.n_cancelled += 1
            self._note(f"  cancelled redundant copy of {self._name(a.task)} on W{w}")

    def _on_worker_down(self, w: int, until: float | None, error: str = "") -> None:
        st = self.ws[w]
        if not st.up:
            return
        st.up, st.down_since = False, self.now
        self.n_crashes += 1
        self.estimator.observe_crash(w)
        self.down_intervals.append((w, self.now, until if until is not None else math.inf))
        a = st.running
        tail = f" (down until {until:.1f})" if until is not None else ""
        if a is not None:
            self._note(f"W{w} UNAVAILABLE{tail} - running {self._name(a.task)} lost after {self.now - a.start:.1f}")
            self._end_attempt_failed(a, CAT_WORKER_UNAVAILABLE)
        else:
            self._note(f"W{w} UNAVAILABLE while idle{tail}" + (f": {error}" if error else ""))
        self._snapshot()

    def _on_worker_up(self, w: int) -> None:
        st = self.ws[w]
        if st.up:
            return
        st.up = True
        self.estimator.observe_recovery(w, self.now - st.down_since)
        for i, (ww, s, e) in enumerate(self.down_intervals):
            if ww == w and math.isinf(e):
                self.down_intervals[i] = (ww, s, self.now)
        self._note(f"W{w} available again")

    # ------------------------------------------------------------------ decisions
    def _decide(self) -> None:
        had_ready = any(s == READY for s in self.status)
        view = SimView(self)
        t0 = _time.perf_counter()
        dec: Decision = self.scheduler.decide(view)
        elapsed = _time.perf_counter() - t0
        if had_ready or dec.revoke:
            self.decision_time += elapsed
            self.n_decisions += 1
        for note in dec.notes:
            self._note(note)
        for tid, w in dec.revoke:
            self._revoke(tid, w)
        for asg in dec.assign:
            self._assign(asg.task, asg.worker, asg.priority, asg.reason)

    def _revoke(self, tid: int, w: int) -> None:
        for aid in list(self.live[tid]):
            a = self.attempts[aid]
            if a.worker == w and a.start is None:
                self._cancel(a, "revoked")
                self.n_revoked += 1
                self._note(f"  reassign: {self._name(tid)} withdrawn from W{w}'s queue")
                if not self.live[tid] and self.status[tid] != DONE:
                    self._make_ready(tid)
                return

    def _assign(self, tid: int, w: int, priority: float | None, reason: str = "") -> None:
        if self.status[tid] in (DONE, WAITING, FAILED_TASK):
            raise ValueError(f"scheduler assigned {self._name(tid)} in state {self.status[tid]}")
        if any(self.attempts[aid].worker == w for aid in self.live[tid]):
            return  # duplicate copy on the same worker is pointless
        kind = "replica" if self.live[tid] else "primary"
        dur = self.workflow.tasks[tid].work / self.workers[w].speed
        a = Attempt(id=next(self._attempt_ids), task=tid, worker=w, kind=kind, queued_at=self.now,
                    expected_duration=dur, reason=reason,
                    est_p_success=self.estimator.p_success(w, dur * self.estimator.perf_ratio(w)))
        self.attempts[a.id] = a
        self.live[tid].add(a.id)
        self.status[tid] = ASSIGNED
        if kind == "replica":
            self.n_replicas += 1
        prio = priority if priority is not None else float(next(self._seq))
        heapq.heappush(self.queues[w], (prio, next(self._seq), a))

    def _dispatch(self) -> None:
        for w, st in enumerate(self.ws):
            if not st.up or st.running is not None or not self.queues[w]:
                continue
            _, _, a = heapq.heappop(self.queues[w])
            a.start = self.now
            st.running = a
            self._note(f"{self._name(a.task)} started on W{w}" + (" (replica)" if a.kind == "replica" else ""))
            self.backend.start(a)

    # ------------------------------------------------------------------ results
    def attempt_records(self) -> list[dict]:
        return [a.as_record(self.workflow) for a in sorted(self.attempts.values(), key=lambda a: a.id)]

    def _result(self, completed: bool) -> RunResult:
        makespan = max(self.finish) if completed else math.inf
        deadline = self.workflow.deadline
        failed = sum(1 for a in self.attempts.values() if a.outcome == "failed")
        recovered = [self.finish[t] - self.first_loss[t] for t in range(len(self.workflow))
                     if math.isfinite(self.first_loss[t]) and math.isfinite(self.finish[t])]
        extra = {"n_task_retries": sum(self.n_task_failures)}
        extra.update(self.scheduler.stats())
        return RunResult(
            scheduler=self.scheduler.name,
            workflow=self.workflow.name,
            n_tasks=len(self.workflow),
            n_workers=len(self.workers),
            completed=completed,
            makespan=makespan,
            deadline=deadline,
            deadline_met=completed and (deadline is None or makespan <= deadline + 1e-9),
            n_attempts=len(self.attempts),
            n_failed_attempts=failed,
            n_replicas=self.n_replicas,
            n_cancelled=self.n_cancelled,
            n_revoked=self.n_revoked,
            n_crashes=self.n_crashes,
            busy_time=self.busy,
            useful_time=self.useful,
            wasted_time=self.wasted,
            decision_time_total=self.decision_time,
            n_decisions=self.n_decisions,
            end_time=self.now,
            tasks_succeeded=sum(1 for s in self.status if s == DONE),
            n_retries=sum(self.retries),
            replicas_started=sum(1 for a in self.attempts.values() if a.kind == "replica" and a.start is not None),
            recovery_time_mean=sum(recovered) / len(recovered) if recovered else 0.0,
            failures_by_category=dict(self.failures_by_category),
            extra=extra,
        )


class Simulation(Engine):
    """Engine + virtual-time backend driven by a seed-deterministic fault trace."""

    def __init__(self, workflow: Workflow, workers: list[WorkerSpec], scheduler: Scheduler,
                 faults: FaultConfig | None = None, seed: int = 0, ref_duration: float | None = None, **kw):
        ref = ref_duration or default_ref_duration(workflow, workers)
        self.faults = faults or FaultConfig.fault_free()
        self.seed = seed
        self.trace = FaultTrace(workers, self.faults, ref, seed)
        super().__init__(workflow, workers, scheduler, SimulatedBackend(self.trace), ref_duration=ref, **kw)


class SimView:
    """Read-only window onto the engine for schedulers."""

    def __init__(self, sim: Engine):
        self._s = sim

    @property
    def now(self) -> float:
        return self._s.now

    @property
    def workflow(self) -> Workflow:
        return self._s.workflow

    @property
    def workers(self) -> list[WorkerSpec]:
        return self._s.workers

    @property
    def estimator(self) -> _EstimatorBase:
        return self._s.estimator

    @property
    def ref_duration(self) -> float:
        return self._s.ref_duration

    def ready_tasks(self) -> list[int]:
        s = self._s
        ready = [t for t, st in enumerate(s.status) if st == READY]
        ready.sort(key=lambda t: (s.ready_since[t], t))
        return ready

    def imminent_tasks(self) -> int:
        """WAITING tasks whose predecessors are all done or already have a live copy:
        demand that will arrive within roughly one task duration."""
        s = self._s
        n = 0
        for t, st in enumerate(s.status):
            if st == WAITING and all(s.status[p] in (DONE, ASSIGNED) for p in s.workflow.tasks[t].preds):
                n += 1
        return n

    def is_up(self, w: int) -> bool:
        return self._s.ws[w].up

    def is_idle(self, w: int) -> bool:
        st = self._s.ws[w]
        return st.up and st.running is None and not self._s.queues[w]

    def failed_on(self, tid: int) -> set[int]:
        return set(self._s.failed_on[tid])

    def n_failures(self, tid: int) -> int:
        return self._s.n_task_failures[tid]

    def copies(self, tid: int) -> list[tuple[int, str]]:
        """(worker, "running" | "queued") for each live copy of a task."""
        s = self._s
        out = []
        for aid in s.live[tid]:
            a = s.attempts[aid]
            out.append((a.worker, "running" if a.start is not None else "queued"))
        return out

    def queued_tasks(self, w: int) -> list[int]:
        return [e[2].task for e in sorted(self._s.queues[w], key=lambda e: (e[0], e[1]))]

    def running_task(self, w: int) -> tuple[int, float, float] | None:
        """(task, start, nominal expected end) of the attempt running on w."""
        a = self._s.ws[w].running
        if a is None:
            return None
        return (a.task, a.start, a.start + a.expected_duration)

    def nominal_duration(self, tid: int, w: int) -> float:
        return self._s.workflow.tasks[tid].work / self._s.workers[w].speed

    def backlog(self, w: int, perf_aware: bool = False, exclude_task: int | None = None) -> float:
        """Expected time from now until w could start a new task (nominal speeds by default).

        ``perf_aware`` scales durations by the observed slowdown ratio and adds the
        expected remaining downtime of a crashed worker.
        """
        s = self._s
        scale = s.estimator.perf_ratio(w) if perf_aware else 1.0
        t = 0.0
        st = s.ws[w]
        if not st.up and perf_aware:
            # Repair times are exponential, i.e. memoryless: however long the worker has
            # been down, the expected remaining downtime is still the mean repair time.
            t += s.estimator.expected_downtime(w)
        if st.running is not None:
            a = st.running
            t += max(a.start + a.expected_duration * scale - s.now, 0.0)
        for _, _, a in s.queues[w]:
            if a.task != exclude_task:
                t += a.expected_duration * scale
        return t

    def wait_before(self, w: int, tid: int, perf_aware: bool = False) -> float:
        """Expected time until queued task ``tid`` starts on w (only work queued ahead of it)."""
        s = self._s
        entry = next((e for e in s.queues[w] if e[2].task == tid), None)
        if entry is None:
            return self.backlog(w, perf_aware)
        key = (entry[0], entry[1])
        total = self.backlog(w, perf_aware)
        scale = s.estimator.perf_ratio(w) if perf_aware else 1.0
        behind = sum(a.expected_duration * scale for p, q, a in s.queues[w] if (p, q) >= key)
        return total - behind
