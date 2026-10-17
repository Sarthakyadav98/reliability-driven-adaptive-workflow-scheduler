"""Discrete-event execution engine shared by every scheduler.

Semantics
---------
* Each worker runs one attempt at a time and has a priority queue of attempts
  assigned to it. Idle, up workers immediately start their best queued attempt.
* A task becomes READY when all predecessors are done and it has no live copy.
  The scheduler's ``decide`` is called after every event time and assigns READY
  tasks to worker queues. Assigning a task that already has a live copy creates
  a replica; the first copy to succeed wins and the others are cancelled.
* Crashes come from a ``FaultTrace``, which is independent of scheduling.
  A crash kills the running attempt (its task becomes READY again unless another
  copy is alive). Queued attempts wait for recovery unless the scheduler revokes
  them, which is how reassignment works.
* The fault & resource monitor feeds uptime, crash, recovery and attempt outcomes
  into the reliability estimator, which every scheduler can read.
"""

from __future__ import annotations

import heapq
import itertools
import math
import time as _time
from dataclasses import dataclass

from .faults import FaultConfig, FaultTrace
from .models import Attempt, RunResult, WorkerSpec, Workflow
from .reliability import GammaPoissonEstimator, _EstimatorBase
from .schedulers.base import Decision, Scheduler

WAITING, READY, ASSIGNED, DONE = "waiting", "ready", "assigned", "done"


@dataclass
class _WorkerState:
    up: bool = True
    down_since: float = 0.0
    running: Attempt | None = None
    crash_k: int = 0


def default_ref_duration(workflow: Workflow, workers: list[WorkerSpec]) -> float:
    mean_speed = sum(w.speed for w in workers) / len(workers)
    return (workflow.total_work / len(workflow)) / mean_speed


def default_estimator(workers: list[WorkerSpec], ref_duration: float) -> GammaPoissonEstimator:
    # Operator's nominal prior: a typical task fails ~10% of the time. Deliberately
    # independent of the scenario's true failure rate.
    prior_rate = -math.log(0.9) / ref_duration
    return GammaPoissonEstimator(len(workers), prior_rate=prior_rate,
                                 prior_exposure=8 * ref_duration, half_life=25 * ref_duration)


class Simulation:
    def __init__(
        self,
        workflow: Workflow,
        workers: list[WorkerSpec],
        scheduler: Scheduler,
        faults: FaultConfig | None = None,
        seed: int = 0,
        estimator: _EstimatorBase | None = None,
        ref_duration: float | None = None,
        max_time: float | None = None,
        record: bool = False,
    ):
        self.workflow = workflow
        self.workers = workers
        self.scheduler = scheduler
        self.faults = faults or FaultConfig.fault_free()
        self.seed = seed
        self.ref_duration = ref_duration or default_ref_duration(workflow, workers)
        self.trace = FaultTrace(workers, self.faults, self.ref_duration, seed)
        self.estimator = estimator or default_estimator(workers, self.ref_duration)
        lower_bound = workflow.total_work / sum(w.speed for w in workers)
        self.max_time = max_time or 200.0 * max(lower_bound, self.ref_duration)
        self.record = record

        n = len(workflow)
        self.now = 0.0
        self._heap: list = []
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
        self.n_task_failures = [0] * n

        self.busy = self.useful = self.wasted = 0.0
        self.n_crashes = self.n_revoked = self.n_replicas = self.n_cancelled = 0
        self.decision_time = 0.0
        self.n_decisions = 0
        self.log: list[tuple[float, str]] = []
        self.reliability_timeline: list[tuple[float, list[dict]]] = []
        self.down_intervals: list[tuple[int, float, float]] = []

    # ------------------------------------------------------------------ utils
    def _push(self, t: float, kind: str, data) -> None:
        heapq.heappush(self._heap, (t, next(self._seq), kind, data))

    def _note(self, msg: str) -> None:
        if self.record:
            self.log.append((self.now, msg))

    def _snapshot(self) -> None:
        if self.record:
            self.reliability_timeline.append((self.now, self.estimator.snapshot(self.ref_duration)))

    def _make_ready(self, tid: int) -> None:
        self.status[tid] = READY
        self.ready_since[tid] = self.now

    # ------------------------------------------------------------------ run
    def run(self) -> RunResult:
        self.scheduler.setup(SimView(self))
        for w in range(len(self.workers)):
            self._schedule_next_crash(w)
        for tid in self.workflow.entry_tasks:
            self._make_ready(tid)
        self._snapshot()
        self._decide()
        self._dispatch()

        n_done = 0
        total = len(self.workflow)
        while self._heap and n_done < total:
            t = self._heap[0][0]
            if t > self.max_time:
                break
            self._advance(t)
            while self._heap and self._heap[0][0] == t:
                _, _, kind, data = heapq.heappop(self._heap)
                if kind == "complete":
                    n_done += self._on_complete(data)
                elif kind == "crash":
                    self._on_crash(*data)
                elif kind == "recover":
                    self._on_recover(*data)
            if n_done < total:
                self._decide()
                self._dispatch()
        return self._result(n_done == total)

    def _advance(self, t: float) -> None:
        dt = t - self.now
        if dt > 0:
            for w, st in enumerate(self.ws):
                if st.up:
                    self.estimator.observe_uptime(w, dt)
        self.now = t

    def _schedule_next_crash(self, w: int) -> None:
        crash, recover = self.trace.crash_event(w, self.ws[w].crash_k)
        if not math.isinf(crash):
            self._push(crash, "crash", (w, self.ws[w].crash_k))

    # ------------------------------------------------------------------ events
    def _on_complete(self, aid: int) -> int:
        a = self.attempts[aid]
        if a.outcome is not None:
            return 0  # attempt was killed or cancelled earlier
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
        self._note(f"T{tid} finished on W{w} (took {dur:.1f})")
        for other in list(self.live[tid]):
            self._cancel(self.attempts[other], "cancelled")
        for s in self.workflow.tasks[tid].succs:
            self.remaining_preds[s] -= 1
            if self.remaining_preds[s] == 0:
                self._make_ready(s)
        self._snapshot()
        return 1

    def _cancel(self, a: Attempt, outcome: str) -> None:
        """Withdraw a live copy: running copies are killed, queued copies removed."""
        w = a.worker
        if a.start is not None:  # running
            a.end = self.now
            spent = a.end - a.start
            self.busy += spent
            self.wasted += spent
            self.ws[w].running = None
        else:
            self.queues[w] = [e for e in self.queues[w] if e[2] is not a]
            heapq.heapify(self.queues[w])
        a.outcome = outcome
        self.live[a.task].discard(a.id)
        if outcome == "cancelled":
            self.n_cancelled += 1
            self._note(f"  cancelled redundant copy of T{a.task} on W{w}")

    def _on_crash(self, w: int, k: int) -> None:
        st = self.ws[w]
        _, recover = self.trace.crash_event(w, k)
        st.up, st.down_since = False, self.now
        self.n_crashes += 1
        self.estimator.observe_crash(w)
        self._push(recover, "recover", (w, k))
        self.down_intervals.append((w, self.now, recover))
        a = st.running
        if a is not None:
            a.end, a.outcome = self.now, "failed"
            spent = a.end - a.start
            self.busy += spent
            self.wasted += spent
            st.running = None
            self.estimator.observe_attempt(w, a.expected_duration, spent, False)
            tid = a.task
            self.live[tid].discard(a.id)
            self.failed_on[tid].add(w)
            self.n_task_failures[tid] += 1
            self._note(f"W{w} CRASHED (down until {recover:.1f}) - running T{tid} lost after {spent:.1f}")
            if not self.live[tid] and self.status[tid] != DONE:
                self._make_ready(tid)
        else:
            self._note(f"W{w} CRASHED while idle (down until {recover:.1f})")
        self._snapshot()

    def _on_recover(self, w: int, k: int) -> None:
        st = self.ws[w]
        st.up = True
        self.estimator.observe_recovery(w, self.now - st.down_since)
        st.crash_k = k + 1
        self._schedule_next_crash(w)
        self._note(f"W{w} recovered")

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
            self._assign(asg.task, asg.worker, asg.priority)

    def _revoke(self, tid: int, w: int) -> None:
        for aid in list(self.live[tid]):
            a = self.attempts[aid]
            if a.worker == w and a.start is None:
                self._cancel(a, "revoked")
                self.n_revoked += 1
                self._note(f"  reassign: T{tid} withdrawn from W{w}'s queue")
                if not self.live[tid] and self.status[tid] != DONE:
                    self._make_ready(tid)
                return

    def _assign(self, tid: int, w: int, priority: float | None) -> None:
        if self.status[tid] in (DONE, WAITING):
            raise ValueError(f"scheduler assigned T{tid} in state {self.status[tid]}")
        if any(self.attempts[aid].worker == w for aid in self.live[tid]):
            return  # duplicate copy on the same worker is pointless
        kind = "replica" if self.live[tid] else "primary"
        a = Attempt(id=next(self._attempt_ids), task=tid, worker=w, kind=kind, queued_at=self.now,
                    expected_duration=self.workflow.tasks[tid].work / self.workers[w].speed)
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
            end = self.trace.finish_time(w, self.now, a.expected_duration)
            self._push(end, "complete", a.id)
            self._note(f"T{a.task} started on W{w}" + (" (replica)" if a.kind == "replica" else ""))

    # ------------------------------------------------------------------ results
    def _result(self, completed: bool) -> RunResult:
        makespan = max(self.finish) if completed else math.inf
        deadline = self.workflow.deadline
        failed = sum(1 for a in self.attempts.values() if a.outcome == "failed")
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
            extra=extra,
        )


class SimView:
    """Read-only window onto the simulation for schedulers."""

    def __init__(self, sim: Simulation):
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
