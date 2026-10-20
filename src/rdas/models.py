"""Core data models: tasks, workflows (DAGs), workers, attempts and run results."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Task:
    """A unit of computation. Its duration on a worker is ``work / worker.speed``.

    ``name``, ``type`` and ``payload`` matter for real execution: ``type`` must be a
    registered task type of the worker agents and ``payload`` is passed to it.
    """

    id: int
    work: float
    preds: tuple[int, ...] = ()
    succs: tuple[int, ...] = ()
    name: str = ""
    type: str = "sleep"
    payload: dict = field(default_factory=dict, compare=False, hash=False)

    @property
    def label(self) -> str:
        return self.name or f"T{self.id}"


@dataclass
class Workflow:
    """A DAG of tasks. Task ids are ``0..n-1`` and index into ``tasks``."""

    tasks: list[Task]
    name: str = "workflow"
    deadline: float | None = None

    def __post_init__(self) -> None:
        for i, t in enumerate(self.tasks):
            if t.id != i:
                raise ValueError(f"task at index {i} has id {t.id}; ids must be 0..n-1")
            if not t.work > 0:
                raise ValueError(f"task {t.label} must have positive work")
        self._topo = self._compute_topo_order()

    def __len__(self) -> int:
        return len(self.tasks)

    @property
    def topo_order(self) -> list[int]:
        return self._topo

    @property
    def entry_tasks(self) -> list[int]:
        return [t.id for t in self.tasks if not t.preds]

    @property
    def exit_tasks(self) -> list[int]:
        return [t.id for t in self.tasks if not t.succs]

    @property
    def total_work(self) -> float:
        return sum(t.work for t in self.tasks)

    def _compute_topo_order(self) -> list[int]:
        indeg = [len(t.preds) for t in self.tasks]
        stack = [t.id for t in self.tasks if indeg[t.id] == 0]
        order: list[int] = []
        while stack:
            tid = stack.pop()
            order.append(tid)
            for s in self.tasks[tid].succs:
                indeg[s] -= 1
                if indeg[s] == 0:
                    stack.append(s)
        if len(order) != len(self.tasks):
            raise ValueError("workflow graph contains a cycle")
        return order

    @staticmethod
    def from_edges(works: list[float], edges: list[tuple[int, int]], name: str = "workflow",
                   names: list[str] | None = None, types: list[str] | None = None,
                   payloads: list[dict] | None = None) -> "Workflow":
        """Build a workflow from per-task work amounts and ``(pred, succ)`` edges."""
        n = len(works)
        preds: list[list[int]] = [[] for _ in range(n)]
        succs: list[list[int]] = [[] for _ in range(n)]
        for u, v in sorted(set(edges)):
            if not (0 <= u < n and 0 <= v < n) or u == v:
                raise ValueError(f"invalid edge {(u, v)}")
            succs[u].append(v)
            preds[v].append(u)
        tasks = [Task(i, float(works[i]), tuple(preds[i]), tuple(succs[i]),
                      name=names[i] if names else "",
                      type=types[i] if types else "sleep",
                      payload=dict(payloads[i]) if payloads else {})
                 for i in range(n)]
        return Workflow(tasks, name=name)


@dataclass(frozen=True)
class WorkerSpec:
    """Static description of a worker as the scheduler sees it.

    ``speed`` is the nominal speed that every scheduler knows. ``failure_factor``
    scales the fleet-wide crash rate for this worker; it is ground truth used by
    the simulated fault injector only and is never shown to schedulers.
    """

    id: int
    speed: float
    failure_factor: float = 1.0
    label: str = ""


# Failure taxonomy. Only worker-related categories update worker reliability:
# a bug in the task itself says nothing about the worker that ran it.
WORKER_UNAVAILABLE = "worker_unavailable"  # crash / container stopped / agent unreachable
INJECTED = "injected"  # configured (demo / experiment) failure on a live worker
TIMEOUT = "timeout"  # attempt exceeded its time limit
DISPATCH_ERROR = "dispatch_error"  # network / HTTP error talking to the worker
TASK_ERROR = "task_error"  # the task itself raised (bad input, bug)
WORKER_RELATED = frozenset({WORKER_UNAVAILABLE, INJECTED, TIMEOUT, DISPATCH_ERROR})
FAILURE_CATEGORIES = (WORKER_UNAVAILABLE, INJECTED, TIMEOUT, DISPATCH_ERROR, TASK_ERROR)


@dataclass
class Attempt:
    """One execution (or replica) of a logical task on a worker.

    Logical task state lives in the engine. An attempt only records what happened
    to this copy. Every attempt has a unique id, so late or duplicate events for an
    attempt that is already finished are ignored.
    """

    id: int
    task: int
    worker: int
    kind: str  # "primary" | "replica"
    queued_at: float
    start: float | None = None
    end: float | None = None
    outcome: str | None = None  # "success" | "failed" | "cancelled" | "revoked"
    expected_duration: float = 0.0
    category: str | None = None  # failure category when outcome == "failed"
    est_p_success: float | None = None  # reliability estimate in force at placement
    reason: str = ""  # fault-tolerance decision that created this attempt

    def as_record(self, workflow: Workflow) -> dict:
        t = workflow.tasks[self.task]
        return {
            "attempt": self.id, "task": t.label, "task_id": self.task, "worker": self.worker,
            "kind": self.kind, "queued_at": self.queued_at, "start": self.start, "end": self.end,
            "outcome": self.outcome, "category": self.category,
            "expected_duration": self.expected_duration,
            "actual_duration": (self.end - self.start) if self.start is not None and self.end is not None else None,
            "est_p_success": self.est_p_success, "reason": self.reason,
        }


@dataclass
class RunResult:
    """Metrics for one run (simulated or real execution)."""

    scheduler: str
    workflow: str
    n_tasks: int
    n_workers: int
    completed: bool  # workflow success: every logical task succeeded
    makespan: float  # inf when the workflow did not complete
    deadline: float | None
    deadline_met: bool
    n_attempts: int
    n_failed_attempts: int
    n_replicas: int  # replica attempts queued
    n_cancelled: int
    n_revoked: int
    n_crashes: int
    busy_time: float
    useful_time: float
    wasted_time: float
    decision_time_total: float
    n_decisions: int
    end_time: float = 0.0  # when the run stopped (completion or terminal failure)
    tasks_succeeded: int = 0
    n_retries: int = 0  # times a logical task was re-queued after all its copies failed
    replicas_started: int = 0  # replica attempts that actually began executing
    recovery_time_mean: float = 0.0  # first loss of a task -> its successful completion
    failures_by_category: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)

    @property
    def resource_overhead(self) -> float:
        return self.wasted_time / self.useful_time if self.useful_time > 0 else 0.0

    @property
    def mean_decision_ms(self) -> float:
        return 1000.0 * self.decision_time_total / self.n_decisions if self.n_decisions else 0.0

    @property
    def task_success_rate(self) -> float:
        return self.tasks_succeeded / self.n_tasks if self.n_tasks else 1.0

    @property
    def utilization(self) -> float:
        """Busy time / available worker time over [0, end_time] (includes wasted work)."""
        span = self.end_time * self.n_workers
        return self.busy_time / span if span > 0 else 0.0

    def as_row(self) -> dict:
        row = {k: v for k, v in self.__dict__.items() if k not in ("extra", "failures_by_category")}
        row["resource_overhead"] = self.resource_overhead
        row["mean_decision_ms"] = self.mean_decision_ms
        row["task_success_rate"] = self.task_success_rate
        row["utilization"] = self.utilization
        for cat, n in self.failures_by_category.items():
            row[f"fail_{cat}"] = n
        row.update(self.extra)
        return row
