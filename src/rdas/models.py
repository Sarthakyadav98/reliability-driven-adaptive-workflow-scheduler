"""Core data models: tasks, workflows (DAGs) and worker specifications."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Task:
    """A unit of computation. Its duration on a worker is ``work / worker.speed``."""

    id: int
    work: float
    preds: tuple[int, ...] = ()
    succs: tuple[int, ...] = ()


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
    def from_edges(works: list[float], edges: list[tuple[int, int]], name: str = "workflow") -> "Workflow":
        """Build a workflow from per-task work amounts and ``(pred, succ)`` edges."""
        n = len(works)
        preds: list[list[int]] = [[] for _ in range(n)]
        succs: list[list[int]] = [[] for _ in range(n)]
        for u, v in sorted(set(edges)):
            if not (0 <= u < n and 0 <= v < n) or u == v:
                raise ValueError(f"invalid edge {(u, v)}")
            succs[u].append(v)
            preds[v].append(u)
        tasks = [Task(i, float(works[i]), tuple(preds[i]), tuple(succs[i])) for i in range(n)]
        return Workflow(tasks, name=name)


@dataclass(frozen=True)
class WorkerSpec:
    """Static description of a worker as the scheduler sees it.

    ``speed`` is the nominal speed that every scheduler knows. ``failure_factor``
    scales the fleet-wide crash rate for this worker; it is ground truth used by
    the fault injector only and is never shown to schedulers.
    """

    id: int
    speed: float
    failure_factor: float = 1.0
    label: str = ""


@dataclass
class Attempt:
    """One execution (or replica) of a task on a worker."""

    id: int
    task: int
    worker: int
    kind: str  # "primary" | "replica"
    queued_at: float
    start: float | None = None
    end: float | None = None
    outcome: str | None = None  # "success" | "failed" | "cancelled" | "revoked"
    expected_duration: float = 0.0


@dataclass
class RunResult:
    """Metrics for one simulation run."""

    scheduler: str
    workflow: str
    n_tasks: int
    n_workers: int
    completed: bool
    makespan: float
    deadline: float | None
    deadline_met: bool
    n_attempts: int
    n_failed_attempts: int
    n_replicas: int
    n_cancelled: int
    n_revoked: int
    n_crashes: int
    busy_time: float
    useful_time: float
    wasted_time: float
    decision_time_total: float
    n_decisions: int
    extra: dict = field(default_factory=dict)

    @property
    def resource_overhead(self) -> float:
        return self.wasted_time / self.useful_time if self.useful_time > 0 else 0.0

    @property
    def mean_decision_ms(self) -> float:
        return 1000.0 * self.decision_time_total / self.n_decisions if self.n_decisions else 0.0

    def as_row(self) -> dict:
        row = {k: v for k, v in self.__dict__.items() if k != "extra"}
        row["resource_overhead"] = self.resource_overhead
        row["mean_decision_ms"] = self.mean_decision_ms
        row.update(self.extra)
        return row
