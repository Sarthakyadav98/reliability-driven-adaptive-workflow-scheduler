"""Workflow (DAG) generators and graph helpers."""

from __future__ import annotations

import numpy as np

from .models import Workflow


def layered_dag(
    n_tasks: int,
    rng: np.random.Generator,
    max_width: int = 6,
    edge_prob: float = 0.3,
    skip_prob: float = 0.05,
    work_range: tuple[float, float] = (5.0, 25.0),
    name: str | None = None,
) -> Workflow:
    """Random layered DAG, the usual synthetic-workflow model in DAG-scheduling papers.

    Tasks are split into levels of 1..max_width tasks. Every task below the top
    level gets at least one parent in the level directly above, extra parents
    from that level with ``edge_prob``, and occasional skip edges from earlier
    levels with ``skip_prob``.
    """
    if n_tasks < 1:
        raise ValueError("n_tasks must be >= 1")
    levels: list[list[int]] = []
    next_id = 0
    while next_id < n_tasks:
        width = int(rng.integers(1, max_width + 1))
        width = min(width, n_tasks - next_id)
        levels.append(list(range(next_id, next_id + width)))
        next_id += width

    edges: set[tuple[int, int]] = set()
    for li in range(1, len(levels)):
        above = levels[li - 1]
        earlier = [t for lvl in levels[: li - 1] for t in lvl]
        for v in levels[li]:
            edges.add((int(rng.choice(above)), v))
            for u in above:
                if rng.random() < edge_prob:
                    edges.add((u, v))
            for u in earlier:
                if rng.random() < skip_prob:
                    edges.add((u, v))

    works = rng.uniform(*work_range, size=n_tasks).round(2).tolist()
    return Workflow.from_edges(works, sorted(edges), name=name or f"layered-{n_tasks}")


def fork_join(
    n_branches: int,
    n_stages: int,
    rng: np.random.Generator,
    work_range: tuple[float, float] = (5.0, 25.0),
) -> Workflow:
    """Repeated fork-join: source -> n_branches parallel tasks -> join, ``n_stages`` times."""
    edges: list[tuple[int, int]] = []
    works: list[float] = []
    prev_join = None
    tid = 0
    for _ in range(n_stages):
        fork = tid
        works.append(float(rng.uniform(*work_range)))
        if prev_join is not None:
            edges.append((prev_join, fork))
        tid += 1
        branch_ids = list(range(tid, tid + n_branches))
        for b in branch_ids:
            works.append(float(rng.uniform(*work_range)))
            edges.append((fork, b))
        tid += n_branches
        join = tid
        works.append(float(rng.uniform(*work_range)))
        for b in branch_ids:
            edges.append((b, join))
        tid += 1
        prev_join = join
    return Workflow.from_edges([round(w, 2) for w in works], edges, name=f"forkjoin-{n_branches}x{n_stages}")


def demo_workflow() -> Workflow:
    """Fixed 12-task workflow used in the review demo (deterministic, easy to draw)."""
    works = [10, 14, 8, 12, 16, 9, 11, 13, 7, 15, 10, 6]
    edges = [
        (0, 1), (0, 2), (0, 3),
        (1, 4), (1, 5), (2, 5), (2, 6), (3, 6), (3, 7),
        (4, 8), (5, 8), (5, 9), (6, 9), (7, 10),
        (8, 11), (9, 11), (10, 11),
    ]
    return Workflow.from_edges([float(w) for w in works], edges, name="demo-12")


def upward_rank(wf: Workflow, mean_cost: list[float]) -> list[float]:
    """HEFT upward rank (Topcuoglu et al. 2002, eq. 7) with zero communication cost.

    rank_u(i) = mean_cost(i) + max_{s in succ(i)} rank_u(s)
    """
    rank = [0.0] * len(wf)
    for tid in reversed(wf.topo_order):
        t = wf.tasks[tid]
        rank[tid] = mean_cost[tid] + max((rank[s] for s in t.succs), default=0.0)
    return rank


def downward_rank(wf: Workflow, mean_cost: list[float]) -> list[float]:
    """rank_d(i) = max_{p in pred(i)} (rank_d(p) + mean_cost(p)), i.e. longest path from an entry to i."""
    rank = [0.0] * len(wf)
    for tid in wf.topo_order:
        t = wf.tasks[tid]
        rank[tid] = max((rank[p] + mean_cost[p] for p in t.preds), default=0.0)
    return rank


def criticality(wf: Workflow, mean_cost: list[float]) -> list[float]:
    """(rank_u + rank_d) / critical-path length: 1.0 on the critical path, lower where there is slack."""
    ru = upward_rank(wf, mean_cost)
    rd = downward_rank(wf, mean_cost)
    cp = max(ru[t] for t in wf.entry_tasks)
    return [(ru[i] + rd[i]) / cp if cp > 0 else 1.0 for i in range(len(wf))]
