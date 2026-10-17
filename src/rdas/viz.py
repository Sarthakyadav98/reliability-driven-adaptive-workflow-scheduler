"""Plots: sweep comparison charts, Gantt charts and reliability timelines."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# Fixed scheduler -> colour mapping so every chart in the report is consistent.
COLORS = {
    "round-robin": "#9aa0a6",
    "least-loaded": "#7b8794",
    "heft": "#e8710a",
    "eft-retry": "#f9ab00",
    "replicate-2": "#a142f4",
    "adaptive": "#1a73e8",
    "adaptive-qubo": "#12b5cb",
    "adaptive-no-replication": "#5e97f6",
    "adaptive-no-reassign": "#8ab4f8",
}


def _color(name: str) -> str:
    return COLORS.get(name, "#444444")


def plot_sweep(summary: pd.DataFrame, out: Path) -> None:
    out = Path(out)
    panels = [
        ("makespan", "Mean makespan", "makespan_ci"),
        ("deadline_met", "Deadline met (fraction of runs)", None),
        ("recovery_overhead", "Recovery overhead (makespan - fault-free)", None),
        ("resource_overhead", "Resource overhead (wasted / useful work)", None),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    for ax, (col, title, err) in zip(axes.flat, panels):
        for name, g in summary.groupby("scheduler", sort=False):
            g = g.sort_values("failure_prob")
            ax.plot(g.failure_prob, g[col], marker="o", ms=4, lw=1.8, label=name, color=_color(name))
            if err:
                ax.fill_between(g.failure_prob, g[col] - g[err], g[col] + g[err], alpha=0.12,
                                color=_color(name))
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Task failure probability")
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "sweep.png", dpi=150)
    plt.close(fig)


def plot_gantt(sim, path: Path, title: str = "") -> None:
    """Gantt chart of every attempt: successes, failures (crash) and cancelled replicas."""
    fig, ax = plt.subplots(figsize=(12, 0.6 * len(sim.workers) + 1.6))
    styles = {"success": ("#1a73e8", None), "failed": ("#d93025", "//"), "cancelled": ("#bdc1c6", ".."),
              "revoked": ("#bdc1c6", None)}
    for w, start, end in sim.down_intervals:
        end = min(end, max(sim.finish) if sim.finish else end)
        ax.barh(w, end - start, left=start, height=0.8, color="#000000", alpha=0.08)
    for a in sim.attempts.values():
        if a.start is None or a.end is None:
            continue
        color, hatch = styles.get(a.outcome, ("#444", None))
        ax.barh(a.worker, a.end - a.start, left=a.start, height=0.6, color=color, hatch=hatch,
                edgecolor="white", linewidth=0.8)
        if a.end - a.start > 2.5:
            ax.text(a.start + (a.end - a.start) / 2, a.worker, f"T{a.task}", ha="center", va="center",
                    fontsize=7, color="white" if a.outcome == "success" else "#202124")
    if sim.workflow.deadline:
        ax.axvline(sim.workflow.deadline, color="#d93025", ls="--", lw=1)
        ax.text(sim.workflow.deadline, len(sim.workers) - 0.4, " deadline", color="#d93025", fontsize=8)
    ax.set_yticks(range(len(sim.workers)))
    ax.set_yticklabels([f"W{w.id} {w.label}".strip() for w in sim.workers])
    ax.invert_yaxis()
    ax.set_xlabel("Simulated time")
    ax.set_title(title or sim.scheduler.name)
    from matplotlib.patches import Patch

    ax.legend(handles=[Patch(color="#1a73e8", label="succeeded"),
                       Patch(facecolor="#d93025", hatch="//", label="lost to crash"),
                       Patch(facecolor="#bdc1c6", hatch="..", label="cancelled replica"),
                       Patch(color="#000000", alpha=0.08, label="worker down")],
              loc="lower right", fontsize=8, frameon=False, ncol=4)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_reliability(sim, path: Path, title: str = "Estimated worker reliability") -> None:
    """R_j over time as the monitor observes crashes and successes."""
    if not sim.reliability_timeline:
        return
    fig, ax = plt.subplots(figsize=(10, 3.6))
    for w in range(len(sim.workers)):
        ts = [t for t, snap in sim.reliability_timeline]
        rs = [snap[w]["R"] for t, snap in sim.reliability_timeline]
        label = f"W{w} {sim.workers[w].label}".strip()
        ax.step(ts, rs, where="post", label=label, lw=1.8)
    for w, start, _ in sim.down_intervals:
        ax.axvline(start, color="#d93025", alpha=0.25, lw=1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Simulated time")
    ax.set_ylabel("R (P(success), ref. task)")
    ax.set_title(title)
    ax.legend(fontsize=8, frameon=False, ncol=4, loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_dag(workflow, path: Path) -> None:
    import networkx as nx

    g = nx.DiGraph()
    for t in workflow.tasks:
        g.add_node(t.id, work=t.work)
        for s in t.succs:
            g.add_edge(t.id, s)
    for t in workflow.tasks:  # layer = longest path from an entry task
        g.nodes[t.id]["layer"] = 0
    for tid in workflow.topo_order:
        for s in workflow.tasks[tid].succs:
            g.nodes[s]["layer"] = max(g.nodes[s]["layer"], g.nodes[tid]["layer"] + 1)
    pos = nx.multipartite_layout(g, subset_key="layer")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    nx.draw_networkx(g, pos, ax=ax, node_color="#e8f0fe", edgecolors="#1a73e8", node_size=720,
                     labels={t.id: f"T{t.id}\n{t.work:g}" for t in workflow.tasks}, font_size=7,
                     arrowsize=12)
    ax.set_title(f"{workflow.name}: {len(workflow)} tasks (label: id / work)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
