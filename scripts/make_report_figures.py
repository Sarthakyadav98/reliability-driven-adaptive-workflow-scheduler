"""Draw the as-implemented architecture diagram used in the report.

    python scripts/make_report_figures.py   # -> docs/figures/architecture_implemented.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

INK, MUTED, ACCENT, SOFT, OK, BAR = "#17202b", "#5a6675", "#2c55c9", "#e3eafb", "#2b8a5a", "#eef1f5"


def box(ax, x, y, w, h, title, lines, face="#ffffff", edge=INK, title_color=INK, size=8.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.1", fc=face, ec=edge, lw=1.4))
    ax.text(x + w / 2, y + h - 0.22, title, ha="center", va="top", fontsize=10.2, weight="bold", color=title_color)
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.62 - i * 0.29, ln, ha="center", va="top", fontsize=size, color=MUTED)


def arrow(ax, a, b, color=INK, rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12, lw=1.4, color=color,
                                 connectionstyle=f"arc3,rad={rad}"))


def label(ax, x, y, text, color=INK, ha="center"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=8.2, color=color,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))


def main() -> None:
    fig, ax = plt.subplots(figsize=(13, 6.4))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6.4)
    ax.axis("off")

    # top row
    box(ax, 0.15, 3.85, 2.2, 2.3, "Workflow", ["JSON (validated)", "or DAG generator", "work, type, payload,", "deadline"])
    box(ax, 2.95, 3.85, 2.75, 2.3, "Engine + monitor", ["logical task states", "attempts, bounded retries", "failure taxonomy",
                                                        "idempotent events", "metrics, event records"],
        face=SOFT, edge=ACCENT, title_color=ACCENT)
    ax.add_patch(FancyBboxPatch((6.45, 3.7), 6.4, 2.6, boxstyle="round,pad=0.02,rounding_size=0.12", fc="none",
                                ec=MUTED, lw=1.0, linestyle=(0, (4, 3))))
    ax.text(9.65, 6.18, "Scheduler = fault-tolerance policy + placement (+ reassignment), one shared cost model",
            ha="center", fontsize=8.4, color=MUTED, style="italic")
    box(ax, 6.6, 3.85, 1.9, 2.05, "Reliability", ["Gamma-Poisson", "(default)", "Beta-Bernoulli", "EMA"])
    box(ax, 8.75, 3.85, 1.95, 2.05, "FT policy", ["execute / retry /", "replicate: k_i", "+ reason", "4 policies"])
    box(ax, 10.95, 3.85, 1.75, 2.05, "Placement", ["RR, least-loaded", "online EFT", "reliability HEFT", "QUBO + SA, exact"])

    arrow(ax, (2.35, 5.0), (2.95, 5.0))
    arrow(ax, (8.5, 4.87), (8.75, 4.87))
    arrow(ax, (10.7, 4.87), (10.95, 4.87))
    arrow(ax, (5.7, 5.55), (6.6, 5.55))
    label(ax, 6.12, 5.78, "view")
    arrow(ax, (6.45, 4.2), (5.7, 4.2), color=ACCENT)
    label(ax, 6.07, 3.98, "decision", color=ACCENT)

    # backend interface bar
    ax.add_patch(FancyBboxPatch((2.95, 2.75), 9.9, 0.5, boxstyle="round,pad=0.02,rounding_size=0.08", fc=BAR, ec=MUTED, lw=1.0))
    ax.text(7.9, 3.0, "ExecutionBackend interface:  start  ·  cancel  ·  next_events  "
                      "(succeeded / failed + category / worker_unavailable / worker_available)",
            ha="center", va="center", fontsize=8.5, color=INK)
    arrow(ax, (3.9, 3.85), (3.9, 3.25))
    label(ax, 3.55, 3.55, "start / cancel", ha="right")
    arrow(ax, (4.9, 3.25), (4.9, 3.85), color=OK)
    label(ax, 5.25, 3.55, "events", color=OK, ha="left")

    # backends
    box(ax, 2.95, 0.25, 4.3, 2.05, "Simulation backend", ["virtual time", "seed-deterministic fault trace:",
                                                          "crashes, repair, slowdowns, degradation"])
    box(ax, 7.6, 0.25, 5.25, 2.05, "HTTP backend: real workers", ["worker agents as processes or Docker containers",
                                                                  "allowlisted tasks, injected failures, timeouts",
                                                                  "health checks; admin crash or container stop"])
    arrow(ax, (5.1, 2.75), (5.1, 2.3))
    arrow(ax, (10.2, 2.75), (10.2, 2.3))
    ax.text(0.2, 1.2, "Both backends share the\nengine, estimator, policies,\nplacements and metrics:\nonly execution differs.",
            fontsize=8.6, color=MUTED, va="center")

    out = Path("docs/figures/architecture_implemented.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print(out)


if __name__ == "__main__":
    main()
