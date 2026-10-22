"""Draw the as-implemented architecture diagram used in the report.

    python scripts/make_report_figures.py   # -> docs/figures/architecture_implemented.png

Text sizes are chosen so the figure stays readable at 0.9 of the report text width.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

INK, MUTED, ACCENT, SOFT, OK, BAR = "#17202b", "#4a5563", "#2c55c9", "#e3eafb", "#2b8a5a", "#eef1f5"
TITLE, BODY, NOTE = 14.5, 12, 11.5
LINE = 0.40


def box(ax, x, y, w, h, title, lines, face="#ffffff", edge=INK, title_color=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.1", fc=face, ec=edge, lw=1.6))
    ax.text(x + w / 2, y + h - 0.22, title, ha="center", va="top", fontsize=TITLE, weight="bold", color=title_color)
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.78 - i * LINE, ln, ha="center", va="top", fontsize=BODY, color=MUTED)


def arrow(ax, a, b, color=INK, rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=16, lw=1.7, color=color,
                                 connectionstyle=f"arc3,rad={rad}"))


def label(ax, x, y, text, color=INK, ha="center"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=NOTE, color=color,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))


def main() -> None:
    W, H = 14.0, 8.2
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    # top row
    top_y, top_h = 4.55, 2.85
    box(ax, 0.1, top_y, 2.35, top_h, "Workflow", ["JSON (validated)", "or DAG generator", "work, type,", "deadline"])
    box(ax, 3.0, top_y, 2.9, top_h, "Engine + monitor", ["task states", "attempts, retries", "failure taxonomy",
                                                         "metrics, records"],
        face=SOFT, edge=ACCENT, title_color=ACCENT)
    ax.add_patch(FancyBboxPatch((6.55, 4.4), 7.35, 3.6, boxstyle="round,pad=0.02,rounding_size=0.12", fc="none",
                                ec=MUTED, lw=1.2, linestyle=(0, (4, 3))))
    ax.text(10.22, 7.78, "Scheduler: policy + placement + reassignment", ha="center", fontsize=NOTE,
            color=MUTED, style="italic")
    box(ax, 6.7, top_y, 2.15, 2.75, "Reliability", ["Gamma-Poisson", "(default)", "Beta-Bernoulli", "EMA"])
    box(ax, 9.1, top_y, 2.15, 2.75, "FT policy", ["execute / retry", "/ replicate", "+ reason", "4 policies"])
    box(ax, 11.5, top_y, 2.3, 2.75, "Placement", ["RR, least-loaded", "online EFT", "reliability HEFT", "QUBO + SA"])

    arrow(ax, (2.45, 5.95), (3.0, 5.95))
    arrow(ax, (8.85, 5.9), (9.1, 5.9))
    arrow(ax, (11.25, 5.9), (11.5, 5.9))
    arrow(ax, (5.9, 6.7), (6.7, 6.7))
    label(ax, 6.3, 7.0, "view")
    arrow(ax, (6.7, 5.0), (5.9, 5.0), color=ACCENT)
    label(ax, 6.3, 4.72, "decision", color=ACCENT)

    # backend interface bar
    ax.add_patch(FancyBboxPatch((3.0, 3.3), 10.9, 0.62, boxstyle="round,pad=0.02,rounding_size=0.08", fc=BAR,
                                ec=MUTED, lw=1.2))
    ax.text(8.45, 3.61, "ExecutionBackend: start · cancel · next_events (succeeded / failed / worker down / up)",
            ha="center", va="center", fontsize=BODY, color=INK)
    arrow(ax, (3.9, 4.55), (3.9, 3.92))
    label(ax, 3.55, 4.23, "start / cancel", ha="right")
    arrow(ax, (5.0, 3.92), (5.0, 4.55), color=OK)
    label(ax, 5.3, 4.23, "events", color=OK, ha="left")

    # backends
    box(ax, 3.0, 0.15, 4.7, 2.55, "Simulation backend", ["virtual time", "seeded fault trace:", "crashes, repair, slowdowns"])
    box(ax, 8.05, 0.15, 5.85, 2.55, "HTTP backend: real workers", ["agents as processes or Docker containers",
                                                                   "allowlisted tasks, injected failures",
                                                                   "health checks, crash or stop"])
    arrow(ax, (5.35, 3.3), (5.35, 2.7))
    arrow(ax, (11.0, 3.3), (11.0, 2.7))
    ax.text(0.15, 1.45, "Both backends share\nthe engine, estimator,\npolicies, placements\nand metrics.",
            fontsize=NOTE, color=MUTED, va="center", linespacing=1.4)

    out = Path("docs/figures/architecture_implemented.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.3)
    fig.savefig(out, dpi=200)
    print(out)


if __name__ == "__main__":
    main()
