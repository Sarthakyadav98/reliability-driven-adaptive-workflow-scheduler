"""RQ2 benchmark: solution quality and solve time of the assignment backends vs problem size.

For decision-scoped problems of n ready tasks x m workers (n*m binary variables)
we compare the classical greedy, QUBO + simulated annealing (dwave-samplers)
and the exact optimum (brute force, where enumerable). Instances follow the
adaptive scheduler's cost structure: c_ij = w_i * E_ij and q_ikj = w_k * d_ij.

    python scripts/qubo_benchmark.py --out results/qubo
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from rdas.qubo import AssignmentProblem, QuboSolver, brute_force, greedy  # noqa: E402


def scheduler_like_problem(n: int, m: int, rng: np.random.Generator) -> AssignmentProblem:
    speed = rng.uniform(0.6, 1.6, size=m)
    p_rate = rng.lognormal(0, 1, size=m) * 0.01
    backlog = rng.exponential(5.0, size=m)
    work = rng.uniform(5, 25, size=n)
    weight = rng.uniform(0.3, 1.0, size=n) ** 2
    d = work[:, None] / speed[None, :]
    p = np.exp(-p_rate[None, :] * d)
    e = backlog[None, :] + d + (1 - p) * (0.5 * d + work.mean())
    c = weight[:, None] * e / 10.0
    q = np.zeros((n, n, m))
    for i in range(n):
        for k in range(i + 1, n):
            q[i, k] = weight[k] * d[i] / 10.0
    return AssignmentProblem(list(range(n)), list(range(m)), c, q)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/qubo")
    ap.add_argument("--instances", type=int, default=10)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    sizes = [(2, 3), (3, 3), (3, 4), (4, 4), (5, 4), (4, 6), (6, 4), (5, 6), (6, 6), (8, 6), (8, 8), (10, 8)]
    solvers = {"qubo-sa (20 reads)": QuboSolver(num_reads=20, num_sweeps=500, seed=0),
               "qubo-sa (100 reads)": QuboSolver(num_reads=100, num_sweeps=1000, seed=0)}
    rows = []
    for n, m in sizes:
        for inst in range(a.instances):
            rng = np.random.default_rng([n, m, inst])
            prob = scheduler_like_problem(n, m, rng)
            t0 = time.perf_counter()
            g = greedy(prob)
            t_g = time.perf_counter() - t0
            exact_e = None
            t_x = None
            if m ** n <= 60_000:
                t0 = time.perf_counter()
                exact_e = prob.energy(brute_force(prob, limit=60_000))
                t_x = time.perf_counter() - t0
            base = {"n_tasks": n, "n_workers": m, "n_vars": n * m, "instance": inst, "exact": exact_e}
            rows.append({**base, "method": "greedy", "energy": prob.energy(g), "seconds": t_g, "violations": 0})
            if exact_e is not None:
                rows.append({**base, "method": "exact (brute force)", "energy": exact_e, "seconds": t_x,
                             "violations": 0})
            for name, solver in solvers.items():
                assign, info = solver.solve(prob)
                rows.append({**base, "method": name, "energy": info.energy, "seconds": info.seconds,
                             "violations": info.violations})
        print(f"  {n}x{m} done", flush=True)

    df = pd.DataFrame(rows)
    # Reference = exact optimum where known, else the best energy any method found.
    best = df.groupby(["n_tasks", "n_workers", "instance"]).energy.transform("min")
    ref = df["exact"].fillna(best)
    df["gap_pct"] = 100 * (df["energy"] - ref) / ref
    df["optimal"] = df["gap_pct"] < 1e-6
    df.to_csv(out / "qubo_benchmark.csv", index=False)

    summary = df.groupby(["n_vars", "n_tasks", "n_workers", "method"]).agg(
        mean_gap_pct=("gap_pct", "mean"), optimal_rate=("optimal", "mean"),
        mean_ms=("seconds", lambda s: 1000 * s.mean()), violations=("violations", "sum")).reset_index()
    summary.to_csv(out / "qubo_summary.csv", index=False)
    with pd.option_context("display.width", 200, "display.precision", 3):
        print(summary.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for method, g in summary.groupby("method"):
        g = g.sort_values("n_vars")
        axes[0].plot(g.n_vars, g.mean_gap_pct, marker="o", label=method)
        axes[1].plot(g.n_vars, g.mean_ms, marker="o", label=method)
    axes[0].set_title("Mean gap to best known optimum (%)")
    axes[1].set_title("Mean solve time (ms)")
    axes[1].set_yscale("log")
    for ax in axes:
        ax.set_xlabel("Binary variables (tasks x workers)")
        ax.grid(alpha=0.3)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "qubo_benchmark.png", dpi=150)
    penalty_sweep(out, a.instances)
    print(f"written to {out}")


def penalty_sweep(out: Path, instances: int) -> None:
    """Penalty-weight sensitivity: A = scale x (safe bound), on problems where some tasks need
    2 copies. Reports the *raw* feasibility rate (before repair), the quality of the repaired
    solution vs the exact optimum, and build / anneal time separately."""
    rows = []
    for n, m in [(4, 4), (6, 4), (4, 6)]:  # small enough to brute-force with 2-copy tasks
        for inst in range(instances):
            rng = np.random.default_rng([n, m, inst, 7])
            prob = scheduler_like_problem(n, m, rng)
            prob.k = [2 if rng.random() < 0.4 else 1 for _ in range(n)]
            exact = prob.energy(brute_force(prob, limit=200_000))
            for scale in [0.1, 0.25, 0.5, 1.0, 2.0, 4.0]:
                solver = QuboSolver(num_reads=50, num_sweeps=1000, seed=inst, penalty_scale=scale)
                assign, info = solver.solve(prob)
                rows.append({"size": f"{n}x{m}", "penalty_scale": scale, "feasible": info.feasible,
                             "gap_pct": 100 * (info.energy - exact) / exact,
                             "build_ms": 1000 * info.build_seconds, "anneal_ms": 1000 * info.anneal_seconds})
    df = pd.DataFrame(rows)
    df.to_csv(out / "penalty_sweep.csv", index=False)
    summary = df.groupby(["size", "penalty_scale"]).agg(
        feasibility_rate=("feasible", "mean"), mean_gap_pct=("gap_pct", "mean"),
        build_ms=("build_ms", "mean"), anneal_ms=("anneal_ms", "mean")).reset_index()
    summary.to_csv(out / "penalty_sweep_summary.csv", index=False)
    print("\nPenalty sweep (A = scale x safe bound; feasibility = raw annealer output, before repair):")
    with pd.option_context("display.width", 200, "display.precision", 3):
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
