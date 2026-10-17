"""Experiment harness: same workflows, worker pools, fault traces and seeds for every scheduler.

Usage:
    python -m rdas.experiments --fail 0,0.1,0.2,0.3 --seeds 20 --out results/sweep
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .engine import Simulation, default_estimator, default_ref_duration
from .faults import Degradation, FaultConfig
from .schedulers import BASELINES, make_scheduler
from .schedulers.heft import heft_plan
from .workers import make_pool
from .workflow import fork_join, layered_dag

DEFAULT_SCHEDULERS = BASELINES + ["adaptive", "adaptive-qubo"]


@dataclass
class ExperimentConfig:
    schedulers: list[str] = field(default_factory=lambda: list(DEFAULT_SCHEDULERS))
    failure_probs: list[float] = field(default_factory=lambda: [0.0, 0.05, 0.1, 0.2, 0.3])
    seeds: list[int] = field(default_factory=lambda: list(range(10)))
    n_tasks: int = 60
    n_workers: int = 6
    workflow: str = "layered"  # layered | forkjoin
    deadline_factor: float = 1.5  # deadline = factor x fault-free HEFT plan makespan
    mttr_factor: float = 1.0  # mean repair time, in units of the mean task duration
    slowdown_rate: float = 0.0  # slowdown episodes per mean-task-duration per worker
    fast_is_flaky: float = 0.7
    failure_spread: float = 1.0
    degrade_fraction: float = 0.0  # fraction of workers whose crash rate jumps x8 mid-run
    # Workflows executed on the same pool before the measured one, sharing the
    # reliability estimator (the monitor keeps running between workflows).
    # 0 = cold start: no history at all.
    warmup_workflows: int = 0
    scheduler_options: dict = field(default_factory=dict)


def build_instance(cfg: ExperimentConfig, seed: int):
    rng = np.random.default_rng([seed, 1])
    if cfg.workflow == "forkjoin":
        stages = max(1, cfg.n_tasks // (cfg.n_workers + 2))
        wf = fork_join(cfg.n_workers, stages, rng)
    else:
        wf = layered_dag(cfg.n_tasks, rng)
    pool = make_pool(cfg.n_workers, np.random.default_rng([seed, 2]),
                     failure_spread=cfg.failure_spread, fast_is_flaky=cfg.fast_is_flaky)
    plan = heft_plan(wf, pool)
    wf.deadline = cfg.deadline_factor * plan.makespan
    return wf, pool, plan


def fault_config(cfg: ExperimentConfig, p: float, ref: float, plan_makespan: float, seed: int,
                 n_workers: int) -> FaultConfig:
    degr = []
    if cfg.degrade_fraction > 0 and p > 0:
        rng = np.random.default_rng([seed, 3])
        k = max(1, round(cfg.degrade_fraction * n_workers))
        for w in rng.choice(n_workers, size=k, replace=False):
            degr.append(Degradation(int(w), at=float(rng.uniform(0.2, 0.5)) * plan_makespan, factor=8.0))
    return FaultConfig(
        task_failure_prob=p,
        mttr=cfg.mttr_factor * ref,
        slowdown_rate=cfg.slowdown_rate / ref if p > 0 else 0.0,
        slowdown_duration=3 * ref,
        degradations=degr,
    )


def _make(cfg: ExperimentConfig, name: str, seed: int):
    if name.startswith("adaptive"):
        return make_scheduler(name, seed=seed, **cfg.scheduler_options)
    return make_scheduler(name)


def warm_estimator(cfg: ExperimentConfig, name: str, wf, pool, p: float, seed: int, ref: float):
    """Run ``cfg.warmup_workflows`` other workflows on the same pool and return the shared estimator."""
    est = default_estimator(pool, ref)
    if not name.startswith("adaptive"):
        return est  # baselines never read the estimator
    for k in range(cfg.warmup_workflows):
        wseed = 10_000 + 97 * seed + k
        warm_wf = layered_dag(len(wf), np.random.default_rng([wseed, 1]))
        warm_plan = heft_plan(warm_wf, pool)
        faults = fault_config(cfg, p, ref, warm_plan.makespan, wseed, len(pool))
        Simulation(warm_wf, pool, _make(cfg, name, wseed), faults, seed=wseed, estimator=est).run()
    return est


def run_grid(cfg: ExperimentConfig, progress: bool = True) -> pd.DataFrame:
    rows = []
    probs = sorted(set(cfg.failure_probs) | {0.0})  # fault-free runs give recovery overhead
    t_start = time.perf_counter()
    for seed in cfg.seeds:
        wf, pool, plan = build_instance(cfg, seed)
        ref = default_ref_duration(wf, pool)
        for p in probs:
            faults = fault_config(cfg, p, ref, plan.makespan, seed, len(pool))
            for name in cfg.schedulers:
                est = warm_estimator(cfg, name, wf, pool, p, seed, ref)
                res = Simulation(wf, pool, _make(cfg, name, seed), faults, seed=seed, estimator=est).run()
                row = res.as_row()
                row.update(seed=seed, failure_prob=p, heft_plan_makespan=plan.makespan)
                rows.append(row)
        if progress:
            print(f"  seed {seed} done ({time.perf_counter() - t_start:.1f}s)", flush=True)
    df = pd.DataFrame(rows)
    base = df[df.failure_prob == 0.0].set_index(["scheduler", "seed"])["makespan"]
    df["fault_free_makespan"] = [base.get((s, sd), np.nan) for s, sd in zip(df.scheduler, df.seed)]
    df["recovery_overhead"] = df["makespan"] - df["fault_free_makespan"]
    df["normalized_makespan"] = df["makespan"] / df["heft_plan_makespan"]
    if set(cfg.failure_probs) != set(probs):
        df = df[df.failure_prob.isin(cfg.failure_probs)]
    return df.reset_index(drop=True)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    def ci95(x):
        x = x.dropna()
        return 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0

    g = df.groupby(["failure_prob", "scheduler"])
    out = g.agg(
        runs=("makespan", "size"),
        completed=("completed", "mean"),
        makespan=("makespan", "mean"),
        makespan_ci=("makespan", ci95),
        norm_makespan=("normalized_makespan", "mean"),
        deadline_met=("deadline_met", "mean"),
        recovery_overhead=("recovery_overhead", "mean"),
        resource_overhead=("resource_overhead", "mean"),
        failed_attempts=("n_failed_attempts", "mean"),
        replicas=("n_replicas", "mean"),
        reassignments=("n_revoked", "mean"),
        decision_ms=("mean_decision_ms", "mean"),
    )
    return out.reset_index()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Run a scheduler comparison sweep.")
    ap.add_argument("--schedulers", default=",".join(DEFAULT_SCHEDULERS))
    ap.add_argument("--fail", default="0,0.05,0.1,0.2,0.3", help="task failure probabilities")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--tasks", type=int, default=60)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--workflow", default="layered", choices=["layered", "forkjoin"])
    ap.add_argument("--deadline-factor", type=float, default=1.5)
    ap.add_argument("--mttr-factor", type=float, default=1.0)
    ap.add_argument("--slowdown-rate", type=float, default=0.0)
    ap.add_argument("--degrade-fraction", type=float, default=0.0)
    ap.add_argument("--warmup", type=int, default=0, help="warm-up workflows sharing the estimator")
    ap.add_argument("--out", default="results/sweep")
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args(argv)

    cfg = ExperimentConfig(
        schedulers=[s.strip() for s in a.schedulers.split(",") if s.strip()],
        failure_probs=[float(x) for x in a.fail.split(",")],
        seeds=list(range(a.seed_offset, a.seed_offset + a.seeds)),
        n_tasks=a.tasks, n_workers=a.workers, workflow=a.workflow,
        deadline_factor=a.deadline_factor, mttr_factor=a.mttr_factor,
        slowdown_rate=a.slowdown_rate, degrade_fraction=a.degrade_fraction,
        warmup_workflows=a.warmup,
    )
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Running {len(cfg.schedulers)} schedulers x {len(cfg.failure_probs)} failure rates x "
          f"{len(cfg.seeds)} seeds -> {out}")
    df = run_grid(cfg)
    summary = summarize(df)
    df.to_csv(out / "runs.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    with pd.option_context("display.width", 200, "display.max_columns", 30, "display.precision", 3):
        print(summary.to_string(index=False))
    if not a.no_plots:
        from .viz import plot_sweep

        plot_sweep(summary, out)
        print(f"plots written to {out}")


if __name__ == "__main__":
    main()
