"""Review 2 live demo: one coherent story, about 5 minutes.

    python scripts/demo.py            # full demo, figures in results/demo/
    python scripts/demo.py --quick    # skip the pauses

1. Generate a workflow (12-task DAG) and a heterogeneous worker pool.
2. Fault-free baseline (HEFT) to show the normal schedule.
3. Inject crashes on W1, the fast-but-flaky worker.
4. Show how each scheduler recovers (HEFT waits, EFT retries, replicate-2 over-provisions).
5. Run the adaptive scheduler: reliability estimates fall after crashes and the
   placement / replication decisions change, with explanations printed.
6. Compare metrics side by side.
7. Show one real decision from that run as a QUBO, solved by greedy, SA and exact.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from rdas import FaultConfig, ScriptedCrash, Simulation
from rdas.engine import default_estimator, default_ref_duration
from rdas.qubo import QuboSolver, brute_force, greedy, to_bqm
from rdas.schedulers import AdaptiveScheduler, make_scheduler
from rdas.schedulers.heft import heft_plan
from rdas.viz import plot_dag, plot_gantt, plot_reliability
from rdas.workers import demo_pool
from rdas.workflow import demo_workflow, layered_dag

OUT = Path("results/demo")


def banner(title: str, quick: bool) -> None:
    print("\n" + "=" * 78 + f"\n  {title}\n" + "=" * 78)
    if not quick:
        input("  [press Enter] ")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    wf = demo_workflow()
    pool = demo_pool()
    plan = heft_plan(wf, pool)
    wf.deadline = round(1.4 * plan.makespan, 1)

    # 1 ---------------------------------------------------------------------
    banner("1. Workflow and worker pool", a.quick)
    print(f"Workflow '{wf.name}': {len(wf)} tasks, {sum(len(t.succs) for t in wf.tasks)} dependencies, "
          f"deadline {wf.deadline}")
    for t in wf.tasks:
        print(f"  T{t.id:<2} work={t.work:>5.1f}  depends on {list(t.preds) or '-'}")
    print("\nWorkers (speed is known to every scheduler; reliability is hidden ground truth):")
    for w in pool:
        print(f"  W{w.id} {w.label:<14} speed={w.speed:<4} hidden crash-rate factor={w.failure_factor}")
    plot_dag(wf, OUT / "01_dag.png")

    # 2 ---------------------------------------------------------------------
    banner("2. Fault-free schedule (HEFT)", a.quick)
    sim = Simulation(wf, pool, make_scheduler("heft"), FaultConfig.fault_free(), seed=0, record=True)
    res = sim.run()
    print(f"HEFT plan makespan {plan.makespan:.1f}; simulated {res.makespan:.1f}. HEFT favours fast W1:")
    print("  tasks planned on W1:", [t for t in range(len(wf)) if plan.proc[t] == 1])
    plot_gantt(sim, OUT / "02_heft_fault_free.png", "HEFT, no faults")

    # 3 ---------------------------------------------------------------------
    crashes = [ScriptedCrash(1, at=8.0 + 20.0 * k, downtime=10.0) for k in range(10)]
    faults = FaultConfig(task_failure_prob=0.0, scripted_crashes=crashes)
    banner("3. Inject faults: W1 (fast-flaky) crashes every 20 time units from t=8 (10 units down)", a.quick)
    print("Same fault trace for every scheduler (common random numbers -> fair comparison).")
    print("This is an illustrative scenario; the statistical comparison is the multi-seed sweep.")

    # 4 ---------------------------------------------------------------------
    banner("4. How the baselines cope", a.quick)
    results = {}
    for name in ["heft", "eft-retry", "replicate-2"]:
        s = Simulation(wf, pool, make_scheduler(name), faults, seed=0, record=True)
        r = s.run()
        results[name] = r
        print(f"\n--- {name}: makespan {r.makespan:.1f} ({'met' if r.deadline_met else 'MISSED'} deadline), "
              f"lost attempts {r.n_failed_attempts}, replicas {r.n_replicas}")
        for t, msg in s.log:
            if "CRASH" in msg or "lost" in msg:
                print(f"   t={t:6.1f}  {msg}")
        plot_gantt(s, OUT / f"04_{name}.png", f"{name} with W1 crashes")

    # 5 ---------------------------------------------------------------------
    banner("5. Adaptive scheduler: monitor -> reliability estimate -> decision", a.quick)
    captured = []
    sched = AdaptiveScheduler(explain=True, seed=0)
    original = sched._problem

    def capture(view, batch, workers, avail):
        prob = original(view, batch, workers, avail)
        if len(batch) >= 3:
            captured.append((view.now, prob))
        return prob

    sched._problem = capture
    s = Simulation(wf, pool, sched, faults, seed=0, record=True)
    r = s.run()
    results["adaptive (cold)"] = r
    for t, msg in s.log:
        if any(k in msg for k in ("->", "CRASH", "REPLICATE", "reassign", "withdrawn")):
            print(f"   t={t:6.1f}  {msg}")
    print("\nReliability estimate R (P(success) of an average task) after the run:")
    for row in s.estimator.snapshot(s.ref_duration):
        print(f"  W{row['worker']}: R={row['R']:.2f}  crashes seen={row['crashes']}")
    plot_gantt(s, OUT / "05_adaptive_cold.png", "adaptive (cold start) with W1 crashes")
    plot_reliability(s, OUT / "05_reliability_cold.png", "Estimated reliability, cold start")
    print("\nCold start: a 12-task workflow ends before much evidence accumulates, so the")
    print("scheduler mostly behaves like EFT. In operation the monitor keeps running between")
    print("workflows, so next we warm it with one earlier workflow on the same pool.")

    banner("5b. Adaptive scheduler with history (monitor observed one earlier workflow)", a.quick)
    ref = default_ref_duration(wf, pool)
    est = default_estimator(pool, ref)
    history = layered_dag(30, np.random.default_rng(42), name="history-30")
    hist_crashes = [ScriptedCrash(1, at=8.0 + 20.0 * k, downtime=10.0) for k in range(25)]
    Simulation(history, pool, make_scheduler("adaptive"),
               FaultConfig(task_failure_prob=0.0, scripted_crashes=hist_crashes),
               seed=1, estimator=est, ref_duration=ref).run()
    print("Reliability estimate before the demo workflow starts:")
    for row in est.snapshot(ref):
        print(f"  W{row['worker']}: R={row['R']:.2f}  crashes seen={row['crashes']}")
    s = Simulation(wf, pool, AdaptiveScheduler(explain=True, seed=0), faults, seed=0, record=True,
                   estimator=est, ref_duration=ref)
    r = s.run()
    results["adaptive (warm)"] = r
    for t, msg in s.log:
        if any(k in msg for k in ("->", "CRASH", "REPLICATE", "reassign", "withdrawn")):
            print(f"   t={t:6.1f}  {msg}")
    plot_gantt(s, OUT / "05_adaptive_warm.png", "adaptive (with history) with W1 crashes")
    plot_reliability(s, OUT / "05_reliability_warm.png", "Estimated reliability, with history")

    # 6 ---------------------------------------------------------------------
    banner("6. Comparison (same workflow, same faults)", a.quick)
    print(f"{'scheduler':<18}{'makespan':>10}{'deadline':>10}{'lost':>6}{'replicas':>10}{'wasted':>8}"
          f"{'res.ovh':>9}")
    for name, r in results.items():
        print(f"{name:<18}{r.makespan:>10.1f}{('met' if r.deadline_met else 'MISSED'):>10}"
              f"{r.n_failed_attempts:>6}{r.n_replicas:>10}{r.wasted_time:>8.1f}{r.resource_overhead:>9.2f}")
    print("\n(Statistical comparison over many seeds and failure rates: python -m rdas.experiments)")

    # 7 ---------------------------------------------------------------------
    banner("7. Quantum track: one real decision from step 5 as a QUBO", a.quick)
    if not captured:
        print("no multi-task decision captured")
        return
    t_dec, prob = captured[0]
    print(f"Decision at t={t_dec:.1f}: tasks {['T%d' % t for t in prob.tasks]} x workers "
          f"{['W%d' % w for w in prob.workers]} -> {prob.n * prob.m} binary variables x_ij")
    print("Linear cost c_ij (reliability-aware expected completion, criticality-weighted):")
    for i, t in enumerate(prob.tasks):
        print(f"  T{t:<3}" + "".join(f"{prob.c[i, j]:8.2f}" for j in range(prob.m)))
    bqm = to_bqm(prob)
    print(f"QUBO: {len(bqm.variables)} variables, {len(bqm.quadratic)} quadratic terms, "
          f"one-hot penalty A={prob.penalty_weight():.2f}")
    g = greedy(prob)
    sa, info = QuboSolver(num_reads=50, num_sweeps=1000, seed=1).solve(prob)
    ex = brute_force(prob)

    def fmt(assign):
        return ", ".join(f"T{prob.tasks[i]}->W{prob.workers[j]}" for i, j in enumerate(assign))

    print(f"  greedy (classical) : {fmt(g)}   cost {prob.energy(g):.3f}")
    print(f"  QUBO + SA          : {fmt(sa)}   cost {prob.energy(sa):.3f}  ({info.seconds * 1000:.1f} ms,"
          f" {info.violations} constraint violations)")
    print(f"  exact (brute force): {fmt(ex)}   cost {prob.energy(ex):.3f}")
    print("\nSame objective, three interchangeable backends. No quantum advantage is claimed: "
          "RQ2 measures quality vs solve time as the problem grows (scripts/qubo_benchmark.py).")
    print(f"\nFigures written to {OUT.resolve()}")


if __name__ == "__main__":
    main()
