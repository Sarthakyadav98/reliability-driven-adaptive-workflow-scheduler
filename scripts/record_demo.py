"""Record the demo scenario in simulation and build the replay viewer.

    python scripts/record_demo.py                 # -> results/viewer/simulation.html
    python scripts/record_demo.py --live results/live   # also build a viewer from rdas.live output

Scenario: the 12-task demo workflow on the 4-worker demo pool. W1 is the fastest
worker and the flakiest; it crashes every 20 time units from t=8 (10 units down).
The adaptive scheduler starts with history from one earlier workflow on the same
pool (the monitor keeps running between workflows). Illustrative, hand-picked
scenario; the statistical comparison is in docs/RESULTS.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rdas import FaultConfig, ScriptedCrash, Simulation
from rdas.engine import default_estimator, default_ref_duration
from rdas.export import run_record
from rdas.pool import demo_pool
from rdas.schedulers import make_scheduler
from rdas.schedulers.heft import heft_plan
from rdas.viewer import build_viewer
from rdas.workflow import demo_workflow, layered_dag

SCENARIO = "W1 (fast, flaky) crashes every 20 time units from t=8, down 10 units each time"


def record_simulation(out_dir: Path) -> list[dict]:
    wf = demo_workflow()
    pool = demo_pool()
    wf.deadline = round(1.4 * heft_plan(wf, pool).makespan, 1)
    faults = FaultConfig(task_failure_prob=0.0,
                         scripted_crashes=[ScriptedCrash(1, at=8.0 + 20.0 * k, downtime=10.0) for k in range(10)])
    runs = []
    for name, label, warm in [("eft-retry", "eft-retry (reliability-blind retry)", False),
                              ("adaptive", "adaptive (with history)", True),
                              ("replicate-2", "replicate-2 (every task twice)", False),
                              ("heft", "static HEFT", False)]:
        ref = default_ref_duration(wf, pool)
        est = default_estimator(pool, ref)
        if warm:  # one earlier workflow on the same pool, same crash pattern
            history = layered_dag(30, np.random.default_rng(42), name="history-30")
            crashes = [ScriptedCrash(1, at=8.0 + 20.0 * k, downtime=10.0) for k in range(25)]
            Simulation(history, pool, make_scheduler(name), FaultConfig(task_failure_prob=0.0, scripted_crashes=crashes),
                       seed=1, estimator=est, ref_duration=ref).run()
        sim = Simulation(wf, pool, make_scheduler(name, explain=True), faults, seed=0, estimator=est,
                         ref_duration=ref, record=True)
        res = sim.run()
        rec = run_record(sim, res, scenario=SCENARIO, label=label)
        (out_dir / f"{name}.json").write_text(json.dumps(rec), encoding="utf-8")
        runs.append(rec)
        print(f"  {label:38s} completed={res.completed} makespan={res.makespan:.1f}")
    return runs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/viewer")
    ap.add_argument("--live", default="", help="rdas.live output folder (contains <scheduler>/run.json)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = record_simulation(out)
    page = build_viewer(runs, out / "simulation.html", title="Schedule replay: simulated crashes",
                        subtitle=f"demo-12 · 12 tasks · 4 workers · {SCENARIO}",
                        note="One illustrative, hand-picked scenario. Statistical results: 30 held-out instances "
                             "per cell in the experiment results.")
    print(f"viewer: {page.resolve()}")
    if a.live:
        live_runs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(Path(a.live).glob("*/run.json"))]
        live_runs.sort(key=lambda r: (r["scheduler"].startswith("adaptive"), r["scheduler"]))  # baseline left
        if live_runs:
            page = build_viewer(live_runs, out / "live.html", title="Schedule replay: real worker containers",
                                note="Real execution on HTTP worker agents. Single demonstration runs; timings "
                                     "include HTTP and host noise.")
            print(f"viewer: {page.resolve()}")


if __name__ == "__main__":
    main()
