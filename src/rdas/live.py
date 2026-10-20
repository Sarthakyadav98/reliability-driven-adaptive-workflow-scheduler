"""Run a workflow on real worker agents (local processes or Docker containers).

    python -m rdas.live --workflow workflows/demo_dag.json \
        --workers http://127.0.0.1:8001,http://127.0.0.1:8002,http://127.0.0.1:8003 \
        --compare eft-retry,adaptive --chaos 1@6:8 --seconds-per-unit 0.25

``--chaos W@T:D`` makes worker index W unavailable at engine time T for D engine
time units (via the agent's /admin/fail endpoint). That is the reproducible
equivalent of ``docker compose stop worker-2``, applied identically to every
compared scheduler.

Writes, per scheduler: ``events.json`` (every attempt with outcome, failure
category, estimated vs actual duration and the reliability estimate used),
``log.txt`` (timeline) and ``metrics.json``, plus a ``comparison.csv`` across
schedulers.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import threading
import time
from pathlib import Path

from .backends.http import HttpBackend, crash_worker, discover_workers, wait_until_healthy
from .engine import Engine, default_estimator, default_ref_duration
from .export import write_run
from .schedulers import make_scheduler
from .workflow import load_workflow_json


def parse_chaos(spec: str | None) -> list[tuple[int, float, float]]:
    out = []
    for item in (spec or "").split(","):
        item = item.strip()
        if not item:
            continue
        w, rest = item.split("@")
        t, d = rest.split(":")
        out.append((int(w), float(t), float(d)))
    return out


def run_live(workflow_path: str, urls: list[str], scheduler: str, seconds_per_unit: float = 0.25,
             chaos: list[tuple[int, float, float]] | None = None, out_dir: str | None = None,
             estimator=None, quiet: bool = False, max_retries: int = 3):
    wf = load_workflow_json(workflow_path)
    wait_until_healthy(urls)
    workers = discover_workers(urls)
    ref = default_ref_duration(wf, workers)
    est = estimator or default_estimator(workers, ref)
    backend = HttpBackend(urls, seconds_per_unit=seconds_per_unit)
    t_wall = time.monotonic()

    def show(t, msg):
        if not quiet:
            print(f"  [{time.monotonic() - t_wall:6.2f}s | t={t:6.1f}] {msg}", flush=True)

    engine = Engine(wf, workers, make_scheduler(scheduler, explain=True), backend, estimator=est,
                    ref_duration=ref, record=True, on_event=show, max_retries=max_retries)
    timers = []
    for w, t, d in chaos or []:
        def fire(w=w, d=d):
            show(engine.now, f"CHAOS: making W{w} ({workers[w].label}) unavailable for {d:g} units")
            crash_worker(urls[w], d * seconds_per_unit)
        timer = threading.Timer(t * seconds_per_unit, fire)
        timer.daemon = True
        timers.append(timer)
    for timer in timers:
        timer.start()
    try:
        res = engine.run()
    finally:
        for timer in timers:
            timer.cancel()
    res.extra["wall_seconds"] = round(time.monotonic() - t_wall, 2)
    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "events.json").write_text(json.dumps(engine.attempt_records(), indent=2), encoding="utf-8")
        (out / "log.txt").write_text("\n".join(f"t={t:7.2f}  {m}" for t, m in engine.log), encoding="utf-8")
        row = {k: (None if isinstance(v, float) and math.isinf(v) else v) for k, v in res.as_row().items()}
        row["workers"] = [{"index": w.id, "label": w.label, "speed": w.speed} for w in workers]
        row["final_reliability"] = est.snapshot(ref)
        (out / "metrics.json").write_text(json.dumps(row, indent=2, default=str), encoding="utf-8")
        chaos_txt = ", ".join(f"W{w} down at t={t:g} for {d:g}" for w, t, d in chaos or []) or "no scripted crash"
        write_run(out / "run.json", engine, res, scenario=f"real workers ({len(urls)}); {chaos_txt}")
    return res, engine


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Run a workflow on real HTTP worker agents.")
    ap.add_argument("--workflow", default="workflows/demo_dag.json")
    ap.add_argument("--workers", default="http://127.0.0.1:8001,http://127.0.0.1:8002,http://127.0.0.1:8003")
    ap.add_argument("--scheduler", default="adaptive")
    ap.add_argument("--compare", default="", help="comma-separated schedulers to run one after another")
    ap.add_argument("--seconds-per-unit", type=float, default=0.25)
    ap.add_argument("--chaos", default="", help="W@T:D[,W@T:D...] worker index, start, duration (engine units)")
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--out", default="results/live")
    a = ap.parse_args(argv)
    urls = [u.strip() for u in a.workers.split(",") if u.strip()]
    names = [s.strip() for s in a.compare.split(",") if s.strip()] or [a.scheduler]
    chaos = parse_chaos(a.chaos)
    rows = []
    for name in names:
        print(f"\n=== {name} on {len(urls)} real workers ===", flush=True)
        res, _ = run_live(a.workflow, urls, name, a.seconds_per_unit, chaos, f"{a.out}/{name}",
                          max_retries=a.max_retries)
        rows.append(res.as_row())
        print(f"--> {name}: completed={res.completed} makespan={res.makespan:.1f} units "
              f"({res.extra['wall_seconds']}s wall), failed attempts={res.n_failed_attempts}, "
              f"replicas={res.n_replicas}, retries={res.n_retries}, failures={res.failures_by_category}")
        time.sleep(1.0)  # let any chaos-induced downtime end before the next scheduler starts
    Path(a.out).mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with open(Path(a.out) / "comparison.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)
    print(f"\nresults written to {a.out}/")


if __name__ == "__main__":
    main()
