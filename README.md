# rdas: Reliability-Driven Adaptive Workflow Scheduling

BTP, IIIT Kottayam. Guide: Dr. Shajulin Benedict.
Team: Sarthak Yadav, Anirudh, Sreyas, Sahithi.

A workflow (DAG) scheduler that tracks each worker's reliability online and
adapts **where tasks run** and **how faults are handled** (retry, reassignment,
replication). The task-to-worker assignment step can be swapped between a
classical greedy solver and a **decision-scoped QUBO** solved with simulated
annealing (the hybrid quantum-classical track).

Everything runs in a discrete-event simulator with a controllable fault
injector, so all schedulers are compared on identical workflows and identical
fault traces.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"
pytest -q                      # 49 tests
```

## Run

```bash
python scripts/demo.py                    # Review 2 live demo (press Enter between steps; --quick to skip)
python -m rdas.experiments --seeds 10     # scheduler x failure-rate sweep -> results/sweep/
python scripts/review2_experiments.py     # held-out evaluation, 3 scenarios -> results/review2/
python scripts/qubo_benchmark.py          # RQ2: greedy vs QUBO+SA vs exact -> results/qubo/
python scripts/tune.py                    # weight tuning on dev seeds 0-9
```

Every run writes CSVs and PNG plots under `results/` (git-ignored; regenerate any time).

## Layout

```
src/rdas/
  models.py          Task, Workflow, WorkerSpec, Attempt, RunResult
  workflow.py        DAG generators, upward/downward rank, criticality
  workers.py         heterogeneous worker pools
  faults.py          fault model + seed-deterministic fault traces
  reliability.py     Gamma-Poisson (Bayesian) and EMA reliability estimators
  engine.py          shared discrete-event engine + monitor + read-only SimView
  qubo.py            decision-scoped assignment problem, QUBO (dimod), SA, greedy, exact
  schedulers/
    base.py          Scheduler / Decision / Assignment interface
    simple.py        Round Robin, Least-Loaded
    heft.py          HEFT (Topcuoglu et al. 2002), EFT-retry, static replication
    adaptive.py      the adaptive scheduler (core contribution)
  experiments.py     sweep harness (CLI: python -m rdas.experiments)
  viz.py             sweep plots, Gantt charts, reliability timelines, DAG drawing
scripts/             demo, Review 2 experiments, QUBO benchmark, tuning
tests/               engine invariants, HEFT, faults, estimators, QUBO correctness
docs/DESIGN.md       full technical specification (models, formulas, metrics, deviations)
```

## Adding a scheduler

Subclass `rdas.schedulers.base.Scheduler`, implement `decide(view) -> Decision`
and register it in `rdas/schedulers/__init__.py`. It is then available to the
experiment harness by name. Use `view.ready_tasks()`, `view.backlog(w)`,
`view.estimator.p_success(w, d)` and the other `SimView` methods. Never mutate
engine state directly.

## Status (Review 2)

| Item | State |
|---|---|
| Simulator: DAG generator, worker pool, fault injector, monitor | done, tested |
| Baselines: RR, Least-Loaded, HEFT, EFT-retry, replicate-2 | done, tested |
| Reliability estimator (Gamma-Poisson + EMA ablation) | done, tested |
| Adaptive scheduler: placement, reassignment, replication | done, tested |
| QUBO backend (dimod + dwave-samplers SA), validated vs exact | done, tested |
| Experiment harness, plots, held-out evaluation | done |
| Docker real-execution testbed (V3) | next phase |
| WfCommons/Pegasus traces, checkpointing, D-Wave Leap (V4) | next phase / stretch |

See `docs/DESIGN.md` for formulas and `docs/RESULTS.md` for the current numbers.
