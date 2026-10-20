# rdas: Reliability-Driven Adaptive Workflow Scheduling

BTP, IIIT Kottayam. Guide: Dr. Shajulin Benedict.
Team: Sarthak Yadav, Anirudh, Sreyas, Sahithi.

A workflow (DAG) scheduler that tracks each worker's reliability online and
makes two separate decisions per ready task:

1. **Fault tolerance:** how much protection the task needs. A policy returns *execute / retry / replicate*, a replica count and a reason.
2. **Placement:** which workers run the copies. Round robin, EFT, Reliability-Aware HEFT, or a **decision-scoped QUBO** solved with simulated annealing (the hybrid quantum-classical track).

The same engine runs the workflow either in a **seed-deterministic
simulator** (for statistics) or on **real worker agents** (local processes or
Docker containers) that execute real tasks and fail for real.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"
pytest -q                      # 74 tests (engine invariants, FT policies, QUBO vs exact, HTTP agents)
```

If you skip activation, call the venv's Python directly: `.venv\Scripts\python.exe ...`.

## Run: simulation

```bash
python scripts/demo.py                    # Review 2 walkthrough (press Enter between steps; --quick to skip)
python -m rdas.experiments --seeds 10     # scheduler x failure-rate sweep -> results/sweep/
python scripts/review2_experiments.py     # held-out evaluation, 3 scenarios -> results/review2/
python scripts/paired_stats.py            # paired comparisons (only runs both schedulers completed)
python scripts/sensitivity.py             # estimator / prior sensitivity
python scripts/qubo_benchmark.py          # RQ2: greedy vs QUBO+SA vs exact, penalty sweep
python scripts/tune.py                    # weight tuning on dev seeds 0-9
```

## Run: real execution

**Local processes (no Docker):**

```bash
python scripts/local_cluster.py           # terminal 1: starts 3 worker agents from configs/workers.json
python -m rdas.live --compare eft-retry,adaptive --chaos 1@8:10   # terminal 2
```

`--chaos 1@8:10` makes worker index 1 unavailable at t=8 for 10 time units,
identically for every compared scheduler. Results go to `results/live/<scheduler>/`:
`events.json` (every attempt: outcome, failure category, expected vs actual
duration, reliability estimate used), `log.txt` and `metrics.json`, plus
`comparison.csv`.

**Docker Compose (1 orchestrator + 3 worker containers):**

```bash
docker compose build                       # Windows: see note below
docker compose up -d --wait worker-1 worker-2 worker-3
docker compose run --rm orchestrator       # eft-retry vs adaptive with a scripted crash -> results/docker/
docker compose stop worker-2               # optional: stop a real container during a run
docker compose down
```

> **Windows note:** on some Docker Desktop setups BuildKit copies files from this
> folder into the image as empty files. If `docker compose run` fails or the agents
> exit immediately, build with
> `powershell -ExecutionPolicy Bypass -File scripts\docker_build.ps1` instead,
> which streams the build context as a tar. (`scripts/docker_build.sh` on Linux/macOS.)

Worker API: `GET /health`, `GET /capabilities`, `POST /execute`,
`POST /cancel/<attempt>`, `GET /tasks/<attempt>`, and (opt-in) `POST /admin/fail`.
Task types are a strict allowlist: `sleep`, `cpu_benchmark`, `hash_payload`,
`fail_once`, `slow_task`. Payloads are data, never code.

## Workflows as JSON

```json
{"name": "demo", "deadline": 70,
 "tasks": [{"id": "ingest", "type": "sleep", "work": 6, "parents": []},
           {"id": "validate", "type": "fail_once", "work": 6, "parents": ["ingest"],
            "payload": {"fail_on_attempt": 1}}]}
```

`rdas.workflow.load_workflow_json` rejects duplicate ids, unknown or self
parents, cycles, non-positive work and task types outside the allowlist. See
`workflows/demo_dag.json`.

## Layout

```
src/rdas/
  models.py           Task, Workflow, WorkerSpec, Attempt, RunResult, failure taxonomy
  workflow.py         DAG generators, JSON loader, upward/downward rank, criticality
  pool.py             heterogeneous worker pools (simulation)
  faults.py           fault model + seed-deterministic fault traces
  reliability.py      Gamma-Poisson (default), Beta-Bernoulli, EMA estimators
  engine.py           shared engine: logical task state, attempts, bounded retries, metrics, SimView
  backends/           simulation.py (virtual time) | http.py (real worker agents)
  fault_tolerance.py  FT policies -> FTDecision(task, action, replicas, reason, avoid)
  qubo.py             decision-scoped assignment problem with replica counts, QUBO, SA, greedy, exact
  schedulers/
    cost.py           shared reliability-aware cost model
    placement.py      round robin, least-loaded, EFT, reliability-aware / QUBO / exact placement
    policy.py         PolicyScheduler = FT policy + placement (+ reassignment)
    adaptive.py       the adaptive scheduler (core contribution)
    heft.py           static HEFT (Topcuoglu et al. 2002) + EFT-based baselines
    simple.py         Round Robin, Least-Loaded
  workers/            agent.py (HTTP worker), task_registry.py (allowlisted task types)
  experiments.py      simulation sweeps (python -m rdas.experiments)
  live.py             real-execution runner (python -m rdas.live)
  viz.py              sweep plots, Gantt charts, reliability timelines, DAG drawing
scripts/              demo, experiments, paired stats, sensitivity, QUBO benchmark, tuning, cluster/docker helpers
workflows/            JSON workflows          configs/  worker pool config
docs/                 DESIGN.md (spec) · RESULTS.md (numbers) · AUDIT.md (vs team spec) · REVIEW2_PLAN.md
```

## Adding a scheduler

Compose one: `PolicyScheduler(placement, ft_policy, reassignment=...)` with any
`Placement` from `schedulers/placement.py` and any policy from
`fault_tolerance.py`. Then register it in `rdas/schedulers/__init__.py`. For a
completely different algorithm, subclass `Scheduler` and implement
`decide(view) -> Decision`. Never mutate engine state directly.

## Status (Review 2)

| Item | State |
|---|---|
| Engine, failure taxonomy, bounded retries, idempotent events | done, tested |
| Simulation backend + fault injector | done, tested |
| Real execution: HTTP agents, local cluster, Docker Compose | done, tested (local processes and containers) |
| Baselines: RR, Least-Loaded, HEFT, EFT-retry, replicate-2, threshold FT | done, tested |
| Reliability estimators (Gamma-Poisson, Beta-Bernoulli, EMA) + sensitivity | done |
| Adaptive scheduler + Reliability-Aware HEFT + ablations | done, tested |
| QUBO with replica counts, feasibility reporting, penalty sweep | done, tested |
| Held-out evaluation (30 seeds x 3 scenarios), paired statistics | done (`docs/RESULTS.md`) |
| Repeated real-execution sweep, data passing between tasks, WfCommons traces | next |

See `docs/AUDIT.md` for the full requirement-by-requirement status.
