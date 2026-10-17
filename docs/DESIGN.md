# Design Specification: what the code actually implements

This is the single-page technical spec agreed for Review 2. It describes the
implemented system, so the report's Methodology and Implementation chapters
should be written from it. If the code changes, update this file in the same commit.

---

## 1. Architecture → code map

| Architecture component (Review 1 diagram) | Module | Notes |
|---|---|---|
| Workflow Generator | `rdas/workflow.py` | Random layered DAGs, fork-join, fixed 12-task demo DAG |
| Worker Pool | `rdas/workers.py`, `rdas/models.py::WorkerSpec` | Heterogeneous speed, hidden per-worker crash-rate factor |
| Fault Injector | `rdas/faults.py` | Poisson crashes, exponential repair, slowdowns, degradation, scripted crashes |
| Fault & Resource Monitor | `rdas/engine.py` (event handlers) | Feeds uptime, crash, recovery and attempt outcomes to the estimator |
| Reliability & Performance Estimator | `rdas/reliability.py` | Gamma-Poisson (Bayesian, forgetting, fleet-pooled prior); EMA as ablation |
| Adaptive Scheduling Controller | `rdas/schedulers/adaptive.py` | Placement + reassignment + replication policy |
| Pluggable Optimization Backend | `rdas/qubo.py` | `greedy` (classical), `qubo` (QUBO + SA), `exact` (brute force), same objective |
| Baselines | `rdas/schedulers/simple.py`, `heft.py` | Round Robin, Least-Loaded, HEFT, EFT-retry, replicate-2 |
| Evaluation Harness | `rdas/experiments.py`, `rdas/viz.py` | Same workflows, pools, fault traces and seeds for every scheduler |

**Core rule:** a scheduler only *returns decisions* (`Decision`: assignments and
revocations). The single shared engine executes them, and one metrics path
scores every scheduler.

## 2. Models

- **Task** `T_i`: `work_i` compute units, predecessor/successor lists. Duration on worker j: `work_i / speed_j`. Communication cost is zero; HEFT's own paper flags this simplification too.
- **Workflow**: a DAG with ids `0..n-1`. Deadline = `1.5 x` the fault-free HEFT plan makespan (configurable).
- **Worker** `W_j`: nominal `speed_j`, which all schedulers know, and a hidden `failure_factor_j`, which only the fault injector knows. Pools are generated so that fast workers tend to be flaky (`fast_is_flaky=0.7`). That creates the speed vs reliability trade-off speed-only schedulers miss.

## 3. Execution semantics (engine)

- Discrete-event simulation (own heap-based engine; SimPy was not needed and would complicate preempting a running attempt on a crash).
- Each worker runs one attempt at a time and has a priority queue of assigned attempts. An idle, up worker immediately starts its best-priority queued attempt.
- A task becomes **READY** when all its predecessors are done and it has no live copy. `decide()` runs after every event time.
- Assigning a task that already has a live copy creates a **replica**. The first copy to succeed wins and the others are cancelled (running copies are killed and their time counts as wasted).
- A **crash** kills the running attempt; its task returns to READY (retry) unless another copy is alive. Queued attempts wait for recovery unless the scheduler **revokes** them (reassignment).

## 4. Fault model

- Crashes: Poisson process per worker with rate `lambda_j = lambda_bar * failure_factor_j`, where `lambda_bar = -ln(1 - p) / ref_duration`. So `p` is the probability that an average task on an average worker is hit by a crash. Sweep `p` in {0, 0.05, 0.1, 0.2, 0.3} (the 5–30% range from the proposal).
- Repair: exponential with mean `MTTR = 1 x` mean task duration (default).
- Optional: slowdown episodes (2–4x slower), degradation (a worker's rate x8 from a random time), scripted crashes (demo, tests).
- **Common random numbers:** traces depend only on `(workers, config, seed)`, never on scheduling, so every scheduler on a seed faces identical faults.

## 5. Reliability & performance estimator

Gamma-Poisson posterior on each worker's crash rate, with exponential forgetting:

```
lambda_hat_j = (a0 + n_j) / (b0 + E_j)        n_j = discounted crashes, E_j = discounted up-time
P(success of a task of duration d on j) = exp(-lambda_hat_j * d)
R_j (display) = P(success) for an average-length task
```

- Prior: `b0 = 8 x ref_duration` of pseudo-exposure. The prior mean is the **fleet-pooled** rate (empirical Bayes), shrunk toward an operator prior of 10% task failure. The scheduler is never told the true `p`.
- Forgetting half-life `25 x ref_duration`, so a worker that degrades or recovers is tracked.
- Performance: EMA of actual / expected duration (catches slowdowns).
- Ablation estimator: `EMAEstimator`, `R <- 0.2*outcome + 0.8*R` (the Review 1 example: 0.9 → 0.72 → 0.776).

Why Gamma-Poisson over plain EMA: it is the conjugate Bayesian model for the
exponential failure model `R(t) = e^{-lambda t}` used in reliability-aware
scheduling. It accounts for task length (long tasks are riskier), carries
uncertainty for workers with little history, and forgets old evidence.

## 6. Adaptive scheduler (core contribution)

At each decision:

1. **Reassignment**: for every queued (not running) single-copy task, compare its expected completion on its current worker with the best alternative. Move it if the worker is down and the alternative is better, or if the alternative is better by more than `0.3 x` the task duration (hysteresis against thrashing).
2. **Placement**: ready tasks in HEFT upward-rank order, in batches of up to 6, are placed by the optimization backend on the cost model below.
3. **Replication (fault-handling policy)**: per task, compare *single copy + retry* against *add a replica on the best other up worker*, and replicate only if the gain is positive.

Cost model (task i, worker j):

```
d_ij = work_i / speed_j * perf_j
p_ij = exp(-lambda_hat_j * d_ij)
E_ij = avail_j + d_ij + beta * (1 - p_ij) * (d_ij/2 + r_i)        r_i = mean duration (retry)
w_i  = criticality_i ^ gamma     criticality = (rank_u + rank_d) / critical-path length
```

Replication rule:

```
time(single)    = f_a + beta (1 - p_a)(d_a/2 + r)
time(replicate) = p1 f1 + (1-p1) p2 f2 + (1-p1)(1-p2)(f2 + beta r)      (copies ordered by finish)
gain = w_i * (time saved) - rho * (extra busy time)
rho  = base_resource_weight + resource_weight * (waiting + imminent tasks per worker)
```

`rho` is the opportunity cost of busy time. It is high under backlog and near
zero on idle capacity, for example at the tail of a workflow, where speculative
backups are cheap. Replicas only go to up workers and must be able to start
before the primary is expected to finish (otherwise they are just a pre-booked retry).

Parameters (`risk_weight beta=1`, `resource_weight=2`, `base_resource_weight=0.05`,
`gamma=2`) were chosen by `scripts/tune.py` on **dev seeds 0–9**. All reported
results use **held-out seeds 100+**.

## 7. Decision-scoped QUBO

Variables `x_ij` = task i (current batch) on worker j.

```
H = sum_ij c_ij x_ij + sum_{i<k} sum_j q_ikj x_ij x_kj + A sum_i (sum_j x_ij - 1)^2
c_ij  = w_i * E_ij / ref                 (same cost model as the classical backend)
q_ikj = w_k * d_ij / ref                 (k waits for i if both go to j)
A     = 1.5 x (largest possible saving from breaking one-hot) + 1
```

- Solved with `dwave.samplers.SimulatedAnnealingSampler`. Any `dimod` sampler, including D-Wave Leap hardware for V4, can be passed in.
- Validated: BQM energy equals the objective on every feasible assignment; the `dimod.ExactSolver` ground state is one-hot and equals the brute-force optimum; SA hits the optimum on small instances (`tests/test_qubo.py`).
- Size is bounded by the batch (≤6 tasks) and by at most 8 workers, so ≤48 variables. This is the decision-scoped design, in contrast to whole-workflow QUBOs that hit the ~7-activity hardware wall (Pérez Armas et al., 2024).
- Capacity constraint: not needed in this model (single-slot workers with queues). Queueing interaction is captured by `q` instead.

## 8. Baselines

| Name | What it does | Weakness it isolates |
|---|---|---|
| `round-robin` | Cycle through up workers | Ignores load, speed, reliability |
| `least-loaded` | Up worker with least backlog | Ignores speed, reliability |
| `heft` | Static HEFT plan (rank_u + insertion EFT); failed task resubmitted to the same planned worker | Assumes fixed, reliable workers |
| `eft-retry` | Online HEFT-style list scheduling, retry anywhere | Reliability-blind; same machinery as adaptive, so this is the cleanest ablation |
| `replicate-2` | Every task on its 2 earliest-finish workers, plus retry | Uniform static fault tolerance |

## 9. Metrics

| Metric | Definition |
|---|---|
| Makespan | time the last task finishes |
| Deadline met | makespan ≤ workflow deadline |
| Recovery overhead | makespan − fault-free makespan of the same scheduler on the same workflow |
| Resource overhead | wasted busy time (failed + cancelled attempts) / useful busy time |
| Failed attempts, replicas, reassignments | counts |
| Decision latency | wall-clock time inside `decide()` per decision |
| QUBO solve time, gap to greedy | per solver call (QUBO variant) |

## 10. Deliberate deviations from the original plan

- **Docker testbed (V3) is deferred to after Review 2**, following the post-Review 1 advice to get the simulator right first. The engine's decision/execution split is the seam where a Docker backend plugs in.
- **Checkpointing** is not implemented (retry, replication and reassignment only).
- **"Stability-Aware Scheduling"** from the teammate's diagram is not implemented: no precise algorithm was found in the literature yet.
- **Static FT baseline** = uniform 2-replication. Revisit once the literature matrix identifies the most common conventional policy.

## 11. Known limitations (state these in the report)

- Zero communication cost; single-slot workers; synthetic DAGs (WfCommons traces not yet loaded).
- Gains appear mainly at higher failure rates (≥ 0.2). At low rates the adaptive scheduler is roughly level with EFT-retry. With little history (cold start, small workflows) it behaves much like EFT.
- When a fast worker is flaky but its crashes rarely hit tasks, caution can cost time (see the demo scenario note in the README).
- QUBO + SA quality degrades with size relative to the exact optimum. Penalty-weight tuning is open work.
