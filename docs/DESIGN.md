# Design Specification: what the code actually implements

The technical spec for Review 2. It describes the implemented system, so the
report's Methodology and Implementation chapters should be written from it. If
the code changes, update this file in the same commit. Requirement-by-requirement
status against the team architecture spec: `docs/AUDIT.md`.

---

## 1. Architecture → code map

```
Workflow (JSON / generator) ─► Engine (logical task state, event loop, monitor) ◄─── events ───┐
                                 │  ready tasks + state                                        │
                                 ▼                                                             │
              Reliability estimator ─► Scheduler = Fault-tolerance policy ─► Placement         │
                                         (how many copies, why)      (which workers)           │
                                 │  Decision (assignments, revocations)                        │
                                 ▼                                                             │
                     ExecutionBackend: Simulation (virtual time)  |  HTTP agents (processes / Docker)
```

| Component | Module | Notes |
|---|---|---|
| Workflow generator / loader | `workflow.py` | layered DAGs, fork-join, demo DAG, **validated JSON loader** |
| Worker pool | `pool.py`, `models.WorkerSpec` | heterogeneous speed, hidden failure factor (simulation) |
| Engine + monitor | `engine.py` | logical task states, attempts, bounded retries, metrics, `SimView` for schedulers |
| Execution backends | `backends/simulation.py`, `backends/http.py` | same normalized events: `succeeded`, `failed` (+ category), `worker_unavailable`, `worker_available` |
| Fault injector (simulation) | `faults.py` | Poisson crashes, repair, slowdowns, degradation, scripted crashes |
| Worker agent (real execution) | `workers/agent.py`, `workers/task_registry.py` | stdlib HTTP server; allowlisted task types |
| Reliability estimator | `reliability.py` | Gamma-Poisson (default), Beta-Bernoulli, EMA |
| Cost model | `schedulers/cost.py` | one reliability-aware model shared by placement, QUBO, FT policy and reassignment |
| Fault-tolerance engine | `fault_tolerance.py` | `RetryOnly`, `StaticReplication`, `ThresholdFT`, `ExpectedCostFT` → `FTDecision` |
| Placement backends | `schedulers/placement.py` | round robin, least-loaded, EFT, `CostPlacement` (greedy = Reliability-Aware HEFT, QUBO + SA, exact) |
| Composed scheduler | `schedulers/policy.py` | FT policy + placement (+ reassignment); `adaptive.py` configures the core contribution |
| Static HEFT | `schedulers/heft.py` | canonical HEFT plan (only scheduler with its own planner) |
| QUBO | `qubo.py` | decision-scoped assignment with replica counts k_i |
| Experiments / live runs | `experiments.py`, `live.py`, `scripts/` | simulation sweeps; real-execution runs |

**Core rules.** A scheduler only *returns decisions*; one engine executes them
on either backend; one metrics path scores everything. The fault-tolerance
policy decides *how much protection*, the placement decides *where*, and any
policy can be paired with any placement.

## 2. Models

- **Task**: `work` units, predecessors and successors; for real execution also `name`, `type` (allowlisted) and `payload`. Duration on worker j is `work / speed_j`. Communication cost is zero.
- **Workflow**: a DAG; deadline = 1.5 × the fault-free HEFT plan makespan in sweeps, or given in the JSON.
- **Worker**: nominal `speed` (known to schedulers; read from `/capabilities` in real execution) and a hidden `failure_factor` (simulation only). Single slot plus a priority queue.
- **Attempt**: one copy of a logical task on a worker, with a unique id, outcome, failure category, expected vs actual duration, the reliability estimate used at placement, and the FT reason.

## 3. Execution semantics

- Logical task states: WAITING → READY → ASSIGNED → DONE | FAILED. READY means all parents succeeded and no live copy exists. `decide()` runs after every batch of events.
- Assigning a task that already has a live copy creates a **replica**. The first success wins and the other copies are cancelled (running ones are killed, and `/cancel` is sent to real workers).
- `worker_unavailable` kills the running attempt (`worker_unavailable` category). Queued attempts wait unless the scheduler revokes them (**reassignment**). `failed` events carry a category: `injected`, `timeout`, `dispatch_error` or `task_error`.
- When every copy of a task has failed, the task is retried, at most `max_retries` times (default 5). After that the task FAILS and the workflow stops (workflow success = false).
- **Idempotent events:** events for attempts that are unknown, finished or cancelled are ignored.
- **Failure taxonomy → reliability:** only worker-related categories (`worker_unavailable`, `injected`, `timeout`, `dispatch_error`) update the estimator. `task_error` (the task's own bug) is retried without blaming the worker.

### Simulation backend
Seed-deterministic `FaultTrace`: per-worker Poisson crash process with rate
`lambda_j = lambda_bar · failure_factor_j`, `lambda_bar = −ln(1−p) / ref_duration`,
and exponential repair (mean = 1 mean task duration). Optional slowdowns,
degradation and scripted crashes. Faults never depend on scheduling decisions
(common random numbers), so every scheduler on a seed sees the same faults.

### HTTP backend (real execution)
- Engine time = wall seconds / `seconds_per_unit`.
- One dispatcher thread per running attempt (`POST /execute`, synchronous, with a per-attempt timeout of 4× the expected duration, at least 2 s).
- A health thread polls `/health` every 0.25 s. 503, a refused connection or a reset connection → `worker_unavailable`; healthy again → `worker_available`.
- Agents emulate speed by running each task for `work · spu / speed` seconds of real work (sleep, CPU hashing, data hashing).
- Configured failures are seeded per (worker, task, attempt number); `fail_once` fails deterministically on a chosen attempt. `/admin/fail` simulates a crash; `docker compose stop` is a real one.

## 4. Reliability & performance estimator

Default: Gamma-Poisson posterior on each worker's failure rate with exponential forgetting:

```
lambda_hat_j = (a0 + n_j) / (b0 + E_j)      n_j = discounted failures, E_j = discounted up-time
P(success, duration d) = exp(-lambda_hat_j · d)
```

- `b0 = 8 × ref_duration` of pseudo-exposure. The prior mean is the fleet-pooled rate (empirical Bayes), shrunk toward an operator prior of 10% task failure. Half-life is 25 × ref_duration.
- Performance: EMA of actual / expected duration per worker.
- Alternatives: `BetaBernoulliEstimator` (`R = (a+s)/(a+b+s+f)`, a=8, b=2; the team baseline) and `EMAEstimator`. `scripts/sensitivity.py` compares them and their priors (RESULTS §5).
- The estimate in force at each placement is stored on the attempt (`est_p_success`).

## 5. Shared cost model (`schedulers/cost.py`)

```
d_ij = work_i / speed_j · perf_j
p_ij = P(success of i on j)
E_ij = avail_j + d_ij + beta (1 − p_ij)(d_ij / 2 + r_i)     r_i = mean duration (retry cost)
w_i  = criticality_i ^ gamma                                criticality = (rank_u + rank_d) / CP length
slack_i = deadline − now − rank_u(i)
```

## 6. Fault-tolerance policies (`fault_tolerance.py`)

Each returns `FTDecision(task, action ∈ {execute, retry, replicate}, replicas, reason, avoid)`.

| Policy | Rule |
|---|---|
| `RetryOnly` | 1 copy; after a failure, retry (placement re-chooses the worker) |
| `StaticReplication(k)` | always k copies, plus retry |
| `ThresholdFT` (team rule) | replicate (2) if criticality ≥ 0.68 and (mean R < 0.82 or slack ≤ 1 mean duration). A retry avoids the failed worker(s) |
| `ExpectedCostFT` (ours) | replicate iff `w_i · (time saved) > rho · (extra busy time)`, see below. A retry avoids the failed worker(s) |

Expected-cost rule, with the two best candidates by `E_ij` (a = primary) and
copies ordered by expected finish f1 ≤ f2:

```
time(single)    = f_a + beta (1 − p_a)(d_a/2 + r)
time(replicate) = p1 f1 + (1−p1) p2 f2 + (1−p1)(1−p2)(f2 + beta r)
rho = base_resource_weight + resource_weight · (waiting + imminent tasks per worker)
```

A replica only goes to an up worker that can start it before the primary is
expected to finish. FT decisions inside a batch see a provisional load map, so
later tasks see earlier primaries. `max_replicas` = 2; retries are bounded by the engine.

## 7. Placement backends (`schedulers/placement.py`)

All place k distinct copies per task, honour `avoid` when enough other workers
exist, and update the shared load map across batches.

- `RoundRobinPlacement`, `LeastLoadedPlacement`, `EFTPlacement`: reliability-blind.
- `CostPlacement("greedy")`: **Reliability-Aware HEFT**. Rank order; each copy goes to the cheapest column of the batch objective below.
- `CostPlacement("qubo")`: the same objective as a QUBO, solved by `dwave-samplers` simulated annealing.
- `CostPlacement("exact")`: brute force (validation).

## 8. Decision-scoped QUBO (`qubo.py`)

```
H = Σ_ij c_ij x_ij + Σ_{i<k} Σ_j q_ikj x_ij x_kj + A Σ_i (Σ_j x_ij − k_i)²
c_ij  = w_i E_ij / ref        (+ a large constant on avoided workers)
q_ikj = w_k d_ij / ref        (k waits for i if both are on j; sharing is charged, not forbidden)
A     = penalty_scale × (1.5 × largest possible saving from dropping a copy + 1)
```

- Copies of a task are on distinct workers by construction (one variable per task-worker pair).
- Decoded samples are checked: a task with ≠ k_i copies is **infeasible**, counted (`qubo_feasibility_rate`), then repaired with its cheapest columns.
- Build, anneal and decode times are measured separately.
- Validated: BQM energy equals the objective on every feasible assignment (including k=2), the `dimod.ExactSolver` ground state is feasible and equals brute force, and SA hits the optimum on small instances.
- Size ≤ 6 tasks × 8 workers = 48 variables, in contrast to whole-workflow QUBOs that hit the ~7-activity hardware wall (Pérez Armas et al., 2024).

## 9. Scheduler registry

| name | placement | fault tolerance | reassign |
|---|---|---|---|
| `round-robin` | round robin | retry | - |
| `least-loaded` | least backlog | retry | - |
| `heft` | static HEFT plan | resubmit to planned worker | - |
| `eft-retry` | online EFT | retry | - |
| `replicate-2` | online EFT | always 2 copies | - |
| `eft-threshold` | online EFT | threshold rule | - |
| `reliability-heft` | reliability-aware | retry | - |
| `adaptive` | reliability-aware | expected-cost | yes |
| `adaptive-qubo` | QUBO + SA | expected-cost | yes |
| `adaptive-threshold` | reliability-aware | threshold rule | yes |
| `adaptive-no-replication` | reliability-aware | retry (+ avoid) | yes |
| `adaptive-no-reassign` | reliability-aware | expected-cost | - |

`reliability-heft` vs `eft-retry` is the *reliability term on/off* ablation;
the `adaptive-*` variants ablate the fault-tolerance parts. Parameters
(`beta=1`, `resource_weight=2`, `base_resource_weight=0.05`, `gamma=2`) come
from `scripts/tune.py` on **dev seeds 0–9**. Results use **held-out seeds 100+**.

## 10. Metrics

| Metric | Definition |
|---|---|
| Makespan | last task finish; undefined (NaN) if the workflow failed |
| Workflow success | all tasks succeeded within the retry limit |
| Task success rate | succeeded tasks / tasks |
| Deadline met | completed and makespan ≤ workflow deadline |
| Recovery overhead | makespan − fault-free makespan of the same scheduler on the same workflow |
| Recovery time | per task: first loss → successful completion (mean over affected tasks) |
| Resource overhead | wasted busy time (failed + cancelled attempts) / useful busy time |
| Utilization | busy time / (workers × run length) |
| Retries / failed attempts / failures by category | counts |
| Replicas requested / queued / started / shortfall | from FT decisions vs what actually ran |
| Decision latency | wall-clock time in `decide()` |
| QUBO | feasibility rate, variables, build / anneal ms, gap to greedy |

## 11. Deviations and limitations

- **Default estimator** is Gamma-Poisson, not the team's Beta-Bernoulli (duration-aware). Both are available; the sensitivity study shows similar performance.
- **QUBO cost** is the shared expected-completion model rather than the `alpha·ET + beta·risk + gamma·lateness` sum; it uses the same signals in one consistent model.
- Single-slot workers; no per-task-type runtime model; task outputs are not forwarded to successors; no global replication budget; no workflow-level reliability target; no checkpointing.
- Synthetic DAGs (WfCommons traces not loaded); zero communication cost; workflow-level deadlines only.
- Real-execution results are demonstration runs; containers on one host are not independent failure domains.
- "Stability-Aware Scheduling" (Review 1 diagram) is not implemented: no precise algorithm was found in the literature.
