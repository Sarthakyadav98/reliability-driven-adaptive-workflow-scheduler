# Audit against the team architecture spec (`Anirudh/README.md`)

Audit date: 2026-10-20. Each requirement in Anirudh's README is checked
against this repository.

**Legend:** ✅ done · 🟡 partial or done differently (with the reason) · ❌ not done yet

Every ✅ points at the code and the test or command that shows it working.

## 0. Starting point

Anirudh's README assumes his baseline (`Anirudh/rdas_full_architecture/rdas_full`,
533 lines) is kept and extended. This repository had already grown past that
baseline (shared engine, 5 baselines, Bayesian estimator, validated QUBO,
held-out evaluation). So instead of porting our work into his code, we
**rebuilt our architecture around his component boundaries** and **kept the
ideas from his baseline that hold up**:

| From his baseline | Here |
|---|---|
| FT engine separate from assignment, returning action + replica count `k_t` | `rdas/fault_tolerance.py`: every policy returns `FTDecision(task, action, replicas, reason, avoid)` |
| Threshold rule: replicate if critical and (unreliable or low slack) | `ThresholdFT`, run as the `eft-threshold` / `adaptive-threshold` schedulers |
| Beta-Bernoulli estimator, prior a=8, b=2 | `BetaBernoulliEstimator` (`--estimator beta-bernoulli`), in the sensitivity study |
| QUBO replica-count penalty `A(Σ_w x_tw − k_t)²` | `rdas/qubo.py` (`AssignmentProblem.k`) |
| HEFT / Round Robin / QUBO+SA behind one `assign` interface | `Placement` backends behind `PolicyScheduler` |

**Issues in his baseline that this codebase avoids** (worth knowing before the review):

1. *Unfair failure draws.* His engine draws each failure from one RNG in dispatch order (`self.rng.random()` in `_dispatch`). Different schedulers therefore see different failures, which violates his own fair-comparison rule. Here, faults come from a seed-deterministic `FaultTrace` that is independent of scheduling (common random numbers).
2. *Capacity penalty forbids sequential tasks.* His QUBO's `B` term penalizes any two batch tasks on the same worker, exactly what his README's safeguard warns against. Here, sharing a worker is charged the expected waiting time (`q_ikj`), not forbidden.
3. *"HEFT" is a proxy.* His HEFT orders by a criticality proxy and picks the earliest-available worker. Here, `heft` follows the paper's two phases (upward rank + insertion-based EFT; tests check schedule validity, not yet the paper's worked example), and the online variant is labelled HEFT-style (`eft-retry`).
4. *Failure handling.* He counts a task failure only after all events for that task have drained, and has a deadlock fallback that downgrades replicas. Here, logical-task and attempt states are separate, and late or duplicate events are ignored (`test_late_and_duplicate_events_are_ignored`).

## 1. Goal and research questions

| Requirement | Status | Where / evidence |
|---|---|---|
| Two separate decisions: fault tolerance (how much protection) vs assignment (where) | ✅ | `fault_tolerance.py` (policy) + `schedulers/placement.py` (where), composed in `schedulers/policy.py` |
| Q: does reliability-aware assignment beat HEFT / RR? | ✅ answered | `docs/RESULTS.md` §2: vs HEFT/RR yes. The reliability term *alone* vs online EFT: no significant effect (ablation) |
| Q: when does replication help / hurt? | ✅ answered | `replicate-2`, `eft-threshold` and expected-cost replication compared (RESULTS §2, §3) |
| Q: does QUBO+SA justify its overhead? | ✅ answered | `scripts/qubo_benchmark.py` + `adaptive-qubo` (RESULTS §4): not at current settings |
| Q: do decisions improve with estimates from real outcomes? | ✅ shown | live/Docker runs update the estimator from real attempt outcomes; `warm` scenario in simulation |

## 2. Architecture

| Component | Status | Where |
|---|---|---|
| Workflow manager (DAG state, deps) | ✅ | `engine.py` logical task states; `workflow.py` |
| Scheduling engine, event loop, bounded window | ✅ | `engine.py`; window = batch of ≤6 ready tasks (`batch_size`) |
| Reliability & criticality | ✅ | `reliability.py`, `schedulers/cost.py` (rank_u, criticality, slack) |
| Runtime monitor (start / end / failure) | ✅ | engine event handlers + `Attempt` records (`engine.attempt_records()`) |
| Fault-tolerance engine (run / retry / replicate) | ✅ | `fault_tolerance.py` |
| Assignment interface → HEFT / RR / QUBO+SA | ✅ | `Placement`: `EFTPlacement`, `RoundRobinPlacement`, `LeastLoadedPlacement`, `CostPlacement(greedy / qubo / exact)` |
| Simulation backend (virtual events) | ✅ | `backends/simulation.py` |
| Docker backend (worker HTTP agents) | ✅ | `backends/http.py`, `workers/agent.py`, `docker-compose.yml` |
| Outcome processing → state / reliability | ✅ | `engine.py` `_on_succeeded / _on_failed / _on_worker_down` |
| Metrics + CSV/JSON | ✅ | `RunResult.as_row()`, `experiments.py` (CSV), `live.py` (events.json, metrics.json, comparison.csv) |
| **Rule:** only assignment + backend interchangeable, the rest shared | ✅ | one `Engine` for every scheduler and both backends. Static `heft` keeps its own planner by definition (it is a static plan). |

## 3. Online scheduling loop

| Step | Status | Notes |
|---|---|---|
| 1 Load + validate DAG | ✅ | `load_workflow_json` (tests: duplicate ids, unknown/self parents, cycles, bad types, work ≤ 0) |
| 2 Ready only when all parents succeeded | ✅ | invariant checked in every run of `test_every_scheduler_completes_correctly` and in the live test |
| 3 Refresh availability, speed, slots, reliability, load | 🟡 | all but **slots**: workers are single-slot (queue per worker) |
| 4 Estimate ET, criticality, slack, risk | ✅ | `CostModel` |
| 5 Bounded window of K ready tasks | ✅ | `batch_size` (default 6) |
| 6 Protection action | ✅ | `FTDecision` |
| 7 Scheduler placements | ✅ | `Placement.place` |
| 8 Dispatch via simulation or Docker | ✅ | `ExecutionBackend.start` |
| 9 Completion / failure / timeout / worker-unavailable | ✅ | normalized events, failure taxonomy |
| 10 First replica wins, cancel losers | ✅ | `_on_succeeded` → `_cancel` (+ `POST /cancel/<id>` on real workers) |
| 11 Update reliability + runtime estimates | ✅ | estimator `observe_*`; per-worker perf ratio |
| 12 Unlock successors; stop at failure limit | ✅ | bounded retries → task FAILED → workflow fails cleanly |
| 13 Metrics + event logs | ✅ | |
| Replica = attempt, not DAG node; requested vs launched recorded | ✅ | `replicas_requested`, `n_replicas` (queued), `replicas_started`, `replica_shortfall` (`test_replica_shortfall_is_recorded_when_too_few_workers`) |

## 4. Component map (his file names → ours)

| His | Ours |
|---|---|
| `workflow.py` (+ JSON loader) | `workflow.py` (+ `load_workflow_json`), `models.py` |
| `engine.py` | `engine.py` |
| `reliability.py` | `reliability.py` (Gamma-Poisson default, Beta-Bernoulli, EMA) |
| `fault_tolerance.py` | `fault_tolerance.py` |
| `schedulers/base.py` `assign(context)` | `schedulers/placement.py::Placement.place(ctx, items)` + `schedulers/base.py::Scheduler.decide(view)` |
| `schedulers/classical.py` | `schedulers/heft.py`, `schedulers/simple.py`, `EFTPlacement` |
| `schedulers/qubo.py` | `qubo.py` + `CostPlacement("qubo")` |
| `schedulers/reliability_aware.py` | `CostPlacement("greedy")`, registered as `reliability-heft` |
| `backends/{base,simulation,docker}.py` | `backends/{base,simulation,http}.py` (HTTP backend works for containers and local processes) |
| `workers/agent.py`, `workers/task_registry.py` | same paths |
| `monitoring.py` | engine event handlers + `Attempt.as_record` |
| `metrics.py` | `models.RunResult`, `experiments.summarize`, `scripts/paired_stats.py` |
| `config.py` | 🟡 no separate module: CLI flags (`rdas.experiments`, `rdas.live`) and `RDAS_*` env vars for agents |

## 5. Reliability and execution-time model

| Requirement | Status | Notes |
|---|---|---|
| Beta-Bernoulli `R=(a+s)/(a+b+s+f)`, prior 8/2 | ✅ | `BetaBernoulliEstimator` |
| Prior is an assumption → sensitivity tests | ✅ | `scripts/sensitivity.py` (RESULTS §5): 8 estimator/prior variants, makespan within ±4% |
| Store the estimate used for every placement | ✅ | `Attempt.est_p_success` (in `events.json`) |
| Only worker-related failures update reliability | ✅ | taxonomy in `models.py`; `test_task_errors_do_not_count_against_the_worker` |
| Default estimator | 🟡 deliberate | **Gamma-Poisson** (duration-aware: long tasks are riskier, forgetting, fleet-pooled prior) instead of Beta-Bernoulli. The sensitivity study shows both perform similarly here. |
| ET = work / speed | ✅ | |
| Docker: record actual wall-clock duration | ✅ | `actual_duration` per attempt, plus the agent's own `duration_s` |
| Moving average by **task type** and worker | 🟡 | per-worker slowdown EMA only (not per task type) |
| Conservative `mean + rho*std` | ❌ | future work |
| Criticality / slack defined consistently, thresholds configurable, tuned on dev seeds | ✅ | `slack = deadline − now − rank_u`; `ThresholdFT(...)` args; `scripts/tune.py` on seeds 0–9, evaluation on 100–129 |

## 6. Assignment algorithms

| Requirement | Status | Notes |
|---|---|---|
| A Round Robin | ✅ | `round-robin` |
| B HEFT (call it HEFT-style unless canonical) | ✅ | `heft` = canonical static HEFT (rank_u + insertion); `eft-retry` = online HEFT-style |
| C QUBO + SA, decision-scoped | ✅ | `adaptive-qubo` |
| QUBO linear cost `alpha ET + beta (1−R) C + gamma lateness` | 🟡 different form | ours: `c = w_i · E_ij`, with `E_ij` the reliability-aware *expected completion* (backlog + duration + failure-risk term) and `w_i = criticality^gamma`. Same signals, one consistent model shared by greedy, QUBO and the FT policy. Deadline lateness is not a QUBO term. |
| Replica-count penalty `A(Σx − k)²`, distinct workers | ✅ | `to_bqm`; `test_qubo_replica_count_constraint` |
| Capacity that does not forbid sequential tasks | ✅ | queue-wait interaction `q_ikj` |
| Validate decoded solutions, report infeasible | ✅ | `decode` counts violations; `qubo_feasibility_rate`, `qubo_infeasible` in every run |
| Penalty sweep; build vs anneal time separately | ✅ | `scripts/qubo_benchmark.py` → `penalty_sweep_summary.csv`; `mean_solver_build_ms`, `mean_solver_anneal_ms` |
| Single vs multiple restarts as a controlled experiment | ✅ | 20 vs 100 SA reads in the benchmark |
| Not a quantum result | ✅ | stated everywhere |
| D Reliability-Aware HEFT + ablation with the reliability term off | ✅ | `reliability-heft` vs `eft-retry` (identical except the reliability term) |
| E Min-Min (optional) | ❌ | not needed so far |

## 7. Fault-tolerance policy

| Requirement | Status | Notes |
|---|---|---|
| Separate from assignment; decision = task, action, replica count, reason | ✅ | reason is logged on every attempt |
| Ordinary task: one attempt | ✅ | |
| Critical + (low reliability or low slack) → 2 replicas | ✅ | `ThresholdFT`. Our `ExpectedCostFT` replaces fixed thresholds with an expected-cost test |
| Replication **budget** | 🟡 | per-task `max_replicas`; no global budget. `ExpectedCostFT` charges replicas by load pressure instead |
| Failure → retry / reassign to a *different* worker, up to a maximum | ✅ | `reassign_on_retry` + `avoid`; `max_retries` (default 5; live default 3); `test_failed_task_is_retried_on_a_different_worker`, `test_retries_are_bounded_and_workflow_fails_cleanly` |
| Too few distinct workers → launch what is feasible, record shortfall | ✅ | `replica_shortfall` |
| First success wins, cancel the rest; never unbounded | ✅ | |
| Workflow-level reliability target (RWSUT idea) | ❌ | later, as the README suggests |

## 8. Docker implementation

| Requirement | Status | Notes |
|---|---|---|
| Compose: 1 orchestrator + 3 worker agents, private network | ✅ | `docker-compose.yml`; **tested** on Docker Desktop 29.7 / Compose v5 |
| One image, per-worker env (id, speed, failure behaviour) | ✅ | `RDAS_WORKER_ID / SPEED / FAILURE_RATE / SEED / ENABLE_ADMIN` |
| No Docker socket in the orchestrator; HTTP dispatch | ✅ | |
| `GET /health`, `GET /capabilities`, `POST /execute`, `GET /tasks/{id}` | ✅ | plus `POST /cancel/{id}` and `POST /admin/fail` (simulated crash, opt-in) |
| Result record fields, UTC ISO-8601 | ✅ | `test_agent_endpoints` |
| Allowlisted task types: sleep, cpu_benchmark, process_chunk / hash, fail_once, slow_task | ✅ | `workers/task_registry.py` (`hash_payload` is the process-chunk task) |
| Validate types, sizes, timeouts; no arbitrary code | ✅ | 64 KB payload cap, bounded rounds/factors, timeout per attempt |
| Pass small results to successors (data dependencies) | ❌ | outputs are recorded, not forwarded yet |
| Speed and slots configured separately | 🟡 | speed yes; slots fixed at 1 |
| Failure taxonomy: task exception, injected, timeout, unavailable, network/dispatch | ✅ | `task_error`, `injected`, `timeout`, `worker_unavailable`, `dispatch_error` |
| Deterministic `fail_once`, then seeded probabilistic failures | ✅ | probabilistic failures seeded per (worker, task, attempt number) |
| Container stop = process unavailability (stated) | ✅ | tested: `docker compose stop worker-2` mid-run → `worker_unavailable` detected in about 0.1 s, run completes |
| Backend interface dispatch / poll_events / cancel | ✅ | named `start / next_events / cancel` |
| Shared event types | 🟡 | `succeeded`, `failed` (with category, so `timeout` is a category), `worker_unavailable`, `worker_available`. `started` / `cancelled` are engine log entries, not backend events |
| Logical vs attempt state; idempotent late events | ✅ | |
| Suggested `RDAS_*` env vars for the orchestrator | 🟡 | orchestrator is configured by CLI flags (`--scheduler`, `--max-retries`, ...) |

**Windows note:** on our machine, Docker Desktop's BuildKit file sync copied every
source file into the image as **0 bytes**. `scripts/docker_build.ps1` streams the
build context as a tar and avoids it.

## 9. Metrics and fair evaluation

| Metric | Status | Field |
|---|---|---|
| Makespan | ✅ | `makespan` (NaN when the workflow failed; never averaged with failed runs) |
| Workflow success rate | ✅ | `completed` |
| Task success rate | ✅ | `task_success_rate` |
| Deadline violations (which) | 🟡 | workflow-level deadline (`deadline_met`); task-level deadlines not modelled |
| Retry count | ✅ | `n_retries` (logical re-queues), `n_failed_attempts` |
| Replication overhead, requested vs launched | ✅ | `replicas_requested`, `n_replicas`, `replicas_started`, `resource_overhead` |
| Worker utilization (window defined) | ✅ | busy / (workers × [0, end]) |
| Scheduler runtime separate from execution | ✅ | `mean_decision_ms` |
| QUBO feasibility rate | ✅ | `qubo_feasibility_rate` |
| Recovery time | ✅ | `recovery_time_mean`: first loss of a task → its successful completion |
| Actual vs predicted duration | ✅ | `events.json` |

| Protocol rule | Status |
|---|---|
| Identical DAGs, workers, deadlines, pre-generated failures | ✅ (simulation) |
| Fault schedule independent of event order | ✅ `FaultTrace` |
| More than 5 seeds, mean ± CI | ✅ 30 held-out seeds, paired 95% CIs |
| Tune on dev seeds, report held-out | ✅ |
| Ablations: reliability on/off, FT on/off, QUBO penalty sweep | ✅ |
| Docker: repeat runs, document host and limits, cold vs warm timings | ❌ single demo runs so far |

## 10. Supplied papers

| Paper | Status |
|---|---|
| RWSUT (dynamic reliability requirement) | ❌ workflow-level target not implemented; estimates already update from outcomes |
| Energy + reliability + deadline (slack-aware, rolling horizon) | ✅ slack-aware `ThresholdFT`; the decision window is a rolling horizon. Energy deferred. |
| Empirical QUBO evaluation | ✅ feasibility, penalty sweep, solver variants, exact comparison |
| Hybrid QA job shop (allocation separate from scheduling) | ✅ the QUBO only assigns workers; readiness and dispatch live in the engine |
| Reliability in ML clusters (failure taxonomy) | ✅ taxonomy and per-attempt records |
| Quantum scheduling survey | ✅ evaluation framing, no advantage claimed |
| IEEE 11264338 | ❌ metadata still unverified: read the PDF before citing |

## 11. Milestones

| Milestone | Status |
|---|---|
| M1 Stabilize: state transitions, retries, replicas, QUBO feasibility, configurable priors / penalties / retries | ✅ |
| M2 JSON DAG loader | ✅ |
| M3 Docker worker agents | ✅ |
| M4 Failure-recovery demo (fail_once, timeouts, reassignment, replicas, cancellation, late events) | ✅ (late events tested at engine level) |
| M5 Reliability-Aware HEFT + ablations | ✅ |
| M6 Rigorous experiments, reproducible commands | 🟡 simulation ✅; Docker results are demo runs, not a repeated sweep |

## 12. Open items, by priority

1. Repeated Docker runs (5–10 per scheduler) with host specs → a small real-execution results table.
2. Pass task outputs to successors (data dependencies) in the live backend.
3. Tune the QUBO penalty scale on **dev** seeds (0.5× looked better on held-out seeds, but must not be tuned there).
4. Multi-slot workers; per-task-type runtime estimates.
5. Workflow-level reliability target (RWSUT), replication budget.
6. Verify IEEE 11264338 and the 2026 papers before citing.
