# Results (Review 2)

All numbers come from this repository and are reproducible with the commands
below. Weights were tuned on **dev seeds 0–9**; everything reported here uses
**held-out seeds 100–129** (30 instances per cell). This version reflects the
architecture refactor of 2026-10-20 (shared fault-tolerance engine, bounded
retries, k-replica QUBO). It supersedes the earlier numbers.

```bash
python scripts/review2_experiments.py   # simulation, 3 scenarios        -> results/review2/
python scripts/paired_stats.py          # paired comparisons              -> results/review2/paired_adaptive.csv
python scripts/sensitivity.py           # estimator / prior sensitivity   -> results/sensitivity/
python scripts/qubo_benchmark.py        # RQ2 + penalty sweep             -> results/qubo/
python -m rdas.live --compare eft-retry,adaptive --chaos 1@8:10   # real worker processes / containers
```

**Simulation setup:** 60-task random layered DAGs; 6 heterogeneous workers
(speed 0.6–1.6, fast workers tend to be flaky); Poisson crashes with mean
repair time = 1 mean task duration; deadline = 1.5× the fault-free HEFT plan;
at most 5 retries per task (then the task and its workflow fail).
`p` = probability that an average task is hit by a crash. Every scheduler sees
identical workflows and fault traces per seed.

**How to read the comparisons:** makespan is compared **paired, only on runs
that both schedulers completed** (`scripts/paired_stats.py`). Averaging the
completed runs alone would favour a scheduler that fails its hardest
instances. Workflow success is reported separately.

## 1. Headline: adaptive vs each baseline (paired % change in makespan; negative = adaptive faster)

| Scenario | vs | p=0 | 0.05 | 0.10 | 0.20 | 0.30 |
|---|---|---|---|---|---|---|
| cold | replicate-2 | **−14.7 ±1.8** | **−9.6 ±2.3** | **−5.8 ±2.0** | **−4.6 ±3.0** | **−4.2 ±3.2** |
| cold | HEFT | +5.7 ±1.3 | **−6.8 ±3.3** | **−13.9 ±5.1** | **−31.3 ±4.8** | **−38.0 ±4.2** |
| cold | eft-threshold (team rule policy) | 0.0 ±0.7 | +1.1 ±1.7 | −0.6 ±1.6 | **−3.2 ±3.0** | **−7.0 ±3.8** |
| cold | eft-retry | +0.3 ±0.6 | +1.8 ±1.7 | −1.0 ±2.0 | −1.0 ±2.4 | −0.7 ±4.0 |
| warm (2 prior workflows) | eft-retry | −0.2 ±0.3 | +0.1 ±1.4 | −1.8 ±2.2 | −0.4 ±3.2 | **−3.8 ±3.6** |
| degrading workers | eft-retry | +0.3 ±0.6 | **−2.6 ±2.3** | −1.7 ±2.6 | **−4.7 ±2.8** | −2.7 ±3.5 |
| degrading workers | replicate-2 | **−14.7 ±1.8** | **−7.9 ±1.8** | **−7.0 ±2.4** | **−5.2 ±2.5** | **−7.8 ±3.4** |

±: 95% CI of the paired difference. **Bold**: CI excludes 0. Full table
(every baseline, win rates, deadlines, overheads): `docs/paired_vs_baselines.csv`.

### Workflow success (fraction of 30 runs that completed)

| Scenario, p | adaptive | eft-retry | replicate-2 | eft-threshold | HEFT |
|---|---|---|---|---|---|
| cold, 0.30 | **1.00** | 0.90 | 1.00 | 1.00 | 0.63 |
| warm, 0.30 | **1.00** | 0.90 | 1.00 | 1.00 | 0.63 |
| degrade, 0.20 | 0.93 | 0.97 | **1.00** | 1.00 | 0.20 |
| degrade, 0.30 | 0.83 | 0.80 | **1.00** | 0.97 | 0.07 |

### Deadline met and resource overhead (cold start)

| p | deadline met: adaptive / eft-retry / replicate-2 / HEFT | resource overhead (wasted / useful): adaptive / eft-retry / replicate-2 |
|---|---|---|
| 0.10 | 0.97 / 0.90 / 0.93 / 0.70 | 0.11 / 0.07 / 0.71 |
| 0.20 | 0.67 / 0.73 / 0.57 / 0.03 | 0.20 / 0.12 / 0.65 |
| 0.30 | 0.27 / 0.27 / 0.13 / 0.00 | 0.26 / 0.21 / 0.63 |

## 2. What the results say

1. **Against uniform replication (`replicate-2`), the adaptive scheduler is better on time and cost.** It is 4–15% faster in every scenario and failure rate (all significant) and wastes 1.6–60× less work (overhead 0.01–0.38 vs 0.6–0.83; the gap is widest at low failure rates). **But** with degrading workers at p ≥ 0.2, blanket replication completes more workflows (100% vs 93% / 83%). Two copies of everything is the most robust option when workers deteriorate quickly, and it pays for that with 60% extra work.
2. **Against static HEFT, online reliability-aware scheduling wins by 7–38% once faults appear**, and HEFT's workflow success collapses (63% at p=0.3; 7% with degrading workers), because failed tasks keep returning to their planned worker. HEFT is 5–6% better with no faults (a full static plan with insertion).
3. **Against the team's threshold policy (`eft-threshold`)**, expected-cost replication is 3–10% faster at p ≥ 0.2 with about half the replication overhead (0.26 vs 0.52 at p=0.3, cold). The threshold rule replicates 28–46 tasks per run at high failure rates; ours replicates 8–11.
4. **Against `eft-retry`, the reliability-blind twin**, the advantage is small: significant only with history (warm, p=0.3: −3.8%, and 100% vs 90% workflow success) and with degrading workers (−2.6% at p=0.05, −4.7% at p=0.2). Cold start is a tie (+1.8% at p=0.05, not significant).
5. **Ablation, reliability term on/off (`reliability-heft` vs `eft-retry`): no consistent effect.** One cell is significantly better (warm p=0.1, −2.3%), one significantly worse (cold p=0.05, +2.4%), and the rest are ties. **The gains come from the fault-tolerance side** (selective replication, reassignment off crashed workers, retrying elsewhere), not from reliability-weighted placement alone.

**Status of RQ1:** supported against fixed fault-tolerance policies (static
replication, threshold replication, static planning). Only partly supported
against a strong dynamic retry baseline: the benefit appears with history and
when workers degrade.

## 3. Replication behaviour (when does it help?)

Replicas per 60-task run, cold start:

| p | adaptive (expected-cost) | eft-threshold | replicate-2 |
|---|---|---|---|
| 0.00 | 0.7 | 1.0 | 60 |
| 0.10 | 4.2 | 3.8 | 60.5 |
| 0.30 | 11.1 | 45.8 | 64.9 |

The expected-cost policy scales replication with observed risk and with free
capacity (load pressure). Under backlog, replicas delay other work, which is
why the threshold and uniform policies lose time at high failure rates despite
completing more workflows when workers degrade.

## 4. RQ2: decision-scoped QUBO

**Inside the scheduler** (cold, 30 held-out seeds, paired vs the greedy backend on the same cost model):

| p | QUBO penalty A = 1.0× bound | A = 0.5× bound | raw feasibility | solve time (build + anneal) |
|---|---|---|---|---|
| 0.0 | +3.5 ±0.7% | +3.3 ±1.1% | 100% | ≈ 0.7 + 7.6 ms |
| 0.1 | +2.1 ±2.4% | +0.2 ±2.2% | 100% | ≈ 0.7 + 7.0 ms |
| 0.3 | +1.7 ±3.2% | −0.2 ±3.3% | 100% / 98.6% | ≈ 0.5 + 5.3 ms |

On average 17 binary variables per decision (≤ 48 by design). The greedy
backend decides in about 0.5 ms; the QUBO adds about 8 ms per batch.

**Standalone benchmark** (`scripts/qubo_benchmark.py`, scheduler-like instances, 10 per size):

| variables | greedy gap | SA 20 reads gap / ms | SA 100 reads gap / ms |
|---|---|---|---|
| 9 (3×3) | 8.5% | 2.1% / 3.5 | **0.0%** / 19 |
| 16 (4×4) | 6.4% | 3.7% / 5 | 1.9% / 32 |
| 24 (6×4) | 16.9% | 8.4% / 7 | 4.8% / 46 |
| 36 (6×6) | 7.5% | 13.0% / 11 | 5.1% / 72 |

**Penalty sweep** (2-copy tasks included; `docs/penalty_sweep_summary.csv`):
at A = 0.1× the safe bound, 0% of raw annealer outputs are feasible. At
0.25–0.5× they are 60–100% feasible with the best gaps (0.7–3%). At ≥1× they
are always feasible, but the gap grows to 4–7% (a stiffer landscape).

**Reading:** a better per-decision objective does not automatically mean a
better makespan, because the batch objective is itself an approximation. At
the default penalty, QUBO+SA is slightly worse than greedy and about 15×
slower per decision. A softer penalty (0.5×) removes most of the gap under
faults. That was measured on held-out seeds, so it must be re-tuned on dev
seeds before it becomes the default. **No quantum advantage is claimed:** this
is classical simulated annealing.

## 5. Estimator and prior sensitivity (`docs/sensitivity_summary.csv`)

Adaptive scheduler, 20 held-out seeds, p ∈ {0.1, 0.3}:

| Estimator / prior | makespan p=0.1 | makespan p=0.3 | workflow success p=0.3 |
|---|---|---|---|
| eft-retry (reference) | 294.1 | 372.6* | 0.85 |
| Gamma-Poisson default (b0=8, half-life 25) | 288.7 | 376.2 | 1.00 |
| Gamma-Poisson weak / strong prior (b0=2 / 32) | 294.3 / 287.3 | 380.3 / 364.6 | 1.00 / 1.00 |
| Gamma-Poisson fast / slow forgetting | 292.6 / 288.9 | 378.2 / 372.5 | 1.00 / 0.95 |
| Beta-Bernoulli a=8, b=2 (team baseline) | 290.7 | 379.8 | 0.95 |
| Beta-Bernoulli weak / strong | 294.8 / 288.8 | 376.9 / 384.5 | 1.00 / 0.90 |
| EMA α=0.2 | 304.9 | 375.0 | 1.00 |

\*Mean over completed runs only. eft-retry failed 15% of these workflows, so
this number excludes its hardest instances (see the note above §1).

Makespan varies by about ±4% across priors and estimators; the method is not
fragile to the prior. EMA (duration-agnostic, no uncertainty) is the weakest at
p=0.1. The Gamma-Poisson default is among the best and never lost a workflow
at p=0.3 here.

## 6. Real execution (HTTP worker agents in Docker)

`docker compose run --rm orchestrator`: 3 worker containers (speed 1.0 / 1.5 / 0.8;
configured failure rates 0.02 / 0.30 / 0.0), the 10-task `workflows/demo_dag.json`
(includes a deterministic `fail_once` task), and worker-2 made unavailable at
t=8 for 10 units. Sample logs: `docs/live_sample/`.

| scheduler | completed | makespan (units) | wall time | failed attempts (unavailable / injected) |
|---|---|---|---|---|
| eft-retry | yes | 43.9 | 6.6 s | 4 (1 / 3) |
| adaptive | yes | 44.7 | 6.7 s | 4 (1 / 3) |

`docker compose stop worker-2` mid-run (a real container stop, not the admin
endpoint) was detected as `worker_unavailable` in about 0.1 s. The lost attempt
was retried on another container and the workflow completed.

These are **single demonstration runs**, not a statistical result: the timings
include HTTP and host noise. A repeated real-execution sweep is open work.

## 7. Figures

| File | Shows |
|---|---|
| `figures/sweep_cold.png`, `sweep_warm.png`, `sweep_degrade.png` | makespan, workflow success, deadline met, recovery / resource overhead, utilization vs failure rate |
| `figures/qubo_benchmark.png` | QUBO quality gap and solve time vs size |
| `figures/demo_heft.png` | static HEFT resubmitting T4 to the crashing worker until it fails |
| `figures/demo_eft_retry.png`, `demo_adaptive_warm.png` | retry vs adaptive on the same crash trace |
| `figures/demo_reliability_warm.png` | the estimator's R_j over time |

## 8. Threats to validity / limitations

- Synthetic layered DAGs; zero communication cost; single-slot workers; workflow-level (not task-level) deadlines.
- Independent Poisson crashes. Correlated failures are not modelled; containers on one host are not independent failure domains.
- Weights tuned on 10 dev seeds. Held-out gains are smaller than dev gains (regression to the mean), which is why only held-out numbers are reported.
- Bounded retries (5) turn rare long failure streaks into workflow failures. Workflow success depends on this limit.
- The demo scenario (`scripts/demo.py`) is hand-picked to show the mechanism. Real-execution numbers are single runs.
