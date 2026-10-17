# Preliminary Results (Review 2)

All numbers come from the implementation in this repository and are
reproducible with the commands below. Weights were tuned on **dev seeds 0–9**.
Everything reported here uses **held-out seeds 100–129** (30 instances per cell).

```bash
python scripts/review2_experiments.py     # -> results/review2/{cold,warm,degrade}/
python scripts/qubo_benchmark.py          # -> results/qubo/
python scripts/demo.py                    # -> results/demo/
```

**Setup:** 60-task random layered DAGs, 6 heterogeneous workers (speed 0.6–1.6;
fast workers tend to be flaky), Poisson crashes with mean repair time = 1 mean
task duration, deadline = 1.5x the fault-free HEFT plan. `p` is the probability
that an average task is hit by a crash. Every scheduler sees identical
workflows and fault traces per seed.

## 1. Headline (paired, adaptive vs baseline, % change in makespan; negative = adaptive faster)

| Scenario | vs | p=0 | 0.05 | 0.10 | 0.20 | 0.30 |
|---|---|---|---|---|---|---|
| cold start | replicate-2 | **−14.2 ±1.7** | **−7.9 ±2.5** | **−4.9 ±2.5** | **−5.1 ±2.5** | −2.5 ±3.4 |
| cold start | HEFT | +6.3 ±1.6 | **−4.9 ±3.7** | **−13.3 ±4.7** | **−32.6 ±4.9** | **−43.2 ±4.8** |
| cold start | EFT-retry | +0.9 ±0.8 | +3.8 ±1.9 | +0.1 ±2.9 | −1.4 ±2.4 | +0.1 ±4.1 |
| warm (2 prior workflows) | EFT-retry | −0.1 ±0.3 | +1.8 ±1.5 | −0.7 ±2.3 | −1.1 ±2.7 | −2.8 ±3.7 |
| degrading workers | EFT-retry | +0.9 ±0.8 | −0.2 ±2.1 | −0.7 ±2.8 | −2.6 ±2.6 | −2.1 ±4.3 |
| degrading workers | replicate-2 | **−14.2 ±1.7** | **−5.5 ±2.2** | **−6.0 ±2.5** | −3.4 ±2.9 | **−6.0 ±4.2** |

±: 95% CI of the paired difference. **Bold**: CI excludes 0. Full table:
`docs/paired_vs_baselines.csv`.

### Resource overhead (wasted / useful work), cold start

| p | adaptive | EFT-retry | replicate-2 | HEFT |
|---|---|---|---|---|
| 0.00 | 0.01 | 0.00 | 0.83 | 0.00 |
| 0.10 | 0.08 | 0.07 | 0.71 | 0.06 |
| 0.30 | 0.24 | 0.20 | 0.63 | 0.29 |

### Deadline met (fraction of 30 runs)

| Scenario, p | adaptive | EFT-retry | replicate-2 | HEFT |
|---|---|---|---|---|
| cold, 0.10 | 1.00 | 0.90 | 0.93 | 0.70 |
| cold, 0.20 | 0.80 | 0.73 | 0.57 | 0.03 |
| degrade, 0.05 | 0.97 | 0.87 | 0.97 | 0.23 |
| degrade, 0.30 | 0.23 | 0.13 | 0.03 | 0.00 |

## 2. What the results say (honest reading)

1. **Against static fault tolerance (uniform 2-replication), the adaptive scheduler wins clearly.** It is faster in every scenario and failure rate (significant in 12 of 15 cells), uses 3–100x less redundant work, and meets deadlines more often. This is the "adaptive vs one fixed fault-tolerance policy" claim of RQ1.
2. **Against static HEFT, reliability-aware online scheduling wins by 5–79% once faults appear.** HEFT is 5–6% better fault-free (a full static plan with insertion), but it has no response to faults. With degrading workers it fails to finish within the time limit in up to 37% of runs at p=0.3, because failed tasks keep going back to the planned worker.
3. **Against EFT-retry, the reliability-blind twin of our scheduler with the same priorities and placement machinery, the difference is not yet statistically significant.** The trend favours adaptive at high failure rates, with history, and under degradation (wins 57–70% of seeds; deadline-met rate up to +10 points). At low failure with no history it is slightly worse (+3.8% at p=0.05, cold).
4. **Ablation:** removing reassignment hurts most (e.g. warm, p=0.3: 393 vs 378). Removing replication changes little. The replication policy is the part to improve next.

So, at Review 2, RQ1 is **partially supported**: adaptive fault handling clearly
beats fixed policies (static replication, static planning). The extra benefit
of reliability estimation over a strong dynamic retry baseline is small and
needs (a) more history and heterogeneity, (b) a better replication policy and
(c) real-trace workflows to confirm or refute.

## 3. RQ2: decision-scoped QUBO (greedy vs QUBO+SA vs exact)

Synthetic decisions with the scheduler's cost structure, 10 instances per size:

| variables (tasks x workers) | greedy gap | SA (20 reads) gap / ms | SA (100 reads) gap / ms | exact ms |
|---|---|---|---|---|
| 9 (3x3) | 8.5% | 2.1% / 3.5 | **0.0%** / 19 | 0.1 |
| 16 (4x4) | 6.4% | 3.7% / 5.1 | 1.9% / 32 | 1.2 |
| 24 (6x4) | 16.9% | 8.4% / 7.2 | 4.8% / 46 | 30 |
| 36 (6x6) | 7.5% | 13.0% / 10.8 | 5.1% / 72 | 317 |
| 80 (10x8) | 2.0%* | 7.3%* / 23 | 1.0%* / 161 | n/a |

*gap to the best solution any method found (no exact optimum).
Zero one-hot constraint violations in all runs. Inside the simulator
(cold scenario), `adaptive-qubo` builds QUBOs of 17 variables on average,
solved in about 6 ms each. It is 2–4% slower than the greedy variant at
p ≤ 0.2 (e.g. 344 vs 332 at p=0.2) and level at p=0.3 (384 vs 389), at
about 10x the decision latency (2–3 ms vs 0.2 ms per decision). Better
per-decision objective values do not automatically translate into better
makespans: the batch objective is itself an approximation.

Reading: per-decision problems are small (≤48 variables by design), which
is exactly where SA reaches the optimum and greedy leaves 6–17% on the
table. SA quality degrades with size; the large one-hot penalty makes the
landscape rugged, so penalty tuning is the next experiment. No quantum
advantage is claimed: real annealing hardware is the V4 stretch goal.

## 4. Figures

| File | Shows |
|---|---|
| `figures/sweep_cold.png`, `sweep_warm.png`, `sweep_degrade.png` | makespan, deadline-met, recovery and resource overhead vs failure rate |
| `figures/qubo_benchmark.png` | QUBO quality gap and solve time vs size |
| `figures/demo_heft.png` | static HEFT resubmitting T4 to the crashing worker for 150+ time units |
| `figures/demo_eft_retry.png`, `demo_adaptive_warm.png` | retry vs adaptive on the same crash trace |
| `figures/demo_reliability_warm.png` | the estimator's R_j over time |
| `figures/demo_dag.png` | the 12-task demo workflow |

## 5. Threats to validity / limitations

- Synthetic layered DAGs and zero communication cost; real WfCommons traces are next.
- Single fault model (independent Poisson crashes). Correlated failures are not modelled.
- 6-worker pools; scaling to larger pools is untested.
- Weights tuned on 10 dev seeds. Held-out gains are smaller than dev gains (regression to the mean), which is why only held-out numbers are reported.
- Demo scenario (`scripts/demo.py`) is illustrative and hand-picked to show the mechanism. With different crash periods the warm adaptive scheduler can also lose to EFT-retry (e.g. period 26: 53.5 vs 50.5).
