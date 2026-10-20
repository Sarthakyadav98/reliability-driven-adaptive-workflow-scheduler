# Reassignment must respect "retry elsewhere" (found in live testing, 20 Oct 2026)

**Bug.** After a task fails on worker W, the fault-tolerance policy retries it elsewhere
(`avoid = failed_on(task)`). The adaptive scheduler's reassignment step did not apply the
same rule. When W came back and looked faster, reassignment moved the queued retry straight
back onto W. In the live Docker demo (`--chaos 1@8:10`), `validate` bounced onto worker-2
(30% configured failure rate) twice and failed both times, so adaptive took about 55-56 units against about 44
for EFT-retry in two of three runs.

**Fix** (`schedulers/policy.py`, `_reassign`): never reassign a task to a worker it already
failed on.

**Effect on the held-out evaluation** (seeds 100-129, same protocol, `scripts/review2_experiments.py`
then `scripts/paired_stats.py`; full table `docs/paired_vs_baselines_reassign_fix.csv`).
Paired % makespan change of adaptive vs baseline (negative = adaptive faster, * = 95% CI excludes 0):

| scenario | vs | p | published | with fix | adaptive completed |
|---|---|---|---|---|---|
| cold | adaptive-no-reassign | 0.1 | -1.0 | -0.9 | 1.00 → 1.00 |
| cold | adaptive-no-reassign | 0.2 | -2.7* | -2.5* | 1.00 → 1.00 |
| cold | adaptive-no-reassign | 0.3 | -4.7* | -6.0* | 1.00 → 1.00 |
| cold | eft-retry | 0.1 | -1.0 | -1.0 | 1.00 → 1.00 |
| cold | eft-retry | 0.2 | -1.0 | -0.7 | 1.00 → 1.00 |
| cold | eft-retry | 0.3 | -0.7 | -2.6 | 1.00 → 1.00 |
| cold | eft-threshold | 0.1 | -0.6 | -0.6 | 1.00 → 1.00 |
| cold | eft-threshold | 0.2 | -3.2* | -2.9 | 1.00 → 1.00 |
| cold | eft-threshold | 0.3 | -7.0* | -8.1* | 1.00 → 1.00 |
| cold | heft | 0.1 | -13.9* | -13.9* | 1.00 → 1.00 |
| cold | heft | 0.2 | -31.3* | -31.1* | 1.00 → 1.00 |
| cold | heft | 0.3 | -38.0* | -38.6* | 1.00 → 1.00 |
| cold | reliability-heft | 0.1 | +0.8 | +0.8 | 1.00 → 1.00 |
| cold | reliability-heft | 0.2 | -1.7 | -1.5 | 1.00 → 1.00 |
| cold | reliability-heft | 0.3 | -1.2 | -2.7 | 1.00 → 1.00 |
| cold | replicate-2 | 0.1 | -5.8* | -5.8* | 1.00 → 1.00 |
| cold | replicate-2 | 0.2 | -4.6* | -4.3* | 1.00 → 1.00 |
| cold | replicate-2 | 0.3 | -4.2* | -5.5* | 1.00 → 1.00 |
| degrade | adaptive-no-reassign | 0.1 | -2.6* | -2.4* | 1.00 → 1.00 |
| degrade | adaptive-no-reassign | 0.2 | -4.6* | -4.2* | 0.93 → 1.00 |
| degrade | adaptive-no-reassign | 0.3 | -3.8* | -2.8* | 0.83 → 0.87 |
| degrade | eft-retry | 0.1 | -1.7 | -1.6 | 1.00 → 1.00 |
| degrade | eft-retry | 0.2 | -4.7* | -4.0* | 0.93 → 1.00 |
| degrade | eft-retry | 0.3 | -2.7 | -2.0 | 0.83 → 0.87 |
| degrade | eft-threshold | 0.1 | -2.6 | -2.3 | 1.00 → 1.00 |
| degrade | eft-threshold | 0.2 | -10.0* | -9.3* | 0.93 → 1.00 |
| degrade | eft-threshold | 0.3 | -9.4* | -8.6* | 0.83 → 0.87 |
| degrade | heft | 0.1 | -27.4* | -27.5* | 1.00 → 1.00 |
| degrade | heft | 0.2 | -30.8* | -30.6* | 0.93 → 1.00 |
| degrade | heft | 0.3 | -41.1* | -39.5* | 0.83 → 0.87 |
| degrade | reliability-heft | 0.1 | -1.7 | -1.5 | 1.00 → 1.00 |
| degrade | reliability-heft | 0.2 | -3.9* | -2.8* | 0.93 → 1.00 |
| degrade | reliability-heft | 0.3 | -4.1* | -1.9 | 0.83 → 0.87 |
| degrade | replicate-2 | 0.1 | -7.0* | -6.8* | 1.00 → 1.00 |
| degrade | replicate-2 | 0.2 | -5.2* | -5.0* | 0.93 → 1.00 |
| degrade | replicate-2 | 0.3 | -7.8* | -6.9* | 0.83 → 0.87 |
| warm | adaptive-no-reassign | 0.1 | -0.1 | +0.3 | 1.00 → 1.00 |
| warm | adaptive-no-reassign | 0.2 | -1.4 | -1.8 | 1.00 → 1.00 |
| warm | adaptive-no-reassign | 0.3 | -2.4 | -0.4 | 1.00 → 1.00 |
| warm | eft-retry | 0.1 | -1.8 | -1.4 | 1.00 → 1.00 |
| warm | eft-retry | 0.2 | -0.4 | -0.8 | 1.00 → 1.00 |
| warm | eft-retry | 0.3 | -3.8* | -1.6 | 1.00 → 1.00 |
| warm | eft-threshold | 0.1 | -1.4 | -1.0 | 1.00 → 1.00 |
| warm | eft-threshold | 0.2 | -2.7 | -3.1* | 1.00 → 1.00 |
| warm | eft-threshold | 0.3 | -8.6* | -6.8* | 1.00 → 1.00 |
| warm | heft | 0.1 | -14.8* | -14.4* | 1.00 → 1.00 |
| warm | heft | 0.2 | -31.0* | -31.1* | 1.00 → 1.00 |
| warm | heft | 0.3 | -39.1* | -38.3* | 1.00 → 1.00 |
| warm | reliability-heft | 0.1 | +0.6 | +1.0 | 1.00 → 1.00 |
| warm | reliability-heft | 0.2 | -0.2 | -0.6 | 1.00 → 1.00 |
| warm | reliability-heft | 0.3 | -2.6* | -0.6 | 1.00 → 1.00 |
| warm | replicate-2 | 0.1 | -6.6* | -6.2* | 1.00 → 1.00 |
| warm | replicate-2 | 0.2 | -4.3* | -4.6* | 1.00 → 1.00 |
| warm | replicate-2 | 0.3 | -5.9* | -4.0* | 1.00 → 1.00 |

Summary: small gains at high failure rates in the cold and degrading scenarios, plus higher completion
under degradation (0.83 → 0.87 at p = 0.3, 0.93 → 1.00 at p = 0.2). The significant warm-start gain over
EFT-retry at p = 0.3 (−3.8%) becomes −1.6% (n.s.). Live demo: 2 bounces → none; adaptive is still
slower than EFT-retry in most single live runs, because one 30% failure draw decides a 10-task run.

**Status.** Kept on this branch so `main` matches the Review 2 report. To adopt it: merge, rerun
`review2_experiments.py`, `paired_stats.py`, `sensitivity.py` and `qubo_benchmark.py`, and update
RESULTS.md, the report and the slides.
