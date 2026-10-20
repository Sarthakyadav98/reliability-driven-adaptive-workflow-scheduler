# Review 2 demo guide: RDAS Studio

About 8 minutes. Four tabs, shown in this order: **Scenario lab → Live cluster → Held-out
results → Recorded runs** (the last one only if Docker misbehaves).

## Before the panel arrives (5 minutes)

```bash
cd codebase
docker compose up -d --wait worker-1 worker-2 worker-3   # 3 worker containers on ports 8001-8003
python -m rdas.studio                                     # opens http://127.0.0.1:8765
```

- Live cluster tab → **Check workers**: three green rows (worker-1, worker-2, worker-3).
- Do one practice run in each tab, then reload the page so it starts clean.
- Projector: full screen (F11). The **Theme** button switches light/dark; light reads better on most projectors.
- If anything fails: **Recorded runs** has the committed replays and needs no Docker.

## 1. Scenario lab (about 3 minutes)

**On screen.** Left: the three setup steps. Right: the outcome table and the replay.

**Say.** "This runs our real scheduling engine in simulation, the same code as the experiments.
The demo workflow has 12 tasks on 4 workers. W1 is the fastest worker and also the
flakiest: it crashes every 20 time units. That is the situation a speed-only scheduler gets wrong."

**Click.** Leave the defaults (EFT-retry, replicate-2, HEFT, adaptive; warm start on) → **Run simulation**.

**Point at the outcome table.**
- One-line summary: adaptive finishes in 60.8 time units, 5.7% faster than EFT-retry and 48% faster than static HEFT.
- HEFT, 117.3: it computed its plan once and keeps sending work to W1 while W1 is crashing.
- Replicate-2, 61.2: almost as fast, but it ran 11 replicas and wasted about half of its useful work (0.48).
- Adaptive used 2 replicas and lost only 2 attempts.
- Every scheduler misses the deadline here (1.4 × HEFT's fault-free plan) because W1 is down half the time. That is expected, not a bug.

**Replay (scroll down).** Left column EFT-retry, right column adaptive, on one shared clock.
Press **Space** to play, or drag the slider to about t = 30.
- Hatched bars = W1 down. Red = an attempt lost in a crash. Dashed outline = a replica.
- Left: EFT-retry keeps putting tasks on W1 and loses them.
- Right: the "R" under each worker is its online reliability estimate. W1 falls to about 0.5, and the
  adaptive scheduler steers work away from it ("avoided earliest-finish W1").
- The decision feed on the right gives the reason for every decision, e.g. *reassign T5: W1 is down → W3*.

**Optional.** Click **Random fleet** (seed 100 is one of the held-out instances, p = 0.3), select
adaptive-qubo, and run. "The QUBO backend makes the same decisions through a QUBO solved by
simulated annealing. It is always feasible and competitive, but not better, and costs a few ms per decision."

**Expected questions.**
- *Is this cherry-picked?* "Yes, this one scenario is illustrative. The statistics are in Held-out results."
- *What is warm start?* "The reliability monitor keeps running between workflows, so it has seen W1 fail before."

## 2. Live cluster (about 2 minutes)

**Say.** "Now the same engine runs real tasks on three Docker containers over HTTP.
We make worker-2 (W1, the fast one, with a 30% configured failure rate) unavailable at t = 8 for 10 units. This
is identical for every scheduler, and equivalent to `docker compose stop`."

**Click.** **Check workers** (three green rows) → **Run on real workers**. It takes about 25 seconds for 3 schedulers.

**While the log streams**, point at:
- Yellow `CHAOS` line: the crash is injected.
- Red lines: `UNAVAILABLE` (crash detected as `worker_unavailable` in about 0.1 s) and `FAILED [injected]`.
- `[retry]` with `avoided earliest-finish W1`: the retry goes to a different worker.

**Result table.** HEFT **failed**: retries exhausted, because it keeps resubmitting to its planned
worker. EFT-retry and adaptive **complete**.

**Be upfront (important).** Adaptive is sometimes slower than EFT-retry here (e.g. 56 vs 44).
"This is one 10-task run: a single 30% failure draw decides the makespan, so live runs vary.
The live demo shows recovery on real infrastructure, not speed. Speed is measured statistically on the next tab."

## 3. Held-out results (about 2 minutes)

**Say.** "We tuned on seeds 0-9 and report only held-out seeds 100-129: 30 instances per point,
60-task DAGs, 6 heterogeneous workers, identical faults for every scheduler. Comparisons are paired,
on runs that both schedulers completed."

**Top cards (cold start, p = 0.3).** 38.0% faster than static HEFT, 7.0% faster than threshold FT,
4.2% faster than replicate-2 with far less waste (0.26 vs 0.63), and +0.7% vs EFT-retry, which is
**not significant**, though adaptive completed 100% of workflows against 90%.

**Chart.** x = task-failure probability; below 0 = adaptive faster; bars = 95% CI; filled points = significant.
- HEFT falls away as failures increase: the cost of a static plan.
- Replicate-2 costs 15% even with no faults: replicating everything is expensive.
- EFT-retry stays near 0: our gain over a good dynamic baseline is small.
- Click **Degrading**: replicate-2 completes more workflows when workers deteriorate quickly.
- Click the **Reliability term only** chip: no consistent effect. The gains come from the fault-tolerance decisions, not from the estimate alone.

**Table (pick p).** The completion, deadline and overhead columns, plus the ablations at the bottom.
Without reassignment, adaptive loses about 5% at p = 0.3, so reassignment matters.

**Read the yellow "Honest limits" box aloud.** The panel will respect it.

## 4. Recorded runs (fallback)

- *Live Docker review run*: the documented four-scheduler run in the report and slides (adaptive 44.5,
  adaptive-QUBO 44.2, EFT-retry 43.7, HEFT failed).
- *Simulated crashes*: the same scenario as the Scenario lab demo.

## Known issue to mention if asked

Live testing found that reassignment could move a retried task back onto the worker it had just
failed on. The fix and its held-out evaluation are on branch `fix/reassign-respects-avoid`
(`docs/REASSIGN_FIX.md`). It is not on `main`, so the code matches the numbers in the report.

## After the demo

```bash
docker compose down        # stop the worker containers; Ctrl+C stops the studio
```
