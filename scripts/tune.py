"""Grid-search the adaptive scheduler's weights on DEV seeds only.

The chosen values are then evaluated on held-out seeds (see README), so the
reported numbers are not tuned to the test instances.

    python scripts/tune.py --seeds 10 --warmup 2
"""

from __future__ import annotations

import argparse
import itertools

import pandas as pd

from rdas.experiments import ExperimentConfig, run_grid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--fail", default="0,0.05,0.1,0.2,0.3")
    a = ap.parse_args()
    probs = [float(x) for x in a.fail.split(",")]
    seeds = list(range(a.seeds))  # dev seeds 0..n-1; evaluation uses 100+

    base = run_grid(ExperimentConfig(schedulers=["eft-retry"], failure_probs=probs, seeds=seeds), progress=False)
    base_ms = base.groupby("failure_prob").makespan.mean()

    rows = []
    grid = itertools.product([1.0, 2.0], [0.5, 1.0, 2.0, 4.0], [0.05, 0.1, 0.3])
    for risk, res_w, base_w in grid:
        opts = dict(risk_weight=risk, resource_weight=res_w, base_resource_weight=base_w)
        cfg = ExperimentConfig(schedulers=["adaptive"], failure_probs=probs, seeds=seeds,
                               warmup_workflows=a.warmup, scheduler_options=opts)
        df = run_grid(cfg, progress=False)
        ms = df.groupby("failure_prob").makespan.mean()
        ratio = (ms / base_ms)
        row = dict(opts, mean_ratio=ratio.mean(), worst_ratio=ratio.max(),
                   overhead=df.resource_overhead.mean(), deadline=df.deadline_met.mean())
        row.update({f"r@{p}": ratio[p] for p in probs})
        rows.append(row)
        print({k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()}, flush=True)
    out = pd.DataFrame(rows).sort_values("mean_ratio")
    with pd.option_context("display.width", 200, "display.precision", 3):
        print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
