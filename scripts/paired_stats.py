"""Paired comparison of the adaptive scheduler against each baseline.

Makespan is compared only on (seed, failure rate) pairs that BOTH schedulers
completed. A plain mean over completed runs would be biased in favour of a
scheduler that fails its hardest instances. Workflow success is reported
separately.

    python scripts/paired_stats.py results/review2 --target adaptive
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def paired(df: pd.DataFrame, target: str, base: str, p: float) -> dict:
    a = df[(df.scheduler == target) & (df.failure_prob == p)].set_index("seed")
    b = df[(df.scheduler == base) & (df.failure_prob == p)].set_index("seed")
    both = a.index.intersection(b.index)
    a, b = a.loc[both], b.loc[both]
    ok = a.completed & b.completed
    d = 100 * (a.makespan[ok] - b.makespan[ok]) / b.makespan[ok]
    ci = 1.96 * d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan
    return dict(vs=base, p=p, n_pairs=int(ok.sum()), mean_pct=d.mean(), ci95=ci,
                significant=bool(abs(d.mean()) > ci) if len(d) > 1 else False, win_rate=(d < 0).mean(),
                success_target=a.completed.mean(), success_base=b.completed.mean(),
                deadline_target=a.deadline_met.mean(), deadline_base=b.deadline_met.mean(),
                overhead_target=a.resource_overhead.mean(), overhead_base=b.resource_overhead.mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default="results/review2")
    ap.add_argument("--target", default="adaptive")
    a = ap.parse_args()
    rows = []
    for scen in sorted(p for p in Path(a.root).iterdir() if (p / "runs.csv").exists()):
        df = pd.read_csv(scen / "runs.csv")
        for base in sorted(set(df.scheduler) - {a.target}):
            for p in sorted(df.failure_prob.unique()):
                rows.append({"scenario": scen.name, **paired(df, a.target, base, p)})
    out = pd.DataFrame(rows)
    out.to_csv(Path(a.root) / f"paired_{a.target}.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_rows", 500, "display.precision", 2):
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
