"""Review 2 evaluation on HELD-OUT seeds 100-129 (weights were tuned on seeds 0-9).

Scenarios:
  cold     - no reliability history (worst case for the adaptive scheduler)
  warm     - monitor observed 2 earlier workflows on the same pool
  degrade  - one third of the workers become 8x more crash-prone partway through the run

    python scripts/review2_experiments.py            # all scenarios -> results/review2/
    python scripts/review2_experiments.py --seeds 10 # quicker
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from rdas.experiments import ExperimentConfig, run_grid, summarize
from rdas.schedulers import BASELINES
from rdas.viz import plot_sweep

ABLATIONS = ["adaptive-no-replication", "adaptive-no-reassign"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--out", default="results/review2")
    ap.add_argument("--scenarios", default="cold,warm,degrade")
    a = ap.parse_args()
    seeds = list(range(100, 100 + a.seeds))
    probs = [0.0, 0.05, 0.1, 0.2, 0.3]
    scenarios = {
        "cold": ExperimentConfig(schedulers=BASELINES + ["adaptive", "adaptive-qubo"] + ABLATIONS,
                                 failure_probs=probs, seeds=seeds),
        "warm": ExperimentConfig(schedulers=BASELINES + ["adaptive"] + ABLATIONS,
                                 failure_probs=probs, seeds=seeds, warmup_workflows=2),
        "degrade": ExperimentConfig(schedulers=BASELINES + ["adaptive"] + ABLATIONS,
                                    failure_probs=probs, seeds=seeds, degrade_fraction=0.34),
    }
    for name in a.scenarios.split(","):
        cfg = scenarios[name]
        out = Path(a.out) / name
        out.mkdir(parents=True, exist_ok=True)
        print(f"\n=== scenario {name}: {len(cfg.schedulers)} schedulers x {len(probs)} rates x {len(seeds)} seeds")
        df = run_grid(cfg, progress=False)
        summary = summarize(df)
        df.to_csv(out / "runs.csv", index=False)
        summary.to_csv(out / "summary.csv", index=False)
        (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
        plot_sweep(summary[~summary.scheduler.isin(ABLATIONS)], out)
        cols = ["failure_prob", "scheduler", "makespan", "makespan_ci", "deadline_met", "recovery_overhead",
                "resource_overhead", "replicas", "decision_ms"]
        with pd.option_context("display.width", 200, "display.precision", 3):
            print(summary[cols].to_string(index=False))


if __name__ == "__main__":
    main()
