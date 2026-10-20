"""Reliability-estimator and prior sensitivity (requirement: "document and sensitivity-test
reliability priors").

Runs the adaptive scheduler with different estimators and priors on held-out
seeds and reports makespan, deadline-met rate and workflow success, alongside
the reliability-blind eft-retry reference.

    python scripts/sensitivity.py --seeds 20 --out results/sensitivity
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from rdas.experiments import ExperimentConfig, run_grid

VARIANTS = [
    # label, estimator, options (Gamma-Poisson exposure / half-life in mean-task-duration units)
    ("gamma-poisson (default: b0=8, half-life 25)", "gamma-poisson", {}),
    ("gamma-poisson weak prior b0=2", "gamma-poisson", {"prior_exposure": 2}),
    ("gamma-poisson strong prior b0=32", "gamma-poisson", {"prior_exposure": 32}),
    ("gamma-poisson fast forgetting (half-life 8)", "gamma-poisson", {"half_life": 8}),
    ("gamma-poisson slow forgetting (half-life 100)", "gamma-poisson", {"half_life": 100}),
    ("beta-bernoulli a=8,b=2 (team baseline)", "beta-bernoulli", {"a": 8, "b": 2}),
    ("beta-bernoulli weak a=2,b=0.5", "beta-bernoulli", {"a": 2, "b": 0.5}),
    ("beta-bernoulli strong a=32,b=8", "beta-bernoulli", {"a": 32, "b": 8}),
    ("ema alpha=0.2", "ema", {"alpha": 0.2}),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--fail", default="0.1,0.3")
    ap.add_argument("--out", default="results/sensitivity")
    a = ap.parse_args()
    seeds = list(range(100, 100 + a.seeds))
    probs = [float(x) for x in a.fail.split(",")]
    frames = []
    ref = run_grid(ExperimentConfig(schedulers=["eft-retry"], failure_probs=probs, seeds=seeds), progress=False)
    ref["variant"] = "eft-retry (reliability-blind reference)"
    frames.append(ref)
    for label, est, opts in VARIANTS:
        cfg = ExperimentConfig(schedulers=["adaptive"], failure_probs=probs, seeds=seeds,
                               estimator=est, estimator_options=opts)
        df = run_grid(cfg, progress=False)
        df["variant"] = label
        frames.append(df)
        print(f"  done: {label}", flush=True)
    df = pd.concat(frames, ignore_index=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "runs.csv", index=False)
    summary = df.groupby(["failure_prob", "variant"], sort=False).agg(
        workflow_success=("completed", "mean"), makespan=("makespan", "mean"),
        deadline_met=("deadline_met", "mean"), resource_overhead=("resource_overhead", "mean"),
        replicas=("n_replicas", "mean")).reset_index()
    summary.to_csv(out / "summary.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_colwidth", 60, "display.precision", 3):
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
