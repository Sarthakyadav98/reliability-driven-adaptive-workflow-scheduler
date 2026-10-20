"""Scheduler registry. Every scheduler runs on the same engine and metrics code.

Except static HEFT, every scheduler = fault-tolerance policy + placement backend:

| name                    | placement           | fault tolerance        | reassign |
|-------------------------|---------------------|------------------------|----------|
| round-robin             | round robin         | retry                  | -        |
| least-loaded            | least backlog       | retry                  | -        |
| heft                    | static HEFT plan    | resubmit to plan       | -        |
| eft-retry               | online EFT          | retry                  | -        |
| replicate-2             | online EFT          | always 2 copies        | -        |
| eft-threshold           | online EFT          | threshold rule         | -        |
| reliability-heft        | reliability-aware   | retry                  | -        |
| adaptive                | reliability-aware   | expected-cost          | yes      |
| adaptive-qubo           | QUBO + SA           | expected-cost          | yes      |
| adaptive-threshold      | reliability-aware   | threshold rule         | yes      |
| adaptive-no-replication | reliability-aware   | retry                  | yes      |
| adaptive-no-reassign    | reliability-aware   | expected-cost          | -        |
"""

from __future__ import annotations

from ..fault_tolerance import RetryOnly, ThresholdFT
from .adaptive import AdaptiveScheduler
from .base import Assignment, Decision, Scheduler
from .cost import CostModel
from .heft import HEFT, EFTRetry, StaticReplication, ThresholdReplication, heft_plan
from .placement import CostPlacement
from .policy import PolicyScheduler
from .simple import LeastLoaded, RoundRobin


def _reliability_heft(**kw):
    return PolicyScheduler(CostPlacement("greedy"), RetryOnly(), reassignment=False,
                           cost=CostModel(), name="reliability-heft")


def _adaptive_threshold(**kw):
    return PolicyScheduler(CostPlacement("greedy"), ThresholdFT(), reassignment=True,
                           cost=CostModel(), name="adaptive-threshold")


# Factories take keyword options (e.g. seed, explain); baselines ignore them.
REGISTRY = {
    "round-robin": lambda **kw: RoundRobin(),
    "least-loaded": lambda **kw: LeastLoaded(),
    "heft": lambda **kw: HEFT(),
    "eft-retry": lambda **kw: EFTRetry(),
    "replicate-2": lambda **kw: StaticReplication(k=2),
    "eft-threshold": lambda **kw: ThresholdReplication(),
    "reliability-heft": _reliability_heft,
    "adaptive": lambda **kw: AdaptiveScheduler(backend="greedy", **kw),
    "adaptive-qubo": lambda **kw: AdaptiveScheduler(backend="qubo", **kw),
    "adaptive-threshold": _adaptive_threshold,
    # Ablations: which part of the adaptive mechanism actually helps?
    "adaptive-no-replication": lambda **kw: AdaptiveScheduler(
        replication=False, name="adaptive-no-replication", **kw),
    "adaptive-no-reassign": lambda **kw: AdaptiveScheduler(
        reassignment=False, name="adaptive-no-reassign", **kw),
}

BASELINES = ["round-robin", "least-loaded", "heft", "eft-retry", "replicate-2"]


def make_scheduler(name: str, **kw) -> Scheduler:
    if name not in REGISTRY:
        raise ValueError(f"unknown scheduler {name!r}; choose from {sorted(REGISTRY)}")
    return REGISTRY[name](**kw)


__all__ = [
    "AdaptiveScheduler", "Assignment", "BASELINES", "Decision", "EFTRetry", "HEFT", "LeastLoaded",
    "PolicyScheduler", "REGISTRY", "RoundRobin", "Scheduler", "StaticReplication", "heft_plan",
    "make_scheduler",
]
