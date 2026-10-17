"""rdas: Reliability-Driven Adaptive workflow Scheduling with hybrid quantum-classical optimisation."""

from .engine import SimView, Simulation
from .faults import Degradation, FaultConfig, FaultTrace, ScriptedCrash
from .models import RunResult, Task, WorkerSpec, Workflow
from .reliability import EMAEstimator, GammaPoissonEstimator

__version__ = "0.1.0"

__all__ = [
    "Degradation", "EMAEstimator", "FaultConfig", "FaultTrace", "GammaPoissonEstimator", "RunResult",
    "ScriptedCrash", "SimView", "Simulation", "Task", "WorkerSpec", "Workflow",
]
