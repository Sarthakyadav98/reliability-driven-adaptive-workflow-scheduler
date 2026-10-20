"""Interchangeable execution backends: virtual-time simulation and real HTTP worker agents."""

from .base import FAILED, SUCCEEDED, WORKER_AVAILABLE, WORKER_UNAVAILABLE, Event, ExecutionBackend
from .simulation import SimulatedBackend

__all__ = ["Event", "ExecutionBackend", "FAILED", "SUCCEEDED", "SimulatedBackend",
           "WORKER_AVAILABLE", "WORKER_UNAVAILABLE"]
