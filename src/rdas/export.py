"""Export a finished run as one JSON record (input for the replay viewer, ``rdas.viewer``).

Works for simulated and real-execution runs alike: everything comes from the
engine's own state, so both backends produce the same format.
"""

from __future__ import annotations

import json
import math
from pathlib import Path


def _num(x):
    if isinstance(x, float) and (math.isinf(x) or math.isnan(x)):
        return None
    return x


def run_record(engine, result, scenario: str = "", label: str | None = None) -> dict:
    wf = engine.workflow
    row = {k: _num(v) for k, v in result.as_row().items() if isinstance(v, (int, float, str, bool, type(None)))}
    return {
        "format": "rdas-run/1",
        "label": label or result.scheduler,
        "scheduler": result.scheduler,
        "scenario": scenario,
        "end_time": _num(engine.now),
        "workflow": {
            "name": wf.name,
            "deadline": wf.deadline,
            "tasks": [{"id": t.id, "label": t.label, "work": t.work, "type": t.type, "preds": list(t.preds)}
                      for t in wf.tasks],
        },
        "workers": [{"id": w.id, "label": w.label or f"W{w.id}", "speed": w.speed} for w in engine.workers],
        "attempts": [{k: _num(v) for k, v in rec.items()} for rec in engine.attempt_records()],
        "down": [[w, s, _num(e)] for w, s, e in engine.down_intervals],
        "log": [[t, m] for t, m in engine.log],
        "reliability": [[t, [round(r["R"], 4) for r in snap]] for t, snap in engine.reliability_timeline],
        "metrics": row,
    }


def write_run(path, engine, result, scenario: str = "", label: str | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(run_record(engine, result, scenario, label)), encoding="utf-8")
    return path
