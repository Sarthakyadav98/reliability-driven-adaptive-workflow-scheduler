import json
import re

import numpy as np

from rdas import FaultConfig, Simulation
from rdas.export import run_record
from rdas.pool import make_pool
from rdas.schedulers import make_scheduler
from rdas.viewer import build_viewer
from rdas.workflow import layered_dag


def _record(name):
    wf = layered_dag(15, np.random.default_rng(0))
    pool = make_pool(3, np.random.default_rng(1))
    sim = Simulation(wf, pool, make_scheduler(name, explain=True), FaultConfig(task_failure_prob=0.3), seed=2,
                     record=True)
    return run_record(sim, sim.run(), scenario="test")


def test_run_record_is_plain_json_and_complete():
    rec = _record("adaptive")
    text = json.dumps(rec, allow_nan=False)  # no inf / NaN may leak into the viewer data
    back = json.loads(text)
    assert back["format"] == "rdas-run/1"
    assert len(back["workflow"]["tasks"]) == 15 and len(back["workers"]) == 3
    assert back["attempts"] and back["log"] and back["reliability"]
    assert all(len(r[1]) == 3 for r in back["reliability"])


def test_viewer_embeds_data_safely(tmp_path):
    a, b = _record("eft-retry"), _record("adaptive")
    b["label"] = "evil </script><script>alert(1)</script>"
    out = build_viewer([a, b], tmp_path / "v.html", title="t")
    html = out.read_text(encoding="utf-8")
    assert "__RDAS_DATA__" not in html
    blob = re.search(r'<script id="rdas-data" type="application/json">(.*?)</script>', html, re.S).group(1)
    data = json.loads(blob)  # the embedded block parses and was not cut short by the label
    assert [r["scheduler"] for r in data["runs"]] == ["eft-retry", "adaptive"]
    assert data["runs"][1]["label"].startswith("evil </script>")
