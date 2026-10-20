"""RDAS Studio: simulation API, results loader and HTTP endpoints."""

import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from rdas import studio


def test_simulate_demo_orders_baseline_left_and_embeds_viewer():
    res = studio.simulate({"preset": "demo", "schedulers": ["heft", "adaptive", "eft-retry"], "warm": True,
                           "crashes": [{"worker": 1, "at": 8, "downtime": 10}]})
    assert [r["scheduler"] for r in res["rows"]] == ["eft-retry", "adaptive", "heft"]
    assert all(r["completed"] for r in res["rows"])
    assert "W1 down at t=8 for 10" in res["faults"]
    assert '"format": "rdas-run/1"' in res["viewer"] and "__RDAS_DATA__" not in res["viewer"]


def test_simulate_fleet_matches_experiment_instance_and_clamps():
    res = studio.simulate({"preset": "fleet", "n_tasks": 500, "n_workers": 4, "seed": 100, "failure_prob": 0.9,
                           "schedulers": ["eft-retry", "adaptive", "replicate-2", "heft", "round-robin"]})
    assert len(res["rows"]) == studio.LIMITS["schedulers"]
    assert all(r["n_tasks"] == studio.LIMITS["n_tasks"][1] for r in res["rows"])
    assert "probability 0.5" in res["faults"]


def test_simulate_rejects_empty_scheduler_list():
    with pytest.raises(ValueError):
        studio.simulate({"preset": "demo", "schedulers": ["no-such-scheduler"]})


def test_held_out_results_load_from_docs():
    rows = studio.held_out_results()["rows"]
    heft = [r for r in rows if r["scenario"] == "cold" and r["vs"] == "heft" and r["p"] == 0.3]
    assert heft and heft[0]["mean"] < 0 and heft[0]["sig"]


def test_http_endpoints():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), studio.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        page = urllib.request.urlopen(base + "/").read().decode()
        assert "RDAS Studio" in page
        meta = json.loads(urllib.request.urlopen(base + "/api/meta").read())
        assert "adaptive" in [s["name"] for s in meta["schedulers"]]
        req = urllib.request.Request(base + "/api/simulate", method="POST",
                                     data=json.dumps({"preset": "demo", "schedulers": ["eft-retry"]}).encode(),
                                     headers={"Content-Type": "application/json"})
        assert json.loads(urllib.request.urlopen(req).read())["rows"][0]["scheduler"] == "eft-retry"
        bad = urllib.request.Request(base + "/api/simulate", method="POST", data=b'{"schedulers": []}')
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(bad)
        assert e.value.code == 400
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(base + "/recorded/..%2fREADME.md")
    finally:
        srv.shutdown()
