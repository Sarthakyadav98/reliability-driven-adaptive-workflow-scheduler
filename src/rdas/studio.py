"""RDAS Studio: a local web console for the review demonstration.

    python -m rdas.studio            # opens http://127.0.0.1:8765

Four views, all driven by the same engine as the experiments:

* Scenario lab: configure a workflow, worker pool and fault script, run up to
  four schedulers in simulation and replay them side by side.
* Live cluster: check the HTTP worker agents (Docker Compose or
  scripts/local_cluster.py), then run schedulers on them with a scripted crash.
* Held-out results: the paired comparisons from docs/paired_vs_baselines.csv.
* Recorded runs: the replay pages committed under docs/demo/.

Standard library only (http.server). Binds to 127.0.0.1 by default.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import math
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from . import FaultConfig, ScriptedCrash, Simulation
from .backends.http import _request
from .engine import default_estimator, default_ref_duration
from .experiments import ExperimentConfig, build_instance, fault_config, warm_estimator
from .export import run_record
from .live import run_live
from .pool import demo_pool
from .schedulers import REGISTRY, make_scheduler
from .schedulers.heft import heft_plan
from .viewer import build_viewer_html
from .workflow import demo_workflow, layered_dag

PAGE = Path(__file__).with_name("studio.html")
REPO = Path(__file__).resolve().parents[2]  # source checkout: <repo>/src/rdas/studio.py

SCHEDULER_INFO = {
    "round-robin": "Round Robin, retry on failure",
    "least-loaded": "Least backlog, retry on failure",
    "heft": "Static HEFT plan, resubmit to planned worker",
    "eft-retry": "Online EFT, reliability-blind retry",
    "replicate-2": "Online EFT, every task twice",
    "eft-threshold": "Online EFT, threshold replication rule",
    "reliability-heft": "Reliability-aware placement, retry only",
    "adaptive": "Reliability-aware placement + expected-cost FT + reassignment",
    "adaptive-qubo": "Adaptive with QUBO + simulated annealing placement",
    "adaptive-threshold": "Adaptive placement with the threshold rule",
    "adaptive-no-replication": "Ablation: adaptive without replication",
    "adaptive-no-reassign": "Ablation: adaptive without reassignment",
}

LIMITS = {"n_tasks": (5, 120), "n_workers": (2, 10), "failure_prob": (0.0, 0.5), "schedulers": 4, "crashes": 20}


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _num(x):
    return None if isinstance(x, float) and (math.isinf(x) or math.isnan(x)) else x


def _find(rel: str) -> Path | None:
    for base in (REPO, Path.cwd()):
        p = base / rel
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------- scenario lab
def _order(names: list[str]) -> list[str]:
    """Baseline in the left column (eft-retry, the closest one, if chosen) and the first adaptive scheduler on the right."""
    base = sorted((n for n in names if not n.startswith("adaptive")), key=lambda n: n != "eft-retry")
    adapt = [n for n in names if n.startswith("adaptive")]
    if base and adapt:
        return [base[0], adapt[0]] + base[1:] + adapt[1:]
    return names


def simulate(spec: dict) -> dict:
    names = [n for n in spec.get("schedulers", []) if n in REGISTRY][:LIMITS["schedulers"]]
    if not names:
        raise ValueError("choose at least one scheduler")
    preset = spec.get("preset", "demo")
    seed = int(spec.get("seed", 100))
    p = float(_clamp(float(spec.get("failure_prob", 0.0)), *LIMITS["failure_prob"]))
    crashes = [ScriptedCrash(int(c["worker"]), float(c["at"]), float(c["downtime"]))
               for c in spec.get("crashes", [])[:LIMITS["crashes"]]]

    if preset == "demo":
        wf, pool = demo_workflow(), demo_pool()
        wf.deadline = round(1.4 * heft_plan(wf, pool).makespan, 1)
        cfg = ExperimentConfig(warmup_workflows=0)
        ref = default_ref_duration(wf, pool)
        faults = FaultConfig(task_failure_prob=p, mttr=ref)
        scenario = "demo-12 workflow on 4 hand-picked workers"
    else:
        cfg = ExperimentConfig(
            n_tasks=int(_clamp(int(spec.get("n_tasks", 60)), *LIMITS["n_tasks"])),
            n_workers=int(_clamp(int(spec.get("n_workers", 6)), *LIMITS["n_workers"])),
            degrade_fraction=0.34 if spec.get("degrade") else 0.0,
            warmup_workflows=2 if spec.get("warm") else 0)
        wf, pool, plan = build_instance(cfg, seed)
        ref = default_ref_duration(wf, pool)
        faults = fault_config(cfg, p, ref, plan.makespan, seed, len(pool))
        scenario = (f"layered DAG, {len(wf)} tasks, {len(pool)} workers, seed {seed}"
                    + (", degrading workers" if cfg.degrade_fraction else "")
                    + (", 2 prior workflows" if cfg.warmup_workflows else ""))
    faults.scripted_crashes = faults_crashes = [c for c in crashes if 0 <= c.worker < len(pool)]
    fault_txt = f"task-failure probability {p:g}" + (
        "; " + ", ".join(f"W{c.worker} down at t={c.at:g} for {c.downtime:g}" for c in faults.scripted_crashes)
        if faults.scripted_crashes else "")

    runs, rows = [], []
    for name in _order(names):
        if preset == "demo":
            est = default_estimator(pool, ref)
            if spec.get("warm") and (name.startswith("adaptive") or name.startswith("reliability")):
                # As scripts/record_demo.py: one earlier 30-task workflow on the same pool and crash pattern.
                history = layered_dag(30, np.random.default_rng(42), name="history-30")
                hist_faults = FaultConfig(task_failure_prob=p, mttr=ref, scripted_crashes=[
                    ScriptedCrash(c.worker, c.at + k * 100.0, c.downtime) for k in range(3) for c in faults_crashes])
                Simulation(history, pool, make_scheduler(name), hist_faults, seed=1, estimator=est,
                           ref_duration=ref).run()
            sched = make_scheduler(name, explain=True)
        else:
            est = warm_estimator(cfg, name, wf, pool, p, seed, ref)
            sched = make_scheduler(name, seed=seed, explain=True)  # as experiments._make, plus decision log
        t0 = time.perf_counter()
        sim = Simulation(wf, pool, sched, faults, seed=seed, estimator=est, ref_duration=ref, record=True,
                         max_retries=cfg.max_retries)
        res = sim.run()
        wall = time.perf_counter() - t0
        runs.append(run_record(sim, res, scenario=f"{scenario}; {fault_txt}", label=name))
        row = {k: _num(v) for k, v in res.as_row().items() if isinstance(v, (int, float, str, bool, type(None)))}
        row["wall_ms"] = round(1000 * wall, 1)
        rows.append(row)

    html = build_viewer_html(runs, title="Schedule replay", subtitle=f"{scenario} · {fault_txt}",
                             note="Simulated run of the scenario configured in RDAS Studio. One instance: "
                                  "illustrative, not a statistical result.")
    return {"scenario": scenario, "faults": fault_txt, "deadline": _num(wf.deadline),
            "workers": [{"id": w.id, "label": w.label or f"W{w.id}", "speed": w.speed} for w in pool],
            "rows": rows, "viewer": html}


# ---------------------------------------------------------------- live cluster
def check_workers(urls: list[str]) -> list[dict]:
    out = []
    for i, url in enumerate(urls):
        item = {"index": i, "url": url, "ok": False}
        try:
            code, caps = _request(url.rstrip("/") + "/capabilities", timeout=1.5)
            if code == 200:
                item.update(ok=True, worker_id=caps.get("worker_id", f"w{i}"), speed=caps.get("speed"),
                            failure_rate=caps.get("failure_rate"))
            else:
                item["error"] = f"HTTP {code}"
        except OSError as e:
            item["error"] = str(e.__class__.__name__)
        out.append(item)
    return out


class _LineSink(io.TextIOBase):
    def __init__(self, job):
        self.job, self.buf = job, ""

    def write(self, s):
        self.buf += s
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            if line.strip():
                with self.job["lock"]:
                    self.job["log"].append(line)
        return len(s)


JOB: dict = {"state": "idle", "log": [], "rows": [], "viewer": None, "error": None, "lock": threading.Lock()}


def _live_job(spec: dict) -> None:
    urls = spec["urls"]
    names = [n for n in spec["schedulers"] if n in REGISTRY][:LIMITS["schedulers"]]
    chaos = [(int(c["worker"]), float(c["at"]), float(c["downtime"])) for c in spec.get("crashes", [])]
    spu = float(_clamp(float(spec.get("seconds_per_unit", 0.15)), 0.05, 1.0))
    wf_path = _find("workflows/demo_dag.json")
    out_root = Path.cwd() / "results" / "studio" / time.strftime("%Y%m%d-%H%M%S")
    sink = _LineSink(JOB)
    runs = []
    try:
        if wf_path is None:
            raise FileNotFoundError("workflows/demo_dag.json not found; start the studio from the repository")
        for name in _order(names):
            sink.write(f"=== {name} on {len(urls)} real workers ===\n")
            with contextlib.redirect_stdout(sink):
                res, engine = run_live(str(wf_path), urls, name, spu, chaos, str(out_root / name))
            runs.append(json.loads((out_root / name / "run.json").read_text(encoding="utf-8")))
            row = {k: _num(v) for k, v in res.as_row().items() if isinstance(v, (int, float, str, bool, type(None)))}
            row["wall_ms"] = round(1000 * res.extra.get("wall_seconds", 0.0), 1)
            with JOB["lock"]:
                JOB["rows"].append(row)
            sink.write(f"--> {name}: completed={res.completed} makespan={res.makespan:.2f} units\n")
            time.sleep(1.0)  # let chaos-induced downtime end before the next scheduler
        chaos_txt = ", ".join(f"W{w} down at t={t:g} for {d:g}" for w, t, d in chaos) or "no scripted crash"
        html = build_viewer_html(runs, title="Schedule replay: real worker agents",
                                 subtitle=f"demo workflow · {len(urls)} workers · {chaos_txt}",
                                 note="Real execution on HTTP worker agents. Single runs; timings include HTTP "
                                      f"and host noise. Files: {out_root}")
        with JOB["lock"]:
            JOB.update(state="done", viewer=html)
    except Exception as e:  # surfaced in the UI
        with JOB["lock"]:
            JOB.update(state="error", error=f"{e.__class__.__name__}: {e}")


def start_live(spec: dict) -> dict:
    with JOB["lock"]:
        if JOB["state"] == "running":
            raise RuntimeError("a live run is already in progress")
        JOB.update(state="running", log=[], rows=[], viewer=None, error=None)
    spec["urls"] = [u.strip() for u in spec.get("urls", []) if u.strip()]
    if not spec["urls"]:
        JOB["state"] = "idle"
        raise ValueError("no worker URLs")
    threading.Thread(target=_live_job, args=(spec,), daemon=True).start()
    return {"state": "running"}


def live_status(since: int = 0) -> dict:
    with JOB["lock"]:
        return {"state": JOB["state"], "log": JOB["log"][since:], "n": len(JOB["log"]), "rows": list(JOB["rows"]),
                "error": JOB["error"], "viewer": JOB["viewer"] if JOB["state"] == "done" else None}


# ---------------------------------------------------------------- results and recordings
def held_out_results() -> dict:
    path = _find("docs/paired_vs_baselines.csv")
    if path is None:
        return {"rows": [], "source": None}
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({"scenario": r["scenario"], "vs": r["vs"], "p": float(r["p"]),
                         "n": int(r["n_pairs"]), "mean": float(r["mean_pct"]), "ci": float(r["ci95"]),
                         "sig": r["significant"] == "True", "win": float(r["win_rate"]),
                         "succ_a": float(r["success_target"]), "succ_b": float(r["success_base"]),
                         "dl_a": float(r["deadline_target"]), "dl_b": float(r["deadline_base"]),
                         "ovh_a": float(r["overhead_target"]), "ovh_b": float(r["overhead_base"])})
    return {"rows": rows, "source": "docs/paired_vs_baselines.csv"}


RECORDINGS = {
    "replay_docker_review.html": "Live Docker review run: HEFT, EFT-retry, adaptive, adaptive-QUBO",
    "replay_simulation.html": "Simulated crashes: fast, flaky worker crashes every 20 units",
    "replay_docker.html": "Earlier live Docker run: EFT-retry vs adaptive",
}


def recordings() -> list[dict]:
    folder = _find("docs/demo")
    if folder is None:
        return []
    return [{"file": f, "title": t} for f, t in RECORDINGS.items() if (folder / f).exists()]


# ---------------------------------------------------------------- HTTP
class Handler(BaseHTTPRequestHandler):
    server_version = "RDASStudio/1"

    def log_message(self, fmt, *args):  # quieter console
        pass

    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200):
        self._send(code, json.dumps(obj).encode("utf-8"), "application/json")

    def do_GET(self):
        path, _, query = self.path.partition("?")
        if path in ("/", "/index.html"):
            return self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        if path == "/api/meta":
            return self._json({"schedulers": [{"name": n, "info": SCHEDULER_INFO.get(n, "")} for n in REGISTRY],
                               "demo_workers": [{"id": w.id, "label": w.label, "speed": w.speed}
                                                for w in demo_pool()],
                               "recordings": recordings(), "limits": LIMITS})
        if path == "/api/results":
            return self._json(held_out_results())
        if path == "/api/live/status":
            since = int(dict(kv.split("=", 1) for kv in query.split("&") if "=" in kv).get("since", 0))
            return self._json(live_status(since))
        if path.startswith("/recorded/"):
            name = path.rsplit("/", 1)[-1]
            folder = _find("docs/demo")
            if name in RECORDINGS and folder and (folder / name).exists():
                return self._send(200, (folder / name).read_bytes(), "text/html; charset=utf-8")
        self._json({"error": "not found"}, 404)

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            spec = json.loads(self.rfile.read(n) or b"{}")
            if self.path == "/api/simulate":
                return self._json(simulate(spec))
            if self.path == "/api/live/check":
                return self._json({"workers": check_workers(spec.get("urls", []))})
            if self.path == "/api/live/run":
                return self._json(start_live(spec))
            self._json({"error": "not found"}, 404)
        except (ValueError, KeyError, TypeError, RuntimeError) as e:
            self._json({"error": str(e)}, 400)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="RDAS Studio: local demo console.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args(argv)
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    url = f"http://{a.host}:{a.port}/"
    print(f"RDAS Studio running at {url}  (Ctrl+C to stop)")
    if not a.no_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
