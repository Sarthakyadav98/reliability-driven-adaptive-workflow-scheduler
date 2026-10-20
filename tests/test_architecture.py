"""Tests for the requirements in the team architecture spec (Anirudh/README.md)."""

import itertools
import json
import math
import threading

import dimod
import numpy as np
import pytest

from rdas import FaultConfig, ScriptedCrash, Simulation, WorkerSpec
from rdas.backends.base import FAILED, SUCCEEDED, Event
from rdas.backends.simulation import SimulatedBackend
from rdas.engine import Engine
from rdas.faults import FaultTrace
from rdas.models import Workflow
from rdas.pool import make_pool
from rdas.qubo import AssignmentProblem, brute_force, decode, greedy, to_bqm
from rdas.reliability import BetaBernoulliEstimator
from rdas.schedulers import make_scheduler
from rdas.workflow import layered_dag, load_workflow_json


# ---------------------------------------------------------------- fault-tolerance engine
def _decisions(name, wf, pool, faults=None):
    """Run a scheduler and collect the FT reason recorded on every attempt."""
    sim = Simulation(wf, pool, make_scheduler(name), faults or FaultConfig.fault_free(), seed=0)
    sim.run()
    return sim


def test_ft_decision_is_recorded_on_every_attempt():
    wf = layered_dag(20, np.random.default_rng(0))
    pool = make_pool(4, np.random.default_rng(1))
    sim = _decisions("replicate-2", wf, pool)
    assert all("static policy" in a.reason for a in sim.attempts.values())
    assert all(a.est_p_success is not None and 0 < a.est_p_success <= 1 for a in sim.attempts.values())


def test_threshold_policy_replicates_only_critical_tasks():
    wf = layered_dag(30, np.random.default_rng(0))
    wf.deadline = 1.0  # impossible deadline: every critical task has low slack
    pool = make_pool(4, np.random.default_rng(1))
    sim = _decisions("eft-threshold", wf, pool)
    replicated = {a.task for a in sim.attempts.values() if a.kind == "replica"}
    crit = sim.scheduler.cost.crit
    assert replicated, "critical low-slack tasks should be replicated"
    assert all(crit[t] >= 0.68 for t in replicated)


def test_replica_shortfall_is_recorded_when_too_few_workers():
    wf = Workflow.from_edges([5.0, 5.0], [(0, 1)])
    pool = [WorkerSpec(0, 1.0)]  # replicate-2 asks for 2 copies, only 1 worker exists
    res = Simulation(wf, pool, make_scheduler("replicate-2"), FaultConfig.fault_free()).run()
    assert res.completed
    assert res.extra["replicas_requested"] == 2 and res.extra["replica_shortfall"] == 2
    assert res.n_replicas == 0 and res.replicas_started == 0


def test_retries_are_bounded_and_workflow_fails_cleanly():
    wf = Workflow.from_edges([10.0], [])
    pool = [WorkerSpec(0, 1.0)]
    crashes = [ScriptedCrash(0, at=5.0 + 20 * k, downtime=10.0) for k in range(20)]
    res = Simulation(wf, pool, make_scheduler("eft-retry"),
                     FaultConfig(task_failure_prob=0.0, scripted_crashes=crashes), max_retries=2).run()
    assert not res.completed and math.isinf(res.makespan)
    assert res.n_retries == 2 and res.n_failed_attempts == 3
    assert res.task_success_rate == 0.0


def test_failed_task_is_retried_on_a_different_worker():
    # T0 fails on W1 (task_error from a scripted backend); the adaptive policy retries elsewhere.
    wf = Workflow.from_edges([4.0, 4.0], [(0, 1)])
    pool = [WorkerSpec(0, 1.0), WorkerSpec(1, 1.6)]

    class FailFirstOnFastWorker(SimulatedBackend):
        def start(self, attempt):
            if attempt.worker == 1 and attempt.task == 0:
                self._push(self.engine.now + 1.0, Event(FAILED, attempt=attempt.id, worker=1, category="injected"))
            else:
                super().start(attempt)

    eng = Engine(wf, pool, make_scheduler("adaptive"),
                 FailFirstOnFastWorker(FaultTrace(pool, FaultConfig.fault_free(), 4.0, 0)))
    res = eng.run()
    assert res.completed
    t0 = sorted((a for a in eng.attempts.values() if a.task == 0), key=lambda a: a.id)
    assert t0[0].worker == 1 and t0[0].outcome == "failed"
    assert t0[1].worker == 0 and t0[1].outcome == "success"  # reassigned, not retried on W1


# ---------------------------------------------------------------- engine semantics
def test_late_and_duplicate_events_are_ignored():
    wf = layered_dag(15, np.random.default_rng(2))
    pool = make_pool(3, np.random.default_rng(3))
    sim = Simulation(wf, pool, make_scheduler("replicate-2"), FaultConfig.fault_free(), seed=0)
    res = sim.run()
    before = (sim.status[:], res.n_failed_attempts, sim.n_task_failures[:])
    for a in sim.attempts.values():  # replay every attempt as a late success AND failure
        sim._handle(Event(SUCCEEDED, attempt=a.id, worker=a.worker))
        sim._handle(Event(FAILED, attempt=a.id, worker=a.worker, category="injected"))
    assert (sim.status, sim.n_task_failures) == (before[0], before[2])
    assert all(s == "done" for s in sim.status)


def test_task_errors_do_not_count_against_the_worker():
    wf = Workflow.from_edges([4.0], [])
    pool = [WorkerSpec(0, 1.0)]

    class TaskBugOnce(SimulatedBackend):
        def start(self, attempt):
            if attempt.id == 0:
                self._push(self.engine.now + 1.0, Event(FAILED, attempt=0, worker=0, category="task_error"))
            else:
                super().start(attempt)

    est = BetaBernoulliEstimator(1)
    eng = Engine(wf, pool, make_scheduler("eft-retry"), TaskBugOnce(FaultTrace(pool, FaultConfig.fault_free(), 4.0, 0)),
                 estimator=est)
    res = eng.run()
    assert res.completed and res.failures_by_category["task_error"] == 1
    assert est.failures[0] == 0 and est.crashes[0] == 0  # reliability untouched by the task's own bug
    assert est.p_success(0, 1) > 0.8


def test_new_metrics_are_reported():
    wf = layered_dag(30, np.random.default_rng(4))
    pool = make_pool(4, np.random.default_rng(5))
    res = Simulation(wf, pool, make_scheduler("adaptive"), FaultConfig(task_failure_prob=0.3), seed=1).run()
    row = res.as_row()
    for key in ("task_success_rate", "utilization", "recovery_time_mean", "replicas_started", "n_retries",
                "fail_worker_unavailable", "replicas_requested"):
        assert key in row
    assert 0 < row["utilization"] <= 1
    if res.n_retries:
        assert res.recovery_time_mean > 0


def test_beta_bernoulli_prior():
    est = BetaBernoulliEstimator(1, a=8, b=2)
    assert est.p_success(0, 10) == pytest.approx(0.8)
    est.observe_attempt(0, 1, 1, False)
    assert est.p_success(0, 10) == pytest.approx(8 / 11)


# ---------------------------------------------------------------- QUBO with replica counts
def test_qubo_replica_count_constraint():
    rng = np.random.default_rng(0)
    n, m = 3, 4
    c = rng.uniform(0.5, 2.0, size=(n, m))
    q = np.zeros((n, n, m))
    for i in range(n):
        for k in range(i + 1, n):
            q[i, k] = rng.uniform(0.1, 1.0, size=m)
    p = AssignmentProblem(list(range(n)), list(range(m)), c, q, k=[2, 1, 2])
    bqm = to_bqm(p)
    for combo in itertools.product(*[itertools.combinations(range(m), k) for k in p.k]):
        sample = {(i, j): int(j in combo[i]) for i in range(n) for j in range(m)}
        assert bqm.energy(sample) == pytest.approx(p.energy(combo))
    ground, viol = decode(p, dimod.ExactSolver().sample(bqm).first.sample)
    assert viol == 0 and p.feasible(ground)
    assert p.energy(ground) == pytest.approx(p.energy(brute_force(p)))
    g = greedy(p)
    assert p.feasible(g) and all(len(set(s)) == k for s, k in zip(g, p.k))


# ---------------------------------------------------------------- JSON workflows
def test_demo_json_workflow_loads():
    wf = load_workflow_json("workflows/demo_dag.json")
    assert len(wf) == 10 and wf.deadline == 70
    assert wf.tasks[wf.entry_tasks[0]].name == "ingest"
    assert {t.type for t in wf.tasks} <= {"sleep", "cpu_benchmark", "hash_payload", "fail_once"}


@pytest.mark.parametrize("tasks,match", [
    ([{"id": "a", "work": 1}, {"id": "a", "work": 1}], "duplicate"),
    ([{"id": "a", "work": 1, "parents": ["x"]}], "unknown task"),
    ([{"id": "a", "work": 1, "parents": ["b"]}, {"id": "b", "work": 1, "parents": ["a"]}], "cycle"),
    ([{"id": "a", "work": 1, "type": "rm -rf /"}], "not allowed"),
    ([{"id": "a", "work": 0}], "positive"),
    ([{"id": "a", "work": 1, "parents": ["a"]}], "itself"),
])
def test_json_workflow_validation(tasks, match):
    with pytest.raises(ValueError, match=match):
        load_workflow_json({"tasks": tasks})


# ---------------------------------------------------------------- real execution (HTTP agents)
@pytest.fixture
def agents():
    from rdas.workers.agent import WorkerAgent, serve

    specs = [("worker-1", 1.0, 0.0), ("worker-2", 1.5, 0.0), ("worker-3", 0.8, 0.0)]
    servers, urls = [], []
    for wid, speed, fr in specs:
        srv = serve(WorkerAgent(wid, speed, fr, enable_admin=True), "127.0.0.1", 0)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        servers.append(srv)
        urls.append(f"http://127.0.0.1:{srv.server_address[1]}")
    yield urls
    for srv in servers:
        srv.shutdown()
        srv.server_close()


def test_agent_endpoints(agents):
    from rdas.backends.http import _request

    code, caps = _request(agents[0] + "/capabilities")
    assert code == 200 and caps["worker_id"] == "worker-1" and "fail_once" in caps["task_types"]
    code, err = _request(agents[0] + "/execute", {"task_id": "x", "attempt_id": "1", "task_type": "shell",
                                                  "work": 1})
    assert code == 400  # not on the allowlist
    code, rec = _request(agents[0] + "/execute", {"task_id": "x", "attempt_id": "2", "task_type": "sleep",
                                                  "work": 1, "seconds_per_unit": 0.01})
    assert code == 200 and rec["status"] == "succeeded" and rec["started_at"].endswith("+00:00")
    code, rec = _request(agents[0] + "/execute", {"task_id": "y", "attempt_id": "3", "task_type": "hash_payload",
                                                  "work": 1, "seconds_per_unit": 0.01,
                                                  "payload": {"data": "abc", "rounds": "lots"}})
    assert rec["status"] == "failed" and rec["category"] == "task_error"
    _request(agents[0] + "/admin/fail", {"duration_s": 0.3})
    assert _request(agents[0] + "/health")[0] == 503


def test_live_run_recovers_from_injected_failure_and_crash(agents, tmp_path):
    from rdas.live import run_live

    spec = {"name": "t", "tasks": [
        {"id": "a", "type": "sleep", "work": 4},
        {"id": "b", "type": "fail_once", "work": 4, "parents": ["a"]},
        {"id": "c", "type": "cpu_benchmark", "work": 6, "parents": ["a"]},
        {"id": "d", "type": "sleep", "work": 3, "parents": ["b", "c"]}]}
    path = tmp_path / "wf.json"
    path.write_text(json.dumps(spec))
    # Crash worker index 1 (the fast one) shortly after the start.
    res, eng = run_live(str(path), agents, "adaptive", seconds_per_unit=0.03, chaos=[(1, 1.0, 6.0)],
                        out_dir=str(tmp_path / "out"), quiet=True)
    assert res.completed
    assert res.failures_by_category["injected"] >= 1  # fail_once
    recs = eng.attempt_records()
    done = {r["task"]: r for r in recs if r["outcome"] == "success"}
    assert done["d"]["start"] >= max(done["b"]["end"], done["c"]["end"]) - 1e-6  # parents first
    assert (tmp_path / "out" / "events.json").exists() and (tmp_path / "out" / "metrics.json").exists()
