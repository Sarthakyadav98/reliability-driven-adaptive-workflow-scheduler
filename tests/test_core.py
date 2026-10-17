import math

import numpy as np
import pytest

from rdas import EMAEstimator, FaultConfig, FaultTrace, GammaPoissonEstimator, ScriptedCrash, Simulation, WorkerSpec
from rdas.models import Workflow
from rdas.schedulers import REGISTRY, make_scheduler
from rdas.schedulers.heft import _earliest_slot, heft_plan
from rdas.workers import demo_pool, make_pool
from rdas.workflow import criticality, demo_workflow, fork_join, layered_dag, upward_rank


# ---------------------------------------------------------------- workflows
def test_layered_dag_is_acyclic_and_connected_downward():
    wf = layered_dag(80, np.random.default_rng(0))
    assert len(wf) == 80
    pos = {t: i for i, t in enumerate(wf.topo_order)}
    for t in wf.tasks:
        for s in t.succs:
            assert pos[t.id] < pos[s]
    # every non-entry task has a parent
    assert all(t.preds for t in wf.tasks if t.id not in wf.entry_tasks)


def test_cycle_rejected():
    with pytest.raises(ValueError):
        Workflow.from_edges([1, 1], [(0, 1), (1, 0)])


def test_upward_rank_on_chain():
    wf = Workflow.from_edges([1, 2, 3], [(0, 1), (1, 2)])
    assert upward_rank(wf, [1, 2, 3]) == [6, 5, 3]
    assert criticality(wf, [1, 2, 3]) == [1.0, 1.0, 1.0]


def test_fork_join_shape():
    wf = fork_join(3, 2, np.random.default_rng(0))
    assert len(wf) == 2 * (3 + 2)
    assert len(wf.entry_tasks) == 1 and len(wf.exit_tasks) == 1


# ---------------------------------------------------------------- HEFT
def test_insertion_slot():
    busy = [(0, 2), (5, 8)]
    assert _earliest_slot(busy, 0, 3) == 2  # fits in gap 2-5
    assert _earliest_slot(busy, 0, 4) == 8  # does not fit, append
    assert _earliest_slot([], 1.5, 1) == 1.5


def test_heft_plan_valid():
    wf = layered_dag(40, np.random.default_rng(3))
    pool = make_pool(4, np.random.default_rng(4))
    plan = heft_plan(wf, pool)
    for t in wf.tasks:
        dur = t.work / pool[plan.proc[t.id]].speed
        assert math.isclose(plan.finish[t.id] - plan.start[t.id], dur)
        for p in t.preds:
            assert plan.start[t.id] >= plan.finish[p] - 1e-9
    for j in range(len(pool)):
        iv = sorted((plan.start[t], plan.finish[t]) for t in range(len(wf)) if plan.proc[t] == j)
        for (s1, e1), (s2, e2) in zip(iv, iv[1:]):
            assert s2 >= e1 - 1e-9


def test_heft_chain_uses_fastest_worker():
    wf = Workflow.from_edges([10, 10, 10], [(0, 1), (1, 2)])
    pool = [WorkerSpec(0, 1.0), WorkerSpec(1, 2.0)]
    plan = heft_plan(wf, pool)
    assert plan.proc == [1, 1, 1]
    assert math.isclose(plan.makespan, 15.0)


# ---------------------------------------------------------------- faults
def test_fault_trace_deterministic_and_rate():
    pool = [WorkerSpec(0, 1.0, failure_factor=1.0)]
    cfg = FaultConfig(task_failure_prob=0.2, mttr=1.0)
    a = FaultTrace(pool, cfg, ref_duration=10.0, seed=5)
    b = FaultTrace(pool, cfg, ref_duration=10.0, seed=5)
    assert [a.crash_event(0, k) for k in range(20)] == [b.crash_event(0, k) for k in range(20)]
    # Empirical crash rate over up-time matches lambda = -ln(0.8)/10
    events = [a.crash_event(0, k) for k in range(3000)]
    up = events[0][0] + sum(c - prev_r for (_, prev_r), (c, _) in zip(events, events[1:]))
    assert abs(len(events) / up - (-math.log(0.8) / 10)) / (-math.log(0.8) / 10) < 0.08


def test_scripted_crash_and_degradation():
    pool = [WorkerSpec(0, 1.0)]
    cfg = FaultConfig(task_failure_prob=0.0, scripted_crashes=[ScriptedCrash(0, at=5.0, downtime=2.0)])
    tr = FaultTrace(pool, cfg, 10.0, seed=0)
    assert tr.crash_event(0, 0) == (5.0, 7.0)
    assert math.isinf(tr.crash_event(0, 1)[0])


def test_slowdown_integration():
    pool = [WorkerSpec(0, 1.0)]
    tr = FaultTrace(pool, FaultConfig(task_failure_prob=0.0, slowdown_rate=1e-9), 10.0, seed=0)
    tr._slow[0] = [(2.0, 4.0, 2.0)]  # 2x slower between t=2 and t=4
    tr._slow_cursor[0] = 1e9
    # 5 units of work from t=0: 2 units by t=2, 1 unit during [2,4], remaining 2 by t=6
    assert math.isclose(tr.finish_time(0, 0.0, 5.0), 6.0)
    assert math.isclose(tr.finish_time(0, 5.0, 1.0), 6.0)


# ---------------------------------------------------------------- reliability
def test_ema_matches_review1_example():
    est = EMAEstimator(1, alpha=0.2, prior=0.9)
    est.observe_attempt(0, 1, 1, success=False)
    assert math.isclose(est.p_success(0, 1), 0.72)
    est.observe_attempt(0, 1, 1, success=True)
    assert math.isclose(est.p_success(0, 1), 0.776)


def test_gamma_poisson_reacts_and_recovers():
    est = GammaPoissonEstimator(2, prior_rate=0.01, prior_exposure=50, half_life=100, pooled=False)
    r0 = est.p_success(0, 10)
    est.observe_crash(0)
    assert est.p_success(0, 10) < r0  # crash lowers reliability
    assert est.p_success(1, 10) == r0  # other worker unaffected
    low = est.p_success(0, 10)
    est.observe_uptime(0, 500)
    assert est.p_success(0, 10) > low  # clean uptime restores it


# ---------------------------------------------------------------- engine
def _check_run_invariants(sim):
    done = {a.task: a for a in sim.attempts.values() if a.outcome == "success"}
    assert len(done) == len(sim.workflow)
    for t in sim.workflow.tasks:
        for p in t.preds:
            assert done[t.id].start >= done[p].end - 1e-9, "precedence violated"
    for w in range(len(sim.workers)):
        iv = sorted((a.start, a.end) for a in sim.attempts.values() if a.worker == w and a.start is not None)
        for (s1, e1), (s2, e2) in zip(iv, iv[1:]):
            assert s2 >= e1 - 1e-9, "worker ran two attempts at once"
    for w, s, e in sim.down_intervals:
        for a in sim.attempts.values():
            if a.worker == w and a.start is not None and a.outcome == "success":
                assert a.end <= s + 1e-9 or a.start >= e - 1e-9, "task succeeded across a crash"


@pytest.mark.parametrize("name", [n for n in REGISTRY if n != "adaptive-qubo"])
@pytest.mark.parametrize("p", [0.0, 0.2])
def test_every_scheduler_completes_correctly(name, p):
    wf = layered_dag(40, np.random.default_rng(1))
    pool = make_pool(5, np.random.default_rng(2))
    sim = Simulation(wf, pool, make_scheduler(name), FaultConfig(task_failure_prob=p), seed=3)
    res = sim.run()
    assert res.completed
    _check_run_invariants(sim)
    if p == 0.0:
        assert res.n_failed_attempts == 0


def test_runs_are_deterministic():
    wf = layered_dag(30, np.random.default_rng(1))
    pool = make_pool(4, np.random.default_rng(2))
    r1 = Simulation(wf, pool, make_scheduler("adaptive"), FaultConfig(task_failure_prob=0.2), seed=9).run()
    r2 = Simulation(wf, pool, make_scheduler("adaptive"), FaultConfig(task_failure_prob=0.2), seed=9).run()
    timing = {"decision_time_total", "mean_decision_ms", "solver_time", "mean_solver_ms"}
    row1 = {k: v for k, v in r1.as_row().items() if k not in timing}
    row2 = {k: v for k, v in r2.as_row().items() if k not in timing}
    assert row1 == row2


def test_crash_causes_retry_and_heft_waits_for_planned_worker():
    wf = demo_workflow()
    pool = demo_pool()
    plan = heft_plan(wf, pool)
    t0 = 0
    w = plan.proc[t0]
    faults = FaultConfig(task_failure_prob=0.0, scripted_crashes=[ScriptedCrash(w, at=1.0, downtime=30.0)])
    sim = Simulation(wf, pool, make_scheduler("heft"), faults, seed=0)
    res = sim.run()
    assert res.completed and res.n_failed_attempts == 1
    retry = [a for a in sim.attempts.values() if a.task == t0 and a.outcome == "success"][0]
    assert retry.worker == w and retry.start >= 31.0  # static HEFT waited for recovery

    sim2 = Simulation(wf, pool, make_scheduler("eft-retry"), faults, seed=0)
    res2 = sim2.run()
    assert res2.completed and res2.makespan < res.makespan  # retry elsewhere is faster


def test_replicas_are_cancelled_when_one_copy_wins():
    wf = demo_workflow()
    pool = demo_pool()
    sim = Simulation(wf, pool, make_scheduler("replicate-2"), FaultConfig.fault_free(), seed=0)
    res = sim.run()
    assert res.completed
    assert res.n_replicas == len(wf)
    for t in range(len(wf)):
        wins = [a for a in sim.attempts.values() if a.task == t and a.outcome == "success"]
        assert len(wins) == 1


def test_adaptive_learns_risk_aware_routing():
    """W1 is the fastest worker but crashes every 15 time units. Once the estimator has
    learned that, the adaptive scheduler keeps W1 for *short* tasks (expected lost work
    grows with duration) and finishes sooner than the reliability-blind eft-retry."""
    pool = [WorkerSpec(0, 1.0), WorkerSpec(1, 1.6), WorkerSpec(2, 1.0)]
    crashes = [ScriptedCrash(1, at=t, downtime=3.0) for t in np.arange(5, 600, 15)]
    faults = FaultConfig(task_failure_prob=0.0, scripted_crashes=crashes)

    def run(name, wf):
        sim = Simulation(wf, pool, make_scheduler(name), faults, seed=0)
        res = sim.run()
        late = [a for a in sim.attempts.values() if a.start is not None and a.start > 100 and a.worker == 1]
        return res.makespan, np.mean([wf.tasks[a.task].work for a in late])

    total_a = total_b = 0.0
    for seed in range(4):
        wf = layered_dag(60, np.random.default_rng(seed))
        mk_a, work_a = run("adaptive", wf)
        mk_b, work_b = run("eft-retry", wf)
        total_a += mk_a
        total_b += mk_b
        assert work_a < work_b  # shorter tasks go to the flaky worker
    assert total_a < total_b
