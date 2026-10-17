import itertools

import dimod
import numpy as np
import pytest

from rdas import FaultConfig, Simulation
from rdas.qubo import AssignmentProblem, QuboSolver, brute_force, decode, greedy, to_bqm
from rdas.schedulers import make_scheduler
from rdas.workers import make_pool
from rdas.workflow import layered_dag


def random_problem(n, m, seed):
    rng = np.random.default_rng(seed)
    c = rng.uniform(0.5, 3.0, size=(n, m))
    q = np.zeros((n, n, m))
    for i in range(n):
        for k in range(i + 1, n):
            q[i, k] = rng.uniform(0.2, 1.5, size=m)
    return AssignmentProblem(list(range(n)), list(range(m)), c, q)


@pytest.mark.parametrize("seed", range(5))
def test_bqm_energy_equals_objective_on_feasible_assignments(seed):
    p = random_problem(3, 3, seed)
    bqm = to_bqm(p)
    for assign in itertools.product(range(p.m), repeat=p.n):
        sample = {(i, j): int(assign[i] == j) for i in range(p.n) for j in range(p.m)}
        assert bqm.energy(sample) == pytest.approx(p.energy(assign))


@pytest.mark.parametrize("seed", range(5))
def test_exact_bqm_minimum_is_feasible_and_optimal(seed):
    p = random_problem(3, 3, seed)  # 9 binary variables -> 512 states
    ss = dimod.ExactSolver().sample(to_bqm(p))
    assign, violations = decode(p, ss.first.sample)
    assert violations == 0  # penalty weight large enough: ground state is one-hot
    assert p.energy(assign) == pytest.approx(p.energy(brute_force(p)))


@pytest.mark.parametrize("seed", range(5))
def test_simulated_annealing_finds_optimum_on_small_instances(seed):
    p = random_problem(4, 3, seed)
    assign, info = QuboSolver(num_reads=50, num_sweeps=1000, seed=seed).solve(p)
    assert info.violations == 0
    assert p.energy(assign) == pytest.approx(p.energy(brute_force(p)), rel=1e-9)


def test_greedy_is_feasible_and_never_better_than_exact():
    for seed in range(10):
        p = random_problem(4, 3, seed)
        g = greedy(p)
        assert len(g) == p.n and all(0 <= j < p.m for j in g)
        assert p.energy(g) >= p.energy(brute_force(p)) - 1e-9


def test_qubo_backed_adaptive_scheduler_completes():
    wf = layered_dag(25, np.random.default_rng(1))
    pool = make_pool(4, np.random.default_rng(2))
    res = Simulation(wf, pool, make_scheduler("adaptive-qubo", qubo_reads=5, qubo_sweeps=200),
                     FaultConfig(task_failure_prob=0.2), seed=3).run()
    assert res.completed
    assert res.extra["n_solver_calls"] > 0
    assert res.extra["qubo_violations"] == 0
