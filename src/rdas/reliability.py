"""Online reliability and performance estimation (the feedback signal of the adaptive loop).

The engine's fault & resource monitor feeds every estimator the same
observations: worker uptime, crashes, recoveries and task-attempt outcomes.
Baseline schedulers simply ignore the estimates.
"""

from __future__ import annotations

import math


class _EstimatorBase:
    def __init__(self, n_workers: int, perf_alpha: float = 0.3, prior_mttr: float = 15.0):
        self.n_workers = n_workers
        self.perf_alpha = perf_alpha
        self._perf = [1.0] * n_workers  # EMA of actual / expected duration
        self._mttr = [prior_mttr] * n_workers
        self.crashes = [0] * n_workers
        self.successes = [0] * n_workers
        self.failures = [0] * n_workers

    def observe_uptime(self, worker: int, dt: float) -> None:
        pass

    def observe_crash(self, worker: int) -> None:
        self.crashes[worker] += 1

    def observe_failure(self, worker: int) -> None:
        """A worker-related failure while the worker stayed up (injected fault, timeout,
        dispatch error). Counted as a failure event, like a crash."""
        self.observe_crash(worker)

    def observe_recovery(self, worker: int, downtime: float) -> None:
        self._mttr[worker] = 0.5 * self._mttr[worker] + 0.5 * downtime

    def observe_attempt(self, worker: int, expected: float, actual: float, success: bool) -> None:
        if success:
            self.successes[worker] += 1
            if expected > 0:
                ratio = actual / expected
                a = self.perf_alpha
                self._perf[worker] = a * ratio + (1 - a) * self._perf[worker]
        else:
            self.failures[worker] += 1

    def perf_ratio(self, worker: int) -> float:
        """Observed slowdown multiplier: 1.0 means the worker runs at nominal speed."""
        return self._perf[worker]

    def expected_downtime(self, worker: int) -> float:
        return self._mttr[worker]

    def p_success(self, worker: int, duration: float) -> float:
        raise NotImplementedError

    def reliability(self, worker: int, ref_duration: float) -> float:
        """Scalar R_j in [0, 1] for display: P(success) for a reference-length task."""
        return self.p_success(worker, ref_duration)

    def snapshot(self, ref_duration: float) -> list[dict]:
        return [
            {"worker": j, "R": self.reliability(j, ref_duration), "perf": self._perf[j],
             "crashes": self.crashes[j]}
            for j in range(self.n_workers)
        ]


class GammaPoissonEstimator(_EstimatorBase):
    """Bayesian crash-rate estimate with exponential forgetting.

    Crashes are modelled as a Poisson process with unknown rate lambda_j, with a
    conjugate Gamma(a0, b0) prior: a0 / b0 = ``prior_rate`` and b0 =
    ``prior_exposure`` (how many time units of evidence the prior is worth).
    The posterior mean is

        lambda_hat_j = (a0 + n_j) / (b0 + E_j)

    where n_j and E_j are crash count and up-time exposure, both discounted with
    half-life ``half_life`` so the estimate tracks workers whose reliability
    changes. A task of duration d then succeeds with probability exp(-lambda d).
    ``kappa`` > 0 adds kappa posterior standard deviations to the rate, making
    the scheduler cautious about workers with little history.

    With ``pooled=True`` (empirical Bayes) the per-worker prior mean is not the
    fixed ``prior_rate`` but the fleet-wide estimate (all crashes / all exposure,
    itself shrunk toward ``prior_rate``). Each worker's estimate starts at what
    the fleet is actually doing and departs from it as its own evidence builds.
    """

    name = "gamma-poisson"

    def __init__(self, n_workers: int, prior_rate: float, prior_exposure: float = 100.0,
                 half_life: float = 300.0, kappa: float = 0.0, pooled: bool = True, **kw):
        super().__init__(n_workers, **kw)
        self.prior_rate = max(prior_rate, 1e-6)
        self.b0 = prior_exposure
        self.half_life = half_life
        self.kappa = kappa
        self.pooled = pooled
        self._n = [0.0] * n_workers
        self._e = [0.0] * n_workers

    @property
    def a0(self) -> float:
        return self.fleet_rate() * self.b0

    def fleet_rate(self) -> float:
        if not self.pooled:
            return self.prior_rate
        return (self.prior_rate * self.b0 + sum(self._n)) / (self.b0 + sum(self._e))

    def observe_uptime(self, worker: int, dt: float) -> None:
        if dt <= 0:
            return
        decay = 0.5 ** (dt / self.half_life) if self.half_life else 1.0
        self._n[worker] *= decay
        # Exposure accrued uniformly over dt, discounted continuously.
        if self.half_life:
            k = math.log(2) / self.half_life
            self._e[worker] = self._e[worker] * decay + (1 - decay) / k
        else:
            self._e[worker] += dt

    def observe_crash(self, worker: int) -> None:
        super().observe_crash(worker)
        self._n[worker] += 1.0

    def failure_rate(self, worker: int) -> float:
        a = self.a0 + self._n[worker]
        b = self.b0 + self._e[worker]
        return a / b + self.kappa * math.sqrt(a) / b

    def p_success(self, worker: int, duration: float) -> float:
        return math.exp(-self.failure_rate(worker) * max(duration, 0.0))

    def snapshot(self, ref_duration: float) -> list[dict]:
        snap = super().snapshot(ref_duration)
        for row in snap:
            row["rate"] = self.failure_rate(row["worker"])
        return snap


class EMAEstimator(_EstimatorBase):
    """Exponential moving average of attempt outcomes: R <- alpha*outcome + (1-alpha)*R.

    Simpler and duration-agnostic. Kept as the ablation and as the
    "first estimator" explained in Review 1.
    """

    name = "ema"

    def __init__(self, n_workers: int, alpha: float = 0.2, prior: float = 0.9, **kw):
        super().__init__(n_workers, **kw)
        self.alpha = alpha
        self._r = [prior] * n_workers

    def observe_attempt(self, worker: int, expected: float, actual: float, success: bool) -> None:
        super().observe_attempt(worker, expected, actual, success)
        self._r[worker] = self.alpha * (1.0 if success else 0.0) + (1 - self.alpha) * self._r[worker]

    def p_success(self, worker: int, duration: float) -> float:
        return self._r[worker]


class BetaBernoulliEstimator(_EstimatorBase):
    """Beta-Bernoulli posterior mean of attempt success (Anirudh's baseline estimator):

        R_w = (a + s_w) / (a + b + s_w + f_w)

    with prior a = 8, b = 2 (initial estimate 0.8). Duration-agnostic: a long
    task is assumed to be as safe as a short one. Only worker-related failures
    reach ``observe_attempt``, because the engine filters out task errors.
    """

    name = "beta-bernoulli"

    def __init__(self, n_workers: int, a: float = 8.0, b: float = 2.0, **kw):
        super().__init__(n_workers, **kw)
        self.a = a
        self.b = b

    def p_success(self, worker: int, duration: float) -> float:
        s, f = self.successes[worker], self.failures[worker]
        return (self.a + s) / (self.a + self.b + s + f)


ESTIMATORS = {"gamma-poisson": GammaPoissonEstimator, "ema": EMAEstimator, "beta-bernoulli": BetaBernoulliEstimator}
