# Review 2: remaining work and ownership (updated 2026-10-20)

**Built:** simulator and real-execution engine (Docker / local agents), 12
scheduler variants including Anirudh's threshold policy and Reliability-Aware
HEFT, three estimators, the k-replica QUBO with feasibility reporting, held-out
evaluation with paired statistics, a sensitivity study and a penalty sweep. See
`docs/AUDIT.md` for status against the team spec.

What's left before the review is **understanding, a few targeted
extensions and the write-up**. Each person owns an area, but **everyone must
be able to explain the whole demo**.

## Everyone (first day)

- [ ] `pip install -e ".[dev]"`, `pytest -q` (76 tests), then `python scripts/demo.py`.
- [ ] Run the real-execution demo once: `python scripts/local_cluster.py` + `python -m rdas.live --compare eft-retry,adaptive --chaos 1@8:10`.
- [ ] Read `docs/DESIGN.md` and `docs/RESULTS.md`. Each person explains one section to the others.
- [ ] Agree on the headline claims. Use only the held-out numbers in `RESULTS.md`, including the negative ones (degrading workers: replicate-2 completes more workflows; the reliability term alone has no effect).

## Sarthak: integration, demo, report lead

- [ ] Rehearse the demo: simulation story (`scripts/demo.py`) + live Docker run, about 7 minutes total.
- [ ] Load 1–2 real workflow shapes (WfCommons Montage / Epigenomics) as JSON and rerun the sweep.
- [ ] Report chapters: Architecture, Implementation, Results.

## Anirudh: real execution + fault tolerance (his spec)

- [ ] Review `docs/AUDIT.md` against your README and flag anything misread.
- [ ] Repeated Docker runs: 5–10 per scheduler with the same chaos schedule. Record host specs and container limits, and add the table to RESULTS §6 (open item 1).
- [ ] Forward task outputs to successors in the live backend (data dependencies, open item 2).
- [ ] Report chapters: Fault model, Fault-tolerance policies, Docker testbed.

## Sreyas + Sahithi: QUBO track

- [ ] Tune `penalty_scale` on **dev seeds 0–9** (held-out results suggest 0.5× beats 1×, but they cannot be used for tuning). Then rerun `adaptive-qubo` on the held-out seeds.
- [ ] Extend the penalty sweep to larger instances, with SA schedule options (beta range, sweeps).
- [ ] Compare decision-scoped QUBO size (`mean_solver_vars` ≈ 17) with the whole-workflow formulation size in Pérez Armas et al. (2024): the decision-scoped novelty argument, with numbers.
- [ ] Report chapters: QUBO formulation, RQ2 results.

## Literature (all)

- [ ] Verify IEEE arnumber 11264338 and the 2026 papers before citing (metadata still unconfirmed).
- [ ] Literature matrix rows for RWSUT, the energy/reliability paper and the QUBO HPC-mapping paper. Does any of them already combine online reliability estimation with adaptive retry / replicate / reassign? Adjust the novelty claim if so.

## Report skeleton

Abstract · Introduction · Problem & objectives · Literature review (verified) ·
Architecture (DESIGN §1–3) · Methodology (DESIGN §4–8) · Implementation
(simulation + real execution) · Experimental setup (RESULTS header, DESIGN §10) ·
Results (RESULTS §1–6) · Limitations & next steps (DESIGN §11, AUDIT §12) ·
References (IEEE).

Clearly separate **implemented / tested / planned**.
