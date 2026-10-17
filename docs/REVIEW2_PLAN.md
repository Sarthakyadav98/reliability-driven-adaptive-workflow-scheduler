# Review 2: remaining work and ownership

The working prototype is in place: simulator, 5 baselines, adaptive scheduler,
QUBO backend, experiment harness and demo. What's left before the review is
**understanding, verification and write-up**, plus a few targeted extensions.
Each person owns an area, but **everyone must be able to explain the whole demo**.

## Everyone (first 1–2 days)

- [ ] Set up the repo (`pip install -e ".[dev]"`, `pytest -q`) and run `python scripts/demo.py`.
- [ ] Read `docs/DESIGN.md` end to end. Each person explains one section to the others.
- [ ] Agree on the final numbers to present (from `results/review2/`, held-out seeds only).

## Sarthak: simulator + integration + demo owner

- [ ] Rehearse the demo (`scripts/demo.py`) until it runs in under 6 minutes with narration.
- [ ] Load 1–2 real WfCommons/Pegasus workflow shapes (Montage, Epigenomics) into `rdas/workflow.py` and rerun the sweep on them. This answers the "only synthetic DAGs?" question.
- [ ] Report chapters: System Architecture, Implementation.

## Anirudh: fault model + baselines + Docker groundwork

- [ ] Verify the HEFT implementation against the paper's Fig. 2 pseudocode (`schedulers/heft.py`), and add the paper's 10-task example as a test (needs communication costs set to zero, or add comm costs).
- [ ] From the literature matrix, confirm the most common static fault-tolerance policy. If it is not uniform 2-replication, add it as a baseline.
- [ ] Start the Docker execution backend spike: one container per worker, a task = CPU-bound sleep/compute, kill via `docker kill`. Target: the same `Decision` interface drives containers (post-Review 2 V3).
- [ ] Report chapters: Fault model, Experimental setup.

## Sreyas + Sahithi: QUBO track

- [ ] Penalty-weight study: SA quality drops as size grows (see `results/qubo/`). Try a smaller `A` (just above the bound), more sweeps, and the `dwave-samplers` beta schedule options. Report the gap/time curves.
- [ ] Measure the QUBO variable/qubit count per decision across a sweep (already logged as `mean_solver_vars`) and compare with the whole-workflow formulation size in Pérez Armas et al. (2024). This is the decision-scoped novelty argument, with numbers.
- [ ] Optional: try `dwave-samplers` `TabuSampler` / `SteepestDescentSolver` as extra classical baselines for the same QUBO.
- [ ] Report chapters: QUBO formulation, RQ2 results.

## Literature (all, split by topic)

- [ ] Read the 4 recent papers flagged after Review 1 (Ren & Yao TSC 2026; RWSUT JKSU-CIS 2026; CCPE 2026 adaptive FT; EPJ QT 2026 quantum scheduling survey). Verify each exists before citing.
- [ ] Add rows to the literature matrix. Check whether any of them already does online reliability estimation + adaptive choice of retry/replicate/reassign, and adjust the novelty claim if needed.

## Report skeleton (owner: whoever finishes their part first)

Abstract · Introduction · Problem & objectives · Literature review (verified) ·
Architecture · Methodology (`DESIGN.md` §2–7) · Implementation · Experimental
setup (§4, §9) · Preliminary results (`RESULTS.md`) · Limitations & next steps
(`DESIGN.md` §10–11) · References (IEEE).

Clearly separate **implemented / tested / planned**.
