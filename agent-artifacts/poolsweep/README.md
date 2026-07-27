# poolsweep — the interpolated-pool / spread sweep

Resolves the confound EVSCORE-1 named as its own decisive missing experiment: is the cross-source
transfer-ceiling collapse (rho(AALTO_BASE, COMMUNITY_BASE) 0.8350 -> 0.2654 from random to
near-optimal layouts) caused by NEAR-OPTIMALITY, or by reduced effective degrees of freedom?

**Answer: neither, as originally posed.** The operative variable is which DIRECTION of variation
the pool retains, not how much. Decomposing each source pair into consensus C = (zA+zB)/2 and
disagreement D = (zA-zB)/2, rho(A,B) is a monotone function of the ratio C/D (Spearman +0.9991
over 49 random-lineage cells). The Pareto archive restricts C by 10.9x while restricting D by only
3.7x -- so it is selection on the shared factor, which is what "optimize predicted time" does
mechanically. A pool of pure random permutations matched to the archive's C and D (no search, no
archive ancestry) reproduces the collapse: rho +0.1078 vs the archive's +0.2184. At matched C/D the
lineage residual is +0.0061 (Wilcoxon p = 0.4697) -- near-optimality adds nothing beyond its effect
on C/D.

Read `state/poolsweep/report.md` for the full argument. Every number is MODELLED ONLY (tau saturated
at 1.0, Phase-D cancelled) and lives on the `.native` surface frame, corpus named per row.

Run any phase with the worktree's own venv:

    PYTHONPATH=$PWD/agent-artifacts/poolsweep uv run --no-sync python \
        agent-artifacts/poolsweep/final.py --bank 200000 --n 400 --out /tmp/out.json
