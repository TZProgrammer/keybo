# Learnings

Methodology lessons worth keeping, written for a future reader who has the
PREREGISTRATIONS.md trail but not the context. One section per learning.

## Augmented Chebyshev scalarization (P17, charter 40cf881)

**Problem it solves.** A multi-objective search needs a single number to
optimize. The obvious choice — a weighted sum `Σ w_g·n_g` over normalized
objectives — has a known blind spot: for any weight vector, the optimum of a
weighted sum is a point where a hyperplane supports the Pareto front. Points in
*concave* regions of the front (the "dents") are never the optimum of ANY
weighted sum, no matter the weights. If your final pick rule is min-max ("best
worst-axis"), its optimum often lives exactly in such a dent — balanced
solutions are the ones that trade a little everywhere, which is what a dent is.
Through P16 we searched weighted sums but picked by min-max: an
objective/pick mismatch that made the searcher structurally unable to find what
the picker wanted.

**The fix.** Chebyshev (weighted L∞) scalarization minimizes the *worst*
weighted deviation from an ideal point instead of the sum:

    fit(x) = max_g [ w_g · n_g(x) ]        n_g(x) = (loss_g(x) − BEST_g) / (QREF_g − BEST_g)

where `BEST_g` is the per-gauge ideal (best value seen on a reference pool) and
`QREF_g` a fixed bad reference (qwerty), so `n_g` ≈ "fraction of the
qwerty-gap you're giving up." Unlike the weighted sum, every Pareto-optimal
point — including dents — is the minimizer of the Chebyshev objective for some
weight vector. Sweeping weights therefore sweeps the *whole* front.

**Why "augmented."** Pure max has two practical defects. (1) It can return
*weakly* Pareto-optimal points: if one gauge is the max, the optimizer is
indifferent to free improvements in the others. (2) It's flat almost
everywhere — a local move that improves a non-worst gauge changes nothing, so
SA gets no gradient signal until the worst axis itself moves. The augmentation
adds a small L1 term:

    fit(x) = max_g [ w_g · n_g(x) ] + ρ · Σ_g [ w_g · n_g(x) ]      (P17: ρ = 0.05)

The sum term breaks ties toward strict Pareto improvements and gives the
annealer a slope everywhere, while ρ small keeps the min-max character (ρ too
large degrades back to a weighted sum). ρ = 0.05 was registered before results.

**How P17 used it.** 44 arms with weights drawn from Dirichlet(1,1,1,1) (front
sweep) + 6 equal-weight arms (aim at the pick rule itself) + 4 warm starts,
each SA 12×16k + exhaustive 2-opt; then the top-10 pool rows got an exhaustive
2-opt + 3-cycle polish on the PURE equal-weight min-max (the actual pick rule,
ρ = 0 — safe for local search because at a near-optimum the ties (1) and
flatness (2) matter less than fidelity to the true criterion).

**Evidence it mattered.** The winner (keybo-c30m) improved same-pool max-regret
by 3.79pp over the best weighted-sum-era candidate; no raw weighted-sum arm in
any campaign (P14–P16, dozens of arms) had found it, and even raw Chebyshev
arms needed the min-max polish to land on it. The convex-hull gap was real,
measurable, and worth more than any surface or budget improvement we tried.

**Reusable rule of thumb.** If you select by criterion C, search with C (or a
smoothed version of it). Any mismatch between search objective and selection
criterion is a standing invitation for the searcher to systematically miss the
selector's optimum.

## SEL-1: audit the pick rule itself (charter 255cf03)

A selection rule is a modeling choice like any other and can be studied without
new data: run many defensible rules (min-max under several normalizations, L2
distance-to-ideal, mean regret, Borda, Copeland, random-preference win rate,
speed-first duals) on the same pool and perturb (leave-one-gauge-out, jackknife
the pool, drop a suspect gauge). What it caught here: (1) rules split into a
worst-axis camp and a consensus camp; (2) the consensus camp's champion rode a
hidden *double-count* — oxey1, oxey2 and wfd all price finger travel, giving
the travel cluster 3 of 5 votes; (3) min-max/L2/Copeland were the most
perturbation-stable (0.96). Reform adopted for P17+: drop wfd as a pick axis
(keep as report row), publish random-preference win share + Copeland as
non-binding diagnostics. Note: there is no "SEL-2" — SEL-1 is the only
selection-rule study to date; its reform is what P17 adopted.

## Matched charsets, or the comparison is the artifact (K31→K30M)

The K31 experiment "beat semimak on oxeylyzer-2" — until the charset was
matched, at which point the win reversed. The gap had come from a dof-encoding
convention (which character sits pinned on the quote slot), not from layout
quality. Cross-tool comparisons are only meaningful when every layout carries
the identical character set and every tool sees every key. Corollary caught the
same week: a gauge that can't see a key (genkey/keymeow are 3×10) silently
flatters any layout that parks its junk on the invisible slot.

## Pick numbers must reproduce from the run artifact (K30M erratum)

The K30M doc's speed column was ~0.2pp off because it was restated through a
different convention than the JSON it cited. Every number in a doc table should
be recomputable by one script from the named runs/*.json; if a convention
changes (time-saved vs speedup), restate the whole column and say so in an
erratum line rather than mixing conventions.
