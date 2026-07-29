"""Is either metric INFORMATIVE or REDUNDANT? (FT round, 2026-07-28)

Answers the four questions the brief asks, in the order the prereg registered them:

1. **Correlation** against the 15-gauge frame, predicted ms/char, and each other. The registered
   bar: R2 > 0.95 on the frame means "a restatement, do not add it as a gauge".
2. **Does it break ties the frame cannot see?** ``alt``/``imbalance`` are hand-partition
   invariants and ``sfr`` is a permutation invariant, so layouts related by a within-hand
   permutation tie on them BY CONSTRUCTION. A metric that separates such a pair adds an axis.
3. **The user's cost claim (C2)**: "pinky use is mostly fine as long as it stays home." Tested by
   regressing predicted ms/char on pinky-total vs pinky-off-home over a randomized layout pool,
   **with a frequency control**, because ``bad_scissor``'s +0.41 ms effect had bigram frequency
   explaining more variance than any geometric axis.
4. **Is it optimizable?** A cheap 1-swap greedy probe — NOT a full search.

Everything compared against comes from the SHIPPED ``keybo analyze --json`` path via
``ft_board.py``; nothing here re-derives a gauge.

Run: ``PYTHONPATH=src python agent-artifacts/ft_analysis.py <board.json> <out.json>``
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

GAUGES = (
    "sfr",
    "sfb",
    "sfs",
    "sfb-dist",
    "sfs-dist",
    "lsb",
    "lsb-dist",
    "alt",
    "roll",
    "sr-roll",
    "redir",
    "scissor",
    "imbalance",
    "oxey-style",
    "comfort",
)

#: Gauges that are invariant to a WITHIN-HAND permutation of the keys, so any two layouts related
#: by one tie on them by construction. `sfr` is invariant to ANY permutation of key identities
#: within a finger; `alt`/`imbalance` depend only on the hand partition.
TIE_INVARIANTS = ("alt", "imbalance", "sfr")


def pearson(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y) -> float:
    """Rank correlation — reported beside every Pearson r because this field has leverage points.

    ``qwerty``/``qwerty30m`` sit at travel_total ≈5.1e8 while every optimized board is 2.2–2.8e8,
    a 1.9x gap. Those two points ALONE produce a Pearson +0.82 between travel and ms/char; drop
    them and it is **−0.87**, and Spearman over all 18 is **−0.09**. A magnitude-sensitive
    statistic on this field is not trustworthy on its own.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    rank = lambda v: np.argsort(np.argsort(v)).astype(float)  # noqa: E731
    return float(np.corrcoef(rank(x), rank(y))[0, 1])


def leverage_audit(values, ms, drop_indices: list[int], label: str) -> dict:
    """Is a correlation's SIGN stable when the high-leverage layouts are removed?

    Added after this harness's own first pass reported travel_total ~ ms/char at r=+0.82 and the
    subset check flipped it to −0.87. A sign that flips under a two-point deletion is an artifact,
    and reporting only the full-field number would have shipped a wrong constant attached to an
    intuitive ("more travel is slower") and therefore unquestioned conclusion.
    """
    values, ms = np.asarray(values, float), np.asarray(ms, float)
    keep = [i for i in range(len(values)) if i not in set(drop_indices)]
    full, subset = pearson(values, ms), pearson(values[keep], ms[keep])
    return {
        "dropped": label,
        "pearson_full": full,
        "pearson_without_leverage": subset,
        "spearman_full": spearman(values, ms),
        "spearman_without_leverage": spearman(values[keep], ms[keep]),
        "sign_flips": bool(full * subset < 0),
        "verdict": (
            "SIGN FLIPS under leverage deletion — the full-field Pearson is an ARTIFACT and must "
            "not be quoted as a relationship"
            if full * subset < 0
            else "sign is stable under leverage deletion"
        ),
    }


def r2_on_frame(target, predictors: dict[str, list[float]]) -> dict:
    """OLS R2 of ``target`` on the given predictors + intercept, standardized.

    Reports ``n``/``k`` alongside because with 18 layouts and 15 predictors R2 is nearly
    guaranteed to be high — the ledger's own registered warning that the 19-gauge frame has an
    effective dof of only ~4-5. An adjusted R2 is therefore reported too, and a high raw R2 on
    k=15/n=18 is NOT evidence of redundancy on its own.
    """
    y = np.asarray(target, float)
    names = sorted(predictors)
    design = np.column_stack([np.asarray(predictors[n], float) for n in names])
    design = (design - design.mean(0)) / np.where(design.std(0) == 0, 1, design.std(0))
    design = np.column_stack([np.ones(len(y)), design])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ beta
    ss_res, ss_tot = float((residual**2).sum()), float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")
    n, k = len(y), len(names)
    adjusted = 1 - (1 - r2) * (n - 1) / (n - k - 1) if n - k - 1 > 0 else float("nan")
    return {"r2": r2, "adjusted_r2": adjusted, "n": n, "k": k, "dof_warning": bool(n - k - 1 <= 2)}


def single_best(target, predictors: dict[str, list[float]]) -> list[tuple[str, float]]:
    """Every predictor's |r| with the target, worst-to-best — the honest redundancy check.

    A high multi-predictor R2 on n=18 is nearly free; a single |r| near 1.0 is not.
    """
    scored = [(name, abs(pearson(values, target))) for name, values in predictors.items()]
    return sorted((s for s in scored if s[1] == s[1]), key=lambda kv: -kv[1])


def tie_breaking(board: dict) -> dict:
    """Find layout PAIRS the frame ties by construction, and ask whether the new metrics split.

    A tie is declared when two layouts agree on all of TIE_INVARIANTS to within 1e-9. Reported
    with the gap each new metric shows on that pair, so "it discriminates" is a number.
    """
    names = sorted(board)
    ties = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            ga, gb = board[a]["gauges"], board[b]["gauges"]
            if any(abs(ga[g] - gb[g]) > 1e-9 for g in TIE_INVARIANTS):
                continue
            ra, rb = board[a], board[b]
            ties.append(
                {
                    "pair": [a, b],
                    "tied_on": {g: ga[g] for g in TIE_INVARIANTS},
                    "travel_total_gap": rb["travel"]["total"] - ra["travel"]["total"],
                    "travel_total_gap_pct": (
                        100.0
                        * (rb["travel"]["total"] - ra["travel"]["total"])
                        / ra["travel"]["total"]
                    ),
                    "travel_max_share_gap": (
                        rb["travel"]["dispersion"]["max_share"]
                        - ra["travel"]["dispersion"]["max_share"]
                    ),
                    "travel_pinky_share_gap": (
                        rb["travel"]["dispersion"]["pinky_share"]
                        - ra["travel"]["dispersion"]["pinky_share"]
                    ),
                    "pinky_off_home_gap": (
                        rb["off_home"]["pinky"]["off_home"] - ra["off_home"]["pinky"]["off_home"]
                    ),
                    "ms_per_char_gap": (
                        (rb["time"]["ms_per_char"] - ra["time"]["ms_per_char"])
                        if ra["time"] and rb["time"]
                        else None
                    ),
                }
            )
    return {
        "invariants": list(TIE_INVARIANTS),
        "why": (
            "alt/imbalance depend only on the HAND PARTITION and sfr only on within-finger key "
            "identity, so a within-hand permutation cannot move them. A metric that separates "
            "such a pair is measuring something the frame cannot express."
        ),
        "pairs": ties,
    }


def cost_claim(pool: list[dict]) -> dict:
    """Test (C2): does predicted ms/char track pinky-TOTAL or pinky-OFF-HOME?

    ⚠ The registered caveat: ``bad_scissor``'s +0.41 ms effect had bigram FREQUENCY explaining
    more variance than any geometric axis. So the geometric predictors are always reported
    ALONGSIDE a frequency control (``sfb`` — same-finger bigram mass — is the frequency-structure
    proxy available on every row), and the honest question is what the geometric term adds ON TOP
    of it, not what it explains alone.

    ⚠⚠ **``n_pool`` IS NOT AN EVIDENCE COUNT, and every R2 below is R2 against a MODEL'S
    PREDICTIONS — not against measured time.** The pool is N layouts scored by the shipped k31
    surface, and that surface's generalization unit is **4 LAYOUTS** (verified on disk:
    ``bistrokes31_v1.tsv`` and ``tristrokes31_cond_v1.tsv`` each hold exactly
    ``{azerty, dvorak, qwerty, qwertz}``; **2202 and 16643 rows** — counted with ``awk END{NR}``,
    and note the FIRST LINE OF EACH FILE IS DATA, NOT A HEADER, so a ``wc -l`` minus one
    undercounts by exactly one). So 160 rows are 160 evaluations of one fitted function, not 160
    independent observations of typing cost, and ``n=160`` massively overstates the evidential
    weight of these numbers.

    **SCOPE TRAVELS WITH THE COUNT.** "n = 4" here means *distinct LAYOUTS in the Aalto/k31
    tables that the shipped time surface was fitted on* — not participants (there are ~55k PIDs on
    that side, which are not independent units for LAYOUT-level generalization), and not the
    separate community frame (which this module never touches). A bare "n = 4" is unusable; the
    scope is the load-bearing half.

    Direction of the consequence, stated so it cannot be read the flattering way: this **weakens
    any positive finding** here and **strengthens** the negative one. The verdict this function
    reaches is "C2 is unsupported on this evidence" — an insufficient-evidence claim — so a
    smaller true n reinforces it. Had the increments come out large, the same caveat would have
    barred calling them a measured effect.
    """
    ms = [row["ms_per_char"] for row in pool]
    total = [row["pinky_usage"] for row in pool]
    off = [row["pinky_off_home"] for row in pool]
    fraction = [row["pinky_off_fraction"] for row in pool]
    control = [row["sfb"] for row in pool]

    def incremental(named: dict[str, list[float]]) -> float:
        return r2_on_frame(ms, named)["r2"]

    base = incremental({"sfb": control})
    return {
        "n_pool": len(pool),
        "marginal_abs_r": {
            "pinky_usage_total": abs(pearson(total, ms)),
            "pinky_off_home": abs(pearson(off, ms)),
            "pinky_off_fraction": abs(pearson(fraction, ms)),
            "frequency_control_sfb": abs(pearson(control, ms)),
        },
        "r2": {
            "sfb_alone_CONTROL": base,
            "pinky_total_alone": incremental({"total": total}),
            "pinky_off_home_alone": incremental({"off": off}),
            "sfb_plus_pinky_total": incremental({"sfb": control, "total": total}),
            "sfb_plus_pinky_off_home": incremental({"sfb": control, "off": off}),
            "sfb_plus_both": incremental({"sfb": control, "total": total, "off": off}),
        },
        "increment_over_control": {
            "pinky_total": incremental({"sfb": control, "total": total}) - base,
            "pinky_off_home": incremental({"sfb": control, "off": off}) - base,
        },
        "reading": (
            "C2 is supported only if pinky_off_home adds materially MORE over the frequency "
            "control than pinky_total does. If both increments are small, the claim is "
            "UNSUPPORTED on this evidence — which is a complete answer."
        ),
        # The n that actually bounds these numbers, carried in the artifact so a later reader
        # cannot pick up `n_pool` as though it were an evidence count.
        "effective_generalization_unit": {
            # The SCOPE is part of the number: this is distinct LAYOUTS in the Aalto/k31 tables the
            # shipped surface was fitted on -- not participants, not the community frame.
            "scope": "distinct LAYOUTS in the Aalto/k31 tables the shipped time surface is fit on",
            "n": 4,
            "which": ["azerty", "dvorak", "qwerty", "qwertz"],
            "not_this": (
                "NOT participants (~55k PIDs on the Aalto side are not independent units for "
                "LAYOUT-level generalization), and NOT the separate community frame"
            ),
            "verified": (
                "cut -f1 on /local/home/zegertho/keybo-e2e/bistrokes31_v1.tsv (2202 rows) and "
                "tristrokes31_cond_v1.tsv (16643 rows) — both yield exactly these four. Rows "
                "counted with `awk END{NR}`; the FIRST LINE OF EACH FILE IS DATA, NOT A HEADER, "
                "so `wc -l` minus one undercounts by one"
            ),
            "target_space": "LOGRAT (confirmed in all six k31 meta sidecars)",
            "so": (
                "every r2/increment here is against MODEL PREDICTIONS, not measured time; n_pool "
                "is a sampling density over one fitted surface, NOT an evidence count"
            ),
        },
    }


def generalization(pool: list[dict]) -> dict:
    """Is the PINKY special, or is this the pinky instance of 'off-home use of a weak finger'?"""
    ms = [row["ms_per_char"] for row in pool]
    out = {}
    for finger in ("pinky", "ring", "middle", "index"):
        out[finger] = {
            "off_home_abs_r_with_ms": abs(pearson([row[f"{finger}_off_home"] for row in pool], ms)),
            "usage_abs_r_with_ms": abs(pearson([row[f"{finger}_usage"] for row in pool], ms)),
        }
    return {
        "per_finger": out,
        "reading": (
            "If ring behaves like pinky, the user has found the pinky INSTANCE of a broader "
            "off-home-weak-finger axis, which is the bigger finding. If the pinky separates, it "
            "is genuinely special."
        ),
    }


def one_swap_probe(seed_name: str, seed_layout: str, bigrams, trigrams) -> dict:
    """Greedy 1-swap descent on travel_total — is the metric even MOVABLE? Not a full search.

    Registered as a movability probe, not an optimization: the question is whether the metric can
    be pushed at all and WHAT it asks for, because a descriptor whose minimizer the time model
    dislikes should not become an objective (WSCISSOR-GEN-1: optimizing a severity axis is
    optimizing the ruler).
    """
    from keybo.analysis.finger_travel import FingerTravel
    from keybo.geometry import ROW_STAGGERED_30
    from keybo.layout import Layout

    travel = FingerTravel(bigrams)
    layout = Layout(seed_layout, ROW_STAGGERED_30)
    start = travel.total(layout)
    characters = list(layout.chars)
    history = []
    current = start
    for _ in range(12):  # a cheap probe: at most 12 accepted swaps
        best = None
        for i, first in enumerate(characters):
            for second in characters[i + 1 :]:
                layout.swap(first, second)
                candidate = travel.total(layout)
                layout.undo()
                if candidate < current - 1e-9 and (best is None or candidate < best[0]):
                    best = (candidate, first, second)
        if best is None:
            break
        current, first, second = best
        layout.swap(first, second)
        history.append({"swap": [first, second], "travel_total": current})
    return {
        "seed": seed_name,
        "seed_travel_total": start,
        "final_travel_total": current,
        "reduction_pct": 100.0 * (start - current) / start,
        "accepted_swaps": len(history),
        "history": history,
        "final_layout": "".join(layout.chars),
        "converged": len(history) < 12,
    }


def main(board_path: str, out_path: str) -> None:
    sys.path.insert(0, str(ROOT / "src"))
    from keybo.data.corpus import load_frequencies, production_corpus_dir
    from keybo.testkit import assert_module_under

    assert_module_under("keybo.analysis.finger_travel", ROOT / "src")

    board = json.loads(Path(board_path).read_text())["rows"]
    names = sorted(board)

    # --- 1. correlations over the 18-layout field ---
    predictors = {g: [board[n]["gauges"][g] for n in names] for g in GAUGES}
    travel_total = [board[n]["travel"]["total"] for n in names]
    travel_max = [board[n]["travel"]["dispersion"]["max_share"] for n in names]
    travel_pinky = [board[n]["travel"]["dispersion"]["pinky_share"] for n in names]
    travel_gini = [board[n]["travel"]["dispersion"]["gini"] for n in names]
    pinky_off = [board[n]["off_home"]["pinky"]["off_home"] for n in names]
    pinky_total = [board[n]["off_home"]["pinky"]["usage"] for n in names]
    pinky_fraction = [board[n]["off_home"]["pinky"]["off_fraction"] for n in names]
    ms = [board[n]["time"]["ms_per_char"] for n in names]

    targets = {
        "travel_total": travel_total,
        "travel_max_share": travel_max,
        "travel_pinky_share": travel_pinky,
        "travel_gini": travel_gini,
        "pinky_off_home": pinky_off,
        "pinky_usage_total": pinky_total,
        "pinky_off_fraction": pinky_fraction,
    }
    # The two qwerty boards are high-leverage on any magnitude statistic (travel_total ≈5.1e8 vs
    # 2.2-2.8e8 for every optimized board). Every ms/char correlation therefore carries a signed
    # value, a rank value, and a leverage audit — see `leverage_audit`.
    leverage = [i for i, n in enumerate(names) if n.startswith("qwerty")]
    correlations = {
        name: {
            "on_gauge_frame": r2_on_frame(values, predictors),
            "closest_single_gauges": single_best(values, predictors)[:5],
            "abs_r_with_ms_per_char": abs(pearson(values, ms)),
            "signed_r_with_ms_per_char": pearson(values, ms),
            "spearman_with_ms_per_char": spearman(values, ms),
            "leverage_audit": leverage_audit(values, ms, leverage, "qwerty + qwerty30m"),
        }
        for name, values in targets.items()
    }
    # and against each other — two metrics, one column each only if they differ
    correlations["cross"] = {
        "travel_total_vs_pinky_off_home": pearson(travel_total, pinky_off),
        "travel_pinky_share_vs_pinky_off_home": pearson(travel_pinky, pinky_off),
        "pinky_total_vs_pinky_off_home": pearson(pinky_total, pinky_off),
    }

    # --- 2. tie-breaking ---
    ties = tie_breaking(board)

    # --- 3. the cost claim, on a RANDOMIZED pool (18 layouts cannot support a regression) ---
    corpus_dir = production_corpus_dir(None)
    bigrams = load_frequencies(str(corpus_dir / "bigrams.txt"))
    trigrams = load_frequencies(str(corpus_dir / "trigrams.txt"))
    pool = build_pool(board, bigrams, trigrams)
    claim = cost_claim(pool)
    general = generalization(pool)

    # --- 4. movability ---
    probe = one_swap_probe("graphite", board["graphite"]["layout"], bigrams, trigrams)
    probe_time = probe_time_check(probe, trigrams)

    Path(out_path).write_text(
        json.dumps(
            {
                "generated_by": "agent-artifacts/ft_analysis.py",
                "field": names,
                "correlations": correlations,
                "tie_breaking": ties,
                "cost_claim_C2": claim,
                "generalization_beyond_pinky": general,
                "movability_probe": probe,
                "movability_time_check": probe_time,
            },
            indent=1,
        )
    )
    print(f"wrote {out_path}", file=sys.stderr)


def build_pool(board: dict, bigrams, trigrams, size: int = 160, seed: int = 20260728) -> list[dict]:
    """A randomized layout pool with per-row metrics + predicted ms/char.

    The 18-layout field is far too small to regress ms/char on anything (n=18, and the campaign's
    own registered warning is that the gauge frame has ~4-5 effective dof). The pool is built by
    random swaps away from real boards so it spans a wide range of the metrics rather than only
    the optimized corner where every candidate already has good pinkies.
    """
    from keybo.analysis.finger_travel import FingerTravel, OffHomeUsage
    from keybo.analysis.kmstats import KmStats
    from keybo.analysis.timecard import default_surface
    from keybo.geometry import ROW_STAGGERED_30
    from keybo.layout import Layout

    rng = random.Random(seed)
    travel = FingerTravel(bigrams)
    off_home = OffHomeUsage(bigrams)
    surface = default_surface(90.0, None)
    kms = KmStats(bigrams, {}, trigrams)

    seeds = [board[n]["layout"] for n in sorted(board)]
    seen: set[str] = set()
    rows = []
    print(f"building pool of {size} layouts…", file=sys.stderr)
    while len(rows) < size:
        base = list(rng.choice(seeds))
        for _ in range(rng.randint(0, 14)):
            i, j = rng.randrange(30), rng.randrange(30)
            base[i], base[j] = base[j], base[i]
        candidate = "".join(base)
        if candidate in seen:
            continue
        seen.add(candidate)
        layout = Layout(candidate, ROW_STAGGERED_30)
        usage, off = off_home.usage(layout), off_home.off_home(layout)
        report = travel.report(layout)
        stats = dict(kms.stats(candidate))
        rows.append(
            {
                "layout": candidate,
                "ms_per_char": surface.card(candidate).ms_per_char,
                "travel_total": report["total"],
                "travel_pinky_share": report["dispersion"]["pinky_share"],
                "travel_max_share": report["dispersion"]["max_share"],
                "sfb": stats["sfb"],
                "sfb_dist": stats["sfb-dist"],
                **{
                    f"{kind}_usage": usage[f"L-{kind}"] + usage[f"R-{kind}"]
                    for kind in ("pinky", "ring", "middle", "index")
                },
                **{
                    f"{kind}_off_home": off[f"L-{kind}"] + off[f"R-{kind}"]
                    for kind in ("pinky", "ring", "middle", "index")
                },
                "pinky_usage": usage["L-pinky"] + usage["R-pinky"],
                "pinky_off_home": off["L-pinky"] + off["R-pinky"],
                "pinky_off_fraction": (
                    100.0
                    * (off["L-pinky"] + off["R-pinky"])
                    / (usage["L-pinky"] + usage["R-pinky"])
                    if (usage["L-pinky"] + usage["R-pinky"])
                    else 0.0
                ),
            }
        )
        if len(rows) % 40 == 0:
            print(f"  {len(rows)}/{size}", file=sys.stderr)
    return rows


def probe_time_check(probe: dict, trigrams) -> dict:
    """Does the travel-minimizing board the probe found actually type FASTER?

    The whole point of asking. If travel drops several percent and predicted time does not
    improve (or worsens), the metric is a descriptor whose minimizer the time model does not
    want — WSCISSOR-GEN-1's "optimizing the ruler" result, and the reason to report it as a
    diagnostic rather than wire it into the objective.
    """
    from keybo.analysis.timecard import default_surface

    surface = default_surface(90.0, None)
    before = surface.card(_seed_of(probe)).ms_per_char
    after = surface.card(probe["final_layout"]).ms_per_char
    return {
        "seed_ms_per_char": before,
        "final_ms_per_char": after,
        "delta_ms_per_char": after - before,
        "travel_reduction_pct": probe["reduction_pct"],
        "time_got_worse": bool(after > before),
    }


def _seed_of(probe: dict) -> str:
    """The probe's starting layout — recovered from its own history, never re-typed."""
    layout = list(probe["final_layout"])
    for step in reversed(probe["history"]):
        first, second = step["swap"]
        i, j = layout.index(first), layout.index(second)
        layout[i], layout[j] = layout[j], layout[i]
    return "".join(layout)


if __name__ == "__main__":
    main(
        sys.argv[1] if len(sys.argv) > 1 else "agent-artifacts/ft_board.json",
        sys.argv[2] if len(sys.argv) > 2 else "agent-artifacts/ft_analysis.json",
    )
