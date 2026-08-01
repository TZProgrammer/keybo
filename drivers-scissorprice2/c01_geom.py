"""Gate 1: the GEOMETRY of `scissor` -- floor, unconstrained optimum, and the support size.

Three questions that decide whether a shadow price is even identifiable, all measured BEFORE
any slope is computed (and before pre-registration, so the prereg's thresholds are grounded
in measured noise rather than guesses):

  Q1  What is the achievable scissor FLOOR (by descent)? Is the field at it?  [PRICEBAND's §4
      premise correction, re-asked for this gauge -- and I must not inherit its answer.]
  Q2  Where does the UNCONSTRAINED speed optimum sit in scissor? (interior vs boundary hint)
  Q3  How big is the support? `scissor` fires on how many of the 870 ordered pairs, and what
      is the COARSEST possible granularity of the gauge -- i.e. is a "per pp" unit even
      meaningful at this mass?
"""
import _env  # noqa: F401
import json

import boards
import fastgauge
import numpy as np
import search
from _env import ART

from keybo.features import classify as C
from keybo.geometry import ROW_STAGGERED_30


class Obj(search.Objective):
    """`search.Objective` with `scissor` added. Inherits `sweep`/`ms`/`sfb` unchanged, so the
    2-opt and cap-descent machinery is PRICEBAND-1's byte-for-byte -- only the gauge callable
    passed to `cap_min_ms` differs. That reuse is the point: if I re-implemented the search I
    would be comparing two estimators, not two gauges."""

    def scissor(self, p):
        return self.fg.scissor_only(p[:30])


def descend_gauge(obj, p, gauge, max_sweeps=300):
    """Steepest descent on the gauge alone -- gives a witness for the floor (upper bound)."""
    cur = gauge(p)
    for _ in range(max_sweeps):
        P = search.swap_perms(p)
        gs = np.array([gauge(q) for q in P])
        k = int(np.argmin(gs))
        if gs[k] < cur - 1e-12:
            p, cur = P[k], float(gs[k])
        else:
            return p, cur
    return p, cur


def main():
    fs, w1, w2 = _env.verify_evaluators(boards.FIELD)
    fg = fastgauge.FastGauges()
    obj = Obj(fs, fg)
    gauge = lambda q: fg.scissor_only(q[:30])  # noqa: E731
    rng = np.random.default_rng(20260801)

    # ---- Q3 first: the SUPPORT. Pure geometry, no search, so it is exact. ----
    g = ROW_STAGGERED_30
    slots = g.slots
    n_ordered = sum(1 for i in range(30) for j in range(30) if i != j)
    scis_pairs = [(i, j) for i in range(30) for j in range(30)
                  if i != j and C.is_scissor(g, slots[i], slots[j])]
    sfb_pairs = [(i, j) for i in range(30) for j in range(30)
                 if i != j and g.same_finger(slots[i][0], slots[j][0])]
    support = {
        "n_ordered_distinct_slot_pairs": n_ordered,
        "n_scissor_pairs": len(scis_pairs),
        "n_sfb_pairs": len(sfb_pairs),
        "scissor_pairs_pct_of_ordered": 100.0 * len(scis_pairs) / n_ordered,
    }
    print("== Q3 support (exact geometry) ==")
    print(f"  ordered distinct slot pairs: {n_ordered}")
    print(f"  scissor fires on {len(scis_pairs)} ({support['scissor_pairs_pct_of_ordered']:.2f}%)")
    print(f"  sfb fires on     {len(sfb_pairs)}")

    # ---- Q1: the floor, by descent. 30 random + 13 field-seeded, as PRICEBAND did. ----
    print("\n== Q1 scissor floor by descent ==")
    floors = []
    for r in range(30):
        p = search.random_perm(rng)
        _, v = descend_gauge(obj, p, gauge)
        floors.append(v)
    field_floors = {}
    for n, s in boards.FIELD.items():
        p = fs.perm(s)
        _, v = descend_gauge(obj, p, gauge)
        field_floors[n] = v
        floors.append(v)
    floor = float(min(floors))
    field_scis = {n: fg.scissor_only(fg.perm(s)) for n, s in boards.FIELD.items()}
    opt_scis = {n: v for n, v in field_scis.items() if n != boards.OFF_FRONTIER}
    print(f"  floor (witness, upper bound on true min) = {floor:.6f}")
    print(f"  n descents = {len(floors)}; random-start best = {min(floors[:30]):.6f}")
    print(f"  optimized field scissor range = [{min(opt_scis.values()):.4f}, "
          f"{max(opt_scis.values()):.4f}], median {float(np.median(list(opt_scis.values()))):.4f}")
    print(f"  median field distance ABOVE the floor = "
          f"{float(np.median([v - floor for v in opt_scis.values()])):.4f} pp")

    # ---- Q2: where does the UNCONSTRAINED speed optimum sit in scissor? ----
    print("\n== Q2 unconstrained speed optimum: its scissor value ==")
    uncon = []
    for r in range(12):
        p = search.random_perm(rng)
        p, m = search.two_opt_ms(obj, p)
        uncon.append((m, float(gauge(p))))
    uncon.sort()
    for m, sc in uncon:
        print(f"    ms={m:.4f}  scissor={sc:.4f}")
    best8 = uncon[:8]
    print(f"  8 best random-2-opt boards: scissor "
          f"[{min(s for _, s in best8):.4f}, {max(s for _, s in best8):.4f}], "
          f"median {float(np.median([s for _, s in best8])):.4f}")
    print(f"  ms/char sd over the {len(uncon)} descents = {float(np.std([m for m, _ in uncon], ddof=1)):.4f}"
          "   <- the SEARCH-seed noise scale, re-measured for this gauge")

    # ---- Q3b: granularity. What is the SMALLEST nonzero scissor step, and how many
    #      distinct scissor values does one board's 435-swap neighbourhood even have? ----
    print("\n== Q3b granularity of the gauge in the near-optimal band ==")
    gran = {}
    for n in ("arm-B", "flagship-c3", "graphite"):
        p = fs.perm(boards.FIELD[n])
        P = search.swap_perms(p)
        vals = np.array([gauge(q) for q in P])
        base = float(gauge(p))
        d = np.abs(vals - base)
        nz = d[d > 1e-12]
        gran[n] = {
            "base": base,
            "n_distinct_neighbour_values": int(len(np.unique(np.round(vals, 10)))),
            "n_zero_change": int((d <= 1e-12).sum()),
            "min_nonzero_abs_step": float(nz.min()) if len(nz) else None,
            "median_abs_step": float(np.median(nz)) if len(nz) else None,
            "max_abs_step": float(d.max()),
            "neighbour_range": [float(vals.min()), float(vals.max())],
        }
        print(f"  {n:<13} base={base:.4f}  distinct={gran[n]['n_distinct_neighbour_values']:>4}"
              f"  no-change={gran[n]['n_zero_change']:>4}"
              f"  |step| min={gran[n]['min_nonzero_abs_step']:.2e}"
              f" med={gran[n]['median_abs_step']:.4f} max={gran[n]['max_abs_step']:.4f}"
              f"  range=[{vals.min():.4f},{vals.max():.4f}]")

    out = {
        "fasteval_worst": w1, "fastgauge_worst": w2,
        "support": support,
        "scissor_floor_witness": floor,
        "n_floor_descents": len(floors),
        "floor_from_random_only": float(min(floors[:30])),
        "field_seeded_floors": field_floors,
        "field_scissor": field_scis,
        "optimized_field_scissor_range": [min(opt_scis.values()), max(opt_scis.values())],
        "optimized_field_scissor_median": float(np.median(list(opt_scis.values()))),
        "median_field_pp_above_floor": float(np.median([v - floor for v in opt_scis.values()])),
        "unconstrained_2opt": [{"ms": m, "scissor": s} for m, s in uncon],
        "unconstrained_best8_scissor_range": [min(s for _, s in best8), max(s for _, s in best8)],
        "unconstrained_best8_scissor_median": float(np.median([s for _, s in best8])),
        "search_seed_sd_ms": float(np.std([m for m, _ in uncon], ddof=1)),
        "granularity": gran,
    }
    with open(ART + "/s01_geom.json", "w") as f:
        json.dump(out, f, indent=1)
    print("\nwrote s01_geom.json")


if __name__ == "__main__":
    main()
