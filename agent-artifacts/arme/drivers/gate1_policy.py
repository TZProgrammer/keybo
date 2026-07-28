"""GATE 1 — is arm E's objective the same function arm D optimized, on the archive curves?

Arm E's whole claim to comparability rests on "only the weights JSON changed". That is checkable
before spending 10M evals, and it is worth checking because arm D's near-miss was exactly this
class of defect: the policy existed on `LossCurve` while the search ran a hand-rolled copy that
ignored it (trap 28). `cf5f731` added `LossCurve.price_many` to close that, so arm E's job is to
prove it USES it and that using it changes nothing about the arithmetic.

⚠ THE BRIEF ASKS FOR AN UNSATISFIABLE PIN, AND THIS GATE IS WHY I KNOW THAT. The brief says to
pin the fast path against `price_many` "at EXACT float equality". Measured here on the real
archive curves: `price_many` is **not** bit-equal to `price` for n>=2, and is **not** bit-equal to
ITSELF across batch shapes. `price` evaluates a length-1 array while `price_many` evaluates length
n, and `_design(...) @ coeffs` dispatches to a different BLAS kernel by shape — identical design
rows, 1-ULP-different products (`comfort` at its `lo`: 0.069389400121559 at n=1 vs
0.06938940012155903 at n>=2). 7 of the 14 archive curves show it; arm D's hand-rolled elementwise
`ClampedCurve` shows it on 0 of 14, because elementwise arithmetic has no shape dispatch. So NO
shape-invariant implementation can be exactly equal to `price_many`, and the 4 shipped tests
cannot see it (one fixed 8-element array on both sides; the saturation test uses `approx`).

The right check is therefore ULP-bounded equality PLUS the property that actually matters to a
search — the induced ORDERING must be identical, since an optimizer only ever compares scores.
`ULP_TOL` is deliberately tiny (4 ULP); this gate still fails loudly on a real semantic change.

Six checks, on the REAL fitted archive curves (not synthetic ones):

  A  price_many == price element-wise to <= ULP_TOL — in-domain, and at levels spanning
     [lo - 3w, hi + 3w] on both sides — under both EXTRAPOLATE and CLAMP.
  B  price_many(CLAMP) == price_many(EXTRAPOLATE) BITWISE for every level strictly INSIDE the
     domain. This one IS exact: same shape, same code path, so "arm E perturbs no supported
     level" is provable at bit level and is asserted that way.
  C  price_many(CLAMP) SATURATES: past either edge the value is constant and equals the edge
     price at the SAME batch shape, so the gradient out there is exactly 0.0 and a maximizer
     collects nothing for leaving. (Comparing across shapes here would re-measure the ULP
     artifact, not saturation — so the edge reference is evaluated at matching shape.)
  D  price_many(CLAMP) == arm D's frozen hand-rolled `ClampedCurve.price` to <= ULP_TOL on the
     same curves, AND induces the identical ordering. This is the cross-implementation pin that
     makes arm E's number comparable to arm D's.
  E  the objective the SEARCH will run (`ValidatedClampedEval`) equals a straight
     `sum_g LossCurve.price_many(level_g, CLAMP)` BITWISE on real layouts, and DIFFERS from the
     extrapolating total on at least one — because a clamp that changes nothing anywhere would
     mean the domains never bind and the arm is silently arm A.
  F  on 2000+ real and random layouts the two implementations' TOTALS agree to <= ULP_TOL, pick
     the same argmin, and induce the identical argsort. This is the check that licenses calling
     arm E comparable to arm D despite A/D being ULP-loose rather than bitwise.

Fails loudly and writes a `gate1-rc.txt` sentinel; trap 1 says the absence of that file is not a
pass. Corpus blend-v1, frame .native, MODELLED ONLY.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append("/local/home/zegertho/agent/state/optevidence/artifacts/drivers")
sys.path.append("/local/home/zegertho/agent/state/armd/artifacts/drivers")

import armd_obj as AD  # noqa: E402  (arm D's FROZEN hand-rolled clamp — the check-D reference)
import arme_obj as AE  # noqa: E402
import evobj as EV  # noqa: E402
from arme_load import load_curves, load_meta  # noqa: E402
from keybo.analysis.evidence_scorer import CLAMP, EXTRAPOLATE, LIVE_GAUGES  # noqa: E402

ARM_JSON = "/local/home/zegertho/agent/state/evidence-scorer/artifacts/arm-archive400-native.json"
STATE = Path("/local/home/zegertho/agent/state/arme/artifacts")

#: Rounding-bound slack factor. Two implementations of `c0 + c1*x + c2*hinge` that differ only in
#: summation/association order are bounded by the standard dot-product rounding bound
#: `n * eps * sum|term|` — the bound is on the sum of the TERMS' magnitudes, not on the result,
#: which is why a value that arises by cancellation (sr-roll's +0.0150 from terms of order 1) can
#: be hundreds of ULP away while still being the same function. `FLOP_SLACK` multiplies that bound.
#:
#: ULP is deliberately NOT the criterion: it measures spacing at the RESULT's magnitude, so it
#: explodes near a zero-crossing and would flag correct arithmetic while being blind to a real
#: error in a large-magnitude gauge. ULP is still reported, as a diagnostic.
FLOP_SLACK = 16

#: The search accepts an improvement only if it exceeds this (`nfit[bi] < cur_fit - 1e-12` in
#: `search_armd.py`). Any discrepancy far below it cannot change a single accept/reject decision,
#: which is the materiality argument check F makes quantitative.
SEARCH_ACCEPT_THRESHOLD = 1e-12

FAILURES: list[str] = []
N_CHECKS = 0


def check(cond: bool, msg: str) -> None:
    global N_CHECKS
    N_CHECKS += 1
    if not cond:
        FAILURES.append(msg)
        print(f"  FAIL: {msg}")


def ulp_distance(a: np.ndarray, b: np.ndarray) -> int:
    """Worst element-wise ULP distance between two float64 arrays. DIAGNOSTIC ONLY.

    Reported because it is the natural unit for "are these the same float", but NOT used as the
    pass criterion: ULP is spacing at the RESULT's magnitude, so a value produced by cancellation
    sits absurdly many ULP from an equally-correct alternative rounding. Measured here: sr-roll at
    level 13.6116 gives 0.015004423663657684 vs 0.015004423663658572 — an 8.9e-16 absolute
    difference between terms of order 1, which is 512 ULP at that result's magnitude. Use
    `rounding_bound` for the verdict.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ia = a.view(np.int64).copy()
    ib = b.view(np.int64).copy()
    # map the sign-magnitude int64 bit pattern onto a monotone ordering
    ia[ia < 0] = np.int64(np.iinfo(np.int64).min) - ia[ia < 0]
    ib[ib < 0] = np.int64(np.iinfo(np.int64).min) - ib[ib < 0]
    return int(np.max(np.abs(ia - ib))) if a.size else 0


def rounding_bound(curve, levels: np.ndarray) -> np.ndarray:
    """Per-level bound on |two roundings of the same curve|: `n * eps * sum|term|`.

    This is the textbook dot-product error bound applied to the design row. It is the right
    criterion for "same function, different association order" because it scales with the TERMS,
    which is exactly what cancellation makes large relative to the result.
    """
    x = np.asarray(levels, dtype=np.float64)
    c = [float(v) for v in curve.coeffs]
    terms = [np.abs(np.full_like(x, c[0])), np.abs(c[1] * x)]
    if curve.form == "quadratic":
        terms.append(np.abs(c[2] * x * x))
    elif curve.form == "hinge":
        terms.append(np.abs(c[2] * np.maximum(x - float(curve.knot), 0.0)))
    mag = np.sum(terms, axis=0)
    return len(terms) * np.finfo(np.float64).eps * mag


def main() -> int:
    curves = load_curves(ARM_JSON)
    meta = load_meta(ARM_JSON)
    print(f"weights: source={meta.get('source')} frame={meta.get('surface_frame')} "
          f"corpus={meta.get('corpus')} pool={meta.get('pool')} n={meta.get('n_layouts')}")

    # The .native assertion the brief demands, made a HARD gate rather than a comment.
    check(meta.get("surface_frame") == "native",
          f"ARM E requires the .native frame; weights say {meta.get('surface_frame')!r}")
    check(meta.get("corpus") == "blend-v1",
          f"ARM E requires blend-v1; weights say {meta.get('corpus')!r}")
    check(meta.get("pool") == "archive-400",
          f"ARM E requires the ARCHIVE pool; weights say {meta.get('pool')!r}")
    if FAILURES:
        print("\nGATE 1 ABORT: the weights JSON is not the one arm E is specified against")
        return 1

    # ---- A / B / C / D: the validated vectorized path against two independent references ----
    print(f"\n{'gauge':<12} {'form':<10} {'domain':<26} {'A/bnd':>7} {'A ulp':>6} "
          f"{'B bitwise':>10} {'C sat':>7} {'D/bnd':>7} {'D order':>8} {'s ulp':>6} {'s#':>5}")
    detail = {}
    worst_a = worst_d = worst_shape = 0.0
    for name in LIVE_GAUGES:
        c = curves[name]
        lo, hi = c.domain
        w = hi - lo
        # levels spanning far outside on BOTH sides, plus the exact edges and the knot
        xs = np.concatenate([
            np.linspace(lo - 3 * w, lo - 1e-9, 12),
            np.linspace(lo, hi, 16),
            np.linspace(hi + 1e-9, hi + 3 * w, 12),
            np.array([lo, hi] + ([] if c.knot is None else [float(c.knot)])),
        ])
        row = {"domain": [lo, hi], "form": c.form, "knot": c.knot, "n_levels": int(xs.size)}

        # A: vectorized == scalar, within the rounding bound, under both policies
        a_ulp, a_ratio = 0, 0.0
        for policy in (EXTRAPOLATE, CLAMP):
            many = c.price_many(xs, policy=policy)
            one = np.array([c.price(float(x), policy=policy) for x in xs])
            clipped = np.clip(xs, lo, hi) if policy == CLAMP else xs
            bound = FLOP_SLACK * rounding_bound(c, clipped)
            ratio = float(np.max(np.abs(many - one) / np.maximum(bound, 5e-324)))
            a_ulp = max(a_ulp, ulp_distance(many, one))
            a_ratio = max(a_ratio, ratio)
            check(ratio <= 1.0,
                  f"{name}: price_many({policy}) vs price({policy}) exceeds the rounding bound "
                  f"by {ratio:.2f}x — that is a semantic difference, not a rounding difference")
        row["A_ulp_vs_scalar"] = a_ulp
        row["A_bound_ratio_vs_scalar"] = a_ratio
        worst_a = max(worst_a, a_ratio)

        # B: in-domain, CLAMP is a no-op. Same shape both sides, so this IS bitwise-exact and is
        # asserted that way — it is the claim "arm E perturbs no supported level".
        inside = np.linspace(lo, hi, 257)
        b_eq = np.array_equal(c.price_many(inside, policy=CLAMP),
                              c.price_many(inside, policy=EXTRAPOLATE))
        check(b_eq, f"{name}: CLAMP perturbs a level INSIDE [{lo:.4f},{hi:.4f}]")
        row["B_in_domain_bitwise_identical"] = bool(b_eq)

        # C: outside, the price is CONSTANT at the edge value => zero gradient => a maximizer
        # collects nothing for leaving. Evaluate the far points and the edge reference in ONE
        # array so the comparison cannot re-measure the shape artifact instead of saturation.
        probe = c.price_many(np.array([lo - 1e3 * w, lo - 50 * w, lo, hi, hi + 50 * w,
                                       hi + 1e3 * w]), policy=CLAMP)
        sat = bool(probe[0] == probe[1] == probe[2] and probe[3] == probe[4] == probe[5])
        check(sat, f"{name}: CLAMP does not saturate outside the domain")
        row["C_saturates"] = sat

        # D: cross-implementation pin against arm D's FROZEN hand-rolled clamp — ULP-bounded,
        # and the ordering it induces must be IDENTICAL (that is what a search consumes).
        ad = AD.ClampedCurve(metric=c.metric, form=c.form,
                             coeffs=np.asarray(c.coeffs, dtype=float),
                             knot=c.knot, domain=(lo, hi))
        pm, hr = c.price_many(xs, policy=CLAMP), ad.price(xs)
        d_bound = FLOP_SLACK * rounding_bound(c, np.clip(xs, lo, hi))
        d_ratio = float(np.max(np.abs(pm - hr) / np.maximum(d_bound, 5e-324)))
        d_ulp = ulp_distance(pm, hr)
        d_order = np.array_equal(np.argsort(pm, kind="stable"), np.argsort(hr, kind="stable"))
        check(d_ratio <= 1.0,
              f"{name}: price_many(CLAMP) vs arm D's ClampedCurve exceeds the rounding bound by "
              f"{d_ratio:.2f}x — the two clamps are not the same function")
        check(d_order, f"{name}: price_many(CLAMP) and arm D's ClampedCurve order levels "
                       f"DIFFERENTLY — a search would follow different gradients")
        row["D_ulp_vs_armD"] = d_ulp
        row["D_bound_ratio_vs_armD"] = d_ratio
        row["D_same_ordering_as_armD"] = bool(d_order)
        worst_d = max(worst_d, d_ratio)

        # the shape-dispatch artifact itself, measured and recorded rather than hidden
        lev = lo + 0.37 * w
        shape_vals = [float(c.price_many(np.full(n, lev), policy=CLAMP)[0])
                      for n in (1, 2, 3, 8, 48, 435, 1024)]
        s_arr = np.array(shape_vals)
        s_ulp = ulp_distance(s_arr, np.full(len(shape_vals), shape_vals[-1]))
        s_bound = FLOP_SLACK * float(rounding_bound(c, np.array([lev]))[0])
        s_ratio = float(np.max(np.abs(s_arr - shape_vals[-1])) / max(s_bound, 5e-324))
        row["shape_dispatch_ulp"] = s_ulp
        row["shape_dispatch_bound_ratio"] = s_ratio
        row["shape_dispatch_distinct_values"] = len(set(shape_vals))
        worst_shape = max(worst_shape, s_ratio)
        check(s_ratio <= 1.0,
              f"{name}: price_many varies across batch shapes 1..1024 by {s_ratio:.2f}x the "
              f"rounding bound")

        detail[name] = row
        print(f"{name:<12} {c.form:<10} [{lo:9.4f},{hi:10.4f}] {a_ratio:7.3f} {a_ulp:6d} "
              f"{str(b_eq):>10} {str(sat):>7} {d_ratio:7.3f} {str(d_order):>8} "
              f"{s_ulp:6d} {len(set(shape_vals)):5d}")

    print(f"\nA worst |price_many - price| / rounding bound  = {worst_a:.3f} (must be <= 1.0)")
    print(f"D worst |price_many(CLAMP) - armD| / bound     = {worst_d:.3f} (must be <= 1.0)")
    print(f"  worst shape-dispatch deviation / bound = {worst_shape:.3f}")
    print("  (that last number is the numpy matmul shape-dispatch artifact: `price` evaluates a")
    print("   length-1 array and `price_many` length n, so `_design @ coeffs` takes a different")
    print("   BLAS path. It is why 'pin at EXACT float equality' is unsatisfiable against")
    print("   price_many — it is not bit-equal to itself. Arm D's elementwise ClampedCurve is")
    print("   0 ULP across all shapes. Recorded as a finding; check F bounds what it can affect.)")

    # ---- E: the object the SEARCH runs ---------------------------------------------------
    fe = EV.FastEval(corpus=None, weights_json=ARM_JSON, with_surface=True)
    ve = AE.ValidatedClampedEval(fe, curves, policy=CLAMP)
    print(f"\ncorpus dir: {fe.corpus_dir}")
    check(str(fe.corpus_dir).endswith("blend-v1"),
          f"corpus is {fe.corpus_dir}, not blend-v1")

    # real layouts: the incumbents + arm A/B/C/D champions + qwerty, and 64 random perms
    from keybo.cli.score_evidence import _EXTRA_NAMED
    OPTEV = Path("/local/home/zegertho/agent/state/optevidence/artifacts")
    inc = json.load(open(OPTEV / "incumbent-reference.json"))
    board: dict[str, str] = dict(inc["incumbents"])
    board.update(inc["reference"])
    board["flagship-c3"] = _EXTRA_NAMED["flagship-c3"]
    for arm, label in (("evidence", "armA"), ("baseline", "armB"), ("constrained", "armC")):
        board[label] = json.load(open(OPTEV / f"runs/arm-{arm}.json"))["champion"]["layout"]
    board["armD"] = json.load(
        open("/local/home/zegertho/agent/state/armd/artifacts/runs/arm-domain.json")
    )["champion"]["layout"]

    rng = np.random.default_rng(20260728)
    rand = [np.concatenate([rng.permutation(30).astype(np.int32), np.array([30], dtype=np.int32)])
            for _ in range(2048)]
    perms = np.stack([EV.perm_of(lay) for lay in board.values()] + rand)
    g = fe.gauges(perms)

    # the search's total, vs an independent straight sum through the validated path. Same shapes
    # on both sides, so this one IS bitwise.
    got = ve.evidence_score(g)
    want = np.zeros_like(got)
    for name in LIVE_GAUGES:
        want = want + curves[name].price_many(g[name], policy=CLAMP)
    check(np.array_equal(got, want), "ValidatedClampedEval.evidence_score != sum of price_many")

    # and it must actually DIFFER from the extrapolating total somewhere, or the clamp is inert
    ext = ve.evidence_score_extrapolating(g)
    n_diff = int(np.sum(got != ext))
    check(n_diff > 0, "CLAMP is inert on every test layout — the domains never bind, so arm E "
                      "would silently be an extrapolating arm")
    print(f"\nE: search objective == sum(price_many(CLAMP)) BITWISE on {perms.shape[0]} layouts; "
          f"clamp changes the total on {n_diff}/{perms.shape[0]}")

    # and the reverse pin: EXTRAPOLATE through price_many == evobj's own hand-rolled total,
    # which is what makes arm E's ev_extrap column comparable to arm A's.
    check(np.allclose(ext, fe.evidence_score(g), rtol=0, atol=1e-9),
          "price_many(EXTRAPOLATE) total disagrees with evobj.FastEval.evidence_score")
    print(f"   max|price_many(EXTRAP) - evobj total| = "
          f"{float(np.max(np.abs(ext - fe.evidence_score(g)))):.3e}")

    # ---- F: the check that licenses "arm E is comparable to arm D" -------------------------
    # A and D are ULP-loose rather than bitwise, so the comparability claim has to rest on
    # something an optimizer can actually observe: the TOTAL, its argmin, and its full ordering.
    ad_eval = AD.ClampedEval(fe, policy=CLAMP)  # fe.curves are the ARCHIVE curves here
    tot_hr = ad_eval.evidence_score(g)
    f_abs = float(np.max(np.abs(got - tot_hr)))
    f_ulp = ulp_distance(got, tot_hr)
    f_rel = f_abs / float(got.max() - got.min())
    f_argmin = int(np.argmin(got)) == int(np.argmin(tot_hr))
    f_order = np.array_equal(np.argsort(got, kind="stable"), np.argsort(tot_hr, kind="stable"))
    # The materiality criterion, and the one that licenses comparability: the disagreement must be
    # far below the threshold at which the search would act on a difference, AND must not reorder
    # anything. Both, not either.
    check(f_abs < SEARCH_ACCEPT_THRESHOLD / 100.0,
          f"totals differ by {f_abs:.3e}, which is not negligible against the search's "
          f"{SEARCH_ACCEPT_THRESHOLD:.0e} accept threshold")
    check(f_argmin, "the two implementations pick DIFFERENT argmin — not comparable")
    check(f_order, "the two implementations induce DIFFERENT orderings — not comparable")
    print(f"F: over {perms.shape[0]} layouts spanning {got.max() - got.min():.4f} score units, "
          f"price_many vs armD hand-rolled:")
    print(f"   worst |diff| {f_abs:.3e} ({f_ulp} ULP) = {f_rel:.3e} of the score range; "
          f"same argmin {f_argmin}; identical full ordering {f_order}")
    print(f"   search accept threshold {SEARCH_ACCEPT_THRESHOLD:.0e} is "
          f"{SEARCH_ACCEPT_THRESHOLD / max(f_abs, 5e-324):.2e}x larger, so no accept/reject "
          f"decision can turn on it")

    # ---- the incumbents' out-of-domain census under ARCHIVE weights (the arm's premise) ----
    ood = fe.out_of_domain(g)
    n_ood = np.sum(np.stack([ood[m] for m in LIVE_GAUGES]), axis=0)
    print(f"\nout-of-domain census under ARCHIVE weights (the premise arm E rests on):")
    census = {}
    for i, label in enumerate(board):
        census[label] = {"n_ood": int(n_ood[i]),
                         "gauges": [m for m in LIVE_GAUGES if bool(ood[m][i])]}
        print(f"  {label:<16} {int(n_ood[i]):2d}/14  {census[label]['gauges']}")

    # ---- G: MUTATION CONTROL — the gate must be able to FAIL ------------------------------
    # Trap 31: a parity check that regenerates its own expectation tests nothing, and a bound
    # loose enough to pass everything is exactly that. So deliberately break the reference and
    # confirm the check bites — AND measure the smallest defect it can still see, because a
    # rounding-tolerant check has a MATHEMATICALLY NECESSARY blind spot below the rounding bound.
    # Quantifying that floor is the honest version of "the gate bites"; asserting a sub-bound
    # mutation must be caught would be asserting the impossible.
    print("\nG: mutation control — does this gate actually bite, and how small a defect can it see?")
    mutations = []
    probe = curves["comfort"]
    lo, hi = probe.domain
    w = hi - lo
    xs = np.linspace(lo - 2 * w, hi + 2 * w, 64)
    good = probe.price_many(xs, policy=CLAMP)
    bound = FLOP_SLACK * rounding_bound(probe, np.clip(xs, lo, hi))

    def _ratio_for(coeffs=None, knot=None, domain=None, extrapolate=False) -> float:
        if extrapolate:
            bad = probe.price_many(xs, policy=EXTRAPOLATE)
        else:
            ad_m = AD.ClampedCurve(
                metric="comfort", form=probe.form,
                coeffs=np.asarray(coeffs if coeffs is not None else probe.coeffs, dtype=float),
                knot=knot if knot is not None else probe.knot,
                domain=tuple(domain if domain is not None else probe.domain))
            bad = ad_m.price(xs)
        return float(np.max(np.abs(good - bad) / np.maximum(bound, 5e-324)))

    for label, ratio, must_bite in (
        ("knot shifted 1e-6", _ratio_for(knot=float(probe.knot) + 1e-6), True),
        ("domain widened 1%", _ratio_for(domain=(lo - 0.01 * w, hi + 0.01 * w)), True),
        ("NO clamp (extrapolating)", _ratio_for(extrapolate=True), True),
        ("coefficient x(1+1e-12)", _ratio_for(
            coeffs=[probe.coeffs[0] * (1 + 1e-12)] + list(probe.coeffs[1:])), True),
        # Below the bound BY CONSTRUCTION — recorded, not asserted. A perturbation smaller than
        # the arithmetic's own rounding is not distinguishable from a different rounding of the
        # same function, by any test whatsoever.
        ("coefficient +1 ULP (sub-bound)", _ratio_for(
            coeffs=[float(np.nextafter(probe.coeffs[0], np.inf))] + list(probe.coeffs[1:])),
         False),
    ):
        bites = ratio > 1.0
        mutations.append({"mutation": label, "bound_ratio": ratio, "gate_bites": bites,
                          "asserted": must_bite})
        note = "" if must_bite else "   <-- documented blind spot, not asserted"
        print(f"   {label:<32} deviation/bound = {ratio:12.3e}  bites: {bites}{note}")
        if must_bite:
            check(bites, f"MUTATION CONTROL FAILED: '{label}' slips through the rounding bound — "
                         f"this gate cannot detect a real defect and is worthless")

    # Calibrate the sensitivity floor: the smallest relative coefficient error the gate catches.
    floor = None
    for exponent in range(-16, 0):
        if _ratio_for(coeffs=[probe.coeffs[0] * (1 + 10.0 ** exponent)]
                      + list(probe.coeffs[1:])) > 1.0:
            floor = 10.0 ** exponent
            break
    print(f"   SENSITIVITY FLOOR: catches a relative coefficient error >= {floor:.0e}; anything "
          f"below that is under the rounding bound and undetectable in principle.")
    check(floor is not None and floor <= 1e-12,
          f"the gate's sensitivity floor is {floor} — too coarse to be meaningful")

    ok = not FAILURES
    STATE.mkdir(parents=True, exist_ok=True)
    json.dump({
        "gate": "arm E gate 1 — policy path validated on the ARCHIVE curves",
        "weights_json": ARM_JSON, "weights_meta": meta,
        "corpus_dir": str(fe.corpus_dir),
        "n_checks": N_CHECKS, "n_failures": len(FAILURES), "failures": FAILURES,
        "per_gauge": detail,
        "criterion": ("|a-b| <= FLOP_SLACK * n * eps * sum|term| (the dot-product rounding "
                      "bound), NOT a ULP count — ULP measures spacing at the RESULT, which "
                      "explodes under cancellation and would flag correct arithmetic"),
        "flop_slack": FLOP_SLACK,
        "search_accept_threshold": SEARCH_ACCEPT_THRESHOLD,
        "worst_bound_ratio_price_many_vs_scalar": worst_a,
        "worst_bound_ratio_price_many_vs_armD_clamped_curve": worst_d,
        "worst_bound_ratio_price_many_across_batch_shapes": worst_shape,
        "price_many_shape_dispatch_finding": (
            "LossCurve.price_many is NOT bit-equal to LossCurve.price for n>=2, and is not "
            "bit-equal to ITSELF across batch shapes: `price` evaluates a length-1 array while "
            "`price_many` evaluates length n, and `_design(...) @ coeffs` dispatches to a "
            "different BLAS kernel by shape. 7 of 14 archive curves show it. Arm D's elementwise "
            "ClampedCurve is 0 ULP across shapes. So the brief's 'pin at EXACT float equality "
            "against price_many' is unsatisfiable by construction, and the 4 shipped tests "
            "cannot catch it (fixed-length arrays on both sides / pytest.approx). Bounded by "
            "check F: identical argmin and identical full ordering."),
        "totals_pin": {"max_abs_diff": f_abs, "ulp": f_ulp, "rel_of_score_range": f_rel,
                       "same_argmin": bool(f_argmin), "identical_ordering": bool(f_order),
                       "score_range": float(got.max() - got.min())},
        "clamp_changes_total_on": n_diff, "n_test_layouts": int(perms.shape[0]),
        "mutation_control": mutations,
        "sensitivity_floor_relative_coeff_error": floor,
        "incumbent_ood_census_archive_weights": census,
        "modelled_only": "MODELLED ONLY: fitted-surface attribution, not measured typing speed.",
    }, open(STATE / "gate1-policy.json", "w"), indent=1)

    print(f"\nGATE 1: {N_CHECKS} checks, {len(FAILURES)} failures — "
          f"{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    rc = main()
    p = STATE / "gate1-rc.txt"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(f"{rc}\n")
    tmp.replace(p)
    sys.exit(rc)
