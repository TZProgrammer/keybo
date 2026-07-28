"""ARM H design probes, run BEFORE the prereg is written. These measure the HARNESS, not
the answer: eval throughput (to size the budget) and the batch-shape sensitivity of
`oxey-style` (my OBJECTIVE, so a "strict improvement" must exceed its own numerical noise --
OXEYFIX-1 recorded that oxey-style is summation-order sensitive in its last digits, and
MARGIN-GATE-1's rule is that a strict win needs a RESOLVABLE margin).

Also measures the STRUCTURAL question the prereg needs a number for: how much `oxey-style`
headroom survives once the six axes oxey RESTATES (trap 27) are held at arm B's level.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE))
import evobj as EV  # noqa: E402

ARMB = "flmpg-yuo,sntdcireahkxbwv'.jzq"
TARGET = "oxey-style"
OXEY_COMPONENTS = ("sfb", "lsb", "scissor", "imbalance", "redir", "alt")


def swap_neighbours(base: np.ndarray, k: int = 1) -> np.ndarray:
    """All layouts at EXACTLY one transposition from `base` (435 of them)."""
    pairs = [(i, j) for i in range(30) for j in range(i + 1, 30)]
    out = np.repeat(base[None, :], len(pairs), axis=0)
    for r, (i, j) in enumerate(pairs):
        out[r, i], out[r, j] = base[j], base[i]
    return out


def main() -> int:
    fe = EV.FastEval(corpus=None, weights_json=None, with_surface=True)
    assert str(Path(fe.corpus_dir).resolve()).startswith("/tmp/armh"), fe.corpus_dir
    out: dict = {}
    b = EV.perm_of(ARMB)

    # ---- 1. throughput at several batch sizes ----
    rng = np.random.default_rng(4242)
    tp = {}
    for n in (64, 435, 4096, 32768, 131072):
        pool = np.stack([np.concatenate([rng.permutation(30), [30]]).astype(np.int32)
                         for _ in range(n)])
        t0 = time.time()
        fe.gauges(pool)
        dt = time.time() - t0
        tp[n] = {"wall_s": dt, "evals_per_s": n / dt}
    out["throughput_single_process"] = tp

    # ---- 2. BATCH-SHAPE SENSITIVITY of every gauge, and of oxey-style especially ----
    # The same layout scored alone, in a pair, in a 435-batch, in a 100k batch. Any
    # difference is float summation order, and it BOUNDS the margin a strict win needs.
    nb = swap_neighbours(b)
    probes = {
        "alone": np.stack([b]),
        "with_1_other": np.stack([b, nb[0]]),
        "in_435": np.concatenate([np.stack([b]), nb[:434]]),
        "in_20000": np.concatenate([np.stack([b]), np.stack([
            np.concatenate([rng.permutation(30), [30]]).astype(np.int32)
            for _ in range(19_999)])]),
    }
    per_shape = {}
    for name, batch in probes.items():
        g = fe.gauges(batch)
        per_shape[name] = {k: float(g[k][0]) for k in
                           (TARGET, "_ms_per_char", "sfb", "comfort", "roll")}
    ref = per_shape["alone"]
    out["batch_shape_sensitivity"] = {
        "per_shape": per_shape,
        "max_abs_dev_from_alone": {
            k: max(abs(v[k] - ref[k]) for v in per_shape.values())
            for k in ref
        },
    }

    # ---- 3. determinism: same batch twice, bit-exact? ----
    g1 = fe.gauges(nb)
    g2 = fe.gauges(nb)
    out["repeat_same_batch_max_abs_diff"] = {
        k: float(np.max(np.abs(g1[k] - g2[k]))) for k in (TARGET, "_ms_per_char")}

    # ---- 4. THE 1-SWAP BALL AROUND ARM B, EXHAUSTIVE. Prereg input: does even ONE of the
    # 435 nearest neighbours satisfy the 13 hard constraints? (No result of the ARM H search
    # is used here -- this is frozen-geometry enumeration around a frozen layout.)
    from keybo.analysis.evidence_scorer import EXPECTED_SIGN, LIVE_GAUGES
    live = list(LIVE_GAUGES)
    con = [g for g in live if g != TARGET]
    gb = fe.gauges(np.stack([b]))
    ref_g = {g: float(gb[g][0]) for g in live}
    ref_ms = float(gb["_ms_per_char"][0])
    gn = fe.gauges(nb)
    dirs = {g: float(EXPECTED_SIGN[g]) for g in live}

    viol = np.zeros(len(nb))
    n_viol_axes = np.zeros(len(nb), dtype=int)
    per_axis_viol_count = {}
    for g in con:
        ex = dirs[g] * (gn[g] - ref_g[g])
        bad = ex > 1e-12
        per_axis_viol_count[g] = int(bad.sum())
        n_viol_axes += bad
        viol += np.maximum(ex, 0.0) / max(abs(ref_g[g]), 1e-9)
    ms_bad = gn["_ms_per_char"] > ref_ms + 1e-12
    feas13 = n_viol_axes == 0
    out["one_swap_ball"] = {
        "n": int(len(nb)),
        "n_satisfying_all_13_axes": int(feas13.sum()),
        "n_satisfying_13_axes_AND_ms_le_armB": int((feas13 & ~ms_bad).sum()),
        "n_with_ms_le_armB": int((~ms_bad).sum()),
        "min_n_violated_axes": int(n_viol_axes.min()),
        "hist_n_violated_axes": {int(k): int(v) for k, v in
                                 zip(*np.unique(n_viol_axes, return_counts=True),
                                     strict=True)},
        "per_axis_violation_count_of_435": per_axis_viol_count,
        "best_oxey_in_ball": float(gn[TARGET].min()),
        "armB_oxey": ref_g[TARGET],
        "n_with_oxey_below_armB": int((gn[TARGET] < ref_g[TARGET]).sum()),
    }
    # of those that improve oxey, how many axes do they break?
    imp = gn[TARGET] < ref_g[TARGET]
    if imp.any():
        out["one_swap_ball"]["among_oxey_improvers"] = {
            "n": int(imp.sum()),
            "min_n_violated_axes": int(n_viol_axes[imp].min()),
            "min_violation_sum": float(viol[imp].min()),
        }

    # ---- 5. THE STRUCTURAL PREREG NUMBER (trap 27): how much oxey headroom survives when
    # the six axes oxey RESTATES are all held at <= arm B? Measured on a large random pool
    # (no ARM H result involved).
    big = np.stack([np.concatenate([rng.permutation(30), [30]]).astype(np.int32)
                    for _ in range(200_000)])
    gg = fe.gauges(big)
    ok6 = np.ones(len(big), dtype=bool)
    for g in OXEY_COMPONENTS:
        ok6 &= dirs[g] * (gg[g] - ref_g[g]) <= 0
    ok13 = np.ones(len(big), dtype=bool)
    for g in con:
        ok13 &= dirs[g] * (gg[g] - ref_g[g]) <= 0
    out["structural_headroom_random_pool"] = {
        "n_pool": int(len(big)),
        "n_holding_6_oxey_components": int(ok6.sum()),
        "n_holding_all_13": int(ok13.sum()),
        "oxey_min_unconstrained": float(gg[TARGET].min()),
        "oxey_min_holding_6": (float(gg[TARGET][ok6].min()) if ok6.any() else None),
        "oxey_min_holding_13": (float(gg[TARGET][ok13].min()) if ok13.any() else None),
        "armB_oxey": ref_g[TARGET],
        "note": ("a RANDOM pool is far from the near-optimal band; this bounds nothing about "
                 "the search, it only shows whether holding oxey's own components pins oxey."),
    }

    json.dump(out, open(HERE / "design-probe.json", "w"), indent=1, sort_keys=True)
    print(json.dumps(out, indent=1, sort_keys=True))
    print(f"\nWROTE {HERE / 'design-probe.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
