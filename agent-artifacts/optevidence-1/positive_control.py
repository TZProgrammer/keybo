"""POSITIVE CONTROL for the fast evaluator — must pass before any search runs.

Three gates, each asserted to a stated tolerance:

  G1  every one of the 14 gauges matches `GaugeContext.vector` (the analyzer's own path)
  G2  the evidence score and its out-of-domain set match `EvidenceWeights.score_layout`
  G3  ms/char matches `TimeSurface.card(...).ms_per_char`

Plus a MUTATION check (trap 31): a deliberately corrupted kernel must make G1 FAIL. A gate
that cannot fail is not a gate.

Run: uv run --no-sync python positive_control.py
Writes an rc sentinel (trap 1: absence of a sentinel is NOT a pass).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
import evobj as EV  # noqa: E402

from keybo.analysis.evidence_scorer import LIVE_GAUGES, gauge_context  # noqa: E402
from keybo.analysis.timecard import TimeSurface  # noqa: E402
from keybo.data.corpus import load_frequencies, production_corpus_dir  # noqa: E402

ARM = "/local/home/zegertho/agent/state/evidence-scorer/artifacts/arm-random400-native.json"
OUT = Path("/local/home/zegertho/agent/state/optevidence/artifacts")

INCUMBENTS = {
    "keybo-lsb": "pyuo,vgdnlhiea.cstrmkj-z'fwbxq",
    "lsb-sib": "fyou,vgdnlheaikcstrmzj'.-pwbxq",
    "archive-1843": "pyou,vgdnmheai.cstlrjz'k-fwbxq",
    "archive-1846": "pyou,vgdnmheai.cstrlkq'z-fbwjx",
    "keybo-lsb+lm": "pyuo,vgdnmhiea.cstrlkj-z'fwbxq",
}
PROBES = {
    "qwerty30m": EV.C30M,
    "graphite": "bldwz'foujnrtsgyhaeixqmcvkp,.-",
    "semimak": "flhvz'wuoysrntkcdeaixjbmqpg,.-",
    **INCUMBENTS,
}


def build_evidence_weights():
    """Reconstruct an `EvidenceWeights` from the frozen arm JSON — the reference scorer."""
    from keybo.analysis.evidence_scorer import EvidenceWeights, LossCurve

    w = json.load(open(ARM))["weights"]
    curves = {}
    for g in w["weights"]:
        curves[g["metric"]] = LossCurve(
            metric=g["metric"], form=g["form"], coeffs=list(g["coeffs"]), knot=g["knot"],
            domain=tuple(g["valid_domain"]), observed_range=tuple(g["observed_range"]),
            weight=g["weight_ms_per_unit"], weight_ci=tuple(g["weight_ci95"]),
            r2=g["r2"], r2_linear=g["r2_linear"], mean_abs_shap=g["mean_abs_shap_ms"],
            shap_share_pct=g["shap_share_pct"],
        )
    return EvidenceWeights(
        source=w["source"], frame=w["surface_frame"], corpus=w["corpus"],
        corpus_sha256=w["corpus_sha256"], surface_sha256=w["surface_sha256"],
        n_layouts=w["n_layouts"], pool_label=w["pool"], curves=curves,
        clusters={k: v["members"] if isinstance(v, dict) and "members" in v else v
                  for k, v in w["clusters"].items()},
        cluster_shap_share_pct={}, cluster_weight={},
        effective_dof=w["effective_dof"],
        surrogate_r2_in_sample=w["surrogate_r2_in_sample"],
        surrogate_r2_holdout=w["surrogate_r2_holdout"],
        base_value=w["base_value_ms_per_trigram"], notes=w["notes"],
    )


def main() -> int:
    t0 = time.time()
    failures: list[str] = []

    # ---- FRAME ASSERTION (the brief's #1 trap) ----
    arm = json.load(open(ARM))
    assert arm["weights"]["surface_frame"] == "native", arm["weights"]["surface_frame"]
    assert arm["surface_frame"] == "native", arm["surface_frame"]
    print(f"frame asserted: native | source={arm['weights']['source']} | pool={arm['weights']['pool']}")

    directory = production_corpus_dir(None)
    print(f"corpus: {directory}")
    from keybo.data.corpus import corpus_identity
    ident = corpus_identity(directory)
    for name, sha in arm["weights"]["corpus_sha256"].items():
        assert ident["sha256"][name] == sha, f"corpus {name} sha mismatch: {ident['sha256'][name]} != {sha}"
    print("corpus sha256 matches the arm JSON on all 4 tables ✓")

    print(f"\nbuilding FastEval ... ", end="", flush=True)
    fe = EV.FastEval(corpus=None, weights_json=ARM, with_surface=True)
    print(f"{time.time() - t0:.1f}s")

    ctx = gauge_context(None)
    weights = build_evidence_weights()
    trigrams = load_frequencies(str(directory / "trigrams.txt"))
    surface = TimeSurface(trigrams, target_wpm=90.0)

    names = list(PROBES)
    perms = np.stack([EV.perm_of(PROBES[n]) for n in names])
    fast = fe.gauges(perms)
    fast_score = fe.evidence_score(fast)
    fast_ood = fe.out_of_domain(fast)

    print(f"\n=== G1: 14 gauges vs GaugeContext.vector (tolerance 1e-9 relative) ===")
    worst = {g: 0.0 for g in LIVE_GAUGES}
    for i, name in enumerate(names):
        ref = ctx.vector(PROBES[name])
        for g in LIVE_GAUGES:
            rel = abs(fast[g][i] - ref[g]) / max(abs(ref[g]), 1e-12)
            worst[g] = max(worst[g], rel)
    for g in LIVE_GAUGES:
        flag = "✓" if worst[g] < 1e-9 else "✗ FAIL"
        print(f"  {g:<11s} max rel err {worst[g]:.3e}  {flag}")
        if worst[g] >= 1e-9:
            failures.append(f"G1 {g} rel={worst[g]:.3e}")

    print(f"\n=== G2: evidence score + out-of-domain vs EvidenceWeights.score_layout ===")
    for i, name in enumerate(names):
        ref = weights.score_layout(PROBES[name], ctx)
        d = abs(fast_score[i] - ref["score"])
        ood_fast = {g for g in LIVE_GAUGES if fast_ood[g][i]}
        ood_ref = set(ref["out_of_domain"])
        ok = d < 1e-9 and ood_fast == ood_ref
        print(f"  {name:<14s} fast={fast_score[i]:+10.6f} ref={ref['score']:+10.6f} "
              f"|d|={d:.2e}  ood_fast={sorted(ood_fast)} ood_ref={sorted(ood_ref)} "
              f"{'✓' if ok else '✗ FAIL'}")
        if not ok:
            failures.append(f"G2 {name} d={d:.2e} ood {sorted(ood_fast)} vs {sorted(ood_ref)}")

    print(f"\n=== G3: ms/char vs TimeSurface.card (tolerance 1e-9 relative) ===")
    for i, name in enumerate(names):
        card = surface.card(PROBES[name])
        rel = abs(fast["_ms_per_char"][i] - card.ms_per_char) / card.ms_per_char
        ok = rel < 1e-9
        print(f"  {name:<14s} fast={fast['_ms_per_char'][i]:.6f} ref={card.ms_per_char:.6f} "
              f"rel={rel:.3e} coverage={card.coverage_pct:.2f}%  {'✓' if ok else '✗ FAIL'}")
        if not ok:
            failures.append(f"G3 {name} rel={rel:.3e}")

    # ---- MUTATION check (trap 31): the gate must be able to fail ----
    print(f"\n=== MUTATION: corrupt one kernel cell, G1 must FAIL ===")
    saved = fe.KB[0, 0 * EV.NS + 1]
    fe.KB[0, 0 * EV.NS + 1] = saved + 1.0
    mutated = fe.gauges(perms)
    ref0 = ctx.vector(PROBES[names[0]])
    rel_mut = abs(mutated["sfb"][0] - ref0["sfb"]) / max(abs(ref0["sfb"]), 1e-12)
    fe.KB[0, 0 * EV.NS + 1] = saved
    bites = rel_mut >= 1e-9
    print(f"  mutated sfb rel err = {rel_mut:.3e}  -> gate {'BITES ✓' if bites else 'DID NOT BITE ✗ FAIL'}")
    if not bites:
        failures.append("MUTATION did not bite — G1 cannot fail")
    restored = fe.gauges(perms)
    assert abs(restored["sfb"][0] - ref0["sfb"]) < 1e-12, "restore failed"

    # ---- SPEED ----
    print(f"\n=== speed ===")
    rng = np.random.default_rng(0)
    for B in (256, 2048):
        batch = np.stack([np.concatenate([rng.permutation(30).astype(np.int32), [30]]) for _ in range(B)])
        t = time.time()
        g = fe.gauges(batch)
        fe.evidence_score(g)
        dt = time.time() - t
        print(f"  B={B:5d}: {dt:6.3f}s -> {B / dt:9.1f} evals/s/core")

    # ---- reproduce the arm's own `scored` block, if present ----
    print(f"\n=== cross-check vs the arm JSON's own `scored` block ===")
    for row in json.load(open(ARM)).get("scored", []):
        lay = row["layout"]
        if len(lay) != 30 or set(lay) != set(EV.C30M):
            print(f"  {row['name']:<14s} skipped (not C30M)")
            continue
        p = EV.perm_of(lay)[None]
        s = float(fe.evidence_score(fe.gauges(p))[0])
        d = abs(s - row["score"])
        ok = d < 1e-9
        print(f"  {row['name']:<14s} fast={s:+10.6f} json={row['score']:+10.6f} |d|={d:.2e} "
              f"{'✓' if ok else '✗ FAIL'}")
        if not ok:
            failures.append(f"ARM-JSON {row['name']} d={d:.2e}")

    rc = 1 if failures else 0
    print(f"\n{'=' * 70}")
    if failures:
        print(f"FAILURES ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
    else:
        print("ALL GATES PASSED — fast evaluator is exact on 8 layouts x 14 gauges + score + ms/char,")
        print("and the mutation check confirms the gate can fail.")
    print(f"elapsed {time.time() - t0:.1f}s")
    (OUT / "positive-control-rc.txt").write_text(str(rc) + "\n")
    return rc


if __name__ == "__main__":
    sys.exit(main())
