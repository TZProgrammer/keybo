"""F5 WARM-START CROSS-SEEDING -- the conservative-direction falsifier.

Every cap is re-seeded from EVERY other cap's incumbent board (repair the seed down under this
cap, then re-descend). This can only LOWER F_hat at the caps where the cold search converged
worse, and hence only SHRINK the estimated price. **This gate cut PRICEBAND-1's own sfb estimate
2.2x (+1.9636 -> +0.9022)**, so the warm number -- not the cold one -- is the headline.

Also seeds from the 14 FIELD boards, which is legitimate here (unlike in the cold frontier,
where uniform effort per cap is required): warm-starting is applied to EVERY cap from the SAME
donor pool, so no cap gets a private head start.

Usage: c08_warm.py <cold-frontier-tag> <out-tag>
"""
import _env  # noqa: F401
import json
import sys
import time

import boards
import fastgauge
import numpy as np
import search
from _env import ART


class Obj(search.Objective):
    def scissor(self, p):
        return self.fg.scissor_only(p[:30])


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "s04"
    out_tag = sys.argv[2] if len(sys.argv) > 2 else "s08"
    fr = json.load(open(f"{ART}/{tag}_frontier.json"))
    band = [float(c) for c in fr["caps_band"]]
    inert = [float(c) for c in fr["caps_inert"]]
    caps = band + inert
    R = len(fr["reps"])

    fs, w1, w2 = _env.verify_evaluators(boards.FIELD)
    fg = fastgauge.FastGauges()
    obj = Obj(fs, fg)
    gauge = lambda q: fg.scissor_only(q[:30])  # noqa: E731

    # ---- the DONOR POOL: every cold incumbent from every (rep, cap) + the 14 field boards ----
    donors = []
    for r in range(R):
        for k, v in fr["reps"][r].items():
            if v["perm"] is not None:
                donors.append(np.array(v["perm"], dtype=np.intp))
    for n, s in boards.FIELD.items():
        donors.append(fs.perm(s))
    print(f"== F5 warm cross-seeding: {len(donors)} donors x {len(caps)} caps, R={R} replicates ==")

    out = {"tag_cold": tag, "n_donors": len(donors), "caps_band": band,
           "caps_inert": [str(c) for c in inert], "R": R,
           "fasteval_worst": w1, "fastgauge_worst": w2, "reps": []}
    t_all = time.time()
    for rep in range(R):
        rep_out = {}
        # Replicate-specific donor SUBSET (disjoint-ish blocks) so replicates stay independent
        # rather than all collapsing onto the identical warm optimum.
        rng = np.random.default_rng(555000 + rep)
        idx = rng.permutation(len(donors))[: max(8, len(donors) // 2)]
        sub = [donors[i] for i in idx]
        for c in caps:
            t0 = time.time()
            best, bp = np.inf, None
            for d in sub:
                p = d.copy()
                p, feas = search.drive_under_cap(obj, p, c, gauge=gauge)
                if not feas:
                    continue
                p, m = search.cap_two_opt(obj, p, c, gauge=gauge)
                if gauge(p) > c + 1e-9:
                    continue
                if m < best:
                    best, bp = float(m), p.copy()
            # 3-opt polish the warm incumbent, as in the cold arm
            if bp is not None:
                bp, best = search.cap_three_opt(obj, bp, c, gauge=gauge)
                best = float(best)
            rep_out[str(c)] = {
                "best_ms": None if bp is None else float(best),
                "scissor_at_best": None if bp is None else float(gauge(bp)),
                "perm": None if bp is None else [int(x) for x in bp],
                "n_donors_used": len(sub),
                "sec": time.time() - t0,
            }
            r_ = rep_out[str(c)]
            cold = fr["reps"][rep][str(c)]["best_ms"]
            print(f"  rep{rep} cap={c:<6} warm={r_['best_ms']:.4f} cold={cold:.4f} "
                  f"delta={r_['best_ms']-cold:+.4f} scis={r_['scissor_at_best']:.4f} "
                  f"{r_['sec']:.0f}s", flush=True)
        out["reps"].append(rep_out)
        with open(f"{ART}/{out_tag}_warm.json", "w") as f:
            json.dump(out, f, indent=1)
        print(f"  -- warm replicate {rep} done, {time.time()-t_all:.0f}s", flush=True)
    out["total_sec"] = time.time() - t_all
    with open(f"{ART}/{out_tag}_warm.json", "w") as f:
        json.dump(out, f, indent=1)
    with open(f"{ART}/{out_tag}_DONE", "w") as f:
        f.write("0\n")
    print(f"\nwrote {out_tag}_warm.json in {out['total_sec']:.0f}s")


if __name__ == "__main__":
    main()
