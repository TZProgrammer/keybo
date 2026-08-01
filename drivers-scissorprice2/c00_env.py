"""Gate 0: environment + evaluator verification, and VERIFY THE BRIEF'S FIELD SCISSOR VALUES.

My brief transcribed 12 field scissor values and told me to verify them myself. This does.
"""
import _env  # noqa: F401  (must be first)
import json

import boards
import fastgauge
from _env import ART

LAY = boards.FIELD


def main():
    fs, w1, w2 = _env.verify_evaluators(LAY)
    fg = fastgauge.FastGauges()

    # THE BRIEF'S TRANSCRIPTION -- verify, do not trust.
    brief = {
        "flagship-c3": 0.089, "lsb-sib": 0.107, "archive-1846": 0.110,
        "archive-1843": 0.120, "keybo-lsb": 0.143, "arm-A": 0.175,
        "keybo-c30m": 0.228, "BALL-1": 0.257, "arm-B": 0.257,
        "semimak": 0.428, "p16-balance": 0.482, "graphite": 0.517,
        "qwerty30m": 1.583,
    }
    measured, ms = {}, {}
    for n, s in LAY.items():
        p = fg.perm(s)
        measured[n] = fg.scissor_only(p)
        ms[n] = fs.ms_per_char(s)
    print("\n== field scissor: BRIEF vs MEASURED ==")
    print(f"{'layout':<14}{'brief':>9}{'measured':>11}{'|diff|':>10}{'ms/char':>11}")
    worst_brief = 0.0
    for n in sorted(measured, key=lambda k: measured[k]):
        b = brief.get(n)
        d = abs(b - measured[n]) if b is not None else float("nan")
        if b is not None:
            worst_brief = max(worst_brief, d)
        bs = f"{b:>9.3f}" if b is not None else f"{'--':>9}"
        print(f"{n:<14}{bs}{measured[n]:>11.4f}{d:>10.4f}{ms[n]:>11.4f}")
    print(f"worst |brief - measured| over the 13 transcribed: {worst_brief:.4f} "
          f"(brief quoted 3 dp, so <=0.0005 is exact agreement)")

    out = {
        "fasteval_worst_abserr_vs_card": w1,
        "fastgauge_worst_abserr_vs_pattern_shares": w2,
        "field_scissor_measured": measured,
        "field_ms_per_char": ms,
        "brief_transcription": brief,
        "worst_abs_diff_brief_vs_measured": worst_brief,
        "brief_transcription_verdict": (
            "EXACT to the 3 dp quoted" if worst_brief <= 5e-4 else "DISAGREES"
        ),
        "field_scissor_min_optimized": min(measured[b] for b in boards.OPTIMIZED),
        "field_scissor_max_optimized": max(measured[b] for b in boards.OPTIMIZED),
    }
    with open(ART + "/s00_env.json", "w") as f:
        json.dump(out, f, indent=1)
    print("\nwrote s00_env.json")


if __name__ == "__main__":
    main()
