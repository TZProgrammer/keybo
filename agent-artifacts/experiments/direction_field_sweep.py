"""DIRECTION-FIELD-1: how common is a high-frequency trigram placed as a redirect?

Q4 of the brief, generalized past ``you``: for every layout in the registry plus the six
adoption candidates, take the top-N highest-frequency corpus trigrams and ask how much
corpus mass lands in an order the community's own predicate calls a direction reversal --
AND how much of that mass a single transposition of the trigram's own letters could move
out of the redirect class. The second quantity is what makes it actionable: a redirect
whose repair is unavailable is not a defect the optimizer could have avoided by a swap.

Also decomposes the whole-layout A/B of DIRECTION-SWAP-1 into the term the user is
talking about and the terms they are not, because the whole-layout ms/char delta is a SUM
over 114,920 trigrams and the word ``you`` is one of them.

All predicates and tables are the shipped ones (``kmstats``, ``timecard.TimeSurface``).
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

from keybo.analysis.kmstats import _KEYS, _is_redirect, _is_roll  # noqa: E402
from keybo.analysis.timecard import TimeSurface  # noqa: E402
from keybo.cli.analyze import _EXTRA_NAMED  # noqa: E402
from keybo.data.corpus import load_frequencies, production_corpus_dir  # noqa: E402
from keybo.layouts import NAMED_LAYOUTS  # noqa: E402
from keybo.testkit import assert_module_under  # noqa: E402

#: The six layouts the campaign has on the adoption table, from the parent's frozen set.
#: Strings are pasted from artifacts and VERIFIED here (charset + length) rather than trusted.
CANDIDATES = {
    "armB": "flmpg-yuo,sntdcireahkxbwv'.jzq",
    "BALL-1": "flmpg-yuo,sntcdireahkxbwv'.jzq",
    "MID": "flmpg.yuo,sntcdireahkxbwv'-jzq",
    "armH-hdln": "flmpg-,uoysntcdireahkxvwb.'jzq",
}

C30M = "qwertyuiopasdfghjkl'zxcvbnm,.-"
CLASSIC = "qwertyuiopasdfghjkl;zxcvbnm,./"


def _validate(name: str, lay: str) -> str:
    """Length + distinctness are FATAL; the charset is reported, not enforced.

    dvorak is deliberately neither C30M nor CLASSIC (it carries both ``;`` and ``'`` and
    lacks ``-``) -- ``cli/analyze`` documents that and renders its charset-dependent cells
    N/A rather than refusing the layout. The gauges this sweep uses (kmstats predicates,
    the measured-keystroke surface) are charset-agnostic, so refusing dvorak here would
    drop a real row for no reason. What must NOT pass silently is a mistyped string, which
    is what the length and distinctness checks catch.
    """
    if len(lay) != 30:
        raise SystemExit(f"{name}: length {len(lay)}, expected 30")
    if len(set(lay)) != 30:
        dupes = sorted({c for c in lay if lay.count(c) > 1})
        raise SystemExit(f"{name}: not a permutation; repeated {dupes}")
    if set(lay) == set(C30M):
        return "C30M"
    if set(lay) == set(CLASSIC):
        return "CLASSIC"
    return "other"


def tri_class(lay30: str, tri: str) -> str | None:
    """kmstats class of an ordered character trigram on a layout, or None if unscorable."""
    slot_of = {ch: i for i, ch in enumerate(lay30)}
    if not all(ch in slot_of for ch in tri):
        return None
    a, b, c = (_KEYS[slot_of[ch]] for ch in tri)
    if _is_redirect(a, b, c):
        return "redir"
    if _is_roll(a, b, c):
        return "sr-roll" if a.row == b.row == c.row else "roll"
    if a.hand != b.hand and a.hand == c.hand:
        return "alt"
    return "other"


def repairable(lay30: str, tri: str) -> tuple[str, str] | None:
    """A transposition of two of the trigram's OWN letters that clears the redirect."""
    if tri_class(lay30, tri) != "redir":
        return None
    for x, y in itertools.combinations(sorted(set(tri)), 2):
        chars = list(lay30)
        i, j = chars.index(x), chars.index(y)
        chars[i], chars[j] = chars[j], chars[i]
        if tri_class("".join(chars), tri) != "redir":
            return (x, y)
    return None


def sweep(lay30: str, top: list[tuple[str, int]]) -> dict:
    """Redirect mass among the top corpus trigrams, and how much of it is 1-swap repairable."""
    seen = redir = rep = 0
    worst: list[dict] = []
    for tri, freq in top:
        cls = tri_class(lay30, tri)
        if cls is None:
            continue
        seen += freq
        if cls == "redir":
            redir += freq
            fix = repairable(lay30, tri)
            if fix:
                rep += freq
            worst.append({"tri": tri, "freq": freq, "repair": list(fix) if fix else None})
    worst.sort(key=lambda r: -r["freq"])
    return {
        "scored_mass": seen,
        "redirect_mass": redir,
        "redirect_pct": 100.0 * redir / seen if seen else 0.0,
        "repairable_mass": rep,
        "repairable_pct": 100.0 * rep / seen if seen else 0.0,
        "top_redirects": worst[:10],
    }


def main() -> int:
    assert_module_under("keybo", REPO)
    registry = {**NAMED_LAYOUTS, **_EXTRA_NAMED, **CANDIDATES}
    charsets = {name: _validate(name, lay) for name, lay in registry.items()}
    print(
        f"validated {len(registry)} layouts (length 30, all-distinct); charsets: "
        + ", ".join(f"{k}={v}" for k, v in sorted(charsets.items()) if v == "other")
        + " (rest C30M/CLASSIC)"
    )

    tri_freqs = load_frequencies(str(production_corpus_dir(None) / "trigrams.txt"))
    letters = set("abcdefghijklmnopqrstuvwxyz")
    # Letter-only trigrams: a trigram containing space or punctuation is not a
    # three-finger same-hand run in the sense the user is describing.
    top = sorted(
        ((t, f) for t, f in tri_freqs.items() if len(t) == 3 and set(t) <= letters),
        key=lambda kv: -kv[1],
    )[:200]
    print(f"top {len(top)} letter-only trigrams, mass {sum(f for _, f in top):,}")
    print(f"  highest: {', '.join(f'{t}={f:,}' for t, f in top[:8])}")

    rows = {name: sweep(lay, top) for name, lay in sorted(registry.items())}
    print(
        f"\n{'layout':14s} {'redir % of top-200':>19s} {'1-swap repairable %':>20s}  worst redirect"
    )
    for name, r in sorted(rows.items(), key=lambda kv: -kv[1]["redirect_pct"]):
        w = r["top_redirects"][0] if r["top_redirects"] else None
        tag = (
            f"{w['tri']!r} ({w['freq']:,})" + (" fixable" if w and w["repair"] else "")
            if w
            else "-"
        )
        print(f"{name:14s} {r['redirect_pct']:18.3f}% {r['repairable_pct']:19.3f}%  {tag}")

    # --- decomposition of the whole-layout A/B ------------------------------------------
    print("\nbuilding TimeSurface for the decomposition...")
    surface = TimeSurface(tri_freqs, target_wpm=90.0)
    decomp = {}
    for name, base, var in (
        ("keybo-lsb", _EXTRA_NAMED["keybo-lsb"], "pyou,vgdnlhiea.cstrmkj-z'fwbxq"),
        ("BALL-1", CANDIDATES["BALL-1"], "flmpg-you,sntcdireahkxbwv'.jzq"),
    ):
        _validate(f"{name}-variant", var)
        if sum(a != b for a, b in zip(base, var, strict=True)) != 2:
            raise SystemExit(f"{name}: variant is not a 2-slot transposition of the base")
        cb, cv = surface.card(base), surface.card(var)
        # per-trigram signed contribution, so the total delta can be attributed
        sb = {ch: i for i, ch in enumerate(base)}
        sv = {ch: i for i, ch in enumerate(var)}
        sb[" "] = sv[" "] = surface._n - 1
        contrib = []
        for tri, freq in surface.tri.items():
            try:
                a1, b1, c1 = sb[tri[0]], sb[tri[1]], sb[tri[2]]
                a2, b2, c2 = sv[tri[0]], sv[tri[1]], sv[tri[2]]
            except KeyError:
                continue
            d = (
                (surface._T2[a2, b2] + surface._Tc[a2, b2, c2])
                - (surface._T2[a1, b1] + surface._Tc[a1, b1, c1])
            ) * freq
            if d:
                contrib.append((tri, freq, d))
        contrib.sort(key=lambda r: r[2])
        total = sum(d for _, _, d in contrib)
        you = next((r for r in contrib if r[0] == "you"), None)
        decomp[name] = {
            "base": base,
            "variant": var,
            "base_ms_per_char": cb.ms_per_char,
            "variant_ms_per_char": cv.ms_per_char,
            "delta_ms_per_char": cv.ms_per_char - cb.ms_per_char,
            "delta_total_ms": total,
            "you_delta_total_ms": you[2] if you else None,
            "you_freq": you[1] if you else None,
            "n_trigrams_moved": len(contrib),
            "best_10": [(t, f, d) for t, f, d in contrib[:10]],
            "worst_10": [(t, f, d) for t, f, d in contrib[-10:]],
        }
        print(f"\n{name}: {base} -> {var}")
        print(
            f"  ms/char {cb.ms_per_char:.6f} -> {cv.ms_per_char:.6f}  ({cv.ms_per_char - cb.ms_per_char:+.6f})"
        )
        print(f"  total delta {total:+,.0f} ms over {len(contrib):,} trigrams that moved")
        if you:
            print(
                f"  'you' alone: freq {you[1]:,}  delta {you[2]:+,.0f} ms "
                f"({100 * you[2] / total if total else float('nan'):+.1f}% of the total delta)"
            )
        print("  biggest IMPROVEMENTS:", ", ".join(f"{t}{d:+,.0f}" for t, _, d in contrib[:5]))
        print("  biggest REGRESSIONS:", ", ".join(f"{t}{d:+,.0f}" for t, _, d in contrib[-5:]))

    dest = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "direction_field_sweep.json"
    dest.write_text(json.dumps({"field": rows, "decomposition": decomp}, indent=1))
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
