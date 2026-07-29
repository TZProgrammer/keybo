"""DIRECTION-SWAP-1: the decisive whole-layout test of the user's proposed swap.

Q2 of the brief: if the models price the redirect, why did the optimizer not avoid it?
The cheap decisive test is to build the variant that makes the word ``you`` a monotone
roll and score BOTH through the shipped path, on every gauge.

The swap is derived, never typed: for a layout whose {y,u,o} sit on one hand in an order
that makes ``you`` a direction reversal, exchange the two characters whose exchange makes
the finger path monotone. That is a pure transposition of two slots, so the variant has
the same charset (hence every gauge remains scorable) and differs from its parent in
exactly two positions -- asserted here, because a swap that silently changed the charset
would make the two rows incomparable.

Everything is scored by calling ``keybo.cli.analyze.main`` with ``--json``: the SHIPPED
command, same corpus, same frame, no reimplementation.
"""

from __future__ import annotations

import contextlib
import io
import itertools
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

from keybo.analysis.kmstats import _KEYS, _is_redirect  # noqa: E402
from keybo.cli.__main__ import main as keybo_main  # noqa: E402
from keybo.cli.analyze import _EXTRA_NAMED  # noqa: E402
from keybo.geometry import ROW_STAGGERED_30 as G  # noqa: E402
from keybo.layouts import NAMED_LAYOUTS  # noqa: E402
from keybo.testkit import assert_module_under  # noqa: E402

REGISTRY = {**NAMED_LAYOUTS, **_EXTRA_NAMED}

#: The trigram whose feel the user is reporting on, and the word that actually carries mass.
WORD = "you"


def word_is_redirect(lay30: str, word: str = WORD) -> bool:
    """Whether ``word`` is a same-hand direction reversal on this layout (SHIPPED predicate)."""
    slot_of = {ch: i for i, ch in enumerate(lay30)}
    if not all(ch in slot_of for ch in word):
        return False
    return _is_redirect(*(_KEYS[slot_of[ch]] for ch in word))


def swap(lay30: str, x: str, y: str) -> str:
    """Exchange the slots of two characters. Derived; the result is asserted, not trusted."""
    i, j = lay30.index(x), lay30.index(y)
    chars = list(lay30)
    chars[i], chars[j] = chars[j], chars[i]
    out = "".join(chars)
    if set(out) != set(lay30):
        raise SystemExit(f"swap changed the charset: {lay30!r} -> {out!r}")
    if sum(a != b for a, b in zip(lay30, out, strict=True)) != 2:
        raise SystemExit(f"swap did not move exactly two slots: {lay30!r} -> {out!r}")
    return out


def repair_swap(lay30: str, word: str = WORD) -> tuple[str, str] | None:
    """The transposition among ``word``'s letters that stops ``word`` being a redirect.

    Returns the pair to exchange, or None if no single transposition of the word's own
    letters fixes it (so "swap two of these three keys" is not available on this layout).
    """
    if not word_is_redirect(lay30, word):
        return None
    for a, b in itertools.combinations(sorted(set(word)), 2):
        if not word_is_redirect(swap(lay30, a, b), word):
            return (a, b)
    return None


def analyze(specs: list[str], ref: str) -> dict:
    """Run the SHIPPED ``keybo analyze --json`` in-process and return its parsed output.

    Dispatches through ``keybo.cli.__main__.main`` -- the console-script entry point -- so
    the argument parser and subcommand dispatch under test are the ones users invoke.
    """
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = keybo_main(["analyze", *specs, "--ref", ref, "--json"])
    if rc != 0:
        raise SystemExit(f"shipped analyze exited {rc}")
    return json.loads(buf.getvalue())


def _fingers(lay30: str, word: str = WORD) -> str:
    slot_of = {ch: i for i, ch in enumerate(lay30)}
    return "->".join(G.finger(G.slots[slot_of[ch]][0]).name for ch in word)


def main() -> int:
    assert_module_under("keybo", REPO)
    affected = {n: s for n, s in sorted(REGISTRY.items()) if word_is_redirect(s)}
    print(f"layouts on which the word {WORD!r} is a same-hand redirect: {list(affected)}")

    pairs = []
    for name, lay in affected.items():
        fix = repair_swap(lay)
        if fix is None:
            print(f"  {name:14s} NO single transposition of y/u/o fixes it — skipping A/B")
            continue
        variant = swap(lay, *fix)
        pairs.append((name, lay, variant, fix))
        print(
            f"  {name:14s} swap {fix[0]}<->{fix[1]}: {_fingers(lay)} (redirect) "
            f"-> {_fingers(variant)} (roll)"
        )

    results = []
    for name, base, variant, fix in pairs:
        # Score BOTH in ONE shipped invocation, so corpus/frame/models are identical by
        # construction rather than by assumption.
        out = analyze([base, variant], ref="qwerty")
        rb, rv = out["rows"][base], out["rows"][variant]
        results.append(
            {
                "name": name,
                "base": base,
                "variant": variant,
                "swap": list(fix),
                "corpus": out["corpus"],
                "base_row": rb,
                "variant_row": rv,
            }
        )

        def ms(row):
            return row["time"]["ms_per_char"] if row.get("time") else None

        print(f"\n{name}: {base}  ->  {variant}   (swap {fix[0]}<->{fix[1]})")
        print(f"  ms/char       {ms(rb):.6f}  ->  {ms(rv):.6f}   delta {ms(rv) - ms(rb):+.6f}")
        for gauge in (
            "redir",
            "roll",
            "sr-roll",
            "alt",
            "sfb",
            "sfs",
            "lsb",
            "scissor",
            "oxey-style",
            "comfort",
            "imbalance",
        ):
            a, b = rb["gauges"].get(gauge), rv["gauges"].get(gauge)
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                print(f"  {gauge:<12}  {a:>10.5f}  ->  {b:>10.5f}   delta {b - a:+.5f}")

    dest = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "direction_swap_ab.json"
    dest.write_text(json.dumps({"word": WORD, "results": results}, indent=1))
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
