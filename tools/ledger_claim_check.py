#!/usr/bin/env python3
"""Check PREREGISTRATIONS.md's git claims against the actual tree.

WHY THIS EXISTS
---------------
Twice in one session a ledger entry was cited to a subagent as if its code were
reachable from ``origin/main``, and twice it was not:

* ``RULER-1`` (``7354ebe``) — the commit message says "gauge objective wired",
  but the commit changes only ``PREREGISTRATIONS.md``. ``git grep gauge_objective
  origin/main -- src/`` returns nothing.
* ``A11`` — recorded CLOSED on branch ``wire-invariance-guard`` (``e735b02``),
  which is not an ancestor of ``origin/main``.

Both false premises reached a subagent brief and cost real work. A grep over
agent state dirs for "ledger only" / "commit message says" hits 29 workspaces,
so this is a recurring class, not two accidents.

WHAT IT CHECKS
--------------
1. **Reachability.** Every SHA the ledger names must resolve, and is reported as
   reachable-from-origin/main or not.
2. **Disclosure.** A SHA that is NOT reachable is only a DEFECT if the ledger
   presents it as landed. An entry that says "local only", "unmerged",
   "unpushed" and so on is *correctly* disclosing a local branch — that is this
   project's normal state, since landing is a human gate. So an undisclosed
   unreachable SHA is the finding; a disclosed one is fine.
3. **Wiring claims.** A line claiming a symbol is "wired"/"wired into" must have
   that symbol present in ``origin/main``'s ``src/`` — unless the same line
   discloses that it is local.

The exit code is 0 unless an UNDISCLOSED unreachable SHA or an undisclosed
wiring claim is found, so this can gate CI later without flagging the ~100
correctly-disclosed local references that already exist.

FIRST-RUN RESULT, AND AN HONEST LIMIT (2026-08-01)
--------------------------------------------------
Run over the 10,700-line ledger: 112 SHAs reachable, 54 local-and-correctly-
disclosed, 5 unresolvable, 11 flagged. **Every one of the 11 was audited by hand
and NONE is a real defect** — they are *provenance* references ("the defect lives
in ``79cb175``", "pinned at ``011dd41``", "the prerequisite was cherry-picked
``f6c4ba7``"), which name a commit without claiming it is on mainline.

So the SHA-reachability half of this tool has a **high false-positive rate and did
NOT catch the two failures that motivated it.** ``RULER-1`` was caught by the
*wiring* half in spirit (``gauge_objective`` absent from ``origin/main``'s
``src/``) — and note that check currently reports 0 because the ledger discloses
those entries correctly; it is the *brief* citing them that lost the caveat, and
this tool does not read briefs.

Kept anyway, with the limit stated, because:

* the **wiring check** is precise (symbol-in-tree is unambiguous) and cheap;
* the **unresolvable** list is genuinely useful — 5 SHAs name commits that exist
  in no clone here, so those citations are unverifiable by anyone;
* the disclosure-vocabulary audit it forced is itself the finding: this ledger
  discloses locality in at least a dozen different phrasings.

**Do not treat a clean run as evidence that a ledger claim is true.** The real
guard for the motivating failure is checking a SHA *at the moment you cite it in
a brief*, which is a process step, not a batch job.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

#: Phrases that disclose "this is not on mainline". Matched case-insensitively
#: against the line carrying the SHA (and the line before it, since entries wrap).
_DISCLOSURE = (
    "local only",
    "local, unmerged",
    "local + unmerged",
    "localonly",
    "unmerged",
    "unpushed",
    "not pushed",
    "nothing pushed",
    "stays local",
    "stays LOCAL",
    "remains local",
    "local branch",
    "ledger only",
    "ledger-only",
    "never landed",
    "did not land",
    "not on main",
    "is not on",
    "no upstream",
    "branch local",
    # Disclosure vocabulary this project actually uses, found by running the
    # checker and auditing its first-run false positives one by one. A bare
    # "(local)" and "USER-GATED" are real disclosures; missing them made the
    # tool over-flag, which is the failure mode that gets a checker ignored.
    "(local",
    "user-gated",
    "user gated",
    "human's call",
    "humans call",
    "the human's",
    "stays the human",
    "remains the human",
    "not landed",
    "not merged",
    "awaiting",
    "pending",
)

#: A line asserting something was wired/landed into the shipped tree.
_WIRING = re.compile(r"\bwired\b|\bis now WIRED\b|\bwire[sd]? into\b", re.IGNORECASE)

_SHA = re.compile(r"`([0-9a-f]{7,40})`")

#: Symbols worth cross-checking when a line claims wiring. Extend as the ledger grows.
_WIRING_SYMBOLS = {
    "gauge_objective": "gauge-objective",
    "sg_dist": "sg_dist",
    "skipgram_span": "skipgram_span",
    "lateral_span": "lateral_span",
    "three_opt": "three-opt",
    "all_distinct": "all_distinct",
}


def _git(*args: str) -> tuple[int, str]:
    proc = subprocess.run(
        ["git", *args], capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1]
    )
    return proc.returncode, proc.stdout.strip()


def _resolves(sha: str) -> bool:
    return _git("cat-file", "-e", f"{sha}^{{commit}}")[0] == 0


def _reachable(sha: str, ref: str) -> bool:
    return _git("merge-base", "--is-ancestor", sha, ref)[0] == 0


def _symbol_in_ref_src(symbol: str, ref: str) -> bool:
    return _git("grep", "-q", symbol, ref, "--", "src/")[0] == 0


def _disclosed(context: str) -> bool:
    low = context.lower()
    return any(p.lower() in low for p in _DISCLOSURE)


def check(ledger: Path, ref: str) -> int:
    lines = ledger.read_text(errors="replace").splitlines()
    unresolved: list[tuple[int, str]] = []
    undisclosed: list[tuple[int, str]] = []
    disclosed_local = 0
    reachable = 0
    wiring_bad: list[tuple[int, str, str]] = []

    for i, line in enumerate(lines, start=1):
        # Two lines of context: entries wrap, so a disclosure can sit on either.
        context = "\n".join(lines[max(0, i - 2) : i + 1])

        for sha in _SHA.findall(line):
            if not _resolves(sha):
                unresolved.append((i, sha))
            elif _reachable(sha, ref):
                reachable += 1
            elif _disclosed(context):
                disclosed_local += 1
            else:
                undisclosed.append((i, sha))

        if _WIRING.search(line):
            for symbol, _label in _WIRING_SYMBOLS.items():
                if (
                    symbol in line
                    and not _symbol_in_ref_src(symbol, ref)
                    and not _disclosed(context)
                ):
                    wiring_bad.append((i, symbol, line.strip()[:90]))

    print(f"ledger: {ledger}   ref: {ref}")
    print(f"  SHAs reachable from {ref}      : {reachable}")
    print(f"  SHAs local + CORRECTLY disclosed: {disclosed_local}")
    print(f"  SHAs that do not resolve at all : {len(unresolved)}")
    print(f"  SHAs UNREACHABLE + UNDISCLOSED  : {len(undisclosed)}   <- defects")
    print(f"  wiring claims not in {ref} src/ : {len(wiring_bad)}   <- defects")

    if unresolved:
        print("\nUNRESOLVED (SHA names no commit in this clone):")
        for ln, sha in unresolved:
            print(f"  line {ln}: {sha}")
    if undisclosed:
        print(f"\nUNDISCLOSED UNREACHABLE (reads as landed, is not on {ref}):")
        for ln, sha in undisclosed:
            print(f"  line {ln}: {sha}")
    if wiring_bad:
        print(f"\nWIRING CLAIMED BUT SYMBOL ABSENT FROM {ref} src/:")
        for ln, symbol, snippet in wiring_bad:
            print(f"  line {ln}: {symbol}  |  {snippet}")

    return 1 if (undisclosed or wiring_bad) else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--ledger",
        default=str(Path(__file__).resolve().parents[1] / "PREREGISTRATIONS.md"),
    )
    ap.add_argument("--ref", default="origin/main")
    args = ap.parse_args()
    return check(Path(args.ledger), args.ref)


if __name__ == "__main__":
    sys.exit(main())
