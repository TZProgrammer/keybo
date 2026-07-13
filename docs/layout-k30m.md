# K30M — the matched community charset (2026-07-13)

Per owner direction, the K31 exile structure is retired as a product frame (a letter
pushed off the main block is still a letter, typed worse) in favor of **matching
semimak/graphite's charset exactly**: 26 letters + `' , . -` on the standard 30
slots, with `;` and `/` off the layout. Every comparison against the community
entries is now same-charset, same-tools, no embeddings. Trail: PREREGISTRATIONS.md
6dc4727 → d34b250. Models: the K31-trained surfaces (position-based, strictly more
data; no retrain needed). Objective coverage *rises* ~0.75pp vs the classic charset
(`'`+`-` carry 10× the corpus mass of `;`+`/`).

## The honest apples-to-apples result

With the charset finally matched, the claim the project can make is:

**Our layouts hold a ~1.0–1.4pp measured-typing-speed advantage over semimak and
graphite, at community scores between dvorak and the community frontier — but
semimak and graphite remain the better pure community-metric layouts.**

| gauge | semimak | graphite | **P16-balance** | **P16-speed** |
|---|---|---|---|---|
| speed vs qwerty30M (wpm 90) | +2.79% | +2.61% | +3.75% | **+3.97%** |
| genkey Score (↓) | **27.7** | 29.5 | 30.8 | 43.3* |
| oxeylyzer-1 (repl, ↑) | 0.365 | **0.460** | 0.415 | 0.294 |
| oxeylyzer-2 (repl, ↑) | **−190.4B** | −199.1B | −234.1B | −312.9B |
| keymeow sfb / lsb | **0.89** / 1.27 | 1.23 / 0.57 | 1.29 / 1.27 | 2.44 / 0.89 |

*speed numbers restated from the pick's regret table (speed_regret over the pool
best, qwerty30M reference). P16-speed's community numbers are unconstrained.

The earlier K31 result that appeared to beat semimak on oxeylyzer-2 came from a
dof-convention difference (what sits pinned on the quote slot), which this frame
eliminates — recorded in the outcome block, and a good example of why the matched
charset was worth building.

## The two K30M candidates

**P16-balance** (`E10-r888303`, five-gauge min-max regret 6.38% — best in a pool
where P13STAB-win* scores 9.6% and semimak 30.9% due to its speed deficit):

```
f r l w g   ' u y o k
s n t d c   . i e a h
v x m p b   , - j q z
```

**P16-speed** (`SPD-r888301`, the C30M speed frontier, +0.07pp over the P10-w0.5
embedding):

```
k o y u ,   v d m n l
h e i a p   c s t r f
q j - . '   g w b x z
```

Structural note: **the left-hand vowel home block recurs** — P16-speed puts
`h e i a` on the left home with `c s t r` on the right, and P16-balance mirrors
the same split (`. i e a h` home-right block reversed). The K31 vowel-side flip
was not an extra-key artifact after all: on both new frames the search finds
left-vowel and right-vowel arrangements speed-equivalent, so vowel side is part
of the degenerate plateau, not an invariant. (The classic-charset P10 family's
`naei`-right was one basin, not the optimum.)

## Status

Candidates, not deliverables. The charset decision (classic 30 with `;`/`/` vs
C30M with `'`/`-`) defines what a keybo layout is and sits with the owner. C30M
arguments: +0.75pp objective coverage, exact community comparability, `'` far more
frequent than `;`. Classic arguments: `;`/`/` are real programming keys; the
published P10-w0.5 lineage is classic.
