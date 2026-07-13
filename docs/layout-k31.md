# K31 — the apostrophe joins the keyset (2026-07-13)

The optimization space is now 31 movable keys: the 30-key block plus the ANSI quote
slot (home row, right pinky's outer column). This levels the structural convention
semimak and graphite already use — `'` on a real key, a rare letter exiled to the
leftover slot — which had been costing our layouts on every apostrophe-bearing
community corpus (~6–8% of our oxeylyzer-2 score). Full migration trail:
PREREGISTRATIONS.md 2542bc4 → ed3f267 → 270d492 (gates A–F, all passed: 30-key
features bit-identical, v5 data exactly reproduced + 220k apostrophe occurrences,
LOLO taus 1.0, rho/ceiling 1.0135).

## The headline structural finding

**Every K31 search arm — including the speed-only arm with no community term —
voluntarily pulls `'` into the 30-key block and exiles a rare letter (z or j) to
the quote slot.** The community convention is speed-optimal on our measured-time
surface too. The K31 speed-only winner is essentially P11-w0.5 with `'` adopted
and `z` exiled, and it is *faster* than P10-w0.5's K31 embedding.

## The two K31 candidates

**P15-balance** (five-gauge min-max regret pick, `E10-r888203`):

```
f y u , .   v d p n l
h i e a o   c s t r m    j on the quote slot
k / ; ' q   g w b x z
```

**P15-speed** (speed-only arm, `SPD-r888202`):

```
g c d l k   . , y o u
s r t h m   p n i e a    z on the quote slot
q x w b v   f ' j ; /
```

Notable: the balance pick puts the vowel block on the LEFT hand (`hieao` home-left)
— the first pick in the campaign to break the `naei`-right invariant; the K31
degrees of freedom are enough to flip the mirror.

## Board (K31 surface, exact tools; genkey sees the 30-block only — convention noted)

| gauge | P10-w0.5+' | P13STAB-win+' | P14-coopt+' | **P15-balance** | **P15-speed** |
|---|---|---|---|---|---|
| speed vs qwerty31 (wpm 90) | +3.66% | +3.52% | +3.49% | +3.67% | **+3.75%** |
| worst community axis | 12.2% | 9.0% | 7.8% | **3.4%** | 23.3% |
| genkey Score (↓) | 33.7 | 31.0 | **30.9** | 33.6 | 43.3 |
| oxeylyzer-1 (repl, ↑) | 0.333* | 0.367* | 0.400* | **0.420** | 0.258 |
| oxeylyzer-2 (repl, ↑) | −260.8B* | −245.1B* | −238.5B* | **−213.3B** | −320.6B |
| keymeow sfb / lsb | 1.18 / 0.60 | **1.07 / 0.09** | 1.23 / 0.53 | 1.33 / 1.93 | 1.85 / 0.41 |

*30-key-era repl numbers (their K31 embeddings pin ' to the quote slot, which is
exactly what those measurements assumed).

Reading: on the K31 surface the old 30-key incumbents lose their speed edge (the
apostrophe traffic they park on the pinky-outer slot now counts), and P15-balance
strictly beats all three on speed *and* worst-community-axis simultaneously —
the first candidate to do that. Its costs: genkey (33.6, worse than P13win/P14)
and keymeow sfb/lsb. P15-speed maximizes the surface but abandons community
metrics entirely (worst axis 23%).

## Status

Both are **candidates**, not deliverables. Adopting K31 as the published keyset —
retiring 30-key P10-w0.5 as the primary — is a one-way decision (it changes what a
"keybo layout" is) and sits with the owner. Deferred work if adopted: F5M-LR
quality surface retrain on K31; P13-style stability study on the K31 plateau;
genkey/keymeow 31-key extensions (their models are 3×10).
