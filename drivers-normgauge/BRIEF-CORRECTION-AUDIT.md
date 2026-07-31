# normgauge: audit of the parent's two mid-turn brief corrections

Per the parent's standing instruction ("keep re-deriving any figure I hand you, treat my
verifications as claims"), I re-derived both corrections on disk myself. **One is confirmed. The
other contains its own wrong constant — the eighth instance of the campaign's signature failure,
and it is inside a correction issued to fix that very failure.**

---

## CORRECTION 1 (bigram direction channel) — CONFIRMED, and IRRELEVANT to my arm

Not audited in detail because **my work cannot touch it**: the normalized gauges are a pure
**trigram QAP objective** over the fitted `.standardized` surfaces
(`fit = Σ_t F[t]·S[p(t₀),p(t₁),p(t₂)]`). `grep -c oxey` returns **0** in all three of
`src/keybo/scoring/model_norm.py`, `drivers-normgauge/build_anchors.py`,
`drivers-normgauge/weight_evidence.py`. No bigram feature vector, no `oxey` weight, no roll or
redirect term enters any anchor, weight, or blend.

⚠ One place it DOES touch my output, and it is presentational only: `blend-report.json` reports the
shipped **15-gauge frame**, which includes `oxey-style`, `roll`, `sr-roll` and `redir`. Those are
**reported** columns, not inputs to the gauge or the weights, so the correction changes no number I
derived. My report already states the `oxey-style` values are freshly computed by today's code.

---

## CORRECTION 2 (data-frame counts) — ITS DIRECTION IS RIGHT, TWO OF ITS NUMBERS ARE NOT

### 🟢 Confirmed: the fold count is 4, not 11

My own scan (`/tmp/ng_rederive.py`, counting labels directly out of the TSVs):

| file | rows (ngram-filtered) | labels | labels are |
|---|---|---|---|
| `bistrokes31_v1.tsv` | **2,202** | **4** | azerty, dvorak, qwerty, qwertz |
| `tristrokes31_cond_v1.tsv` | **16,643** | **4** | same four |
| `bistrokes_community.tsv` | **5,445** | **12** | 12 `label@geometry#submitter` strings |
| `tristrokes_last_community.tsv` | **38,516** | **12** | same twelve |

So "3,098 rows / 11 layouts / 1 participant per layout" is indeed not a real frame, and the AALTO
generalization unit is **4**. **This strengthens rather than weakens my own arm's caveats**, exactly
as the parent said it would.

### 🔴 WRONG #1: "9 distinct participants" for COMMUNITY. It is **7**.

* **From the data:** distinct pids in `bistrokes_community.tsv` = **7** → `{200001…200007}`.
* **From the registered provenance** (`data/community/processed/ingest_report.json`): `pids` has
  exactly **7** entries — alite, andrewcastro, ddn, grzegorzkulesza, octahedron, richarddavison, vg.

**Mechanism, which is the useful part.** `9` is what you get from a naive
`label.rsplit("#", 1)[1]` over the 12 labels — it yields 9 distinct *strings*, three of which are
the **same person**:

```
grzegorzkulesza  ·  grzegorzkulesza+pseudo  ·  grzegorzkulesza+rareboost
```

`+pseudo` and `+rareboost` are **CORPUS TAGS appended after the submitter name**, not other people.
`data/community/README.md` says so explicitly: they are "ingested with corpus tags so non-natural
text never pools silently with natural text." So **9 counts label-variants, 7 counts humans.**

⚠ **The error direction matters and it is the dangerous one: 7 < 9, so every "too few independent
units" argument gets STRONGER.** Do not weaken such a claim on the strength of `9`.

Note this is *the same fact* my own AMENDMENT 1 A1.1 corrected in the opposite direction: my prereg
said "n=7 community participants" when the **4-label rowStagger training subset** carries only
**4** pids (200001, 200003, 200006, 200007). Both numbers are right for their own scope:
**7 = the whole community file · 4 = the subset the COMMUNITY surface was fitted on.** The scope,
not the count, is what has to travel with the number.

### 🔴 WRONG #2 (minor, but it is a retyped constant): the row counts are off by one

Correction 2 gives `bistrokes31_v1 rows=2201` and `tristrokes31_cond_v1 rows=16642`; I count
**2,202** and **16,643**. I checked for the obvious cause and it is **not** a header line — the
first line of `bistrokes31_v1.tsv` is a data row (`qwerty\t((-4, 3), (-5, 2))\twa\t93274\t…`). Most
likely a `wc -l`-minus-one habit. Immaterial to any conclusion, but it is exactly the class of
number the campaign has agreed to stop retyping.

---

## WHAT I CHANGED IN MY OWN WORK AS A RESULT

**Nothing in any anchor, weight, or blend.** My arm never used the "11 layouts / 3,098 rows /
1 participant per layout" frame, and never used a bigram feature or an `oxey` weight as an input. My
own participant counts were measured in-code from the stroke tables, not taken from the brief —
`n_participants` in `weight-evidence.json` is emitted by `held_out()`, which is why it already reads
**4** (training subset) and **55,404** (AALTO) rather than any briefed figure.

The audit does confirm one thing I should state more sharply in my report: **the AALTO side's 55,404
"participants" are not 55,404 independent units for generalization purposes** — the shipped surface
is fitted over **4 layouts**, so layout-level generalization rests on 4 folds. That makes my
AMENDMENT 2 caveat (the AALTO-side bootstrap interval is a *lower bound* on uncertainty) **more**
warranted, not less.
