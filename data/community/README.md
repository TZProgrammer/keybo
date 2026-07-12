# Community typing data (Kiakl collection)

Community-contributed monkeytype keystroke captures, collected via the "Submit your
Kiakl times" form. Ingested 2026-07-12 under the KIAKL-INGEST registration
(PREREGISTRATIONS.md) — dedup, validity, labeling, and wpm rules were fixed there
BEFORE processing.

## Layout

```
raw/        the untouched uploads (zips + the loose GK json)
processed/  production-schema stroke TSVs + ingest_report.json
```

Regenerate `processed/` at any time:

```
python -m keybo.data.community_cli <extracted-raw-dir> data/community/processed
```

(`keybo/data/community.py` holds the ingestion logic + rules; the raw zips are the
source of truth — `processed/` is derived and reproducible.)

## What's in it (ingest_report.json has exact numbers)

- **7 submitters**, ~3.3k deduped sessions, ~530k keystroke events, ~450k valid
  bigram samples across **8 distinct layouts** (colemak-dh, a recurva variant,
  colemak, an mtgap variant, qwerty, and 3 unidentified customs), on three
  physical geometries (rowStagger / ortholinear / angleMod).
- Source schema: per-event `{key, interval(ms, press-to-press), correct}`,
  session-level layout string + keyboardType. **No release timestamps** (hold=-1),
  no per-test text.
- Layout labels are `<name>@<geometry>#<submitter>` — layout, physical geometry,
  and typist are kept visible; nothing is pooled silently.
- Community pids start at 200001 (disjoint from aalto participant ids).

## Known confounds (recorded at ingestion; see KIAKL-INGEST)

- Layout ⇄ typist confounded: most layouts have exactly one submitter.
- Ortholinear/angleMod geometry differs from the ROW_STAGGERED_30 the feature
  schema assumes; those labels carry the geometry tag for downstream filtering.
- Tiny volume vs the aalto dataset (~0.4% of its samples); enthusiast population;
  monkeytype word-mode text.
- A few degenerate short sessions produce absurd session-wpm (max ~3400); the
  standard 40–140 wpm cell bucketing excludes them downstream.
- `raw/gk_typingdataColemkaDH_ortholinear.json` duplicates in-form-zip content
  (sessionID-verified); two form jsons are empty lists.
- `raw/gk_typingdata.zip` holds 136 NEW sessions from the same submitter:
  `typingdata0003.json` (colemak-dh, pseudo-words → label tag `+pseudo`) and
  `typingdata1278.json` (colemak-dh, rare-char-boosted words → `+rareboost`) are
  ingested with corpus tags so non-natural text never pools silently with natural
  text. `typingdata.json` (qwerty, capture version 1.1.0) is **unusable**: that
  capture version masks key identity (key ∈ {0,2,3} category codes, `correct`
  always false) — no ngram extraction is possible; it drops out via the wpm>0
  rule and is recorded here so nobody re-mines it.

Any MODEL use of this data (LODO extension, QIN certification, practice terms,
cross-layout validation) requires its own preregistration.
