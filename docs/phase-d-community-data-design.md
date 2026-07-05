# Phase D design: breaking the 4-layout family ceiling (DRAFT — user approval required before any outreach)

**Status: draft for review. Nothing here has been posted, shared, or built externally.**

## Why this is the decisive step

Every validation result so far lives inside the QWERTY-adjacent family (qwerty, azerty,
qwertz, dvorak — and dvorak carries most of the "different geometry" burden with 64
self-selected typists). "Our layout is best" is currently an extrapolation for any typist
of colemak, workman, graphite, or a from-scratch layout. One novel holdout family would
convert the central claim from extrapolation to measurement. The analysis machinery
(schema, LOLO harness, ceilings, CIs) is DONE — this is purely a data-acquisition problem.

## What the data must contain (requirements, from the pipeline's schema)

Per keystroke: press timestamp (ms), release timestamp (optional but wanted — hold),
the key's character, the expected text, a participant id, and the participant's layout.
That is exactly the Dhakal-dump shape our `process-data` already ingests — a collector
that emits the same TSV means ZERO new pipeline code.

Per participant: layout name + string (to build the char map), self-reported years on the
layout and rough WPM, keyboard form factor (row-staggered / ortho / split — ortho data is
a bonus that would also unlock OQ-6).

Volume target: ~15 minutes of transcription typing (≈15–20 standard sentences) per
participant; **10–20 participants per layout family** puts a new fold at dvorak-scale
(64 typists, ceiling .67) or better. Colemak alone plausibly reaches 30+ volunteers.

## Collection instrument (build plan, ~1–2 days)

A single static web page (no backend beyond a POST endpoint or even a "download your
file + submit" flow):
- shows sentences (reuse the Dhakal sentence set for comparability),
- records keydown/keyup with `performance.now()` timestamps,
- writes the exact `*_keystrokes.txt` TSV format client-side,
- collects the participant metadata line + explicit consent text,
- transmits nothing else (no IP logging beyond hosting defaults; data published openly
  in anonymized form — pids are sequential integers).

Hosting: GitHub Pages + a serverless collector (or manual file submission for v1).

## Recruitment (ALL user-gated; drafts only)

Communities that would plausibly volunteer (they run layout experiments recreationally):
r/KeyboardLayouts, the AKL (Alternate Keyboard Layouts) Discord, colemak.com forums,
r/Colemak, r/typing. Draft post (for the user to edit/approve/post — NOT posted):

> **Help validate a data-driven keyboard layout model (15 min of typing)**
> I've been building an open-source, measurement-first layout optimizer (keybo). Its
> model is trained on the 136M-keystroke Aalto dataset, but that data is ~99% QWERTY —
> so every cross-layout claim is validated only within the QWERTY family. If you type
> colemak / workman / graphite / semimak / anything else fluently, 15 minutes of
> transcription typing would let us test the model on a layout family it has NEVER seen.
> Data is anonymous (an integer id + your layout name + timing), published openly, and
> the analysis code + preregistered acceptance criteria are already public in the repo.

Preregistered analysis (to be entered in PREREGISTRATIONS.md before the first file
arrives): the collected family becomes a pure holdout; the shipped model (trained ONLY on
the Aalto data) predicts it; acceptance = ρ ≥ 0.8 × the new family's split-half ceiling
and τ consistent with the within-family ordering. Publishing the criteria BEFORE
collecting is what makes a pass decisive.

## The n=1 longitudinal self-test (companion protocol — user participation gated)

Written protocol (details in the repo when approved): baseline 3×15-min qwerty sessions;
switch to the candidate layout with a logger (the same web page works offline); 15 min/day;
prediction registered in advance: the practice-curve plateau should approach the model's
predicted % gain for the user's WPM band. Even n=1 tests the direction of the central
claim on the person the layout is actually for.

## What I can do without approval (none of it externally visible)

Build the collector page in the repo; write the preregistration entry; prepare the
anonymization + ingest script (Dhakal-format passthrough). What I cannot do: post the
recruitment text anywhere, stand up public hosting, or contact communities — those are
the user's calls.
