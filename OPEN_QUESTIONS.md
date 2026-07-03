# Open Questions

Unresolved *conceptual/modeling* questions — things where the right answer isn't obvious and
a wrong choice quietly produces worse layouts. These are distinct from `TODO.md` (concrete
work items). Each question records why it matters, the positions, a current lean, and how we
would actually resolve it (usually an experiment, not an argument).

Status legend: 🔴 open · 🟡 leaning · 🟢 decided (move the decision to a design doc + TODO).

---

## OQ-1 🟡 Should n-gram frequency be a model FEATURE, or only an objective WEIGHT?

**The distinction.** Frequency plays two roles that are easy to conflate:
1. **Weight** in the objective: `fitness = Σ time(ngram) × freq(ngram)`. Uncontroversial — we
   optimize for text people actually type. Keep.
2. **Feature** of the time model: `time = model(geometry, freq, wpm)`. **This is the question.**

**Why it matters for optimization (not just fit).** A bigram's frequency (e.g. `th` ≈ 9.7M)
is a property of the *language* — fixed no matter where T and H sit. The model is trained on
QWERTY-family layouts, so it can't cleanly separate "fast because frequent (muscle memory)"
from "fast because of where it sits on QWERTY (geometry)." When the optimizer moves keys and
still feeds the fixed frequency, the learned "frequent → fast" bonus is partly a disguised
*QWERTY-geometry* bonus applied to a layout nobody has practiced — biasing rankings toward
layouts that keep frequent bigrams in QWERTY-favorable spots. Fitting the training data well
(high R²) is **not** the same as ranking novel layouts well, which is the actual goal.

**Positions.**
- *Keep it (it transfers):* if we optimize for a hypothetical fully-proficient user, they'd
  have muscle memory for their language's frequent bigrams wherever those bigrams sit — so
  "frequent → fast" is a property of (language + proficiency) and transfers to any mastered
  layout. Frequency is then a legitimate, even necessary, feature.
- *Drop it (weight only):* model `time = f(geometry, wpm)` as pure biomechanics; let frequency
  live only in the weight. Removes the train/serve *semantic* mismatch, avoids overfitting to
  trained layouts, and makes the objective corpus swappable per user without retraining.

**Current lean: 🟡 weight, not feature.** Two supporting points: (a) the freq *feature*
saturates at corpus scale (see OQ-2 / audit finding #6) — it barely differentiates layouts as
a feature while staying load-bearing as a weight; (b) internal inconsistency (OQ-3). Cost of
dropping: lower training R² and reliance on a "proficient user" idealization.

**How to resolve:** leave-one-layout-out validation (see OQ-5). Train with vs. without the
frequency feature; whichever ranks the held-out layout's known relative speed better wins.
This is an experiment, and it depends on the eval harness (TODO).

**Blocks:** the "proper" fix for audit finding #5 (constituent bg/sg frequency features).
Keep = join real corpus freqs into training; Drop = delete those features. Until decided, the
scorer feeds the training-time default (1.0) on both sides — consistent, inert, not skewed.

---

## OQ-2 🔴 The `freq` feature saturates and is dual-purpose — is that acceptable, or a smell?

`freq` is used **both** as a model input feature *and* as the summation weight. As a feature it
saturates: XGBoost was trained on keystroke-data occurrence counts (~1–500 in realistic rows),
so every real corpus bigram (thousands→millions) lands in the same top bin and the feature is
effectively constant across the corpus — it can't help rank layouts. Largely a facet of OQ-1;
if OQ-1 resolves to "weight only," this dissolves. If we keep frequency as a feature, we must
at least (a) put the *feature* and the *weight* on the same distribution (OQ-3) and (b) handle
the scale mismatch (log-transform? train on corpus-scale frequencies?).

---

## OQ-3 🔴 Which frequency distribution — and should the feature-freq match the weight-freq?

Today the model is **trained** with frequencies from the 136M keystroke *dataset* (what
test-takers typed) but **scored/weighted** with *iWeb* corpus frequencies — two different
distributions. And (your point) neither necessarily matches a given user's real typing (code,
chat, non-English). Sub-questions:
- If frequency is a feature (OQ-1), the training feature-freq and the scoring weight-freq
  **must** come from the same distribution or the model is applied off-distribution.
- Should the objective corpus be **user-configurable** (weight by *your* text: prose vs. code
  vs. another language)? This is arguably the highest-leverage "best layout for ME" feature.
- iWeb is licensed/not-freely-redistributable; the derived frequency files are committed, but a
  swappable-corpus feature needs a documented way to generate new ones.

---

## OQ-4 🔴 Is the objective (predicted typing time) the right thing to optimize at all?

The whole pipeline optimizes *predicted speed*. But the original paper itself notes comfort,
effort, and injury-avoidance may matter more to users, and that speed gains are marginal (~6%).
Should the objective be multi-term (speed + effort + comfort penalties like SFBs/scissors/
redirects as first-class costs), and/or Pareto rather than single-scalar? Related: the geometry
features (SFB, scissor, LSB, redirect) are computed but only enter via their learned effect on
*time* — some may deserve to be explicit *comfort* costs regardless of predicted time.

---

## OQ-5 🔴 How do we validate that a model ranks NOVEL layouts well (not just fits data)?

There is currently **no generalization test**. The data has only 4 layouts (all QWERTY-family),
so a model can fit them well and still misrank a novel layout. Proposed method:
**leave-one-layout-out** — train on 3 layouts, predict the 4th, check whether predicted ranking/
times match the held-out layout's observed times. This is the harness that actually answers
OQ-1 and grounds every "layout X is N% faster" claim. Until it exists, treat cross-layout
speed numbers as unvalidated. (Build item in TODO.)

---

## OQ-6 🔵 Do we ever need non-row-stagger geometry (ortho / column-stagger / thumb keys)?

Decided *for now* during the rewrite design: **no** — the data is 30-key row-staggered, so any
other geometry would produce unvalidated extrapolation, and thumb/key-count changes have zero
supporting data. Geometry is isolated behind one object so this is extensible later. Re-open
only if we collect data on other physical layouts. (Kept here as a recorded decision, not an
active question.)
