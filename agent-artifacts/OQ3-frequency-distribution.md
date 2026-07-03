# OQ-3 — Which frequency distribution? Should feature-freq match weight-freq? Per-user corpora?

**Status: 🔴 open — part decision, part small tooling.** Not blocking anything today (the freq
feature is inert per OQ-1/OQ-2), but it becomes load-bearing the moment OQ-1 resolves "keep",
and the per-user-corpus half is a product feature worth having regardless.

## The three sub-questions and best current answers

### (a) Feature-freq vs weight-freq distribution mismatch
🟢 VERIFIED: training freqs come from the 136M keystroke dataset's occurrence counts;
scoring weights come from iWeb. Different sources, wildly different scales (~1–500 vs 10³–10⁷).
**Answer:** if OQ-1 = weight-only, the mismatch disappears (no feature). If keep: the feature
must be drawn from the same distribution at train and serve — practically, join iWeb counts
onto training rows (log-scaled), which the schema already makes possible since rows carry the
ngram string.

### (b) Which corpus should the WEIGHT use?
iWeb is a reasonable general-English default and its derived counts are already committed.
But the weight literally defines "what typing we optimize" — and (user's point) no fixed
corpus matches a specific person typing code, chat, or another language.
**Answer: make the weight swappable (it already is, mechanically)** — `--bigram-freqs`/
`--trigram-freqs` accept any file. What's missing is a supported way to GENERATE a personal
frequency file. That's a small, high-value tool: `keybo corpus-freqs <textfile(s)>` that
counts n-grams over a user-supplied text sample (their git commits, chat logs, prose) and
writes the same `ngram<TAB>count` format. 🟠 INFERRED: for programmers the divergence from
iWeb is large (symbols, camelCase patterns), so this likely changes optimal layouts
materially — worth an experiment (below).

### (c) Licensing / reproducibility of the committed iWeb-derived files
The derived counts are committed; iWeb itself is licensed and non-redistributable. Document
clearly (README) that these are derived aggregates, and that `corpus-freqs` gives users a
clean-room path to their own files. 🟡 The derived-counts-in-repo situation is common practice
for n-gram stats but worth an explicit note rather than silence.

## Definitive close

1. **(a)** closes with OQ-1's experiment (weight-only → moot; keep → implement the join and
   assert train/serve freq ranges overlap in a schema test).
2. **(b)** closes by building `corpus-freqs` (small: read text, slide 2/3-char windows over
   the allowed charset, count, write TSV) + one experiment: optimize a layout under iWeb
   weights vs. under a code-heavy corpus's weights; report layout diff and predicted-time
   delta of each layout under the *other* corpus. If the optimal layouts differ materially
   (they will 🟠), per-user corpora graduate from nice-to-have to a headline feature.
3. **(c)** closes with a README paragraph — no experiment needed.
