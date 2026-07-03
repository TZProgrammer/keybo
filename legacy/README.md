# Legacy code (pre-rewrite)

This directory holds the original `keybo` scripts as they existed before the 2026-07
rewrite. They are kept for reference only. **Nothing in `keybo/` imports from here.**

The rewrite (see `docs/specs/2026-07-03--rewrite-design.md`) replaced these with a proper
package. Notable differences and the reasons the old files were retired:

- `classifier.py` — `Keyboard` + `Classifier`; split into `keybo.geometry`, `keybo.layout`,
  and the classifier predicates folded into `keybo.features`.
- `scorer.py` — `BigramXGBoostScorer` / `TrigramXGBoostScorer` / `FreyaScorer`. Replaced by
  `keybo.scoring`. `FreyaScorer` (the paper's hand-tuned cost function) is pre-fork legacy
  and was not carried forward.
- `wpm_conditioned_model.py`, `trigram_model.py` — two divergent copies of the feature +
  training code. Unified into `keybo.features` (one pipeline) and `keybo.training`.
- `optimizer.py`, `simulated_annealing.py`, `two_opt.py`, `three_opt.py` — folded into
  `keybo.optimize`.
- `process_dataset.py` — rebuilt as `keybo.data.keystrokes` (the old version crashed against
  its own `Keyboard` class).
- `print_stats.py`, `hyperparameter_tuning.py`, `utils.py` — replaced by `keybo.cli.score`,
  `keybo.training.tune`, and `keybo.data.corpus`.
- `trigram_model.ipynb` — exploratory notebook using the old cost-function fitting.
- `Graph_Utils/` — one-off plotting/color scripts.
- `bigram_model.pkl`, `trigram_model.pkl` — old pickled models; not loadable by the new
  code (which uses versioned JSON artifacts) and intentionally not preserved.
