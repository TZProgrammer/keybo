# OQ-6 — Do we ever need non-row-stagger geometry (ortho / column-stagger / thumb keys)?

**Status: 🟢 decided (no, for now) — recorded with the reopening condition.**

## The decision and its basis

Decided during the rewrite design (user-approved): the package models exactly one geometry,
`ROW_STAGGERED_30`, because that is the only geometry the training data contains.

- **Thumb/extra keys: zero signal.** Every participant pressed exactly one thumb key (space)
  in one place; there is no data about where thumb keys *should* go.
- **Ortho/column-stagger: computable but unvalidated.** Features would produce numbers for an
  ortho board, but every training sample is row-staggered — the predictions would be pure
  extrapolation with no way to check them. Confident-looking garbage.
- The constraint is honest: *the model can only be as good as its data.*

## What the design already provides for a future reopening

Geometry is a single value object (`keybo/geometry.py`) — positions, stagger offsets, finger
map, space position — and everything else queries it. Supporting a second geometry is
mechanically an "add another `Geometry` instance + plumb a `--geometry` flag" change; the hard
part is (and will remain) DATA, not code.

## Reopening condition (the definitive close is already done; this is the un-close trigger)

Reopen if and only if a dataset with per-key timings on a non-row-stagger physical board
becomes available (e.g. a Kiakl-style crowdsource from ergo-board users, or a lab study).
Then: add the new `Geometry`, retrain with a `geometry` tag per row, and extend the OQ-5
harness with leave-one-GEOMETRY-out — the same validation logic, one level up.
