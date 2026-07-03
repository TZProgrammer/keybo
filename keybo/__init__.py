"""keybo — a data-driven keyboard layout optimizer.

The package is organized around a few plug-and-play seams:

- ``keybo.geometry``  — the physical board (key positions, finger map, distances).
- ``keybo.layout``    — a character-to-position assignment over a geometry.
- ``keybo.features``  — the single n-gram feature pipeline shared by data processing,
                        training, and scoring (guards against train/serve skew).
- ``keybo.models``    — ``TypingModel`` implementations (features -> predicted time).
- ``keybo.scoring``   — ``IScorer`` implementations (layout -> fitness).
- ``keybo.optimize``  — ``IOptimizer`` implementations (simulated annealing, local search).
- ``keybo.data``      — corpus + keystroke dataset loading/processing.
- ``keybo.training``  — fitting and tuning typing models.
- ``keybo.cli``       — thin command-line entry points.
"""

__version__ = "0.1.0"
