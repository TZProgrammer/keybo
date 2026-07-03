"""The scorer seam: a layout in, a fitness number out."""

from __future__ import annotations

from abc import ABC, abstractmethod

from keybo.layout import Layout


class IScorer(ABC):
    """Scores a layout. Lower fitness means faster predicted typing of the corpus."""

    @abstractmethod
    def fitness(self, layout: Layout) -> float:
        """Return the layout's fitness (total predicted typing time over the corpus)."""
