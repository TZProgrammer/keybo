"""The optimizer seam."""

from __future__ import annotations

from abc import ABC, abstractmethod

from keybo.layout import Layout
from keybo.scoring.base import IScorer


class IOptimizer(ABC):
    """Searches for a layout minimizing a scorer's fitness, returning the best found."""

    @abstractmethod
    def optimize(self, layout: Layout, scorer: IScorer) -> Layout:
        """Optimize ``layout`` in place and return the best layout found."""
