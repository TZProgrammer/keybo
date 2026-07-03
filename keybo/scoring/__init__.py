"""Scorers: turn a layout into a single fitness number (lower = faster to type)."""

from keybo.scoring.base import IScorer
from keybo.scoring.model_scorer import BigramModelScorer, TrigramModelScorer

__all__ = ["BigramModelScorer", "IScorer", "TrigramModelScorer"]
