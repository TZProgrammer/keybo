"""A layout: which character sits on which physical key.

:class:`Layout` assigns characters to the slots of a :class:`~keybo.geometry.Geometry`. It
is the mutable object the optimizer explores by swapping keys.

Swaps are tracked on an explicit stack so that an arbitrarily long sequence of swaps can be
undone one at a time in LIFO order. This fixes the original 3-opt corruption bug: the old
implementation remembered only the most recent swapped pair, so undoing a multi-swap move
silently left the board in a wrong-but-plausible permutation.
"""

from __future__ import annotations

import random

from keybo.geometry import Geometry, Position


class Layout:
    def __init__(self, chars: str, geometry: Geometry) -> None:
        if len(chars) != len(geometry.slots):
            raise ValueError(
                f"layout has {len(chars)} characters but geometry has {len(geometry.slots)} slots"
            )
        if len(set(chars)) != len(chars):
            raise ValueError("layout characters must be unique")

        self.geometry = geometry
        self._char_to_pos: dict[str, Position] = dict(zip(chars, geometry.slots, strict=True))
        self._pos_to_char: dict[Position, str] = {p: c for c, p in self._char_to_pos.items()}
        self._swaps: list[tuple[str, str]] = []

    # --- lookup ---------------------------------------------------------------------

    def pos(self, char: str) -> Position:
        """The physical position of ``char``."""
        return self._char_to_pos[char]

    def key_at(self, x: int, y: int) -> str | None:
        """The character at position ``(x, y)``, or ``None`` if nothing is there."""
        return self._pos_to_char.get((x, y))

    def as_dict(self) -> dict[str, Position]:
        """A copy of the character→position mapping (handy for tests and snapshots)."""
        return dict(self._char_to_pos)

    @property
    def chars(self) -> tuple[str, ...]:
        """Characters in canonical slot order."""
        return tuple(self._pos_to_char[slot] for slot in self.geometry.slots)

    # --- mutation -------------------------------------------------------------------

    def swap(self, k1: str, k2: str) -> None:
        """Exchange the positions of two characters, recording the move for undo."""
        p1 = self._char_to_pos[k1]
        p2 = self._char_to_pos[k2]
        self._char_to_pos[k1], self._char_to_pos[k2] = p2, p1
        self._pos_to_char[p1], self._pos_to_char[p2] = k2, k1
        self._swaps.append((k1, k2))

    def undo(self) -> None:
        """Reverse the most recent swap. Raises ``IndexError`` if there is none."""
        k1, k2 = self._swaps.pop()
        # A swap is its own inverse; re-apply it without re-recording.
        p1 = self._char_to_pos[k1]
        p2 = self._char_to_pos[k2]
        self._char_to_pos[k1], self._char_to_pos[k2] = p2, p1
        self._pos_to_char[p1], self._pos_to_char[p2] = k2, k1

    def random_swap(self, rng: random.Random) -> None:
        """Swap two distinct characters chosen by the given RNG (seedable)."""
        k1, k2 = rng.sample(list(self._char_to_pos), 2)
        self.swap(k1, k2)

    # --- rendering ------------------------------------------------------------------

    def render(self) -> str:
        """Render as rows (top to bottom), space-separated, matching the geometry."""
        rows: dict[int, list[tuple[int, str]]] = {}
        for (x, y), char in self._pos_to_char.items():
            if y > 0:
                rows.setdefault(y, []).append((x, char))
        lines = []
        for y in sorted(rows, reverse=True):
            chars = [c for _, c in sorted(rows[y], key=lambda xc: xc[0])]
            lines.append(" ".join(chars))
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"Layout({''.join(self.chars)!r})"
