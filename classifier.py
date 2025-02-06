"""
classifier.py – Keyboard layout and classifier utilities.

This module defines:
  - Keyboard: A class that represents a keyboard layout, supports key swaps,
    and computes key positions with row offsets.
  - Classifier: A helper class that loads trigram frequency data and provides
    various keyboard-related predicates.
  - A test() function to quickly inspect the keyboards.
"""

from math import atan2, degrees
from random import sample
from itertools import product
from typing import Dict, List, Set, Tuple, Optional
import numpy as np
import os


class Keyboard:
    """
    Represents a keyboard layout.

    The layout string should contain 60 characters: the first 30 for lowercase
    keys (arranged in three rows of 10) and the next 30 for their uppercase versions.
    Row numbering follows the convention used in feature extraction:
      - Top row: row 3
      - Middle row: row 2
      - Bottom row: row 1
    An extra key (the space) is added at position (0, 0).

    Row offsets (used in geometric calculations) are stored in a dictionary.
    """

    def __init__(
        self,
        data_size: int,  # not used beyond bookkeeping
        layout: str = "qwertyuiopasdfghjkl;zxcvbnm,./QWERTYUIOPASDFGHJKL:ZXCVBNM<>?",
    ) -> None:
        self.data_size = data_size
        # Row offsets (for rows 1, 2, 3) as used in the feature extraction code.
        # These match the offsets used in get_bistroke_features.
        self.row_offsets: Dict[int, float] = {1: 0.5, 2: 0.0, 3: -0.25}
        self.chars: str = layout
        self.swap_pair: Tuple[str, str] = ("", "")
        self.affected_indices = range(data_size)
        self.key_count: int = 30

        # The first 30 characters are considered lowercase keys.
        self.lowercase: str = layout[:30]
        self.uppercase: str = layout[30:]
        self.lower_to_upper: Dict[str, str] = dict(zip(self.lowercase, self.uppercase))
        self.upper_to_lower: Dict[str, str] = dict(zip(self.uppercase, self.lowercase))

        # Map x-coordinate (the key’s horizontal position) to a finger name.
        self.x_to_finger: Dict[int, str] = {
            5: "lp",
            4: "lr",
            3: "lm",
            2: "li",
            1: "li",
            -1: "ri",
            -2: "ri",
            -3: "rm",
            -4: "rr",
            -5: "rp",
        }

        # Build a mapping from key to (x, y) position.
        self.key_to_pos: Dict[str, Tuple[int, int]] = {}
        # Create three rows of 10 keys each from the lowercase layout.
        rows: List[str] = [self.lowercase[i * 10 : (i + 1) * 10] for i in range(3)]
        # Row numbering: top row is row 3, then row 2, then row 1.
        for y_index, row in enumerate(rows):
            row_number = 3 - y_index
            for x_index, k in enumerate(row):
                # Shift x positions so that keys in the left half are negative.
                new_x: int = x_index - 5 if x_index < 5 else x_index - 4
                self.key_to_pos[k] = (new_x, row_number)
        # Add the space key at (0, 0).
        self.key_to_pos[" "] = (0, 0)

        # Build the inverse mapping.
        self.pos_to_key: Dict[Tuple[int, int], str] = {
            pos: key for key, pos in self.key_to_pos.items()
        }

    def __repr__(self) -> str:
        """
        Returns a string representation of the current (lowercase) key layout.
        The keys are arranged by row (top to bottom) and sorted by their x position.
        """
        rows = {}
        for key, (x, y) in self.key_to_pos.items():
            if y > 0:  # only display letter keys (exclude space)
                rows.setdefault(y, []).append((x, key))
        result_lines = []
        # Sort rows by row number descending (top first)
        for row_num in sorted(rows.keys(), reverse=True):
            # Sort keys by x coordinate
            row_keys = [key for x, key in sorted(rows[row_num], key=lambda tup: tup[0])]
            result_lines.append(" ".join(row_keys))
        return "\n".join(result_lines)

    def get_ngrams(self, n: int) -> Set[str]:
        """
        Generate all n-grams from the layout, optionally filtered by a current swap.
        If a nonempty swap is set, only include n-grams that contain one of the swapped keys.
        """
        ngrams = set()
        for swap in self.swap_pair:
            for combo in product(self.chars, repeat=n):
                if swap == "" or swap in combo:
                    ngrams.add("".join(combo))
        return ngrams

    def undo_swap(self) -> None:
        """Undo the current swap (by swapping the same keys again)."""
        self.swap(*self.swap_pair)

    def random_swap(self) -> None:
        """Randomly choose two distinct keys from the lowercase keys and swap them."""
        k1, k2 = sample(list(self.lowercase), 2)
        self.swap(k1, k2)

    def swap(self, k1: str, k2: str) -> None:
        """
        Swap the positions of keys k1 and k2.
        The swap_pair attribute is updated accordingly.
        """
        self.swap_pair = (k1, k2)
        pos1 = self.key_to_pos.get(k1)
        pos2 = self.key_to_pos.get(k2)
        if pos1 is None or pos2 is None:
            raise ValueError(
                f"Cannot swap keys '{k1}' and '{k2}'; one or both keys not found."
            )
        self.key_to_pos[k1], self.key_to_pos[k2] = pos2, pos1
        self.pos_to_key[pos1], self.pos_to_key[pos2] = k2, k1

    def get_key(self, x: int, y: int) -> Optional[str]:
        """Return the key at position (x, y), or None if not found."""
        return self.pos_to_key.get((x, y))

    def get_pos(self, k: str) -> Tuple[int, int]:
        """
        Return the (x, y) position of key k.
        If k is an uppercase letter, return the position of its lowercase equivalent.
        """
        if k in self.uppercase:
            k = self.upper_to_lower.get(k, k)
        pos = self.key_to_pos.get(k)
        if pos is None:
            raise ValueError(f"Key '{k}' not found in layout.")
        return pos

    def get_col(self, k: str) -> int:
        """Return the x-coordinate (column) of key k."""
        return self.get_pos(k)[0]

    def get_finger(self, k: str) -> str:
        """Return the finger (as a string code) used for key k."""
        x = self.get_pos(k)[0]
        try:
            return self.x_to_finger[x]
        except KeyError:
            raise ValueError(f"No finger mapping found for key '{k}' at x = {x}.")

    def get_row(self, k: str) -> int:
        """Return the y-coordinate (row) of key k."""
        return self.get_pos(k)[1]

    def get_hand(self, k: str) -> int:
        """
        Return a numeric indicator for the hand used to type key k.
        For nonzero x positions, return the sign (+1 for right, -1 for left).
        For keys at x == 0 (e.g. the space), return 0.
        """
        x = self.get_pos(k)[0]
        if x == 0:
            return 0
        return int(x / abs(x))


class Classifier:
    """
    Provides helper methods for keyboard-based feature analysis.

    On initialization, this class reads trigram frequency data from 'trigrams.txt'
    (only considering trigrams made up entirely of a set of valid lowercase characters)
    and builds several keyboard layouts.
    """

    def __init__(self, kb: str = "qwerty") -> None:
        trigrams: List[str] = []
        tg_freqs: List[int] = []
        tg_percentages: Dict[int, int] = {}
        valid_chars = set("qwertyuiopasdfghjkl'zxcvbnm,.-")

        trigram_path = "trigrams.txt"
        if not os.path.exists(trigram_path):
            raise FileNotFoundError(f"Cannot find file: {trigram_path}")

        with open(trigram_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                trigram, freq_str = parts[:2]
                if all(c in valid_chars for c in trigram):
                    trigrams.append(trigram)
                    try:
                        tg_freqs.append(int(freq_str))
                    except ValueError:
                        tg_freqs.append(0)

        total_count = sum(tg_freqs)
        elapsed = 0
        # Build a percentage-to-index mapping.
        for i, freq in enumerate(tg_freqs):
            percentage = int(100 * (elapsed / total_count)) if total_count else 0
            tg_percentages[percentage + 1] = i
            elapsed += freq

        # Use a coverage cutoff (100% by default)
        tg_coverage = 100
        cutoff = tg_percentages.get(tg_coverage, len(tg_freqs))
        tg_freqs = np.array(tg_freqs[:cutoff])
        trigrams = trigrams[:cutoff]
        print("Processed trigram data: using", len(trigrams), "trigrams.")

        data_size = len(trigrams)
        # Build several common keyboard layouts.
        self.keyboards: Dict[str, Keyboard] = {
            "qwerty": Keyboard(
                data_size=data_size,
                layout="qwertyuiopasdfghjkl;zxcvbnm,./QWERTYUIOPASDFGHJKL:ZXCVBNM<>?",
            ),
            "azerty": Keyboard(
                data_size=data_size,
                layout="azertyuiopqsdfghjkl;wxcvbnm,./AZERTYUIOPQSDFGHJKL:WXCVBNM<>?",
            ),
            "dvorak": Keyboard(
                data_size=data_size,
                layout="',.pyfgcrlaoeuidhtns;qjkxbmwvzPYFGCRL?+|AOEUIDHTNS:QJKXBMWVZ",
            ),
            "qwertz": Keyboard(
                data_size=data_size,
                layout="qwertzuiopasdfghjklöyxcvbnm,.-QWERTZUIOPASDFGHJKLÖYXCVBNM;:_",
            ),
        }
        if kb not in self.keyboards:
            raise ValueError(
                f"Keyboard layout '{kb}' is not available. Choose from: {list(self.keyboards.keys())}."
            )
        self.kb: Keyboard = self.keyboards[kb]

    def is_pinky(self, k: str) -> bool:
        return abs(self.kb.get_pos(k)[0]) == 5

    def is_ring(self, k: str) -> bool:
        return abs(self.kb.get_pos(k)[0]) == 4

    def is_middle(self, k: str) -> bool:
        return abs(self.kb.get_pos(k)[0]) == 3

    def is_bottom(self, k: str) -> bool:
        return self.kb.get_pos(k)[1] == 1

    def is_homerow(self, k: str) -> bool:
        return self.kb.get_pos(k)[1] == 2

    def is_top(self, k: str) -> bool:
        return self.kb.get_pos(k)[1] == 3

    def is_index(self, k: str) -> bool:
        return abs(self.kb.get_pos(k)[0]) in (1, 2)

    def same_col(self, bg: Tuple[str, str]) -> bool:
        return self.kb.get_col(bg[0]) == self.kb.get_col(bg[1])

    def same_hand(self, bg: Tuple[str, str]) -> bool:
        return self.kb.get_hand(bg[0]) == self.kb.get_hand(bg[1])

    def inwards_rotation(self, bg: Tuple[str, str]) -> bool:
        if self.same_hand(bg):
            # Determine which key is more on the outside.
            if abs(self.kb.get_col(bg[0])) < abs(self.kb.get_col(bg[1])):
                outer, inner = bg[1], bg[0]
            elif abs(self.kb.get_col(bg[0])) > abs(self.kb.get_col(bg[1])):
                outer, inner = bg[0], bg[1]
            else:
                return False

            if self.kb.get_row(outer) > self.kb.get_row(inner):
                return True
        return False

    def get_rotation(self, bg: Tuple[str, str]) -> Optional[float]:
        """
        Compute the rotation angle (in degrees) between two keys in a bigram.
        Returns None if the keys have the same column.
        """
        if not self.same_hand(bg):
            return None

        # Determine outer and inner keys by comparing x coordinates.
        if abs(self.kb.get_col(bg[0])) < abs(self.kb.get_col(bg[1])):
            outer, inner = bg[1], bg[0]
        elif abs(self.kb.get_col(bg[0])) > abs(self.kb.get_col(bg[1])):
            outer, inner = bg[0], bg[1]
        else:
            return None

        x1, y1 = self.kb.get_pos(outer)
        x2, y2 = self.kb.get_pos(inner)
        offset1 = self.kb.row_offsets.get(y1, 0)
        offset2 = self.kb.row_offsets.get(y2, 0)
        # Avoid multiplying by zero; if hand is ambiguous (x==0), return 0.
        hand_factor = self.kb.get_hand(bg[0]) or 1
        angle = degrees(
            atan2((y1 - y2), ((x1 + offset1) - (x2 + offset2)) * hand_factor)
        )
        return round(angle, 2)

    def outwards_rotation(self, bg: Tuple[str, str]) -> bool:
        if self.same_hand(bg):
            if abs(self.kb.get_col(bg[0])) < abs(self.kb.get_col(bg[1])):
                outer, inner = bg[1], bg[0]
            elif abs(self.kb.get_col(bg[0])) > abs(self.kb.get_col(bg[1])):
                outer, inner = bg[0], bg[1]
            else:
                return False
            if self.kb.get_row(outer) < self.kb.get_row(inner):
                return True
        return False

    def is_adjacent(self, bg: Tuple[str, str]) -> bool:
        return abs(self.kb.get_col(bg[0]) - self.kb.get_col(bg[1])) == 1

    def get_dx(self, bg: Tuple[str, str]) -> float:
        x1, y1 = self.kb.get_pos(bg[0])
        x2, y2 = self.kb.get_pos(bg[1])
        offset1 = self.kb.row_offsets.get(y1, 0)
        offset2 = self.kb.row_offsets.get(y2, 0)
        return abs((x1 + offset1) - (x2 + offset2))

    def get_dy(self, bg: Tuple[str, str]) -> int:
        return abs(self.kb.get_row(bg[0]) - self.kb.get_row(bg[1]))

    def get_distance(self, bg: Tuple[str, str], ex: float = 2) -> float:
        dx = self.get_dx(bg)
        dy = self.get_dy(bg)
        return (dx**ex + dy**ex) ** (1 / ex)

    def is_scissor(self, bg: Tuple[str, str]) -> bool:
        """
        Returns True if the bigram has a vertical separation of 2,
        the keys are not typed with the same finger, and both keys
        are on the same hand.
        """
        return (
            self.get_dy(bg) == 2 and (not self.same_finger(bg)) and self.same_hand(bg)
        )

    def same_finger(self, bg: Tuple[str, str]) -> bool:
        if bg[0] != bg[1]:
            return self.kb.get_finger(bg[0]) == self.kb.get_finger(bg[1])
        return False


def test() -> None:
    """
    A simple test routine to compare available keyboard layouts.
    """
    c = Classifier()
    keys = list(c.keyboards.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            k1 = keys[i]
            k2 = keys[j]
            common = "".join(
                v
                for idx, v in enumerate(c.keyboards[k2].lowercase)
                if idx < len(c.keyboards[k1].lowercase)
                and v == c.keyboards[k1].lowercase[idx]
            )
            print(f"{k1} <=> {k2}: common keys: {common}")
            print(f"{k1} layout:\n{c.keyboards[k1]}\n")
            print(f"{k2} layout:\n{c.keyboards[k2]}\n")


if __name__ == "__main__":
    test()
