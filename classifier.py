"""
classifier.py – Keyboard layout and classifier utilities.

This module defines:
  - Keyboard: A class representing a keyboard layout.
  - Classifier: A helper class that loads trigram frequency data and provides
    keyboard-related predicates.
  - A test() function to inspect the keyboards.
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
    """
    def __init__(self, data_size: int, layout: str) -> None:
        if len(layout) != 60:
            raise ValueError("Layout string must be exactly 60 characters long.")
        self.data_size = data_size
        self.row_offsets: Dict[int, float] = {1: 0.5, 2: 0.0, 3: -0.25}
        self.chars: str = layout
        self.swap_pair: Tuple[str, str] = ("", "")
        self.key_count: int = 30
        self.lowercase: str = layout[:30]
        self.uppercase: str = layout[30:]
        self.lower_to_upper: Dict[str, str] = dict(zip(self.lowercase, self.uppercase))
        self.upper_to_lower: Dict[str, str] = dict(zip(self.uppercase, self.lowercase))
        self.x_to_finger: Dict[int, str] = {
            5: "lp", 4: "lr", 3: "lm", 2: "li", 1: "li",
            -1: "ri", -2: "ri", -3: "rm", -4: "rr", -5: "rp",
        }
        self.key_to_pos: Dict[str, Tuple[int, int]] = {}
        rows: List[str] = [self.lowercase[i * 10: (i + 1) * 10] for i in range(3)]
        for y_index, row in enumerate(rows):
            row_number = 3 - y_index  # top row is row 3
            for x_index, k in enumerate(row):
                new_x: int = x_index - 5 if x_index < 5 else x_index - 4
                self.key_to_pos[k] = (new_x, row_number)
        self.key_to_pos[" "] = (0, 0)
        self.pos_to_key: Dict[Tuple[int, int], str] = {pos: key for key, pos in self.key_to_pos.items()}

    def __repr__(self) -> str:
        rows = {}
        for key, (x, y) in self.key_to_pos.items():
            if y > 0:
                rows.setdefault(y, []).append((x, key))
        result_lines = []
        for row_num in sorted(rows.keys(), reverse=True):
            row_keys = [key for x, key in sorted(rows[row_num], key=lambda tup: tup[0])]
            result_lines.append(" ".join(row_keys))
        return "\n".join(result_lines)

    def get_ngrams(self, n: int) -> Set[str]:
        ngrams = set()
        for swap in self.swap_pair:
            for combo in product(self.chars, repeat=n):
                if swap == "" or swap in combo:
                    ngrams.add("".join(combo))
        return ngrams

    def undo_swap(self) -> None:
        self.swap(*self.swap_pair)

    def random_swap(self) -> None:
        k1, k2 = sample(list(self.lowercase), 2)
        self.swap(k1, k2)

    def swap(self, k1: str, k2: str) -> None:
        self.swap_pair = (k1, k2)
        pos1 = self.key_to_pos.get(k1)
        pos2 = self.key_to_pos.get(k2)
        if pos1 is None or pos2 is None:
            raise ValueError(f"Cannot swap keys '{k1}' and '{k2}'; one or both keys not found.")
        self.key_to_pos[k1], self.key_to_pos[k2] = pos2, pos1
        self.pos_to_key[pos1], self.pos_to_key[pos2] = k2, k1

    def get_key(self, x: int, y: int) -> Optional[str]:
        return self.pos_to_key.get((x, y))

    def get_pos(self, k: str) -> Tuple[int, int]:
        if k in self.uppercase:
            k = self.upper_to_lower.get(k, k)
        return self.key_to_pos.get(k, (0, 0))

    def get_col(self, k: str) -> int:
        return self.get_pos(k)[0]

    def get_finger(self, k: str) -> str:
        x = self.get_pos(k)[0]
        try:
            return self.x_to_finger[x]
        except KeyError:
            raise ValueError(f"No finger mapping found for key '{k}' at x = {x}.")

    def get_row(self, k: str) -> int:
        return self.get_pos(k)[1]

    def get_hand(self, k: str) -> int:
        x = self.get_pos(k)[0]
        if x == 0:
            return 0
        return int(x / abs(x))

class Classifier:
    """
    Provides helper methods for keyboard-based feature analysis.
    """
    def __init__(self, kb: str = "qwerty") -> None:
        # Allowed characters now include both lower and upper case.
        valid_chars = set("qwertyuiopasdfghjkl;zxcvbnm,./QWERTYUIOPASDFGHJKL:ZXCVBNM<>? ")
        trigrams = []
        tg_freqs = []
        tg_percentages = {}
        trigram_path = "trigrams.txt"
        if not os.path.exists(trigram_path):
            raise FileNotFoundError(f"Cannot find file: {trigram_path}")
        with open(trigram_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                trigram, freq_str = parts[:2]
                trigram = trigram.strip()  # do NOT convert to lowercase
                if len(trigram) != 3:
                    continue
                if all(c in valid_chars for c in trigram):
                    trigrams.append(trigram)
                    try:
                        tg_freqs.append(int(freq_str))
                    except ValueError:
                        tg_freqs.append(0)
        total_count = sum(tg_freqs)
        elapsed = 0
        for i, freq in enumerate(tg_freqs):
            percentage = int(100 * (elapsed / total_count)) if total_count else 0
            tg_percentages[percentage + 1] = i
            elapsed += freq
        tg_coverage = 100
        cutoff = tg_percentages.get(tg_coverage, len(tg_freqs))
        tg_freqs = np.array(tg_freqs[:cutoff])
        trigrams = trigrams[:cutoff]
        print("Processed trigram data: using", len(trigrams), "trigrams.")
        data_size = len(trigrams)
        self.keyboards: Dict[str, Keyboard] = {
            "qwerty": Keyboard(data_size=data_size,
                               layout="qwertyuiopasdfghjkl;zxcvbnm,./QWERTYUIOPASDFGHJKL:ZXCVBNM<>?"),
            "azerty": Keyboard(data_size=data_size,
                               layout="azertyuiopqsdfghjkl;wxcvbnm,./AZERTYUIOPQSDFGHJKL:WXCVBNM<>?"),
            "dvorak": Keyboard(data_size=data_size,
                               layout="',.pyfgcrlaoeuidhtns;qjkxbmwvzPYFGCRL?+|AOEUIDHTNS:QJKXBMWVZ"),
            "qwertz": Keyboard(data_size=data_size,
                               layout="qwertzuiopasdfghjklöyxcvbnm,.-QWERTZUIOPASDFGHJKLÖYXCVBNM;:_"),
        }
        if kb not in self.keyboards:
            raise ValueError(f"Keyboard layout '{kb}' is not available. Choose from: {list(self.keyboards.keys())}.")
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

    def same_hand(self, bg: Tuple[str, str]) -> bool:
        return self.kb.get_hand(bg[0]) == self.kb.get_hand(bg[1])
    
    def different_hand(self, bg: Tuple[str, str]) -> bool:
        return self.kb.get_hand(bg[0]) != self.kb.get_hand(bg[1])
    
    def inwards_rotation(self, bg: Tuple[str, str]) -> bool:
        if self.same_hand(bg):
            if abs(self.kb.get_col(bg[0])) < abs(self.kb.get_col(bg[1])):
                outer, inner = bg[1], bg[0]
            elif abs(self.kb.get_col(bg[0])) > abs(self.kb.get_col(bg[1])):
                outer, inner = bg[0], bg[1]
            else:
                return False
            if self.kb.get_row(outer) > self.kb.get_row(inner):
                return True
        return False

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

    def get_rotation(self, bg: Tuple[str, str]) -> Optional[float]:
        if not self.same_hand(bg):
            return None
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
        hand_factor = self.kb.get_hand(bg[0]) or 1
        angle = atan2((y1 - y2), ((x1 + offset1) - (x2 + offset2)) * hand_factor)
        return round(degrees(angle), 2)

    def get_distance(self, bg: Tuple[str, str], ex: float = 2) -> float:
        dx = abs(self.kb.get_col(bg[0]) - self.kb.get_col(bg[1]))
        dy = abs(self.kb.get_row(bg[0]) - self.kb.get_row(bg[1]))
        return (dx**ex + dy**ex) ** (1/ex)

def test() -> None:
    c = Classifier()
    keys = list(c.keyboards.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            k1 = keys[i]
            k2 = keys[j]
            common = "".join(v for idx, v in enumerate(c.keyboards[k2].lowercase)
                             if idx < len(c.keyboards[k1].lowercase)
                             and v == c.keyboards[k1].lowercase[idx])
            print(f"{k1} <=> {k2}: common keys: {common}")
            print(f"{k1} layout:\n{c.keyboards[k1]}\n")
            print(f"{k2} layout:\n{c.keyboards[k2]}\n")

if __name__ == "__main__":
    test()
