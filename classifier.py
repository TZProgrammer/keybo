from math import atan2, degrees
from random import sample
from itertools import product


class Keyboard:
    def __init__(
        self, layout="qwertyuiopasdfghjkl;zxcvbnm,./QWERTYUIOPASDFGHJKL:ZXCVBNM<>?"
    ):
        self.row_offsets = [-0.25, 0, 0.5]
        self.chars = layout
        self.swap_pair = ["", ""]
        self.key_count = 30
        self.lowercase = layout[:30]
        self.uppercase = layout[30:]
        self.lower_to_upper = dict(zip(self.lowercase, self.uppercase))
        self.upper_to_lower = dict(zip(self.uppercase, self.lowercase))

        self.x_to_finger = {
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

        self.key_to_pos = {}

        for y, row in enumerate(
            [self.lowercase[i * 10 : i * 10 + 10] for i in range(3)]
        ):
            for x, k in enumerate(row):
                new_x = x - 5 if x < 5 else x - 4

                self.key_to_pos[k] = (new_x, 3 - y)

        self.key_to_pos[" "] = (0, 0)
        self.pos_to_key = dict(zip(self.key_to_pos.values(), self.key_to_pos.keys()))

    def __repr__(self):
        rows = [list(self.pos_to_key.values())[i : i + 10] for i in range(0, 30, 10)]
        return "\n".join([" ".join(row) for row in rows])

    def get_ngrams(self, n):
        return set(
            "".join(combo)
            for swap in self.swap_pair
            for combo in product(self.chars, repeat=n)
            if swap in combo or swap == ""
        )

    def undo_swap(self):
        self.swap(*self.swap_pair)

    def random_swap(self):
        self.swap(*sample(self.lowercase, 2))

    def swap(self, k1, k2):
        self.swap_pair = (k1, k2)
        x1, y1 = self.key_to_pos[k1]
        x2, y2 = self.key_to_pos[k2]

        self.key_to_pos[k1] = (x2, y2)
        self.key_to_pos[k2] = (x1, y1)
        self.pos_to_key[(x1, y1)] = k2
        self.pos_to_key[(x2, y2)] = k1

    def get_key(self, x, y):
        return self.pos_to_key[(x, y)]

    def get_pos(self, k):
        if k in self.uppercase:
            return self.key_to_pos[self.upper_to_lower[k]]
        return self.key_to_pos[k]

    def get_col(self, k):
        return self.key_to_pos[k][0]

    def get_finger(self, k):
        return self.x_to_finger[self.key_to_pos[k][0]]

    def get_row(self, k):
        return self.key_to_pos[k][1]

    def get_hand(self, k):
        return abs(self.key_to_pos[k][0]) / self.key_to_pos[k][0]


class Classifier:
    def __init__(self, kb="qwerty"):
        self.keyboards = {
            "qwerty": Keyboard(
                "qwertyuiopasdfghjkl;zxcvbnm,./QWERTYUIOPASDFGHJKL:ZXCVBNM<>?"
            ),
            "azerty": Keyboard(
                "azertyuiopqsdfghjkl;wxcvbnm,./AZERTYUIOPQSDFGHJKL:WXCVBNM<>?"
            ),
            "dvorak": Keyboard(
                "',.pyfgcrlaoeuidhtns;qjkxbmwvzPYFGCRL?+|AOEUIDHTNS:QJKXBMWVZ"
            ),
            "qwertz": Keyboard(
                "qwertzuiopasdfghjklöyxcvbnm,.-QWERTZUIOPASDFGHJKLÖYXCVBNM;:_"
            ),
        }

        self.kb = self.keyboards[kb]

    def is_pinky(self, k):
        return abs(self.kb.get_pos(k)[0]) == 5

    def is_ring(self, k):
        return abs(self.kb.get_pos(k)[0]) == 4

    def is_middle(self, k):
        return abs(self.kb.get_pos(k)[0]) == 3

    def is_bottom(self, k):
        return abs(self.kb.get_pos(k)[1]) == 1

    def is_homerow(self, k):
        return abs(self.kb.get_pos(k)[1]) == 2

    def is_top(self, k):
        return abs(self.kb.get_pos(k)[1]) == 3

    def is_index(self, k):
        return abs(self.kb.get_pos(k)[0]) in (2, 1)

    def same_col(self, bg):
        return self.kb.get_col(bg[0]) == self.kb.get_col(bg[1])

    def same_hand(self, bg):
        return self.kb.get_hand(bg[0]) == self.kb.get_hand(bg[1])

    def inwards_rotation(self, bg):
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

    def get_rotation(self, bg):
        outer, inner = bg[1], bg[0]

        if self.same_hand(bg):
            if abs(self.kb.get_col(bg[0])) < abs(self.kb.get_col(bg[1])):
                outer, inner = bg[1], bg[0]
            elif abs(self.kb.get_col(bg[0])) > abs(self.kb.get_col(bg[1])):
                outer, inner = bg[0], bg[1]
            else:
                return None

            x1, y1 = self.kb.get_pos(outer)
            x2, y2 = self.kb.get_pos(inner)

            return round(
                degrees(
                    atan2(
                        (y1 - y2),
                        (
                            (x1 + self.kb.row_offsets[3 - y1])
                            - (x2 + self.kb.row_offsets[3 - y2])
                        )
                        * self.kb.get_hand(bg[0]),
                    )
                )
            )

        return None

    def outwards_rotation(self, bg):
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

    def is_adjacent(self, bg):
        return abs(self.kb.get_col(bg[0]) - self.kb.get_col(bg[1])) == 1

    def get_dx(self, bg):
        x1, y1 = self.kb.get_pos(bg[0])
        x2, y2 = self.kb.get_pos(bg[1])

        return abs(
            (x1 + self.kb.row_offsets[3 - y1]) - (x2 + self.kb.row_offsets[3 - y2])
        )

    def get_dy(self, bg):
        return abs(self.kb.get_row(bg[0]) - self.kb.get_row(bg[1]))

    def get_distance(self, bg, ex):
        return ((self.get_dx(bg)) ** ex + (self.get_dy(bg)) ** ex) ** 0.5

    def is_scissor(self, bg):
        return (
            self.get_dy(bg) == 2
            and not self.same_finger(bg)
            and self.kb.get_hand(bg[0]) == self.kb.get_hand(bg[1])
        )  # get_dx(bg) < 2 and get_dy(bg) == 2 and not same_finger(bg)

    def same_finger(self, bg):
        return bg[0] != bg[1] and self.kb.get_finger(bg[0]) == self.kb.get_finger(bg[1])


def test():
    c = Classifier()

    for i in range(len(c.keyboards)):
        for j in range(i + 1, len(c.keyboards)):
            k1 = list(c.keyboards.keys())[i]
            k2 = list(c.keyboards.keys())[j]

            print(
                k1,
                "<=>",
                k2,
                "".join(
                    [
                        v
                        for i, v in enumerate(c.keyboards[k2].lowercase)
                        if v == c.keyboards[k1].lowercase[i]
                    ]
                ),
            )

            print(c.keyboards[k1])
