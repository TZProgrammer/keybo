from conversion import *
import re

# Used for my poster
#bg_col = linear_srgb_to_oklab(18, 21, 31) # poster dark dark blue
#fg_col = linear_srgb_to_oklab(29, 200, 255) # poster blue
# fg_col = linear_srgb_to_oklab(255, 0, 64) # red
# fg_col = linear_srgb_to_oklab(0, 170, 0) # green
fg_col = linear_srgb_to_oklab(255, 255, 0) # blue
bg_col = linear_srgb_to_oklab(255, 0, 0)


table = """\cellcolor[HTML]{24314b}\\textbf{M} & \cellcolor[HTML]{316799}\\textbf{R} & \cellcolor[HTML]{3091ca}\\textbf{T} & \cellcolor[HTML]{283a59}\\textbf{C} & \cellcolor[HTML]{20293f}\\textbf{W} & \cellcolor[HTML]{1b2131}\\textbf{,} & \cellcolor[HTML]{181e2c}\\textbf{K} & \cellcolor[HTML]{3183ba}\\textbf{A} & \cellcolor[HTML]{1dc8ff}\\textbf{E} & \cellcolor[HTML]{131722}\\textbf{'} \\\\ \hline
\cellcolor[HTML]{2d4a71}\\textbf{L} & \cellcolor[HTML]{3173a8}\\textbf{N} & \cellcolor[HTML]{2b4366}\\textbf{D} & \cellcolor[HTML]{316a9d}\\textbf{S} & \cellcolor[HTML]{1b2132}\\textbf{V} & \cellcolor[HTML]{222c43}\\textbf{Y} & \cellcolor[HTML]{283958}\\textbf{U} & \cellcolor[HTML]{3180b7}\\textbf{O} & \cellcolor[HTML]{3173a9}\\textbf{I} & \cellcolor[HTML]{222e45}\\textbf{G} \\\\ \hline
\cellcolor[HTML]{2e507a}\\textbf{H} & \cellcolor[HTML]{131621}\\textbf{X} & \cellcolor[HTML]{12141e}\\textbf{Z} & \cellcolor[HTML]{1e2538}\\textbf{B} & \cellcolor[HTML]{232e46}\\textbf{F} & \cellcolor[HTML]{1b2233}\\textbf{.} & \cellcolor[HTML]{131722}\\textbf{-} & \cellcolor[HTML]{12141e}\\textbf{Q} & \cellcolor[HTML]{12151f}\\textbf{J} & \cellcolor[HTML]{222d45}\\textbf{P} \\\\ \hline"""

freqs = {
    "M" : 2.22,
    "R" : 5.85,
    "T" : 8.44,
    "C" : 2.83,
    "W" : 1.63,
    "," : 0.99,
    "K" : 0.75,
    "A" : 7.57,
    "E" : 11.47,
    "'" : 0.26,
    "L" : 3.97,
    "N" : 6.59,
    "D" : 3.45,
    "S" : 6.05,
    "V" : 1.01,
    "Y" : 1.84,
    "U" : 2.79,
    "O" : 7.41,
    "I" : 6.63,
    "G" : 1.93,
    "H" : 4.38,
    "X" : 0.2,
    "Z" : 0.09,
    "B" : 1.32,
    "F" : 1.99,
    "." : 1.08,
    "-" : 0.26,
    "Q" : 0.09,
    "J" : 0.11,
    "P" : 1.92,
}

low_f = min(freqs.values())
high_f = max(freqs.values())

def get_col(freq):
    lerp = lambda a, b, t : a + (b-a) * t
    t = (freq-low_f)/(high_f-low_f)

    col = [lerp(a, b, t) for a, b in zip(bg_col, fg_col)]

    return rgb_to_hex(*oklab_to_linear_srgb(col[0], col[1], col[2]))

for r in table.split("\n"):
    cells = r.split("&")
    cells = [
        re.sub
        (
            r"\{[a-z\d]*\}",
            "{"+get_col(freqs[re.search(r"\{([A-Z.,'-])\}", c).group(1)])[1:]+"}",
            c
        )
        for c in r.split("&")
    ]

    print("&".join(cells))

# s = ""
# for c, f in freqs.items():
#     s += c*int(f*100)
# 
# with open("out", "w") as f:
#     f.write(s)

print("".join(r[0].lower() for r in sorted(freqs.items(), key=lambda x: -x[1]) if r[0].isalpha()))