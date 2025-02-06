# Dependencies
import numpy as np
from scipy.optimize import curve_fit
from classifier import Classifier
from collections import defaultdict

wpm = 80

# Getting frequency information
trigram_to_freq = defaultdict(int)
bigram_to_freq = defaultdict(int)
skipgram_to_freq = defaultdict(int)

with open("trigrams.txt") as f:
    for k, v in (l.split("\t") for l in f):
        if not any([c in "QWERTYUIOPASDFGHJKL:ZXCVBNM<>? " for c in k]):
            trigram_to_freq[k] = int(v)
    trigrams = list(trigram_to_freq.keys())

    percentages = [0] * 100
    total_count = sum(trigram_to_freq.values())
    elapsed = 0

    for i, tg in enumerate(trigrams):
        percentage = int(100 * (elapsed / total_count))
        percentages[percentage] = i

        elapsed += trigram_to_freq[tg]

    print(percentages)

print(trigrams)

with open("bigrams.txt") as f:
    for k, v in (l.split("\t") for l in f):
        bigram_to_freq[k] = int(v)

with open("1-skip.txt") as f:
    for k, v in (l.split("\t") for l in f):
        skipgram_to_freq[k] = int(v)

# Getting positional/time information


# Helper function IQR average for time processing later
def get_iqr_avg(data):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    new_data = [x for x in data if x >= lower_bound and x <= upper_bound]

    return sum(new_data) / len(new_data)


# because eval is so freaking slow
def str_to_tuple(s):
    return tuple(map(int, s.strip("()").split(", ")))


tg_min_samples = 35

with open("tristrokes.tsv") as f1:
    tristroke_data = [
        (eval(a), b, *[s for x in d if ((s := str_to_tuple(x))[0] >= wpm)])
        for (a, b, c, *d) in (l.strip().split("\t") for l in f1)
        if (not any([c in "QWERTYUIOPASDFGHJKL:ZXCVBNM<>? " for c in b]))
    ]  #  and not any([char in " QWERTYUIOPASDFGHJKL:ZXCVBNM<>?" for char in b]))
    tristroke_data = [td for td in tristroke_data if (len(td) - 2 >= tg_min_samples)]

bg_min_samples = 50

with open("bistrokes.tsv") as f2:
    bistroke_data = [
        (eval(a), b, *[s for x in c if ((s := str_to_tuple(x))[0] >= wpm)])
        for (a, b, *c) in (l.strip().split("\t") for l in f2)
        if (not any([c in "QWERTYUIOPASDFGHJKL:ZXCVBNM<>? " for c in b]))
    ]  # and not any([char in "" for char in b])
    bistroke_data = [bd for bd in bistroke_data if (len(bd) - 2 >= bg_min_samples)]

# Generating bigram features
bg_data_size = len(bistroke_data)
bg_class = Classifier()

# features lists
bg_freqs, bg_times = np.zeros(bg_data_size), np.zeros(bg_data_size)
bg_space1, bg_space2 = np.zeros(bg_data_size), np.zeros(bg_data_size)
bg_caps1, bg_caps2 = np.zeros(bg_data_size), np.zeros(bg_data_size)
bg_bottom1, bg_bottom2 = np.zeros(bg_data_size), np.zeros(bg_data_size)
bg_home1, bg_home2 = np.zeros(bg_data_size), np.zeros(bg_data_size)
bg_top1, bg_top2 = np.zeros(bg_data_size), np.zeros(bg_data_size)
bg_pinky1, bg_pinky2 = np.zeros(bg_data_size), np.zeros(bg_data_size)
bg_ring1, bg_ring2 = np.zeros(bg_data_size), np.zeros(bg_data_size)
bg_middle1, bg_middle2 = np.zeros(bg_data_size), np.zeros(bg_data_size)
bg_index1, bg_index2 = np.zeros(bg_data_size), np.zeros(bg_data_size)
bg_sfb, bg_scb, bg_shb = (
    np.zeros(bg_data_size),
    np.zeros(bg_data_size),
    np.zeros(bg_data_size),
)
bg_adjacent, bg_lateral, bg_scissor, bg_lsb = (
    np.zeros(bg_data_size),
    np.zeros(bg_data_size),
    np.zeros(bg_data_size),
    np.zeros(bg_data_size),
)
bg_dy, bg_dx = np.zeros(bg_data_size), np.zeros(bg_data_size)
bg_labels = [""] * bg_data_size
layout_col = ["red" for _ in range(bg_data_size)]

ml = float("inf")


def get_bistroke_features(pos, bigram):
    ((ax, ay), (bx, by)) = pos

    col = "red"

    if ((ax, ay), (bx, by)) == tuple(
        [bg_class.keyboards["qwerty"].get_pos(c) for c in bigram]
    ):
        col = "green"

    freq = bigram_to_freq[bigram]
    label = bigram

    cap1 = bigram[0] in "QWERTYUIOPASDFGHJKLZXCVBNM<>?:"
    cap2 = bigram[0] in "QWERTYUIOPASDFGHJKLZXCVBNM<>?:"

    # Row features
    space1 = ay == 0
    space2 = by == 0

    bottom1 = ay == 1
    bottom2 = by == 1

    home1 = ay == 2
    home2 = by == 2

    top1 = ay == 3
    top2 = by == 3

    # Column features
    pinky1 = abs(ax) == 5
    pinky2 = abs(bx) == 5
    ring1 = abs(ax) == 4
    ring2 = abs(bx) == 4
    middle1 = abs(ax) == 3
    middle2 = abs(bx) == 3
    index1 = abs(ax) in (1, 2)
    index2 = abs(bx) in (1, 2)

    row_offsets = [0.5, 0, -0.25]

    dy = abs(ay - by)
    dx = abs((ax + row_offsets[ay - 1]) - (bx + row_offsets[by - 1]))
    # LSB = middle and index adjacent, then dx > 1.5

    # bg classification
    shb = False

    if not (space1 or space2):
        shb = (ax // abs(ax)) == (bx // abs(bx))

    scb = ax == bx
    sfb = scb or (shb and (abs(ax) in (1, 2) and abs(bx) in (1, 2)))

    lateral = (abs(bx) == 1) and (not shb) | (abs(bx) == 1)  #  & shb and abs(ax) != 1
    adjacent = shb * (abs(ax - bx) == 1 | (index1 and middle2) | (index2 and middle1))
    lsb = shb * (
        (index1 and middle2) | (index2 and middle1) and dx > 1.5
    )  # maybe only index 2

    # Scissor classification
    # A scissor is a bigram with a Δy==2, and a row-stagger Δx <= 1
    # or a bigram where the long finger curls and the short finger stretches 'xq'
    bg_scissor = dy == 2 and dx <= 1

    # middle and ring don't like curling, index and pinky don't like to stretch
    bg_scissor |= (pinky1 and top1 and ring2 and bottom2) or (
        ring1 and bottom1 and pinky2 and top2
    )
    bg_scissor |= (ring1 and top1 and middle2 and bottom2) or (
        middle1 and bottom1 and ring2 and top2
    )
    bg_scissor |= (index1 and top1 and middle2 and bottom2) or (
        middle1 and bottom1 and index1 and top1
    )

    bg_scissor &= adjacent

    return (
        freq,
        space1,
        space2,
        bottom1,
        bottom2,
        home1,
        home2,
        top1,
        top2,
        pinky1,
        pinky2,
        ring1,
        ring2,
        middle1,
        middle2,
        index1,
        index2,
        lateral,
        shb,
        sfb,
        adjacent,
        bg_scissor,
        lsb,
        cap1,
        cap2,
        dy,
        dx,
        label,
        col,
    )


for i, bistroke in enumerate(bistroke_data):
    pos, bigram, freq, *bistroke_times = bistroke
    (
        bg_freqs[i],
        bg_space1[i],
        bg_space2[i],
        bg_bottom1[i],
        bg_bottom2[i],
        bg_home1[i],
        bg_home2[i],
        bg_top1[i],
        bg_top2[i],
        bg_pinky1[i],
        bg_pinky2[i],
        bg_ring1[i],
        bg_ring2[i],
        bg_middle1[i],
        bg_middle2[i],
        bg_index1[i],
        bg_index2[i],
        bg_lateral[i],
        bg_shb[i],
        bg_sfb[i],
        bg_adjacent[i],
        bg_scissor[i],
        bg_lsb[i],
        bg_caps1[i],
        bg_caps2[i],
        bg_dy[i],
        bg_dx[i],
        bg_labels[i],
        layout_col[i],
    ) = get_bistroke_features(pos, bigram)[:30]
    arr = [t[1] for t in bistroke_times]
    ml = min(len(arr), ml)
    bg_times[i] = get_iqr_avg(arr)

bg_features = [
    bg_freqs,
    bg_space1,
    bg_space2,
    bg_bottom1,
    bg_bottom2,
    bg_home1,
    bg_home2,
    bg_top1,
    bg_top2,
    bg_pinky1,
    bg_pinky2,
    bg_ring1,
    bg_ring2,
    bg_middle1,
    bg_middle2,
    bg_index1,
    bg_index2,
    bg_lateral,
    bg_shb,
    bg_sfb,
    bg_adjacent,
    bg_scissor,
    bg_lsb,
    bg_caps1,
    bg_caps2,
    bg_dy,
    bg_dx,
]  # , bg_labels, layout_col


def bg_penalty(
    features,
    p0,
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
    p7,
    p8,
    p9,
    p10,
    p11,
    p12,
    p13,
    p14,
    p15,
    p16,
    p17,
    p18,
    p19,
    p20,
    p21,
    p22,
    p23,
    p24,
    p25,
    p26,
    p27,
    p28,
    p29,
    p30,
    p31,
    p32,
    p33,
    p34,
    p35,
    p36,
    p37,
    p38,
    p39,
    p40,
    p41,
    p42,
    p43,
    p44,
    p45,
):
    (
        bg_freqs,
        bg_space1,
        bg_space2,
        bg_bottom1,
        bg_bottom2,
        bg_home1,
        bg_home2,
        bg_top1,
        bg_top2,
        bg_pinky1,
        bg_pinky2,
        bg_ring1,
        bg_ring2,
        bg_middle1,
        bg_middle2,
        bg_index1,
        bg_index2,
        bg_lateral,
        bg_shb,
        bg_sfb,
        bg_adjacent,
        bg_scissor,
        bg_lsb,
        bg_caps1,
        bg_caps2,
        bg_dy,
        bg_dx,
    ) = features

    freq_pen = p0 * np.log(np.clip(bg_freqs + p1, 1e-8, None)) + p2

    # Row penalties
    base_row_pen = p3 * (bg_home2 + bg_top2) + p4 * (bg_top2 + bg_bottom2)
    shb_row_pen = p5 * (bg_home2 + bg_top2) + p6 * (bg_top2 + bg_bottom2)
    alt_row_pen = p7 * (bg_home2 + bg_top2) + p8 * (bg_top2 + bg_bottom2)
    sfb_row_pen = p9 * (bg_home2 + bg_top2) + p10 * (bg_top2 + bg_bottom2)

    # Finger penalties
    sfb_finger_pen = (
        p11 * bg_pinky2
        + p12 * bg_ring2
        + p13 * bg_middle2
        + p14 * (bg_index2 - bg_lateral)
        + p15 * bg_lateral
    )
    base_finger_pen = (
        p16 * bg_pinky2
        + p17 * bg_ring2
        + p18 * bg_middle2
        + p19 * (bg_index2 - bg_lateral)
        + p20 * bg_lateral
    )
    shb_finger_pen = (
        p21 * bg_pinky2
        + p22 * bg_ring2
        + p23 * bg_middle2
        + p24 * (bg_index2 - bg_lateral)
        + p25 * bg_lateral
    )
    alt_finger_pen = (
        p26 * bg_pinky2
        + p27 * bg_ring2
        + p28 * bg_middle2
        + p29 * (bg_index2 - bg_lateral)
        + p30 * bg_lateral
    )

    # Aggregate penalties for classes
    shb_pen = (shb_finger_pen) * (shb_row_pen)
    alt_pen = (alt_finger_pen) * (alt_row_pen)
    sfb_pen = (sfb_finger_pen) * (sfb_row_pen)

    # class penalties
    base_weight = 1 + (base_row_pen * base_finger_pen)
    shb_weight = (bg_shb * (1 - bg_sfb)) * (shb_pen)
    dist_pen = (bg_dx**2 + bg_dy**2) ** 0.5 + p31
    sfb_weight = bg_sfb * sfb_pen * dist_pen
    alt_weight = (1 - bg_shb) * alt_pen

    return freq_pen * (base_weight + alt_weight + shb_weight + sfb_weight)


bg_popt, bg_pcov = curve_fit(
    bg_penalty, bg_features, bg_times, method="lm", maxfev=750000
)  # "trf" p0=initial_guess

sum_of_squares = np.sum((bg_times - np.mean(bg_times)) ** 2)

new_y = bg_penalty(bg_features, *bg_popt)
residuals = bg_times - new_y
r2 = 1 - np.sum((residuals) ** 2) / sum_of_squares

print("R^2:", r2)
print("MAE:", np.mean(np.abs(residuals)))

bg_popt, bg_pcov = curve_fit(
    bg_penalty, bg_features, bg_times, method="lm", maxfev=750000
)  # "trf" p0=initial_guess

sum_of_squares = np.sum((bg_times - np.mean(bg_times)) ** 2)

new_y = bg_penalty(bg_features, *bg_popt)
residuals = bg_times - new_y
r2 = 1 - np.sum((residuals) ** 2) / sum_of_squares

print("R^2:", r2)
print("MAE:", np.mean(np.abs(residuals)))

# These can be used in simulated annealing.py
print(list(bg_popt))

import matplotlib.pyplot as plt

plt.figure()

xx, yy, ll, fit_y, c = zip(
    *sorted(
        [
            r
            for r in zip(bg_freqs, bg_times, bg_labels, new_y, layout_col)
            if r[-1] != "blue"
        ],
        key=lambda x: x[0],
        reverse=True,
    )
)
xx = list((range(len(xx))))
scatter = plt.scatter(xx, yy, s=50, c=c)

for x, y, l in zip(xx, yy, ll):
    plt.annotate(f"'{l}'", (x, y))

plt.plot(xx, fit_y, c="black")
# plt.axhline(0, color='black', linewidth=0.5)
# plt.scatter(freqs, times-new_y, c="red")
plt.xlabel("Frequency Index")
plt.ylabel("Average Typing Time (Milliseconds)")
# plt.xscale("log")

plt.show()

# Generating trigram features
tg_data_size = len(tristroke_data)

# features lists
tg_freqs, tg_times = np.zeros(tg_data_size), np.zeros(tg_data_size)
tg_bg1_prediction, tg_bg2_prediction = np.zeros(tg_data_size), np.zeros(tg_data_size)
tg_sg_features = [() for _ in range(tg_data_size)]
tg_bg1, tg_bg2, tg_sg = [""] * tg_data_size, [""] * tg_data_size, [""] * tg_data_size
tg_labels, tg_col = [""] * tg_data_size, ["red"] * tg_data_size
tg_redirect, tg_bad = np.zeros(tg_data_size), np.zeros(tg_data_size)
tg_sht = np.zeros(tg_data_size)
sg_freqs = np.zeros(tg_data_size)

for i, tristroke in enumerate(tristroke_data):
    ((ax, ay), (bx, by), (cx, cy)), trigram, *tristroke_time = tristroke

    tg_freqs[i] = trigram_to_freq[trigram]
    tg_times[i] = get_iqr_avg([t[1] for t in tristroke_time])
    tg_bg1[i], tg_bg2[i], tg_sg[i] = trigram[:2], trigram[1:], trigram[::2]

    tg_bg1_prediction[i] = bg_penalty(
        get_bistroke_features(((ax, ay), (bx, by)), tg_bg1[i])[:-2], *bg_popt
    )
    tg_bg2_prediction[i] = bg_penalty(
        get_bistroke_features(((bx, by), (cx, cy)), tg_bg2[i])[:-2], *bg_popt
    )
    tg_sg_features[i] = get_bistroke_features(((ax, ay), (cx, cy)), tg_sg[i])[:-2]
    tg_labels[i] = trigram

    if ((ax, ay), (bx, by), (cx, cy)) == tuple(
        [bg_class.keyboards["qwerty"].get_pos(c) for c in trigram]
    ):
        tg_col[i] = "green"

    sg_freqs[i] = skipgram_to_freq[tg_sg[i]]

    if 0 not in (ax, bx, cx):
        tg_sht[i] = ax // abs(ax) == bx // abs(bx) == cx // abs(cx)

    if tg_sht[i]:
        tg_redirect[i] = (abs(ax) < abs(bx) and abs(cx) < abs(bx)) | (
            abs(ax) > abs(bx) and abs(cx) > abs(bx)
        )
        tg_bad[i] = tg_redirect[i] * (
            not any([abs(x) in (1, 2) for x in (abs(ax), abs(bx), abs(cx))])
        )
        tg_redirect[i] *= 1 - tg_bad[i]

tg_features = [
    tg_freqs,
    tg_bg1_prediction,
    tg_bg2_prediction,
    tg_sht,
    tg_redirect,
    tg_bad,
    sg_freqs,
    *[r for r in np.stack(tg_sg_features, axis=1)],
]

print(list(v[0] for v in zip(tg_labels, tg_bad) if v[1]))


def tg_penalty(
    tg_features, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15
):
    (
        tg_freqs,
        tg_bg1_prediction,
        tg_bg2_prediction,
        tg_sht,
        tg_redirect,
        tg_bad,
        sg_freq,
        bg_freqs,
        bg_space1,
        bg_space2,
        bg_bottom1,
        bg_bottom2,
        bg_home1,
        bg_home2,
        bg_top1,
        bg_top2,
        bg_pinky1,
        bg_pinky2,
        bg_ring1,
        bg_ring2,
        bg_middle1,
        bg_middle2,
        bg_index1,
        bg_index2,
        bg_lateral,
        bg_shb,
        bg_sfb,
        bg_adjacent,
        bg_scissor,
        bg_lsb,
        bg_caps1,
        bg_caps2,
        bg_dy,
        bg_dx,
    ) = tg_features

    freq_pen = p0 * np.log(tg_freqs + p1) + p2
    sfs_row_pen = p3 * (bg_home2 + bg_top2) + p4 * (bg_bottom2 + bg_top2)
    sfs_finger_pen = (
        p5 * bg_pinky2
        + p6 * bg_ring2
        + p7 * bg_middle2
        + p8 * (bg_index2 - bg_lateral)
        + p9 * (bg_lateral)
    )
    dist_pen = (bg_dx**2 + bg_dy**2) ** 0.5 + p10
    sfs_weight = bg_sfb * (sfs_row_pen + sfs_finger_pen) * dist_pen

    return tg_bg1_prediction + tg_bg2_prediction + sfs_weight + p11


mask = np.isnan(tg_bg2_prediction)

for p in np.array(tg_labels)[mask]:
    print(p)

print([v for v in np.isnan(tg_features).sum(axis=1)])  # Number of nulls per column

tg_popt, tg_pcov = curve_fit(
    tg_penalty, tg_features, tg_times, method="trf", maxfev=750000
)

sum_of_squares = np.sum((tg_times - np.mean(tg_times)) ** 2)

new_y = tg_penalty(tg_features, *tg_popt)
residuals = tg_times - new_y
r2 = 1 - np.sum((residuals) ** 2) / sum_of_squares

print("R^2:", r2)
print("MAE:", np.mean(np.abs(residuals)))

print(list(tg_popt))

import matplotlib.pyplot as plt

plt.figure()

xx, yy, ll, fit_y, cc = zip(
    *sorted(
        [
            r
            for r in zip(tg_freqs, tg_times, tg_labels, new_y, tg_col)
            if r[-1] != "blue"
        ],
        key=lambda x: x[0],
        reverse=True,
    )
)
xx = list((range(len(xx))))
scatter = plt.scatter(xx, yy, s=50, c=cc)

for x, y, l in zip(xx, yy, ll):
    plt.annotate(f"'{l}'", (x, y))

plt.plot(xx, fit_y, c="black")
# plt.axhline(0, color='black', linewidth=0.5)
# plt.scatter(freqs, times-new_y, c="red")
plt.xlabel("Frequency Index")
plt.ylabel("Average Typing Time (Milliseconds)")
# plt.xscale("log")

plt.show()
