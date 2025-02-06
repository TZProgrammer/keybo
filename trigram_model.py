"""
typing_model.py – An improved and extensible typing analysis script.

Usage:
    python typing_model.py [--model curve_fit|xgboost]
"""

import argparse
import ast
import os
from collections import defaultdict
from typing import Any, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Try to import xgboost if available.
try:
    import xgboost as xgb
except ImportError:
    xgb = None

# Import your custom classifier (make sure classifier.py is in your PYTHONPATH)
from classifier import Classifier

################################################################################
# Global constants and data structures
################################################################################

wpm: int = 80  # words per minute threshold

# These globals will be set by load_ngram_frequencies()
trigram_to_freq: dict = {}
bigram_to_freq: dict = {}
skipgram_to_freq: dict = {}

################################################################################
# 1. Data Loading Functions
################################################################################


def load_ngram_frequencies() -> Tuple[dict, dict, dict, List[str]]:
    """
    Loads frequency data for trigrams, bigrams, and skipgrams from text files.
    Returns:
      (trigram_to_freq, bigram_to_freq, skipgram_to_freq, list_of_trigrams).
    """
    trigram_to_freq_local = defaultdict(int)
    trigrams: List[str] = []
    trigram_path = "trigrams.txt"
    if not os.path.exists(trigram_path):
        raise FileNotFoundError(f"Cannot find file: {trigram_path}")
    with open(trigram_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            k, v = parts[:2]
            # Only keep trigrams that do NOT contain any unwanted uppercase/punctuation characters.
            if not any(c in "QWERTYUIOPASDFGHJKL:ZXCVBNM<>? " for c in k):
                trigram_to_freq_local[k] = int(v)
                trigrams.append(k)

    # (Optional) You can use the percentages list for indexing.
    percentages = [0] * 100
    total_count = sum(trigram_to_freq_local.values())
    elapsed = 0
    for i, tg in enumerate(trigrams):
        pct = int(100 * (elapsed / total_count)) if total_count else 0
        if pct < 100:
            percentages[pct] = i
        elapsed += trigram_to_freq_local[tg]
    print("Trigram percentages:", percentages)
    print("Trigrams loaded:", trigrams[:10], "..." if len(trigrams) > 10 else "")

    # Load bigram frequencies.
    bigram_to_freq_local = defaultdict(int)
    bigram_path = "bigrams.txt"
    if not os.path.exists(bigram_path):
        raise FileNotFoundError(f"Cannot find file: {bigram_path}")
    with open(bigram_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            k, v = parts[:2]
            bigram_to_freq_local[k] = int(v)

    # Load skipgram frequencies.
    skipgram_to_freq_local = defaultdict(int)
    skipgram_path = "1-skip.txt"
    if not os.path.exists(skipgram_path):
        raise FileNotFoundError(f"Cannot find file: {skipgram_path}")
    with open(skipgram_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            k, v = parts[:2]
            skipgram_to_freq_local[k] = int(v)

    return trigram_to_freq_local, bigram_to_freq_local, skipgram_to_freq_local, trigrams


def str_to_tuple(s: str) -> Tuple[int, int]:
    """
    Converts a string like '(1, 2)' into a tuple of ints (1, 2).
    """
    return tuple(map(int, s.strip("()").split(", ")))


def get_iqr_avg(data: List[float]) -> float:
    """
    Returns the mean of data after removing outliers (beyond 1.5*IQR).
    If all data is filtered, returns the unfiltered mean.
    """
    if not data:
        return 0.0
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered = [x for x in data if lower_bound <= x <= upper_bound]
    return np.mean(filtered) if filtered else np.mean(data)


def load_tristroke_data(
    filepath: str, wpm_threshold: int, tg_min_samples: int = 35
) -> List[Any]:
    """
    Loads tristroke (trigram) data from a TSV file.
    Each line is expected to have: (positions) [tab] trigram [tab] [stroke times...]
    Only keeps stroke times if the first value meets the wpm threshold.
    """
    data = []
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Cannot find file: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            a, trigram, *rest = parts
            # Skip trigrams that contain unwanted characters (e.g. uppercase or punctuation).
            if any(ch in "QWERTYUIOPASDFGHJKL:ZXCVBNM<>? " for ch in trigram):
                continue
            strokes = []
            for x in rest:
                try:
                    parsed = str_to_tuple(x)
                except Exception:
                    continue
                if parsed[0] >= wpm_threshold:
                    strokes.append(parsed)
            if len(strokes) < tg_min_samples:
                continue
            try:
                positions = ast.literal_eval(a)
            except (SyntaxError, ValueError):
                continue
            data.append((positions, trigram, *strokes))
    return data


def load_bistroke_data(
    filepath: str, wpm_threshold: int, bg_min_samples: int = 50
) -> List[Any]:
    """
    Loads bistroke (bigram) data from a TSV file.
    Each line is expected to have: (positions) [tab] bigram [tab] [stroke times...]
    Only keeps stroke times if the first value meets the wpm threshold.
    """
    data = []
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Cannot find file: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            a, bigram, *rest = parts
            if any(ch in "QWERTYUIOPASDFGHJKL:ZXCVBNM<>? " for ch in bigram):
                continue
            strokes = []
            for x in rest:
                try:
                    parsed = str_to_tuple(x)
                except Exception:
                    continue
                if parsed[0] >= wpm_threshold:
                    strokes.append(parsed)
            if len(strokes) < bg_min_samples:
                continue
            try:
                positions = ast.literal_eval(a)
            except (SyntaxError, ValueError):
                continue
            data.append((positions, bigram, *strokes))
    return data


################################################################################
# 2. Feature Extraction
################################################################################

# Instantiate the classifier once.
bg_class = Classifier()


def get_bistroke_features(pos: Tuple[Any, Any], bigram: str) -> Tuple[Any, ...]:
    """
    Given a pair of key positions and a bigram string, returns a tuple of 29 values:
      – The first 27 are numeric feature values,
      – Followed by a label (the bigram) and a color (for plotting).
    """
    ((ax, ay), (bx, by)) = pos
    # Compute the expected positions based on the standard "qwerty" layout.
    expected = tuple(bg_class.kb.get_pos(c) for c in bigram)
    col = "red"
    if ((ax, ay), (bx, by)) == expected:
        col = "green"

    freq = bigram_to_freq.get(bigram, 1)
    label = bigram
    cap1 = bigram[0] in "QWERTYUIOPASDFGHJKLZXCVBNM<>?:"
    cap2 = bigram[1] in "QWERTYUIOPASDFGHJKLZXCVBNM<>?:"

    space1 = ay == 0
    space2 = by == 0
    bottom1 = ay == 1
    bottom2 = by == 1
    home1 = ay == 2
    home2 = by == 2
    top1 = ay == 3
    top2 = by == 3

    pinky1 = abs(ax) == 5
    pinky2 = abs(bx) == 5
    ring1 = abs(ax) == 4
    ring2 = abs(bx) == 4
    middle1 = abs(ax) == 3
    middle2 = abs(bx) == 3
    index1 = abs(ax) in (1, 2)
    index2 = abs(bx) in (1, 2)

    # Use the same row offsets as in the classifier (consistent with feature extraction).
    row_offsets = {1: 0.5, 2: 0.0, 3: -0.25}
    offset_ax = row_offsets.get(ay, 0.0)
    offset_bx = row_offsets.get(by, 0.0)
    dy = abs(ay - by)
    dx = abs((ax + offset_ax) - (bx + offset_bx))

    shb = False
    if not (space1 or space2) and ax != 0 and bx != 0:
        shb = (ax // abs(ax)) == (bx // abs(bx))
    scb = ax == bx
    sfb = scb or (shb and (abs(ax) in (1, 2) and abs(bx) in (1, 2)))
    lateral = ((abs(bx) == 1) and (not shb)) or (abs(bx) == 1)
    adjacent = shb and (
        (abs(ax - bx) == 1) or (index1 and middle2) or (index2 and middle1)
    )
    lsb = shb and (((index1 and middle2) or (index2 and middle1)) and dx > 1.5)

    bg_scissor = dy == 2 and dx <= 1
    bg_scissor |= (pinky1 and top1 and ring2 and bottom2) or (
        ring1 and bottom1 and pinky2 and top2
    )
    bg_scissor |= (ring1 and top1 and middle2 and bottom2) or (
        middle1 and bottom1 and ring2 and top2
    )
    bg_scissor |= (index1 and top1 and middle2 and bottom2) or (
        middle1 and bottom1 and index1 and top1
    )
    bg_scissor = bg_scissor and adjacent

    return (
        freq,  # feature 0
        space1,  # feature 1
        space2,  # feature 2
        bottom1,  # feature 3
        bottom2,  # feature 4
        home1,  # feature 5
        home2,  # feature 6
        top1,  # feature 7
        top2,  # feature 8
        pinky1,  # feature 9
        pinky2,  # feature 10
        ring1,  # feature 11
        ring2,  # feature 12
        middle1,  # feature 13
        middle2,  # feature 14
        index1,  # feature 15
        index2,  # feature 16
        lateral,  # feature 17
        shb,  # feature 18
        sfb,  # feature 19
        adjacent,  # feature 20
        bg_scissor,  # feature 21
        lsb,  # feature 22
        cap1,  # feature 23
        cap2,  # feature 24
        dy,  # feature 25
        dx,  # feature 26
        label,  # for reference
        col,  # for plotting
    )


def extract_bigram_features(
    bistroke_data: List[Any],
) -> Tuple[Tuple[Any, ...], np.ndarray, List[str], List[str]]:
    """
    Loops over all bistroke data and extracts bigram features.
    Returns:
      (features_tuple, bg_times, bg_labels, layout_col)
    """
    n = len(bistroke_data)
    bg_freqs = np.zeros(n)
    bg_space1 = np.zeros(n, dtype=bool)
    bg_space2 = np.zeros(n, dtype=bool)
    bg_bottom1 = np.zeros(n, dtype=bool)
    bg_bottom2 = np.zeros(n, dtype=bool)
    bg_home1 = np.zeros(n, dtype=bool)
    bg_home2 = np.zeros(n, dtype=bool)
    bg_top1 = np.zeros(n, dtype=bool)
    bg_top2 = np.zeros(n, dtype=bool)
    bg_pinky1 = np.zeros(n, dtype=bool)
    bg_pinky2 = np.zeros(n, dtype=bool)
    bg_ring1 = np.zeros(n, dtype=bool)
    bg_ring2 = np.zeros(n, dtype=bool)
    bg_middle1 = np.zeros(n, dtype=bool)
    bg_middle2 = np.zeros(n, dtype=bool)
    bg_index1 = np.zeros(n, dtype=bool)
    bg_index2 = np.zeros(n, dtype=bool)
    bg_lateral = np.zeros(n, dtype=bool)
    bg_shb = np.zeros(n, dtype=bool)
    bg_sfb = np.zeros(n, dtype=bool)
    bg_adjacent = np.zeros(n, dtype=bool)
    bg_scissor = np.zeros(n, dtype=bool)
    bg_lsb = np.zeros(n, dtype=bool)
    bg_caps1 = np.zeros(n, dtype=bool)
    bg_caps2 = np.zeros(n, dtype=bool)
    bg_dy = np.zeros(n)
    bg_dx = np.zeros(n)
    bg_labels = []
    layout_col = []
    bg_times = np.zeros(n)

    for i, row in enumerate(bistroke_data):
        pos, bigram, *stroke_info = row
        if len(stroke_info) < 1:
            continue
        # The first stroke_info value is not used further.
        _ = stroke_info[0]
        stroke_times = stroke_info[1:]
        feats = get_bistroke_features(pos, bigram)
        # Unpack features (first 27 values) and then label and color.
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
            label,
            col,
        ) = feats

        bg_labels.append(label)
        layout_col.append(col)
        # Extract the second element of each stroke tuple as the stroke time.
        times_list = [t[1] for t in stroke_times if len(t) > 1]
        bg_times[i] = get_iqr_avg(times_list)

    features_tuple = (
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
    )
    return features_tuple, bg_times, bg_labels, layout_col


def predict_single(model: Any, feature_tuple: Tuple[Any, ...]) -> float:
    """
    Predict a single sample by wrapping each feature in a 1-element np.array.
    """
    wrapped = tuple(np.array([f]) for f in feature_tuple)
    pred = model.predict(wrapped)
    return pred[0]


def extract_trigram_features(
    tristroke_data: List[Any], bg_model: Any
) -> Tuple[Tuple[Any, ...], np.ndarray, List[str], List[str]]:
    """
    For each tristroke (trigram), extract features and use the fitted bigram model
    to predict times for the overlapping bigrams.
    Returns:
      (features_tuple, tg_times, tg_labels, tg_colors)
    """
    n = len(tristroke_data)
    tg_freqs = np.zeros(n)
    tg_bg1_prediction = np.zeros(n)
    tg_bg2_prediction = np.zeros(n)
    tg_sht = np.zeros(n, dtype=bool)
    tg_redirect = np.zeros(n, dtype=bool)
    tg_bad = np.zeros(n, dtype=bool)
    sg_freqs = np.zeros(n)
    tg_sg_features_list = []
    tg_labels = []
    tg_col = []
    tg_times = np.zeros(n)

    for i, row in enumerate(tristroke_data):
        try:
            (pos1, pos2, pos3), trigram, *time_info = row
        except ValueError:
            continue
        tg_freqs[i] = trigram_to_freq.get(trigram, 1)
        times_list = [t[1] for t in time_info if len(t) > 1]
        tg_times[i] = get_iqr_avg(times_list)

        bg1 = trigram[:2]
        bg2 = trigram[1:]
        sg = trigram[::2]

        # Use full features (all 27 numeric values) by dropping label and color.
        bg1_feats = get_bistroke_features((pos1, pos2), bg1)[:-2]
        tg_bg1_prediction[i] = predict_single(bg_model, bg1_feats)

        bg2_feats = get_bistroke_features((pos2, pos3), bg2)[:-2]
        tg_bg2_prediction[i] = predict_single(bg_model, bg2_feats)

        # For skipgram features, we similarly drop the label and color.
        sg_feats = get_bistroke_features((pos1, pos3), sg)[:-2]
        tg_sg_features_list.append(sg_feats)

        expected = tuple(bg_class.kb.get_pos(c) for c in trigram)
        col = "green" if ((pos1, pos2, pos3) == expected) else "red"
        tg_col.append(col)

        sg_freqs[i] = skipgram_to_freq.get(sg, 1)

        ax, _ = pos1
        bx, _ = pos2
        cx, _ = pos3
        if 0 not in (ax, bx, cx):
            try:
                side1 = ax // abs(ax)
                side2 = bx // abs(bx)
                side3 = cx // abs(cx)
                tg_sht[i] = side1 == side2 == side3
            except ZeroDivisionError:
                tg_sht[i] = False

        if tg_sht[i]:
            tg_redirect[i] = (abs(ax) < abs(bx) and abs(cx) < abs(bx)) or (
                abs(ax) > abs(bx) and abs(cx) > abs(bx)
            )
            tg_bad[i] = tg_redirect[i] and (
                not any(abs(x) in (1, 2) for x in (ax, bx, cx))
            )
            tg_redirect[i] = tg_redirect[i] and (not tg_bad[i])

        tg_labels.append(trigram)

    # Convert the list of skipgram feature tuples into an array.
    tg_sg_features_array = np.array(tg_sg_features_list)  # shape: (n, 27)
    # Unpack the 27 skipgram feature arrays.
    skipgram_features = tuple(
        tg_sg_features_array[:, j] for j in range(tg_sg_features_array.shape[1])
    )

    # The complete trigram feature tuple contains 7 arrays (from individual trigram features)
    # plus 27 arrays (skipgram features) = 34 total elements.
    features_tuple = (
        tg_freqs,
        tg_bg1_prediction,
        tg_bg2_prediction,
        tg_sht,
        tg_redirect,
        tg_bad,
        sg_freqs,
    ) + skipgram_features

    return features_tuple, tg_times, tg_labels, tg_col


################################################################################
# 3. Model (Penalty) Functions & TypingModel Wrapper
################################################################################


def bg_penalty(features: Tuple[np.ndarray, ...], *p: float) -> np.ndarray:
    """
    Bigram penalty function: expects 32 parameters.
    'features' is a tuple of 27 arrays from extract_bigram_features().
    """
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

    index2_f = bg_index2.astype(float)
    lateral_f = bg_lateral.astype(float)

    freq_pen = p[0] * np.log(np.clip(bg_freqs + p[1], 1e-8, None)) + p[2]
    base_row_pen = p[3] * (bg_home2 + bg_top2) + p[4] * (bg_top2 + bg_bottom2)
    shb_row_pen = p[5] * (bg_home2 + bg_top2) + p[6] * (bg_top2 + bg_bottom2)
    alt_row_pen = p[7] * (bg_home2 + bg_top2) + p[8] * (bg_top2 + bg_bottom2)
    sfb_row_pen = p[9] * (bg_home2 + bg_top2) + p[10] * (bg_top2 + bg_bottom2)
    sfb_finger_pen = (
        p[11] * bg_pinky2
        + p[12] * bg_ring2
        + p[13] * bg_middle2
        + p[14] * (index2_f - lateral_f)
        + p[15] * bg_lateral
    )
    base_finger_pen = (
        p[16] * bg_pinky2
        + p[17] * bg_ring2
        + p[18] * bg_middle2
        + p[19] * (index2_f - lateral_f)
        + p[20] * bg_lateral
    )
    shb_finger_pen = (
        p[21] * bg_pinky2
        + p[22] * bg_ring2
        + p[23] * bg_middle2
        + p[24] * (index2_f - lateral_f)
        + p[25] * bg_lateral
    )
    alt_finger_pen = (
        p[26] * bg_pinky2
        + p[27] * bg_ring2
        + p[28] * bg_middle2
        + p[29] * (index2_f - lateral_f)
        + p[30] * bg_lateral
    )
    shb_pen = shb_finger_pen * shb_row_pen
    alt_pen = alt_finger_pen * alt_row_pen
    sfb_pen = sfb_finger_pen * sfb_row_pen
    base_weight = 1 + (base_row_pen * base_finger_pen)
    shb_weight = (bg_shb * (1 - bg_sfb)) * shb_pen
    dist_pen = np.sqrt(bg_dx**2 + bg_dy**2) + p[31]
    sfb_weight = bg_sfb * sfb_pen * dist_pen
    alt_weight = (1 - bg_shb) * alt_pen
    return freq_pen * (base_weight + alt_weight + shb_weight + sfb_weight)


def tg_penalty(tg_features: Tuple[np.ndarray, ...], *p: float) -> np.ndarray:
    """
    Trigram penalty function: expects 12 parameters.
    'tg_features' is a tuple from extract_trigram_features() that contains:
      7 trigram-level arrays followed by 27 skipgram feature arrays.
    """
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

    index2_f = bg_index2.astype(float)
    lateral_f = bg_lateral.astype(float)

    freq_pen = p[0] * np.log(tg_freqs + p[1]) + p[2]
    sfs_row_pen = p[3] * (bg_home2 + bg_top2) + p[4] * (bg_bottom2 + bg_top2)
    sfs_finger_pen = (
        p[5] * bg_pinky2
        + p[6] * bg_ring2
        + p[7] * bg_middle2
        + p[8] * (index2_f - lateral_f)
        + p[9] * bg_lateral
    )
    dist_pen = np.sqrt(bg_dx**2 + bg_dy**2) + p[10]
    sfs_weight = bg_sfb * (sfs_row_pen + sfs_finger_pen) * dist_pen
    return tg_bg1_prediction + tg_bg2_prediction + sfs_weight + p[11]


class TypingModel:
    """
    A wrapper for a regression model using either a custom penalty function (via curve_fit)
    or an XGBoost regressor.
    """

    def __init__(
        self,
        model_type: str = "curve_fit",
        penalty_function: Any = None,
        fit_kwargs: dict = None,
        **kwargs,
    ) -> None:
        self.model_type = model_type
        self.penalty_function = penalty_function
        self.fit_kwargs = fit_kwargs if fit_kwargs else {}
        self.params = None
        if model_type == "xgboost":
            if xgb is None:
                raise ImportError("xgboost is not installed.")
            self.model = xgb.XGBRegressor(**kwargs)

    def _convert_features(self, features: Tuple[Any, ...]) -> Tuple[np.ndarray, ...]:
        """
        Convert each element in features (expected as a tuple/list) to a NumPy array.
        """
        return tuple(np.asarray(f) for f in features)

    def fit(self, features: Tuple[Any, ...], y: np.ndarray) -> None:
        if self.model_type == "curve_fit":
            features = self._convert_features(features)
            self.params, self.pcov = curve_fit(
                self.penalty_function, features, y, **self.fit_kwargs
            )
        elif self.model_type == "xgboost":
            X = np.column_stack(features)
            self.model.fit(X, y)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def predict(self, features: Tuple[Any, ...]) -> np.ndarray:
        if self.model_type == "curve_fit":
            features = self._convert_features(features)
            return self.penalty_function(features, *self.params)
        elif self.model_type == "xgboost":
            X = np.column_stack(features)
            return self.model.predict(X)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def evaluate(self, features: Tuple[Any, ...], y: np.ndarray) -> Tuple[float, float]:
        y_pred = self.predict(features)
        residuals = y - y_pred
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        mae = np.mean(np.abs(residuals))
        return r2, mae


################################################################################
# 4. Plotting
################################################################################


def plot_results(
    x: np.ndarray,
    y: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str],
    colors: List[str],
    xlabel: str,
    ylabel: str,
    title: str,
    annotate_threshold: int = 25,
) -> None:
    """
    Plots a scatter plot (data) with a line (fit). If number of points is small,
    annotates the points with labels.
    """
    plt.figure(figsize=(9, 6))
    plt.scatter(x, y, s=40, c=colors, alpha=0.8, label="Data")
    plt.plot(x, y_pred, color="black", linewidth=2, label="Fit")
    if len(x) <= annotate_threshold:
        for xi, yi, lab in zip(x, y, labels):
            plt.annotate(
                f"'{lab}'",
                (xi, yi),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
            )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


################################################################################
# 5. Main
################################################################################


def main() -> None:
    parser = argparse.ArgumentParser(description="Typing Model Training")
    parser.add_argument(
        "--model",
        type=str,
        default="curve_fit",
        choices=["curve_fit", "xgboost"],
        help="Choose model type",
    )
    args = parser.parse_args()

    global trigram_to_freq, bigram_to_freq, skipgram_to_freq
    (trigram_to_freq, bigram_to_freq, skipgram_to_freq, trigrams) = (
        load_ngram_frequencies()
    )

    tristroke_data = load_tristroke_data("tristrokes.tsv", wpm)
    bistroke_data = load_bistroke_data("bistrokes.tsv", wpm)

    # Extract bigram features.
    bg_features, bg_times, bg_labels, layout_col = extract_bigram_features(
        bistroke_data
    )

    if args.model == "curve_fit":
        bg_fit_kwargs = {"method": "lm", "maxfev": 750000, "p0": np.ones(32)}
        bg_model = TypingModel(
            model_type="curve_fit",
            penalty_function=bg_penalty,
            fit_kwargs=bg_fit_kwargs,
        )
    else:
        bg_model = TypingModel(model_type="xgboost")
    bg_model.fit(bg_features, bg_times)
    bg_r2, bg_mae = bg_model.evaluate(bg_features, bg_times)
    print(f"Bigram Model - R^2: {bg_r2:.4f}, MAE: {bg_mae:.3f}")

    # Sort by bigram frequency for plotting.
    sorted_idx = np.argsort(bg_features[0])
    x_sorted = bg_features[0][sorted_idx]
    y_sorted = bg_times[sorted_idx]
    bg_pred = bg_model.predict(bg_features)
    pred_sorted = bg_pred[sorted_idx]
    labels_sorted = [bg_labels[i] for i in sorted_idx]
    colors_sorted = [layout_col[i] for i in sorted_idx]
    plot_results(
        x_sorted,
        y_sorted,
        pred_sorted,
        labels_sorted,
        colors_sorted,
        "Bigram Frequency",
        "Avg Typing Time (ms)",
        "Bigram Model Fit",
    )

    # Extract trigram features.
    tg_features, tg_times, tg_labels, tg_colors = extract_trigram_features(
        tristroke_data, bg_model
    )

    if args.model == "curve_fit":
        tg_fit_kwargs = {"method": "trf", "maxfev": 750000, "p0": np.ones(12)}
        tg_model = TypingModel(
            model_type="curve_fit",
            penalty_function=tg_penalty,
            fit_kwargs=tg_fit_kwargs,
        )
    else:
        tg_model = TypingModel(model_type="xgboost")
    tg_model.fit(tg_features, tg_times)
    tg_r2, tg_mae = tg_model.evaluate(tg_features, tg_times)
    print(f"Trigram Model - R^2: {tg_r2:.4f}, MAE: {tg_mae:.3f}")

    sorted_idx_tg = np.argsort(tg_features[0])
    x_sorted_tg = tg_features[0][sorted_idx_tg]
    y_sorted_tg = tg_times[sorted_idx_tg]
    tg_pred = tg_model.predict(tg_features)
    pred_sorted_tg = tg_pred[sorted_idx_tg]
    labels_sorted_tg = [tg_labels[i] for i in sorted_idx_tg]
    colors_sorted_tg = [tg_colors[i] for i in sorted_idx_tg]
    plot_results(
        x_sorted_tg,
        y_sorted_tg,
        pred_sorted_tg,
        labels_sorted_tg,
        colors_sorted_tg,
        "Trigram Frequency",
        "Avg Typing Time (ms)",
        "Trigram Model Fit",
    )


if __name__ == "__main__":
    main()
