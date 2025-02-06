"""
typing_model.py – An improved and extensible typing analysis script.

Usage:
    python typing_model.py --model curve_fit|xgboost
      [--trigrams_file TRIGRAMS] [--bigrams_file BIGRAMS]
      [--skip_file SKIP] [--tristrokes_file TRISTROKES]
      [--bistrokes_file BISTROKES]
"""

import argparse
import ast
import os
import pickle  # For saving the final model
from collections import defaultdict
from typing import Any, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.model_selection import KFold  # for k-fold CV

# Try to import xgboost if available.
try:
    import xgboost as xgb
except ImportError:
    xgb = None

# Import the classifier module.
from classifier import Classifier

################################################################################
# Global constants
################################################################################

WPM_THRESHOLD = 80  # words per minute threshold

################################################################################
# Data Loading Functions
################################################################################

def load_ngram_frequencies(trigrams_file: str, bigrams_file: str, skip_file: str
                            ) -> Tuple[dict, dict, dict, List[str]]:
    """
    Loads frequency data for trigrams, bigrams, and skipgrams from text files.
    Returns:
      (trigram_to_freq, bigram_to_freq, skipgram_to_freq, list_of_trigrams).
    """
    trigram_to_freq = defaultdict(int)
    trigrams = []
    if not os.path.exists(trigrams_file):
        raise FileNotFoundError(f"Cannot find file: {trigrams_file}")
    with open(trigrams_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            k, v = parts[:2]
            # Only keep trigrams that do NOT contain any unwanted uppercase/punctuation characters.
            if not any(c in "QWERTYUIOPASDFGHJKL:ZXCVBNM<>? " for c in k):
                trigram_to_freq[k] = int(v)
                trigrams.append(k)

    total_count = sum(trigram_to_freq.values())
    percentages = [0] * 100
    elapsed = 0
    for i, tg in enumerate(trigrams):
        pct = int(100 * (elapsed / total_count)) if total_count else 0
        if pct < 100:
            percentages[pct] = i
        elapsed += trigram_to_freq[tg]
    print("Trigram percentages:", percentages)
    print("Trigrams loaded:", trigrams[:10], "..." if len(trigrams) > 10 else "")

    # Load bigram frequencies.
    bigram_to_freq = defaultdict(int)
    if not os.path.exists(bigrams_file):
        raise FileNotFoundError(f"Cannot find file: {bigrams_file}")
    with open(bigrams_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            k, v = parts[:2]
            bigram_to_freq[k] = int(v)

    # Load skipgram frequencies.
    skipgram_to_freq = defaultdict(int)
    if not os.path.exists(skip_file):
        raise FileNotFoundError(f"Cannot find file: {skip_file}")
    with open(skip_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            k, v = parts[:2]
            skipgram_to_freq[k] = int(v)

    return trigram_to_freq, bigram_to_freq, skipgram_to_freq, trigrams


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


def load_tristroke_data(filepath: str, wpm_threshold: int, tg_min_samples: int = 35
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


def load_bistroke_data(filepath: str, wpm_threshold: int, bg_min_samples: int = 50
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
# Feature Extraction (using list accumulation)
################################################################################

# Instantiate the classifier once.
bg_classifier = Classifier()

def get_bistroke_features(pos: Tuple[Any, Any], bigram: str) -> Tuple[Any, ...]:
    """
    Given a pair of key positions and a bigram string, returns a tuple of 29 values:
      – The first 27 are numeric feature values,
      – Followed by a label (the bigram) and a color (for plotting).
    """
    ((ax, ay), (bx, by)) = pos
    expected = tuple(bg_classifier.kb.get_pos(c) for c in bigram)
    col = "red"
    if ((ax, ay), (bx, by)) == expected:
        col = "green"

    # Lookup frequency from a bigram frequency dictionary; this is set later.
    freq = bg_classifier.bigram_freq.get(bigram, 1)
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

    row_offsets = {1: 0.5, 2: 0.0, 3: -0.25}
    offset_ax = row_offsets.get(ay, 0.0)
    offset_bx = row_offsets.get(by, 0.0)
    dy = abs(ay - by)
    dx = abs((ax + offset_ax) - (bx + offset_bx))

    shb = False
    if not (space1 or space2) and ax != 0 and bx != 0:
        shb = (ax // abs(ax)) == (bx // abs(bx))
    scb = ax == bx
    sfb = scb or (shb and ((index1 and middle2) or (index2 and middle1)))
    lateral = ((abs(bx) == 1) and (not shb)) or (abs(bx) == 1)
    adjacent = shb and ((abs(ax - bx) == 1) or (index1 and middle2) or (index2 and middle1))
    lsb = shb and (((index1 and middle2) or (index2 and middle1)) and dx > 1.5)

    bg_scissor = ((dy == 2 and dx <= 1) or
                  ((pinky1 and top1 and ring2 and bottom2) or (ring1 and bottom1 and pinky2 and top2)) or
                  ((ring1 and top1 and middle2 and bottom2) or (middle1 and bottom1 and ring2 and top2)) or
                  ((index1 and top1 and middle2 and bottom2) or (middle1 and bottom1 and index1 and top1)))
    bg_scissor = bg_scissor and adjacent

    return (
        freq,         # feature 0
        space1,       # feature 1
        space2,       # feature 2
        bottom1,      # feature 3
        bottom2,      # feature 4
        home1,        # feature 5
        home2,        # feature 6
        top1,         # feature 7
        top2,         # feature 8
        pinky1,       # feature 9
        pinky2,       # feature 10
        ring1,        # feature 11
        ring2,        # feature 12
        middle1,      # feature 13
        middle2,      # feature 14
        index1,       # feature 15
        index2,       # feature 16
        lateral,      # feature 17
        shb,          # feature 18
        sfb,          # feature 19
        adjacent,     # feature 20
        bg_scissor,   # feature 21
        lsb,          # feature 22
        cap1,         # feature 23
        cap2,         # feature 24
        dy,           # feature 25
        dx,           # feature 26
        label,        # label (for reference)
        col,          # color (for plotting)
    )

def update_classifier_bigram_freq(bigram_to_freq: dict) -> None:
    """Update the classifier instance with bigram frequency data."""
    bg_classifier.bigram_freq = bigram_to_freq

def extract_bigram_features(bistroke_data: List[Any]
                           ) -> Tuple[Tuple[np.ndarray, ...], np.ndarray, List[str], List[str]]:
    """
    Loops over all bistroke data and extracts bigram features.
    Returns:
      (features_tuple, bg_times, bg_labels, layout_col)
    """
    # Use lists to accumulate samples.
    freqs, space1s, space2s, bottom1s, bottom2s = [], [], [], [], []
    home1s, home2s, top1s, top2s = [], [], [], []
    pinky1s, pinky2s, ring1s, ring2s = [], [], [], []
    middle1s, middle2s, index1s, index2s = [], [], [], []
    laterals, shbs, sfb_vals, adjacents = [], [], [], []
    scissors, lsbs, caps1s, caps2s = [], [], [], []
    dys, dxs = [], []
    labels, colors = [], []
    times_list_final = []

    for row in bistroke_data:
        try:
            pos, bigram, *stroke_info = row
        except ValueError:
            continue
        if len(stroke_info) < 1:
            continue

        feats = get_bistroke_features(pos, bigram)
        (freq, sp1, sp2, bot1, bot2, hom1, hom2, top1, top2,
         p1, p2, r1, r2, m1, m2, i1, i2, lateral, shb, sfb, adjacent,
         scissor, lsb, cap1, cap2, dy, dx, label, col) = feats

        freqs.append(freq)
        space1s.append(float(sp1))
        space2s.append(float(sp2))
        bottom1s.append(float(bot1))
        bottom2s.append(float(bot2))
        home1s.append(float(hom1))
        home2s.append(float(hom2))
        top1s.append(float(top1))
        top2s.append(float(top2))
        pinky1s.append(float(p1))
        pinky2s.append(float(p2))
        ring1s.append(float(r1))
        ring2s.append(float(r2))
        middle1s.append(float(m1))
        middle2s.append(float(m2))
        index1s.append(float(i1))
        index2s.append(float(i2))
        laterals.append(float(lateral))
        shbs.append(float(shb))
        sfb_vals.append(float(sfb))
        adjacents.append(float(adjacent))
        scissors.append(float(scissor))
        lsbs.append(float(lsb))
        caps1s.append(float(cap1))
        caps2s.append(float(cap2))
        dys.append(float(dy))
        dxs.append(float(dx))
        labels.append(label)
        colors.append(col)
        times = [t[1] for t in stroke_info if len(t) > 1]
        times_list_final.append(get_iqr_avg(times))

    bg_features = (
        np.array(freqs),
        np.array(space1s),
        np.array(space2s),
        np.array(bottom1s),
        np.array(bottom2s),
        np.array(home1s),
        np.array(home2s),
        np.array(top1s),
        np.array(top2s),
        np.array(pinky1s),
        np.array(pinky2s),
        np.array(ring1s),
        np.array(ring2s),
        np.array(middle1s),
        np.array(middle2s),
        np.array(index1s),
        np.array(index2s),
        np.array(laterals),
        np.array(shbs),
        np.array(sfb_vals),
        np.array(adjacents),
        np.array(scissors),
        np.array(lsbs),
        np.array(caps1s),
        np.array(caps2s),
        np.array(dys),
        np.array(dxs),
    )
    return bg_features, np.array(times_list_final), labels, colors


def predict_single(model: Any, feature_tuple: Tuple[Any, ...]) -> float:
    """
    Predict a single sample by wrapping each feature in a 1-element np.array.
    """
    wrapped = tuple(np.array([f]) for f in feature_tuple)
    pred = model.predict(wrapped)
    return pred[0]


def extract_trigram_features(tristroke_data: List[Any], bg_model: Any,
                             trigram_to_freq: dict, skipgram_to_freq: dict
                            ) -> Tuple[Tuple[np.ndarray, ...], np.ndarray, List[str], List[str]]:
    """
    For each tristroke (trigram), extract features and use the fitted bigram model
    to predict times for the overlapping bigrams.
    Returns:
      (features_tuple, tg_times, tg_labels, tg_colors)
    """
    tg_freqs = []
    tg_bg1_prediction = []
    tg_bg2_prediction = []
    tg_sht = []
    tg_redirect = []
    tg_bad = []
    sg_freqs = []
    skip_features_list = []
    tg_labels = []
    tg_colors = []
    tg_times = []

    for row in tristroke_data:
        try:
            (pos1, pos2, pos3), trigram, *time_info = row
        except ValueError:
            continue
        tg_freqs.append(trigram_to_freq.get(trigram, 1))
        times = [t[1] for t in time_info if len(t) > 1]
        tg_times.append(get_iqr_avg(times))

        bg1 = trigram[:2]
        bg2 = trigram[1:]
        sg = trigram[::2]

        bg1_feats = get_bistroke_features((pos1, pos2), bg1)[:-2]
        tg_bg1_prediction.append(predict_single(bg_model, bg1_feats))

        bg2_feats = get_bistroke_features((pos2, pos3), bg2)[:-2]
        tg_bg2_prediction.append(predict_single(bg_model, bg2_feats))

        sg_feats = get_bistroke_features((pos1, pos3), sg)[:-2]
        skip_features_list.append(sg_feats)

        expected = tuple(bg_classifier.kb.get_pos(c) for c in trigram)
        col = "green" if ((pos1, pos2, pos3) == expected) else "red"
        tg_colors.append(col)

        sg_freqs.append(skipgram_to_freq.get(sg, 1))

        ax, _ = pos1
        bx, _ = pos2
        cx, _ = pos3
        if 0 not in (ax, bx, cx):
            try:
                side1 = ax // abs(ax)
                side2 = bx // abs(bx)
                side3 = cx // abs(cx)
                sht = (side1 == side2 == side3)
            except ZeroDivisionError:
                sht = False
        else:
            sht = False
        tg_sht.append(float(sht))

        if sht:
            redir = (abs(ax) < abs(bx) and abs(cx) < abs(bx)) or (abs(ax) > abs(bx) and abs(cx) > abs(bx))
            tg_redirect.append(float(redir))
            bad = redir and (not any(abs(x) in (1, 2) for x in (ax, bx, cx)))
            tg_bad.append(float(bad))
        else:
            tg_redirect.append(0.0)
            tg_bad.append(0.0)

        tg_labels.append(trigram)

    skip_features_array = np.array(skip_features_list)  # shape: (n, 27)
    skip_features = tuple(skip_features_array[:, j] for j in range(skip_features_array.shape[1]))

    features_tuple = (
        np.array(tg_freqs),
        np.array(tg_bg1_prediction),
        np.array(tg_bg2_prediction),
        np.array(tg_sht),
        np.array(tg_redirect),
        np.array(tg_bad),
        np.array(sg_freqs),
    ) + skip_features  # Total of 7 + 27 = 34 elements

    return features_tuple, np.array(tg_times), tg_labels, tg_colors

################################################################################
# Model (Penalty) Functions & TypingModel Wrapper
################################################################################

def bg_penalty(features: Tuple[np.ndarray, ...], *p: float) -> np.ndarray:
    """
    Bigram penalty function: expects 32 parameters.
    'features' is a tuple of 27 arrays from extract_bigram_features().
    """
    (bg_freqs, bg_space1, bg_space2, bg_bottom1, bg_bottom2, bg_home1, bg_home2,
     bg_top1, bg_top2, bg_pinky1, bg_pinky2, bg_ring1, bg_ring2, bg_middle1,
     bg_middle2, bg_index1, bg_index2, bg_lateral, bg_shb, bg_sfb, bg_adjacent,
     bg_scissor, bg_lsb, bg_caps1, bg_caps2, bg_dy, bg_dx) = features

    index2_f = bg_index2.astype(float)
    lateral_f = bg_lateral.astype(float)

    freq_pen = p[0] * np.log(np.clip(bg_freqs + p[1], 1e-8, None)) + p[2]
    base_row_pen = p[3] * (bg_home2 + bg_top2) + p[4] * (bg_top2 + bg_bottom2)
    shb_row_pen = p[5] * (bg_home2 + bg_top2) + p[6] * (bg_top2 + bg_bottom2)
    alt_row_pen = p[7] * (bg_home2 + bg_top2) + p[8] * (bg_top2 + bg_bottom2)
    sfb_row_pen = p[9] * (bg_home2 + bg_top2) + p[10] * (bg_top2 + bg_bottom2)
    sfb_finger_pen = (p[11] * bg_pinky2 +
                      p[12] * bg_ring2 +
                      p[13] * bg_middle2 +
                      p[14] * (index2_f - lateral_f) +
                      p[15] * bg_lateral)
    base_finger_pen = (p[16] * bg_pinky2 +
                       p[17] * bg_ring2 +
                       p[18] * bg_middle2 +
                       p[19] * (index2_f - lateral_f) +
                       p[20] * bg_lateral)
    shb_finger_pen = (p[21] * bg_pinky2 +
                      p[22] * bg_ring2 +
                      p[23] * bg_middle2 +
                      p[24] * (index2_f - lateral_f) +
                      p[25] * bg_lateral)
    alt_finger_pen = (p[26] * bg_pinky2 +
                      p[27] * bg_ring2 +
                      p[28] * bg_middle2 +
                      p[29] * (index2_f - lateral_f) +
                      p[30] * bg_lateral)
    shb_pen = shb_finger_pen * shb_row_pen
    alt_pen = alt_finger_pen * alt_row_pen
    sfb_pen = bg_sfb * p[31] * (np.sqrt(bg_dx**2 + bg_dy**2) + 1)
    base_weight = 1 + (base_row_pen * base_finger_pen)
    shb_weight = (bg_shb * (1 - bg_sfb)) * shb_pen
    alt_weight = (1 - bg_shb) * alt_pen
    return freq_pen * (base_weight + alt_weight + shb_weight + sfb_pen)


def tg_penalty(tg_features: Tuple[np.ndarray, ...], *p: float) -> np.ndarray:
    """
    Trigram penalty function: expects 12 parameters.
    'tg_features' is a tuple from extract_trigram_features() that contains:
      7 trigram-level arrays followed by 27 skipgram feature arrays.
    """
    (tg_freqs, tg_bg1_prediction, tg_bg2_prediction, tg_sht, tg_redirect, tg_bad,
     sg_freq, bg_freqs, bg_space1, bg_space2, bg_bottom1, bg_bottom2, bg_home1,
     bg_home2, bg_top1, bg_top2, bg_pinky1, bg_pinky2, bg_ring1, bg_ring2,
     bg_middle1, bg_middle2, bg_index1, bg_index2, bg_lateral, bg_shb, bg_sfb,
     bg_adjacent, bg_scissor, bg_lsb, bg_caps1, bg_caps2, bg_dy, bg_dx) = tg_features

    index2_f = bg_index2.astype(float)
    lateral_f = bg_lateral.astype(float)

    freq_pen = p[0] * np.log(tg_freqs + p[1]) + p[2]
    sfs_row_pen = p[3] * (bg_home2 + bg_top2) + p[4] * (bg_bottom2 + bg_top2)
    sfs_finger_pen = (p[5] * bg_pinky2 +
                      p[6] * bg_ring2 +
                      p[7] * bg_middle2 +
                      p[8] * (index2_f - lateral_f) +
                      p[9] * bg_lateral)
    dist_pen = np.sqrt(bg_dx**2 + bg_dy**2) + p[10]
    sfs_weight = bg_sfb * (sfs_row_pen + sfs_finger_pen) * dist_pen
    return tg_bg1_prediction + tg_bg2_prediction + sfs_weight + p[11]


class TypingModel:
    """
    A wrapper for a regression model using either a custom penalty function (via curve_fit)
    or an XGBoost regressor.
    """

    def __init__(self, model_type: str = "curve_fit", penalty_function: Any = None,
                 fit_kwargs: dict = None, **kwargs) -> None:
        self.model_type = model_type
        self.penalty_function = penalty_function
        self.fit_kwargs = fit_kwargs if fit_kwargs else {}
        self.params = None
        if model_type == "xgboost":
            if xgb is None:
                raise ImportError("xgboost is not installed.")
            self.model = xgb.XGBRegressor(**kwargs)

    def _convert_features(self, features: Tuple[Any, ...]) -> Tuple[np.ndarray, ...]:
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
# Cross-Validation Function
################################################################################

def cross_validate_model(model_args: dict, features: Tuple[np.ndarray, ...],
                         y: np.ndarray, n_splits: int = 5) -> Tuple[float, float]:
    """
    Perform k-fold cross validation for a given model configuration.
    model_args: dictionary of arguments to pass to TypingModel (e.g., model_type, penalty_function, etc.)
    features: tuple of numpy arrays (each with shape (n,))
    y: target numpy array (shape (n,))
    Returns:
      (average_r2, average_mae)
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    r2_scores = []
    mae_scores = []
    n = len(y)
    for train_index, test_index in kf.split(np.arange(n)):
        train_features = tuple(f[train_index] for f in features)
        test_features = tuple(f[test_index] for f in features)
        y_train = y[train_index]
        y_test = y[test_index]
        # Create a new model instance for each fold.
        model = TypingModel(**model_args)
        model.fit(train_features, y_train)
        r2, mae = model.evaluate(test_features, y_test)
        r2_scores.append(r2)
        mae_scores.append(mae)
    return np.mean(r2_scores), np.mean(mae_scores)

################################################################################
# Plotting
################################################################################

def plot_results(x: np.ndarray, y: np.ndarray, y_pred: np.ndarray,
                 labels: List[str], colors: List[str],
                 xlabel: str, ylabel: str, title: str,
                 annotate_threshold: int = 25) -> None:
    plt.figure(figsize=(9, 6))
    plt.scatter(x, y, s=40, c=colors, alpha=0.8, label="Data")
    plt.plot(x, y_pred, color="black", linewidth=2, label="Fit")
    if len(x) <= annotate_threshold:
        for xi, yi, lab in zip(x, y, labels):
            plt.annotate(f"'{lab}'", (xi, yi), xytext=(0, 5),
                         textcoords="offset points", ha="center")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

################################################################################
# Main
################################################################################

def main() -> None:
    parser = argparse.ArgumentParser(description="Typing Model Training")
    parser.add_argument("--model", type=str, default="curve_fit",
                        choices=["curve_fit", "xgboost"], help="Choose model type")
    parser.add_argument("--trigrams_file", type=str, default="trigrams.txt",
                        help="Path to trigram frequencies file")
    parser.add_argument("--bigrams_file", type=str, default="bigrams.txt",
                        help="Path to bigram frequencies file")
    parser.add_argument("--skip_file", type=str, default="1-skip.txt",
                        help="Path to skipgram frequencies file")
    parser.add_argument("--tristrokes_file", type=str, default="tristrokes.tsv",
                        help="Path to tristroke data file")
    parser.add_argument("--bistrokes_file", type=str, default="bistrokes.tsv",
                        help="Path to bistroke data file")
    args = parser.parse_args()

    # Load frequency data.
    trigram_to_freq, bigram_to_freq, skipgram_to_freq, trigrams = load_ngram_frequencies(
        args.trigrams_file, args.bigrams_file, args.skip_file)
    update_classifier_bigram_freq(bigram_to_freq)

    # Load stroke data.
    tristroke_data = load_tristroke_data(args.tristrokes_file, WPM_THRESHOLD)
    bistroke_data = load_bistroke_data(args.bistrokes_file, WPM_THRESHOLD)

    ###########################################################
    # BIGRAM MODEL: Cross-Validation and Final Training & Saving
    ###########################################################
    bg_features, bg_times, bg_labels, layout_col = extract_bigram_features(bistroke_data)

    if args.model == "curve_fit":
        bg_model_args = {
            "model_type": "curve_fit",
            "penalty_function": bg_penalty,
            "fit_kwargs": {"method": "lm", "maxfev": 750000, "p0": np.ones(32)}
        }
    else:
        bg_model_args = {"model_type": "xgboost"}

    bg_cv_r2, bg_cv_mae = cross_validate_model(bg_model_args, bg_features, bg_times, n_splits=5)
    print(f"Bigram Model CV - R^2: {bg_cv_r2:.4f}, MAE: {bg_cv_mae:.3f}")

    # Train final bigram model on the whole data.
    bg_model = TypingModel(**bg_model_args)
    bg_model.fit(bg_features, bg_times)
    # Save the final bigram model.
    with open("bigram_model.pkl", "wb") as f:
        pickle.dump(bg_model, f)
    print("Final bigram model saved to 'bigram_model.pkl'.")

    # Plot bigram results.
    sorted_idx = np.argsort(bg_features[0])
    x_sorted = bg_features[0][sorted_idx]
    y_sorted = bg_times[sorted_idx]
    bg_pred = bg_model.predict(bg_features)
    pred_sorted = bg_pred[sorted_idx]
    labels_sorted = [bg_labels[i] for i in sorted_idx]
    colors_sorted = [layout_col[i] for i in sorted_idx]
    plot_results(x_sorted, y_sorted, pred_sorted, labels_sorted, colors_sorted,
                 "Bigram Frequency", "Avg Typing Time (ms)", "Bigram Model Fit")

    ###########################################################
    # TRIGRAM MODEL: Cross-Validation and Final Training & Saving
    ###########################################################
    tg_features, tg_times, tg_labels, tg_colors = extract_trigram_features(
        tristroke_data, bg_model, trigram_to_freq, skipgram_to_freq)

    if args.model == "curve_fit":
        tg_model_args = {
            "model_type": "curve_fit",
            "penalty_function": tg_penalty,
            "fit_kwargs": {"method": "trf", "maxfev": 750000, "p0": np.ones(12)}
        }
    else:
        tg_model_args = {"model_type": "xgboost"}

    tg_cv_r2, tg_cv_mae = cross_validate_model(tg_model_args, tg_features, tg_times, n_splits=5)
    print(f"Trigram Model CV - R^2: {tg_cv_r2:.4f}, MAE: {tg_cv_mae:.3f}")

    # Train final trigram model on the whole data.
    tg_model = TypingModel(**tg_model_args)
    tg_model.fit(tg_features, tg_times)
    # Save the final trigram model.
    with open("trigram_model.pkl", "wb") as f:
        pickle.dump(tg_model, f)
    print("Final trigram model saved to 'trigram_model.pkl'.")

    # Plot trigram results.
    sorted_idx_tg = np.argsort(tg_features[0])
    x_sorted_tg = tg_features[0][sorted_idx_tg]
    y_sorted_tg = tg_times[sorted_idx_tg]
    tg_pred = tg_model.predict(tg_features)
    pred_sorted_tg = tg_pred[sorted_idx_tg]
    labels_sorted_tg = [tg_labels[i] for i in sorted_idx_tg]
    colors_sorted_tg = [tg_colors[i] for i in sorted_idx_tg]
    plot_results(x_sorted_tg, y_sorted_tg, pred_sorted_tg, labels_sorted_tg,
                 colors_sorted_tg, "Trigram Frequency", "Avg Typing Time (ms)", "Trigram Model Fit")

if __name__ == "__main__":
    main()
