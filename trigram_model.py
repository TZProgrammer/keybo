"""
typing_model.py – An improved typing analysis script with extra features,
regularized XGBoost hyperparameters, and an option to predict the last‐letter time vs. the full sequence.

Usage:
    python typing_model.py --model curve_fit|xgboost [--predict_mode full|last]
      [--trigrams_file TRIGRAMS] [--bigrams_file BIGRAMS]
      [--skip_file SKIP] [--tristrokes_file TRISTROKES]
      [--bistrokes_file BISTROKES]
"""

import argparse
import ast
import os
import pickle
from collections import defaultdict
from typing import Any, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.model_selection import KFold

try:
    import xgboost as xgb
except ImportError:
    xgb = None

from classifier import Classifier

# Global constant
WPM_THRESHOLD = 80

##########################################################################
# For key features, define a mapping from key characters to numeric IDs.
##########################################################################
ALL_KEYS = list("qwertyuiopasdfghjkl;zxcvbnm,./ QWERTYUIOPASDFGHJKL:ZXCVBNM<>?")
# (The space key is included in our data.)
char_to_id = {c: i for i, c in enumerate(ALL_KEYS)}


##########################################################################
# Data Loading and Utility Functions
##########################################################################
def load_ngram_frequencies(
    trigrams_file: str, bigrams_file: str, skip_file: str
) -> Tuple[dict, dict, dict, List[str]]:
    trigram_to_freq = defaultdict(int)
    trigrams = []
    # Allow both lower and upper case letters (plus punctuation and space)
    allowed_chars = "qwertyuiopasdfghjkl;zxcvbnm,./QWERTYUIOPASDFGHJKL:ZXCVBNM<>? "
    if not os.path.exists(trigrams_file):
        raise FileNotFoundError(f"Cannot find file: {trigrams_file}")
    with open(trigrams_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            k, v = parts[:2]
            k = k.strip()  # do NOT convert to lowercase
            # Only accept keys that are exactly three characters long and valid.
            if len(k) != 3 or not all(c in allowed_chars for c in k):
                continue
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

    bigram_to_freq = defaultdict(int)
    if not os.path.exists(bigrams_file):
        raise FileNotFoundError(f"Cannot find file: {bigrams_file}")
    with open(bigrams_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            k, v = parts[:2]
            k = k.strip()  # keep case
            if len(k) != 2 or not all(c in allowed_chars for c in k):
                continue
            bigram_to_freq[k] = int(v)

    skipgram_to_freq = defaultdict(int)
    if not os.path.exists(skip_file):
        raise FileNotFoundError(f"Cannot find file: {skip_file}")
    with open(skip_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            k, v = parts[:2]
            k = k.strip()  # keep case
            if not all(c in allowed_chars for c in k):
                continue
            skipgram_to_freq[k] = int(v)

    return trigram_to_freq, bigram_to_freq, skipgram_to_freq, trigrams


def str_to_tuple(s: str) -> Tuple[int, ...]:
    return tuple(map(int, s.strip("()").split(", ")))


def get_iqr_avg(data: List[float]) -> float:
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
    data = []
    allowed_chars = "qwertyuiopasdfghjkl;zxcvbnm,./QWERTYUIOPASDFGHJKL:ZXCVBNM<>? "
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Cannot find file: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            a, trigram, *rest = parts
            trigram = trigram.strip()  # preserve case
            if len(trigram) != 3 or not all(ch in allowed_chars for ch in trigram):
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
    data = []
    allowed_chars = "qwertyuiopasdfghjkl;zxcvbnm,./QWERTYUIOPASDFGHJKL:ZXCVBNM<>? "
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Cannot find file: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            a, bigram, *rest = parts
            bigram = bigram.strip()  # preserve case
            if len(bigram) != 2 or not all(c in allowed_chars for c in bigram):
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


##########################################################################
# Feature Extraction (with predict_mode option)
##########################################################################

# Instantiate the classifier once.
bg_classifier = Classifier()


def get_bistroke_features(pos: Tuple[Any, Any], bigram: str) -> Tuple[Any, ...]:
    """
    Extended bigram feature extraction.

    Returns a tuple containing:
      - 27 original features (frequency, row/column booleans, dx, dy, etc.)
      - 4 derived features (rotation angle, inwards/outwards flags, Euclidean distance)
      - 6 raw key features (key IDs and raw coordinates)
      - 1 different-hand flag (1.0 if the two keys are typed with different hands, 0.0 otherwise)
      - Followed by the label (the bigram) and a color.

    Total length = 27 + 4 + 6 + 1 + 2 = 40.
    """
    ((ax, ay), (bx, by)) = pos
    k1, k2 = bigram[0], bigram[1]

    expected = tuple(bg_classifier.kb.get_pos(c) for c in bigram)
    col = "green" if ((ax, ay), (bx, by)) == expected else "red"
    freq = bg_classifier.bigram_freq.get(bigram, 1)
    cap1 = k1.isupper()
    cap2 = k2.isupper()
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
    lateral = abs(bx) == 1
    adjacent = shb and (
        (abs(ax - bx) == 1) or (index1 and middle2) or (index2 and middle1)
    )
    lsb = shb and (((index1 and middle2) or (index2 and middle1)) and dx > 1.5)
    bg_scissor = (
        (dy == 2 and dx <= 1)
        or (
            (pinky1 and top1 and ring2 and bottom2)
            or (ring1 and bottom1 and pinky2 and top2)
        )
        or (
            (ring1 and top1 and middle2 and bottom2)
            or (middle1 and bottom1 and ring2 and top2)
        )
        or (
            (index1 and top1 and middle2 and bottom2)
            or (middle1 and bottom1 and index1 and top1)
        )
    )
    bg_scissor = bg_scissor and adjacent

    angle = bg_classifier.get_rotation((k1, k2))
    if angle is None:
        angle = 0.0
    inwards = bg_classifier.inwards_rotation((k1, k2))
    outwards = bg_classifier.outwards_rotation((k1, k2))
    distance = bg_classifier.get_distance((k1, k2))

    k1_id = char_to_id.get(k1, -1)
    k2_id = char_to_id.get(k2, -1)
    diff_hand = 1.0 if bg_classifier.different_hand((k1, k2)) else 0.0

    orig_features = (
        freq,
        float(space1),
        float(space2),
        float(bottom1),
        float(bottom2),
        float(home1),
        float(home2),
        float(top1),
        float(top2),
        float(pinky1),
        float(pinky2),
        float(ring1),
        float(ring2),
        float(middle1),
        float(middle2),
        float(index1),
        float(index2),
        float(lateral),
        float(shb),
        float(sfb),
        float(adjacent),
        float(bg_scissor),
        float(lsb),
        float(cap1),
        float(cap2),
        float(dy),
        float(dx),
    )
    derived_features = (float(angle), float(inwards), float(outwards), float(distance))
    raw_key_info = (
        float(k1_id),
        float(k2_id),
        float(ax),
        float(ay),
        float(bx),
        float(by),
    )
    new_feature = (float(diff_hand),)

    return orig_features + derived_features + raw_key_info + new_feature + (bigram, col)


def extract_bigram_features(
    bistroke_data: List[Any], predict_mode: str = "full"
) -> Tuple[Tuple[np.ndarray, ...], np.ndarray, List[str], List[str]]:
    """
    Loops over all bistroke data and extracts extended bigram features.
    For bigrams the target is computed as the IQR average of the last bigram
    measurement for each trial—thus, regardless of the predict_mode, the result
    is the same.
    """
    orig_list = [[] for _ in range(27)]
    derived_list = [[] for _ in range(4)]
    raw_key_list = [[] for _ in range(6)]
    diff_hand_list = []
    labels = []
    colors = []
    times_list_final = []

    for row in bistroke_data:
        try:
            pos, bigram, *stroke_info = row
        except ValueError:
            continue
        if len(stroke_info) < 1:
            continue

        feats = get_bistroke_features(pos, bigram)
        for i in range(27):
            orig_list[i].append(feats[i])
        for i in range(4):
            derived_list[i].append(feats[27 + i])
        for i in range(6):
            raw_key_list[i].append(feats[31 + i])
        diff_hand_list.append(feats[37])
        labels.append(feats[38])
        colors.append(feats[39])

        # For each trial, take the last bigram measurement (whether there's one or more).
        stroke_times = [t[-1] for t in stroke_info if len(t) > 0]
        # Then average across all trials using the IQR average.
        times_list_final.append(get_iqr_avg(stroke_times))

    bg_features = tuple(
        np.array(lst)
        for lst in (orig_list + derived_list + raw_key_list + [diff_hand_list])
    )
    return bg_features, np.array(times_list_final), labels, colors


def extract_trigram_features(
    tristroke_data: List[Any],
    bg_model: Any,
    trigram_to_freq: dict,
    skipgram_to_freq: dict,
    predict_mode: str = "full",
) -> Tuple[Tuple[np.ndarray, ...], np.ndarray, List[str], List[str]]:
    """
    For each tristroke, extract features.
    In "full" mode, the target is defined as the IQR average of the full trigram times.
    In "last" mode, the target is defined as the IQR average of the last bigram times,
    computed for each trial as:
          last_bigram_time = full_trigram_time - first_bigram_time
    (It is assumed that each trial in the tristroke data provides either two measurements:
     [first_bigram_time, full_trigram_time] or one measurement, in which case that value is used.)
    """
    tg_freqs = []
    tg_bg1_prediction = []
    tg_bg2_prediction = []
    tg_sht = []
    tg_redirect = []
    tg_bad = []
    sg_freqs = []
    skip_features_list = []
    tg_times = []
    tg_labels = []
    tg_colors = []

    for row in tristroke_data:
        try:
            (pos1, pos2, pos3), trigram, *time_info = row
        except ValueError:
            continue
        tg_freqs.append(trigram_to_freq.get(trigram, 1))

        bg1 = trigram[:2]
        bg2 = trigram[1:]
        # placeholders for bigram predictions
        _ = get_bistroke_features((pos1, pos2), bg1)[:-2]
        _ = get_bistroke_features((pos2, pos3), bg2)[:-2]
        tg_bg1_prediction.append(0)
        tg_bg2_prediction.append(0)

        sg = trigram[::2]
        sg_feats = get_bistroke_features((pos1, pos3), sg)[:-2]
        skip_features_list.append(sg_feats)
        sg_freqs.append(skipgram_to_freq.get(sg, 1))

        ax, _ = pos1
        bx, _ = pos2
        cx, _ = pos3
        if 0 not in (ax, bx, cx):
            try:
                side1 = ax // abs(ax)
                side2 = bx // abs(bx)
                side3 = cx // abs(cx)
                sht = side1 == side2 == side3
            except ZeroDivisionError:
                sht = False
        else:
            sht = False
        tg_sht.append(float(sht))
        if sht:
            redir = (abs(ax) < abs(bx) and abs(cx) < abs(bx)) or (
                abs(ax) > abs(bx) and abs(cx) > abs(bx)
            )
            tg_redirect.append(float(redir))
            bad = redir and (not any(abs(x) in (1, 2) for x in (ax, bx, cx)))
            tg_bad.append(float(bad))
        else:
            tg_redirect.append(0.0)
            tg_bad.append(0.0)

        expected = tuple(bg_classifier.kb.get_pos(c) for c in trigram)
        col = "green" if ((pos1, pos2, pos3) == expected) else "red"
        tg_colors.append(col)
        tg_labels.append(trigram)

        # --- Corrected target computation for trigram features ---
        if predict_mode == "last":
            # For each trial in the tristroke data, compute the duration of the last bigram.
            last_bigram_durations = []
            for t in time_info:
                try:
                    if len(t) >= 2:
                        # Assume t[0] is the time for the first bigram and t[1] is the full trigram time.
                        last_bigram_durations.append(t[1] - t[0])
                    elif len(t) == 1:
                        # If only one measurement is available, use it.
                        last_bigram_durations.append(t[0])
                except Exception:
                    continue
            tg_times.append(get_iqr_avg(last_bigram_durations))
        else:
            # Full mode: use the full trigram times.
            full_times = []
            for t in time_info:
                try:
                    if len(t) >= 2:
                        full_times.append(t[1])
                    elif len(t) == 1:
                        full_times.append(t[0])
                except Exception:
                    continue
            tg_times.append(get_iqr_avg(full_times))
        # ----------------------------------------------------------

    skip_features_array = np.array(skip_features_list)
    num_skip_features = skip_features_array.shape[1]
    skip_features = tuple(skip_features_array[:, j] for j in range(num_skip_features))

    tg_level = (
        np.array(tg_freqs),
        np.array(tg_bg1_prediction),
        np.array(tg_bg2_prediction),
        np.array(tg_sht),
        np.array(tg_redirect),
        np.array(tg_bad),
        np.array(sg_freqs),
    )
    features_tuple = tg_level + skip_features
    return features_tuple, np.array(tg_times), tg_labels, tg_colors


def model_predict_single(model: Any, feature_tuple: Tuple[Any, ...]) -> float:
    wrapped = tuple(np.array([f]) for f in feature_tuple)
    return model.predict(wrapped)[0]


##########################################################################
# Model (Penalty) Functions & TypingModel Wrapper
##########################################################################
def bg_penalty(features: Tuple[np.ndarray, ...], *p: float) -> np.ndarray:
    # Note: This simple model only uses frequency.
    trimmed = features[:27]
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
    ) = trimmed
    freq_pen = p[0] * np.log(np.clip(bg_freqs + p[1], 1e-8, None)) + p[2]
    return freq_pen


def tg_penalty(tg_features: Tuple[np.ndarray, ...], *p: float) -> np.ndarray:
    # A simple linear model on a couple of placeholder features.
    trimmed = tg_features[:7]
    return p[0] * trimmed[1] + p[1] * trimmed[2] + p[2]


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

    def evaluate(
        self, features: Tuple[np.ndarray, ...], y: np.ndarray
    ) -> Tuple[float, float]:
        y_pred = self.predict(features)
        residuals = y - y_pred
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        mae = np.mean(np.abs(residuals))
        return r2, mae


##########################################################################
# Cross-Validation Function
##########################################################################
def cross_validate_model(
    model_args: dict, features: Tuple[np.ndarray, ...], y: np.ndarray, n_splits: int = 5
) -> Tuple[float, float]:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    r2_scores = []
    mae_scores = []
    n = len(y)
    for train_index, test_index in kf.split(np.arange(n)):
        train_features = tuple(f[train_index] for f in features)
        test_features = tuple(f[test_index] for f in features)
        y_train = y[train_index]
        y_test = y[test_index]
        model = TypingModel(**model_args)
        model.fit(train_features, y_train)
        r2, mae = model.evaluate(test_features, y_test)
        r2_scores.append(r2)
        mae_scores.append(mae)
    return np.mean(r2_scores), np.mean(mae_scores)


##########################################################################
# Plotting
##########################################################################
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


##########################################################################
# Main
##########################################################################
def main() -> None:
    parser = argparse.ArgumentParser(description="Typing Model Training")
    parser.add_argument(
        "--model",
        type=str,
        default="curve_fit",
        choices=["curve_fit", "xgboost"],
        help="Choose model type",
    )
    parser.add_argument(
        "--predict_mode",
        type=str,
        default="full",
        choices=["full", "last"],
        help="Predict the full trigram time ('full') or only the last bigram time ('last').",
    )
    parser.add_argument(
        "--trigrams_file",
        type=str,
        default="trigrams.txt",
        help="Path to trigram frequencies file",
    )
    parser.add_argument(
        "--bigrams_file",
        type=str,
        default="bigrams.txt",
        help="Path to bigram frequencies file",
    )
    parser.add_argument(
        "--skip_file",
        type=str,
        default="1-skip.txt",
        help="Path to skipgram frequencies file",
    )
    parser.add_argument(
        "--tristrokes_file",
        type=str,
        default="tristrokes.tsv",
        help="Path to tristroke data file",
    )
    parser.add_argument(
        "--bistrokes_file",
        type=str,
        default="bistrokes.tsv",
        help="Path to bistroke data file",
    )
    args = parser.parse_args()

    # Load frequency data.
    trigram_to_freq, bigram_to_freq, skipgram_to_freq, trigrams = (
        load_ngram_frequencies(args.trigrams_file, args.bigrams_file, args.skip_file)
    )
    bg_classifier.bigram_freq = bigram_to_freq

    tristroke_data = load_tristroke_data(args.tristrokes_file, WPM_THRESHOLD)
    bistroke_data = load_bistroke_data(args.bistrokes_file, WPM_THRESHOLD)

    # Bigram Model
    bg_features, bg_times, bg_labels, layout_col = extract_bigram_features(
        bistroke_data, predict_mode=args.predict_mode
    )
    if args.model == "curve_fit":
        bg_model_args = {
            "model_type": "curve_fit",
            "penalty_function": bg_penalty,
            "fit_kwargs": {"method": "lm", "maxfev": 750000, "p0": np.ones(3)},
        }
    else:
        xgb_params = {
            "max_depth": 3,
            "min_child_weight": 1,
            "gamma": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1,
            "n_estimators": 100,
            "learning_rate": 0.1,
            "objective": "reg:squarederror",
            "verbosity": 0,
        }
        bg_model_args = {"model_type": "xgboost", **xgb_params}
    bg_cv_r2, bg_cv_mae = cross_validate_model(
        bg_model_args, bg_features, bg_times, n_splits=5
    )
    print(f"Bigram Model CV - R^2: {bg_cv_r2:.4f}, MAE: {bg_cv_mae:.3f}")
    bg_model = TypingModel(**bg_model_args)
    bg_model.fit(bg_features, bg_times)
    with open("bigram_model.pkl", "wb") as f:
        pickle.dump(bg_model, f)
    print("Final bigram model saved to 'bigram_model.pkl'.")
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

    # Trigram Model
    tg_features, tg_times, tg_labels, tg_colors = extract_trigram_features(
        tristroke_data,
        bg_model,
        trigram_to_freq,
        skipgram_to_freq,
        predict_mode=args.predict_mode,
    )
    if args.model == "curve_fit":
        tg_model_args = {
            "model_type": "curve_fit",
            "penalty_function": tg_penalty,
            "fit_kwargs": {"method": "trf", "maxfev": 750000, "p0": np.ones(3)},
        }
    else:
        tg_model_args = {"model_type": "xgboost", **xgb_params}
    tg_cv_r2, tg_cv_mae = cross_validate_model(
        tg_model_args, tg_features, tg_times, n_splits=5
    )
    print(f"Trigram Model CV - R^2: {tg_cv_r2:.4f}, MAE: {tg_cv_mae:.3f}")
    tg_model = TypingModel(**tg_model_args)
    tg_model.fit(tg_features, tg_times)
    with open("trigram_model.pkl", "wb") as f:
        pickle.dump(tg_model, f)
    print("Final trigram model saved to 'trigram_model.pkl'.")
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
