import argparse
import ast
import os
import pickle
import json
from collections import defaultdict
from typing import Any, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from utils import load_ngram_frequencies

try:
    import xgboost as xgb
except ImportError:
    xgb = None

from classifier import Classifier

# Global constant for WPM threshold.
WPM_THRESHOLD = 80

# For key features, map key characters to numeric IDs.
ALL_KEYS = list("qwertyuiopasdfghjkl;zxcvbnm,./ QWERTYUIOPASDFGHJKL:ZXCVBNM<>?")
char_to_id = {c: i for i, c in enumerate(ALL_KEYS)}


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


def load_tristroke_data(filepath: str, wpm_threshold: int, tg_min_samples: int = 35) -> List[Any]:
    data = []
    allowed_chars = "qwertyuiopasdfghjkl';zxcvbnm,./QWERTYUIOPASDFGHJKL:ZXCVBNM<>? "
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Cannot find file: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue
            pos, trigram, num_occurances, *occurances = parts
            if len(trigram) != 3 or not all(ch in allowed_chars for ch in trigram):
                continue
            wpm_type_time_list = []
            for x in occurances:
                try:
                    parsed = str_to_tuple(x)
                except Exception:
                    continue
                if parsed[0] >= wpm_threshold:
                    wpm_type_time_list.append(parsed)
            if len(wpm_type_time_list) < tg_min_samples:
                continue
            try:
                positions = ast.literal_eval(pos)
            except (SyntaxError, ValueError):
                continue
            data.append((positions, trigram, *wpm_type_time_list))
    return data


def load_bistroke_data(filepath: str, wpm_threshold: int, bg_min_samples: int = 50) -> List[Any]:
    data = []
    allowed_chars = "qwertyuiopasdfghjkl';zxcvbnm,./QWERTYUIOPASDFGHJKL:ZXCVBNM<>? "
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Cannot find file: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue
            pos, bigram, num_occurances, *occurances = parts
            bigram = bigram.strip()
            if len(bigram) != 2 or not all(c in allowed_chars for c in bigram):
                continue
            wps_type_time_list = []
            for x in occurances:
                try:
                    parsed = str_to_tuple(x)
                except Exception:
                    continue
                if parsed[0] >= wpm_threshold:
                    wps_type_time_list.append(parsed)
            if len(wps_type_time_list) < bg_min_samples:
                continue
            try:
                positions = ast.literal_eval(pos)
            except (SyntaxError, ValueError):
                continue
            data.append((positions, bigram, *wps_type_time_list))
    return data


def get_bistroke_features(bg_classifier: Classifier, bigram: str, freq: int) -> Tuple[Any, ...]:
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
    (ax, ay) = bg_classifier.kb.get_pos(k1)
    (bx, by) = bg_classifier.kb.get_pos(k2)

    offset_ax = row_offsets.get(ay, 0.0)
    offset_bx = row_offsets.get(by, 0.0)
    dy = abs(ay - by)
    dx = abs((ax + offset_ax) - (bx + offset_bx))
    shb = False
    if not (space1 or space2) and ax != 0 and bx != 0:
        shb = (ax // abs(ax)) == (bx // abs(bx))
    scb = (ax == bx)
    sfb = scb or (shb and ((index1 and middle2) or (index2 and middle1)))
    lateral = (abs(bx) == 1)
    adjacent = shb and ((abs(ax - bx) == 1) or (index1 and middle2) or (index2 and middle1))
    lsb = shb and (((index1 and middle2) or (index2 and middle1)) and dx > 1.5)
    bg_scissor = ((dy == 2 and dx <= 1) or
                  ((pinky1 and top1 and ring2 and bottom2) or (ring1 and bottom1 and pinky2 and top2)) or
                  ((ring1 and top1 and middle2 and bottom2) or (middle1 and bottom1 and ring2 and top2)) or
                  ((index1 and top1 and middle2 and bottom2) or (middle1 and bottom1 and index1 and top1)))
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
    raw_key_info = (float(k1_id), float(k2_id), float(ax), float(ay), float(bx), float(by))
    new_feature = (float(diff_hand),)
    return orig_features + derived_features + raw_key_info + new_feature + (bigram, col)


def extract_bigram_features(bistroke_data: List[Any], bigram_to_freq: dict, bg_classifier):
    all_features = []
    all_times = []
    all_labels = []
    all_colors = []
    for row in bistroke_data:
        try:
            pos, bigram, *occurrences = row
        except ValueError:
            continue
        if not occurrences:
            continue
        feats = get_bistroke_features(bg_classifier, bigram, bigram_to_freq.get(bigram, 1))
        base_features = list(feats[:38])
        wpm_groups = defaultdict(list)
        for occ in occurrences:
            try:
                if len(occ) > 1:
                    wpm_groups[occ[0]].append(occ[1])
            except Exception:
                continue
        for wpm, time_list in wpm_groups.items():
            avg_time = get_iqr_avg(time_list)
            sample_features = base_features + [float(wpm)]
            all_features.append(sample_features)
            all_times.append(avg_time)
            all_labels.append(feats[38])
            all_colors.append(feats[39])
    if all_features:
        features_array = np.array(all_features)
        features_tuple = tuple(features_array[:, i] for i in range(features_array.shape[1]))
    else:
        features_tuple = tuple()
    return features_tuple, np.array(all_times), all_labels, all_colors


def extract_trigram_features(tristroke_data: List[Any], bg_classifier: Classifier, trigram_to_freq: dict, skipgram_to_freq: dict, bigram_to_freq: dict):
    all_features = []
    all_times = []
    all_labels = []
    all_colors = []
    for row in tristroke_data:
        try:
            (pos1, pos2, pos3), trigram, *occurrences = row
        except ValueError:
            continue
        tg_freq = float(trigram_to_freq.get(trigram, 1))
        ax, _ = pos1
        bx, _ = pos2
        cx, _ = pos3
        try:
            sht = (ax != 0 and bx != 0 and cx != 0 and (ax // abs(ax) == bx // abs(bx) == cx // abs(cx)))
        except ZeroDivisionError:
            sht = False
        sht_val = 1.0 if sht else 0.0
        if sht:
            redirect_val = 1.0 if ((abs(ax) < abs(bx) and abs(cx) < abs(bx)) or (abs(ax) > abs(bx) and abs(cx) > abs(bx))) else 0.0
            bad_val = 1.0 if (redirect_val and (not any(abs(x) in (1, 2) for x in (ax, bx, cx)))) else 0.0
        else:
            redirect_val, bad_val = 0.0, 0.0
        sg = trigram[::2]
        sg_feats = get_bistroke_features(bg_classifier, sg, bigram_to_freq.get(sg, 1))
        sg_freq = float(skipgram_to_freq.get(sg, 1))
        bg1 = trigram[:2]
        bg2 = trigram[1:]
        bg1_feats = get_bistroke_features(bg_classifier, bg1, bigram_to_freq.get(bg1, 1))
        bg2_feats = get_bistroke_features(bg_classifier, bg2, bigram_to_freq.get(bg2, 1))
        bg1_numeric = list(bg1_feats[:38])
        bg2_numeric = list(bg2_feats[:38])
        base_features = [tg_freq, sht_val, redirect_val, bad_val, sg_freq] + list(sg_feats[:38]) + bg1_numeric + bg2_numeric
        wpm_groups = defaultdict(list)
        for occ in occurrences:
            try:
                if len(occ) > 1:
                    wpm_groups[occ[0]].append(occ[1])
            except Exception:
                continue
        for wpm, time_list in wpm_groups.items():
            avg_time = get_iqr_avg(time_list)
            sample_features = base_features + [float(wpm)]
            all_features.append(sample_features)
            all_times.append(avg_time)
            expected = tuple(bg_classifier.kb.get_pos(c) for c in trigram)
            col = "green" if ((pos1, pos2, pos3) == expected) else "red"
            all_labels.append(trigram)
            all_colors.append(col)
    if all_features:
        features_array = np.array(all_features)
        features_tuple = tuple(features_array[:, i] for i in range(features_array.shape[1]))
    else:
        features_tuple = tuple()
    return features_tuple, np.array(all_times), all_labels, all_colors


# --- Shared feature vector creation functions used both for training and inference ---

ROW_OFFSETS = {1: 0.5, 2: 0.0, 3: -0.25}

def create_bigram_feature_vector(bg_classifier, bigram, target_wpm, bigram_to_freq):
    """
    Given a keyboard object and a bigram, returns a feature vector (as a NumPy array)
    that consists of the numeric features (the first 38 entries from get_bistroke_features)
    plus the target WPM. This same function is used in training and during inference.
    """
    feats = get_bistroke_features(bg_classifier, bigram, bigram_to_freq.get(bigram, 1))
    numeric_features = list(feats[:38])
    return np.array(numeric_features + [float(target_wpm)]).reshape(1, -1)


def create_trigram_feature_vector(bg_classifier, trigram, target_wpm, trigram_to_freq, skipgram_to_freq, bigram_to_freq):
    """
    Given a keyboard, trigram, and target WPM (plus frequency data), build the trigram feature vector.
    This function encapsulates the (duplicated) logic from both training and inference.
    """
    pos_list = [bg_classifier.kb.get_pos(c) for c in trigram]
    tg_freq = float(trigram_to_freq.get(trigram, 1))
    ax, ay = pos_list[0]
    bx, by = pos_list[1]
    cx, cy = pos_list[2]
    try:
        sht = (ax != 0 and bx != 0 and cx != 0 and (ax // abs(ax) == bx // abs(bx) == cx // abs(cx)))
    except ZeroDivisionError:
        sht = False
    sht_val = 1.0 if sht else 0.0
    if sht:
        redirect_val = 1.0 if ((abs(ax) < abs(bx) and abs(cx) < abs(bx)) or (abs(ax) > abs(bx) and abs(cx) > abs(bx))) else 0.0
        bad_val = 1.0 if (redirect_val and (not any(abs(x) in (1, 2) for x in (ax, bx, cx)))) else 0.0
    else:
        redirect_val, bad_val = 0.0, 0.0
    sg = trigram[::2]
    sg_feats = get_bistroke_features(bg_classifier, sg, bigram_to_freq.get(sg, 1))
    sg_numeric = list(sg_feats[:38])
    bg1 = trigram[:2]
    bg2 = trigram[1:]
    bg1_feats = get_bistroke_features(bg_classifier, bg1, bigram_to_freq.get(bg1, 1))
    bg2_feats = get_bistroke_features(bg_classifier, bg2, bigram_to_freq.get(bg2, 1))
    bg1_numeric = list(bg1_feats[:38])
    bg2_numeric = list(bg2_feats[:38])
    sg_freq = float(skipgram_to_freq.get(sg, 1))
    base_features = [tg_freq, sht_val, redirect_val, bad_val, sg_freq] + sg_numeric + bg1_numeric + bg2_numeric
    return np.array(base_features + [float(target_wpm)]).reshape(1, -1)


# --- XGBoost Model Wrapper and Training Utilities ---

class TypingModel:
    """
    A simple wrapper around an XGBoost regressor.
    """
    def __init__(self, **kwargs) -> None:
        if xgb is None:
            raise ImportError("xgboost is not installed.")
        self.model = xgb.XGBRegressor(**kwargs)

    def fit(self, features: Tuple[Any, ...], y: np.ndarray) -> None:
        X = np.column_stack([np.asarray(f) for f in features])
        self.model.fit(X, y)

    def predict(self, features) -> np.ndarray:
        # If features is already a 2D numpy array, use it directly.
        if isinstance(features, np.ndarray) and features.ndim == 2:
            X = features
        else:
            # Otherwise assume it's a tuple (or list) of arrays.
            X = np.column_stack([np.asarray(f) for f in features])
        return self.model.predict(X)

    def evaluate(self, features: Tuple[np.ndarray, ...], y: np.ndarray) -> Tuple[float, float]:
        y_pred = self.predict(features)
        residuals = y - y_pred
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        mae = np.mean(np.abs(residuals))
        return r2, mae


def cross_validate_model(model_args: dict, features: Tuple[np.ndarray, ...], y: np.ndarray, n_splits: int = 5) -> Tuple[float, float]:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    r2_scores = []
    mae_scores = []
    n = len(y)
    for train_index, test_index in kf.split(np.arange(n)):
        print("Begin CV Split")
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


def plot_results(x: np.ndarray, y: np.ndarray, y_pred: np.ndarray, labels: List[str], colors: List[str],
                 xlabel: str, ylabel: str, title: str, annotate_threshold: int = 25) -> None:
    plt.figure(figsize=(9, 6))
    plt.scatter(x, y, s=40, c=colors, alpha=0.8, label="Data")
    plt.plot(x, y_pred, color="black", linewidth=2, label="Fit")
    if len(x) <= annotate_threshold:
        for xi, yi, lab in zip(x, y, labels):
            plt.annotate(f"'{lab}'", (xi, yi), xytext=(0, 5), textcoords="offset points", ha="center")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Typing Model Training with XGBoost (full mode only)")
    parser.add_argument("--trigrams_file", type=str, default="trigrams.txt", help="Path to trigram frequencies file")
    parser.add_argument("--bigrams_file", type=str, default="bigrams.txt", help="Path to bigram frequencies file")
    parser.add_argument("--skip_file", type=str, default="1-skip.txt", help="Path to skipgram frequencies file")
    parser.add_argument("--tristrokes_file", type=str, default="tristrokes.tsv", help="Path to tristroke data file")
    parser.add_argument("--bistrokes_file", type=str, default="bistrokes.tsv", help="Path to bistroke data file")
    parser.add_argument("--hyperparams_file", type=str, default=None, help="Path to JSON file with hyperparameters")
    args = parser.parse_args()

    if args.hyperparams_file is not None and os.path.exists(args.hyperparams_file):
        with open(args.hyperparams_file, "r") as f:
            xgb_params = json.load(f)
        print(f"Loaded hyperparameters from {args.hyperparams_file}: {xgb_params}")
    else:
        xgb_params = {
            "max_depth": 5,
            "min_child_weight": 1,
            "gamma": 0.1,
            "subsample": 0.6,
            "colsample_bytree": 0.6,
            "reg_alpha": 0.1,
            "reg_lambda": 1,
            "n_estimators": 20,
            "learning_rate": 0.9,
            "objective": "reg:squarederror",
            "verbosity": 0,
        }
        print("Using default hyperparameters.")

    # Load frequency data.
    trigram_to_freq, bigram_to_freq, skipgram_to_freq, trigrams = load_ngram_frequencies(
        args.trigrams_file, args.bigrams_file, args.skip_file
    )
    bg_classifier = Classifier()

    tristroke_data = load_tristroke_data(args.tristrokes_file, WPM_THRESHOLD)
    bistroke_data = load_bistroke_data(args.bistrokes_file, WPM_THRESHOLD)

    # Bigram Model Training.
    bg_features, bg_times, bg_labels, layout_col = extract_bigram_features(bistroke_data, bigram_to_freq, bg_classifier)
    bg_cv_r2, bg_cv_mae = cross_validate_model(xgb_params, bg_features, bg_times, n_splits=5)
    print(f"Bigram Model CV - R^2: {bg_cv_r2:.4f}, MAE: {bg_cv_mae:.3f}")
    bg_model = TypingModel(**xgb_params)
    bg_model.fit(bg_features, bg_times)
    train_r2, train_mae = bg_model.evaluate(bg_features, bg_times)
    print(f"Final Training Metrics for Bigram Model - R^2: {train_r2:.4f}, MAE: {train_mae:.3f}")
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
    plot_results(x_sorted, y_sorted, pred_sorted, labels_sorted, colors_sorted,
                 "Bigram Frequency", "Avg Typing Time (ms)", "Bigram Model Fit")

    # Trigram Model Training.
    tg_features, tg_times, tg_labels, tg_colors = extract_trigram_features(tristroke_data, bg_classifier, trigram_to_freq, skipgram_to_freq, bigram_to_freq)
    tg_cv_r2, tg_cv_mae = cross_validate_model(xgb_params, tg_features, tg_times, n_splits=5)
    print(f"Trigram Model CV - R^2: {tg_cv_r2:.4f}, MAE: {tg_cv_mae:.3f}")
    tg_model = TypingModel(**xgb_params)
    tg_model.fit(tg_features, tg_times)
    train_r2_tg, train_mae_tg = tg_model.evaluate(tg_features, tg_times)
    print(f"Final Training Metrics for Trigram Model - R^2: {train_r2_tg:.4f}, MAE: {train_mae_tg:.3f}")
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
    plot_results(x_sorted_tg, y_sorted_tg, pred_sorted_tg, labels_sorted_tg, colors_sorted_tg,
                 "Trigram Frequency", "Avg Typing Time (ms)", "Trigram Model Fit")


if __name__ == "__main__":
    main()
