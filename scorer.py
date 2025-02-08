from abc import ABC
import pickle
import numpy as np
from collections import defaultdict

from utils import load_ngram_frequencies

# Import shared functions and classes from the training module.
from wpm_conditioned_model import (
    create_bigram_feature_vector,
    create_trigram_feature_vector,
)
from classifier import Classifier

# === Process Trigram and Bigram Frequency Data ===
trigrams, tg_freqs = [], []
tg_percentages = {}

keyboard_chars = "qwertyuiopasdfghjkl'zxcvbnm,.-"
with open("trigrams.txt") as f:
    for trigram, freq in (l.split("\t") for l in f):
        if all(c in keyboard_chars for c in trigram):
            trigrams.append(trigram)
            tg_freqs.append(int(freq))
    total_count = sum(tg_freqs)
    elapsed = 0
    for i in range(len(tg_freqs)):
        percentage = int(100 * (elapsed / total_count))
        tg_percentages[percentage + 1] = i
        elapsed += tg_freqs[i]

# Trim to a chosen coverage percentage.
tg_coverage = 100  
tg_freqs = np.array(tg_freqs[: tg_percentages[tg_coverage]])
#print(tg_freqs)
trigrams = trigrams[: tg_percentages[tg_coverage]]
#print("Processed trigram data")

# (Global variable used by FreyaScorer.)
data_size = len(trigrams)

# Load bigram frequencies.
bigram_to_freq = defaultdict(int)
with open("bigrams.txt") as f:
    for k, v in (l.split("\t") for l in f):
        bigram_to_freq[k] = int(v)
#print("Processed bigram data")


# --- Define the scorer interface and classes ---

class IScorer(ABC):
    def get_fitness(self, keyboard) -> int:
        ...


class BigramXGBoostScorer(IScorer):
    def __init__(self, bigram_model_path="bigram_model.pkl", bigrams_file="bigrams.txt", target_wpm=100):
        self.target_wpm = target_wpm

        self.bg_classifier = Classifier()

        with open(bigram_model_path, 'rb') as f:
            self.bg_model = pickle.load(f)
        #print("Bigram XGBoost model loaded.")
        # Load frequency data (using our shared loader)
        _, self.bigram_to_freq, _, _ = load_ngram_frequencies("dummy_bigrams.txt", bigrams_file, "dummy_skips.txt")

    def get_fitness(self, keyboard):
        total_score = 0
        keyboard_chars = "qwertyuiopasdfghjkl'zxcvbnm,.- "
        bigrams_to_score = [c1 + c2 for c1 in keyboard_chars for c2 in keyboard_chars]
        for bigram in bigrams_to_score:
            if len(bigram) != 2:
                continue
            try:
                # Build the feature vector using the shared function.
                self.bg_classifier.kb = keyboard
                feature_vector = create_bigram_feature_vector(self.bg_classifier, bigram, self.target_wpm, bigram_to_freq)
                predicted_time = self.bg_model.predict(feature_vector)[0]
                total_score += predicted_time * self.bigram_to_freq.get(bigram, 1)
            except KeyError:
                total_score += 10000  # Penalize if a key is missing.
        return int(total_score)


class TrigramXGBoostScorer(IScorer):
    def __init__(self, trigram_model_path="trigram_model.pkl", trigram_freq_file="trigrams.txt", target_wpm=100):
        self.target_wpm = target_wpm

        self.bg_classifier = Classifier()

        #print("Loading Trigram XGBoost model...")
        with open(trigram_model_path, 'rb') as f:
            self.tg_model = pickle.load(f)
        self.trigram_to_freq, self.bigram_to_freq, self.skipgram_to_freq, self.trigrams = load_ngram_frequencies(trigram_freq_file, "bigrams_file", "1-skip.txt")

    def get_fitness(self, keyboard):
        total_score = 0
        for trigram in self.trigrams:
            try:
                # Build the trigram feature vector using the shared function.
                self.bg_classifier.kb = keyboard
                feature_vector = create_trigram_feature_vector(
                    self.bg_classifier, trigram, self.target_wpm, self.trigram_to_freq, self.skipgram_to_freq, self.bigram_to_freq 
                )
                predicted_time = self.tg_model.predict(feature_vector)[0]
                total_score += predicted_time * float(self.trigram_to_freq.get(trigram, 1))
            except KeyError:
                total_score += 100000 * float(self.trigram_to_freq.get(trigram, 1))
        return int(total_score)

class FreyaScorer(IScorer):
    def __init__(self) -> None:
        self.bg_times = {}

        # Penalties
        self.bg_p = [
            -13.006290200857604,
            -1110.6975118720209,
            301.79595431794627,
            0.2373529046003713,
            -0.3483169122234296,
            -0.44869126126093584,
            0.5183895781213851,
            -0.3675121927036903,
            0.21114436760532884,
            -0.11053904996244854,
            0.4969806603740193,
            0.2270610038405438,
            0.24377147190687995,
            0.24058025690698362,
            0.18606952101779842,
            0.13566910965517368,
            0.4269095628160329,
            1.0659786791611865,
            0.9072922405602595,
            0.8267421145473832,
            0.683055916507772,
            0.4223096257759204,
            0.9144166084516641,
            1.0013425051520004,
            0.5423592762848283,
            0.3124915691983608,
            0.7636142108419756,
            1.246192084201704,
            1.0748870559594512,
            0.9737199188685832,
            0.761229963934242,
            2.932362314618125,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
        ]

        self.tg_p = [
            1.0,
            1.0,
            -30.949166125789187,
            10.49968914570526,
            1.3812508525050837,
            3.557707209342576,
            -1.5392605080726125,
            -1.3101250448185828,
            -11.732288602506747,
            1.8697907379041518,
            19.60346057380557,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
        ]

        self.tg_bg1_prediction = np.zeros(data_size)
        self.tg_bg2_prediction = np.zeros(data_size)
        self.tg_redirect = np.zeros(data_size)
        self.tg_bad = np.zeros(data_size)
        self.sg_bottom = np.zeros(data_size)
        self.sg_home = np.zeros(data_size)
        self.sg_top = np.zeros(data_size)
        self.sg_pinky = np.zeros(data_size)
        self.sg_ring = np.zeros(data_size)
        self.sg_middle = np.zeros(data_size)
        self.sg_index = np.zeros(data_size)
        self.sg_lateral = np.zeros(data_size)
        self.sg_dx = np.zeros(data_size)
        self.sg_dy = np.zeros(data_size)
        self.sg_sfs = np.zeros(data_size)

    def get_fitness(self, keyboard) -> int:
        return int(np.sum(self.update_trigram_times(keyboard) * tg_freqs))

    def get_bg_features(self, keyboard, bg):
        ((ax, ay), (bx, by)) = [keyboard.get_pos(c) for c in bg]

        freq = bigram_to_freq[bg]

        # Row features
        bottom = by == 1
        home = by == 2
        top = by == 3

        # Column features
        pinky = abs(bx) == 5
        ring = abs(bx) == 4
        middle = abs(bx) == 3
        index = abs(bx) in (1, 2)

        dy = abs(ay - by)
        dx = abs((ax + row_offsets[ay - 1]) - (bx + row_offsets[by - 1]))

        lateral = abs(bx) == 1

        # Classifications
        shb = (ax // abs(ax)) == (bx // abs(bx))
        sfb = (ax == bx) or (shb and (abs(ax) in (1, 2) and abs(bx) in (1, 2)))

        return (
            freq,
            bottom,
            home,
            top,
            pinky,
            ring,
            middle,
            index,
            shb,
            sfb,
            lateral,
            dx,
            dy,
        )

    def get_bg_time(self, keyboard, bg):
        if bg not in self.bg_times:
            (
                bg_freq,
                bg_bottom,
                bg_home,
                bg_top,
                bg_pinky,
                bg_ring,
                bg_middle,
                bg_index,
                bg_shb,
                bg_sfb,
                bg_lateral,
                bg_dx,
                bg_dy,
            ) = self.get_bg_features(keyboard, bg)

            freq_pen = (
                self.bg_p[0] * np.log(np.clip(bg_freq + self.bg_p[1], a_min=1, a_max=None))
                + self.bg_p[2]
            )

            # Row penalties
            base_row_pen = self.bg_p[3] * (bg_home + bg_top) + self.bg_p[4] * (
                bg_top + bg_bottom
            )
            shb_row_pen = self.bg_p[5] * (bg_home + bg_top) + self.bg_p[6] * (
                bg_top + bg_bottom
            )
            alt_row_pen = self.bg_p[7] * (bg_home + bg_top) + self.bg_p[8] * (
                bg_top + bg_bottom
            )
            sfb_row_pen = self.bg_p[9] * (bg_home + bg_top) + self.bg_p[10] * (
                bg_top + bg_bottom
            )

            # Finger penalties
            sfb_finger_pen = (
                self.bg_p[11] * bg_pinky
                + self.bg_p[12] * bg_ring
                + self.bg_p[13] * bg_middle
                + self.bg_p[14] * (bg_index - bg_lateral)
                + self.bg_p[15] * bg_lateral
            )
            base_finger_pen = (
                self.bg_p[16] * bg_pinky
                + self.bg_p[17] * bg_ring
                + self.bg_p[18] * bg_middle
                + self.bg_p[19] * (bg_index - bg_lateral)
                + self.bg_p[20] * bg_lateral
            )
            shb_finger_pen = (
                self.bg_p[21] * bg_pinky
                + self.bg_p[22] * bg_ring
                + self.bg_p[23] * bg_middle
                + self.bg_p[24] * (bg_index - bg_lateral)
                + self.bg_p[25] * bg_lateral
            )
            alt_finger_pen = (
                self.bg_p[26] * bg_pinky
                + self.bg_p[27] * bg_ring
                + self.bg_p[28] * bg_middle
                + self.bg_p[29] * (bg_index - bg_lateral)
                + self.bg_p[30] * bg_lateral
            )

            # Aggregate penalties for classes
            shb_pen = shb_finger_pen * (shb_row_pen)
            alt_pen = alt_finger_pen * (alt_row_pen)
            sfb_pen = sfb_finger_pen * (sfb_row_pen)

            # class penalties
            base_weight = (
                1
                + (base_row_pen * base_finger_pen)
            )
            shb_weight = (
                (bg_shb * (1 - bg_sfb)) * (shb_pen)
            )

            dist_pen = (bg_dx**2 + bg_dy**2) ** 0.5 + self.bg_p[31]
            sfb_weight = bg_sfb * sfb_pen * dist_pen
            alt_weight = (1 - bg_shb) * alt_pen

            self.bg_times[bg] = freq_pen * (
                base_weight + alt_weight + shb_weight + sfb_weight
            )

        return self.bg_times[bg]

    def update_tg_features(self, keyboard):
        for i in keyboard.affected_indices:
            tg = trigrams[i]

            # extracting position
            ((ax, ay), (bx, by), (cx, cy)) = [keyboard.get_pos(c) for c in tg]

            # getting bigram times
            self.tg_bg1_prediction[i] = self.get_bg_time(keyboard, tg[:2])
            self.tg_bg2_prediction[i] = self.get_bg_time(keyboard, tg[1:])

            # Skipgram Penalties

            # row features
            self.sg_bottom[i] = cy == 1
            self.sg_home[i] = cy == 2
            self.sg_top[i] = cy == 3

            # Column features
            self.sg_pinky[i] = abs(cx) == 5
            self.sg_ring[i] = abs(cx) == 4
            self.sg_middle[i] = abs(cx) == 3
            self.sg_index[i] = abs(cx) in (1, 2)
            self.sg_lateral[i] = abs(cx) == 1

            # SFS
            if ax != 0 and cx != 0:
                self.sg_sfs[i] = (ax / abs(ax)) == (cx / abs(cx))
                self.sg_sfs[i] *= (abs(ax) == abs(cx)) or (abs(ax) in (1, 2) and self.sg_index[i])

            self.sg_dy[i] = abs(ay - by)
            self.sg_dx[i] = abs((ax + row_offsets[ay - 1]) - (bx + row_offsets[by - 1]))

        return (
            self.tg_bg1_prediction,
            self.tg_bg2_prediction,
            self.sg_bottom,
            self.sg_home,
            self.sg_top,
            self.sg_pinky,
            self.sg_ring,
            self.sg_middle,
            self.sg_index,
            self.sg_lateral,
            self.sg_sfs,
        )

    def update_trigram_times(self, keyboard):
        (
            self.tg_bg1_prediction,
            self.tg_bg2_prediction,
            self.sg_bottom,
            self.sg_home,
            self.sg_top,
            self.sg_pinky,
            self.sg_ring,
            self.sg_middle,
            self.sg_index,
            self.sg_lateral,
            self.sg_sfs,
        ) = self.update_tg_features(keyboard)

        # finger penalty
        sfs_row_pen = self.tg_p[3] * (self.sg_top + self.sg_home) + self.tg_p[4] * (self.sg_top + self.sg_bottom)

        sfs_finger_pen = (
            self.tg_p[5] * self.sg_pinky
            + self.tg_p[6] * self.sg_ring
            + self.tg_p[7] * self.sg_middle
            + self.tg_p[8] * (self.sg_index - self.sg_lateral)
            + self.tg_p[9] * self.sg_lateral
        )

        dist_pen = (self.sg_dx**2 + self.sg_dy**2) ** 0.5 + self.tg_p[10]

        # sfs penalty
        sfs_weight = self.sg_sfs * (sfs_row_pen + sfs_finger_pen)  * dist_pen

        return (
            # freq_pen *
            self.tg_bg1_prediction
            + self.tg_bg2_prediction
            + sfs_weight
            + self.tg_p[10]
        )
