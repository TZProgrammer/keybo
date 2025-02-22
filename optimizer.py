from abc import ABC
import copy
import numpy as np
from math import ceil, exp, log
from random import random, sample

### INIT ###
initial_temp = None
tg_coverage = 100  # the percentage of trigrams to use
keyboard_chars = "qwertyuiopasdfghjkl';zxcvbnm,./ "

# Process trigrams and frequencies
trigrams, tg_freqs = [], []
row_offsets = [0.5, 0, -0.25]
tg_percentages = {}

with open("trigrams.txt") as f:
    valid_c = keyboard_chars
    for trigram, freq in (line.split("\t") for line in f):
        if all(c in valid_c for c in trigram):
            trigrams.append(trigram)
            tg_freqs.append(int(freq))
    total_count = sum(tg_freqs)
    elapsed = 0
    for i, freq in enumerate(tg_freqs):
        percentage = int(100 * (elapsed / total_count))
        tg_percentages[percentage + 1] = i
        elapsed += freq

# Trim trigram data to the chosen coverage percentage.
tg_freqs = np.array(tg_freqs[: tg_percentages[tg_coverage]])
trigrams = trigrams[: tg_percentages[tg_coverage]]
print("Processed trigram data")

# Global variable used for trigram penalties.
data_size = len(trigrams)

class IOptimizer(ABC):
    def optimize(self, keyboard, scorer):
        ...

class Optimizer(IOptimizer):
    def __init__(self, keyboard, scorer, a=0.999):
        self.a = a
        self.t0 = 0
        self.cooling_schedule = "default"
        # Initialize background scores for bigrams.
        self.bg_scores = {bg: 0 for bg in keyboard.get_ngrams(2)}
        self.new_bg_scores = {}
        self.fitness = 0
        self.prev_fitness = 0
        self.tg_coverage = tg_coverage  # the percentage of trigrams to use
        self.keyboard = keyboard  # will hold the best keyboard at end
        
        # Initialize best configuration with a deep copy.
        self.best_fitness = None
        self.best_keyboard = copy.deepcopy(keyboard)
        
        # Get initial fitness and set as best.
        self.get_fitness(keyboard, scorer)
        self.accept()
        self.best_fitness = self.fitness
        self.best_keyboard = copy.deepcopy(keyboard)
        
        # Estimate the initial temperature using a standard approach.
        self.temp = self.get_initial_temperature(keyboard, scorer, x0=0.8)

    def optimize(self, keyboard, scorer):
        stopping_point = self.get_stopping_point(keyboard)
        stays = 0
        outer_iter = 0  # Counter for outer iterations

        while stays < stopping_point:
            outer_iter += 1
            # Perform a series of swaps equal to the number of keys.
            for _ in range(keyboard.key_count):
                self.swap(keyboard)
                self.get_fitness(keyboard, scorer)
                delta = self.fitness - self.prev_fitness

                if delta < 0:
                    self.accept()
                    stays = 0
                elif random() < exp(-delta / self.temp):  # Metropolis criterion.
                    self.accept()
                    stays = max(0, stays - 1)
                else:
                    self.reject(keyboard, scorer)
                    stays += 1

                # Update best keyboard if an improvement is found.
                if self.best_fitness is None or self.fitness < self.best_fitness:
                    self.best_fitness = self.fitness
                    self.best_keyboard = copy.deepcopy(keyboard)

            self.cool()
            progress_pct = (stays / stopping_point) * 100
            print(f"Outer Iteration {outer_iter}: Fitness = {self.fitness}, Temp = {self.temp:.2f}, "
                  f"Rejection streak = {stays}/{stopping_point} ({progress_pct:.1f}% complete)")
            print(keyboard)

        # Store the best configuration found.
        self.keyboard = self.best_keyboard
        return self.best_keyboard

    def swap(self, keyboard, k1=None, k2=None):
        if (k1 is not None) and (k2 is not None):
            keyboard.swap(k1, k2)
        else:
            keyboard.random_swap()

    def accept(self):
        self.bg_scores.update(self.new_bg_scores)

    def reject(self, keyboard, scorer):
        keyboard.undo_swap()
        scorer.get_fitness(keyboard)
        self.fitness = self.prev_fitness
        self.new_bg_scores = {}

    def get_initial_temperature(self, keyboard, scorer, x0):
        """
        Estimate the initial temperature T0 such that the average uphill move Δ has acceptance probability x0,
        i.e. T0 = -avg(Δ) / log(x0).
        """
        print("Estimating initial temperature...")

        current_fitness = scorer.get_fitness(keyboard)
        lowercase = keyboard.lowercase
        n = len(lowercase)
        total_pairs = n * (n - 1) // 2
        sample_size = min(1000, total_pairs)
        all_pairs = [(lowercase[i], lowercase[j]) for i in range(n) for j in range(i+1, n)]
        swap_samples = sample(all_pairs, sample_size)

        uphill_deltas = []
        for k1, k2 in swap_samples:
            keyboard.swap(k1, k2)
            new_fitness = scorer.get_fitness(keyboard)
            delta = new_fitness - current_fitness
            if delta > 0:
                uphill_deltas.append(delta)
            keyboard.undo_swap()

        if uphill_deltas:
            avg_delta = sum(uphill_deltas) / len(uphill_deltas)
            T0 = -avg_delta / log(x0)
        else:
            T0 = 1.0

        print(f"Initial temperature determined: t0 = {T0}")
        return T0

    def cool(self):
        self.temp *= self.a

    def get_stopping_point(self, keyboard):
        swaps = keyboard.key_count * (keyboard.key_count - 1) / 2
        euler_mascheroni = 0.5772156649
        return ceil(swaps * (log(swaps) + euler_mascheroni) + 0.5)

    def get_fitness(self, keyboard, scorer):
        self.prev_fitness = self.fitness
        self.fitness = scorer.get_fitness(keyboard)
