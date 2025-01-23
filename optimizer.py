from abc import ABC

import numpy as np
from math import ceil, exp, log
from random import random

### INIT ###
initial_temp = None
tg_coverage = 100  # the percentage of tg's to use
keyboard_chars = "qwertyuiopasdfghjkl'zxcvbnm,.-"

# getting trigrams and their frequencies
trigrams, tg_freqs = [], []
row_offsets = [0.5, 0, -0.25]
tg_percentages = {}

with open("trigrams.txt") as f:
    valid_c = keyboard_chars

    for trigram, freq in (l.split("\t") for l in f):
        if all([c in valid_c for c in trigram]):
            trigrams.append(trigram)
            tg_freqs.append(int(freq))

    percentages = [0] * 100
    total_count = sum(tg_freqs)
    elapsed = 0

    for i in range(len(tg_freqs)):
        percentage = int(100 * (elapsed / total_count))
        tg_percentages[percentage + 1] = i
        elapsed += tg_freqs[i]

# trimming our tg data to the amount of data we'll actually be processing
tg_freqs = np.array(tg_freqs[: tg_percentages[tg_coverage]])
trigrams = trigrams[: tg_percentages[tg_coverage]]
print("Processed trigram data")

# trigram penalties
data_size = len(trigrams)

class IOptimizer(ABC):
    def optimize(self, keyboard, scorer):
        ...

class Optimizer(IOptimizer):
    def __init__(self, keyboard, scorer, a=0.995):
        self.a = a
        self.t0 = 0
        self.cooling_schedule = "default"
        self.bg_scores = {bg : 0 for bg in keyboard.get_ngrams(2)}
        self.new_bg_scores = {}
        self.fitness = 0
        self.prev_fitness = 0
        self.tg_coverage = tg_coverage  # the percentage of tg's to use

        self.get_fitness(keyboard, scorer)
        self.accept()

        self.temp = self.get_initial_temperature(keyboard, scorer, 0.8, 0.01)

    def optimize(self, keyboard, scorer):
        stopping_point = self.get_stopping_point(keyboard=keyboard)

        stays = 0
        while stays < stopping_point:
            # markov chain
            for _ in range(keyboard.key_count):
                self.swap(keyboard=keyboard)
                self.get_fitness(keyboard, scorer)

                delta = self.fitness - self.prev_fitness

                if delta < 0:
                    self.accept()
                    stays = 0
                # Metropolis criterion
                elif random() < exp(-delta / self.temp):
                    self.accept()
                    stays -= 1
                else:
                    self.reject(keyboard=keyboard, scorer=scorer)
                    stays += 1

            self.cool()
            print(self.fitness, f"@{self.tg_coverage}% a={self.a}")
            print(keyboard)

        return keyboard


    def swap(self, keyboard, k1=None, k2=None):
        if (k1 is not None) and (k2 is not None):
            keyboard.swap(k1, k2)
        else:
            keyboard.random_swap()

        keyboard.affected_indices = [
            i
            for i, tg in enumerate(trigrams)
            if any([c in keyboard.swap_pair for c in tg])
        ]

    def accept(self):
        self.bg_scores.update(self.new_bg_scores)

    def reject(self, keyboard, scorer):
        keyboard.undo_swap()
        scorer.get_fitness(keyboard)

        self.fitness = self.prev_fitness
        self.new_bg_scores = {}

    def get_initial_temperature(self, keyboard, scorer, x0, epsilon=0.01):
        global initial_temp
        print("getting initial temperature")

        # An initial guess for t1
        tn = self.fitness
        acceptance_probability = 0

        # Repeat guess
        while abs(acceptance_probability - x0) > epsilon:
            energies = []

            # test all possible swaps
            for i, k1 in enumerate(keyboard.lowercase[:-1]):
                for k2 in keyboard.lowercase[i + 1 :]:
                    self.swap(keyboard, k1, k2)
                    self.get_fitness(keyboard, scorer)

                    delta = self.fitness - self.prev_fitness

                    # Keep track of transition energies for each positive transition
                    if delta > 0:
                        energies.append(self.fitness)

                    self.reject(keyboard=keyboard, scorer=scorer)

            # Calculate acceptance probability
            acceptance_probability = sum(
                [exp(-(e_after / tn)) for e_after in energies]
            ) / (len(energies) * exp(-(self.prev_fitness / tn)))

            tn = tn * (log(acceptance_probability) / log(x0))

        print(f"initial temperature found, t0 = {tn}")
        initial_temp = tn

        return tn

    def cool(self):
        self.temp *= self.a

    # Calculate a stopping time for the annealing process based on the number of swaps (coupon collector's problem).
    def get_stopping_point(self, keyboard):
        swaps = keyboard.key_count * (keyboard.key_count - 1) / 2
        euler_mascheroni = 0.5772156649

        return ceil(swaps * (log(swaps) + euler_mascheroni) + 0.5)

    def get_fitness(self, keyboard, scorer):
        self.prev_fitness = self.fitness
        self.fitness = scorer.get_fitness(keyboard)
