from abc import ABC
import copy
import numpy as np
from math import ceil, exp, log
from random import random, sample
from itertools import permutations, combinations
from tqdm import tqdm

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

    def simulated_annealing(self, keyboard, scorer):
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

    def optimize(self, keyboard, scorer):
        self.simulated_annealing(keyboard, scorer)

        self.local_improvement_2opt(keyboard, scorer)

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

    def local_improvement_2opt(self, keyboard, scorer):
        """
        Perform a 2-opt local improvement pass. Iterate over all pairs of keys and accept a swap
        if it improves the fitness. Continue until no improvement is possible.
        """
        improvement = True
        iteration = 0
        while improvement:
            improvement = False
            iteration += 1
            # Iterate over all unique pairs.
            for i in range(keyboard.key_count):
                for j in range(i+1, keyboard.key_count):
                    current_fitness = scorer.get_fitness(keyboard)
                    key1 = keyboard.lowercase[i]
                    key2 = keyboard.lowercase[j]
                    keyboard.swap(key1, key2)
                    new_fitness = scorer.get_fitness(keyboard)
                    if new_fitness < current_fitness:
                        improvement = True
                        print(f"2-opt improvement iteration {iteration}: Fitness improved from {current_fitness} to {new_fitness}")
                        if new_fitness < self.best_fitness:
                            self.best_fitness = new_fitness
                            self.best_keyboard = copy.deepcopy(keyboard)
                            print("2-opt: New best keyboard found:")
                            print(keyboard)
                            print(f"Score: {new_fitness}")
                        break  # Exit inner loop to restart scan.
                    else:
                        keyboard.undo_swap()
                if improvement:
                    break
        self.keyboard = self.best_keyboard
        return self.keyboard

    def local_improvement_3opt(self, keyboard, scorer):
        """
        Perform a 3-opt local improvement pass. Iterate over all unique triples of keys and try reordering
        them (using a sequence of swap() calls) if it improves the fitness. A tqdm progress bar is used
        to indicate progress for each 3-opt iteration.
        """

        improvement = True
        iteration = 0
        n = keyboard.key_count
        total_triples = (n * (n - 1) * (n - 2)) // 6  # total number of unique triples

        while improvement:
            improvement = False
            iteration += 1
            print(f"\nStarting 3-opt iteration {iteration}")

            # Use tqdm to wrap the combinations iterator.
            for i, j, k in tqdm(combinations(range(n), 3), total=total_triples, desc=f"3-opt iteration {iteration}"):
                current_fitness = scorer.get_fitness(keyboard)
                # Get the current letters at these positions.
                orig = [keyboard.lowercase[i], keyboard.lowercase[j], keyboard.lowercase[k]]
                
                found_improvement = False
                # Try each permutation (skip the identity permutation).
                for new_order in permutations(orig):
                    if list(new_order) == orig:
                        continue

                    # Determine the minimal swap sequence to achieve the new order.
                    a, b, c = orig
                    swap_sequence = None
                    if new_order == (b, a, c):
                        swap_sequence = [(i, j)]
                    elif new_order == (a, c, b):
                        swap_sequence = [(j, k)]
                    elif new_order == (c, b, a):
                        swap_sequence = [(i, k)]
                    elif new_order == (b, c, a):
                        swap_sequence = [(i, j), (j, k)]
                    elif new_order == (c, a, b):
                        swap_sequence = [(i, k), (j, k)]
                    else:
                        continue  # Should not occur

                    performed_swaps = 0
                    # Apply the swaps in the sequence.
                    for pos1, pos2 in swap_sequence:
                        key1 = keyboard.lowercase[pos1]
                        key2 = keyboard.lowercase[pos2]
                        keyboard.swap(key1, key2)
                        performed_swaps += 1

                    new_fitness = scorer.get_fitness(keyboard)
                    if new_fitness < current_fitness:
                        improvement = True
                        found_improvement = True
                        print(f"3-opt improvement iteration {iteration}: Fitness improved from {current_fitness} to {new_fitness}")
                        if new_fitness < self.best_fitness:
                            self.best_fitness = new_fitness
                            self.best_keyboard = copy.deepcopy(keyboard)
                            print("3-opt: New best keyboard found:")
                            print(keyboard)
                            print(f"Score: {new_fitness}")
                        break  # Break out of the permutation loop.
                    else:
                        # Undo the swaps if no improvement.
                        for _ in range(performed_swaps):
                            keyboard.undo_swap()

                if found_improvement:
                    # Restart scanning from the beginning if an improvement was found.
                    break

            if not improvement:
                print(f"3-opt iteration {iteration}: No improvement found, local optimum reached.")

        self.keyboard = self.best_keyboard
        return self.keyboard
