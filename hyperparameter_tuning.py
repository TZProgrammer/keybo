"""
Hyperparameter Tuning via scikit-learn's RandomizedSearchCV for Typing Model

Usage:
    python hyperparam_tuning_sklearn.py --model_type bigram --n_iter 30 --cv 5 \
        --trigrams_file trigrams.txt --bigrams_file bigrams.txt --skip_file 1-skip.txt \
        --tristrokes_file tristrokes.tsv --bistrokes_file bistrokes.tsv \
        --output_file best_hyperparams.json
"""

import argparse
import json
import os

import numpy as np

# For cross-validation searching.
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, mean_absolute_error
from scipy.stats import uniform, randint

# If xgboost is installed with the scikit-learn API, you can import:
try:
    from xgboost import XGBRegressor
except ImportError:
    raise ImportError("Please install xgboost (pip install xgboost) to run this script.")

# Import your existing data/feature extraction from your main module.
# Adjust this import if your main code is in a different file or structure.
from classifier import Classifier
from utils import load_ngram_frequencies

from wpm_conditioned_model import (
    load_tristroke_data,
    load_bistroke_data,
    extract_bigram_features,
    extract_trigram_features,
    WPM_THRESHOLD,
)

def main():
    parser = argparse.ArgumentParser(description="Randomized Search with scikit-learn for Typing Model")
    parser.add_argument("--model_type", type=str, choices=["bigram", "trigram"], default="bigram",
                        help="Which model to tune (bigram or trigram)")
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
    parser.add_argument("--output_file", type=str, default="best_hyperparams.json",
                        help="JSON file to save the best hyperparameters")
    parser.add_argument("--n_iter", type=int, default=100,
                        help="Number of parameter settings that are sampled in RandomizedSearchCV")
    parser.add_argument("--cv", type=int, default=5,
                        help="Number of cross-validation folds")
    args = parser.parse_args()

    # Load frequencies.
    trigram_to_freq, bigram_to_freq, skipgram_to_freq, _ = load_ngram_frequencies(
        args.trigrams_file, args.bigrams_file, args.skip_file
    )

    bg_classifier = Classifier()

    # Load data and features depending on model_type.
    if args.model_type == "bigram":
        bistroke_data = load_bistroke_data(args.bistrokes_file, WPM_THRESHOLD)
        features, times, _, _ = extract_bigram_features(bistroke_data, bigram_to_freq, bg_classifier)
    else:  # "trigram"
        tristroke_data = load_tristroke_data(args.tristrokes_file, WPM_THRESHOLD)
        features, times, _, _ = extract_trigram_features(tristroke_data, bg_classifier, trigram_to_freq, skipgram_to_freq, bigram_to_freq)

    # Convert features from tuple-of-arrays to a single 2D array for scikit-learn
    # Each element of 'features' is an array for one feature dimension: shape (#samples,)
    # So we stack them horizontally with column_stack:
    X = np.column_stack(features)
    y = times

    # Set up a param_distributions dict for RandomizedSearchCV.
    # You can tailor these ranges or distributions as you wish:
    param_distributions = {
        "max_depth": randint(2, 7),  # integer range [2..9]
        "learning_rate": uniform(0.001, 0.1),  # float in [0.001..0.101)
        "min_child_weight": randint(1, 6),     # integer range [1..5]
        "subsample": uniform(0.5, 0.5),        # float in [0.5..1.0)
        "colsample_bytree": uniform(0.5, 0.5), # float in [0.5..1.0)
        "gamma": uniform(0, 1),               # float in [0..1)
        "reg_alpha": uniform(0, 2),           # float in [0..2)
        "reg_lambda": uniform(0, 2),          # float in [0..2)
        # If you want to tune n_estimators in a random range, you could do:
        "n_estimators": randint(100, 1000),
    }

    # Create an XGBRegressor with default config (or specify any baseline).
    # Some parameters could be fixed if you prefer, and *only* tune certain ones above.
    xgb_model = XGBRegressor(
        objective="reg:squarederror",
        verbosity=0,     # silent mode
        tree_method="auto"
    )

    mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)

    # Create RandomizedSearchCV instance.
    # n_iter = number of random configurations to try.
    # cv = number of folds.
    # scoring = negative MAE so that higher = better in scikit-learn’s internal sense.
    # n_jobs = -1 to use all CPU cores (optional).
    random_search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_distributions,
        n_iter=args.n_iter,
        scoring=mae_scorer,
        cv=args.cv,
        refit=True,
        verbose=1,
        #random_state=42,
        n_jobs=-1
    )

    # Fit the search. This will take some time depending on n_iter and data size.
    print(f"Starting RandomizedSearchCV with n_iter={args.n_iter}, cv={args.cv}...")
    random_search.fit(X, y)

    # Best results
    best_params = random_search.best_params_
    best_score = -random_search.best_score_  # remember we used neg_mean_absolute_error
    print(f"\nDone! Best MAE: {best_score:.4f}")
    print(f"Best Hyperparameters: {best_params}")

    # Save best hyperparameters to JSON
    with open(args.output_file, "w") as f:
        json.dump(best_params, f, indent=4)

    print(f"Best hyperparameters saved to '{args.output_file}'.")

if __name__ == "__main__":
    main()
