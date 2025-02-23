import argparse
import json
import logging
import os
import sys
from typing import Tuple

import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, mean_absolute_error
from scipy.stats import uniform, randint

try:
    from xgboost import XGBRegressor
except ImportError:
    raise ImportError("Please install xgboost (pip install xgboost) to run this script.")

from classifier import Classifier
from utils import load_ngram_frequencies
from wpm_conditioned_model import (
    load_tristroke_data,
    load_bistroke_data,
    extract_bigram_features,
    extract_trigram_features,
    WPM_THRESHOLD,
)


def setup_logging() -> None:
    """Set up logging for the script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Randomized Search with scikit-learn for Typing Model (using GPU)"
    )
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
                        help="Number of parameter settings sampled in RandomizedSearchCV")
    parser.add_argument("--cv", type=int, default=5,
                        help="Number of cross-validation folds")
    parser.add_argument("--random_seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU ID to use for training")
    return parser.parse_args()


def load_data_and_features(args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load ngram frequencies, extract features, and return a feature matrix X and target y.
    Exits if any file is missing or no features can be extracted.
    """
    # Check frequency files exist.
    for file_path in (args.trigrams_file, args.bigrams_file, args.skip_file):
        if not os.path.exists(file_path):
            logging.error("Frequency file '%s' does not exist.", file_path)
            sys.exit(1)

    trigram_to_freq, bigram_to_freq, skipgram_to_freq, _ = load_ngram_frequencies(
        args.trigrams_file, args.bigrams_file, args.skip_file
    )
    logging.info("Loaded ngram frequencies.")

    bg_classifier = Classifier()

    # Depending on model type, load data and extract features.
    if args.model_type == "bigram":
        if not os.path.exists(args.bistrokes_file):
            logging.error("Bistrokes file '%s' does not exist.", args.bistrokes_file)
            sys.exit(1)
        bistroke_data = load_bistroke_data(args.bistrokes_file, WPM_THRESHOLD)
        features, times, _, _ = extract_bigram_features(bistroke_data, bigram_to_freq, bg_classifier)
    else:  # trigram
        if not os.path.exists(args.tristrokes_file):
            logging.error("Tristrokes file '%s' does not exist.", args.tristrokes_file)
            sys.exit(1)
        tristroke_data = load_tristroke_data(args.tristrokes_file, WPM_THRESHOLD)
        features, times, _, _ = extract_trigram_features(tristroke_data, bg_classifier, trigram_to_freq, skipgram_to_freq, bigram_to_freq)

    # Ensure that features were extracted.
    if not features or features[0].size == 0:
        logging.error("No features were extracted. Check your data and thresholds.")
        sys.exit(1)

    # Stack the tuple-of-arrays into a 2D feature matrix.
    X = np.column_stack(features)
    y = times
    logging.info("Extracted features shape: %s", X.shape)
    return X, y


def main() -> None:
    setup_logging()
    args = parse_args()
    logging.info("Starting hyperparameter tuning for the %s model using GPU (ID: %d).", args.model_type, args.gpu_id)

    X, y = load_data_and_features(args)

    # Define parameter distributions.
    param_distributions = {
        "max_depth": randint(5, 7),
        "learning_rate": uniform(0.003, 0.03),
        "min_child_weight": randint(1, 6),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4),
        "gamma": uniform(0, 0.5),
        "reg_alpha": uniform(0, 2),
        "reg_lambda": uniform(0, 2),
        "n_estimators": randint(600, 1000),
    }

    # Create the base XGBRegressor with GPU acceleration.
    xgb_model = XGBRegressor(
        objective="reg:squarederror",
        verbosity=0,
        tree_method="gpu_hist",
        predictor="gpu_predictor",
        gpu_id=args.gpu_id,
        random_state=args.random_seed
    )

    mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)

    random_search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_distributions,
        n_iter=args.n_iter,
        scoring=mae_scorer,
        cv=args.cv,
        refit=True,
        verbose=1,
        random_state=args.random_seed,
        n_jobs=-1
    )

    logging.info("Starting RandomizedSearchCV with n_iter=%d and cv=%d...", args.n_iter, args.cv)
    random_search.fit(X, y)

    best_params = random_search.best_params_
    best_score = -random_search.best_score_  # convert from negative MAE
    logging.info("RandomizedSearchCV completed.")
    logging.info("Best MAE: %.4f", best_score)
    logging.info("Best Hyperparameters: %s", best_params)

    try:
        with open(args.output_file, "w") as f:
            json.dump(best_params, f, indent=4)
        logging.info("Best hyperparameters saved to '%s'.", args.output_file)
    except IOError as e:
        logging.error("Failed to save hyperparameters to '%s'. Error: %s", args.output_file, str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
