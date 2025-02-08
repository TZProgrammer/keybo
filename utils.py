from collections import defaultdict
from typing import Any, List, Tuple
import os

##########################################################################
# Data Loading and Utility Functions
##########################################################################
def load_ngram_frequencies(
    trigrams_file: str, bigrams_file: str, skip_file: str
) -> Tuple[dict, dict, dict, List[str]]:
    trigram_to_freq = defaultdict(int)
    trigrams = []
    allowed_chars = "qwertyuiopasdfghjkl';zxcvbnm,./QWERTYUIOPASDFGHJKL:ZXCVBNM<>? "
    if os.path.exists(trigrams_file):
        with open(trigrams_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                k, v = parts[:2]
                k = k.strip()  # do NOT convert to lowercase
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
    #print("Trigram percentages:", percentages)
    #print("Trigrams loaded:", trigrams[:10], "..." if len(trigrams) > 10 else "")

    bigram_to_freq = defaultdict(int)
    if os.path.exists(bigrams_file):
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
    #print("Bigrams loaded:", len(bigram_to_freq))

    skipgram_to_freq = defaultdict(int)
    if os.path.exists(skip_file):
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
    #print("Skipgram loaded:", len(skipgram_to_freq))

    return trigram_to_freq, bigram_to_freq, skipgram_to_freq, trigrams

