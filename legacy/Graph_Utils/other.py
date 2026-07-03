from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from collections import defaultdict
from conversion import merge, rgb_to_hex, okhsl_to_srgb, oklab_to_linear_srgb, linear_srgb_to_oklab, srgb_to_okhsl
from math import log10
from matplotlib.colors import Normalize

high_h, high_s, high_l = srgb_to_okhsl(252, 231, 44)
high_col = (high_h, high_s, high_l*0.97)
low_col = srgb_to_okhsl(67, 13, 85)

lower_chars = {c1 : c2 for c1, c2 in zip("ETAOINSRHLDCUMFGPYWBVLXJZQ:\"<>?", "etaoinsrhldcumfgpywbvkxjzq;',./")}

def lower(s):
    return "".join([lower_chars[c] if c in lower_chars else c for c in s])

chars = "etaoinsrhldcumfgpywbvkxjzq;'-,./"

# get bigram freqs
bigram_freqs = defaultdict(int)

with open('bigrams.txt') as f:
    for l in f:
        bigram, freq = l.split('\t')
        bigram = lower(bigram)

        if all(c in chars for c in bigram):
            bigram_freqs[bigram] += int(freq)
            bigram_freqs[bigram[1]+bigram[0]] += int(freq)

# get skipgram freqs
skipgram_freqs = defaultdict(int)

with open('1-skip.txt') as f:
    for l in f:
        skipgram, freq = l.split('\t')
        skipgram = lower(skipgram)

        if all(c in chars for c in skipgram):
            skipgram_freqs[skipgram] += int(freq)
            skipgram_freqs[skipgram[1]+skipgram[0]] += int(freq)

# Define max frequency normalization
high_f = max(*skipgram_freqs.values(), *bigram_freqs.values()) + 10000
bigram_norm = Normalize(vmin=0, vmax=high_f)
bigram_cmap = plt.cm.get_cmap('viridis')

# Chart creation with regular cmap
for target in chars:
    labels = [target + c for c in chars]

    indices = sorted(range(len(labels)), key=lambda i: -(bigram_freqs[labels[i]]))  # sort by frequency
    labels = [labels[i] for i in indices]

    bigram_vals = [bigram_freqs[l] for l in labels]
    skipgram_vals = [skipgram_freqs[l] for l in labels]

    # Map colors to values using the colormap
    bigram_colors = [bigram_cmap(bigram_norm(val)) for val in bigram_vals]
    skipgram_colors = [bigram_cmap(bigram_norm(val)) for val in skipgram_vals]

    fig, ax = plt.subplots(figsize=(19, 9))

    # Plot each bar type
    bigram_bar = ax.bar(labels, bigram_vals, label='bigram', color=bigram_colors, edgecolor='white')
    skipgram_bar = ax.bar(labels, skipgram_vals, label='skipgram', bottom=bigram_vals, color=skipgram_colors, edgecolor='white')

    plt.ylabel('Cumulative Occurrence Count', fontsize=18)
    plt.title(f"{target.upper()} - Bigram (Bottom) vs. Skipgram (Top) Occurrences", fontsize=24)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    # Add a colorbar for the colormap
    sm = plt.cm.ScalarMappable(cmap=bigram_cmap, norm=bigram_norm)
    cbar = plt.colorbar(sm, ax=ax)
    cbar.ax.set_ylabel('Frequency Intensity', fontsize=18, rotation=270, labelpad=20)

    # Save the figure
    plt.show()
    #plt.savefig(f'Graph_Utils/out/{target.upper()}_Skipgram_Bars.png', format='png', dpi=300)
    #print(f"Exported {target.upper()}_Skipgram_Bars.png")
    #plt.close()
