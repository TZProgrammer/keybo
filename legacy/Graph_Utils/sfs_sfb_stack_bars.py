from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from collections import defaultdict
from conversion import merge, rgb_to_hex, okhsl_to_srgb, oklab_to_linear_srgb, linear_srgb_to_oklab, srgb_to_okhsl
from math import log10

high_h, high_s, high_l = srgb_to_okhsl(252, 231, 44)
high_col = (high_h, high_s, high_l*0.97)
low_col = srgb_to_okhsl(67, 13, 85)

lower_chars = {c1 : c2 for c1, c2 in zip("ETAOINSRHLDCUMFGPYWBVLXJZQ:\"<>?", "etaoinsrhldcumfgpywbvkxjzq;',./")}

def lower(s):
    return "".join([lower_chars[c] if c in lower_chars else c for c in s])

chars = "etaoinsrhldcumfgpywbvkxjzq;',./"

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

high_f = max(*skipgram_freqs.values(), *bigram_freqs.values())+10000
high_scale = max(bigram_freqs[c1+c2]+skipgram_freqs[c1+c2] for c1 in chars for c2 in chars)*1

def get_col(freq, offset=0):
    t = (freq)/high_f

    h, s, l = merge(low_col, high_col, t)

    return rgb_to_hex(*okhsl_to_srgb(h, s, min(l+offset, 1)))

bigram_cmap = LinearSegmentedColormap.from_list("custom", [get_col(v, 0.09) for v in range(0, high_f, 20000)])
# skipgram_cmap = LinearSegmentedColormap.from_list("custom",  [get_col(v, -0.05) for v in range(0, high_f, 20000)])

# Chart
for target in chars:
    labels = [target+c for c in chars]

    indices = sorted(range(len(labels)), key=lambda i: -(bigram_freqs[labels[i]])) # skipgram_freqs[labels[i]]+
    labels = [labels[i] for i in indices]

    bigram_colors = [get_col(bigram_freqs[l], 0.09) for l in labels]
    skipgram_colors = [get_col(skipgram_freqs[l], -0.09) for l in labels]

    fig, ax = plt.subplots(figsize=(22, 9))
    # ax.set_ylim(0, high_scale) # 10**int(log10(high_scale)+0.5)

    # Plot each bar type
    bigram_vals = [bigram_freqs[l] for l in labels]
    skipgram_vals = [skipgram_freqs[l] for l in labels]
    bigram_bar = ax.bar(labels, bigram_vals, label='bigram',  color=bigram_colors, edgecolor='white')
    skipgram_bar = ax.bar(labels, skipgram_vals, label='skipgram', bottom=bigram_vals, color=skipgram_colors, edgecolor='white')

    plt.ylabel('Cummulitative Occurence Count', fontsize=18)
    # plt.xlabel('Skipgram', fontsize=24)
    plt.title(f"{target.upper()} - Bigram (Bottom) vs. Skipgram (Top) Occurences", fontsize=24)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    # Gradient legend explanation
    bigram_sm = plt.cm.ScalarMappable(cmap=bigram_cmap, norm=plt.Normalize(vmin=0, vmax=high_f))
    bigram_cbar = plt.colorbar(bigram_sm)
    bigram_cbar.ax.set_ylabel('Frequency Intensity', fontsize=18, rotation=270, labelpad=20)

    #skipgram_sm = plt.cm.ScalarMappable(cmap=skipgram_cmap, norm=plt.Normalize(vmin=0, vmax=high_f))
    #skipgram_cbar = plt.colorbar(skipgram_sm)
    # skipgram_cbar.ax.set_ylabel('Skipgram Frequency Intensity', fontsize=16)

    
    plt.savefig(f'Graph_Utils/out/{target.upper()}_Stats.png', format='png', dpi=300)
    # plt.show()
    print(f"Exported {target.upper()}_Stats.png")
    plt.close()
