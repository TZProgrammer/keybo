import matplotlib.pyplot as plt
from collections import defaultdict
from conversion import merge, rgb_to_hex, okhsl_to_srgb, oklab_to_linear_srgb, linear_srgb_to_oklab, srgb_to_okhsl

high_col = srgb_to_okhsl(252, 231, 44)
low_col = srgb_to_okhsl(67, 13, 85)

chars = "etaoinsrhldcumfgpywbvkxjzq"
freqs = defaultdict(int)

with open('1-skip.txt') as f:
    for l in f:
        skipgram, freq = l.split('\t')

        if skipgram.isalpha():
            freqs[skipgram] = int(freq)

high_f = max(freqs.values())

def get_col(freq):
    t = (freq)/high_f
    return rgb_to_hex(*okhsl_to_srgb(*merge(low_col, high_col, t)))

for target in chars:
    labels = [target+c for c in chars]
    colors = [get_col(freqs[l]) for l in labels]

    plt.figure(figsize=(18, 9))
    plt.bar(labels, [freqs[l] for l in labels], color=colors)
    plt.ylabel('Frequency', fontsize=24)
    # plt.xlabel('Skipgram', fontsize=24)
    plt.title(f"Skipgram Frequencies for {target.upper()}", fontsize=32)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.savefig(f'Graph_Utils/out/{target.upper()}_Skipgram_Bars.png', format='png', dpi=300)

    plt.close()
