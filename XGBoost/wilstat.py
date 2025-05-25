import numpy as np
from scipy.stats import wilcoxon

# Test F1 scores for the 10 samples
f1_scores = [
    0.19686800894854586,
    0.16744186046511628,
    0.3034134007585335,
    0.15455594002306805,
    0.1692420897718911,
    0.19508196721311474,
    0.21311475409836064,
    0.19853431045969352,
    0.23575129533678757,
    0.2288135593220339
]

# Perform pairwise Wilcoxon signed-rank tests
results = []
for i in range(len(f1_scores)):
    for j in range(i + 1, len(f1_scores)):
        stat, p_value = wilcoxon([f1_scores[i]], [f1_scores[j]])
        results.append((i + 1, j + 1, stat, p_value))

# Print results
print(f"{'Sample 1':<10} {'Sample 2':<10} {'Statistic':<10} {'P-value':<10}")
for result in results:
    print(f"{result[0]:<10} {result[1]:<10} {result[2]:<10.4f} {result[3]:<10.4f}")
