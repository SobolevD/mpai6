import numpy as np
from scipy.stats import entropy


def get_entropy(q):
    hist, bins = np.histogram(q, bins=[el for el in range(int(q.min()), int(q.max() + 1), 1)])
    W = hist / q.size
    entrpy = entropy(W, base=2)
    return entrpy
