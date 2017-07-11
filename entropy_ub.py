import numpy as np
import itertools

from util import calculate_entropy, information_gain
from collections import Counter


def find_best_split(n, k):
    x = list(np.random.choice(range(k), size=n, p=[0.8,0.05,0.1,0.05]))
    print(Counter(x))
    prior_entropy = calculate_entropy(x)

    best_ig = 0
    best_left = None
    best_right = None

    for m in range(1, len(x)):
        for left in set(itertools.combinations(x, m)):
            copy = x.copy()
            for elt in left:
                copy.remove(elt)
            right = copy

            ig = information_gain(left, right, prior_entropy)

            if ig > best_ig:
                best_ig = ig
                best_left = left
                best_right = right

    print(best_ig)
    print(best_left)
    print(best_right)


find_best_split(25, 4)