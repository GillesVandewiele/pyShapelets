import numpy as np
from scipy.stats import entropy
from collections import Counter


def euclidian_distance(a, b, min_dist=np.inf):
    if min_dist == np.inf:
        return np.linalg.norm(a-b, ord=2)

    # If a min_dist is given, we apply early stopping
    # TODO: bench if this early stopping every performs better than a numpy function
    dist = 0
    for x, y in zip(a, b):
        dist += (float(x)-float(y))**2
        if dist >= min_dist: return np.inf
    return dist


def calculate_entropy(probabilities):
    # This takes logarithm with base 10 instead of 2 (should not be a problem)
    return entropy(probabilities)
    #return sum([-prob * np.log(prob)/np.log(2) if prob != 0 else 0 for prob in probabilities])


def calculate_dict_entropy(data):
    cntr = Counter(data)
    total = sum(cntr.values())
    for key in cntr:
        cntr[key] /= total
    return calculate_entropy(list(cntr.values()))


def information_gain(labels_left, labels_right, prior_entropy):
    total_length = len(labels_left) + len(labels_right)
    ig_left = float(len(labels_left)) / total_length * calculate_dict_entropy(labels_left)
    ig_right = float(len(labels_right)) / total_length * calculate_dict_entropy(labels_right)
    return prior_entropy - (ig_left + ig_right)


def subsequence_dist(time_serie, sub_serie):
    if len(sub_serie) < len(time_serie):
        min_dist, min_idx = float("inf"), -1
        for i in range(len(time_serie)-len(sub_serie)+1):
            dist = euclidian_distance(sub_serie, time_serie[i:i + len(sub_serie)], min_dist)
            if dist is not None and dist < min_dist: min_dist, min_idx = dist, i
        return min_dist, min_idx
    else:
        return None, None
