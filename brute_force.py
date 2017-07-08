import itertools
import numpy as np
from collections import Counter
import time

from util import subsequence_dist, calculate_entropy, calculate_dict_entropy, information_gain


# TODO: re-write most of the code (optimize it --> numpy/scipy) and implement some pre-pruning stuff


def generate_candidates(timeseries, labels, max_len, min_len):
    candidates, l = [], max_len
    while l >= min_len:
        for time_serie, label in zip(timeseries, labels):
            for k in range(len(time_serie)-l+1):
                candidates.append((time_serie[k:k+l], label))
        l -= 1
    return candidates


def check_candidate(timeseries, labels, shapelet, min_prune_length=5):
    distances = []
    for time_serie, label in zip(timeseries, labels):
        d, idx = subsequence_dist(time_serie, shapelet)
        distances.append((d, label))
        # TODO: Here we can do entropy pre-pruning, but it requires distances to be sorted!
        # we set the distances to be either 0.0 or max+1.
        if len(distances) > min_prune_length:
            pass
    return find_best_split_point(sorted(distances, key=lambda x: x[0]))


def find_best_split_point(distances):
    labels_all = [x[1] for x in distances]
    prior_entropy = calculate_dict_entropy(labels_all)
    best_distance, max_ig = 0, 0
    for i in range(len(distances)):
        labels_left = [x[1] for x in distances[:i]]
        labels_right = [x[1] for x in distances[i:]]
        ig = information_gain(labels_left, labels_right, prior_entropy)
        if ig > max_ig:
            best_distance, max_ig = distances[i][0], ig
    return max_ig, best_distance


def find_shapelets_bf(timeseries, labels, max_len=100, min_len=1, verbose=True):
    candidates = generate_candidates(timeseries, labels, max_len, min_len)
    bsf_gain, bsf_shapelet = 0, None
    unique_labels = np.unique(labels)
    # TODO: Not a good ub, best ub is probably when we can split of the majority class from the others
    gain_ub = calculate_entropy([1./len(unique_labels)] * len(unique_labels))
    if verbose: candidates_length = len(candidates)
    for idx, candidate in enumerate(candidates):
        gain, dist = check_candidate(timeseries, labels, candidate[0])

        if verbose: print(idx, '/', candidates_length, ":", gain, dist)

        if gain > bsf_gain:
            bsf_gain, bsf_shapelet = gain, candidate[0]
            if verbose: print('Found new best shapelet with gain & dist:', bsf_gain, dist)

        if bsf_gain >= gain_ub: break  # We won't find any better solution than this

    return bsf_shapelet