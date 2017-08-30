import itertools
import numpy as np
from collections import Counter
import time

from pyshapelets.util import util


def generate_candidates(timeseries, labels, max_len, min_len):
    candidates, l = [], max_len
    while l >= min_len:
        for time_serie, label in zip(timeseries, labels):
            for k in range(len(time_serie)-l+1):
                candidates.append((time_serie[k:k+l], label))
        l -= 1
    return candidates


def entropy_pre_prune(label_counter, distances):
    # TODO: Can we again calculate this mathematically?
    # TODO: rewrite this ugly thing
    max_ig = 0
    for left, right in util.partitions(list(label_counter.items()), 2):
        left_keys = [x[0] for x in left]
        left_values = [x[1] for x in left]
        right_keys = [x[0] for x in right]
        right_values = [x[1] for x in right]
        sorted_distances = sorted(distances, key=lambda x: x[0])
        left_labels = [[x] * y for (x, y) in zip(left_keys, left_values)][0]
        left_list = list(zip([0] * sum(left_values), left_labels))
        right_labels = [[x] * y for (x, y) in zip(right_keys, right_values)][0]
        right_list = list(zip([sorted_distances[-1][0]] * sum(right_values), right_labels))
        concat = left_list + sorted_distances + right_list
        ig = find_best_split_point(concat)[0]
        if ig > max_ig: max_ig = ig
        left_labels = [[x] * y for (x, y) in zip(left_keys, left_values)][0]
        left_list = list(zip([sorted_distances[-1][0]] * sum(left_values), left_labels))
        right_labels = [[x] * y for (x, y) in zip(right_keys, right_values)][0]
        right_list = list(zip([0] * sum(right_values), right_labels))
        concat = right_list + sorted_distances + left_list
        ig = find_best_split_point(concat)[0]
        if ig > max_ig: max_ig = ig
    return max_ig


def check_candidate(timeseries, labels, shapelet, min_prune_length=20, best_ig=None):
    distances = []
    # cntr = Counter(labels)
    for time_serie, label in zip(timeseries, labels):
        d, idx = util.subsequence_dist(time_serie, shapelet)
        distances.append((d, label))
        # max_ig = None
        # if best_ig is not None:
        #     cntr[label] -= 1
        #     if len(distances) > min_prune_length:
        #         max_ig = entropy_pre_prune(cntr, distances)
        # if max_ig is not None and max_ig <= best_ig:
        #     return 0, 0

    return find_best_split_point(sorted(distances, key=lambda x: x[0]))


def find_best_split_point(distances):
    labels_all = [x[1] for x in distances]
    prior_entropy = util.calculate_dict_entropy(labels_all)
    best_distance, max_ig = 0, 0
    for i in range(len(distances)-1):
        while distances[i] == distances[i+1]: i+=1  # skip equal distances
        labels_left = [x[1] for x in distances[:i+1]]
        labels_right = [x[1] for x in distances[i+1:]]
        ig = util.information_gain(labels_left, labels_right, prior_entropy)
        if ig > max_ig:
            best_distance, max_ig = distances[i][0], ig
    return max_ig, best_distance


def find_shapelets_bf(timeseries, labels, max_len=100, min_len=1, verbose=True):
    candidates = generate_candidates(timeseries, labels, max_len, min_len)
    bsf_gain, bsf_shapelet = 0, None
    gain_ub = util.information_gain_ub(labels)
    print(Counter(labels), 'UPPER BOUND =', gain_ub)
    if verbose: candidates_length = len(candidates)
    total_start = time.time()
    block_start = time.time()
    for idx, candidate in enumerate(candidates):
        gain, dist = check_candidate(timeseries, labels, candidate[0], best_ig=None)

        if verbose: print(idx, '/', candidates_length, ":", gain, dist)

        if not idx % 10:
            block_end = time.time()
            print('10 iterations took', block_end-block_start)
            block_start = block_end

        if gain > bsf_gain:
            bsf_gain, bsf_shapelet = gain, candidate[0]
            if verbose: print('Found new best shapelet with gain & dist:', bsf_gain, dist)

        if bsf_gain >= gain_ub: break  # We won't find any better solution than this

    total_time = time.time() - total_start
    print('Brute force algorithm took', total_time)
    print(total_time/idx, 'per iteration')
    return bsf_shapelet
