from collections import Counter
import time
import numpy as np

from shapelet_extraction.brute_force import find_best_split_point
from util.util import calculate_stats, subsequence_dist_fast, information_gain_ub


def check_candidate(timeseries, labels, shapelet, stats):
    distances = []
    for index, time_serie, label in enumerate(zip(timeseries, labels)):
        d, idx = subsequence_dist_fast(time_serie, shapelet, stats[index])
        distances.append((d, label))

    gain, dist = find_best_split_point(sorted(distances, key=lambda x: x[0]))
    distances_left = []
    distances_right = []
    for index, time_serie in enumerate(timeseries):
        d = subsequence_dist_fast(time_serie, shapelet, stats[index])
        if d <= dist:
            distances_left.append(d)
        else:
            distances_right.append(d)

    return gain, dist, np.mean(distances_right) - np.mean(distances_left)


def find_shapelets_fast(timeseries, labels, max_len=100, min_len=1, verbose=True):
    m = min([len(x) for x in timeseries])  # The min. length of a timeseries in the dataset
    max_ig, max_gap, best_dist, best_shapelet = 0, 0, 0, None
    stats = {}
    block_start = time.time()
    total_start = time.time()
    gain_ub = information_gain_ub(labels)
    for idx, timeserie1 in enumerate(timeseries):

        if verbose: print(idx, '/', len(timeseries))

        for idx2, timeserie2 in enumerate(timeseries):
            stats[idx2] = calculate_stats(timeserie1, timeserie2)

        for l in range(min(m, min_len), min(m, max_len)):
            # TODO: implement pre-pruning

            for i in range(len(timeserie1)-l):
                candidate = timeserie1[i:i+l]
                gain, dist, gap = check_candidate(timeseries, labels, candidate, stats)

                if gain > max_ig or (gain == max_ig and gap > max_gap):
                    max_ig = gain
                    max_gap = gap
                    best_dist = dist
                    best_shapelet = candidate

        if not idx % 1:
            block_end = time.time()
            print('10 iterations took', block_end-block_start)
            block_start = block_end

        if max_ig >= gain_ub: break  # We won't find any better solution than this

    total_time = time.time() - total_start
    print('Brute force algorithm took', total_time)
    print(total_time/idx, 'per iteration')
    return best_shapelet, best_dist
