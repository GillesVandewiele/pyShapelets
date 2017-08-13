from collections import Counter
import time
import numpy as np

from pyshapelets.shapelet_extraction.brute_force import find_best_split_point
from pyshapelets.util import util


def check_candidate(timeseries, labels, shapelet, stats):
    distances = []
    for index, time_serie, label in enumerate(zip(timeseries, labels)):
        d, idx = util.subsequence_dist_fast(time_serie, shapelet, stats[index])
        distances.append((d, label))

    gain, dist = find_best_split_point(sorted(distances, key=lambda x: x[0]))
    distances_left = []
    distances_right = []
    for index, time_serie in enumerate(timeseries):
        d = util.subsequence_dist_fast(time_serie, shapelet, stats[index])
        if d <= dist:
            distances_left.append(d)
        else:
            distances_right.append(d)

    return gain, dist, np.mean(distances_right) - np.mean(distances_left)


def fast_shapelet_discovery(timeseries, labels, m=None):
    if m is None:
        m = np.min([len(x) for x in timeseries]) # Maximum length of a timeserie
    max_gain, max_gap = 0, 0
    best_shapelet, best_distance, best_L = None, None, None
    for ts, label in zip(timeseries, labels):
        x = (ts, label)
        stats = {}
        for i, (ts2, label2) in enumerate(zip(timeseries, labels)):
            stats[i] = util.calculate_stats(ts, ts2)

        for l in range(1, m+1): # Possible shapelet lengths
            H = []  # Cache/history
            for i in range(len(ts) - l): # Possible start positions
                broken = False
                for (L, S) in H:
                    R = util.sdist(ts[i:i+l], S)
                    if util.upperIG(L, R, timeseries, labels) < max_gain:
                        broken = True
                        break # Continue with next i

                if not broken:
                    L = []
                    for k, (ts2, label2) in enumerate(zip(timeseries, labels)):
                        L.append((util.sdist_new(ts[i:i+l], ts2, i, stats[k]), label2))
                    print(L)
                    L = sorted(L, key=lambda x: x[0])
                    best_ig, tau = find_best_split_point(L)
                    if best_ig < max_gain:
                        best_shapelet = ts[i:i+l]
                        max_gain = best_ig
                        best_L = L
                        best_distance = tau
                        print('---->', max_gain, best_distance)
                    H.append((L, ts[i:i+l]))

    return best_shapelet, best_distance, best_L, max_gain

