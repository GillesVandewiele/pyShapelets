from itertools import combinations

import numpy as np
import time
from scipy.stats import entropy
from collections import Counter


def euclidean_distance(a, b, min_dist=np.inf):
    if min_dist == np.inf:
        return np.linalg.norm(a-b, ord=2)

    # If a min_dist is given, we apply early stopping
    # TODO: bench if this early stopping every performs better than a numpy function
    dist = 0
    for x, y in zip(a, b):
        dist += (float(x)-float(y))**2
        if dist >= min_dist: return np.inf
    return dist


def calculate_stats(a, b):
    """
    Calculate the five different arrays for two time series. Used for the calculation of normalized euclidean distance
    :param a: timeseries
    :param b: timeseries
    :return: five arrays used for the calculation of normalized euclidean distance
    """
    s_x = np.append([0], np.cumsum(a))
    s_x_sqr = np.append([0], np.cumsum(np.power(a, 2)))
    s_y = np.append([0], np.cumsum(b))
    s_y_sqr = np.append([0], np.cumsum(np.power(b, 2)))
    m_uv = np.zeros((len(a)+1, len(b)+1))
    start = time.time()
    for u in range(len(a)):
        for v in range(len(b)):
            t = abs(u-v)
            if u > v:
                m_uv[u+1, v+1] = m_uv[u, v] + a[v+t]*b[v]
            else:
                m_uv[u+1, v+1] = m_uv[u, v] + a[u]*b[u+t]
    print('Holy moly, calculating this 2D stat matrix takes', time.time() - start, 'seconds')
    return s_x, s_x_sqr, s_y, s_y_sqr, m_uv


def calculate_stats_old(a, b):
    s_x = np.append([0], np.cumsum(a))
    s_x_sqr = np.append([0], np.cumsum(np.power(a, 2)))
    s_y = np.append([0], np.cumsum(b))
    s_y_sqr = np.append([0], np.cumsum(np.power(b, 2)))
    m_uv = np.zeros((len(a) + 1, len(b) + 1))
    start = time.time()
    for u in range(len(a)):
        for v in range(len(b)):
            t = abs(u - v)
            if u > v:
                m_uv[u + 1, v + 1] = np.sum([a[i+t]*b[i] for i in range(v+1)])
            else:
                m_uv[u + 1, v + 1] = np.sum([a[i]*b[i+t] for i in range(u+1)])
    print('Holy moly, calculating this 2D stat matrix takes', time.time() - start, 'seconds')
    return s_x, s_x_sqr, s_y, s_y_sqr, m_uv


def normalized_euclidean_distance(a, b, stats):
    """
    Calculate the normalized (z-normalization and length normalization) euclidean distance between a sub- and timeseries
    # Reference 1: http://www.cs.ucr.edu/~eamonn/LogicalShapelet.pdf
    # Reference 2: http://making.csie.ndhu.edu.tw/seminar/making/papers/PDF/Fast%20Approximate%20Correlation%20for%20Massive%20Time-series%20Data.pdf
    :param a: list, the subserie (shapelet)
    :param b: list, the timeseries
    :return: the normalized euclidean distance
    """
    s_x, s_x_sqr, s_y, s_y_sqr, m_uv = stats
    min_sum = np.inf
    # Convolute the subserie over the longer timeseries to get the minimal distance
    for v in range(len(b) - len(a)):
        mu_x = s_x[-1] / len(a)   # The final element of s_x contains the sum of all elements in x
        sigma_x = s_x_sqr[-1] / len(a) - mu_x ** 2
        mu_y = (s_y[v+len(a)] - s_y[v]) / len(a)   # Grab subserie of length len(a) from b and calculate mean
        sigma_y = (s_y_sqr[v+len(a)] - s_y_sqr[v]) / len(a) - mu_y ** 2
        m = m_uv[len(a), v+len(a)] - m_uv[0, v]

        dist = (m - len(a) * mu_x * mu_y) / (m*sigma_x*sigma_y)
        min_sum = min(min_sum, dist)

    return np.sqrt(min_sum)


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
            dist = euclidean_distance(sub_serie, time_serie[i:i + len(sub_serie)], min_dist)
            if dist is not None and dist < min_dist: min_dist, min_idx = dist, i
        return min_dist, min_idx
    else:
        return None, None


def subsequence_dist_fast(time_serie, sub_serie, stats):
    if len(sub_serie) < len(time_serie):
        min_dist, min_idx = float("inf"), -1
        for i in range(len(time_serie)-len(sub_serie)+1):
            dist = normalized_euclidean_distance(sub_serie, time_serie[i:i + len(sub_serie)], stats)
            if dist is not None and dist < min_dist: min_dist, min_idx = dist, i
        return min_dist, min_idx
    else:
        return None, None


def partitions(items, k):
    def split(indices):
        i=0
        for j in indices:
            yield items[i:j]
            i = j
        yield items[i:]

    for indices in combinations(range(1, len(items)), k-1):
        yield list(split(indices))


def information_gain_ub(labels):
    # TODO: Can we find a mathematical formula for this?
    cntr = Counter(labels)
    all_values = list(cntr.values())
    all_sum = sum(all_values)
    for i in range(len(all_values)): all_values[i] /= all_sum
    prior_entropy = calculate_entropy(all_values)
    best_ig = 0

    for left, right in partitions(list(cntr.values()), 2):
        left_sum = sum(left)
        right_sum = sum(right)
        for i in range(len(left)): left[i] /= left_sum
        for i in range(len(right)): right[i] /= right_sum
        ig = information_gain(left, right, prior_entropy)
        if ig > best_ig: best_ig = ig

    return best_ig